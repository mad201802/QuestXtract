"""The evaluation harness: measure the pipeline against the labelled set.

NFR2 gives a number to hit (recall >= 50% on the team's labelled set), so the metric runs
in CI. Sections 6 through 9 of architecture.md are blind without it.

A prediction matches a label when their time ranges overlap and the overlap covers at least
half the shorter range. Time-based, not text-based, because ASR text will differ from what a
human typed and the ASR is not what is being evaluated here. Matching is one-to-one, best
overlap first, so one long sentence cannot satisfy two labels.

The labelled *dataset* lives under ``tests/data/<name>.labels.jsonl``, next to the
``<name>.transcript.json`` it is timed against and, for a recording whose board was read, the
``<name>.visual.json`` board states; this module is the code that reads it.

:func:`evaluate_vision` does the same for the vision model, against the labelled keyframes in
``tests/data/vision/<name>.frames.jsonl`` (architecture.md 4.4).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Config
from .llm import JUDGE_PROMPT_VERSION
from .models import Context, Transcript, VisualContext, Word
from .pipeline import STAGES, run
from .vision import read_keyframe

_MIN_OVERLAP = 0.5
_AFTER_TRANSCRIBE = 2  # stages 1 and 2 are skipped: the transcript is committed
_SWEEP = tuple(round(i * 0.05, 2) for i in range(21))

Span = tuple[float, float]

# Lecture types a labelled frame can have: the layout probe's kinds (architecture.md 6.2) that
# are a board or slides, so a label's kind goes into the prompt as the probe's would.
KINDS = ("slides", "blackboard", "whiteboard", "handwriting")
_FORMULA_MAX_ERROR = 0.25  # edits per token at which a labelled formula still counts as read
_UNREADABLE = "[unreadable]"  # what prompt v0 asks for in place of a word it cannot read
# A LaTeX token: a command (\frac, \{), a number, or any other single character.
_LATEX_TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|\d+|\S")
# Spacing and sizing LaTeX that changes how a formula looks, not what it says.
_LATEX_NOISE = re.compile(r"\\(?:(?:left|right|displaystyle|q?quad)(?![a-zA-Z])|[,;:! ])|\$")


def overlap_ratio(prediction: Span, label: Span) -> float:
    """How much of the shorter of the two spans their overlap covers, in ``[0, 1]``."""
    start = max(prediction[0], label[0])
    end = min(prediction[1], label[1])
    shorter = min(prediction[1] - prediction[0], label[1] - label[0])
    if end <= start or shorter <= 0:
        return 0.0
    return (end - start) / shorter


def overlaps(prediction: Span, label: Span, *, minimum: float = _MIN_OVERLAP) -> bool:
    """Whether a predicted span matches a labelled one.

    They match when the overlap covers at least ``minimum`` of the shorter of the two.
    """
    ratio = overlap_ratio(prediction, label)
    return ratio > 0.0 and ratio >= minimum


def match(predictions: Sequence[Span], labels: Sequence[Span]) -> dict[int, int]:
    """Pair predictions with labels one-to-one, highest overlap first.

    Returns:
        Prediction index to label index, for every matched prediction.

    """
    scored = sorted(
        (
            (ratio, i, j)
            for i, p in enumerate(predictions)
            for j, lab in enumerate(labels)
            if (ratio := overlap_ratio(p, lab)) >= _MIN_OVERLAP
        ),
        reverse=True,
    )
    pairs: dict[int, int] = {}
    for _, i, j in scored:
        if i not in pairs and j not in pairs.values():
            pairs[i] = j
    return pairs


def evaluate(
    paths: Sequence[str | Path], config: Config | None = None, *, board: bool = False
) -> dict[str, Any]:
    """Score the pipeline against every labelled recording under ``paths``.

    ``paths`` are ``*.labels.jsonl`` files or directories holding them. A label file with no
    ``<name>.transcript.json`` beside it cannot be scored and is listed under ``skipped``.
    ``config`` defaults to ``Config()``; pass one with ``detector="llm"`` to score the LLM
    detector (its answers are cached, so a second run makes no calls). With ``board``, the
    board states in ``<name>.visual.json`` beside a transcript go into the run, as if vision
    had read them; a recording without that file runs without them.

    Returns:
        The report: the ``detector`` that ran, the gate ``threshold``, the recordings run
        with ``board`` states, per-recording and pooled ``total`` metrics, and a pooled
        ``sweep`` over the confidence threshold.

    Raises:
        FileNotFoundError: If a path does not exist.
        ValueError: If no label file under ``paths`` has a transcript.

    """
    config = config or Config()
    threshold = config.min_confidence
    runs: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[Word, ...]]] = {}
    skipped: list[str] = []
    with_board: list[str] = []
    for labels_path in _label_files(paths, "*.labels.jsonl"):
        name = labels_path.name.removesuffix(".labels.jsonl")
        transcript_path = labels_path.with_name(f"{name}.transcript.json")
        if not transcript_path.is_file():
            skipped.append(str(labels_path))
            continue
        transcript = Transcript.from_file(transcript_path)
        if name in runs:
            raise ValueError(f"duplicate recording name: {name}")
        visual_path = labels_path.with_name(f"{name}.visual.json")
        visual = _read_visual(visual_path) if board and visual_path.is_file() else ()
        if visual:
            with_board.append(name)
        predictions = _predict(transcript, config, visual)
        runs[name] = (_read_labels(labels_path), predictions, transcript.words)
    if not runs:
        raise ValueError(f"no *.labels.jsonl with a matching *.transcript.json under {paths}")

    def pooled(t: float) -> dict[str, Any]:
        return _metrics(_sum([_counts(*r, t) for r in runs.values()]))

    return {
        "detector": _detector(config),
        "threshold": threshold,
        "board": with_board,
        "recordings": {name: _metrics(_counts(*r, threshold)) for name, r in runs.items()},
        "total": pooled(threshold),
        "sweep": [
            {"threshold": t, "recall": m["recall"], "precision": m["precision"], "f1": m["f1"]}
            for t in _SWEEP
            for m in [pooled(t)]
        ],
        "skipped": skipped,
    }


def format_report(report: dict[str, Any]) -> str:
    """Render :func:`evaluate`'s report as a plain-text table plus its failure cases."""
    rows = [*report["recordings"].items(), ("TOTAL", report["total"])]
    lines = [
        f"detector {report['detector']}, threshold {report['threshold']:.2f}"
        + (f", board states for {', '.join(report['board'])}" if report["board"] else ""),
        f"{'recording':<22}{'content':>8}{'tp':>5}{'fp':>5}{'fn':>5}"
        f"{'recall':>8}{'prec':>7}{'f1':>7}{'?-rate':>8}",
    ]
    for name, m in rows:
        lines.append(
            f"{name:<22}{m['content_labels']:>8}{m['tp']:>5}{m['fp']:>5}{m['fn']:>5}"
            f"{m['recall']:>8.2f}{m['precision']:>7.2f}{m['f1']:>7.2f}"
            f"{m['question_mark_rate']:>8.2f}"
        )
    breakdown = ", ".join(
        f"{k} {v}" for k, v in report["total"]["false_positive_breakdown"].items()
    )
    lines += ["", f"false positives by label kind: {breakdown or 'none'}", "", "sweep:"]
    lines += [
        f"  {s['threshold']:.2f}  recall {s['recall']:.2f}  precision {s['precision']:.2f}"
        f"  f1 {s['f1']:.2f}"
        for s in report["sweep"]
    ]
    for name, m in report["recordings"].items():
        lines += ["", f"{name}: missed content questions"]
        lines += [f"  {x['start']:8.1f}  {x['text']}" for x in m["false_negatives"]]
        lines += [f"{name}: false positives"]
        lines += [
            f"  {x['start']:8.1f}  {x['kind']:<10} {x['confidence']:.2f}  {x['text']}"
            for x in m["false_positives"]
        ]
    if report["skipped"]:
        lines += ["", "skipped (no transcript): " + ", ".join(report["skipped"])]
    return "\n".join(lines)


def _label_files(paths: Sequence[str | Path], pattern: str) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out += sorted(path.glob(pattern))
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    return out


def _read_labels(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _read_visual(path: Path) -> tuple[VisualContext, ...]:
    """Read the board states of a ``<name>.visual.json`` (``scripts/board_states.py``)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(VisualContext(**state) for state in data["board_states"])


def _detector(config: Config) -> str:
    """Name what found the questions, so a pasted report says which model and prompt."""
    if config.detector != "llm":
        return config.detector
    return f"llm {config.judge_model} (prompt {JUDGE_PROMPT_VERSION})"


def _predict(
    transcript: Transcript, config: Config, visual: tuple[VisualContext, ...] = ()
) -> list[dict[str, Any]]:
    """Run stages 3 onward with the emission gate open, so the sweep sees every candidate.

    ``visual`` stands in for what the vision stage would have read; that stage itself has no
    video to read here and leaves it as it is.
    """
    config = config.replace(min_confidence=0.0)
    ctx = Context(source=Path("eval"), config=config, transcript=transcript, visual=visual)
    ctx.duration = transcript.duration
    ctx = run(ctx, STAGES[_AFTER_TRANSCRIBE:])
    return [
        {"start": p.start, "end": p.end, "confidence": p.confidence, "text": p.question.text}
        for p in ctx.pairs
    ]


def _counts(
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    words: tuple[Word, ...],
    threshold: float,
) -> dict[str, Any]:
    """Raw counts and failure cases for one recording at one threshold."""
    kept = [p for p in predictions if p["confidence"] >= threshold]
    pairs = match([(p["start"], p["end"]) for p in kept], [(x["start"], x["end"]) for x in labels])
    found = {j for j in pairs.values() if labels[j]["kind"] == "content"}
    content = [j for j, x in enumerate(labels) if x["kind"] == "content"]
    blind = [j for j in content if labels[j].get("split", "seed") == "blind"]
    false_positives = [
        {**p, "kind": labels[pairs[i]]["kind"] if i in pairs else "unlabelled"}
        for i, p in enumerate(kept)
        if pairs.get(i) not in found
    ]
    return {
        "content_labels": len(content),
        "predictions": len(kept),
        "tp": len(found),
        "fp": len(false_positives),
        "fn": len(content) - len(found),
        "blind_labels": len(blind),
        "blind_tp": len(found.intersection(blind)),
        "marked": sum(1 for j in content if _has_mark(labels[j], words)),
        "kinds": Counter(p["kind"] for p in false_positives),
        "false_negatives": [
            {k: labels[j][k] for k in ("start", "end", "text")} for j in content if j not in found
        ],
        "false_positives": false_positives,
    }


def _has_mark(label: dict[str, Any], words: tuple[Word, ...]) -> bool:
    """Whether the transcript words under ``label`` carry a '?' (architecture.md 4.2)."""
    return any("?" in w.text for w in words if w.start < label["end"] and w.end > label["start"])


def _sum(counts: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {k: 0 for k in counts[0] if isinstance(counts[0][k], int)}
    total.update(kinds=Counter(), false_negatives=[], false_positives=[])
    for c in counts:
        for k, v in c.items():
            total[k] += v
    return total


def _metrics(c: dict[str, Any]) -> dict[str, Any]:
    recall = _ratio(c["tp"], c["content_labels"])
    precision = _ratio(c["tp"], c["predictions"])
    f1 = _ratio(2 * recall * precision, recall + precision)
    return {
        "content_labels": c["content_labels"],
        "predictions": c["predictions"],
        "tp": c["tp"],
        "fp": c["fp"],
        "fn": c["fn"],
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(f1, 3),
        "false_positive_breakdown": dict(c["kinds"].most_common()),
        "question_mark_rate": round(_ratio(c["marked"], c["content_labels"]), 3),
        "blind_slice_recall": (
            round(_ratio(c["blind_tp"], c["blind_labels"]), 3) if c["blind_labels"] else None
        ),
        "false_negatives": c["false_negatives"],
        "false_positives": c["false_positives"],
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def character_error_rate(predicted: str, reference: str) -> float:
    """Edits per reference character to turn ``predicted`` into ``reference``, in ``[0, 1]``.

    Both are casefolded and their whitespace collapsed first: line breaks and spacing on a board
    are not something to read wrong. Capped at 1, so one answer full of invented text cannot
    outweigh the rest of a mean. An empty reference scores 0 against an empty answer, else 1.
    """
    predicted, reference = _plain(predicted), _plain(reference)
    if not reference:
        return 0.0 if not predicted else 1.0
    return min(1.0, _distance(predicted, reference) / len(reference))


def formula_found(formula: str, latex: str | None) -> bool:
    r"""Whether a labelled ``formula`` is in a model's ``latex`` answer.

    Both are normalised (``\left``/``\right``, spacing commands and ``$`` removed, ``\dfrac``
    and ``\tfrac`` read as ``\frac``) and split into tokens: a command, a number, or any other
    single character. The formula counts as found when some run of whole tokens in the answer is
    within one edit per four of its tokens: the answer may hold several formulas on one line, and
    a brace more or less is not a misreading. A formula of three tokens or fewer must match
    exactly, and never inside a longer token: a labelled ``x`` is not in ``\max``.
    """
    needle, haystack = _latex(formula), _latex(latex or "")
    if not needle:
        return True
    return _distance(needle, haystack, free_ends=True) <= _FORMULA_MAX_ERROR * len(needle)


def evaluate_vision(paths: Sequence[str | Path], config: Config | None = None) -> dict[str, Any]:
    """Score the vision model against every labelled keyframe under ``paths``.

    ``paths`` are ``*.frames.jsonl`` files or directories holding them; each line names a PNG
    beside the file. Every frame is read with ``config.reading_model`` (the vision model, or the
    OCR model with ``vision_reader = "ocr"``) through
    :func:`~questxtract.vision.read_keyframe`, as the vision stage reads a keyframe when the
    layout probe got the frame's kind right: the label's kind in the prompt, no speech. Answers
    are cached per request, so a second run makes no calls. A frame the model could not read
    counts as an empty answer, and a blank board it could not read as not read.

    A frame labelled blank (no text) has no character error rate: any text at all would score 1,
    the same as misreading a full board. It is counted as read when the answer's text is empty
    or only ``[unreadable]``.

    Returns:
        The report: the ``model`` and ``prompt`` version, metrics per lecture type under
        ``kinds`` and pooled under ``total`` (frames, failed readings, mean character error rate
        over the frames with text, blank frames and how many were read as blank, formulas
        labelled and found, formula recall), and the score of every frame.

    Raises:
        FileNotFoundError: If a path or a labelled frame does not exist.
        ValueError: If a label line is malformed, or no frame is labelled under ``paths``.

    """
    config = config or Config()
    frames: list[dict[str, Any]] = []
    for labels_path in _label_files(paths, "*.frames.jsonl"):
        for number, label in enumerate(_read_labels(labels_path), 1):
            _check_frame(label, f"{labels_path}:{number}")
            png = labels_path.with_name(label["frame"])
            if not png.is_file():
                raise FileNotFoundError(f"{labels_path}:{number}: no frame {png.name} beside it")
            answer = read_keyframe(config, png.read_bytes(), kind=label["kind"])
            text = answer["text"] if answer else ""
            latex = answer["latex"] if answer else None
            blank = not label["text"].strip()
            frames.append(
                {
                    "frame": label["frame"],
                    "kind": label["kind"],
                    "failed": answer is None,
                    "blank": blank,
                    "read_blank": blank
                    and answer is not None
                    and not text.replace(_UNREADABLE, "").strip(),
                    "cer": None if blank else round(character_error_rate(text, label["text"]), 3),
                    "formulas": len(label["latex"]),
                    "missed": [f for f in label["latex"] if not formula_found(f, latex)],
                }
            )
    if not frames:
        raise ValueError(f"no labelled frame in a *.frames.jsonl under {paths}")
    return {
        "model": config.reading_model,
        "prompt": "v0",  # what read_keyframe sends without a previous state (6.5)
        "kinds": {
            kind: _vision_metrics([f for f in frames if f["kind"] == kind])
            for kind in KINDS
            if any(f["kind"] == kind for f in frames)
        },
        "total": _vision_metrics(frames),
        "frames": frames,
    }


def format_vision_report(report: dict[str, Any]) -> str:
    """Render :func:`evaluate_vision`'s report as a table per lecture type, then per frame."""
    rows = [*report["kinds"].items(), ("TOTAL", report["total"])]
    lines = [
        f"vision model {report['model']} (prompt {report['prompt']})",
        f"{'lecture type':<14}{'frames':>7}{'failed':>7}{'CER':>7}{'blank':>7}{'formulas':>10}"
        f"{'found':>7}{'recall':>8}",
    ]
    for name, m in rows:
        cer = "-" if m["cer"] is None else f"{m['cer']:.2f}"
        blank = f"{m['blank_read']}/{m['blank']}" if m["blank"] else "-"
        recall = "-" if m["formula_recall"] is None else f"{m['formula_recall']:.2f}"
        lines.append(
            f"{name:<14}{m['frames']:>7}{m['failed']:>7}{cer:>7}{blank:>7}{m['formulas']:>10}"
            f"{m['formulas_found']:>7}{recall:>8}"
        )
    lines += ["", "per frame, worst text first (a blank board: read as blank or not):"]
    for f in sorted(report["frames"], key=_worst_first):
        if f["blank"]:
            score = "blank" if f["read_blank"] else "NOT blank"
        else:
            score = f"{f['cer']:.2f}"
        found = f"{f['formulas'] - len(f['missed'])}/{f['formulas']}"
        failed = "  FAILED" if f["failed"] else ""
        lines.append(f"  {score:>9}  {found:>5}  {f['kind']:<11} {f['frame']}{failed}")
        lines += [f"             missed: {formula}" for formula in f["missed"]]
    return "\n".join(lines)


def _worst_first(frame: dict[str, Any]) -> float:
    """Sort key: a blank board read as text first, then by character error rate."""
    if frame["blank"]:
        return 0.0 if frame["read_blank"] else -2.0
    return -float(frame["cer"])


def _check_frame(label: Any, where: str) -> None:
    """Raise ``ValueError`` naming ``where`` unless ``label`` is one well-formed frame label."""
    problem = None
    if not isinstance(label, dict):
        problem = "not a JSON object"
    elif (
        not isinstance(label.get("frame"), str)
        or Path(label["frame"]).name != label["frame"]
        or not label["frame"].endswith(".png")
    ):
        problem = "frame must be the name of a .png file beside the labels"
    elif label.get("kind") not in KINDS:
        problem = f"kind must be one of {', '.join(KINDS)}"
    elif not isinstance(label.get("text"), str):
        problem = "text must be a string"
    elif not isinstance(label.get("latex"), list) or not all(
        isinstance(x, str) and x.strip() for x in label["latex"]
    ):
        problem = "latex must be a list of formulas"
    elif not (label.get("diagram") is None or isinstance(label["diagram"], str)):
        problem = "diagram must be a string or null"
    if problem:
        raise ValueError(f"{where}: {problem}")


def _vision_metrics(frames: list[dict[str, Any]]) -> dict[str, Any]:
    formulas = sum(f["formulas"] for f in frames)
    found = formulas - sum(len(f["missed"]) for f in frames)
    rates = [f["cer"] for f in frames if not f["blank"]]
    return {
        "frames": len(frames),
        "failed": sum(f["failed"] for f in frames),
        "cer": round(sum(rates) / len(rates), 3) if rates else None,
        "blank": sum(f["blank"] for f in frames),
        "blank_read": sum(f["read_blank"] for f in frames),
        "formulas": formulas,
        "formulas_found": found,
        "formula_recall": round(found / formulas, 3) if formulas else None,
    }


def _plain(text: str) -> str:
    return " ".join(text.casefold().split())


def _latex(formula: str) -> list[str]:
    """Normalise a formula and split it into tokens, before whitespace can join two of them."""
    formula = _LATEX_NOISE.sub("", formula)
    formula = re.sub(r"\\[dt]frac(?![a-zA-Z])", r"\\frac", formula)
    return _LATEX_TOKEN.findall(formula)


def _distance(a: Sequence[str], b: Sequence[str], *, free_ends: bool = False) -> int:
    """Count the edits (insert, delete, substitute) that turn ``b`` into ``a``.

    With ``free_ends``, ``b`` is a haystack: count the fewest edits that turn some run of it
    into ``a``, so a match may start and end anywhere in ``b`` at no cost.
    """
    previous = [0] * (len(b) + 1) if free_ends else list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return min(previous) if free_ends else previous[-1]

"""The evaluation harness: measure the pipeline against the labelled set.

NFR2 gives a number to hit (recall >= 50% on the team's labelled set), so the metric runs
in CI. Sections 6 through 9 of architecture.md are blind without it.

A prediction matches a label when their time ranges overlap and the overlap covers at least
half the shorter range. Time-based, not text-based, because ASR text will differ from what a
human typed and the ASR is not what is being evaluated here. Matching is one-to-one, best
overlap first, so one long sentence cannot satisfy two labels.

The labelled *dataset* lives under ``tests/data/<name>.labels.jsonl``, next to the
``<name>.transcript.json`` it is timed against; this module is the code that reads it.

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
from .models import Context, Transcript, Word
from .pipeline import STAGES, run
from .vision import VISION_PROMPT_VERSION, _caption

_MIN_OVERLAP = 0.5
_AFTER_TRANSCRIBE = 2  # stages 1 and 2 are skipped: the transcript is committed
_SWEEP = tuple(round(i * 0.05, 2) for i in range(21))

Span = tuple[float, float]

KINDS = ("slides", "blackboard", "whiteboard", "tablet")  # lecture types, as prompt v0 names them
_FORMULA_MAX_ERROR = 0.25  # edits per character at which a labelled formula still counts as read
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


def evaluate(paths: Sequence[str | Path], config: Config | None = None) -> dict[str, Any]:
    """Score the pipeline against every labelled recording under ``paths``.

    ``paths`` are ``*.labels.jsonl`` files or directories holding them. A label file with no
    ``<name>.transcript.json`` beside it cannot be scored and is listed under ``skipped``.
    ``config`` defaults to ``Config()``; pass one with ``detector="llm"`` to score the LLM
    detector (its answers are cached, so a second run makes no calls).

    Returns:
        The report: the ``detector`` that ran, the gate ``threshold``, per-recording and
        pooled ``total`` metrics, and a pooled ``sweep`` over the confidence threshold.

    Raises:
        FileNotFoundError: If a path does not exist.
        ValueError: If no label file under ``paths`` has a transcript.

    """
    config = config or Config()
    threshold = config.min_confidence
    runs: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[Word, ...]]] = {}
    skipped: list[str] = []
    for labels_path in _label_files(paths, "*.labels.jsonl"):
        name = labels_path.name.removesuffix(".labels.jsonl")
        transcript_path = labels_path.with_name(f"{name}.transcript.json")
        if not transcript_path.is_file():
            skipped.append(str(labels_path))
            continue
        transcript = Transcript.from_file(transcript_path)
        if name in runs:
            raise ValueError(f"duplicate recording name: {name}")
        runs[name] = (_read_labels(labels_path), _predict(transcript, config), transcript.words)
    if not runs:
        raise ValueError(f"no *.labels.jsonl with a matching *.transcript.json under {paths}")

    def pooled(t: float) -> dict[str, Any]:
        return _metrics(_sum([_counts(*r, t) for r in runs.values()]))

    return {
        "detector": _detector(config),
        "threshold": threshold,
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
        f"detector {report['detector']}, threshold {report['threshold']:.2f}",
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


def _detector(config: Config) -> str:
    """Name what found the questions, so a pasted report says which model and prompt."""
    if config.detector != "llm":
        return config.detector
    return f"llm {config.judge_model} (prompt {JUDGE_PROMPT_VERSION})"


def _predict(transcript: Transcript, config: Config) -> list[dict[str, Any]]:
    """Run stages 3 onward with the emission gate open, so the sweep sees every candidate."""
    config = config.replace(min_confidence=0.0)
    ctx = Context(source=Path("eval"), config=config, transcript=transcript)
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

    Both are normalised (whitespace, ``\left``/``\right``, spacing commands and ``$`` removed,
    ``\dfrac`` and ``\tfrac`` read as ``\frac``), and the formula counts as found when some
    stretch of the answer is within one edit per four of its characters: the answer may hold
    several formulas on one line, and a brace more or less is not a misreading.
    """
    needle, haystack = _latex(formula), _latex(latex or "")
    if not needle:
        return True
    return _substring_distance(needle, haystack) <= _FORMULA_MAX_ERROR * len(needle)


def evaluate_vision(paths: Sequence[str | Path], config: Config | None = None) -> dict[str, Any]:
    """Score the vision model against every labelled keyframe under ``paths``.

    ``paths`` are ``*.frames.jsonl`` files or directories holding them; each line names a PNG
    beside the file. Every frame is read with ``config.vision_model`` as the vision stage reads
    a keyframe (same prompt, frame kind ``unknown``, no speech), cached per request, so a second
    run makes no calls. A frame the model could not read counts as an empty answer.

    Returns:
        The report: the ``model`` and ``prompt`` version, metrics per lecture type under
        ``kinds`` and pooled under ``total`` (frames, failed readings, mean character error
        rate, formulas labelled and found, formula recall), and the score of every frame.

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
            answer = _caption(config, png.read_bytes(), "")
            text = answer["text"] if answer else ""
            latex = answer["latex"] if answer else None
            frames.append(
                {
                    "frame": label["frame"],
                    "kind": label["kind"],
                    "failed": answer is None,
                    "cer": round(character_error_rate(text, label["text"]), 3),
                    "formulas": len(label["latex"]),
                    "missed": [f for f in label["latex"] if not formula_found(f, latex)],
                }
            )
    if not frames:
        raise ValueError(f"no labelled frame in a *.frames.jsonl under {paths}")
    return {
        "model": config.vision_model,
        "prompt": VISION_PROMPT_VERSION,
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
        f"{'lecture type':<14}{'frames':>7}{'failed':>7}{'CER':>7}{'formulas':>10}{'found':>7}"
        f"{'recall':>8}",
    ]
    for name, m in rows:
        recall = "-" if m["formula_recall"] is None else f"{m['formula_recall']:.2f}"
        lines.append(
            f"{name:<14}{m['frames']:>7}{m['failed']:>7}{m['cer']:>7.2f}{m['formulas']:>10}"
            f"{m['formulas_found']:>7}{recall:>8}"
        )
    lines += ["", "per frame, worst text first:"]
    for f in sorted(report["frames"], key=lambda f: -f["cer"]):
        found = f"{f['formulas'] - len(f['missed'])}/{f['formulas']}"
        failed = "  FAILED" if f["failed"] else ""
        lines.append(f"  {f['cer']:.2f}  {found:>5}  {f['kind']:<10} {f['frame']}{failed}")
        lines += [f"        missed: {formula}" for formula in f["missed"]]
    return "\n".join(lines)


def _check_frame(label: Any, where: str) -> None:
    """Raise ``ValueError`` naming ``where`` unless ``label`` is one well-formed frame label."""
    problem = None
    if not isinstance(label, dict):
        problem = "not a JSON object"
    elif not isinstance(label.get("frame"), str) or Path(label["frame"]).name != label["frame"]:
        problem = "frame must be a file name beside the labels"
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
    return {
        "frames": len(frames),
        "failed": sum(f["failed"] for f in frames),
        "cer": round(sum(f["cer"] for f in frames) / len(frames), 3),
        "formulas": formulas,
        "formulas_found": found,
        "formula_recall": round(found / formulas, 3) if formulas else None,
    }


def _plain(text: str) -> str:
    return " ".join(text.casefold().split())


def _latex(formula: str) -> str:
    formula = _LATEX_NOISE.sub("", formula)
    formula = re.sub(r"\\[dt]frac(?![a-zA-Z])", r"\\frac", formula)
    return "".join(formula.split())


def _distance(a: str, b: str) -> int:
    """Count the edits between two strings: insertions, deletions and substitutions."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _substring_distance(needle: str, haystack: str) -> int:
    """Count the fewest edits that turn some stretch of ``haystack`` into ``needle``."""
    previous = [0] * (len(haystack) + 1)  # a match may start anywhere in the haystack
    for i, x in enumerate(needle, 1):
        current = [i]
        for j, y in enumerate(haystack, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return min(previous)

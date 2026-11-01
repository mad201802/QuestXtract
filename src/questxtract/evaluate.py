"""The evaluation harness: measure the pipeline against the labelled set."""

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
_AFTER_TRANSCRIBE = 2
_SWEEP = tuple(round(i * 0.05, 2) for i in range(21))

Span = tuple[float, float]

KINDS = ("slides", "blackboard", "whiteboard", "handwriting")
_FORMULA_MAX_ERROR = 0.25
_UNREADABLE = "[unreadable]"
_LATEX_TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|\d+|\S")
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
    """Whether a predicted span matches a labelled one."""
    ratio = overlap_ratio(prediction, label)
    return ratio > 0.0 and ratio >= minimum


def match(predictions: Sequence[Span], labels: Sequence[Span]) -> dict[int, int]:
    """Pair predictions with labels one-to-one, highest overlap first."""
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
    """Score the pipeline against every labelled recording under ``paths``."""
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
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(VisualContext(**state) for state in data["board_states"])


def _detector(config: Config) -> str:
    if config.detector != "llm":
        return config.detector
    return f"llm {config.judge_model} (prompt {JUDGE_PROMPT_VERSION})"


def _predict(
    transcript: Transcript, config: Config, visual: tuple[VisualContext, ...] = ()
) -> list[dict[str, Any]]:
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
    """Edits per reference character to turn ``predicted`` into ``reference``, in ``[0, 1]``."""
    predicted, reference = _plain(predicted), _plain(reference)
    if not reference:
        return 0.0 if not predicted else 1.0
    return min(1.0, _distance(predicted, reference) / len(reference))


def formula_found(formula: str, latex: str | None) -> bool:
    """Whether a labelled ``formula`` is in a model's ``latex`` answer."""
    needle, haystack = _latex(formula), _latex(latex or "")
    if not needle:
        return True
    return _distance(needle, haystack, free_ends=True) <= _FORMULA_MAX_ERROR * len(needle)


def evaluate_vision(paths: Sequence[str | Path], config: Config | None = None) -> dict[str, Any]:
    """Score the vision model against every labelled keyframe under ``paths``."""
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
        "prompt": "v0",
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
    if frame["blank"]:
        return 0.0 if frame["read_blank"] else -2.0
    return -float(frame["cer"])


def _check_frame(label: Any, where: str) -> None:
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
    formula = _LATEX_NOISE.sub("", formula)
    formula = re.sub(r"\\[dt]frac(?![a-zA-Z])", r"\\frac", formula)
    return _LATEX_TOKEN.findall(formula)


def _distance(a: Sequence[str], b: Sequence[str], *, free_ends: bool = False) -> int:
    previous = [0] * (len(b) + 1) if free_ends else list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return min(previous) if free_ends else previous[-1]

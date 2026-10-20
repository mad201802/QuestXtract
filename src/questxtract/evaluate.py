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
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Config
from .llm import JUDGE_PROMPT_VERSION
from .models import Context, Transcript, VisualContext, Word
from .pipeline import STAGES, run

_MIN_OVERLAP = 0.5
_AFTER_TRANSCRIBE = 2  # stages 1 and 2 are skipped: the transcript is committed
_SWEEP = tuple(round(i * 0.05, 2) for i in range(21))

Span = tuple[float, float]


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
    for labels_path in _label_files(paths):
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


def _label_files(paths: Sequence[str | Path]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out += sorted(path.glob("*.labels.jsonl"))
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

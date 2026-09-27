"""Writing and reading ``questions.json`` (pipeline.md 12.1).

Export is not a stage: it runs on the finished ``Context``.

One rule governs the shape: **do not drop ``None`` on serialisation.** A fixed key set is
emitted, always, ``null`` included, so consumers never have to defensively ``.get()`` a
missing ``answer_text``. The format is also the edit surface (FR5): the editor lives in a
wrapper around this package, and re-running the pipeline merges human edits back by id,
which is why sentence ids are content hashes rather than indices.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .models import Context, QAPair, SchemaError, Segment, Sentence, VisualContext
from .score import confidence_label

SCHEMA_VERSION = "1.1"  # 1.0 -> 1.1: added start_hms/end_hms to the fixed question key set

_CONFIG_KEYS = ("model_size", "language", "judge_model", "vision_model")


def result_to_dict(ctx: Context) -> dict[str, Any]:
    """Build the exported representation of a finished context."""
    by_id = {s.id: s for s in ctx.sentences}
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": str(ctx.source),
            "duration_seconds": ctx.duration or None,
            "hash": ctx.source_hash or None,
        },
        "config": {key: getattr(ctx.config, key) for key in _CONFIG_KEYS},
        "stats": dict(ctx.stats),
        "questions": [_question(pair, by_id, ctx.segments) for pair in ctx.pairs],
    }


def _format_hms(seconds: float) -> str:
    """Render ``seconds`` as ``M:SS``, or ``H:MM:SS`` once the recording is an hour long.

    For humans checking a question against the source recording (a YouTube ``&t=`` link,
    scrubbing a video player); ``start``/``end`` stay the machine-usable seconds this is
    derived from - filtering (issue #16) and seeking both want the number, not this string.
    """
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _question(
    pair: QAPair, by_id: dict[str, Sentence], segments: Sequence[Segment]
) -> dict[str, Any]:
    """One question object, with every documented key present."""
    sentence = by_id.get(pair.question.sentence_id)
    original = sentence.text if sentence and sentence.text != pair.question.text else None
    return {
        "id": pair.id,
        "text": pair.question.text,
        "original_text": original if pair.verdict == "revise" else None,
        "start": pair.start,
        "end": pair.end,
        "start_hms": _format_hms(pair.start),
        "end_hms": _format_hms(pair.end),
        "confidence": pair.confidence,
        "confidence_label": confidence_label(pair.confidence),
        "question_type": pair.question.question_type,
        "answer": _answer(pair, by_id),
        "segment_id": sentence.segment_index if sentence else None,
        "sentence_id": pair.question.sentence_id,
        "visual": _visual(sentence, segments),
        "verdict": pair.verdict,
        "reasons": list(pair.question.reasons) + list(pair.reasons),
    }


def _answer(pair: QAPair, by_id: dict[str, Sentence]) -> dict[str, Any] | None:
    """Return the answer with the span of its own sentences, or ``None`` when none was found."""
    if not pair.answer_text:
        return None
    covered = [by_id[sentence_id] for sentence_id in pair.answer_sentence_ids]
    return {
        "text": pair.answer_text,
        "start": min(s.start for s in covered),
        "end": max(s.end for s in covered),
        "sentence_ids": list(pair.answer_sentence_ids),
    }


def _visual(sentence: Sentence | None, segments: Sequence[Segment]) -> dict[str, Any] | None:
    """Return the board state overlapping this question, or ``None`` when vision was off.

    The join lives on :class:`~questxtract.models.Segment` (pipeline.md 5); the export
    surfaces the first state overlapping the question's own segment.
    """
    if sentence is None or not 0 <= sentence.segment_index < len(segments):
        return None
    states = segments[sentence.segment_index].visual
    if not states:
        return None
    state: VisualContext = states[0]
    return {
        "start": state.start,
        "end": state.end,
        "keyframe_path": state.keyframe_path,
        "description": state.description,
        "text": state.text,
        "latex": state.latex,
    }


def result_to_json(ctx: Context) -> str:
    """Serialise a finished context to the exported JSON format."""
    return json.dumps(result_to_dict(ctx), indent=2, ensure_ascii=False)


def write_questions(ctx: Context, path: str | Path) -> None:
    """Write a finished context to ``path`` as ``questions.json`` (UTF-8)."""
    Path(path).write_text(result_to_json(ctx), encoding="utf-8")


def read_questions(path: str | Path) -> dict[str, Any]:
    """Read a ``questions.json`` back, edited or not.

    Returns the parsed document rather than a ``Context``: the export is a projection of a
    finished run, not a snapshot of one, so it does not round-trip into a runnable context.

    Raises:
        SchemaError: If the file is not JSON or does not match the exported schema.

    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SchemaError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise SchemaError(f"{path}: expected an object with a 'questions' list")
    return data

"""Writing and reading ``questions.json`` (architecture.md 12).

Export is not a stage: it runs on the finished ``Context``.

One rule governs the shape: **do not drop ``None`` on serialisation.** A fixed key set is
emitted, always, ``null`` included, so consumers never have to defensively ``.get()`` a
missing ``answer``. Segments and board states are listed once and referenced by id. The
format is also the edit surface (FR5): ``review`` is the one block people own, and a re-run
merges it back by question id (#15). Formal description: ``questions.schema.json``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .models import Context, QAPair, SchemaError, Segment, Sentence, VisualContext
from .score import confidence_label

SCHEMA_VERSION = "2.0"
SCHEMA_PATH = Path(__file__).with_name("questions.schema.json")

_CONFIG_KEYS = ("model_size", "language", "min_confidence", "judge_model", "vision_model")
_SOURCE = "audio"  # questions are only read from speech until the vision milestones (M2/M3)


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
        "segments": [_segment(index, segment) for index, segment in enumerate(ctx.segments)],
        "visuals": [_visual(state) for state in ctx.visual],
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


def _segment(index: int, segment: Segment) -> dict[str, Any]:
    """One segment: its span and the board states joined to it, not its text."""
    return {
        "id": index,
        "start": segment.start,
        "end": segment.end,
        "topic": None,  # reserved for topic filtering (#16, FR11); no topics exist yet
        "visual_ids": [state.id for state in segment.visual],
    }


def _visual(state: VisualContext) -> dict[str, Any]:
    """One board state, listed once however many questions point at it."""
    return {
        "id": state.id,
        "start": state.start,
        "end": state.end,
        "keyframe_path": state.keyframe_path,
        "description": state.description,
        "text": state.text,
        "latex": state.latex,
        "confidence": state.confidence,
    }


def _question(
    pair: QAPair, by_id: dict[str, Sentence], segments: Sequence[Segment]
) -> dict[str, Any]:
    """One question object, with every documented key present."""
    sentence = by_id.get(pair.question.sentence_id)
    original = sentence.text if sentence and sentence.text != pair.question.text else None
    segment = _segment_of(sentence, segments)
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
        "source": _SOURCE,
        "answer": _answer(pair, by_id),
        "segment_id": segment,
        "sentence_id": pair.question.sentence_id,
        "visual_ids": [] if segment is None else [s.id for s in segments[segment].visual],
        "verdict": pair.verdict,
        "reasons": list(pair.question.reasons) + list(pair.reasons),
        "review": {"status": "pending", "text": None, "note": None, "metadata": {}},
    }


def _segment_of(sentence: Sentence | None, segments: Sequence[Segment]) -> int | None:
    """Return the index of the sentence's segment, or ``None`` when it has none."""
    if sentence is None or not 0 <= sentence.segment_index < len(segments):
        return None
    return sentence.segment_index


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
        SchemaError: If the file is not JSON, was written by another major schema version,
            or lacks the ``segments``, ``visuals`` or ``questions`` list.

    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SchemaError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: expected a JSON object")
    version = str(data.get("schema_version", ""))
    major = SCHEMA_VERSION.split(".")[0]
    if version.split(".")[0] != major:
        raise SchemaError(
            f"{path}: schema version {version or 'missing'}, but this questxtract reads "
            f"{major}.x; re-run `questxtract process` to regenerate it"
        )
    for key in ("segments", "visuals", "questions"):
        if not isinstance(data.get(key), list):
            raise SchemaError(f"{path}: expected a '{key}' list")
    return data

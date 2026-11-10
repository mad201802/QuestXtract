"""Writing and reading ``questions.json`` (architecture.md 12)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .llm import JUDGE_PROMPT_VERSION
from .models import (
    Context,
    QAPair,
    SchemaError,
    Segment,
    Sentence,
    VisualContext,
    overlapping,
)
from .score import confidence_label

SCHEMA_VERSION = "2.4"
SCHEMA_PATH = Path(__file__).with_name("questions.schema.json")

_CONFIG_KEYS = ("model_size", "language", "min_confidence")
#: The review statuses that take a question out of every export (questions.schema.json).
REJECTED = ("deleted", "false_positive")


def result_to_dict(ctx: Context, keyframes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build the exported representation of a finished context."""
    by_id = {s.id: s for s in ctx.sentences}
    llm = ctx.config.detector == "llm"
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": str(ctx.source),
            "duration_seconds": ctx.duration or None,
            "hash": ctx.source_hash or None,
        },
        "config": {
            **{key: getattr(ctx.config, key) for key in _CONFIG_KEYS},
            "detector": ctx.config.detector,
            "judge_model": ctx.config.judge_model if llm else None,
            "judge_prompt_version": JUDGE_PROMPT_VERSION if llm else None,
            "judge_reasoning_effort": ctx.config.reasoning_effort if llm else None,
            "vision_model": ctx.config.vision_model if ctx.config.vision_enabled else None,
            "vision_reasoning_effort": (
                ctx.config.vision_effort if ctx.config.vision_enabled else None
            ),
            "vision_ocr_model": (
                ctx.config.vision_ocr_model
                if ctx.config.vision_enabled and ctx.config.vision_reader == "ocr"
                else None
            ),
        },
        "stats": dict(ctx.stats),
        "segments": [_segment(index, segment) for index, segment in enumerate(ctx.segments)],
        "visuals": [_visual(state, keyframes or {}) for state in ctx.visual],
        "questions": [_question(pair, by_id, ctx.segments) for pair in ctx.pairs],
    }


def _format_hms(seconds: float) -> str:
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _segment(index: int, segment: Segment) -> dict[str, Any]:
    return {
        "id": index,
        "start": segment.start,
        "end": segment.end,
        "topic": None,
        "visual_ids": [state.id for state in segment.visual],
    }


def _visual(state: VisualContext, keyframes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "id": state.id,
        "start": state.start,
        "end": state.end,
        "keyframe_path": keyframes.get(state.keyframe_path, state.keyframe_path),
        "description": state.description,
        "text": state.text,
        "latex": state.latex,
        "confidence": state.confidence,
    }


def _question(
    pair: QAPair, by_id: dict[str, Sentence], segments: Sequence[Segment]
) -> dict[str, Any]:
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
        "source": "combined" if "board_overlap" in pair.question.reasons else "audio",
        "answer": _answer(pair, by_id),
        "segment_id": segment,
        "sentence_id": pair.question.sentence_id,
        "visual_ids": _visual_ids(pair, segment, segments),
        "verdict": pair.verdict,
        "reasons": list(pair.question.reasons) + list(pair.reasons),
        "review": {"status": "pending", "text": None, "note": None, "metadata": {}},
    }


def _visual_ids(pair: QAPair, segment: int | None, segments: Sequence[Segment]) -> list[str]:
    if segment is None:
        return []
    states = segments[segment].visual
    return [s.id for s in overlapping(states, pair.start, pair.end) or states]


def _segment_of(sentence: Sentence | None, segments: Sequence[Segment]) -> int | None:
    if sentence is None or not 0 <= sentence.segment_index < len(segments):
        return None
    return sentence.segment_index


def _answer(pair: QAPair, by_id: dict[str, Sentence]) -> dict[str, Any] | None:
    if not pair.answer_text:
        return None
    covered = [by_id[sentence_id] for sentence_id in pair.answer_sentence_ids]
    return {
        "text": pair.answer_text,
        "start": min(s.start for s in covered),
        "end": max(s.end for s in covered),
        "sentence_ids": list(pair.answer_sentence_ids),
    }


def result_to_json(ctx: Context, keyframes: Mapping[str, str] | None = None) -> str:
    """Serialise a finished context to the exported JSON format."""
    return json.dumps(result_to_dict(ctx, keyframes), indent=2, ensure_ascii=False)


def write_questions(ctx: Context, path: str | Path) -> None:
    """Write a finished context to ``path`` as ``questions.json`` (UTF-8), with its keyframes.

    Each board state's keyframe is copied into a folder next to ``path``, named after it
    (``lecture.questions.json`` gets ``lecture.keyframes/<visual id><suffix>``), and
    ``keyframe_path`` is written relative to ``path``, so the file and its folder can be moved
    or shared together, and lectures written to one folder keep their images apart
    (architecture.md 12.1).

    Raises:
        FileNotFoundError: If a board state's keyframe is gone from the cache.

    """
    path = Path(path)
    cache = Path(ctx.config.cache_dir)
    keyframes = _copy_keyframes(ctx.visual, path.parent, _keyframes_dir(path), cache)
    path.write_text(result_to_json(ctx, keyframes), encoding="utf-8")


def _keyframes_dir(path: str | Path) -> str:
    stem = Path(path).name.removesuffix(".json").removesuffix(".questions")
    return f"{stem}.keyframes"


def _copy_keyframes(
    visual: Sequence[VisualContext], folder: Path, name: str, cache: Path
) -> dict[str, str]:
    copied: dict[str, str] = {}
    for state in visual:
        if not state.keyframe_path or state.keyframe_path in copied:
            continue
        source = cache / state.keyframe_path
        if not source.is_file():
            raise FileNotFoundError(f"{source}: keyframe of board state {state.id} is missing")
        image = f"{state.id}{source.suffix}"
        (folder / name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, folder / name / image)
        copied[state.keyframe_path] = f"{name}/{image}"
    return copied


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
    except (json.JSONDecodeError, UnicodeError, RecursionError) as exc:
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

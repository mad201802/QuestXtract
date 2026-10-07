"""Writing and reading ``questions.json`` (architecture.md 12).

Export is not a stage: it runs on the finished ``Context``.

One rule governs the shape: **do not drop ``None`` on serialisation.** A fixed key set is
emitted, always, ``null`` included, so consumers never have to defensively ``.get()`` a
missing ``answer``. Segments and board states are listed once and referenced by id. The
format is also the edit surface (FR5): ``review`` is the one block people own. A re-run does
not carry those edits over yet; merging them back by question id is #15. Keyframes travel
with the file: :func:`write_questions` copies them next to it. Formal description:
``questions.schema.json``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import Context, QAPair, SchemaError, Segment, Sentence, VisualContext
from .score import confidence_label
from .vision import overlapping

SCHEMA_VERSION = "2.1"
SCHEMA_PATH = Path(__file__).with_name("questions.schema.json")
KEYFRAMES_DIR = "keyframes"  # next to questions.json; keyframe_path is relative to that file

_CONFIG_KEYS = ("model_size", "language", "min_confidence")
_SOURCE = "audio"  # questions are only read from speech until the vision milestones (M2/M3)


def result_to_dict(ctx: Context, keyframes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build the exported representation of a finished context.

    Args:
        ctx: The finished context.
        keyframes: Where each keyframe was copied, by its cache path; see
            :func:`write_questions`. Without it, ``keyframe_path`` is the cache path.

    """
    by_id = {s.id: s for s in ctx.sentences}
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": str(ctx.source),
            "duration_seconds": ctx.duration or None,
            "hash": ctx.source_hash or None,
        },
        "config": {
            **{key: getattr(ctx.config, key) for key in _CONFIG_KEYS},
            # null when the stage is off, so the file never names a model that did not run
            "judge_model": ctx.config.judge_model if ctx.config.detector == "llm" else None,
            "vision_model": ctx.config.vision_model if ctx.config.vision_enabled else None,
        },
        "stats": dict(ctx.stats),
        "segments": [_segment(index, segment) for index, segment in enumerate(ctx.segments)],
        "visuals": [_visual(state, keyframes or {}) for state in ctx.visual],
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


def _visual(state: VisualContext, keyframes: Mapping[str, str]) -> dict[str, Any]:
    """One board state, listed once however many questions point at it.

    Only the written path changes; ``id`` stays the state's own, so the ``visual_ids`` of
    segments and questions keep pointing at it.
    """
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
        "visual_ids": _visual_ids(pair, segment, segments),
        "verdict": pair.verdict,
        "reasons": list(pair.question.reasons) + list(pair.reasons),
        "review": {"status": "pending", "text": None, "note": None, "metadata": {}},
    }


def _visual_ids(pair: QAPair, segment: int | None, segments: Sequence[Segment]) -> list[str]:
    """Return the ids of the board states on screen while the question was asked.

    A segment runs up to 90 s, so its board states can postdate the question (#40). Falls
    back to every state of the segment when none overlaps the question itself; empty when
    vision is off or the question has no segment.
    """
    if segment is None:
        return []
    states = segments[segment].visual
    return [s.id for s in overlapping(states, pair.start, pair.end) or states]


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


def result_to_json(ctx: Context, keyframes: Mapping[str, str] | None = None) -> str:
    """Serialise a finished context to the exported JSON format."""
    return json.dumps(result_to_dict(ctx, keyframes), indent=2, ensure_ascii=False)


def write_questions(ctx: Context, path: str | Path) -> None:
    """Write a finished context to ``path`` as ``questions.json`` (UTF-8), with its keyframes.

    Each board state's keyframe is copied to ``keyframes/<visual id><suffix>`` next to
    ``path``, and ``keyframe_path`` is written relative to ``path``, so the file and its folder
    can be moved or shared together (architecture.md 12.1).

    Raises:
        FileNotFoundError: If a board state's keyframe is gone from the cache.

    """
    path = Path(path)
    keyframes = _copy_keyframes(ctx.visual, path.parent, Path(ctx.config.cache_dir))
    path.write_text(result_to_json(ctx, keyframes), encoding="utf-8")


def _copy_keyframes(visual: Sequence[VisualContext], folder: Path, cache: Path) -> dict[str, str]:
    """Copy every keyframe into ``folder/keyframes``; map each cache path to its new one.

    A relative ``keyframe_path`` is relative to the cache directory (#59), so a cached run
    reused from another working directory still finds its images.
    """
    copied: dict[str, str] = {}
    for state in visual:
        if not state.keyframe_path or state.keyframe_path in copied:
            continue
        source = cache / state.keyframe_path  # an absolute keyframe_path stays as it is
        if not source.is_file():
            raise FileNotFoundError(f"{source}: keyframe of board state {state.id} is missing")
        name = f"{state.id}{source.suffix}"
        (folder / KEYFRAMES_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, folder / KEYFRAMES_DIR / name)
        copied[state.keyframe_path] = f"{KEYFRAMES_DIR}/{name}"
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

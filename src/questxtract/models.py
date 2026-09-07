"""Core data types and JSON (de)serialisation for QuestXtract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TranscriptSegment:
    """A chunk of transcribed speech with its position in the recording.

    Args:
        text: The transcribed text of the segment.
        start: Start time in seconds from the beginning of the recording.
        end: End time in seconds from the beginning of the recording.

    """

    text: str
    start: float
    end: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptSegment:
        """Build a segment from a plain dict (e.g. an ASR engine's JSON output)."""
        return cls(text=str(data["text"]), start=float(data["start"]), end=float(data["end"]))


@dataclass(frozen=True)
class Question:
    """A question detected in a transcript.

    Args:
        text: The question sentence as spoken.
        start: Start time in seconds of the segment the question was found in.
        end: End time in seconds of that segment.
        confidence: Detection confidence in ``[0, 1]``.
        reasons: Heuristics that fired, kept for debugging and manual review.

    """

    text: str
    start: float
    end: float
    confidence: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


def questions_to_json(questions: list[Question]) -> str:
    """Serialise questions to the exported JSON format."""
    return json.dumps({"questions": [asdict(q) for q in questions]}, indent=2, ensure_ascii=False)


def write_questions(questions: list[Question], path: str | Path) -> None:
    """Write questions to ``path`` as JSON (UTF-8)."""
    Path(path).write_text(questions_to_json(questions), encoding="utf-8")


def read_segments(path: str | Path) -> list[TranscriptSegment]:
    """Read a transcript JSON file.

    Accepts either a bare list of segments or an object with a ``segments`` key,
    which is what most ASR engines emit.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = data["segments"] if isinstance(data, dict) else data
    return [TranscriptSegment.from_dict(s) for s in segments]

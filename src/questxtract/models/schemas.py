"""Core data types for QuestXtract."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


class SchemaError(ValueError):
    """Raised when JSON input does not match the expected schema."""


@dataclass(frozen=True, kw_only=True)
class Word:
    """A single recognised word with its own timestamps.

    Args:
        text: The word as transcribed.
        start: Start time in seconds from the beginning of the recording.
        end: End time in seconds from the beginning of the recording.

    """

    text: str
    start: float
    end: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Word:
        """Build a word from a plain dict; accepts Whisper's ``word`` key or ``text``."""
        try:
            text = data["word"] if "word" in data else data["text"]
            return cls(text=str(text).strip(), start=float(data["start"]), end=float(data["end"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"invalid word object: {exc}") from exc


@dataclass(frozen=True, kw_only=True)
class TranscriptSegment:
    """A chunk of transcribed speech with its position in the recording.

    Args:
        text: The transcribed text of the segment.
        start: Start time in seconds from the beginning of the recording.
        end: End time in seconds from the beginning of the recording.
        words: Word-level timestamps when the ASR engine provides them; empty otherwise.

    """

    text: str
    start: float
    end: float
    words: tuple[Word, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptSegment:
        """Build a segment from a plain dict (e.g. an ASR engine's JSON output)."""
        try:
            return cls(
                text=str(data["text"]),
                start=float(data["start"]),
                end=float(data["end"]),
                words=tuple(Word.from_dict(w) for w in data.get("words", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"invalid transcript segment: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready representation."""
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class Question:
    """A question detected in a transcript.

    Args:
        id: Stable identifier, derived from ``start`` and ``question`` when not given, so
            edits and re-runs of the same recording can be matched up.
        question: The question sentence as spoken.
        start: Start time in seconds of the segment the question was found in.
        end: End time in seconds of that segment.
        confidence: Detection confidence in ``[0, 1]``.
        source: Provenance of the detection (``"audio"`` for transcript-based detectors).
        reasons: Heuristics that fired, kept for debugging and manual review.

    """

    id: str = ""
    question: str
    start: float
    end: float
    confidence: float
    source: str = "audio"
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Fill in a deterministic ``id`` when none was given."""
        if not self.id:
            digest = hashlib.sha1(f"{self.start:.3f}|{self.question}".encode()).hexdigest()
            object.__setattr__(self, "id", digest[:12])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        """Build a question from its JSON representation."""
        try:
            return cls(
                id=str(data.get("id", "")),
                question=str(data["question"]),
                start=float(data["start"]),
                end=float(data["end"]),
                confidence=float(data["confidence"]),
                source=str(data.get("source", "audio")),
                reasons=tuple(str(r) for r in data.get("reasons", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"invalid question object: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready representation."""
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class LectureResult:
    """Everything extracted from one recording: the unit that is exported and edited.

    Args:
        lecture: Name of the recording (or transcript) the questions came from.
        questions: The detected questions, in recording order.

    """

    lecture: str
    questions: tuple[Question, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LectureResult:
        """Build a result from its JSON representation."""
        try:
            questions = data["questions"]
            if not isinstance(questions, list):
                raise TypeError("'questions' must be a list")
            return cls(
                lecture=str(data["lecture"]),
                questions=tuple(Question.from_dict(q) for q in questions),
            )
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"invalid lecture result: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready representation."""
        return asdict(self)

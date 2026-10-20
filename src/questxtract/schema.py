"""Issue #9: the flat internal Question schema.

This defines ``Question``: a flat, validated, serialisable shape for "one question."

**Not produced by the pipeline.** Nothing in the package builds a ``Question`` today; the
pipeline's output is ``questions.json``, owned by ``export.py``, whose ``review`` block is
where people edit. That format needs provenance (``id``, ``sentence_id``,
``answer_sentence_ids``) that this type deliberately omits. ``Question`` is a lighter internal
shape: useful anywhere in the pipeline that wants "a question with a timestamp and a
confidence" without the full export record's edit-tracking fields. If a use for it as a
public/API shape shows up later, build it *from* the export record rather than growing this
type to duplicate it.

Acceptance criteria this satisfies:
    1. schema is defined                -> the ``Question`` dataclass below
    2. schema supports timestamps       -> ``start`` / ``end``, validated ``end > start``
    3. schema supports confidence       -> ``confidence``, validated to ``[0, 1]``
    4. schema can be serialised to JSON -> ``to_dict`` / ``from_dict``
    5. schema is documented             -> this module's and the class's docstrings
    6. tests validate valid/invalid     -> see ``test_schema.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import SchemaError

#: Where a question was detected from. "combined" when the board on screen raised its
#: score; "visual" is reserved for questions read from the board alone.
_VALID_SOURCES = frozenset({"audio", "visual", "combined"})


@dataclass(frozen=True, kw_only=True)
class Question:
    """One extracted question, in the wire schema this issue defines.

    Every field the *base* schema promises is required and validated at construction
    time. The *extended* fields (``visual_context``, ``topic``, ``answer``) all default
    to their "not present yet" value, so old callers constructing only the base fields
    keep working once the schema grows - this is what "later can be extended to ..." in
    the issue means in practice.

    Args:
        question: The question text, verbatim from the transcript. Never invented -
            the pipeline only ever extracts spans that were actually said.
        start: Start time in seconds from the beginning of the recording. Must be >= 0.
        end: End time in seconds. Must be strictly greater than ``start``.
        confidence: Detection confidence in ``[0, 1]``.
        source: Where the question was detected. One of ``"audio"``, ``"visual"``,
            ``"combined"``; ``"visual"`` is not produced yet.
        visual_context: Board or slide context overlapping the question. Empty tuple
            until the vision stage (M2/M3) fills it in.
        topic: A topic or segment label for this question, or ``None`` until the
            segmentation-to-topic mapping exists.
        answer: The answer text if the pipeline found one, else ``None``. Absence of an
            answer is not an error: FR6/FR7 ask for questions with timestamps, not
            necessarily answered ones.

    Raises:
        SchemaError: At construction time, if any field violates the schema (see
            ``__post_init__``).

    """

    question: str
    start: float
    end: float
    confidence: float
    source: str = "audio"
    visual_context: tuple[Any, ...] = ()
    topic: str | None = None
    answer: str | None = None

    def __post_init__(self) -> None:
        """Validate every field's own invariant, collecting all violations at once.

        Raises:
            SchemaError: Listing every rule this object breaks, so a caller building
                a ``Question`` programmatically sees every problem in one error rather
                than fixing them one at a time.

        """
        errors: list[str] = []
        if not isinstance(self.question, str) or not self.question.strip():
            errors.append("'question' must be a non-empty string")
        if not isinstance(self.start, (int, float)) or self.start < 0:
            errors.append("'start' must be a number >= 0")
        if not isinstance(self.end, (int, float)) or self.end <= self.start:
            errors.append("'end' must be a number greater than 'start'")
        if not isinstance(self.confidence, (int, float)) or not 0.0 <= self.confidence <= 1.0:
            errors.append("'confidence' must be a number between 0 and 1")
        if self.source not in _VALID_SOURCES:
            errors.append(f"'source' must be one of {sorted(_VALID_SOURCES)}, got {self.source!r}")
        if self.topic is not None and not isinstance(self.topic, str):
            errors.append("'topic' must be a string or null")
        if self.answer is not None and not isinstance(self.answer, str):
            errors.append("'answer' must be a string or null")
        if errors:
            raise SchemaError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain, JSON-ready dict.

        Every documented key is always present, ``null`` included, matching the
        project's export convention (export.py): a consumer never has to defensively
        ``.get()`` a key that might be missing.
        """
        return {
            "question": self.question,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source": self.source,
            "visual_context": list(self.visual_context),
            "topic": self.topic,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        """Build a ``Question`` from a plain dict, e.g. one loaded from JSON.

        Accepts a dict with only the base fields (``question``, ``start``, ``end``,
        ``confidence``); the extended fields fall back to their defaults when absent,
        which is what makes the schema backward-compatible as it grows.

        Raises:
            SchemaError: If ``data`` is not a dict, a required field is missing, a
                field has the wrong type, or a value fails validation (delegated to
                ``__post_init__``).

        """
        if not isinstance(data, dict):
            raise SchemaError(f"expected an object, got {type(data).__name__}")
        try:
            return cls(
                question=str(data["question"]),
                start=float(data["start"]),
                end=float(data["end"]),
                confidence=float(data["confidence"]),
                source=str(data.get("source", "audio")),
                visual_context=tuple(data.get("visual_context") or ()),
                topic=data.get("topic"),
                answer=data.get("answer"),
            )
        except KeyError as exc:
            raise SchemaError(f"missing required field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"invalid question object: {exc}") from exc

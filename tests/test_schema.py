"""Tests for the issue #7 Question schema.

Covers: base-schema construction, extended-field round-tripping, JSON
serialisation, and rejection of every invalid shape the schema documents.
"""

from __future__ import annotations

import json

import pytest

from questxtract.models import SchemaError
from questxtract.schema import Question

# --- valid objects -----------------------------------------------------------


def test_base_schema_round_trips_through_dict() -> None:
    q = Question(question="What is a stable sort?", start=125.4, end=128.1, confidence=0.87)
    d = q.to_dict()

    assert d == {
        "question": "What is a stable sort?",
        "start": 125.4,
        "end": 128.1,
        "confidence": 0.87,
        "source": "audio",
        "visual_context": [],
        "topic": None,
        "answer": None,
    }
    assert Question.from_dict(d) == q


def test_base_schema_survives_actual_json_encode_decode() -> None:
    """Acceptance criterion 4: schema can be serialised to JSON (not just dicts)."""
    q = Question(question="Why does entropy increase?", start=10.0, end=12.5, confidence=0.6)
    blob = json.dumps(q.to_dict())
    restored = Question.from_dict(json.loads(blob))
    assert restored == q


def test_extended_fields_round_trip() -> None:
    q = Question(
        question="What does this term mean?",
        start=200.0,
        end=203.2,
        confidence=0.75,
        source="visual",
        visual_context=({"text": "S = k log W", "keyframe": "42.jpg"},),
        topic="thermodynamics",
        answer="A measure of disorder in the system.",
    )
    d = q.to_dict()
    assert d["source"] == "visual"
    assert d["visual_context"] == [{"text": "S = k log W", "keyframe": "42.jpg"}]
    assert d["topic"] == "thermodynamics"
    assert d["answer"] == "A measure of disorder in the system."
    assert Question.from_dict(d) == q


def test_from_dict_accepts_base_fields_only() -> None:
    """Backward compatibility: old producers sending only the base schema still work."""
    minimal = {"question": "Does anyone know why?", "start": 1.0, "end": 2.0, "confidence": 0.5}
    q = Question.from_dict(minimal)
    assert q.source == "audio"
    assert q.visual_context == ()
    assert q.topic is None
    assert q.answer is None


# --- invalid objects -----------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message_fragment"),
    [
        ({"question": ""}, "question"),
        ({"question": "   "}, "question"),
        ({"start": -1.0}, "start"),
        ({"start": 5.0, "end": 5.0}, "end"),
        ({"start": 5.0, "end": 4.0}, "end"),
        ({"confidence": 1.5}, "confidence"),
        ({"confidence": -0.01}, "confidence"),
        ({"source": "telepathy"}, "source"),
        ({"topic": 123}, "topic"),
        ({"answer": 456}, "answer"),
    ],
)
def test_invalid_field_raises_schema_error(
    overrides: dict[str, object], message_fragment: str
) -> None:
    base: dict[str, object] = {
        "question": "What is a stable sort?",
        "start": 1.0,
        "end": 3.0,
        "confidence": 0.5,
    }
    base.update(overrides)
    with pytest.raises(SchemaError, match=message_fragment):
        Question(**base)  # type: ignore[arg-type]


def test_multiple_violations_are_all_reported() -> None:
    with pytest.raises(SchemaError) as exc_info:
        Question(question="", start=-1.0, end=-2.0, confidence=5.0)
    message = str(exc_info.value)
    assert "question" in message
    assert "start" in message
    assert "end" in message
    assert "confidence" in message


def test_from_dict_rejects_non_dict_input() -> None:
    with pytest.raises(SchemaError, match="expected an object"):
        Question.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_from_dict_missing_required_field_raises() -> None:
    with pytest.raises(SchemaError, match="missing required field"):
        Question.from_dict({"question": "x", "start": 1.0, "end": 2.0})  # no confidence


def test_from_dict_wrong_type_raises() -> None:
    with pytest.raises(SchemaError, match="invalid question object"):
        Question.from_dict(
            {"question": "x", "start": "not-a-number", "end": 2.0, "confidence": 0.5}
        )


def test_question_is_immutable() -> None:
    """Frozen, matching every other type in models.py (architecture.md 5, D3)."""
    q = Question(question="x?", start=0.0, end=1.0, confidence=0.5)
    with pytest.raises(AttributeError):
        q.confidence = 0.9  # type: ignore[misc]

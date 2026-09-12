import pytest

from questxtract import LectureResult, Question, SchemaError


def test_question_id_is_deterministic():
    a = Question(question="What is mass?", start=1.5, end=3.0, confidence=1.0)
    b = Question(question="What is mass?", start=1.5, end=3.0, confidence=0.5)
    assert a.id == b.id
    assert a.id != Question(question="What is mass?", start=1.6, end=3.0, confidence=1.0).id


def test_fields_are_keyword_only():
    with pytest.raises(TypeError):
        Question("q", 0.0, 1.0, 0.5)  # type: ignore[misc]


def test_question_roundtrips_through_dict():
    q = Question(question="q", start=0.0, end=1.0, confidence=0.9, reasons=("mark",))
    assert Question.from_dict(q.to_dict()) == q


def test_lecture_result_rejects_bad_shapes():
    with pytest.raises(SchemaError):
        LectureResult.from_dict({"lecture": "x", "questions": {}})
    with pytest.raises(SchemaError):
        LectureResult.from_dict({"lecture": "x", "questions": [None]})
    with pytest.raises(SchemaError):
        LectureResult.from_dict({"lecture": "x", "questions": ["not a question"]})
    with pytest.raises(SchemaError):
        Question.from_dict({"question": "q"})

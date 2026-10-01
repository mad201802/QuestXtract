import pytest

from questxtract import SchemaError, questions_to_exerplaza


def _question(
    identifier: str,
    verdict: str,
    *,
    review: dict[str, object] | None = None,
) -> dict[str, object]:
    question: dict[str, object] = {
        "id": identifier,
        "text": f"original {identifier}",
        "start": 1.0,
        "end": 2.0,
        "confidence": 0.8,
        "answer": None,
        "question_type": "what",
        "verdict": verdict,
    }
    if review is not None:
        question["review"] = review
    return question


def _document() -> dict[str, object]:
    return {
        "source": {"path": "lecture.mp4", "hash": "video-1", "duration_seconds": 20.0},
        "segments": [],
        "visuals": [],
        "questions": [
            _question("q1", "unjudged"),
            _question("q2", "keep"),
            _question("q3", "revise", review={"status": "accepted", "text": "edited q3"}),
            _question("q4", "reject"),
            _question("q5", "keep", review={"status": "deleted"}),
            _question("q6", "keep", review={"status": "false_positive"}),
        ],
    }


def test_default_export_excludes_only_explicit_reviewer_rejections() -> None:
    result = questions_to_exerplaza(_document())
    exported = result["quiz"]["questions"]

    assert [question["id"] for question in exported] == ["q1", "q2", "q3", "q4"]
    assert exported[2]["question"] == "edited q3"


def test_only_keep_is_an_explicit_restrictive_filter() -> None:
    result = questions_to_exerplaza(_document(), only_keep=True)

    assert [question["id"] for question in result["quiz"]["questions"]] == ["q2"]


def test_missing_review_is_compatible() -> None:
    document = {"source": {"hash": "video-1"}, "questions": [_question("q1", "unjudged")]}

    assert len(questions_to_exerplaza(document)["quiz"]["questions"]) == 1


def test_video_id_is_required_without_source_hash() -> None:
    document = {"source": {"hash": None}, "questions": []}

    with pytest.raises(SchemaError, match="video_id is required"):
        questions_to_exerplaza(document)



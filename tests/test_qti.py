"""Tests for the QTI model and its ``questions.json`` mapping (issue #67)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
from jsonschema import Draft202012Validator

from questxtract.export import SCHEMA_PATH
from questxtract.qti import QtiExportError, questions_to_qti
from questxtract.qti_v3 import QTI_NS, write_qti_package

_Q = f"{{{QTI_NS}}}"


def _document(*questions: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "2.1",
        "config": {"language": "en"},
        "segments": [],
        "visuals": [],
        "questions": list(questions),
    }


def _question(identifier: str = "q_abc123", **overrides: Any) -> dict[str, Any]:
    value = {
        "id": identifier,
        "text": "What is 2 < 3 & 4 > 1?",
        "question_type": "what",
        "answer": None,
        "sentence_id": "s_abc123",
        "confidence": 0.82,
        "verdict": "keep",
    }
    value.update(overrides)
    return value


def _ids(document: dict[str, Any], **kwargs: Any) -> list[str | None]:
    return [
        item.provenance.source_question_id if item.provenance else None
        for item in questions_to_qti(document, **kwargs).items
    ]


def _item_root(document: dict[str, Any], tmp_path: Path, identifier: str) -> ElementTree.Element:
    output = tmp_path / "questions.zip"
    write_qti_package(questions_to_qti(document), output)
    with zipfile.ZipFile(output) as package:
        return ElementTree.fromstring(package.read(f"items/qti-{identifier}.xml"))


def test_empty_document_creates_empty_test() -> None:
    assert questions_to_qti(_document()).items == ()


def test_a_question_becomes_a_human_scored_text_entry_item() -> None:
    item = questions_to_qti(_document(_question())).items[0]
    assert item.identifier == "qti-q_abc123"
    assert item.linguistic_question_type == "what"
    assert item.interaction.interaction_type == "extendedTextInteraction"
    assert not item.has_answer_key
    assert item.options == ()
    assert item.points is None
    assert item.provenance is not None
    assert item.provenance.source_answer is None


def test_a_schema_valid_document_gives_extended_text_items(tmp_path: Path) -> None:
    review = {"status": "pending", "text": None, "note": None, "metadata": {}}
    question = {
        "id": "q_abc123",
        "text": "What is shown?",
        "original_text": None,
        "start": 1,
        "end": 2,
        "start_hms": "0:01",
        "end_hms": "0:02",
        "confidence": 0.8,
        "confidence_label": "high",
        "question_type": "what",
        "source": "audio",
        "answer": {"text": "A plot.", "start": 2, "end": 3, "sentence_ids": ["s_def456"]},
        "segment_id": None,
        "sentence_id": "s_abc123",
        "visual_ids": [],
        "verdict": "keep",
        "reasons": [],
        "review": review,
    }
    document = {
        "schema_version": "2.1",
        "source": {"path": "lecture.mp4", "duration_seconds": 10, "hash": None},
        "config": {
            "model_size": "turbo",
            "language": "en",
            "min_confidence": 0.5,
            "judge_model": None,
            "vision_model": None,
        },
        "stats": {},
        "segments": [],
        "visuals": [],
        "questions": [question, {**question, "id": "q_def456", "sentence_id": "s_def456"}],
    }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)

    items = questions_to_qti(document).items

    assert [item.interaction.interaction_type for item in items] == [
        "extendedTextInteraction",
        "extendedTextInteraction",
    ]
    assert items[0].provenance is not None
    assert items[0].provenance.source_answer == "A plot."
    root = _item_root(document, tmp_path, "q_abc123")
    assert root.find(f".//{_Q}qti-extended-text-interaction") is not None


def test_an_open_answer_record_gives_no_answer_key(tmp_path: Path) -> None:
    question = _question(correct_response={"text": "Because it is stable.", "option_ids": []})
    root = _item_root(_document(question), tmp_path, "q_abc123")
    assert root.find(f".//{_Q}qti-correct-response") is None
    assert root.find(f"{_Q}qti-response-processing") is None


def test_quiz_keys_on_a_question_are_ignored(tmp_path: Path) -> None:
    question = _question(
        interaction_type="choice_single",
        options=[{"id": "a", "text": "A"}],
        correct_response={"text": None, "option_ids": ["a", "b"]},
        feedback={"correct": 1},
        points=5,
        visual={"latex": r"\alpha"},
        visual_ids=7,
    )
    root = _item_root(_document(question), tmp_path, "q_abc123")
    assert root.find(f".//{_Q}qti-extended-text-interaction") is not None
    assert root.find(f".//{_Q}qti-correct-response") is None
    assert root.find(f"{_Q}qti-response-processing") is None
    assert root.find(f"{_Q}qti-modal-feedback") is None


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"text": "What is\x07 this?"}, r"questions\[0\]\.text"),
        ({"text": "Lone \ud800 surrogate?"}, r"questions\[0\]\.text"),
        ({"review": {"status": "pending", "text": "Edited\x0c?"}}, r"questions\[0\]\.review\.text"),
    ],
)
def test_a_character_illegal_in_xml_fails_naming_the_field(
    overrides: dict[str, Any], field: str
) -> None:
    with pytest.raises(QtiExportError, match=f"{field}: character U\\+"):
        questions_to_qti(_document(_question(**overrides)))


def test_tabs_and_newlines_are_allowed() -> None:
    item = questions_to_qti(_document(_question(text="What is\tthis,\nreally?"))).items[0]
    assert item.prompt == "What is\tthis,\nreally?"


def test_default_export_includes_all_unrejected_verdicts() -> None:
    document = _document(
        _question("q_keep"),
        _question("q_revise", verdict="revise"),
        _question("q_reject", verdict="reject"),
        _question("q_unjudged", verdict="unjudged"),
    )
    assert _ids(document) == ["q_keep", "q_revise", "q_reject", "q_unjudged"]


def test_default_export_excludes_reviewer_rejections() -> None:
    document = _document(
        _question("q_deleted", review={"status": "deleted"}),
        _question("q_false_positive", review={"status": "false_positive"}),
        _question("q_accepted", review={"status": "accepted"}),
    )
    assert _ids(document) == ["q_accepted"]


def test_only_keep_exports_only_keep_verdicts() -> None:
    document = _document(
        _question("q_keep", verdict="keep"),
        _question("q_revise", verdict="revise"),
        _question("q_reject", verdict="reject"),
        _question("q_unjudged", verdict="unjudged"),
        _question("q_deleted", verdict="keep", review={"status": "deleted"}),
    )
    assert _ids(document, only_keep=True) == ["q_keep"]


def test_missing_review_and_orphaned_reviews_are_accepted() -> None:
    document = _document(_question(review=None))
    document["orphaned_reviews"] = [{"id": "q_old", "review": {"status": "accepted"}}]
    assert _ids(document) == ["q_abc123"]


def test_revise_review_text_is_exported() -> None:
    review = {"status": "pending", "text": "Edited wording", "note": None, "metadata": {}}
    item = questions_to_qti(_document(_question(verdict="revise", review=review))).items[0]
    assert item.prompt == "Edited wording"


def test_empty_review_text_keeps_the_spoken_text() -> None:
    review = {"status": "pending", "text": "  ", "note": None, "metadata": {}}
    item = questions_to_qti(_document(_question(review=review))).items[0]
    assert item.prompt == "What is 2 < 3 & 4 > 1?"


def test_invalid_questions_fail_readably() -> None:
    with pytest.raises(QtiExportError, match="invalid XML identifier"):
        questions_to_qti(_document(_question("bad id")))
    with pytest.raises(QtiExportError, match="duplicate question identifier"):
        questions_to_qti(_document(_question(), _question()))
    with pytest.raises(QtiExportError, match=r"text: expected a non-empty string"):
        questions_to_qti(_document(_question(text=" ")))
    with pytest.raises(QtiExportError, match="'questions' list"):
        questions_to_qti({"questions": None})


def test_identifiers_are_stable() -> None:
    document = _document(_question())
    assert questions_to_qti(document) == questions_to_qti(document)


def test_special_characters_are_escaped_and_read_back(tmp_path: Path) -> None:
    root = _item_root(_document(_question()), tmp_path, "q_abc123")
    prompt = root.find(f".//{_Q}qti-prompt")
    assert prompt is not None
    assert prompt.text == "What is 2 < 3 & 4 > 1?"

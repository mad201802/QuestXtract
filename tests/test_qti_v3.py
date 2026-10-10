"""Tests for the QTI 3.0 package writer (issue #67)."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

from questxtract.qti import (
    AssessmentItem,
    AssessmentOption,
    AssessmentTest,
    Feedback,
    Interaction,
    QtiExportError,
    ResponseDeclaration,
    questions_to_qti,
)
from questxtract.qti_v3 import CP_NS, QTI_NS, export_qti, write_qti_package

_Q = f"{{{QTI_NS}}}"
_CP = f"{{{CP_NS}}}"


def _questions_file(tmp_path: Path) -> Path:
    question = {
        "id": "q_abc123",
        "text": "What is a stable sort?",
        "question_type": "what",
        "answer": None,
        "sentence_id": "s_abc123",
        "confidence": 0.82,
        "verdict": "keep",
    }
    document = {
        "schema_version": "2.1",
        "config": {"language": "en"},
        "segments": [],
        "visuals": [],
        "questions": [question, {**question, "id": "q_def456", "text": "Why?"}],
    }
    path = tmp_path / "lecture.questions.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _choice_item(**overrides: Any) -> AssessmentItem:
    value: dict[str, Any] = {
        "identifier": "qti-q_choice",
        "title": "Question q_choice",
        "prompt": "Which sort is stable?",
        "language": "en",
        "linguistic_question_type": "what",
        "response_declarations": (
            ResponseDeclaration(
                identifier="RESPONSE",
                base_type="identifier",
                cardinality="single",
                correct_values=("a",),
            ),
        ),
        "interaction": Interaction(
            interaction_type="choiceInteraction",
            response_identifier="RESPONSE",
            min_choices=1,
            max_choices=1,
        ),
        "options": (
            AssessmentOption(identifier="a", text="Merge sort"),
            AssessmentOption(identifier="b", text="Quicksort"),
        ),
        "points": 5,
    }
    value.update(overrides)
    return AssessmentItem(**value)


def _package(tmp_path: Path, *items: AssessmentItem) -> zipfile.ZipFile:
    output = tmp_path / "package.zip"
    write_qti_package(AssessmentTest(identifier="t", title="T", items=items), output)
    return zipfile.ZipFile(output)


def _item(tmp_path: Path, item: AssessmentItem) -> ElementTree.Element:
    with _package(tmp_path, item) as package:
        return ElementTree.fromstring(package.read(f"items/{item.identifier}.xml"))


def _outcome(root: ElementTree.Element, identifier: str) -> ElementTree.Element:
    node = root.find(f"{_Q}qti-outcome-declaration[@identifier='{identifier}']")
    assert node is not None
    return node


def _default(outcome: ElementTree.Element) -> str | None:
    return outcome.findtext(f"{_Q}qti-default-value/{_Q}qti-value")


def test_two_exports_at_different_times_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _questions_file(tmp_path)
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    export_qti(source, tmp_path / "a.zip")
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    export_qti(source, tmp_path / "b.zip")
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()


def test_every_entry_has_the_fixed_date_and_is_deflated(tmp_path: Path) -> None:
    export_qti(_questions_file(tmp_path), tmp_path / "out.zip")
    with zipfile.ZipFile(tmp_path / "out.zip") as package:
        infos = package.infolist()
    assert [info.filename for info in infos] == [
        "items/qti-q_abc123.xml",
        "items/qti-q_def456.xml",
        "assessmentTest.xml",
        "imsmanifest.xml",
        "metadata/questxtract.json",
    ]
    for info in infos:
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.compress_type == zipfile.ZIP_DEFLATED
        assert info.create_system == 3
        assert info.external_attr >> 16 == 0o100644


def test_xml_uses_the_default_namespace_and_declares_its_schema(tmp_path: Path) -> None:
    export_qti(_questions_file(tmp_path), tmp_path / "out.zip")
    with zipfile.ZipFile(tmp_path / "out.zip") as package:
        raw = {name: package.read(name).decode("utf-8") for name in package.namelist()}
    for name in ("items/qti-q_abc123.xml", "assessmentTest.xml"):
        assert f'xmlns="{QTI_NS}"' in raw[name]
        assert f'xsi:schemaLocation="{QTI_NS} https://purl.imsglobal.org/' in raw[name]
    assert f'xmlns="{CP_NS}"' in raw["imsmanifest.xml"]
    assert f'xsi:schemaLocation="{CP_NS} https://purl.imsglobal.org/' in raw["imsmanifest.xml"]
    assert 'xml:lang="en"' in raw["items/qti-q_abc123.xml"]
    assert "xml:lang" not in raw["assessmentTest.xml"]
    for name, text in raw.items():
        if name.endswith(".xml"):
            for prefix in ("<qti:", "<cp:", "ns0:"):
                assert prefix not in text, (name, prefix)


def test_every_xml_entry_parses(tmp_path: Path) -> None:
    item = _choice_item(feedback=Feedback(correct="Yes.", incorrect="No.", general="See 3.2."))
    with _package(tmp_path, item, _choice_item(identifier="qti-q_other")) as package:
        for name in package.namelist():
            if name.endswith(".xml"):
                ElementTree.fromstring(package.read(name))


def test_points_set_the_maximum_and_a_correct_answer_scores_it(tmp_path: Path) -> None:
    root = _item(tmp_path, _choice_item())
    maximum = _outcome(root, "MAXSCORE")
    assert maximum.get("base-type") == "float"
    assert _default(maximum) == "5.0"
    score = _outcome(root, "SCORE")
    assert score.get("normal-maximum") == "5.0"
    assert score.get("normal-minimum") == "0.0"
    assert score.get("external-scored") is None
    assert _default(score) == "0.0"
    interaction = root.find(f".//{_Q}qti-choice-interaction")
    assert interaction is not None
    assert (interaction.get("min-choices"), interaction.get("max-choices")) == ("1", "1")

    processing = root.find(f"{_Q}qti-response-processing")
    assert processing is not None
    assert processing.get("template") is None
    response_if = processing.find(f"{_Q}qti-response-condition/{_Q}qti-response-if")
    assert response_if is not None
    assert response_if.find(f"{_Q}qti-match/{_Q}qti-correct[@identifier='RESPONSE']") is not None
    on_correct = response_if.find(f"{_Q}qti-set-outcome-value[@identifier='SCORE']")
    assert on_correct is not None
    assert on_correct.find(f"{_Q}qti-variable[@identifier='MAXSCORE']") is not None
    on_wrong = processing.find(
        f".//{_Q}qti-response-else/{_Q}qti-set-outcome-value[@identifier='SCORE']"
    )
    assert on_wrong is not None
    assert on_wrong.findtext(f"{_Q}qti-base-value") == "0.0"
    assert root.find(f".//{_Q}qti-set-outcome-value[@identifier='FEEDBACK']") is None


def test_an_item_without_points_is_worth_one(tmp_path: Path) -> None:
    root = _item(tmp_path, _choice_item(points=None))
    assert _default(_outcome(root, "MAXSCORE")) == "1.0"
    assert _outcome(root, "SCORE").get("normal-maximum") is None


def test_feedback_follows_the_response_and_comes_after_processing(tmp_path: Path) -> None:
    root = _item(tmp_path, _choice_item(feedback=Feedback(correct="Yes.", incorrect="No.")))
    feedback_outcome = _outcome(root, "FEEDBACK")
    assert feedback_outcome.get("base-type") == "identifier"
    processing = root.find(f"{_Q}qti-response-processing")
    assert processing is not None
    branches = {
        branch.tag.removeprefix(_Q): branch.findtext(
            f"{_Q}qti-set-outcome-value[@identifier='FEEDBACK']/{_Q}qti-base-value"
        )
        for branch in processing.iter()
        if branch.tag in {f"{_Q}qti-response-if", f"{_Q}qti-response-else"}
    }
    assert branches == {"qti-response-if": "correct", "qti-response-else": "incorrect"}

    children = [child.tag.removeprefix(_Q) for child in root]
    assert children.index("qti-item-body") < children.index("qti-response-processing")
    assert children.index("qti-response-processing") < children.index("qti-modal-feedback")
    modal = root.findall(f"{_Q}qti-modal-feedback")
    assert [node.get("identifier") for node in modal] == ["correct", "incorrect"]
    assert {node.get("outcome-identifier") for node in modal} == {"FEEDBACK"}
    assert {node.get("show-hide") for node in modal} == {"show"}
    assert [node.findtext(f"{_Q}qti-content-body") for node in modal] == ["Yes.", "No."]


def test_an_item_without_an_answer_key_is_scored_by_a_person(tmp_path: Path) -> None:
    item = questions_to_qti(json.loads(_questions_file(tmp_path).read_text("utf-8"))).items[0]
    root = _item(tmp_path, item)
    assert _outcome(root, "SCORE").get("external-scored") == "human"
    assert _default(_outcome(root, "MAXSCORE")) == "1.0"
    assert root.find(f"{_Q}qti-response-processing") is None
    assert root.find(f".//{_Q}qti-correct-response") is None


def test_the_manifest_lists_every_file_and_its_references_resolve(tmp_path: Path) -> None:
    export_qti(_questions_file(tmp_path), tmp_path / "out.zip")
    with zipfile.ZipFile(tmp_path / "out.zip") as package:
        names = set(package.namelist())
        manifest = ElementTree.fromstring(package.read("imsmanifest.xml"))
        metadata = json.loads(package.read("metadata/questxtract.json"))
    assert manifest.findtext(f"{_CP}metadata/{_CP}schemaversion") == "3.0.0"
    resources = manifest.findall(f"{_CP}resources/{_CP}resource")
    files = {node.get("href") for node in manifest.iter(f"{_CP}file")}
    assert files == names - {"imsmanifest.xml"}
    identifiers = {node.get("identifier") for node in resources}
    references = {node.get("identifierref") for node in manifest.iter(f"{_CP}dependency")}
    assert references
    assert references <= identifiers
    sidecar = manifest.find(f"{_CP}resources/{_CP}resource[@identifier='questxtract-metadata']")
    assert sidecar is not None
    assert (sidecar.get("type"), sidecar.get("href")) == ("extension", "metadata/questxtract.json")
    assert metadata["items"]["qti-q_abc123"] == {
        "source_question_id": "q_abc123",
        "source_sentence_id": "s_abc123",
        "linguistic_question_type": "what",
        "interaction_type": "extendedTextInteraction",
        "source_answer": None,
        "metadata": {"confidence": "0.82"},
    }


def test_an_output_in_a_missing_folder_fails_readably(tmp_path: Path) -> None:
    with pytest.raises(QtiExportError, match="cannot write the QTI package"):
        export_qti(_questions_file(tmp_path), tmp_path / "missing" / "out.zip")


def test_an_unreadable_questions_file_fails_readably(tmp_path: Path) -> None:
    with pytest.raises(QtiExportError, match="missing.json"):
        export_qti(tmp_path / "missing.json", tmp_path / "out.zip")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(QtiExportError, match="not valid JSON"):
        export_qti(bad, tmp_path / "out.zip")

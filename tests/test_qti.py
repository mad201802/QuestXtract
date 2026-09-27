import json
import zipfile
from xml.etree import ElementTree

import pytest

from questxtract.cli import main
from questxtract.qti import QTI_NS, QtiExportError, questions_to_qti, write_qti_package


def _document(*questions):
    return {"schema_version": "1.1", "config": {"language": "en"}, "questions": list(questions)}


def _question(identifier="q_abc123", **overrides):
    value = {
        "id": identifier,
        "text": "What is 2 < 3 & 4 > 1?",
        "question_type": "what",
        "answer": None,
        "sentence_id": "s_abc123",
        "confidence": 0.82,
        "verdict": "keep",
        "visual": {"latex": r"\alpha < \beta"},
    }
    value.update(overrides)
    return value


def test_empty_document_creates_empty_test():
    assessment = questions_to_qti(_document())
    assert assessment.items == ()


def test_keep_question_is_free_text_and_not_automatically_scored():
    assessment = questions_to_qti(_document(_question()))
    item = assessment.items[0]
    assert item.identifier == "qti-q_abc123"
    assert item.linguistic_question_type == "what"
    assert item.interaction_type == "extendedTextInteraction"
    assert item.correct_response is None
    assert item.options == ()
    assert item.source_answer is None


def test_revise_and_reject_are_not_exported():
    assessment = questions_to_qti(
        _document(
            _question("q_keep"),
            _question("q_revise", verdict="revise"),
            _question("q_reject", verdict="reject"),
        )
    )
    assert [item.source_question_id for item in assessment.items] == ["q_keep"]


def test_unjudged_requires_explicit_opt_in():
    document = _document(_question(verdict="unjudged"))
    assert questions_to_qti(document).items == ()
    assert len(questions_to_qti(document, include_unjudged=True).items) == 1


def test_unsupported_identifier_fails():
    with pytest.raises(QtiExportError, match="invalid XML identifier"):
        questions_to_qti(_document(_question("bad id")))


def test_unsupported_objective_fields_fail():
    with pytest.raises(QtiExportError, match="choice options"):
        questions_to_qti(_document(_question(options=[{"identifier": "a", "text": "A"}])))
    with pytest.raises(QtiExportError, match="correct responses"):
        questions_to_qti(_document(_question(correct_response="yes")))


def test_identifiers_are_stable():
    document = _document(_question())
    first = questions_to_qti(document).items[0].identifier
    second = questions_to_qti(document).items[0].identifier
    assert first == second


def test_package_has_coherent_xml_and_manifest(tmp_path):
    output = tmp_path / "questions.zip"
    write_qti_package(_document(_question(), _question("q_def456")), output)
    with zipfile.ZipFile(output) as package:
        names = set(package.namelist())
        assert names == {
            "imsmanifest.xml",
            "assessmentTest.xml",
            "metadata/questxtract.json",
            "items/qti-q_abc123.xml",
            "items/qti-q_def456.xml",
        }
        item_root = ElementTree.fromstring(package.read("items/qti-q_abc123.xml"))
        assert item_root.tag == f"{{{QTI_NS}}}qti-assessment-item"
        assert item_root.find(f"{{{QTI_NS}}}qti-response-declaration") is not None
        assert item_root.find(f".//{{{QTI_NS}}}qti-extended-text-interaction") is not None
        assert item_root.find(f".//{{{QTI_NS}}}qti-correct-response") is None
        test_root = ElementTree.fromstring(package.read("assessmentTest.xml"))
        refs = test_root.findall(f".//{{{QTI_NS}}}qti-assessment-item-ref")
        assert len(refs) == 2
        manifest_root = ElementTree.fromstring(package.read("imsmanifest.xml"))
        assert manifest_root.tag.endswith("manifest")
        resources = list(manifest_root.iter())
        assert sum(element.tag.endswith("resource") for element in resources) == 3


def test_latex_and_special_characters_are_preserved(tmp_path):
    output = tmp_path / "questions.zip"
    write_qti_package(_document(_question()), output)
    with zipfile.ZipFile(output) as package:
        root = ElementTree.fromstring(package.read("items/qti-q_abc123.xml"))
        paragraph = root.find(".//p")
        assert paragraph is not None
        assert paragraph.text == "What is 2 < 3 & 4 > 1?"
        metadata = json.loads(package.read("metadata/questxtract.json"))
        assert metadata["items"]["qti-q_abc123"]["latex"] == r"\alpha < \beta"


def test_cli_exports_qti_and_reports_invalid_input(tmp_path, capsys):
    source = tmp_path / "questions.json"
    output = tmp_path / "questions.zip"
    source.write_text(json.dumps(_document(_question())), encoding="utf-8")
    assert main(["qti", str(source), "-o", str(output)]) == 0
    assert output.exists()
    assert "QTI package" in capsys.readouterr().err

    assert main(["qti", str(tmp_path / "missing.json"), "-o", str(output)]) == 1
    assert capsys.readouterr().err.startswith("error: qti: ")

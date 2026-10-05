import json
import zipfile
from xml.etree import ElementTree

import pytest

from questxtract.cli import main
from questxtract.qti import QTI_NS, QtiExportError, questions_to_qti, write_qti_package
from questxtract.qti_v21 import QTI_NS as QTI_21_NS
from questxtract.qti_v21 import write_qti_package as write_qti_21_package


def _document(*questions):
    return {
        "schema_version": "2.0",
        "config": {"language": "en"},
        "segments": [],
        "visuals": [],
        "questions": list(questions),
    }


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


def _choice_question(identifier="q_choice", interaction_type="choice_single", **overrides):
    value = _question(
        identifier,
        interaction_type=interaction_type,
        options=[{"id": "a", "text": "A"}, {"id": "b", "text": "B"}],
        correct_response={"text": None, "option_ids": ["a"]},
        points=1,
    )
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


def test_default_export_includes_all_unrejected_verdicts():
    assessment = questions_to_qti(
        _document(
            _question("q_keep"),
            _question("q_revise", verdict="revise"),
            _question("q_reject", verdict="reject"),
        )
    )
    assert [item.source_question_id for item in assessment.items] == [
        "q_keep",
        "q_revise",
        "q_reject",
    ]


def test_default_export_includes_unjudged():
    assert len(questions_to_qti(_document(_question(verdict="unjudged"))).items) == 1


def test_default_export_excludes_reviewer_rejections():
    document = _document(
        _question("q_deleted", review={"status": "deleted"}),
        _question("q_false_positive", review={"status": "false_positive"}),
        _question("q_accepted", review={"status": "accepted"}),
    )
    assert [item.source_question_id for item in questions_to_qti(document).items] == ["q_accepted"]


def test_only_keep_exports_only_keep_verdicts():
    document = _document(
        _question("q_keep", verdict="keep"),
        _question("q_revise", verdict="revise"),
        _question("q_reject", verdict="reject"),
        _question("q_unjudged", verdict="unjudged"),
        _question("q_deleted", verdict="keep", review={"status": "deleted"}),
    )
    assert [
        item.source_question_id for item in questions_to_qti(document, only_keep=True).items
    ] == ["q_keep"]


def test_missing_review_and_orphaned_reviews_are_compatible():
    document = _document(_question(review=None))
    document["orphaned_reviews"] = [{"id": "q_old", "review": {"status": "accepted"}}]
    assert [item.source_question_id for item in questions_to_qti(document).items] == ["q_abc123"]


def test_unsupported_identifier_fails():
    with pytest.raises(QtiExportError, match="invalid XML identifier"):
        questions_to_qti(_document(_question("bad id")))


def test_supported_choice_profiles_build_answer_keys_and_points():
    single = questions_to_qti(_document(_choice_question()))
    multiple = questions_to_qti(
        _document(
            _choice_question(
                "q_multiple",
                interaction_type="choice_multiple",
                correct_response={"text": None, "option_ids": ["a", "b"]},
            )
        )
    )
    assert single.items[0].interaction.min_choices == 1
    assert single.items[0].interaction.max_choices == 1
    assert single.items[0].correct_response == "a"
    assert single.items[0].score == 1
    assert multiple.items[0].interaction.min_choices == 1
    assert multiple.items[0].interaction.max_choices is None
    assert multiple.items[0].response_declarations[0].correct_values == ("a", "b")


def test_unsupported_interaction_and_inconsistent_choice_data_fail():
    with pytest.raises(QtiExportError, match="unsupported value"):
        questions_to_qti(_document(_question(interaction_type="numeric")))
    with pytest.raises(QtiExportError, match="at least two options"):
        questions_to_qti(_document(_question(interaction_type="choice_single")))
    with pytest.raises(QtiExportError, match="does not support choice options"):
        questions_to_qti(_document(_question(options=[{"id": "a", "text": "A"}])))
    with pytest.raises(QtiExportError, match="expected at least one option"):
        questions_to_qti(
            _document(_choice_question(correct_response={"text": None, "option_ids": []}))
        )
    with pytest.raises(QtiExportError, match="accepts one option"):
        questions_to_qti(
            _document(_choice_question(correct_response={"text": None, "option_ids": ["a", "b"]}))
        )


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


def test_qti_21_package_has_the_same_complete_layout(tmp_path):
    output = tmp_path / "questions-2.1.zip"
    write_qti_21_package(_document(_question(), _question("q_def456")), output)
    with zipfile.ZipFile(output) as package:
        assert set(package.namelist()) == {
            "imsmanifest.xml",
            "assessmentTest.xml",
            "metadata/questxtract.json",
            "items/qti-q_abc123.xml",
            "items/qti-q_def456.xml",
        }
        item_root = ElementTree.fromstring(package.read("items/qti-q_abc123.xml"))
        assert item_root.tag == f"{{{QTI_21_NS}}}assessmentItem"
        assert item_root.find(f"{{{QTI_21_NS}}}responseDeclaration") is not None
        manifest_root = ElementTree.fromstring(package.read("imsmanifest.xml"))
        resources = list(manifest_root.iter())
        assert sum(element.tag.endswith("resource") for element in resources) == 3
        metadata = json.loads(package.read("metadata/questxtract.json"))
        assert metadata["format"] == "questxtract-qti-2.1"


def test_cli_exports_qti_and_reports_invalid_input(tmp_path, capsys):
    source = tmp_path / "questions.json"
    output = tmp_path / "questions.zip"
    source.write_text(json.dumps(_document(_question())), encoding="utf-8")
    assert main(["qti", str(source), "-o", str(output)]) == 0
    assert output.exists()
    assert "QTI 3.0 package" in capsys.readouterr().err
    strict_output = tmp_path / "strict.zip"
    source.write_text(
        json.dumps(
            _document(
                _question("q_keep", verdict="keep"),
                _question("q_revise", verdict="revise"),
            )
        ),
        encoding="utf-8",
    )
    assert main(["qti", str(source), "-o", str(strict_output), "--only-keep"]) == 0
    capsys.readouterr()
    with zipfile.ZipFile(strict_output) as package:
        assert "items/qti-q_keep.xml" in package.namelist()
        assert "items/qti-q_revise.xml" not in package.namelist()

    assert main(["qti", str(tmp_path / "missing.json"), "-o", str(output)]) == 1
    assert capsys.readouterr().err.startswith("error: qti: ")


def test_cli_can_select_qti_21(tmp_path):
    source = tmp_path / "questions.json"
    output = tmp_path / "questions-2.1.zip"
    source.write_text(json.dumps(_document(_question())), encoding="utf-8")

    assert main(["qti", str(source), "-o", str(output), "--qti-version", "2.1"]) == 0

    with zipfile.ZipFile(output) as package:
        root = ElementTree.fromstring(package.read("items/qti-q_abc123.xml"))
    assert root.tag == f"{{{QTI_21_NS}}}assessmentItem"

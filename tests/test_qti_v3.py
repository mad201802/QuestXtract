"""Tests for the QTI 3.0 package writer (issue #67).

Every package written here is validated against the official 1EdTech XSDs, from the local
mirror `scripts/fetch_qti_xsd.py` downloads; the tests never reach the network. Without the
mirror they skip, unless QUESTXTRACT_REQUIRE_QTI_XSD is set, as in CI, where they fail.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
import xmlschema
from jsonschema import Draft202012Validator
from scripts.fetch_qti_xsd import ASI_XSD, CP_XSD, missing, uri_mapper, xsd_dir

from questxtract.cli import main
from questxtract.export import read_questions
from questxtract.qti import QtiExportError
from questxtract.qti_package import export_qti
from questxtract.qti_v3 import CP_NS, QTI_NS
from questxtract.quiz_schema import QUIZ_SCHEMA_PATH, read_quiz

_Q = f"{{{QTI_NS}}}"
_CP = f"{{{CP_NS}}}"
_DATA = Path(__file__).parent / "data"

MakeQuiz = Callable[..., dict[str, Any]]
Schemas = tuple[xmlschema.XMLSchema, xmlschema.XMLSchema]


@pytest.fixture(scope="session")
def schemas() -> Schemas:
    """The item-and-test schema and the content-package schema, compiled once, offline."""
    root = xsd_dir()
    if missing(root):
        message = f"QTI 3.0 XSDs not in {root}: run `uv run python scripts/fetch_qti_xsd.py`"
        if os.environ.get("QUESTXTRACT_REQUIRE_QTI_XSD"):
            pytest.fail(message)
        pytest.skip(message)
    mapper = uri_mapper(root)
    asi, cp = (
        xmlschema.XMLSchema(mapper(url), uri_mapper=mapper, allow="local")
        for url in (ASI_XSD, CP_XSD)
    )
    return asi, cp


def _export(tmp_path: Path, quiz: dict[str, Any], name: str = "out.zip") -> dict[str, bytes]:
    source = tmp_path / "quiz.json"
    source.write_text(json.dumps(quiz), encoding="utf-8")
    export_qti(source, tmp_path / name)
    with zipfile.ZipFile(tmp_path / name) as package:
        return {entry: package.read(entry) for entry in package.namelist()}


def _assert_conformant(package: dict[str, bytes], schemas: Schemas) -> None:
    """Validate every XML entry against its XSD, then check what an XSD cannot express."""
    asi, cp = schemas
    for name, data in package.items():
        if name.endswith(".xml"):
            schema = cp if name == "imsmanifest.xml" else asi
            errors = [str(error).splitlines()[0] for error in schema.iter_errors(io.BytesIO(data))]
            assert not errors, (name, errors)

    names = set(package)
    manifest = ElementTree.fromstring(package["imsmanifest.xml"])
    resources = manifest.findall(f"{_CP}resources/{_CP}resource")
    identifiers = [node.get("identifier") for node in resources]
    assert len(identifiers) == len(set(identifiers))
    assert {node.get("href") for node in resources} <= names
    assert {node.get("href") for node in manifest.iter(f"{_CP}file")} == names - {"imsmanifest.xml"}
    assert {node.get("identifierref") for node in manifest.iter(f"{_CP}dependency")} <= set(
        identifiers
    )

    test = ElementTree.fromstring(package["assessmentTest.xml"])
    references = list(test.iter(f"{_Q}qti-assessment-item-ref"))
    assert {node.get("href") for node in references} <= names
    assert len({node.get("identifier") for node in references}) == len(references)

    for name in names:
        if not name.startswith("items/"):
            continue
        item = ElementTree.fromstring(package[name])
        assert f"items/{item.get('identifier')}.xml" == name
        declared = {node.get("identifier") for node in item.iter(f"{_Q}qti-response-declaration")}
        outcomes = {node.get("identifier") for node in item.iter(f"{_Q}qti-outcome-declaration")}
        choices = {node.get("identifier") for node in item.iter(f"{_Q}qti-simple-choice")}
        interaction = item.find(f".//{_Q}qti-choice-interaction")
        assert interaction is not None
        assert interaction.get("response-identifier") in declared
        correct = item.findall(f".//{_Q}qti-correct-response/{_Q}qti-value")
        assert correct
        assert {node.text for node in correct} <= choices
        for modal in item.iter(f"{_Q}qti-modal-feedback"):
            assert modal.get("identifier") in choices
            assert modal.get("outcome-identifier") in outcomes


def _item(package: dict[str, bytes], identifier: str = "quiz_q_aaa") -> ElementTree.Element:
    return ElementTree.fromstring(package[f"items/{identifier}.xml"])


def _outcome(root: ElementTree.Element, identifier: str) -> ElementTree.Element:
    node = root.find(f"{_Q}qti-outcome-declaration[@identifier='{identifier}']")
    assert node is not None
    return node


def _default(outcome: ElementTree.Element) -> str | None:
    return outcome.findtext(f"{_Q}qti-default-value/{_Q}qti-value")


@pytest.mark.parametrize("source_ids", [("q_aaa", "q_bbb"), ()])
def test_a_quiz_gives_a_package_the_xsds_accept(
    tmp_path: Path, make_quiz: MakeQuiz, schemas: Schemas, source_ids: tuple[str, ...]
) -> None:
    _assert_conformant(_export(tmp_path, make_quiz(*source_ids)), schemas)


def test_every_instruction_still_gives_a_conformant_package(
    tmp_path: Path, make_quiz: MakeQuiz, schemas: Schemas
) -> None:
    quiz = make_quiz("q_aaa", "q_bbb")
    quiz["instructions"] = {
        "text": "Pick one answer.",
        "shuffle_questions": True,
        "shuffle_options": True,
        "show_feedback": False,
    }
    quiz["questions"][0]["points"] = 0
    _assert_conformant(_export(tmp_path, quiz), schemas)


def test_a_quiz_on_a_real_export_gives_a_conformant_package(
    tmp_path: Path, make_quiz: MakeQuiz, schemas: Schemas
) -> None:
    """process on a committed transcript, a quiz on its questions, then qti, all through the CLI.

    The quiz is what the quiz command (#79) is meant to write: one item per question, the stem
    spoken as it was said, the question's own times. It passes the schema and read_quiz with
    the questions file, so the rules on source questions and spoken stems run too.
    """
    transcript = _DATA / "ttk4225_2026_09_14.transcript.json"
    assert main(["process", str(transcript), "-o", "lecture.questions.json"]) == 0
    questions = read_questions(tmp_path / "lecture.questions.json")
    asked = questions["questions"]
    assert len(asked) > 10

    quiz = make_quiz(*[question["id"] for question in asked])
    for item, question in zip(quiz["questions"], asked, strict=True):
        item["prompt"]["text"] = question["original_text"] or question["text"]
        item["provenance"].update(start=question["start"], end=question["end"])
    quiz["questions"][1]["review"]["status"] = "rejected"
    Draft202012Validator(json.loads(QUIZ_SCHEMA_PATH.read_text(encoding="utf-8"))).validate(quiz)
    (tmp_path / "lecture.quiz.json").write_text(json.dumps(quiz), encoding="utf-8")
    read_quiz(tmp_path / "lecture.quiz.json", questions)

    assert main(["qti", "lecture.quiz.json", "-o", "lecture.zip"]) == 0

    with zipfile.ZipFile(tmp_path / "lecture.zip") as package:
        contents = {name: package.read(name) for name in package.namelist()}
    _assert_conformant(contents, schemas)
    kept = [question for index, question in enumerate(asked) if index != 1]
    assert [name for name in contents if name.startswith("items/")] == [
        f"items/quiz_{question['id']}.xml" for question in kept
    ]
    metadata = json.loads(contents["metadata/questxtract.json"])
    for question in kept:
        item = metadata["items"][f"quiz_{question['id']}"]
        assert (item["start"], item["end"]) == (question["start"], question["end"])
        prompt = _item(contents, f"quiz_{question['id']}").findtext(f".//{_Q}qti-prompt")
        assert prompt == (question["original_text"] or question["text"])


def test_the_validator_rejects_a_broken_item(
    tmp_path: Path, make_quiz: MakeQuiz, schemas: Schemas
) -> None:
    raw = _export(tmp_path, make_quiz("q_aaa"))["items/quiz_q_aaa.xml"]
    broken = raw.replace(b'max-choices="1"', b'max-choices="one"')
    assert not schemas[0].is_valid(io.BytesIO(broken))


def test_two_exports_at_different_times_are_byte_identical(
    tmp_path: Path, make_quiz: MakeQuiz, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    first = _export(tmp_path, make_quiz("q_aaa"), "a.zip")
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    second = _export(tmp_path, make_quiz("q_aaa"), "b.zip")
    assert first == second
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()


def test_xml_uses_the_default_namespace_and_declares_its_schema(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    package = _export(tmp_path, make_quiz("q_aaa"))
    raw = {name: data.decode("utf-8") for name, data in package.items()}
    for name in ("items/quiz_q_aaa.xml", "assessmentTest.xml"):
        assert f'xmlns="{QTI_NS}"' in raw[name]
        assert f'xsi:schemaLocation="{QTI_NS} {ASI_XSD}"' in raw[name]
    assert f'xmlns="{CP_NS}"' in raw["imsmanifest.xml"]
    assert f'xsi:schemaLocation="{CP_NS} {CP_XSD}"' in raw["imsmanifest.xml"]
    assert 'xml:lang="en"' in raw["items/quiz_q_aaa.xml"]
    for name, text in raw.items():
        if name.endswith(".xml"):
            for prefix in ("<qti:", "<cp:", "ns0:"):
                assert prefix not in text, (name, prefix)


def test_a_choice_item_has_its_options_key_and_points(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    root = _item(_export(tmp_path, make_quiz("q_aaa")))
    interaction = root.find(f".//{_Q}qti-choice-interaction")
    assert interaction is not None
    assert (interaction.get("max-choices"), interaction.get("shuffle")) == ("1", "false")
    assert interaction.findtext(f"{_Q}qti-prompt") == "Which sort is stable: 2 < 3 & 4 > 1?"
    choices = interaction.findall(f"{_Q}qti-simple-choice")
    assert [(node.get("identifier"), node.text) for node in choices] == [
        ("a", "Merge sort"),
        ("b", "Quicksort"),
        ("c", "Heapsort"),
    ]
    response = root.find(f"{_Q}qti-response-declaration[@identifier='RESPONSE']")
    assert response is not None
    assert (response.get("base-type"), response.get("cardinality")) == ("identifier", "single")
    assert response.findtext(f"{_Q}qti-correct-response/{_Q}qti-value") == "a"
    maximum = _outcome(root, "MAXSCORE")
    assert _default(maximum) == "2.0"
    score = _outcome(root, "SCORE")
    assert (score.get("normal-maximum"), score.get("normal-minimum")) == ("2.0", "0.0")
    assert score.get("external-scored") is None
    assert _default(score) == "0.0"


def test_a_correct_answer_scores_the_maximum_and_any_other_zero(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    processing = _item(_export(tmp_path, make_quiz("q_aaa"))).find(f"{_Q}qti-response-processing")
    assert processing is not None
    assert processing.get("template") is None
    response_if = processing.find(f"{_Q}qti-response-condition/{_Q}qti-response-if")
    assert response_if is not None
    assert response_if.find(f"{_Q}qti-match/{_Q}qti-correct[@identifier='RESPONSE']") is not None
    on_correct = response_if.find(f"{_Q}qti-set-outcome-value[@identifier='SCORE']")
    assert on_correct is not None
    assert on_correct.find(f"{_Q}qti-variable[@identifier='MAXSCORE']") is not None
    on_wrong = processing.find(
        f"{_Q}qti-response-condition/{_Q}qti-response-else/"
        f"{_Q}qti-set-outcome-value[@identifier='SCORE']"
    )
    assert on_wrong is not None
    assert on_wrong.findtext(f"{_Q}qti-base-value") == "0.0"


def test_an_item_worth_zero_has_no_normal_range(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["points"] = 0
    root = _item(_export(tmp_path, quiz))
    assert _default(_outcome(root, "MAXSCORE")) == "0.0"
    assert _outcome(root, "SCORE").get("normal-maximum") is None


def test_the_chosen_option_shows_its_explanation_then_the_general_one(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    root = _item(_export(tmp_path, make_quiz("q_aaa")))
    assert _outcome(root, "FEEDBACK").get("base-type") == "identifier"
    assert _default(_outcome(root, "FEEDBACK")) is None
    processing = root.find(f"{_Q}qti-response-processing")
    assert processing is not None
    chosen = processing.find(f"{_Q}qti-set-outcome-value[@identifier='FEEDBACK']")
    assert chosen is not None
    assert chosen.find(f"{_Q}qti-variable[@identifier='RESPONSE']") is not None

    children = [child.tag.removeprefix(_Q) for child in root]
    assert children.index("qti-item-body") < children.index("qti-response-processing")
    assert children.index("qti-response-processing") < children.index("qti-modal-feedback")
    modal = root.findall(f"{_Q}qti-modal-feedback")
    assert [node.get("identifier") for node in modal] == ["a", "b", "c"]
    assert {node.get("outcome-identifier") for node in modal} == {"FEEDBACK"}
    assert {node.get("show-hide") for node in modal} == {"show"}
    paragraphs = [node.text for node in modal[1].iter(f"{_Q}p")]
    assert paragraphs == [
        "Partitioning can reorder equal keys.",
        "A stable sort keeps equal keys in order.",
    ]


def test_the_test_follows_the_instructions(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    plain = ElementTree.fromstring(_export(tmp_path, quiz, "plain.zip")["assessmentTest.xml"])
    control = plain.find(f"{_Q}qti-test-part/{_Q}qti-item-session-control")
    assert control is not None
    assert control.get("show-feedback") == "true"
    assert plain.find(f".//{_Q}qti-ordering") is None
    assert plain.find(f".//{_Q}qti-rubric-block") is None

    quiz["instructions"] = {
        "text": "Pick one answer.",
        "shuffle_questions": True,
        "shuffle_options": True,
        "show_feedback": False,
    }
    package = _export(tmp_path, quiz, "shuffled.zip")
    test = ElementTree.fromstring(package["assessmentTest.xml"])
    control = test.find(f"{_Q}qti-test-part/{_Q}qti-item-session-control")
    assert control is not None
    assert control.get("show-feedback") == "false"
    section = test.find(f".//{_Q}qti-assessment-section")
    assert section is not None
    assert section.get("title") == "Sorting"
    ordering = section.find(f"{_Q}qti-ordering")
    assert ordering is not None
    assert ordering.get("shuffle") == "true"
    rubric = section.find(f"{_Q}qti-rubric-block")
    assert rubric is not None
    assert rubric.get("view") == "candidate"
    assert rubric.findtext(f"{_Q}qti-content-body/{_Q}p") == "Pick one answer."
    interaction = _item(package).find(f".//{_Q}qti-choice-interaction")
    assert interaction is not None
    assert interaction.get("shuffle") == "true"


def test_rejected_items_are_not_in_the_package(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa", "q_bbb")
    quiz["questions"][0]["review"]["status"] = "rejected"
    package = _export(tmp_path, quiz)
    assert [name for name in package if name.startswith("items/")] == ["items/quiz_q_bbb.xml"]


def test_the_sidecar_keeps_provenance_review_and_origins(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    quiz = make_quiz("q_aaa")
    manifest = ElementTree.fromstring(_export(tmp_path, quiz)["imsmanifest.xml"])
    assert manifest.findtext(f"{_CP}metadata/{_CP}schemaversion") == "3.0.0"
    sidecar = manifest.find(f"{_CP}resources/{_CP}resource[@identifier='questxtract-metadata']")
    assert sidecar is not None
    assert (sidecar.get("type"), sidecar.get("href")) == ("extension", "metadata/questxtract.json")
    metadata = json.loads(_export(tmp_path, quiz)["metadata/questxtract.json"])
    assert metadata["format"] == "questxtract-qti-3.0"
    assert metadata["quiz"]["id"] == "quiz_lecture"
    assert metadata["quiz"]["generation"] == quiz["generation"]
    assert metadata["quiz"]["source"] == quiz["source"]
    item = metadata["items"]["quiz_q_aaa"]
    assert (item["source_question_id"], item["start"], item["end"]) == ("q_aaa", 1.5, 4.0)
    assert item["review_status"] == "pending"
    assert item["origin"]["prompt"] == "spoken"
    assert item["origin"]["options"] == {"a": "generated", "b": "generated", "c": "generated"}


def test_a_newer_quiz_with_an_unsupported_item_fails_naming_it(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    quiz = make_quiz("q_aaa", "q_bbb", schema_version="1.1")
    quiz["questions"][1]["interaction_type"] = "choice_multiple"
    with pytest.raises(QtiExportError, match=r"questions\[1\]\.interaction_type"):
        _export(tmp_path, quiz)
    assert not (tmp_path / "out.zip").exists()


def test_nan_and_infinity_in_the_file_fail_instead_of_reaching_the_xml(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    source = tmp_path / "quiz.json"
    for literal in ("NaN", "Infinity"):
        text = json.dumps(make_quiz("q_aaa")).replace('"points": 2', f'"points": {literal}')
        source.write_text(text, encoding="utf-8")
        with pytest.raises(QtiExportError, match=r"questions\[0\]\.points: expected a finite"):
            export_qti(source, tmp_path / "out.zip")
    assert not (tmp_path / "out.zip").exists()

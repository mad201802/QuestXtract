"""Tests for the QTI 2.1 package writer (issue #163).

Every package written here is validated against the official 1EdTech XSDs, ASI 2.1.2 and the QTI
2.1 content-package profile, from the local mirror `scripts/fetch_qti_xsd.py` downloads; the
tests never reach the network. Without the mirror they skip, unless QUESTXTRACT_REQUIRE_QTI_XSD
is set, as in CI, where they fail.
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
from scripts.fetch_qti_xsd import ASI_XSD_V2P1, CP_XSD_V2P1, missing, uri_mapper, xsd_dir

from questxtract import qti_v3
from questxtract.cli import main
from questxtract.export import read_questions
from questxtract.qti import METADATA_PATH, QtiExportError
from questxtract.qti_package import export_qti
from questxtract.qti_v2 import CP_NS, QTI_NS
from questxtract.quiz_schema import QUIZ_SCHEMA_PATH, read_quiz

_Q = f"{{{QTI_NS}}}"
_CP = f"{{{CP_NS}}}"
_V3 = f"{{{qti_v3.QTI_NS}}}"
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_DATA = Path(__file__).parent / "data"
_SPECIAL = "<b>bold</b> & \"double\" 'single' ]]> Größe 日本 🙂"

MakeQuiz = Callable[..., dict[str, Any]]
Schemas = tuple[xmlschema.XMLSchema, xmlschema.XMLSchema]


@pytest.fixture(scope="session")
def schemas() -> Schemas:
    """The 2.1 item-and-test schema and content-package schema, compiled once, offline."""
    root = xsd_dir()
    if missing(root):
        message = f"QTI XSDs not in {root}: run `uv run python scripts/fetch_qti_xsd.py`"
        if os.environ.get("QUESTXTRACT_REQUIRE_QTI_XSD"):
            pytest.fail(message)
        pytest.skip(message)
    mapper = uri_mapper(root)
    asi, cp = (
        xmlschema.XMLSchema(mapper(url), uri_mapper=mapper, allow="local")
        for url in (ASI_XSD_V2P1, CP_XSD_V2P1)
    )
    return asi, cp


def _export(
    tmp_path: Path, quiz: dict[str, Any], name: str = "out.zip", version: str = "2.1"
) -> dict[str, bytes]:
    source = tmp_path / "quiz.json"
    source.write_text(json.dumps(quiz), encoding="utf-8")
    export_qti(source, tmp_path / name, version=version)
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
    references = list(test.iter(f"{_Q}assessmentItemRef"))
    assert {node.get("href") for node in references} <= names
    assert len({node.get("identifier") for node in references}) == len(references)

    for name in names:
        if not name.startswith("items/"):
            continue
        item = ElementTree.fromstring(package[name])
        assert f"items/{item.get('identifier')}.xml" == name
        declared = {node.get("identifier") for node in item.iter(f"{_Q}responseDeclaration")}
        outcomes = {node.get("identifier") for node in item.iter(f"{_Q}outcomeDeclaration")}
        choices = {node.get("identifier") for node in item.iter(f"{_Q}simpleChoice")}
        interaction = item.find(f".//{_Q}choiceInteraction")
        assert interaction is not None
        assert interaction.get("responseIdentifier") in declared
        correct = item.findall(f".//{_Q}correctResponse/{_Q}value")
        assert correct
        assert {node.text for node in correct} <= choices
        for modal in item.iter(f"{_Q}modalFeedback"):
            assert modal.get("identifier") in choices
            assert modal.get("outcomeIdentifier") in outcomes


# Every shape the writer can produce: instructions absent and present, shuffles off and on,
# feedback shown and hidden, points above zero and zero, rejected items, XML special characters.
_SHAPES = (
    "two items",
    "no items",
    "instructions text, both shuffles, no feedback",
    "both shuffles, no instructions text",
    "an item worth 0 points",
    "a rejected item",
    "XML special characters",
)


def _shape(make_quiz: MakeQuiz, shape: str) -> dict[str, Any]:
    quiz = make_quiz() if shape == "no items" else make_quiz("q_aaa", "q_bbb")
    if shape == "instructions text, both shuffles, no feedback":
        quiz["instructions"] = {
            "text": "Pick one answer.",
            "shuffle_questions": True,
            "shuffle_options": True,
            "show_feedback": False,
        }
    elif shape == "both shuffles, no instructions text":
        quiz["instructions"].update(shuffle_questions=True, shuffle_options=True)
    elif shape == "an item worth 0 points":
        quiz["questions"][0]["points"] = 0
    elif shape == "a rejected item":
        quiz["questions"][0]["review"]["status"] = "rejected"
    elif shape == "XML special characters":
        quiz["title"] = f"Sorting {_SPECIAL}"
        quiz["instructions"]["text"] = f"Answer {_SPECIAL}"
        for item in quiz["questions"]:
            item["prompt"]["text"] = f"Which {_SPECIAL}?"
            for option in item["options"]:
                option["text"]["text"] += f" {_SPECIAL}"
            item["solution"]["explanation"]["text"] += f" {_SPECIAL}"
            for explanation in item["solution"]["option_explanations"].values():
                explanation["text"] += f" {_SPECIAL}"
    return quiz


def _item(package: dict[str, bytes], identifier: str = "quiz_q_aaa") -> ElementTree.Element:
    return ElementTree.fromstring(package[f"items/{identifier}.xml"])


def _outcome(root: ElementTree.Element, identifier: str) -> ElementTree.Element:
    node = root.find(f"{_Q}outcomeDeclaration[@identifier='{identifier}']")
    assert node is not None
    return node


def _default(outcome: ElementTree.Element) -> str | None:
    return outcome.findtext(f"{_Q}defaultValue/{_Q}value")


# What _content reads, by version: the same element or attribute under its 2.1 and 3.0 names.
_NAMESPACES = {"2.1": _Q, "3.0": _V3}
_NAMES = {
    "2.1": {
        "part": "testPart",
        "control": "itemSessionControl",
        "show_feedback": "showFeedback",
        "section": "assessmentSection",
        "ordering": "ordering",
        "rubric": "rubricBlock",
        "ref": "assessmentItemRef",
        "interaction": "choiceInteraction",
        "prompt": "prompt",
        "choice": "simpleChoice",
        "max_choices": "maxChoices",
        "response": "responseDeclaration",
        "correct": "correctResponse",
        "outcome": "outcomeDeclaration",
        "default": "defaultValue",
        "value": "value",
        "modal": "modalFeedback",
    },
    "3.0": {
        "part": "qti-test-part",
        "control": "qti-item-session-control",
        "show_feedback": "show-feedback",
        "section": "qti-assessment-section",
        "ordering": "qti-ordering",
        "rubric": "qti-rubric-block",
        "ref": "qti-assessment-item-ref",
        "interaction": "qti-choice-interaction",
        "prompt": "qti-prompt",
        "choice": "qti-simple-choice",
        "max_choices": "max-choices",
        "response": "qti-response-declaration",
        "correct": "qti-correct-response",
        "outcome": "qti-outcome-declaration",
        "default": "qti-default-value",
        "value": "qti-value",
        "modal": "qti-modal-feedback",
    },
}


def _content(package: dict[str, bytes], version: str) -> dict[str, Any]:
    """What a package tells the candidate and the LMS, in terms both versions share."""
    q, names = _NAMESPACES[version], _NAMES[version]

    def path(*keys: str) -> str:
        return "/".join(q + names[key] for key in keys)

    test = ElementTree.fromstring(package["assessmentTest.xml"])
    control = test.find(path("part", "control"))
    section = test.find(path("part", "section"))
    assert control is not None and section is not None
    ordering = section.find(path("ordering"))
    rubric = section.find(path("rubric"))
    items = {}
    for name, data in package.items():
        if not name.startswith("items/"):
            continue
        root = ElementTree.fromstring(data)
        interaction = root.find(".//" + path("interaction"))
        maximum = root.find(path("outcome") + "[@identifier='MAXSCORE']")
        assert interaction is not None and maximum is not None
        items[name] = {
            "identifier": root.get("identifier"),
            "title": root.get("title"),
            "language": root.get(_XML_LANG),
            "prompt": interaction.findtext(path("prompt")),
            "shuffle": interaction.get("shuffle"),
            "max_choices": interaction.get(names["max_choices"]),
            "options": [
                (node.get("identifier"), node.text) for node in interaction.findall(path("choice"))
            ],
            "correct": root.findtext(path("response", "correct", "value")),
            "points": maximum.findtext(path("default", "value")),
            "feedback": [
                (node.get("identifier"), [paragraph.text for paragraph in node.iter(f"{q}p")])
                for node in root.findall(path("modal"))
            ],
        }
    sidecar = json.loads(package[METADATA_PATH])
    return {
        "entries": list(package),
        "title": (test.get("identifier"), test.get("title"), section.get("title")),
        "show_feedback": control.get(names["show_feedback"]),
        "shuffle_items": None if ordering is None else ordering.get("shuffle"),
        "instructions": None if rubric is None else [node.text for node in rubric.iter(f"{q}p")],
        "references": [
            (node.get("identifier"), node.get("href")) for node in section.iter(path("ref"))
        ],
        "items": items,
        "sidecar": {key: value for key, value in sidecar.items() if key != "format"},
    }


@pytest.mark.parametrize("shape", _SHAPES)
def test_every_shape_gives_a_package_the_xsds_accept(
    tmp_path: Path, make_quiz: MakeQuiz, schemas: Schemas, shape: str
) -> None:
    _assert_conformant(_export(tmp_path, _shape(make_quiz, shape)), schemas)


@pytest.mark.parametrize("shape", _SHAPES)
def test_the_package_holds_what_the_3_0_package_holds(
    tmp_path: Path, make_quiz: MakeQuiz, shape: str
) -> None:
    """The same items, ids, order, texts, key, points, feedback and instructions as in 3.0."""
    quiz = _shape(make_quiz, shape)
    v2 = _content(_export(tmp_path, quiz, "v2.zip"), "2.1")
    v3 = _content(_export(tmp_path, quiz, "v3.zip", version="3.0"), "3.0")
    assert v2 == v3


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

    assert main(["qti", "lecture.quiz.json", "-o", "lecture.zip", "--qti-version", "2.1"]) == 0

    with zipfile.ZipFile(tmp_path / "lecture.zip") as package:
        contents = {name: package.read(name) for name in package.namelist()}
    _assert_conformant(contents, schemas)
    kept = [question for index, question in enumerate(asked) if index != 1]
    assert [name for name in contents if name.startswith("items/")] == [
        f"items/quiz_{question['id']}.xml" for question in kept
    ]
    metadata = json.loads(contents[METADATA_PATH])
    for question in kept:
        item = metadata["items"][f"quiz_{question['id']}"]
        assert (item["start"], item["end"]) == (question["start"], question["end"])
        prompt = _item(contents, f"quiz_{question['id']}").findtext(f".//{_Q}prompt")
        assert prompt == (question["original_text"] or question["text"])


def test_the_validators_reject_a_broken_item_and_an_extension_sidecar(
    tmp_path: Path, make_quiz: MakeQuiz, schemas: Schemas
) -> None:
    package = _export(tmp_path, make_quiz("q_aaa"))
    broken = package["items/quiz_q_aaa.xml"].replace(b'maxChoices="1"', b'maxChoices="one"')
    assert not schemas[0].is_valid(io.BytesIO(broken))
    # The 2.1 profile has no extension resource type, which is why the sidecar is webcontent.
    extension = package["imsmanifest.xml"].replace(b'type="webcontent"', b'type="extension"')
    assert not schemas[1].is_valid(io.BytesIO(extension))


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
        assert f'xsi:schemaLocation="{QTI_NS} {ASI_XSD_V2P1}"' in raw[name]
    assert f'xmlns="{CP_NS}"' in raw["imsmanifest.xml"]
    assert f'xsi:schemaLocation="{CP_NS} {CP_XSD_V2P1}"' in raw["imsmanifest.xml"]
    assert 'xml:lang="en"' in raw["items/quiz_q_aaa.xml"]
    for name, text in raw.items():
        if name.endswith(".xml"):
            for prefix in ("<qti:", "<cp:", "ns0:"):
                assert prefix not in text, (name, prefix)


def test_names_are_camel_case_without_the_qti_prefix(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = _shape(make_quiz, "instructions text, both shuffles, no feedback")
    for name, data in _export(tmp_path, quiz).items():
        if not name.endswith(".xml"):
            continue
        for node in ElementTree.fromstring(data).iter():
            tag = node.tag.rpartition("}")[2]
            attributes = [attribute.rpartition("}")[2] for attribute in node.attrib]
            assert "-" not in tag, (name, tag)
            assert not [attribute for attribute in attributes if "-" in attribute], (name, tag)


def test_a_choice_item_has_its_options_key_and_points(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    root = _item(_export(tmp_path, make_quiz("q_aaa")))
    assert root.tag == f"{_Q}assessmentItem"
    assert (root.get("identifier"), root.get("title")) == ("quiz_q_aaa", "Question q_aaa")
    assert (root.get("adaptive"), root.get("timeDependent")) == ("false", "false")
    interaction = root.find(f"{_Q}itemBody/{_Q}choiceInteraction")
    assert interaction is not None
    assert interaction.get("responseIdentifier") == "RESPONSE"
    assert (interaction.get("maxChoices"), interaction.get("shuffle")) == ("1", "false")
    assert interaction.findtext(f"{_Q}prompt") == "Which sort is stable: 2 < 3 & 4 > 1?"
    choices = interaction.findall(f"{_Q}simpleChoice")
    assert [(node.get("identifier"), node.text) for node in choices] == [
        ("a", "Merge sort"),
        ("b", "Quicksort"),
        ("c", "Heapsort"),
    ]
    response = root.find(f"{_Q}responseDeclaration[@identifier='RESPONSE']")
    assert response is not None
    assert (response.get("baseType"), response.get("cardinality")) == ("identifier", "single")
    assert response.findtext(f"{_Q}correctResponse/{_Q}value") == "a"
    maximum = _outcome(root, "MAXSCORE")
    assert (maximum.get("baseType"), _default(maximum)) == ("float", "2.0")
    score = _outcome(root, "SCORE")
    assert (score.get("normalMaximum"), score.get("normalMinimum")) == ("2.0", "0.0")
    assert (score.get("baseType"), _default(score)) == ("float", "0.0")


def test_a_correct_answer_scores_the_maximum_and_any_other_zero(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    processing = _item(_export(tmp_path, make_quiz("q_aaa"))).find(f"{_Q}responseProcessing")
    assert processing is not None
    # The standard template is named for engines that support only templates; the item's own
    # rules below are preferred by every engine that runs them.
    assert processing.get("template") == (
        "http://www.imsglobal.org/question/qti_v2p1/rptemplates/match_correct"
    )
    response_if = processing.find(f"{_Q}responseCondition/{_Q}responseIf")
    assert response_if is not None
    match = response_if.find(f"{_Q}match")
    assert match is not None
    assert match.find(f"{_Q}variable[@identifier='RESPONSE']") is not None
    assert match.find(f"{_Q}correct[@identifier='RESPONSE']") is not None
    on_correct = response_if.find(f"{_Q}setOutcomeValue[@identifier='SCORE']")
    assert on_correct is not None
    assert on_correct.find(f"{_Q}variable[@identifier='MAXSCORE']") is not None
    on_wrong = processing.find(
        f"{_Q}responseCondition/{_Q}responseElse/{_Q}setOutcomeValue[@identifier='SCORE']"
    )
    assert on_wrong is not None
    value = on_wrong.find(f"{_Q}baseValue")
    assert value is not None
    assert (value.get("baseType"), value.text) == ("float", "0.0")


def test_an_item_worth_zero_has_no_normal_range(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["points"] = 0
    root = _item(_export(tmp_path, quiz))
    assert _default(_outcome(root, "MAXSCORE")) == "0.0"
    score = _outcome(root, "SCORE")
    assert (score.get("normalMaximum"), score.get("normalMinimum")) == (None, None)


def test_the_chosen_option_shows_its_explanation_then_the_general_one(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    root = _item(_export(tmp_path, make_quiz("q_aaa")))
    assert _outcome(root, "FEEDBACK").get("baseType") == "identifier"
    assert _default(_outcome(root, "FEEDBACK")) is None  # no response, no feedback
    processing = root.find(f"{_Q}responseProcessing")
    assert processing is not None
    chosen = processing.find(f"{_Q}setOutcomeValue[@identifier='FEEDBACK']")
    assert chosen is not None
    assert chosen.find(f"{_Q}variable[@identifier='RESPONSE']") is not None

    children = [child.tag.removeprefix(_Q) for child in root]
    assert children.index("itemBody") < children.index("responseProcessing")
    assert children.index("responseProcessing") < children.index("modalFeedback")
    modal = root.findall(f"{_Q}modalFeedback")
    assert [node.get("identifier") for node in modal] == ["a", "b", "c"]
    assert {node.get("outcomeIdentifier") for node in modal} == {"FEEDBACK"}
    assert {node.get("showHide") for node in modal} == {"show"}
    # 2.1 has no content body: the paragraphs are the feedback's own children.
    assert [[child.tag for child in node] for node in modal] == [[f"{_Q}p", f"{_Q}p"]] * 3
    assert [node.text for node in modal[1]] == [
        "Partitioning can reorder equal keys.",
        "A stable sort keeps equal keys in order.",
    ]


def test_the_test_follows_the_instructions(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    plain = ElementTree.fromstring(_export(tmp_path, quiz, "plain.zip")["assessmentTest.xml"])
    assert (plain.get("identifier"), plain.get("title")) == ("quiz_lecture", "Sorting")
    part = plain.find(f"{_Q}testPart")
    assert part is not None
    assert (part.get("navigationMode"), part.get("submissionMode")) == ("linear", "individual")
    control = part.find(f"{_Q}itemSessionControl")
    assert control is not None
    assert control.get("showFeedback") == "true"
    assert plain.find(f".//{_Q}ordering") is None
    assert plain.find(f".//{_Q}rubricBlock") is None

    quiz["instructions"] = {
        "text": "Pick one answer.",
        "shuffle_questions": True,
        "shuffle_options": True,
        "show_feedback": False,
    }
    package = _export(tmp_path, quiz, "shuffled.zip")
    test = ElementTree.fromstring(package["assessmentTest.xml"])
    control = test.find(f"{_Q}testPart/{_Q}itemSessionControl")
    assert control is not None
    assert control.get("showFeedback") == "false"
    section = test.find(f".//{_Q}assessmentSection")
    assert section is not None
    assert (section.get("title"), section.get("visible")) == ("Sorting", "true")
    ordering = section.find(f"{_Q}ordering")
    assert ordering is not None
    assert ordering.get("shuffle") == "true"
    rubric = section.find(f"{_Q}rubricBlock")
    assert rubric is not None
    assert (rubric.get("view"), rubric.get("use")) == ("candidate", "instructions")
    # No content body: the paragraph is the rubric block's own child.
    assert [(child.tag, child.text) for child in rubric] == [(f"{_Q}p", "Pick one answer.")]
    interaction = _item(package).find(f".//{_Q}choiceInteraction")
    assert interaction is not None
    assert interaction.get("shuffle") == "true"


@pytest.mark.parametrize(("shuffle_questions", "shuffle_options"), [(True, False), (False, True)])
def test_each_shuffle_maps_on_its_own(
    tmp_path: Path, make_quiz: MakeQuiz, shuffle_questions: bool, shuffle_options: bool
) -> None:
    quiz = make_quiz("q_aaa")
    quiz["instructions"].update(
        shuffle_questions=shuffle_questions, shuffle_options=shuffle_options
    )
    package = _export(tmp_path, quiz)
    test = ElementTree.fromstring(package["assessmentTest.xml"])
    assert (test.find(f".//{_Q}assessmentSection/{_Q}ordering") is not None) == shuffle_questions
    interaction = _item(package).find(f".//{_Q}choiceInteraction")
    assert interaction is not None
    assert interaction.get("shuffle") == ("true" if shuffle_options else "false")


def test_rejected_items_are_not_in_the_package(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa", "q_bbb")
    quiz["questions"][0]["review"]["status"] = "rejected"
    package = _export(tmp_path, quiz)
    assert [name for name in package if name.startswith("items/")] == ["items/quiz_q_bbb.xml"]
    test = ElementTree.fromstring(package["assessmentTest.xml"])
    assert [node.get("href") for node in test.iter(f"{_Q}assessmentItemRef")] == [
        "items/quiz_q_bbb.xml"
    ]
    manifest = ElementTree.fromstring(package["imsmanifest.xml"])
    assert manifest.find(f".//{_CP}resource[@identifier='quiz_q_aaa']") is None
    assert "quiz_q_aaa" not in json.loads(package[METADATA_PATH])["items"]


def test_xml_special_characters_are_escaped(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = _shape(make_quiz, "XML special characters")
    package = _export(tmp_path, quiz)
    test = ElementTree.fromstring(package["assessmentTest.xml"])
    assert test.get("title") == f"Sorting {_SPECIAL}"
    assert test.findtext(f".//{_Q}rubricBlock/{_Q}p") == f"Answer {_SPECIAL}"
    root = _item(package)
    assert root.findtext(f".//{_Q}prompt") == f"Which {_SPECIAL}?"
    assert [node.text for node in root.iter(f"{_Q}simpleChoice")][0] == f"Merge sort {_SPECIAL}"
    assert [node.text for node in root.findall(f"{_Q}modalFeedback[@identifier='b']/{_Q}p")] == [
        f"Partitioning can reorder equal keys. {_SPECIAL}",
        f"A stable sort keeps equal keys in order. {_SPECIAL}",
    ]
    assert json.loads(package[METADATA_PATH])["quiz"]["id"] == "quiz_lecture"


@pytest.mark.parametrize(
    ("path", "character"),
    [
        ("title", "\x00"),
        ("instructions.text", "\x0b"),
        ("questions[0].prompt.text", "\x01"),
        ("questions[0].options[1].text.text", "￾"),
        ("questions[0].solution.explanation.text", "￿"),
        ("questions[0].solution.option_explanations.b.text", "\ud800"),
    ],
)
def test_a_character_xml_does_not_allow_fails_with_its_path(
    tmp_path: Path, make_quiz: MakeQuiz, path: str, character: str
) -> None:
    quiz = make_quiz("q_aaa")
    item = quiz["questions"][0]
    container: dict[str, Any] = {
        "title": quiz,
        "instructions.text": quiz["instructions"],
        "questions[0].prompt.text": item["prompt"],
        "questions[0].options[1].text.text": item["options"][1]["text"],
        "questions[0].solution.explanation.text": item["solution"]["explanation"],
        "questions[0].solution.option_explanations.b.text": (
            item["solution"]["option_explanations"]["b"]
        ),
    }[path]
    key = "title" if path == "title" else "text"
    container[key] = f"before {character} after"
    escaped = path.replace(".", r"\.").replace("[", r"\[").replace("]", r"\]")
    code = f"{ord(character):04X}"
    with pytest.raises(QtiExportError, match=rf"^{escaped}: character U\+{code} is not allowed"):
        _export(tmp_path, quiz)
    assert not (tmp_path / "out.zip").exists()


def test_the_manifest_follows_the_qti_2_1_profile(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    manifest = ElementTree.fromstring(
        _export(tmp_path, make_quiz("q_aaa", "q_bbb"))["imsmanifest.xml"]
    )
    assert manifest.findtext(f"{_CP}metadata/{_CP}schema") == "QTIv2.1 Package"
    assert manifest.findtext(f"{_CP}metadata/{_CP}schemaversion") == "1.0.0"
    resources = manifest.findall(f"{_CP}resources/{_CP}resource")
    assert [(node.get("identifier"), node.get("type"), node.get("href")) for node in resources] == [
        ("quiz_lecture", "imsqti_test_xmlv2p1", "assessmentTest.xml"),
        ("quiz_q_aaa", "imsqti_item_xmlv2p1", "items/quiz_q_aaa.xml"),
        ("quiz_q_bbb", "imsqti_item_xmlv2p1", "items/quiz_q_bbb.xml"),
        ("questxtract-metadata", "webcontent", METADATA_PATH),
    ]
    dependencies = resources[0].findall(f"{_CP}dependency")
    assert [node.get("identifierref") for node in dependencies] == ["quiz_q_aaa", "quiz_q_bbb"]


def test_the_sidecar_keeps_provenance_review_and_origins(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    quiz = make_quiz("q_aaa")
    metadata = json.loads(_export(tmp_path, quiz)[METADATA_PATH])
    assert metadata["format"] == "questxtract-qti-2.1"
    assert metadata["quiz"]["id"] == "quiz_lecture"
    assert metadata["quiz"]["generation"] == quiz["generation"]
    assert metadata["quiz"]["source"] == quiz["source"]
    item = metadata["items"]["quiz_q_aaa"]
    assert (item["source_question_id"], item["start"], item["end"]) == ("q_aaa", 1.5, 4.0)
    assert item["review_status"] == "pending"
    assert item["origin"]["prompt"] == "spoken"
    assert item["origin"]["options"] == {"a": "generated", "b": "generated", "c": "generated"}

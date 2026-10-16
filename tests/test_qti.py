"""Tests for the QTI model and its ``quiz.json`` mapping (issue #67)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from questxtract.qti import QtiExportError, quiz_to_qti
from questxtract.quiz_schema import QUIZ_SCHEMA_PATH

MakeQuiz = Callable[..., dict[str, Any]]


def _ids(document: dict[str, Any]) -> list[str]:
    return [item.identifier for item in quiz_to_qti(document).items]


def test_the_fixture_is_a_valid_quiz(make_quiz: MakeQuiz) -> None:
    schema = json.loads(QUIZ_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(make_quiz("q_aaa", "q_bbb"))


def test_an_empty_quiz_gives_an_empty_test(make_quiz: MakeQuiz) -> None:
    assessment = quiz_to_qti(make_quiz())
    assert assessment.items == ()
    assert (assessment.identifier, assessment.title) == ("quiz_lecture", "Sorting")


def test_an_item_becomes_a_single_choice_item_with_its_key_and_feedback(
    make_quiz: MakeQuiz,
) -> None:
    item = quiz_to_qti(make_quiz("q_aaa")).items[0]
    assert (item.identifier, item.title, item.language) == ("quiz_q_aaa", "Question q_aaa", "en")
    assert item.prompt == "Which sort is stable: 2 < 3 & 4 > 1?"
    assert item.interaction.interaction_type == "choiceInteraction"
    assert item.interaction.max_choices == 1
    (declaration,) = item.response_declarations
    assert (declaration.base_type, declaration.cardinality) == ("identifier", "single")
    assert declaration.correct_values == ("a",)
    assert [(option.identifier, option.text) for option in item.options] == [
        ("a", "Merge sort"),
        ("b", "Quicksort"),
        ("c", "Heapsort"),
    ]
    assert item.options[1].feedback == "Partitioning can reorder equal keys."
    assert item.explanation == "A stable sort keeps equal keys in order."
    assert item.points == 2.0
    assert item.provenance is not None
    assert (item.provenance.source_question_id, item.provenance.start) == ("q_aaa", 1.5)


def test_text_origins_and_the_review_status_go_to_the_metadata(make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["options"][2]["text"]["origin"] = "spoken"
    metadata = quiz_to_qti(quiz).items[0].metadata
    assert metadata == {
        "review_status": "pending",
        "origin": {
            "prompt": "spoken",
            "explanation": "generated",
            "options": {"a": "generated", "b": "generated", "c": "spoken"},
            "option_explanations": {"a": "generated", "b": "generated", "c": "generated"},
        },
    }


def test_instructions_reach_the_test(make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    quiz["instructions"] = {
        "text": "Pick one answer.",
        "shuffle_questions": True,
        "shuffle_options": True,
        "show_feedback": False,
    }
    assessment = quiz_to_qti(quiz)
    assert assessment.instructions == "Pick one answer."
    assert assessment.shuffle_items
    assert not assessment.show_feedback
    assert assessment.items[0].interaction.shuffle


def test_rejected_items_are_left_out(make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_pending", "q_accepted", "q_rejected")
    quiz["questions"][1]["review"]["status"] = "accepted"
    quiz["questions"][2]["review"]["status"] = "rejected"
    quiz["questions"][2]["prompt"] = None  # a rejected item is not read further
    quiz["questions"][2]["interaction_type"] = "choice_multiple"
    assert _ids(quiz) == ["quiz_q_pending", "quiz_q_accepted"]


def test_zero_points_are_allowed(make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["points"] = 0
    outcomes = quiz_to_qti(quiz).items[0].outcome_declarations
    score = next(outcome for outcome in outcomes if outcome.identifier == "SCORE")
    assert score.normal_maximum is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda q: q.update(questions=None), "'questions' list"),
        (lambda q: q.update(id="quiz lecture"), r"^id: invalid XML identifier"),
        (lambda q: q.update(language="English, UK"), r"^language: expected a language tag"),
        (lambda q: q.update(title=" "), r"^title: expected a non-empty string"),
        (lambda q: q.update(instructions=[]), r"^instructions: expected an object"),
        (
            lambda q: q["instructions"].update(show_feedback="yes"),
            r"^instructions\.show_feedback: expected true or false",
        ),
        (lambda q: q["questions"].append(3), r"^questions\[1\]: expected an object"),
        (
            lambda q: q["questions"][0].update(review=None),
            r"^questions\[0\]\.review: expected an object",
        ),
        (
            lambda q: q["questions"][0].update(id="quiz_q a"),
            r"^questions\[0\]\.id: invalid XML identifier",
        ),
        (
            lambda q: q["questions"][0]["prompt"].update(text=""),
            r"^questions\[0\]\.prompt\.text: expected a non-empty string",
        ),
        (
            lambda q: q["questions"][0]["prompt"].update(origin="llm"),
            r"^questions\[0\]\.prompt\.origin: expected 'spoken' or 'generated'",
        ),
        (
            lambda q: q["questions"][0].update(options=[]),
            r"^questions\[0\]\.options: expected a non-empty list",
        ),
        (
            lambda q: q["questions"][0]["options"][1].update(id="a"),
            r"^questions\[0\]\.options\[1\]\.id: 'a' repeats another option",
        ),
        (
            lambda q: q["questions"][0]["solution"]["option_explanations"].pop("c"),
            r"^questions\[0\]\.solution\.option_explanations\.c: expected an object",
        ),
        (
            lambda q: q["questions"][0]["correct_response"].update(option_ids=["d"]),
            r"^questions\[0\]\.correct_response\.option_ids: expected the id of one option",
        ),
        (
            lambda q: q["questions"][0].update(interaction_type="choice_multiple"),
            r"^questions\[0\]\.interaction_type: 'choice_multiple' is not supported",
        ),
        (
            lambda q: q["questions"][0].update(points=True),
            r"^questions\[0\]\.points: expected a finite number >= 0",
        ),
        (
            lambda q: q["questions"][0].update(points=float("nan")),
            r"^questions\[0\]\.points: expected a finite number >= 0, got nan",
        ),
        (
            lambda q: q["questions"][0]["provenance"].update(end=None),
            r"^questions\[0\]\.provenance: expected finite numbers",
        ),
        (
            lambda q: q["questions"][0]["provenance"].update(start=float("-inf")),
            r"^questions\[0\]\.provenance: expected finite numbers",
        ),
    ],
)
def test_a_malformed_field_fails_naming_its_path(
    make_quiz: MakeQuiz, mutate: Callable[[dict[str, Any]], Any], message: str
) -> None:
    quiz = make_quiz("q_aaa")
    mutate(quiz)
    with pytest.raises(QtiExportError, match=message):
        quiz_to_qti(quiz)


@pytest.mark.parametrize(
    ("location", "text"),
    [
        ("prompt", "What is\x07 this?"),
        ("prompt", "Lone \ud800 surrogate?"),
        ("explanation", "Because\x0c."),
    ],
)
def test_a_character_illegal_in_xml_fails_naming_the_field(
    make_quiz: MakeQuiz, location: str, text: str
) -> None:
    quiz = make_quiz("q_aaa")
    item = quiz["questions"][0]
    target = item["prompt"] if location == "prompt" else item["solution"]["explanation"]
    target["text"] = text
    with pytest.raises(QtiExportError, match=rf"{location}\.text: character U\+"):
        quiz_to_qti(quiz)


def test_tabs_and_newlines_are_allowed(make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["prompt"]["text"] = "What is\tthis,\nreally?"
    assert quiz_to_qti(quiz).items[0].prompt == "What is\tthis,\nreally?"


def test_identifiers_are_unique_across_the_test_and_its_items(make_quiz: MakeQuiz) -> None:
    with pytest.raises(QtiExportError, match=r"questions\[1\]\.id: 'quiz_q_aaa' is already"):
        quiz_to_qti(make_quiz("q_aaa", "q_aaa"))
    with pytest.raises(QtiExportError, match=r"questions\[0\]\.id: 'quiz_q_x' is already"):
        quiz_to_qti(make_quiz("q_x", id="quiz_q_x"))


def test_the_mapping_is_stable(make_quiz: MakeQuiz) -> None:
    assert quiz_to_qti(make_quiz("q_aaa")) == quiz_to_qti(make_quiz("q_aaa"))

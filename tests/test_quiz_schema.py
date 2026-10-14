import json
import re
from importlib.resources import files

import pytest
from jsonschema import Draft202012Validator

from questxtract import SchemaError, read_quiz, validate_quiz
from questxtract.export import result_to_dict
from questxtract.pipeline import STAGES, run
from questxtract.quiz_schema import QUIZ_SCHEMA_PATH, QUIZ_SCHEMA_VERSION

DROP = object()


def _text(text, origin="generated"):
    return {"text": text, "origin": origin}


def _option(option_id, text):
    return {"id": option_id, "text": _text(text)}


def _item(source_id, prompt, start, end):
    return {
        "id": "quiz_" + source_id,
        "prompt": _text(prompt, "spoken"),
        "interaction_type": "choice_single",
        "points": 1,
        "options": [_option("a", "the right one"), _option("b", "wrong"), _option("c", "wrong")],
        "correct_response": {"option_ids": ["a"]},
        "solution": {
            "explanation": _text("Because."),
            "option_explanations": {
                "a": _text("Correct."),
                "b": _text("A misconception."),
                "c": _text("Another one."),
            },
        },
        "provenance": {"source_question_id": source_id, "start": start, "end": end},
        "review": {"status": "pending", "note": None, "metadata": {}},
    }


# What validate_quiz reads from a questions.json: ids, text and original_text. q_bbbbbb was
# reworded by the LLM judge (verdict "revise"), so only original_text is what was said.
QUESTIONS = {
    "schema_version": "2.1",
    "questions": [
        {"id": "q_aaaaaa", "text": "What is a stable sort?", "original_text": None},
        {
            "id": "q_bbbbbb",
            "text": "Why use merge sort?",
            "original_text": "So why would we, um, use merge sort?",
        },
    ],
}


def _quiz():
    return {
        "schema_version": QUIZ_SCHEMA_VERSION,
        "id": "quiz_lecture",
        "title": "Sorting",
        "language": "en",
        "status": "draft",
        "source": {
            "questions_path": "lecture.questions.json",
            "questions_schema_version": "2.1",
            "questions_hash": "9f2a",
        },
        "instructions": {
            "shuffle_questions": False,
            "shuffle_options": True,
            "show_feedback": True,
        },
        "generation": {"method": "llm", "model": "gpt-5.6-terra", "prompt_version": "v0"},
        "questions": [
            _item("q_aaaaaa", "What is a stable sort?", 2.2, 4.0),
            _item("q_bbbbbb", "So why would we, um, use merge sort?", 10.0, 12.5),
        ],
    }


def _mutate(document, path, value):
    """Set (or with ``DROP`` delete) the value at a dotted path such as ``questions.0.id``."""
    *parents, last = path.split(".")
    target = document
    for key in parents:
        target = target[int(key)] if isinstance(target, list) else target[key]
    key = int(last) if isinstance(target, list) else last
    if value is DROP:
        del target[key]
    else:
        target[key] = value


@pytest.fixture(scope="module")
def validator():
    schema = json.loads(QUIZ_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_a_valid_quiz_passes_the_schema_and_the_validator(validator):
    validator.validate(_quiz())
    validate_quiz(_quiz(), QUESTIONS)
    validate_quiz(_quiz())


def test_an_empty_quiz_passes(validator):
    quiz = _quiz()
    quiz["questions"] = []
    validator.validate(quiz)
    validate_quiz(quiz, QUESTIONS)


def test_a_quiz_from_a_real_questions_file_passes(validator, context):
    questions = result_to_dict(run(context, STAGES[2:]))
    question = questions["questions"][0]
    quiz = _quiz()
    quiz["questions"] = [
        _item(question["id"], question["text"], question["start"], question["end"])
    ]
    validator.validate(quiz)
    validate_quiz(quiz, questions)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("description", "Sorting algorithms."),
        ("schema_version", "1.4"),
        ("source.questions_path", "../exports/lecture.questions.json"),
        ("instructions.text", "Pick one answer."),
        ("generation.method", "manual"),
        ("generation.model", None),
        ("generation.prompt_version", None),
        ("questions.0.points", 0),
        ("questions.0.review.status", "rejected"),
        ("questions.0.review.note", "Too easy."),
        ("questions.0.review.metadata", {"difficulty": "easy"}),
    ],
)
def test_the_schema_accepts_what_the_format_allows(validator, path, value):
    quiz = _quiz()
    _mutate(quiz, path, value)
    validator.validate(quiz)
    validate_quiz(quiz, QUESTIONS)


TWO_OPTIONS = [_option("a", "right"), _option("b", "wrong")]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("schema_version", DROP),
        ("questions", DROP),
        ("extra", 1),
        ("schema_version", "2.0"),
        ("id", "lecture"),
        ("title", ""),
        ("description", ""),
        ("language", ""),
        ("status", "final"),
        ("questions", {}),
        ("source.questions_hash", DROP),
        ("source.question_ids", ["q_aaaaaa"]),
        ("source.questions_path", ""),
        ("source.questions_path", "/home/me/lecture.questions.json"),
        ("source.questions_path", "C:\\Users\\me\\lecture.questions.json"),
        ("source.questions_path", "\\\\server\\share\\lecture.questions.json"),
        ("source.questions_schema_version", ""),
        ("instructions.show_feedback", DROP),
        ("instructions.shuffle_options", "yes"),
        ("instructions.text", ""),
        ("instructions.extra", True),
        ("generation.prompt_version", DROP),
        ("generation.method", "magic"),
        ("generation.model", ""),
        ("generation.generated_at", "2026-10-14T09:00:00Z"),
        ("questions.0.review", DROP),
        ("questions.0.feedback", {"correct": "Well done."}),
        ("questions.0.id", "quiz_lecture"),
        ("questions.0.interaction_type", "choice_multiple"),
        ("questions.0.points", -1),
        ("questions.0.prompt", "What is a stable sort?"),
        ("questions.0.prompt.origin", DROP),
        ("questions.0.prompt.origin", "paraphrased"),
        ("questions.0.prompt.text", ""),
        ("questions.0.prompt.extra", 1),
        ("questions.0.options", TWO_OPTIONS),
        ("questions.0.options.0.text", DROP),
        ("questions.0.options.0.id", "1a"),
        ("questions.0.options.0.extra", 1),
        ("questions.0.correct_response.option_ids", ["a", "b"]),
        ("questions.0.correct_response.option_ids", []),
        ("questions.0.correct_response.option_ids", ["1a"]),
        ("questions.0.correct_response.extra", 1),
        ("questions.0.solution.explanation", DROP),
        ("questions.0.solution.extra", 1),
        ("questions.0.solution.option_explanations", {"a": _text("Yes."), "b": _text("No.")}),
        ("questions.0.solution.option_explanations.1a", _text("No.")),
        ("questions.0.solution.option_explanations.a", "Correct."),
        ("questions.0.provenance.end", DROP),
        ("questions.0.provenance.source_question_id", "lecture_1"),
        ("questions.0.provenance.start", -1),
        ("questions.0.provenance.start", True),
        ("questions.0.provenance.end", -0.5),
        ("questions.0.provenance.extra", 1),
        ("questions.0.review.status", "deleted"),
        ("questions.0.review.note", ""),
        ("questions.0.review.metadata", None),
        ("questions.0.review.extra", 1),
    ],
)
def test_the_schema_rejects_what_the_format_does_not_allow(validator, path, value):
    quiz = _quiz()
    _mutate(quiz, path, value)
    assert not validator.is_valid(quiz)


def _correct_option_missing(quiz):
    quiz["questions"][0]["correct_response"]["option_ids"] = ["d"]


def _option_without_explanation(quiz):
    quiz["questions"][0]["options"].append(_option("d", "wrong"))


def _explanation_without_option(quiz):
    quiz["questions"][0]["solution"]["option_explanations"]["d"] = _text("No such option.")


def _option_id_repeated(quiz):
    quiz["questions"][0]["options"][2]["id"] = "b"


def _id_not_derived(quiz):
    quiz["questions"][1]["id"] = "quiz_q_aaaaaa"


def _two_items_one_source(quiz):
    quiz["questions"][1]["id"] = "quiz_q_aaaaaa"
    quiz["questions"][1]["provenance"]["source_question_id"] = "q_aaaaaa"


def _source_not_in_questions(quiz):
    quiz["questions"][1]["id"] = "quiz_q_cccccc"
    quiz["questions"][1]["provenance"]["source_question_id"] = "q_cccccc"


def _end_before_start(quiz):
    quiz["questions"][0]["provenance"]["end"] = 1.0


def _spoken_stem_reworded(quiz):
    quiz["questions"][0]["prompt"]["text"] = "Which sorts are stable?"


def _spoken_stem_from_revised_text(quiz):
    quiz["questions"][1]["prompt"]["text"] = "Why use merge sort?"


@pytest.mark.parametrize(
    ("breaks", "message", "needs_questions"),
    [
        (
            _correct_option_missing,
            "questions[0].correct_response.option_ids[0]: 'd' is not an option",
            False,
        ),
        (
            _option_without_explanation,
            "questions[0].solution.option_explanations: no entry for option 'd'",
            False,
        ),
        (
            _explanation_without_option,
            "questions[0].solution.option_explanations.d: no option 'd'",
            False,
        ),
        (_option_id_repeated, "questions[0].options[2].id: 'b' repeats options[1].id", False),
        (_id_not_derived, "questions[1].id: expected 'quiz_q_bbbbbb'", False),
        (
            _two_items_one_source,
            "questions[1].provenance.source_question_id: 'q_aaaaaa' is already the source of "
            "questions[0]",
            False,
        ),
        (
            _source_not_in_questions,
            "questions[1].provenance.source_question_id: 'q_cccccc' is not in questions.json",
            True,
        ),
        (_end_before_start, "questions[0].provenance.end: 1.0 is before start 2.2", False),
        (_spoken_stem_reworded, "questions[0].prompt.text: origin 'spoken'", True),
        (_spoken_stem_from_revised_text, "questions[1].prompt.text: origin 'spoken'", True),
    ],
)
def test_validate_quiz_names_the_item_and_field(validator, breaks, message, needs_questions):
    quiz = _quiz()
    breaks(quiz)
    assert validator.is_valid(quiz)  # each rule is one JSON Schema cannot express
    with pytest.raises(SchemaError, match=re.escape(message)):
        validate_quiz(quiz, QUESTIONS)
    if needs_questions:
        validate_quiz(quiz)  # rules 5 and 7 are skipped without questions.json
    else:
        with pytest.raises(SchemaError, match=re.escape(message)):
            validate_quiz(quiz)


def test_validate_quiz_lists_every_problem_one_per_line():
    quiz = _quiz()
    _correct_option_missing(quiz)
    _id_not_derived(quiz)
    quiz["questions"][1]["provenance"]["end"] = 5.0
    with pytest.raises(SchemaError) as caught:
        validate_quiz(quiz)
    assert str(caught.value).splitlines() == [
        "questions[0].correct_response.option_ids[0]: 'd' is not an option",
        "questions[1].provenance.end: 5.0 is before start 10.0",
        "questions[1].id: expected 'quiz_q_bbbbbb' (from provenance.source_question_id), "
        "got 'quiz_q_aaaaaa'",
    ]


def test_a_spoken_stem_is_compared_with_whitespace_and_case_normalised():
    quiz = _quiz()
    quiz["questions"][0]["prompt"]["text"] = "  what is a STABLE\n sort? "
    validate_quiz(quiz, QUESTIONS)


def test_a_reworded_stem_marked_generated_passes():
    quiz = _quiz()
    quiz["questions"][1]["prompt"] = _text("Why use merge sort?")
    validate_quiz(quiz, QUESTIONS)


@pytest.mark.parametrize(
    ("path", "value", "questions"),
    [
        ("questions", None, None),
        ("questions.0", None, None),
        ("questions.0", {}, None),
        ("questions.0.options", "abc", None),
        ("questions.0.options.0.id", ["a"], None),
        ("questions.0.correct_response.option_ids", [["a"]], None),
        ("questions.0.correct_response", None, None),
        ("questions.0.solution.option_explanations", [], None),
        ("questions.0.provenance", None, None),
        ("questions.0.provenance.start", "0", None),
        ("questions.0.provenance.end", True, None),
        ("questions.0.provenance.source_question_id", 5, None),
        ("questions.0.prompt", None, {"questions": None}),
        ("title", "Sorting", {"questions": [None, {"id": 5}, {"id": "q_aaaaaa", "text": None}]}),
    ],
)
def test_validate_quiz_reports_a_broken_shape_instead_of_crashing(path, value, questions):
    quiz = _quiz()
    _mutate(quiz, path, value)
    with pytest.raises(SchemaError):
        validate_quiz(quiz, questions)


def test_validate_quiz_reports_a_document_that_is_not_an_object():
    with pytest.raises(SchemaError, match="questions: expected a list"):
        validate_quiz([])


def test_read_quiz_reads_a_valid_file(tmp_path):
    path = tmp_path / "lecture.quiz.json"
    path.write_text(json.dumps(_quiz()), encoding="utf-8")
    assert read_quiz(path) == _quiz()
    assert read_quiz(str(path), QUESTIONS) == _quiz()


def test_read_quiz_accepts_a_newer_minor_and_refuses_another_major(tmp_path):
    path = tmp_path / "quiz.json"
    quiz = _quiz()
    quiz["schema_version"] = "1.7"
    path.write_text(json.dumps(quiz), encoding="utf-8")
    assert read_quiz(path)["schema_version"] == "1.7"
    quiz["schema_version"] = "2.0"
    path.write_text(json.dumps(quiz), encoding="utf-8")
    with pytest.raises(SchemaError, match="schema version 2.0, but this questxtract reads 1.x"):
        read_quiz(path)


@pytest.mark.parametrize(
    "content", [b"not json", b"\xff\xfe", b"[]", b'"quiz"', b'{"questions": []}']
)
def test_malformed_quiz_file_raises_schema_error(tmp_path, content):
    path = tmp_path / "quiz.json"
    path.write_bytes(content)
    with pytest.raises(SchemaError):
        read_quiz(path)


def test_read_quiz_prefixes_every_problem_with_the_file(tmp_path):
    quiz = _quiz()
    _correct_option_missing(quiz)
    _source_not_in_questions(quiz)
    path = tmp_path / "quiz.json"
    path.write_text(json.dumps(quiz), encoding="utf-8")
    with pytest.raises(SchemaError) as caught:
        read_quiz(path, QUESTIONS)
    lines = str(caught.value).splitlines()
    assert len(lines) == 2
    assert all(line.startswith(f"{path}: questions[") for line in lines)


def test_the_schema_ships_in_the_package():
    assert QUIZ_SCHEMA_PATH.is_file()
    assert files("questxtract").joinpath("quiz.schema.json").is_file()

"""Tests for the ExerPlaza TeX export (issue #178)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from questxtract import TexExportError, export_tex
from questxtract.tex import quiz_to_tex

MakeQuiz = Callable[..., dict[str, Any]]

# The contract with ExerPlaza: what export_tex writes for _quiz(). Copy it into ExerPlaza's
# tests/backend/test_files/ when either side changes the format.
GOLDEN = r"""% QuestXtract quiz for ExerPlaza (IndexedQuestion format), upload at /questions/upload/tex.
% quiz: quiz_lecture, quiz.json schema 1.0
% title: Sorting
% Texts are verbatim, not LaTeX-escaped: ExerPlaza shows them with MathJax.

\begin{IndexedQuestion}
% questxtract: quiz_lecture/quiz_q_aaa
\QuestionContentUnits{Sorting}
\QuestionType{multiple choice}
\QuestionLanguage{english}
\QuestionDisclosability{only me}
\QuestionBody{Which sort is stable: 2 < 3 & 4 > 1?}
\QuestionPotentialAnswers{
\correctanswer Merge sort
\answer Quicksort
\answer Heapsort
}
\QuestionNotesForTheTeachers{QuestXtract quiz_lecture/quiz_q_aaa. Question q_aaa at 0:01-0:04 of the lecture. Stem spoken; generated text by gpt-5.6-terra, prompt v0.}
\begin{IndexedSolution}
\SolutionDisclosability{only me}
\SolutionBody{A stable sort keeps equal keys in order.

- Merge sort (correct) — Merging never swaps equal keys.
- Quicksort — Partitioning can reorder equal keys.
- Heapsort — Sifting can reorder equal keys.}
\end{IndexedSolution}
\end{IndexedQuestion}

\begin{IndexedQuestion}
% questxtract: quiz_lecture/quiz_q_bbb
\QuestionContentUnits{Sorting}
\QuestionType{multiple choice}
\QuestionLanguage{english}
\QuestionDisclosability{everybody}
\QuestionBody{Which sort is stable: 2 < 3 & 4 > 1?}
\QuestionPotentialAnswers{
\correctanswer Merge sort
\answer Quicksort
\answer Heapsort
}
\QuestionNotesForTheTeachers{QuestXtract quiz_lecture/quiz_q_bbb. Question q_bbb at 0:01-0:04 of the lecture. Stem spoken; generated text by gpt-5.6-terra, prompt v0.}
\begin{IndexedSolution}
\SolutionDisclosability{everybody}
\SolutionBody{A stable sort keeps equal keys in order.

- Merge sort (correct) — Merging never swaps equal keys.
- Quicksort — Partitioning can reorder equal keys.
- Heapsort — Sifting can reorder equal keys.}
\end{IndexedSolution}
\end{IndexedQuestion}
"""  # noqa: E501 - the file's lines are as long as its texts

# Copied from ExerPlaza's exerplaza/backend/quiz/latex_parser.py v2.1: how it finds a block and
# the answers in its \QuestionPotentialAnswers section.
_PARSER_BLOCK = re.compile(r"\\begin\{IndexedQuestion\}([\s\S]+?)\\end\{IndexedQuestion\}")
_PARSER_ANSWER = re.compile(
    r"\s*\\(correctanswer|penalizableanswer|answer)(?:\[(\-?\d+)\])?\s+"
    r"((?:(?!\\correctanswer|\\penalizableanswer|\\answer).)+)",
    re.MULTILINE,
)
_ANSWERS = re.compile(r"\\QuestionPotentialAnswers\{\n(.*?)\n\}\n", re.DOTALL)


def _quiz(make_quiz: MakeQuiz) -> dict[str, Any]:
    """A pending, an accepted and a rejected item."""
    quiz = make_quiz("q_aaa", "q_bbb", "q_ccc")
    quiz["questions"][1]["review"]["status"] = "accepted"
    quiz["questions"][2]["review"]["status"] = "rejected"
    return quiz


def _export(tmp_path: Path, quiz: dict[str, Any], *units: str, name: str = "quiz.tex") -> str:
    source = tmp_path / "quiz.json"
    source.write_text(json.dumps(quiz), encoding="utf-8")
    export_tex(source, tmp_path / name, content_units=units or ("Sorting",))
    return (tmp_path / name).read_bytes().decode("utf-8")


def _fails(tmp_path: Path, quiz: dict[str, Any], message: str, *units: str) -> str:
    """Export, expect a TexExportError matching ``message``, and check nothing was written."""
    with pytest.raises(TexExportError, match=message) as caught:
        _export(tmp_path, quiz, *units)
    assert not (tmp_path / "quiz.tex").exists()
    return str(caught.value)


def _blocks(tex: str) -> list[str]:
    return _PARSER_BLOCK.findall(tex)


def test_the_export_is_the_golden_file(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    assert _export(tmp_path, _quiz(make_quiz)) == GOLDEN


def test_rejected_items_get_no_block_and_the_rest_keep_document_order(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    quiz = _quiz(make_quiz)
    quiz["questions"].reverse()  # rejected, accepted, pending
    references = re.findall(r"^% questxtract: (\S+)$", _export(tmp_path, quiz), re.MULTILINE)
    assert references == ["quiz_lecture/quiz_q_bbb", "quiz_lecture/quiz_q_aaa"]


def test_every_block_starts_with_its_reference(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    blocks = _blocks(_export(tmp_path, _quiz(make_quiz)))
    assert [block.splitlines()[1] for block in blocks] == [
        "% questxtract: quiz_lecture/quiz_q_aaa",
        "% questxtract: quiz_lecture/quiz_q_bbb",
    ]


def test_the_review_status_sets_both_disclosabilities(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    pending, accepted = _blocks(_export(tmp_path, _quiz(make_quiz)))
    for block, expected in ((pending, "only me"), (accepted, "everybody")):
        assert f"\\QuestionDisclosability{{{expected}}}" in block
        assert f"\\SolutionDisclosability{{{expected}}}" in block


def test_an_unknown_review_status_fails_naming_it(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["review"]["status"] = "draft"
    _fails(tmp_path, quiz, r"questions\[0\]\.review\.status: expected 'pending' or 'accepted'")


@pytest.mark.parametrize(
    ("language", "expected"), [("en", "english"), ("en-GB", "english"), ("it", "italian")]
)
def test_the_language_tag_gives_exerplazas_name(
    tmp_path: Path, make_quiz: MakeQuiz, language: str, expected: str
) -> None:
    tex = _export(tmp_path, make_quiz("q_aaa", language=language))
    assert f"\\QuestionLanguage{{{expected}}}\n" in tex


def test_another_language_fails_and_writes_nothing(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    _fails(tmp_path, make_quiz("q_aaa", language="de"), r"^language: .*'de'")


def test_options_keep_their_order_and_one_line_each(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    item = quiz["questions"][0]
    item["correct_response"]["option_ids"] = ["b"]
    item["options"][2]["text"]["text"] = "Heap\nsort,  in\r\nplace"
    answers = _ANSWERS.search(_export(tmp_path, quiz))
    assert answers is not None
    assert answers.group(1).splitlines() == [
        "\\answer Merge sort",
        "\\correctanswer Quicksort",
        "\\answer Heap sort, in place",
    ]


def test_exerplazas_parser_reads_back_every_option_and_one_correct(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    quiz = _quiz(make_quiz)
    quiz["questions"][0]["options"][1]["text"]["text"] = "Quick\nsort with $\\{x\\}$"
    for block in _blocks(_export(tmp_path, quiz)):
        section = _ANSWERS.search(block)
        assert section is not None
        parsed = _PARSER_ANSWER.findall(section.group(1))
        assert [text.strip() for _, _, text in parsed] == [
            "Merge sort",
            "Quick sort with $\\{x\\}$" if "quiz_q_aaa" in block else "Quicksort",
            "Heapsort",
        ]
        assert [kind for kind, _, _ in parsed].count("correctanswer") == 1


def test_the_solution_holds_the_explanation_then_one_line_per_option(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["solution"]["explanation"]["text"] = "Stable:\r\nequal keys stay."
    quiz["questions"][0]["solution"]["option_explanations"]["b"]["text"] = "It can\nswap."
    solution = re.search(r"\\SolutionBody\{(.*?)\}\n", _export(tmp_path, quiz), re.DOTALL)
    assert solution is not None
    assert solution.group(1) == (
        "Stable:\nequal keys stay.\n\n"
        "- Merge sort (correct) — Merging never swaps equal keys.\n"
        "- Quicksort — It can swap.\n"
        "- Heapsort — Sifting can reorder equal keys."
    )


def test_the_prompt_is_verbatim_with_lf_line_endings(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["prompt"]["text"] = "Is $\\{1,2\\}$  50% countable?\r\nAnd\rwhy?"
    tex = _export(tmp_path, quiz)
    assert "\\QuestionBody{Is $\\{1,2\\}$  50% countable?\nAnd\nwhy?}\n" in tex


def _set_text(quiz: dict[str, Any], location: str, text: str) -> None:
    item = quiz["questions"][1]
    target = {
        "prompt": item["prompt"],
        "option": item["options"][2]["text"],
        "explanation": item["solution"]["explanation"],
        "option explanation": item["solution"]["option_explanations"]["b"],
    }[location]
    target["text"] = text


_PATHS = {
    "prompt": r"questions\[1\]\.prompt\.text",
    "option": r"questions\[1\]\.options\[2\]\.text",
    "explanation": r"questions\[1\]\.solution\.explanation\.text",
    "option explanation": r"questions\[1\]\.solution\.option_explanations\.b\.text",
}


@pytest.mark.parametrize("location", list(_PATHS))
@pytest.mark.parametrize("text", ["Is $\\{x\\}$ a set}?", "Is {x a set?", "Is }x{ a set?"])
def test_unbalanced_braces_fail_naming_the_text(
    tmp_path: Path, make_quiz: MakeQuiz, location: str, text: str
) -> None:
    quiz = make_quiz("q_rejected", "q_aaa")
    quiz["questions"][0]["review"]["status"] = "rejected"  # the path is the document's index
    _set_text(quiz, location, text)
    _fails(tmp_path, quiz, rf"^{_PATHS[location]}: unbalanced braces$")


@pytest.mark.parametrize("location", list(_PATHS))
@pytest.mark.parametrize(
    "command",
    [
        "\\begin{IndexedSolution}",
        "\\end{IndexedQuestion}",
        "\\QuestionBody{x}",
        "\\SolutionBody{x}",
        "\\answer",
        "\\correctanswer",
        "\\penalizableanswer",
    ],
)
def test_a_command_of_the_format_fails_naming_the_text(
    tmp_path: Path, make_quiz: MakeQuiz, location: str, command: str
) -> None:
    quiz = make_quiz("q_rejected", "q_aaa")
    quiz["questions"][0]["review"]["status"] = "rejected"
    _set_text(quiz, location, f"See {command} here")
    _fails(tmp_path, quiz, rf"^{_PATHS[location]}: .* is a command of the ExerPlaza TeX format$")


def test_a_title_can_hold_braces_but_not_a_command(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    tex = _export(tmp_path, make_quiz("q_aaa", title="Sets {1,\n 2"), name="braces.tex")
    assert "% title: Sets {1, 2\n" in tex
    _fails(tmp_path, make_quiz("q_aaa", title="\\begin{IndexedQuestion}"), r"^title: ")


@pytest.mark.parametrize(
    ("units", "message"),
    [
        ((), r"^content units: expected at least one name$"),
        (("Sorting", " "), r"^content unit ' ': expected a non-empty name$"),
        (("Sort\ning",), r"^content unit 'Sort\\ning': expected one line$"),
        (("Sort{ing}",), r"^content unit 'Sort\{ing\}': braces are not allowed$"),
    ],
)
def test_a_bad_content_unit_fails_and_writes_nothing(
    tmp_path: Path, make_quiz: MakeQuiz, units: tuple[str, ...], message: str
) -> None:
    source = tmp_path / "quiz.json"
    source.write_text(json.dumps(make_quiz("q_aaa")), encoding="utf-8")
    with pytest.raises(TexExportError, match=message):
        export_tex(source, tmp_path / "quiz.tex", content_units=units)
    assert not (tmp_path / "quiz.tex").exists()


def test_content_units_are_stripped_and_kept_in_order(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    tex = _export(tmp_path, make_quiz("q_aaa"), " Sorting ", "Stability")
    assert "\\QuestionContentUnits{Sorting}{Stability}\n" in tex


def test_a_choice_multiple_item_fails_without_mentioning_qti(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    quiz = make_quiz("q_aaa", "q_bbb", schema_version="1.1")
    quiz["questions"][1]["interaction_type"] = "choice_multiple"
    message = _fails(
        tmp_path, quiz, r"questions\[1\]\.interaction_type: 'choice_multiple' is not supported"
    )
    assert "QTI" not in message


def test_two_runs_give_the_same_bytes(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    _export(tmp_path, _quiz(make_quiz), name="a.tex")
    _export(tmp_path, _quiz(make_quiz), name="b.tex")
    data = (tmp_path / "a.tex").read_bytes()
    assert data == (tmp_path / "b.tex").read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert data.endswith(b"}\n") and not data.endswith(b"\n\n")


def test_an_empty_quiz_gives_the_header_only(make_quiz: MakeQuiz) -> None:
    tex = quiz_to_tex(make_quiz(), ["Sorting"])
    assert tex == "\n".join(GOLDEN.splitlines()[:4]) + "\n"


@pytest.mark.parametrize(
    ("generation", "by"),
    [
        ({"method": "llm", "model": "gpt-4o", "prompt_version": None}, "gpt-4o"),
        ({"method": "manual", "model": None, "prompt_version": None}, "method manual"),
    ],
)
def test_the_notes_name_the_model_or_else_the_method(
    make_quiz: MakeQuiz, generation: dict[str, Any], by: str
) -> None:
    tex = quiz_to_tex(make_quiz("q_aaa", generation=generation), ["Sorting"])
    assert f"; generated text by {by}.}}\n" in tex


@pytest.mark.parametrize(
    ("start", "end", "clock"),
    [(0, 59.99, "0:00-0:59"), (754.2, 770.9, "12:34-12:50"), (3599.5, 3725, "59:59-1:02:05")],
)
def test_the_notes_give_the_times_in_whole_seconds(
    make_quiz: MakeQuiz, start: float, end: float, clock: str
) -> None:
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["provenance"].update(start=start, end=end)
    quiz["questions"][0]["prompt"]["origin"] = "generated"
    tex = quiz_to_tex(quiz, ["Sorting"])
    assert f"Question q_aaa at {clock} of the lecture. Stem generated;" in tex


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda q: q.update(generation=None), r"^generation: expected an object$"),
        (
            lambda q: q["generation"].update(model=""),
            r"^generation\.model: expected a non-empty string or null$",
        ),
        (
            lambda q: q["generation"].update(model=None, method=None),
            r"^generation\.method: expected a non-empty string when model is null$",
        ),
        (lambda q: q["generation"].update(model="gpt}"), r"^generation\.model: unbalanced"),
        (lambda q: q.update(schema_version="1.0\n%"), r"^schema_version: expected a version"),
        (
            lambda q: q["questions"][0]["provenance"].update(start=-1),
            r"^questions\[0\]\.provenance\.start: expected >= 0",
        ),
    ],
)
def test_a_bad_value_for_the_notes_or_header_fails_naming_it(
    make_quiz: MakeQuiz, mutate: Callable[[dict[str, Any]], Any], message: str
) -> None:
    quiz = make_quiz("q_aaa")
    mutate(quiz)
    with pytest.raises(TexExportError, match=message):
        quiz_to_tex(quiz, ["Sorting"])


def test_an_unreadable_quiz_or_output_fails_readably(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    with pytest.raises(TexExportError, match="missing.json"):
        export_tex(tmp_path / "missing.json", tmp_path / "quiz.tex", content_units=["Sorting"])
    _fails(tmp_path, make_quiz("q_aaa", schema_version="2.0"), r"schema version 2\.0")
    with pytest.raises(TexExportError, match="cannot write the TeX file"):
        _export(tmp_path, make_quiz("q_aaa"), name="missing/quiz.tex")

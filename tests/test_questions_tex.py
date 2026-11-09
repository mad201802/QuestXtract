from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from questxtract import QuestionsTexError, export_questions_tex
from questxtract.cli import main
from questxtract.questions_tex import escape, questions_to_tex

TRANSCRIPT = Path(__file__).parent / "data" / "mit_18_100b_l1.transcript.json"


def _question(qid: str, text: str, **fields: Any) -> dict[str, Any]:
    question: dict[str, Any] = {
        "id": qid,
        "text": text,
        "original_text": None,
        "start": 12.0,
        "end": 15.0,
        "start_hms": "0:12",
        "end_hms": "0:15",
        "confidence": 0.82,
        "confidence_label": "high",
        "answer": None,
        "visual_ids": [],
        "review": {"status": "pending", "text": None, "note": None, "metadata": {}},
    }
    question.update(fields)
    return question


def _document(*questions: dict[str, Any], visuals: list[dict[str, Any]] | None = None) -> dict:
    return {
        "schema_version": "2.0",
        "source": {"path": "lectures/sorting_1.mp4"},
        "segments": [],
        "visuals": visuals or [],
        "questions": list(questions),
    }


def _rejected(question: dict[str, Any]) -> dict[str, Any]:
    question["review"]["status"] = "rejected"
    return question


def test_rejected_questions_are_left_out_and_the_rest_keep_their_order():
    tex = questions_to_tex(
        _document(
            _question("q1", "First question?"),
            _rejected(_question("q2", "Rejected question?")),
            _question("q3", "Third question?"),
        )
    )

    assert tex.count("\\item ") == 2
    assert tex.index("First question?") < tex.index("Third question?")
    assert "Rejected question?" not in tex
    assert "\\title{Questions asked in sorting\\_1.mp4}" in tex
    assert "2 questions, found by QuestXtract." in tex


@pytest.mark.parametrize(
    ("char", "latex"),
    [
        ("%", "\\%"),
        ("&", "\\&"),
        ("_", "\\_"),
        ("#", "\\#"),
        ("$", "\\$"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
        ("\\", "\\textbackslash{}"),
    ],
)
def test_every_special_character_is_escaped(char: str, latex: str):
    assert escape(f"a{char}b") == f"a{latex}b"
    assert f"a{latex}b" in questions_to_tex(_document(_question("q1", f"a{char}b")))


def test_characters_beyond_latin_1_become_latex_or_a_question_mark():
    assert escape("ω ≤ 2") == "\\ensuremath{\\omega} \\ensuremath{\\leq} 2"
    assert escape("“x” – ①") == "``x'' -- (1)"
    assert escape("é ü ø") == "é ü ø"  # Latin-1 passes through for inputenc
    assert escape("中 😀") == "? ?"
    assert escape("a\n  b") == "a b"


def test_an_answer_and_the_board_appear_in_their_entry_and_only_there():
    board = {
        "id": "v_1",
        "start": 10.0,
        "end": 70.0,
        "text": "stable: (3,a) (3,b)",
        "description": "two passes",
        "latex": "\\int_0^1 x\\,dx \\end{verbatim}",
    }
    answer = {"text": "Equal keys keep their order.", "start": 16.0, "end": 20.0}
    tex = questions_to_tex(
        _document(
            _question("q1", "With context?", answer=answer, visual_ids=["v_1"]),
            _question("q2", "Without context?"),
            visuals=[board],
        )
    )
    first, second = tex.split("\\item ")[1:]

    assert "\\textit{Answer} [0:16--0:20]: Equal keys keep their order.\\par" in first
    assert "\\textit{Board} [0:10--1:10]: stable: (3,a) (3,b). two passes\\par" in first
    assert "\\begin{verbatim}\n\\int_0^1 x\\,dx \\end {verbatim}\n\\end{verbatim}" in first
    assert "Answer" not in second and "Board" not in second


def test_a_reviewed_question_shows_the_teachers_text_and_notes_the_spoken_one():
    edited = _question("q1", "so why does it matter here")
    edited["review"]["text"] = "Why does stability matter?"

    entry = questions_to_tex(_document(edited)).split("\\item ")[1]

    assert "Why does stability matter?\\par" in entry
    assert "\\textit{Spoken:} so why does it matter here\\par" in entry


def test_the_time_and_confidence_head_each_entry():
    late = _question("q1", "Late?", start_hms="1:01:40", end_hms="1:01:45", confidence=0.5)
    late["confidence_label"] = "medium"

    assert "\\item \\textbf{[1:01:40--1:01:45]} \\quad confidence medium (0.50)\\par" in (
        questions_to_tex(_document(late))
    )


def test_the_same_file_gives_the_same_bytes(tmp_path: Path):
    source = tmp_path / "lecture.questions.json"
    assert main(["process", str(TRANSCRIPT), "-o", str(source)]) == 0

    count = export_questions_tex(source, tmp_path / "a.tex")
    export_questions_tex(source, tmp_path / "b.tex")

    data = (tmp_path / "a.tex").read_bytes()
    assert count > 0 and data == (tmp_path / "b.tex").read_bytes()
    assert data.startswith(b"\\documentclass") and data.endswith(b"\\end{document}\n")
    assert b"\r" not in data


@pytest.mark.parametrize("content", ["not json", '{"questions": []}', None])
def test_a_file_that_is_not_a_questions_json_fails_and_writes_nothing(
    tmp_path: Path, content: str | None
):
    source = tmp_path / "questions.json"
    if content is not None:
        source.write_text(content, encoding="utf-8")

    with pytest.raises(QuestionsTexError):
        export_questions_tex(source, tmp_path / "out.tex")
    assert not (tmp_path / "out.tex").exists()


def test_no_question_left_fails_and_writes_nothing(tmp_path: Path):
    source = tmp_path / "lecture.questions.json"
    source.write_text(json.dumps(_document(_rejected(_question("q1", "Gone?")))), "utf-8")

    with pytest.raises(QuestionsTexError, match="no question to write"):
        export_questions_tex(source, tmp_path / "out.tex")
    assert not (tmp_path / "out.tex").exists()


def test_the_command_writes_next_to_the_input_and_says_how_many(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    source = tmp_path / "lecture.questions.json"
    assert main(["process", str(TRANSCRIPT), "-o", str(source)]) == 0
    capsys.readouterr()

    assert main(["questions-tex", str(source)]) == 0

    written = tmp_path / "lecture.questions.tex"
    assert written.is_file()
    err = capsys.readouterr().err
    assert err.startswith("Questions TeX, ") and err.endswith(f"questions -> {written}\n")


def test_the_command_reports_a_bad_file_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["questions-tex", str(tmp_path / "missing.json"), "-o", "x.tex"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: questions-tex: ") and err.count("\n") == 1

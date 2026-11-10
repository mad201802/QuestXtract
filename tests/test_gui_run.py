from __future__ import annotations

import json
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from questxtract.cli import build_parser, main
from questxtract.config import Config
from questxtract.gui import run as gui
from questxtract.pipeline import Cancelled

DATA = Path(__file__).parent / "data"
TRANSCRIPT = DATA / "ttk4225_2026_09_14.transcript.json"


def _parse(argv: list[str]) -> Any:
    return build_parser().parse_args(argv)


def test_the_form_values_become_the_flags_the_cli_takes():
    argv = gui.to_argv(
        Path("lecture.mp4"),
        "questions",
        {
            "threshold": "0.6",
            "detector": "llm",
            "llm_model": "gpt-4o",
            "llm_reasoning": "server",
            "vision": True,
            "vision_model": "Qwen/Qwen3.8-27B",
            "from_scratch": True,
        },
        output_path=Path("out.json"),
    )

    args = _parse(argv)

    assert args.command == "process" and args.input == "lecture.mp4"
    assert (args.threshold, args.detector, args.llm_model) == (0.6, "llm", "gpt-4o")
    assert (args.llm_reasoning, args.vision) == ("server", True)
    assert args.vision_model == "Qwen/Qwen3.8-27B"
    assert args.from_scratch is True and args.output == "out.json"
    assert "--quiz" not in argv


def test_an_untouched_value_is_not_passed_so_the_settings_file_applies():
    argv = gui.to_argv(Path("lecture.mp4"), "questions", {"threshold": None, "llm_model": ""})

    assert argv == ["process", "lecture.mp4"]


def test_the_quiz_output_adds_the_quiz_flag_and_its_options_only_there():
    values = {"strictness": "strict", "options": 4, "quiz_model": "m"}
    quiz = gui.to_argv(Path("lecture.mp4"), "quiz", values)
    questions = gui.to_argv(Path("lecture.mp4"), "questions", values)

    assert "--quiz" in quiz and quiz[quiz.index("--strictness") + 1] == "strict"
    assert _parse(quiz).quiz is True and _parse(quiz).options == 4
    assert "--quiz" not in questions and "--strictness" not in questions


def test_the_transcript_output_takes_only_transcribe_options():
    argv = gui.to_argv(Path("lecture.mp4"), "transcript", {"model": "tiny", "detector": "llm"})

    assert argv == ["transcribe", "lecture.mp4", "--model", "tiny"]
    assert _parse(argv).command == "transcribe"


def test_an_llm_detector_without_a_model_is_refused_before_running():
    with pytest.raises(ValueError, match="needs a model name") as refused:
        gui.to_argv(Path("lecture.mp4"), "questions", {"detector": "llm", "llm_model": ""})
    assert "leave the field empty" not in str(refused.value)  # it is empty already
    assert Config().judge_model in str(refused.value)


def test_only_recordings_for_a_transcript_and_recordings_or_transcripts_for_questions():
    assert gui.source_problem(TRANSCRIPT, "questions") is None
    assert gui.source_problem(TRANSCRIPT, "transcript") == (
        f"{TRANSCRIPT}: expected an .mp4 recording or an .mp3 recording"
    )
    assert "no such file" in gui.source_problem(Path("missing.mp4"), "questions")


def test_the_app_writes_the_same_file_as_the_cli(tmp_path: Path):
    app = tmp_path / "app.questions.json"
    cli_out = tmp_path / "cli.questions.json"

    gui.run(TRANSCRIPT, "questions", {"threshold": 0.4}, output_path=app)
    assert main(["process", str(TRANSCRIPT), "--threshold", "0.4", "-o", str(cli_out)]) == 0

    assert app.read_bytes() == cli_out.read_bytes()


def test_progress_names_every_stage_and_ends_done(tmp_path: Path):
    seen: list[str] = []

    gui.run(
        TRANSCRIPT, "questions", {}, tmp_path / "q.json", progress=lambda name, _: seen.append(name)
    )

    assert seen[-1] == "done" and "sentences" in seen
    assert seen[0] != "done"


def test_cancel_stops_before_the_next_stage_and_writes_nothing(tmp_path: Path):
    output = tmp_path / "q.json"
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(Cancelled):
        gui.run(TRANSCRIPT, "questions", {}, output, stop=stop)

    assert not output.exists()


def test_a_bad_value_gives_the_cli_one_line_message(tmp_path: Path):
    with pytest.raises(ValueError) as caught:
        gui.run(TRANSCRIPT, "questions", {"threshold": 2.0}, tmp_path / "q.json")

    assert str(caught.value) == "min_confidence / --threshold must be between 0 and 1"
    gui.run(TRANSCRIPT, "questions", {"threshold": 0.4}, tmp_path / "q.json")
    assert (tmp_path / "q.json").is_file()


def test_a_missing_key_says_which_key_in_one_line(tmp_path: Path):
    with pytest.raises(ValueError) as caught:
        gui.run(TRANSCRIPT, "questions", {"detector": "llm"}, tmp_path / "q.json")

    message = str(caught.value)
    assert "API_KEY" in message and "\n" not in message


def test_the_quiz_output_writes_questions_and_quiz_next_to_each_other(
    tmp_path: Path, fake_llm: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("IDUN_API_KEY", "ntnu-test")
    fake_llm(
        {
            "quiz_material": True,
            "stem": "Which property makes a sort stable?",
            "correct_answer": "Equal keys keep their order",
            "correct_explanation": "That is the definition of stability.",
            "explanation": "Stability is about the order of equal keys.",
            "distractor_1": "It sorts in place",
            "why_wrong_1": "In place is about memory, not order.",
            "distractor_2": "It runs in linear time",
            "why_wrong_2": "Speed says nothing about stability.",
        }
    )
    output = tmp_path / "lecture.questions.json"

    result = gui.run(TRANSCRIPT, "quiz", {"llm_url": "http://localhost:11434/v1"}, output)

    assert result.path == tmp_path / "lecture.quiz.json" and result.path.is_file()
    assert output.is_file()
    assert "quiz items" in result.summary


def test_the_qti_output_exports_the_quiz_the_quiz_output_made(
    tmp_path: Path, fake_llm: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile

    monkeypatch.setenv("IDUN_API_KEY", "ntnu-test")
    fake_llm(
        {
            "quiz_material": True,
            "stem": "Which property makes a sort stable?",
            "correct_answer": "Equal keys keep their order",
            "correct_explanation": "That is the definition of stability.",
            "explanation": "Stability is about the order of equal keys.",
            "distractor_1": "It sorts in place",
            "why_wrong_1": "In place is about memory, not order.",
            "distractor_2": "It runs in linear time",
            "why_wrong_2": "Speed says nothing about stability.",
        }
    )
    quiz = gui.run(
        TRANSCRIPT,
        "quiz",
        {"llm_url": "http://localhost:11434/v1"},
        tmp_path / "lecture.questions.json",
    ).path
    package = tmp_path / "lecture.qti.zip"

    result = gui.run(quiz, "qti", {}, package)

    assert result.path == package and zipfile.is_zipfile(package)
    assert any(name.startswith("items/") for name in zipfile.ZipFile(package).namelist())


def test_an_export_takes_a_quiz_a_recording_or_a_transcript_and_nothing_else(
    tmp_path: Path, make_quiz: Callable[..., Any]
) -> None:
    quiz = tmp_path / "lecture.quiz.json"
    quiz.write_text(json.dumps(make_quiz("q_aaa")), encoding="utf-8")
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")

    for output in ("qti", "tex"):
        assert gui.source_problem(quiz, output) is None
        assert gui.source_problem(TRANSCRIPT, output) is None
        assert "expected an .mp4" in str(gui.source_problem(notes, output))
    assert gui.is_quiz(quiz) and not gui.is_quiz(TRANSCRIPT)


def test_the_qti_output_needs_a_file_to_write(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a file"):
        gui.to_argv(tmp_path / "quiz.json", "qti", {})


def test_an_export_shows_the_quiz_settings_and_only_tex_asks_for_content_units(tmp_path: Path):
    shown = {
        output: {o.dest for o in gui.OPTIONS if o.kind != "implicit" and gui.applies(o, output)}
        for output in ("quiz", "qti", "tex")
    }

    assert shown["qti"] == shown["quiz"]
    assert shown["tex"] == shown["quiz"] | {"content_units"}
    with pytest.raises(ValueError, match="between 0 and 1"):
        gui.check(TRANSCRIPT, "questions", {"threshold": 2.0}, tmp_path / "q.json")
    gui.check(TRANSCRIPT, "questions", {"threshold": 0.4}, tmp_path / "q.json")


_QUIZ_ANSWER = {
    "quiz_material": True,
    "stem": "Which property makes a sort stable?",
    "correct_answer": "Equal keys keep their order",
    "correct_explanation": "That is the definition of stability.",
    "explanation": "Stability is about the order of equal keys.",
    "distractor_1": "It sorts in place",
    "why_wrong_1": "In place is about memory, not order.",
    "distractor_2": "It runs in linear time",
    "why_wrong_2": "Speed says nothing about stability.",
}
_LOCAL = {"llm_url": "http://localhost:11434/v1"}


def _lecture(tmp_path: Path) -> Path:
    """A copy of the transcript, so the files an export writes beside it stay in tmp_path."""
    source = tmp_path / "lecture.transcript.json"
    shutil.copyfile(TRANSCRIPT, source)
    return source


def test_the_qti_output_runs_from_a_transcript_to_the_package_in_one_run(
    tmp_path: Path, fake_llm: Callable[..., Any]
) -> None:
    fake_llm(_QUIZ_ANSWER)
    package = tmp_path / "lecture.qti.zip"

    result = gui.run(_lecture(tmp_path), "qti", _LOCAL, package)

    assert result.path == package and zipfile.is_zipfile(package)
    assert (tmp_path / "lecture.questions.json").is_file()
    assert (tmp_path / "lecture.quiz.json").is_file()
    assert "quiz items" in result.summary and "QTI 3.0 package" in result.summary


def test_the_tex_output_runs_from_a_transcript_with_the_content_units(
    tmp_path: Path, fake_llm: Callable[..., Any]
) -> None:
    fake_llm(_QUIZ_ANSWER)
    tex = tmp_path / "lecture.tex"

    result = gui.run(_lecture(tmp_path), "tex", {**_LOCAL, "content_units": " A ; ;B "}, tex)

    text = tex.read_text(encoding="utf-8")
    assert result.path == tex
    assert text.count("\\begin{IndexedQuestion}") >= 1
    assert text.count("\\begin{IndexedQuestion}") == text.count("\\QuestionContentUnits{A}{B}\n")
    assert "ExerPlaza TeX" in result.summary


def test_the_tex_output_needs_a_content_unit_before_it_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one ExerPlaza content unit"):
        gui.check(_lecture(tmp_path), "tex", {"content_units": " ; "}, tmp_path / "l.tex")


def test_an_export_of_a_finished_quiz_asks_no_model(
    tmp_path: Path, make_quiz: Callable[..., Any]
) -> None:
    quiz = tmp_path / "lecture.quiz.json"
    quiz.write_text(json.dumps(make_quiz("q_aaa")), encoding="utf-8")
    tex = tmp_path / "lecture.tex"

    assert gui.to_argv(quiz, "tex", {"content_units": "A"}, tex)[0] == "tex"
    result = gui.run(quiz, "tex", {"content_units": "A"}, tex)  # no key, no fake model

    assert result.path == tex and tex.is_file()
    assert not (tmp_path / "lecture.questions.json").exists()


def test_a_stop_after_the_quiz_keeps_the_quiz_and_writes_no_export(
    tmp_path: Path, fake_llm: Callable[..., Any]
) -> None:
    fake_llm(_QUIZ_ANSWER)
    quiz = tmp_path / "lecture.quiz.json"
    package = tmp_path / "lecture.qti.zip"

    with pytest.raises(Cancelled, match="before the export"):
        gui.run(_lecture(tmp_path), "qti", _LOCAL, package, stop=quiz.exists)

    assert quiz.is_file() and not package.exists()


def test_an_export_without_a_key_stops_before_transcription_and_writes_nothing(
    tmp_path: Path,
) -> None:
    source = _lecture(tmp_path)

    with pytest.raises(ValueError, match="IDUN_API_KEY"):
        gui.run(source, "qti", {}, tmp_path / "lecture.qti.zip")

    assert sorted(path.name for path in tmp_path.iterdir()) == [source.name]


def test_the_questions_tex_box_writes_the_tex_beside_the_questions(tmp_path: Path) -> None:
    questions = tmp_path / "lecture.questions.json"

    result = gui.run(TRANSCRIPT, "questions", {"questions_tex": True}, questions)

    tex = tmp_path / "lecture.questions.tex"
    assert result.path == questions and questions.is_file() and tex.is_file()
    assert "Questions TeX" in result.summary
    assert tex.read_text(encoding="utf-8").startswith("\\documentclass")


def test_without_the_box_no_tex_is_written(tmp_path: Path) -> None:
    questions = tmp_path / "lecture.questions.json"

    gui.run(TRANSCRIPT, "questions", {"questions_tex": False}, questions)

    assert questions.is_file() and not (tmp_path / "lecture.questions.tex").exists()


def test_a_stop_during_the_last_stage_writes_no_file(tmp_path: Path):
    output = tmp_path / "q.json"
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 9

    with pytest.raises(Cancelled):
        gui.run(TRANSCRIPT, "questions", {}, output, stop=stop)

    assert not output.exists()


def test_a_bad_argument_is_a_one_line_error_and_prints_nothing(
    capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(ValueError, match="invalid float value") as caught:
        gui.check(TRANSCRIPT, "questions", {"threshold": "abc"})

    assert str(caught.value).startswith("argument -t/--threshold")
    assert capsys.readouterr() == ("", "")


def test_the_judge_model_is_passed_only_with_the_llm_detector() -> None:
    rules = gui.to_argv(TRANSCRIPT, "questions", {"llm_model": "some/model"})
    judged = gui.to_argv(TRANSCRIPT, "questions", {"detector": "llm", "llm_model": "some/model"})

    assert "--llm-model" not in rules
    assert judged[judged.index("--llm-model") + 1] == "some/model"

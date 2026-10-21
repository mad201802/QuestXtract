"""The window's logic, headless (issue #124): form values become the CLI's own run.

Uses the committed transcripts, so there is no ASR and no network. The window itself is not
tested here: these tests cover what it does with the values.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from questxtract.cli import build_parser, main
from questxtract.gui import run as gui
from questxtract.pipeline import Cancelled

DATA = Path(__file__).parent / "data"
TRANSCRIPT = DATA / "ttk4225_2026_09_14.transcript.json"


def _parse(argv: list[str]) -> Any:
    return build_parser().parse_args(argv)


def test_the_form_values_become_the_flags_the_cli_takes():
    """AC: every value the form sets is the flag the CLI reads it from, and parses back."""
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
    """AC: a control the user did not touch is left out, as on the command line."""
    argv = gui.to_argv(Path("lecture.mp4"), "questions", {"threshold": None, "llm_model": ""})

    assert argv == ["process", "lecture.mp4"]


def test_the_quiz_output_adds_the_quiz_flag_and_its_options_only_there():
    """AC: the Quiz output is `process --quiz`; quiz options appear for it and nowhere else."""
    values = {"strictness": "strict", "options": 4, "quiz_model": "m"}
    quiz = gui.to_argv(Path("lecture.mp4"), "quiz", values)
    questions = gui.to_argv(Path("lecture.mp4"), "questions", values)

    assert "--quiz" in quiz and quiz[quiz.index("--strictness") + 1] == "strict"
    assert _parse(quiz).quiz is True and _parse(quiz).options == 4
    assert "--quiz" not in questions and "--strictness" not in questions


def test_the_transcript_output_takes_only_transcribe_options():
    """AC: a transcript is `transcribe`; the LLM and vision controls do not apply to it."""
    argv = gui.to_argv(Path("lecture.mp4"), "transcript", {"model": "tiny", "detector": "llm"})

    assert argv == ["transcribe", "lecture.mp4", "--model", "tiny"]
    assert _parse(argv).command == "transcribe"


def test_an_llm_detector_without_a_model_is_refused_before_running():
    """AC: an empty model with the LLM detector is refused with a message, not sent."""
    with pytest.raises(ValueError, match="needs a model name"):
        gui.to_argv(Path("lecture.mp4"), "questions", {"detector": "llm", "llm_model": ""})


def test_only_recordings_for_a_transcript_and_recordings_or_transcripts_for_questions():
    """AC: a file of the wrong kind is refused with the CLI's own message."""
    assert gui.source_problem(TRANSCRIPT, "questions") is None
    assert gui.source_problem(TRANSCRIPT, "transcript") == (
        f"{TRANSCRIPT}: expected an .mp4 recording or an .mp3 recording"
    )
    assert "no such file" in gui.source_problem(Path("missing.mp4"), "questions")


def test_the_app_writes_the_same_file_as_the_cli(tmp_path: Path):
    """AC: the same options give byte-identical questions.json from the app and the CLI."""
    app = tmp_path / "app.questions.json"
    cli_out = tmp_path / "cli.questions.json"

    gui.run(TRANSCRIPT, "questions", {"threshold": 0.4}, output_path=app)
    assert main(["process", str(TRANSCRIPT), "--threshold", "0.4", "-o", str(cli_out)]) == 0

    assert app.read_bytes() == cli_out.read_bytes()


def test_progress_names_every_stage_and_ends_done(tmp_path: Path):
    """AC: the window's progress comes from the pipeline's own callback."""
    seen: list[str] = []

    gui.run(
        TRANSCRIPT, "questions", {}, tmp_path / "q.json", progress=lambda name, _: seen.append(name)
    )

    assert seen[-1] == "done" and "sentences" in seen  # the pipeline's stage names
    assert seen[0] != "done"  # the first report is the first stage, not the end


def test_cancel_stops_before_the_next_stage_and_writes_nothing(tmp_path: Path):
    """AC: Cancel takes effect between stages; a stopped run leaves no file behind."""
    output = tmp_path / "q.json"
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2  # let two stages start, then ask to stop

    with pytest.raises(Cancelled):
        gui.run(TRANSCRIPT, "questions", {}, output, stop=stop)

    assert not output.exists()


def test_a_bad_value_gives_the_cli_one_line_message(tmp_path: Path):
    """AC: errors are the CLI's one-line text, never a traceback, and the run can be retried."""
    with pytest.raises(ValueError) as caught:
        gui.run(TRANSCRIPT, "questions", {"threshold": 2.0}, tmp_path / "q.json")

    assert str(caught.value) == "min_confidence / --threshold must be between 0 and 1"
    gui.run(TRANSCRIPT, "questions", {"threshold": 0.4}, tmp_path / "q.json")  # fix and run again
    assert (tmp_path / "q.json").is_file()


def test_a_missing_key_says_which_key_in_one_line(tmp_path: Path):
    """AC: with no key for the endpoint the message names the variable; no key value shows."""
    with pytest.raises(ValueError) as caught:
        gui.run(TRANSCRIPT, "questions", {"detector": "llm"}, tmp_path / "q.json")

    message = str(caught.value)
    assert "API_KEY" in message and "\n" not in message  # OPENAI on main, NTNU after #174


def test_the_quiz_output_writes_questions_and_quiz_next_to_each_other(
    tmp_path: Path, fake_llm: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
):
    """AC: Quiz gives the same two files `process --quiz` writes, and the window gets the quiz."""
    monkeypatch.setenv("QUESTXTRACT_NTNU_API_KEY", "ntnu-test")
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

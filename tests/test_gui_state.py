"""The graphical app's form state, headless (issue #124): values, enabled controls, progress.

No tkinter widgets here, so these run in CI on every platform. The one tkinter use is the
drop-path split, which needs only a bare Tcl interpreter and no display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from questxtract.gui import state
from questxtract.gui.options import OPTIONS, OUTPUTS, Option

DATA = Path(__file__).parent / "data"
TRANSCRIPT = DATA / "ttk4225_2026_09_14.transcript.json"


def _option(dest: str) -> Option:
    (option,) = [o for o in OPTIONS if o.dest == dest]
    return option


def test_a_control_at_its_default_is_not_passed():
    """AC: a setting left at its Config default is None, so questxtract.toml still applies."""
    raw: dict[str, Any] = {
        "model": "turbo",  # the Config default, as the entry starts
        "threshold": "0.4",
        "detector": "rules",
        "from_scratch": False,
        "vision": "Default",
        "output": "",
        "config": "",
    }

    values = state.form_values(raw)

    assert values["model"] is None
    assert values["threshold"] is None
    assert values["detector"] is None
    assert values["from_scratch"] is None
    assert values["vision"] is None
    assert values["output"] is None and values["config"] is None


def test_a_changed_control_is_passed_as_the_user_typed_it():
    """AC: a value the user changed reaches the CLI; tri-state and checkbox map to flags."""
    raw: dict[str, Any] = {
        "model": "tiny",
        "threshold": " 0.6 ",
        "detector": "llm",
        "from_scratch": True,
        "vision": "On",
        "vision_reader": "ocr",
        "config": "C:/Users/Ana Lu/questxtract.toml",
    }

    values = state.form_values(raw)

    assert values["model"] == "tiny"
    assert values["threshold"] == "0.6"
    assert values["detector"] == "llm"
    assert values["from_scratch"] is True
    assert values["vision"] is True
    assert values["vision_reader"] == "ocr"
    assert values["config"] == "C:/Users/Ana Lu/questxtract.toml"


def test_an_empty_llm_model_stays_empty_so_the_run_refuses_it():
    """AC: an empty model is passed as "", not replaced by the default, so it is refused."""
    values = state.form_values({"llm_model": ""})

    assert values["llm_model"] == ""


def test_the_quiz_flag_is_not_a_control_and_is_not_in_the_values():
    """The implicit --quiz option is the Quiz output's job, not a form field."""
    values = state.form_values({"quiz": True, "threshold": "0.5"})

    assert "quiz" not in values


def test_a_transcript_is_not_offered_for_a_transcript_input():
    """AC: given a .json transcript, the transcript output is not offered."""
    assert state.output_offered("transcript", TRANSCRIPT) is False
    assert state.output_offered("questions", TRANSCRIPT) is True
    assert state.output_offered("transcript", Path("lecture.mp4")) is True
    assert state.output_offered("transcript", None) is True


def test_qti_needs_a_default_file_next_to_the_quiz_and_the_others_do_not():
    """AC: QTI writes a .qti.zip beside the quiz.json; the other outputs let the CLI choose."""
    quiz = Path("out") / "lecture.quiz.json"

    assert state.default_output(quiz, "qti") == Path("out") / "lecture.qti.zip"
    assert state.default_output(quiz, "questions") is None
    assert state.default_output(None, "qti") is None


def test_run_is_disabled_until_a_source_is_chosen():
    """AC: Run needs a file; the window shows why it is not enabled."""
    problem = state.run_problem(None, "questions", {}, None)

    assert problem is not None and "Drop a recording" in problem


def test_a_wrong_file_shows_the_cli_message_and_run_stays_disabled():
    """AC: a file of the wrong kind gives the CLI's own message, in one line."""
    problem = state.run_problem(TRANSCRIPT, "transcript", {}, None)

    assert problem == f"{TRANSCRIPT}: expected an .mp4 recording or an .mp3 recording"


def test_a_threshold_out_of_range_is_refused_before_running():
    """AC: a threshold outside 0 to 1 is refused with a message, before anything runs."""
    problem = state.run_problem(TRANSCRIPT, "questions", {"threshold": "2"}, None)

    assert problem == "min_confidence / --threshold must be between 0 and 1"


def test_an_llm_detector_without_a_model_is_refused_before_running():
    """AC: an empty LLM model with the llm detector is refused, with a message."""
    problem = state.run_problem(TRANSCRIPT, "questions", {"detector": "llm", "llm_model": ""}, None)

    assert problem is not None and "needs a model name" in problem


def test_a_valid_run_has_no_problem():
    """AC: with a valid source and values, Run is enabled (no problem text)."""
    assert state.run_problem(TRANSCRIPT, "questions", {"threshold": "0.4"}, None) is None


def test_a_bad_settings_file_is_a_one_line_problem(tmp_path: Path):
    """A --config that does not exist is shown in the window, not raised."""
    problem = state.run_problem(
        TRANSCRIPT, "questions", {"config": str(tmp_path / "missing.toml")}, None
    )

    assert problem is not None and "\n" not in problem


def test_one_line_joins_a_message_so_the_window_never_shows_a_second_line():
    """AC: the status line is one line of text."""
    assert state.one_line("first\n\n  second \n") == "first second"
    assert state.one_line("   \n") == "the run failed"


def test_progress_moves_through_the_stages_and_ends_at_one():
    """AC: the bar follows the pipeline's own progress names, from zero to done."""
    fractions = [
        state.overall_fraction("ingest", 0.0),
        state.overall_fraction("transcribe", 0.5),
        state.overall_fraction("score", 1.0),
        state.overall_fraction("quiz", 1.0),
        state.overall_fraction("done", 1.0),
    ]

    assert fractions == sorted(fractions)
    assert fractions[0] == pytest.approx(0.0)
    assert fractions[-1] == 1.0
    assert all(0.0 <= value <= 1.0 for value in fractions)


def test_an_unknown_progress_name_does_not_move_the_bar():
    """A stage the window does not know yet counts as the start of the run, never as an error."""
    assert state.overall_fraction("some_new_stage", 0.7) == 0.0


def test_stage_labels_are_the_cli_words_and_unknown_names_are_readable():
    """AC: the window words a stage as the CLI's progress bar does."""
    assert state.stage_label("transcribe") == "Transcribing"
    assert state.stage_label("done") == "Done"
    assert state.stage_label("some_new_stage") == "Some new stage"


def test_the_hint_names_the_default_the_form_starts_from():
    """AC: the form shows each option's default from Config, not a number of its own."""
    assert state.hint(_option("model")) == "turbo"
    assert state.hint(_option("threshold")) == "0.4"
    assert state.hint(_option("from_scratch")) == "off"
    assert state.hint(_option("output")) == "next to the input"
    assert state.hint(_option("llm_reasoning")) == "server"  # the judge sends no setting (#179)


def test_every_output_has_a_label_and_a_command():
    """The radio buttons and the command table name the same outputs."""
    assert tuple(state.OUTPUT_LABELS) == tuple(OUTPUTS)


def test_a_drop_with_spaces_in_the_path_is_one_path():
    """AC: a path with spaces, braced by Tk in a drop, is read as one file."""
    tkinter = pytest.importorskip("tkinter")
    interpreter = tkinter.Tcl()
    data = "{/Users/Ana Lu/lecture one.mp4} /tmp/plain.mp3"

    paths = state.dropped_paths(data, interpreter.splitlist)

    assert paths == [Path("/Users/Ana Lu/lecture one.mp4"), Path("/tmp/plain.mp3")]


def test_open_folder_uses_the_platform_opener(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """AC: Open folder uses the system's opener: open on macOS, xdg-open on Linux."""
    calls: list[list[str]] = []
    monkeypatch.setattr(state.subprocess, "Popen", lambda argv: calls.append(argv))

    monkeypatch.setattr(state.sys, "platform", "darwin")
    state.open_folder(tmp_path)
    monkeypatch.setattr(state.sys, "platform", "linux")
    state.open_folder(tmp_path)

    assert calls == [["open", str(tmp_path)], ["xdg-open", str(tmp_path)]]


def test_the_quiz_output_does_not_reach_the_end_when_the_pipeline_is_done():
    """The pipeline's `done` comes before the quiz; with the Quiz output the bar waits there."""
    quiz_start = state.overall_fraction("done", 1.0, quiz_follows=True)

    assert 0.0 < quiz_start < 1.0
    assert state.overall_fraction("done", 1.0) == 1.0
    assert state.overall_fraction("quiz", 0.0) == pytest.approx(quiz_start)
    assert state.overall_fraction("quiz", 1.0) == 1.0

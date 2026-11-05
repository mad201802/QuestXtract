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
    raw: dict[str, Any] = {
        "model": "turbo",
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
    values = state.form_values({"llm_model": ""})

    assert values["llm_model"] == ""


def test_the_quiz_flag_is_not_a_control_and_is_not_in_the_values():
    values = state.form_values({"quiz": True, "threshold": "0.5"})

    assert "quiz" not in values


def test_a_transcript_is_not_offered_for_a_transcript_input():
    assert state.output_offered("transcript", TRANSCRIPT) is False
    assert state.output_offered("questions", TRANSCRIPT) is True
    assert state.output_offered("transcript", Path("lecture.mp4")) is True
    assert state.output_offered("transcript", None) is True


def test_qti_needs_a_default_file_next_to_the_quiz_and_the_others_do_not():
    quiz = Path("out") / "lecture.quiz.json"

    assert state.default_output(quiz, "qti") == Path("out") / "lecture.qti.zip"
    assert state.default_output(quiz, "questions") is None
    assert state.default_output(None, "qti") is None


def test_run_is_disabled_until_a_source_is_chosen():
    problem = state.run_problem(None, "questions", {}, None)

    assert problem is not None and "Drop a recording" in problem


def test_a_wrong_file_shows_the_cli_message_and_run_stays_disabled():
    problem = state.run_problem(TRANSCRIPT, "transcript", {}, None)

    assert problem == f"{TRANSCRIPT}: expected an .mp4 recording or an .mp3 recording"


def test_a_threshold_out_of_range_is_refused_before_running():
    problem = state.run_problem(TRANSCRIPT, "questions", {"threshold": "2"}, None)

    assert problem == "min_confidence / --threshold must be between 0 and 1"


def test_an_llm_detector_without_a_model_is_refused_before_running():
    problem = state.run_problem(TRANSCRIPT, "questions", {"detector": "llm", "llm_model": ""}, None)

    assert problem is not None and "needs a model name" in problem


def test_a_valid_run_has_no_problem():
    assert state.run_problem(TRANSCRIPT, "questions", {"threshold": "0.4"}, None) is None


def test_a_bad_settings_file_is_a_one_line_problem(tmp_path: Path):
    problem = state.run_problem(
        TRANSCRIPT, "questions", {"config": str(tmp_path / "missing.toml")}, None
    )

    assert problem is not None and "\n" not in problem


def test_one_line_joins_a_message_so_the_window_never_shows_a_second_line():
    assert state.one_line("first\n\n  second \n") == "first second"
    assert state.one_line("   \n") == "the run failed"


def test_progress_moves_through_the_stages_and_ends_at_one():
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
    assert state.overall_fraction("some_new_stage", 0.7) == 0.0


def test_stage_labels_are_the_cli_words_and_unknown_names_are_readable():
    assert state.stage_label("transcribe") == "Transcribing"
    assert state.stage_label("done") == "Done"
    assert state.stage_label("some_new_stage") == "Some new stage"


def test_the_hint_names_the_default_the_form_starts_from():
    assert state.hint(_option("model")) == "turbo"
    assert state.hint(_option("threshold")) == "0.4"
    assert state.hint(_option("from_scratch")) == "off"
    assert state.hint(_option("output")) == "next to the input"
    assert state.hint(_option("llm_reasoning")) == "server"


def test_every_output_has_a_label_and_a_command():
    assert tuple(state.OUTPUT_LABELS) == tuple(OUTPUTS)


def test_a_drop_with_spaces_in_the_path_is_one_path():
    tkinter = pytest.importorskip("tkinter")
    interpreter = tkinter.Tcl()
    data = "{/Users/Ana Lu/lecture one.mp4} /tmp/plain.mp3"

    paths = state.dropped_paths(data, interpreter.splitlist)

    assert paths == [Path("/Users/Ana Lu/lecture one.mp4"), Path("/tmp/plain.mp3")]


def test_open_folder_uses_the_platform_opener(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[list[str]] = []
    monkeypatch.setattr(state.subprocess, "Popen", lambda argv: calls.append(argv))

    monkeypatch.setattr(state.sys, "platform", "darwin")
    state.open_folder(tmp_path)
    monkeypatch.setattr(state.sys, "platform", "linux")
    state.open_folder(tmp_path)

    assert calls == [["open", str(tmp_path)], ["xdg-open", str(tmp_path)]]


def test_the_quiz_output_does_not_reach_the_end_when_the_pipeline_is_done():
    quiz_start = state.overall_fraction("done", 1.0, quiz_follows=True)

    assert 0.0 < quiz_start < 1.0
    assert state.overall_fraction("done", 1.0) == 1.0
    assert state.overall_fraction("quiz", 0.0) == pytest.approx(quiz_start)
    assert state.overall_fraction("quiz", 1.0) == 1.0


def test_the_window_starts_from_its_start_values_without_a_settings_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert state.start_text(_option("detector")) == "llm"
    assert state.start_tristate(_option("vision")) == "On"
    assert state.hint(_option("detector")) == "llm"


def test_a_settings_file_decides_the_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "questxtract.toml").write_text('detector = "rules"\n', encoding="utf-8")

    assert state.start_text(_option("detector")) == "rules"
    assert state.start_tristate(_option("vision")) == "Default"
    assert state.form_values({"detector": "rules"})["detector"] is None


def test_the_key_window_follows_the_endpoint_a_run_would_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    chosen = tmp_path / "openrouter.toml"
    chosen.write_text('base_url = "openrouter"\n', encoding="utf-8")
    ntnu, openrouter = "https://llm.hpc.ntnu.no/v1", "https://openrouter.ai/api/v1"

    assert state.key_endpoint({"llm_url": None, "config": None}) == ntnu
    assert state.key_endpoint({"llm_url": None, "config": str(chosen)}) == openrouter
    assert state.key_endpoint({"llm_url": "ollama", "config": str(chosen)}) == (
        "http://localhost:11434/v1"
    )
    Path("questxtract.toml").write_text('base_url = "openai"\n', encoding="utf-8")
    assert state.key_endpoint({"llm_url": None, "config": None}) == "https://api.openai.com/v1"
    missing = str(tmp_path / "missing.toml")
    assert state.key_endpoint({"llm_url": None, "config": missing}) == ntnu


def test_no_reasoning_setting_shows_as_server_and_is_not_passed(tmp_path, monkeypatch):
    """#259: the Reasoning box was empty; "server" is the default and passes no flag."""
    monkeypatch.chdir(tmp_path)
    option = next(o for o in OPTIONS if o.dest == "llm_reasoning")
    assert state.start_text(option) == "server"
    assert state.form_values({"llm_reasoning": "server"})["llm_reasoning"] is None

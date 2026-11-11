from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from questxtract.cli import main
from questxtract.gui import i18n, state
from questxtract.gui import run as gui_run
from questxtract.gui.options import OPTIONS, WHISPER_MODELS
from questxtract.pipeline import Cancelled

gui_window = pytest.importorskip("questxtract.gui.window")
tkinter = pytest.importorskip("tkinter")

DATA = Path(__file__).parent / "data"
TRANSCRIPT = DATA / "ttk4225_2026_09_14.transcript.json"


def _has_display() -> bool:
    return sys.platform != "linux" or bool(os.environ.get("DISPLAY"))


pytestmark = pytest.mark.skipif(
    not _has_display(),
    reason="no display here: the window is checked where a display exists",
)


@pytest.fixture(autouse=True)
def english() -> Iterator[None]:
    i18n.set_language("en")
    yield
    i18n.set_language("en")


@pytest.fixture(autouse=True)
def errors(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record the error pop-ups instead of opening them, so no test waits for a click."""
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gui_window.messagebox,
        "showerror",
        lambda title, message, **_: shown.append((title, message)),
    )
    return shown


@pytest.fixture
def root() -> Iterator[Any]:
    try:
        top = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"no display could be opened: {exc}")
    yield top
    with contextlib.suppress(tkinter.TclError):
        top.destroy()


@pytest.fixture
def window(root: Any) -> gui_window.QuestXtractWindow:
    window = gui_window.QuestXtractWindow(root, drag_and_drop=False)
    window.fields["detector"].set("rules")
    window.fields["vision"].set(i18n.tr("tristate.default"))
    return window


def _pump(root: Any, done: Callable[[], bool], timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while not done():
        root.update()
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the run to end")
        time.sleep(0.02)


def _enabled(button: Any) -> bool:
    return not button.instate(["disabled"])


def test_the_form_has_a_control_for_every_option_the_table_shows(window):
    assert set(window.fields) == {o.dest for o in OPTIONS if o.kind != "implicit"}


def test_run_is_disabled_until_a_valid_file_is_chosen(window):
    assert not _enabled(window.run_button)
    assert "Drop a recording" in window.hint.cget("text")


def test_a_transcript_enables_run_and_the_transcript_output_is_not_offered(window):
    window.choose_source(TRANSCRIPT)

    assert _enabled(window.run_button)
    assert not _enabled(window.output_buttons["transcript"])
    assert window.hint.cget("text") == ""


def test_a_wrong_file_shows_the_cli_message_in_the_window(window, tmp_path: Path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a lecture", encoding="utf-8")

    window.choose_source(notes)

    assert not _enabled(window.run_button)
    assert "expected an .mp4 recording or an .mp3 recording" in window.hint.cget("text")


def test_a_threshold_out_of_range_is_refused_before_running(window):
    window.choose_source(TRANSCRIPT)
    window.fields["threshold"].set("2")
    window._refresh()

    assert not _enabled(window.run_button)
    assert "between 0 and 1" in window.hint.cget("text")


def test_the_output_choice_shows_only_the_options_that_output_takes(window):
    shown = lambda dest: bool(window.option_rows[dest].grid_info())  # noqa: E731

    window.output.set("questions")
    window._refresh()
    assert shown("threshold") and not shown("strictness")

    window.output.set("transcript")
    window._refresh()
    assert shown("model") and not shown("threshold")

    window.output.set("quiz")
    window._refresh()
    assert shown("strictness") and shown("threshold")

    window.output.set("qti")
    window._refresh()
    assert shown("strictness") and shown("threshold") and not shown("content_units")

    window.output.set("tex")
    window._refresh()
    assert shown("strictness") and shown("content_units")


def test_a_run_writes_the_file_the_cli_writes_and_shows_its_path(
    window, root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    opened: list[Path] = []
    monkeypatch.setattr(state, "open_folder", opened.append)
    app = tmp_path / "app.questions.json"
    cli_out = tmp_path / "cli.questions.json"
    window.choose_source(TRANSCRIPT)
    window.fields["output"].set(str(app))

    window._run()
    assert not _enabled(window.run_button)
    _pump(root, lambda: not window.busy)

    assert main(["process", str(TRANSCRIPT), "-o", str(cli_out)]) == 0
    assert app.read_bytes() == cli_out.read_bytes()
    assert str(app) in window.status.cget("text")
    assert window.progress_var.get() == pytest.approx(100.0)
    window._open_folder()
    assert opened == [tmp_path]


def test_cancel_stops_the_run_between_stages_and_says_so(
    window, root, monkeypatch: pytest.MonkeyPatch
):

    def stoppable(*_: Any, progress: Any = None, stop: Callable[[], bool], **__: Any) -> None:
        progress("ingest", 0.0)
        while not stop():
            time.sleep(0.01)
        raise Cancelled("stopped before transcribe")

    monkeypatch.setattr(gui_run, "run", stoppable)
    window.choose_source(TRANSCRIPT)

    window._run()
    window._cancel()
    _pump(root, lambda: not window.busy)

    assert "Cancelled. No partial file was written." in window.status.cget("text")
    assert not _enabled(window.cancel_button)
    assert _enabled(window.run_button)


def test_an_error_is_one_line_and_the_run_can_be_fixed_and_run_again(
    window, root, errors, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    window.choose_source(TRANSCRIPT)
    window.fields["output"].set(str(tmp_path / "q.json"))
    window.fields["detector"].set("llm")

    window._run()
    _pump(root, lambda: not window.busy)

    ((title, message),) = errors  # #259: a pop-up, not text at the bottom
    assert title == i18n.tr("error.title")
    assert "API_KEY" in message and "\n" not in message and "Traceback" not in message
    window.fields["detector"].set("rules")
    window._run()
    _pump(root, lambda: not window.busy)
    assert (tmp_path / "q.json").is_file()


def test_the_drop_box_takes_the_first_file_of_a_braced_drop(window, tmp_path: Path):
    folder = tmp_path / "Ana Lu"
    folder.mkdir()
    copy = folder / "lecture one.transcript.json"
    shutil.copy(TRANSCRIPT, copy)
    data = f"{{{copy}}} {{{TRANSCRIPT}}}"

    window._on_drop(SimpleNamespace(data=data, action="copy"))

    assert window.source == copy
    assert _enabled(window.run_button)


def test_without_drag_and_drop_the_window_says_so(window):
    assert "not available" in window.drop_note.cget("text")


def test_the_window_opens_on_the_root_it_is_given(root):
    gui_window.QuestXtractWindow(root, drag_and_drop=False)

    assert root.title() == "QuestXtract"


def test_closing_during_a_run_stops_it_and_then_closes_the_window(
    root, monkeypatch: pytest.MonkeyPatch
):
    stopped: list[bool] = []

    def stoppable(*_: Any, progress: Any = None, stop: Callable[[], bool], **__: Any) -> None:
        progress("transcribe", 0.2)
        while not stop():
            time.sleep(0.01)
        stopped.append(True)
        raise Cancelled("stopped before sentences")

    monkeypatch.setattr(gui_run, "run", stoppable)
    window = gui_window.QuestXtractWindow(root, drag_and_drop=False)
    window.choose_source(TRANSCRIPT)
    window._run()

    window._close()
    _pump(root, lambda: not window.alive)

    assert stopped == [True]


def test_the_language_switch_relabels_the_window_and_keeps_the_form(window, root):
    window.choose_source(TRANSCRIPT)
    window.fields["threshold"].set("0.5")
    window.fields["detector"].set("llm")
    window.language.set("nb")

    window._switch_language()

    assert window.run_button.cget("text") == "Kjør"
    assert window.output_buttons["qti"].cget("text") == "QTI-pakke"
    assert window.fields["threshold"].get() == "0.5"
    assert window.fields["detector"].get() == "llm"
    assert tr_shown(window, "vision") == "Standard"
    assert window._values()["vision"] is None
    assert "(standard: på)" in window.option_labels["vision"][0].cget("text")


def test_a_run_in_norsk_finishes_in_norsk(window, root, tmp_path: Path):
    window.choose_source(TRANSCRIPT)
    window.fields["output"].set(str(tmp_path / "q.json"))
    window.language.set("it")
    window._switch_language()

    window._run()
    _pump(root, lambda: not window.busy)

    assert window.stage_label.cget("text") == "Fatto"
    assert window.status.cget("text").startswith("Scritto: ")
    assert (tmp_path / "q.json").is_file()


def tr_shown(window: Any, dest: str) -> str:
    return str(window.fields[dest].get())


def test_the_start_view_is_simple_and_the_options_are_behind_advanced_settings(window):
    assert not window.options_frame.winfo_manager()
    assert window.advanced_button.cget("text").startswith("Advanced settings")

    window._toggle_advanced()
    assert window.options_frame.winfo_manager() == "grid"
    shown = window.advanced_button.cget("text")

    window._toggle_advanced()
    assert not window.options_frame.winfo_manager()
    assert shown != window.advanced_button.cget("text")


def test_reading_the_board_is_on_the_start_view_not_behind_advanced_settings(window):
    window.output.set("questions")
    window._refresh()

    assert not window.options_frame.winfo_manager()
    assert window.option_rows["vision"].master is not window.options_frame
    assert window.option_rows["vision"].grid_info()


def test_the_window_starts_with_the_llm_judge_and_reading_the_board_on(root):
    window = gui_window.QuestXtractWindow(root, drag_and_drop=False)

    assert window.fields["detector"].get() == "llm"
    assert tr_shown(window, "vision") == "On"
    values = window._values()
    assert values["detector"] == "llm" and values["vision"] is True


def test_a_language_switch_keeps_the_advanced_view_open(window):
    window._toggle_advanced()
    window.language.set("it")

    window._switch_language()

    assert window.options_frame.winfo_manager() == "grid"
    assert window.advanced_button.cget("text").startswith("Impostazioni avanzate")


def test_cancel_can_be_clicked_while_a_run_is_busy_and_not_before_or_after(
    window, root, monkeypatch: pytest.MonkeyPatch
):

    def stoppable(*_: Any, progress: Any = None, stop: Callable[[], bool], **__: Any) -> None:
        progress("ingest", 0.0)
        while not stop():
            time.sleep(0.01)
        raise Cancelled("stopped before transcribe")

    monkeypatch.setattr(gui_run, "run", stoppable)
    window.choose_source(TRANSCRIPT)
    assert not _enabled(window.cancel_button)

    window._run()
    assert _enabled(window.cancel_button)
    window.fields["threshold"].set("0.5")
    assert _enabled(window.cancel_button)

    window.cancel_button.invoke()
    assert not _enabled(window.cancel_button)
    window.fields["threshold"].set("0.6")
    assert not _enabled(window.cancel_button)
    _pump(root, lambda: not window.busy)
    assert not _enabled(window.cancel_button) and _enabled(window.run_button)


def test_the_quiz_output_keeps_the_bar_below_the_end_until_the_quiz_is_done(window, root):
    window.run_output = "quiz"
    window._handle("progress", ("score", 1.0))
    window._handle("progress", ("done", 1.0))

    assert window.progress_var.get() < 100
    assert window.stage_label.cget("text") != "Done"
    window.run_output = "questions"
    window._handle("progress", ("done", 1.0))
    assert window.progress_var.get() == 100


def test_the_note_about_a_drop_of_many_files_goes_with_the_next_file(window, tmp_path: Path):
    copy = tmp_path / "again.transcript.json"
    shutil.copy(TRANSCRIPT, copy)

    window._on_drop(SimpleNamespace(data=f"{{{copy}}} {{{TRANSCRIPT}}}", action="copy"))
    assert "One file at a time" in window.drop_note.cget("text")
    window.language.set("it")
    window._switch_language()
    assert "Un file alla volta" in window.drop_note.cget("text")

    window.choose_source(TRANSCRIPT)
    assert "alla volta" not in window.drop_note.cget("text")
    assert "non disponibile" in window.drop_note.cget("text")


def test_a_failed_drag_and_drop_start_leaves_one_root_and_binds_the_variables_to_it(
    monkeypatch: pytest.MonkeyPatch,
):
    import types

    class Failing(tkinter.Tk):
        def __init__(self) -> None:
            super().__init__()
            raise tkinter.TclError("cannot load tkdnd")

    fake = types.ModuleType("tkinterdnd2")
    fake.TkinterDnD = types.SimpleNamespace(Tk=Failing)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tkinterdnd2", fake)

    root, drag_and_drop = gui_window._new_root()
    try:
        assert drag_and_drop is False
        assert tkinter._default_root is root
        var = tkinter.StringVar()
        assert var._root is root
    finally:
        root.destroy()


def _counting_checks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = state.run_problem

    def counted(*args: Any, **kwargs: Any) -> str | None:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(state, "run_problem", counted)
    return calls


def test_typing_is_checked_once_after_a_pause(window, monkeypatch):
    window.choose_source(TRANSCRIPT)
    calls = _counting_checks(monkeypatch)

    for text in ("0", "0.", "0.5"):
        window.fields["threshold"].set(text)
    assert calls == []

    time.sleep(gui_window._TYPING_PAUSE_MS / 1000 + 0.1)
    window.root.update()
    assert len(calls) == 1


def test_an_unchanged_form_is_not_checked_again(window, monkeypatch):
    window.choose_source(TRANSCRIPT)
    calls = _counting_checks(monkeypatch)

    window._refresh()
    window._refresh()
    assert calls == []


def test_run_checks_a_value_typed_just_before_it(window):
    window.choose_source(TRANSCRIPT)
    window.fields["threshold"].set("2")

    window._run()

    assert not window.busy
    assert "between 0 and 1" in window.hint.cget("text")


def test_a_new_output_does_not_reuse_the_file_typed_for_the_last(window, tmp_path):
    window.choose_source(TRANSCRIPT)
    questions = tmp_path / "lecture.questions.json"
    window.fields["output"].set(str(questions))
    assert window._output_path() == questions

    window.output.set("qti")
    assert window.fields["output"].get() == ""
    assert window._output_path() != questions


def test_the_tex_output_waits_for_a_content_unit_and_offers_a_tex_file(
    window, monkeypatch, tmp_path, make_quiz
):
    quiz = tmp_path / "lecture.quiz.json"
    quiz.write_text(json.dumps(make_quiz("q_aaa")), encoding="utf-8")
    window.choose_source(quiz)
    window.output.set("tex")
    window._refresh()

    assert not _enabled(window.run_button)
    assert "content unit" in window.hint.cget("text")
    assert window._output_path() == tmp_path / "lecture.tex"

    window.fields["content_units"].set("Sorting; Stability")
    window._refresh()
    assert _enabled(window.run_button)

    asked: dict[str, Any] = {}
    monkeypatch.setattr(
        gui_window.filedialog, "asksaveasfilename", lambda **kwargs: asked.update(kwargs) or ""
    )
    output = next(option for option in OPTIONS if option.dest == "output")
    window._browse_into(output, window.fields["output"])
    assert asked["defaultextension"] == ".tex"


def test_the_questions_tex_box_is_for_the_questions_output_only(window):
    window.choose_source(TRANSCRIPT)
    window.questions_tex.set(True)

    window.output.set("questions")
    window._refresh()
    assert _enabled(window.questions_tex_box) and window._values()["questions_tex"] is True

    window.output.set("quiz")
    window._refresh()
    assert not _enabled(window.questions_tex_box) and window._values()["questions_tex"] is False


def test_browse_for_the_qti_package_suggests_a_zip(window, monkeypatch):
    asked: dict[str, Any] = {}
    monkeypatch.setattr(
        gui_window.filedialog, "asksaveasfilename", lambda **kwargs: asked.update(kwargs) or ""
    )
    output = next(option for option in OPTIONS if option.dest == "output")

    window.output.set("qti")
    window._browse_into(output, window.fields["output"])

    assert asked["defaultextension"] == ".zip"


def test_a_settings_file_decides_where_the_form_starts(root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "questxtract.toml").write_text('detector = "rules"\n', encoding="utf-8")

    window = gui_window.QuestXtractWindow(root, drag_and_drop=False)

    assert window.fields["detector"].get() == "rules"
    assert window._values()["detector"] is None
    assert window._values()["vision"] is None


def test_the_judge_model_is_open_only_for_the_llm_detector(window):
    window.fields["llm_model"].set("some/model")
    window.fields["detector"].set("rules")
    window._refresh()
    assert not _enabled(window.controls["llm_model"])
    assert "--llm-model" not in gui_run.to_argv(TRANSCRIPT, "questions", window._values())

    window.fields["detector"].set("llm")
    window._refresh()
    assert _enabled(window.controls["llm_model"])
    assert "--llm-model" in gui_run.to_argv(TRANSCRIPT, "questions", window._values())


def _key_dialog(root: Any, monkeypatch: pytest.MonkeyPatch, answer: str, **kwargs: Any):
    from questxtract.gui import key_dialog, keys

    monkeypatch.setattr(keys, "check_key", lambda key, url: answer)
    return key_dialog.KeyDialog(root, **kwargs)


def test_the_window_has_a_key_button_that_opens_the_key_window(window, monkeypatch):
    opened: list[object] = []
    monkeypatch.setattr(gui_window, "KeyDialog", lambda root, **kwargs: opened.append(kwargs))

    window._open_key_dialog()

    assert len(opened) == 1 and opened[0]["on_change"] == window._refresh
    assert opened[0]["url"] == "https://llm.hpc.ntnu.no/v1"
    assert any(
        str(widget.cget("text")) == i18n.tr("button.api_key")
        for widget, key in window.static_texts
        if key == "button.api_key"
    )


def test_the_key_window_is_for_the_endpoint_in_the_llm_endpoint_field(window, monkeypatch):
    opened: list[dict[str, Any]] = []
    monkeypatch.setattr(gui_window, "KeyDialog", lambda root, **kwargs: opened.append(kwargs))
    assert tuple(window.controls["llm_url"].cget("values")) == (
        "ntnu",
        "openrouter",
        "openai",
        "ollama",
    )

    window.fields["llm_url"].set("openrouter")
    window._open_key_dialog()
    window.fields["llm_url"].set("http://localhost:8000/v1")
    window._open_key_dialog()

    assert [o["url"] for o in opened] == [
        "https://openrouter.ai/api/v1",
        "http://localhost:8000/v1",
    ]


def test_with_the_field_at_its_default_the_key_window_follows_the_settings_file(
    window, monkeypatch, tmp_path
):
    opened: list[dict[str, Any]] = []
    monkeypatch.setattr(gui_window, "KeyDialog", lambda root, **kwargs: opened.append(kwargs))
    chosen = tmp_path / "settings.toml"
    chosen.write_text('base_url = "openrouter"\n', encoding="utf-8")

    window.fields["config"].set(str(chosen))
    window._open_key_dialog()

    assert opened[0]["url"] == "https://openrouter.ai/api/v1"


def test_the_ntnu_key_window_lists_the_steps_and_opens_the_request_page(root, monkeypatch):
    from questxtract.config import NTNU_KEY_PAGE
    from questxtract.gui import key_dialog

    opened: list[str] = []
    dialog = _key_dialog(root, monkeypatch, "accepted")
    monkeypatch.setattr(key_dialog.webbrowser, "open", opened.append)

    assert dialog.steps is not None and dialog.link.cget("text") == NTNU_KEY_PAGE
    texts = [
        child.cget("text") for child in dialog.steps.winfo_children() if "text" in child.keys()
    ]
    assert texts[0] == i18n.tr("key.steps_title")
    assert any("sk-" in text for text in texts)
    dialog.open_key_page()
    assert opened == [NTNU_KEY_PAGE]


def test_a_pasted_ntnu_key_without_sk_gets_a_note(root, monkeypatch):
    dialog = _key_dialog(root, monkeypatch, "accepted")

    dialog.key.set("abc123")
    assert dialog.prefix_note.cget("text") == i18n.tr("key.prefix_hint")
    dialog.key.set("  sk-abc123 ")
    assert dialog.prefix_note.cget("text") == ""


def test_another_endpoint_gets_no_ntnu_steps(root, monkeypatch):
    dialog = _key_dialog(root, monkeypatch, "accepted", url="https://openrouter.ai/api/v1")

    assert dialog.steps is None
    dialog.key.set("abc123")
    assert dialog.prefix_note.cget("text") == ""


def test_the_key_window_saves_openrouters_key_as_its_own(root, monkeypatch):
    from questxtract.config import Config, save_user_key, user_env_path

    save_user_key("ntnu-saved")
    dialog = _key_dialog(root, monkeypatch, "accepted", url="https://openrouter.ai/api/v1")
    assert dialog.top.title() == "API key: OpenRouter"
    dialog.key.set("sk-or-typed")

    dialog.save()
    _pump(root, lambda: not dialog.busy, timeout=5)

    assert dialog.message.cget("text") == "Key saved. OpenRouter accepted it."
    assert Config(base_url="openrouter").api_key == "sk-or-typed"
    assert Config().api_key == "ntnu-saved"
    assert 'OPENROUTER_API_KEY="sk-or-typed"' in user_env_path().read_text(encoding="utf-8")


def test_save_in_the_key_window_stores_the_key_and_empties_the_field(root, monkeypatch):
    from questxtract.config import Config

    dialog = _key_dialog(root, monkeypatch, "accepted")
    dialog.key.set("ntnu-typed")

    dialog.save()
    _pump(root, lambda: not dialog.busy, timeout=5)

    assert Config().api_key == "ntnu-typed"
    assert dialog.message.cget("text") == i18n.tr("key.saved", endpoint="NTNU")
    assert dialog.key.get() == ""
    assert dialog.source.cget("text") == i18n.tr("key.source.file")


def test_a_refused_key_is_not_saved_and_says_so(root, monkeypatch):
    from questxtract.config import Config

    dialog = _key_dialog(root, monkeypatch, "refused")
    dialog.key.set("wrong")

    dialog.save()
    _pump(root, lambda: not dialog.busy, timeout=5)

    assert Config().api_key is None
    assert dialog.message.cget("text") == i18n.tr("key.refused", endpoint="NTNU")


def test_test_only_tests_and_an_unusable_key_is_told_at_once(root, monkeypatch):
    from questxtract.config import Config

    dialog = _key_dialog(root, monkeypatch, "unreachable")
    dialog.key.set("has space")
    dialog.test()
    assert "key" in dialog.message.cget("text") and not dialog.busy

    dialog.key.set("fine")
    dialog.test()
    _pump(root, lambda: not dialog.busy, timeout=5)

    assert dialog.message.cget("text") == i18n.tr("key.unreachable", endpoint="NTNU")
    assert Config().api_key is None


def test_remove_in_the_key_window_removes_the_saved_key(root, monkeypatch):
    from questxtract.config import Config

    dialog = _key_dialog(root, monkeypatch, "accepted")
    dialog.key.set("ntnu-typed")
    dialog.save()
    _pump(root, lambda: not dialog.busy, timeout=5)

    dialog.remove()
    assert dialog.message.cget("text") == i18n.tr("key.removed")
    assert Config().api_key is None
    dialog.remove()
    assert dialog.message.cget("text") == i18n.tr("key.nothing_to_remove")


def test_the_settings_open_below_run_so_run_stays_in_view(window):
    """#259: opening the settings pushed Run off a small screen."""
    window._toggle_advanced()
    run_row = window.run_button.master.grid_info()["row"]
    assert int(run_row) < int(window.options_frame.grid_info()["row"])


def test_the_window_never_grows_past_the_screen_with_the_settings_open(window, root):
    """#259: the longest list of settings (the quiz's) scrolls instead."""
    window.output.set("quiz")
    window._toggle_advanced()
    root.update()
    window._fit_options()
    root.update()
    assert root.winfo_reqheight() <= root.winfo_screenheight()


def test_the_whisper_model_is_picked_from_a_list(window):
    """#259: a typed name was only checked once transcription started."""
    control = window.controls["model"]
    assert str(control.cget("state")) == "readonly"
    assert tuple(str(v) for v in control.cget("values")) == WHISPER_MODELS


def test_each_output_is_explained_in_plain_words_and_in_each_language(window):
    """#259: the outputs were four bare names."""
    shown = set()
    for name in state.OUTPUT_LABELS:
        window.output.set(name)
        window._refresh()
        shown.add(window.output_about.cget("text"))
    assert len(shown) == len(state.OUTPUT_LABELS) and "" not in shown
    window.language.set("nb")
    window._switch_language()
    assert window.output_about.cget("text") == i18n.tr(f"about.{window.output.get()}")


def test_the_drop_area_shows_the_chosen_file_and_keeps_the_path_on_hover(window):
    """#259: the drop area kept saying "Drop a file" after a file was chosen."""
    assert window.drop_box.cget("text") == i18n.tr("drop.help")
    window.choose_source(TRANSCRIPT)
    assert TRANSCRIPT.name in window.drop_box.cget("text")
    assert window.source_tip.text == str(TRANSCRIPT)
    window.language.set("it")
    window._switch_language()
    assert window.drop_box.cget("text") == i18n.tr("drop.chosen", name=TRANSCRIPT.name)


def test_the_window_and_the_key_window_stay_inside_the_screen(window, root):
    """#259: with the settings open, the window ran past the bottom of a laptop screen."""
    root.geometry(f"+0+{root.winfo_screenheight() - 200}")
    window._toggle_advanced()
    root.update()
    window._show_options()
    root.update()
    for shown in (root, gui_window.KeyDialog(root).top):
        shown.update()
        assert shown.winfo_y() >= 0
        assert shown.winfo_y() + shown.winfo_height() <= shown.winfo_screenheight()
        assert shown.winfo_x() + shown.winfo_width() <= shown.winfo_screenwidth()


def test_an_output_without_settings_says_so_at_the_top(window, root, tmp_path, make_quiz):
    """#259: QTI showed an empty settings box with its content floating in the middle."""
    quiz = tmp_path / "lecture.quiz.json"
    quiz.write_text(json.dumps(make_quiz("q_aaa")), encoding="utf-8")
    window.choose_source(quiz)  # a finished quiz: the export makes no quiz, so no settings
    window._toggle_advanced()
    for name, empty in (("qti", True), ("questions", False)):
        window.output.set(name)
        window._refresh()
        root.update()
        window._fit_options()
        root.update()
        assert bool(window.no_settings.winfo_ismapped()) is empty
        assert window.options_canvas.yview()[0] == 0.0
        assert float(str(window.options_canvas.cget("scrollregion")).split()[1]) == 0.0


def test_switching_the_output_never_resizes_the_window(window, root):
    """#259: each output's explanation and settings made the window jump."""
    window.choose_source(TRANSCRIPT)
    for advanced in (False, True):
        if advanced:
            window._toggle_advanced()
        sizes = set()
        for name in state.OUTPUT_LABELS:
            window.output.set(name)
            window._refresh()
            root.update()
            window._fit_options()
            root.update()
            sizes.add((root.winfo_reqwidth(), root.winfo_reqheight()))
        assert len(sizes) == 1, sizes


def test_enter_starts_a_run_and_escape_closes_the_settings(window, root, monkeypatch):
    """#259: the keyboard could not start a run or close the settings."""
    runs: list[None] = []
    monkeypatch.setattr(window, "_run", lambda: runs.append(None))
    window._toggle_advanced()
    window.run_button.focus_force()
    root.update()

    window.run_button.event_generate("<Return>", when="now")
    window.run_button.event_generate("<Escape>", when="now")

    assert runs == [None]
    assert not window.advanced_open


def test_the_settings_scroll_with_the_wheel_on_linux_too(window, root):
    """#259: on Linux the wheel sends Button-4 and Button-5, which nothing was bound to."""
    window._toggle_advanced()
    root.update()
    canvas = window.options_canvas

    canvas.event_generate("<Enter>", when="now")
    assert all(canvas.bind_all(name) for name in ("<MouseWheel>", "<Button-4>", "<Button-5>"))
    canvas.event_generate("<Leave>", when="now")
    assert not canvas.bind_all("<Button-5>")

"""The graphical app's window on a real display (issue #124).

Skipped where there is no display: CI on Linux is headless, so there the window is only imported.
The logic the window drives is tested headless in ``test_gui_state.py`` and ``test_gui_run.py``.
Run these locally with a display (or ``xvfb-run -a pytest tests/test_gui_window.py``).
"""

from __future__ import annotations

import contextlib
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
from questxtract.gui.options import OPTIONS
from questxtract.pipeline import Cancelled

# The window imports tkinter, so this module is skipped (with the reason) where Tk is missing.
gui_window = pytest.importorskip("questxtract.gui.window")
tkinter = pytest.importorskip("tkinter")

DATA = Path(__file__).parent / "data"
TRANSCRIPT = DATA / "ttk4225_2026_09_14.transcript.json"


def _has_display() -> bool:
    return sys.platform != "linux" or bool(os.environ.get("DISPLAY"))


pytestmark = pytest.mark.skipif(
    not _has_display(),
    reason="no display here (headless CI): the window is checked where a display exists",
)


@pytest.fixture(autouse=True)
def english() -> Iterator[None]:
    """Each window test starts in English, and a language switch does not leak to the next."""
    i18n.set_language("en")
    yield
    i18n.set_language("en")


@pytest.fixture
def root() -> Iterator[Any]:
    """A Tk root for one test; a Tk that cannot open skips the test with its reason."""
    try:
        top = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"no display could be opened: {exc}")
    yield top
    with contextlib.suppress(tkinter.TclError):  # a test may have closed the window already
        top.destroy()


@pytest.fixture
def window(root: Any) -> gui_window.QuestXtractWindow:
    """The window with no drag and drop, as on a platform where it cannot start.

    Set back to the CLI's offline defaults: the runs here must not call a model.
    """
    window = gui_window.QuestXtractWindow(root, drag_and_drop=False)
    window.fields["detector"].set("rules")
    window.fields["vision"].set(i18n.tr("tristate.default"))
    return window


def _pump(root: Any, done: Callable[[], bool], timeout: float = 120.0) -> None:
    """Let Tk run its events (the window polls the worker) until ``done`` or the timeout."""
    deadline = time.monotonic() + timeout
    while not done():
        root.update()
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the run to end")
        time.sleep(0.02)


def _enabled(button: Any) -> bool:
    return not button.instate(["disabled"])


def test_the_form_has_a_control_for_every_option_the_table_shows(window):
    """AC: the form is built from OPTIONS only; no widget is hand-copied."""
    assert set(window.fields) == {o.dest for o in OPTIONS if o.kind != "implicit"}


def test_run_is_disabled_until_a_valid_file_is_chosen(window):
    """AC: with no file Run is disabled and the window says what to do."""
    assert not _enabled(window.run_button)
    assert "Drop a recording" in window.hint.cget("text")


def test_a_transcript_enables_run_and_the_transcript_output_is_not_offered(window):
    """AC: a .json transcript enables Run; the transcript output is disabled for it."""
    window.choose_source(TRANSCRIPT)

    assert _enabled(window.run_button)
    assert not _enabled(window.output_buttons["transcript"])
    assert window.hint.cget("text") == ""


def test_a_wrong_file_shows_the_cli_message_in_the_window(window, tmp_path: Path):
    """AC: a file with another extension keeps Run disabled and shows the CLI's own words."""
    notes = tmp_path / "notes.txt"
    notes.write_text("not a lecture", encoding="utf-8")

    window.choose_source(notes)

    assert not _enabled(window.run_button)
    assert "expected an .mp4 recording or an .mp3 recording" in window.hint.cget("text")


def test_a_threshold_out_of_range_is_refused_before_running(window):
    """AC: a threshold outside 0 to 1 disables Run and says why."""
    window.choose_source(TRANSCRIPT)
    window.fields["threshold"].set("2")
    window._refresh()  # the checks wait for typing to pause; a test does not wait for that

    assert not _enabled(window.run_button)
    assert "between 0 and 1" in window.hint.cget("text")


def test_the_output_choice_shows_only_the_options_that_output_takes(window):
    """AC: each output shows its own controls: the quiz options for Quiz, none for QTI."""
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
    assert not any(shown(dest) for dest in window.option_rows)


def test_a_run_writes_the_file_the_cli_writes_and_shows_its_path(
    window, root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """AC: the window's run is the CLI's run: same bytes, the path shown, Open folder enabled."""
    opened: list[Path] = []
    monkeypatch.setattr(state, "open_folder", opened.append)
    app = tmp_path / "app.questions.json"
    cli_out = tmp_path / "cli.questions.json"
    window.choose_source(TRANSCRIPT)
    window.fields["output"].set(str(app))

    window._run()
    assert not _enabled(window.run_button)  # disabled while the run is busy
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
    """AC: Cancel asks the run to stop; the window ends with a Cancelled line, no file."""

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
    window, root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """AC: a failure shows the one-line message (no traceback); fixing the input runs again."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    window.choose_source(TRANSCRIPT)
    window.fields["output"].set(str(tmp_path / "q.json"))
    window.fields["detector"].set("llm")

    window._run()
    _pump(root, lambda: not window.busy)

    status = window.status.cget("text")
    assert "API_KEY" in status and "\n" not in status and "Traceback" not in status
    window.fields["detector"].set("rules")
    window._run()
    _pump(root, lambda: not window.busy)
    assert (tmp_path / "q.json").is_file()


def test_the_drop_box_takes_the_first_file_of_a_braced_drop(window, tmp_path: Path):
    """AC: a dropped path with spaces, braced by Tk, is read as one file."""
    folder = tmp_path / "Ana Lu"
    folder.mkdir()
    copy = folder / "lecture one.transcript.json"
    shutil.copy(TRANSCRIPT, copy)
    data = f"{{{copy}}} {{{TRANSCRIPT}}}"

    window._on_drop(SimpleNamespace(data=data, action="copy"))

    assert window.source == copy
    assert _enabled(window.run_button)


def test_without_drag_and_drop_the_window_says_so(window):
    """AC: where drop cannot start, the window says it is not available; Browse still works."""
    assert "not available" in window.drop_note.cget("text")


def test_the_window_opens_on_the_root_it_is_given(root):
    """The window is built on the root it is given and its title is the app's name."""
    gui_window.QuestXtractWindow(root, drag_and_drop=False)

    assert root.title() == "QuestXtract"


def test_closing_during_a_run_stops_it_and_then_closes_the_window(
    root, monkeypatch: pytest.MonkeyPatch
):
    """Closing mid-run asks the run to stop, so its temporary files are removed; then it closes."""
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
    """AC: the switch shows the window in another language and keeps what the user entered."""
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
    assert "(standard: av)" in window.option_labels["vision"][0].cget("text")


def test_a_run_in_norsk_finishes_in_norsk(window, root, tmp_path: Path):
    """The status line and the stage name change language while the run is shown."""
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
    """Return the text a tri-state control shows now."""
    return str(window.fields[dest].get())


def test_the_start_view_is_simple_and_the_options_are_behind_advanced_settings(window):
    """AC: at start only the essentials show; the options form is one click away."""
    assert not window.options_frame.winfo_manager()  # hidden
    assert window.advanced_button.cget("text").startswith("Advanced settings")

    window._toggle_advanced()
    assert window.options_frame.winfo_manager() == "grid"
    shown = window.advanced_button.cget("text")

    window._toggle_advanced()
    assert not window.options_frame.winfo_manager()
    assert shown != window.advanced_button.cget("text")


def test_reading_the_board_is_on_the_start_view_not_behind_advanced_settings(window):
    """The vision choice is outside the hidden options form and follows the output choice."""
    window.output.set("questions")
    window._refresh()

    assert not window.options_frame.winfo_manager()
    assert window.option_rows["vision"].master is not window.options_frame
    assert window.option_rows["vision"].grid_info()


def test_the_window_starts_with_the_llm_judge_and_reading_the_board_on(root):
    """The start values: the LLM detector and vision on, so Run passes both flags."""
    window = gui_window.QuestXtractWindow(root, drag_and_drop=False)

    assert window.fields["detector"].get() == "llm"
    assert tr_shown(window, "vision") == "On"
    values = window._values()
    assert values["detector"] == "llm" and values["vision"] is True


def test_a_language_switch_keeps_the_advanced_view_open(window):
    """The switch re-labels the Advanced settings button without closing the form."""
    window._toggle_advanced()
    window.language.set("it")

    window._switch_language()

    assert window.options_frame.winfo_manager() == "grid"
    assert window.advanced_button.cget("text").startswith("Impostazioni avanzate")


def test_cancel_can_be_clicked_while_a_run_is_busy_and_not_before_or_after(
    window, root, monkeypatch: pytest.MonkeyPatch
):
    """The Cancel button is enabled during a run, disabled once asked, and again at the end."""

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
    window.fields["threshold"].set("0.5")  # a refresh while busy must not change it
    assert _enabled(window.cancel_button)

    window.cancel_button.invoke()
    assert not _enabled(window.cancel_button)
    window.fields["threshold"].set("0.6")  # ...nor bring it back once Cancel was asked
    assert not _enabled(window.cancel_button)
    _pump(root, lambda: not window.busy)
    assert not _enabled(window.cancel_button) and _enabled(window.run_button)


def test_the_quiz_output_keeps_the_bar_below_the_end_until_the_quiz_is_done(window, root):
    """The pipeline's `done` is not the end of a Quiz run: the bar and label wait for the quiz."""
    window.run_output = "quiz"
    window._handle("progress", ("score", 1.0))
    window._handle("progress", ("done", 1.0))

    assert window.progress_var.get() < 100
    assert window.stage_label.cget("text") != "Done"
    window.run_output = "questions"
    window._handle("progress", ("done", 1.0))
    assert window.progress_var.get() == 100


def test_the_note_about_a_drop_of_many_files_goes_with_the_next_file(window, tmp_path: Path):
    """The 'one file at a time' note is about one drop: the next file does not keep it."""
    copy = tmp_path / "again.transcript.json"
    shutil.copy(TRANSCRIPT, copy)

    window._on_drop(SimpleNamespace(data=f"{{{copy}}} {{{TRANSCRIPT}}}", action="copy"))
    assert "One file at a time" in window.drop_note.cget("text")
    window.language.set("it")
    window._switch_language()
    assert "Un file alla volta" in window.drop_note.cget("text")

    window.choose_source(TRANSCRIPT)
    assert "alla volta" not in window.drop_note.cget("text")
    assert "non disponibile" in window.drop_note.cget("text")  # unavailable stays, in Italian


def test_a_failed_drag_and_drop_start_leaves_one_root_and_binds_the_variables_to_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """AC: when tkdnd will not load, the half-built root is removed and the plain one is used."""
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
        assert tkinter._default_root is root  # the orphan is gone; this is the only root
        var = tkinter.StringVar()
        assert var._root is root
    finally:
        root.destroy()


def _counting_checks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the window's calls to the run check."""
    calls: list[int] = []
    real = state.run_problem

    def counted(*args: Any, **kwargs: Any) -> str | None:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(state, "run_problem", counted)
    return calls


def test_typing_is_checked_once_after_a_pause(window, monkeypatch):
    """Keystrokes in a row cost one check, after typing pauses, not one per key."""
    window.choose_source(TRANSCRIPT)
    calls = _counting_checks(monkeypatch)

    for text in ("0", "0.", "0.5"):
        window.fields["threshold"].set(text)
    assert calls == []  # nothing checked while typing

    time.sleep(gui_window._TYPING_PAUSE_MS / 1000 + 0.1)
    window.root.update()
    assert len(calls) == 1


def test_an_unchanged_form_is_not_checked_again(window, monkeypatch):
    """A refresh with the same form, file and settings reuses the last check."""
    window.choose_source(TRANSCRIPT)
    calls = _counting_checks(monkeypatch)

    window._refresh()
    window._refresh()
    assert calls == []  # choose_source already checked this form


def test_run_checks_a_value_typed_just_before_it(window):
    """Run during the typing pause re-checks, so a bad value never starts a run."""
    window.choose_source(TRANSCRIPT)
    window.fields["threshold"].set("2")  # its check has not run yet

    window._run()

    assert not window.busy
    assert "between 0 and 1" in window.hint.cget("text")


def test_a_new_output_does_not_reuse_the_file_typed_for_the_last(window, tmp_path):
    """A path typed for Questions is not written by Transcript or QTI."""
    window.choose_source(TRANSCRIPT)
    questions = tmp_path / "lecture.questions.json"
    window.fields["output"].set(str(questions))
    assert window._output_path() == questions

    window.output.set("qti")
    assert window.fields["output"].get() == ""
    assert window._output_path() != questions


def test_browse_for_the_qti_package_suggests_a_zip(window, monkeypatch):
    """The save dialog names the ZIP for QTI, not a JSON file."""
    asked: dict[str, Any] = {}
    monkeypatch.setattr(
        gui_window.filedialog, "asksaveasfilename", lambda **kwargs: asked.update(kwargs) or ""
    )
    output = next(option for option in OPTIONS if option.dest == "output")

    window.output.set("qti")
    window._browse_into(output, window.fields["output"])

    assert asked["defaultextension"] == ".zip"

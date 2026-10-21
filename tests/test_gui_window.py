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
    """The window with no drag and drop, as on a platform where it cannot start."""
    return gui_window.QuestXtractWindow(root, drag_and_drop=False)


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

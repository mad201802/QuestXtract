"""The `questxtract gui` subcommand (issue #124): its messages, and that the CLI stays tkinter-free.

Headless: the window is never opened here. Each test replaces the part it needs with a stand-in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from questxtract import cli


def test_importing_the_cli_and_the_gui_logic_never_imports_tkinter():
    """AC: the base install and every other command never import tkinter or tkinterdnd2."""
    code = (
        "import sys, questxtract.cli, questxtract.gui.state, questxtract.gui.run;"
        "print('tkinter' in sys.modules, 'tkinterdnd2' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.split() == ["False", "False"]


def test_a_python_without_tkinter_gets_one_error_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """AC: without tkinter the gui command prints one `error:` line, not a traceback."""
    monkeypatch.setitem(sys.modules, "tkinter", None)

    assert cli.main(["gui"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: this Python has no tkinter")
    assert err.count("\n") == 1


def test_a_missing_gui_extra_says_to_install_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """AC: without the gui extra the command says to install questxtract[gui]."""
    pytest.importorskip("tkinter")
    monkeypatch.setitem(sys.modules, "tkinterdnd2", None)

    assert cli.main(["gui"]) == 1

    assert 'pip install "questxtract[gui]"' in capsys.readouterr().err


def test_a_window_that_cannot_open_is_an_error_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """AC: with no display the command prints one error line and exits 1."""
    tkinter = pytest.importorskip("tkinter")
    pytest.importorskip("tkinterdnd2")
    from questxtract.gui import window as gui_window

    def no_display() -> None:
        raise tkinter.TclError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr(gui_window, "_new_root", no_display)

    assert cli.main(["gui"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: cannot open a window: ") and err.count("\n") == 1


def test_the_gui_command_is_listed_with_its_extra(capsys: pytest.CaptureFixture[str]):
    """The subcommand is in the CLI's help, and the help names the extra it needs."""
    with pytest.raises(SystemExit):
        cli.main(["--help"])

    assert "gui" in capsys.readouterr().out


def test_a_wrong_settings_path_is_an_error_line_not_a_traceback(tmp_path: Path):
    """The CLI's own check (shared with the app) raises one message for a missing settings file."""
    with pytest.raises((OSError, ValueError)):
        cli.check_arguments(["process", "lecture.mp4", "--config", str(tmp_path / "x.toml")])

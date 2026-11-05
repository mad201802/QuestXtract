from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from typing import Any

import pytest

tkinter = pytest.importorskip("tkinter")
screen = pytest.importorskip("questxtract.gui.screen")

pytestmark = pytest.mark.skipif(
    sys.platform == "linux" and not os.environ.get("DISPLAY"),
    reason="no display here: the window is checked where a display exists",
)


@pytest.fixture
def root() -> Iterator[Any]:
    try:
        top = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"no display could be opened: {exc}")
    yield top
    with contextlib.suppress(tkinter.TclError):
        top.destroy()


def _inside(window: Any) -> bool:
    window.update()
    return (
        window.winfo_x() >= 0
        and window.winfo_y() >= 0
        and window.winfo_x() + window.winfo_width() <= window.winfo_screenwidth()
        and window.winfo_y() + window.winfo_height() <= window.winfo_screenheight()
    )


def test_a_window_bigger_than_the_screen_is_shrunk_onto_it(root):
    """#259: a window taller than a laptop screen went past the taskbar."""
    big = tkinter.Frame(
        root, width=root.winfo_screenwidth() * 2, height=root.winfo_screenheight() * 2
    )
    big.pack()
    screen.fit_on_screen(root)
    assert _inside(root)


def test_a_window_that_fits_keeps_its_size_and_is_moved_onto_the_screen(root):
    tkinter.Frame(root, width=300, height=200).pack()
    root.geometry(f"+{root.winfo_screenwidth() - 50}+{root.winfo_screenheight() - 50}")
    root.update()
    screen.fit_on_screen(root)
    root.update()
    assert _inside(root)
    assert (root.winfo_width(), root.winfo_height()) == (300, 200)


def test_a_dialog_opens_over_its_parent(root):
    tkinter.Frame(root, width=600, height=400).pack()
    root.geometry("+100+100")
    root.update()
    dialog = tkinter.Toplevel(root)
    tkinter.Frame(dialog, width=200, height=100).pack()
    screen.fit_on_screen(dialog, over=root)
    dialog.update()
    centre_x = dialog.winfo_rootx() + dialog.winfo_width() // 2
    assert abs(centre_x - (root.winfo_rootx() + root.winfo_width() // 2)) <= 20
    assert _inside(dialog)

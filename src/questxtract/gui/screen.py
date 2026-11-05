"""Keep every window of the app inside the screen (#259)."""

from __future__ import annotations

import tkinter as tk

__all__ = ["fit_on_screen"]

_MARGIN_PX = 90  # room for the taskbar and the window's border


def fit_on_screen(window: tk.Tk | tk.Toplevel, *, over: tk.Misc | None = None) -> None:
    """Move ``window``, and shrink it only if it is too big, so all of it is on the screen.

    ``over`` centres it on that window first, as for a dialog.
    """
    window.geometry("")  # back to the size its content asks for
    window.update_idletasks()
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight() - _MARGIN_PX
    width = min(window.winfo_reqwidth(), screen_w)
    height = min(window.winfo_reqheight(), screen_h)
    if over is not None:
        x = over.winfo_rootx() + (over.winfo_width() - width) // 2
        y = over.winfo_rooty() + (over.winfo_height() - height) // 2
    else:
        x, y = window.winfo_x(), window.winfo_y()
    x = max(0, min(x, screen_w - width))
    y = max(0, min(y, screen_h - height))
    too_big = (width, height) != (window.winfo_reqwidth(), window.winfo_reqheight())
    size = f"{width}x{height}" if too_big else ""  # a fitting window keeps growing freely
    window.geometry(f"{size}+{x}+{y}")

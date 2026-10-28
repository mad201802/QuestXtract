"""The NTNU key window (issue #189): paste a key, test it, save it or remove it.

The logic is :mod:`.keys`; this module only draws it. The key is read from the field when a button
is pressed and is never shown, logged or put in a message. The test asks NTNU's service on a
worker thread; the main thread reads its answer with ``after()``, as the main window does.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from . import keys
from .i18n import tr

__all__ = ["KeyDialog"]

_POLL_MS = 100


class KeyDialog:
    """A modal window over ``parent``: a masked field, Test, Save, Remove and Close."""

    def __init__(self, parent: tk.Misc, *, on_change: Callable[[], None] | None = None) -> None:
        """Open the window; ``on_change`` is called after a key was saved or removed."""
        self.on_change = on_change
        self.answers: queue.Queue[str] = queue.Queue()
        self.busy = False
        self.alive = True
        self.top = tk.Toplevel(parent)
        self.top.title(tr("key.title"))
        self.top.transient(parent)  # type: ignore[call-overload]
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.top.grab_set()  # modal: the main window waits until this one is closed
        frame = ttk.Frame(self.top, padding=8)
        frame.grid(sticky="nsew")
        ttk.Label(frame, text=tr("key.explain"), wraplength=420).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(frame, text=tr("key.label")).grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.key = tk.StringVar(master=self.top, value="")
        self.entry = ttk.Entry(frame, textvariable=self.key, show="*", width=48)
        self.entry.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(frame, text=tr("key.file", path=keys.key_file()), wraplength=420).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        self.source = ttk.Label(frame, text="", wraplength=420)
        self.source.grid(row=3, column=0, columnspan=2, sticky="w")
        self.message = ttk.Label(frame, text="", wraplength=420)
        self.message.grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=2, sticky="e", pady=(8, 0))
        self.test_button = ttk.Button(buttons, text=tr("button.key_test"), command=self.test)
        self.save_button = ttk.Button(buttons, text=tr("button.key_save"), command=self.save)
        remove = ttk.Button(buttons, text=tr("button.key_remove"), command=self.remove)
        close = ttk.Button(buttons, text=tr("button.key_close"), command=self.close)
        for column, button in enumerate((self.test_button, self.save_button, remove, close)):
            button.grid(row=0, column=column, padx=(6, 0))
        frame.columnconfigure(1, weight=1)
        self._show_source()
        self.entry.focus_set()
        self.top.after(_POLL_MS, self._poll)

    # --- actions -----------------------------------------------------------------------

    def test(self) -> None:
        """Ask NTNU whether the typed key works; save nothing."""
        self._ask(keys.check_key)

    def save(self) -> None:
        """Test the typed key, then save it unless NTNU refused it."""
        self._ask(keys.save_key)

    def remove(self) -> None:
        """Remove the saved key."""
        try:
            removed = keys.remove_key()
        except (OSError, ValueError) as exc:
            self._say(str(exc))
            return
        self._say(tr("key.removed" if removed else "key.nothing_to_remove"))
        self._changed()

    def close(self) -> None:
        """Close the window; an answer still on its way is dropped."""
        self.alive = False
        self.top.destroy()

    # --- helpers -----------------------------------------------------------------------

    def _ask(self, work: Callable[[str], str]) -> None:
        """Run ``work(key)`` on a worker thread and show the code it returns as words."""
        if self.busy:
            return
        text = self.key.get()
        try:
            text = keys.valid_key(text)  # an empty or unusable key is told at once, no request
        except ValueError as exc:
            self._say(str(exc))
            return
        self.busy = True
        self._say(tr("key.testing"))
        self.test_button.state(["disabled"])
        self.save_button.state(["disabled"])

        def worker() -> None:
            try:
                self.answers.put(work(text))
            except (OSError, ValueError) as exc:
                self.answers.put(f"!{exc}")
            except Exception:  # an answer must come; the text may hold the key, so it is not shown
                self.answers.put("unreachable")

        threading.Thread(target=worker, daemon=True).start()

    def _poll(self) -> None:
        """Show an answer from the worker, then look again."""
        if not self.alive:
            return
        try:
            code = self.answers.get_nowait()
        except queue.Empty:
            self.top.after(_POLL_MS, self._poll)
            return
        self.busy = False
        self.test_button.state(["!disabled"])
        self.save_button.state(["!disabled"])
        if code.startswith("!"):
            self._say(code[1:])
        else:
            self._say(tr(f"key.{code}"))
            if code in ("saved", "saved_unchecked"):
                self.key.set("")  # the key is not kept in the window once it is saved
                self._changed()
        self.top.after(_POLL_MS, self._poll)

    def _changed(self) -> None:
        self._show_source()
        if self.on_change is not None:
            self.on_change()

    def _show_source(self) -> None:
        """Say where a run would find the key now."""
        try:
            where = keys.key_source() or "none"
        except (OSError, ValueError) as exc:
            self.source.configure(text=str(exc))
            return
        self.source.configure(text=tr(f"key.source.{where}"))

    def _say(self, text: str) -> None:
        self.message.configure(text=text)

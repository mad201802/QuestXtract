"""The graphical app's window (issue #124): tkinter widgets over :mod:`.state` and :mod:`.run`.

The window runs no pipeline itself. Run starts a worker thread that calls
:func:`questxtract.gui.run.run`, which calls the CLI's own code. The worker only puts events on a
queue; the main thread reads that queue with ``after()``, so Tk is used from one thread only.

The words come from :mod:`.i18n`. A language switch re-labels every widget in place: texts
that come from the window's state are kept as functions and evaluated again.

The look is Tk's classic ttk theme. It approximates Windows XP; it is not the XP theme.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

from ..cli import RunResult
from ..pipeline import Cancelled
from . import i18n, state
from . import run as gui_run
from .i18n import tr
from .options import OPTIONS, Option

__all__ = ["run_app"]

_POLL_MS = 100


def run_app() -> int:
    """Open the window and run it until it is closed.

    Raises:
        OSError: If no window can be opened here, for example with no display.

    """
    try:
        root, drag_and_drop = _new_root()
    except tk.TclError as exc:
        raise OSError(f"cannot open a window: {exc}") from None
    QuestXtractWindow(root, drag_and_drop=drag_and_drop)
    root.mainloop()
    return 0


def _new_root() -> tuple[tk.Tk, bool]:
    """Return the window's root and whether drag and drop can work on this system.

    tkinterdnd2 gives a root that accepts drops; where its binary will not load, the plain root
    is used and the window says drop is not available.
    """
    try:
        from tkinterdnd2 import TkinterDnD

        return TkinterDnD.Tk(), True
    except (ImportError, tk.TclError):
        return tk.Tk(), False


class QuestXtractWindow:
    """The one window: a file, an output, the options, Run, Cancel, progress and the result."""

    def __init__(self, root: tk.Tk, *, drag_and_drop: bool) -> None:
        """Build the window on ``root``; ``drag_and_drop`` says whether the root takes drops."""
        self.root = root
        self.source: Path | None = None
        self.busy = False
        self.result_path: Path | None = None
        self.cancel = threading.Event()
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.output = tk.StringVar(value="questions")
        self.language = tk.StringVar(value=i18n.language())
        self.fields: dict[str, tk.StringVar | tk.BooleanVar] = {}
        self.option_rows: dict[str, ttk.Frame] = {}
        self.option_labels: dict[str, tuple[ttk.Label, Option]] = {}
        self.tristate_combos: dict[str, ttk.Combobox] = {}
        self.output_buttons: dict[str, ttk.Radiobutton] = {}
        self.static_texts: list[tuple[Any, str]] = []
        # Texts that depend on the window's state; evaluated again on a language switch.
        self.drop_text: Callable[[], str] | None = None
        self.status_text: Callable[[], str] = lambda: ""
        self.stage_text: Callable[[], str] = lambda: ""
        self.closing = False
        self.alive = True
        self.advanced_open = False
        root.protocol("WM_DELETE_WINDOW", self._close)
        style = ttk.Style(root)
        if "classic" in style.theme_names():
            style.theme_use("classic")
        self._build()
        self._enable_drop(drag_and_drop)
        self._retranslate()
        root.after(_POLL_MS, self._poll)

    # --- layout ------------------------------------------------------------------------

    def _text(self, widget: Any, key: str) -> None:
        """Give ``widget`` the message ``key`` now, and again whenever the language changes."""
        self.static_texts.append((widget, key))
        widget.configure(text=tr(key))

    def _build(self) -> None:
        """Lay out the groups top to bottom: language, file, output, options, Run and progress."""
        self.root.columnconfigure(0, weight=1)
        self.root.title("QuestXtract")
        frame = ttk.Frame(self.root, padding=8)
        frame.grid(sticky="nsew")

        bar = ttk.Frame(frame)
        bar.grid(row=0, sticky="e")
        for column, (code, name) in enumerate(i18n.LANGUAGES.items()):
            button = ttk.Radiobutton(
                bar,
                text=name,
                value=code,
                variable=self.language,
                command=self._switch_language,
            )
            button.grid(row=0, column=column, padx=(8, 0))

        source = ttk.LabelFrame(frame, padding=8)
        self._text(source, "group.source")
        source.grid(row=1, sticky="ew")
        self.drop_box = tk.Label(source, relief="sunken", bd=2, bg="white", height=2, width=54)
        self._text(self.drop_box, "drop.help")
        self.drop_box.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.source_label = ttk.Label(source, wraplength=440)
        self.source_label.grid(row=1, column=0, sticky="w", pady=(6, 0))
        browse = ttk.Button(source, command=self._browse)
        self._text(browse, "button.browse")
        browse.grid(row=1, column=1, pady=(6, 0))
        self.drop_note = ttk.Label(source, text="", wraplength=440)
        self.drop_note.grid(row=2, column=0, columnspan=2, sticky="w")

        outputs = ttk.LabelFrame(frame, padding=8)
        self._text(outputs, "group.output")
        outputs.grid(row=2, sticky="ew", pady=(8, 0))
        for column, name in enumerate(state.OUTPUT_LABELS):
            button = ttk.Radiobutton(
                outputs, value=name, variable=self.output, command=self._refresh
            )
            self._text(button, f"output.{name}")
            button.grid(row=0, column=column, sticky="w", padx=(0, 18))
            self.output_buttons[name] = button

        # Everything else is one click away: the options form starts hidden.
        options = ttk.LabelFrame(frame, padding=8)
        self.options_frame = options
        self._text(options, "group.options")
        options.grid(row=3, sticky="ew", pady=(8, 0))
        files = ttk.Label(options, wraplength=620)
        self._text(files, "output.files")
        files.grid(row=0, column=0, sticky="w", pady=(0, 6))
        shown = [option for option in OPTIONS if option.kind != "implicit"]
        for row, option in enumerate(shown, start=1):
            line = self._option_row(options, option)
            line.grid(row=row, column=0, sticky="ew", pady=1)
            self.option_rows[option.dest] = line
        options.grid_remove()

        actions = ttk.Frame(frame)
        actions.grid(row=4, sticky="ew", pady=(8, 0))
        self.run_button = ttk.Button(actions, command=self._run, width=10)
        self._text(self.run_button, "button.run")
        self.run_button.grid(row=0, column=0)
        self.cancel_button = ttk.Button(actions, command=self._cancel, width=10, state="disabled")
        self._text(self.cancel_button, "button.cancel")
        self.cancel_button.grid(row=0, column=1, padx=(6, 0))
        self.open_button = ttk.Button(actions, command=self._open_folder, state="disabled")
        self._text(self.open_button, "button.open_folder")
        self.open_button.grid(row=0, column=2, padx=(6, 0))
        actions.columnconfigure(3, weight=1)
        self.advanced_button = ttk.Button(actions, command=self._toggle_advanced)
        self.advanced_button.grid(row=0, column=3, sticky="e")

        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(frame, maximum=100, variable=self.progress_var).grid(
            row=5, sticky="ew", pady=(8, 0)
        )
        self.stage_label = ttk.Label(frame, text="")
        self.stage_label.grid(row=6, sticky="w")
        self.hint = ttk.Label(frame, text="", wraplength=460, foreground="#444444")
        self.hint.grid(row=7, sticky="w", pady=(4, 0))
        self.status = ttk.Label(frame, text="", wraplength=460)
        self.status.grid(row=8, sticky="w", pady=(4, 0))

    def _option_row(self, parent: tk.Misc, option: Option) -> ttk.Frame:
        """One control with its label, built from the option table and its Config default."""
        line = ttk.Frame(parent)
        label = ttk.Label(line, width=46, anchor="w")
        label.grid(row=0, column=0, sticky="w")
        self.option_labels[option.dest] = (label, option)
        var: tk.StringVar | tk.BooleanVar
        if option.kind == "bool":
            var = tk.BooleanVar(value=False)
            control: tk.Widget = ttk.Checkbutton(line, variable=var)
        elif option.kind == "tristate":
            var = tk.StringVar(value=tr("tristate.default"))
            combo = ttk.Combobox(line, textvariable=var, state="readonly", width=26)
            self.tristate_combos[option.dest] = combo
            control = combo
            self._set_tristate_values(option.dest)
        elif option.kind == "choice":
            var = tk.StringVar(value=state.default_text(option))
            control = ttk.Combobox(
                line, textvariable=var, values=option.choices, state="readonly", width=26
            )
        else:
            initial = "" if option.kind in ("file", "path") else state.default_text(option)
            var = tk.StringVar(value=initial)
            control = ttk.Entry(line, textvariable=var, width=36)
        control.grid(row=0, column=1, sticky="w")
        if option.kind in ("file", "path"):
            browse = ttk.Button(line, command=lambda: self._browse_into(option, var))
            self._text(browse, "button.browse")
            browse.grid(row=0, column=2, padx=(4, 0))
        self.fields[option.dest] = var
        var.trace_add("write", lambda *_: self._refresh())
        return line

    def _set_tristate_values(self, dest: str) -> None:
        """Give a tri-state control its choices in the current language."""
        labels = [tr(f"tristate.{name.lower()}") for name in state.TRISTATE_LABELS]
        self.tristate_combos[dest].configure(values=labels)

    def _enable_drop(self, drag_and_drop: bool) -> None:
        """Register the drop box, or say on the window that drop is not available here."""
        if drag_and_drop:
            try:
                from tkinterdnd2 import DND_FILES

                target: Any = self.drop_box  # tkinterdnd2 adds these methods to Tk widgets
                target.drop_target_register(DND_FILES)
                target.dnd_bind("<<Drop>>", self._on_drop)
                return
            except tk.TclError:
                pass
        self.drop_text = lambda: tr("drop.unavailable")

    # --- what the controls mean --------------------------------------------------------

    def choose_source(self, path: Path) -> None:
        """Use ``path`` as the recording or transcript, and show it."""
        self.source = path
        self.source_label.configure(text=str(path))
        if not state.output_offered(self.output.get(), path):
            self.output.set("questions")
        self._refresh()

    def _toggle_advanced(self) -> None:
        """Show or hide the options form; the window grows or shrinks to fit."""
        self.advanced_open = not self.advanced_open
        if self.advanced_open:
            self.options_frame.grid()
        else:
            self.options_frame.grid_remove()
        self._retranslate_advanced_button()

    def _retranslate_advanced_button(self) -> None:
        """Label the Advanced settings button for what it will do, in the current language."""
        key = "button.advanced_hide" if self.advanced_open else "button.advanced_show"
        self.advanced_button.configure(text=tr(key))

    def _tristate_canonical(self, dest: str) -> str:
        """Return the English name of a tri-state control's shown choice (its internal value)."""
        shown = self.fields[dest].get()
        for name in state.TRISTATE_LABELS:
            if tr(f"tristate.{name.lower()}") == shown:
                return name
        return state.TRISTATE_LABELS[0]

    def _values(self) -> dict[str, Any]:
        """Return the form as the values the CLI takes."""
        raw: dict[str, Any] = {dest: var.get() for dest, var in self.fields.items()}
        for dest in self.tristate_combos:
            raw[dest] = self._tristate_canonical(dest)
        return state.form_values(raw)

    def _output_path(self) -> Path | None:
        """Return the file to write: the output field, else the output's default (QTI only)."""
        text = str(self.fields["output"].get()).strip()
        if text:
            return Path(text)
        return state.default_output(self.source, self.output.get())

    def _problem(self) -> str | None:
        """Why Run is disabled now, or ``None``."""
        return state.run_problem(
            self.source, self.output.get(), self._values(), self._output_path()
        )

    def _refresh(self) -> None:
        """Bring every control in line with the form: shown rows, enabled buttons, the hint."""
        output = self.output.get()
        for name, button in self.output_buttons.items():
            offered = state.output_offered(name, self.source)
            button.state(["!disabled"] if offered else ["disabled"])
        for option in OPTIONS:
            row = self.option_rows.get(option.dest)
            if row is not None:
                if gui_run.applies(option, output):
                    row.grid()
                else:
                    row.grid_remove()
        problem = None if self.busy else self._problem()
        self.run_button.state(["disabled"] if self.busy or problem else ["!disabled"])
        self.hint.configure(text=problem or "")

    # --- language ----------------------------------------------------------------------

    def _switch_language(self) -> None:
        """Show the window in the language the switch picked; the form keeps its values."""
        choices = {dest: self._tristate_canonical(dest) for dest in self.tristate_combos}
        i18n.set_language(self.language.get())
        for dest, name in choices.items():
            self._set_tristate_values(dest)
            var: tk.Variable = self.fields[dest]
            var.set(tr(f"tristate.{name.lower()}"))
        self._retranslate()

    def _retranslate(self) -> None:
        """Put every text of the window in the current language."""
        self.root.title("QuestXtract")
        self._retranslate_advanced_button()
        for widget, key in self.static_texts:
            widget.configure(text=tr(key))
        for dest, (label, option) in self.option_labels.items():
            text = tr("opt." + dest)
            label.configure(text=tr("control.label", label=text, default=state.hint(option)))
        self.source_label.configure(
            text=tr("source.none") if self.source is None else str(self.source)
        )
        self.drop_note.configure(text=self.drop_text() if self.drop_text else "")
        self.status.configure(text=self.status_text())
        self.stage_label.configure(text=self.stage_text())
        self._refresh()

    # --- actions -----------------------------------------------------------------------

    def _browse(self) -> None:
        """Ask the system dialog for a recording or transcript."""
        name = filedialog.askopenfilename(
            parent=self.root,
            title=tr("dialog.source"),
            filetypes=[
                (tr("filetype.recordings"), "*.mp4 *.mp3 *.json"),
                (tr("filetype.all"), "*.*"),
            ],
        )
        if name:
            self.choose_source(Path(name))

    def _browse_into(self, option: Option, var: tk.Variable) -> None:
        """Ask the system dialog for the file an option names, and put it in its control."""
        if option.dest == "output":
            name = filedialog.asksaveasfilename(
                parent=self.root,
                title=tr("dialog.output"),
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), (tr("filetype.all"), "*.*")],
            )
        else:
            name = filedialog.askopenfilename(
                parent=self.root,
                title=tr("dialog.settings"),
                filetypes=[(tr("filetype.settings"), "*.toml"), (tr("filetype.all"), "*.*")],
            )
        if name:
            var.set(name)

    def _on_drop(self, event: Any) -> str:
        """Take the first file of a drop; tkinterdnd2 passes the paths as one braced string."""
        paths = state.dropped_paths(str(event.data), self.root.tk.splitlist)
        if paths:
            self.choose_source(paths[0])
            if len(paths) > 1:
                self.drop_text = lambda: tr("drop.one_file")
                self.drop_note.configure(text=self.drop_text())
        return str(getattr(event, "action", "copy"))

    def _run(self) -> None:
        """Start the run on a worker thread; the window keeps answering while it works."""
        source = self.source
        if source is None or self.busy:
            return
        self.busy = True
        self.cancel.clear()
        self.result_path = None
        self.progress_var.set(0.0)
        self._set_stage(lambda: tr("status.starting"))
        self._set_status(lambda: tr("status.long_run"))
        args = (source, self.output.get(), self._values(), self._output_path())
        threading.Thread(target=self._work, args=args, daemon=True).start()
        self._refresh()

    def _work(
        self,
        source: Path,
        output: str,
        values: dict[str, Any],
        output_path: Path | None,
    ) -> None:
        """Run the CLI's run and report only through the queue; never touch Tk here."""

        def report(stage: str, fraction: float) -> None:
            self.events.put(("progress", (stage, fraction)))

        try:
            result = gui_run.run(
                source, output, values, output_path, progress=report, stop=self.cancel.is_set
            )
        except Cancelled:
            self.events.put(("cancelled", None))
        except (OSError, ValueError) as exc:
            self.events.put(("error", state.one_line(str(exc))))
        except Exception as exc:  # a bug, not a user error: one line here, the rest in the terminal
            traceback.print_exc()
            self.events.put(("error", tr("status.unexpected", name=type(exc).__name__)))
        else:
            self.events.put(("done", result))

    def _close(self) -> None:
        """Close the window; a run in progress is stopped first, so its temporary files go."""
        if not self.busy:
            self._destroy()
            return
        self.closing = True
        self._cancel()
        self._set_status(lambda: tr("status.closing"))

    def _cancel(self) -> None:
        """Ask the run to stop before its next stage; a running stage finishes first."""
        self.cancel.set()
        self.cancel_button.state(["disabled"])
        self._set_status(lambda: tr("status.stopping"))

    def _open_folder(self) -> None:
        """Show the folder of the written file in the system's file manager."""
        if self.result_path is None:
            return
        try:
            state.open_folder(self.result_path.parent)
        except OSError as exc:
            message = str(exc)
            self._set_status(lambda: tr("status.open_failed", error=message))

    def _set_status(self, text: Callable[[], str]) -> None:
        """Show ``text`` on the status line now, and again on a language switch."""
        self.status_text = text
        self.status.configure(text=text())

    def _set_stage(self, text: Callable[[], str]) -> None:
        """Show ``text`` as the progress stage now, and again on a language switch."""
        self.stage_text = text
        self.stage_label.configure(text=text())

    # --- events from the worker --------------------------------------------------------

    def _destroy(self) -> None:
        """Destroy the window and stop polling the worker."""
        self.alive = False
        self.root.destroy()

    def _poll(self) -> None:
        """Handle what the worker reported since the last look, then look again later."""
        if not self.alive:
            return
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle(kind, payload)
        self.root.after(_POLL_MS, self._poll)

    def _handle(self, kind: str, payload: Any) -> None:
        """Show one worker event: progress while running, the result or error when it ends."""
        if kind == "progress":
            stage, fraction = payload
            self.progress_var.set(100 * state.overall_fraction(stage, fraction))
            self._set_stage(lambda: state.stage_label(stage))
            return
        self.busy = False
        self.cancel.clear()
        self.cancel_button.state(["disabled"])
        if self.closing:
            self._destroy()
            return
        if kind == "done":
            self._show_result(payload)
        elif kind == "cancelled":
            self._set_status(lambda: tr("status.cancelled"))
            self._set_stage(lambda: tr("stage.cancelled"))
        else:
            self._set_status(lambda: payload)
            self._set_stage(lambda: tr("stage.stopped"))
        self._refresh()

    def _show_result(self, result: RunResult) -> None:
        """Show the written path, the count and any warnings, and enable Open folder."""
        self.result_path = result.path
        path = result.path

        def text() -> str:
            lines = [result.summary]
            if path is not None:
                lines.insert(0, tr("status.written", path=path))
            lines.extend(result.warnings)
            return "\n".join(lines)

        self._set_status(text)
        self.open_button.state(["!disabled"] if path is not None else ["disabled"])
        self.progress_var.set(100.0)
        self._set_stage(lambda: tr("stage.done"))

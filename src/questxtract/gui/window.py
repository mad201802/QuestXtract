"""The graphical app's window (issue #124): tkinter widgets over :mod:`.state` and :mod:`.run`.

The window runs no pipeline itself. Run starts a worker thread that calls
:func:`questxtract.gui.run.run`, which calls the CLI's own code. The worker only puts events on a
queue; the main thread reads that queue with ``after()``, so Tk is used from one thread only.

The look is Tk's classic ttk theme. It approximates Windows XP; it is not the XP theme.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

from ..cli import RunResult
from ..pipeline import Cancelled
from . import run as gui_run
from . import state
from .options import OPTIONS, Option

__all__ = ["run_app"]

_POLL_MS = 100
_DROP_HELP = "Drop an .mp4, .mp3 or .json file here"


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
        self.fields: dict[str, tk.StringVar | tk.BooleanVar] = {}
        self.option_rows: dict[str, ttk.Frame] = {}
        self.output_buttons: dict[str, ttk.Radiobutton] = {}
        self.closing = False
        self.alive = True
        root.title("QuestXtract")
        root.protocol("WM_DELETE_WINDOW", self._close)
        style = ttk.Style(root)
        if "classic" in style.theme_names():
            style.theme_use("classic")
        self._build()
        self._enable_drop(drag_and_drop)
        self._refresh()
        root.after(_POLL_MS, self._poll)

    # --- layout ------------------------------------------------------------------------

    def _build(self) -> None:
        """Lay out the groups top to bottom: file, output, options, then Run and progress."""
        self.root.columnconfigure(0, weight=1)
        frame = ttk.Frame(self.root, padding=8)
        frame.grid(sticky="nsew")

        source = ttk.LabelFrame(frame, text="1. Recording or transcript", padding=8)
        source.grid(sticky="ew")
        self.drop_box = tk.Label(
            source, text=_DROP_HELP, relief="sunken", bd=2, bg="white", height=2, width=54
        )
        self.drop_box.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.source_label = ttk.Label(source, text="No file chosen.", wraplength=440)
        self.source_label.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(source, text="Browse...", command=self._browse).grid(
            row=1, column=1, pady=(6, 0)
        )
        self.drop_note = ttk.Label(source, text="", wraplength=440)
        self.drop_note.grid(row=2, column=0, columnspan=2, sticky="w")

        outputs = ttk.LabelFrame(frame, text="2. Output", padding=8)
        outputs.grid(sticky="ew", pady=(8, 0))
        for column, (name, label) in enumerate(state.OUTPUT_LABELS.items()):
            button = ttk.Radiobutton(
                outputs,
                text=label,
                value=name,
                variable=self.output,
                command=self._refresh,
            )
            button.grid(row=0, column=column, sticky="w", padx=(0, 18))
            self.output_buttons[name] = button
        ttk.Label(
            outputs,
            text="Files: <name>.questions.json, <name>.transcript.json, <name>.quiz.json "
            "(beside the questions) and <name>.qti.zip, next to the input unless given below.",
            wraplength=680,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        options = ttk.LabelFrame(
            frame, text="3. Options (a setting left at its default is not passed)", padding=8
        )
        options.grid(sticky="ew", pady=(8, 0))
        shown = [option for option in OPTIONS if option.kind != "implicit"]
        for row, option in enumerate(shown):
            line = self._option_row(options, option)
            line.grid(row=row, column=0, sticky="ew", pady=1)
            self.option_rows[option.dest] = line

        actions = ttk.Frame(frame)
        actions.grid(sticky="ew", pady=(8, 0))
        self.run_button = ttk.Button(actions, text="Run", command=self._run, width=10)
        self.run_button.grid(row=0, column=0)
        self.cancel_button = ttk.Button(
            actions, text="Cancel", command=self._cancel, width=10, state="disabled"
        )
        self.cancel_button.grid(row=0, column=1, padx=(6, 0))
        self.open_button = ttk.Button(
            actions, text="Open folder", command=self._open_folder, state="disabled"
        )
        self.open_button.grid(row=0, column=2, padx=(6, 0))

        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(frame, maximum=100, variable=self.progress_var).grid(
            sticky="ew", pady=(8, 0)
        )
        self.stage_label = ttk.Label(frame, text="")
        self.stage_label.grid(sticky="w")
        self.hint = ttk.Label(frame, text="", wraplength=460, foreground="#444444")
        self.hint.grid(sticky="w", pady=(4, 0))
        self.status = ttk.Label(frame, text="", wraplength=460)
        self.status.grid(sticky="w", pady=(4, 0))

    def _option_row(self, parent: tk.Misc, option: Option) -> ttk.Frame:
        """One control with its label, built from the option table and its Config default."""
        line = ttk.Frame(parent)
        label = f"{option.label} (default: {state.hint(option)})"
        ttk.Label(line, text=label, width=46, anchor="w").grid(row=0, column=0, sticky="w")
        var: tk.StringVar | tk.BooleanVar
        if option.kind == "bool":
            var = tk.BooleanVar(value=False)
            control: tk.Widget = ttk.Checkbutton(line, variable=var)
        elif option.kind in ("choice", "tristate"):
            values = option.choices or state.TRISTATE_LABELS
            initial = (
                state.TRISTATE_LABELS[0]
                if option.kind == "tristate"
                else state.default_text(option)
            )
            var = tk.StringVar(value=initial)
            control = ttk.Combobox(
                line, textvariable=var, values=values, state="readonly", width=26
            )
        else:
            initial = "" if option.kind in ("file", "path") else state.default_text(option)
            var = tk.StringVar(value=initial)
            control = ttk.Entry(line, textvariable=var, width=36)
        control.grid(row=0, column=1, sticky="w")
        if option.kind in ("file", "path"):
            browse = ttk.Button(
                line, text="Browse...", command=lambda: self._browse_into(option, var)
            )
            browse.grid(row=0, column=2, padx=(4, 0))
        self.fields[option.dest] = var
        var.trace_add("write", lambda *_: self._refresh())
        return line

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
        self.drop_note.configure(text="Drag and drop is not available on this system: use Browse.")

    # --- what the controls mean --------------------------------------------------------

    def choose_source(self, path: Path) -> None:
        """Use ``path`` as the recording or transcript, and show it."""
        self.source = path
        self.source_label.configure(text=str(path))
        if not state.output_offered(self.output.get(), path):
            self.output.set("questions")
        self._refresh()

    def _values(self) -> dict[str, Any]:
        """Return the form as the values the CLI takes."""
        return state.form_values({dest: var.get() for dest, var in self.fields.items()})

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

    # --- actions -----------------------------------------------------------------------

    def _browse(self) -> None:
        """Ask the system dialog for a recording or transcript."""
        name = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a recording or transcript",
            filetypes=[("Recordings and transcripts", "*.mp4 *.mp3 *.json"), ("All files", "*.*")],
        )
        if name:
            self.choose_source(Path(name))

    def _browse_into(self, option: Option, var: tk.Variable) -> None:
        """Ask the system dialog for the file an option names, and put it in its control."""
        if option.dest == "output":
            name = filedialog.asksaveasfilename(
                parent=self.root,
                title="Where to write the output",
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            )
        else:
            name = filedialog.askopenfilename(
                parent=self.root,
                title="Settings file",
                filetypes=[("Settings", "*.toml"), ("All files", "*.*")],
            )
        if name:
            var.set(name)

    def _on_drop(self, event: Any) -> str:
        """Take the first file of a drop; tkinterdnd2 passes the paths as one braced string."""
        paths = state.dropped_paths(str(event.data), self.root.tk.splitlist)
        if paths:
            self.choose_source(paths[0])
            if len(paths) > 1:
                self.drop_note.configure(text="One file at a time: the first one is used.")
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
        self.stage_label.configure(text="Starting...")
        self.status.configure(text="A run can take as long as the recording. Cancel stops it.")
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
            self.events.put(("error", f"unexpected {type(exc).__name__}; see the terminal"))
        else:
            self.events.put(("done", result))

    def _close(self) -> None:
        """Close the window; a run in progress is stopped first, so its temporary files go."""
        if not self.busy:
            self._destroy()
            return
        self.closing = True
        self._cancel()
        self.status.configure(text="Stopping after the current step, then the window closes...")

    def _cancel(self) -> None:
        """Ask the run to stop before its next stage; a running stage finishes first."""
        self.cancel.set()
        self.cancel_button.state(["disabled"])
        self.status.configure(text="Stopping after the current step...")

    def _open_folder(self) -> None:
        """Show the folder of the written file in the system's file manager."""
        if self.result_path is None:
            return
        try:
            state.open_folder(self.result_path.parent)
        except OSError as exc:
            self.status.configure(text=f"Could not open the folder: {exc}")

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
            self.stage_label.configure(text=state.stage_label(stage))
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
            self.status.configure(text="Cancelled. No partial file was written.")
            self.stage_label.configure(text="Cancelled")
        else:
            self.status.configure(text=payload)
            self.stage_label.configure(text="Stopped")
        self._refresh()

    def _show_result(self, result: RunResult) -> None:
        """Show the written path, the count and any warnings, and enable Open folder."""
        self.result_path = result.path
        lines = [result.summary]
        if result.path is not None:
            lines.insert(0, f"Written: {result.path}")
        lines.extend(result.warnings)
        self.status.configure(text="\n".join(lines))
        self.open_button.state(["!disabled"] if result.path is not None else ["disabled"])
        self.progress_var.set(100.0)
        self.stage_label.configure(text="Done")

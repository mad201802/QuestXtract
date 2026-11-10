"""The graphical app's window (issue #124): tkinter widgets over :mod:`.state` and :mod:`.run`."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from ..cli import RunResult
from ..config import ENDPOINTS
from ..pipeline import Cancelled
from . import i18n, state
from . import run as gui_run
from .i18n import tr
from .key_dialog import KeyDialog
from .options import EXPORTS, OPTIONS, OUTPUTS, Option
from .screen import fit_on_screen

__all__ = ["run_app"]

_POLL_MS = 100
_TYPING_PAUSE_MS = 150
_OPTIONS_TOP_PX = 420  # where the settings start, until the window is on screen
_MIN_OPTIONS_PX = 160  # the settings box never gets shorter than this
_SCREEN_MARGIN_PX = 90  # room for the taskbar and the window's border
_OUTPUT_OPTION = next(option for option in OPTIONS if option.dest == "output")
# Controls on the start view, by their row under the output buttons; the rest is Advanced.
_START_VIEW: dict[str, int] = {"vision": 2, "content_units": 3}
# Controls an export of a finished quiz.json still takes.
_EXPORT_CONTROLS: frozenset[str] = frozenset({"content_units"})
# What the save dialog offers for an output's file, by command.
_SAVE_KINDS: dict[str, tuple[str, str]] = {"qti": ("ZIP", ".zip"), "tex": ("TeX", ".tex")}


def run_app() -> int:
    """Open the window and run it until it is closed."""
    try:
        root, drag_and_drop = _new_root()
    except tk.TclError as exc:
        raise OSError(f"cannot open a window: {exc}") from None
    QuestXtractWindow(root, drag_and_drop=drag_and_drop)
    root.after_idle(fit_on_screen, root)  # on any screen size, the whole window shows
    root.mainloop()
    return 0


def _new_root() -> tuple[tk.Tk, bool]:
    before = getattr(tk, "_default_root", None)
    try:
        from tkinterdnd2 import TkinterDnD

        return TkinterDnD.Tk(), True
    except ImportError:
        pass
    except tk.TclError:
        # A failed tkdnd load leaves a half-built default Tk root behind; remove it.
        orphan = getattr(tk, "_default_root", None)
        if orphan is not None and orphan is not before:
            orphan.destroy()
    return tk.Tk(), False


def _file_stamp(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


class QuestXtractWindow:
    """The one window: a file, an output, the options, Run, Cancel, progress and the result."""

    def __init__(self, root: tk.Tk, *, drag_and_drop: bool) -> None:
        """Build the window on ``root``; ``drag_and_drop`` says whether the root takes drops."""
        self.root = root
        self.source: Path | None = None
        self.busy = False
        self._refresh_job: str | None = None
        self._problem_memo: tuple[object, str | None] | None = None
        self.result_path: Path | None = None
        self.cancel = threading.Event()
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.output = tk.StringVar(master=root, value="questions")
        self.language = tk.StringVar(master=root, value=i18n.language())
        self.fields: dict[str, tk.StringVar | tk.BooleanVar] = {}
        self.option_rows: dict[str, ttk.Frame] = {}
        self.controls: dict[str, Any] = {}
        self.option_labels: dict[str, tuple[ttk.Label, Option]] = {}
        self.tristate_combos: dict[str, ttk.Combobox] = {}
        self.output_buttons: dict[str, ttk.Radiobutton] = {}
        self.static_texts: list[tuple[Any, str]] = []
        self.drop_unavailable = False
        self.drop_extra: Callable[[], str] | None = None
        self.run_output = "questions"
        self.status_text: Callable[[], str] = lambda: ""
        self.stage_text: Callable[[], str] = lambda: ""
        self.closing = False
        self.alive = True
        self.advanced_open = False
        self._options_full_height = 0  # set once every settings row exists
        self._options_full_width = 0
        root.protocol("WM_DELETE_WINDOW", self._close)
        style = ttk.Style(root)
        if "classic" in style.theme_names():
            style.theme_use("classic")
        self._build()
        self.output.trace_add("write", lambda *_: self._output_changed())
        self._enable_drop(drag_and_drop)
        self._retranslate()
        root.after(_POLL_MS, self._poll)

    def _text(self, widget: Any, key: str) -> None:
        self.static_texts.append((widget, key))
        widget.configure(text=tr(key))

    def _build(self) -> None:
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
        # The drop area, Browse and the chosen file on one row, to save room (#259)
        self.drop_box = tk.Label(source, relief="sunken", bd=2, bg="white", height=2, width=36)
        self.drop_box.grid(row=0, column=0, sticky="ew")
        browse = ttk.Button(source, command=self._browse)
        self._text(browse, "button.browse")
        browse.grid(row=0, column=1, padx=(8, 0))
        self.source_tip = _Tooltip(self.drop_box)  # the full path, on hover
        self.drop_note = ttk.Label(source, text="", wraplength=440)
        self.drop_note.grid(row=1, column=0, columnspan=3, sticky="w")

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
        self.output_about = tk.Label(  # a fixed box, so the window keeps its size (#259)
            outputs,
            height=3,
            width=90,
            wraplength=620,
            anchor="nw",
            justify="left",
            foreground="#444444",
        )
        self.output_about.grid(  # what the chosen output makes, in plain words (#259)
            row=1, column=0, columnspan=len(state.OUTPUT_LABELS), sticky="w", pady=(4, 6)
        )

        options = ttk.LabelFrame(frame, padding=8)
        self.options_frame = options
        self._text(options, "group.options")
        options.grid(row=9, sticky="ew", pady=(8, 0))  # below Run, so it never hides it (#259)
        body = self._scrolling(options)  # scrolls instead of growing past the screen (#259)
        files = ttk.Label(body, wraplength=620)
        self._text(files, "output.files")
        files.grid(row=0, column=0, sticky="w", pady=(0, 6))
        shown = [option for option in OPTIONS if option.kind != "implicit"]
        row = 0
        for option in shown:
            if option.dest in _START_VIEW:  # on the start view, under the outputs
                line = self._option_row(outputs, option)
                line.grid(
                    row=_START_VIEW[option.dest],
                    column=0,
                    columnspan=len(state.OUTPUT_LABELS),
                    sticky="w",
                )
            else:
                row += 1
                line = self._option_row(body, option)
                line.grid(row=row, column=0, sticky="ew", pady=1)
            self.option_rows[option.dest] = line
        self.no_settings = ttk.Label(body, wraplength=620, foreground="#444444")
        self._text(self.no_settings, "options.none")
        self.no_settings.grid(row=row + 1, column=0, sticky="w", pady=(6, 0))
        body.update_idletasks()  # every row is shown now: this is the tallest list
        self._options_full_height = body.winfo_reqheight()
        self._options_full_width = body.winfo_reqwidth()  # and the widest
        self.outputs_frame = outputs
        options.grid_remove()

        actions = ttk.Frame(frame)
        actions.grid(row=3, sticky="ew", pady=(8, 0))
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
        key_button = ttk.Button(actions, command=self._open_key_dialog)
        self._text(key_button, "button.api_key")
        key_button.grid(row=0, column=4, sticky="e", padx=(0, 6))
        self.advanced_button = ttk.Button(actions, command=self._toggle_advanced)
        self.advanced_button.grid(row=0, column=5, sticky="e")

        self.progress_var = tk.DoubleVar(master=self.root, value=0.0)
        ttk.Progressbar(frame, maximum=100, variable=self.progress_var).grid(
            row=5, sticky="ew", pady=(8, 0)
        )
        self.stage_label = ttk.Label(frame, text="")
        self.stage_label.grid(row=6, sticky="w")
        self.hint = tk.Label(  # two lines' room, so a longer hint never resizes the window (#259)
            frame,
            text="",
            height=2,
            wraplength=460,
            anchor="nw",
            justify="left",
            foreground="#444444",
        )
        self.hint.grid(row=7, sticky="w", pady=(4, 0))
        self.status = ttk.Label(frame, text="", wraplength=460)
        self.status.grid(row=8, sticky="w", pady=(4, 0))

    def _scrolling(self, box: ttk.LabelFrame) -> ttk.Frame:
        """Return a frame in ``box`` that scrolls once it is taller than the screen allows."""
        canvas = tk.Canvas(box, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(box, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        box.columnconfigure(0, weight=1)
        body = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=body, anchor="nw")
        self.options_canvas = canvas

        def fit(_event: object = None) -> None:
            top = canvas.winfo_rooty() or _OPTIONS_TOP_PX  # 0 until the window is shown
            room = self.root.winfo_screenheight() - top - _SCREEN_MARGIN_PX
            height = min(
                max(body.winfo_reqheight(), self._options_full_height),
                max(_MIN_OPTIONS_PX, room),
            )
            canvas.configure(
                # never shorter than the box, so a short list starts at the top (#259)
                scrollregion=(0, 0, body.winfo_reqwidth(), max(body.winfo_reqheight(), height)),
                # the same size for every output, whatever line is widest (#259)
                width=max(body.winfo_reqwidth(), self._options_full_width),
                height=height,
            )

        def wheel(event: Any) -> None:
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        body.bind("<Configure>", fit)
        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", wheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
        self._fit_options = fit
        return body

    def _option_row(self, parent: tk.Misc, option: Option) -> ttk.Frame:
        line = ttk.Frame(parent)
        label = ttk.Label(line, width=46, anchor="w", wraplength=330)  # wraps, never cut off
        label.grid(row=0, column=0, sticky="w")
        self.option_labels[option.dest] = (label, option)
        var: tk.StringVar | tk.BooleanVar
        if option.kind == "bool":
            var = tk.BooleanVar(master=line, value=False)
            control: tk.Widget = ttk.Checkbutton(line, variable=var)
        elif option.kind == "tristate":
            start = state.start_tristate(option)
            var = tk.StringVar(master=line, value=tr(f"tristate.{start.lower()}"))
            combo = ttk.Combobox(line, textvariable=var, state="readonly", width=26)
            self.tristate_combos[option.dest] = combo
            control = combo
            self._set_tristate_values(option.dest)
        elif option.kind == "choice":
            var = tk.StringVar(master=line, value=state.start_text(option))
            control = ttk.Combobox(
                line, textvariable=var, values=option.choices, state="readonly", width=26
            )
        elif option.menu:  # a list to pick from, so a name cannot be mistyped (#259)
            var = tk.StringVar(master=line, value=state.start_text(option))
            control = ttk.Combobox(
                line, textvariable=var, values=option.menu, state="readonly", width=26
            )
        elif option.kind == "endpoint":  # a short name from the list, or any URL typed in
            var = tk.StringVar(master=line, value=state.start_text(option))
            control = ttk.Combobox(line, textvariable=var, values=tuple(ENDPOINTS), width=34)
        else:
            initial = "" if option.kind in ("file", "path") else state.start_text(option)
            var = tk.StringVar(master=line, value=initial)
            control = ttk.Entry(line, textvariable=var, width=36)
        control.grid(row=0, column=1, sticky="w")
        self.controls[option.dest] = control
        if option.kind in ("file", "path"):
            browse = ttk.Button(line, command=lambda: self._browse_into(option, var))
            self._text(browse, "button.browse")
            browse.grid(row=0, column=2, padx=(4, 0))
        self.fields[option.dest] = var
        var.trace_add("write", lambda *_: self._schedule_refresh())
        return line

    def _set_tristate_values(self, dest: str) -> None:
        labels = [tr(f"tristate.{name.lower()}") for name in state.TRISTATE_LABELS]
        self.tristate_combos[dest].configure(values=labels)

    def _enable_drop(self, drag_and_drop: bool) -> None:
        if drag_and_drop:
            try:
                from tkinterdnd2 import DND_FILES

                target: Any = self.drop_box  # tkinterdnd2 adds these methods to Tk widgets
                target.drop_target_register(DND_FILES)
                target.dnd_bind("<<Drop>>", self._on_drop)
                return
            except tk.TclError:
                pass
        self.drop_unavailable = True

    def choose_source(self, path: Path) -> None:
        """Use ``path`` as the recording or transcript, and show it."""
        self.source = path
        self.drop_box.configure(text=self._drop_text())  # the box shows the file it holds
        self.source_tip.text = str(path)
        self.drop_extra = None
        self.drop_note.configure(text=self._drop_note())
        if not state.output_offered(self.output.get(), path):
            self.output.set("questions")
        self._refresh()

    def _open_key_dialog(self) -> None:
        KeyDialog(self.root, url=state.key_endpoint(self._values()), on_change=self._refresh)

    def _toggle_advanced(self) -> None:
        self.advanced_open = not self.advanced_open
        if self.advanced_open:
            self.options_frame.grid()
            self.root.after_idle(self._show_options)  # its place on screen is known only now
        else:
            self.options_frame.grid_remove()
            self.root.after_idle(fit_on_screen, self.root)
        self._retranslate_advanced_button()

    def _show_options(self) -> None:
        self._fit_options()
        fit_on_screen(self.root)  # the window grew: move it up if its bottom left the screen

    def _retranslate_advanced_button(self) -> None:
        key = "button.advanced_hide" if self.advanced_open else "button.advanced_show"
        self.advanced_button.configure(text=tr(key))

    def _drop_text(self) -> str:
        """Return what the drop area says: what to do, or the file it holds."""
        if self.source is None:
            return tr("drop.help")
        return tr("drop.chosen", name=self.source.name)

    def _show_error(self, message: str) -> None:
        """Show a failed run in a pop-up the user closes, not as text at the bottom (#259)."""
        messagebox.showerror(tr("error.title"), message, parent=self.root)

    def _drop_note(self) -> str:
        if self.drop_extra is not None:
            return self.drop_extra()
        return tr("drop.unavailable") if self.drop_unavailable else ""

    def _tristate_canonical(self, dest: str) -> str:
        shown = self.fields[dest].get()
        for name in state.TRISTATE_LABELS:
            if tr(f"tristate.{name.lower()}") == shown:
                return name
        return state.TRISTATE_LABELS[0]

    def _values(self) -> dict[str, Any]:
        raw: dict[str, Any] = {dest: var.get() for dest, var in self.fields.items()}
        for dest in self.tristate_combos:
            raw[dest] = self._tristate_canonical(dest)
        return state.form_values(raw)

    def _output_path(self) -> Path | None:
        output = self.output.get()
        text = str(self.fields["output"].get()).strip()
        if text and gui_run.applies(_OUTPUT_OPTION, output):
            return Path(text)
        return state.default_output(self.source, output)

    def _output_changed(self) -> None:
        field = self.fields["output"]
        if isinstance(field, tk.StringVar):
            field.set("")
        self._refresh()

    def _problem(self) -> str | None:
        values = self._values()
        output = self.output.get()
        output_path = self._output_path()
        settings = Path(values.get("config") or "questxtract.toml")
        stamps = (_file_stamp(self.source), _file_stamp(settings), _file_stamp(Path(".env")))
        key = (self.source, output, tuple(sorted(values.items())), output_path, stamps)
        if self._problem_memo is None or self._problem_memo[0] != key:
            self._problem_memo = (key, state.run_problem(self.source, output, values, output_path))
        return self._problem_memo[1]

    def _schedule_refresh(self) -> None:
        if self._refresh_job is not None:
            self.root.after_cancel(self._refresh_job)
        self._refresh_job = self.root.after(_TYPING_PAUSE_MS, self._refresh)

    def _refresh(self) -> None:
        self._refresh_job = None
        output = self.output.get()
        self.output_about.configure(text=tr(f"about.{output}"))
        for name, button in self.output_buttons.items():
            offered = state.output_offered(name, self.source)
            button.state(["!disabled"] if offered else ["disabled"])
        any_setting = False
        # an export of a finished quiz makes no quiz, so the quiz settings do not apply (#283)
        quiz_source = (
            OUTPUTS[output] in EXPORTS and self.source is not None and gui_run.is_quiz(self.source)
        )
        for option in OPTIONS:
            row = self.option_rows.get(option.dest)
            if row is not None:
                shown = gui_run.applies(option, output)
                if quiz_source and option.dest not in _EXPORT_CONTROLS:
                    shown = False
                if shown:
                    row.grid()
                    any_setting = any_setting or option.dest not in _START_VIEW
                else:
                    row.grid_remove()
        # an output without settings says so, instead of showing an empty box (#259)
        if any_setting:
            self.no_settings.grid_remove()
        else:
            self.no_settings.grid()
        judge_model = self.controls.get("llm_model")
        if judge_model is not None:
            judging = self.fields["detector"].get() == "llm"
            judge_model.state(["!disabled"] if judging else ["disabled"])
        problem = None if self.busy else self._problem()
        self.run_button.state(["disabled"] if self.busy or problem else ["!disabled"])
        cancellable = self.busy and not self.cancel.is_set()
        self.cancel_button.state(["!disabled"] if cancellable else ["disabled"])
        self.hint.configure(text=problem or "")

    def _switch_language(self) -> None:
        choices = {dest: self._tristate_canonical(dest) for dest in self.tristate_combos}
        i18n.set_language(self.language.get())
        for dest, name in choices.items():
            self._set_tristate_values(dest)
            var: tk.Variable = self.fields[dest]
            var.set(tr(f"tristate.{name.lower()}"))
        self._retranslate()

    def _retranslate(self) -> None:
        self.root.title("QuestXtract")
        self._retranslate_advanced_button()
        for widget, key in self.static_texts:
            widget.configure(text=tr(key))
        for dest, (label, option) in self.option_labels.items():
            text = tr("opt." + dest)
            label.configure(text=tr("control.label", label=text, default=state.hint(option)))
        self._keep_start_view_room()
        self.drop_box.configure(text=self._drop_text())
        self.drop_note.configure(text=self._drop_note())
        self.status.configure(text=self.status_text())
        self.stage_label.configure(text=self.stage_text())
        self._refresh()

    def _keep_start_view_room(self) -> None:
        """Keep each start-view line's room for outputs that hide it, so the window never jumps.

        Measured after the labels have their words, since a long label wraps to two lines.
        """
        self.root.update_idletasks()
        for dest, line_row in _START_VIEW.items():
            height = self.option_rows[dest].winfo_reqheight()
            self.outputs_frame.rowconfigure(line_row, minsize=height)

    def _browse(self) -> None:
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
        if option.dest == "output":
            kind = _SAVE_KINDS.get(OUTPUTS[self.output.get()], ("JSON", ".json"))
            name = filedialog.asksaveasfilename(
                parent=self.root,
                title=tr("dialog.output"),
                defaultextension=kind[1],
                filetypes=[(kind[0], f"*{kind[1]}"), (tr("filetype.all"), "*.*")],
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
        paths = state.dropped_paths(str(event.data), self.root.tk.splitlist)
        if paths:
            self.choose_source(paths[0])
            if len(paths) > 1:
                self.drop_extra = lambda: tr("drop.one_file")
                self.drop_note.configure(text=self._drop_note())
        return str(getattr(event, "action", "copy"))

    def _run(self) -> None:
        source = self.source
        if source is None or self.busy:
            return
        if self._problem() is not None:
            self._refresh()
            return
        self.busy = True
        self.run_output = self.output.get()
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
        except Exception as exc:
            traceback.print_exc()
            self.events.put(("error", tr("status.unexpected", name=type(exc).__name__)))
        else:
            self.events.put(("done", result))

    def _close(self) -> None:
        if not self.busy:
            self._destroy()
            return
        self.closing = True
        self._cancel()
        self._set_status(lambda: tr("status.closing"))

    def _cancel(self) -> None:
        self.cancel.set()
        self.cancel_button.state(["disabled"])
        self._set_status(lambda: tr("status.stopping"))

    def _open_folder(self) -> None:
        if self.result_path is None:
            return
        try:
            state.open_folder(self.result_path.parent)
        except OSError as exc:
            message = str(exc)
            self._set_status(lambda: tr("status.open_failed", error=message))

    def _set_status(self, text: Callable[[], str]) -> None:
        self.status_text = text
        self.status.configure(text=text())

    def _set_stage(self, text: Callable[[], str]) -> None:
        self.stage_text = text
        self.stage_label.configure(text=text())

    def _destroy(self) -> None:
        self.alive = False
        self.root.destroy()

    def _poll(self) -> None:
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
        if kind == "progress":
            stage, fraction = payload
            quiz_follows = self.run_output == "quiz" or OUTPUTS[self.run_output] in EXPORTS
            self.progress_var.set(
                100 * state.overall_fraction(stage, fraction, quiz_follows=quiz_follows)
            )
            if not (stage == "done" and quiz_follows):
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
            self._set_status(lambda: "")
            self._set_stage(lambda: tr("stage.failed"))
            self._show_error(payload)
        self._refresh()

    def _show_result(self, result: RunResult) -> None:
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


class _Tooltip:
    """A small box with ``text`` that appears while the pointer is over ``widget``."""

    def __init__(self, widget: tk.Widget) -> None:
        self.widget = widget
        self.text = ""
        self.box: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event: object = None) -> None:
        if not self.text or self.box is not None:
            return
        self.box = tk.Toplevel(self.widget)
        self.box.wm_overrideredirect(True)
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        screen_w = self.widget.winfo_screenwidth()
        tk.Label(
            self.box,
            text=self.text,
            bg="#ffffe0",
            relief="solid",
            bd=1,
            padx=4,
            wraplength=screen_w // 2,  # a long path wraps instead of running off the screen
        ).pack()
        self.box.update_idletasks()
        x = max(0, min(x, screen_w - self.box.winfo_reqwidth()))
        self.box.wm_geometry(f"+{x}+{y}")

    def _hide(self, _event: object = None) -> None:
        if self.box is not None:
            self.box.destroy()
            self.box = None

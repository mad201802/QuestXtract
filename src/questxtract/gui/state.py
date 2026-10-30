"""The window's state without widgets (issue #124): what the form holds and what Run needs.

Nothing here imports tkinter, so every function is tested headless (``tests/test_gui_state.py``).
The window reads its widgets into plain values and asks this module what they mean.

A setting left at its ``Config`` default is not passed to the CLI, so a ``questxtract.toml``
still applies. Choosing the default again in the form does not override the file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import GUI_START_DETECTOR, GUI_START_VISION, Config, endpoint_url, load_config
from ..pipeline import STAGES, stage_info
from .i18n import has, tr
from .options import OPTIONS, OUTPUTS, Option
from .run import check, source_problem

__all__ = [
    "OUTPUT_LABELS",
    "TRISTATE_LABELS",
    "default_output",
    "default_text",
    "dropped_paths",
    "form_values",
    "hint",
    "one_line",
    "start_text",
    "start_tristate",
    "open_folder",
    "output_offered",
    "overall_fraction",
    "run_problem",
    "stage_label",
]

#: The radio buttons, in order. Each one writes a file named after its command (docs/cli.md).
OUTPUT_LABELS: dict[str, str] = {
    "questions": "Questions",
    "transcript": "Transcript",
    "quiz": "Quiz",
    "qti": "QTI package",
}
#: The choices of an option that is a tri-state: the CLI's flag pair, or no flag at all.
TRISTATE_LABELS: tuple[str, ...] = ("Default", "On", "Off")
_TRISTATE_VALUES: dict[str, bool | None] = {"Default": None, "On": True, "Off": False}
# The order the pipeline runs its stages in, then the quiz (which runs after them).
_STAGE_ORDER: tuple[str, ...] = tuple(stage_info(stage).name for stage in STAGES) + ("quiz",)


def default_text(option: Option) -> str:
    """Return the text a control starts with: the ``Config`` default, or "" for no setting."""
    if option.field is None:
        return ""
    value = getattr(Config(), option.field)
    return "" if value is None else str(value)


#: What the window starts with where it differs from the CLI's default (set in config.py). A
#: control left at its start value is passed as a flag, so the run is the command line it shows.
START_VALUES: dict[str, str] = {
    "detector": GUI_START_DETECTOR,
    "vision": "On" if GUI_START_VISION else "Off",
}


_SETTINGS_FILE = Path("questxtract.toml")  # the one the CLI reads when --config is not given


def start_text(option: Option) -> str:
    """Return the text a control starts with.

    With a ``questxtract.toml`` here, the file's value (or the default it leaves), so an
    untouched form runs what the file says. Without one, the window's start values.
    """
    if option.field is None:
        return ""
    if _SETTINGS_FILE.is_file():
        try:
            value = getattr(load_config(), option.field)
        except (OSError, ValueError):  # the run check reports the file's problem
            return default_text(option)
        return "" if value is None else str(value)
    return START_VALUES.get(option.dest, default_text(option))


def start_tristate(option: Option) -> str:
    """Return the choice a tri-state control starts with: the file decides when there is one."""
    if _SETTINGS_FILE.is_file():
        return TRISTATE_LABELS[0]
    return START_VALUES.get(option.dest, TRISTATE_LABELS[0])


def hint(option: Option) -> str:
    """Return the value the control starts with, as a short phrase beside it."""
    if option.dest == "output":
        return tr("hint.next_input")
    if option.kind == "tristate" and start_tristate(option) != TRISTATE_LABELS[0]:
        return tr("hint.on") if start_tristate(option) == "On" else tr("hint.off")
    if option.kind in ("bool", "tristate"):
        return tr("hint.on") if default_text(option) == "True" else tr("hint.off")
    text = start_text(option)
    if option.dest == "llm_reasoning" and not text:
        return "server"  # "" sends no setting; the CLI spells that server
    return text or tr("hint.unset")


def form_values(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Turn what the controls hold into the values :func:`questxtract.gui.run.to_argv` takes.

    A control at its default becomes ``None`` (not passed). A text control that is empty is
    passed as ``""``, so an empty LLM model is refused rather than replaced by the default.
    """
    values: dict[str, Any] = {}
    for option in OPTIONS:
        if option.kind == "implicit" or option.dest not in raw:
            continue
        value = raw[option.dest]
        if option.kind == "bool":
            values[option.dest] = True if value else None
        elif option.kind == "tristate":
            values[option.dest] = _TRISTATE_VALUES.get(str(value))
        elif option.kind in ("file", "path"):
            text = str(value).strip()
            values[option.dest] = text or None
        else:
            text = str(value).strip()
            values[option.dest] = None if text == default_text(option) else text
    return values


def key_endpoint(values: Mapping[str, Any]) -> str | None:
    """Return the endpoint a run with the form's ``values`` sends the judge to, for the key window.

    The LLM endpoint field when it is passed, else the settings file's ``base_url`` (the one
    chosen in ``config``, or ``questxtract.toml`` here), else the default: as the CLI decides.
    """
    if values.get("llm_url"):
        return endpoint_url(str(values["llm_url"]))
    chosen = values.get("config")
    try:
        return load_config(Path(chosen) if chosen else None).base_url
    except (OSError, ValueError):  # the run check reports the file's problem
        return Config().base_url


def output_offered(output: str, source: Path | None) -> bool:
    """Whether the radio for ``output`` is enabled: a transcript is not made from a transcript."""
    return not (output == "transcript" and source is not None and source.suffix.lower() == ".json")


def default_output(source: Path | None, output: str) -> Path | None:
    """Return the file a run writes when the output field is empty; ``None`` lets the CLI choose.

    Only the QTI output needs a name chosen here: its CLI command requires ``-o``.
    """
    if source is None or OUTPUTS[output] != "qti":
        return None
    return source.with_name(f"{source.stem.removesuffix('.quiz')}.qti.zip")


def run_problem(
    source: Path | None,
    output: str,
    values: Mapping[str, Any],
    output_path: Path | None,
) -> str | None:
    """Why Run is disabled, in the CLI's own words; ``None`` when the run can start.

    Checks the file for the output, then the values the CLI would refuse, so a bad value shows
    before anything runs.
    """
    if source is None:
        return tr("hint.drop_first")
    problem = source_problem(source, output)
    if problem is not None:
        return one_line(problem)
    try:
        check(source, output, values, output_path)
    except (OSError, ValueError) as exc:
        return one_line(str(exc))
    return None


def one_line(text: str) -> str:
    """Join a message into one line, so the window never shows a multi-line error."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines) or tr("status.failed")


def overall_fraction(stage: str, fraction: float, *, quiz_follows: bool = False) -> float:
    """Where the whole run is, 0 to 1, from the pipeline's stage name and its fraction.

    The stage names are the pipeline's own progress names (architecture.md 11.3); a name this
    module does not know counts as the start of the run. The pipeline reports ``done`` when its
    stages end; with the Quiz output the quiz still follows, so that is where the quiz starts.
    """
    if stage == "done":
        return _STAGE_ORDER.index("quiz") / len(_STAGE_ORDER) if quiz_follows else 1.0
    if stage not in _STAGE_ORDER:
        return 0.0
    done = min(max(fraction, 0.0), 1.0)
    return (_STAGE_ORDER.index(stage) + done) / len(_STAGE_ORDER)


def stage_label(stage: str) -> str:
    """Return the words the window shows for a progress name, in the current language."""
    if has(f"stage.{stage}"):
        return tr(f"stage.{stage}")
    return stage.replace("_", " ").capitalize()


def dropped_paths(data: str, splitlist: Callable[[str], Sequence[str]]) -> list[Path]:
    """Return the paths in a drop. Tk braces a path that has spaces, so Tk splits the text.

    Args:
        data: The text tkinterdnd2 gives in its drop event.
        splitlist: The interpreter's ``splitlist`` (``root.tk.splitlist``), which understands
            the braces; splitting on spaces would break ``/Users/Ana Lu/lecture.mp4``.

    """
    return [Path(item) for item in splitlist(data) if item]


def open_folder(folder: Path) -> None:
    """Open ``folder`` in the system's file manager.

    Raises:
        OSError: If the opener is not there, for example ``xdg-open`` on a bare Linux system.

    """
    if sys.platform == "win32":
        os.startfile(folder)  # noqa: S606 - the OS opens a folder the app wrote
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])  # noqa: S603, S607 - a fixed program, our folder
    else:
        subprocess.Popen(["xdg-open", str(folder)])  # noqa: S603, S607 - a fixed program, our folder

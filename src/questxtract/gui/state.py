"""The window's state without widgets (issue #124): what the form holds and what Run needs."""

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
from .options import EXPORTS, OPTIONS, OUTPUTS, Option
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

OUTPUT_LABELS: dict[str, str] = {
    "questions": "Questions",
    "transcript": "Transcript",
    "quiz": "Quiz",
    "qti": "QTI package",
    "tex": "TeX (ExerPlaza)",
}
TRISTATE_LABELS: tuple[str, ...] = ("Default", "On", "Off")
_TRISTATE_VALUES: dict[str, bool | None] = {"Default": None, "On": True, "Off": False}
_STAGE_ORDER: tuple[str, ...] = tuple(stage_info(stage).name for stage in STAGES) + ("quiz",)


def default_text(option: Option) -> str:
    """Return the text a control starts with: the ``Config`` default, or "" for no setting."""
    if option.field is None:
        return ""
    return _shown(option, getattr(Config(), option.field))


def _shown(option: Option, value: Any) -> str:
    """Return the text a control shows for ``value``; no reasoning setting is "server"."""
    if value is None:
        return ""
    if option.dest == "llm_reasoning" and value == "":
        return "server"
    return str(value)


START_VALUES: dict[str, str] = {
    "detector": GUI_START_DETECTOR,
    "vision": "On" if GUI_START_VISION else "Off",
}


_SETTINGS_FILE = Path("questxtract.toml")


def start_text(option: Option) -> str:
    """Return the text a control starts with."""
    if option.field is None:
        return ""
    if _SETTINGS_FILE.is_file():
        try:
            value = getattr(load_config(), option.field)
        except (OSError, ValueError):
            return default_text(option)
        return _shown(option, value)
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
    return start_text(option) or tr("hint.unset")


def form_values(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Turn what the controls hold into the values :func:`questxtract.gui.run.to_argv` takes."""
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
    """Return the endpoint a run with the form's ``values`` sends the judge to."""
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

    An export goes next to its source, named after the lecture: ``lecture.mp4``,
    ``lecture.transcript.json`` and ``lecture.quiz.json`` all give ``lecture.qti.zip`` or
    ``lecture.tex``.
    """
    if source is None or OUTPUTS[output] not in EXPORTS:
        return None
    base = source.name
    for suffix in (".quiz.json", ".transcript.json", ".questions.json"):
        base = base.removesuffix(suffix)
    if base == source.name:
        base = source.stem
    extension = ".qti.zip" if OUTPUTS[output] == "qti" else ".tex"
    return source.with_name(base + extension)


def run_problem(
    source: Path | None,
    output: str,
    values: Mapping[str, Any],
    output_path: Path | None,
) -> str | None:
    """Why Run is disabled, in the CLI's own words; ``None`` when the run can start."""
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
    """Where the whole run is, 0 to 1, from the pipeline's stage name and its fraction."""
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
    """Return the paths in a drop."""
    return [Path(item) for item in splitlist(data) if item]


def open_folder(folder: Path) -> None:
    """Open ``folder`` in the system's file manager."""
    if sys.platform == "win32":
        os.startfile(folder)  # noqa: S606 - the OS opens a folder the app wrote
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])  # noqa: S603, S607 - a fixed program, our folder
    else:
        subprocess.Popen(["xdg-open", str(folder)])  # noqa: S603, S607 - a fixed program, our folder

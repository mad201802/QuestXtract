"""The window's logic, without widgets (issue #124): form values in, the CLI's own run out.

The app never runs a pipeline of its own. It builds the arguments a user would type on the
command line (:func:`to_argv`) and hands them to :func:`questxtract.cli.execute`, so parsing,
defaults, checks and output are the CLI's. Tested headless in ``tests/test_gui_run.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..cli import PROCESS_INPUTS, TRANSCRIBE_INPUTS, RunResult, check_input, execute
from ..models import Progress
from ..quiz_schema import read_quiz
from .options import OPTIONS, OUTPUTS, TRANSCRIBE_DESTS, Option


def inputs_for(output: str) -> tuple[str, ...]:
    """Return the file suffixes an output takes: recordings for a transcript, a quiz.json for QTI.

    A transcript output only takes recordings; the QTI output takes a quiz.json, checked by
    name in :func:`source_problem`.
    """
    command = OUTPUTS[output]
    if command == "qti":
        return (".json",)
    return TRANSCRIBE_INPUTS if command == "transcribe" else PROCESS_INPUTS


def source_problem(path: Path, output: str) -> str | None:
    """Why ``path`` cannot be run for ``output``, in the CLI's own words; ``None`` if it can."""
    if OUTPUTS[output] == "qti":
        if not path.is_file():
            return f"{path}: no such file"
        try:
            read_quiz(path)
        except (OSError, ValueError):
            return f"{path}: expected a quiz.json written by the Quiz output"
        return None
    try:
        check_input(path, inputs_for(output))
    except (OSError, ValueError) as exc:
        return str(exc)
    return None


def _applies(option: Option, output: str) -> bool:
    """Whether a control is shown for ``output``: by its command, then by its own outputs."""
    if OUTPUTS[output] == "transcribe" and option.dest not in TRANSCRIBE_DESTS:
        return False
    return not option.outputs or output in option.outputs


def to_argv(
    source: Path,
    output: str,
    values: Mapping[str, Any],
    output_path: Path | None = None,
) -> list[str]:
    """Return the command line a user would type for this run (``questxtract process ...``).

    ``values`` is keyed by the option's destination; a value left as ``None`` (or an empty
    text field) is not passed, so the settings file and the defaults apply exactly as on the
    command line. ``output_path`` is ``-o``: the questions file, or the transcript.

    Raises:
        ValueError: If the form asks for an LLM detector without a model.

    """
    if output not in OUTPUTS:
        raise ValueError(f"unknown output {output!r}")
    if values.get("detector") == "llm" and values.get("llm_model") == "":
        raise ValueError(
            "the LLM detector needs a model name, or leave the field empty for the default"
        )
    command = OUTPUTS[output]
    if command == "qti":
        if output_path is None:
            raise ValueError("the QTI output needs a file to write the package to")
        return ["qti", str(source), "--output", str(output_path)]
    argv = [command, str(source)]
    if output_path is not None:
        argv += ["--output", str(output_path)]
    if output == "quiz":
        argv.append("--quiz")
    for option in OPTIONS:
        if option.kind == "implicit" or option.dest == "output" or not _applies(option, output):
            continue
        value = values.get(option.dest)
        if value is None or value == "":
            continue
        flag = max((part.strip() for part in option.flag.split(",")), key=len)
        if option.kind == "bool":
            if value:
                argv.append(flag)
        elif option.kind == "tristate":
            argv.append("--vision" if value else "--no-vision")
        else:
            argv += [flag, str(value)]
    return argv


def run(
    source: Path,
    output: str,
    values: Mapping[str, Any],
    output_path: Path | None = None,
    *,
    progress: Progress | None = None,
    stop: Callable[[], bool] | None = None,
) -> RunResult:
    """Run the CLI's own ``process`` or ``transcribe`` for the form's values.

    Raises:
        ValueError, OSError: with the CLI's own one-line message; the window shows it as is.
        questxtract.pipeline.Cancelled: if ``stop`` asked for it between stages.

    """
    argv = to_argv(source, output, values, output_path)
    return execute(argv, progress=progress, stop=stop)

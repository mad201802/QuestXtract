"""The window's logic, without widgets (issue #124): form values in, the CLI's own run out."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..cli import (
    PROCESS_INPUTS,
    TRANSCRIBE_INPUTS,
    RunResult,
    check_arguments,
    check_input,
    execute,
)
from ..config import Config
from ..models import Progress
from ..quiz_schema import read_quiz
from .options import OPTIONS, OUTPUTS, TRANSCRIBE_DESTS, Option


def inputs_for(output: str) -> tuple[str, ...]:
    """Return the file suffixes an output takes."""
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


def applies(option: Option, output: str) -> bool:
    """Whether a control is shown for ``output``: by its command, then by its own outputs."""
    if OUTPUTS[output] == "qti":
        return False
    if OUTPUTS[output] == "transcribe" and option.dest not in TRANSCRIBE_DESTS:
        return False
    return not option.outputs or output in option.outputs


def to_argv(
    source: Path,
    output: str,
    values: Mapping[str, Any],
    output_path: Path | None = None,
) -> list[str]:
    """Return the command line a user would type for this run (``questxtract process ...``)."""
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
    judging = (values.get("detector") or Config().detector) == "llm"
    for option in OPTIONS:
        if option.kind == "implicit" or option.dest == "output" or not applies(option, output):
            continue
        if option.dest == "llm_model" and not judging:
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


def check(
    source: Path,
    output: str,
    values: Mapping[str, Any],
    output_path: Path | None = None,
) -> None:
    """Raise the CLI's own error for a run that would be refused, before any work starts."""
    check_arguments(to_argv(source, output, values, output_path))


def run(
    source: Path,
    output: str,
    values: Mapping[str, Any],
    output_path: Path | None = None,
    *,
    progress: Progress | None = None,
    stop: Callable[[], bool] | None = None,
) -> RunResult:
    """Run the CLI's own ``process`` or ``transcribe`` for the form's values."""
    argv = to_argv(source, output, values, output_path)
    return execute(argv, progress=progress, stop=stop)

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
from ..pipeline import Cancelled
from ..quiz_schema import read_quiz
from .options import EXPORTS, OPTIONS, OUTPUTS, TRANSCRIBE_DESTS, Option

# Options of the export commands, never passed to ``process``.
_EXPORT_ONLY: frozenset[str] = frozenset({"content_units"})


def inputs_for(output: str) -> tuple[str, ...]:
    """Return the file suffixes an output takes."""
    command = OUTPUTS[output]
    return TRANSCRIBE_INPUTS if command == "transcribe" else PROCESS_INPUTS


def is_quiz(path: Path) -> bool:
    """Whether ``path`` is a ``quiz.json`` an export can take as it is."""
    if path.suffix.lower() != ".json" or not path.is_file():
        return False
    try:
        read_quiz(path)
    except (OSError, ValueError):
        return False
    return True


def source_problem(path: Path, output: str) -> str | None:
    """Why ``path`` cannot be run for ``output``, in the CLI's own words; ``None`` if it can.

    An export takes a ``quiz.json`` as it is, and a recording or transcript through the quiz.
    """
    if OUTPUTS[output] in EXPORTS and is_quiz(path):
        return None
    try:
        check_input(path, inputs_for(output))
    except (OSError, ValueError) as exc:
        return str(exc)
    return None


def applies(option: Option, output: str) -> bool:
    """Whether a control is shown for ``output``: by its command, then by its own outputs.

    An export shows the Quiz output's controls, since from a recording it makes the quiz first.
    """
    command = OUTPUTS[output]
    if command == "transcribe" and option.dest not in TRANSCRIBE_DESTS:
        return False
    if not option.outputs or output in option.outputs:
        return True
    return command in EXPORTS and "quiz" in option.outputs


def content_units(values: Mapping[str, Any]) -> list[str]:
    """Return the content units of the form's field: names separated by ";", blanks dropped."""
    text = str(values.get("content_units") or "")
    return [name.strip() for name in text.split(";") if name.strip()]


def export_argv(
    quiz: Path,
    output: str,
    values: Mapping[str, Any],
    output_path: Path | None,
) -> list[str]:
    """Return the ``qti`` or ``tex`` command line that exports ``quiz``."""
    command = OUTPUTS[output]
    if output_path is None:
        raise ValueError(f"the {output.upper()} output needs a file to write to")
    argv = [command, str(quiz), "--output", str(output_path)]
    if command == "tex":
        units = content_units(values)
        if not units:
            raise ValueError(
                "the TeX output needs at least one ExerPlaza content unit; separate several with ;"
            )
        for unit in units:
            argv += ["--content-unit", unit]
    return argv


def to_argv(
    source: Path,
    output: str,
    values: Mapping[str, Any],
    output_path: Path | None = None,
) -> list[str]:
    """Return the command line a user would type for this run (``questxtract process ...``).

    For an export from a recording or transcript this is the ``process --quiz`` run that makes
    the quiz; :func:`run` exports it after. Its ``questions.json`` and ``quiz.json`` go next to
    the source, as they do for the Quiz output without a file named.
    """
    if output not in OUTPUTS:
        raise ValueError(f"unknown output {output!r}")
    if values.get("detector") == "llm" and values.get("llm_model") == "":
        raise ValueError(
            f"the LLM detector needs a model name; type one, such as {Config().judge_model}"
        )
    command = OUTPUTS[output]
    if command in EXPORTS:
        if is_quiz(source):
            return export_argv(source, output, values, output_path)
        export_argv(source, output, values, output_path)  # its problems show before the run
        command, output_path = "process", None
    argv = [command, str(source)]
    if output_path is not None:
        argv += ["--output", str(output_path)]
    if output == "quiz" or OUTPUTS[output] in EXPORTS:
        argv.append("--quiz")
    judging = (values.get("detector") or Config().detector) == "llm"
    for option in OPTIONS:
        if option.kind == "implicit" or option.dest == "output" or not applies(option, output):
            continue
        if option.dest in _EXPORT_ONLY:
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
    """Run the CLI's own commands for the form's values.

    An export from a recording or transcript is one run of two commands: ``process --quiz``,
    then ``qti`` or ``tex`` on the quiz it wrote. A stop between them leaves the quiz and no
    export file. The Questions output with ``questions_tex`` set also runs ``questions-tex``
    on the ``questions.json`` it wrote (#284).

    Raises:
        Cancelled: If ``stop`` asked for it before a stage or before the export.

    """
    argv = to_argv(source, output, values, output_path)
    if output == "questions" and values.get("questions_tex"):
        return _with_questions_tex(execute(argv, progress=progress, stop=stop), stop)
    if OUTPUTS[output] not in EXPORTS or argv[0] != "process":
        return execute(argv, progress=progress, stop=stop)
    made = execute(argv, progress=progress, stop=stop)
    if made.path is None:  # pragma: no cover - the quiz always has a file next to the source
        raise ValueError("the quiz step wrote no file to export")
    if stop is not None and stop():
        raise Cancelled("stopped before the export")
    exported = execute(export_argv(made.path, output, values, output_path))
    return RunResult(
        path=exported.path,
        summary=f"{made.summary}; {exported.summary}",
        warnings=made.warnings,
    )


def _with_questions_tex(made: RunResult, stop: Callable[[], bool] | None) -> RunResult:
    """Write the ``.tex`` of the questions file ``made`` wrote, beside it."""
    if made.path is None:  # pragma: no cover - the app always names a file
        return made
    if stop is not None and stop():
        raise Cancelled("stopped before the .tex")
    tex = made.path.with_name(f"{made.path.name.removesuffix('.json')}.tex")
    written = execute(["questions-tex", str(made.path), "--output", str(tex)])
    return RunResult(
        path=made.path, summary=f"{made.summary}; {written.summary}", warnings=made.warnings
    )

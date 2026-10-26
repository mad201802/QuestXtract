"""The options the graphical app shows, one table, checked against the CLI (issue #124).

The app runs the CLI, so its flags reach the same settings through the CLI's own mapping
(:data:`questxtract.cli.FLAG_FIELDS`). Nothing in this module imports tkinter: the parity test
(``tests/test_gui_options.py``) runs headless in CI.

Every option of ``process`` and ``transcribe`` is either in :data:`OPTIONS` (the app shows it, with
the default that ``Config`` gives it) or in :data:`EXCLUDED` (the app does not show it, with a
reason). A new CLI option fails the test until one of the two is chosen for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["text", "number", "choice", "bool", "tristate", "file", "path", "implicit"]

# The outputs the app offers (issue #124 and its quiz addition): which CLI command each one runs.
OUTPUTS: dict[str, str] = {
    "questions": "process",
    "transcript": "transcribe",
    "quiz": "process",
    "qti": "qti",
}
# Options `transcribe` takes; every other option belongs to `process`.
TRANSCRIBE_DESTS: frozenset[str] = frozenset({"config", "output", "model", "from_scratch"})


@dataclass(frozen=True, kw_only=True)
class Option:
    """One control in the app, for the CLI option with destination ``dest``.

    ``field`` is the ``Config`` field whose default the control starts from, or ``None`` when the
    option is not a setting (a file or folder to pick). ``choices`` are the CLI's own choices.
    """

    dest: str
    flag: str
    kind: Kind
    field: str | None = None
    choices: tuple[str, ...] = ()
    # The outputs this control is shown for; empty means every output of its command.
    outputs: tuple[str, ...] = ()


OPTIONS: tuple[Option, ...] = (
    Option(dest="config", flag="--config", kind="file"),
    Option(dest="output", flag="-o, --output", kind="path"),
    Option(dest="model", flag="-m, --model", kind="text", field="model_size"),
    Option(
        dest="from_scratch",
        flag="--from-scratch",
        kind="bool",
        field="from_scratch",
    ),
    Option(
        dest="threshold",
        flag="-t, --threshold",
        kind="number",
        field="min_confidence",
    ),
    Option(
        dest="detector",
        flag="--detector",
        kind="choice",
        field="detector",
        choices=("rules", "llm"),
    ),
    Option(dest="llm_model", flag="--llm-model", kind="text", field="judge_model"),
    Option(dest="llm_url", flag="--llm-url", kind="text", field="base_url"),
    Option(
        dest="llm_reasoning",
        flag="--llm-reasoning",
        kind="choice",
        field="reasoning_effort",
        choices=("none", "low", "medium", "high", "server"),
    ),
    Option(
        dest="vision",
        flag="--vision, --no-vision",
        kind="tristate",
        field="vision_enabled",
    ),
    Option(
        dest="vision_model",
        flag="--vision-model",
        kind="text",
        field="vision_model",
    ),
    Option(
        dest="vision_reader",
        flag="--vision-reader",
        kind="choice",
        field="vision_reader",
        choices=("vlm", "ocr"),
    ),
    Option(
        dest="quiz",
        flag="--quiz",
        kind="implicit",
        outputs=("quiz",),
    ),
    Option(
        dest="strictness",
        flag="--strictness",
        kind="choice",
        field="quiz_strictness",
        choices=("strict", "lenient"),
        outputs=("quiz",),
    ),
    Option(
        dest="options",
        flag="--options",
        kind="number",
        field="quiz_options",
        outputs=("quiz",),
    ),
    Option(
        dest="quiz_model",
        flag="--quiz-model",
        kind="text",
        field="quiz_model",
        outputs=("quiz",),
    ),
)

# Options the app leaves out, with the reason. The parity test requires every CLI option to be
# either in OPTIONS or here.
EXCLUDED: dict[str, str] = {
    "input": "the recording or transcript is chosen with the drop box, not a flag",
    "recording": "the recording is chosen with the drop box, not a flag",
}

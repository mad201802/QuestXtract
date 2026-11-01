"""The options the graphical app shows, one table, checked against the CLI (issue #124)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal[
    "text", "endpoint", "number", "choice", "bool", "tristate", "file", "path", "implicit"
]

OUTPUTS: dict[str, str] = {
    "questions": "process",
    "transcript": "transcribe",
    "quiz": "process",
    "qti": "qti",
}
TRANSCRIBE_DESTS: frozenset[str] = frozenset({"config", "output", "model", "from_scratch"})


@dataclass(frozen=True, kw_only=True)
class Option:
    """One control in the app, for the CLI option with destination ``dest``."""

    dest: str
    flag: str
    kind: Kind
    field: str | None = None
    choices: tuple[str, ...] = ()
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
    Option(dest="llm_url", flag="--llm-url", kind="endpoint", field="base_url"),
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

EXCLUDED: dict[str, str] = {
    "input": "the recording or transcript is chosen with the drop box, not a flag",
    "recording": "the recording is chosen with the drop box, not a flag",
}

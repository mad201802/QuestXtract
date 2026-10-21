"""Every pipeline setting in one flat frozen record (architecture.md 11.1).

Flat, not nested: that is what lets a stage's ``uses=(...)`` be a tuple of names rather
than a path expression. Stages read config only through ``ctx.config``; nothing else in
the package reads the environment.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, get_args, get_type_hints

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_PUNCTUATION_PROMPT = (
    "The following is a university lecture. Transcribe it with full punctuation, "
    "including question marks on every question asked."
)

NTNU_LLM_URL = "https://llm.hpc.ntnu.no/v1"  # NTNU's LLM service: the default for vision
VISION_KEY_ENV = "QUESTXTRACT_VISION_API_KEY"  # the key for vision's own endpoint


@dataclass(frozen=True, kw_only=True)
class Config:
    """Pipeline settings, with every documented default in one place."""

    # --- 2.2 transcription ---
    asr_backend: str = "faster_whisper"
    model_size: str = "turbo"
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = True
    compute_type: str = "int8"
    batch_size: int = 4
    initial_prompt: str = _PUNCTUATION_PROMPT

    # --- 2.4 vision-informed second ASR pass ---
    hotwords_from_vision: bool = False  # not read, not planned (#62)

    # --- 3.1 sentences ---
    max_sentence_words: int = 60
    force_split_min_gap_seconds: float = 0.6

    # --- 3.2 segments ---
    segment_max_gap_seconds: float = 2.0
    segment_max_duration_seconds: float = 90.0
    segment_max_chars: int = 800

    # --- 6 vision ---
    vision_enabled: bool = False  # unless an endpoint is configured
    vision_fps: float = 1.0
    vision_model: str = "Qwen/Qwen3.8-27B-FP8"
    vision_reader: str = "vlm"  # or "ocr": vision_ocr_model reads text and formulas only (6.5)
    vision_ocr_model: str = "zai-org/GLM-OCR"  # its name on NTNU's LLM service
    vision_base_url: str | None = None  # None: base_url when set, else NTNU_LLM_URL
    vision_states_per_minute: float = 3.0  # board states allowed per minute of recording
    vision_min_states: int = 80  # never fewer, however short the recording
    vision_change_fraction: float = 0.02  # share of pixels that must change
    vision_change_hold: int = 3  # samples (at vision_fps) a new view must hold

    # --- 7 detection ---
    min_question_score: float = 0.45
    max_question_chars: int = 350
    board_overlap_delta: float = 0.0  # off: 0.10 moved nothing at the gate (architecture.md 7.3)

    # --- 8 answer search (answers beyond the adjacent seed come from the LLM detector) ---
    answer_seed_max_gap_seconds: float = 1.0

    # --- 9 scoring ---
    min_confidence: float = 0.40

    # --- 10 the LLM layer ---
    detector: str = "rules"  # or "llm": the judge reads the transcript and picks the questions
    base_url: str | None = None  # None is OpenAI; "http://localhost:11434/v1" is a local Ollama
    judge_model: str = "gpt-4o"
    judge_window: int = 10  # sentences of lecture on each side of a candidate
    judge_batch_size: int = 8  # nearby candidates judged in one request
    judge_parallel: int = 1  # requests in flight; a local server queues them past the deadline
    judge_reasoning_effort: str = ""  # "" sends no setting (gpt-4o takes none); or none..high
    timeout_seconds: int = 120  # per LLM request, start to answer, retries included

    # --- 12.3 quiz (a separate command; shares base_url and judge_reasoning_effort) ---
    quiz_model: str = "gpt-4o"
    quiz_strictness: str = "lenient"  # or "strict": only questions the lecture answered
    quiz_options: int = 3  # options per item, one correct; 3 to 5

    # --- 11.2 cache ---
    cache_dir: str = ".questxtract"
    from_scratch: bool = False

    @property
    def api_key(self) -> str | None:
        """Read the key from the environment or cwd's .env, never storing it in Config."""
        if "OPENAI_API_KEY" in os.environ:
            return os.environ["OPENAI_API_KEY"]
        return _dotenv_api_key()

    @property
    def reading_model(self) -> str:
        """The model that reads each keyframe: ``vision_model``, or the OCR model with ``ocr``.

        The layout probe always asks ``vision_model``: an OCR model cannot answer it.
        """
        return self.vision_ocr_model if self.vision_reader == "ocr" else self.vision_model

    @property
    def vision_own_endpoint(self) -> bool:
        """Whether vision has an endpoint of its own, apart from the judge's ``base_url``.

        True when ``vision_base_url`` is set, or when ``base_url`` is not (vision then goes to
        NTNU's LLM service while the judge stays on OpenAI). False when only ``base_url`` is set:
        vision follows it with the judge's key, as before vision had its own endpoint.
        """
        return self.vision_base_url is not None or self.base_url is None

    @property
    def vision_url(self) -> str | None:
        """The endpoint vision reads the board through (architecture.md 10.1)."""
        if not self.vision_own_endpoint:
            return self.base_url
        return self.vision_base_url or NTNU_LLM_URL

    @property
    def vision_api_key(self) -> str | None:
        """The key for :attr:`vision_url`, never stored in Config.

        ``QUESTXTRACT_VISION_API_KEY`` from the environment or cwd's ``.env`` for vision's own
        endpoint, so the OpenAI key never goes to another service; the judge's key when vision
        follows ``base_url``.
        """
        if not self.vision_own_endpoint:
            return self.api_key
        if VISION_KEY_ENV in os.environ:
            return os.environ[VISION_KEY_ENV]
        return _dotenv_api_key(VISION_KEY_ENV)

    def replace(self, **changes: Any) -> Config:
        """Return a copy with ``changes`` applied.

        Raises:
            TypeError: If a name is not a config field, so a typo fails loudly rather
                than silently keeping the default.

        """
        known = {f.name for f in fields(self)}
        unknown = set(changes) - known
        if unknown:
            raise TypeError(f"unknown config field(s): {', '.join(sorted(unknown))}")
        return Config(**{**{f.name: getattr(self, f.name) for f in fields(self)}, **changes})


def load_config(path: Path | None = None) -> Config:
    """Read a flat TOML file over the standard settings.

    An explicit path must exist. Without one, read ``questxtract.toml`` in the
    current directory if present, otherwise return the defaults. Relative values
    such as ``cache_dir`` remain relative to the current working directory.

    Only declared field names and their annotated types are accepted. Integer
    values are also accepted for float fields; booleans are not numbers here.
    Omit an optional key to keep its default: TOML has no null value.

    Raises:
        OSError: If the selected file cannot be read.
        ValueError: If the file is malformed, contains secrets or unknown keys,
            or gives a field a value of the wrong type. Values are never echoed.

    """
    source = Path("questxtract.toml") if path is None else path
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        if path is None:
            return Config()
        raise
    except UnicodeError:
        raise ValueError(f"{source}: expected UTF-8 text") from None

    text = text.removeprefix("\ufeff")  # Windows editors may write a UTF-8 BOM.
    try:
        values = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        # Parser messages can quote input. Keep only the location, never a value.
        location = re.search(r"\(at line (\d+), column (\d+)\)$", str(exc))
        line = int(location[1]) if location else text.count("\n") + 1
        raise ValueError(f"{source}: invalid TOML at line {line}") from None

    for key in values:
        if any(part in key.casefold() for part in ("key", "token", "secret", "password")):
            raise ValueError(
                f"{source}: {key}: secrets belong in OPENAI_API_KEY or {VISION_KEY_ENV}, not TOML"
            )

    defaults = Config()
    try:
        defaults.replace(**values)  # Reuse the existing unknown-field check.
    except TypeError as exc:
        raise ValueError(f"{source}: {exc}") from None

    annotations = get_type_hints(Config)
    for key, value in values.items():
        annotation = annotations[key]
        allowed = get_args(annotation) or (annotation,)
        if type(value) in allowed:
            continue
        if float in allowed and type(value) is int:
            values[key] = float(value)
            continue
        expected = " or ".join(t.__name__ for t in allowed if t is not type(None))
        names: dict[type[Any], str] = {str: "string", dict: "table", list: "array"}
        actual = names.get(type(value), type(value).__name__)
        raise ValueError(f"{source}: {key}: expected {expected}, got {actual}")
    return defaults.replace(**values)


def _dotenv_api_key(name: str = "OPENAI_API_KEY") -> str | None:
    """Read only the key ``name``; never execute or expand .env contents."""
    path = Path(".env")
    try:
        text = path.read_text(encoding="utf-8").removeprefix("\ufeff")
    except FileNotFoundError:
        return None
    except UnicodeError:
        raise ValueError(f"{path}: expected UTF-8 text") from None
    key_value = None
    for number, line in enumerate(text.splitlines(), 1):
        variable, separator, value = line.strip().removeprefix("export ").partition("=")
        if not separator or variable.strip() != name:
            continue
        try:
            parts = shlex.split(value, comments=True, posix=True)
        except ValueError:
            raise ValueError(f"{path}: line {number}: invalid {name} quoting") from None
        if len(parts) > 1:
            raise ValueError(f"{path}: line {number}: quote {name} values with spaces")
        key_value = parts[0] if parts else ""
        if any(marker in key_value for marker in ("$", "`")):
            raise ValueError(
                f"{path}: line {number}: expansion is not supported; "
                f"set {name} in the environment instead"
            )
    return key_value

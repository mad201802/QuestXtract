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
    hotwords_from_vision: bool = False  # not read yet, #62

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
    vision_model: str = "gpt-5.6-luna"
    vision_max_states: int = 80
    vision_change_fraction: float = 0.02  # share of pixels that must change
    vision_change_hold: int = 3  # samples (at vision_fps) a new view must hold

    # --- 7 detection ---
    min_question_score: float = 0.45
    max_question_chars: int = 350

    # --- 8 answer search (answers beyond the adjacent seed come from the LLM detector) ---
    answer_seed_max_gap_seconds: float = 1.0

    # --- 9 scoring ---
    min_confidence: float = 0.40

    # --- 10 the LLM layer ---
    detector: str = "rules"  # or "llm": the judge reads the transcript and picks the questions
    base_url: str | None = None  # None is OpenAI; "http://localhost:11434/v1" is a local Ollama
    judge_model: str = "gpt-5.6-terra"
    judge_window: int = 10  # sentences of lecture on each side of a candidate
    judge_batch_size: int = 8  # nearby candidates judged in one request
    judge_parallel: int = 1  # requests in flight; a local server queues them past the deadline
    judge_reasoning_effort: str = "none"  # "" leaves it to the server
    timeout_seconds: int = 120  # per LLM request, start to answer, retries included

    # --- 12.3 quiz (a separate command; shares base_url and judge_reasoning_effort) ---
    quiz_model: str = "gpt-5.6-terra"
    quiz_strictness: str = "strict"  # only questions the lecture answered; or "lenient"
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
            raise ValueError(f"{source}: {key}: secrets belong in OPENAI_API_KEY, not TOML")

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


def _dotenv_api_key() -> str | None:
    """Read only OPENAI_API_KEY; never execute or expand .env contents."""
    path = Path(".env")
    try:
        text = path.read_text(encoding="utf-8").removeprefix("\ufeff")
    except FileNotFoundError:
        return None
    except UnicodeError:
        raise ValueError(f"{path}: expected UTF-8 text") from None
    key_value = None
    for number, line in enumerate(text.splitlines(), 1):
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if not separator or name.strip() != "OPENAI_API_KEY":
            continue
        try:
            parts = shlex.split(value, comments=True, posix=True)
        except ValueError:
            raise ValueError(f"{path}: line {number}: invalid OPENAI_API_KEY quoting") from None
        if len(parts) > 1:
            raise ValueError(f"{path}: line {number}: quote OPENAI_API_KEY values with spaces")
        key_value = parts[0] if parts else ""
        if any(marker in key_value for marker in ("$", "`")):
            raise ValueError(
                f"{path}: line {number}: expansion is not supported; "
                "set OPENAI_API_KEY in the environment instead"
            )
    return key_value

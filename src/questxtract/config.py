"""Every pipeline setting in one flat frozen record (architecture.md 11.1)."""

from __future__ import annotations

import os
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, get_args, get_type_hints
from urllib.parse import urlsplit

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_PUNCTUATION_PROMPT = (
    "The following is a university lecture. Transcribe it with full punctuation, "
    "including question marks on every question asked."
)

NTNU_LLM_URL = "https://llm.hpc.ntnu.no/v1"
OPENAI_URL = "https://api.openai.com/v1"
NTNU_KEY_ENV = "IDUN_API_KEY"
OLD_NTNU_KEY_ENV = "QUESTXTRACT_NTNU_API_KEY"
VISION_KEY_ENV = "QUESTXTRACT_VISION_API_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
LLM_KEY_ENV = "QUESTXTRACT_LLM_API_KEY"
OLLAMA_URL = "http://localhost:11434/v1"
ENDPOINTS = {
    "ntnu": NTNU_LLM_URL,
    "openrouter": OPENROUTER_URL,
    "openai": OPENAI_URL,
    "ollama": OLLAMA_URL,
}
EXERPLAZA_EMAIL_ENV = "QUESTXTRACT_EXERPLAZA_EMAIL"
EXERPLAZA_PASSWORD_ENV = "QUESTXTRACT_EXERPLAZA_PASSWORD"  # noqa: S105
DEEPSEEK = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
GUI_START_DETECTOR = "llm"
GUI_START_VISION = True
QWEN = "Qwen/Qwen3.8-27B-FP8"


@dataclass(frozen=True, kw_only=True)
class Config:
    """Pipeline settings, with every documented default in one place."""

    asr_backend: str = "faster_whisper"
    model_size: str = "turbo"
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = True
    compute_type: str = "int8"
    batch_size: int = 4
    initial_prompt: str = _PUNCTUATION_PROMPT

    max_sentence_words: int = 60
    force_split_min_gap_seconds: float = 0.6

    segment_max_gap_seconds: float = 2.0
    segment_max_duration_seconds: float = 90.0
    segment_max_chars: int = 800

    vision_enabled: bool = False
    vision_fps: float = 1.0
    vision_model: str = DEEPSEEK
    vision_reader: str = "vlm"
    vision_ocr_model: str = "zai-org/GLM-OCR"
    vision_base_url: str | None = None
    vision_reasoning_effort: str | None = "none"
    vision_states_per_minute: float = 3.0
    vision_min_states: int = 80
    vision_change_fraction: float = 0.02
    vision_change_hold: int = 3

    min_question_score: float = 0.45
    max_question_chars: int = 350
    board_overlap_delta: float = 0.0

    answer_seed_max_gap_seconds: float = 1.0

    min_confidence: float = 0.40

    detector: str = "rules"
    base_url: str | None = NTNU_LLM_URL
    judge_model: str = QWEN
    judge_window: int = 10
    judge_batch_size: int = 8
    judge_parallel: int = 1
    reasoning_effort: str = ""
    timeout_seconds: int = 120

    quiz_model: str = QWEN
    quiz_strictness: str = "lenient"
    quiz_options: int = 3

    cache_dir: str = ".questxtract"
    from_scratch: bool = False

    def __post_init__(self) -> None:
        """Turn a short endpoint name in ``base_url`` or ``vision_base_url`` into its URL."""
        object.__setattr__(self, "base_url", endpoint_url(self.base_url))
        object.__setattr__(self, "vision_base_url", endpoint_url(self.vision_base_url))

    @property
    def api_key(self) -> str | None:
        """The key for ``base_url`` (the judge and the quiz), never stored in Config."""
        return endpoint_key(self.base_url)

    @property
    def reading_model(self) -> str:
        """The model that reads each keyframe: ``vision_model``, or the OCR model with ``ocr``.

        The layout probe always asks ``vision_model``: an OCR model cannot answer it.
        """
        return self.vision_ocr_model if self.vision_reader == "ocr" else self.vision_model

    @property
    def vision_url(self) -> str | None:
        """The endpoint vision reads the board through (architecture.md 10.1).

        ``vision_base_url`` when set, else ``base_url``: by default NTNU's service for both.
        """
        return self.vision_base_url if self.vision_base_url is not None else self.base_url

    @property
    def vision_effort(self) -> str:
        """The ``reasoning_effort`` sent to ``vision_model``: the layout probe and its readings.

        ``vision_reasoning_effort`` when set, else ``reasoning_effort``. The OCR model gets none.
        """
        if self.vision_reasoning_effort is None:
            return self.reasoning_effort
        return self.vision_reasoning_effort

    @property
    def vision_api_key(self) -> str | None:
        """The key for :attr:`vision_url`, never stored in Config.

        A ``vision_base_url`` of its own on a host without a key of its own (not NTNU's,
        OpenAI's or OpenRouter's) reads ``QUESTXTRACT_VISION_API_KEY``, so the judge's key is
        not sent to a host set up for vision alone.
        """
        if self.vision_base_url is None:
            return endpoint_key(self.base_url)
        return endpoint_key(self.vision_base_url, other=VISION_KEY_ENV)

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
    """Read a flat TOML file over the standard settings."""
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
                f"{source}: {key}: secrets belong in the endpoint's key variable, such as "
                f"{NTNU_KEY_ENV} or {OPENROUTER_KEY_ENV}, not TOML"
            )

    if "judge_reasoning_effort" in values:
        if "reasoning_effort" in values:
            raise ValueError(
                f"{source}: set reasoning_effort only (judge_reasoning_effort is its old name)"
            )
        values["reasoning_effort"] = values.pop("judge_reasoning_effort")

    defaults = Config()
    try:
        defaults.replace(**values)
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


def _host(url: str) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = -1
    if port not in (None, 80, 443):
        host = f"{host}:{port}"
    return host


def endpoint_url(url: str | None) -> str | None:
    """Return the URL of a short endpoint name (:data:`ENDPOINTS`, any case), else ``url``."""
    if not isinstance(url, str):  # None, or a wrong type load_config reports
        return url
    return ENDPOINTS.get(url.strip().lower(), url)


def is_ntnu(url: str | None) -> bool:
    """Whether ``url`` is NTNU's LLM service."""
    return url is not None and _host(url) == _host(NTNU_LLM_URL)


def is_openai(url: str | None) -> bool:
    """Whether ``url`` is OpenAI: ``None`` (the SDK's own default) or :data:`OPENAI_URL`."""
    return url is None or _host(url) == _host(OPENAI_URL)


def is_openrouter(url: str | None) -> bool:
    """Whether ``url`` is OpenRouter."""
    return url is not None and _host(url) == _host(OPENROUTER_URL)


def sends_key_unencrypted(url: str | None) -> bool:
    """Whether ``url`` is NTNU's, OpenAI's or OpenRouter's host over plain ``http``."""
    known = is_ntnu(url) or is_openrouter(url) or (url is not None and is_openai(url))
    return known and urlsplit(str(url).strip()).scheme.lower() != "https"


def key_names(url: str | None, *, other: str = LLM_KEY_ENV) -> tuple[str, ...]:
    """Return the variables the key for ``url`` is read from, the first one set winning."""
    if is_ntnu(url):
        return (NTNU_KEY_ENV, OLD_NTNU_KEY_ENV)
    if is_openai(url):
        return (OPENAI_KEY_ENV,)
    if is_openrouter(url):
        return (OPENROUTER_KEY_ENV,)
    return (other,)


def endpoint_key(url: str | None, *, other: str = LLM_KEY_ENV) -> str | None:
    """Read the key for ``url`` from the environment, cwd's ``.env`` or the user's key file."""
    names = key_names(url, other=other)
    for name in names:
        key = _read_key(name)
        if key is not None:
            return key
    for name in names:
        key = _dotenv_api_key(name, user_env_path())
        if key is not None:
            return key
    return None


def _read_key(name: str) -> str | None:
    if name in os.environ:
        return os.environ[name]
    return _dotenv_api_key(name)


def exerplaza_credentials() -> tuple[str | None, str | None]:
    """Read the ExerPlaza email and password from the environment or cwd's ``.env``.

    Like the API keys: never a command-line argument, never stored in a ``Config``. The app's
    key file holds API keys only, so it is not read here.
    """
    return _read_key(EXERPLAZA_EMAIL_ENV), _read_key(EXERPLAZA_PASSWORD_ENV)


def user_env_path() -> Path:
    """Return the file the app saves API keys in: ``questxtract/.env`` in the config folder."""
    home = Path.home()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        configured = os.environ.get("XDG_CONFIG_HOME", "")
        base = (
            Path(configured) if configured and Path(configured).is_absolute() else home / ".config"
        )
    return base / "questxtract" / ".env"


def valid_key(key: str) -> str:
    """Return ``key`` without surrounding whitespace, or raise why the ``.env`` cannot hold it."""
    key = key.strip()
    if not key:
        raise ValueError("the key is empty")
    if any(char.isspace() or char in "\\\"'$`" for char in key):
        raise ValueError("the key has a space, quote, backslash, $ or backtick; copy it again")
    return key


def save_user_key(key: str, name: str = NTNU_KEY_ENV) -> Path:
    """Save ``key`` in :func:`user_env_path` as ``name``, keeping every other line."""
    key = valid_key(key)
    path = user_env_path()
    lines = _env_lines(path)
    entry = f'{name}="{key}"'
    replaced = False
    kept: list[str] = []
    for line in lines:
        if _variable(line) != name:
            kept.append(line)
        elif not replaced:
            kept.append(entry)
            replaced = True
    if not replaced:
        kept.append(entry)
    _write_env(path, kept)
    return path


def remove_user_key(name: str = NTNU_KEY_ENV) -> bool:
    """Remove ``name`` from :func:`user_env_path`, keeping every other line."""
    path = user_env_path()
    lines = _env_lines(path)
    kept = [line for line in lines if _variable(line) != name]
    if len(kept) == len(lines):
        return False
    if any(line.strip() for line in kept):
        _write_env(path, kept)
    else:
        path.unlink()
    return True


def _variable(line: str) -> str:
    return line.strip().removeprefix("export ").partition("=")[0].strip() if "=" in line else ""


def _env_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").removeprefix("\ufeff").splitlines()
    except FileNotFoundError:
        return []
    except UnicodeError:
        raise ValueError(f"{path}: expected UTF-8 text") from None


def _write_env(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            out.write("\n".join(lines) + "\n")
        os.chmod(name, 0o600)
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _dotenv_api_key(name: str = OPENAI_KEY_ENV, path: Path | None = None) -> str | None:
    path = path if path is not None else Path(".env")
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

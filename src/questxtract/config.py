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

NTNU_LLM_URL = "https://llm.hpc.ntnu.no/v1"  # NTNU's LLM service: the default endpoint
OPENAI_URL = "https://api.openai.com/v1"  # OpenAI, the documented fallback (also base_url None)
NTNU_KEY_ENV = "IDUN_API_KEY"  # the key for NTNU's LLM service
OLD_NTNU_KEY_ENV = "QUESTXTRACT_NTNU_API_KEY"  # its former name, still read when the new is unset
VISION_KEY_ENV = "QUESTXTRACT_VISION_API_KEY"  # for a vision_base_url of another service
OPENAI_KEY_ENV = "OPENAI_API_KEY"  # the key for OpenAI and base_url's other endpoints
DEEPSEEK = "deepseek-ai/DeepSeek-V4.1-Flash-1017"  # vision's default, on NTNU's service
# The graphical app opens with the LLM judge and the board reading on (#124). The CLI's own
# defaults are the ones above; these only set where the window starts.
GUI_START_DETECTOR = "llm"
GUI_START_VISION = True
QWEN = "Qwen/Qwen3.8-27B-FP8"  # the judge's and the quiz's default, on NTNU's service


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
    vision_model: str = DEEPSEEK
    vision_reader: str = "vlm"  # or "ocr": vision_ocr_model reads text and formulas only (6.5)
    vision_ocr_model: str = "zai-org/GLM-OCR"  # its name on NTNU's LLM service
    vision_base_url: str | None = None  # None: vision follows base_url
    # DeepSeek thinks unless told "none"; None follows reasoning_effort; never sent to OCR
    vision_reasoning_effort: str | None = "none"
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
    base_url: str | None = NTNU_LLM_URL  # None or OPENAI_URL is OpenAI; or e.g. a local Ollama
    judge_model: str = QWEN
    judge_window: int = 10  # sentences of lecture on each side of a candidate
    judge_batch_size: int = 8  # nearby candidates judged in one request
    judge_parallel: int = 1  # requests in flight; a local server queues them past the deadline
    # Sent with every judge and quiz request, and vision's when vision_reasoning_effort is
    # None. "" sends no setting, for a model that rejects "none" (Qwen) or takes none
    # (gpt-4o); "none" switches DeepSeek's thinking off.
    reasoning_effort: str = ""
    timeout_seconds: int = 120  # per LLM request, start to answer, retries included

    # --- 12.3 quiz (a separate command; shares base_url and reasoning_effort) ---
    quiz_model: str = QWEN
    quiz_strictness: str = "lenient"  # or "strict": only questions the lecture answered
    quiz_options: int = 3  # options per item, one correct; 3 to 5

    # --- 11.2 cache ---
    cache_dir: str = ".questxtract"
    from_scratch: bool = False

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

        A ``vision_base_url`` of its own that is neither NTNU's service nor OpenAI (another
        hosted vision model) reads ``QUESTXTRACT_VISION_API_KEY``, so neither the OpenAI nor
        the NTNU key is sent to a host set up for vision alone.
        """
        url = self.vision_url
        if self.vision_base_url is not None and not is_ntnu(url) and not is_openai(url):
            return _read_key(VISION_KEY_ENV)
        return endpoint_key(url)

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
                f"{source}: {key}: secrets belong in {NTNU_KEY_ENV} or {OPENAI_KEY_ENV}, not TOML"
            )

    if "judge_reasoning_effort" in values:  # its old name, from before it covered every request
        if "reasoning_effort" in values:
            raise ValueError(
                f"{source}: set reasoning_effort only (judge_reasoning_effort is its old name)"
            )
        values["reasoning_effort"] = values.pop("judge_reasoning_effort")

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


def _host(url: str) -> str:
    """Return the host of ``url``, lower case, default port dropped.

    Neither scheme nor path counts: a key belongs to the service, however its URL is written.
    """
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:  # not a number: not a service we know
        port = -1
    if port not in (None, 80, 443):
        host = f"{host}:{port}"
    return host


def is_ntnu(url: str | None) -> bool:
    """Whether ``url`` is NTNU's LLM service."""
    return url is not None and _host(url) == _host(NTNU_LLM_URL)


def is_openai(url: str | None) -> bool:
    """Whether ``url`` is OpenAI: ``None`` (the SDK's own default) or :data:`OPENAI_URL`."""
    return url is None or _host(url) == _host(OPENAI_URL)


def endpoint_key(url: str | None) -> str | None:
    """Read the key for ``url`` from the environment, cwd's ``.env`` or the user's key file.

    NTNU's service has a key of its own, ``IDUN_API_KEY`` (or its former name,
    ``QUESTXTRACT_NTNU_API_KEY``, when that is unset), so the OpenAI key never
    goes there and the NTNU key never goes anywhere else. Every other endpoint (OpenAI,
    OpenRouter, a local server) reads ``OPENAI_API_KEY``, which a local server may leave unset.
    The key is never stored in a ``Config``.
    """
    if not is_ntnu(url):
        return _read_key(OPENAI_KEY_ENV)
    names = (NTNU_KEY_ENV, OLD_NTNU_KEY_ENV)
    for name in names:
        key = _read_key(name)
        if key is not None:
            return key
    for name in names:  # the app's key file comes after both names in the environment and .env
        key = _dotenv_api_key(name, user_env_path())
        if key is not None:
            return key
    return None


def _read_key(name: str) -> str | None:
    """Return ``name`` from the environment, else from cwd's ``.env``."""
    if name in os.environ:
        return os.environ[name]
    return _dotenv_api_key(name)


def user_env_path() -> Path:
    """Return the file the app saves the NTNU key in: ``questxtract/.env`` in the config folder.

    The folder is ``%APPDATA%`` on Windows, ``~/Library/Application Support`` on macOS and
    ``$XDG_CONFIG_HOME`` (else ``~/.config``) elsewhere, so a key saved once is found from any
    working directory.
    """
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
    """Return ``key`` without surrounding whitespace, or raise why the ``.env`` cannot hold it.

    Raises:
        ValueError: If the key is empty, or holds whitespace, a quote, a backslash, ``$`` or a
            backtick, which the ``.env`` reader refuses or reads differently.

    """
    key = key.strip()
    if not key:
        raise ValueError("the key is empty")
    if any(char.isspace() or char in "\\\"'$`" for char in key):
        raise ValueError("the key has a space, quote, backslash, $ or backtick; copy it again")
    return key


def save_user_key(key: str) -> Path:
    """Save the NTNU key in :func:`user_env_path` as ``IDUN_API_KEY``, keeping every other line.

    Replaces an earlier line for the key and writes the file in one step with mode ``0600`` where
    the system has it. The value is written in double quotes.

    Returns:
        The file written.

    Raises:
        ValueError: If :func:`valid_key` refuses the key.
        OSError: If the file cannot be read or written.

    """
    key = valid_key(key)
    path = user_env_path()
    lines = _env_lines(path)
    entry = f'{NTNU_KEY_ENV}="{key}"'
    replaced = False
    kept: list[str] = []
    for line in lines:
        if _variable(line) != NTNU_KEY_ENV:
            kept.append(line)
        elif not replaced:  # the reader takes the last line, so later copies would win: drop them
            kept.append(entry)
            replaced = True
    if not replaced:
        kept.append(entry)
    _write_env(path, kept)
    return path


def remove_user_key() -> bool:
    """Remove ``IDUN_API_KEY`` from :func:`user_env_path`, keeping every other line.

    The file is deleted when nothing but blank lines is left.

    Returns:
        Whether a key line was there.

    Raises:
        OSError: If the file cannot be read or written.

    """
    path = user_env_path()
    lines = _env_lines(path)
    kept = [line for line in lines if _variable(line) != NTNU_KEY_ENV]
    if len(kept) == len(lines):
        return False
    if any(line.strip() for line in kept):
        _write_env(path, kept)
    else:
        path.unlink()
    return True


def _variable(line: str) -> str:
    """Return the variable a ``.env`` line sets, cut as the reader cuts it, or ``""``."""
    return line.strip().removeprefix("export ").partition("=")[0].strip() if "=" in line else ""


def _env_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").removeprefix("\ufeff").splitlines()
    except FileNotFoundError:
        return []
    except UnicodeError:
        raise ValueError(f"{path}: expected UTF-8 text") from None


def _write_env(path: Path, lines: list[str]) -> None:
    """Write ``lines`` to ``path`` through a temporary file in its folder, private to the user.

    The folder is made private too, also one created before. On Windows ``chmod`` changes
    neither: the user's profile folder already keeps other users out.
    """
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
    """Read only the key ``name`` from ``path`` (cwd's ``.env``); never execute or expand it."""
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

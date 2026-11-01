"""The OpenAI-compatible endpoints of the judge and vision (architecture.md 10.1)."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import Config

# Sent to an endpoint without a key: with None the SDK would read OPENAI_API_KEY itself.
NO_KEY = "local"


class InvalidAnswer(ValueError):
    """The model answered, twice, with something other than the JSON object asked for."""


class Stalled(ConnectionError):
    """The endpoint gave no answer within the deadline, twice."""


def route(config: Config, *, vision: bool = False) -> tuple[str | None, str | None]:
    """Return the endpoint and key a request goes to: the judge's, or vision's."""
    if vision:
        return config.vision_url, config.vision_api_key
    return config.base_url, config.api_key


class Refused(ConnectionError):
    """The endpoint refused the request itself (HTTP 400), as for a parameter it does not take."""


def client(config: Config, *, vision: bool = False) -> Any:
    """Build the OpenAI-compatible client for the judge, or for vision with ``vision=True``."""
    try:
        from openai import OpenAI  # noqa: PLC0415 - imported lazily so the base install is light
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise RuntimeError("the LLM layer needs `pip install questxtract[llm]`") from exc
    base_url, api_key = route(config, vision=vision)
    return OpenAI(
        api_key=api_key or NO_KEY,
        base_url=base_url,
        timeout=config.timeout_seconds,
    )


def ask(
    config: Config,
    request: dict[str, Any],
    *,
    name: str,
    valid: Callable[[dict[str, Any]], bool] | None = None,
    usage: Counter[str] | None = None,
    vision: bool = False,
    convert: Callable[[Any], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Send ``request`` and return the JSON object it answers with; cached on the request."""
    base_url, key_for_errors = route(config, vision=vision)
    key = json.dumps([base_url, request], sort_keys=True)
    path = Path(config.cache_dir) / name / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
    if not config.from_scratch and path.is_file():
        try:
            cached = _parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            cached = None
        if cached is not None and (valid is None or valid(cached)):
            return cached

    llm = client(config, vision=vision)
    from openai import OpenAIError  # noqa: PLC0415 - the optional extra, as in client()

    endpoint = base_url or "OpenAI"
    api_key = getattr(llm, "api_key", None) or key_for_errors
    if api_key == NO_KEY:  # the placeholder is no secret, and hiding it mangles "localhost"
        api_key = None

    def failure(detail: str, kind: type[ConnectionError] = ConnectionError) -> ConnectionError:
        # Providers may echo credentials in errors; keep diagnostics but hide the key.
        message = f"{name}: {endpoint}: {detail}"
        return kind(message.replace(api_key, "[REDACTED]") if api_key else message)

    stalled: TimeoutError | None = None
    # Once more after a stall or invalid JSON; the SDK retries HTTP errors itself.
    for _ in range(2):
        started = time.perf_counter()
        try:
            response = _create(llm, request, config.timeout_seconds)
        except TimeoutError as exc:
            _count(usage, name, None, started)
            stalled = exc
            continue
        except OpenAIError as exc:
            _count(usage, name, None, started)
            refused = getattr(exc, "status_code", None) == 400
            raise failure(str(exc), Refused if refused else ConnectionError) from None
        _count(usage, name, response, started)
        stalled = None
        if convert is None:
            content = response.choices[0].message.content
            answer = _parse(content)
        else:
            answer = convert(response.choices[0])
            content = json.dumps(answer, ensure_ascii=False)
        if answer is not None and (valid is None or valid(answer)):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            except OSError:
                pass
            return answer
    if stalled is not None:
        raise failure(f"{stalled}, twice", Stalled) from None
    raise InvalidAnswer(f"{name}: {request['model']} did not return the expected JSON, twice")


def _count(usage: Counter[str] | None, name: str, response: Any, started: float) -> None:
    if usage is None:
        return
    tokens = getattr(response, "usage", None)
    usage[f"{name}_requests"] += 1
    usage[f"{name}_input_tokens"] += getattr(tokens, "prompt_tokens", None) or 0
    usage[f"{name}_output_tokens"] += getattr(tokens, "completion_tokens", None) or 0
    usage[f"{name}_ms"] += round((time.perf_counter() - started) * 1000)


def _create(llm: Any, request: dict[str, Any], seconds: float) -> Any:
    outcome: list[Any] = []

    def call() -> None:
        try:
            outcome.append(llm.chat.completions.create(**request))
        except Exception as exc:
            outcome.append(exc)

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    worker.join(seconds)
    if not outcome:
        raise TimeoutError(f"no answer within {seconds:g} s")
    if isinstance(outcome[0], Exception):
        raise outcome[0]
    return outcome[0]


def _parse(content: str | None) -> dict[str, Any] | None:
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None

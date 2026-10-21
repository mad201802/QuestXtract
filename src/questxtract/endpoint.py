"""The OpenAI-compatible endpoint the judge and vision share (architecture.md 10.1).

Not a stage: it holds the request code both model calls need, so neither stage imports the
other (architecture.md 11.1 rule 1). Each stage keeps its own prompt and schema, and decides
what a failure means: the judge ends the run, vision drops the one keyframe.
"""

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


class InvalidAnswer(ValueError):
    """The model answered, twice, with something other than the JSON object asked for.

    A ``ValueError`` so existing callers keep working. It is its own type so a caller that
    tolerates a bad answer (vision) does not also tolerate a setup error, such as an unreadable
    ``.env``, which is a plain ``ValueError`` raised before any request.
    """


class Stalled(ConnectionError):
    """The endpoint gave no answer within the deadline, twice.

    A ``ConnectionError`` so existing callers keep working. It is its own type so a caller that
    skips one slow request (the quiz) does not also skip an unreachable endpoint or a bad key,
    which every later request would hit the same way.
    """


def client(config: Config) -> Any:
    """Build the OpenAI-compatible client.

    ``base_url`` is what makes the offline story work without a second code path: any local
    OpenAI-compatible server (Ollama, llama.cpp, vLLM, LM Studio) plugs in with zero new code.

    Raises:
        RuntimeError: If the optional ``questxtract[llm]`` extra is not installed.

    """
    try:
        from openai import OpenAI  # noqa: PLC0415 - imported lazily so the base install is light
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise RuntimeError("the LLM layer needs `pip install questxtract[llm]`") from exc
    return OpenAI(
        api_key=config.api_key or "local",
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )


def ask(
    config: Config,
    request: dict[str, Any],
    *,
    name: str,
    valid: Callable[[dict[str, Any]], bool] | None = None,
    usage: Counter[str] | None = None,
) -> dict[str, Any]:
    """Send ``request`` and return the JSON object it answers with; cached on the request.

    The cache key is the endpoint plus the whole request, so any change to what the model
    would see asks again, and nothing else does. Entries live under ``cache_dir/<name>/``;
    ``name`` also prefixes every error. Only a valid answer is cached: a JSON object that
    ``valid``, when given, accepts. Anything else counts as invalid JSON, asked once more and
    never cached, so a rejected answer is not replayed on the next run. A cache that cannot be
    read or written only costs a repeat request: the answer already paid for is returned.

    ``usage``, when given, adds what the calls cost: ``<name>_requests`` (every request sent,
    a repeat or a failed one included), ``<name>_input_tokens`` and ``<name>_output_tokens``
    (the API's ``usage``; 0 from a server that reports none) and ``<name>_ms``. A cached
    answer adds nothing.

    Raises:
        RuntimeError: If the optional ``questxtract[llm]`` extra is not installed.
        ConnectionError: If the endpoint fails after the SDK's own retries.
        Stalled: If it gives no answer within the deadline, twice (a ``ConnectionError``).
        InvalidAnswer: If the model answers with something other than a JSON object twice.
        ValueError: If the settings cannot be used, such as an unreadable ``.env``.

    """
    key = json.dumps([config.base_url, request], sort_keys=True)
    path = Path(config.cache_dir) / name / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
    if not config.from_scratch and path.is_file():
        try:
            cached = _parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            cached = None
        if cached is not None and (valid is None or valid(cached)):
            return cached  # a damaged or rejected entry is asked again

    llm = client(config)  # first: without the extra, it raises the install hint
    from openai import OpenAIError  # noqa: PLC0415 - the optional extra, as in client()

    endpoint = config.base_url or "OpenAI"
    api_key = getattr(llm, "api_key", None) or config.api_key

    def failure(detail: str, kind: type[ConnectionError] = ConnectionError) -> ConnectionError:
        # Providers may echo credentials in errors; keep diagnostics but hide the key.
        message = f"{name}: {endpoint}: {detail}"
        return kind(message.replace(api_key, "[REDACTED]") if api_key else message)

    stalled: TimeoutError | None = None
    # Once more after a stall or invalid JSON; the SDK retries HTTP errors itself. A stall is
    # usually the provider: the same model answers in seconds before and after one.
    for _ in range(2):
        started = time.perf_counter()
        try:
            response = _create(llm, request, config.timeout_seconds)
        except TimeoutError as exc:
            _count(usage, name, None, started)
            stalled = exc
            continue
        except OpenAIError as exc:
            _count(usage, name, None, started)  # sent, though it failed
            raise failure(str(exc)) from None
        _count(usage, name, response, started)
        stalled = None
        content = response.choices[0].message.content
        answer = _parse(content)
        if answer is not None and (valid is None or valid(answer)):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            except OSError:
                pass  # a read-only or full cache loses the saving, not the answer just paid for
            return answer
    if stalled is not None:
        raise failure(f"{stalled}, twice", Stalled) from None
    raise InvalidAnswer(f"{name}: {request['model']} did not return the expected JSON, twice")


def _count(usage: Counter[str] | None, name: str, response: Any, started: float) -> None:
    """Add one sent request, its tokens and its time to ``usage``."""
    if usage is None:
        return
    tokens = getattr(response, "usage", None)
    usage[f"{name}_requests"] += 1
    usage[f"{name}_input_tokens"] += getattr(tokens, "prompt_tokens", None) or 0
    usage[f"{name}_output_tokens"] += getattr(tokens, "completion_tokens", None) or 0
    usage[f"{name}_ms"] += round((time.perf_counter() - started) * 1000)


def _create(llm: Any, request: dict[str, Any], seconds: float) -> Any:
    """Make one request, with a deadline on the whole call.

    The SDK's timeout is per read, and a server that keeps the connection alive while the model
    works never trips it: OpenRouter sends filler bytes, with no overall limit, so a stalled
    request would hang the run. The call runs in a daemon thread and is abandoned past the
    deadline; the process does not wait for it on exit.

    Raises:
        TimeoutError: If no answer arrived within ``seconds``, the SDK's retries included.

    """
    outcome: list[Any] = []

    def call() -> None:
        try:
            outcome.append(llm.chat.completions.create(**request))
        except Exception as exc:  # re-raised in the caller's thread below
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
    """Return a response's JSON object, or ``None`` when it is not one."""
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None

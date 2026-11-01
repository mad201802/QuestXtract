"""The API key window's logic without widgets (#189, #250): test, save, remove and show."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from ..config import (
    NTNU_LLM_URL,
    _dotenv_api_key,
    _host,
    is_ntnu,
    is_openai,
    is_openrouter,
    key_names,
    remove_user_key,
    save_user_key,
    sends_key_unencrypted,
    user_env_path,
    valid_key,
)

__all__ = [
    "check_key",
    "endpoint_name",
    "key_file",
    "key_source",
    "remove_key",
    "save_key",
    "valid_key",
]

_TIMEOUT_SECONDS = 10.0


def endpoint_name(url: str | None = NTNU_LLM_URL) -> str:
    """Return the endpoint as the window names it: NTNU, OpenAI, OpenRouter, or its host."""
    if is_ntnu(url):
        return "NTNU"
    if is_openai(url):
        return "OpenAI"
    if is_openrouter(url):
        return "OpenRouter"
    return _host(url or "") or str(url)


def check_key(key: str, url: str | None = NTNU_LLM_URL) -> str:
    """Ask the endpoint at ``url`` whether it accepts ``key``: one request that needs the key."""
    if sends_key_unencrypted(url):
        return "insecure"
    try:
        import openai  # noqa: PLC0415 - the optional questxtract[llm] extra
    except ImportError:
        return "no_llm_extra"
    client = openai.OpenAI(
        api_key=key.strip(), base_url=url, timeout=_TIMEOUT_SECONDS, max_retries=0
    )
    try:
        if is_openrouter(url):
            # OpenRouter lists its models to anyone: https://openrouter.ai/docs/api-reference/limits
            client.get("/key", cast_to=object)
        else:
            client.models.list()
    except (openai.AuthenticationError, openai.PermissionDeniedError):
        return "refused"
    except openai.APIError:
        return "unreachable"
    return "accepted"


def save_key(
    key: str, url: str | None = NTNU_LLM_URL, check: Callable[[str], str] | None = None
) -> str:
    """Test ``key`` against ``url``, then save it as that endpoint's key unless it was refused."""
    key = valid_key(key)
    outcome = check(key) if check is not None else check_key(key, url)
    if outcome in ("refused", "insecure"):
        return outcome
    save_user_key(key, key_names(url)[0])
    return "saved" if outcome == "accepted" else "saved_unchecked"


def remove_key(url: str | None = NTNU_LLM_URL) -> bool:
    """Remove the saved key of ``url``; return whether there was one."""
    return remove_user_key(key_names(url)[0])


def key_source(url: str | None = NTNU_LLM_URL) -> str | None:
    """Say where a run would read the key of ``url`` from, without returning it."""
    names = key_names(url)
    for name in names:
        if name in os.environ:
            return "environment"
        if _dotenv_api_key(name) is not None:
            return "dotenv"
    if any(_dotenv_api_key(name, user_env_path()) is not None for name in names):
        return "file"
    return None


def key_file() -> Path:
    """Return the file the key is saved in, to show the user where it is."""
    return user_env_path()

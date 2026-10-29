"""The NTNU key window's logic without widgets (issue #189): test, save, remove and show."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from ..config import (
    NTNU_KEY_ENV,
    NTNU_LLM_URL,
    OLD_NTNU_KEY_ENV,
    _dotenv_api_key,
    remove_user_key,
    save_user_key,
    user_env_path,
    valid_key,
)

__all__ = ["check_key", "key_file", "key_source", "remove_key", "save_key", "valid_key"]

_TIMEOUT_SECONDS = 10.0


def check_key(key: str) -> str:
    """Ask NTNU's service whether it accepts ``key``: one request that lists its models."""
    try:
        import openai  # noqa: PLC0415 - the optional questxtract[llm] extra
    except ImportError:
        return "no_llm_extra"
    client = openai.OpenAI(
        api_key=key.strip(), base_url=NTNU_LLM_URL, timeout=_TIMEOUT_SECONDS, max_retries=0
    )
    try:
        client.models.list()
    except (openai.AuthenticationError, openai.PermissionDeniedError):
        return "refused"
    except openai.APIError:
        return "unreachable"
    return "accepted"


def save_key(key: str, check: Callable[[str], str] | None = None) -> str:
    """Test ``key``, then save it unless the service refused it."""
    key = valid_key(key)
    outcome = (check or check_key)(key)
    if outcome == "refused":
        return "refused"
    save_user_key(key)
    return "saved" if outcome == "accepted" else "saved_unchecked"


def remove_key() -> bool:
    """Remove the saved key; return whether there was one."""
    return remove_user_key()


def key_source() -> str | None:
    """Say where a run would read the NTNU key from, without returning it."""
    names = (NTNU_KEY_ENV, OLD_NTNU_KEY_ENV)
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

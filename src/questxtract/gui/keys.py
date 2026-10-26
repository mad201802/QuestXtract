"""The NTNU key window's logic without widgets (issue #189): test, save, remove and show.

The key lives in the per-user file of :func:`questxtract.config.user_env_path`, in plain text,
as in any ``.env``. It is never printed, logged or put in an argument list: the functions here
return short codes, and the window turns them into words (``key.*`` in :mod:`.i18n`).

Nothing here imports tkinter, so it is tested headless (``tests/test_gui_keys.py``).
"""

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
    """Ask NTNU's service whether it accepts ``key``: one request that lists its models.

    Returns:
        ``accepted``; ``refused`` (the service answered 401 or 403); ``unreachable`` (no
        answer, or any other answer: the NTNU network or VPN is the usual cause); or
        ``no_llm_extra`` when ``questxtract[llm]`` is not installed.

    """
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
    """Test ``key``, then save it unless the service refused it.

    ``check`` replaces :func:`check_key` (the tests give it a fake).
    A key that cannot be tested (no network, no ``llm`` extra) is saved anyway: a teacher off
    the NTNU network can still set it up.

    Returns:
        ``saved`` when the service accepted it, ``saved_unchecked`` when it could not be
        tested, or ``refused`` when it was not saved.

    Raises:
        ValueError: If the key is empty or holds a character the ``.env`` reader refuses.
        OSError: If the file cannot be written.

    """
    key = valid_key(key)  # a key the file cannot hold fails here, before any request
    outcome = (check or check_key)(key)
    if outcome == "refused":
        return "refused"  # an earlier saved key stays as it was
    save_user_key(key)
    return "saved" if outcome == "accepted" else "saved_unchecked"


def remove_key() -> bool:
    """Remove the saved key; return whether there was one."""
    return remove_user_key()


def key_source() -> str | None:
    """Say where a run would read the NTNU key from, without returning it.

    Returns:
        ``environment``, ``dotenv`` (``.env`` in the working directory), ``file`` (the app's
        key file), or ``None`` for no key. The order is the one a run uses.

    """
    names = (NTNU_KEY_ENV, OLD_NTNU_KEY_ENV)
    if any(name in os.environ for name in names):
        return "environment"
    if any(_dotenv_api_key(name) is not None for name in names):
        return "dotenv"
    if any(_dotenv_api_key(name, user_env_path()) is not None for name in names):
        return "file"
    return None


def key_file() -> Path:
    """Return the file the key is saved in, to show the user where it is."""
    return user_env_path()

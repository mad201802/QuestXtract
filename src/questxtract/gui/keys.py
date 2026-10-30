"""The API key window's logic without widgets (#189, #250): test, save, remove and show.

Each function works on the key of one endpoint, NTNU's service unless ``url`` says otherwise,
under that endpoint's variable (:func:`questxtract.config.key_names`). The keys live in the
per-user file of :func:`questxtract.config.user_env_path`, in plain text, as in any ``.env``. A
key is never printed, logged or put in an argument list: the functions here return short codes,
and the window turns them into words (``key.*`` in :mod:`.i18n`).

Nothing here imports tkinter, so it is tested headless (``tests/test_gui_keys.py``).
"""

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
    """Ask the endpoint at ``url`` whether it accepts ``key``: one request that lists its models.

    Returns:
        ``accepted``; ``refused`` (the service answered 401 or 403); ``unreachable`` (no
        answer, or any other answer: for NTNU's service, the NTNU network or VPN is the usual
        cause); or ``no_llm_extra`` when ``questxtract[llm]`` is not installed.

    """
    try:
        import openai  # noqa: PLC0415 - the optional questxtract[llm] extra
    except ImportError:
        return "no_llm_extra"
    client = openai.OpenAI(
        api_key=key.strip(), base_url=url, timeout=_TIMEOUT_SECONDS, max_retries=0
    )
    try:
        client.models.list()
    except (openai.AuthenticationError, openai.PermissionDeniedError):
        return "refused"
    except openai.APIError:
        return "unreachable"
    return "accepted"


def save_key(
    key: str, url: str | None = NTNU_LLM_URL, check: Callable[[str], str] | None = None
) -> str:
    """Test ``key`` against ``url``, then save it as that endpoint's key unless it was refused.

    ``check`` replaces :func:`check_key` (the tests give it a fake). The keys of the other
    endpoints in the file are kept.
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
    outcome = check(key) if check is not None else check_key(key, url)
    if outcome == "refused":
        return "refused"  # an earlier saved key stays as it was
    save_user_key(key, key_names(url)[0])
    return "saved" if outcome == "accepted" else "saved_unchecked"


def remove_key(url: str | None = NTNU_LLM_URL) -> bool:
    """Remove the saved key of ``url``; return whether there was one."""
    return remove_user_key(key_names(url)[0])


def key_source(url: str | None = NTNU_LLM_URL) -> str | None:
    """Say where a run would read the key of ``url`` from, without returning it.

    Returns:
        ``environment``, ``dotenv`` (``.env`` in the working directory), ``file`` (the app's
        key file), or ``None`` for no key. The order is the one a run uses.

    """
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

"""The API key window's logic (#189, #250), headless: test, save, remove, where it is read from."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from questxtract.config import Config, user_env_path
from questxtract.gui import keys

MARKER = "sk-marker-4f9c1e"


def _fake_openai(monkeypatch: pytest.MonkeyPatch, raises: type[Exception] | None = None) -> list:
    """Install an ``openai`` that records its calls and raises ``raises`` on listing models."""
    calls: list[dict[str, object]] = []

    class APIError(Exception):
        pass

    class Auth(APIError):
        pass

    class Denied(APIError):
        pass

    class Connection(APIError):
        pass

    class OpenAI:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)
            self.models = types.SimpleNamespace(list=lambda: self._list("models"))

        def get(self, path: str, *, cast_to: object) -> list:
            return self._list(path)

        def _list(self, path: str) -> list:
            calls.append({"asked": path})
            if raises is not None:
                raise {"auth": Auth, "denied": Denied, "connection": Connection}[raises]("x")
            return []

    module = types.ModuleType("openai")
    module.OpenAI = OpenAI  # type: ignore[attr-defined]
    module.APIError = APIError  # type: ignore[attr-defined]
    module.AuthenticationError = Auth  # type: ignore[attr-defined]
    module.PermissionDeniedError = Denied  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return calls


@pytest.mark.parametrize(
    ("raises", "expected"),
    [(None, "accepted"), ("auth", "refused"), ("denied", "refused"), ("connection", "unreachable")],
)
def test_the_check_tells_accepted_refused_and_unreachable(monkeypatch, raises, expected):
    calls = _fake_openai(monkeypatch, raises)  # type: ignore[arg-type]

    assert keys.check_key(f" {MARKER} ") == expected

    assert calls == [
        {
            "api_key": MARKER,
            "base_url": "https://llm.hpc.ntnu.no/v1",
            "timeout": 10.0,
            "max_retries": 0,
        },
        {"asked": "models"},
    ]


def test_the_check_says_when_the_llm_extra_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)  # makes `import openai` fail

    assert keys.check_key(MARKER) == "no_llm_extra"


@pytest.mark.parametrize(
    ("answer", "result", "saved"),
    [
        ("accepted", "saved", True),
        ("unreachable", "saved_unchecked", True),
        ("no_llm_extra", "saved_unchecked", True),
        ("refused", "refused", False),
    ],
)
def test_save_tests_first_and_saves_unless_the_key_was_refused(answer, result, saved):
    assert keys.save_key(f" {MARKER}\n", check=lambda key: answer) == result

    assert user_env_path().exists() is saved
    assert Config().api_key == (MARKER if saved else None)


def test_a_refused_key_leaves_the_earlier_saved_key_as_it_was():
    keys.save_key("good", check=lambda key: "accepted")

    assert keys.save_key("bad", check=lambda key: "refused") == "refused"

    assert Config().api_key == "good"


def test_an_unusable_key_is_refused_before_any_request():
    asked: list[str] = []

    with pytest.raises(ValueError, match="key"):
        keys.save_key("a b", check=lambda key: asked.append(key) or "accepted")

    assert asked == []
    assert not user_env_path().exists()


def test_remove_says_whether_there_was_a_key():
    assert keys.remove_key() is False
    keys.save_key("k", check=lambda key: "accepted")
    assert keys.remove_key() is True
    assert Config().api_key is None


def test_the_source_follows_the_order_a_run_uses(monkeypatch):
    assert keys.key_source() is None
    keys.save_key("k", check=lambda key: "accepted")
    assert keys.key_source() == "file"
    Path(".env").write_text("IDUN_API_KEY=d\n", encoding="utf-8")
    assert keys.key_source() == "dotenv"
    monkeypatch.setenv("QUESTXTRACT_NTNU_API_KEY", "old-name")
    assert keys.key_source() == "dotenv"  # IDUN_API_KEY in .env wins over the old name
    monkeypatch.setenv("IDUN_API_KEY", "e")
    assert keys.key_source() == "environment"


def test_the_saved_key_is_in_no_argument_the_app_builds_and_not_in_the_config(tmp_path):
    from questxtract.gui import run as gui_run

    keys.save_key(MARKER, check=lambda key: "accepted")
    source = tmp_path / "lecture.json"
    argv = gui_run.to_argv(source, "quiz", {"detector": "llm"}, tmp_path / "out.json")

    assert Config().api_key == MARKER  # a run does read it ...
    assert MARKER not in " ".join(argv) + repr(Config())  # ... and it goes nowhere else


@pytest.mark.parametrize(("raises", "expected"), [(None, "accepted"), ("auth", "refused")])
def test_openrouter_is_asked_for_the_key_not_the_model_list(monkeypatch, raises, expected):
    """#250, found live: OpenRouter lists its models to any key, so the check asks GET /key."""
    calls = _fake_openai(monkeypatch, raises)  # type: ignore[arg-type]

    assert keys.check_key(MARKER, "https://openrouter.ai/api/v1") == expected

    assert calls[0]["base_url"] == "https://openrouter.ai/api/v1"
    assert calls[1] == {"asked": "/key"}


def test_another_endpoint_is_asked_for_its_model_list(monkeypatch):
    calls = _fake_openai(monkeypatch)

    assert keys.check_key(MARKER, "http://localhost:8000/v1") == "accepted"

    assert calls[1:] == [{"asked": "models"}]


def test_each_endpoints_key_is_saved_under_its_own_name_and_the_others_kept(monkeypatch):
    """#250 AC8: OpenRouter's key is saved as OPENROUTER_API_KEY; the NTNU key stays."""
    tested: list[tuple[str, str | None]] = []
    monkeypatch.setattr(keys, "check_key", lambda key, url: tested.append((key, url)) or "accepted")
    openrouter = "https://openrouter.ai/api/v1"
    keys.save_key("ntnu-key")
    assert keys.save_key("sk-or", openrouter) == "saved"

    assert tested == [("ntnu-key", "https://llm.hpc.ntnu.no/v1"), ("sk-or", openrouter)]
    lines = user_env_path().read_text(encoding="utf-8").splitlines()
    assert lines == ['IDUN_API_KEY="ntnu-key"', 'OPENROUTER_API_KEY="sk-or"']
    assert (Config().api_key, Config(base_url="openrouter").api_key) == ("ntnu-key", "sk-or")
    assert keys.key_source(openrouter) == "file"

    assert keys.remove_key(openrouter) is True
    assert keys.key_source(openrouter) is None and keys.key_source() == "file"


@pytest.mark.parametrize(
    ("url", "name"),
    [
        ("https://llm.hpc.ntnu.no/v1", "NTNU"),
        ("https://api.openai.com/v1", "OpenAI"),
        ("https://openrouter.ai/api/v1", "OpenRouter"),
        ("http://localhost:11434/v1", "localhost:11434"),
        ("https://llm.example.org/v1", "llm.example.org"),
    ],
)
def test_the_window_names_the_endpoint(url, name):
    assert keys.endpoint_name(url) == name


def test_a_key_for_another_host_is_saved_as_the_generic_key():
    keys.save_key("vllm-key", "http://localhost:8000/v1", check=lambda key: "accepted")

    assert 'QUESTXTRACT_LLM_API_KEY="vllm-key"' in user_env_path().read_text(encoding="utf-8")
    assert Config(base_url="http://localhost:8000/v1").api_key == "vllm-key"


@pytest.mark.parametrize(
    "url",
    ["http://openrouter.ai/api/v1", "HTTP://llm.hpc.ntnu.no/v1", "http://api.openai.com/v1"],
)
def test_a_known_service_over_http_is_not_sent_the_key(monkeypatch, url):
    """Review of #252: the key would travel unencrypted, so nothing is asked or saved."""
    calls = _fake_openai(monkeypatch)

    assert keys.check_key(MARKER, url) == "insecure"
    assert keys.save_key(MARKER, url) == "insecure"

    assert calls == [] and not user_env_path().exists()


def test_a_local_server_over_http_is_fine(monkeypatch):
    _fake_openai(monkeypatch)
    assert keys.check_key(MARKER, "http://localhost:11434/v1") == "accepted"

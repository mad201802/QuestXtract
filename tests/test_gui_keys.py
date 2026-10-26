"""The NTNU key window's logic (issue #189), headless: test, save, remove, where it is read from."""

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
            self.models = types.SimpleNamespace(list=self._list)

        def _list(self) -> list:
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
        }
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

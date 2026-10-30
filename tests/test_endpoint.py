"""Tests for the shared endpoint (#53). Its request loop is tested through its two callers:
the judge in tests/test_llm.py and the vision call in tests/test_vision.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import openai
import pytest

from questxtract import Config, endpoint
from questxtract.config import NTNU_LLM_URL, OLLAMA_URL, OPENAI_URL, OPENROUTER_URL


def test_the_client_points_at_the_configured_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the client makes no request; the endpoint and timeout come from Config."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")  # the SDK's own fallback must not apply
    client = endpoint.client(Config(base_url="http://localhost:11434/v1", timeout_seconds=30))
    assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"
    assert client.timeout == 30
    assert client.api_key == endpoint.NO_KEY


def test_an_error_from_a_local_server_keeps_its_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review of #252: the placeholder key is not redacted, so "localhost" stays readable."""

    def down(**kwargs: Any) -> Any:
        def create(**request: Any) -> Any:
            raise openai.OpenAIError("Connection error: http://localhost:11434/v1 refused")

        return SimpleNamespace(
            api_key=kwargs["api_key"],
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )

    monkeypatch.setattr(openai, "OpenAI", down)
    with pytest.raises(ConnectionError) as error:
        _ask(Config(base_url="ollama"), vision=False)
    assert "http://localhost:11434/v1" in str(error.value)
    assert "[REDACTED]" not in str(error.value)


class _Recorder:
    """Stands in for ``openai.OpenAI``: records each client's endpoint and key, answers ``{}``."""

    def __init__(self) -> None:
        self.clients: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.clients.append(kwargs)
        message = SimpleNamespace(content="{}")

        def create(**request: Any) -> Any:
            self.requests.append(request)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

        return SimpleNamespace(
            api_key=kwargs["api_key"],
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorded = _Recorder()
    monkeypatch.setattr(openai, "OpenAI", recorded)
    return recorded


def _ask(config: Config, *, vision: bool) -> None:
    endpoint.ask(config, {"model": "m", "messages": []}, name="t", vision=vision)


def test_by_default_every_request_goes_to_ntnu_with_its_key(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#173 AC1: the judge and vision on NTNU's service, one key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("IDUN_API_KEY", "ntnu-key")

    _ask(Config(), vision=False)
    _ask(Config(from_scratch=True), vision=True)  # the same request, so past the cache

    assert [(c["base_url"], c["api_key"]) for c in recorder.clients] == 2 * [
        (NTNU_LLM_URL, "ntnu-key")
    ]


def test_openai_gets_the_openai_key_and_no_reasoning_setting_with_server(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#173 AC2: the fallback; the NTNU key never leaves for another endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("IDUN_API_KEY", "ntnu-key")
    config = Config(base_url=OPENAI_URL, reasoning_effort="")

    _ask(config, vision=False)
    _ask(config.replace(from_scratch=True), vision=True)

    assert [(c["base_url"], c["api_key"]) for c in recorder.clients] == 2 * [
        (OPENAI_URL, "sk-openai")
    ]
    assert all("reasoning_effort" not in r for r in recorder.requests)


def test_only_base_url_still_sends_vision_there_with_the_judges_key(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#162 AC: a config from before vision had its own endpoint keeps working."""
    monkeypatch.setenv("QUESTXTRACT_LLM_API_KEY", "sk-local")
    monkeypatch.setenv("IDUN_API_KEY", "ntnu-key")

    _ask(Config(base_url="http://localhost:11434/v1"), vision=True)

    (vision,) = recorder.clients
    assert (vision["base_url"], vision["api_key"]) == ("http://localhost:11434/v1", "sk-local")


def test_vision_base_url_takes_vision_elsewhere_with_that_endpoints_key(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("IDUN_API_KEY", "ntnu-key")
    config = Config(base_url="http://localhost:11434/v1", vision_base_url=NTNU_LLM_URL)

    _ask(config, vision=False)
    _ask(config, vision=True)

    judge, vision = recorder.clients
    assert (judge["base_url"], judge["api_key"]) == ("http://localhost:11434/v1", "local")
    assert (vision["base_url"], vision["api_key"]) == (NTNU_LLM_URL, "ntnu-key")


def _every_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in (
        ("IDUN_API_KEY", "ntnu-key"),
        ("QUESTXTRACT_NTNU_API_KEY", "old-ntnu-key"),
        ("OPENAI_API_KEY", "sk-openai"),
        ("OPENROUTER_API_KEY", "sk-or"),
    ):
        monkeypatch.setenv(name, value)


def _judge_and_vision(recorder: _Recorder, config: Config) -> list[tuple[str, str]]:
    config = config.replace(from_scratch=True)  # each request asks, past the cache
    _ask(config, vision=False)
    _ask(config, vision=True)
    return [(c["base_url"], c["api_key"]) for c in recorder.clients]


def test_openrouter_by_its_name_gets_its_own_key_for_the_judge_and_vision(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#250 AC1: OPENROUTER_API_KEY goes to OpenRouter, and no other service's key does."""
    _every_key(monkeypatch)
    assert _judge_and_vision(recorder, Config(base_url="openrouter")) == 2 * [
        (OPENROUTER_URL, "sk-or")
    ]


@pytest.mark.parametrize("url", ["ollama", "http://localhost:8080/v1", "http://127.0.0.1:1234/v1"])
def test_a_local_server_is_sent_no_other_services_key(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """#250 AC3: with every hosted key set, a local server needs none and gets none of them."""
    _every_key(monkeypatch)
    sent = _judge_and_vision(recorder, Config(base_url=url))
    assert [key for _, key in sent] == ["local", "local"]
    assert sent[0][0] == (OLLAMA_URL if url == "ollama" else url)


def test_another_host_gets_the_generic_key_and_runs_without_one(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#250 AC4: QUESTXTRACT_LLM_API_KEY for any host without a key of its own."""
    _every_key(monkeypatch)
    url = "https://llm.example.org/v1"
    assert _judge_and_vision(recorder, Config(base_url=url)) == 2 * [(url, "local")]

    monkeypatch.setenv("QUESTXTRACT_LLM_API_KEY", "generic")
    recorder.clients.clear()
    assert _judge_and_vision(recorder, Config(base_url=url)) == 2 * [(url, "generic")]


def test_the_vision_key_is_hidden_in_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDUN_API_KEY", "ntnu-secret")

    def refused(**kwargs: Any) -> Any:
        def create(**request: Any) -> Any:
            raise openai.OpenAIError(f"bad key {kwargs['api_key']}")

        return SimpleNamespace(
            api_key=kwargs["api_key"],
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )

    monkeypatch.setattr(openai, "OpenAI", refused)
    with pytest.raises(ConnectionError) as error:
        _ask(Config(), vision=True)
    assert "ntnu-secret" not in str(error.value) and "[REDACTED]" in str(error.value)
    assert NTNU_LLM_URL in str(error.value)

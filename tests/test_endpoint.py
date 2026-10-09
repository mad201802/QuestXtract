"""Tests for the shared endpoint (#53). Its request loop is tested through its two callers:
the judge in tests/test_llm.py and the vision call in tests/test_vision.py."""

from __future__ import annotations

from questxtract import Config, endpoint


def test_the_client_points_at_the_configured_endpoint():
    """Building the client makes no request; the endpoint and timeout come from Config."""
    client = endpoint.client(Config(base_url="http://localhost:11434/v1", timeout_seconds=30))
    assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"
    assert client.timeout == 30

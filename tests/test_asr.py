"""Tests for the model-independent ASR interface.

Deliberately never loads a real Whisper model: CI runs on three OSes with no GPU, so
every test here exercises the *interface* (registry, resolution, wiring into the
`transcribe` stage) using a fake backend. `FasterWhisperBackend` itself is only checked
for correct *registration*, never actually invoked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from questxtract import asr
from questxtract.asr import AsrBackend, FasterWhisperBackend, register_backend, resolve_backend
from questxtract.config import Config
from questxtract.models import Context, Transcript
from questxtract.transcribe import transcribe


class _FakeBackend:
    """A minimal `AsrBackend` for tests: no model, no network, no GPU."""

    def transcribe(self, source: Path, config: Config) -> list[dict[str, Any]]:
        del source, config  # unused - the fake ignores both, which the interface permits
        return [{"text": "What is a stable sort?", "start": 0.0, "end": 2.0}]


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """Snapshot/restore the backend registry so tests can't leak into each other."""
    before = dict(asr._REGISTRY)
    yield
    asr._REGISTRY.clear()
    asr._REGISTRY.update(before)


# --- the interface and registry -----------------------------------------------


def test_fake_backend_satisfies_the_protocol_at_runtime() -> None:
    assert isinstance(_FakeBackend(), AsrBackend)


def test_register_and_resolve_backend_round_trips() -> None:
    register_backend("fake")(_FakeBackend)
    backend = resolve_backend("fake")
    assert isinstance(backend, _FakeBackend)


def test_resolve_unknown_backend_raises_with_available_list() -> None:
    register_backend("fake")(_FakeBackend)
    with pytest.raises(ValueError, match="unknown ASR backend"):
        resolve_backend("does-not-exist")


def test_faster_whisper_backend_is_registered_by_default() -> None:
    """Acceptance criterion 4: the default engine is swappable, not hardcoded elsewhere."""
    backend = resolve_backend("faster_whisper")
    assert isinstance(backend, FasterWhisperBackend)


# --- wiring into the transcribe stage -------------------------------------------


def test_transcribe_stage_uses_the_configured_backend(tmp_path: Path) -> None:
    """Acceptance criteria 1-3: called through the interface, timestamps present,
    backend selection comes from config, not a hardcoded import."""
    register_backend("fake")(_FakeBackend)
    source = tmp_path / "lecture.mp4"
    source.touch()
    config = Config().replace(asr_backend="fake")
    ctx = Context(source=source, config=config)

    result = transcribe(ctx)

    assert isinstance(result.transcript, Transcript)
    assert len(result.transcript.words) > 0
    assert result.transcript.words[0].text == "What"
    assert result.transcript.asr_segments[0].start == 0.0
    assert result.transcript.asr_segments[0].end == 2.0


def test_transcribe_stage_rejects_an_unconfigured_backend(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.touch()
    config = Config().replace(asr_backend="does-not-exist")
    ctx = Context(source=source, config=config)
    with pytest.raises(ValueError, match="unknown ASR backend"):
        transcribe(ctx)


def test_config_asr_backend_defaults_to_faster_whisper() -> None:
    """Acceptance criterion 3: configuration, not a hardcoded engine name in code."""
    assert Config().asr_backend == "faster_whisper"


def test_config_replace_can_swap_the_backend() -> None:
    config = Config().replace(asr_backend="fake")
    assert config.asr_backend == "fake"
    assert Config().asr_backend == "faster_whisper"  # original untouched (frozen)

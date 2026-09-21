"""Tests for the model-independent ASR interface.

Deliberately never loads a real Whisper model: CI runs on three OSes with no GPU, so
every test here exercises the *interface* (registry, resolution, wiring into the
`transcribe` stage) using a fake backend. `FasterWhisperBackend` itself is checked for
correct *registration*, plus its own progress-fraction math against a stubbed model and
stubbed audio extraction - never against a real Whisper model or real audio.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
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

    def transcribe(
        self,
        source: Path,
        config: Config,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[dict[str, Any]]:
        del source, config  # unused - the fake ignores both, which the interface permits
        if on_progress:
            on_progress(0.5)
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


class _FakeWord:
    def __init__(self, word: str, start: float, end: float) -> None:
        self.word, self.start, self.end, self.probability = word, start, end, 1.0


class _FakeSegment:
    def __init__(self, text: str, start: float, end: float, words: list[_FakeWord] | None) -> None:
        self.text, self.start, self.end, self.words = text, start, end, words
        self.avg_logprob = self.no_speech_prob = self.compression_ratio = 0.0


def test_faster_whisper_backend_reports_word_end_over_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #14: the real progress math on the real backend, not the fake - word.end /
    info.duration, capped at 0.99 (the caller decides when a stage is truly finished), and
    falling back to the segment's own end when faster-whisper gave it no words."""
    source = tmp_path / "lecture.mp4"
    source.touch()
    hi, there = _FakeWord("Hi", 0.0, 1.0), _FakeWord("there", 1.0, 2.0)
    segments = [
        _FakeSegment("Hi there", 0.0, 2.0, [hi, there]),
        _FakeSegment("(silence)", 2.0, 10.0, words=None),  # no words -> falls back to segment.end
    ]
    info = type("Info", (), {"duration": 10.0})()

    class _FakeModel:
        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            return segments, info

    monkeypatch.setattr(asr, "_load_model", lambda *a, **k: _FakeModel())

    @contextmanager
    def _fake_extract_audio(path: Path) -> Any:
        yield path

    monkeypatch.setattr(asr, "extract_audio", _fake_extract_audio)

    seen: list[float] = []
    result = FasterWhisperBackend().transcribe(source, Config(), on_progress=seen.append)

    assert seen == [0.1, 0.2, 0.99]
    assert len(result) == 2


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


def test_transcribe_stage_forwards_backend_progress_to_ctx_progress(tmp_path: Path) -> None:
    """Issue #14 AC1/AC4: a slow backend's intra-stage progress reaches ctx.progress,
    tagged with this stage's name, as real fractions arrive - not just start/done."""
    register_backend("fake")(_FakeBackend)
    source = tmp_path / "lecture.mp4"
    source.touch()
    config = Config().replace(asr_backend="fake")
    seen: list[tuple[str, float]] = []
    ctx = Context(
        source=source, config=config, progress=lambda name, frac: seen.append((name, frac))
    )

    transcribe(ctx)

    assert seen == [("transcribe", 0.5)]


def test_transcribe_stage_works_without_a_progress_callback(tmp_path: Path) -> None:
    """Issue #14 AC2: progress reporting is optional and must not break plain calls."""
    register_backend("fake")(_FakeBackend)
    source = tmp_path / "lecture.mp4"
    source.touch()
    config = Config().replace(asr_backend="fake")
    ctx = Context(source=source, config=config)  # no progress callback

    result = transcribe(ctx)

    assert isinstance(result.transcript, Transcript)


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

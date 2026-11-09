from __future__ import annotations

import os
import sys
import types
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
    def transcribe(
        self,
        source: Path,
        config: Config,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[dict[str, Any]]:
        del source, config
        if on_progress:
            on_progress(0.5)
        return [{"text": "What is a stable sort?", "start": 0.0, "end": 2.0}]


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    before = dict(asr._REGISTRY)
    yield
    asr._REGISTRY.clear()
    asr._REGISTRY.update(before)


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
    source = tmp_path / "lecture.mp4"
    source.touch()
    hi, there = _FakeWord("Hi", 0.0, 1.0), _FakeWord("there", 1.0, 2.0)
    segments = [
        _FakeSegment("Hi there", 0.0, 2.0, [hi, there]),
        _FakeSegment("(silence)", 2.0, 10.0, words=None),
    ]
    info = type("Info", (), {"duration": 10.0})()

    class _FakeModel:
        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            return segments, info

    monkeypatch.setattr(asr, "_load_model", lambda *a, **k: _FakeModel())

    @contextmanager
    def _fake_extract_audio(path: Path, progress: Any = None) -> Any:
        progress(0.5)
        yield path

    monkeypatch.setattr(asr, "extract_audio", _fake_extract_audio)

    seen: list[float] = []
    result = FasterWhisperBackend().transcribe(source, Config(), on_progress=seen.append)

    assert seen == pytest.approx([0.05, 0.19, 0.28, 0.99])
    assert len(result) == 2


def test_transcribe_stage_uses_the_configured_backend(tmp_path: Path) -> None:
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
    register_backend("fake")(_FakeBackend)
    source = tmp_path / "lecture.mp4"
    source.touch()
    config = Config().replace(asr_backend="fake")
    ctx = Context(source=source, config=config)

    result = transcribe(ctx)

    assert isinstance(result.transcript, Transcript)


def test_transcribe_stage_rejects_an_unconfigured_backend(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.touch()
    config = Config().replace(asr_backend="does-not-exist")
    ctx = Context(source=source, config=config)
    with pytest.raises(ValueError, match="unknown ASR backend"):
        transcribe(ctx)


def test_load_model_allows_a_second_openmp_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK")
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = lambda *a, **k: object()  # type: ignore[attr-defined]
    fake.BatchedInferencePipeline = lambda model: model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(asr, "_MODEL_CACHE", {})

    asr._load_model("tiny", "int8")

    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"


def test_config_asr_backend_defaults_to_faster_whisper() -> None:
    assert Config().asr_backend == "faster_whisper"


def test_config_replace_can_swap_the_backend() -> None:
    config = Config().replace(asr_backend="fake")
    assert config.asr_backend == "fake"
    assert Config().asr_backend == "faster_whisper"

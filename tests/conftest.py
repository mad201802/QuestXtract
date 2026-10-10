import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import pytest

from questxtract import Config, Context, Sentence, Transcript, Word, endpoint


@pytest.fixture(autouse=True)
def _isolated_from_the_developer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the developer's OPENAI_API_KEY, .env and questxtract.toml out of every test.

    Each test runs in its own empty ``tmp_path`` with no key in the environment, so the
    cwd-relative reads (``.env``, ``questxtract.toml``, the ``.questxtract`` cache) see
    nothing of the checkout. Build paths to test files from ``__file__``, never the cwd.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def _make_mp4(path: Path, *, audio: bool = True) -> Path:
    """Write a tiny real MP4: a tenth of a second of silence, or a single video frame."""
    with av.open(str(path), mode="w", format="mp4") as container:
        if audio:
            stream = container.add_stream("aac", rate=8000)
            stream.layout = "mono"
            frame = av.AudioFrame(format="s16", layout="mono", samples=800)
            frame.sample_rate = 8000
            frame.planes[0].update(b"\0" * frame.planes[0].buffer_size)
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        else:
            stream = container.add_stream("mpeg4", rate=1)
            stream.width = 16
            stream.height = 16
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame(16, 16, "yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    return path


@pytest.fixture
def make_mp4() -> Callable[..., Path]:
    return _make_mp4


def _make_mp3(path: Path) -> Path:
    """Write a tiny real MP3: a tenth of a second of silence."""
    with av.open(str(path), mode="w", format="mp3") as container:
        stream = container.add_stream("libmp3lame", rate=16000)
        stream.layout = "mono"
        frame = av.AudioFrame(format="s16", layout="mono", samples=1600)
        frame.sample_rate = 16000
        frame.planes[0].update(b"\0" * frame.planes[0].buffer_size)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture
def make_mp3() -> Callable[..., Path]:
    return _make_mp3


def _words(spec: list[tuple[str, float, float]]) -> tuple[Word, ...]:
    return tuple(Word(text=t, start=s, end=e) for t, s, e in spec)


@pytest.fixture
def transcript() -> Transcript:
    """Three sentences over two ASR chunks, with real word timings.

    The silence between "sort?" (ends 4.0) and "It" (starts 6.0) is exactly
    ``Config.segment_max_gap_seconds``, and the gap rule is strict, so this fixture is one
    segment by a hair. Every test that leans on that - here and in test_export, test_detect,
    test_vision and test_pipeline - would flip if the timings moved. Leave them alone.
    """
    return Transcript(
        words=_words(
            [
                ("Today", 0.0, 0.5),
                ("we", 0.5, 0.8),
                ("cover", 0.8, 1.4),
                ("sorting.", 1.4, 2.0),
                ("What", 2.2, 2.6),
                ("is", 2.6, 2.8),
                ("a", 2.8, 2.9),
                ("stable", 2.9, 3.5),
                ("sort?", 3.5, 4.0),
                ("It", 6.0, 6.3),
                ("keeps", 6.3, 6.8),
                ("equal", 6.8, 7.3),
                ("elements", 7.3, 8.0),
                ("in", 8.0, 8.2),
                ("order.", 8.2, 9.0),
            ]
        ),
        language="en",
        duration=9.0,
    )


@pytest.fixture
def sentences(transcript: Transcript) -> tuple[Sentence, ...]:
    from questxtract.sentences import split_sentences

    return split_sentences(transcript)


@pytest.fixture
def context(transcript: Transcript) -> Context:
    return Context(source=Path("lecture.mp4"), config=Config(), transcript=transcript)


class FakeLLM:
    """Stands in for the OpenAI client: replays canned replies in order, repeating the last
    one, and records every request. A reply is a JSON-able object, a raw string, or an
    exception to raise."""

    def __init__(self, replies: tuple[Any, ...]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.chat = self.completions = self

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeLLM]:
    """Install a :class:`FakeLLM` as the endpoint's client: the judge and vision, no network."""

    def install(*replies: Any) -> FakeLLM:
        client = FakeLLM(replies)
        monkeypatch.setattr(endpoint, "client", lambda config: client)
        return client

    return install

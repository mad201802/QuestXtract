import io
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from questxtract import Config, Context, Sentence, Transcript, Word, llm


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


# --- synthetic slide lecture and a stub VLM, for the vision stage (#55) ---------------------


@dataclass(frozen=True)
class Slide:
    """One synthetic slide: what it looks like, and what the stub VLM "reads" off it."""

    background: tuple[int, int, int]
    ink: tuple[int, int, int]
    digit: int
    caption: dict[str, Any]


SLIDES = (
    Slide(
        background=(245, 245, 240),
        ink=(20, 20, 20),
        digit=1,
        caption={
            "description": "Title slide: Sorting",
            "text": "Sorting",
            "latex": None,
            "confidence": 0.9,
        },
    ),
    Slide(
        background=(40, 70, 160),
        ink=(250, 250, 250),
        digit=2,
        caption={
            "description": "Definition of a stable sort",
            "text": "Stable: equal keys keep their order",
            "latex": None,
            "confidence": 0.8,
        },
    ),
    Slide(
        background=(20, 40, 28),
        ink=(235, 235, 200),
        digit=3,
        caption={
            "description": "Chalkboard with the merge sort running time",
            "text": "T(n) = 2T(n/2) + n",
            "latex": r"T(n) = 2T(\tfrac{n}{2}) + n = O(n \log n)",
            "confidence": 0.7,
        },
    ),
)

# 3x5 bitmap digits, drawn as the "text" on each slide.
_DIGITS = {
    1: ("010", "110", "010", "010", "111"),
    2: ("111", "001", "111", "100", "111"),
    3: ("111", "001", "111", "001", "111"),
}


@dataclass(frozen=True)
class SyntheticLecture:
    """A generated MP4 whose slide changes are known exactly.

    Slides change on whole seconds, so the default 1 fps sampling lands on every change and
    the right board states are exactly ``states``.
    """

    path: Path
    slides: tuple[Slide, ...]
    slide_seconds: float

    @property
    def duration(self) -> float:
        return len(self.slides) * self.slide_seconds

    @property
    def changes(self) -> tuple[float, ...]:
        """Timestamps where one slide gives way to the next."""
        return tuple(i * self.slide_seconds for i in range(1, len(self.slides)))

    @property
    def states(self) -> tuple[tuple[float, float], ...]:
        """The ``(start, end)`` lifetime of every slide."""
        bounds = (0.0, *self.changes, self.duration)
        return tuple(zip(bounds, bounds[1:], strict=False))


def _draw_slide(slide: Slide, width: int, height: int) -> np.ndarray:
    """Paint a slide: background, a title bar, a box, and its number as a bitmap digit."""
    image = np.empty((height, width, 3), np.uint8)
    image[:] = slide.background
    image[6:14, 8 : width - 8] = slide.ink  # title bar
    image[30:70, 70 + 10 * slide.digit : 110 + 10 * slide.digit] = slide.ink  # a shape
    for row, bits in enumerate(_DIGITS[slide.digit]):
        for col, bit in enumerate(bits):
            if bit == "1":
                y, x = 30 + 8 * row, 16 + 8 * col
                image[y : y + 8, x : x + 8] = slide.ink
    return image


def make_slides_mp4(
    path: Path,
    slides: tuple[Slide, ...] = SLIDES,
    *,
    slide_seconds: float = 2.0,
    fps: int = 5,
    size: tuple[int, int] = (160, 96),
) -> SyntheticLecture:
    """Write a silent slide-show MP4: ``slide_seconds`` per slide, a video and an audio track.

    The audio is there because ``ingest`` refuses a recording without it; it is silence, and
    the tests swap ASR out anyway.
    """
    width, height = size
    frames_per_slide = round(slide_seconds * fps)
    with av.open(str(path), mode="w", format="mp4") as container:
        video = container.add_stream("mpeg4", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        audio = container.add_stream("aac", rate=8000)  # both streams before the first mux
        audio.layout = "mono"
        index = 0
        for slide in slides:
            frame = av.VideoFrame.from_ndarray(_draw_slide(slide, width, height), format="rgb24")
            for _ in range(frames_per_slide):
                frame.pts = index
                index += 1
                for packet in video.encode(frame):
                    container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

        samples = 1024
        for i in range(int(len(slides) * slide_seconds * 8000) // samples):
            chunk = av.AudioFrame(format="s16", layout="mono", samples=samples)
            chunk.sample_rate, chunk.pts = 8000, i * samples
            chunk.planes[0].update(b"\0" * chunk.planes[0].buffer_size)
            for packet in audio.encode(chunk):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    return SyntheticLecture(path=path, slides=slides, slide_seconds=slide_seconds)


@pytest.fixture
def make_slides() -> Callable[..., SyntheticLecture]:
    return make_slides_mp4


@pytest.fixture(scope="session")
def slides_lecture(tmp_path_factory: pytest.TempPathFactory) -> SyntheticLecture:
    """Three slides of two seconds each, changing at 2.0 s and 4.0 s. Built once per run."""
    return make_slides_mp4(tmp_path_factory.mktemp("synthetic") / "slides.mp4")


@dataclass
class StubVlm:
    """Stands in for the vision model: deterministic JSON per frame, no network, no key.

    It "reads" a keyframe by its background colour and returns that slide's canned caption,
    so a caption also proves the keyframe came from the right slide. Every call is recorded.
    """

    slides: tuple[Slide, ...] = SLIDES
    calls: list[tuple[bytes, str]] = field(default_factory=list)

    def slide_of(self, image: bytes) -> Slide:
        """Return the slide whose background is closest to the image's bottom-right corner."""
        with av.open(io.BytesIO(image)) as container:
            pixels = next(container.decode(video=0)).to_ndarray(format="rgb24")
        corner = pixels[-4, -4].astype(int)
        return min(self.slides, key=lambda s: int(np.abs(corner - s.background).sum()))

    def __call__(self, config: Config, image: bytes, transcript: str = "") -> dict[str, Any]:
        del config  # unused: the stub answers the same whatever model is configured
        self.calls.append((image, transcript))
        return dict(self.slide_of(image).caption)


@pytest.fixture
def stub_vlm(monkeypatch: pytest.MonkeyPatch) -> StubVlm:
    """Swap ``llm.describe_board`` for a :class:`StubVlm`, the one seam vision calls out on."""
    stub = StubVlm()
    monkeypatch.setattr(llm, "describe_board", stub)
    return stub


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No API key and no network: any attempt to open a connection fails the test."""

    def _refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"test tried to reach the network: {args}")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


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

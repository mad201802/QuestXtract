"""Stage 5: optional enrichment of the audio with what is on the board."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import av
import numpy as np

from . import endpoint
from .config import Config
from .models import (
    Context,
    Layout,
    Progress,
    Segment,
    Sentence,
    VisualContext,
    overlapping,
    stage,
)

_COMPARE_HEIGHT = 480
_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # ITU-R BT.601 luma weights
_NOISE = 32

_REUSE_KINDS = ("slides", "handwriting", "screencast")
# measured on the codec noise of TTK4225 2026-09-14 (architecture.md 6.5)
_REUSE_CANDIDATE = 0.01
_DETAIL_SQUARES = ((4, 2, 20.0), (8, 4, 11.0))
_DETAIL_COLOUR = 12.0
_YCBCR = np.array(  # ITU-R BT.601 RGB to Y, Cb and Cr, without offsets: only differences matter
    [[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]], dtype=np.float32
)

# ink rule measured on #133, the 0.7 limit on the #58 frames (D27)
_INK_WINDOW = 15
_INK_CONTRAST = 20
_INK_KEPT = 0.7
_NEW_INK_CELL = 8
_NEW_INK_SHARE = 0.2

VISION_PROMPT_VERSION = "v2"
_SPEECH_CHARS = 1500
_GIVE_UP_AFTER = 3

_FIELDS = ("description", "text", "latex", "confidence")

# GLM-OCR's task prompt, measured in #52 and #143
_OCR_PROMPT = "Text Recognition:"
_OCR_UNMEASURED = 0.5
_OCR_FORMULA = re.compile(r"\$\$(.+?)\$\$|\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)", re.DOTALL)
_OCR_CIRCLED = re.compile(r"\\textcircled\{([^}]*)\}")
_OCR_COMMAND = re.compile(r"\\(?:mathrm|text|mathbf|mathit)\{([^}]*)\}|\\([A-Za-z]+)")
_NO_LOGPROBS: set[str | None] = set()
_OCR_LABEL = re.compile(r"\\textcircled\{[^}]*\}|[\d.,\s]*\\mathrm\{[^}]*\}|\\[A-Za-z]+|\S")

_PROMPT = """\
You read one frame from a lecture recording: a slide, a blackboard, a whiteboard or tablet notes.
Describe what is visible on the board or slide, for someone who cannot see it.

Frame kind: {kind}

What the lecturer said while this frame was on screen (context only):
{speech}

Rules:
- Report only what is visible in the image. The speech helps you read unclear writing, but never
  copy words, formulas or claims from it that are not on the board.
- "text": every piece of text on the board or slide, handwritten or printed, in reading order,
  one line per line on the board. Keep the wording and spelling you see. Mark a word you cannot
  read as [unreadable]. Empty string when there is no text.
- "latex": the formulas on the board as LaTeX, one formula per line. null when there is no
  formula. Do not write LaTeX for plain words.
- "description": one or two sentences on what the frame shows: the slide or board topic, and any
  diagram, graph or figure (its parts and how they connect). No text transcription here.
- "confidence": a number from 0 to 1 for how sure you are of "text" and "latex" together. Use a
  low value for blur, glare, a lecturer blocking the board, or handwriting you had to guess.
- Ignore the lecturer, the room, the camera and any on-screen interface.
- An empty or blank board is a valid answer: empty "text", null "latex", a short "description"
  saying so, confidence 1 if you are sure it is blank.

Answer with one JSON object and nothing else: \
{"description": ..., "text": ..., "latex": ..., "confidence": ...}\
"""

_PREVIOUS = """\
The board state before this one, as read from its own frame (context only):
Description: {previous_description}
Text:
{previous_text}
LaTeX:
{previous_latex}
Where new writing appeared since then: {new_ink}

"""
_PREVIOUS_RULE = """\
- The previous board state helps you keep names and notation the same and see what is new. It is
  not this frame: report everything visible in this frame, also what was there before, and never
  copy anything from the previous state that is no longer visible.
"""
_V0_FIRST_RULE = "  copy words, formulas or claims from it that are not on the board.\n"
_PROMPT_V2 = _PROMPT.replace("Rules:\n", _PREVIOUS + "Rules:\n").replace(
    _V0_FIRST_RULE, _V0_FIRST_RULE + _PREVIOUS_RULE
)
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "board",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "text": {"type": "string"},
                "latex": {"type": ["string", "null"]},
                "confidence": {"type": "number"},
            },
            "required": list(_FIELDS),
            "additionalProperties": False,
        },
    },
}

Frame = np.ndarray[Any, np.dtype[np.uint8]]
Ink = np.ndarray[Any, np.dtype[np.bool_]]
Crop = tuple[float, float, float, float]

_FULL_FRAME: Crop = (0.0, 0.0, 1.0, 1.0)
_KINDS = ("slides", "blackboard", "whiteboard", "handwriting", "screencast", "unknown")
_PROBE_AT = (0.2, 0.5, 0.8)
_PROBE_WIDTH = 640
_REGION_PAD = 0.02
_REGION_MIN_AREA = 0.1

_LAYOUT_PROMPT = """\
You see three frames taken from across one lecture recording. Find where the lecture content is:
the slides, the board or the written notes.

Rules:
- "content_region": the box around the content as [x0, y0, x1, y1], fractions of the frame's
  width and height measured from its top-left corner, each from 0 to 1. Leave out the lecturer's
  camera, the room around the board, and on-screen interface such as toolbars, a video-call
  window or a taskbar. Give one box that holds the content in all three frames. If the content
  fills the frame, answer [0, 0, 1, 1].
- "kind": what the content is.
  "slides": prepared slides with printed text and figures.
  "blackboard": chalk on a blackboard.
  "whiteboard": marker on a whiteboard.
  "handwriting": writing on a tablet or on paper under a camera, also over slides.
  "screencast": a computer screen with code, a terminal or an application.
  "unknown": none of these, or you cannot tell.

Answer with one JSON object and nothing else: {"content_region": [x0, y0, x1, y1], "kind": ...}\
"""

_LAYOUT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "layout",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "content_region": {"type": "array", "items": {"type": "number"}},
                "kind": {"type": "string", "enum": list(_KINDS)},
            },
            "required": ["content_region", "kind"],
            "additionalProperties": False,
        },
    },
}


@stage("layout")
def layout(ctx: Context) -> Context:
    """Find where the content is and what kind it is, once per recording (architecture.md 6.2)."""
    if not ctx.config.vision_enabled or not has_video(ctx.source):
        return ctx
    try:
        frames = _probe_frames(ctx.source)
    except (av.error.FFmpegError, OSError) as exc:
        return _fallback(ctx, f"the recording could not be read: {exc}")
    if not frames:
        return _fallback(ctx, "the recording has no frame to show")
    images = [{"type": "image_url", "image_url": {"url": _data_url(png)}} for png in frames]
    request: dict[str, Any] = {
        "model": ctx.config.vision_model,
        "messages": [
            {"role": "system", "content": _LAYOUT_PROMPT},
            {"role": "user", "content": images},
        ],
        "response_format": _LAYOUT_SCHEMA,
    }
    if ctx.config.vision_effort:
        request["reasoning_effort"] = ctx.config.vision_effort
    usage: Counter[str] = Counter()
    ctx.stats.update(dict.fromkeys(_cost_keys("layout"), 0))
    try:
        answer = endpoint.ask(ctx.config, request, name="layout", usage=usage, vision=True)
    except (ConnectionError, endpoint.InvalidAnswer) as exc:
        return _fallback(ctx, str(exc))
    finally:
        ctx.stats.update(usage)
    found = _layout_from(answer)
    if found is None:
        return _fallback(ctx, f"{ctx.config.vision_model} answered with a malformed layout")
    ctx.layout = found
    return ctx


def _fallback(ctx: Context, why: str) -> Context:
    warnings.warn(
        f"vision: layout probe failed, using the full frame: {why}", UserWarning, stacklevel=3
    )
    ctx.layout = Layout()
    return ctx


def _probe_frames(path: Path) -> list[bytes]:
    pngs: list[bytes] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = container.duration / av.time_base if container.duration else 0.0
        for fraction in _PROBE_AT if duration else (0.0,):
            target = duration * fraction
            container.seek(int(target * av.time_base))
            for frame in container.decode(stream):
                if frame.time is not None and frame.time + 1e-6 < target:
                    continue
                width = min(frame.width, _PROBE_WIDTH)
                height = round(frame.height * width / frame.width / 2) * 2 or 2
                # av < 18 (on Python 3.10) types to_ndarray as a union of dtypes; rgb24 is uint8.
                picture: Frame = frame.reformat(width=width, height=height).to_ndarray(  # type: ignore[assignment, unused-ignore]
                    format="rgb24"
                )
                pngs.append(_png(picture))
                break
    return pngs


def _layout_from(answer: dict[str, Any]) -> Layout | None:
    region, kind = answer.get("content_region"), answer.get("kind")
    if kind not in _KINDS or not isinstance(region, list) or len(region) != 4:
        return None
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in region
    ):
        return None
    x0, y0, x1, y1 = (float(v) for v in region)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        return None
    if (x1 - x0) * (y1 - y0) < _REGION_MIN_AREA:
        return Layout(kind=str(kind))
    padded = (
        max(0.0, x0 - _REGION_PAD),
        max(0.0, y0 - _REGION_PAD),
        min(1.0, x1 + _REGION_PAD),
        min(1.0, y1 + _REGION_PAD),
    )
    return Layout(content_region=padded, kind=str(kind))


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


@stage("vision")
def vision(ctx: Context) -> Context:
    """Capture board states from the recording into ``ctx.visual`` (architecture.md 6.3 to 6.5)."""
    if not ctx.config.vision_enabled or not has_video(ctx.source):
        return ctx
    config, report = ctx.config, ctx.progress
    views = _cut(ctx)
    reuse = ctx.layout.kind in _REUSE_KINDS
    carry = config.vision_reader != "ocr"
    readings: list[_Reading] = []
    visual: list[VisualContext] = []
    before: tuple[Ink, dict[str, Any]] | None = None
    failed = reused = 0
    usage: Counter[str] = Counter()
    for index, view in enumerate(views):
        if report:
            report("vision", 0.5 + 0.5 * index / len(views))
        picture = _decoded(view.keyframe) if reuse or carry else None
        small = None if picture is None else _for_comparison(picture)
        ink = _ink(small) if carry and small is not None else None
        fields = _earlier(picture, small, readings, views) if reuse else None
        if fields is not None:
            reused += 1
        else:
            speech = _speech(ctx.sentences, view.start, view.end)
            fields = read_keyframe(
                config,
                view.keyframe,
                speech,
                kind=ctx.layout.kind,
                usage=usage,
                previous=_previous(before, ink),
            )
            if fields is None:
                failed += 1
                before = None
                if not visual and failed >= _GIVE_UP_AFTER:
                    warnings.warn(
                        f"vision: the first {failed} keyframes could not be read, so the rest "
                        "are not sent; check vision_model or vision_ocr_model, the API key and "
                        "the endpoint",
                        UserWarning,
                        stacklevel=2,
                    )
                    break
                continue
            if reuse and small is not None:
                readings.append(_Reading(small=small, view=index, fields=fields))
        before = None if ink is None else (ink, fields)
        path = _store(view.keyframe, config, ctx.source_hash)
        visual.append(VisualContext(start=view.start, end=view.end, keyframe_path=path, **fields))
    ctx.visual = tuple(visual)
    ctx.stats["vision_failed"] = len(views) - len(visual)
    ctx.stats["vision_reused"] = reused
    ctx.stats.update({**dict.fromkeys(_cost_keys("vision"), 0), **usage})
    return ctx


def _cost_keys(name: str) -> tuple[str, ...]:
    return tuple(f"{name}_{key}" for key in ("requests", "input_tokens", "output_tokens", "ms"))


@dataclass(frozen=True, kw_only=True)
class _Reading:
    small: Frame
    view: int
    fields: dict[str, Any]


def _earlier(
    picture: Frame | None,
    small: Frame | None,
    readings: Sequence[_Reading],
    views: Sequence[_View],
) -> dict[str, Any] | None:
    if picture is None or small is None:
        return None
    candidates = [r for r in readings if r.small.shape == small.shape]
    if not candidates:
        return None
    closest = min(candidates, key=lambda r: _changed(r.small, small))
    if _changed(closest.small, small) > _REUSE_CANDIDATE:
        return None
    before = _decoded(views[closest.view].keyframe)
    if before is None or before.shape != picture.shape or not _same_page(before, picture):
        return None
    return closest.fields


def _same_page(before: Frame, after: Frame) -> bool:
    change = (before.astype(np.float32) - after.astype(np.float32)) @ _YCBCR.T
    largest = max(side for side, _, _ in _DETAIL_SQUARES)
    border = ((0, -change.shape[0] % largest), (0, -change.shape[1] % largest), (0, 0))
    change = np.pad(change, border, mode="edge")
    for side, step, limit in _DETAIL_SQUARES:
        means = _square_means(change, side, step)
        if means[..., 0].max() > limit:
            return False
    return bool(_square_means(change[..., 1:], largest, largest // 2).max() <= _DETAIL_COLOUR)


def _square_means(change: np.ndarray[Any, Any], side: int, step: int) -> np.ndarray[Any, Any]:
    height, width, channels = change.shape[0] // step, change.shape[1] // step, change.shape[2]
    cells = change.reshape(height, step, width, step, channels).sum(axis=(1, 3))
    span = side // step
    rows, columns = height - span + 1, width - span + 1
    squares = sum(cells[i : i + rows, j : j + columns] for i in range(span) for j in range(span))
    means: np.ndarray[Any, Any] = np.abs(np.asarray(squares)) / side**2
    return means


def _decoded(png: bytes) -> Frame | None:
    try:
        codec = av.CodecContext.create("png", "r")
        frames = [frame for packet in codec.parse(png) for frame in codec.decode(packet)]
    except (av.error.FFmpegError, ValueError):
        return None
    if not frames or not isinstance(frames[0], av.VideoFrame):
        return None
    return np.asarray(frames[0].to_ndarray(format="rgb24"), dtype=np.uint8)  # rgb24 is uint8


@dataclass(frozen=True, kw_only=True)
class PreviousState:
    """The board state before the one being read, as the captioning prompt gets it (6.5)."""

    description: str
    text: str
    latex: str | None
    new_ink: Crop | None = None


def _previous(before: tuple[Ink, dict[str, Any]] | None, now: Ink | None) -> PreviousState | None:
    if before is None or now is None:
        return None
    was, fields = before
    if _ink_kept(was, now) < _INK_KEPT:
        return None
    return PreviousState(
        description=fields["description"],
        text=fields["text"],
        latex=fields["latex"],
        new_ink=_new_ink(now & ~was),
    )


def _ink_kept(was: Ink, now: Ink) -> float:
    if was.shape != now.shape or not was.any():
        return 0.0
    return float((was & now).sum() / was.sum())


def _ink(small: Frame) -> Ink:
    side, height, width = _INK_WINDOW, small.shape[0], small.shape[1]
    gray = small.astype(np.float64)
    sums = np.pad(np.pad(gray, side // 2, mode="edge").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    window = (
        sums[side : side + height, side : side + width]
        - sums[:height, side : side + width]
        - sums[side : side + height, :width]
        + sums[:height, :width]
    )
    ink: Ink = np.abs(gray - window / side**2) > _INK_CONTRAST
    return ink


def _new_ink(new: Ink) -> Crop | None:
    cell = _NEW_INK_CELL
    height, width = new.shape[0] // cell * cell, new.shape[1] // cell * cell
    share = new[:height, :width].reshape(height // cell, cell, width // cell, cell).mean((1, 3))
    rows, columns = np.nonzero(share >= _NEW_INK_SHARE)
    if not len(rows):
        return None
    return (
        round(int(columns.min()) * cell / new.shape[1], 2),
        round(int(rows.min()) * cell / new.shape[0], 2),
        round((int(columns.max()) + 1) * cell / new.shape[1], 2),
        round((int(rows.max()) + 1) * cell / new.shape[0], 2),
    )


def _cut(ctx: Context) -> list[_View]:
    config, report = ctx.config, ctx.progress
    region = ctx.layout.content_region
    crop = None if region == _FULL_FRAME else region
    limit = state_limit(config, ctx.duration)
    header = {
        "content_region": list(region),
        "vision_fps": config.vision_fps,
        "vision_change_fraction": config.vision_change_fraction,
        "vision_change_hold": config.vision_change_hold,
        "max_views": limit,
        "duration": ctx.duration,
    }
    path = Path(config.cache_dir) / ctx.source_hash / "views.json"
    if ctx.source_hash and not config.from_scratch:
        cached = _load_views(path, header, Path(config.cache_dir))
        if cached is not None:
            return cached

    def sampling(_: str, fraction: float) -> None:
        if report:
            report("vision", fraction / 2)

    samples = _sample_frames(ctx.source, config.vision_fps, crop=crop, progress=sampling)
    views = _views(
        _from_audio_start(samples, _audio_start(ctx.source)),
        change=config.vision_change_fraction,
        hold=config.vision_change_hold,
        max_views=limit,
        end=ctx.duration or None,
    )
    if ctx.source_hash:
        entries = []
        for view in views:
            stored = _store(view.keyframe, config, ctx.source_hash)
            if not stored:
                return views
            entries.append(
                {
                    "start": view.start,
                    "end": view.end,
                    "keyframe": stored,
                    "keyframe_time": view.keyframe_time,
                }
            )
        try:
            path.write_text(json.dumps({"config": header, "views": entries}), encoding="utf-8")
        except OSError:
            pass
    return views


def _from_audio_start(
    samples: Iterable[tuple[float, Frame]], origin: float
) -> Iterator[tuple[float, Frame]]:
    before: Frame | None = None
    for time, frame in samples:
        shifted = time - origin
        if shifted < 0:
            before = frame
            continue
        if before is not None:
            if shifted > 0:
                yield 0.0, before
            before = None
        yield shifted, frame
    if before is not None:
        yield 0.0, before


def state_limit(config: Config, duration: float) -> int:
    """Return how many board states a recording of ``duration`` seconds may have."""
    if config.vision_states_per_minute <= 0 or config.vision_min_states < 1:
        raise ValueError("vision: vision_states_per_minute and vision_min_states must be positive")
    scaled = math.ceil(duration / 60 * config.vision_states_per_minute - 1e-9)
    return max(config.vision_min_states, scaled)


def _load_views(path: Path, header: dict[str, Any], cache: Path) -> list[_View] | None:
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("config") != header:
            return None
        views = []
        for entry in stored["views"]:
            keyframe = (cache / entry["keyframe"]).read_bytes()
            if _content_name(keyframe) != PurePosixPath(entry["keyframe"]).name:
                return None
            views.append(
                _View(
                    start=entry["start"],
                    end=entry["end"],
                    keyframe=keyframe,
                    keyframe_time=entry["keyframe_time"],
                )
            )
        return views
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _audio_start(path: Path) -> float:
    try:
        with av.open(str(path)) as container:
            audio = container.streams.audio
            if not audio or audio[0].start_time is None or audio[0].time_base is None:
                return 0.0
            return float(audio[0].start_time * audio[0].time_base)
    except (av.error.FFmpegError, OSError):
        return 0.0


def _speech(sentences: Sequence[Sentence], start: float, end: float) -> str:
    text = " ".join(s.text.strip() for s in sentences if s.start < end and s.end > start)
    if len(text) <= _SPEECH_CHARS:
        return text
    return text[:_SPEECH_CHARS].rsplit(" ", 1)[0] + " ..."


def _store(png: bytes, config: Config, source_hash: str) -> str:
    name = _content_name(png)
    relative = PurePosixPath(source_hash or "unhashed", "keyframes", name)
    path = Path(config.cache_dir) / relative
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or _content_name(path.read_bytes()) != name:
            partial = path.with_suffix(".tmp")
            partial.write_bytes(png)
            partial.replace(path)
    except OSError:
        return ""
    return relative.as_posix()


def _content_name(png: bytes) -> str:
    return f"{hashlib.sha256(png).hexdigest()[:16]}.png"


def has_video(path: Path) -> bool:
    """Whether ``path`` is a recording with a video track; ``False`` for anything else."""
    try:
        with av.open(str(path)) as container:
            return bool(container.streams.video)
    except (av.error.FFmpegError, OSError):
        return False


def _sample_frames(
    path: Path, fps: float, *, crop: Crop | None = None, progress: Progress | None = None
) -> Iterator[tuple[float, Frame]]:
    if fps <= 0:
        raise ValueError("vision: vision_fps must be positive")
    if crop is not None:
        x0, y0, x1, y1 = crop
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"vision: crop {crop} is not a box inside the frame")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: recording contains no video track")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration = container.duration / av.time_base if container.duration else 0.0
        due = 0
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-6 < due / fps:
                continue
            due = int(frame.time * fps + 1e-6) + 1
            picture: Frame = frame.to_ndarray(format="rgb24")  # type: ignore[assignment, unused-ignore]
            if crop is not None:
                height, width = picture.shape[:2]
                picture = picture[
                    round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)
                ]
            if progress and duration:
                progress("vision", min(1.0, frame.time / duration))
            yield frame.time, picture


def _for_comparison(frame: Frame) -> Frame:
    gray = frame @ _GRAY
    factor = max(1, round(gray.shape[0] / _COMPARE_HEIGHT))
    height, width = gray.shape[0] // factor * factor, gray.shape[1] // factor * factor
    blocks = gray[:height, :width].reshape(height // factor, factor, width // factor, factor)
    small: Frame = blocks.mean(axis=(1, 3)).round().astype(np.uint8)
    return small


@dataclass(frozen=True, kw_only=True)
class _View:
    start: float
    end: float
    keyframe: bytes
    keyframe_time: float


def _views(
    samples: Iterable[tuple[float, Frame]],
    *,
    change: float,
    hold: int,
    max_views: int,
    end: float | None = None,
) -> list[_View]:
    if not 0 < change < 1 or hold < 1 or max_views < 1:
        raise ValueError(
            "vision: vision_change_fraction must be in (0, 1), and vision_change_hold and "
            "the board-state limit positive"
        )
    views: list[_View] = []
    reference: Frame | None = None
    best: tuple[float, float, Frame] = (0.0, 0.0, np.zeros((1, 1, 3), np.uint8))
    pending: list[tuple[float, Frame, Frame]] = []
    start = previous = last = 0.0
    seen = 0
    for time, frame in samples:
        small = _for_comparison(frame)
        previous, last, seen = last, time, seen + 1
        if reference is None:
            reference, start, best = small, time, (_sharpness(small), time, frame)
        elif _changed(small, reference) <= change:
            pending = []
            if (score := _sharpness(small)) > best[0]:
                best = (score, time, frame)
        else:
            if pending and _changed(small, pending[-1][1]) > change:
                pending = []
            pending.append((time, small, frame))
            if len(pending) >= hold:
                views.append(_close(start, pending[0][0], best))
                start, reference = pending[0][0], pending[0][1]
                best = max(((_sharpness(s), t, f) for t, s, f in pending), key=lambda b: b[0])
                pending = []
    if reference is None:
        return []
    step = last - previous if seen > 1 else 1.0
    views.append(_close(start, max(end, last) if end is not None else last + step, best))
    return _merge_shortest(views, max_views)


def _changed(before: Frame, after: Frame) -> float:
    return float((np.abs(before.astype(np.int16) - after.astype(np.int16)) > _NOISE).mean())


def _sharpness(gray: Frame) -> float:
    values = gray.astype(np.float32)
    return float(np.abs(np.diff(values, axis=0)).mean() + np.abs(np.diff(values, axis=1)).mean())


def _close(start: float, end: float, best: tuple[float, float, Frame]) -> _View:
    return _View(start=start, end=end, keyframe=_png(best[2]), keyframe_time=best[1])


def _png(frame: Frame) -> bytes:
    picture = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
    codec = av.CodecContext.create("png", "w")
    codec.width, codec.height, codec.pix_fmt = picture.width, picture.height, "rgb24"
    return b"".join(bytes(packet) for packet in [*codec.encode(picture), *codec.encode(None)])


def _merge_shortest(views: list[_View], max_views: int) -> list[_View]:
    views = list(views)
    while len(views) > max_views:
        i = min(range(len(views)), key=lambda k: _length(views[k]))
        if i == len(views) - 1 or i > 0 and _length(views[i - 1]) <= _length(views[i + 1]):
            i -= 1
        left, right = views[i], views[i + 1]
        kept = left if _length(left) >= _length(right) else right
        views[i : i + 2] = [
            _View(
                start=left.start,
                end=right.end,
                keyframe=kept.keyframe,
                keyframe_time=kept.keyframe_time,
            )
        ]
    return views


def _length(view: _View) -> float:
    return view.end - view.start


def read_keyframe(
    config: Config,
    png: bytes,
    speech: str = "",
    *,
    kind: str = "unknown",
    usage: Counter[str] | None = None,
    previous: PreviousState | None = None,
) -> dict[str, Any] | None:
    """Read one keyframe with the vision model (architecture.md 6.5)."""
    image = _data_url(png)
    if config.vision_reader == "ocr":
        return _read_ocr(config, image, usage)
    values = {"kind": kind, "speech": speech.strip() or "-"}
    if previous is not None:
        box = previous.new_ink
        values |= {
            "previous_description": previous.description.strip() or "-",
            "previous_text": previous.text.strip() or "-",
            "previous_latex": (previous.latex or "").strip() or "-",
            "new_ink": "not found"
            if box is None
            else f"[{', '.join(f'{v:.2f}' for v in box)}] (x0, y0, x1, y1: fractions of this "
            "frame's width and height from its top-left corner)",
        }
    template = _PROMPT if previous is None else _PROMPT_V2
    prompt = _PLACEHOLDER.sub(lambda m: values.get(m[1], m[0]), template)
    request: dict[str, Any] = {
        "model": config.vision_model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]},
        ],
        "response_format": _SCHEMA,
    }
    if config.vision_effort:
        request["reasoning_effort"] = config.vision_effort
    try:
        answer = endpoint.ask(
            config, request, name="vision", valid=_readable, usage=usage, vision=True
        )
    except (ConnectionError, endpoint.InvalidAnswer) as exc:
        warnings.warn(f"vision: keyframe left out: {exc}", UserWarning, stacklevel=2)
        return None
    description, text, latex, confidence = (answer[k] for k in _FIELDS)
    return {
        "description": description.strip(),
        "text": text.strip(),
        "latex": (latex or "").strip() or None,
        "confidence": min(1.0, max(0.0, float(confidence))),
    }


def _read_ocr(config: Config, image: str, usage: Counter[str] | None) -> dict[str, Any] | None:
    request: dict[str, Any] = {
        "model": config.vision_ocr_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": _OCR_PROMPT},
                ],
            }
        ],
    }
    url = config.vision_url
    if url not in _NO_LOGPROBS:
        request["logprobs"] = True
    try:
        try:
            answer = _ask_ocr(config, request, usage)
        except endpoint.Refused:
            if "logprobs" not in request:
                raise
            # Some servers do not take logprobs: ask without, and confidence is "not measured".
            _NO_LOGPROBS.add(url)
            del request["logprobs"]
            answer = _ask_ocr(config, request, usage)
    except (ConnectionError, endpoint.InvalidAnswer) as exc:
        warnings.warn(f"vision: keyframe left out: {exc}", UserWarning, stacklevel=3)
        return None
    return {key: answer[key] for key in _FIELDS}


def _ask_ocr(config: Config, request: dict[str, Any], usage: Counter[str] | None) -> dict[str, Any]:
    return endpoint.ask(
        config,
        request,
        name="vision",
        valid=_readable,
        usage=usage,
        vision=True,
        convert=_from_ocr,
    )


def _from_ocr(choice: Any) -> dict[str, Any] | None:
    content = getattr(choice.message, "content", None)
    if not isinstance(content, str):
        return None
    formulas = ocr_formulas(content)
    text = _OCR_FORMULA.sub(lambda m: _plain_label(m[1] or m[2]), content)
    text = "\n".join(" ".join(line.split()) for line in text.splitlines())
    return {
        "description": "",
        "text": text.strip(),
        "latex": "\n".join(formulas) or None,
        "confidence": _token_confidence(choice),
    }


def _plain_label(segment: str) -> str:
    segment = segment.strip()
    if not _OCR_LABEL.fullmatch(segment):
        return ""
    segment = _OCR_CIRCLED.sub(r"(\1)", segment)
    return _OCR_COMMAND.sub(lambda m: f" {m[1]}" if m[1] is not None else m[2], segment).strip()


def ocr_formulas(text: str) -> list[str]:
    """Return the formulas in OCR text: its ``$...$`` and ``$$...$$`` segments, labels left out."""
    formulas: list[str] = []
    for match in _OCR_FORMULA.finditer(text):
        formula = (match[1] or match[2]).strip()
        if formula and not _OCR_LABEL.fullmatch(formula):
            formulas.append(formula)
    return formulas


def _token_confidence(choice: Any) -> float:
    tokens = getattr(getattr(choice, "logprobs", None), "content", None) or ()
    values = [t.logprob for t in tokens if isinstance(getattr(t, "logprob", None), (int, float))]
    if not values:
        return _OCR_UNMEASURED
    return round(min(1.0, max(0.0, sum(math.exp(v) for v in values) / len(values))), 3)


def _readable(answer: dict[str, Any]) -> bool:
    if not all(key in answer for key in _FIELDS):
        return False
    description, text, latex, confidence = (answer[k] for k in _FIELDS)
    return (
        isinstance(description, str)
        and isinstance(text, str)
        and (latex is None or isinstance(latex, str))
        and not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and math.isfinite(confidence)
    )


@stage("attach_visual")
def attach_visual(ctx: Context) -> Context:
    """Join board states to the segments they overlap, by interval intersection."""
    if not ctx.visual:
        return ctx
    ctx.segments = tuple(
        _with_visual(s, overlapping(ctx.visual, s.start, s.end)) for s in ctx.segments
    )
    ctx.stats["board_states"] = len(ctx.visual)
    return ctx


def _with_visual(segment: Segment, visual: tuple[VisualContext, ...]) -> Segment:
    return Segment(
        sentence_start=segment.sentence_start,
        sentence_end=segment.sentence_end,
        start=segment.start,
        end=segment.end,
        text=segment.text,
        visual=visual,
        close_reason=segment.close_reason,
    )

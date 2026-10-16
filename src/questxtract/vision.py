"""Stage 5: optional enrichment of the audio with what is on the board.

Three stages. ``layout`` asks the model once where the content is and what it is (6.2), so
``vision`` can crop to it and tell the captioning prompt which kind of frame it reads.

Capture and the join are two stages, not one, because of architecture.md 11.1 rule 4:
capture is expensive and caches as ``vision.json``, while the interval join is eight lines
and always recomputes. Nailing one board state to one segment at capture time would bake in
a wrong join that every consumer inherits.

All three no-op themselves when vision is off, rather than being filtered out by the runner
(rule 3).
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import av
import numpy as np

from . import endpoint
from .config import Config
from .models import Context, Layout, Progress, Segment, Sentence, VisualContext, stage

_COMPARE_HEIGHT = 480  # architecture.md 6.3: change detection compares frames at about 480p
_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # ITU-R BT.601 luma weights
_NOISE = 32  # grey levels a pixel must move by to count as changed, above codec noise

# The prompt and the schema are module-level, version-tagged, so a prompt change is a reviewable
# diff. Both are part of every request, so a change to either asks again (endpoint.ask's cache).
VISION_PROMPT_VERSION = "v0"  # tests/data/vision/prompt_v0.txt, chosen in #52
_SPEECH_CHARS = 1500  # the speech sent with one keyframe: its own view's words, kept short
_GIVE_UP_AFTER = 3  # failed readings in a row, before any success, that stop the captioning

_FIELDS = ("description", "text", "latex", "confidence")

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
Crop = tuple[float, float, float, float]  # normalised (x0, y0, x1, y1), from the layout probe

_FULL_FRAME: Crop = (0.0, 0.0, 1.0, 1.0)
_KINDS = ("slides", "blackboard", "whiteboard", "handwriting", "screencast", "unknown")
_PROBE_AT = (0.2, 0.5, 0.8)  # where in the recording the probe's three frames are taken
_PROBE_WIDTH = 640  # the region is in fractions, so a small frame shows it as well
_REGION_PAD = 0.02  # VLM boxes are loose; a margin keeps the board's edges in the crop
_REGION_MIN_AREA = 0.1  # a smaller box is more likely a misreading than the content

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
    """Find where the content is and what kind it is, once per recording (architecture.md 6.2).

    A no-op unless ``config.vision_enabled``, and for a source with no video track. Three
    frames from across the recording go to ``vision_model`` in one request; the answer sets
    ``ctx.layout``. Any failure of the endpoint, the model or the recording leaves the full
    frame and ``unknown`` with a warning: the probe only sharpens vision, so it never ends the
    run. Not stage-cached: the request is cached by :func:`endpoint.ask`, which keeps only a
    valid answer, so a probe that failed is asked again on the next run.

    Raises:
        RuntimeError: If the optional ``questxtract[llm]`` extra is not installed.
        ValueError: If the settings cannot be used, such as an unreadable ``.env``.

    """
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
    try:
        answer = endpoint.ask(ctx.config, request, name="layout")
    except (ConnectionError, endpoint.InvalidAnswer) as exc:
        return _fallback(ctx, str(exc))
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
    """Take the frames at ``_PROBE_AT`` of the recording, as PNG, ``_PROBE_WIDTH`` wide at most.

    Seeks to each one rather than decoding the whole recording. Without a known duration,
    only the first frame.
    """
    pngs: list[bytes] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = container.duration / av.time_base if container.duration else 0.0
        for fraction in _PROBE_AT if duration else (0.0,):
            target = duration * fraction
            container.seek(int(target * av.time_base))  # to the keyframe at or before it
            for frame in container.decode(stream):
                if frame.time is not None and frame.time + 1e-6 < target:
                    continue
                width = min(frame.width, _PROBE_WIDTH)
                height = round(frame.height * width / frame.width / 2) * 2 or 2
                picture: Frame = frame.reformat(width=width, height=height).to_ndarray(
                    format="rgb24"
                )
                pngs.append(_png(picture))
                break
    return pngs


def _layout_from(answer: dict[str, Any]) -> Layout | None:
    """Return the :class:`Layout` an answer gives, or ``None`` when it is malformed.

    The box is padded by ``_REGION_PAD`` on each side, within the frame. A box smaller than
    ``_REGION_MIN_AREA`` of the frame is not trusted: the kind is kept, the crop is not.
    """
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
    """Capture board states from the recording into ``ctx.visual`` (architecture.md 6.3 to 6.5).

    A no-op unless ``config.vision_enabled``, and for a source with no video track (an MP3, or
    a transcript): there is no board to read. Otherwise the recording is sampled at
    ``vision_fps`` and cut into stable views, one keyframe each; each keyframe is read by
    ``vision_model`` together with what was said while its view was on screen, and stored in
    the cache. A reading that fails leaves that one view out with a warning; when the first
    readings all fail, the rest are not sent. Frames are cropped to the ``layout`` stage's
    ``content_region`` and its ``kind`` goes into the prompt; one answer holds for the whole
    recording. De-occlusion is not built.
    """
    if not ctx.config.vision_enabled or not has_video(ctx.source):
        return ctx
    config, report = ctx.config, ctx.progress
    views = _cut(ctx)
    visual: list[VisualContext] = []
    failed = 0
    for index, view in enumerate(views):
        if report:
            report("vision", 0.5 + 0.5 * index / len(views))
        speech = _speech(ctx.sentences, view.start, view.end)
        fields = _caption(config, view.keyframe, speech, kind=ctx.layout.kind)
        if fields is None:
            failed += 1
            if not visual and failed >= _GIVE_UP_AFTER:
                warnings.warn(
                    f"vision: the first {failed} keyframes could not be read, so the rest are "
                    "not sent; check vision_model, the API key and the endpoint",
                    UserWarning,
                    stacklevel=2,
                )
                break
            continue
        path = _store(view.keyframe, config, ctx.source_hash)
        visual.append(VisualContext(start=view.start, end=view.end, keyframe_path=path, **fields))
    ctx.visual = tuple(visual)
    ctx.stats["vision_failed"] = len(views) - len(visual)
    return ctx


def _cut(ctx: Context) -> list[_View]:
    """Sample the recording and cut it into views, cached on the sampling settings.

    Frames are cropped to the ``layout`` stage's ``content_region``. Sampling decodes the
    whole video, minutes for a lecture, while the views depend only on the recording, the
    crop and these settings, not on the transcript or the model. Entries live in
    ``cache_dir/<source_hash>/views.json``, keyframes beside them (:func:`_store`); without a
    source hash nothing is cached, as for the runner's own cache (architecture.md 11.2).
    """
    config, report = ctx.config, ctx.progress
    region = ctx.layout.content_region
    crop = None if region == _FULL_FRAME else region
    header = {
        "content_region": list(region),
        "vision_fps": config.vision_fps,
        "vision_change_fraction": config.vision_change_fraction,
        "vision_change_hold": config.vision_change_hold,
        "vision_max_states": config.vision_max_states,
        "duration": ctx.duration,
    }
    path = Path(config.cache_dir) / ctx.source_hash / "views.json"
    if ctx.source_hash and not config.from_scratch:
        cached = _load_views(path, header, Path(config.cache_dir))
        if cached is not None:
            return cached

    def sampling(_: str, fraction: float) -> None:
        if report:
            report("vision", fraction / 2)  # sampling is the first half of the stage

    samples = _sample_frames(ctx.source, config.vision_fps, crop=crop, progress=sampling)
    views = _views(
        _from_audio_start(samples, _audio_start(ctx.source)),
        change=config.vision_change_fraction,
        hold=config.vision_change_hold,
        max_views=config.vision_max_states,
        end=ctx.duration or None,
    )
    if ctx.source_hash:
        entries = []
        for view in views:
            stored = _store(view.keyframe, config, ctx.source_hash)
            if not stored:
                return views  # the cache cannot be written: keep the views, cache nothing
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
    """Shift sample times so that zero is the audio start, the transcript's clock.

    What was on screen before the audio starts is never spoken about, so those samples are
    dropped, except the last one: the board visible when the audio starts, at time zero. Every
    other sample keeps its own, later time, so no view can end where it starts.
    """
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
    if before is not None:  # the picture ends before the audio starts
        yield 0.0, before


def _load_views(path: Path, header: dict[str, Any], cache: Path) -> list[_View] | None:
    """Read cached views, or ``None`` when they are missing, stale or damaged.

    A keyframe is named by the start of its content hash (:func:`_store`), so one whose bytes
    no longer match its name, such as a write cut short, makes the whole entry stale.
    """
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
    """When the audio starts on the container's clock, in seconds: the transcript's zero.

    Word times count from the first audio sample, while frame times are the video stream's
    own, so a file whose streams do not start at zero (a trimmed recording) would put board
    states beside the wrong speech. Subtracting the audio start keeps a real offset between
    sound and picture, which zeroing the video stream alone would remove.
    """
    try:
        with av.open(str(path)) as container:
            audio = container.streams.audio
            if not audio or audio[0].start_time is None or audio[0].time_base is None:
                return 0.0
            return float(audio[0].start_time * audio[0].time_base)
    except (av.error.FFmpegError, OSError):
        return 0.0


def _speech(sentences: Sequence[Sentence], start: float, end: float) -> str:
    """Return what was said while a view was on screen, cut to ``_SPEECH_CHARS`` at a word."""
    text = " ".join(s.text.strip() for s in sentences if s.start < end and s.end > start)
    if len(text) <= _SPEECH_CHARS:
        return text
    return text[:_SPEECH_CHARS].rsplit(" ", 1)[0] + " ..."


def _store(png: bytes, config: Config, source_hash: str) -> str:
    """Write a keyframe into the cache; return its path relative to ``cache_dir`` (#59).

    Named by its content, so a re-run writes nothing new, and a file whose bytes do not match
    its name is written again. Written through a temporary file, so an interrupted write
    leaves no half image under the real name. Relative to the cache, so a cached result still
    finds its images from another working directory and board-state ids do not depend on where
    the cache lives. ``""`` when the cache cannot be written: the reading is kept, only the
    image is missing.
    """
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
    """Return a keyframe's file name: the start of its SHA-256, so the name checks the content."""
    return f"{hashlib.sha256(png).hexdigest()[:16]}.png"


def has_video(path: Path) -> bool:
    """Whether ``path`` is a recording with a video track; ``False`` for anything else."""
    try:
        with av.open(str(path)) as container:
            return bool(container.streams.video)
    except (av.error.FFmpegError, OSError):
        return False


def overlapping(
    visual: Sequence[VisualContext], start: float, end: float
) -> tuple[VisualContext, ...]:
    """Board states whose lifetime overlaps the half-open interval ``[start, end)``."""
    return tuple(v for v in visual if v.start < end and v.end > start)


def _sample_frames(
    path: Path, fps: float, *, crop: Crop | None = None, progress: Progress | None = None
) -> Iterator[tuple[float, Frame]]:
    """Yield ``(timestamp, frame)`` at ``fps`` from the first video stream (architecture.md 6.3).

    Each frame is the full-colour RGB picture, cropped to ``crop`` when given, so a frame that
    ends up a keyframe can go to the model as is; :func:`_for_comparison` makes the small copy.
    Streams: one decoded frame at a time, never the whole video. Every frame is decoded and
    only the sampled ones are converted, since seeking saves nothing at one sample a second.
    A sample is the first frame at or after each multiple of ``1 / fps``, so timestamps do
    not drift. Close the generator if you stop early: it holds the file open.

    Raises:
        ValueError: If ``fps`` is not positive, ``crop`` is not inside the frame, or the
            recording has no video stream (on the first ``next()``).

    """
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
        stream.thread_type = "AUTO"  # decoding is the cost; let FFmpeg use every core
        duration = container.duration / av.time_base if container.duration else 0.0
        due = 0
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-6 < due / fps:
                continue
            due = int(frame.time * fps + 1e-6) + 1
            picture: Frame = frame.to_ndarray(format="rgb24")
            if crop is not None:
                height, width = picture.shape[:2]
                picture = picture[
                    round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)
                ]
            if progress and duration:
                progress("vision", min(1.0, frame.time / duration))
            yield frame.time, picture


def _for_comparison(frame: Frame) -> Frame:
    """Grayscale, downscaled by a whole factor to about 480 rows, for change detection.

    Block averaging rather than dropping pixels also smooths codec noise between frames.
    """
    gray = frame @ _GRAY
    factor = max(1, round(gray.shape[0] / _COMPARE_HEIGHT))
    height, width = gray.shape[0] // factor * factor, gray.shape[1] // factor * factor
    blocks = gray[:height, :width].reshape(height // factor, factor, width // factor, factor)
    small: Frame = blocks.mean(axis=(1, 3)).round().astype(np.uint8)
    return small


@dataclass(frozen=True, kw_only=True)
class _View:
    """One stable view: a slide or board as it stood, with the keyframe that shows it.

    Args:
        start: First sample of the view, in seconds.
        end: Where the next view starts, or the end of the last sample.
        keyframe: The view's sharpest sample, PNG-encoded at full resolution.
        keyframe_time: When that sample was taken, in seconds.

    """

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
    """Cut sampled frames into stable views, one keyframe each (architecture.md 6.4).

    A sample belongs to the current view while at most ``change`` of its pixels differ from
    the view's first sample. A new view starts at the first differing sample, but only once
    ``hold`` samples in a row differ from the old view and resemble each other: a person
    walking past never resembles themselves a second later, so they start nothing. Of the
    samples that match a view, the sharpest is its keyframe, so a frame with someone in
    front of the board is not chosen. Only a few full frames are held at a time. More than
    ``max_views`` views are merged, shortest first.

    ``hold`` counts samples, so its length in seconds depends on the sampling rate: 3 samples
    are 3 s at 1 fps but 0.6 s at 5 fps. The last view ends at ``end``, the recording's end
    time; without it, one sample interval after the last sample (one second when there is a
    single sample), so no view is ever empty.

    Raises:
        ValueError: If ``change`` is not in ``(0, 1)``, or ``hold`` or ``max_views`` is not
            positive.

    """
    if not 0 < change < 1 or hold < 1 or max_views < 1:
        raise ValueError(
            "vision: vision_change_fraction must be in (0, 1), and vision_change_hold and "
            "vision_max_states positive"
        )
    views: list[_View] = []
    reference: Frame | None = None  # the current view's first sample, small
    best: tuple[float, float, Frame] = (0.0, 0.0, np.zeros((1, 1, 3), np.uint8))
    pending: list[tuple[float, Frame, Frame]] = []  # (time, small, full): a new view forming
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
                pending = []  # still moving: not a view yet
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
    """Share of pixels whose grey level moved by more than the codec noise."""
    return float((np.abs(before.astype(np.int16) - after.astype(np.int16)) > _NOISE).mean())


def _sharpness(gray: Frame) -> float:
    """Mean gradient: higher for crisp text and lines, lower for blur and empty board."""
    values = gray.astype(np.float32)
    return float(np.abs(np.diff(values, axis=0)).mean() + np.abs(np.diff(values, axis=1)).mean())


def _close(start: float, end: float, best: tuple[float, float, Frame]) -> _View:
    """Finish a view, encoding its keyframe only now that it is final."""
    return _View(start=start, end=end, keyframe=_png(best[2]), keyframe_time=best[1])


def _png(frame: Frame) -> bytes:
    """Encode an RGB frame as PNG, lossless so text stays readable for the model."""
    picture = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
    codec = av.CodecContext.create("png", "w")
    codec.width, codec.height, codec.pix_fmt = picture.width, picture.height, "rgb24"
    return b"".join(bytes(packet) for packet in [*codec.encode(picture), *codec.encode(None)])


def _merge_shortest(views: list[_View], max_views: int) -> list[_View]:
    """Merge the shortest view into its shorter neighbour until at most ``max_views`` remain.

    The merged view spans both and keeps the keyframe of the longer one.
    """
    views = list(views)
    while len(views) > max_views:
        i = min(range(len(views)), key=lambda k: _length(views[k]))
        if i == len(views) - 1 or i > 0 and _length(views[i - 1]) <= _length(views[i + 1]):
            i -= 1  # merge with the left neighbour
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


def _caption(
    config: Config, png: bytes, speech: str, *, kind: str = "unknown"
) -> dict[str, Any] | None:
    """Read one keyframe with the vision model (architecture.md 6.5).

    ``png`` is the keyframe as :class:`_View` holds it; ``speech`` is what was said while its
    view was on screen; ``kind`` is what the layout probe says the frame shows. Returns the
    :class:`VisualContext` fields the model reads, or ``None`` with a warning when the call
    fails or a field is malformed: a board description only enriches questions the pipeline
    already found, so unlike the judge, vision never ends the run. Cached on the request,
    image included (:func:`endpoint.ask`). Only what the endpoint or the model did wrong is
    tolerated; a setting that cannot be used, such as an unreadable ``.env``, is a user error
    that would drop every keyframe the same way, so it is raised.

    Raises:
        RuntimeError: If the optional ``questxtract[llm]`` extra is not installed.
        ValueError: If the settings cannot be used, such as an unreadable ``.env``.

    """
    image = _data_url(png)
    # As prompt v0 was measured in #52: the prompt, with its kind and speech filled in, as the
    # system message, and the image alone as the user message.
    prompt = _PROMPT.replace("{kind}", kind).replace("{speech}", speech.strip() or "-")
    request: dict[str, Any] = {
        "model": config.vision_model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]},
        ],
        "response_format": _SCHEMA,
    }
    try:
        answer = endpoint.ask(config, request, name="vision", valid=_readable)
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


def _readable(answer: dict[str, Any]) -> bool:
    """Whether an answer has every board field with a usable type; nothing else is cached."""
    if not all(key in answer for key in _FIELDS):
        return False  # a missing "latex" is not null: the schema asks for every key
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
    """Return ``segment`` with its board states attached; it is frozen, so copy."""
    return Segment(
        sentence_start=segment.sentence_start,
        sentence_end=segment.sentence_end,
        start=segment.start,
        end=segment.end,
        text=segment.text,
        visual=visual,
        close_reason=segment.close_reason,
    )

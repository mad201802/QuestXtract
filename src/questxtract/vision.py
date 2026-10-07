"""Stage 5: optional enrichment of the audio with what is on the board.

Two stages, not one, because of architecture.md 11.1 rule 4: capture is expensive and caches
as ``vision.json``, while the interval join is eight lines and always recomputes. Nailing
one board state to one segment at capture time would bake in a wrong join that every
consumer inherits.

Both no-op themselves when vision is off, rather than being filtered out by the runner
(rule 3).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np

from .models import Context, Progress, Segment, VisualContext, stage

_COMPARE_HEIGHT = 480  # architecture.md 6.3: change detection compares frames at about 480p
_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # ITU-R BT.601 luma weights
_NOISE = 32  # grey levels a pixel must move by to count as changed, above codec noise

Frame = np.ndarray[Any, np.dtype[np.uint8]]
Crop = tuple[float, float, float, float]  # normalised (x0, y0, x1, y1), from the layout probe


@stage(
    "vision",
    cache="visual",
    uses=(
        "vision_fps",
        "vision_model",
        "vision_max_states",
        "vision_change_fraction",
        "vision_change_hold",
    ),
)
def vision(ctx: Context) -> Context:
    """Capture board states from the recording into ``ctx.visual``.

    A no-op unless ``config.vision_enabled``, which is false unless an endpoint is
    configured, and for a source with no video track (an MP3, or a transcript): there is no
    board to read. The layout probe, frame sampling, de-occlusion, board-state segmentation
    and VLM captioning of architecture.md 6.2 to 6.5 are not implemented yet (#59).
    """
    if not ctx.config.vision_enabled or not _has_video(ctx.source):
        return ctx
    # ponytail: #59, frame sampling, keyframes and VLM captioning (architecture.md 6.2-6.5).
    raise NotImplementedError("vision is not implemented yet (#59); set vision_enabled=False")


def _has_video(path: Path) -> bool:
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

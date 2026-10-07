"""Stage 5: optional enrichment of the audio with what is on the board.

Two stages, not one, because of architecture.md 11.1 rule 4: capture is expensive and caches
as ``vision.json``, while the interval join is eight lines and always recomputes. Nailing
one board state to one segment at capture time would bake in a wrong join that every
consumer inherits.

Both no-op themselves when vision is off, rather than being filtered out by the runner
(rule 3).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import av
import numpy as np

from .models import Context, Progress, Segment, VisualContext, stage

_COMPARE_HEIGHT = 480  # architecture.md 6.3: change detection compares frames at about 480p
_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # ITU-R BT.601 luma weights

Frame = np.ndarray[Any, np.dtype[np.uint8]]
Crop = tuple[float, float, float, float]  # normalised (x0, y0, x1, y1), from the layout probe


@stage("vision", cache="visual", uses=("vision_fps", "vision_model", "vision_max_states"))
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

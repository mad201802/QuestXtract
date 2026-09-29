"""Stage 5: optional enrichment of the audio with what is on the board.

Two stages, not one, because of pipeline.md 11.1 rule 4: capture is expensive and caches
as ``vision.json``, while the interval join is eight lines and always recomputes. Nailing
one board state to one segment at capture time would bake in a wrong join that every
consumer inherits.

Both no-op themselves when vision is off, rather than being filtered out by the runner
(rule 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import av
import numpy as np

from . import llm
from .models import Context, Segment, Transcript, VisualContext, stage

_ANALYSIS_HEIGHT = 480  # pipeline.md 6.3: compare boards at ~480p, caption the full frame
_INK_DELTA = 32  # grey levels a pixel must move by to count as changed, above codec noise
_CHANGE_FRACTION = 0.02  # share of changed pixels that starts a new board state


@dataclass(frozen=True)
class BoardState:
    """One board state before captioning: its lifetime and its keyframe.

    Args:
        start: Start of the lifetime in seconds.
        end: End of the lifetime in seconds.
        keyframe: The state's last sample, PNG-encoded: the completed board.
        change: Share of pixels that changed when this state began; ``1.0`` for the first.

    """

    start: float
    end: float
    keyframe: bytes
    change: float = 1.0


def board_states(path: Path, fps: float = 1.0, max_states: int = 80) -> list[BoardState]:
    """Cut a recording into board states by sampling it and diffing consecutive samples.

    Samples the first video stream at ``fps`` and starts a new state wherever more than
    2 % of the pixels changed. The slides regime of pipeline.md 6.1, where a transition
    changes many pixels at once. When there are more than ``max_states``, only the largest
    changes are kept as boundaries.

    Raises:
        ValueError: If ``fps`` or ``max_states`` is not positive, or there is no video.

    """
    # ponytail: layout probe (6.2), de-occlusion and the camera-motion gate (6.3) and the
    # ink-count curve with erase events (6.4). A lecturer walking past a board cuts a state.
    if fps <= 0 or max_states <= 0:
        raise ValueError("vision: vision_fps and vision_max_states must be positive")
    states: list[BoardState] = []
    previous: np.ndarray[Any, np.dtype[np.uint8]] | None = None
    keyframe: av.VideoFrame | None = None
    next_sample, end = 0.0, 0.0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: recording contains no video track")
        stream = container.streams.video[0]
        step = 1.0 / float(stream.average_rate or fps)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            end = frame.time + step
            if frame.time + 1e-6 < next_sample:
                continue
            next_sample = frame.time + 1.0 / fps
            gray = _gray(frame)
            change = 1.0 if previous is None else _changed(previous, gray)
            if previous is None or change > _CHANGE_FRACTION:
                if keyframe is not None:
                    states[-1] = _close(states[-1], frame.time, keyframe)
                states.append(
                    BoardState(start=frame.time, end=frame.time, keyframe=b"", change=change)
                )
            keyframe, previous = frame, gray
    if keyframe is not None:
        states[-1] = _close(states[-1], end, keyframe)
    return _limit(states, max_states)


def _gray(frame: av.VideoFrame) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Return the frame as a grayscale array no taller than the analysis height."""
    if frame.height > _ANALYSIS_HEIGHT:
        width = round(frame.width * _ANALYSIS_HEIGHT / frame.height / 2) * 2
        frame = frame.reformat(width=width, height=_ANALYSIS_HEIGHT)
    return cast("np.ndarray[Any, np.dtype[np.uint8]]", frame.to_ndarray(format="gray"))


def _changed(
    before: np.ndarray[Any, np.dtype[np.uint8]], after: np.ndarray[Any, np.dtype[np.uint8]]
) -> float:
    """Share of pixels whose grey level moved by more than the codec noise floor."""
    delta = np.abs(before.astype(np.int16) - after.astype(np.int16))
    return float((delta > _INK_DELTA).mean())


def _close(state: BoardState, end: float, keyframe: av.VideoFrame) -> BoardState:
    """Finish ``state`` at ``end`` with ``keyframe`` as its completed board."""
    return BoardState(start=state.start, end=end, keyframe=_png(keyframe), change=state.change)


def _png(frame: av.VideoFrame) -> bytes:
    """Encode one frame as PNG, at full resolution, for the vision model."""
    rgb = frame.reformat(format="rgb24")
    codec = av.CodecContext.create("png", "w")
    codec.width, codec.height, codec.pix_fmt = rgb.width, rgb.height, "rgb24"
    return b"".join(bytes(packet) for packet in [*codec.encode(rgb), *codec.encode(None)])


def _limit(states: list[BoardState], max_states: int) -> list[BoardState]:
    """Merge states until at most ``max_states`` remain, keeping the largest changes.

    A merged state keeps the keyframe of its last part: the board as it ended up.
    """
    if len(states) <= max_states:
        return states
    by_change = sorted(range(1, len(states)), key=lambda i: states[i].change, reverse=True)
    starts = [0, *sorted(by_change[: max_states - 1])]
    ends = [*starts[1:], len(states)]
    return [
        BoardState(
            start=states[first].start,
            end=states[last - 1].end,
            keyframe=states[last - 1].keyframe,
            change=states[first].change,
        )
        for first, last in zip(starts, ends, strict=True)
    ]


@stage("vision", cache="visual", uses=("vision_fps", "vision_model", "vision_max_states"))
def vision(ctx: Context) -> Context:
    """Capture board states from the recording into ``ctx.visual``.

    A no-op unless ``config.vision_enabled``, which is false unless an endpoint is
    configured. Cuts the recording into board states with :func:`board_states`, then
    captions each keyframe with one :func:`questxtract.llm.describe_board` call, passing
    what was said while the board was on screen.
    """
    if not ctx.config.vision_enabled:
        return ctx
    config = ctx.config
    states = board_states(ctx.source, config.vision_fps, config.vision_max_states)
    visual = []
    for index, state in enumerate(states):
        if ctx.progress:
            ctx.progress("vision", index / len(states))
        spoken = _spoken(ctx.transcript, state.start, state.end)
        reply = llm.describe_board(config, state.keyframe, spoken)
        visual.append(_visual(state, reply))
    ctx.visual = tuple(visual)
    return ctx


def _spoken(transcript: Transcript | None, start: float, end: float) -> str:
    """Return the words said during ``[start, end)``; nothing without a transcript."""
    if transcript is None:
        return ""
    return " ".join(w.text for w in transcript.words if w.start < end and w.end > start)


def _visual(state: BoardState, reply: dict[str, Any]) -> VisualContext:
    """Build a board state from the model's reply, tolerating missing or odd keys."""
    try:
        confidence = min(max(float(reply.get("confidence") or 0.0), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    latex = reply.get("latex")
    return VisualContext(
        start=state.start,
        end=state.end,
        description=str(reply.get("description") or ""),
        text=str(reply.get("text") or ""),
        latex=str(latex) if latex else None,
        confidence=confidence,
    )


def overlapping(
    visual: Sequence[VisualContext], start: float, end: float
) -> tuple[VisualContext, ...]:
    """Board states whose lifetime overlaps the half-open interval ``[start, end)``."""
    return tuple(v for v in visual if v.start < end and v.end > start)


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

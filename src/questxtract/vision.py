"""Stage 5: optional enrichment of the audio with what is on the board.

Two stages, not one, because of architecture.md 11.1 rule 4: capture is expensive and caches
as ``vision.json``, while the interval join is eight lines and always recomputes. Nailing
one board state to one segment at capture time would bake in a wrong join that every
consumer inherits.

Both no-op themselves when vision is off, rather than being filtered out by the runner
(rule 3).
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import Context, Segment, VisualContext, stage


@stage("vision", cache="visual", uses=("vision_fps", "vision_model", "vision_max_states"))
def vision(ctx: Context) -> Context:
    """Capture board states from the recording into ``ctx.visual``.

    A no-op unless ``config.vision_enabled``, which is false unless an endpoint is
    configured. The layout probe, frame sampling, de-occlusion, board-state segmentation
    and VLM captioning of architecture.md 6.2 to 6.5 land with the M2/M3 milestones.
    """
    if not ctx.config.vision_enabled:
        return ctx
    # ponytail: frame sampling, keyframes and VLM captioning (architecture.md 6.2-6.5), #59.
    raise NotImplementedError("vision: board reading lands with the M2/M3 milestones")


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

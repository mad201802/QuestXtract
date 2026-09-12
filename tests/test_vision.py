import pytest

from questxtract import VisualContext
from questxtract.segments import segments as segments_stage
from questxtract.sentences import sentences as sentences_stage
from questxtract.vision import attach_visual, overlapping, vision

STATES = (
    VisualContext(start=0.0, end=5.0, description="first board"),
    VisualContext(start=5.0, end=20.0, description="second board"),
)


def test_overlapping_uses_half_open_intervals():
    assert overlapping(STATES, 1.0, 2.0) == (STATES[0],)
    assert overlapping(STATES, 4.0, 6.0) == STATES
    assert overlapping(STATES, 5.0, 6.0) == (STATES[1],)  # touching at 5.0 is not overlapping
    assert overlapping(STATES, 30.0, 40.0) == ()


def test_attach_visual_is_a_no_op_without_board_states(context):
    ctx = segments_stage(sentences_stage(context))
    before = ctx.segments
    assert attach_visual(ctx).segments is before


def test_attach_visual_joins_states_to_the_segments_they_overlap(context):
    ctx = segments_stage(sentences_stage(context))
    ctx.visual = STATES
    ctx = attach_visual(ctx)
    assert ctx.segments[0].visual == STATES  # the one segment spans 0.0-9.0
    assert ctx.stats["board_states"] == 2


def test_vision_no_ops_when_disabled(context):
    assert vision(context) is context


def test_vision_refuses_when_enabled_but_unimplemented(context):
    context.config = context.config.replace(vision_enabled=True)
    with pytest.raises(NotImplementedError, match="M2/M3"):
        vision(context)

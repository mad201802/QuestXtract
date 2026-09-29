"""The vision stage, offline: a synthetic slide video from conftest and a stub VLM.

No test here reaches a model. ``stub_vlm`` replaces ``llm.describe_board``, the stage's only
way out, and ``offline`` fails any test that opens a socket or finds an API key.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from questxtract import Config, Context, VisualContext, asr, llm, process
from questxtract.export import result_to_dict
from questxtract.segments import segments as segments_stage
from questxtract.sentences import sentences as sentences_stage
from questxtract.vision import (
    BoardState,
    _limit,
    _visual,
    attach_visual,
    board_states,
    overlapping,
    vision,
)

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


# --- the synthetic lecture ---------------------------------------------------------------


def test_synthetic_lecture_is_built_at_test_time_in_under_two_seconds(tmp_path, make_slides):
    began = time.perf_counter()
    lecture = make_slides(tmp_path / "slides.mp4")
    assert time.perf_counter() - began < 2.0
    assert lecture.path.is_file()


def test_synthetic_lecture_exposes_its_slide_changes(slides_lecture):
    assert slides_lecture.changes == (2.0, 4.0)
    assert slides_lecture.states == ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0))
    assert slides_lecture.duration == 6.0


# --- board states --------------------------------------------------------------------------


@pytest.mark.parametrize("fps", [1.0, 5.0])
def test_board_states_cut_on_the_known_slide_changes(slides_lecture, fps):
    states = board_states(slides_lecture.path, fps=fps)
    assert [(s.start, s.end) for s in states] == list(slides_lecture.states)


def test_board_states_of_a_tall_recording_are_compared_downscaled(tmp_path, make_slides):
    lecture = make_slides(tmp_path / "tall.mp4", slide_seconds=1.0, fps=2, size=(160, 544))
    assert [(s.start, s.end) for s in board_states(lecture.path)] == list(lecture.states)


def test_board_states_reject_nonpositive_settings(slides_lecture):
    with pytest.raises(ValueError, match="positive"):
        board_states(slides_lecture.path, fps=0)
    with pytest.raises(ValueError, match="positive"):
        board_states(slides_lecture.path, max_states=0)


def test_board_states_need_a_video_track(tmp_path, make_mp4):
    with pytest.raises(ValueError, match="no video track"):
        board_states(make_mp4(tmp_path / "audio_only.mp4"))


def test_capping_board_states_keeps_the_largest_changes():
    states = [
        BoardState(start=0.0, end=1.0, keyframe=b"a"),
        BoardState(start=1.0, end=2.0, keyframe=b"b", change=0.05),
        BoardState(start=2.0, end=3.0, keyframe=b"c", change=0.60),
        BoardState(start=3.0, end=4.0, keyframe=b"d", change=0.10),
    ]
    assert _limit(states, 4) is states
    merged = _limit(states, 2)
    assert [(s.start, s.end, s.keyframe) for s in merged] == [(0.0, 2.0, b"b"), (2.0, 4.0, b"d")]


def test_capped_board_states_still_cover_the_whole_recording(slides_lecture):
    (only,) = board_states(slides_lecture.path, max_states=1)
    assert (only.start, only.end) == (0.0, slides_lecture.duration)


# --- captioning ----------------------------------------------------------------------------


def test_vision_captions_every_board_state_with_the_stub(slides_lecture, stub_vlm, offline):
    ctx = Context(source=slides_lecture.path, config=Config(vision_enabled=True))
    ctx = vision(ctx)
    assert [(v.start, v.end) for v in ctx.visual] == list(slides_lecture.states)
    assert [v.description for v in ctx.visual] == [
        s.caption["description"] for s in slides_lecture.slides
    ]
    assert ctx.visual[2].latex == slides_lecture.slides[2].caption["latex"]
    assert [transcript for _, transcript in stub_vlm.calls] == ["", "", ""]  # no transcript


def test_every_keyframe_is_read_as_its_own_slide(slides_lecture, stub_vlm):
    keyframes = [s.keyframe for s in board_states(slides_lecture.path)]
    assert all(k.startswith(b"\x89PNG") for k in keyframes)
    assert [stub_vlm.slide_of(k) for k in keyframes] == list(slides_lecture.slides)


def test_vision_reports_progress_per_board_state(slides_lecture, stub_vlm, offline):
    seen: list[tuple[str, float]] = []
    ctx = Context(
        source=slides_lecture.path,
        config=Config(vision_enabled=True),
        progress=lambda name, fraction: seen.append((name, fraction)),
    )
    vision(ctx)
    assert seen == [("vision", 0.0), ("vision", 1 / 3), ("vision", 2 / 3)]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ({}, VisualContext(start=0.0, end=1.0)),
        ({"confidence": "high", "latex": ""}, VisualContext(start=0.0, end=1.0)),
        (
            {"confidence": 7, "text": 42},
            VisualContext(start=0.0, end=1.0, text="42", confidence=1.0),
        ),
        ({"confidence": -1, "latex": "x^2"}, VisualContext(start=0.0, end=1.0, latex="x^2")),
    ],
)
def test_a_malformed_caption_still_gives_a_well_formed_board_state(reply, expected):
    assert _visual(BoardState(start=0.0, end=1.0, keyframe=b""), reply) == expected


# --- STAGES end to end, vision on, against the stub ----------------------------------------

_SPOKEN = [
    {"text": "Today we cover sorting.", "start": 0.2, "end": 1.6},
    {"text": "What is a stable sort?", "start": 2.2, "end": 3.6},
    {"text": "It keeps equal elements in order.", "start": 4.2, "end": 5.8},
]


@pytest.fixture
def fake_asr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canned speech timed against the slides: one sentence per slide."""

    class _Backend:
        def transcribe(
            self,
            source: Path,
            config: Config,
            on_progress: Callable[[float], None] | None = None,
        ) -> list[dict[str, Any]]:
            del source, config, on_progress
            return _SPOKEN

    monkeypatch.setitem(asr._REGISTRY, "faster_whisper", _Backend)


def test_stages_run_end_to_end_with_vision_on_against_the_stub(
    tmp_path, slides_lecture, stub_vlm, fake_asr, offline
):
    config = Config(vision_enabled=True, cache_dir=str(tmp_path / "cache"))
    ctx = process(slides_lecture.path, config)

    # vision: one board state per slide, captioned from its own keyframe
    assert [(v.start, v.end) for v in ctx.visual] == list(slides_lecture.states)
    assert [v.text for v in ctx.visual] == [s.caption["text"] for s in slides_lecture.slides]
    assert [transcript for _, transcript in stub_vlm.calls] == [s["text"] for s in _SPOKEN]

    # attach_visual: the lone segment spans every slide
    (segment,) = ctx.segments
    assert segment.visual == ctx.visual
    assert ctx.stats["board_states"] == 3

    # export: the question points at the slide on screen while it was asked
    (question,) = result_to_dict(ctx)["questions"]
    assert question["text"] == "What is a stable sort?"
    assert question["visual_ids"] == [ctx.visual[1].id]

    # a second run replays vision.json instead of calling the model again
    again = process(slides_lecture.path, config)
    assert again.visual == ctx.visual
    assert len(stub_vlm.calls) == 3


# --- the llm.py helper the stub replaces ---------------------------------------------------


def _fake_client(monkeypatch: pytest.MonkeyPatch, content: str | None) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "client", lambda config: fake)
    return requests


def test_describe_board_sends_the_image_and_transcript_to_the_vision_model(monkeypatch):
    reply = {"description": "a slide", "text": "x", "latex": None, "confidence": 0.5}
    requests = _fake_client(monkeypatch, json.dumps(reply))

    assert llm.describe_board(Config(), b"\x89PNG", "what is x?") == reply
    (request,) = requests
    assert request["model"] == Config().vision_model
    assert request["response_format"] == {"type": "json_object"}
    text, image = request["messages"][1]["content"]
    assert "what is x?" in text["text"]
    assert image["image_url"]["url"] == "data:image/png;base64,iVBORw=="


@pytest.mark.parametrize("content", [None, "[]", '"a string"'])
def test_describe_board_refuses_a_reply_that_is_not_an_object(monkeypatch, content):
    _fake_client(monkeypatch, content)
    with pytest.raises(ValueError, match="JSON object"):
        llm.describe_board(Config(), b"")

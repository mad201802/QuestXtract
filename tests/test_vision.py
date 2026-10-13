import base64
import inspect
from pathlib import Path

import av
import numpy as np
import openai
import pytest

from questxtract import Config, Context, Transcript, VisualContext, Word, endpoint
from questxtract import vision as vision_module
from questxtract.export import result_to_dict, write_questions
from questxtract.pipeline import run
from questxtract.segments import segments as segments_stage
from questxtract.sentences import sentences as sentences_stage
from questxtract.vision import (
    _PROMPT,
    VISION_PROMPT_VERSION,
    _caption,
    _for_comparison,
    _merge_shortest,
    _sample_frames,
    _speech,
    _store,
    _View,
    _views,
    attach_visual,
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


@pytest.mark.parametrize("name", ["lecture.mp3", "audio_only.mp4", "lecture.transcript.json"])
def test_vision_has_nothing_to_read_without_a_video_track(
    context, tmp_path, make_mp3, make_mp4, name
):
    """#117 AC: an MP3, an MP4 holding only audio, or a transcript: no board, no error."""
    path = tmp_path / name
    if name.endswith(".mp3"):
        make_mp3(path)
    elif name.endswith(".mp4"):
        make_mp4(path)  # audio only
    else:
        path.write_text("{}", encoding="utf-8")
    context.source = path
    context.config = context.config.replace(vision_enabled=True)
    assert vision(context) is context and context.visual == ()


def _video(path, seconds=10, rate=25, size=(320, 240)):
    """A small MP4 whose frame brightness is its second: 0, 20, 40, ..."""
    width, height = size
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(seconds * rate):
            picture = np.full((height, width, 3), i // rate * 20, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_sampling_takes_one_frame_per_second(tmp_path):
    """#54 AC: a 10 s video at 1 fps gives 10 frames, each within 0.1 s of 0..9."""
    samples = list(_sample_frames(_video(tmp_path / "v.mp4"), 1.0))
    assert [round(t) for t, _ in samples] == list(range(10))
    assert all(abs(t - round(t)) <= 0.1 for t, _ in samples)
    assert all(frame.shape == (240, 320, 3) for _, frame in samples)  # full colour
    brightness = [int(frame.mean()) for _, frame in samples]
    assert brightness == sorted(brightness) and brightness[-1] > brightness[0]  # second by second


def test_sampling_follows_the_rate(tmp_path):
    assert len(list(_sample_frames(_video(tmp_path / "v.mp4", seconds=2), 5.0))) == 10


def test_sampling_crops_every_frame(tmp_path):
    """#54 AC: a crop box gives the cropped dimensions."""
    samples = _sample_frames(_video(tmp_path / "v.mp4", seconds=2), 1.0, crop=(0.25, 0, 0.75, 0.5))
    assert {frame.shape for _, frame in samples} == {(120, 160, 3)}


def test_sampling_streams(tmp_path):
    """#54 AC: frames come one at a time, so memory stays flat on a long recording."""
    samples = _sample_frames(_video(tmp_path / "v.mp4"), 1.0)
    assert inspect.isgenerator(samples)
    next(samples)
    samples.close()  # releases the file; on Windows the temp dir could not be removed otherwise


def test_sampling_reports_progress(tmp_path):
    seen: list[tuple[str, float]] = []
    list(_sample_frames(_video(tmp_path / "v.mp4"), 1.0, progress=lambda s, f: seen.append((s, f))))
    assert len(seen) == 10 and {s for s, _ in seen} == {"vision"}
    assert [f for _, f in seen] == sorted(f for _, f in seen) and 0.8 <= seen[-1][1] <= 1.0


@pytest.mark.parametrize(
    ("fps", "crop"), [(0.0, None), (1.0, (0.5, 0, 0.5, 1)), (1.0, (0, 0, 1.2, 1))]
)
def test_sampling_rejects_bad_settings(tmp_path, fps, crop):
    with pytest.raises(ValueError):
        next(_sample_frames(_video(tmp_path / "v.mp4", seconds=1), fps, crop=crop))


def test_sampling_rejects_an_audio_only_recording(tmp_path, make_mp4):
    with pytest.raises(ValueError, match="no video"):
        next(_sample_frames(make_mp4(tmp_path / "a.mp4"), 1.0))


def test_comparison_frames_are_gray_and_about_480p():
    assert _for_comparison(np.zeros((1080, 1920, 3), np.uint8)).shape == (540, 960)
    assert _for_comparison(np.zeros((240, 320, 3), np.uint8)).shape == (240, 320)  # never upscaled
    white = _for_comparison(np.full((960, 1280, 3), 255, np.uint8))
    assert white.shape == (480, 640) and white.dtype == np.uint8 and int(white.min()) == 255


def _slide(index, size=(240, 320)):
    """A light slide with a dark block whose place says which slide it is."""
    picture = np.full((*size, 3), 200, dtype=np.uint8)
    top = 20 + 60 * index
    picture[top : top + 40, 40:280] = 30
    return picture


def _slides_with_a_walker(path, rate=25):
    """12 s: slides 0, 1, 2 from 0, 4 and 8 s; someone walks across slide 0 from 1 to 3.5 s."""
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for i in range(12 * rate):
            second = i / rate
            picture = _slide(int(second // 4))
            if 1.0 <= second < 3.5:
                left = int((second - 1.0) * 100)  # 100 px a second, 60 px wide
                picture[30:230, left : left + 60] = 10
            for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _cut(samples, change=0.02, hold=3, max_views=80):
    return _views(samples, change=change, hold=hold, max_views=max_views)


def test_one_view_per_slide_and_a_walker_starts_none(tmp_path):
    """#56 AC: one keyframe per slide, each change within 1 s; a person walking past a static
    slide adds no view."""
    views = _cut(_sample_frames(_slides_with_a_walker(tmp_path / "v.mp4"), 1.0))
    assert [round(v.start) for v in views] == [0, 4, 8]
    assert [v.end for v in views[:-1]] == [v.start for v in views[1:]]
    assert views[-1].end == pytest.approx(12.0, abs=0.1)
    assert all(v.keyframe[1:4] == b"PNG" for v in views)
    assert not 1.0 <= views[0].keyframe_time < 3.5  # never the frame with the walker in it


def test_more_views_than_the_cap_are_merged_shortest_first():
    """#56 AC: at most max_views come out, covering the whole recording."""
    lengths = [3, 9, 4, 6, 3, 5]  # seconds per slide, sampled at 1 fps
    samples, time = [], 0
    for index, length in enumerate(lengths):
        samples += [(float(time + k), _slide(index % 3)) for k in range(length)]
        time += length
    views = _cut(samples, max_views=3)
    assert len(views) == 3
    assert views[0].start == 0.0 and views[-1].end == 30.0
    assert [v.end for v in views[:-1]] == [v.start for v in views[1:]]


def test_a_merged_view_keeps_the_longer_keyframe():
    short = _View(start=0.0, end=2.0, keyframe=b"short", keyframe_time=1.0)
    long = _View(start=2.0, end=10.0, keyframe=b"long", keyframe_time=5.0)
    (merged,) = _merge_shortest([short, long], 1)
    assert (merged.start, merged.end, merged.keyframe) == (0.0, 10.0, b"long")


def test_a_change_must_hold_to_start_a_view():
    """Two odd samples are a flicker, not a slide: with hold 3 they start nothing."""
    samples = [(float(t), _slide(1 if t in (3, 4) else 0)) for t in range(10)]
    assert len(_cut(samples)) == 1
    assert len(_cut(samples, hold=2)) == 3


def test_no_samples_no_views():
    assert _cut([]) == []


@pytest.mark.parametrize(
    ("change", "hold", "max_views"), [(0.0, 3, 80), (1.0, 3, 80), (0.02, 0, 80), (0.02, 3, 0)]
)
def test_views_reject_bad_settings(change, hold, max_views):
    with pytest.raises(ValueError):
        _cut([], change=change, hold=hold, max_views=max_views)


def test_the_last_view_ends_at_the_recording_end_and_is_never_empty():
    """A single sample still gets a view with a length; the caller's end time wins."""
    (only,) = _cut([(0.0, _slide(0))])
    assert (only.start, only.end) == (0.0, 1.0)
    samples = [(float(t), _slide(0)) for t in range(5)]
    assert _cut(samples)[-1].end == 5.0  # one sample interval after the last sample
    assert _views(samples, change=0.02, hold=3, max_views=80, end=5.7)[-1].end == 5.7


# --- the vision call (#53) ---

PNG = b"\x89PNG keyframe"  # only sent, never decoded


def _board(**fields):
    return {
        "description": "A slide on sorting.",
        "text": "Stable sort",
        "latex": None,
        "confidence": 0.8,
        **fields,
    }


@pytest.mark.parametrize(("latex", "kept"), [(" x^2 ", "x^2"), ("", None), (None, None)])
def test_a_keyframe_is_read_into_board_fields(fake_llm, tmp_path, latex, kept):
    """AC: valid JSON gives the VisualContext fields; the request carries the model, the
    prompt with the speech, and the image as a PNG data URL."""
    client = fake_llm(_board(description=" A slide. ", latex=latex, confidence=1.4))
    config = Config(cache_dir=str(tmp_path), vision_model="qwen3-vl")
    fields = _caption(config, PNG, " so we sort ")
    assert fields == {
        "description": "A slide.",
        "text": "Stable sort",
        "latex": kept,
        "confidence": 1.0,
    }
    assert VisualContext(start=0.0, end=5.0, **fields).latex == kept
    (request,) = client.requests
    assert request["model"] == "qwen3-vl"
    assert request["response_format"]["json_schema"]["strict"] is True
    system, user = request["messages"]
    # As prompt v0 was measured (#52): the filled-in prompt as system, the image alone as user.
    assert system["content"] == _PROMPT.replace("{kind}", "unknown").replace(
        "{speech}", "so we sort"
    )
    assert user["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()},
        },
    ]


@pytest.mark.parametrize(
    "replies",
    [
        (openai.OpenAIError("Connection error."),),
        (TimeoutError("no answer within 120 s"),),  # a stall, twice
        ("not json", "[]"),
        (_board(text=None),),
        (_board(latex=3),),
        (_board(confidence="high"),),
        (_board(confidence=True),),
        (_board(confidence=float("nan")),),
    ],
)
def test_a_failed_reading_leaves_the_keyframe_out_with_a_warning(fake_llm, tmp_path, replies):
    """AC: vision is never required, so no failure escapes; the run goes on without it."""
    fake_llm(*replies)
    with pytest.warns(UserWarning, match="vision: keyframe left out") as caught:
        assert _caption(Config(cache_dir=str(tmp_path)), PNG, "") is None
    assert len(caught) == 1


def test_a_keyframe_read_once_is_not_sent_again(fake_llm, tmp_path):
    """AC: cached on the request, image included; a failed call is not cached."""
    config = Config(cache_dir=str(tmp_path))
    client = fake_llm(openai.OpenAIError("Connection error."), _board())
    with pytest.warns(UserWarning):
        assert _caption(config, PNG, "") is None
    assert _caption(config, PNG, "") == _caption(config, PNG, "") is not None
    assert len(client.requests) == 2
    assert len(list((tmp_path / "vision").glob("*.json"))) == 1
    _caption(config, PNG + b"another board", "")
    assert len(client.requests) == 3


def test_without_the_llm_extra_the_install_hint_is_raised(monkeypatch, tmp_path):
    def missing(config):
        raise RuntimeError("the LLM layer needs `pip install questxtract[llm]`")

    monkeypatch.setattr(endpoint, "client", missing)
    with pytest.raises(RuntimeError, match=r"questxtract\[llm\]"):
        _caption(Config(cache_dir=str(tmp_path)), PNG, "")


def test_a_setting_that_cannot_be_used_is_raised_not_dropped(monkeypatch, tmp_path, recwarn):
    """An unreadable .env is a user error that would drop every keyframe the same way, so it
    is not turned into a warning per keyframe; only what the endpoint or model did is."""

    def unusable(config):
        raise ValueError(".env: line 1: expansion is not supported")

    monkeypatch.setattr(endpoint, "client", unusable)
    with pytest.raises(ValueError, match="expansion is not supported"):
        _caption(Config(cache_dir=str(tmp_path)), PNG, "")
    assert not recwarn.list


def test_a_cache_that_cannot_be_written_still_returns_the_answer(fake_llm, tmp_path, recwarn):
    """The answer is already paid for: a read-only or full cache only costs a repeat request."""
    blocker = tmp_path / "cache"
    blocker.write_text("a file where the cache folder should go", encoding="utf-8")
    client = fake_llm(_board())
    fields = _caption(Config(cache_dir=str(blocker)), PNG, "")
    assert fields is not None and fields["text"] == "Stable sort"
    assert len(client.requests) == 1
    assert not recwarn.list


def test_a_cache_entry_that_is_not_text_is_asked_again(fake_llm, tmp_path):
    config = Config(cache_dir=str(tmp_path))
    client = fake_llm(_board())
    assert _caption(config, PNG, "") is not None
    (entry,) = (tmp_path / "vision").glob("*.json")
    entry.write_bytes(b"\xff\xfe\x00 not utf-8")
    assert _caption(config, PNG, "") is not None
    assert len(client.requests) == 2
    assert entry.read_text(encoding="utf-8").startswith("{")  # replaced by the new answer


def test_the_prompt_is_prompt_v0():
    """The prompt chosen in #52, word for word; a change needs a new version and file."""
    assert VISION_PROMPT_VERSION == "v0"
    v0 = Path(__file__).parent / "data" / "vision" / "prompt_v0.txt"
    assert _PROMPT == v0.read_text(encoding="utf-8").rstrip("\n")


# --- the vision stage (#59) ---


def _slides(path, count, seconds=3, rate=10):
    """``count`` slides, ``seconds`` each, nothing moving."""
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for i in range(count * seconds * rate):
            picture = _slide(i // (seconds * rate))
            for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _lecture(context, video, cache, **config):
    """A cacheable context for ``video``, with the fixture's 9 s of speech and vision on."""
    return Context(
        source=video,
        source_hash="abc123",
        transcript=context.transcript,
        duration=12.0,
        config=Config(cache_dir=str(cache), **{"vision_enabled": True, **config}),
    )


VISION_STAGES = (sentences_stage, segments_stage, vision, attach_visual)


def test_vision_reads_one_board_state_per_slide(context, fake_llm, tmp_path):
    """#59 AC: a slides video gives one state per slide with its time on screen; each keyframe
    is in the cache under a path relative to it, and the export links states to segments."""
    client = fake_llm(_board())
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    ctx = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)

    assert [(round(s.start), round(s.end)) for s in ctx.visual] == [(0, 4), (4, 8), (8, 12)]
    assert {s.text for s in ctx.visual} == {"Stable sort"}
    assert len(client.requests) == 3
    for state in ctx.visual:
        assert state.keyframe_path.startswith("abc123/keyframes/")
        assert (tmp_path / "cache" / state.keyframe_path).read_bytes()[1:4] == b"PNG"
    first = client.requests[0]["messages"][0]["content"]
    assert "Today we cover sorting. What is a stable sort?" in first  # said during slide 0
    assert "keeps equal elements" not in first  # said during slide 1

    assert result_to_dict(ctx)["segments"][0]["visual_ids"] == [s.id for s in ctx.visual]
    write_questions(ctx, tmp_path / "out" / "lecture.questions.json")
    assert len(list((tmp_path / "out" / "lecture.keyframes").iterdir())) == 3


def test_a_second_run_sends_nothing(context, fake_llm, tmp_path):
    """#59 AC: a re-run with the same settings reads the stage cache: zero calls."""
    client = fake_llm(_board())
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    first = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    again = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert again.visual == first.visual
    assert len(client.requests) == 3


def test_turning_vision_on_after_a_run_without_it_reads_the_board(context, fake_llm, tmp_path):
    """The empty result of a run with vision off is not reused once vision is on."""
    client = fake_llm(_board())
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    off = run(_lecture(context, video, tmp_path / "cache", vision_enabled=False), VISION_STAGES)
    assert off.visual == () and not client.requests
    on = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert len(on.visual) == 3


def test_readings_that_failed_are_asked_again_on_the_next_run(context, fake_llm, tmp_path):
    """Jesper on PR #128: a run with the endpoint down must not cache an empty result."""
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    fake_llm(openai.OpenAIError("Connection error."))
    with pytest.warns(UserWarning):
        down = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert down.visual == () and down.stats["vision_failed"] == 3
    client = fake_llm(_board())
    up = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert len(up.visual) == 3 and up.stats["vision_failed"] == 0
    assert len(client.requests) == 3


def test_a_rejected_answer_is_not_cached(fake_llm, tmp_path):
    """A malformed answer counts as invalid JSON: asked once more, never replayed."""
    config = Config(cache_dir=str(tmp_path))
    fake_llm(_board(confidence="high"))
    with pytest.warns(UserWarning, match="keyframe left out"):
        assert _caption(config, PNG, "") is None
    assert not list((tmp_path / "vision").glob("*.json"))
    client = fake_llm(_board())
    assert _caption(config, PNG, "") is not None
    assert len(client.requests) == 1


def test_a_new_transcript_reads_the_board_again(context, fake_llm, tmp_path):
    """The speech is part of every request, so other speech is another reading."""
    client = fake_llm(_board())
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    other = _lecture(context, video, tmp_path / "cache")
    other.transcript = Transcript(
        words=(Word(text="Something", start=0.5, end=1.0), Word(text="else.", start=1.0, end=1.5)),
        duration=12.0,
    )
    run(other, VISION_STAGES)
    assert len(client.requests) == 6  # every slide had other speech, so all three again


def test_the_views_are_cached_so_a_second_run_samples_nothing(
    context, fake_llm, tmp_path, monkeypatch
):
    fake_llm(_board())
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    first = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)

    def no_sampling(*args, **kwargs):
        raise AssertionError("sampled again")

    monkeypatch.setattr(vision_module, "_sample_frames", no_sampling)
    assert run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES).visual == first.visual
    changed = _lecture(context, video, tmp_path / "cache", vision_change_hold=2)
    with pytest.raises(AssertionError, match="sampled again"):
        run(changed, VISION_STAGES)  # other sampling settings cut again


def _offset_lecture(path, offset=10.0, rate=10):
    """4 s of one slide whose video and audio streams both start at ``offset`` seconds."""
    with av.open(str(path), "w", format="mp4") as container:
        video = container.add_stream("mpeg4", rate=rate)
        video.width, video.height, video.pix_fmt = 320, 240, "yuv420p"
        audio = container.add_stream("aac", rate=8000)
        audio.layout = "mono"
        for i in range(4 * rate):
            frame = av.VideoFrame.from_ndarray(_slide(0), format="rgb24")
            frame.pts = int(offset * rate) + i
            for packet in video.encode(frame):
                container.mux(packet)
        for i in range(32):
            sound = av.AudioFrame(format="fltp", layout="mono", samples=1024)
            sound.sample_rate = 8000
            sound.pts = int(offset * 8000) + i * 1024
            sound.planes[0].update(np.zeros(1024, np.float32).tobytes())
            for packet in audio.encode(sound):
                container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode():
                container.mux(packet)
    return path


def test_board_states_share_the_transcript_clock(context, fake_llm, tmp_path):
    """Jesper on PR #128: frame times count from the audio start, like the word times."""
    client = fake_llm(_board())
    video = _offset_lecture(tmp_path / "trimmed.mp4")
    with av.open(str(video)) as container:  # the file really starts late
        stream = container.streams.video[0]
        assert stream.start_time * stream.time_base > 9
    ctx = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    (state,) = ctx.visual
    assert state.start < 1.0
    assert ctx.segments[0].visual == (state,)
    assert "Today we cover sorting." in client.requests[0]["messages"][0]["content"]


def test_a_failed_reading_leaves_only_its_slide_out(context, fake_llm, tmp_path):
    client = fake_llm(_board(), openai.OpenAIError("Connection error."), _board())
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    with pytest.warns(UserWarning, match="keyframe left out"):
        ctx = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert [round(s.start) for s in ctx.visual] == [0, 8]
    assert ctx.stats["vision_failed"] == 1
    assert len(client.requests) == 3


def test_when_the_first_readings_all_fail_the_rest_are_not_sent(context, fake_llm, tmp_path):
    """Milos on #59: a wrong model name or key would otherwise pay for every keyframe."""
    client = fake_llm(openai.OpenAIError("model not found"))
    video = _slides(tmp_path / "lecture.mp4", count=4)
    with pytest.warns(UserWarning) as caught:
        ctx = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert ctx.visual == () and ctx.stats["vision_failed"] == 4  # three failed, one not sent
    assert len(client.requests) == 3
    assert "the first 3 keyframes could not be read" in str(caught[-1].message)


def test_failures_after_a_reading_do_not_stop_the_rest(context, fake_llm, tmp_path):
    client = fake_llm(_board(), openai.OpenAIError("Connection error."))
    video = _slides(tmp_path / "lecture.mp4", count=4)
    with pytest.warns(UserWarning, match="keyframe left out"):
        ctx = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert len(ctx.visual) == 1
    assert len(client.requests) == 4


def test_vision_reports_its_progress(context, fake_llm, tmp_path):
    fake_llm(_board())
    ctx = _lecture(context, _slides_with_a_walker(tmp_path / "lecture.mp4"), tmp_path / "cache")
    seen = []
    run(ctx, VISION_STAGES, progress=lambda name, fraction: seen.append((name, fraction)))
    fractions = [f for name, f in seen if name == "vision"]
    assert fractions == sorted(fractions) and 0.0 <= fractions[0] and fractions[-1] <= 1.0
    assert any(f > 0.5 for f in fractions)  # the reading half, not only the sampling


def test_the_speech_for_a_view_is_what_overlaps_it_cut_at_a_word(sentences):
    assert _speech(sentences, 0.0, 4.0) == "Today we cover sorting. What is a stable sort?"
    assert _speech(sentences, 20.0, 30.0) == ""
    long = [type(sentences[0])(text="word " * 400, word_start=0, word_end=1, start=0, end=1)]
    cut = _speech(long, 0.0, 1.0)
    assert cut.endswith("word ...") and len(cut) <= 1504


def test_a_keyframe_is_stored_once_by_its_content(tmp_path):
    config = Config(cache_dir=str(tmp_path))
    path = _store(PNG, config, "abc123")
    assert path == _store(PNG, config, "abc123") != _store(PNG + b"other", config, "abc123")
    assert (tmp_path / path).read_bytes() == PNG
    (tmp_path / "blocked").write_text("a file where the cache should be", encoding="utf-8")
    assert _store(PNG, Config(cache_dir=str(tmp_path / "blocked")), "abc123") == ""


# --- Jesper's review of PR #131 ---


def test_an_answer_without_latex_is_rejected_not_cached(fake_llm, tmp_path):
    """A missing key breaks the schema: left out with a warning, never a crash or a cache hit."""
    config = Config(cache_dir=str(tmp_path))
    answer = _board()
    del answer["latex"]
    fake_llm(answer)
    with pytest.warns(UserWarning, match="keyframe left out"):
        assert _caption(config, PNG, "") is None
    assert not list((tmp_path / "vision").glob("*.json"))


def _late_audio_lecture(path, audio_start=5.0, rate=10):
    """10 s of video from 0 s: a title card until 1 s, then a slide; audio from ``audio_start``."""
    with av.open(str(path), "w", format="mp4") as container:
        video = container.add_stream("mpeg4", rate=rate)
        video.width, video.height, video.pix_fmt = 320, 240, "yuv420p"
        audio = container.add_stream("aac", rate=8000)
        audio.layout = "mono"
        for i in range(10 * rate):
            frame = av.VideoFrame.from_ndarray(_slide(0 if i < rate else 2), format="rgb24")
            frame.pts = i
            for packet in video.encode(frame):
                container.mux(packet)
        for i in range(40):
            sound = av.AudioFrame(format="fltp", layout="mono", samples=1024)
            sound.sample_rate = 8000
            sound.pts = int(audio_start * 8000) + i * 1024
            sound.planes[0].update(np.zeros(1024, np.float32).tobytes())
            for packet in audio.encode(sound):
                container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode():
                container.mux(packet)
    return path


def test_what_was_on_screen_before_the_audio_makes_no_empty_view(context, fake_llm, tmp_path):
    """Jesper on PR #131: a title card before the audio starts must not leave a 0-0 view."""
    client = fake_llm(_board())
    video = _late_audio_lecture(tmp_path / "late.mp4")
    ctx = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    (state,) = ctx.visual  # only the slide on screen once the audio runs
    assert state.start == 0.0 and state.end > state.start
    assert ctx.segments[0].visual == (state,)
    assert len(client.requests) == 1


def test_a_damaged_keyframe_in_the_cache_is_sampled_again(context, fake_llm, tmp_path):
    """Jesper on PR #131: a cut-short PNG is not sent to the model; it is replaced."""
    fake_llm(_board())
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    first = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    damaged = tmp_path / "cache" / first.visual[0].keyframe_path
    good = damaged.read_bytes()
    damaged.write_bytes(good[:8])

    client = fake_llm(_board())
    again = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert again.visual == first.visual and not client.requests  # same images, so cache hits
    assert damaged.read_bytes() == good

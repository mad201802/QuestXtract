import base64
import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import openai
import pytest

from questxtract import Config, Context, Layout, Transcript, VisualContext, Word, endpoint
from questxtract import vision as vision_module
from questxtract.export import result_to_dict, write_questions
from questxtract.pipeline import run
from questxtract.segments import segments as segments_stage
from questxtract.sentences import sentences as sentences_stage
from questxtract.vision import (
    _LAYOUT_PROMPT,
    _PROMPT,
    VISION_PROMPT_VERSION,
    _changed,
    _decoded,
    _for_comparison,
    _merge_shortest,
    _png,
    _Reading,
    _same_page,
    _sample_frames,
    _speech,
    _store,
    _View,
    _views,
    attach_visual,
    layout,
    read_keyframe,
    vision,
)

STATES = (
    VisualContext(start=0.0, end=5.0, description="first board"),
    VisualContext(start=5.0, end=20.0, description="second board"),
)


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
    fields = read_keyframe(config, PNG, " so we sort ")
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
        assert read_keyframe(Config(cache_dir=str(tmp_path)), PNG, "") is None
    assert len(caught) == 1


def test_a_keyframe_read_once_is_not_sent_again(fake_llm, tmp_path):
    """AC: cached on the request, image included; a failed call is not cached."""
    config = Config(cache_dir=str(tmp_path))
    client = fake_llm(openai.OpenAIError("Connection error."), _board())
    with pytest.warns(UserWarning):
        assert read_keyframe(config, PNG, "") is None
    assert read_keyframe(config, PNG, "") == read_keyframe(config, PNG, "") is not None
    assert len(client.requests) == 2
    assert len(list((tmp_path / "vision").glob("*.json"))) == 1
    read_keyframe(config, PNG + b"another board", "")
    assert len(client.requests) == 3


def test_without_the_llm_extra_the_install_hint_is_raised(monkeypatch, tmp_path):
    def missing(config, **_):
        raise RuntimeError("the LLM layer needs `pip install questxtract[llm]`")

    monkeypatch.setattr(endpoint, "client", missing)
    with pytest.raises(RuntimeError, match=r"questxtract\[llm\]"):
        read_keyframe(Config(cache_dir=str(tmp_path)), PNG, "")


def test_a_setting_that_cannot_be_used_is_raised_not_dropped(monkeypatch, tmp_path, recwarn):
    """An unreadable .env is a user error that would drop every keyframe the same way, so it
    is not turned into a warning per keyframe; only what the endpoint or model did is."""

    def unusable(config, **_):
        raise ValueError(".env: line 1: expansion is not supported")

    monkeypatch.setattr(endpoint, "client", unusable)
    with pytest.raises(ValueError, match="expansion is not supported"):
        read_keyframe(Config(cache_dir=str(tmp_path)), PNG, "")
    assert not recwarn.list


def test_a_cache_that_cannot_be_written_still_returns_the_answer(fake_llm, tmp_path, recwarn):
    """The answer is already paid for: a read-only or full cache only costs a repeat request."""
    blocker = tmp_path / "cache"
    blocker.write_text("a file where the cache folder should go", encoding="utf-8")
    client = fake_llm(_board())
    fields = read_keyframe(Config(cache_dir=str(blocker)), PNG, "")
    assert fields is not None and fields["text"] == "Stable sort"
    assert len(client.requests) == 1
    assert not recwarn.list


def test_a_cache_entry_that_is_not_text_is_asked_again(fake_llm, tmp_path):
    config = Config(cache_dir=str(tmp_path))
    client = fake_llm(_board())
    assert read_keyframe(config, PNG, "") is not None
    (entry,) = (tmp_path / "vision").glob("*.json")
    entry.write_bytes(b"\xff\xfe\x00 not utf-8")
    assert read_keyframe(config, PNG, "") is not None
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
        assert read_keyframe(config, PNG, "") is None
    assert not list((tmp_path / "vision").glob("*.json"))
    client = fake_llm(_board())
    assert read_keyframe(config, PNG, "") is not None
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
    cropped = _lecture(context, video, tmp_path / "cache")
    cropped.layout = Layout(content_region=(0.0, 0.0, 0.5, 1.0), kind="slides")
    with pytest.raises(AssertionError, match="sampled again"):
        run(cropped, VISION_STAGES)  # another crop cuts again


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


# --- the layout probe (#57) ---

LAYOUT_STAGES = (sentences_stage, segments_stage, layout, vision, attach_visual)


def _probe(**fields):
    return {"content_region": [0.5, 0.0, 1.0, 0.5], "kind": "slides", **fields}


def test_the_probe_sends_three_frames_from_across_the_recording(context, fake_llm, tmp_path):
    """One request: the prompt, and frames at 20, 50 and 80 % of the recording, small."""
    client = fake_llm(_probe())
    video = _video(tmp_path / "v.mp4", size=(800, 600))  # brightness 20 per second
    ctx = layout(_lecture(context, video, tmp_path / "cache", vision_model="qwen3-vl"))
    assert ctx.layout == Layout(content_region=(0.48, 0.0, 1.0, 0.52), kind="slides")
    (request,) = client.requests
    assert request["model"] == "qwen3-vl"
    assert request["response_format"]["json_schema"]["strict"] is True
    system, user = request["messages"]
    assert system["content"] == _LAYOUT_PROMPT
    brightness = []
    for part in user["content"]:
        png = base64.b64decode(part["image_url"]["url"].removeprefix("data:image/png;base64,"))
        (tmp_path / "f.png").write_bytes(png)
        with av.open(str(tmp_path / "f.png")) as image:
            picture = next(image.decode(video=0)).to_ndarray(format="rgb24")
        assert picture.shape == (480, 640, 3)
        brightness.append(round(float(picture.mean()) / 20))
    assert brightness == [2, 5, 8]


@pytest.mark.parametrize(
    "replies",
    [
        (openai.OpenAIError("Connection error."),),
        (TimeoutError("no answer within 120 s"),),  # a stall, twice
        ("not json", "[]"),  # bad JSON, twice
        (_probe(kind="chalk"),),
        (_probe(kind=None),),
        (_probe(content_region=[0.1, 0.1, 0.9]),),
        (_probe(content_region="the left half"),),
        (_probe(content_region=[0.6, 0.0, 0.4, 1.0]),),  # x0 > x1
        (_probe(content_region=[0, 0, 500, 1000]),),  # pixels, not fractions
        (_probe(content_region=[0, 0, True, 1]),),
        (_probe(content_region=[0, 0, float("nan"), 1]),),
    ],
)
def test_a_failed_probe_falls_back_to_the_full_frame(context, fake_llm, tmp_path, replies):
    """#57 AC: when the probe fails or returns bad JSON, the full frame and unknown are used."""
    fake_llm(*replies)
    ctx = _lecture(context, _video(tmp_path / "v.mp4", seconds=2), tmp_path / "cache")
    with pytest.warns(UserWarning, match="layout probe failed, using the full frame") as caught:
        ctx = layout(ctx)
    assert ctx.layout == Layout(content_region=(0.0, 0.0, 1.0, 1.0), kind="unknown")
    assert len(caught) == 1


def test_after_a_failed_probe_the_run_continues_on_the_full_frame(context, fake_llm, tmp_path):
    """#57 AC: the run goes on: board states are still read, uncropped and of kind unknown."""
    client = fake_llm("not json", "not json", _board())
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    with pytest.warns(UserWarning, match="layout probe failed"):
        ctx = run(_lecture(context, video, tmp_path / "cache"), LAYOUT_STAGES)
    assert len(ctx.visual) == 3
    assert all("Frame kind: unknown" in r["messages"][0]["content"] for r in client.requests[2:])
    assert _png_size(tmp_path / "cache" / ctx.visual[0].keyframe_path) == (320, 240)


def test_the_probe_crops_the_frames_and_names_the_kind(context, fake_llm, tmp_path):
    """#57: the crop box goes to the sampler, the kind to the captioning prompt."""
    client = fake_llm(_probe(content_region=[0.0, 0.0, 0.48, 0.48], kind="blackboard"), _board())
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    ctx = run(_lecture(context, video, tmp_path / "cache"), LAYOUT_STAGES)
    assert ctx.layout == Layout(content_region=(0.0, 0.0, 0.5, 0.5), kind="blackboard")
    assert ctx.visual and all(
        "Frame kind: blackboard" in r["messages"][0]["content"] for r in client.requests[1:]
    )
    assert _png_size(tmp_path / "cache" / ctx.visual[0].keyframe_path) == (160, 120)


def test_a_cached_probe_is_not_asked_again(context, fake_llm, tmp_path):
    """#57 AC: given a cached probe result, a second run makes no probe call."""
    client = fake_llm(_probe())
    video = _video(tmp_path / "v.mp4", seconds=2)
    first = layout(_lecture(context, video, tmp_path / "cache"))
    again = layout(_lecture(context, video, tmp_path / "cache"))
    assert again.layout == first.layout != Layout()
    assert len(client.requests) == 1
    assert len(list((tmp_path / "cache" / "layout").glob("*.json"))) == 1


def test_a_failed_probe_is_asked_again_on_the_next_run(context, fake_llm, tmp_path):
    client = fake_llm(openai.OpenAIError("Connection error."), _probe())
    video = _video(tmp_path / "v.mp4", seconds=2)
    with pytest.warns(UserWarning, match="layout probe failed"):
        assert layout(_lecture(context, video, tmp_path / "cache")).layout == Layout()
    assert layout(_lecture(context, video, tmp_path / "cache")).layout.kind == "slides"
    assert len(client.requests) == 2


def test_a_box_too_small_to_trust_keeps_the_kind_but_not_the_crop(context, fake_llm, tmp_path):
    fake_llm(_probe(content_region=[0.4, 0.4, 0.6, 0.6], kind="whiteboard"))
    ctx = layout(_lecture(context, _video(tmp_path / "v.mp4", seconds=2), tmp_path / "cache"))
    assert ctx.layout == Layout(kind="whiteboard")


def test_the_padding_stays_inside_the_frame(context, fake_llm, tmp_path):
    fake_llm(_probe(content_region=[0, 0, 1, 1], kind="screencast"))
    ctx = layout(_lecture(context, _video(tmp_path / "v.mp4", seconds=2), tmp_path / "cache"))
    assert ctx.layout == Layout(kind="screencast")


def test_the_probe_no_ops_when_vision_is_off_or_there_is_no_video(
    context, fake_llm, tmp_path, make_mp3
):
    """#57: the full frame and unknown when vision is off, and no call."""
    client = fake_llm(_probe())
    video = _video(tmp_path / "v.mp4", seconds=2)
    off = _lecture(context, video, tmp_path / "cache", vision_enabled=False)
    assert layout(off).layout == Layout()
    audio = _lecture(context, make_mp3(tmp_path / "a.mp3"), tmp_path / "cache")
    assert layout(audio).layout == Layout()
    assert not client.requests


def test_a_recording_that_cannot_be_decoded_falls_back(context, fake_llm, tmp_path, monkeypatch):
    client = fake_llm(_probe())

    def broken(path):
        raise OSError("read error")

    monkeypatch.setattr("questxtract.vision._probe_frames", broken)
    ctx = _lecture(context, _video(tmp_path / "v.mp4", seconds=2), tmp_path / "cache")
    with pytest.warns(UserWarning, match="could not be read"):
        assert layout(ctx).layout == Layout()
    monkeypatch.setattr("questxtract.vision._probe_frames", lambda path: [])
    with pytest.warns(UserWarning, match="no frame"):
        assert layout(ctx).layout == Layout()
    assert not client.requests


def test_a_setting_that_cannot_be_used_stops_the_probe(context, monkeypatch, tmp_path):
    """As for captioning: an unreadable .env is a user error, raised, not a fallback."""

    def unusable(config, **_):
        raise ValueError(".env: line 1: expansion is not supported")

    monkeypatch.setattr(endpoint, "client", unusable)
    ctx = _lecture(context, _video(tmp_path / "v.mp4", seconds=2), tmp_path / "cache")
    with pytest.raises(ValueError, match="expansion is not supported"):
        layout(ctx)


def _png_size(path):
    with av.open(str(path)) as image:
        frame = next(image.decode(video=0))
        return frame.width, frame.height


# --- PR #131 ---


def test_an_answer_without_latex_is_rejected_not_cached(fake_llm, tmp_path):
    """A missing key breaks the schema: left out with a warning, never a crash or a cache hit."""
    config = Config(cache_dir=str(tmp_path))
    answer = _board()
    del answer["latex"]
    fake_llm(answer)
    with pytest.warns(UserWarning, match="keyframe left out"):
        assert read_keyframe(config, PNG, "") is None
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


# --- what the vision calls cost (#123) ---


def test_stats_record_what_the_readings_cost(context, fake_llm, tmp_path):
    """#123: requests, tokens and time per caller; a cached re-run reads as no calls."""
    client = fake_llm(_board())
    client.usage = SimpleNamespace(prompt_tokens=700, completion_tokens=150)
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    ctx = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert ctx.stats["vision_requests"] == 3
    assert (ctx.stats["vision_input_tokens"], ctx.stats["vision_output_tokens"]) == (2100, 450)
    assert ctx.stats["vision_ms"] >= 0

    again = run(_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    cost = ("vision_requests", "vision_input_tokens", "vision_output_tokens", "vision_ms")
    assert [again.stats[k] for k in cost] == [0, 0, 0, 0]


def test_the_layout_probe_records_its_cost_even_when_it_fails(context, fake_llm, tmp_path):
    client = fake_llm(openai.OpenAIError("Connection error."))
    client.usage = SimpleNamespace(prompt_tokens=2000, completion_tokens=40)
    video = _slides_with_a_walker(tmp_path / "lecture.mp4")
    with pytest.warns(UserWarning, match="layout probe failed"):
        ctx = layout(_lecture(context, video, tmp_path / "cache"))
    assert ctx.stats["layout_requests"] == 1  # sent, though nothing came back
    assert ctx.stats["layout_input_tokens"] == 0
    client = fake_llm(_probe())
    client.usage = SimpleNamespace(prompt_tokens=2000, completion_tokens=40)
    ctx = layout(_lecture(context, video, tmp_path / "cache"))
    assert (ctx.stats["layout_requests"], ctx.stats["layout_input_tokens"]) == (1, 2000)


# --- reusing a reading when a slide comes back (#138) ---


def _glyph(page, x, y, segments):
    """Draw a seven-segment digit, 24 x 44 px, with segments named a to g."""
    bars = {
        "a": (y, y + 5, x, x + 24),
        "b": (y, y + 24, x + 19, x + 24),
        "c": (y + 20, y + 44, x + 19, x + 24),
        "d": (y + 39, y + 44, x, x + 24),
        "e": (y + 20, y + 44, x, x + 5),
        "f": (y, y + 24, x, x + 5),
        "g": (y + 20, y + 25, x, x + 24),
    }
    for name in segments:
        y0, y1, x0, x1 = bars[name]
        page[y0:y1, x0:x1] = 40


def _page(*, digit="3", point=0, sign="<", note=0, colour=(200, 40, 40)):
    """A 640 x 360 page of text-like lines with the small details a reuse must not miss.

    The details have the sizes they have on a 720p page, since the detail check works in
    pixels; the page is smaller only to keep the tests fast. ``point`` is the side of a decimal
    point in px, off the 4 px grid; ``note`` draws a handwritten stroke this many grey levels
    darker than the page; ``colour`` keeps the grey level of the default red when it is
    (0, 130, 100).
    """
    page = np.full((360, 640, 3), 240, np.uint8)
    page[20:50, 40:600] = 30  # the title
    for row, width in enumerate((450, 520, 360, 490)):
        page[70 + 30 * row : 77 + 30 * row, 40 : 40 + width] = 40  # a line of text
    page[195:209, 40:350] = colour
    _glyph(page, 40, 250, "abcdg" if digit == "3" else "abcdefg")
    for row in range(25):  # a 25 px "<" or ">", 4 px thick
        slant = abs(row - 12) * 20 // 12
        column = 110 + (slant if sign == "<" else 20 - slant)
        page[255 + row, column : column + 4] = 40
    if point:
        page[271 : 271 + point, 71 : 71 + point] = 40
    for row in range(30 if note else 0):  # a handwritten stroke
        page[310 + row, 200 + 3 * row : 208 + 3 * row] = 240 - note
    return page


def _other_page():
    page = np.full((360, 640, 3), 240, np.uint8)
    page[150:180, 100:550] = 30
    page[210:300, 150:450] = 120
    return page


def _pages(path, pages, seconds=4, rate=2, quantizer=8):
    """``pages`` in order, ``seconds`` each, at a fixed quantizer (2 to 31). The default 8 gives
    about the codec noise of a compressed screen recording, and keeps a 2 px dot."""
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.height, stream.width = pages[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"qmin": str(quantizer), "qmax": str(quantizer)}
        for page in pages:
            for _ in range(seconds * rate):
                for packet in stream.encode(av.VideoFrame.from_ndarray(page, format="rgb24")):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _paged_lecture(context, video, cache, kind="slides", **config):
    """:func:`_lecture` with the layout probe's answer already given."""
    ctx = _lecture(context, video, cache, **config)
    ctx.layout = Layout(kind=kind)
    return ctx


@pytest.mark.parametrize("kind", ["slides", "handwriting", "screencast"])
def test_a_page_that_comes_back_is_read_once(context, fake_llm, tmp_path, kind):
    """#138 AC: A, B, A gives three board states but two readings; the third copies the first,
    through the codec noise of a compressed screen recording (``_pages``)."""
    client = fake_llm(_board(text="page A"), _board(text="page B"))
    video = _pages(tmp_path / "lecture.mp4", [_page(), _other_page(), _page()])
    ctx = run(_paged_lecture(context, video, tmp_path / "cache", kind), VISION_STAGES)

    first, middle, again = ctx.visual
    assert len(client.requests) == 2
    assert again.text == first.text == "page A" and middle.text == "page B"
    assert (round(again.start), round(again.end)) == (8, 12)
    assert len({s.id for s in ctx.visual}) == 3
    # other speech on the return, so the request cache could not have caught it
    assert _speech(ctx.sentences, first.start, first.end) != _speech(
        ctx.sentences, again.start, again.end
    )
    assert ctx.stats["vision_reused"] == 1
    assert ctx.stats["vision_requests"] == 2  # a copied reading costs no request (#123)


def test_under_harder_compression_a_page_is_read_again(context, fake_llm, tmp_path):
    """#138: noise above the detail limits is not told from a change, so the page is read again."""
    client = fake_llm(_board(text="page A"), _board(text="page B"), _board(text="page A again"))
    pages = [_page(), _other_page(), _page()]
    video = _pages(tmp_path / "lecture.mp4", pages, quantizer=24)
    ctx = run(_paged_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert len(client.requests) == 3 and ctx.stats["vision_reused"] == 0


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"digit": "8"}, id="digit"),
        pytest.param({"point": 4}, id="decimal-point-4px"),
        pytest.param({"point": 2}, id="decimal-point-2px"),
        pytest.param({"sign": ">"}, id="inequality-sign"),
        pytest.param({"note": 48}, id="low-contrast-note"),
        pytest.param({"colour": (0, 130, 100)}, id="colour-of-equal-grey"),
    ],
)
def test_a_page_back_with_a_small_change_is_read_again(context, fake_llm, tmp_path, change):
    """#138 AC: a change the views' measure barely sees still gets a new reading."""
    small = _changed(_for_comparison(_page()), _for_comparison(_page(**change)))
    assert small < 0.02  # under vision_change_fraction: the views alone would call it the same

    client = fake_llm(_board(text="page A"), _board(text="page B"), _board(text="page A changed"))
    video = _pages(tmp_path / "lecture.mp4", [_page(), _other_page(), _page(**change)])
    ctx = run(_paged_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert len(client.requests) == 3
    assert ctx.visual[2].text == "page A changed"
    assert ctx.stats["vision_reused"] == 0


def test_a_reading_that_failed_is_not_reused(context, fake_llm, tmp_path):
    """#138 AC: only a reading that worked is copied; a later return is asked again."""
    client = fake_llm(
        openai.OpenAIError("Connection error."), _board(text="page B"), _board(text="page A")
    )
    video = _pages(tmp_path / "lecture.mp4", [_page(), _other_page(), _page()])
    with pytest.warns(UserWarning, match="keyframe left out"):
        ctx = run(_paged_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert len(client.requests) == 3
    assert [s.text for s in ctx.visual] == ["page B", "page A"]
    assert ctx.stats["vision_reused"] == 0 and ctx.stats["vision_failed"] == 1


@pytest.mark.parametrize("kind", ["blackboard", "whiteboard", "unknown"])
def test_boards_and_unknown_frames_are_read_every_time(context, fake_llm, tmp_path, kind):
    """#138 AC: a board, or what the probe could not name, is never copied (#133 owns boards)."""
    client = fake_llm(_board(text="page A"), _board(text="page B"), _board(text="page A again"))
    video = _pages(tmp_path / "lecture.mp4", [_page(), _other_page(), _page()])
    ctx = run(_paged_lecture(context, video, tmp_path / "cache", kind), VISION_STAGES)
    assert len(client.requests) == 3
    assert ctx.stats["vision_reused"] == 0


def test_a_rerun_from_the_caches_reuses_the_same_way(context, fake_llm, tmp_path):
    """#138 AC: views from views.json and readings from the request cache give the same states."""
    client = fake_llm(_board(text="page A"), _board(text="page B"))
    video = _pages(tmp_path / "lecture.mp4", [_page(), _other_page(), _page()])
    first = run(_paged_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    again = run(_paged_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert again.visual == first.visual
    assert len(client.requests) == 2  # the second run asks nothing
    assert again.stats["vision_reused"] == 1


def test_from_scratch_still_reuses_within_the_run(context, fake_llm, tmp_path):
    """#138 AC: --from-scratch skips the caches, not the reuse, which is never cached."""
    client = fake_llm(_board(text="page A"), _board(text="page B"))
    video = _pages(tmp_path / "lecture.mp4", [_page(), _other_page(), _page()])
    run(_paged_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    fresh = run(
        _paged_lecture(context, video, tmp_path / "cache", from_scratch=True), VISION_STAGES
    )
    assert len(client.requests) == 4  # A and B asked again, the return copied again
    assert fresh.stats["vision_reused"] == 1


def test_the_lookup_keeps_only_small_copies_and_readings(context, fake_llm, tmp_path, monkeypatch):
    """#138 AC: what the run holds per reading is a small grey copy, an index and the fields.
    Full-resolution pictures are decoded while their view is handled (the current keyframe for
    its small copy, an earlier one again for the detail check) and never kept."""
    assert {f.name for f in dataclasses.fields(_Reading)} == {"small", "view", "fields"}
    kept = []

    def recorded(**fields):
        kept.append(_Reading(**fields))
        return kept[-1]

    monkeypatch.setattr(vision_module, "_Reading", recorded)
    fake_llm(_board(text="page A"), _board(text="page B"))
    double = np.ones((2, 2, 1), np.uint8)  # 720p, so the small copy is smaller than the keyframe
    pages = [np.kron(page, double) for page in (_page(), _other_page(), _page())]
    video = _pages(tmp_path / "lecture.mp4", pages)
    run(_paged_lecture(context, video, tmp_path / "cache"), VISION_STAGES)

    assert [r.view for r in kept] == [0, 1]  # the reused third page added nothing
    for reading in kept:
        assert reading.small.ndim == 2 and reading.small.shape == (360, 640)  # 720p, halved
        assert set(reading.fields) == {"description", "text", "latex", "confidence"}


@pytest.mark.parametrize("size", [(720, 1280), (363, 645)])
def test_a_mark_at_the_border_is_a_change(size):
    """#138: the last rows and columns, past a whole square, are compared too."""
    before = np.full((*size, 3), 240, np.uint8)
    after = before.copy()
    after[-2:, -2:] = 40  # a 2 px mark in the bottom right corner
    assert _same_page(before, before.copy())
    assert not _same_page(before, after)


def test_a_keyframe_that_cannot_be_decoded_is_read_as_usual(
    context, fake_llm, tmp_path, monkeypatch
):
    """#138: no picture, no reuse; the reading itself does not need to decode the PNG."""
    assert _decoded(PNG) is None
    page = _page()
    assert np.array_equal(_decoded(_png(page)), page)

    monkeypatch.setattr(vision_module, "_decoded", lambda png: None)
    client = fake_llm(_board(text="page A"), _board(text="page B"), _board(text="page A again"))
    video = _pages(tmp_path / "lecture.mp4", [_page(), _other_page(), _page()])
    ctx = run(_paged_lecture(context, video, tmp_path / "cache"), VISION_STAGES)
    assert len(client.requests) == 3 and ctx.stats["vision_reused"] == 0

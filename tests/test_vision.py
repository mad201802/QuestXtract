import inspect

import av
import numpy as np
import pytest

from questxtract import VisualContext
from questxtract.segments import segments as segments_stage
from questxtract.sentences import sentences as sentences_stage
from questxtract.vision import (
    _for_comparison,
    _merge_shortest,
    _sample_frames,
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


def test_vision_refuses_when_enabled_but_unimplemented(context):
    context.config = context.config.replace(vision_enabled=True)
    with pytest.raises(NotImplementedError, match="#59"):
        vision(context)


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

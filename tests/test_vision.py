import inspect

import av
import numpy as np
import pytest

from questxtract import VisualContext
from questxtract.segments import segments as segments_stage
from questxtract.sentences import sentences as sentences_stage
from questxtract.vision import _for_comparison, _sample_frames, attach_visual, overlapping, vision

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


def test_vision_refuses_when_enabled_but_unimplemented(context, tmp_path, make_mp4):
    context.source = make_mp4(tmp_path / "lecture.mp4", audio=False)  # has a video track
    context.config = context.config.replace(vision_enabled=True)
    with pytest.raises(NotImplementedError, match="#59"):
        vision(context)


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

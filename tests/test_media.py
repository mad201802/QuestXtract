import wave

import av
import pytest

import questxtract.media as media_module
from questxtract.media import _HASH_BYTES, MediaError, extract_audio, ingest, source_hash


def _make_mp4(path, *, audio=True):
    with av.open(str(path), mode="w", format="mp4") as container:
        if audio:
            stream = container.add_stream("aac", rate=8000)
            stream.layout = "mono"
            frame = av.AudioFrame(format="s16", layout="mono", samples=800)
            frame.sample_rate = 8000
            frame.planes[0].update(b"\0" * frame.planes[0].buffer_size)
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        else:
            stream = container.add_stream("mpeg4", rate=1)
            stream.width = 16
            stream.height = 16
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame(16, 16, "yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


def test_hash_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"lecture")
    assert source_hash(a) == source_hash(a)

    b = tmp_path / "b.bin"
    b.write_bytes(b"lecturf")
    assert source_hash(a) != source_hash(b)


def test_hash_reads_both_ends_of_a_large_file(tmp_path):
    """Only the first and last 8 MiB are read, so a tail-only change must still register."""
    head = b"\x01" * _HASH_BYTES
    tail = b"\x02" * _HASH_BYTES
    big = tmp_path / "big.bin"
    big.write_bytes(head + b"\x00" * 1024 + tail)
    other = tmp_path / "other.bin"
    other.write_bytes(head + b"\x00" * 1024 + tail[:-1] + b"\x03")
    assert source_hash(big) != source_hash(other)


def test_size_alone_changes_the_hash(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"x" * 10)
    b = tmp_path / "b.bin"
    b.write_bytes(b"x" * 11)
    assert source_hash(a) != source_hash(b)


def test_ingest_rejects_a_missing_source(context):
    with pytest.raises(FileNotFoundError):
        ingest(context)


def test_ingest_rejects_a_non_mp4(tmp_path, context):
    media = tmp_path / "lecture.wav"
    media.write_bytes(b"not an mp4")
    context.source = media
    with pytest.raises(MediaError, match="expected an .mp4"):
        ingest(context)


def test_ingest_rejects_an_invalid_mp4(tmp_path, context):
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"not really an mp4")
    context.source = media
    with pytest.raises(MediaError, match="invalid or unreadable"):
        ingest(context)
    assert context.source_hash == source_hash(media)


def test_ingest_records_duration_for_a_valid_mp4(tmp_path, context):
    media = tmp_path / "lecture.mp4"
    _make_mp4(media)
    context.source = media
    result = ingest(context)
    assert result is context
    assert context.duration > 0


def test_ingest_rejects_a_video_without_audio(tmp_path, context):
    media = tmp_path / "lecture.mp4"
    _make_mp4(media, audio=False)
    context.source = media
    with pytest.raises(MediaError, match="no audio track"):
        ingest(context)


def test_ingest_wraps_container_open_errors(tmp_path, context, monkeypatch):
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"not really an mp4")
    context.source = media

    def fail_open(*args, **kwargs):
        raise OSError("cannot open")

    monkeypatch.setattr(media_module.av, "open", fail_open)
    with pytest.raises(MediaError, match="invalid or unreadable"):
        ingest(context)


def test_extract_audio_wraps_container_open_errors(tmp_path, monkeypatch):
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"not really an mp4")

    def fail_open(*args, **kwargs):
        raise OSError("cannot open")

    monkeypatch.setattr(media_module.av, "open", fail_open)
    with pytest.raises(MediaError, match="invalid or unreadable"):
        with extract_audio(media):
            pass


def test_extract_audio_wraps_wav_write_errors(tmp_path, monkeypatch):
    media = tmp_path / "lecture.mp4"
    _make_mp4(media)
    monkeypatch.setattr(media_module, "_write_wav", lambda *args: (_ for _ in ()).throw(OSError))
    with pytest.raises(MediaError, match="audio extraction failed"):
        with extract_audio(media):
            pass


def test_write_wav_flushes_final_encoder_packets(tmp_path, monkeypatch):
    class FakeResampler:
        def __init__(self, **kwargs):
            pass

        def resample(self, frame):
            return []

    class FakeInputContainer:
        def decode(self, audio_stream):
            return []

    class FakeOutputStream:
        layout = "mono"

        def encode(self, *frames):
            return ["final-packet"] if not frames else []

    class FakeDestination:
        def __init__(self):
            self.packets = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def add_stream(self, *args, **kwargs):
            return FakeOutputStream()

        def mux(self, packet):
            self.packets.append(packet)

    destination = FakeDestination()
    monkeypatch.setattr(media_module.av.audio.resampler, "AudioResampler", FakeResampler)
    monkeypatch.setattr(media_module.av, "open", lambda *args, **kwargs: destination)
    media_module._write_wav(FakeInputContainer(), object(), tmp_path / "audio.wav")
    assert destination.packets == ["final-packet"]


def test_extract_audio_creates_a_wav_and_cleans_it_up(tmp_path):
    media = tmp_path / "lecture.mp4"
    _make_mp4(media)
    with extract_audio(media) as audio:
        audio_path = audio
        assert audio.is_file()
        with wave.open(str(audio)) as wav:
            assert wav.getframerate() == 16_000
            assert wav.getnchannels() == 1
            assert wav.getnframes() > 0
    assert not audio_path.exists()


def test_extract_audio_rejects_a_video_without_audio(tmp_path):
    media = tmp_path / "lecture.mp4"
    _make_mp4(media, audio=False)
    with pytest.raises(MediaError, match="no audio track"):
        with extract_audio(media):
            pass

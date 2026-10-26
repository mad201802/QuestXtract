"""Stage 1: validate the source, probe the container, hash it for the cache key.

An ``.mp4`` or an ``.mp3`` is accepted; what the file holds decides the rest, not its name: a
file with only an audio track is transcribed and has no board to read (the vision stage checks
for a video track itself).

PyAV bundles the FFmpeg libraries, so this is not a subprocess and there is nothing for
the user to install. There is no separate audio-extraction stage: the ASR backend calls
:func:`extract_audio` for a temporary WAV (architecture.md 2.1).
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import av

from .models import Context, stage

_HASH_BYTES = 8 * 1024 * 1024  # first and last 8 MiB; full-hashing a 2 GB lecture is wasted I/O
_AUDIO_RATE = 16_000
_SUFFIXES = (".mp4", ".mp3")


class MediaError(ValueError):
    """Raised when a recording is not an MP4 or MP3, cannot be opened, or has no audio."""


def _validate_source(path: Path) -> None:
    """Validate the source path and the supported container type."""
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such file")
    if path.suffix.lower() not in _SUFFIXES:
        raise MediaError(f"{path}: unsupported media type; expected an .mp4 or .mp3 recording")


@contextmanager
def extract_audio(path: Path, progress: Callable[[float], None] | None = None) -> Iterator[Path]:
    """Extract a recording's first audio stream to a temporary 16 kHz mono WAV file.

    ``progress`` is called with the fraction of the recording decoded so far, in ``[0, 1]``.

    The yielded file exists only inside the context manager and is removed when the
    context exits, including when decoding or the consumer raises an exception.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        MediaError: If the path is not an MP4 or MP3, cannot be opened, or has no audio.

    """
    _validate_source(path)
    try:
        container = av.open(str(path))
    except (av.error.FFmpegError, OSError) as exc:
        raise MediaError(f"{path}: invalid or unreadable recording") from exc

    with container:
        audio_stream = next(
            (stream for stream in container.streams if stream.type == "audio"), None
        )
        if audio_stream is None:
            raise MediaError(f"{path}: recording contains no audio track")
        audio_stream = cast(av.audio.stream.AudioStream, audio_stream)

        with tempfile.TemporaryDirectory(prefix="questxtract-audio-") as directory:
            output = Path(directory) / "audio.wav"
            try:
                _write_wav(container, audio_stream, output, progress)
            except (av.error.FFmpegError, OSError) as exc:
                raise MediaError(f"{path}: audio extraction failed") from exc
            yield output


def _write_wav(
    container: av.container.InputContainer,
    audio_stream: av.audio.stream.AudioStream,
    output: Path,
    progress: Callable[[float], None] | None = None,
) -> None:
    """Decode an audio stream and write resampled PCM samples to ``output``."""
    duration = float(container.duration / av.time_base) if container.duration else 0.0
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=_AUDIO_RATE)
    with av.open(str(output), mode="w", format="wav") as destination:
        output_stream = destination.add_stream("pcm_s16le", rate=_AUDIO_RATE)
        output_stream.layout = "mono"
        for frame in container.decode(audio_stream):
            if progress and duration and frame.time is not None:
                progress(min(1.0, frame.time / duration))
            for converted in resampler.resample(frame):
                for packet in output_stream.encode(converted):
                    destination.mux(packet)
        for converted in resampler.resample(None):
            for packet in output_stream.encode(converted):
                destination.mux(packet)
        for packet in output_stream.encode():
            destination.mux(packet)


def source_hash(path: Path) -> str:
    """Content hash of a media file: first and last 8 MiB plus the file size.

    This is the cache key every later stage is filed under (architecture.md 2.1).
    """
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(_HASH_BYTES))
        if size > 2 * _HASH_BYTES:
            handle.seek(-_HASH_BYTES, 2)
            digest.update(handle.read(_HASH_BYTES))
    return digest.hexdigest()[:16]


@stage("ingest")
def ingest(ctx: Context) -> Context:
    """Validate the source and record its hash and duration.

    Raises:
        FileNotFoundError: If the source does not exist.
        MediaError: If the source is not a readable MP4 or MP3 recording with audio.

    """
    _validate_source(ctx.source)
    ctx.source_hash = source_hash(ctx.source)
    try:
        with av.open(str(ctx.source)) as container:
            if not any(stream.type == "audio" for stream in container.streams):
                raise MediaError(f"{ctx.source}: recording contains no audio track")
            ctx.duration = float(container.duration / av.time_base) if container.duration else 0.0
    except (av.error.FFmpegError, OSError) as exc:
        raise MediaError(f"{ctx.source}: invalid or unreadable recording") from exc
    return ctx

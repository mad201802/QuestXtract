"""Stage 1: validate the source, probe the container, hash it for the cache key.

PyAV bundles the FFmpeg libraries, so this is not a subprocess and there is nothing for
the user to install. There is no separate audio-extraction stage: stage 2 hands the
container straight to the decoder (pipeline.md 2.1).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import Context, stage

_HASH_BYTES = 8 * 1024 * 1024  # first and last 8 MiB; full-hashing a 2 GB lecture is wasted I/O


def source_hash(path: Path) -> str:
    """Content hash of a media file: first and last 8 MiB plus the file size.

    This is the cache key every later stage is filed under (pipeline.md 2.1).
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
        NotImplementedError: Container probing needs PyAV, which arrives with issue #7.

    """
    if not ctx.source.is_file():
        raise FileNotFoundError(f"{ctx.source}: no such file")
    ctx.source_hash = source_hash(ctx.source)
    # ponytail: duration needs av.open(); issue #7 replaces this stage body (pipeline.md 2.1).
    raise NotImplementedError("ingest: container probing lands with issue #7")

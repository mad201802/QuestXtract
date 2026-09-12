"""Persisting ASR output: read and write transcript segments as JSON."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..models.schemas import SchemaError, TranscriptSegment


def read_segments(path: str | Path) -> list[TranscriptSegment]:
    """Read a transcript JSON file.

    Accepts either a bare list of segments or an object with a ``segments`` key,
    which is what most ASR engines emit.

    Raises:
        SchemaError: If the file is not JSON or does not contain a list of segments.

    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SchemaError(f"{path}: not valid JSON ({exc})") from exc
    segments = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(segments, list):
        raise SchemaError(f"{path}: expected a list of segments or a 'segments' key holding one")
    return [TranscriptSegment.from_dict(s) for s in segments]


def segments_to_json(segments: Iterable[TranscriptSegment]) -> str:
    """Serialise segments to the transcript JSON format."""
    return json.dumps({"segments": [s.to_dict() for s in segments]}, indent=2, ensure_ascii=False)


def write_segments(segments: Iterable[TranscriptSegment], path: str | Path) -> None:
    """Write segments to ``path`` as JSON (UTF-8)."""
    Path(path).write_text(segments_to_json(segments), encoding="utf-8")

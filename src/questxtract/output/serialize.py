"""JSON export and import of lecture results."""

from __future__ import annotations

import json
from pathlib import Path

from ..models.schemas import LectureResult, SchemaError


def result_to_json(result: LectureResult) -> str:
    """Serialise a lecture result to the exported JSON format."""
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def write_result(result: LectureResult, path: str | Path) -> None:
    """Write a lecture result to ``path`` as JSON (UTF-8)."""
    Path(path).write_text(result_to_json(result), encoding="utf-8")


def read_result(path: str | Path) -> LectureResult:
    """Read a lecture result written by :func:`write_result` (possibly edited since).

    Raises:
        SchemaError: If the file is not JSON or does not match the exported schema.

    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path}: not valid JSON ({exc})") from exc
    return LectureResult.from_dict(data)

"""Orchestration: the stage contract, the stage list, the cache and progress."""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .answer import answer
from .config import Config
from .detect import detect
from .llm import judge
from .media import ingest
from .models import Context, Progress, Stage, StageInfo, Transcript, stage, stage_info
from .score import score
from .segments import segments
from .sentences import sentences
from .transcribe import transcribe
from .vision import attach_visual, layout, vision

__all__ = [
    "STAGES",
    "Cancelled",
    "Stage",
    "StageInfo",
    "cached_transcript",
    "process",
    "run",
    "stage",
    "stage_info",
    "stoppable",
]


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    return value


def _decode(value: Any, annotation: Any) -> Any:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in typing.get_args(annotation) if a is not type(None)]
        return None if value is None else _decode(value, inner[0])
    if origin in (tuple, list):
        (item,) = [a for a in typing.get_args(annotation) if a is not Ellipsis]
        return tuple(_decode(v, item) for v in value)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        if not isinstance(value, dict):
            return value
        hints = typing.get_type_hints(annotation)
        known = {f.name for f in dataclasses.fields(annotation)}
        fields = {k: _decode(v, hints[k]) for k, v in value.items() if k in known}
        return annotation(**fields)
    return value


def cached_transcript(config: Config, source_hash: str) -> Transcript | None:
    """Return the transcript cached for a recording, or ``None`` when there is none."""
    path = Path(config.cache_dir) / source_hash / "transcribe.json"
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        transcript = _decode(stored["data"], Transcript)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return transcript if isinstance(transcript, Transcript) else None


def _cache_path(ctx: Context, info: StageInfo) -> Path:
    return Path(ctx.config.cache_dir) / ctx.source_hash / f"{info.name}.json"


def _header(config: Config, info: StageInfo) -> dict[str, Any]:
    return {name: getattr(config, name) for name in info.uses}


def _cached(fn: Stage, ctx: Context) -> Context:
    info = stage_info(fn)
    assert info.cache is not None  # noqa: S101 - only called for caching stages
    if not ctx.source_hash:
        return fn(ctx)
    path = _cache_path(ctx, info)
    header = _header(ctx.config, info)

    if not ctx.config.from_scratch and path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored = None
        if isinstance(stored, dict) and stored.get("config") == header:
            annotation = typing.get_type_hints(Context)[info.cache]
            setattr(ctx, info.cache, _decode(stored["data"], annotation))
            return ctx

    ctx = fn(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": header, "data": _encode(getattr(ctx, info.cache))}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return ctx


STAGES: tuple[Stage, ...] = (
    ingest,
    transcribe,
    sentences,
    segments,
    layout,
    vision,
    attach_visual,
    detect,
    answer,
    judge,
    score,
)


class Cancelled(Exception):
    """A run was stopped on request (the graphical app's Cancel, #124)."""


def stoppable(progress: Progress | None, stop: Callable[[], bool]) -> Progress:
    """Return a progress callback that raises :class:`Cancelled` once ``stop`` returns true.

    Stages report progress as they go, so a stop takes effect inside a long stage
    (transcribing, reading the board), not only after it.
    """

    def report(name: str, fraction: float) -> None:
        if stop():
            raise Cancelled(f"stopped during {name}")
        if progress is not None:
            progress(name, fraction)

    return report


def run(
    ctx: Context,
    stages: Sequence[Stage] = STAGES,
    progress: Progress | None = None,
    stop: Callable[[], bool] | None = None,
) -> Context:
    """Run ``stages`` over ``ctx`` in order, caching and reporting as each declares.

    List order is execution order. There is no DAG resolver: a stage that needs something
    an earlier stage did not produce fails on the spot with a ``None``. ``stop``, when given, is
    asked before each stage and at each progress report a stage makes; once it returns true,
    :class:`Cancelled` is raised, the running stage writes no cache entry and no later stage
    runs.

    Raises:
        Cancelled: If ``stop`` asked for it before or during a stage.

    """
    progress = progress or ctx.progress
    if stop is not None:
        progress = stoppable(progress, stop)
    ctx.progress = progress
    for fn in stages:
        info = stage_info(fn)
        if stop is not None and stop():
            raise Cancelled(f"stopped before {info.name}")
        if progress:
            progress(info.name, 0.0)
        ctx = _cached(fn, ctx) if info.cache else fn(ctx)
    if progress:
        progress("done", 1.0)
    return ctx


def process(
    path: str | Path,
    config: Config | None = None,
    *,
    stages: Sequence[Stage] = STAGES,
    progress: Progress | None = None,
) -> Context:
    """Run the pipeline over one recording and return the finished :class:`Context`."""
    ctx = Context(source=Path(path), config=config or Config(), progress=progress)
    return run(ctx, stages, progress)

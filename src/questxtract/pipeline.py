"""Orchestration: the stage contract, the stage list, the cache and progress.

Everything in the pipeline is one function with the same ``Context -> Context`` signature.
That sameness is the whole extension mechanism: a caller who wants a different ASR engine,
a custom detector or an extra enrichment step passes a different list to ``run()``.

Four rules hold the contract up (architecture.md 11.1):

1. Stages never import each other; they communicate only through ``Context``.
2. Stages read config only through ``ctx.config``.
3. Optional stages no-op themselves rather than being filtered out by the runner.
4. A cached stage produces exactly one ``Context`` field of JSON-serialisable dataclasses.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .answer import answer
from .config import Config
from .detect import detect
from .llm import judge
from .media import ingest
from .models import Context, Progress, Stage, StageInfo, stage, stage_info
from .score import score
from .segments import segments
from .sentences import sentences
from .transcribe import transcribe
from .vision import attach_visual, vision

__all__ = ["STAGES", "Stage", "StageInfo", "process", "run", "stage", "stage_info"]


# --- cache (architecture.md 11.2) -------------------------------------------------


def _encode(value: Any) -> Any:
    """Turn dataclasses and tuples into JSON-ready values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    return value


def _decode(value: Any, annotation: Any) -> Any:
    """Rebuild dataclasses from JSON, driven by the ``Context`` field's annotation."""
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


def _cache_path(ctx: Context, info: StageInfo) -> Path:
    """Where this stage's cache entry lives for this source."""
    return Path(ctx.config.cache_dir) / ctx.source_hash / f"{info.name}.json"


def _header(config: Config, info: StageInfo) -> dict[str, Any]:
    """Return the config values this stage's cache entry is keyed on."""
    header = {name: getattr(config, name) for name in info.uses}
    if info.version:
        header["version"] = info.version
    return header


def _cached(fn: Stage, ctx: Context) -> Context:
    """Run ``fn``, reusing its cache entry when the header still matches."""
    info = stage_info(fn)
    assert info.cache is not None  # noqa: S101 - only called for caching stages
    if not ctx.source_hash:
        # No content hash, no cache key. Sharing one bucket across sources would replay
        # one recording's results into another's.
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


# --- the pipeline (architecture.md 1.3) -------------------------------------------

#: Reading this list is reading the diagram in architecture.md 1.3.
STAGES: tuple[Stage, ...] = (
    ingest,
    transcribe,
    sentences,
    segments,
    vision,
    attach_visual,
    detect,
    answer,
    judge,
    score,
)


def run(
    ctx: Context, stages: Sequence[Stage] = STAGES, progress: Progress | None = None
) -> Context:
    """Run ``stages`` over ``ctx`` in order, caching and reporting as each declares.

    List order is execution order. There is no DAG resolver: a stage that needs something
    an earlier stage did not produce fails on the spot with a ``None``.
    """
    progress = progress or ctx.progress
    # A stage that reports its own intra-stage progress (transcribe.py) reads ctx.progress
    # directly, not this function's local parameter - keep them in sync so a caller who
    # passes progress= straight to run() (rather than through process(), which already sets
    # both) still gets intra-stage updates, not just the stage-boundary 0.0/1.0 calls below.
    ctx.progress = progress
    for fn in stages:
        info = stage_info(fn)
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

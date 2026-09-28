"""Filtering extracted questions by time range, and experimentally by topic (FR11, #16).

Two entry points over the two shapes a finished run takes: :func:`filter_questions` on a
``Context``, :func:`filter_document` on a ``questions.json`` document from
``read_questions()``. Both keep the same rule and return a new value; the input is never
touched.

A question is in range when ``start <= question.start < end``. The question's start is the
FR7 timestamp, and half-open ranges tile a recording without counting a question twice.
Only the question list shrinks: segments, board states, source, config and stats are left
as they are, so ``segment_id`` and ``visual_ids`` still resolve.

Topic filtering is **experimental**. Nothing produces topic labels yet: a ``Segment`` is a
time unit, not a topic one (architecture.md 3.2), and every exported ``segments[].topic``
is ``null``. ``topic=`` matches a question's segment topic exactly, and warns when the data
carries no topic labels at all, since it then matches nothing.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import warnings
from typing import Any

from .models import Context, SchemaError


def filter_questions(
    ctx: Context,
    *,
    start: float | None = None,
    end: float | None = None,
    topic: str | None = None,
) -> Context:
    """Return a copy of a finished context holding only the questions in ``[start, end)``.

    Args:
        ctx: A context the pipeline has run over; only ``ctx.pairs`` is filtered.
        start: Inclusive lower bound in seconds; ``None`` leaves the range open below.
        end: Exclusive upper bound in seconds; ``None`` leaves the range open above.
        topic: **Experimental.** Keep only questions whose segment carries this topic.
            No stage produces topics yet, so on a ``Context`` this always warns and
            matches nothing.

    Returns:
        A new ``Context`` whose ``pairs`` are the matching questions in their original
        order. Every other field is the input's.

    Raises:
        ValueError: If the range is invalid (see :func:`filter_document`).

    """
    low, high = _check_range(start, end)
    pairs = tuple(p for p in ctx.pairs if _in_range(p.start, low, high))
    if topic is not None:
        # Segment has no topic field: a topic layer would sit over segments (3.2), not in them.
        _warn_no_topics()
        pairs = ()
    return dataclasses.replace(ctx, pairs=pairs, stats=dict(ctx.stats))


def filter_document(
    data: dict[str, Any],
    *,
    start: float | None = None,
    end: float | None = None,
    topic: str | None = None,
) -> dict[str, Any]:
    """Return a copy of a ``questions.json`` document holding only the questions in range.

    Works on edited documents too: ``review`` blocks are carried over as they are.

    Args:
        data: A document as returned by ``read_questions()`` or ``result_to_dict()``.
        start: Inclusive lower bound in seconds; ``None`` leaves the range open below.
        end: Exclusive upper bound in seconds; ``None`` leaves the range open above.
        topic: **Experimental.** Keep only questions whose ``segments[segment_id].topic``
            equals this. Warns when no segment carries a topic, which today is always.

    Returns:
        A deep copy of ``data`` whose ``questions`` list keeps only the matching questions,
        in their original order. Every other key is unchanged.

    Raises:
        ValueError: If ``start`` or ``end`` is not a finite, non-negative number, or
            ``start > end``. A range past the recording's end is not an error.
        SchemaError: If ``data`` has no ``questions`` list, or a question has no numeric
            ``start``.

    """
    low, high = _check_range(start, end)
    questions = data.get("questions")
    if not isinstance(questions, list):
        raise SchemaError("expected a 'questions' list")
    kept = [q for q in questions if _in_range(_question_start(q), low, high)]
    if topic is not None:
        topics = _segment_topics(data.get("segments"))
        if not any(t is not None for t in topics.values()):
            _warn_no_topics()
        kept = [q for q in kept if topics.get(q.get("segment_id")) == topic]
    return copy.deepcopy({**data, "questions": kept})


def _check_range(start: object, end: object) -> tuple[float, float]:
    """Validate the bounds and return them with open sides as ``-inf`` / ``inf``."""
    low = -math.inf if start is None else _bound("start", start)
    high = math.inf if end is None else _bound("end", end)
    if low > high:
        raise ValueError(f"start ({low:g}) must not be after end ({high:g})")
    return low, high


def _bound(name: str, value: object) -> float:
    """Return one bound as a float, or raise when it is not a finite, non-negative number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number of seconds, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return float(value)


def _in_range(when: float, low: float, high: float) -> bool:
    """Half-open membership: ``low <= when < high``."""
    return low <= when < high


def _question_start(question: object) -> float:
    """Return a document question's ``start``, or raise when it has none a filter can use."""
    value = question.get("start") if isinstance(question, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"question without a numeric 'start': {question!r}")
    return float(value)


def _segment_topics(segments: object) -> dict[Any, Any]:
    """Map each document segment's id to its topic, ``None`` included."""
    if not isinstance(segments, list):
        return {}
    return {s.get("id"): s.get("topic") for s in segments if isinstance(s, dict)}


def _warn_no_topics() -> None:
    """Tell the caller that ``topic=`` cannot match, rather than silently returning nothing."""
    warnings.warn(
        "topic filtering is experimental and this data has no topic labels (every segment "
        "topic is null), so no question matches; see #16",
        UserWarning,
        stacklevel=3,
    )

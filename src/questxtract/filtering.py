"""Filtering extracted questions by time range and text, and experimentally by topic."""

from __future__ import annotations

import copy
import dataclasses
import math
import re
import warnings
from typing import Any

from .export import result_to_dict
from .models import Context, SchemaError


def filter_questions(
    ctx: Context,
    *,
    start: float | None = None,
    end: float | None = None,
    topic: str | None = None,
    text: str | None = None,
) -> Context:
    """Return a copy of a finished context holding only the questions in ``[start, end)``.

    Args:
        ctx: A context the pipeline has run over; only ``ctx.pairs`` is filtered.
        start: Inclusive lower bound in seconds; ``None`` leaves the range open below.
        end: Exclusive upper bound in seconds; ``None`` leaves the range open above.
        topic: **Experimental.** Keep only questions whose segment carries this topic.
            No stage produces topics yet, so on a ``Context`` this always warns and
            matches nothing.
        text: Keep only questions where every word of ``text`` appears, spoken or on a
            linked board state (see :func:`filter_document`).

    Returns:
        A new ``Context`` whose ``pairs`` are the matching questions in their original
        order. Every other field is the input's.

    Raises:
        ValueError: If the range or ``text`` is invalid (see :func:`filter_document`).

    """
    low, high = _check_range(start, end)
    words = _query_words(text)
    pairs = tuple(p for p in ctx.pairs if _in_range(p.start, low, high))
    if words:
        doc = result_to_dict(dataclasses.replace(ctx, pairs=pairs))
        visuals = _visuals_by_id(doc["visuals"])
        pairs = tuple(
            p for p, q in zip(pairs, doc["questions"], strict=True) if _matches(q, visuals, words)
        )
    if topic is not None:
        _warn_no_topics()
        pairs = ()
    return dataclasses.replace(ctx, pairs=pairs, stats=dict(ctx.stats))


def filter_document(
    data: dict[str, Any],
    *,
    start: float | None = None,
    end: float | None = None,
    topic: str | None = None,
    text: str | None = None,
) -> dict[str, Any]:
    """Return a copy of a ``questions.json`` document holding only the questions in range.

    Works on edited documents too: ``review`` blocks are carried over as they are.

    Args:
        data: A document as returned by ``read_questions()`` or ``result_to_dict()``.
        start: Inclusive lower bound in seconds; ``None`` leaves the range open below.
        end: Exclusive upper bound in seconds; ``None`` leaves the range open above.
        topic: **Experimental.** Keep only questions whose ``segments[segment_id].topic``
            equals this. Warns when no segment carries a topic, which today is always.
        text: Keep only questions where every word of ``text`` appears as a whole word,
            ignoring case, in the question's ``text``, ``original_text``, ``answer.text`` or
            ``review.text``, or in the ``text``, ``latex`` or ``description`` of a board state
            in its ``visual_ids``. The words may come from different places.

    Returns:
        A deep copy of ``data`` whose ``questions`` list keeps only the matching questions,
        in their original order. Every other key is unchanged.

    Raises:
        ValueError: If ``start`` or ``end`` is not a finite, non-negative number, or
            ``start > end``. A range past the recording's end is not an error. Also if
            ``text`` is not a string or holds no word.
        SchemaError: If ``data`` has no ``questions`` list, or a question has no numeric
            ``start``.

    """
    low, high = _check_range(start, end)
    words = _query_words(text)
    questions = data.get("questions")
    if not isinstance(questions, list):
        raise SchemaError("expected a 'questions' list")
    kept = [q for q in questions if _in_range(_question_start(q), low, high)]
    if words:
        visuals = _visuals_by_id(data.get("visuals"))
        kept = [q for q in kept if _matches(q, visuals, words)]
    if topic is not None:
        topics = _segment_topics(data.get("segments"))
        if not any(t is not None for t in topics.values()):
            _warn_no_topics()
        kept = [q for q in kept if topics.get(q.get("segment_id")) == topic]
    return copy.deepcopy({**data, "questions": kept})


def _check_range(start: object, end: object) -> tuple[float, float]:
    low = -math.inf if start is None else _bound("start", start)
    high = math.inf if end is None else _bound("end", end)
    if low > high:
        raise ValueError(f"start ({low:g}) must not be after end ({high:g})")
    return low, high


def _bound(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number of seconds, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return float(value)


def _in_range(when: float, low: float, high: float) -> bool:
    return low <= when < high


def _question_start(question: object) -> float:
    value = question.get("start") if isinstance(question, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"question without a numeric 'start': {question!r}")
    return float(value)


def _query_words(text: object) -> frozenset[str]:
    if text is None:
        return frozenset()
    if not isinstance(text, str) or not (words := _words(text)):
        raise ValueError(f"text must contain a word, got {text!r}")
    return frozenset(words)


def _words(*values: object) -> set[str]:
    return {w for v in values if isinstance(v, str) for w in re.findall(r"[^\W_]+", v.casefold())}


def _visuals_by_id(visuals: object) -> dict[Any, dict[str, Any]]:
    if not isinstance(visuals, list):
        return {}
    return {v.get("id"): v for v in visuals if isinstance(v, dict)}


def _matches(
    question: dict[str, Any], visuals: dict[Any, dict[str, Any]], words: frozenset[str]
) -> bool:
    answer, review = question.get("answer"), question.get("review")
    found = _words(
        question.get("text"),
        question.get("original_text"),
        answer.get("text") if isinstance(answer, dict) else None,
        review.get("text") if isinstance(review, dict) else None,
    )
    for visual_id in question.get("visual_ids") or ():
        state = visuals.get(visual_id, {})
        found |= _words(state.get("text"), state.get("latex"), state.get("description"))
    return words <= found


def _segment_topics(segments: object) -> dict[Any, Any]:
    if not isinstance(segments, list):
        return {}
    return {s.get("id"): s.get("topic") for s in segments if isinstance(s, dict)}


def _warn_no_topics() -> None:
    warnings.warn(
        "topic filtering is experimental and this data has no topic labels (every segment "
        "topic is null), so no question matches; see #16",
        UserWarning,
        stacklevel=3,
    )

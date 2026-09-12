"""Stage 4: group sentences into segments that hang together topically.

One mode, structural (pipeline.md 3.2). Three consumers: FR11 (filter results by topic or
time range), the QA scoring's segment-relation term, and the visual join in section 6.5.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import Context, Segment, Sentence, stage


def build_segments(
    sentences: Sequence[Sentence],
    *,
    max_gap: float = 2.0,
    max_duration: float = 90.0,
    max_chars: int = 800,
) -> tuple[Segment, ...]:
    """Group consecutive sentences into segments.

    A segment closes on the first of: a silence gap over ``max_gap``, a duration over
    ``max_duration``, or text over ``max_chars``.

    Returns:
        The segments, each owning a half-open range into ``sentences``.

    """
    out: list[Segment] = []
    start = 0
    for i, sentence in enumerate(sentences):
        if i == start:
            continue
        gap = sentence.start - sentences[i - 1].end
        duration = sentence.end - sentences[start].start
        chars = sum(len(s.text) + 1 for s in sentences[start:i])
        if gap > max_gap or duration > max_duration or chars > max_chars:
            out.append(_build(sentences, start, i))
            start = i
    if start < len(sentences):
        out.append(_build(sentences, start, len(sentences)))
    return tuple(out)


def _build(sentences: Sequence[Sentence], lo: int, hi: int) -> Segment:
    """Build a segment over ``sentences[lo:hi]``."""
    return Segment(
        sentence_start=lo,
        sentence_end=hi,
        start=sentences[lo].start,
        end=sentences[hi - 1].end,
        text=" ".join(s.text for s in sentences[lo:hi]),
    )


@stage("segments")
def segments(ctx: Context) -> Context:
    """Group ``ctx.sentences`` into ``ctx.segments`` and stamp each sentence's index."""
    ctx.segments = build_segments(
        ctx.sentences,
        max_gap=ctx.config.segment_max_gap_seconds,
        max_duration=ctx.config.segment_max_duration_seconds,
        max_chars=ctx.config.segment_max_chars,
    )
    indexed = list(ctx.sentences)
    for index, segment in enumerate(ctx.segments):
        for position in range(segment.sentence_start, segment.sentence_end):
            indexed[position] = _with_segment(indexed[position], index)
    ctx.sentences = tuple(indexed)
    ctx.stats["segments"] = len(ctx.segments)
    return ctx


def _with_segment(sentence: Sentence, index: int) -> Sentence:
    """Return ``sentence`` with its ``segment_index`` set; it is frozen, so copy."""
    return Sentence(
        id=sentence.id,
        text=sentence.text,
        word_start=sentence.word_start,
        word_end=sentence.word_end,
        start=sentence.start,
        end=sentence.end,
        segment_index=index,
    )

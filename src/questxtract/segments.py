"""Stage 4: group sentences into segments bounded by silence and by length.

One mode, structural (architecture.md 3.2). A segment is **not** a topic: nothing here measures
what a run is about, and the only consumer that works today - the visual join in 6.5 - wants
a time interval, not a topic. The one that would want topicality, FR11's filter, is unbuilt
(the QA scoring's segment-relation term was dropped, 9.1), and neither this stage nor a
smarter one can move NFR2, since detection reads sentences and never looks at
``segment_index``.

3.2 records what a topical version would cost and why lexical cohesion, not embeddings, is
the first rung if it is ever wanted.
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

    A segment closes on the first of: a silence gap over ``max_gap``, or a sentence that
    would push the run past ``max_duration`` or ``max_chars``. Both length rules measure the
    run *with* the candidate sentence in it, so a closed segment is inside its budget - the
    one exception being a single sentence already over it, which nothing here can split.

    Returns:
        The segments, each owning a half-open range into ``sentences``.

    """
    out: list[Segment] = []
    start = 0
    chars = 0  # joined length of sentences[start:i], kept as a running total
    for i, sentence in enumerate(sentences):
        if i == start:
            # A new segment always takes its first sentence, however wide the silence before
            # it: closing here would emit an empty (start, start) run.
            chars = len(sentence.text)
            continue
        gap = sentence.start - sentences[i - 1].end
        duration = sentence.end - sentences[start].start
        width = chars + 1 + len(sentence.text)  # the +1 is the separator _build joins with
        if gap > max_gap:
            reason = "gap"
        elif duration > max_duration:
            reason = "duration"
        elif width > max_chars:
            reason = "chars"
        else:
            chars = width
            continue
        out.append(_build(sentences, start, i, reason))
        start = i
        chars = len(sentence.text)
    if start < len(sentences):
        out.append(_build(sentences, start, len(sentences), "end"))
    return tuple(out)


def _build(sentences: Sequence[Sentence], lo: int, hi: int, reason: str) -> Segment:
    """Build a segment over ``sentences[lo:hi]``, recording what closed it."""
    return Segment(
        sentence_start=lo,
        sentence_end=hi,
        start=sentences[lo].start,
        end=sentences[hi - 1].end,
        text=" ".join(s.text for s in sentences[lo:hi]),
        close_reason=reason,
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
    # Which rule is doing the cutting. On the labelled recordings a pause closes 95-98% of
    # segments and the length caps are rare backstops (3.2, #41); caps closing more than a
    # few is the sign a new recording does not fit those thresholds.
    for reason in ("gap", "duration", "chars"):
        ctx.stats[f"segments_closed_{reason}"] = sum(
            1 for s in ctx.segments if s.close_reason == reason
        )
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

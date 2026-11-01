"""Stage 4: group sentences into segments bounded by silence and by length."""

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
    """Group consecutive sentences into segments."""
    out: list[Segment] = []
    start = 0
    chars = 0
    for i, sentence in enumerate(sentences):
        if i == start:
            chars = len(sentence.text)
            continue
        gap = sentence.start - sentences[i - 1].end
        duration = sentence.end - sentences[start].start
        width = chars + 1 + len(sentence.text)
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
    for reason in ("gap", "duration", "chars"):
        ctx.stats[f"segments_closed_{reason}"] = sum(
            1 for s in ctx.segments if s.close_reason == reason
        )
    return ctx


def _with_segment(sentence: Sentence, index: int) -> Sentence:
    return Sentence(
        id=sentence.id,
        text=sentence.text,
        word_start=sentence.word_start,
        word_end=sentence.word_end,
        start=sentence.start,
        end=sentence.end,
        segment_index=index,
    )

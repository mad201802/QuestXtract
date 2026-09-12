"""Stage 3: rebuild sentences from the word stream.

ASR segments are not sentences: the engine emits chunks bounded by its decoding window
and by silence, so one chunk routinely holds two questions or half of one. Detection
needs units that read on their own.

We attack the input rather than the output (pipeline.md 3.1): a ``turbo``-sized Whisper
punctuates well, and a punctuated transcript splits on a regex. Boundaries are snapped to
word indices, so a sentence's timings are real measurements rather than interpolations.
"""

from __future__ import annotations

from .models import Context, Sentence, Transcript, Word, stage


def split_sentences(
    transcript: Transcript, *, max_words: int = 60, min_gap: float = 0.6
) -> tuple[Sentence, ...]:
    """Split a transcript's words into sentences.

    Args:
        transcript: The words to split, in recording order.
        max_words: A sentence longer than this with no terminal punctuation is force-split.
        min_gap: Only force-split at a silence gap of at least this many seconds.

    Returns:
        The sentences, each owning a half-open range into ``transcript.words``.

    """
    words = transcript.words
    out: list[Sentence] = []
    for start, end in _boundaries(words):
        for lo, hi in _force_split(words, start, end, max_words, min_gap):
            out.append(_build(words, lo, hi))
    return tuple(out)


def _boundaries(words: tuple[Word, ...]) -> list[tuple[int, int]]:
    r"""Return half-open word ranges, one per sentence, from terminal punctuation.

    Equivalent to pipeline.md 3.1's ``(?<=[.!?])\s+`` split followed by snapping the
    character offsets back to word indices, done in one walk instead of two.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for i, word in enumerate(words):
        if word.text.rstrip().endswith(("?", ".", "!")):
            spans.append((start, i + 1))
            start = i + 1
    if start < len(words):
        spans.append((start, len(words)))
    return spans


def _force_split(
    words: tuple[Word, ...], start: int, end: int, max_words: int, min_gap: float
) -> list[tuple[int, int]]:
    """Split an over-long unpunctuated run at its largest internal gap.

    The one run-on case the regex genuinely cannot see (pipeline.md 3.1).
    """
    if end - start <= max_words:
        return [(start, end)]
    gaps = [(words[i + 1].start - words[i].end, i + 1) for i in range(start, end - 1)]
    widest, cut = max(gaps, default=(0.0, end))
    if widest < min_gap:
        return [(start, end)]
    return _force_split(words, start, cut, max_words, min_gap) + _force_split(
        words, cut, end, max_words, min_gap
    )


def _build(words: tuple[Word, ...], lo: int, hi: int) -> Sentence:
    """Build a sentence over ``words[lo:hi]``; both timings are real measurements."""
    return Sentence(
        text=" ".join(w.text for w in words[lo:hi]).strip(),
        word_start=lo,
        word_end=hi,
        start=words[lo].start,
        end=words[hi - 1].end,
    )


@stage("sentences")
def sentences(ctx: Context) -> Context:
    """Split ``ctx.transcript`` into ``ctx.sentences``."""
    if ctx.transcript is None:
        raise ValueError("sentences: no transcript; stage 2 must run first")
    ctx.sentences = split_sentences(
        ctx.transcript,
        max_words=ctx.config.max_sentence_words,
        min_gap=ctx.config.force_split_min_gap_seconds,
    )
    ctx.stats["sentences"] = len(ctx.sentences)
    return ctx

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

#: Words that end in a full stop without ending a sentence. A missed boundary merges two
#: sentences; a false one leaves a fragment that reads as neither, so the guard is a
#: suppression list: merging is the safer error for a stage whose job is standalone units.
_ABBREV = frozenset("dr prof mr mrs ms vs eg ie etc fig cf al approx".split())


def _is_terminal(word: Word) -> bool:
    """Whether this word ends a sentence, rather than an abbreviation or an ellipsis."""
    text = word.text.rstrip()
    if text.endswith(("?", "!")):
        return True
    if not text.endswith(".") or text.endswith(".."):
        return False
    core = text[:-1]
    if len(core) == 1 and core.isalpha():  # "capital N." - an initial, not a full stop
        return False
    return core.lower().replace(".", "") not in _ABBREV


def split_sentences(
    transcript: Transcript, *, max_words: int = 60, min_gap: float = 0.6
) -> tuple[Sentence, ...]:
    """Split a transcript's words into sentences.

    Args:
        transcript: The words to split, in recording order.
        max_words: Hard cap on a sentence's length. A run-on over it is split even when no
            silence in it qualifies.
        min_gap: A silence of at least this many seconds is a boundary inside a run-on.

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
    character offsets back to word indices, done in one walk instead of two, with the
    abbreviations and initials that regex cannot tell from a full stop held back.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for i, word in enumerate(words):
        if _is_terminal(word):
            spans.append((start, i + 1))
            start = i + 1
    if start < len(words):
        spans.append((start, len(words)))
    return spans


def _force_split(
    words: tuple[Word, ...], start: int, end: int, max_words: int, min_gap: float
) -> list[tuple[int, int]]:
    """Split a run-on at its silences, then cap whatever is still over-long.

    The one case the regex genuinely cannot see (pipeline.md 3.1). A span that both ends
    in real terminal punctuation and fits in ``max_words`` is a sentence and is left
    alone - lecturers pause mid-sentence at the board constantly, so silence on its own
    is not a boundary. Anything else is a run-on the ASR failed to punctuate, and there
    silence is the best evidence available.
    """
    if end - start <= max_words and _is_terminal(words[end - 1]):
        return [(start, end)]
    cuts = [i + 1 for i in range(start, end - 1) if words[i + 1].start - words[i].end >= min_gap]
    bounds = [start, *cuts, end]
    return [
        p
        for lo, hi in zip(bounds, bounds[1:], strict=False)  # pairwise: one shorter by design
        for p in _cap(words, lo, hi, max_words)
    ]


def _cap(words: tuple[Word, ...], lo: int, hi: int, max_words: int) -> list[tuple[int, int]]:
    """Halve a run that no pause broke up, at its widest gap, tie-breaking to the middle.

    The last resort for a run-on with no usable silence in it: ``max_words`` is a cap, not
    a hint. The tie-break is what stops a gapless run degenerating into one-word sentences.
    """
    if hi - lo <= max(1, max_words):
        return [(lo, hi)]
    mid = (lo + hi) / 2
    cut = max(
        (words[i + 1].start - words[i].end, -abs(i + 1 - mid), i + 1) for i in range(lo, hi - 1)
    )[2]
    return _cap(words, lo, cut, max_words) + _cap(words, cut, hi, max_words)


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

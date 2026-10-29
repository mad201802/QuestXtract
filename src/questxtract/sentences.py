"""Stage 3: rebuild sentences from the word stream."""

from __future__ import annotations

from .models import Context, Sentence, Transcript, Word, stage

_ABBREV = frozenset("dr prof mr mrs ms vs eg ie etc fig cf al approx".split())


def _is_terminal(word: str) -> bool:
    text = word.rstrip()
    if text.endswith(("?", "!")):
        return True
    if not text.endswith(".") or text.endswith(".."):
        return False
    core = text[:-1]
    if len(core) == 1 and core.isalpha() and core.isupper():
        return False
    return core.lower().replace(".", "") not in _ABBREV


def split_sentences(
    transcript: Transcript, *, max_words: int = 60, min_gap: float = 0.6
) -> tuple[Sentence, ...]:
    """Split a transcript's words into sentences."""
    words = transcript.words
    out: list[Sentence] = []
    for start, end in _boundaries(words):
        for lo, hi in _force_split(words, start, end, max_words, min_gap):
            out.append(_build(words, lo, hi))
    return tuple(out)


def _boundaries(words: tuple[Word, ...]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for i, word in enumerate(words):
        if _is_terminal(word.text):
            spans.append((start, i + 1))
            start = i + 1
    if start < len(words):
        spans.append((start, len(words)))
    return spans


def _force_split(
    words: tuple[Word, ...], start: int, end: int, max_words: int, min_gap: float
) -> list[tuple[int, int]]:
    if end - start <= max_words and _is_terminal(words[end - 1].text):
        return [(start, end)]
    cuts = [i + 1 for i in range(start, end - 1) if words[i + 1].start - words[i].end >= min_gap]
    bounds = [start, *cuts, end]
    return [
        p
        for lo, hi in zip(bounds, bounds[1:], strict=False)
        for p in _cap(words, lo, hi, max_words)
    ]


def _cap(words: tuple[Word, ...], lo: int, hi: int, max_words: int) -> list[tuple[int, int]]:
    if hi - lo <= max(1, max_words):
        return [(lo, hi)]
    mid = (lo + hi) / 2
    cut = max(
        (words[i + 1].start - words[i].end, -abs(i + 1 - mid), i + 1) for i in range(lo, hi - 1)
    )[2]
    return _cap(words, lo, cut, max_words) + _cap(words, cut, hi, max_words)


def _build(words: tuple[Word, ...], lo: int, hi: int) -> Sentence:
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
    ctx.stats["sentences_unpunctuated"] = sum(1 for s in ctx.sentences if not _is_terminal(s.text))
    return ctx

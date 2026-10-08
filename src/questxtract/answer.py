"""Stage 7: find the answer span for each question candidate.

One rule, the adjacent seed (architecture.md 8). Answers further away come from the LLM
detector (stage 8, ``--detector llm``), not from rules here.

A question with no answer is still a result. architecture.md 9.2 is explicit: FR6 and FR7 ask
for questions with timestamps, not for answered questions.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import Context, QAPair, QuestionCandidate, Sentence, stage

# A confirmation tag, check-in or framing question is not asking for content, so whatever
# follows it is the lecture carrying on, not its answer.
_UNSEEDED = frozenset({"tag", "check_in", "framing"})


def adjacent_seed(
    question: Sentence, following: Sentence | None, *, max_gap: float = 1.0
) -> Sentence | None:
    """Return the sentence right after the question when it reads as its answer.

    Distance 0: the Socratic case, and the highest-precision answer shape there is. Stage 3
    splits *at* the ``?``, so the answer to "What is X? It is Y." is the next sentence, not
    a tail of the question's own (issue #38). Taken only across a near-zero silence, and
    never when that sentence is itself a question.
    """
    if following is None or not following.text.strip():
        return None
    if following.start - question.end > max_gap:
        return None
    if following.text.rstrip().endswith("?"):
        return None
    return following


def pair_answers(
    candidates: Sequence[QuestionCandidate],
    sentences: Sequence[Sentence],
    *,
    max_seed_gap: float = 1.0,
) -> tuple[QAPair, ...]:
    """Build one :class:`QAPair` per candidate, with the adjacent seed where present."""
    index = {s.id: i for i, s in enumerate(sentences)}
    out: list[QAPair] = []
    for candidate in candidates:
        i = index.get(candidate.sentence_id)
        if i is None:
            continue
        sentence = sentences[i]
        following = sentences[i + 1] if i + 1 < len(sentences) else None
        seed = None
        if not _UNSEEDED & set(candidate.reasons):
            seed = adjacent_seed(sentence, following, max_gap=max_seed_gap)
        out.append(
            QAPair(
                question=candidate,
                answer_sentence_ids=(seed.id,) if seed else (),
                answer_text=seed.text.strip() if seed else "",
                confidence=0.0,  # stage 9 fills this in
                start=sentence.start,
                end=sentence.end,
                reasons=("intra_sentence_qa",) if seed else ("no_answer_found",),
            )
        )
    return tuple(out)


@stage("answer")
def answer(ctx: Context) -> Context:
    """Pair each candidate in ``ctx.candidates`` with an answer span into ``ctx.pairs``."""
    ctx.pairs = pair_answers(
        ctx.candidates, ctx.sentences, max_seed_gap=ctx.config.answer_seed_max_gap_seconds
    )
    ctx.stats["answered"] = sum(1 for p in ctx.pairs if p.answer_text)
    return ctx

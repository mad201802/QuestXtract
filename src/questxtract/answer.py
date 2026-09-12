"""Stage 7: find the answer span for each question candidate.

Three channels composing rather than competing (pipeline.md 8): the same-sentence seed,
the forward window and the far-window rescue. Only the first is implemented; the other two
land with the solution-extraction work.

A question with no answer is still a result. Pipeline.md 9.2 is explicit: FR6 and FR7 ask
for questions with timestamps, not for answered questions.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import Context, QAPair, QuestionCandidate, Sentence, stage


def same_sentence_seed(sentence: Sentence, question_text: str) -> str:
    """Text after the ``?`` in the question's own sentence.

    Distance 0, gap 0: the Socratic case, and the highest-precision answer shape there is.
    Returns an empty string when the sentence holds no trailing text.
    """
    if "?" not in sentence.text:
        return ""
    tail = sentence.text.rsplit("?", 1)[1].strip()
    return tail if tail and tail != question_text else ""


def pair_answers(
    candidates: Sequence[QuestionCandidate], sentences: Sequence[Sentence]
) -> tuple[QAPair, ...]:
    """Build one :class:`QAPair` per candidate, with the same-sentence seed where present."""
    by_id = {s.id: s for s in sentences}
    out: list[QAPair] = []
    for candidate in candidates:
        sentence = by_id.get(candidate.sentence_id)
        if sentence is None:
            continue
        seed = same_sentence_seed(sentence, candidate.text)
        out.append(
            QAPair(
                question=candidate,
                answer_sentence_ids=(sentence.id,) if seed else (),
                answer_text=seed,
                confidence=0.0,  # stage 8 fills this in
                start=sentence.start,
                end=sentence.end,
                reasons=("intra_sentence_qa",) if seed else ("no_answer_found",),
            )
        )
    return tuple(out)


@stage("answer")
def answer(ctx: Context) -> Context:
    """Pair each candidate in ``ctx.candidates`` with an answer span into ``ctx.pairs``."""
    # ponytail: forward window + far-window rescue, pipeline.md 8. Swapping the whole search
    # replaces this stage entry; swapping one channel is an edit inside this module.
    ctx.pairs = pair_answers(ctx.candidates, ctx.sentences)
    ctx.stats["answered"] = sum(1 for p in ctx.pairs if p.answer_text)
    return ctx

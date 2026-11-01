"""Stage 9: confidence and the threshold."""

from __future__ import annotations

from collections.abc import Sequence

from .models import Context, QAPair, stage

_QUESTION_WEIGHT = 0.55


def confidence_label(confidence: float) -> str:
    """Bucket a confidence into ``high``, ``medium`` or ``low`` (architecture.md 9.2)."""
    if confidence >= 0.75:
        return "high"
    return "medium" if confidence >= 0.45 else "low"


def score_pairs(
    pairs: Sequence[QAPair], *, min_confidence: float = 0.40, weigh_answers: bool = True
) -> tuple[QAPair, ...]:
    """Assign each pair a confidence and drop those below ``min_confidence``."""
    out: list[QAPair] = []
    for pair in pairs:
        confidence = pair.question.score
        if weigh_answers and not pair.answer_text:
            confidence *= _QUESTION_WEIGHT
        confidence = round(min(1.0, max(0.0, confidence)), 2)
        if confidence >= min_confidence:
            out.append(_rescored(pair, confidence))
    return tuple(out)


def _rescored(pair: QAPair, confidence: float) -> QAPair:
    return QAPair(
        question=pair.question,
        answer_sentence_ids=pair.answer_sentence_ids,
        answer_text=pair.answer_text,
        confidence=confidence,
        start=pair.start,
        end=pair.end,
        verdict=pair.verdict,
        reasons=pair.reasons,
    )


@stage("score")
def score(ctx: Context) -> Context:
    """Score and filter ``ctx.pairs``."""
    ctx.pairs = score_pairs(
        ctx.pairs,
        min_confidence=ctx.config.min_confidence,
        weigh_answers=ctx.config.detector != "llm",
    )
    ctx.stats["emitted"] = len(ctx.pairs)
    return ctx

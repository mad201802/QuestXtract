"""Stage 9: confidence and the threshold.

One number the user can reason about (FR10), plus the judge's verdict. No parallel quality
score, no risk band, no profile-scoped emission gates (architecture.md 9.2).
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import Context, QAPair, stage

_QUESTION_WEIGHT = 0.55
_ANSWER_WEIGHT = 0.45


def confidence_label(confidence: float) -> str:
    """Bucket a confidence into ``high``, ``medium`` or ``low`` (architecture.md 9.2)."""
    if confidence >= 0.75:
        return "high"
    return "medium" if confidence >= 0.45 else "low"


def score_pairs(pairs: Sequence[QAPair], *, min_confidence: float = 0.40) -> tuple[QAPair, ...]:
    """Assign each pair a confidence and drop those below ``min_confidence``.

    A pair with no answer keeps ``0.55 * question_score`` and an empty answer span; it is
    still emitted, because FR6 and FR7 ask for questions with timestamps, not for answered
    questions.
    """
    out: list[QAPair] = []
    for pair in pairs:
        # An answer is not scored on its own (architecture.md 9.1). With the LLM detector,
        # question.score is the model's own confidence.
        confidence = _QUESTION_WEIGHT * pair.question.score
        if pair.answer_text:
            confidence += _ANSWER_WEIGHT * pair.question.score
        confidence = round(min(1.0, max(0.0, confidence)), 2)
        if confidence >= min_confidence:
            out.append(_rescored(pair, confidence))
    return tuple(out)


def _rescored(pair: QAPair, confidence: float) -> QAPair:
    """Return ``pair`` with its confidence set; it is frozen, so copy."""
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
    # No dedupe (architecture.md 9.3).
    ctx.pairs = score_pairs(ctx.pairs, min_confidence=ctx.config.min_confidence)
    ctx.stats["emitted"] = len(ctx.pairs)
    return ctx

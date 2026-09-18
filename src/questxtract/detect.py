"""Stage 6: find the sentences that are questions.

Purely lexical and orthographic. No classifier, no prosody. Every decision leaves a reason
code, which is what makes both the review workflow and the LLM judge's job possible
(pipeline.md 7).

Sentences are only extracted, never invented: every candidate's text is a span of the
transcript.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import Context, QuestionCandidate, Sentence, stage

_WH_WORDS = frozenset({"who", "what", "when", "where", "why", "how", "which", "whose", "whom"})
_AUX_WORDS = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "am",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "have",
        "has",
        "had",
        "any",
        "if",
    }
)

_TYPES = {
    "what": "what",
    "why": "why",
    "how": "how",
    "who": "what",
    "when": "what",
    "where": "what",
    "which": "what",
    "whose": "what",
    "whom": "what",
}

# ponytail: keyword heuristic. pipeline.md 7.1-7.3 specifies the cue tables, the vetoes and
# the full additive scoring table; they land with issue #10 once the labelled set (#11) exists.
_SCORES = {"mark": 0.6, "wh": 0.5, "aux": 0.5}


def score_sentence(sentence: str) -> tuple[float, tuple[str, ...]]:
    """Score how likely a sentence is a question.

    Returns:
        The score in ``[0, 1]`` and the names of the heuristics that fired.

    """
    reasons: list[str] = []
    if sentence.rstrip().endswith("?"):
        reasons.append("mark")
    words = re.findall(r"\w+", sentence.casefold())
    if words:
        if words[0] in _WH_WORDS:
            reasons.append("wh")
        if words[0] in _AUX_WORDS:
            reasons.append("aux")
    score = min(1.0, sum((_SCORES[r] for r in reasons), 0.0))
    return score, tuple(reasons)


def question_type(sentence: str) -> str:
    """Classify a question by its head word."""
    words = re.findall(r"\w+", sentence.casefold())
    if not words:
        return "other"
    if words[0] in _TYPES:
        return _TYPES[words[0]]
    return "yes_no" if words[0] in _AUX_WORDS else "other"


def detect_questions(
    sentences: Sequence[Sentence], *, threshold: float = 0.45, max_chars: int = 350
) -> tuple[QuestionCandidate, ...]:
    """Find the question candidates among ``sentences``.

    Args:
        sentences: The transcript's sentences, in recording order.
        threshold: Minimum score for a sentence to be reported.
        max_chars: A sentence longer than this is rejected outright (pipeline.md 7.3).

    Returns:
        The candidates, in recording order.

    """
    out: list[QuestionCandidate] = []
    for sentence in sentences:
        text = sentence.text.strip()
        if not text or len(text) > max_chars:
            continue
        score, reasons = score_sentence(text)
        if score >= threshold:
            out.append(
                QuestionCandidate(
                    sentence_id=sentence.id,
                    text=text,
                    score=round(score, 2),
                    question_type=question_type(text),
                    reasons=reasons,
                )
            )
    return tuple(out)


@stage("detect")
def detect(ctx: Context) -> Context:
    """Score ``ctx.sentences`` into ``ctx.candidates``."""
    ctx.candidates = detect_questions(
        ctx.sentences,
        threshold=ctx.config.min_question_score,
        max_chars=ctx.config.max_question_chars,
    )
    ctx.stats["candidates"] = len(ctx.candidates)
    return ctx

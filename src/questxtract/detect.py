"""Heuristic question detection over transcript segments."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Question, TranscriptSegment

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

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
    }
)

# ponytail: keyword heuristic, swap for a trained classifier if recall < 50% on the labelled set.
_SCORES = {"mark": 0.6, "wh": 0.5, "aux": 0.5}


def score_sentence(sentence: str) -> tuple[float, tuple[str, ...]]:
    """Score how likely a sentence is a question.

    Returns:
        The confidence in ``[0, 1]`` and the names of the heuristics that fired.

    """
    reasons: list[str] = []
    if sentence.rstrip().endswith("?"):
        reasons.append("mark")
    words = re.findall(r"[a-z']+", sentence.lower())
    if words:
        if words[0] in _WH_WORDS:
            reasons.append("wh")
        if words[0] in _AUX_WORDS:
            reasons.append("aux")
    confidence = min(1.0, sum(_SCORES[r] for r in reasons))
    return confidence, tuple(reasons)


def detect_questions(
    segments: Iterable[TranscriptSegment], threshold: float = 0.5
) -> list[Question]:
    """Find question sentences in transcript segments.

    Sentences are only extracted, never invented: every returned question is a
    verbatim sentence from the transcript.

    Args:
        segments: Transcribed segments, in recording order.
        threshold: Minimum confidence for a sentence to be reported.

    Returns:
        The detected questions with their timestamps, in recording order.

    """
    questions: list[Question] = []
    for segment in segments:
        for sentence in _SENTENCE_SPLIT.split(segment.text.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            confidence, reasons = score_sentence(sentence)
            if confidence >= threshold:
                questions.append(
                    Question(
                        text=sentence,
                        start=segment.start,
                        end=segment.end,
                        confidence=round(confidence, 2),
                        reasons=reasons,
                    )
                )
    return questions

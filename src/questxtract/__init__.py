"""QuestXtract: extract spoken questions from lecture recordings."""

from .detect import detect_questions, score_sentence
from .models import (
    Question,
    TranscriptSegment,
    questions_to_json,
    read_segments,
    write_questions,
)

__version__ = "0.1.0"

__all__ = [
    "Question",
    "TranscriptSegment",
    "__version__",
    "detect_questions",
    "questions_to_json",
    "read_segments",
    "score_sentence",
    "write_questions",
]

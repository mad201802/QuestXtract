"""QuestXtract: extract spoken questions from lecture recordings."""

from .audio.transcripts import read_segments, segments_to_json, write_segments
from .detection.questions import detect_questions, score_sentence
from .models.schemas import LectureResult, Question, SchemaError, TranscriptSegment, Word
from .output.serialize import read_result, result_to_json, write_result

__version__ = "0.1.0"

__all__ = [
    "LectureResult",
    "Question",
    "SchemaError",
    "TranscriptSegment",
    "Word",
    "__version__",
    "detect_questions",
    "read_result",
    "read_segments",
    "result_to_json",
    "score_sentence",
    "segments_to_json",
    "write_result",
    "write_segments",
]

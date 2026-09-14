"""QuestXtract: extract spoken questions from lecture recordings.

The pipeline is a list of ``Context -> Context`` functions (``STAGES``); ``process()`` runs
it over one recording. See ``docs/architecture.md`` for what each stage does and why.
"""

from .config import Config
from .export import read_questions, result_to_json, write_questions
from .media import MediaError, extract_audio
from .models import (
    AsrSegment,
    Context,
    QAPair,
    QuestionCandidate,
    SchemaError,
    Segment,
    Sentence,
    Transcript,
    VisualContext,
    Word,
)
from .pipeline import STAGES, process, run, stage
from .schema import Question

__version__ = "0.1.0"

__all__ = [
    "STAGES",
    "AsrSegment",
    "Config",
    "Context",
    "MediaError",
    "QAPair",
    "Question",
    "QuestionCandidate",
    "SchemaError",
    "Segment",
    "Sentence",
    "Transcript",
    "VisualContext",
    "Word",
    "__version__",
    "process",
    "extract_audio",
    "read_questions",
    "result_to_json",
    "run",
    "stage",
    "write_questions",
]

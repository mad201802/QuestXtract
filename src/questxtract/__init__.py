"""QuestXtract: extract spoken questions from lecture recordings.

The pipeline is a list of ``Context -> Context`` functions (``STAGES``); ``process()`` runs
it over one recording. See ``docs/pipeline.md`` for what each stage does and why.
"""

from .config import Config
from .export import read_questions, result_to_json, write_questions
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

__version__ = "0.1.0"

__all__ = [
    "STAGES",
    "AsrSegment",
    "Config",
    "Context",
    "QAPair",
    "QuestionCandidate",
    "SchemaError",
    "Segment",
    "Sentence",
    "Transcript",
    "VisualContext",
    "Word",
    "__version__",
    "process",
    "read_questions",
    "result_to_json",
    "run",
    "stage",
    "write_questions",
]

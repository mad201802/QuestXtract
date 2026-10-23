"""QuestXtract: extract spoken questions from lecture recordings.

The pipeline is a list of ``Context -> Context`` functions (``STAGES``); ``process()`` runs
it over one recording. See ``docs/architecture.md`` for what each stage does and why.
"""

from importlib.metadata import PackageNotFoundError, version

from .asr import AsrBackend, register_backend, resolve_backend
from .config import Config
from .export import read_questions, result_to_json, write_questions
from .filtering import filter_document, filter_questions
from .media import MediaError, extract_audio
from .models import (
    AsrSegment,
    Context,
    Layout,
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
from .qti import QtiExportError
from .qti_package import export_qti
from .quiz_schema import read_quiz, validate_quiz
from .schema import Question

try:
    __version__ = version("questxtract")  # from pyproject.toml via the installed metadata
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "STAGES",
    "AsrBackend",
    "AsrSegment",
    "Config",
    "Context",
    "Layout",
    "MediaError",
    "QAPair",
    "QtiExportError",
    "Question",
    "QuestionCandidate",
    "SchemaError",
    "Segment",
    "Sentence",
    "Transcript",
    "VisualContext",
    "Word",
    "__version__",
    "extract_audio",
    "export_qti",
    "filter_document",
    "filter_questions",
    "process",
    "read_questions",
    "read_quiz",
    "register_backend",
    "resolve_backend",
    "result_to_json",
    "run",
    "stage",
    "validate_quiz",
    "write_questions",
]

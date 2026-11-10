"""QuestXtract: extract spoken questions from lecture recordings."""

from importlib.metadata import PackageNotFoundError, version

from .asr import AsrBackend, register_backend, resolve_backend
from .config import Config
from .exerplaza import ExerPlazaExportError, ExerPlazaImport, ExerPlazaItem, export_exerplaza
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
from .questions_tex import QuestionsTexError, export_questions_tex
from .quiz_schema import read_quiz, validate_quiz
from .schema import Question
from .tex import TexExportError, export_tex

try:
    __version__ = version("questxtract")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "STAGES",
    "AsrBackend",
    "AsrSegment",
    "Config",
    "Context",
    "ExerPlazaExportError",
    "ExerPlazaImport",
    "ExerPlazaItem",
    "Layout",
    "MediaError",
    "QAPair",
    "QuestionsTexError",
    "QtiExportError",
    "Question",
    "QuestionCandidate",
    "SchemaError",
    "Segment",
    "Sentence",
    "TexExportError",
    "Transcript",
    "VisualContext",
    "Word",
    "__version__",
    "export_exerplaza",
    "extract_audio",
    "export_qti",
    "export_questions_tex",
    "export_tex",
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

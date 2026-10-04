"""Version-independent QTI intermediate representation.

This module deliberately contains no XML or package-format code.  Version-specific
serializers live in :mod:`questxtract.qti_v21` and :mod:`questxtract.qti_v3`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .export import read_questions
from .models import SchemaError

_XML_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class QtiExportError(SchemaError):
    """Raised when a questions document cannot be represented as QTI."""


@dataclass(frozen=True, kw_only=True)
class AssessmentOption:
    """An option belonging to a choice interaction."""

    identifier: str
    text: str


@dataclass(frozen=True, kw_only=True)
class ResponseDeclaration:
    """A response variable and its optional answer key."""

    identifier: str
    base_type: str
    cardinality: str
    correct_values: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Interaction:
    """The interaction presented to a learner."""

    interaction_type: str
    response_identifier: str
    min_choices: int | None = None
    max_choices: int | None = None


@dataclass(frozen=True, kw_only=True)
class OutcomeDeclaration:
    """An outcome variable produced by response processing."""

    identifier: str
    base_type: str = "float"
    cardinality: str = "single"
    default_value: str | None = None


@dataclass(frozen=True, kw_only=True)
class ResponseProcessing:
    """Version-neutral response-processing intent."""

    template: str | None = None


@dataclass(frozen=True, kw_only=True)
class Feedback:
    """Feedback shown for a response outcome."""

    correct: str | None = None
    incorrect: str | None = None
    general: str | None = None


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """Links an exported item back to QuestXtract source data."""

    source_question_id: str
    source_sentence_id: str | None = None
    source_answer: str | None = None
    latex: str | None = None


@dataclass(frozen=True, kw_only=True)
class AssessmentItem:
    """Common QTI assessment-item representation, independent of QTI version."""

    identifier: str
    title: str
    prompt: str
    language: str
    linguistic_question_type: str
    response_declarations: tuple[ResponseDeclaration, ...]
    interaction: Interaction
    options: tuple[AssessmentOption, ...] = ()
    outcome_declarations: tuple[OutcomeDeclaration, ...] = ()
    response_processing: ResponseProcessing | None = None
    feedback: Feedback | None = None
    provenance: Provenance | None = None
    metadata: tuple[tuple[str, str], ...] = ()
    score: float | None = None

    @property
    def interaction_type(self) -> str:
        """Compatibility alias for the interaction's QTI type."""
        return self.interaction.interaction_type

    @property
    def correct_response(self) -> str | None:
        """Compatibility alias for the first answer-key value."""
        declaration = self.response_declarations[0]
        return declaration.correct_values[0] if declaration.correct_values else None

    @property
    def source_question_id(self) -> str:
        """Compatibility alias for provenance."""
        return self.provenance.source_question_id if self.provenance else ""

    @property
    def source_sentence_id(self) -> str | None:
        """Compatibility alias for provenance."""
        return self.provenance.source_sentence_id if self.provenance else None

    @property
    def source_answer(self) -> str | None:
        """Compatibility alias for provenance."""
        return self.provenance.source_answer if self.provenance else None

    @property
    def latex(self) -> str | None:
        """Compatibility alias for provenance."""
        return self.provenance.latex if self.provenance else None


@dataclass(frozen=True, kw_only=True)
class AssessmentTest:
    """Common QTI assessment-test representation."""

    identifier: str
    title: str
    language: str
    items: tuple[AssessmentItem, ...]
    metadata: tuple[tuple[str, str], ...] = ()


def questions_to_qti(
    document: dict[str, Any],
    *,
    only_keep: bool = False,
    language: str | None = None,
) -> AssessmentTest:
    """Convert a questions document to the shared QTI representation."""
    if not isinstance(document, dict) or not isinstance(document.get("questions"), list):
        raise QtiExportError("expected an object with a 'questions' list")
    config = document.get("config")
    if config is not None and not isinstance(config, dict):
        raise QtiExportError("'config' must be an object")
    default_language = language or str((config or {}).get("language") or "en")
    selected: list[AssessmentItem] = []
    seen: set[str] = set()
    for index, raw in enumerate(document["questions"]):
        if not isinstance(raw, dict):
            raise QtiExportError(f"questions[{index}]: expected an object")
        review = raw.get("review")
        status = review.get("status") if isinstance(review, dict) else None
        if status in {"deleted", "false_positive"} or (only_keep and raw.get("verdict") != "keep"):
            continue
        item = _item_from_record(raw, default_language, index)
        if item.identifier in seen:
            raise QtiExportError(f"duplicate question identifier: {item.identifier}")
        seen.add(item.identifier)
        selected.append(item)
    return AssessmentTest(
        identifier="questxtract-test",
        title="QuestXtract questions",
        language=default_language,
        items=tuple(selected),
        metadata=(("format", "questxtract-qti"),),
    )


def _item_from_record(raw: dict[str, Any], language: str, index: int) -> AssessmentItem:
    question_id = raw.get("id")
    prompt = raw.get("text")
    if not isinstance(question_id, str) or not question_id:
        raise QtiExportError(f"questions[{index}].id: expected a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise QtiExportError(f"questions[{index}].text: expected a non-empty string")
    identifier = f"qti-{question_id}"
    if not _XML_ID.fullmatch(identifier):
        raise QtiExportError(f"questions[{index}].id: invalid XML identifier {question_id!r}")

    kind = raw.get("interaction_type", "extended_text")
    if kind not in {"choice_single", "choice_multiple", "extended_text"}:
        raise QtiExportError(f"questions[{index}].interaction_type: unsupported value {kind!r}")

    raw_options = raw.get("options", [])
    if not isinstance(raw_options, list):
        raise QtiExportError(f"questions[{index}].options: expected an array")
    options: list[AssessmentOption] = []
    option_ids: set[str] = set()
    for option_index, option in enumerate(raw_options):
        if (
            not isinstance(option, dict)
            or not isinstance(option.get("id"), str)
            or not isinstance(option.get("text"), str)
        ):
            raise QtiExportError(f"questions[{index}]: choice options require id and text strings")
        if not option["id"] or not option["text"].strip() or not _XML_ID.fullmatch(option["id"]):
            raise QtiExportError(f"questions[{index}].options[{option_index}]: invalid identifier")
        if option["id"] in option_ids:
            raise QtiExportError(
                f"questions[{index}].options[{option_index}]: duplicate identifier"
            )
        option_ids.add(option["id"])
        options.append(AssessmentOption(identifier=option["id"], text=option["text"]))

    if kind == "extended_text" and options:
        raise QtiExportError(
            f"questions[{index}]: extended_text does not support choice options"
        )
    if kind in {"choice_single", "choice_multiple"} and len(options) < 2:
        raise QtiExportError(
            f"questions[{index}]: {kind} requires at least two options"
        )
    interaction_type = {
        "choice_single": "choiceInteraction",
        "choice_multiple": "choiceInteraction",
        "extended_text": "extendedTextInteraction",
    }.get(str(kind), "extendedTextInteraction")
    cardinality = "multiple" if kind == "choice_multiple" else "single"
    base_type = "identifier" if kind in {"choice_single", "choice_multiple"} else "string"
    correct = raw.get("correct_response")
    if correct is not None and not isinstance(correct, dict):
        raise QtiExportError(f"questions[{index}]: correct responses require an object or null")
    correct_values = tuple(correct.get("option_ids", ())) if correct else ()
    if correct and not correct_values:
        raise QtiExportError(
            f"questions[{index}].correct_response.option_ids: expected at least one option"
        )
    if kind == "extended_text" and correct_values:
        raise QtiExportError(
            f"questions[{index}]: extended_text does not support option answer keys"
        )
    if kind == "choice_single" and len(correct_values) > 1:
        raise QtiExportError(
            f"questions[{index}].correct_response.option_ids: choice_single accepts one option"
        )
    if any(not isinstance(value, str) or value not in option_ids for value in correct_values):
        raise QtiExportError(f"questions[{index}].correct_response.option_ids: unknown option")
    declarations = (
        ResponseDeclaration(
            identifier="RESPONSE",
            base_type=base_type,
            cardinality=cardinality,
            correct_values=correct_values,
        ),
    )
    outcome = (OutcomeDeclaration(identifier="SCORE", default_value="0"),)
    feedback_raw = raw.get("feedback")
    feedback = (
        Feedback(**{key: feedback_raw.get(key) for key in ("correct", "incorrect", "general")})
        if isinstance(feedback_raw, dict)
        else None
    )
    answer = raw.get("answer")
    if answer is not None and not isinstance(answer, dict):
        raise QtiExportError(f"questions[{index}].answer: expected an object or null")
    visual = raw.get("visual")
    if visual is not None and not isinstance(visual, dict):
        raise QtiExportError(f"questions[{index}].visual: expected an object or null")
    provenance = Provenance(
        source_question_id=question_id,
        source_sentence_id=raw.get("sentence_id"),
        source_answer=answer.get("text") if answer else None,
        latex=visual.get("latex") if visual else None,
    )
    return AssessmentItem(
        identifier=identifier,
        title=f"Question {question_id}",
        prompt=prompt,
        language=language,
        linguistic_question_type=str(raw.get("question_type", "other")),
        response_declarations=declarations,
        interaction=Interaction(
            interaction_type=interaction_type,
            response_identifier="RESPONSE",
            min_choices=1 if options else None,
            max_choices=1 if options and kind == "choice_single" else None,
        ),
        options=tuple(options),
        outcome_declarations=outcome,
        response_processing=(
            ResponseProcessing(template="match_correct") if correct_values else None
        ),
        feedback=feedback,
        provenance=provenance,
        metadata=(("confidence", str(raw.get("confidence", ""))),),
        score=float(raw["points"]) if isinstance(raw.get("points"), (int, float)) else None,
    )


def export_qti(
    questions_path: str,
    output_path: str,
    *,
    only_keep: bool = False,
    language: str | None = None,
) -> None:
    """Backward-compatible entry point for the QTI 3.0 serializer."""
    from .qti_v3 import write_qti_package

    try:
        document = read_questions(questions_path)
    except (OSError, SchemaError) as exc:
        raise QtiExportError(f"{questions_path}: {exc}") from exc
    write_qti_package(document, output_path, only_keep=only_keep, language=language)


# Compatibility exports: the historical ``questxtract.qti`` API is QTI 3.0.
QTI_NS = "http://www.imsglobal.org/xsd/imsqtiasi_v3p0"


def write_qti_package(
    document: dict[str, Any],
    path: str,
    *,
    only_keep: bool = False,
    language: str | None = None,
) -> None:
    """Write a QTI 3.0 package using the versioned serializer."""
    from .qti_v3 import write_qti_package as write_v3

    write_v3(document, path, only_keep=only_keep, language=language)

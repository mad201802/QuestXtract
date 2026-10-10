"""QTI assessment model and the ``questions.json`` mapping.

This module holds no XML and no package code: :mod:`questxtract.qti_v3` serializes the model
and imports this module, never the other way round.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import SchemaError

_XML_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
# Characters XML 1.0 does not allow, even escaped; ElementTree writes them unchecked.
_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f￾￿\ud800-\udfff]")


class QtiExportError(SchemaError):
    """Raised when a questions file cannot be read or exported as QTI, or the package written."""


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
    """An outcome variable produced by response processing or by a human scorer."""

    identifier: str
    base_type: str = "float"
    cardinality: str = "single"
    default_value: str | None = None
    normal_maximum: float | None = None
    normal_minimum: float | None = None
    external_scored: str | None = None


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


@dataclass(frozen=True, kw_only=True)
class AssessmentItem:
    """One QTI assessment item."""

    identifier: str
    title: str
    prompt: str
    language: str
    linguistic_question_type: str
    response_declarations: tuple[ResponseDeclaration, ...]
    interaction: Interaction
    options: tuple[AssessmentOption, ...] = ()
    feedback: Feedback | None = None
    provenance: Provenance | None = None
    metadata: tuple[tuple[str, str], ...] = ()
    points: float | None = None

    @property
    def has_answer_key(self) -> bool:
        """Whether a response can be scored by matching it against an answer key."""
        return any(declaration.correct_values for declaration in self.response_declarations)

    @property
    def outcome_declarations(self) -> tuple[OutcomeDeclaration, ...]:
        """``SCORE`` and ``MAXSCORE``, plus ``FEEDBACK`` when there is feedback to show.

        ``SCORE`` is scored externally, by a person, when there is no answer key.
        """
        maximum = 1.0 if self.points is None else float(self.points)
        normalised = self.points is not None and self.points > 0  # normal-maximum must be > 0
        outcomes = [
            OutcomeDeclaration(
                identifier="SCORE",
                default_value=repr(0.0),
                normal_maximum=maximum if normalised else None,
                normal_minimum=0.0 if normalised else None,
                external_scored=None if self.has_answer_key else "human",
            ),
            OutcomeDeclaration(identifier="MAXSCORE", default_value=repr(maximum)),
        ]
        feedback = self.feedback
        if feedback is not None and any(
            text is not None for text in (feedback.correct, feedback.incorrect, feedback.general)
        ):
            outcomes.append(
                OutcomeDeclaration(
                    identifier="FEEDBACK", base_type="identifier", default_value="general"
                )
            )
        return tuple(outcomes)


@dataclass(frozen=True, kw_only=True)
class AssessmentTest:
    """An ordered QTI assessment test."""

    identifier: str
    title: str
    items: tuple[AssessmentItem, ...]


def questions_to_qti(document: dict[str, Any], *, only_keep: bool = False) -> AssessmentTest:
    """Convert a ``questions.json`` document to an assessment test of text-entry items.

    Questions a reviewer marked ``deleted`` or ``false_positive`` are left out; with
    ``only_keep``, so is every question whose verdict is not ``keep``.
    """
    if not isinstance(document, dict) or not isinstance(document.get("questions"), list):
        raise QtiExportError("expected an object with a 'questions' list")
    config = document.get("config")
    if config is not None and not isinstance(config, dict):
        raise QtiExportError("'config' must be an object")
    language = str((config or {}).get("language") or "en")
    selected: list[AssessmentItem] = []
    seen: set[str] = set()
    for index, raw in enumerate(document["questions"]):
        if not isinstance(raw, dict):
            raise QtiExportError(f"questions[{index}]: expected an object")
        review = raw.get("review")
        status = review.get("status") if isinstance(review, dict) else None
        if status in {"deleted", "false_positive"} or (only_keep and raw.get("verdict") != "keep"):
            continue
        item = _item_from_question(raw, language, index)
        if item.identifier in seen:
            raise QtiExportError(f"duplicate question identifier: {item.identifier}")
        seen.add(item.identifier)
        selected.append(item)
    return AssessmentTest(
        identifier="questxtract-test",
        title="QuestXtract questions",
        items=tuple(selected),
    )


# ponytail: #79, read choice items, points and feedback from quiz.json instead of questions.json
def _item_from_question(raw: dict[str, Any], language: str, index: int) -> AssessmentItem:
    """Build an extended-text item with no answer key from one question."""
    question_id = raw.get("id")
    if not isinstance(question_id, str) or not question_id:
        raise QtiExportError(f"questions[{index}].id: expected a non-empty string")
    identifier = f"qti-{question_id}"
    if not _XML_ID.fullmatch(identifier):
        raise QtiExportError(f"questions[{index}].id: invalid XML identifier {question_id!r}")
    prompt = raw.get("text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise QtiExportError(f"questions[{index}].text: expected a non-empty string")
    _check_xml_text(prompt, f"questions[{index}].text")
    review = raw.get("review")
    if isinstance(review, dict) and isinstance(review.get("text"), str) and review["text"].strip():
        prompt = review["text"]
        _check_xml_text(prompt, f"questions[{index}].review.text")
    answer = raw.get("answer")
    if answer is not None and not isinstance(answer, dict):
        raise QtiExportError(f"questions[{index}].answer: expected an object or null")
    answer_text = answer.get("text") if answer else None
    sentence_id = raw.get("sentence_id")
    return AssessmentItem(
        identifier=identifier,
        title=f"Question {question_id}",
        prompt=prompt,
        language=language,
        linguistic_question_type=str(raw.get("question_type", "other")),
        response_declarations=(
            ResponseDeclaration(identifier="RESPONSE", base_type="string", cardinality="single"),
        ),
        interaction=Interaction(
            interaction_type="extendedTextInteraction", response_identifier="RESPONSE"
        ),
        provenance=Provenance(
            source_question_id=question_id,
            source_sentence_id=sentence_id if isinstance(sentence_id, str) else None,
            source_answer=answer_text if isinstance(answer_text, str) else None,
        ),
        metadata=(("confidence", str(raw.get("confidence", ""))),),
    )


def _check_xml_text(text: str, field: str) -> None:
    illegal = _XML_ILLEGAL.search(text)
    if illegal:
        raise QtiExportError(
            f"{field}: character U+{ord(illegal.group()):04X} is not allowed in XML"
        )

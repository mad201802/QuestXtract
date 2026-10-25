"""QTI assessment model and the ``quiz.json`` mapping.

This module holds no XML and no package code. The writers, :mod:`questxtract.qti_v2` and
:mod:`questxtract.qti_v3`, serialize the model, and :mod:`questxtract.qti_package` writes the
ZIP; they import this module, never the other way round.

:func:`quiz_to_qti` takes a ``quiz.json`` that :func:`questxtract.quiz_schema.read_quiz` has
read. That checks the rules across fields, not the structure (the JSON Schema is enforced in
the tests only), so every field the mapping uses is checked here and reported with its path.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeGuard

from .models import SchemaError

_XML_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
# xs:language, the type of xml:lang.
_LANGUAGE = re.compile(r"^[A-Za-z]{1,8}(-[A-Za-z0-9]{1,8})*$")
# Characters XML 1.0 does not allow, even escaped; ElementTree writes them unchecked.
_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f￾￿\ud800-\udfff]")
# The package's JSON sidecar: every writer's manifest lists it, and qti_package writes it.
METADATA_PATH = "metadata/questxtract.json"


class QtiExportError(SchemaError):
    """Raised when a quiz file cannot be read or exported as QTI, or the package written."""


@dataclass(frozen=True, kw_only=True)
class AssessmentOption:
    """An option of the choice interaction, with the feedback shown when it is chosen."""

    identifier: str
    text: str
    feedback: str


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """Links an exported item back to the lecture: the source question and its seconds."""

    source_question_id: str
    start: float
    end: float


@dataclass(frozen=True, kw_only=True)
class AssessmentItem:
    """One single-choice QTI assessment item.

    ``correct_option`` is the identifier of the one correct option. ``explanation`` is shown
    after every option's own feedback. ``metadata`` goes to the package's JSON sidecar as it is.
    """

    identifier: str
    title: str
    prompt: str
    language: str
    options: tuple[AssessmentOption, ...]
    correct_option: str
    shuffle_options: bool
    explanation: str
    points: float
    provenance: Provenance
    metadata: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class AssessmentTest:
    """An ordered QTI assessment test.

    ``metadata`` goes to the package's JSON sidecar as it is.
    """

    identifier: str
    title: str
    items: tuple[AssessmentItem, ...]
    instructions: str | None = None
    shuffle_items: bool = False
    show_feedback: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def quiz_to_qti(document: Mapping[str, Any]) -> AssessmentTest:
    """Convert a ``quiz.json`` document to an assessment test of single-choice items.

    Items a reviewer marked ``rejected`` are left out; ``pending`` and ``accepted`` are kept.

    Raises:
        QtiExportError: Naming the field, if one the mapping uses is missing, has the wrong
            type, is not a valid XML identifier or language, or holds a character XML does not
            allow, or if an item's ``interaction_type`` is not ``choice_single``. A newer 1.x
            ``quiz.json`` may add interaction types; they fail here instead of being exported
            as single choice.

    """
    if not isinstance(document, Mapping) or not isinstance(document.get("questions"), list):
        raise QtiExportError("expected an object with a 'questions' list")
    identifier = _identifier(document.get("id"), "id")
    language = document.get("language")
    if not isinstance(language, str) or not _LANGUAGE.fullmatch(language):
        raise QtiExportError(f"language: expected a language tag such as 'en', got {language!r}")
    instructions = _object(document.get("instructions"), "instructions")
    shuffle_options = _flag(instructions, "shuffle_options", "instructions")
    instructions_text = instructions.get("text")
    if instructions_text is not None:
        instructions_text = _string(instructions_text, "instructions.text")

    items: list[AssessmentItem] = []
    seen = {identifier}
    for index, raw in enumerate(document["questions"]):
        path = f"questions[{index}]"
        item = _object(raw, path)
        review = _object(item.get("review"), f"{path}.review")
        if review.get("status") == "rejected":
            continue
        exported = _item(item, path, language, shuffle_options)
        if exported.identifier in seen:
            raise QtiExportError(f"{path}.id: {exported.identifier!r} is already in use")
        seen.add(exported.identifier)
        items.append(exported)

    return AssessmentTest(
        identifier=identifier,
        title=_string(document.get("title"), "title"),
        items=tuple(items),
        instructions=instructions_text,
        shuffle_items=_flag(instructions, "shuffle_questions", "instructions"),
        show_feedback=_flag(instructions, "show_feedback", "instructions"),
        metadata={
            key: document.get(key)
            for key in ("id", "schema_version", "status", "description", "source", "generation")
        },
    )


def _item(
    item: Mapping[str, Any], path: str, language: str, shuffle_options: bool
) -> AssessmentItem:
    """Build one single-choice item from a ``quiz.json`` item."""
    interaction_type = item.get("interaction_type")
    if interaction_type != "choice_single":
        raise QtiExportError(
            f"{path}.interaction_type: {interaction_type!r} is not supported; "
            "only 'choice_single' is supported"
        )
    identifier = _identifier(item.get("id"), f"{path}.id")
    prompt, prompt_origin = _text(item.get("prompt"), f"{path}.prompt")
    solution = _object(item.get("solution"), f"{path}.solution")
    explanation, explanation_origin = _text(
        solution.get("explanation"), f"{path}.solution.explanation"
    )
    explained = _object(solution.get("option_explanations"), f"{path}.solution.option_explanations")

    raw_options = item.get("options")
    if not isinstance(raw_options, list) or not raw_options:
        raise QtiExportError(f"{path}.options: expected a non-empty list")
    options: list[AssessmentOption] = []
    origins: dict[str, dict[str, str]] = {"options": {}, "option_explanations": {}}
    for index, raw_option in enumerate(raw_options):
        option_path = f"{path}.options[{index}]"
        option = _object(raw_option, option_path)
        option_id = _identifier(option.get("id"), f"{option_path}.id")
        text, text_origin = _text(option.get("text"), f"{option_path}.text")
        feedback, feedback_origin = _text(
            explained.get(option_id), f"{path}.solution.option_explanations.{option_id}"
        )
        if any(existing.identifier == option_id for existing in options):
            raise QtiExportError(f"{option_path}.id: {option_id!r} repeats another option")
        options.append(AssessmentOption(identifier=option_id, text=text, feedback=feedback))
        origins["options"][option_id] = text_origin
        origins["option_explanations"][option_id] = feedback_origin

    correct = _object(item.get("correct_response"), f"{path}.correct_response")
    option_ids = correct.get("option_ids")
    if (
        not isinstance(option_ids, list)
        or len(option_ids) != 1
        or option_ids[0] not in {option.identifier for option in options}
    ):
        raise QtiExportError(
            f"{path}.correct_response.option_ids: expected the id of one option, got {option_ids!r}"
        )

    points = item.get("points")
    if not _is_number(points) or points < 0:
        raise QtiExportError(f"{path}.points: expected a finite number >= 0, got {points!r}")
    provenance = _object(item.get("provenance"), f"{path}.provenance")
    source_id = _string(
        provenance.get("source_question_id"), f"{path}.provenance.source_question_id"
    )
    start, end = provenance.get("start"), provenance.get("end")
    if not _is_number(start) or not _is_number(end):
        raise QtiExportError(f"{path}.provenance: expected finite numbers for start and end")
    review = _object(item.get("review"), f"{path}.review")

    return AssessmentItem(
        identifier=identifier,
        title=f"Question {source_id}",
        prompt=prompt,
        language=language,
        options=tuple(options),
        correct_option=option_ids[0],
        shuffle_options=shuffle_options,
        explanation=explanation,
        points=float(points),
        provenance=Provenance(source_question_id=source_id, start=float(start), end=float(end)),
        metadata={
            "review_status": review.get("status"),
            "origin": {
                "prompt": prompt_origin,
                "explanation": explanation_origin,
                **origins,
            },
        },
    )


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QtiExportError(f"{path}: expected an object")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QtiExportError(f"{path}: expected a non-empty string")
    _check_xml_text(value, path)
    return value


def _text(value: object, path: str) -> tuple[str, str]:
    """Return the text and the origin of a ``{text, origin}`` object."""
    text = _object(value, path)
    origin = text.get("origin")
    if origin not in {"spoken", "generated"}:
        raise QtiExportError(f"{path}.origin: expected 'spoken' or 'generated', got {origin!r}")
    return _string(text.get("text"), f"{path}.text"), origin


def _identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or not _XML_ID.fullmatch(value):
        raise QtiExportError(f"{path}: invalid XML identifier {value!r}")
    return value


def _flag(container: Mapping[str, Any], key: str, path: str) -> bool:
    value = container.get(key)
    if not isinstance(value, bool):
        raise QtiExportError(f"{path}.{key}: expected true or false")
    return value


def _is_number(value: object) -> TypeGuard[float]:
    """Whether ``value`` is an int or float that XML can carry: not a bool, NaN or infinity.

    ``json.loads`` reads ``NaN`` and ``Infinity``; written as ``nan`` they are no ``xs:double``.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_xml_text(text: str, field: str) -> None:
    illegal = _XML_ILLEGAL.search(text)
    if illegal:
        raise QtiExportError(
            f"{field}: character U+{ord(illegal.group()):04X} is not allowed in XML"
        )

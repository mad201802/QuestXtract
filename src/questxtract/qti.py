"""QTI 3.0 export for the existing ``questions.json`` format.

The first QTI profile deliberately supports only free-text responses.  The
linguistic ``question_type`` from QuestXtract is retained as metadata and is
never used to infer a QTI interaction or a correct answer.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree

from .export import read_questions
from .models import SchemaError

QTI_NS = "http://www.imsglobal.org/xsd/imsqtiasi_v3p0"
CP_NS = "http://www.imsglobal.org/xsd/qti/qtiv3p0/imscp_v1p1"
QTI_PREFIX = f"{{{QTI_NS}}}"
CP_PREFIX = f"{{{CP_NS}}}"
_XML_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")

ElementTree.register_namespace("", QTI_NS)
ElementTree.register_namespace("", CP_NS)


class QtiExportError(SchemaError):
    """Raised when a questions document cannot be represented as QTI."""


@dataclass(frozen=True, kw_only=True)
class AssessmentOption:
    """An option for a future objective interaction profile."""

    identifier: str
    text: str


@dataclass(frozen=True, kw_only=True)
class AssessmentItem:
    """The small, explicit intermediate representation used by this exporter."""

    identifier: str
    title: str
    prompt: str
    language: str
    linguistic_question_type: str
    interaction_type: str = "extendedTextInteraction"
    correct_response: str | None = None
    options: tuple[AssessmentOption, ...] = ()
    score: float | None = None
    feedback: str | None = None
    source_question_id: str = ""
    source_sentence_id: str | None = None
    source_answer: str | None = None
    latex: str | None = None


@dataclass(frozen=True, kw_only=True)
class AssessmentTest:
    """A QTI test and its ordered assessment items."""

    identifier: str
    title: str
    language: str
    items: tuple[AssessmentItem, ...]


def questions_to_qti(
    document: dict[str, Any],
    *,
    only_keep: bool = False,
    language: str | None = None,
) -> AssessmentTest:
    """Convert a ``questions.json`` document to the supported QTI model.

    By default every question is exported except those explicitly rejected by a reviewer.
    ``only_keep`` opts into exporting only records judged ``keep``.
    """
    if not isinstance(document, dict) or not isinstance(document.get("questions"), list):
        raise QtiExportError("expected an object with a 'questions' list")
    config = document.get("config")
    if config is not None and not isinstance(config, dict):
        raise QtiExportError("'config' must be an object")
    selected: list[AssessmentItem] = []
    seen: set[str] = set()
    default_language = language or str((config or {}).get("language") or "en")
    for index, raw in enumerate(document["questions"]):
        if not isinstance(raw, dict):
            raise QtiExportError(f"questions[{index}]: expected an object")
        review = raw.get("review")
        review_status = review.get("status") if isinstance(review, dict) else None
        if review_status in {"deleted", "false_positive"}:
            continue
        if only_keep and raw.get("verdict") != "keep":
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
    )


def write_qti_package(
    document: dict[str, Any],
    path: str | Path,
    *,
    only_keep: bool = False,
    language: str | None = None,
) -> None:
    """Write a deterministic QTI 3.0 ZIP package."""
    assessment = questions_to_qti(document, only_keep=only_keep, language=language)
    output = Path(path)
    try:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as package:
            item_paths = []
            for item in assessment.items:
                item_path = f"items/{item.identifier}.xml"
                item_paths.append(item_path)
                package.writestr(item_path, _xml_bytes(_item_xml(item)))
            package.writestr("assessmentTest.xml", _xml_bytes(_test_xml(assessment)))
            package.writestr("imsmanifest.xml", _xml_bytes(_manifest_xml(assessment, item_paths)))
            package.writestr("metadata/questxtract.json", _metadata_json(assessment))
    except OSError as exc:
        raise QtiExportError(f"{output}: cannot write QTI package ({exc})") from exc


def export_qti(
    questions_path: str | Path,
    output_path: str | Path,
    *,
    only_keep: bool = False,
    language: str | None = None,
) -> None:
    """Read ``questions.json`` and write a QTI package."""
    try:
        document = read_questions(questions_path)
    except (OSError, SchemaError) as exc:
        raise QtiExportError(f"{questions_path}: {exc}") from exc
    write_qti_package(
        document,
        output_path,
        only_keep=only_keep,
        language=language,
    )


def _item_from_record(raw: dict[str, Any], language: str, index: int) -> AssessmentItem:
    question_id = raw.get("id")
    prompt = raw.get("text")
    if not isinstance(question_id, str) or not question_id:
        raise QtiExportError(f"questions[{index}].id: expected a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise QtiExportError(f"questions[{index}].text: expected a non-empty string")
    if raw.get("options"):
        raise QtiExportError(
            f"questions[{index}]: choice options are not supported by the QTI v1 profile"
        )
    if raw.get("correct_response") is not None:
        raise QtiExportError(
            f"questions[{index}]: objective correct responses are not supported "
            "by the QTI v1 profile"
        )
    identifier = f"qti-{question_id}"
    if not _XML_ID.fullmatch(identifier):
        raise QtiExportError(f"questions[{index}].id: invalid XML identifier {question_id!r}")
    answer = raw.get("answer")
    if answer is not None and not isinstance(answer, dict):
        raise QtiExportError(f"questions[{index}].answer: expected an object or null")
    source_answer = answer.get("text") if answer else None
    if source_answer is not None and not isinstance(source_answer, str):
        raise QtiExportError(f"questions[{index}].answer.text: expected a string")
    visual = raw.get("visual")
    if visual is not None and not isinstance(visual, dict):
        raise QtiExportError(f"questions[{index}].visual: expected an object or null")
    latex = visual.get("latex") if visual else None
    if latex is not None and not isinstance(latex, str):
        raise QtiExportError(f"questions[{index}].visual.latex: expected a string or null")
    confidence = raw.get("confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise QtiExportError(f"questions[{index}].confidence: expected a number or null")
    return AssessmentItem(
        identifier=identifier,
        title=f"Question {question_id}",
        prompt=prompt,
        language=language,
        linguistic_question_type=str(raw.get("question_type", "other")),
        source_question_id=question_id,
        source_sentence_id=raw.get("sentence_id"),
        source_answer=source_answer,
        latex=latex,
        score=float(confidence) if confidence is not None else None,
    )


def _item_xml(item: AssessmentItem) -> ElementTree.Element:
    root = ElementTree.Element(
        f"{QTI_PREFIX}qti-assessment-item",
        {
            "identifier": item.identifier,
            "title": item.title,
            "time-dependent": "false",
            "{http://www.w3.org/XML/1998/namespace}lang": item.language,
        },
    )
    ElementTree.SubElement(
        root,
        f"{QTI_PREFIX}qti-response-declaration",
        {"identifier": "RESPONSE", "base-type": "string", "cardinality": "single"},
    )
    body = ElementTree.SubElement(root, f"{QTI_PREFIX}qti-item-body")
    paragraph = ElementTree.SubElement(body, "p")
    paragraph.text = item.prompt
    ElementTree.SubElement(
        body,
        f"{QTI_PREFIX}qti-extended-text-interaction",
        {"response-identifier": "RESPONSE"},
    )
    return root


def _test_xml(assessment: AssessmentTest) -> ElementTree.Element:
    root = ElementTree.Element(
        f"{QTI_PREFIX}qti-assessment-test",
        {
            "identifier": assessment.identifier,
            "title": assessment.title,
            "{http://www.w3.org/XML/1998/namespace}lang": assessment.language,
        },
    )
    part = ElementTree.SubElement(
        root,
        f"{QTI_PREFIX}qti-test-part",
        {"identifier": "part-1", "navigation-mode": "linear", "submission-mode": "individual"},
    )
    section = ElementTree.SubElement(
        part,
        f"{QTI_PREFIX}qti-assessment-section",
        {"identifier": "section-1", "title": "Questions", "visible": "true"},
    )
    for item in assessment.items:
        ElementTree.SubElement(
            section,
            f"{QTI_PREFIX}qti-assessment-item-ref",
            {"identifier": f"ref-{item.identifier}", "href": f"items/{item.identifier}.xml"},
        )
    return root


def _manifest_xml(assessment: AssessmentTest, item_paths: list[str]) -> ElementTree.Element:
    root = ElementTree.Element(
        f"{CP_PREFIX}manifest",
        {
            "identifier": "questxtract-manifest",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        },
    )
    metadata = ElementTree.SubElement(root, f"{CP_PREFIX}metadata")
    ElementTree.SubElement(metadata, f"{CP_PREFIX}schema").text = "QTI Package"
    ElementTree.SubElement(metadata, f"{CP_PREFIX}schemaversion").text = "3.0.0"
    ElementTree.SubElement(root, f"{CP_PREFIX}organizations")
    resources = ElementTree.SubElement(root, f"{CP_PREFIX}resources")
    test_resource = ElementTree.SubElement(
        resources,
        f"{CP_PREFIX}resource",
        {
            "identifier": assessment.identifier,
            "type": "imsqti_test_xmlv3p0",
            "href": "assessmentTest.xml",
        },
    )
    ElementTree.SubElement(test_resource, f"{CP_PREFIX}file", {"href": "assessmentTest.xml"})
    for item, item_path in zip(assessment.items, item_paths, strict=True):
        resource = ElementTree.SubElement(
            resources,
            f"{CP_PREFIX}resource",
            {"identifier": item.identifier, "type": "imsqti_item_xmlv3p0", "href": item_path},
        )
        ElementTree.SubElement(resource, f"{CP_PREFIX}file", {"href": item_path})
        ElementTree.SubElement(
            test_resource, f"{CP_PREFIX}dependency", {"identifierref": item.identifier}
        )
    return root


def _metadata_json(assessment: AssessmentTest) -> bytes:
    payload = {
        "format": "questxtract-qti-3.0",
        "items": {
            item.identifier: {
                "source_question_id": item.source_question_id,
                "source_sentence_id": item.source_sentence_id,
                "linguistic_question_type": item.linguistic_question_type,
                "interaction_type": item.interaction_type,
                "source_answer": item.source_answer,
                "score": item.score,
                "latex": item.latex,
            }
            for item in assessment.items
        },
    }
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _xml_bytes(root: ElementTree.Element) -> bytes:
    return cast(bytes, ElementTree.tostring(root, encoding="utf-8", xml_declaration=True))

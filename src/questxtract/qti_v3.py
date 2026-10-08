"""QTI 3.0 serializer for the common QuestXtract QTI model."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree

from .models import SchemaError
from .qti import AssessmentItem, AssessmentTest, QtiExportError, questions_to_qti

QTI_NS = "http://www.imsglobal.org/xsd/imsqtiasi_v3p0"
CP_NS = "http://www.imsglobal.org/xsd/qti/qtiv3p0/imscp_v1p1"
QTI_PREFIX = f"{{{QTI_NS}}}"
CP_PREFIX = f"{{{CP_NS}}}"
MATCH_CORRECT_TEMPLATE = "http://www.imsglobal.org/question/qti_v3p0/rptemplates/match_correct"
ElementTree.register_namespace("qti", QTI_NS)
ElementTree.register_namespace("cp", CP_NS)


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
            paths = [f"items/{item.identifier}.xml" for item in assessment.items]
            for item, item_path in zip(assessment.items, paths, strict=True):
                package.writestr(item_path, _xml_bytes(_item_xml(item)))
            package.writestr("assessmentTest.xml", _xml_bytes(_test_xml(assessment)))
            package.writestr("imsmanifest.xml", _xml_bytes(_manifest_xml(assessment, paths)))
            package.writestr("metadata/questxtract.json", _metadata_json(assessment))
    except (OSError, TypeError, ValueError) as exc:
        raise QtiExportError(f"{output}: cannot write QTI package ({exc})") from exc


def export_qti(
    questions_path: str | Path,
    output_path: str | Path,
    *,
    only_keep: bool = False,
    language: str | None = None,
) -> None:
    """Read a questions JSON file and write a QTI 3.0 package."""
    from .export import read_questions

    try:
        document = read_questions(questions_path)
    except (OSError, SchemaError) as exc:
        raise QtiExportError(f"{questions_path}: {exc}") from exc
    write_qti_package(document, output_path, only_keep=only_keep, language=language)


def _item_xml(item: AssessmentItem) -> ElementTree.Element:
    root = ElementTree.Element(
        f"{QTI_PREFIX}qti-assessment-item",
        {
            "identifier": item.identifier,
            "title": item.title,
            "adaptive": "false",
            "time-dependent": "false",
            "{http://www.w3.org/XML/1998/namespace}lang": item.language,
        },
    )
    for declaration in item.response_declarations:
        node = ElementTree.SubElement(
            root,
            f"{QTI_PREFIX}qti-response-declaration",
            {
                "identifier": declaration.identifier,
                "base-type": declaration.base_type,
                "cardinality": declaration.cardinality,
            },
        )
        if declaration.correct_values:
            correct = ElementTree.SubElement(node, f"{QTI_PREFIX}qti-correct-response")
            for value in declaration.correct_values:
                ElementTree.SubElement(correct, f"{QTI_PREFIX}qti-value").text = value
    for outcome in item.outcome_declarations:
        node = ElementTree.SubElement(
            root,
            f"{QTI_PREFIX}qti-outcome-declaration",
            {
                "identifier": outcome.identifier,
                "base-type": outcome.base_type,
                "cardinality": outcome.cardinality,
            },
        )
        if outcome.default_value is not None:
            default_value = ElementTree.SubElement(node, f"{QTI_PREFIX}qti-default-value")
            ElementTree.SubElement(
                default_value, f"{QTI_PREFIX}qti-value"
            ).text = outcome.default_value
    body = ElementTree.SubElement(root, f"{QTI_PREFIX}qti-item-body")
    interaction_name = {
        "extendedTextInteraction": "extended-text-interaction",
        "choiceInteraction": "choice-interaction",
    }.get(item.interaction.interaction_type, item.interaction.interaction_type)
    interaction = ElementTree.SubElement(
        body,
        f"{QTI_PREFIX}qti-{interaction_name}",
        {
            "response-identifier": item.interaction.response_identifier,
            **(
                {"min-choices": str(item.interaction.min_choices)}
                if item.interaction.min_choices is not None
                else {}
            ),
            **(
                {"max-choices": str(item.interaction.max_choices)}
                if item.interaction.max_choices is not None
                else {}
            ),
        },
    )
    ElementTree.SubElement(interaction, f"{QTI_PREFIX}qti-prompt").text = item.prompt
    for option in item.options:
        ElementTree.SubElement(
            interaction,
            f"{QTI_PREFIX}qti-simple-choice",
            {
                "identifier": option.identifier,
            },
        ).text = option.text
    if item.response_processing is not None:
        processing = ElementTree.SubElement(root, f"{QTI_PREFIX}qti-response-processing")
        feedback_enabled = any(
            outcome.identifier == "FEEDBACK" for outcome in item.outcome_declarations
        )
        if not feedback_enabled:
            processing.set("template", MATCH_CORRECT_TEMPLATE)
        else:
            correct_feedback = _feedback_identifier(item, "correct")
            incorrect_feedback = _feedback_identifier(item, "incorrect")
            condition = ElementTree.SubElement(processing, f"{QTI_PREFIX}qti-response-condition")
            response_if = ElementTree.SubElement(condition, f"{QTI_PREFIX}qti-response-if")
            match = ElementTree.SubElement(response_if, f"{QTI_PREFIX}qti-match")
            ElementTree.SubElement(match, f"{QTI_PREFIX}qti-variable", {"identifier": "RESPONSE"})
            ElementTree.SubElement(match, f"{QTI_PREFIX}qti-correct", {"identifier": "RESPONSE"})
            _set_outcome(response_if, "SCORE", "float", "1")
            _set_outcome(response_if, "FEEDBACK", "identifier", correct_feedback)
            response_else = ElementTree.SubElement(condition, f"{QTI_PREFIX}qti-response-else")
            _set_outcome(response_else, "SCORE", "float", "0")
            _set_outcome(response_else, "FEEDBACK", "identifier", incorrect_feedback)
    if item.feedback is not None:
        for identifier, text in (
            ("correct", item.feedback.correct),
            ("incorrect", item.feedback.incorrect),
            ("general", item.feedback.general),
        ):
            if text is not None:
                feedback_node = ElementTree.SubElement(
                    root,
                    f"{QTI_PREFIX}qti-modal-feedback",
                    {
                        "identifier": identifier,
                        "outcome-identifier": "FEEDBACK",
                        "show-hide": "show",
                    },
                )
                ElementTree.SubElement(feedback_node, f"{QTI_PREFIX}qti-content-body").text = text
    return root


def _set_outcome(
    parent: ElementTree.Element,
    identifier: str,
    base_type: str,
    value: str,
) -> None:
    node = ElementTree.SubElement(
        parent,
        f"{QTI_PREFIX}qti-set-outcome-value",
        {"identifier": identifier},
    )
    ElementTree.SubElement(
        node,
        f"{QTI_PREFIX}qti-base-value",
        {"base-type": base_type},
    ).text = value


def _feedback_identifier(item: AssessmentItem, preferred: str) -> str:
    """Return a feedback identifier that has a corresponding modal element."""
    feedback = item.feedback
    if feedback is None:
        return "none"
    if getattr(feedback, preferred) is not None:
        return preferred
    if feedback.general is not None:
        return "general"
    return "none"


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
        {
            "identifier": "part-1",
            "navigation-mode": "linear",
            "submission-mode": "individual",
        },
    )
    section = ElementTree.SubElement(
        part,
        f"{QTI_PREFIX}qti-assessment-section",
        {
            "identifier": "section-1",
            "title": "Questions",
            "visible": "true",
        },
    )
    for item in assessment.items:
        ElementTree.SubElement(
            section,
            f"{QTI_PREFIX}qti-assessment-item-ref",
            {
                "identifier": f"ref-{item.identifier}",
                "href": f"items/{item.identifier}.xml",
            },
        )
    return root


def _manifest_xml(assessment: AssessmentTest, item_paths: list[str]) -> ElementTree.Element:
    root = ElementTree.Element(f"{CP_PREFIX}manifest", {"identifier": "questxtract-manifest"})
    metadata = ElementTree.SubElement(root, f"{CP_PREFIX}metadata")
    ElementTree.SubElement(metadata, f"{CP_PREFIX}schema").text = "QTI Package"
    ElementTree.SubElement(metadata, f"{CP_PREFIX}schemaversion").text = "3.0.0"
    ElementTree.SubElement(root, f"{CP_PREFIX}organizations")
    resources = ElementTree.SubElement(root, f"{CP_PREFIX}resources")
    test = ElementTree.SubElement(
        resources,
        f"{CP_PREFIX}resource",
        {
            "identifier": assessment.identifier,
            "type": "imsqti_test_xmlv3p0",
            "href": "assessmentTest.xml",
        },
    )
    ElementTree.SubElement(test, f"{CP_PREFIX}file", {"href": "assessmentTest.xml"})
    for item, path in zip(assessment.items, item_paths, strict=True):
        resource = ElementTree.SubElement(
            resources,
            f"{CP_PREFIX}resource",
            {
                "identifier": item.identifier,
                "type": "imsqti_item_xmlv3p0",
                "href": path,
            },
        )
        ElementTree.SubElement(resource, f"{CP_PREFIX}file", {"href": path})
        ElementTree.SubElement(test, f"{CP_PREFIX}dependency", {"identifierref": item.identifier})
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
                "metadata": dict(item.metadata),
            }
            for item in assessment.items
        },
    }
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _xml_bytes(root: ElementTree.Element) -> bytes:
    return cast(bytes, ElementTree.tostring(root, encoding="utf-8", xml_declaration=True))

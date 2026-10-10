"""QTI 3.0 package: the item, test and manifest XML, the ZIP layout and ``export_qti``.

The package is deterministic: the same assessment gives the same bytes, so every entry gets a
fixed timestamp and fixed attributes instead of the clock and the platform's.
"""

from __future__ import annotations

import io
import json
import stat
import zipfile
from pathlib import Path
from typing import cast
from xml.etree import ElementTree

from .export import read_questions
from .models import SchemaError
from .qti import AssessmentItem, AssessmentTest, QtiExportError, questions_to_qti

QTI_NS = "http://www.imsglobal.org/xsd/imsqtiasi_v3p0"
CP_NS = "http://www.imsglobal.org/xsd/qti/qtiv3p0/imscp_v1p1"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_QTI_XSD = "https://purl.imsglobal.org/spec/qti/v3p0/schema/xsd/imsqti_asiv3p0_v1p0.xsd"
_CP_XSD = "https://purl.imsglobal.org/spec/qti/v3p0/schema/xsd/imsqtiv3p0_imscpv1p2_v1p0.xsd"
# The namespaces are literal attributes on unqualified tags: ElementTree's
# tostring(default_namespace=...) rejects unqualified attributes, and register_namespace is
# process-global. Every official 1EdTech example uses the default namespace.
_QTI_ROOT = {"xmlns": QTI_NS, "xmlns:xsi": XSI_NS, "xsi:schemaLocation": f"{QTI_NS} {_QTI_XSD}"}
_CP_ROOT = {"xmlns": CP_NS, "xmlns:xsi": XSI_NS, "xsi:schemaLocation": f"{CP_NS} {_CP_XSD}"}
_INTERACTIONS = {
    "extendedTextInteraction": "qti-extended-text-interaction",
    "choiceInteraction": "qti-choice-interaction",
}
_METADATA_PATH = "metadata/questxtract.json"
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # the earliest date a ZIP entry can carry


def export_qti(
    questions_path: str | Path,
    output_path: str | Path,
    *,
    only_keep: bool = False,
) -> None:
    """Read a ``questions.json`` and write it as a QTI 3.0 package.

    Every question becomes a text-entry item scored by a person. Questions a reviewer marked
    ``deleted`` or ``false_positive`` are left out. The same input gives a byte-identical ZIP.

    Args:
        questions_path: A ``questions.json`` written by ``process``, edited or not.
        output_path: Where to write the ZIP. Its folder must exist.
        only_keep: Export only questions whose ``verdict`` is ``keep``.

    Raises:
        QtiExportError: If the questions file cannot be read or is not a ``questions.json``,
            if a question cannot be represented in QTI, or if the package cannot be written.

    """
    try:
        document = read_questions(questions_path)
    except OSError as exc:
        raise QtiExportError(f"{questions_path}: {exc.strerror or exc}") from exc
    except SchemaError as exc:
        raise QtiExportError(str(exc)) from exc
    write_qti_package(questions_to_qti(document, only_keep=only_keep), output_path)


def write_qti_package(assessment: AssessmentTest, path: str | Path) -> None:
    """Write an assessment test as a QTI 3.0 ZIP package."""
    data = _package_bytes(assessment)
    try:
        Path(path).write_bytes(data)
    except OSError as exc:
        raise QtiExportError(
            f"{path}: cannot write the QTI package ({exc.strerror or exc})"
        ) from exc


def _package_bytes(assessment: AssessmentTest) -> bytes:
    item_paths = [f"items/{item.identifier}.xml" for item in assessment.items]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        for item, item_path in zip(assessment.items, item_paths, strict=True):
            _write_entry(package, item_path, _xml_bytes(_item_xml(item)))
        _write_entry(package, "assessmentTest.xml", _xml_bytes(_test_xml(assessment)))
        _write_entry(package, "imsmanifest.xml", _xml_bytes(_manifest_xml(assessment, item_paths)))
        _write_entry(package, _METADATA_PATH, _metadata_json(assessment))
    return buffer.getvalue()


def _write_entry(package: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED  # ZipInfo defaults to STORED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.create_system = 3  # Unix, also on Windows, where ZipInfo would say 0
    package.writestr(info, data)


def _item_xml(item: AssessmentItem) -> ElementTree.Element:
    root = ElementTree.Element(
        "qti-assessment-item",
        {
            **_QTI_ROOT,
            "identifier": item.identifier,
            "title": item.title,
            "adaptive": "false",
            "time-dependent": "false",
            "xml:lang": item.language,
        },
    )
    for declaration in item.response_declarations:
        node = ElementTree.SubElement(
            root,
            "qti-response-declaration",
            {
                "identifier": declaration.identifier,
                "base-type": declaration.base_type,
                "cardinality": declaration.cardinality,
            },
        )
        if declaration.correct_values:
            correct = ElementTree.SubElement(node, "qti-correct-response")
            for value in declaration.correct_values:
                ElementTree.SubElement(correct, "qti-value").text = value
    for outcome in item.outcome_declarations:
        attributes = {
            "identifier": outcome.identifier,
            "base-type": outcome.base_type,
            "cardinality": outcome.cardinality,
        }
        if outcome.normal_maximum is not None:
            attributes["normal-maximum"] = _float(outcome.normal_maximum)
        if outcome.normal_minimum is not None:
            attributes["normal-minimum"] = _float(outcome.normal_minimum)
        if outcome.external_scored is not None:
            attributes["external-scored"] = outcome.external_scored
        node = ElementTree.SubElement(root, "qti-outcome-declaration", attributes)
        if outcome.default_value is not None:
            default_value = ElementTree.SubElement(node, "qti-default-value")
            ElementTree.SubElement(default_value, "qti-value").text = outcome.default_value
    body = ElementTree.SubElement(root, "qti-item-body")
    attributes = {"response-identifier": item.interaction.response_identifier}
    if item.interaction.min_choices is not None:
        attributes["min-choices"] = str(item.interaction.min_choices)
    if item.interaction.max_choices is not None:
        attributes["max-choices"] = str(item.interaction.max_choices)
    interaction = ElementTree.SubElement(
        body, _INTERACTIONS[item.interaction.interaction_type], attributes
    )
    ElementTree.SubElement(interaction, "qti-prompt").text = item.prompt
    for option in item.options:
        ElementTree.SubElement(
            interaction, "qti-simple-choice", {"identifier": option.identifier}
        ).text = option.text
    feedback_enabled = any(
        outcome.identifier == "FEEDBACK" for outcome in item.outcome_declarations
    )
    if item.has_answer_key:
        # Explicit Core response processing (Implementation Guide 3.4.4), not the
        # match_correct template: the template always scores 1, whatever the item is worth.
        processing = ElementTree.SubElement(root, "qti-response-processing")
        condition = ElementTree.SubElement(processing, "qti-response-condition")
        response_if = ElementTree.SubElement(condition, "qti-response-if")
        match = ElementTree.SubElement(response_if, "qti-match")
        ElementTree.SubElement(match, "qti-variable", {"identifier": "RESPONSE"})
        ElementTree.SubElement(match, "qti-correct", {"identifier": "RESPONSE"})
        score = ElementTree.SubElement(
            response_if, "qti-set-outcome-value", {"identifier": "SCORE"}
        )
        ElementTree.SubElement(score, "qti-variable", {"identifier": "MAXSCORE"})
        if feedback_enabled:
            _set_outcome(
                response_if, "FEEDBACK", "identifier", _feedback_identifier(item, "correct")
            )
        response_else = ElementTree.SubElement(condition, "qti-response-else")
        _set_outcome(response_else, "SCORE", "float", _float(0.0))
        if feedback_enabled:
            _set_outcome(
                response_else, "FEEDBACK", "identifier", _feedback_identifier(item, "incorrect")
            )
    if item.feedback is not None:
        for identifier, text in (
            ("correct", item.feedback.correct),
            ("incorrect", item.feedback.incorrect),
            ("general", item.feedback.general),
        ):
            if text is not None:
                feedback_node = ElementTree.SubElement(
                    root,
                    "qti-modal-feedback",
                    {
                        "identifier": identifier,
                        "outcome-identifier": "FEEDBACK",
                        "show-hide": "show",
                    },
                )
                ElementTree.SubElement(feedback_node, "qti-content-body").text = text
    return root


def _set_outcome(parent: ElementTree.Element, identifier: str, base_type: str, value: str) -> None:
    node = ElementTree.SubElement(parent, "qti-set-outcome-value", {"identifier": identifier})
    ElementTree.SubElement(node, "qti-base-value", {"base-type": base_type}).text = value


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
        "qti-assessment-test",
        {**_QTI_ROOT, "identifier": assessment.identifier, "title": assessment.title},
    )
    part = ElementTree.SubElement(
        root,
        "qti-test-part",
        {"identifier": "part-1", "navigation-mode": "linear", "submission-mode": "individual"},
    )
    section = ElementTree.SubElement(
        part,
        "qti-assessment-section",
        {"identifier": "section-1", "title": "Questions", "visible": "true"},
    )
    for item in assessment.items:
        ElementTree.SubElement(
            section,
            "qti-assessment-item-ref",
            {"identifier": f"ref-{item.identifier}", "href": f"items/{item.identifier}.xml"},
        )
    return root


def _manifest_xml(assessment: AssessmentTest, item_paths: list[str]) -> ElementTree.Element:
    root = ElementTree.Element("manifest", {**_CP_ROOT, "identifier": "questxtract-manifest"})
    metadata = ElementTree.SubElement(root, "metadata")
    ElementTree.SubElement(metadata, "schema").text = "QTI Package"
    # Implementation Guide 6.1.1 and the binding (section 8) say schemaversion MUST be 3.0.0,
    # and the 3.0.0 content-package XSD allows no other value.
    ElementTree.SubElement(metadata, "schemaversion").text = "3.0.0"
    ElementTree.SubElement(root, "organizations")
    resources = ElementTree.SubElement(root, "resources")
    test = ElementTree.SubElement(
        resources,
        "resource",
        {
            "identifier": assessment.identifier,
            "type": "imsqti_test_xmlv3p0",
            "href": "assessmentTest.xml",
        },
    )
    ElementTree.SubElement(test, "file", {"href": "assessmentTest.xml"})
    for item, path in zip(assessment.items, item_paths, strict=True):
        resource = ElementTree.SubElement(
            resources,
            "resource",
            {"identifier": item.identifier, "type": "imsqti_item_xmlv3p0", "href": path},
        )
        ElementTree.SubElement(resource, "file", {"href": path})
        ElementTree.SubElement(test, "dependency", {"identifierref": item.identifier})
    # Every file but the manifest needs a resource (Implementation Guide 6.1).
    sidecar = ElementTree.SubElement(
        resources,
        "resource",
        {"identifier": "questxtract-metadata", "type": "extension", "href": _METADATA_PATH},
    )
    ElementTree.SubElement(sidecar, "file", {"href": _METADATA_PATH})
    return root


def _metadata_json(assessment: AssessmentTest) -> bytes:
    items = {}
    for item in assessment.items:
        provenance = item.provenance
        items[item.identifier] = {
            "source_question_id": provenance.source_question_id if provenance else None,
            "source_sentence_id": provenance.source_sentence_id if provenance else None,
            "linguistic_question_type": item.linguistic_question_type,
            "interaction_type": item.interaction.interaction_type,
            "source_answer": provenance.source_answer if provenance else None,
            "metadata": dict(item.metadata),
        }
    payload = {"format": "questxtract-qti-3.0", "items": items}
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _float(value: float) -> str:
    return repr(float(value))


def _xml_bytes(root: ElementTree.Element) -> bytes:
    return cast(bytes, ElementTree.tostring(root, encoding="utf-8", xml_declaration=True))

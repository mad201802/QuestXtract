"""QTI 3.0 XML: the items, the test and the manifest of a package, from an assessment test.

:func:`package_entries` is the whole interface; :mod:`questxtract.qti_package` adds the JSON
sidecar and writes the ZIP.
"""

from __future__ import annotations

from typing import cast
from xml.etree import ElementTree

from .qti import METADATA_PATH, AssessmentItem, AssessmentTest

QTI_NS = "http://www.imsglobal.org/xsd/imsqtiasi_v3p0"
CP_NS = "http://www.imsglobal.org/xsd/qti/qtiv3p0/imscp_v1p1"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_QTI_XSD = "https://purl.imsglobal.org/spec/qti/v3p0/schema/xsd/imsqti_asiv3p0p1_v1p0.xsd"
_CP_XSD = "https://purl.imsglobal.org/spec/qti/v3p0/schema/xsd/imsqtiv3p0_imscpv1p2_v1p0.xsd"
# The namespaces are literal attributes on unqualified tags: ElementTree's
# tostring(default_namespace=...) rejects unqualified attributes, and register_namespace is
# process-global. Every official 1EdTech example uses the default namespace.
_QTI_ROOT = {"xmlns": QTI_NS, "xmlns:xsi": XSI_NS, "xsi:schemaLocation": f"{QTI_NS} {_QTI_XSD}"}
_CP_ROOT = {"xmlns": CP_NS, "xmlns:xsi": XSI_NS, "xsi:schemaLocation": f"{CP_NS} {_CP_XSD}"}


def package_entries(assessment: AssessmentTest) -> list[tuple[str, bytes]]:
    """Return the package's XML entries in ZIP order: the items, the test, then the manifest.

    Each entry is a path in the package and the file's bytes. The sidecar the manifest lists,
    :data:`questxtract.qti.METADATA_PATH`, is not among them.
    """
    item_paths = [f"items/{item.identifier}.xml" for item in assessment.items]
    entries = [
        (item_path, _xml_bytes(_item_xml(item)))
        for item, item_path in zip(assessment.items, item_paths, strict=True)
    ]
    entries.append(("assessmentTest.xml", _xml_bytes(_test_xml(assessment))))
    entries.append(("imsmanifest.xml", _xml_bytes(_manifest_xml(assessment, item_paths))))
    return entries


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
    response = ElementTree.SubElement(
        root,
        "qti-response-declaration",
        {"identifier": "RESPONSE", "base-type": "identifier", "cardinality": "single"},
    )
    correct = ElementTree.SubElement(response, "qti-correct-response")
    ElementTree.SubElement(correct, "qti-value").text = item.correct_option
    # FEEDBACK names the chosen option; it has no default.
    for identifier, base_type, default in (
        ("SCORE", "float", 0.0),
        ("MAXSCORE", "float", item.points),
        ("FEEDBACK", "identifier", None),
    ):
        attributes = {"identifier": identifier, "base-type": base_type, "cardinality": "single"}
        if identifier == "SCORE" and item.points > 0:  # normal-maximum must be > 0
            attributes["normal-maximum"] = _float(item.points)
            attributes["normal-minimum"] = _float(0.0)
        outcome = ElementTree.SubElement(root, "qti-outcome-declaration", attributes)
        if default is not None:
            default_value = ElementTree.SubElement(outcome, "qti-default-value")
            ElementTree.SubElement(default_value, "qti-value").text = _float(default)
    body = ElementTree.SubElement(root, "qti-item-body")
    interaction = ElementTree.SubElement(
        body,
        "qti-choice-interaction",
        {
            "response-identifier": "RESPONSE",
            "shuffle": _bool(item.shuffle_options),
            "max-choices": "1",
        },
    )
    ElementTree.SubElement(interaction, "qti-prompt").text = item.prompt
    for option in item.options:
        ElementTree.SubElement(
            interaction, "qti-simple-choice", {"identifier": option.identifier}
        ).text = option.text

    # Explicit Core response processing (Implementation Guide 3.4.4), not the match_correct
    # template: the template always scores 1, whatever the item is worth.
    processing = ElementTree.SubElement(root, "qti-response-processing")
    condition = ElementTree.SubElement(processing, "qti-response-condition")
    response_if = ElementTree.SubElement(condition, "qti-response-if")
    match = ElementTree.SubElement(response_if, "qti-match")
    ElementTree.SubElement(match, "qti-variable", {"identifier": "RESPONSE"})
    ElementTree.SubElement(match, "qti-correct", {"identifier": "RESPONSE"})
    score = ElementTree.SubElement(response_if, "qti-set-outcome-value", {"identifier": "SCORE"})
    ElementTree.SubElement(score, "qti-variable", {"identifier": "MAXSCORE"})
    response_else = ElementTree.SubElement(condition, "qti-response-else")
    wrong = ElementTree.SubElement(response_else, "qti-set-outcome-value", {"identifier": "SCORE"})
    ElementTree.SubElement(wrong, "qti-base-value", {"base-type": "float"}).text = _float(0.0)

    # FEEDBACK names the chosen option, which shows that option's modal feedback. No response
    # leaves it NULL, and no feedback is shown.
    chosen = ElementTree.SubElement(processing, "qti-set-outcome-value", {"identifier": "FEEDBACK"})
    ElementTree.SubElement(chosen, "qti-variable", {"identifier": "RESPONSE"})
    for option in item.options:
        modal = ElementTree.SubElement(
            root,
            "qti-modal-feedback",
            {
                "identifier": option.identifier,
                "outcome-identifier": "FEEDBACK",
                "show-hide": "show",
            },
        )
        content = ElementTree.SubElement(modal, "qti-content-body")
        ElementTree.SubElement(content, "p").text = option.feedback
        ElementTree.SubElement(content, "p").text = item.explanation
    return root


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
    ElementTree.SubElement(
        part, "qti-item-session-control", {"show-feedback": _bool(assessment.show_feedback)}
    )
    section = ElementTree.SubElement(
        part,
        "qti-assessment-section",
        {"identifier": "section-1", "title": assessment.title, "visible": "true"},
    )
    if assessment.shuffle_items:
        ElementTree.SubElement(section, "qti-ordering", {"shuffle": "true"})
    if assessment.instructions is not None:
        rubric = ElementTree.SubElement(
            section, "qti-rubric-block", {"view": "candidate", "use": "instructions"}
        )
        content = ElementTree.SubElement(rubric, "qti-content-body")
        ElementTree.SubElement(content, "p").text = assessment.instructions
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
        {"identifier": "questxtract-metadata", "type": "extension", "href": METADATA_PATH},
    )
    ElementTree.SubElement(sidecar, "file", {"href": METADATA_PATH})
    return root


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _float(value: float) -> str:
    return repr(float(value))


def _xml_bytes(root: ElementTree.Element) -> bytes:
    return cast(bytes, ElementTree.tostring(root, encoding="utf-8", xml_declaration=True))

"""QTI 2.1 XML: the items, the test and the manifest of a package, from an assessment test.

The same content as :mod:`questxtract.qti_v3`, in the ASI 2.1.2 binding: elements and attributes
are camelCase, without the ``qti-`` prefix, and there is no content body, so modal feedback and
the rubric block hold their paragraphs directly. Response processing also names the standard
``match_correct`` template, for an engine that supports only templates. The manifest follows the
QTI 2.1 profile of content packaging 1.2. :func:`package_entries` is the whole interface;
:mod:`questxtract.qti_package` adds the JSON sidecar and writes the ZIP.
"""

from __future__ import annotations

from typing import cast
from xml.etree import ElementTree

from .qti import METADATA_PATH, AssessmentItem, AssessmentTest

QTI_NS = "http://www.imsglobal.org/xsd/imsqti_v2p1"
CP_NS = "http://www.imsglobal.org/xsd/imscp_v1p1"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_QTI_XSD = "http://www.imsglobal.org/xsd/qti/qtiv2p1/imsqti_v2p1p2.xsd"  # ASI 2.1.2
_CP_XSD = "http://www.imsglobal.org/xsd/qti/qtiv2p1/qtiv2p1_imscpv1p2_v1p0.xsd"
# The standard template that sets SCORE to 1 for the correct response, else 0 (info model 8.1.1).
_MATCH_CORRECT = "http://www.imsglobal.org/question/qti_v2p1/rptemplates/match_correct"
# The namespaces are literal attributes on unqualified tags: ElementTree's
# tostring(default_namespace=...) rejects unqualified attributes, and register_namespace is
# process-global. The official 1EdTech 2.1 examples use the default namespace too.
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
        "assessmentItem",
        {
            **_QTI_ROOT,
            "identifier": item.identifier,
            "title": item.title,
            "adaptive": "false",
            "timeDependent": "false",
            "xml:lang": item.language,
        },
    )
    response = ElementTree.SubElement(
        root,
        "responseDeclaration",
        {"identifier": "RESPONSE", "baseType": "identifier", "cardinality": "single"},
    )
    correct = ElementTree.SubElement(response, "correctResponse")
    ElementTree.SubElement(correct, "value").text = item.correct_option
    # FEEDBACK names the chosen option; it has no default.
    for identifier, base_type, default in (
        ("SCORE", "float", 0.0),
        ("MAXSCORE", "float", item.points),
        ("FEEDBACK", "identifier", None),
    ):
        attributes = {"identifier": identifier, "baseType": base_type, "cardinality": "single"}
        if identifier == "SCORE" and item.points > 0:  # normalMaximum must be > 0
            attributes["normalMaximum"] = _float(item.points)
            attributes["normalMinimum"] = _float(0.0)
        outcome = ElementTree.SubElement(root, "outcomeDeclaration", attributes)
        if default is not None:
            default_value = ElementTree.SubElement(outcome, "defaultValue")
            ElementTree.SubElement(default_value, "value").text = _float(default)
    body = ElementTree.SubElement(root, "itemBody")
    interaction = ElementTree.SubElement(
        body,
        "choiceInteraction",
        {
            "responseIdentifier": "RESPONSE",
            "shuffle": _bool(item.shuffle_options),
            "maxChoices": "1",
        },
    )
    ElementTree.SubElement(interaction, "prompt").text = item.prompt
    for option in item.options:
        ElementTree.SubElement(
            interaction, "simpleChoice", {"identifier": option.identifier}
        ).text = option.text

    # Explicit response processing, so a correct answer scores the item's points and FEEDBACK
    # names the chosen option. It also names the match_correct template: generalized response
    # processing is optional in QTI 2.1 (info model 8.1), and an engine that supports only the
    # standard templates, as Blackboard may, can score with the template instead. When an item
    # gives both, its own rules are preferred (8.2). The template scores 1 and sets no FEEDBACK.
    processing = ElementTree.SubElement(root, "responseProcessing", {"template": _MATCH_CORRECT})
    condition = ElementTree.SubElement(processing, "responseCondition")
    response_if = ElementTree.SubElement(condition, "responseIf")
    match = ElementTree.SubElement(response_if, "match")
    ElementTree.SubElement(match, "variable", {"identifier": "RESPONSE"})
    ElementTree.SubElement(match, "correct", {"identifier": "RESPONSE"})
    score = ElementTree.SubElement(response_if, "setOutcomeValue", {"identifier": "SCORE"})
    ElementTree.SubElement(score, "variable", {"identifier": "MAXSCORE"})
    response_else = ElementTree.SubElement(condition, "responseElse")
    wrong = ElementTree.SubElement(response_else, "setOutcomeValue", {"identifier": "SCORE"})
    ElementTree.SubElement(wrong, "baseValue", {"baseType": "float"}).text = _float(0.0)

    # FEEDBACK names the chosen option, which shows that option's modal feedback. No response
    # leaves it NULL, and no feedback is shown.
    chosen = ElementTree.SubElement(processing, "setOutcomeValue", {"identifier": "FEEDBACK"})
    ElementTree.SubElement(chosen, "variable", {"identifier": "RESPONSE"})
    for option in item.options:
        modal = ElementTree.SubElement(
            root,
            "modalFeedback",
            {"identifier": option.identifier, "outcomeIdentifier": "FEEDBACK", "showHide": "show"},
        )
        ElementTree.SubElement(modal, "p").text = option.feedback
        ElementTree.SubElement(modal, "p").text = item.explanation
    return root


def _test_xml(assessment: AssessmentTest) -> ElementTree.Element:
    root = ElementTree.Element(
        "assessmentTest",
        {**_QTI_ROOT, "identifier": assessment.identifier, "title": assessment.title},
    )
    part = ElementTree.SubElement(
        root,
        "testPart",
        {"identifier": "part-1", "navigationMode": "linear", "submissionMode": "individual"},
    )
    ElementTree.SubElement(
        part, "itemSessionControl", {"showFeedback": _bool(assessment.show_feedback)}
    )
    section = ElementTree.SubElement(
        part,
        "assessmentSection",
        {"identifier": "section-1", "title": assessment.title, "visible": "true"},
    )
    if assessment.shuffle_items:
        ElementTree.SubElement(section, "ordering", {"shuffle": "true"})
    if assessment.instructions is not None:
        rubric = ElementTree.SubElement(
            section, "rubricBlock", {"view": "candidate", "use": "instructions"}
        )
        ElementTree.SubElement(rubric, "p").text = assessment.instructions
    for item in assessment.items:
        ElementTree.SubElement(
            section,
            "assessmentItemRef",
            {"identifier": f"ref-{item.identifier}", "href": f"items/{item.identifier}.xml"},
        )
    return root


def _manifest_xml(assessment: AssessmentTest, item_paths: list[str]) -> ElementTree.Element:
    root = ElementTree.Element("manifest", {**_CP_ROOT, "identifier": "questxtract-manifest"})
    metadata = ElementTree.SubElement(root, "metadata")
    # The QTI 2.1 content-package profile requires both values.
    ElementTree.SubElement(metadata, "schema").text = "QTIv2.1 Package"
    ElementTree.SubElement(metadata, "schemaversion").text = "1.0.0"
    ElementTree.SubElement(root, "organizations")
    resources = ElementTree.SubElement(root, "resources")
    test = ElementTree.SubElement(
        resources,
        "resource",
        {
            "identifier": assessment.identifier,
            "type": "imsqti_test_xmlv2p1",
            "href": "assessmentTest.xml",
        },
    )
    ElementTree.SubElement(test, "file", {"href": "assessmentTest.xml"})
    for item, path in zip(assessment.items, item_paths, strict=True):
        resource = ElementTree.SubElement(
            resources,
            "resource",
            {"identifier": item.identifier, "type": "imsqti_item_xmlv2p1", "href": path},
        )
        ElementTree.SubElement(resource, "file", {"href": path})
        ElementTree.SubElement(test, "dependency", {"identifierref": item.identifier})
    # The sidecar is a resource like every other file. The profile has no extension resource
    # type, so it is webcontent.
    sidecar = ElementTree.SubElement(
        resources,
        "resource",
        {"identifier": "questxtract-metadata", "type": "webcontent", "href": METADATA_PATH},
    )
    ElementTree.SubElement(sidecar, "file", {"href": METADATA_PATH})
    return root


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _float(value: float) -> str:
    return repr(float(value))


def _xml_bytes(root: ElementTree.Element) -> bytes:
    return cast(bytes, ElementTree.tostring(root, encoding="utf-8", xml_declaration=True))

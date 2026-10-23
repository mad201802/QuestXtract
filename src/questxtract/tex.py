"""ExerPlaza TeX: a ``quiz.json`` as IndexedQuestion blocks for ExerPlaza's upload (12.5).

The file is a transport format for ExerPlaza's parser (``exerplaza/backend/quiz/latex_parser.py``
v2.1), not a document to compile. ExerPlaza stores every text as parsed and shows it with
MathJax, so texts are written verbatim, without LaTeX escaping.

The parser reports none of the mistakes that would break an item; it reads the item wrong
instead. So this module guarantees what it needs: each answer on one line, balanced braces in
every text, and no command of the format inside a text.

The items come from :func:`questxtract.qti.quiz_to_qti`, so a quiz exports the same items as
TeX and as QTI.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import SchemaError
from .qti import AssessmentItem, AssessmentTest, QtiExportError, quiz_to_qti
from .quiz_schema import read_quiz

_LANGUAGES = {"en": "english", "it": "italian"}  # the only two ExerPlaza's parser accepts
_DISCLOSABILITY = {"pending": "only me", "accepted": "everybody"}
_FORMAT_COMMAND = re.compile(
    r"\\(?:begin|end)\{Indexed|\\Question|\\Solution|\\(?:correct|penalizable)?answer"
)
_SCHEMA_VERSION = re.compile(r"[0-9]+\.[0-9]+")


class TexExportError(SchemaError):
    """Raised when a quiz file cannot be read or exported as ExerPlaza TeX, or the file written."""


def export_tex(
    quiz_path: str | Path, output_path: str | Path, *, content_units: Sequence[str]
) -> int:
    """Read a ``quiz.json`` and write it as ExerPlaza TeX.

    The file is for ExerPlaza's upload at ``/questions/upload/tex``. Every item becomes one
    ``IndexedQuestion`` block with its options, its solution and notes for the teachers. Items
    a reviewer marked ``rejected`` are left out; ``pending`` items are visible only to their
    author, ``accepted`` ones to everybody. The file is UTF-8 with LF line endings, and the same
    input gives the same bytes. Nothing is written if any check fails.

    Args:
        quiz_path: A ``quiz.json``, schema version 1.x, in English or Italian.
        output_path: Where to write the ``.tex`` file. Its folder must exist.
        content_units: ExerPlaza content units for every item, at least one, each spelled
            exactly as in ExerPlaza: only an exact match links to an existing unit.

    Returns:
        The number of questions written.

    Raises:
        TexExportError: If a content unit is empty, spans lines or holds a brace; if the quiz
            file cannot be read or breaks a rule of ``quiz.json``; if an item cannot be written
            for ExerPlaza (another language, unbalanced braces, a command of the format in a
            text), naming the field; or if the file cannot be written.

    """
    units = _content_units(content_units)
    try:
        document = read_quiz(quiz_path)
    except OSError as exc:
        raise TexExportError(f"{quiz_path}: {exc.strerror or exc}") from exc
    except SchemaError as exc:
        raise TexExportError(str(exc)) from exc
    text, count = _tex(document, units)
    try:
        Path(output_path).write_bytes(text.encode("utf-8"))
    except OSError as exc:
        raise TexExportError(
            f"{output_path}: cannot write the TeX file ({exc.strerror or exc})"
        ) from exc
    return count


def quiz_to_tex(document: Mapping[str, Any], content_units: Sequence[str]) -> str:
    """Convert a ``quiz.json`` document to ExerPlaza TeX, one block per item not rejected.

    Raises:
        TexExportError: Naming the field, as :func:`export_tex` describes.

    """
    return _tex(document, _content_units(content_units))[0]


def _tex(document: Mapping[str, Any], units: tuple[str, ...]) -> tuple[str, int]:
    """Return the file's text and the number of blocks in it."""
    try:
        assessment = quiz_to_qti(document)
    except QtiExportError as exc:
        raise TexExportError(str(exc)) from exc
    language = _language(document.get("language"))
    schema_version = document.get("schema_version")
    if not isinstance(schema_version, str) or not _SCHEMA_VERSION.fullmatch(schema_version):
        raise TexExportError(
            f"schema_version: expected a version such as '1.0', got {schema_version!r}"
        )
    _check(assessment.title, "title", braces=False)
    by = _generated_by(assessment.metadata.get("generation"))

    # Rejected items are not in the assessment, so an item's position there is not its index
    # in the document; error paths need the index.
    index_of: dict[str, int] = {}
    for index, raw in enumerate(document["questions"]):
        identifier = raw.get("id") if isinstance(raw, Mapping) else None
        if isinstance(identifier, str):
            index_of.setdefault(identifier, index)

    header = "\n".join(
        [
            "% QuestXtract quiz for ExerPlaza (IndexedQuestion format), upload at "
            "/questions/upload/tex.",
            f"% quiz: {assessment.identifier}, quiz.json schema {schema_version}",
            f"% title: {_one_line(assessment.title)}",
            "% Texts are verbatim, not LaTeX-escaped: ExerPlaza shows them with MathJax.",
        ]
    )
    blocks = [
        _block(assessment, item, f"questions[{index_of[item.identifier]}]", units, language, by)
        for item in assessment.items
    ]
    return "\n\n".join([header, *blocks]) + "\n", len(blocks)


def _block(
    assessment: AssessmentTest,
    item: AssessmentItem,
    path: str,
    units: tuple[str, ...],
    language: str,
    by: str,
) -> str:
    """One ``IndexedQuestion`` block, without a trailing newline."""
    status = item.metadata.get("review_status")
    if status not in _DISCLOSABILITY:
        raise TexExportError(
            f"{path}.review.status: expected 'pending' or 'accepted', got {status!r}"
        )
    disclosability = _DISCLOSABILITY[status]
    origin = item.metadata.get("origin")
    stem = origin.get("prompt") if isinstance(origin, Mapping) else None
    if stem not in {"spoken", "generated"}:
        raise TexExportError(
            f"{path}.prompt.origin: expected 'spoken' or 'generated', got {stem!r}"
        )

    prompt = _verbatim(item.prompt)
    _check(prompt, f"{path}.prompt.text")
    answers: list[str] = []
    explained: list[str] = []
    for index, option in enumerate(item.options):
        text = _one_line(option.text)
        _check(text, f"{path}.options[{index}].text")
        feedback = _one_line(option.feedback)
        _check(feedback, f"{path}.solution.option_explanations.{option.identifier}.text")
        correct = option.identifier == item.correct_option
        answers.append(("\\correctanswer " if correct else "\\answer ") + text)
        explained.append(f"- {text}{' (correct)' if correct else ''} — {feedback}")
    explanation = _verbatim(item.explanation)
    _check(explanation, f"{path}.solution.explanation.text")

    provenance = item.provenance
    _check(provenance.source_question_id, f"{path}.provenance.source_question_id")
    for key, seconds in (("start", provenance.start), ("end", provenance.end)):
        if seconds < 0:
            raise TexExportError(f"{path}.provenance.{key}: expected >= 0, got {seconds!r}")
    reference = f"{assessment.identifier}/{item.identifier}"
    notes = (
        f"QuestXtract {reference}. Question {provenance.source_question_id} at "
        f"{_clock(provenance.start)}-{_clock(provenance.end)} of the lecture. "
        f"Stem {stem}; generated text by {by}."
    )
    solution = explanation + "\n\n" + "\n".join(explained)
    return "\n".join(
        [
            "\\begin{IndexedQuestion}",
            f"% questxtract: {reference}",
            "\\QuestionContentUnits" + "".join(f"{{{unit}}}" for unit in units),
            "\\QuestionType{multiple choice}",
            f"\\QuestionLanguage{{{language}}}",
            f"\\QuestionDisclosability{{{disclosability}}}",
            f"\\QuestionBody{{{prompt}}}",
            "\\QuestionPotentialAnswers{",
            *answers,
            "}",
            f"\\QuestionNotesForTheTeachers{{{notes}}}",
            "\\begin{IndexedSolution}",
            f"\\SolutionDisclosability{{{disclosability}}}",
            f"\\SolutionBody{{{solution}}}",
            "\\end{IndexedSolution}",
            "\\end{IndexedQuestion}",
        ]
    )


def _content_units(content_units: Sequence[str]) -> tuple[str, ...]:
    """Return the units, stripped, each checked: non-empty, on one line, without braces."""
    if isinstance(content_units, str) or not content_units:
        raise TexExportError("content units: expected at least one name")
    units: list[str] = []
    for unit in content_units:
        name = unit.strip() if isinstance(unit, str) else ""
        if not name:
            raise TexExportError(f"content unit {unit!r}: expected a non-empty name")
        if len(name.splitlines()) > 1:
            raise TexExportError(f"content unit {unit!r}: expected one line")
        if "{" in name or "}" in name:
            raise TexExportError(f"content unit {unit!r}: braces are not allowed")
        units.append(name)
    return tuple(units)


def _language(value: object) -> str:
    """ExerPlaza's name for a language tag's primary subtag."""
    primary = value.split("-")[0].lower() if isinstance(value, str) else None
    if primary not in _LANGUAGES:
        raise TexExportError(
            f"language: ExerPlaza accepts English and Italian only ('en' or 'it'), got {value!r}"
        )
    return _LANGUAGES[primary]


def _generated_by(generation: object) -> str:
    """Who wrote the generated text, for the notes: the model and prompt, or the method."""
    if not isinstance(generation, Mapping):
        raise TexExportError("generation: expected an object")
    values: dict[str, str | None] = {}
    for key in ("method", "model", "prompt_version"):
        value = generation.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise TexExportError(f"generation.{key}: expected a non-empty string or null")
        if value is not None:
            _check(value, f"generation.{key}")
        values[key] = value
    model, prompt_version, method = values["model"], values["prompt_version"], values["method"]
    if model is not None:
        return f"{model}, prompt {prompt_version}" if prompt_version is not None else model
    if method is None:
        raise TexExportError("generation.method: expected a non-empty string when model is null")
    return f"method {method}"


def _check(text: str, path: str, *, braces: bool = True) -> None:
    """Fail unless the parser would read ``text`` back whole: balanced, no format command."""
    command = _FORMAT_COMMAND.search(text)
    if command:
        raise TexExportError(
            f"{path}: {command.group()!r} is a command of the ExerPlaza TeX format"
        )
    if not braces:
        return
    depth = 0
    for char in text:
        depth += 1 if char == "{" else -1 if char == "}" else 0
        if depth < 0:
            break
    if depth != 0:
        raise TexExportError(f"{path}: unbalanced braces")


def _verbatim(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _clock(seconds: float) -> str:
    """Whole seconds, rounded down, as ``m:ss``, or ``h:mm:ss`` from one hour on."""
    hours, rest = divmod(math.floor(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"

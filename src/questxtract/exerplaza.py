"""Map the reviewed ``questions.json`` export to an Exerplaza quiz payload.

The Exerplaza API is not available yes, so this does not authenticate or make network requests.
It defines the small, versioned exchange payload used by the CLI for local review and hand-off.
When the real contract is available, the mapping can be changed here without coupling it to
question extraction or the review format.
"""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any

from .export import read_questions
from .models import SchemaError

EXERPLAZA_FORMAT = "exerplaza"
EXERPLAZA_FORMAT_VERSION = "0.1"


def _slug(value: str) -> str:
    """Return a stable, readable identifier for a quiz title."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "video-quiz"


def _quiz_title(document: dict[str, Any]) -> str:
    source = document.get("source")
    if isinstance(source, dict):
        path = source.get("path")
        if isinstance(path, str) and path:
            return PurePath(path).stem or "Video quiz"
    return "Video quiz"


def _approved_questions(document: dict[str, Any]) -> list[dict[str, Any]]:
    questions = document["questions"]
    approved: list[dict[str, Any]] = []
    for question in questions:
        if not isinstance(question, dict):
            raise SchemaError("questions must contain objects")
        if question.get("verdict") in {"keep", "revise"}:
            approved.append(question)
    return approved


def _video_id(document: dict[str, Any], explicit_id: str | None) -> str:
    if explicit_id:
        return explicit_id
    source = document.get("source")
    if isinstance(source, dict):
        source_hash = source.get("hash")
        if isinstance(source_hash, str) and source_hash:
            return source_hash
    raise SchemaError("video_id is required when source.hash is missing")


def _validate_question(question: dict[str, Any], index: int) -> None:
    prefix = f"questions[{index}]"
    for field in ("id", "text"):
        value = question.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SchemaError(f"{prefix}.{field} must be a non-empty string")
    start = question.get("start")
    end = question.get("end")
    if (
        not isinstance(start, (int, float))
        or isinstance(start, bool)
        or not isinstance(end, (int, float))
        or isinstance(end, bool)
        or end <= start
    ):
        raise SchemaError(f"{prefix}.start and .end must be numbers with end greater than start")
    confidence = question.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 1
    ):
        raise SchemaError(f"{prefix}.confidence must be a number between 0 and 1")
    answer = question.get("answer")
    if answer is not None and (
        not isinstance(answer, dict)
        or (answer.get("text") is not None and not isinstance(answer.get("text"), str))
    ):
        raise SchemaError(f"{prefix}.answer must be an object with optional string text or null")


def questions_to_exerplaza(
    document: dict[str, Any],
    *,
    video_id: str | None = None,
    include_unjudged: bool = False,
) -> dict[str, Any]:
    """Map an exported question document to a local Exerplaza quiz payload.

    By default only questions explicitly approved with ``verdict`` equal to ``"keep"``
    or ``"revise"`` are
    included.  ``include_unjudged`` is intended for previews before UI review,
    not for an upload workflow.  ``reject`` questions are never exported.
    """
    if not isinstance(document, dict) or not isinstance(document.get("questions"), list):
        raise SchemaError("expected an object with a 'questions' list")

    source = document.get("source")
    if not isinstance(source, dict):
        source = {}
    resolved_video_id = _video_id(document, video_id)
    title = _quiz_title(document)
    selected = _approved_questions(document)
    if include_unjudged:
        selected.extend(
            question
            for question in document["questions"]
            if isinstance(question, dict) and question.get("verdict") == "unjudged"
        )

    mapped_questions: list[dict[str, Any]] = []
    for index, question in enumerate(selected):
        _validate_question(question, index)
        answer = question.get("answer")
        mapped_questions.append(
            {
                "id": question.get("id"),
                "question": question.get("text"),
                "answer": answer.get("text") if isinstance(answer, dict) else None,
                "question_type": question.get("question_type"),
                "timestamp": {
                    "start": question.get("start"),
                    "end": question.get("end"),
                },
                "confidence": question.get("confidence"),
            }
        )

    return {
        "format": EXERPLAZA_FORMAT,
        "format_version": EXERPLAZA_FORMAT_VERSION,
        "quiz": {
            "id": resolved_video_id,
            "title": title,
            "video": {
                "id": resolved_video_id,
                "path": source.get("path"),
                "duration_seconds": source.get("duration_seconds"),
            },
            "questions": mapped_questions,
        },
    }


def read_exerplaza_questions(
    path: str,
    *,
    video_id: str | None = None,
    include_unjudged: bool = False,
) -> dict[str, Any]:
    """Read ``questions.json`` and return its Exerplaza quiz mapping."""
    return questions_to_exerplaza(
        read_questions(path),
        video_id=video_id,
        include_unjudged=include_unjudged,
    )

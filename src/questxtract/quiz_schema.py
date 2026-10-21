"""Reading and checking ``quiz.json`` (architecture.md 12.3).

``quiz.schema.json`` describes the shape and ships in the package; the tests enforce it with
``jsonschema``. JSON Schema cannot compare one field with another, so :func:`validate_quiz`
checks those rules in plain Python, which keeps ``jsonschema`` out of the runtime
dependencies. Any shape it needs for a rule and does not find is reported, never raised as a
``KeyError``: whatever JSON comes in, the only error is :class:`SchemaError`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeGuard

from .models import SchemaError

QUIZ_SCHEMA_VERSION = "1.0"
QUIZ_SCHEMA_PATH = Path(__file__).with_name("quiz.schema.json")

_ITEM_PREFIX = "quiz_"


def validate_quiz(document: Mapping[str, Any], questions: Mapping[str, Any] | None = None) -> None:
    """Check the rules of a ``quiz.json`` document that its JSON Schema cannot express.

    Per item: the correct option exists (1), ``option_explanations`` has one entry per option
    and no other key (2), option ids are unique (3), the item id is ``"quiz_"`` plus
    ``provenance.source_question_id`` and no two items share that source (4), and
    ``provenance.end`` is not before ``start`` (6). With ``questions``, also: the source
    question is in that file (5), and a ``prompt`` marked ``spoken`` is the question as it was
    said (7): its ``original_text`` when the LLM judge reworded it, else its ``text``, compared
    with whitespace and case normalised. Options and explanations marked ``spoken`` cannot be
    checked without the transcript.

    The structure itself (keys, types, enums) is the schema's job, ``QUIZ_SCHEMA_PATH``.

    Args:
        document: A parsed ``quiz.json``.
        questions: The parsed ``questions.json`` the quiz was made from, if at hand. Without
            it, rules 5 and 7 are skipped.

    Raises:
        SchemaError: Listing every problem, one per line, each with its path, e.g.
            ``questions[2].correct_response.option_ids[0]: 'd' is not an option``.

    """
    errors: list[str] = []
    spoken = _spoken_texts(questions, errors) if questions is not None else None
    items = document.get("questions") if isinstance(document, Mapping) else None
    if not isinstance(items, list):
        errors.append("questions: expected a list")
        items = []
    first_item_of: dict[str, int] = {}
    for index, item in enumerate(items):
        path = f"questions[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{path}: expected an object")
            continue
        _check_options(path, item, errors)
        source_id = _check_provenance(path, item, errors)
        if source_id is None:
            continue
        expected = _ITEM_PREFIX + source_id
        if item.get("id") != expected:
            errors.append(
                f"{path}.id: expected {expected!r} (from provenance.source_question_id), "
                f"got {item.get('id')!r}"
            )
        if source_id in first_item_of:
            errors.append(
                f"{path}.provenance.source_question_id: {source_id!r} is already the source of "
                f"questions[{first_item_of[source_id]}]"
            )
        else:
            first_item_of[source_id] = index
        if spoken is not None:
            _check_source(path, item, source_id, spoken, errors)
    if errors:
        raise SchemaError("\n".join(errors))


def read_quiz(path: str | Path, questions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read a ``quiz.json`` and check it with :func:`validate_quiz`.

    A newer minor version is read; another major version is refused.

    Args:
        path: The ``quiz.json`` to read.
        questions: The parsed ``questions.json`` it was made from, passed on to
            :func:`validate_quiz`.

    Raises:
        SchemaError: If the file is not JSON or not an object, was written by another major
            schema version, or breaks a rule of :func:`validate_quiz` (one line per problem,
            each starting with ``path``).

    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SchemaError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: expected a JSON object")
    version = str(data.get("schema_version", ""))
    major = QUIZ_SCHEMA_VERSION.split(".")[0]
    if version.split(".")[0] != major:
        raise SchemaError(
            f"{path}: schema version {version or 'missing'}, but this questxtract reads "
            f"{major}.x; re-run `questxtract quiz` to regenerate it"
        )
    try:
        validate_quiz(data, questions)
    except SchemaError as exc:
        raise SchemaError("\n".join(f"{path}: {line}" for line in str(exc).splitlines())) from exc
    return data


def _check_options(path: str, item: Mapping[str, Any], errors: list[str]) -> None:
    """Rules 1 to 3: the correct option exists, one explanation per option, unique ids."""
    options = item.get("options")
    if not isinstance(options, list):
        errors.append(f"{path}.options: expected a list")
        return
    first_option_of: dict[str, int] = {}
    for index, option in enumerate(options):
        option_id = option.get("id") if isinstance(option, Mapping) else None
        if not isinstance(option_id, str):
            errors.append(f"{path}.options[{index}].id: expected a string")
        elif option_id in first_option_of:
            errors.append(
                f"{path}.options[{index}].id: {option_id!r} repeats "
                f"options[{first_option_of[option_id]}].id"
            )
        else:
            first_option_of[option_id] = index

    correct = item.get("correct_response")
    option_ids = correct.get("option_ids") if isinstance(correct, Mapping) else None
    if not isinstance(option_ids, list):
        errors.append(f"{path}.correct_response.option_ids: expected a list")
    else:
        for index, option_id in enumerate(option_ids):
            if not isinstance(option_id, str) or option_id not in first_option_of:
                errors.append(
                    f"{path}.correct_response.option_ids[{index}]: {option_id!r} is not an option"
                )

    solution = item.get("solution")
    explained = solution.get("option_explanations") if isinstance(solution, Mapping) else None
    if not isinstance(explained, Mapping):
        errors.append(f"{path}.solution.option_explanations: expected an object")
        return
    for option_id in first_option_of:
        if option_id not in explained:
            errors.append(f"{path}.solution.option_explanations: no entry for option {option_id!r}")
    for key in explained:
        if key not in first_option_of:
            errors.append(f"{path}.solution.option_explanations.{key}: no option {key!r}")


def _check_provenance(path: str, item: Mapping[str, Any], errors: list[str]) -> str | None:
    """Rule 6, and the source question id for rules 4, 5 and 7, or ``None`` if unusable."""
    provenance = item.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append(f"{path}.provenance: expected an object")
        return None
    start, end = provenance.get("start"), provenance.get("end")
    if not _is_number(start):
        errors.append(f"{path}.provenance.start: expected a number")
    elif not _is_number(end):
        errors.append(f"{path}.provenance.end: expected a number")
    elif end < start:
        errors.append(f"{path}.provenance.end: {end} is before start {start}")
    source_id = provenance.get("source_question_id")
    if not isinstance(source_id, str):
        errors.append(f"{path}.provenance.source_question_id: expected a string")
        return None
    return source_id


def _check_source(
    path: str,
    item: Mapping[str, Any],
    source_id: str,
    spoken: dict[str, str],
    errors: list[str],
) -> None:
    """Rules 5 and 7: the source question exists, and a ``spoken`` stem is what was said."""
    if source_id not in spoken:
        errors.append(
            f"{path}.provenance.source_question_id: {source_id!r} is not in questions.json"
        )
        return
    prompt = item.get("prompt")
    if not isinstance(prompt, Mapping) or prompt.get("origin") != "spoken":
        return
    text = prompt.get("text")
    if not isinstance(text, str) or normalise_spoken(text) != normalise_spoken(spoken[source_id]):
        errors.append(
            f"{path}.prompt.text: origin 'spoken', but it is not question {source_id!r} as it "
            "was said; mark a reworded stem 'generated'"
        )


def _spoken_texts(questions: Mapping[str, Any], errors: list[str]) -> dict[str, str] | None:
    """Each question id in a ``questions.json``, mapped to the question as it was said.

    That is ``original_text`` when the LLM judge reworded the question (verdict ``revise``),
    else ``text``. ``None``, with an error, if the document has no ``questions`` list.
    """
    entries = questions.get("questions") if isinstance(questions, Mapping) else None
    if not isinstance(entries, list):
        errors.append("questions.json: expected a 'questions' list")
        return None
    spoken: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            continue
        original = entry.get("original_text")
        text = original if isinstance(original, str) else entry.get("text")
        spoken[entry["id"]] = text if isinstance(text, str) else ""
    return spoken


def normalise_spoken(text: str) -> str:
    """Text as rule 7 compares it: whitespace and case do not count, punctuation does."""
    return " ".join(text.split()).casefold()


def _is_number(value: object) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

"""Turn the questions of a lecture into multiple-choice quiz items (architecture.md 12.3).

Not a stage, and deliberately not part of ``process``'s pipeline: this is the only part of the
package that writes text that was not spoken (requirement 4.5). Every text it writes says where
it came from (``origin``: ``spoken`` or ``generated``), and every item starts as pending teacher
review. It reads a finished ``questions.json`` and asks the model once per question, through the
same endpoint as the judge and vision (:mod:`questxtract.endpoint`), cached on the request.

The keys of ``quiz.json`` are written in one place, :func:`_quiz_dict` and :func:`_item_dict`,
so the agreed v1 schema only has to be followed there.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import endpoint
from .config import Config
from .models import Sentence
from .quiz_schema import QUIZ_SCHEMA_VERSION, normalise_spoken

QUIZ_PROMPT_VERSION = "v1"

STRICTNESS = ("strict", "lenient")
MIN_OPTIONS, MAX_OPTIONS = 3, 5

_REJECTED = ("deleted", "false_positive")  # review.status values a teacher set to drop a question
_BOARD_CHARS = 1500  # board text sent with one question: the states on screen, kept short
_SPOKEN_BEFORE, _SPOKEN_AFTER = 4, 2  # sentences of lecture sent around the question
_SPOKEN_CHARS = 1500  # and at most this much of them
_IDS = "ABCDE"

_PROMPT = """\
You write one multiple-choice quiz item from a question asked in a university lecture. You get
the question as it was asked, the lecturer's answer when the lecture gave one, what was said
just before and after it, and what was written on the board or slides at that moment. The
speech is an automatic transcript: it can mishear technical words, so tell from the context and
the board what was meant.

Write:
- "quiz_material": false only when the question is addressed to the room ("what would you
  answer, people?"), rhetorical, a check on the class ("do you see how...", "are we together",
  "is that clear"), about organisation, or when neither the question, the lecturer's answer,
  the speech around it nor the board says what it is about. A question that points at
  something ("this one", "here") is still quiz material when the context makes the topic
  clear: then name that topic in the stem, with the actual expression from the board or the
  speech, never a placeholder such as "square" or "star". When you cannot tell what it points
  at, it is false. When false, leave every other field as an empty string.
- "stem": the question for the quiz. Keep the lecturer's wording when it is clear on its own;
  otherwise add only the context a student needs to answer it. Never turn it into another
  question.
- "correct_answer": the right answer as one short option. When the lecturer answered, it must
  say the same as the lecturer's answer, condensed. Write it yourself only when there is none.
  The lecturer's answer is an automatic transcript of speech and can answer a different
  question, or only a step of one. Work the answer out yourself and check the stem against it:
  if the lecturer's answer is wrong for the stem you write, give the correct answer instead.
  A stem never states something false, and never brings in a concept that neither the lecturer
  nor the board used.
- "correct_explanation": one or two sentences on why it is right.
- "distractor_1" to "distractor_{distractors}": wrong options that a student with a typical
  misconception about this topic would choose. Each must be clearly wrong to a student who
  understood the topic, similar in length and style to the correct answer, and different from
  every other option. Never "all of the above" or "none of the above".
- "why_wrong_1" to "why_wrong_{distractors}": for each distractor, the misconception behind it
  and why it is wrong, in one or two sentences.
- "explanation": two or three sentences a student reads after answering: the idea the question
  tests.

Use the board text only as context; do not quote it as an option unless it is the answer.
Write formulas as LaTeX between $ signs. Answer with one JSON object only."""


@dataclass(frozen=True, kw_only=True)
class QuizResult:
    """What :func:`build_quiz` made: the ``quiz.json`` document and what it left out.

    Args:
        quiz: The ``quiz.json`` document, ready to write.
        skipped: Ids of questions that are not quiz material: rejected, check-ins, unanswered
            ones when strict, and those the model judged unusable (addressed to the room,
            rhetorical, or too vague to answer on their own).
        failed: ``(question id, reason)`` for each question the model could not turn into a
            usable item; the rest of the quiz is unaffected.
        usage: What the model calls cost, as :func:`endpoint.ask` counts it: ``quiz_requests``,
            ``quiz_input_tokens``, ``quiz_output_tokens`` and ``quiz_ms``. Cached answers add
            nothing.

    """

    quiz: dict[str, Any]
    skipped: tuple[str, ...] = ()
    failed: tuple[tuple[str, str], ...] = field(default=())
    usage: dict[str, int] = field(default_factory=dict)


def build_quiz(
    document: Mapping[str, Any],
    config: Config,
    *,
    questions_name: str,
    questions_hash: str,
    sentences: Sequence[Sentence] = (),
    progress: Callable[[str, float], None] | None = None,
) -> QuizResult:
    """Build ``quiz.json`` from a ``questions.json`` document (as :func:`read_questions` returns).

    ``questions_name`` and ``questions_hash`` identify the source file in the output; the hash
    is the SHA-256 of its bytes. ``sentences`` are the lecture's, from its transcript: the
    ones around each question go with it as context, and without them the model sees only the
    question, the answer and the board. A question the model fails on (a stall, or an answer that is
    not a usable item, twice) is left out and listed in ``failed``.

    Raises:
        ValueError: If ``quiz_strictness`` or ``quiz_options`` is out of range.
        ConnectionError: If the endpoint cannot be reached or refuses the key.

    """
    if config.quiz_strictness not in STRICTNESS:
        raise ValueError("quiz_strictness must be strict or lenient")
    if not MIN_OPTIONS <= config.quiz_options <= MAX_OPTIONS:
        raise ValueError(f"quiz_options must be between {MIN_OPTIONS} and {MAX_OPTIONS}")
    boards = {v["id"]: v for v in document.get("visuals", []) if isinstance(v, dict)}
    spoken = _Spoken(sentences)
    selected, skipped = _select(document.get("questions", []), config.quiz_strictness)
    items: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []
    usage: Counter[str] = Counter()
    for index, question in enumerate(selected):
        if progress:
            progress("quiz", index / len(selected))
        try:
            answer = endpoint.ask(
                config,
                _request(config, question, boards, spoken),
                name="quiz",
                valid=_usable(config.quiz_options - 1),
                usage=usage,
            )
        except (endpoint.Stalled, endpoint.InvalidAnswer) as exc:
            failed.append((question["id"], str(exc)))
            continue
        if answer["quiz_material"] is False:
            skipped.append(str(question["id"]))
            continue
        items.append(_item_dict(question, answer, config.quiz_options))
    if progress:
        progress("quiz", 1.0)
    quiz = _quiz_dict(document, config, items, questions_name, questions_hash)
    return QuizResult(quiz=quiz, skipped=tuple(skipped), failed=tuple(failed), usage=dict(usage))


def quiz_to_json(quiz: Mapping[str, Any]) -> str:
    """Serialise a quiz document: stable key order, so the same quiz gives the same bytes."""
    return json.dumps(quiz, indent=2, ensure_ascii=False) + "\n"


def file_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes, the identity of the ``questions.json`` a quiz came from."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _select(
    questions: Sequence[Mapping[str, Any]], strictness: str
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Split questions into quiz material and the ids left out, in ``questions.json`` order.

    Always left out: questions a reviewer rejected and classroom check-ins (they are not
    quiz material). ``strict`` also leaves out questions the lecture did not answer.
    """
    selected: list[Mapping[str, Any]] = []
    skipped: list[str] = []
    for question in questions:
        review = question.get("review") or {}
        if (
            review.get("status") in _REJECTED
            or "check_in" in (question.get("reasons") or ())
            or (strictness == "strict" and not question.get("answer"))
        ):
            skipped.append(str(question.get("id")))
        else:
            selected.append(question)
    return selected, skipped


def _request(
    config: Config,
    question: Mapping[str, Any],
    boards: Mapping[str, Mapping[str, Any]],
    spoken: _Spoken,
) -> dict[str, Any]:
    """Build the chat request for one question: the prompt, then the question in context."""
    distractors = config.quiz_options - 1
    answer = question.get("answer") or {}
    lines = [f"Question: {question['text']}"]
    if answer.get("text"):
        lines.append(f"Lecturer's answer: {answer['text']}")
    else:
        lines.append("Lecturer's answer: none in the lecture; write the correct answer yourself.")
    around = spoken.around(str(question.get("sentence_id") or ""))
    lines.append(
        f"Said around it (the question marked >>):\n{around}"
        if around
        else "Said around it: no transcript was given."
    )
    board = _board_text(question.get("visual_ids") or (), boards)
    lines.append(f"Board: {board}" if board else "Board: nothing was linked to this question.")
    request: dict[str, Any] = {
        "model": config.quiz_model,
        "messages": [
            {"role": "system", "content": _PROMPT.replace("{distractors}", str(distractors))},
            {"role": "user", "content": "\n".join(lines)},
        ],
        "response_format": _schema(distractors),
    }
    if config.reasoning_effort:
        request["reasoning_effort"] = config.reasoning_effort
    return request


class _Spoken:
    """The lecture's sentences, to look up what was said around a question."""

    def __init__(self, sentences: Sequence[Sentence]) -> None:
        self.sentences = tuple(sentences)
        self.index = {s.id: i for i, s in enumerate(self.sentences)}

    def around(self, sentence_id: str) -> str:
        """Return the sentences around one, the question marked, or ``""`` when not found.

        Over the length cap, whole sentences go, farthest first: those before the question,
        then those after it. The question itself always stays.
        """
        at = self.index.get(sentence_id)
        if at is None:
            return ""
        first, last = max(0, at - _SPOKEN_BEFORE), min(len(self.sentences), at + _SPOKEN_AFTER + 1)

        def text() -> str:
            return "\n".join(
                (">> " if i == at else "   ") + self.sentences[i].text.strip()
                for i in range(first, last)
            )

        while len(text()) > _SPOKEN_CHARS and first < at:
            first += 1
        while len(text()) > _SPOKEN_CHARS and last > at + 1:
            last -= 1
        return text()


def _board_text(visual_ids: Sequence[str], boards: Mapping[str, Mapping[str, Any]]) -> str:
    """Join the description, text and LaTeX of the linked board states, capped in length."""
    parts: list[str] = []
    for visual_id in visual_ids:
        state = boards.get(visual_id)
        if not state:
            continue
        for key in ("description", "text", "latex"):
            value = state.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(f"{key}: {value.strip()}")
    return "\n".join(parts)[:_BOARD_CHARS]


def _schema(distractors: int) -> dict[str, Any]:
    """Build the response format: a named field per distractor, so a strict decoder fills each."""
    names = ["stem", "correct_answer", "correct_explanation", "explanation"]
    for number in range(1, distractors + 1):
        names += [f"distractor_{number}", f"why_wrong_{number}"]
    properties: dict[str, Any] = {"quiz_material": {"type": "boolean"}}  # decided first
    properties |= {name: {"type": "string"} for name in names}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "quiz_item",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def _usable(distractors: int) -> Callable[[dict[str, Any]], bool]:
    """Whether an answer is usable: a decline, or every field filled and every option different.

    Anything else is rejected, asked once more and never cached (:func:`endpoint.ask`).
    """

    def usable(answer: dict[str, Any]) -> bool:
        if answer.get("quiz_material") is False:
            return True  # a valid answer: the question is left out, and the answer is cached
        if answer.get("quiz_material") is not True:
            return False
        keys = ["stem", "correct_answer", "correct_explanation", "explanation"]
        keys += [
            f"{kind}_{n}" for n in range(1, distractors + 1) for kind in ("distractor", "why_wrong")
        ]
        if not all(isinstance(answer.get(k), str) and answer[k].strip() for k in keys):
            return False
        options = [answer["correct_answer"]] + [
            answer[f"distractor_{n}"] for n in range(1, distractors + 1)
        ]
        return len({_normal(option) for option in options}) == len(options)

    return usable


_MATH_DELIMITERS = re.compile(r"\$|\\[()\[\]]")
_NOT_MEANING = re.compile(r"[^\w\-+=/^<>*(){}\\.,]")  # a point or comma inside a number counts
_SENTENCE_END = " .,;:!?"


def _normal(text: str) -> str:
    """Options compared for sameness: case, spacing, end punctuation, math delimiters: not counted.

    A sign does: ``-2`` and ``2``, or ``s = -1`` and ``s = 1``, are different options; so does a
    decimal point: ``1.5`` and ``15`` are different options.
    """
    text = _MATH_DELIMITERS.sub("", text.casefold()).strip(_SENTENCE_END)
    return _NOT_MEANING.sub("", text)


def _item_dict(
    question: Mapping[str, Any], answer: Mapping[str, Any], options: int
) -> dict[str, Any]:
    """Build one ``quiz.json`` item.

    The correct option's place follows from the question id, so it varies between items and is
    the same on every run.
    """
    source_id = str(question["id"])
    correct_at = int(hashlib.sha256(source_id.encode()).hexdigest(), 16) % options
    distractors = iter(range(1, options))
    texts: list[tuple[str, str]] = []  # (option text, why it is right or wrong)
    for position in range(options):
        if position == correct_at:
            texts.append((answer["correct_answer"], answer["correct_explanation"]))
        else:
            number = next(distractors)
            texts.append((answer[f"distractor_{number}"], answer[f"why_wrong_{number}"]))
    spoken = question.get("original_text") or question["text"]
    stem = answer["stem"].strip()
    return {
        "id": "quiz_" + source_id,
        "prompt": _text(stem, spoken=normalise_spoken(stem) == normalise_spoken(spoken)),
        "interaction_type": "choice_single",
        "points": 1,
        "options": [{"id": _IDS[i], "text": _text(text)} for i, (text, _) in enumerate(texts)],
        "correct_response": {"option_ids": [_IDS[correct_at]]},
        "solution": {
            "explanation": _text(answer["explanation"]),
            "option_explanations": {_IDS[i]: _text(why) for i, (_, why) in enumerate(texts)},
        },
        "provenance": {
            "source_question_id": source_id,
            "start": question.get("start"),
            "end": question.get("end"),
        },
        "review": {"status": "pending", "note": None, "metadata": {}},
    }


def _text(text: str, *, spoken: bool = False) -> dict[str, str]:
    """Mark a text with its origin: ``spoken`` only when it is what was said, word for word.

    Word for word as :func:`quiz_schema.normalise_spoken` compares it, so a stem that lost its
    question mark is ``generated``.
    """
    return {"text": text.strip(), "origin": "spoken" if spoken else "generated"}


def _quiz_dict(
    document: Mapping[str, Any],
    config: Config,
    items: list[dict[str, Any]],
    questions_name: str,
    questions_hash: str,
) -> dict[str, Any]:
    """Build the ``quiz.json`` document around the items; nothing in it changes between runs."""
    base = questions_name.removesuffix(".json").removesuffix(".questions")
    language = (document.get("config") or {}).get("language") or config.language
    return {
        "schema_version": QUIZ_SCHEMA_VERSION,
        "id": f"quiz_{questions_hash[:12]}",
        "title": base,
        "language": language,
        "status": "draft",
        "source": {
            "questions_path": questions_name,
            "questions_schema_version": str(document.get("schema_version", "")),
            "questions_hash": questions_hash,
        },
        "instructions": {
            "shuffle_questions": False,
            "shuffle_options": True,
            "show_feedback": True,
        },
        "generation": {
            "method": "llm",
            "model": config.quiz_model,
            "prompt_version": QUIZ_PROMPT_VERSION,
        },
        "questions": items,
    }

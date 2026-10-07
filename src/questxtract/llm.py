"""Stage 8, the judge: a text LLM decides which candidates are real questions (architecture.md 10).

Off unless ``config.detector == "llm"``; the default ``Config`` makes zero outbound calls and
the rules alone produce the result.

The hard constraint from the requirements (4.5, "the model shall not create questions on
their own"): the model only judges candidates the code chose, by sentence id, and every id
is checked against the request it came from. It may reword a question it kept, and the export
keeps the spoken version beside it (``original_text``); it can never add one that was not
spoken. Quiz generation is generative and lives in a separate command (architecture.md 12.3).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from . import endpoint
from .config import Config
from .models import Context, QAPair, QuestionCandidate, Sentence, stage

# The prompt and the schema are module-level, version-tagged, so a prompt change is a reviewable
# diff. A change to either changes the cache key, so no stale answer is replayed.
JUDGE_PROMPT_VERSION = "v5"

_KINDS = ("content", "rhetorical", "check_in", "procedural")
_FIELDS = ("reason", "kind", "question", "answer_sentence_ids", "confidence")

# Label definitions condensed from docs/annotation_guidelines.md 2, so the model and the
# labelled set draw the same lines.
_PROMPT = """\
You judge the questions asked in a university lecture. The transcript comes from speech
recognition: punctuation and sentence breaks can be wrong. Students are often inaudible, so the
lecturer repeating a student's question in full counts as that question.

You get an excerpt of the lecture, one sentence per line: its id, its start time in seconds, its
text. Lines starting with ">>" are the candidates: judge each of them, and only them, with the
other lines as context. "[said 5x in this lecture]" means the lecturer uses these exact words
that often; a form repeated that much is usually a habit, such as a confirmation tag or a
check-in, rather than a real question.

First ask what the candidate asks about. Who answers it does not matter: lecturers answer most
of their own questions, straight away or after a pause, and that makes none of them rhetorical.

Label each candidate with one kind:
- content: it asks about the subject: a concept, a definition, a property, a fact, a step of
  the argument or the calculation. "What is a group?", "What property do we need here?", "Is
  this matrix invertible?", "Can we swap the sum and the integral?", "What do we get if the
  input doubles?". It counts when the lecturer answers it at once, when it introduces what the
  lecturer explains next, when it leans on the board ("Does it converge or diverge?"), and when
  it is a student's question the lecturer repeats in full.
- rhetorical: it asks about no subject matter of its own. Framing that only announces the next
  step and names nothing ("So how do we do that?", "What does it mean?", "What am I doing
  here?"), questions asked for effect ("Isn't that nice?"), and a confirmation tag closing a
  statement ("..., right?", "..., true?").
- check_in: classroom management, not the subject: "Can everyone hear me?", "Is this too fast
  for anyone?", "Any questions?", "Are we together?", and polls whose answer is not about the
  subject.
- procedural: logistics and managing the exchange: "Can you pass me the marker?", asking for a
  name, calling on someone, and echoing only a fragment of a student's question to check it
  ("Sorting and what?"). A full restatement of a student's question is content.

For each candidate give:
- reason: one short sentence: the subject matter it asks about, or that it asks about none.
- kind: one of the four above.
- question: for a content question with messy wording (fillers, false starts, a preamble, or a
  question that only makes sense with the sentence before it), a clean standalone version that
  keeps its meaning and adds nothing that was not said. Otherwise "".
- answer_sentence_ids: for a content question the excerpt answers, the ids of the sentences
  after it that give the answer, in order. Otherwise [].
- confidence: from 0 to 1, how sure you are that the kind is right.

Answer with JSON only: an object with one key per candidate id, in order, each holding these
five fields, e.g. {"s_1a2b3c": {"reason": "...", "kind": "...", "question": "",
"answer_sentence_ids": [], "confidence": 0.0}}. Judge only the candidates, never other lines.\
"""


def _form(text: str) -> str:
    """Return the words of ``text``, lower-cased: what counts as the same question form."""
    return " ".join(re.findall(r"\w+(?:'\w+)?", text.casefold().replace("\u2019", "'")))


def _batches(
    positions: Sequence[int], total: int, window: int, size: int
) -> list[tuple[list[int], range]]:
    """Group candidate positions into requests, each with one contiguous excerpt.

    Every candidate brings ``window`` sentences on each side. A candidate within ``window``
    sentences of the previous one joins its request, up to ``size`` candidates, so the excerpt
    stays one piece of lecture; any other starts a new request. Excerpts may overlap, which is
    how no candidate loses the lecture around it.
    """
    batches: list[tuple[list[int], range]] = []
    for at in positions:
        low, high = max(0, at - window), min(total, at + window + 1)
        if batches and len(batches[-1][0]) < size and at <= batches[-1][0][-1] + window:
            members, span = batches[-1]
            batches[-1] = ([*members, at], range(span.start, high))
        else:
            batches.append(([at], range(low, high)))
    return batches


def _schema(excerpt: Sequence[Sentence], candidates: Sequence[Sentence]) -> dict[str, Any]:
    """Build the response format of one request: exactly these candidates, fixed choices each.

    Under constrained decoding (OpenAI strict mode, Ollama, vLLM, llama.cpp) the model cannot
    skip a candidate, end early, or cite an answer that is not a later sentence of the excerpt.
    Servers differ in how faithfully they apply a schema, so ``_pairs`` checks it all again.
    """
    ids = [s.id for s in excerpt]

    def judgement(sentence: Sentence) -> dict[str, Any]:
        later = ids[ids.index(sentence.id) + 1 :] or ["none"]
        return {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "kind": {"type": "string", "enum": list(_KINDS)},
                "question": {"type": "string"},
                "answer_sentence_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": later},
                },
                "confidence": {"type": "number"},
            },
            "required": list(_FIELDS),
            "additionalProperties": False,
        }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judgements",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {s.id: judgement(s) for s in candidates},
                "required": [s.id for s in candidates],
                "additionalProperties": False,
            },
        },
    }


def _excerpt(excerpt: Sequence[Sentence], candidates: set[str], repeats: Counter[str]) -> str:
    """Render the excerpt as request lines: candidates marked ">>", with their repeat count."""
    lines = []
    for s in excerpt:
        said = repeats[_form(s.text)] if s.id in candidates else 0
        mark = ">> " if s.id in candidates else "   "
        # From 3 on: on the labelled set no content question repeats that often, while most
        # habitual tags do. A real question asked twice is common.
        note = f" [said {said}x in this lecture]" if said >= 3 else ""
        lines.append(f"{mark}{s.id} {s.start:.0f}s{note} {s.text.strip()}")
    return "\n".join(lines)


def _ask(
    config: Config,
    excerpt: Sequence[Sentence],
    candidates: Sequence[Sentence],
    repeats: Counter[str],
    *,
    again: bool = False,
) -> dict[str, Any]:
    """Ask the model to judge ``candidates`` in ``excerpt``; cached on the request.

    Returns the judgements by candidate id. ``again`` marks a follow-up for candidates a first
    answer left out; it also keeps that request from being byte-identical to the first one,
    whose cached answer would otherwise come back instead. The cache key is the whole request
    (:func:`endpoint.ask`), so any change to the transcript, the sentence split, the prompt,
    the model or the endpoint asks again, and nothing else does.

    Raises:
        ConnectionError: If the endpoint fails after the SDK's own retries, or times out twice.
        ValueError: If the model answers with something other than a JSON object twice.

    """
    request: dict[str, Any] = {
        "model": config.judge_model,
        "messages": [
            {"role": "system", "content": _PROMPT},
            {
                "role": "user",
                "content": _excerpt(excerpt, {s.id for s in candidates}, repeats)
                + ("\n\nJudge every line marked >>, leaving none out." if again else ""),
            },
        ],
        "response_format": _schema(excerpt, candidates),
    }
    if config.judge_reasoning_effort:
        request["reasoning_effort"] = config.judge_reasoning_effort
    return endpoint.ask(config, request, name="judge")


def _pairs(
    judgements: dict[str, Any],
    excerpt: Sequence[Sentence],
    candidates: Sequence[Sentence],
    rules: dict[str, QuestionCandidate],
    stats: dict[str, int],
) -> list[QAPair]:
    """Turn the judgements into pairs, keeping only what traces back to the request.

    A judgement for anything but one of ``candidates``, or malformed in any way, is dropped and
    counted in ``judge_dropped``: never added (requirement 4.5). Kinds other than ``content`` are
    counted in ``judge_rejected``. An answer is later sentences of the excerpt; other ids are
    dropped, not trusted.
    """
    position = {s.id: i for i, s in enumerate(excerpt)}
    asked = {s.id for s in candidates}
    out: list[QAPair] = []
    for sentence_id, judgement in judgements.items():
        fields = _fields(judgement)
        if sentence_id not in asked or fields is None:
            stats["judge_dropped"] += 1
            continue
        kind, question, answer_ids, confidence = fields
        if kind != "content":
            stats["judge_rejected"] += 1
            continue
        at = position[sentence_id]
        sentence = excerpt[at]
        answers = sorted({position[a] for a in answer_ids if position.get(a, -1) > at})
        rule = rules.get(sentence_id)
        text = question.strip() or sentence.text.strip()
        out.append(
            QAPair(
                question=QuestionCandidate(
                    sentence_id=sentence_id,
                    text=text,
                    score=round(confidence, 2),
                    question_type=rule.question_type if rule else "other",
                    reasons=(*(rule.reasons if rule else ()), "llm"),
                ),
                answer_sentence_ids=tuple(excerpt[i].id for i in answers),
                answer_text=" ".join(excerpt[i].text.strip() for i in answers),
                confidence=0.0,  # stage 9 fills this in
                start=sentence.start,
                end=sentence.end,
                verdict="keep" if text == sentence.text.strip() else "revise",
                reasons=("llm_answer",) if answers else ("no_answer_found",),
            )
        )
    return out


def _fields(judgement: Any) -> tuple[str, str, list[Any], float] | None:
    """Unpack one judgement, or return ``None`` when any field has the wrong type or range."""
    if not isinstance(judgement, dict):
        return None
    kind = judgement.get("kind")
    question = judgement.get("question")
    answer_ids = judgement.get("answer_sentence_ids")
    confidence = judgement.get("confidence")
    if (
        kind not in _KINDS
        or not isinstance(question, str)
        or not isinstance(answer_ids, list)
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
    ):
        return None
    # An id is a string; anything else in the list is not trusted as an answer, like an id
    # that is not a later sentence of the excerpt.
    answer_ids = [a for a in answer_ids if isinstance(a, str)]
    return str(kind), question, answer_ids, min(1.0, max(0.0, float(confidence)))


def _batch(
    config: Config,
    excerpt: Sequence[Sentence],
    candidates: Sequence[Sentence],
    repeats: Counter[str],
    rules: dict[str, QuestionCandidate],
) -> tuple[list[QAPair], Counter[str]]:
    """Judge one request's candidates, asking once more for any the answer left out."""
    stats: Counter[str] = Counter()
    judgements = _ask(config, excerpt, candidates, repeats)
    missing = [s for s in candidates if s.id not in judgements]
    if missing:  # a server that ignores the schema can still skip some
        judgements = {**_ask(config, excerpt, missing, repeats, again=True), **judgements}
        stats["judge_skipped"] += sum(1 for s in candidates if s.id not in judgements)
    return _pairs(judgements, excerpt, candidates, rules, stats), stats


@stage("judge")
def judge(ctx: Context) -> Context:
    """Replace the rules' candidates and pairs with the ones the LLM judges real questions.

    A no-op unless ``config.detector == "llm"``. The candidates are every sentence ending in "?"
    plus the rules' own; the model decides each of them in a window of the lecture around it
    (``_batches``). Runs before stage 9, so the ``min_confidence`` cut applies to what the model
    kept. Caches each request itself (``_ask``) instead of declaring ``cache=``: its input is the
    transcript, which a config-only header cannot key.

    Raises:
        ValueError: If ``config.detector`` is neither ``rules`` nor ``llm``, the model does
            not return the expected JSON, or it judged none of the candidates.
        ConnectionError: If the endpoint fails. The user asked for the LLM, so there is no
            silent fallback to the rules.

    """
    detector = ctx.config.detector
    if detector == "rules":
        return ctx
    if detector != "llm":
        raise ValueError(f"unknown detector {detector!r}, expected 'rules' or 'llm'")
    sentences = ctx.sentences
    rules = {c.sentence_id: c for c in ctx.candidates}
    positions = [
        i for i, s in enumerate(sentences) if s.text.rstrip().endswith("?") or s.id in rules
    ]
    repeats = Counter(_form(sentences[i].text) for i in positions)
    batches = _batches(
        positions, len(sentences), ctx.config.judge_window, ctx.config.judge_batch_size
    )
    ctx.stats.update(judge_dropped=0, judge_rejected=0, judge_skipped=0)
    pairs: list[QAPair] = []
    pool = ThreadPoolExecutor(max(1, ctx.config.judge_parallel))
    try:
        futures = [
            pool.submit(
                _batch,
                ctx.config,
                sentences[span.start : span.stop],
                [sentences[i] for i in members],
                repeats,
                rules,
            )
            for members, span in batches
        ]
        for done, future in enumerate(as_completed(futures)):
            if ctx.progress:
                ctx.progress("judge", done / len(batches))
            batch_pairs, batch_stats = future.result()
            pairs += batch_pairs
            ctx.stats.update({k: ctx.stats[k] + v for k, v in batch_stats.items()})
    finally:  # on a failure, the requests not yet sent are not sent
        pool.shutdown(wait=True, cancel_futures=True)
    if positions and ctx.stats["judge_skipped"] == len(positions):
        # Some skipped is a model's lapse; all of them is a server answering in another shape
        # (e.g. the judgements wrapped in one more object), and an empty result would hide it.
        raise ValueError(
            f"judge: {ctx.config.judge_model} judged none of the {len(positions)} candidates; "
            "the endpoint may not follow the response format"
        )
    ctx.pairs = tuple(sorted(pairs, key=lambda p: p.start))  # a follow-up answers out of order
    ctx.candidates = tuple(p.question for p in ctx.pairs)
    ctx.stats["candidates"] = len(ctx.candidates)
    ctx.stats["answered"] = sum(1 for p in pairs if p.answer_text)
    return ctx

"""Stage 8, the judge: an LLM decides which candidates are real questions (architecture.md 10)."""

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

JUDGE_PROMPT_VERSION = "v5"
_STATS = (
    "judge_dropped",
    "judge_rejected",
    "judge_skipped",
    "judge_requests",
    "judge_input_tokens",
    "judge_output_tokens",
    "judge_ms",
)

_KINDS = ("content", "rhetorical", "check_in", "procedural")
_FIELDS = ("reason", "kind", "question", "answer_sentence_ids", "confidence")

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
    return " ".join(re.findall(r"\w+(?:'\w+)?", text.casefold().replace("\u2019", "'")))


def _batches(
    positions: Sequence[int], total: int, window: int, size: int
) -> list[tuple[list[int], range]]:
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
    lines = []
    for s in excerpt:
        said = repeats[_form(s.text)] if s.id in candidates else 0
        mark = ">> " if s.id in candidates else "   "
        # 3: no labelled content question repeats that often (architecture.md 10.2)
        note = f" [said {said}x in this lecture]" if said >= 3 else ""
        lines.append(f"{mark}{s.id} {s.start:.0f}s{note} {s.text.strip()}")
    return "\n".join(lines)


def _ask(
    config: Config,
    excerpt: Sequence[Sentence],
    candidates: Sequence[Sentence],
    repeats: Counter[str],
    *,
    usage: Counter[str],
    again: bool = False,
) -> dict[str, Any]:
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
    if config.reasoning_effort:
        request["reasoning_effort"] = config.reasoning_effort
    return endpoint.ask(config, request, name="judge", usage=usage)


def _pairs(
    judgements: dict[str, Any],
    excerpt: Sequence[Sentence],
    candidates: Sequence[Sentence],
    rules: dict[str, QuestionCandidate],
    stats: dict[str, int],
) -> list[QAPair]:
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
                confidence=0.0,
                start=sentence.start,
                end=sentence.end,
                verdict="keep" if text == sentence.text.strip() else "revise",
                reasons=("llm_answer",) if answers else ("no_answer_found",),
            )
        )
    return out


def _fields(judgement: Any) -> tuple[str, str, list[Any], float] | None:
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
    answer_ids = [a for a in answer_ids if isinstance(a, str)]
    return str(kind), question, answer_ids, min(1.0, max(0.0, float(confidence)))


def _batch(
    config: Config,
    excerpt: Sequence[Sentence],
    candidates: Sequence[Sentence],
    repeats: Counter[str],
    rules: dict[str, QuestionCandidate],
) -> tuple[list[QAPair], Counter[str]]:
    stats: Counter[str] = Counter()
    judgements = _ask(config, excerpt, candidates, repeats, usage=stats)
    missing = [s for s in candidates if s.id not in judgements]
    if missing:
        again = _ask(config, excerpt, missing, repeats, usage=stats, again=True)
        judgements = {**again, **judgements}
        stats["judge_skipped"] += sum(1 for s in candidates if s.id not in judgements)
    return _pairs(judgements, excerpt, candidates, rules, stats), stats


@stage("judge")
def judge(ctx: Context) -> Context:
    """Replace the rules' candidates and pairs with the ones the LLM judges real questions."""
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
    ctx.stats.update(dict.fromkeys(_STATS, 0))
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
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if positions and ctx.stats["judge_skipped"] == len(positions):
        raise ValueError(
            f"judge: {ctx.config.judge_model} judged none of the {len(positions)} candidates; "
            "the endpoint may not follow the response format"
        )
    ctx.pairs = tuple(sorted(pairs, key=lambda p: p.start))
    ctx.candidates = tuple(p.question for p in ctx.pairs)
    ctx.stats["candidates"] = len(ctx.candidates)
    ctx.stats["answered"] = sum(1 for p in pairs if p.answer_text)
    return ctx

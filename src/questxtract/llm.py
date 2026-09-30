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

import hashlib
import json
import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Config
from .models import Context, QAPair, QuestionCandidate, Sentence, stage

# The prompt and the schema are module-level, version-tagged, so a prompt change is a reviewable
# diff. A change to either changes the cache key, so no stale answer is replayed.
JUDGE_PROMPT_VERSION = "v4"

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

Label each candidate with one kind:
- content: a question about the subject that a student should be able to answer after the
  lecture: "What is a field?", "How does a Markov chain work?", "Does anyone know why that
  matters?". It counts when the lecturer answers it straight away, when the lecturer puts it to
  the class and waits, and when it is short and leans on what is on the board ("Is this matrix
  invertible?", "Does it converge or diverge?"). A student's question the lecturer restates in
  full counts too.
- rhetorical: asked for effect, not for an answer. The test: if a student answered it out loud,
  would that derail the lecture rather than fit into it? If so, it is rhetorical. Framing that
  only announces the next step and names no concept ("So how do we do that?", "So why is this
  popular?"), rhetorical confirmation ("Isn't that interesting?"), and a confirmation tag
  closing a statement ("..., right?", "..., true?").
- check_in: classroom management, not the subject: "Can everyone hear me?", "Is this too fast
  for anyone?", "Any questions?", and polls whose answer is not about the subject.
- procedural: logistics and managing the exchange: "Can you pass me the marker?", asking for a
  name, calling on someone, echoing a fragment of a student's question to check it.

For each candidate give:
- reason: one short sentence on what the candidate does in the lecture, before you label it.
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


def client(config: Config) -> Any:
    """Build the OpenAI-compatible client.

    ``base_url`` is what makes the offline story work without a second code path: any local
    OpenAI-compatible server (Ollama, llama.cpp, vLLM, LM Studio) plugs in with zero new code.

    Raises:
        RuntimeError: If the optional ``questxtract[llm]`` extra is not installed.

    """
    try:
        from openai import OpenAI  # noqa: PLC0415 - imported lazily so the base install is light
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise RuntimeError("the LLM layer needs `pip install questxtract[llm]`") from exc
    return OpenAI(
        api_key=config.api_key or "local",
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )


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
        # From 3 on: on the labelled set no content question repeats that often, and 61% of
        # rhetorical ones (habitual tags) do. A real question asked twice is common.
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
    whose cached answer would otherwise come back instead. The cache key is the whole request,
    so any change to the transcript, the sentence split, the prompt, the model or the endpoint
    asks again, and nothing else does.

    Raises:
        ConnectionError: If the endpoint fails after the SDK's own retries, or times out.
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
    key = json.dumps([config.base_url, request], sort_keys=True)
    path = Path(config.cache_dir) / "judge" / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
    if not config.from_scratch and path.is_file():
        cached = _parse(path.read_text(encoding="utf-8"))
        if cached is not None:  # a damaged entry is asked again
            return cached

    llm = client(config)  # first: without the extra, it raises the install hint
    from openai import OpenAIError  # noqa: PLC0415 - the optional extra, as in client()

    for _ in range(2):  # a second try for invalid JSON; the SDK retries network errors itself
        try:
            response = _create(llm, request, config.timeout_seconds)
        except (OpenAIError, TimeoutError) as exc:
            raise ConnectionError(f"judge: {config.base_url or 'OpenAI'}: {exc}") from exc
        content = response.choices[0].message.content
        judgements = _parse(content)
        if judgements is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return judgements
    raise ValueError(f"judge: {config.judge_model} did not return the expected JSON, twice")


def _create(llm: Any, request: dict[str, Any], seconds: float) -> Any:
    """Make one request, with a deadline on the whole call.

    The SDK's timeout is per read, and a server that keeps the connection alive while the model
    works never trips it: OpenRouter sends filler bytes, with no overall limit, so a stalled
    request would hang the run. The call runs in a daemon thread and is abandoned past the
    deadline; the process does not wait for it on exit.

    Raises:
        TimeoutError: If no answer arrived within ``seconds``, the SDK's retries included.

    """
    outcome: list[Any] = []

    def call() -> None:
        try:
            outcome.append(llm.chat.completions.create(**request))
        except Exception as exc:  # re-raised in the caller's thread below
            outcome.append(exc)

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    worker.join(seconds)
    if not outcome:
        raise TimeoutError(f"no answer within {seconds:g} s")
    if isinstance(outcome[0], Exception):
        raise outcome[0]
    return outcome[0]


def _parse(content: str | None) -> dict[str, Any] | None:
    """Return a response's judgements by candidate id, or ``None`` when it is no JSON object."""
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


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
    return str(kind), question, answer_ids, min(1.0, max(0.0, float(confidence)))


@stage("judge")
def judge(ctx: Context) -> Context:
    """Replace the rules' candidates and pairs with the ones the LLM judges real questions.

    A no-op unless ``config.detector == "llm"``. The candidates are every sentence ending in "?"
    plus the rules' own; the model decides each of them in a window of the lecture around it
    (``_batches``). Runs before stage 9, so the ``min_confidence`` cut applies to what the model
    kept. Caches each request itself (``_ask``) instead of declaring ``cache=``: its input is the
    transcript, which a config-only header cannot key.

    Raises:
        ValueError: If ``config.detector`` is neither ``rules`` nor ``llm``, or the model does
            not return the expected JSON.
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
    for done, (members, span) in enumerate(batches):
        if ctx.progress:
            ctx.progress("judge", done / len(batches))
        excerpt = sentences[span.start : span.stop]
        candidates = [sentences[i] for i in members]
        judgements = _ask(ctx.config, excerpt, candidates, repeats)
        missing = [s for s in candidates if s.id not in judgements]
        if missing:  # a server that ignores the schema can still skip some: ask once more
            judgements = {**_ask(ctx.config, excerpt, missing, repeats, again=True), **judgements}
            ctx.stats["judge_skipped"] += sum(1 for s in candidates if s.id not in judgements)
        pairs += _pairs(judgements, excerpt, candidates, rules, ctx.stats)
    ctx.pairs = tuple(sorted(pairs, key=lambda p: p.start))  # a follow-up answers out of order
    ctx.candidates = tuple(p.question for p in ctx.pairs)
    ctx.stats["candidates"] = len(ctx.candidates)
    ctx.stats["answered"] = sum(1 for p in pairs if p.answer_text)
    return ctx

"""Stage 8, the judge: a text LLM picks the questions from the transcript (architecture.md 10).

Off unless ``config.detector == "llm"``; the default ``Config`` makes zero outbound calls and
the rules alone produce the result.

The hard constraint from the requirements (4.5, "the model shall not create questions on
their own"): the model answers with sentence ids, and every id is checked against the
transcript it was sent. It may reword a question it found, and the export keeps the spoken
version beside it (``original_text``); it can never add one that was not spoken. Quiz
generation is generative and lives in a separate command (architecture.md 12.3).
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Config
from .models import Context, QAPair, QuestionCandidate, Segment, Sentence, stage

# The prompt and the schema are module-level constants, version-tagged, so a prompt change is
# a reviewable diff. A change to either changes the cache key, so no stale answer is replayed.
JUDGE_PROMPT_VERSION = "v2"

_KINDS = ("content", "rhetorical", "check_in", "procedural")

# Label definitions condensed from docs/annotation_guidelines.md 2, so the model and the
# labelled set draw the same lines.
_PROMPT = """\
You find the questions asked in a university lecture. The transcript comes from speech
recognition: punctuation and sentence breaks can be wrong. Students are often inaudible, so the
lecturer repeating a student's question in full counts as that question.

Each input line is one sentence: its id, its start time in seconds, its text.

Give an entry for every sentence that asks a question, and for every sentence ending in "?",
labelled with one kind:
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

For each one give:
- sentence_id: the id of the sentence holding the question, copied exactly. Only ids from the
  input, one entry per sentence.
- kind: one of the four above.
- question: for a content question with messy wording (fillers, false starts, a preamble, or a
  question that only makes sense with the sentence before it), a clean standalone version that
  keeps its meaning and adds nothing that was not said. Otherwise "".
- answer_sentence_ids: for a content question the lecture answers, the ids of the sentences
  after it that give the answer, in order. Otherwise [].
- confidence: from 0 to 1, how sure you are that the kind is right.

Answer with JSON only, in this shape: {"questions": [{"sentence_id": "...", "kind": "...",
"question": "...", "answer_sentence_ids": ["..."], "confidence": 0.0}]}. Never invent a
question: every entry must be a sentence of the input.\
"""

_ITEM = {
    "type": "object",
    "properties": {
        "sentence_id": {"type": "string"},
        "kind": {"type": "string", "enum": list(_KINDS)},
        "question": {"type": "string"},
        "answer_sentence_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["sentence_id", "kind", "question", "answer_sentence_ids", "confidence"],
    "additionalProperties": False,
}

# Constrained decoding on the server (OpenAI, Ollama, vLLM, llama.cpp). Servers differ in how
# faithfully they apply it, and a router such as OpenRouter may pick one that ignores it, so the
# prompt spells the shape out too and _parse and _pairs check everything again.
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "questions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"questions": {"type": "array", "items": _ITEM}},
            "required": ["questions"],
            "additionalProperties": False,
        },
    },
}


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


def _chunks(
    sentences: Sequence[Sentence], segments: Sequence[Segment], max_chars: int
) -> list[tuple[Sentence, ...]]:
    """Group whole segments into chunks of at most ``max_chars`` of text.

    A segment wider than ``max_chars`` is a chunk on its own rather than being cut.
    """
    # ponytail: #88, no overlap between chunks, so an answer past a chunk boundary is lost.
    # Boundaries fall on segment breaks, which are silences of 2 s or more.
    chunks: list[list[Sentence]] = [[]]
    size = 0
    for segment in segments:
        width = len(segment.text)
        if chunks[-1] and size + width > max_chars:
            chunks.append([])
            size = 0
        chunks[-1].extend(sentences[segment.sentence_start : segment.sentence_end])
        size += width
    return [tuple(c) for c in chunks if c]


def _ask(config: Config, chunk: Sequence[Sentence], only: Sequence[str] = ()) -> list[Any]:
    """Send one chunk to the model and return its list of entries, cached on the request.

    ``only`` narrows the request to those sentence ids, for the ones a first answer skipped.
    The cache key is the whole request, so any change to the transcript, the sentence split,
    the prompt, the model or the endpoint asks again, and nothing else does.

    Raises:
        ConnectionError: If the endpoint fails after the SDK's own retries.
        ValueError: If the model answers with something other than the expected JSON twice.

    """
    lines = "\n".join(f"{s.id} {s.start:.0f}s {s.text.strip()}" for s in chunk)
    if only:
        lines += f"\n\nGive an entry for each of these sentences, and only these: {', '.join(only)}"
    request: dict[str, Any] = {
        "model": config.judge_model,
        "messages": [
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": lines},
        ],
        "response_format": _RESPONSE_FORMAT,
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
        entries = _parse(content)
        if entries is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return entries
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


def _skipped(entries: Sequence[Any], chunk: Sequence[Sentence]) -> list[str]:
    """Ids of the chunk's sentences ending in "?" that no entry names.

    A model can end its list early with valid JSON, which no schema catches: on a chunk with
    57 such sentences, one answer named 4. Every one of them must come back with some kind.
    """
    named = {e.get("sentence_id") for e in entries if isinstance(e, dict)}
    return [s.id for s in chunk if s.text.rstrip().endswith("?") and s.id not in named]


def _parse(content: str | None) -> list[Any] | None:
    """Return the ``questions`` list of a response, or ``None`` when there is none."""
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError:
        return None
    entries = data.get("questions") if isinstance(data, dict) else None
    return entries if isinstance(entries, list) else None


def _pairs(
    entries: Sequence[Any],
    chunk: Sequence[Sentence],
    rules: dict[str, QuestionCandidate],
    stats: dict[str, int],
) -> list[QAPair]:
    """Turn the model's entries into pairs, keeping only what traces back to ``chunk``.

    An entry naming a sentence that is not in the chunk, or malformed in any way, is dropped
    and counted in ``judge_dropped``: never added (requirement 4.5). Kinds other than
    ``content`` are counted in ``judge_rejected``.
    """
    position = {s.id: i for i, s in enumerate(chunk)}
    seen: set[str] = set()
    out: list[QAPair] = []
    for entry in entries:
        fields = _fields(entry)
        if fields is None or fields[0] not in position or fields[0] in seen:
            stats["judge_dropped"] += 1
            continue
        sentence_id, kind, question, answer_ids, confidence = fields
        seen.add(sentence_id)
        if kind != "content":
            stats["judge_rejected"] += 1
            continue
        at = position[sentence_id]
        sentence = chunk[at]
        # An answer comes after its question; ids from elsewhere are dropped, not trusted.
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
                answer_sentence_ids=tuple(chunk[i].id for i in answers),
                answer_text=" ".join(chunk[i].text.strip() for i in answers),
                confidence=0.0,  # stage 9 fills this in
                start=sentence.start,
                end=sentence.end,
                verdict="keep" if text == sentence.text.strip() else "revise",
                reasons=("llm_answer",) if answers else ("no_answer_found",),
            )
        )
    return out


def _fields(entry: Any) -> tuple[str, str, str, list[Any], float] | None:
    """Unpack one entry, or return ``None`` when any field has the wrong type or range."""
    if not isinstance(entry, dict):
        return None
    sentence_id = entry.get("sentence_id")
    kind = entry.get("kind")
    question = entry.get("question")
    answer_ids = entry.get("answer_sentence_ids")
    confidence = entry.get("confidence")
    if (
        not isinstance(sentence_id, str)
        or kind not in _KINDS
        or not isinstance(question, str)
        or not isinstance(answer_ids, list)
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
    ):
        return None
    return sentence_id, str(kind), question, answer_ids, min(1.0, max(0.0, float(confidence)))


@stage("judge")
def judge(ctx: Context) -> Context:
    """Replace the rules' candidates and pairs with the questions the LLM picks.

    A no-op unless ``config.detector == "llm"``. Runs before stage 9, so the ``min_confidence``
    cut applies to what the model picked. Caches each request itself (``_ask``) instead of
    declaring ``cache=``: its input is the transcript, which a config-only header cannot key.

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
    rules = {c.sentence_id: c for c in ctx.candidates}
    chunks = _chunks(ctx.sentences, ctx.segments, ctx.config.judge_chunk_chars)
    ctx.stats.update(judge_dropped=0, judge_rejected=0, judge_skipped=0)
    pairs: list[QAPair] = []
    for done, chunk in enumerate(chunks):
        if ctx.progress:
            ctx.progress("judge", done / len(chunks))
        entries = _ask(ctx.config, chunk)
        skipped = _skipped(entries, chunk)
        if skipped:  # asked once more, for exactly those; what is still missing is counted
            entries = [*entries, *_ask(ctx.config, chunk, skipped)]
            ctx.stats["judge_skipped"] += len(_skipped(entries, chunk))
        pairs += _pairs(entries, chunk, rules, ctx.stats)
    ctx.pairs = tuple(pairs)
    ctx.candidates = tuple(p.question for p in pairs)
    ctx.stats["candidates"] = len(ctx.candidates)
    ctx.stats["answered"] = sum(1 for p in pairs if p.answer_text)
    return ctx

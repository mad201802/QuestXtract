"""Tests for the LLM detector (#88, #89). A fake client stands in for the endpoint: no network."""

from __future__ import annotations

import json
import threading
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openai
import pytest
from jsonschema import Draft202012Validator

from questxtract import Config, Context, Transcript, endpoint, llm
from questxtract.config import DEEPSEEK
from questxtract.export import SCHEMA_PATH, result_to_dict
from questxtract.pipeline import STAGES, run
from questxtract.sentences import split_sentences

# The `context` fixture: "Today we cover sorting." / "What is a stable sort?" / "It keeps equal
# elements in order." The one candidate is the middle sentence, the one ending in "?".


def _judged(kind: str = "content", **fields: Any) -> dict[str, Any]:
    return {
        "reason": "",
        "kind": kind,
        "question": "",
        "answer_sentence_ids": [],
        "confidence": 0.9,
        **fields,
    }


def _run(context: Context, tmp_path: Path, **config: Any) -> Context:
    context.config = context.config.replace(detector="llm", cache_dir=str(tmp_path), **config)
    return run(context, STAGES[2:])


def _lecture(text: str, seconds_apart: float = 1.0) -> Context:
    """A context over ``text``, one word every ``seconds_apart`` seconds."""
    words = [
        {"word": w, "start": seconds_apart * i, "end": seconds_apart * i + 0.5}
        for i, w in enumerate(text.split())
    ]
    end = seconds_apart * len(words)
    transcript = Transcript.from_dict(
        {"segments": [{"text": text, "start": 0.0, "end": end, "words": words}]}
    )
    return Context(source=Path("t.json"), config=Config(), transcript=transcript)


def test_the_rules_detector_makes_no_calls(context, monkeypatch):
    """AC: the default config makes zero outbound calls and leaves the rules' result alone."""

    def _no_client(config: Config, **_: Any) -> Any:
        raise AssertionError("the default config must not build a client")

    monkeypatch.setattr(endpoint, "client", _no_client)
    ctx = run(context, STAGES[2:])
    assert [p.question.text for p in ctx.pairs] == ["What is a stable sort?"]
    assert ctx.pairs[0].verdict == "unjudged"


def test_a_question_keeps_its_id_across_detectors_and_rewordings(
    context, sentences, fake_llm, tmp_path
):
    """#94 AC: the same spoken question has one id under `rules`, under `llm`, and under a
    second LLM run with another model that words it differently, so review edits matched by
    id stay on it."""
    rules = run(context, STAGES[2:]).pairs[0].id
    client = fake_llm(
        {sentences[1].id: _judged(question="What does a stable sort keep?")},
        {sentences[1].id: _judged(question="What makes a sort stable?")},
    )
    first = _run(context, tmp_path).pairs[0]
    second = _run(context, tmp_path, judge_model="qwen3.5:9b").pairs[0]
    assert client.requests[1]["model"] == "qwen3.5:9b"
    assert first.question.text != second.question.text
    assert rules == first.id == second.id


def test_the_schema_requires_exactly_the_candidates_and_later_answers(
    context, sentences, fake_llm, tmp_path
):
    """AC: under constrained decoding the model can skip no candidate and cite no answer that
    is not a later sentence of its excerpt."""
    client = fake_llm({sentences[1].id: _judged()})
    _run(context, tmp_path)
    (request,) = client.requests
    schema = request["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["required"] == [sentences[1].id]
    answers = schema["schema"]["properties"][sentences[1].id]["properties"]["answer_sentence_ids"]
    assert answers["items"]["enum"] == [sentences[2].id]
    assert request["model"] == DEEPSEEK and request["reasoning_effort"] == "none"  # #173
    lines = request["messages"][1]["content"].splitlines()
    assert lines[0].startswith(f"   {sentences[0].id} 0s ")
    assert lines[1] == f">> {sentences[1].id} 2s What is a stable sort?"

    _run(context, tmp_path, reasoning_effort="low", from_scratch=True)
    assert client.requests[1]["reasoning_effort"] == "low"


def test_an_id_that_was_not_a_candidate_is_dropped_and_counted_never_added(
    context, sentences, fake_llm, tmp_path
):
    """AC, requirement 4.5: the model cannot add a question, not even a real sentence it was
    not asked about."""
    fake_llm(
        {
            "s_000000": _judged(),
            sentences[0].id: _judged(),  # a real sentence, but no candidate
            sentences[1].id: _judged(),
        }
    )
    ctx = _run(context, tmp_path)
    assert [p.question.sentence_id for p in ctx.pairs] == [sentences[1].id]
    assert ctx.stats["judge_dropped"] == 2


def test_malformed_judgements_are_dropped_and_other_kinds_counted(fake_llm, tmp_path):
    ctx = _lecture("Is one? Is two? Is three? Is four? Is five?")
    one, two, three, four, five = split_sentences(ctx.transcript)
    fake_llm(
        {
            one.id: "not an object",
            two.id: _judged(kind="question"),
            three.id: _judged(confidence="high"),
            four.id: _judged(kind="check_in"),
            five.id: _judged(confidence=7),
        }
    )
    ctx = _run(ctx, tmp_path)
    assert ctx.stats["judge_dropped"] == 3 and ctx.stats["judge_rejected"] == 1
    (pair,) = ctx.pairs
    assert pair.question.sentence_id == five.id and pair.question.score == 1.0  # clamped


def test_a_later_answer_is_the_text_of_the_cited_sentences(fake_llm, tmp_path):
    """An answer is extractive: the cited sentences' own text, later ones only."""
    ctx = _lecture("What is a stable sort? Let me draw it. Here. It keeps equal keys in order.")
    question, draw, here, answer = split_sentences(ctx.transcript)
    # An earlier id, an unknown one and a non-string (a server not enforcing the schema) are
    # dropped.
    cited = [answer.id, question.id, "s_000000", {"id": answer.id}]
    fake_llm({question.id: _judged(answer_sentence_ids=cited)})
    (pair,) = _run(ctx, tmp_path).pairs
    assert pair.answer_sentence_ids == (answer.id,)
    assert pair.answer_text == "It keeps equal keys in order."
    assert pair.reasons == ("llm_answer",)


def test_a_rewritten_question_exports_the_spoken_one_beside_it(
    context, sentences, fake_llm, tmp_path
):
    """`text` is the rewrite, `original_text` the transcript, `verdict` revise; and the export
    of an LLM run passes the published schema unchanged (no schema bump)."""
    rewrite = "What makes a sorting algorithm stable?"
    fake_llm({sentences[1].id: _judged(question=rewrite)})
    data = result_to_dict(_run(context, tmp_path))
    (question,) = data["questions"]
    assert question["text"] == rewrite
    assert question["original_text"] == "What is a stable sort?"
    assert question["verdict"] == "revise"
    assert question["reasons"] == ["mark", "wh", "llm", "no_answer_found"]
    assert question["question_type"] == "what"  # carried over from the rules
    assert data["config"]["judge_model"] == Config().judge_model
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(data)


def test_a_question_mark_the_rules_rejected_is_still_a_candidate(fake_llm, tmp_path):
    """Candidates are every sentence ending in "?" plus the rules' own: a confirmation tag
    scores under the rules' gate, and the model still judges it."""
    ctx = _lecture("It is sorted, right?")
    (tag,) = split_sentences(ctx.transcript)
    fake_llm({tag.id: _judged(question=f"  {tag.text}  ")})
    (pair,) = _run(ctx, tmp_path).pairs
    assert pair.verdict == "keep"
    assert pair.question.reasons == ("llm",) and pair.question.question_type == "other"


def test_the_threshold_cuts_the_models_own_confidence(context, sentences, fake_llm, tmp_path):
    """The judge runs before `score`, so `min_confidence` applies to what the model kept, and
    to its confidence as it is: the model saw the answer, so no answer weighting on top."""
    fake_llm({sentences[1].id: _judged(confidence=0.5)})
    (pair,) = _run(context, tmp_path).pairs
    assert pair.confidence == 0.5  # unanswered, and not cut to 0.55 * 0.5

    fake_llm({sentences[1].id: _judged(confidence=0.3)})
    assert _run(context, tmp_path, from_scratch=True).pairs == ()


def test_a_form_repeated_in_the_lecture_carries_its_count(fake_llm, tmp_path):
    """AC: from three times on, a candidate line says how often the lecturer says it."""
    ctx = _lecture("True? Is it bounded? True? Is it bounded? True?")
    client = fake_llm({split_sentences(ctx.transcript)[0].id: _judged(kind="rhetorical")})
    _run(ctx, tmp_path, judge_window=0, judge_batch_size=1)
    lines = [r["messages"][1]["content"] for r in client.requests]
    assert sum("[said 3x in this lecture] True?" in line for line in lines) >= 3
    assert not any("Is it bounded?" in line and "[said" in line for line in lines)


def test_nearby_candidates_share_a_request_and_distant_ones_do_not():
    """AC: a candidate within the window of the previous one joins its request, up to the
    batch size; the excerpt is one piece of lecture covering both windows."""
    assert llm._batches([5, 8, 30], 40, 4, 8) == [([5, 8], range(1, 13)), ([30], range(26, 35))]
    assert llm._batches([5, 8, 9], 40, 4, 2) == [([5, 8], range(1, 13)), ([9], range(5, 14))]
    assert llm._batches([0, 39], 40, 4, 8) == [([0], range(0, 5)), ([39], range(35, 40))]


def test_answers_are_cached_on_the_request(context, sentences, fake_llm, tmp_path):
    """Another threshold makes zero calls; --from-scratch makes them again."""
    client = fake_llm({sentences[1].id: _judged()})
    _run(context, tmp_path)
    _run(context, tmp_path, min_confidence=0.2)
    assert len(client.requests) == 1
    _run(context, tmp_path, from_scratch=True)
    assert len(client.requests) == 2
    _run(context, tmp_path, from_scratch=False, judge_model="qwen3.5:9b")  # another request
    assert len(client.requests) == 3

    for entry in (tmp_path / "judge").iterdir():
        entry.write_text("{truncated", encoding="utf-8")
    _run(context, tmp_path)  # a damaged entry is asked again, not an error
    assert len(client.requests) == 4


def test_a_candidate_the_answer_skipped_is_asked_for_again(fake_llm, tmp_path):
    """AC: a server that ignores the schema can skip a candidate; the skipped ones are asked
    for once more, and what is still missing is counted."""
    ctx = _lecture("Is one? Is two?")
    one, two = split_sentences(ctx.transcript)
    client = fake_llm({one.id: _judged()}, {two.id: _judged()})
    ctx = _run(ctx, tmp_path)
    assert [p.question.sentence_id for p in ctx.pairs] == [one.id, two.id]
    follow_up = client.requests[1]
    assert follow_up["response_format"]["json_schema"]["schema"]["required"] == [two.id]
    assert ctx.stats["judge_skipped"] == 0

    fake_llm({one.id: _judged()}, {})  # two skipped twice: counted, and the run goes on
    assert _run(ctx, tmp_path, from_scratch=True).stats["judge_skipped"] == 1


def test_a_reply_that_judges_no_candidate_fails_loudly(context, sentences, fake_llm, tmp_path):
    """A server answering in another shape skips every candidate: an error, not an empty file."""
    fake_llm({"judgements": {sentences[1].id: _judged()}})
    with pytest.raises(ValueError, match="judged none of the 1 candidates"):
        _run(context, tmp_path)


def test_invalid_json_is_asked_again_once_then_fails(context, sentences, fake_llm, tmp_path):
    """Invalid JSON twice ends the run; there is no silent fallback to the rules."""
    client = fake_llm("not json", {sentences[1].id: _judged()})
    assert len(_run(context, tmp_path).pairs) == 1
    assert len(client.requests) == 2

    fake_llm("not json", '["a list, not an object"]')
    with pytest.raises(ValueError, match="did not return the expected JSON, twice"):
        _run(context, tmp_path, from_scratch=True)


def test_an_endpoint_error_is_a_connection_error(context, fake_llm, tmp_path):
    fake_llm(openai.OpenAIError("Connection error."))
    with pytest.raises(ConnectionError, match="judge: http://localhost:11434/v1: Connection error"):
        _run(context, tmp_path, base_url="http://localhost:11434/v1")


@pytest.mark.parametrize("error_type", [openai.OpenAIError, TimeoutError])
def test_provider_error_hides_key_in_message_and_traceback(
    context, fake_llm, tmp_path, monkeypatch, error_type
):
    secret = "sk-fake-provider-error-secret"
    monkeypatch.setenv("QUESTXTRACT_NTNU_API_KEY", secret)  # the default endpoint's key
    message = f"Request rejected for {secret}; credential {secret} is invalid."
    fake_llm(error_type(message), error_type(message))
    with pytest.raises(ConnectionError) as caught:
        _run(context, tmp_path)
    assert secret not in str(caught.value)
    assert "Request rejected for [REDACTED]; credential [REDACTED] is invalid." in str(caught.value)
    assert secret not in "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )


def test_a_request_that_never_answers_times_out(context, monkeypatch, tmp_path):
    """The SDK's timeout is per read; a server sending keep-alive bytes never trips it, so the
    whole call has a deadline of its own."""
    release = threading.Event()

    def create(**request: Any) -> Any:
        release.wait(5)  # stands in for a request that keeps the connection open

    stalled = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(endpoint, "client", lambda config, **_: stalled)
    try:
        with pytest.raises(ConnectionError, match="no answer within 0 s"):
            _run(context, tmp_path, timeout_seconds=0)
    finally:
        release.set()


def test_a_stalled_request_is_sent_once_more(context, sentences, monkeypatch, tmp_path):
    """A stall is usually the provider: one more try, before the run fails."""
    release = threading.Event()
    calls: list[int] = []
    answer = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({sentences[1].id: _judged()}))
            )
        ]
    )

    def create(**request: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            release.wait(5)  # the first try stalls
        return answer

    flaky = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(endpoint, "client", lambda config, **_: flaky)
    try:
        assert len(_run(context, tmp_path, timeout_seconds=1).pairs) == 1
    finally:
        release.set()
    assert len(calls) == 2


def test_parallel_requests_give_the_same_result_in_recording_order(fake_llm, tmp_path):
    ctx = _lecture("Is one? Is two? Is three?")
    one, two, three = split_sentences(ctx.transcript)
    fake_llm({one.id: _judged(), two.id: _judged(), three.id: _judged()})
    ctx = _run(ctx, tmp_path, judge_window=0, judge_batch_size=1, judge_parallel=3)
    assert [p.question.sentence_id for p in ctx.pairs] == [one.id, two.id, three.id]
    assert ctx.stats["judge_dropped"] == 6  # each request's answer names the other two


def test_an_unknown_detector_fails_loudly(context):
    context.config = context.config.replace(detector="LLM")
    with pytest.raises(ValueError, match="unknown detector 'LLM'"):
        run(context, STAGES[2:])


def test_every_request_reports_progress(fake_llm, tmp_path):
    ctx = _lecture("Is one? Is two? Is three?")
    seen: list[float] = []
    ctx.progress = lambda name, fraction: seen.append(fraction) if name == "judge" else None
    client = fake_llm({split_sentences(ctx.transcript)[0].id: _judged()})
    _run(ctx, tmp_path, judge_window=0, judge_batch_size=1)
    assert len(client.requests) == 1 + 2 * 2  # one per candidate; two and three asked again
    assert seen[0] == 0.0 and max(seen) < 1.0 and len(seen) == 4


def test_the_form_of_a_question_ignores_case_and_punctuation():
    assert llm._form("True?") == llm._form("  true ") == "true"
    assert llm._form("Don’t we, people?") == "don't we people"
    assert Counter(map(llm._form, ["Right?", "right.", "Right, right?"])) == Counter(
        {"right": 2, "right right": 1}
    )


def test_stats_record_the_requests_tokens_and_time_of_the_calls(
    context, sentences, fake_llm, tmp_path
):
    """#123 AC: what the calls cost lands in stats; a cached re-run reads as no calls."""
    client = fake_llm({sentences[1].id: _judged()})
    client.usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=80)
    ctx = _run(context, tmp_path)
    assert ctx.stats["judge_requests"] == 1
    assert (ctx.stats["judge_input_tokens"], ctx.stats["judge_output_tokens"]) == (1200, 80)
    assert ctx.stats["judge_ms"] >= 0
    validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    validator.validate(result_to_dict(ctx))  # stats keys are integers: no schema change

    again = _run(context, tmp_path)  # served from the cache
    cost = ("judge_requests", "judge_input_tokens", "judge_output_tokens", "judge_ms")
    assert [again.stats[k] for k in cost] == [0, 0, 0, 0]


def test_a_server_without_usage_still_counts_the_request(context, sentences, fake_llm, tmp_path):
    """#123 AC: no usage field, the run goes on and the token keys stay 0."""
    fake_llm({sentences[1].id: _judged()})  # FakeLLM sends no usage by default
    ctx = _run(context, tmp_path)
    assert ctx.stats["judge_requests"] == 1
    assert ctx.stats["judge_input_tokens"] == ctx.stats["judge_output_tokens"] == 0

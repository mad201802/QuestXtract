"""Tests for the LLM detector (#88). A fake client stands in for the endpoint: no network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import openai
import pytest
from jsonschema import Draft202012Validator

from questxtract import Config, Context, Sentence, Transcript, llm
from questxtract.export import SCHEMA_PATH, result_to_dict
from questxtract.pipeline import STAGES, run
from questxtract.segments import build_segments
from questxtract.sentences import split_sentences


def _entry(sentence: Sentence, kind: str = "content", **fields: Any) -> dict[str, Any]:
    return {
        "sentence_id": sentence.id,
        "kind": kind,
        "question": "",
        "answer_sentence_ids": [],
        "confidence": 0.9,
        **fields,
    }


def _run(context: Context, tmp_path: Path, **config: Any) -> Context:
    context.config = context.config.replace(detector="llm", cache_dir=str(tmp_path), **config)
    return run(context, STAGES[2:])


def test_the_rules_detector_makes_no_calls(context, monkeypatch):
    """AC: the default config makes zero outbound calls and leaves the rules' result alone."""

    def _no_client(config: Config) -> Any:
        raise AssertionError("the default config must not build a client")

    monkeypatch.setattr(llm, "client", _no_client)
    ctx = run(context, STAGES[2:])
    assert [p.question.text for p in ctx.pairs] == ["What is a stable sort?"]
    assert ctx.pairs[0].verdict == "unjudged"


def test_an_unknown_sentence_id_is_dropped_and_counted_never_added(
    context, sentences, fake_llm, tmp_path
):
    """AC, requirement 4.5: the model cannot add a question that was not spoken."""
    invented = {**_entry(sentences[1]), "sentence_id": "s_000000"}
    fake_llm({"questions": [invented, _entry(sentences[1]), _entry(sentences[1])]})
    ctx = _run(context, tmp_path)
    assert [p.question.sentence_id for p in ctx.pairs] == [sentences[1].id]
    assert ctx.stats["judge_dropped"] == 2  # the invented id and the duplicate


def test_malformed_entries_are_dropped_and_other_kinds_counted(
    context, sentences, fake_llm, tmp_path
):
    fake_llm(
        {
            "questions": [
                "not an object",
                _entry(sentences[0], kind="question"),
                _entry(sentences[0], confidence="high"),
                _entry(sentences[0], kind="check_in"),
                _entry(sentences[1], confidence=7),
            ]
        }
    )
    ctx = _run(context, tmp_path)
    assert ctx.stats["judge_dropped"] == 3 and ctx.stats["judge_rejected"] == 1
    assert ctx.pairs[0].question.score == 1.0  # clamped into [0, 1]


def test_a_later_answer_is_the_text_of_the_cited_sentences(fake_llm, tmp_path):
    """AC: an answer several sentences later is extractive: the cited sentences' own text."""
    text = "What is a stable sort? Let me draw it. Here. It keeps equal keys in order."
    transcript = Transcript.from_dict({"segments": [{"text": text, "start": 0.0, "end": 8.0}]})
    ctx = Context(source=Path("t.json"), config=Config(), transcript=transcript)
    question, draw, here, answer = split_sentences(transcript)
    cited = [answer.id, question.id, "s_000000"]  # an earlier and an unknown id are dropped
    fake_llm({"questions": [_entry(question, answer_sentence_ids=cited)]})
    (pair,) = _run(ctx, tmp_path).pairs
    assert pair.answer_sentence_ids == (answer.id,)
    assert pair.answer_text == "It keeps equal keys in order."
    assert pair.reasons == ("llm_answer",)


def test_a_rewritten_question_exports_the_spoken_one_beside_it(
    context, sentences, fake_llm, tmp_path
):
    """AC: `text` is the rewrite, `original_text` the transcript, `verdict` revise; and the
    export of an LLM run passes the published schema unchanged (no schema bump)."""
    rewrite = "What makes a sorting algorithm stable?"
    fake_llm({"questions": [_entry(sentences[1], question=rewrite)]})
    data = result_to_dict(_run(context, tmp_path))
    (question,) = data["questions"]
    assert question["text"] == rewrite
    assert question["original_text"] == "What is a stable sort?"
    assert question["verdict"] == "revise"
    assert question["reasons"] == ["mark", "wh", "llm", "no_answer_found"]
    assert question["question_type"] == "what"  # carried over from the rules
    assert data["config"]["judge_model"] == Config().judge_model
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(data)


def test_a_question_the_rules_missed_is_kept_verbatim(context, sentences, fake_llm, tmp_path):
    fake_llm({"questions": [_entry(sentences[0], question=f"  {sentences[0].text}  ")]})
    (pair,) = _run(context, tmp_path).pairs
    assert pair.verdict == "keep"
    assert pair.question.reasons == ("llm",) and pair.question.question_type == "other"


def test_the_threshold_cuts_after_the_llm(context, sentences, fake_llm, tmp_path):
    """The judge runs before `score`, so `min_confidence` applies to what the model picked."""
    fake_llm({"questions": [_entry(sentences[1], confidence=0.5)]})
    assert _run(context, tmp_path).pairs == ()  # 0.55 * 0.5, unanswered, is under 0.40


def test_answers_are_cached_on_the_request(context, sentences, fake_llm, tmp_path):
    """AC: another threshold makes zero calls; --from-scratch makes them again."""
    client = fake_llm({"questions": [_entry(sentences[1])]})
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


def test_the_request_asks_for_json_matching_the_schema(context, sentences, fake_llm, tmp_path):
    client = fake_llm({"questions": [_entry(sentences[1])]})
    _run(context, tmp_path)
    (request,) = client.requests
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["reasoning_effort"] == "none"
    lines = request["messages"][1]["content"].splitlines()
    assert lines[1].startswith("s_") and lines[1].endswith("2s What is a stable sort?")

    _run(context, tmp_path, judge_reasoning_effort="", from_scratch=True)
    assert "reasoning_effort" not in client.requests[1]


def test_a_question_the_first_answer_skipped_is_asked_for_again(
    context, sentences, fake_llm, tmp_path
):
    """A model can end its list early with valid JSON: every sentence ending in '?' must come
    back with some kind, so the skipped ones are asked for once more."""
    client = fake_llm({"questions": []}, {"questions": [_entry(sentences[1])]})
    ctx = _run(context, tmp_path)
    assert [p.question.sentence_id for p in ctx.pairs] == [sentences[1].id]
    follow_up = client.requests[1]["messages"][1]["content"]
    assert follow_up.endswith(f"and only these: {sentences[1].id}")
    assert ctx.stats["judge_skipped"] == 0

    fake_llm({"questions": []})  # skipped twice: counted, and the run goes on
    assert _run(context, tmp_path, from_scratch=True).stats["judge_skipped"] == 1


def test_invalid_json_is_asked_again_once_then_fails(context, sentences, fake_llm, tmp_path):
    """AC: invalid JSON twice ends the run; there is no silent fallback to the rules."""
    client = fake_llm("not json", {"questions": [_entry(sentences[1])]})
    assert len(_run(context, tmp_path).pairs) == 1
    assert len(client.requests) == 2

    fake_llm("not json", {"answers": []})
    with pytest.raises(ValueError, match="did not return the expected JSON, twice"):
        _run(context, tmp_path, from_scratch=True)


def test_an_endpoint_error_is_a_connection_error(context, fake_llm, tmp_path):
    fake_llm(openai.OpenAIError("Connection error."))
    with pytest.raises(ConnectionError, match="judge: http://localhost:11434/v1: Connection error"):
        _run(context, tmp_path, base_url="http://localhost:11434/v1")


def test_an_unknown_detector_fails_loudly(context):
    context.config = context.config.replace(detector="LLM")
    with pytest.raises(ValueError, match="unknown detector 'LLM'"):
        run(context, STAGES[2:])


def test_chunks_hold_whole_segments_up_to_the_limit():
    texts = ["One two.", "Three four.", "Five six."]
    transcript = Transcript.from_dict(
        {
            "segments": [
                {"text": t, "start": 10.0 * i, "end": 10.0 * i + 1.0} for i, t in enumerate(texts)
            ]
        }
    )
    sentences = split_sentences(transcript)
    segments = build_segments(sentences)  # 9 s silences: one segment per sentence
    assert len(segments) == 3
    widths = [[len(c) for c in llm._chunks(sentences, segments, limit)] for limit in (100, 19, 1)]
    assert widths == [[3], [2, 1], [1, 1, 1]]  # a segment over the limit is a chunk of its own


def test_every_chunk_is_one_request_and_reports_progress(fake_llm, tmp_path):
    text = "What is one? Two. What is three? Four."
    # 4 s between words, so every sentence is a segment of its own
    words = [
        {"word": w, "start": 5.0 * i, "end": 5.0 * i + 1.0} for i, w in enumerate(text.split())
    ]
    transcript = Transcript.from_dict(
        {"segments": [{"text": text, "start": 0.0, "end": 36.0, "words": words}]}
    )
    seen: list[float] = []
    ctx = Context(
        source=Path("t.json"),
        config=Config(judge_chunk_chars=1),
        transcript=transcript,
        progress=lambda name, fraction: seen.append(fraction) if name == "judge" else None,
    )
    client = fake_llm({"questions": []})
    _run(ctx, tmp_path)
    # one per chunk, plus a follow-up for each of the two chunks whose question went unanswered
    assert len(client.requests) == len(ctx.segments) + 2 > 3
    assert seen[0] == 0.0 and max(seen) < 1.0


def test_the_client_points_at_the_configured_endpoint():
    """Building the client makes no request; the endpoint and timeout come from Config."""
    client = llm.client(Config(base_url="http://localhost:11434/v1", timeout_seconds=30))
    assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"
    assert client.timeout == 30

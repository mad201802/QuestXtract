import json

import pytest

from questxtract import (
    QAPair,
    QuestionCandidate,
    SchemaError,
    Sentence,
    Transcript,
    VisualContext,
    Word,
)


def test_visual_context_id_is_stable_and_survives_a_new_caption():
    board = VisualContext(start=0.0, end=5.0, keyframe_path="frames/1.jpg")
    recaptioned = VisualContext(start=0.0, end=5.0, keyframe_path="frames/1.jpg", description="x")
    assert board.id == recaptioned.id and board.id.startswith("v_")
    assert board.id != VisualContext(start=5.0, end=9.0, keyframe_path="frames/2.jpg").id


def test_transcript_to_dict_round_trips_through_from_dict(transcript):
    """What `questxtract transcribe` writes, `questxtract process` reads back unchanged."""
    chunked = Transcript.from_dict(
        {"segments": [{"text": "What is it?", "start": 1.0, "end": 2.0, "avg_logprob": -0.3}]}
    )
    for original in (transcript, chunked):
        assert Transcript.from_dict(original.to_dict()) == original


def test_transcript_from_file_reads_json_and_rejects_non_json(transcript, tmp_path):
    good = tmp_path / "t.json"
    good.write_text(json.dumps(transcript.to_dict()), encoding="utf-8")
    assert Transcript.from_file(good) == transcript

    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(SchemaError, match="not valid JSON"):
        Transcript.from_file(bad)


def test_reads_whisper_style_words():
    t = Transcript.from_dict(
        {
            "segments": [
                {
                    "text": "What is it?",
                    "start": 1.0,
                    "end": 2.0,
                    "avg_logprob": -0.3,
                    "words": [
                        {"word": " What", "start": 1.0, "end": 1.2, "probability": 0.9},
                        {"word": " is", "start": 1.2, "end": 1.5},
                        {"word": " it?", "start": 1.5, "end": 2.0},
                    ],
                }
            ]
        }
    )
    assert t.words[0] == Word(text="What", start=1.0, end=1.2, confidence=0.9)
    assert t.asr_segments[0].avg_logprob == -0.3
    assert t.duration == 2.0


def test_words_are_synthesised_when_the_engine_gives_none():
    t = Transcript.from_dict([{"text": "a b", "start": 0.0, "end": 2.0}])
    assert [w.text for w in t.words] == ["a", "b"]
    assert t.words[0].start == 0.0 and t.words[1].end == 2.0


@pytest.mark.parametrize(
    "data",
    [
        {"chunks": []},
        {"segments": None},
        {"segments": [{"text": "x"}]},
        {"segments": ["not an object"]},
        "not an object at all",
    ],
)
def test_malformed_transcript_raises_schema_error(data):
    with pytest.raises(SchemaError):
        Transcript.from_dict(data)


def test_sentence_id_is_a_content_hash_and_survives_a_rerun():
    a = Sentence(text="What is mass?", word_start=0, word_end=3, start=1.5, end=3.0)
    b = Sentence(text="What is mass?", word_start=9, word_end=12, start=1.5, end=3.0)
    assert a.id == b.id and a.id.startswith("s_")
    moved = Sentence(text="What is mass?", word_start=0, word_end=3, start=1.6, end=3.0)
    assert moved.id != a.id


def test_timings_are_slices_of_the_word_list(transcript, sentences):
    for s in sentences:
        assert s.start == transcript.words[s.word_start].start
        assert s.end == transcript.words[s.word_end - 1].end


def test_fields_are_keyword_only():
    with pytest.raises(TypeError):
        Word("what", 0.0, 1.0)  # type: ignore[misc]


def test_qapair_id_is_deterministic():
    candidate = QuestionCandidate(sentence_id="s_abc", text="Why?", score=0.6)
    a = QAPair(question=candidate, confidence=0.5, start=1.5, end=2.0)
    b = QAPair(question=candidate, confidence=0.9, start=1.5, end=2.0)
    assert a.id == b.id and a.id.startswith("q_")

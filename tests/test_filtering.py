import copy
import dataclasses
import math
import warnings

import pytest

from questxtract import (
    Config,
    Context,
    QAPair,
    QuestionCandidate,
    SchemaError,
    filter_document,
    filter_questions,
    read_questions,
    write_questions,
)
from questxtract.export import result_to_dict
from questxtract.pipeline import STAGES, run

STARTS = (10.0, 299.9, 300.0, 450.0, 599.9, 600.0, 1200.0)


def _pair(start: float) -> QAPair:
    candidate = QuestionCandidate(
        sentence_id=f"s_{int(start)}", text=f"Question at {start}?", score=0.9
    )
    return QAPair(question=candidate, confidence=0.8, start=start, end=start + 2.0)


@pytest.fixture
def ctx(tmp_path) -> Context:
    return Context(
        source=tmp_path / "lecture.mp4",
        config=Config(),
        pairs=tuple(_pair(s) for s in STARTS),
        stats={"candidates": len(STARTS)},
        duration=1300.0,
    )


@pytest.fixture
def document(ctx) -> dict:
    data = result_to_dict(ctx)
    data["segments"] = [
        {"id": 0, "start": 0.0, "end": 400.0, "topic": None, "visual_ids": []},
        {"id": 1, "start": 400.0, "end": 1300.0, "topic": None, "visual_ids": []},
    ]
    for question in data["questions"]:
        question["segment_id"] = 0 if question["start"] < 400 else 1
    data["questions"][3]["review"] = {
        "status": "edited",
        "text": "Fixed?",
        "note": "hi",
        "metadata": {},
    }
    return data


def _starts_of_ctx(ctx: Context) -> list[float]:
    return [p.start for p in ctx.pairs]


def _starts_of_doc(data: dict) -> list[float]:
    return [q["start"] for q in data["questions"]]


def test_context_range_is_half_open_and_keeps_order(ctx):
    window = filter_questions(ctx, start=300, end=600)
    assert _starts_of_ctx(window) == [300.0, 450.0, 599.9]


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ({"start": 600}, [600.0, 1200.0]),
        ({"end": 300}, [10.0, 299.9]),
        ({}, list(STARTS)),
    ],
)
def test_a_missing_bound_leaves_that_side_open(ctx, document, bounds, expected):
    assert _starts_of_ctx(filter_questions(ctx, **bounds)) == expected
    assert _starts_of_doc(filter_document(document, **bounds)) == expected


def test_context_keeps_every_other_field(ctx):
    window = filter_questions(ctx, start=300, end=600)
    assert window is not ctx
    for field in dataclasses.fields(Context):
        if field.name != "pairs":
            assert getattr(window, field.name) == getattr(ctx, field.name)
    assert window.stats is not ctx.stats


def test_filtering_a_real_run(context):
    finished = run(context, STAGES[2:])
    assert len(filter_questions(finished, start=2.0, end=3.0).pairs) == 1
    assert filter_questions(finished, start=3.0).pairs == ()


def test_document_only_shrinks_the_question_list(document):
    window = filter_document(document, start=300, end=600)
    assert _starts_of_doc(window) == [300.0, 450.0, 599.9]
    for key in ("schema_version", "source", "config", "stats", "segments", "visuals"):
        assert window[key] == document[key]
    assert set(window) == set(document)


def test_document_keeps_review_blocks_as_edited(document):
    window = filter_document(document, start=400)
    assert window["questions"][0]["review"]["status"] == "edited"
    assert window["questions"][0]["review"]["text"] == "Fixed?"


def test_document_from_disk(ctx, tmp_path):
    path = tmp_path / "lecture.questions.json"
    write_questions(ctx, path)
    data = read_questions(path)
    window = filter_document(data, start=300, end=600)
    assert _starts_of_doc(window) == [300.0, 450.0, 599.9]
    for key in ("schema_version", "source", "config", "stats", "segments", "visuals"):
        assert window[key] == data[key]
    assert set(window) == set(data)


def test_inputs_are_left_untouched(ctx, document):
    ctx_before = copy.deepcopy(ctx)
    doc_before = copy.deepcopy(document)
    filter_questions(ctx, start=300, end=600)
    window = filter_document(document, start=300, end=600)
    window["questions"][0]["review"]["status"] = "accepted"
    window["segments"].clear()
    with pytest.warns(UserWarning):
        filter_questions(ctx, topic="sorting")
    with pytest.warns(UserWarning):
        filter_document(document, topic="sorting")
    for bad in ({"start": 600, "end": 300}, {"start": math.nan}):
        with pytest.raises(ValueError):
            filter_questions(ctx, **bad)
        with pytest.raises(ValueError):
            filter_document(document, **bad)
    assert ctx == ctx_before
    assert ctx.pairs == ctx_before.pairs
    assert document == doc_before
    assert document["questions"] == doc_before["questions"]


def test_a_range_past_the_recording_is_not_an_error(ctx, document):
    assert filter_questions(ctx, start=5000).pairs == ()
    assert _starts_of_doc(filter_document(document, start=1000, end=99999)) == [1200.0]


def test_an_empty_range_is_allowed(ctx):
    assert filter_questions(ctx, start=300, end=300).pairs == ()


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ({"start": 600, "end": 300}, "must not be after end"),
        ({"start": -1}, "start must not be negative"),
        ({"end": -0.5}, "end must not be negative"),
        ({"start": math.nan}, "start must be finite"),
        ({"end": math.inf}, "end must be finite"),
        ({"start": "300"}, "start must be a number"),
        ({"end": None, "start": True}, "start must be a number"),
        ({"end": [600]}, "end must be a number"),
    ],
)
def test_invalid_ranges_raise(ctx, document, bounds, message):
    with pytest.raises(ValueError, match=message):
        filter_questions(ctx, **bounds)
    with pytest.raises(ValueError, match=message):
        filter_document(document, **bounds)


def test_document_without_questions_is_a_schema_error():
    with pytest.raises(SchemaError, match="'questions' list"):
        filter_document({"segments": []})


def test_document_question_without_a_start_is_a_schema_error(document):
    document["questions"][0]["start"] = "0:10"
    with pytest.raises(SchemaError, match="numeric 'start'"):
        filter_document(document, start=0)


def test_topic_on_unlabelled_data_warns_and_matches_nothing(ctx, document):
    with pytest.warns(UserWarning, match="experimental"):
        assert filter_questions(ctx, topic="sorting").pairs == ()
    with pytest.warns(UserWarning, match="no topic labels"):
        assert filter_document(document, topic="sorting")["questions"] == []


def test_topic_matches_the_questions_segment_topic(document):
    document["segments"][1]["topic"] = "sorting"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        window = filter_document(document, topic="sorting", end=1000)
    assert _starts_of_doc(window) == [450.0, 599.9, 600.0]

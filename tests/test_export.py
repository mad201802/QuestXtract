import dataclasses
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from questxtract import SchemaError, Transcript, VisualContext, read_questions, write_questions
from questxtract.cli import main
from questxtract.export import SCHEMA_PATH, _format_hms, result_to_dict
from questxtract.pipeline import STAGES, run
from questxtract.vision import attach_visual

QUESTION_KEYS = {
    "id",
    "text",
    "original_text",
    "start",
    "end",
    "start_hms",
    "end_hms",
    "confidence",
    "confidence_label",
    "question_type",
    "source",
    "answer",
    "segment_id",
    "sentence_id",
    "visual_ids",
    "verdict",
    "reasons",
    "review",
}

BOARDS = (
    VisualContext(start=0.0, end=5.0, keyframe_path="frames/1.jpg", description="stable sort"),
    VisualContext(start=5.0, end=20.0, keyframe_path="frames/2.jpg", latex=r"O(n \log n)"),
)


@pytest.fixture
def finished(context):
    return run(context, STAGES[2:])


def test_export_emits_a_fixed_key_set_with_nulls_kept(finished):
    data = result_to_dict(finished)
    assert set(data) == {
        "schema_version",
        "source",
        "config",
        "stats",
        "segments",
        "visuals",
        "questions",
    }
    assert data["config"]["min_confidence"] == finished.config.min_confidence
    question = data["questions"][0]
    assert set(question) == QUESTION_KEYS
    assert question["answer"] is None
    assert question["visual_ids"] == []
    assert question["original_text"] is None
    assert question["verdict"] == "unjudged"
    assert question["source"] == "audio"
    assert question["review"] == {"status": "pending", "text": None, "note": None, "metadata": {}}


def test_export_carries_the_timestamp_and_the_label(finished):
    question = result_to_dict(finished)["questions"][0]
    assert question["text"] == "What is a stable sort?"
    assert question["start"] == 2.2 and question["end"] == 4.0
    assert question["start_hms"] == "0:02" and question["end_hms"] == "0:04"
    assert question["confidence_label"] in {"high", "medium", "low"}
    assert question["question_type"] == "what"


def test_questions_point_into_the_exported_segments(finished):
    data = result_to_dict(finished)
    segment = data["segments"][data["questions"][0]["segment_id"]]
    assert segment == {"id": 0, "start": 0.0, "end": 9.0, "topic": None, "visual_ids": []}


def test_a_question_without_a_known_segment_points_at_none(finished):
    pair = finished.pairs[0]
    orphan = dataclasses.replace(pair.question, sentence_id="s_gone")
    finished.pairs = (dataclasses.replace(pair, question=orphan),)
    question = result_to_dict(finished)["questions"][0]
    assert question["segment_id"] is None and question["visual_ids"] == []


def test_board_states_are_listed_once_and_referenced_by_id(context):
    context.visual = BOARDS
    data = result_to_dict(run(context, STAGES[2:]))
    ids = [board.id for board in BOARDS]
    assert [v["id"] for v in data["visuals"]] == ids
    assert data["visuals"][1]["latex"] == r"O(n \log n)"
    assert data["segments"][0]["visual_ids"] == ids
    assert data["questions"][0]["visual_ids"] == ids  # every overlapping state, not the first


def _answered_by_last_sentence(ctx):
    """Give the first question an answer spanning the transcript's last sentence."""
    later = ctx.sentences[-1]
    ctx.pairs = (
        dataclasses.replace(ctx.pairs[0], answer_text=later.text, answer_sentence_ids=(later.id,)),
    )
    return later


def test_an_answer_carries_the_span_of_its_own_sentences(finished):
    later = _answered_by_last_sentence(finished)
    question = result_to_dict(finished)["questions"][0]
    assert (question["start"], question["end"]) == (2.2, 4.0)
    assert (question["answer"]["start"], question["answer"]["end"]) == (later.start, later.end)
    assert question["answer"]["sentence_ids"] == [later.id]


def test_a_recording_with_no_questions_still_exports_the_full_envelope(context, tmp_path):
    """#13 AC5: no questions is a valid result, not an error or an empty file."""
    context.transcript = Transcript.from_dict(
        {
            "segments": [
                {"text": "Today we cover sorting. It keeps order.", "start": 0.0, "end": 5.0}
            ]
        }
    )
    out = tmp_path / "q.json"
    write_questions(run(context, STAGES[2:]), out)
    data = read_questions(out)
    assert data["questions"] == [] and data["stats"]["emitted"] == 0
    assert data["segments"]  # the lecture is still described


@pytest.fixture
def validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_every_export_shape_matches_the_published_schema(finished, validator):
    """#13 AC2 as a CI check: plain, with an answer and board states, and empty."""
    validator.validate(result_to_dict(finished))
    finished.visual = BOARDS
    finished = attach_visual(finished)
    _answered_by_last_sentence(finished)
    validator.validate(result_to_dict(finished))
    finished.pairs = ()
    validator.validate(result_to_dict(finished))


def test_the_schema_rejects_what_the_format_does_not_allow(finished, validator):
    data = result_to_dict(finished)
    data["questions"][0]["review"]["status"] = "maybe"
    with pytest.raises(ValidationError):
        validator.validate(data)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (2.2, "0:02"), (59.6, "1:00"), (90, "1:30"), (3661, "1:01:01")],
)
def test_format_hms_switches_to_hours_past_an_hour(seconds, expected):
    assert _format_hms(seconds) == expected


def test_write_and_read_questions(tmp_path, finished):
    out = tmp_path / "q.json"
    write_questions(finished, out)
    assert read_questions(out) == json.loads(out.read_text(encoding="utf-8"))


def test_read_questions_accepts_a_newer_minor_and_refuses_another_major(tmp_path, finished):
    out = tmp_path / "q.json"
    data = result_to_dict(finished)
    data["schema_version"] = "2.7"
    out.write_text(json.dumps(data), encoding="utf-8")
    assert read_questions(out)["schema_version"] == "2.7"
    data["schema_version"] = "1.1"
    out.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SchemaError, match="reads 2.x"):
        read_questions(out)


@pytest.mark.parametrize(
    "content",
    ["not json", '{"questions": null}', "[]", '{"schema_version": "2.0", "questions": []}'],
)
def test_malformed_questions_file_raises_schema_error(tmp_path, content):
    path = tmp_path / "q.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SchemaError):
        read_questions(path)


def _transcript_file(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "text": "Today we cover sorting. What is a stable sort?",
                        "start": 0.0,
                        "end": 6.0,
                    },
                    {"text": "It keeps equal elements in order.", "start": 6.0, "end": 9.0},
                    {"text": "Does anyone know why that matters", "start": 9.0, "end": 12.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_roundtrip(tmp_path):
    out = tmp_path / "q.json"
    assert main(["process", str(_transcript_file(tmp_path)), "-o", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "2.0"
    # An aux head without a '?' scores 0.4, under the 0.45 detection gate, so it is never a
    # candidate (architecture.md 7.3).
    assert [q["text"] for q in data["questions"]] == ["What is a stable sort?"]
    assert data["stats"]["candidates"] == 1
    assert data["stats"]["sentences"] == 4


def test_cli_writes_to_stdout_with_a_dash_output(tmp_path, capsys):
    assert main(["process", str(_transcript_file(tmp_path)), "-o", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["questions"]


def test_cli_threshold_filters(tmp_path, capsys):
    assert main(["process", str(_transcript_file(tmp_path)), "-t", "0.2", "-o", "-"]) == 0
    assert len(json.loads(capsys.readouterr().out)["questions"]) == 1
    assert main(["process", str(_transcript_file(tmp_path)), "-t", "0.9", "-o", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["questions"] == []


def test_cli_reports_malformed_transcript_readably(tmp_path, capsys):
    path = tmp_path / "t.json"
    path.write_text('{"chunks": []}', encoding="utf-8")
    assert main(["process", str(path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err


def test_cli_reports_missing_file(tmp_path, capsys):
    assert main(["process", str(tmp_path / "missing.json")]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_unimplemented_subcommands_fail_readably(tmp_path, capsys):
    assert main(["eval", str(tmp_path)]) == 1
    assert capsys.readouterr().err.startswith("error: ")
    assert main(["quiz", str(tmp_path / "q.json")]) == 1
    assert capsys.readouterr().err.startswith("error: ")

import json

import pytest

from questxtract import SchemaError, read_questions, write_questions
from questxtract.cli import main
from questxtract.export import _format_hms, result_to_dict
from questxtract.pipeline import STAGES, run

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
    "answer",
    "segment_id",
    "sentence_id",
    "visual",
    "verdict",
    "reasons",
}


@pytest.fixture
def finished(context):
    return run(context, STAGES[2:])


def test_export_emits_a_fixed_key_set_with_nulls_kept(finished):
    data = result_to_dict(finished)
    assert set(data) == {"schema_version", "source", "config", "stats", "questions"}
    question = data["questions"][0]
    assert set(question) == QUESTION_KEYS
    assert question["answer"] is None
    assert question["visual"] is None
    assert question["original_text"] is None
    assert question["verdict"] == "unjudged"


def test_export_carries_the_timestamp_and_the_label(finished):
    question = result_to_dict(finished)["questions"][0]
    assert question["text"] == "What is a stable sort?"
    assert question["start"] == 2.2 and question["end"] == 4.0
    assert question["start_hms"] == "0:02" and question["end_hms"] == "0:04"
    assert question["confidence_label"] in {"high", "medium", "low"}
    assert question["question_type"] == "what"


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


@pytest.mark.parametrize("content", ["not json", '{"questions": null}', "[]"])
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
    assert data["schema_version"] == "1.1"
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
    assert "#12" in capsys.readouterr().err
    assert main(["quiz", str(tmp_path / "q.json")]) == 1
    assert capsys.readouterr().err.startswith("error: ")

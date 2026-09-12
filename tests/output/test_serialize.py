import json

from questxtract import LectureResult, detect_questions, read_result, write_result, write_segments
from questxtract.cli import main


def test_roundtrip_via_cli(tmp_path, segments):
    transcript = tmp_path / "t.json"
    write_segments(segments, transcript)
    out = tmp_path / "q.json"
    assert main([str(transcript), "-o", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["lecture"] == "t.json"
    assert len(data["questions"]) == 2
    assert data["questions"][0]["source"] == "audio"
    assert data["questions"][0]["id"]


def test_result_roundtrip(tmp_path, segments):
    result = LectureResult(lecture="lecture.mp4", questions=tuple(detect_questions(segments)))
    out = tmp_path / "q.json"
    write_result(result, out)
    assert read_result(out) == result
    assert json.loads(out.read_text(encoding="utf-8"))["questions"][0]["question"] == (
        "What is a stable sort?"
    )


def test_cli_reports_malformed_transcript_readably(tmp_path, capsys):
    transcript = tmp_path / "t.json"
    transcript.write_text('{"chunks": []}', encoding="utf-8")
    assert main([str(transcript)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err


def test_cli_reports_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "missing.json")]) == 1
    assert capsys.readouterr().err.startswith("error: ")

import json

from questxtract import TranscriptSegment, detect_questions, read_segments, write_questions
from questxtract.cli import main

SEGMENTS = [
    TranscriptSegment(text="Today we cover sorting. What is a stable sort?", start=0.0, end=6.0),
    TranscriptSegment(text="It keeps equal elements in order.", start=6.0, end=9.0),
    TranscriptSegment(text="Does anyone know why that matters", start=9.0, end=12.0),
]


def test_detects_marked_and_unmarked_questions():
    questions = detect_questions(SEGMENTS)
    assert [q.text for q in questions] == [
        "What is a stable sort?",
        "Does anyone know why that matters",
    ]
    assert questions[0].start == 0.0
    assert questions[1].confidence < questions[0].confidence


def test_statements_are_not_questions():
    assert detect_questions([SEGMENTS[1]]) == []


def test_threshold_filters_low_confidence():
    assert len(detect_questions(SEGMENTS, threshold=0.9)) == 1


def test_roundtrip_via_cli(tmp_path):
    transcript = tmp_path / "t.json"
    raw = [{"text": s.text, "start": s.start, "end": s.end} for s in SEGMENTS]
    transcript.write_text(json.dumps({"segments": raw}))
    out = tmp_path / "q.json"
    assert main([str(transcript), "-o", str(out)]) == 0
    assert len(json.loads(out.read_text())["questions"]) == 2
    assert read_segments(transcript) == SEGMENTS


def test_write_questions(tmp_path):
    out = tmp_path / "q.json"
    write_questions(detect_questions(SEGMENTS), out)
    assert json.loads(out.read_text())["questions"][0]["text"] == "What is a stable sort?"

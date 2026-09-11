import json

import pytest

from questxtract import SchemaError, TranscriptSegment, Word, read_segments, write_segments


def test_segments_roundtrip(tmp_path, segments):
    path = tmp_path / "t.json"
    write_segments(segments, path)
    assert read_segments(path) == segments


def test_reads_whisper_style_words(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "text": "What is it?",
                        "start": 1.0,
                        "end": 2.0,
                        "words": [{"word": " What", "start": 1.0, "end": 1.2}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert read_segments(path) == [
        TranscriptSegment(
            text="What is it?", start=1.0, end=2.0, words=(Word(text="What", start=1.0, end=1.2),)
        )
    ]


@pytest.mark.parametrize(
    "content",
    ["not json", '{"chunks": []}', '{"segments": null}', '{"segments": [{"text": "x"}]}'],
)
def test_malformed_transcript_raises_schema_error(tmp_path, content):
    path = tmp_path / "t.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SchemaError):
        read_segments(path)

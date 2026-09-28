"""Tests for scripts/scratch_to_labels.py's transcript snapping (issue #12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.scratch_to_labels import main, snap

from questxtract.models import Word

_WORDS = [
    Word(text=t, start=float(i), end=i + 0.5)
    for i, t in enumerate("So what is a field? Well, what is a field? Right.".split())
]


def test_snap_takes_times_from_the_quoted_words() -> None:
    [record] = snap([{"start": 0.0, "text": "What is a field?"}], _WORDS)
    assert (record["start"], record["end"]) == (1.0, 4.5)


def test_snap_picks_the_occurrence_nearest_the_noted_start() -> None:
    [record] = snap([{"start": 7.0, "text": "what is a field"}], _WORDS)
    assert (record["start"], record["end"]) == (6.0, 9.5)


def test_snap_rejects_a_quote_not_in_the_transcript() -> None:
    with pytest.raises(ValueError, match="not in the transcript"):
        snap([{"start": 0.0, "text": "what is a ring"}], _WORDS)


def test_main_writes_snapped_labels(tmp_path: Path) -> None:
    transcript = tmp_path / "t.transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "text": "x",
                        "start": 0.0,
                        "end": 12.0,
                        "words": [{"word": w.text, "start": w.start, "end": w.end} for w in _WORDS],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "notes.txt"
    scratch.write_text("0 | So what is a field? | content\n", encoding="utf-8")
    out = tmp_path / "out.labels.jsonl"
    assert main([str(scratch), "-o", str(out), "--transcript", str(transcript)]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert (record["start"], record["end"], record["kind"]) == (0.0, 4.5, "content")

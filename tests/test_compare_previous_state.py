"""Tests for scripts/compare_previous_state.py (#137): offline, the model replaced."""

from __future__ import annotations

import json

import numpy as np
from scripts.compare_previous_state import compare, copied_lines, format_report

from questxtract import Config
from questxtract.vision import _png


def _board(top: int, rows: int) -> bytes:
    page = np.full((240, 320, 3), 230, np.uint8)
    for row in range(rows):
        page[top + 20 * row : top + 20 * row + 6, 30:290] = 40
    return _png(page)


def _answer(text: str, latex: str | None = None) -> dict[str, object]:
    return {"description": "A board.", "text": text, "latex": latex, "confidence": 0.9}


def test_a_line_from_the_previous_reading_that_is_not_on_the_board_is_copied():
    previous = "Definition of a limit\nlim f(x) = L"
    label = "Theorem 2\nevery convergent sequence is bounded"
    answer = "Definition of a limit\nTheorem 2\nevery convergent sequence is bounded"
    assert copied_lines(answer, previous, label) == ["Definition of a limit"]
    # still on the board, a short line, or a misreading of a line on the board: not copied
    assert copied_lines(answer, previous, previous + "\n" + label) == []
    assert copied_lines("x = 0", "x = 0", label) == []
    assert copied_lines("Theorem 2!", "Theorem 2!", "Theorem 2") == []


def test_each_frame_after_the_first_is_read_alone_and_with_the_previous_one(fake_llm, tmp_path):
    (tmp_path / "rec_00010.0.png").write_bytes(_board(20, 4))
    (tmp_path / "rec_00020.0.png").write_bytes(_board(20, 8))  # the same board, written on
    (tmp_path / "rec_00030.0.png").write_bytes(_board(130, 3))  # erased and written again
    labels = [
        {"frame": "rec_00020.0.png", "text": "first board\nmore", "latex": ["x^2"]},
        {"frame": "rec_00010.0.png", "text": "first board", "latex": []},
        {"frame": "rec_00030.0.png", "text": "second board", "latex": []},
    ]
    (tmp_path / "rec.frames.jsonl").write_text(
        "".join(json.dumps({"kind": "blackboard", "diagram": None, **x}) + "\n" for x in labels),
        encoding="utf-8",
    )
    client = fake_llm(
        _answer("first board"),  # 10 s alone
        _answer("first board\nmore", "x^2"),  # 20 s alone
        _answer("first board\nmore", "x^2"),  # 20 s after 10 s
        _answer("second board"),  # 30 s alone
        _answer("second board\nfirst board\nmore"),  # 30 s after 20 s: copied
    )
    rows = compare(tmp_path, Config(cache_dir=str(tmp_path / "cache")))

    assert [(r["frame"], r["previous_frame"], r["sent"]) for r in rows] == [
        ("rec_00020.0.png", "rec_00010.0.png", True),
        ("rec_00030.0.png", "rec_00020.0.png", False),  # erased: the stage sends none
    ]
    assert len(client.requests) == 5
    assert "Text:\nfirst board\n" in client.requests[2]["messages"][0]["content"]
    assert rows[1]["copied"] == {"v0": [], "v2": ["first board"]}

    report = format_report(rows, "a-model")
    assert "| v2 | 2 | 0 | 0.5 | 1/1 | 1.0 | 1 (1) |" in report  # the copy costs CER
    assert "| v2 | 1 | 0 | 0.0 | 1/1 | 1.0 | 0 (0) |" in report  # the pair the stage keeps
    assert "- `rec_00030.0.png`: first board" in report

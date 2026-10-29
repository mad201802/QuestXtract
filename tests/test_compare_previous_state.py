"""Tests for scripts/compare_previous_state.py (#137, #230): offline, the model replaced."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scripts.compare_previous_state import compare, copied_lines, erased_kept, format_report

from questxtract import Config
from questxtract.evaluate import _check_frame
from questxtract.vision import _INK_KEPT, _decoded, _for_comparison, _ink, _ink_kept, _png

SEQUENCE = Path(__file__).parent / "data" / "vision" / "sequence"


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


def test_an_old_form_of_a_line_that_has_grown_is_copied():
    """#230: the board's line is longer now, so the previous reading's form is no longer on it."""
    assert copied_lines("f(x) = x", "f(x) = x", "f(x) = x^2 + 2x + 1") == ["f(x) = x"]
    # two lines of the label read as one is layout, not a copy
    joined = "h(t) = 1/(1+t)   t ≥ 0"
    assert copied_lines(joined, joined, "h(t) = 1/(1+t)\nt ≥ 0") == []


def test_erased_text_counts_as_kept_in_the_text_or_the_latex():
    erased = ["t ≥ 0", "Y_{FR}(now)", "u * h(now)"]
    assert erased_kept(_answer("h(t) = 1/(1+t)"), erased) == []
    assert erased_kept(_answer("h(t)", r"h(t) = \frac{1}{1+t}" + "\nt \\geq 0"), erased) == [
        "t ≥ 0"
    ]
    assert erased_kept(_answer("y_FR (now)\nU * h (now)"), erased) == erased[1:]
    assert erased_kept(_answer("", r"Y_{FR}(\text{now})"), erased) == ["Y_{FR}(now)"]


def test_each_arm_is_chained_on_its_own_readings(fake_llm, tmp_path):
    (tmp_path / "rec_00010.0.png").write_bytes(_board(20, 4))
    (tmp_path / "rec_00020.0.png").write_bytes(_board(20, 8))  # the same board, written on
    (tmp_path / "rec_00030.0.png").write_bytes(_board(130, 3))  # erased and written again
    labels = [
        {"frame": "rec_00020.0.png", "text": "first board\nmore lines here", "latex": ["x^2"]},
        {"frame": "rec_00010.0.png", "text": "first board", "latex": []},
        {"frame": "rec_00030.0.png", "text": "second board", "latex": [], "erased": ["more"]},
    ]
    (tmp_path / "rec.frames.jsonl").write_text(
        "".join(json.dumps({"kind": "blackboard", "diagram": None, **x}) + "\n" for x in labels),
        encoding="utf-8",
    )
    client = fake_llm(
        _answer("first board"),  # 10 s alone, every arm
        _answer("first board\nmore", "x^2"),  # 20 s alone
        _answer("first board\nmore lines here", "x^2"),  # 20 s after 10 s: v2 and stage
        _answer("second board"),  # 30 s alone: v0, and stage after the erase
        _answer("second board\nfirst board\nmore lines here"),  # 30 s after v2's 20 s
    )
    rows = compare(tmp_path, Config(cache_dir=str(tmp_path / "cache")))

    assert [(r["frame"], r["previous_frame"], r["sent"]) for r in rows] == [
        ("rec_00020.0.png", "rec_00010.0.png", True),
        ("rec_00030.0.png", "rec_00020.0.png", False),  # erased: the stage sends none
    ]
    assert len(client.requests) == 5  # the stage's requests are v2's or v0's, cached
    assert "Text:\nfirst board\n" in client.requests[2]["messages"][0]["content"]
    # v2 is given its own reading of 20 s, not v0's
    assert "Text:\nfirst board\nmore lines here\n" in client.requests[4]["messages"][0]["content"]
    assert rows[1]["answers"]["stage"] == rows[1]["answers"]["v0"]
    assert rows[1]["copied"] == {
        "v0": [],
        "v2": ["first board", "more lines here"],
        "stage": [],
    }
    assert rows[1]["erased"] == {"v0": [], "v2": ["more"], "stage": []}

    report = format_report(rows, "a-model")
    assert "| v2 | 2 | 0 | 0.5 | 1/1 | 1.0 | 2 (1) | 1/1 |" in report  # the copy costs CER
    assert "| stage | 2 | 0 | 0.0 | 1/1 | 1.0 | 0 (0) | 0/1 |" in report
    assert "| stage | 1 | 0 | 0.0 | 1/1 | 1.0 | 0 (0) | 0/0 |" in report  # the pair it sends
    assert "| 0 / 2 / 0 | 0 / 1 / 0 |" in report
    assert "- `rec_00030.0.png`: more lines here" in report
    assert "Erased text v2 still reads:\n\n- `rec_00030.0.png`: more" in report


def _sequence() -> list[dict[str, Any]]:
    lines = (SEQUENCE / "ttk4225_2026_09_14.frames.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in lines.splitlines()]


def test_the_sequence_labels_name_text_erased_since_the_view_before():
    """#230: consecutive views of the stage, each erased item gone from the board since."""
    labels = _sequence()
    assert len(labels) >= 2
    for number, (before, label) in enumerate(pairwise(labels), 2):
        _check_frame(label, f"sequence:{number}")
        assert (SEQUENCE / label["frame"]).is_file()
        assert before["frame"] < label["frame"]  # in time order
        for item in label["erased"]:
            assert item in before["text"] and item not in label["text"], item


def test_the_stage_sends_the_previous_state_across_the_sequences_partial_erase():
    """#230 AC: the partial erase keeps enough ink that the stage reads the next view with v2."""
    inks = [
        _ink(_for_comparison(_decoded((SEQUENCE / x["frame"]).read_bytes()))) for x in _sequence()
    ]
    kept = [_ink_kept(was, now) for was, now in pairwise(inks)]
    erases = [k for k, x in zip(kept, _sequence()[1:], strict=True) if x["erased"]]
    assert erases and all(k >= _INK_KEPT for k in kept)
    assert min(erases) < 0.8  # the right half wiped: 0.79

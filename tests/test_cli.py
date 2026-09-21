"""Tests for the terminal progress bar (issue #14): cli.py._progress_bar / _draw.

Existing CLI tests (test_export.py) check JSON and exit codes but never inspect stderr,
so the renderer itself - stage numbering, completing a stage before advancing, and
closing the line before an error - was previously untested.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from questxtract import cli
from questxtract.cli import main
from questxtract.pipeline import STAGES


def test_draw_renders_fraction_as_filled_blocks(capsys: pytest.CaptureFixture[str]) -> None:
    cli._draw(1, 3, "score", 0.7)
    err = capsys.readouterr().err
    assert err.startswith("\r[1/3]")
    assert "█" * 7 + "░" * 3 in err
    assert "70%" in err


def test_progress_bar_falls_back_to_a_title_cased_label_for_an_unknown_stage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = cli._progress_bar(1)
    report("some_new_stage", 0.0)
    assert "Some new stage" in capsys.readouterr().err


def test_progress_bar_completes_each_stage_before_advancing_to_the_next(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = cli._progress_bar(2)
    report("sentences", 0.0)
    report("sentences", 0.5)
    report("segments", 0.0)
    report("segments", 1.0)
    report("done", 1.0)

    err = capsys.readouterr().err
    first = err.index("[1/2] Splitting sentences")
    hundred = err.index("100%", first)
    second = err.index("[2/2] Grouping segments")
    # the first stage is redrawn to 100% before the bar ever shows the second stage's line
    assert first < hundred < second
    assert err.endswith("\n")  # "done" closes the redrawing line


def _transcript_file(tmp_path: Path) -> Path:
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps({"segments": [{"text": "Hi?", "start": 0.0, "end": 1.0}]}), encoding="utf-8"
    )
    return path


def test_process_closes_the_bar_line_before_an_error_propagates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage failing mid-run must not leave the bar's `\\r`-redrawn line dangling in
    front of the traceback."""

    def _boom(ctx: Any, stages: Any, progress: Callable[[str, float], None]) -> Any:
        del ctx, stages
        progress("sentences", 0.0)
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        main(["process", str(_transcript_file(tmp_path))])

    err = capsys.readouterr().err
    # `process` skips ingest and transcribe: the transcript is already in hand.
    assert f"[1/{len(STAGES) - 2}]" in err and "Splitting sentences" in err
    assert err.endswith("\n")

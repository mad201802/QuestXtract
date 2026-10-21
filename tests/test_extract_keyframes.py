"""Tests for scripts/extract_keyframes.py, whose defaults come from Config."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.extract_keyframes import main


def test_the_options_parse_with_the_current_config(capsys: pytest.CaptureFixture[str]) -> None:
    """#130 review: the parser reads its defaults from Config before any argument is checked."""
    with pytest.raises(SystemExit) as stop:
        main(["--help"])
    assert stop.value.code == 0
    assert "--max-views" in capsys.readouterr().out


def test_a_missing_recording_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "missing.mp4"), "-o", str(tmp_path / "out")]) == 1
    assert "is not a file" in capsys.readouterr().err

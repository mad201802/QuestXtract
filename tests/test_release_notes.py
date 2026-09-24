"""Tests for scripts/release_notes.py (issue #45)."""

from pathlib import Path

import pytest
from scripts.release_notes import (
    ReleaseError,
    changelog_section,
    main,
    project_version,
    release_notes,
    tag_version,
)

PYPROJECT = """\
[project]
name = "questxtract"
version = "0.2.0"

[tool.other]
version = "9.9.9"
"""

CHANGELOG = """\
# Changelog

## Unreleased

- Not yet.

## [0.2.0] - 2026-10-01

### Added

- Automatic PyPI releases.

## 0.1.0

- First release.
"""


@pytest.mark.parametrize(
    ("tag", "version"),
    [("v0.2.0", "0.2.0"), ("v1.10.3", "1.10.3"), ("v0.2.0rc1", "0.2.0rc1")],
)
def test_tag_version(tag: str, version: str) -> None:
    assert tag_version(tag) == version


@pytest.mark.parametrize("tag", ["0.2.0", "v0.2", "release-0.2.0", "v0.2.0-foo"])
def test_tag_version_rejects_malformed(tag: str) -> None:
    with pytest.raises(ReleaseError, match="not of the form"):
        tag_version(tag)


def test_project_version_reads_only_project_table() -> None:
    assert project_version(PYPROJECT) == "0.2.0"


def test_project_version_missing() -> None:
    with pytest.raises(ReleaseError, match="no version"):
        project_version('[tool.other]\nversion = "1.0.0"\n')


def test_changelog_section_stops_at_next_heading() -> None:
    notes = changelog_section(CHANGELOG, "0.2.0")
    assert notes == "### Added\n\n- Automatic PyPI releases.\n"


def test_changelog_section_plain_heading_at_end_of_file() -> None:
    assert changelog_section(CHANGELOG, "0.1.0") == "- First release.\n"


def test_changelog_section_does_not_match_prefix() -> None:
    with pytest.raises(ReleaseError, match="no '## 0.1.0' section"):
        changelog_section("## 0.1.01\n\n- x\n", "0.1.0")


def test_changelog_section_empty() -> None:
    with pytest.raises(ReleaseError, match="empty"):
        changelog_section("## 0.2.0\n\n## 0.1.0\n- x\n", "0.2.0")


def test_release_notes_version_mismatch() -> None:
    with pytest.raises(ReleaseError, match="does not match"):
        release_notes("v0.3.0", PYPROJECT, CHANGELOG)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    return tmp_path


def test_main_writes_notes_file(tmp_path: Path) -> None:
    out = tmp_path / "notes.md"
    assert main(["v0.2.0", "-o", str(out), "--root", str(_repo(tmp_path))]) == 0
    assert "Automatic PyPI releases." in out.read_text(encoding="utf-8")


def test_main_prints_notes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["v0.1.0", "--root", str(_repo(tmp_path))]) == 1  # pyproject says 0.2.0
    assert "does not match" in capsys.readouterr().err
    assert main(["v0.2.0", "--root", str(_repo(tmp_path))]) == 0
    assert "Automatic PyPI releases." in capsys.readouterr().out


def test_main_missing_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["v0.2.0", "--root", str(tmp_path)]) == 1
    assert "release_notes:" in capsys.readouterr().err

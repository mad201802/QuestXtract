"""Tests for scripts/validate_labels.py (issue #11)."""

from __future__ import annotations

from pathlib import Path

from scripts.validate_labels import main, validate_file


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_file_has_no_errors(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "a.labels.jsonl",
        '{"start": 1.0, "end": 2.0, "text": "what is this", "kind": "content"}\n'
        '{"start": 3.0, "end": 4.0, "text": "can you hear me", "kind": "check_in", '
        '"split": "blind"}\n',
    )
    assert validate_file(path) == []


def test_defaults_split_to_seed_when_absent(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "a.labels.jsonl",
        '{"start": 1.0, "end": 2.0, "text": "what is this", "kind": "content"}\n',
    )
    assert validate_file(path) == []


def test_missing_required_fields_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.labels.jsonl", '{"start": 1.0, "end": 2.0}\n')
    errors = validate_file(path)
    messages = [e.message for e in errors]
    assert any("text" in m for m in messages)
    assert any("kind" in m for m in messages)


def test_end_before_start_reported(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "a.labels.jsonl", '{"start": 5.0, "end": 2.0, "text": "x", "kind": "content"}\n'
    )
    errors = validate_file(path)
    assert any("greater than" in e.message for e in errors)


def test_negative_start_reported(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "a.labels.jsonl",
        '{"start": -1.0, "end": 2.0, "text": "x", "kind": "content"}\n',
    )
    errors = validate_file(path)
    assert any("'start' must be >= 0" in e.message for e in errors)


def test_empty_text_reported(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "a.labels.jsonl", '{"start": 1.0, "end": 2.0, "text": "  ", "kind": "content"}\n'
    )
    errors = validate_file(path)
    assert any("non-empty string" in e.message for e in errors)


def test_invalid_kind_reported(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "a.labels.jsonl", '{"start": 1.0, "end": 2.0, "text": "x", "kind": "maybe"}\n'
    )
    errors = validate_file(path)
    assert any("'kind' must be one of" in e.message for e in errors)


def test_invalid_split_reported(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "a.labels.jsonl",
        '{"start": 1.0, "end": 2.0, "text": "x", "kind": "content", "split": "training"}\n',
    )
    errors = validate_file(path)
    assert any("'split' must be one of" in e.message for e in errors)


def test_overlapping_entries_reported(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "a.labels.jsonl",
        '{"start": 10.0, "end": 20.0, "text": "a", "kind": "content"}\n'
        '{"start": 15.0, "end": 25.0, "text": "b", "kind": "content"}\n',
    )
    errors = validate_file(path)
    assert any("overlaps" in e.message for e in errors)


def test_malformed_json_line_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.labels.jsonl", "not json\n")
    errors = validate_file(path)
    assert any("not valid JSON" in e.message for e in errors)


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "a.labels.jsonl",
        '{"start": 1.0, "end": 2.0, "text": "x", "kind": "content"}\n\n\n',
    )
    assert validate_file(path) == []


def test_empty_file_reports_no_entries(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.labels.jsonl", "")
    errors = validate_file(path)
    assert any("no valid labelled entries" in e.message for e in errors)


def test_missing_file_reported_not_raised(tmp_path: Path) -> None:
    errors = validate_file(tmp_path / "does_not_exist.labels.jsonl")
    assert any("could not read file" in e.message for e in errors)


def test_cli_exit_code_zero_on_valid_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "a.labels.jsonl", '{"start": 1.0, "end": 2.0, "text": "x", "kind": "content"}\n'
    )
    assert main([str(path)]) == 0


def test_cli_exit_code_one_on_invalid_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.labels.jsonl", "not json\n")
    assert main([str(path)]) == 1


def test_cli_exit_code_one_when_no_files_found(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert main([str(empty_dir)]) == 1


def test_cli_directory_finds_labels_files(tmp_path: Path) -> None:
    _write(
        tmp_path, "a.labels.jsonl", '{"start": 1.0, "end": 2.0, "text": "x", "kind": "content"}\n'
    )
    _write(tmp_path, "not_a_labels_file.txt", "irrelevant")
    assert main([str(tmp_path)]) == 0

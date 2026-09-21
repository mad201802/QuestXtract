"""Validate a labelled dataset file against the schema `docs/annotation_guidelines.md` defines.

Not part of the evaluation harness (`evaluate.py`, issue #12) - this is standalone tooling for
issue #11: it checks that a `.labels.jsonl` file is well-formed *before* it is committed or
consumed by anything downstream, independent of whether the pipeline that reads it exists yet.

Usage:
    uv run python scripts/validate_labels.py tests/data/*.labels.jsonl
    uv run python scripts/validate_labels.py tests/data/ --recursive

Exit code 0 when every file is valid, 1 otherwise, with every problem in every file reported
in one pass rather than stopping at the first error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

_VALID_KINDS = frozenset({"content", "rhetorical", "check_in", "procedural"})
_VALID_SPLITS = frozenset({"seed", "blind"})


@dataclass(frozen=True)
class LabelError:
    """One problem found in one line of one file."""

    path: Path
    line_number: int
    message: str

    def __str__(self) -> str:
        """Format as ``path:line: message``, matching a compiler-style error line."""
        return f"{self.path}:{self.line_number}: {self.message}"


def _iter_jsonl_files(paths: Sequence[str], *, recursive: bool) -> Iterator[Path]:
    """Expand a mix of files and directories into `.labels.jsonl` file paths."""
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            pattern = "**/*.labels.jsonl" if recursive else "*.labels.jsonl"
            yield from sorted(path.glob(pattern))
        elif path.is_file():
            yield path
        else:
            yield path  # let validate_file report the missing-file error uniformly


def validate_line(
    path: Path, line_number: int, raw_line: str
) -> tuple[dict | None, list[LabelError]]:
    """Parse and validate one JSONL line.

    Returns the parsed record (or ``None`` if parsing failed) and every error found on it.
    """
    errors: list[LabelError] = []
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        return None, [LabelError(path, line_number, f"not valid JSON ({exc})")]

    if not isinstance(record, dict):
        return None, [
            LabelError(path, line_number, f"expected an object, got {type(record).__name__}")
        ]

    for field in ("start", "end", "text", "kind"):
        if field not in record:
            errors.append(LabelError(path, line_number, f"missing required field '{field}'"))

    start = record.get("start")
    end = record.get("end")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        if start < 0:
            errors.append(LabelError(path, line_number, f"'start' must be >= 0, got {start}"))
        if end <= start:
            errors.append(
                LabelError(
                    path, line_number, f"'end' ({end}) must be greater than 'start' ({start})"
                )
            )
    elif "start" in record or "end" in record:
        errors.append(LabelError(path, line_number, "'start' and 'end' must both be numbers"))

    text = record.get("text")
    if "text" in record and (not isinstance(text, str) or not text.strip()):
        errors.append(LabelError(path, line_number, "'text' must be a non-empty string"))

    kind = record.get("kind")
    if "kind" in record and kind not in _VALID_KINDS:
        errors.append(
            LabelError(
                path, line_number, f"'kind' must be one of {sorted(_VALID_KINDS)}, got {kind!r}"
            )
        )

    split = record.get("split", "seed")
    if split not in _VALID_SPLITS:
        errors.append(
            LabelError(
                path, line_number, f"'split' must be one of {sorted(_VALID_SPLITS)}, got {split!r}"
            )
        )

    return record, errors


def validate_file(path: Path) -> list[LabelError]:
    """Validate every line of one `.labels.jsonl` file, plus cross-line invariants.

    Raises no exceptions for a missing or unreadable file: that is reported as a `LabelError`
    like any other problem, so a batch run over many files always finishes and reports
    everything at once.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [LabelError(path, 0, f"could not read file: {exc}")]

    errors: list[LabelError] = []
    records: list[tuple[int, dict]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        record, line_errors = validate_line(path, line_number, raw_line)
        errors.extend(line_errors)
        if record is not None and not line_errors:
            records.append((line_number, record))

    if not records:
        errors.append(LabelError(path, 0, "file contains no valid labelled entries"))

    errors.extend(_check_no_overlaps(path, records))
    return errors


def _check_no_overlaps(path: Path, records: list[tuple[int, dict]]) -> list[LabelError]:
    """Flag entries whose time spans overlap - each labelled span should be one utterance."""
    errors: list[LabelError] = []
    by_start = sorted(records, key=lambda r: r[1]["start"])
    for (line_a, a), (line_b, b) in zip(by_start, by_start[1:], strict=False):
        if b["start"] < a["end"]:
            errors.append(
                LabelError(
                    path,
                    line_b,
                    f"overlaps entry at line {line_a} ({a['start']}-{a['end']} vs "
                    f"{b['start']}-{b['end']})",
                )
            )
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation over every path given. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="`.labels.jsonl` files or directories")
    parser.add_argument("--recursive", action="store_true", help="search directories recursively")
    args = parser.parse_args(argv)

    files = list(_iter_jsonl_files(args.paths, recursive=args.recursive))
    if not files:
        print("error: no .labels.jsonl files found", file=sys.stderr)
        return 1

    total_errors = 0
    for path in files:
        errors = validate_file(path)
        if errors:
            for error in errors:
                print(str(error), file=sys.stderr)
            total_errors += len(errors)
        else:
            print(f"ok: {path}")

    if total_errors:
        print(f"\n{total_errors} problem(s) across {len(files)} file(s)", file=sys.stderr)
        return 1
    print(f"\n{len(files)} file(s) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

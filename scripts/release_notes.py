"""Check a release tag against the package version and print its CHANGELOG section.

Run by `.github/workflows/release.yml` (issue #45) before anything is built or uploaded, so a tag
that disagrees with `pyproject.toml`, or a version with no CHANGELOG entry, fails the release
instead of publishing to PyPI. Stdlib only: the release job runs it without syncing the project.

Usage:
    python scripts/release_notes.py v0.2.0                   # notes to stdout
    python scripts/release_notes.py v0.2.0 -o notes.md       # notes to a file

The CHANGELOG section is the one whose `## ` heading names the version, with or without a
leading `v`, brackets or a trailing date: `## 0.2.0`, `## [0.2.0] - 2026-10-01`, `## v0.2.0`.
It runs until the next `## ` heading. Exit code 0 on success, 1 with a message on stderr otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TAG = re.compile(r"^v(\d+\.\d+\.\d+(?:[-.]?(?:a|b|rc|post|dev)\d*)*)$")


class ReleaseError(Exception):
    """The tag, the package version and the CHANGELOG do not agree."""


def tag_version(tag: str) -> str:
    """Return the version a `vX.Y.Z` tag names, e.g. `v0.2.0` -> `0.2.0`."""
    match = _TAG.match(tag.strip())
    if match is None:
        raise ReleaseError(f"tag {tag!r} is not of the form vX.Y.Z")
    return match.group(1)


def project_version(pyproject: str) -> str:
    """Return `version` from the `[project]` table of a pyproject.toml's text."""
    in_project = False
    for line in pyproject.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
        elif in_project:
            match = re.match(r'^version\s*=\s*"([^"]+)"', stripped)
            if match:
                return match.group(1)
    raise ReleaseError("no version in the [project] table of pyproject.toml")


def changelog_section(changelog: str, version: str) -> str:
    """Return the body of the CHANGELOG section for `version`, without its heading."""
    heading = re.compile(rf"^##\s+\[?v?{re.escape(version)}\]?(?:\s|$)")
    lines = changelog.splitlines()
    for start, line in enumerate(lines):
        if heading.match(line):
            end = next(
                (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
                len(lines),
            )
            body = "\n".join(lines[start + 1 : end]).strip()
            if not body:
                raise ReleaseError(f"CHANGELOG section for {version} is empty")
            return body + "\n"
    raise ReleaseError(
        f"CHANGELOG.md has no '## {version}' section; rename '## Unreleased' before tagging"
    )


def release_notes(tag: str, pyproject: str, changelog: str) -> str:
    """Check `tag` against the package version and return that version's CHANGELOG section."""
    version = tag_version(tag)
    declared = project_version(pyproject)
    if version != declared:
        raise ReleaseError(f"tag {tag} does not match pyproject.toml version {declared}")
    return changelog_section(changelog, version)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", help="the release tag, e.g. v0.2.0")
    parser.add_argument("-o", "--output", type=Path, help="write notes here instead of stdout")
    parser.add_argument("--root", type=Path, default=_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        notes = release_notes(
            args.tag,
            (args.root / "pyproject.toml").read_text(encoding="utf-8"),
            (args.root / "CHANGELOG.md").read_text(encoding="utf-8"),
        )
    except (ReleaseError, OSError) as exc:
        print(f"release_notes: {exc}", file=sys.stderr)
        return 1

    if args.output is None:
        sys.stdout.write(notes)
    else:
        args.output.write_text(notes, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

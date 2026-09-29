r"""Convert a plain-text labelling scratch file into `.labels.jsonl` (issue #11).

Scratch file format, one entry per line, fields separated by ``|``:

    142.0 | what is a supremum | content
    310.5 | does that make sense to everyone | check_in

A pipe delimiter rather than "extra spaces" is deliberate: it is unambiguous even if a
line's spacing is inconsistent, which matters when the file was typed quickly while
watching a video rather than carefully formatted.

- The first field is the start timestamp in seconds.
- The last field is the `kind` (content/rhetorical/check_in/procedural).
- The middle field is the question text (may itself contain any punctuation except ``|``).
- During phase 1 (capture only), leave the third field empty or omit it - a two-field line
  (``142.0 | what is a supremum``) is accepted with `kind` left blank in the output, to be
  filled in during phase 2 without re-running this script.
- `end` is not written by hand: this script sets it to `start + duration`, a rough default
  since exact end timing rarely matters for the phrase-level recall metric. Adjust by hand
  in the output JSONL afterward if a particular span badly needs a tighter/looser end time.

Snapping to a transcript: pass `--transcript PATH` and the middle field must quote the
transcript verbatim (case and punctuation ignored). The quote is located in the transcript's
word stream, nearest the noted start where it occurs more than once, and `start`/`end` are
set from its first and last word. The noted start then only needs to be roughly right, and
the label sits on the same clock `evaluate.py` matches predictions on. A quote that is not in
the transcript is an error, not a silent fallback to the hand-typed time.

Blind-slice windows: pass `--blind START END` (seconds, repeatable) for each 10-minute
window that should be exhaustively-labelled per the guidelines. Any entry whose start falls
inside one of those windows is tagged `"split": "blind"`; everything else defaults to
`"split": "seed"`.

Usage:
    uv run python scripts/scratch_to_labels.py notes.txt \
        -o tests/data/mit_18_100b_l1.labels.jsonl --blind 300 900 --blind 3600 4200
    uv run python scripts/scratch_to_labels.py notes.txt -o out.labels.jsonl \
        --transcript tests/data/mit_18_100b_l1.transcript.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from questxtract.models import Transcript, Word

_VALID_KINDS = frozenset({"content", "rhetorical", "check_in", "procedural"})


def parse_scratch_line(line_number: int, raw_line: str) -> dict | None:
    """Parse one scratch-file line into a label record, or ``None`` for a blank line.

    Raises:
        ValueError: If the line is non-blank but doesn't have at least two ``|``-separated
            fields, or the timestamp or kind can't be parsed. The message names the line.

    """
    line = raw_line.rstrip("\n")
    if not line.strip() or line.strip().startswith("#"):
        return None

    fields = [f.strip() for f in line.split("|")]
    if len(fields) < 2:
        raise ValueError(
            f"line {line_number}: expected 'start | text' or 'start | text | kind' "
            f"(pipe-separated), got: {line!r}"
        )

    start_raw = fields[0]
    text = fields[1]
    kind_raw = fields[2] if len(fields) >= 3 else ""

    try:
        start = float(start_raw)
    except ValueError as exc:
        raise ValueError(f"line {line_number}: '{start_raw}' is not a valid timestamp") from exc

    if not text:
        raise ValueError(f"line {line_number}: question text is empty")

    kind = kind_raw.strip().lower()
    if kind and kind not in _VALID_KINDS:
        raise ValueError(
            f"line {line_number}: kind must be one of {sorted(_VALID_KINDS)} or blank, "
            f"got {kind_raw!r}"
        )

    return {"start": start, "text": text, "kind": kind or None}


def convert(
    lines: Sequence[str], *, default_duration: float, blind_windows: Sequence[tuple[float, float]]
) -> list[dict]:
    """Convert every scratch line into a full label record, sorted by start time."""
    records = []
    for i, raw_line in enumerate(lines, start=1):
        record = parse_scratch_line(i, raw_line)
        if record is None:
            continue
        record["end"] = round(record["start"] + default_duration, 1)
        record["split"] = "blind" if _in_any_window(record["start"], blind_windows) else "seed"
        records.append(record)
    return sorted(records, key=lambda r: r["start"])


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold().replace("\u2019", "'"))


def snap(records: list[dict], words: Sequence[Word]) -> list[dict]:
    """Set each record's ``start``/``end`` from the transcript words its ``text`` quotes.

    Raises:
        ValueError: If a record's text does not occur in ``words``.

    """
    stream = [(tok, i) for i, w in enumerate(words) for tok in _tokens(w.text)]
    tokens = [tok for tok, _ in stream]
    for record in records:
        quote = _tokens(record["text"])
        hits = [
            j for j in range(len(tokens) - len(quote) + 1) if tokens[j : j + len(quote)] == quote
        ]
        if not quote or not hits:
            raise ValueError(f"not in the transcript: {record['text']!r}")
        best = min(hits, key=lambda j: abs(words[stream[j][1]].start - record["start"]))
        record["start"] = words[stream[best][1]].start
        record["end"] = words[stream[best + len(quote) - 1][1]].end
    return sorted(records, key=lambda r: r["start"])


def _in_any_window(start: float, windows: Sequence[tuple[float, float]]) -> bool:
    return any(lo <= start <= hi for lo, hi in windows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the conversion. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scratch_file", help="the plain-text notes file")
    parser.add_argument("-o", "--output", required=True, help="where to write the .labels.jsonl")
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="default span length in seconds when only a start time was noted (default: 3.0)",
    )
    parser.add_argument(
        "--blind",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        action="append",
        default=[],
        help="a blind-slice window in seconds, e.g. --blind 300 900. Repeatable.",
    )
    parser.add_argument(
        "--transcript", help="snap start/end to this transcript's words (text must quote it)"
    )
    args = parser.parse_args(argv)

    try:
        lines = Path(args.scratch_file).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        records = convert(
            lines, default_duration=args.duration, blind_windows=[tuple(w) for w in args.blind]
        )
        if args.transcript:
            records = snap(records, Transcript.from_file(args.transcript).words)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("error: no valid entries found in scratch file", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    blind_count = sum(1 for r in records if r["split"] == "blind")
    print(f"wrote {len(records)} entries ({blind_count} blind) -> {out_path}")
    print("now run: uv run python scripts/validate_labels.py " + str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

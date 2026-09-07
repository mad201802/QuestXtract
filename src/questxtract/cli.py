"""Command line interface: ``questxtract transcript.json -o questions.json``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .detect import detect_questions
from .models import questions_to_json, read_segments, write_questions


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    parser = argparse.ArgumentParser(prog="questxtract", description=__doc__)
    parser.add_argument("transcript", help="transcript JSON file with text/start/end segments")
    parser.add_argument("-o", "--output", help="write questions here instead of stdout")
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.5, help="minimum confidence (default: 0.5)"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    questions = detect_questions(read_segments(args.transcript), threshold=args.threshold)
    if args.output:
        write_questions(questions, args.output)
        print(f"{len(questions)} questions -> {args.output}", file=sys.stderr)
    else:
        print(questions_to_json(questions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line interface: ``questxtract transcript.json -o questions.json``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .audio.transcripts import read_segments
from .detection.questions import detect_questions
from .models.schemas import LectureResult, SchemaError
from .output.serialize import result_to_json, write_result


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

    try:
        segments = read_segments(args.transcript)
    except (OSError, SchemaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = LectureResult(
        lecture=Path(args.transcript).name,
        questions=tuple(detect_questions(segments, threshold=args.threshold)),
    )
    if args.output:
        write_result(result, args.output)
        print(f"{len(result.questions)} questions -> {args.output}", file=sys.stderr)
    else:
        print(result_to_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

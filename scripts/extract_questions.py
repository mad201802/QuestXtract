#!/usr/bin/env python3
"""Run the full pipeline over a recording, end to end, and write questions.json.

Standalone dev script: the CLI's `process` command doesn't accept a raw
recording yet (issue #17), so this fills that gap for real end-to-end testing
(video -> audio -> transcript -> sentences -> ... -> questions) through the
public API, mirroring `questxtract process`'s own options.

Usage: uv run python scripts/extract_questions.py <recording.mp4> [-o out.json] [-t threshold]
"""

from __future__ import annotations

import argparse
import sys

from questxtract.cli import _progress_bar
from questxtract.config import Config
from questxtract.export import result_to_json, write_questions
from questxtract.media import MediaError
from questxtract.pipeline import STAGES, process


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline over ``argv[0]`` and write or print the questions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="an .mp4 recording")
    parser.add_argument("-o", "--output", help="write questions here instead of stdout")
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.40, help="minimum confidence (default: 0.40)"
    )
    parser.add_argument(
        "--from-scratch", action="store_true", help="ignore every cache entry and recompute"
    )
    args = parser.parse_args(argv)

    config = Config().replace(min_confidence=args.threshold, from_scratch=args.from_scratch)
    progress = _progress_bar(len(STAGES))
    try:
        ctx = process(args.recording, config, progress=progress)
    except (FileNotFoundError, MediaError) as exc:
        print(file=sys.stderr)  # close the redrawing bar line before the error
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print(file=sys.stderr)
        raise

    if args.output:
        write_questions(ctx, args.output)
        print(f"{len(ctx.pairs)} questions -> {args.output}", file=sys.stderr)
    else:
        print(result_to_json(ctx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Save both the questions that cleared the confidence threshold and the dropped ones.

`questxtract process` (and extract_questions.py) only ever write the survivors;
score.py's threshold filter silently drops the rest. This runs the same pipeline
with `min_confidence` set to 0.0 so score() drops nobody, then splits its output by
the real threshold - writing one file per side, in the same schema as a normal
questions.json.

Usage: uv run python scripts/dropped_questions.py <transcript.json> [-o out.json] [-t threshold]
Writes <out.json> (kept, default questions.json) and a sibling <out>.dropped.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from questxtract.cli import _load_transcript
from questxtract.config import Config
from questxtract.export import write_questions
from questxtract.models import Context, SchemaError
from questxtract.pipeline import STAGES, run

_AFTER_TRANSCRIBE = 2  # stages 1 and 2 are skipped: the transcript is already in hand


def main(argv: list[str] | None = None) -> int:
    """Split a transcript's candidates into kept/dropped files at the given threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", help="a transcript JSON, as read by `questxtract process`")
    parser.add_argument(
        "-o", "--output", default="questions.json", help="where to write the kept questions"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.40, help="minimum confidence (default: 0.40)"
    )
    args = parser.parse_args(argv)

    try:
        transcript = _load_transcript(args.transcript)
    except (OSError, SchemaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # The real pipeline, with the emission gate wide open: score() then keeps every
    # candidate and the split below is this script's own, not a second scoring path.
    config = Config().replace(min_confidence=0.0)
    ctx = Context(source=Path(args.transcript), config=config, transcript=transcript)
    ctx.duration = transcript.duration
    ctx = run(ctx, STAGES[_AFTER_TRANSCRIBE:])

    kept = tuple(p for p in ctx.pairs if p.confidence >= args.threshold)
    dropped = tuple(p for p in ctx.pairs if p.confidence < args.threshold)

    kept_path = Path(args.output)
    dropped_path = kept_path.with_name(kept_path.stem + ".dropped.json")

    for label, pairs, path in (("kept", kept, kept_path), ("dropped", dropped, dropped_path)):
        ctx.pairs = pairs
        ctx.stats["emitted"] = len(pairs)
        write_questions(ctx, path)
        print(f"{len(pairs)} {label} -> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Save both the questions that cleared the confidence threshold and the dropped ones.

`questxtract process` (and extract_questions.py) only ever write the survivors;
score.py's threshold filter silently drops the rest. This runs the same stages up
to (but not including) score(), scores every candidate itself at threshold 0.0 so
nothing is dropped, then splits the results by the real threshold - writing one
file per side, in the same schema as a normal questions.json.

Usage: uv run python scripts/dropped_questions.py <transcript.json> [-o out.json] [-t threshold]
Writes <out.json> (kept, default questions.json) and a sibling <out>.dropped.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from questxtract.config import Config
from questxtract.export import result_to_dict
from questxtract.models import Context, SchemaError, Transcript
from questxtract.pipeline import STAGES, run
from questxtract.score import score_pairs

_BEFORE_SCORE = 8  # STAGES[2:8]: sentences..answer - everything score() needs, not score() itself


def _load_transcript(path: str) -> Transcript:
    """Read a transcript JSON file, same shape `questxtract process` reads.

    Raises:
        SchemaError: If the file is not JSON or is not a recognised transcript shape.

    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SchemaError(f"{path}: not valid JSON ({exc})") from exc
    return Transcript.from_dict(data)


def _dropped_path(kept_path: Path) -> Path:
    """Return the sibling path for the dropped-candidates file next to ``kept_path``."""
    if kept_path.suffix == ".json":
        return kept_path.with_name(kept_path.stem + ".dropped.json")
    return kept_path.with_name(kept_path.name + ".dropped.json")


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

    config = Config().replace(min_confidence=args.threshold)
    ctx = Context(source=Path(args.transcript), config=config, transcript=transcript)
    ctx.duration = transcript.duration
    ctx = run(ctx, STAGES[2:_BEFORE_SCORE])  # sentences..answer: unscored candidate pairs

    scored = score_pairs(ctx.pairs, min_confidence=0.0)  # score everyone, drop no one here
    kept = tuple(p for p in scored if p.confidence >= config.min_confidence)
    dropped = tuple(p for p in scored if p.confidence < config.min_confidence)

    kept_path = Path(args.output)
    dropped_path = _dropped_path(kept_path)

    ctx.pairs = kept
    ctx.stats["emitted"] = len(kept)
    kept_path.write_text(
        json.dumps(result_to_dict(ctx), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ctx.pairs = dropped
    ctx.stats["emitted"] = len(dropped)
    dropped_path.write_text(
        json.dumps(result_to_dict(ctx), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"{len(kept)} kept -> {kept_path}", file=sys.stderr)
    print(f"{len(dropped)} dropped -> {dropped_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

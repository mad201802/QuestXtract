"""Command line interface: ``process``, ``eval`` and ``quiz``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import Config
from .evaluate import evaluate
from .export import result_to_json, write_questions
from .models import Context, SchemaError, Transcript
from .pipeline import STAGES, run

_AFTER_TRANSCRIBE = 2  # stages 1 and 2 are skipped when a transcript is supplied


def _load_transcript(path: str) -> Transcript:
    """Read a transcript JSON file.

    Raises:
        SchemaError: If the file is not JSON or is not a recognised transcript shape.

    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SchemaError(f"{path}: not valid JSON ({exc})") from exc
    return Transcript.from_dict(data)


def _process(args: argparse.Namespace) -> int:
    """Run the pipeline over a transcript and write ``questions.json``."""
    try:
        transcript = _load_transcript(args.transcript)
    except (OSError, SchemaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    config = Config().replace(min_confidence=args.threshold, from_scratch=args.from_scratch)
    ctx = Context(source=Path(args.transcript), config=config, transcript=transcript)
    ctx.duration = transcript.duration
    # Stages 1 and 2 are skipped: the transcript is already in hand. This is the documented
    # `stages=` seam (architecture.md 11.1), not a workaround.
    ctx = run(ctx, STAGES[_AFTER_TRANSCRIBE:])

    if args.output:
        write_questions(ctx, args.output)
        print(f"{len(ctx.pairs)} questions -> {args.output}", file=sys.stderr)
    else:
        print(result_to_json(ctx))
    return 0


def _eval(args: argparse.Namespace) -> int:
    """Score the pipeline against the labelled set."""
    try:
        report = evaluate(args.paths)
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2) if args.json else report)
    return 0


def _quiz(args: argparse.Namespace) -> int:
    """Turn questions.json into quiz.json. Generative, and deliberately separate."""
    del args
    print("error: quiz: the generative stage is not wired up yet", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="questxtract", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subs = parser.add_subparsers(dest="command", required=True)

    process = subs.add_parser("process", help="extract questions from a recording or transcript")
    process.add_argument("transcript", help="transcript JSON with text/start/end segments")
    process.add_argument("-o", "--output", help="write questions here instead of stdout")
    process.add_argument(
        "-t", "--threshold", type=float, default=0.40, help="minimum confidence (default: 0.40)"
    )
    process.add_argument(
        "--from-scratch", action="store_true", help="ignore every cache entry and recompute"
    )
    process.set_defaults(handler=_process)

    evaluation = subs.add_parser("eval", help="score against the labelled set")
    evaluation.add_argument("paths", nargs="+", help="directories holding *.labels.jsonl")
    evaluation.add_argument("--json", action="store_true", help="machine-readable, for CI")
    evaluation.set_defaults(handler=_eval)

    quiz = subs.add_parser("quiz", help="generate quiz.json from questions.json")
    quiz.add_argument("questions", help="a questions.json written by `process`")
    quiz.add_argument("-o", "--output", help="write the quiz here")
    quiz.set_defaults(handler=_quiz)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    handler: object = args.handler
    assert callable(handler)  # noqa: S101 - set_defaults guarantees it
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

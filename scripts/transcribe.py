#!/usr/bin/env python3
"""Transcribe a recording and save the raw transcript as JSON.

Standalone dev script: the CLI's `process` command doesn't accept a raw
recording yet (issue #17), so this fills that gap for local testing. The
output is exactly the ``{"segments": [...]}`` shape `questxtract process`
reads, so it can be fed straight back in without re-transcribing:

    uv run python scripts/transcribe.py lecture.mp4
    uv run questxtract process lecture.transcript.json -o questions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from questxtract.asr import resolve_backend
from questxtract.cli import _progress_bar
from questxtract.config import Config
from questxtract.media import MediaError


def main(argv: list[str] | None = None) -> int:
    """Transcribe the given recording and write the raw segments to an output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="an .mp4 recording")
    parser.add_argument(
        "output", nargs="?", help="output path (default: <recording>.transcript.json)"
    )
    args = parser.parse_args(argv)

    source = Path(args.recording)
    out = Path(args.output) if args.output else source.with_suffix(".transcript.json")

    config = Config()
    report = _progress_bar(1)

    def on_progress(fraction: float) -> None:
        report("transcribe", fraction)

    try:
        backend = resolve_backend(config.asr_backend)
        # Seed the 0% line before the (possibly slow, on a cold model cache) call below,
        # matching pipeline.run()'s own per-stage 0.0 call - otherwise nothing is drawn
        # until the first real progress update, which can be a long silent wait.
        report("transcribe", 0.0)
        segments = backend.transcribe(source, config, on_progress=on_progress)
    except (FileNotFoundError, MediaError) as exc:
        print(file=sys.stderr)  # close the redrawing bar line before the error
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report("done", 1.0)

    data = {"segments": segments, "language": config.language}
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(segments)} segments -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

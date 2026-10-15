#!/usr/bin/env python3
"""Read the board of a labelled recording into <slug>.visual.json, for `eval --board`.

Runs the layout probe and the vision stage (architecture.md 6) over a recording against its
committed transcript, so the board states keep the clock the labels are timed against, and no
ASR runs. The layout answer, views and readings are cached as in a normal run
(`.questxtract/`), so a second run makes no calls.
Calls the vision model: needs the `[llm]` extra and OPENAI_API_KEY, or `base_url` in the
settings file.

Usage: uv run python scripts/board_states.py <recording.mp4> tests/data/<slug>.transcript.json
Writes tests/data/<slug>.visual.json beside the transcript: the model and prompt version, and
each board state's start, end, description, text, latex and confidence. Keyframes stay in the
cache; the file holds no images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from questxtract.config import load_config
from questxtract.media import source_hash
from questxtract.models import Context, Transcript
from questxtract.pipeline import run
from questxtract.segments import segments
from questxtract.sentences import sentences
from questxtract.vision import VISION_PROMPT_VERSION, has_video, layout, vision


def main(argv: list[str] | None = None) -> int:
    """Write the board states of one recording beside its transcript."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="an .mp4 with a video track")
    parser.add_argument("transcript", type=Path, help="its <slug>.transcript.json")
    parser.add_argument("-c", "--config", type=Path, default=None, help="settings file")
    args = parser.parse_args(argv)

    if not has_video(args.recording):
        print(f"error: {args.recording} has no video track", file=sys.stderr)
        return 1
    config = load_config(args.config).replace(vision_enabled=True)
    transcript = Transcript.from_file(args.transcript)
    ctx = Context(source=args.recording, config=config, transcript=transcript)
    ctx.source_hash = source_hash(args.recording)
    ctx.duration = transcript.duration
    ctx = run(
        ctx,
        (sentences, segments, layout, vision),
        progress=lambda stage, f: print(f"\r{stage} {f:4.0%}", end="", file=sys.stderr),
    )
    print(file=sys.stderr)

    slug = args.transcript.name.removesuffix(".transcript.json")
    output = args.transcript.with_name(f"{slug}.visual.json")
    states = [
        {
            "start": s.start,
            "end": s.end,
            "description": s.description,
            "text": s.text,
            "latex": s.latex,
            "confidence": s.confidence,
        }
        for s in ctx.visual
    ]
    data = {
        "vision_model": config.vision_model,
        "vision_prompt_version": VISION_PROMPT_VERSION,
        "board_states": states,
    }
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = ctx.stats.get("vision_failed", 0)
    print(f"{len(states)} board states, {failed} unread -> {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

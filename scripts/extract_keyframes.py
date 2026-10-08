#!/usr/bin/env python3
"""Write the keyframes of a lecture recording as PNG files, to pick test frames from.

Runs the sampling and stable-view detection of the vision stage (architecture.md 6.3 and 6.4,
`questxtract.vision`) over a recording and saves each view's keyframe, so a few frames per
lecture type can be hand-picked for the model comparison (#52). It calls no model and no
network, and writes nothing into the package cache.

Usage: uv run python scripts/extract_keyframes.py <recording> [-o DIR] [--fps 1.0]
Writes DIR/<recording stem>_<seconds>.png per view (default DIR: keyframes/) and
DIR/<recording stem>.views.json, one entry per view with its start, end and keyframe time.
Defaults for the view detection are those of `Config`; `--max-views` is where to raise the cap
(default 80) to see every view of a long recording.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from questxtract.config import Config
from questxtract.vision import Crop, _sample_frames, _views


def _crop(text: str) -> Crop:
    """Parse ``x0,y0,x1,y1`` as fractions of the frame."""
    try:
        x0, y0, x1, y1 = (float(part) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected four numbers: x0,y0,x1,y1") from exc
    return x0, y0, x1, y1


def main(argv: list[str] | None = None) -> int:
    """Extract one keyframe per stable view of a recording."""
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="an .mp4 with a video track")
    parser.add_argument("-o", "--output", type=Path, default=Path("keyframes"), help="folder")
    parser.add_argument("--fps", type=float, default=defaults.vision_fps, help="samples/second")
    parser.add_argument(
        "--change",
        type=float,
        default=defaults.vision_change_fraction,
        help="share of pixels that must change to start a view",
    )
    parser.add_argument(
        "--hold",
        type=int,
        default=defaults.vision_change_hold,
        help="samples a new view must hold",
    )
    parser.add_argument(
        "--max-views", type=int, default=defaults.vision_max_states, help="merge down to this many"
    )
    parser.add_argument("--crop", type=_crop, default=None, help="x0,y0,x1,y1 as fractions")
    args = parser.parse_args(argv)

    if not args.recording.is_file():
        print(f"error: {args.recording} is not a file", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    samples = _sample_frames(
        args.recording,
        args.fps,
        crop=args.crop,
        progress=lambda _stage, fraction: print(f"\r{fraction:4.0%}", end="", file=sys.stderr),
    )
    try:
        views = _views(samples, change=args.change, hold=args.hold, max_views=args.max_views)
    except ValueError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    finally:
        samples.close()
    print(file=sys.stderr)

    index = []
    for view in views:
        name = f"{args.recording.stem}_{view.keyframe_time:07.1f}.png"
        (args.output / name).write_bytes(view.keyframe)
        index.append(
            {
                "file": name,
                "start": round(view.start, 1),
                "end": round(view.end, 1),
                "keyframe_time": round(view.keyframe_time, 1),
            }
        )
    index_path = args.output / f"{args.recording.stem}.views.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"{len(views)} views -> {args.output}/ (index: {index_path})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

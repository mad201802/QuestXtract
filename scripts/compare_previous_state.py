#!/usr/bin/env python3
"""Read the labelled keyframes with and without the previous board state, for #137.

Each recording's labelled frames (`tests/data/vision/*.frames.jsonl`, #58) are taken in time
order. Every frame after the first is read twice through `vision.read_keyframe`, as a run sends
it: prompt v0 alone, and prompt v2 with the previous labelled frame's v0 reading and the box
around the new ink. The previous state is given even where the vision stage would leave it out
after an erase, so that copying from an erased board shows; the stage's own decision (ink kept,
architecture.md 6.5) is a column, and the table is also given for the pairs it keeps.

Per arm: failed readings, mean character error rate of `text` (frames with text), LaTeX filled
(frames with a labelled formula whose answer has `latex`), formula recall, and copied lines: a
line of the answer that is a line of the previous reading but on no line of this frame's label
(within a character error rate of 1/3). v0 never sees the previous reading, so its copied lines
are the baseline: the same words on both boards.

Calls the vision model: needs the `[llm]` extra and a key, as `eval --score-vision`. Answers are
cached per request, so a second run makes no calls.

Usage: uv run python scripts/compare_previous_state.py tests/data/vision/ [--vision-model NAME]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from questxtract.config import Config, load_config
from questxtract.evaluate import character_error_rate, formula_found
from questxtract.vision import (
    PreviousState,
    _decoded,
    _for_comparison,
    _ink,
    _ink_kept,
    _new_ink,
    _previous,
    read_keyframe,
)

_COPY_MAX_ERROR = 1 / 3  # a line closer than this to a label line is on the board
_COPY_MIN_CHARS = 8  # shorter lines ("a", "x = 0") are on many boards by chance
_ARMS = ("v0", "v2")


def _plain(text: str) -> str:
    return " ".join(text.casefold().split())


def _seconds(label: dict[str, Any]) -> float:
    """Return when a frame was taken, from its name: ``<slug>_<seconds>.png``."""
    return float(str(label["frame"]).rsplit("_", 1)[1].removesuffix(".png"))


def copied_lines(answer: str, previous: str, label: str) -> list[str]:
    """Return the lines of ``answer`` copied from ``previous`` that are not on the board."""
    before = {_plain(line) for line in previous.splitlines()}
    board = [_plain(line) for line in label.splitlines() if _plain(line)]
    copies = []
    for line in answer.splitlines():
        plain = _plain(line)
        if len(plain) < _COPY_MIN_CHARS or plain not in before or plain in _plain(label):
            continue
        if all(character_error_rate(plain, b) > _COPY_MAX_ERROR for b in board):
            copies.append(line.strip())
    return copies


def compare(directory: Path, config: Config) -> list[dict[str, Any]]:
    """Read every labelled frame that has an earlier one in its recording, both ways.

    Returns one row per frame: its label, the stage's decision (``kept``: the share of the
    earlier frame's ink still there; ``sent``: whether the stage would send the previous
    state), and per arm the answer (``None`` when the reading failed) and its copied lines.
    """
    rows: list[dict[str, Any]] = []
    for labels_path in sorted(directory.glob("*.frames.jsonl")):
        lines = labels_path.read_text(encoding="utf-8").splitlines()
        labels = sorted((json.loads(line) for line in lines if line.strip()), key=_seconds)
        before: tuple[str, Any, dict[str, Any] | None] | None = None  # name, ink, reading
        for label in labels:
            png = labels_path.with_name(label["frame"]).read_bytes()
            picture = _decoded(png)
            now = None if picture is None else _ink(_for_comparison(picture))
            alone = read_keyframe(config, png, kind=label["kind"])
            if before is not None and before[2] is not None and now is not None:
                previous_frame, was, reading = before
                same_size = was.shape == now.shape
                previous = PreviousState(
                    description=reading["description"],
                    text=reading["text"],
                    latex=reading["latex"],
                    new_ink=_new_ink(now & ~was) if same_size else None,
                )
                given = read_keyframe(config, png, kind=label["kind"], previous=previous)
                answers = {"v0": alone, "v2": given}
                rows.append(
                    {
                        "frame": label["frame"],
                        "kind": label["kind"],
                        "label": label,
                        "previous_frame": previous_frame,
                        "kept": round(_ink_kept(was, now), 2),
                        "sent": _previous((was, reading), now) is not None,
                        "answers": answers,
                        "copied": {
                            arm: copied_lines(a["text"], reading["text"], label["text"])
                            if a
                            else []
                            for arm, a in answers.items()
                        },
                    }
                )
            before = None if now is None else (label["frame"], now, alone)
    return rows


def metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Failed readings, mean CER, LaTeX filled, formula recall and copied lines for one arm."""
    errors, filled, formulas, found, copies, copying = [], 0, 0, 0, 0, 0
    with_formulas = 0
    for row in rows:
        answer, label = row["answers"][arm], row["label"]
        text = answer["text"] if answer else ""
        latex = answer["latex"] if answer else None
        if label["text"].strip():
            errors.append(character_error_rate(text, label["text"]))
        if label["latex"]:
            with_formulas += 1
            filled += bool(latex)
        formulas += len(label["latex"])
        found += sum(formula_found(f, latex) for f in label["latex"])
        copies += len(row["copied"][arm])
        copying += bool(row["copied"][arm])
    return {
        "frames": len(rows),
        "failed": sum(row["answers"][arm] is None for row in rows),
        "cer": round(statistics.mean(errors), 3) if errors else None,
        "latex_filled": f"{filled}/{with_formulas}",
        "formula_recall": round(found / formulas, 3) if formulas else None,
        "copied_lines": copies,
        "frames_copying": copying,
    }


def format_report(rows: list[dict[str, Any]], model: str) -> str:
    """Render the comparison as Markdown: the tables, each pair's decision, the copied lines."""
    out = [f"Model `{model}`. Each frame read alone (v0) and after the previous labelled one (v2)."]
    groups = [
        ("every pair, previous state forced", rows),
        ("pairs the stage would send (ink kept)", [r for r in rows if r["sent"]]),
    ]
    for title, group in groups:
        out += [
            "",
            f"**{title}**",
            "",
            "| arm | frames | failed | CER | LaTeX filled | formula recall | copied lines "
            "(frames) |",
            "|---|---|---|---|---|---|---|",
        ]
        for arm in _ARMS:
            m = metrics(group, arm)
            out.append(
                f"| {arm} | {m['frames']} | {m['failed']} | {m['cer']} | {m['latex_filled']} | "
                f"{m['formula_recall']} | {m['copied_lines']} ({m['frames_copying']}) |"
            )
    out += ["", "| frame | after | ink kept | stage sends | copied in v0 | copied in v2 |"]
    out.append("|---|---|---|---|---|---|")
    for row in rows:
        v0, v2 = (len(row["copied"][arm]) for arm in _ARMS)
        sends = "yes" if row["sent"] else "no"
        out.append(
            f"| {row['frame']} | {row['previous_frame']} | {row['kept']} | {sends} | {v0} | {v2} |"
        )
    copies = [(row["frame"], line) for row in rows for line in row["copied"]["v2"]]
    if copies:
        out += ["", "Lines v2 copied from the previous reading that are not on the board:", ""]
        out += [f"- `{frame}`: {line}" for frame, line in copies]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Print the comparison for the labelled frames in a directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="holds the *.frames.jsonl and their PNGs")
    parser.add_argument("-c", "--config", type=Path, default=None, help="settings file")
    parser.add_argument("--vision-model", default=None, help="instead of vision_model")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.vision_model:
        config = config.replace(vision_model=args.vision_model)
    rows = compare(args.directory, config)
    if not rows:
        print(f"no recording in {args.directory} has two labelled frames", file=sys.stderr)
        return 1
    print(format_report(rows, config.vision_model))
    return 0


if __name__ == "__main__":
    sys.exit(main())

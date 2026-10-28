#!/usr/bin/env python3
"""Read labelled keyframes with and without the previous board state, for #137 and #230.

Each recording's labelled frames (`*.frames.jsonl`) are taken in time order and read three ways,
each arm chained on its own readings as the vision stage chains a run:

- v0: prompt v0 alone;
- v2: prompt v2 with the arm's previous reading and the box around the new ink, on every pair,
  also where the stage would leave the previous state out after an erase, so that copying from
  an erased board shows;
- stage: as the vision stage sends it (architecture.md 6.5): the previous state only when at
  least `vision._INK_KEPT` of the earlier frame's ink is still ink.

A failed reading gives the next frame no previous state, as in the stage. The first frame of a
recording is read alone in every arm, one call.

Per arm: failed readings, mean character error rate of `text` (frames with text), LaTeX filled
(frames with a labelled formula whose answer has `latex`), formula recall, copied lines and
erased content kept. A copied line is a line of the answer that is a line of the arm's previous
reading but within a character error rate of 1/3 of no line of this frame's label, nor of two
consecutive ones joined (line breaks are layout): gone from the board, or an old form of a line
that has grown since. v0 never sees the previous reading,
so its copied lines are the baseline: the same words on both boards. Erased content is what a
label's optional `erased` list names: text of the previous frame that is no longer on this one
(`tests/data/vision/sequence/`); it counts as kept when the answer's text or LaTeX holds it
within one edit per four characters, ignoring case, spacing and LaTeX markup (a `geq` command
reads as `≥`).

Calls the vision model: needs the `[llm]` extra and a key, as `eval --score-vision`. Answers are
cached per request, so a second run makes no calls.

Usage: uv run python scripts/compare_previous_state.py tests/data/vision/sequence/
       [--vision-model NAME]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

from questxtract.config import Config, load_config
from questxtract.evaluate import _distance, character_error_rate, formula_found
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
_ERASED_MAX_ERROR = 1 / 4  # edits per character at which erased text still counts as kept
_ARMS = ("v0", "v2", "stage")
_NOTATION = re.compile(r"\\(?:text|mathrm|left|right)\b|[\s{}_^\\]")
_SYMBOLS = {r"\geq": "≥", r"\ge": "≥", r"\leq": "≤", r"\le": "≤", r"\cdot": "*", "∗": "*"}


def _plain(text: str) -> str:
    return " ".join(text.casefold().split())


def _seconds(label: dict[str, Any]) -> float:
    """Return when a frame was taken, from its name: ``<slug>_<seconds>.png``."""
    return float(str(label["frame"]).rsplit("_", 1)[1].removesuffix(".png"))


def copied_lines(answer: str, previous: str, label: str) -> list[str]:
    """Return the lines of ``answer`` copied from ``previous`` whose version is not on the board.

    A line of ``previous`` that is on no line of ``label``, nor on two consecutive ones joined:
    gone from the board, or an old, shorter form of a line that has grown since.
    """
    before = {_plain(line) for line in previous.splitlines()}
    lines = [_plain(line) for line in label.splitlines() if _plain(line)]
    board = lines + [f"{a} {b}" for a, b in pairwise(lines)]
    copies = []
    for line in answer.splitlines():
        plain = _plain(line)
        if len(plain) < _COPY_MIN_CHARS or plain not in before:
            continue
        if all(character_error_rate(plain, b) > _COPY_MAX_ERROR for b in board):
            copies.append(line.strip())
    return copies


def _bare(text: str) -> str:
    for command, symbol in _SYMBOLS.items():
        text = re.sub(re.escape(command) + r"(?![a-z])", symbol, text)
    return _NOTATION.sub("", text).casefold()


def erased_kept(answer: dict[str, Any], erased: list[str]) -> list[str]:
    """Return the erased text in ``erased`` that ``answer``'s text or LaTeX still holds."""
    haystack = _bare(answer["text"] + "\n" + (answer["latex"] or ""))
    kept = []
    for item in erased:
        needle = _bare(item)
        if needle and _distance(needle, haystack, free_ends=True) <= _ERASED_MAX_ERROR * len(
            needle
        ):
            kept.append(item)
    return kept


def compare(directory: Path, config: Config) -> list[dict[str, Any]]:
    """Read every labelled frame that has an earlier one in its recording, three ways.

    Returns one row per frame: its label, the stage's decision (``kept``: the share of the
    earlier frame's ink still there; ``sent``: whether the stage sends the previous state), and
    per arm the answer (``None`` when the reading failed), its copied lines and the erased text
    it kept.
    """
    rows: list[dict[str, Any]] = []
    for labels_path in sorted(directory.glob("*.frames.jsonl")):
        lines = labels_path.read_text(encoding="utf-8").splitlines()
        labels = sorted((json.loads(line) for line in lines if line.strip()), key=_seconds)
        was: Any = None  # the ink of the frame before
        readings: dict[str, dict[str, Any] | None] = dict.fromkeys(_ARMS)
        previous_frame = None
        for label in labels:
            png = labels_path.with_name(label["frame"]).read_bytes()
            picture = _decoded(png)
            now = None if picture is None else _ink(_for_comparison(picture))
            alone = read_keyframe(config, png, kind=label["kind"])
            if was is None or now is None:
                answers = dict.fromkeys(_ARMS, alone)
            else:
                new_ink = _new_ink(now & ~was) if was.shape == now.shape else None
                answers = {"v0": alone}
                for arm in ("v2", "stage"):
                    reading = readings[arm]
                    if reading is None:
                        previous = None
                    elif arm == "v2":
                        previous = PreviousState(
                            description=reading["description"],
                            text=reading["text"],
                            latex=reading["latex"],
                            new_ink=new_ink,
                        )
                    else:
                        previous = _previous((was, reading), now)
                    answers[arm] = (
                        alone
                        if previous is None
                        else read_keyframe(config, png, kind=label["kind"], previous=previous)
                    )
                rows.append(
                    {
                        "frame": label["frame"],
                        "kind": label["kind"],
                        "label": label,
                        "previous_frame": previous_frame,
                        "kept": round(_ink_kept(was, now), 2),
                        "sent": readings["stage"] is not None
                        and _previous((was, readings["stage"]), now) is not None,
                        "answers": answers,
                        "copied": {
                            arm: copied_lines(a["text"], b["text"], label["text"])
                            if (a := answers[arm]) and (b := readings[arm])
                            else []
                            for arm in _ARMS
                        },
                        "erased": {
                            arm: erased_kept(a, label.get("erased", [])) if a else []
                            for arm, a in answers.items()
                        },
                    }
                )
            readings = dict(answers)
            was, previous_frame = now, label["frame"]
    return rows


def metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Failed readings, mean CER, LaTeX filled, formula recall, copies and erased text kept."""
    errors, filled, formulas, found, copies, copying = [], 0, 0, 0, 0, 0
    with_formulas = erased = kept = 0
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
        erased += len(label.get("erased", []))
        kept += len(row["erased"][arm])
    return {
        "frames": len(rows),
        "failed": sum(row["answers"][arm] is None for row in rows),
        "cer": round(statistics.mean(errors), 3) if errors else None,
        "latex_filled": f"{filled}/{with_formulas}",
        "formula_recall": round(found / formulas, 3) if formulas else None,
        "copied_lines": copies,
        "frames_copying": copying,
        "erased_kept": f"{kept}/{erased}",
    }


def format_report(rows: list[dict[str, Any]], model: str) -> str:
    """Render the comparison as Markdown: the tables, each pair's decision, copies and erasures."""
    out = [
        f"Model `{model}`. Each frame read alone (v0), after the arm's previous reading on every "
        "pair (v2), and as the stage sends it (stage)."
    ]
    groups = [
        ("every pair", rows),
        ("pairs the stage sends (ink kept)", [r for r in rows if r["sent"]]),
    ]
    for title, group in groups:
        out += [
            "",
            f"**{title}**",
            "",
            "| arm | frames | failed | CER | LaTeX filled | formula recall | copied lines "
            "(frames) | erased kept |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for arm in _ARMS:
            m = metrics(group, arm)
            out.append(
                f"| {arm} | {m['frames']} | {m['failed']} | {m['cer']} | {m['latex_filled']} | "
                f"{m['formula_recall']} | {m['copied_lines']} ({m['frames_copying']}) | "
                f"{m['erased_kept']} |"
            )
    out += [
        "",
        "| frame | after | ink kept | stage sends | erased | copied v0 / v2 / stage "
        "| erased kept v0 / v2 / stage |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        sends = "yes" if row["sent"] else "no"
        erased = ", ".join(f"`{x}`" for x in row["label"].get("erased", [])) or "-"
        copied = " / ".join(str(len(row["copied"][arm])) for arm in _ARMS)
        kept = " / ".join(str(len(row["erased"][arm])) for arm in _ARMS)
        out.append(
            f"| {row['frame']} | {row['previous_frame']} | {row['kept']} | {sends} | {erased} | "
            f"{copied} | {kept} |"
        )
    for arm in ("v2", "stage"):
        copies = [(row["frame"], line) for row in rows for line in row["copied"][arm]]
        if copies:
            out += ["", f"Lines {arm} copied from its previous reading that are not on the board:"]
            out += [""] + [f"- `{frame}`: {line}" for frame, line in copies]
        held = [(row["frame"], item) for row in rows for item in row["erased"][arm]]
        if held:
            out += ["", f"Erased text {arm} still reads:", ""]
            out += [f"- `{frame}`: {item}" for frame, item in held]
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

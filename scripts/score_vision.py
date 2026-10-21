#!/usr/bin/env python3
"""Score runs of scripts/compare_vision.py against the reference reading of the committed frames.

Spike tooling for #143 and #144, not shipped. The score is word-sequence similarity (0 to 1):
the words a reader returned, in order, against the words read off the frame by hand
(``tests/data/vision/results/claude-hand-graded``). LaTeX commands, braces, ``$`` and punctuation
are dropped first, so a formula and its plain-text spelling compare by content. A vision model's
``text`` and ``latex`` count together; an OCR run (``--ocr``) is its plain text. A failed call
scores 0, and frames whose reference has no text (a blank board) are skipped.

Usage: uv run python scripts/score_vision.py spike-results/glm-ocr spike-results/qwen3-vl-2b
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_REFERENCE = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "data"
    / "vision"
    / "results"
    / "claude-hand-graded"
    / "results.jsonl"
)
_GROUPS = {"mit_": "chalk (MIT)", "ttk": "tablet (TTK4225)"}


def words(text: str) -> list[str]:
    """Return the lower-case words and numbers, without LaTeX commands or syntax."""
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[{}$\\^_&|]", " ", text)
    return re.findall(r"[a-z0-9]+", text.lower())


def said(row: dict[str, Any]) -> str:
    """Return what one row read: an OCR run's text, or a vision model's text plus LaTeX."""
    if "answer" not in row:
        return str(row.get("text") or "")
    answer = row["answer"]
    if not isinstance(answer, dict):
        return ""
    return " ".join(str(answer[key]) for key in ("text", "latex") if answer.get(key))


def read(path: Path) -> list[dict[str, Any]]:
    """Return every row of a results.jsonl."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main(argv: list[str] | None = None) -> int:
    """Print one table: the mean score per lecture type and run, with failures and time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="folders holding a results.jsonl")
    args = parser.parse_args(argv)

    reference = {row["frame"]: words(said(row)) for row in read(_REFERENCE)}
    print(f"| Run | {' | '.join(_GROUPS.values())} | All | Failed | Median s/frame |")
    print(f"|---|{'---|' * len(_GROUPS)}---|---|---|")
    for run in args.runs:
        rows = read(run / "results.jsonl")
        scores: dict[str, list[float]] = {name: [] for name in _GROUPS.values()}
        for row in rows:
            ref = reference.get(row["frame"])
            frame = row["frame"]
            group = next((g for prefix, g in _GROUPS.items() if frame.startswith(prefix)), None)
            if not ref or group is None:
                continue
            found = words(said(row)) if row.get("ok") else []
            scores[group].append(SequenceMatcher(None, ref, found).ratio() if found else 0.0)
        everything = [s for group in scores.values() for s in group]
        seconds = [row["seconds"] for row in rows if row.get("seconds")]
        cells = [f"{statistics.mean(v):.2f} ({len(v)})" if v else "-" for v in scores.values()]
        total = f"{statistics.mean(everything):.2f}" if everything else "-"
        failed = sum(1 for row in rows if not row.get("ok"))
        median = f"{statistics.median(seconds):.1f}" if seconds else "-"
        print(f"| {run.name} | {' | '.join(cells)} | **{total}** | {failed} | {median} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The evaluation harness: measure the pipeline against the labelled set.

NFR2 gives a number to hit (recall >= 50% on the team's labelled set), so the metric runs
in CI. Sections 6 through 9 of pipeline.md are blind without it.

A prediction matches a label when their time ranges overlap and the overlap covers at least
half the shorter range. Time-based, not text-based, because ASR text will differ from what a
human typed and the ASR is not what is being evaluated here.

The labelled *dataset* lives under ``tests/data/<name>.labels.jsonl``; this module is the
code that reads it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

_MIN_OVERLAP = 0.5


def overlaps(
    prediction: tuple[float, float], label: tuple[float, float], *, minimum: float = _MIN_OVERLAP
) -> bool:
    """Whether a predicted span matches a labelled one.

    They match when the overlap covers at least ``minimum`` of the shorter of the two.
    """
    start = max(prediction[0], label[0])
    end = min(prediction[1], label[1])
    if end <= start:
        return False
    shorter = min(prediction[1] - prediction[0], label[1] - label[0])
    return shorter > 0 and (end - start) / shorter >= minimum


def evaluate(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Score the pipeline against every labelled recording under ``paths``.

    Raises:
        NotImplementedError: The harness lands with issue #12, once the labelled set (#11)
            exists. Building it before then would be optimising blind.

    """
    # ponytail: recall / precision / f1 / false_positive_breakdown / question_mark_rate /
    # blind_slice_recall plus the min_confidence sweep, pipeline.md 4.2.
    raise NotImplementedError("eval: the harness lands with issue #12")

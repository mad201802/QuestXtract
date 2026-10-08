# Vision spike material (preliminary)

Issue #52. Work in progress: the frames, the comparison and the recommended model are not
final, and this file changes with them. Findings go on the issue and a summary in
[`docs/architecture.md`](../../../docs/architecture.md) 6.1.

## Contents

| File | What it is |
|---|---|
| `prompt_v0.txt` | Captioning prompt v0. `{kind}` is the frame kind (slides, blackboard, whiteboard, tablet, unknown); `{speech}` is the transcript overlapping the frame, as context only. |
| `prompt_v0.schema.json` | The JSON the model must return: `description`, `text`, `latex` (or `null`), `confidence`, matching `VisualContext` in `models.py`. |
| `mit_18_100b_l1_<seconds>.png` | Test frames from the MIT lecture, named by the second of the recording they were taken at (table below). Tablet and diagram frames are not committed yet. |

| Frame | What it tests |
|---|---|
| `mit_18_100b_l1_00072.0.png` | A nearly blank board being wiped: the empty-board answer |
| `mit_18_100b_l1_00746.0.png` | Handwriting and two hand-drawn graphs, lecturer partly in front |
| `mit_18_100b_l1_01507.0.png` | Wide shot of six boards at once, small writing |
| `mit_18_100b_l1_02898.0.png` | Formulas over ghosted chalk from erased boards, camera zoomed on one board |
| `mit_18_100b_l1_03415.0.png` | Inequalities and logic arrows, dust and smudges, lecturer pointing |

## Frames

About 15 keyframes across lecture types: slides, chalkboard, tablet handwriting and at least one
diagram-heavy frame. Whiteboard is not covered.

They are keyframes the vision stage's own sampling would choose, written by
`scripts/extract_keyframes.py` (the stable-view detection of architecture.md 6.4), then
hand-picked. Re-run it on a recording to see every candidate:

```bash
uv run python scripts/extract_keyframes.py lecture.mp4 -o keyframes/ --max-views 400
```

## Licences

Frames are derived from the recording and carry its licence, not the repository's MIT licence
(same rule as `tests/data/README.md`).

| Recording | Frames committed? | Licence |
|---|---|---|
| MIT 18.100B Real Analysis, L1 | yes, 5 frames | CC BY-NC-SA 4.0, attribution: MIT OpenCourseWare, 18.100B Real Analysis |
| TTK4225 2026-09-14 and 2026-09-15 | yes, once picked | the lecturer's own recordings, committed with the lecturer's (Damiano's) permission |
| Brian Douglas video (diagrams) | no | standard YouTube licence: used locally for measuring, never committed |

Only commit a frame of a recording whose licence allows redistributing derivatives.

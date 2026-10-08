# Vision spike material (preliminary)

Issue #52. Work in progress: the frames, the comparison and the recommended model are not
final, and this file changes with them. Findings go on the issue and a summary in
[`docs/architecture.md`](../../../docs/architecture.md) 6.1.

## Contents

| File | What it is |
|---|---|
| `prompt_v0.txt` | Captioning prompt v0. `{kind}` is the frame kind (slides, blackboard, whiteboard, tablet, unknown); `{speech}` is the transcript overlapping the frame, as context only. |
| `prompt_v0.schema.json` | The JSON the model must return: `description`, `text`, `latex` (or `null`), `confidence`, matching `VisualContext` in `models.py`. |
| `mit_18_100b_l1_<seconds>.png` | Test frames from the MIT lecture, named by the second of the recording they were taken at (table below). `ttk4225_2026_09_14_<seconds>.png` are the same for the TTK4225 lecture (640x360, tablet notes over slides). Diagram frames from a third recording are not committed. |

| Frame | What it tests |
|---|---|
| `mit_18_100b_l1_00072.0.png` | A nearly blank board being wiped: the empty-board answer |
| `mit_18_100b_l1_00746.0.png` | Handwriting and two hand-drawn graphs, lecturer partly in front |
| `mit_18_100b_l1_01507.0.png` | Wide shot of six boards at once, small writing |
| `mit_18_100b_l1_02898.0.png` | Formulas over ghosted chalk from erased boards, camera zoomed on one board |
| `mit_18_100b_l1_03415.0.png` | Inequalities and logic arrows, dust and smudges, lecturer pointing |
| `ttk4225_2026_09_14_00257.0.png` | Printed question slide with a hand-drawn sinusoid |
| `ttk4225_2026_09_14_00423.0.png` | Printed question with an integral and handwritten boxes and a sketch |
| `ttk4225_2026_09_14_00758.0.png` | Printed theorem and formula with handwriting below |
| `ttk4225_2026_09_14_01265.0.png` | Handwritten `h(t) = 1/(1+t)`, a graph and a diagram, app toolbar visible |
| `ttk4225_2026_09_14_01731.0.png` | Messy handwriting and sketches, app toolbar visible |

## Frames

About 15 keyframes across lecture types: slides, chalkboard, tablet handwriting and at least one
diagram-heavy frame. Whiteboard is not covered.

They are keyframes the vision stage's own sampling would choose, written by
`scripts/extract_keyframes.py` (the stable-view detection of architecture.md 6.4), then
hand-picked. Re-run it on a recording to see every candidate:

```bash
uv run python scripts/extract_keyframes.py lecture.mp4 -o keyframes/ --max-views 400
```

## Results

Raw answers of the model comparison, one line per call: `results/<run>/results.jsonl`, written by
`scripts/compare_vision.py` (frame, model, seconds, token counts, the answer or the error).
`results/tesseract/` holds the OCR text. The comparison and its caveats are on #52.

| Run | Models | Frames |
|---|---|---|
| `auto`, `low` | OpenAI, image detail auto and low | MIT |
| `openai-ttk-360p` | OpenAI | TTK |
| `idun-mit`, `idun-ttk-360p`, `idun-ttk-720p` | Qwen3.8-27B, Mistral Medium 3.5, Kimi K2.6 instant, on Idun's hosted service | MIT, TTK at 640x360, TTK at 1280x720 (not committed) |
| `idun-mit-speech`, `idun-ttk-360p-speech`, `idun-ttk-720p-speech` | the same, with the words spoken within 20 s of each frame | as above |
| `idun-large-models` | Kimi K2.6, DeepSeek-V4-Flash-Vision-Exp, GLM-5.3-Flash | MIT and TTK at 360p |
| `idun-mistral-repeat` | Mistral Medium 3.5, three repeats | two TTK frames at 720p |
| `qwen3-vl-2b-instruct-local` | Qwen3-VL 2B through Ollama | MIT |

## Licences

Frames are derived from the recording and carry its licence, not the repository's MIT licence
(same rule as `tests/data/README.md`).

| Recording | Frames committed? | Licence |
|---|---|---|
| MIT 18.100B Real Analysis, L1 | yes, 5 frames | CC BY-NC-SA 4.0, attribution: MIT OpenCourseWare, 18.100B Real Analysis |
| TTK4225 2026-09-14 | yes, 5 frames | the lecturer's own recordings, committed with the lecturer's (Damiano's) permission |
| Brian Douglas video (diagrams) | no | standard YouTube licence: used locally for measuring, never committed |

Only commit a frame of a recording whose licence allows redistributing derivatives.

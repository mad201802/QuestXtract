# Labelled keyframes for the vision model

Ground truth for `questxtract eval --score-vision` (#58): 30 keyframes from six lectures, each
labelled with the board text, its formulas as LaTeX and a line on any diagram. Also the captioning
prompts and the raw answers of the model comparison (#52).

```bash
uv run questxtract eval --score-vision tests/data/vision/        # vision_model on NTNU's service, needs IDUN_API_KEY
uv run questxtract eval --score-vision tests/data/vision/ --llm-url URL --vision-model NAME
```

Each frame goes through `vision.read_keyframe` with prompt v0, the label's `kind` and no speech
(architecture.md 4.4). Frames are the full picture, not cropped. Answers are cached. CI does not
score (it needs a real model) but checks the labels (`tests/test_evaluate.py`): every line is well
formed, its PNG exists, and every lecture type has at least 5 frames.

## Layout

```text
tests/data/vision/
  <recording-name>.frames.jsonl      one file per recording, one line per labelled frame
  <recording-name>_<seconds>.png     the frames, named by the second they were taken at
  <recording-name>.links.jsonl       questions labelled with the board state on screen
  prompt_v0.txt, prompt_v1.txt       the captioning prompts of #52
  prompt_v2.txt                      v0 with the previous board state (#137)
  prompt_v0.schema.json              the JSON the model must return
  sequence/                          consecutive views with partial erases (#230)
  results/<run>/results.jsonl        raw answers of the #52 comparison
```

Slugs follow `tests/data/README.md`.

## Schema

```json
{"frame": "mit_18_404j_l1_02009.0.png", "kind": "slides", "text": "Regular Languages – Examples\n...", "latex": ["L(M_1) = \\{w \\mid w \\text{ contains substring } 11\\} = A"], "diagram": "The state diagram of automaton M1: ..."}
```

| Field | Type | Notes |
|---|---|---|
| `frame` | string | PNG file name beside the labels |
| `kind` | string | `slides`, `blackboard`, `whiteboard` or `handwriting` (tablet, or paper under a document camera, also over slides): what the layout probe (architecture.md 6.2) should answer |
| `text` | string | every piece of text readable at full resolution, in reading order, one line per board line; `""` for a blank board |
| `latex` | list of strings | each formula on the board, once, as LaTeX; `[]` when there is none |
| `diagram` | string or null | one line on any diagram, graph or figure; `null` when there is none |

Conventions:

- `text` keeps the board's spelling ("independet" stays). Formulas are also in `text`, typed
  plainly (`x^2`, `∫`, `≤`, `lim_{x→-2}`).
- Words cut off at the frame edge are written as far as visible ("ukushima"). Unreadable writing
  and ghosted chalk from an erased board are left out.
- Text inside a diagram (state names, axis labels) goes in `diagram`.
- Left out: the lecturer, the room and its signs, video overlays (logo, name tag, webcam column)
  and the drawing app's toolbar and status bar. Text printed on a slide, footer included, counts.
- No frame shows a student's face.

The labels were drafted by an LLM (Claude) reading each frame at full resolution and are
provisional until a human checks them.

## Scoring

Per frame, `evaluate.py` computes:

- CER: edits (insert, delete, substitute) per label character of the model's `text`, after
  casefolding and collapsing whitespace, capped at 1.
- Blank boards (label `text` is `""`) have no CER; they count as read when the answer's text is
  empty or only `[unreadable]`, in a column of their own.
- Formula recall: both formulas are normalised (`$`, `\left`, `\right`, `\displaystyle` and
  spacing commands removed, `\dfrac`/`\tfrac` read as `\frac`) and split into tokens (a command, a
  number, or one other character). A labelled formula is found when some run of whole tokens in
  the model's `latex` is within one edit per four of its tokens, so a formula of three tokens or
  fewer must match exactly, and `x` is not found inside `\max`.

A frame the model could not read counts as an empty answer and is listed as failed. The report
gives mean CER, blank boards read and pooled formula recall per lecture type and in total, then
every frame, worst first, with its missed formulas. `diagram` is not scored.

## Frames

| Lecture type | Frames | From |
|---|---|---|
| `blackboard` | 9 | `mit_18_100b_l1` (5), `mit_6_042j_2010_l1` (4) |
| `handwriting` | 9 | `ttk4225_2026_09_14` (5), `math231_2020_r2` (4) |
| `slides` | 6 | `mit_18_404j_l1` |
| `whiteboard` | 6 | `uw_machine_intelligence_l13` |

Keyframes chosen by the vision stage's own sampling, written by `scripts/extract_keyframes.py`
(architecture.md 6.4), then picked by hand for range. To see every candidate:

```bash
uv run python scripts/extract_keyframes.py lecture.mp4 -o keyframes/ --max-views 400
```

| Frame | What it tests |
|---|---|
| `mit_18_100b_l1_00072.0` | A nearly blank board being wiped: the empty-board answer |
| `mit_18_100b_l1_00746.0` | Handwriting and two hand-drawn graphs, lecturer partly in front |
| `mit_18_100b_l1_01507.0` | Wide shot of six boards at once, small writing |
| `mit_18_100b_l1_02898.0` | Formulas over ghosted chalk from erased boards, camera zoomed on one board |
| `mit_18_100b_l1_03415.0` | Inequalities and logic arrows, dust and smudges, lecturer pointing |
| `mit_6_042j_2010_l1_01001.0` | Large numbers and a formula, lecturer writing in front of it, 640x480 |
| `mit_6_042j_2010_l1_01470.0` | Long handwritten sentences, close-up |
| `mit_6_042j_2010_l1_01812.0` | A truth table being started over ghosted chalk |
| `mit_6_042j_2010_l1_02061.0` | Wide shot: a full truth table and a quantified statement, small |
| `ttk4225_2026_09_14_00257.0` | Printed question slide with a hand-drawn sinusoid (640x360) |
| `ttk4225_2026_09_14_00423.0` | Printed question with an integral and handwritten boxes and a sketch |
| `ttk4225_2026_09_14_00758.0` | Printed theorem and formula with handwriting below |
| `ttk4225_2026_09_14_01265.0` | Handwritten `h(t) = 1/(1+t)`, a graph and a diagram, app toolbar visible |
| `ttk4225_2026_09_14_01731.0` | Messy handwriting and sketches, app toolbar visible |
| `math231_2020_r2_00018.0` | Handwritten announcements with highlighting, webcam column |
| `math231_2020_r2_00180.0` | A limit problem with a fraction, started |
| `math231_2020_r2_00306.0` | The same page with three lines of working |
| `math231_2020_r2_00410.0` | Scrolled page: chained limits of fractions, cut off at the top |
| `mit_18_404j_l1_00264.0` | Text-only slide, two columns, presenter inset |
| `mit_18_404j_l1_01009.0` | State diagram, set notation and a check-in label |
| `mit_18_404j_l1_01365.0` | Formal definition with Greek letters and a transition table |
| `mit_18_404j_l1_01785.0` | Dense definitions with subscripts |
| `mit_18_404j_l1_02009.0` | Set-builder notation and a state diagram |
| `mit_18_404j_l1_03284.0` | Proof with products of sets, a struck-out formula and a diagram |
| `uw_machine_intelligence_l13_00571.0` | A diagram half hidden by the lecturer, cut-off words |
| `uw_machine_intelligence_l13_01366.0` | A handwritten comparison table |
| `uw_machine_intelligence_l13_02881.0` | Integral, sum and 3x3 matrices in blue and red, lecturer in front |
| `uw_machine_intelligence_l13_03420.0` | Matrices and a neuron diagram, lecturer covering part |
| `uw_machine_intelligence_l13_03811.0` | Two boards: formulas and 3D volume sketches |
| `uw_machine_intelligence_l13_04659.0` | A pipeline of box sketches with short labels |

## Consecutive views

`sequence/` holds what the previous board state (architecture.md 6.5) is measured on (#230):
consecutive views of one board as the vision stage cuts them, so each view is read after the one
before it, as in a run. The #58 frames above are far apart in time and cannot show that.
`ttk4225_2026_09_14` 1177 s to 1447 s: six views, written by the stage (720p, 1 frame per
second), of one handwritten page filling up, with three partial erases: a fragment at 1266 s,
`t ≥ 0` at 1340 s, and the right half of the page (`Y_FR(now)` and `u * h(now)`) at 1408 s.
Every pair keeps at least 0.79 of the earlier view's ink, so the stage sends the previous state
across each erase.

The labels follow the schema above, with one more field:

| Field | Type | Notes |
|---|---|---|
| `erased` | list of strings | text of the view before, as its `text` writes it, that is no longer on the board; `[]` when nothing was wiped |

```bash
uv run python scripts/compare_previous_state.py tests/data/vision/sequence/
```

reads each view alone (v0), after the previous reading on every pair (v2), and as the stage
sends it, and reports character error rate, formula recall, lines copied from the previous
reading and erased text still read. **Status.** Labelled by an LLM (Claude) from the frames and
the transcript; not yet checked by a human. As in the #58 labels, the numbers and arrow labels of
the diagram (`*h`, `1`, `2`, `3?`) are in `diagram`, not `text`, so a reading that types them
raises its character error rate.

## Question links

`<recording-name>.links.jsonl` labels each question with the board state on screen when it was
asked (#135). `tests/test_export.py` runs the pipeline on the committed transcript and board
states (`../<recording-name>.visual.json`), with no video or network, and checks each detected
question's `visual_ids` against the label.

```json
{"start": 97.27, "end": 98.57, "text": "Is it absolutely integrable?", "state_start": 93.0}
```

| Field | Type | Notes |
|---|---|---|
| `start`, `end`, `text` | number, number, string | copied from a `content` line of `../<recording-name>.labels.jsonl` |
| `state_start` | number | `start` of the board state in `../<recording-name>.visual.json` on screen when the question was asked; when the screen changes during the question, the state the question is about |

`ttk4225_2026_09_14` has 22 questions, biased towards questions near a change of board state,
checked by hand against the video.

## Sources and licences

Frames and labels carry the recording's licence, not the repository's MIT licence. Only
recordings whose licence allows redistributing derivatives are used.

| Slug | Recording | URL | Type | Licence |
|---|---|---|---|---|
| `mit_18_100b_l1` | MIT 18.100B Real Analysis, Lecture 1 | https://www.youtube.com/watch?v=BWvjy0a1Ffg | blackboard | CC BY-NC-SA 4.0, attribution: MIT OpenCourseWare, 18.100B Real Analysis |
| `mit_6_042j_2010_l1` | MIT 6.042J Mathematics for Computer Science, Fall 2010, Lecture 1, Tom Leighton | https://www.youtube.com/watch?v=L3LMbpZIKhQ | blackboard | CC BY-NC-SA, attribution: MIT OpenCourseWare, 6.042J Fall 2010. A clip OCW marks as not covered is not used. |
| `ttk4225_2026_09_14` | TTK4225, 2026-09-14, P03·12 | https://www.youtube.com/watch?v=ufmTACqiE78 | handwriting | the lecturer's own recording, committed with the lecturer's (Damiano's) permission |
| `math231_2020_r2` | Recitation 2 Recording for Math 231, Fall 2020, Linda Green | https://www.youtube.com/watch?v=8XYYUsFnl4A | handwriting | CC BY (YouTube's Creative Commons Attribution licence), attribution: Linda Green |
| `mit_18_404j_l1` | MIT 18.404J Theory of Computation, Fall 2020, Lecture 1, Michael Sipser | https://www.youtube.com/watch?v=9syvZr-9xwk | slides | CC BY-NC-SA, attribution: MIT OpenCourseWare, 18.404J Fall 2020. Only the lecturer's own slides are used. |
| `uw_machine_intelligence_l13` | Machine Intelligence, Lecture 13 (Convolutional Neural Networks), Kimia Lab, University of Waterloo | https://www.youtube.com/watch?v=AVwS3VpMGlo | whiteboard | CC BY (YouTube's Creative Commons Attribution licence), attribution: Kimia Lab |

## Prompts and comparison results (#52)

| File | What it is |
|---|---|
| `prompt_v0.txt` | Captioning prompt v0. `{kind}` is the layout probe's kind; `{speech}` the transcript overlapping the frame, as context. |
| `prompt_v1.txt` | v0 with a stricter `latex` instruction. It did not change how often models fill `latex`, so v0 stays. |
| `prompt_v2.txt` | v0 with the previous board state (#137): `{previous_description}`, `{previous_text}`, `{previous_latex}`, `{new_ink}` (the box around new writing, or "not found"), and a rule never to copy what is no longer visible. Sent when a state follows another without an erase. `scripts/compare_previous_state.py` reads frames with and without it (#137, #230). |
| `prompt_v0.schema.json` | The JSON the model returns: `description`, `text`, `latex` (or `null`), `confidence`, matching `VisualContext`. |

`results/<run>/results.jsonl` holds one line per call (frame, model, seconds, token counts, answer
or error), written by `scripts/compare_vision.py` over the ten MIT 18.100B and TTK4225 frames.
`results/glm-ocr/` and `results/glm-ocr-q8-local/` hold GLM-OCR's plain text, `results/tesseract/`
a classic OCR engine's. Comparisons and caveats: #52, #143, #144.

| Run | Models | Frames |
|---|---|---|
| `auto`, `low` | OpenAI, image detail auto and low | MIT |
| `openai-ttk-360p` | OpenAI | TTK |
| `idun-mit`, `idun-ttk-360p`, `idun-ttk-720p` | Qwen3.8-27B, Mistral Medium 3.5, Kimi K2.6 instant, on Idun's hosted service | MIT, TTK at 640x360, TTK at 1280x720 (not committed) |
| `idun-mit-speech`, `idun-ttk-360p-speech`, `idun-ttk-720p-speech` | the same, with the words spoken within 20 s of each frame | as above |
| `idun-large-models` | Kimi K2.6, DeepSeek-V4-Flash-Vision-Exp, GLM-5.3-Flash | MIT and TTK at 360p |
| `idun-mistral-repeat` | Mistral Medium 3.5, three repeats | two TTK frames at 720p |
| `deepseek-173` | DeepSeek-V4.1-Flash-1017 on Idun with thinking off, three repeat runs of `eval --score-vision` and `eval --detector llm` (`repeat-runs.txt`), plus first full runs with thinking on and off (`first-run-*.txt`) including Qwen3.8-27B and gpt-4o | 30 keyframes; the three labelled lectures |
| `idun-prompt-v1`, `openai-prompt-v1` | Qwen3.8-27B, Mistral Medium 3.5, Kimi K2.6 instant, GLM-5.3-Flash on Idun; gpt-5.6-luna and gpt-4.1-mini on OpenAI, all with `prompt_v1.txt` | MIT and TTK at 360p |
| `claude-hand-graded` | Claude Sonnet 5.5 reading each frame and answering in the v0 format. Not independent: the same reader wrote the reference text. | MIT and TTK |
| `claude-independent-agent` | A fresh Claude agent given only the prompt and the frames | MIT and TTK |
| `qwen3-vl-2b-instruct-local` | Qwen3-VL 2B through Ollama | MIT |
| `glm-ocr` | GLM-OCR (`zai-org/GLM-OCR`, 0.9B) on Idun's hosted service, prompt "Text Recognition:" | MIT and TTK at 360p |
| `glm-ocr-q8-local` | GLM-OCR `glm-ocr:q8_0` through Ollama on an 8 GB MacBook Air M2; `ollama-ps.txt` holds the memory in use | MIT and TTK at 360p |

To rerun a local comparison, `scripts/compare_vision.py --help` and `scripts/score_vision.py
--help`. `score_vision.py` scores word-sequence similarity (0 to 1), not CER or formula recall.

# Labelled keyframes for the vision model

Ground truth for `questxtract eval --score-vision` (#58): 30 keyframes from six lectures, each labelled
with the text on the board, its formulas as LaTeX and a line on any diagram, so the vision model
can be scored the way `tests/data/*.labels.jsonl` score question detection. The prompt and the
raw answers of the model comparison (#52) live here too.

```bash
uv run questxtract eval --score-vision tests/data/vision/        # gpt-5.6-luna, needs OPENAI_API_KEY
uv run questxtract eval --score-vision tests/data/vision/ --llm-url URL --vision-model NAME
```

The model is `vision_model` (`--vision-model`, or `questxtract.toml`). Each frame goes through
`vision.read_keyframe`, the function the vision stage reads a keyframe with, as a run sends it
when the layout probe got the frame's kind right: prompt v0 with the label's `kind`, no speech.
Frames are the full picture, not cropped to the probe's content region. Answers are cached, so a
second run makes no calls. CI never runs it: it needs a real model. CI does check the labels
(`tests/test_evaluate.py`): every line is well formed, its PNG exists, and every lecture type has
at least 5 frames.

## Layout

```text
tests/data/vision/
  <recording-name>.frames.jsonl      one file per recording, one line per labelled frame
  <recording-name>_<seconds>.png     the frames, named by the second they were taken at
  <recording-name>.links.jsonl       questions labelled with the board state on screen
  prompt_v0.txt, prompt_v1.txt       the captioning prompts of #52
  prompt_v2.txt                      v0 with the previous board state (#137)
  prompt_v0.schema.json              the JSON the model must return
  results/<run>/results.jsonl        raw answers of the #52 comparison
```

Slugs follow `tests/data/README.md`: a short, stable name per recording.

## Schema

```json
{"frame": "mit_18_404j_l1_02009.0.png", "kind": "slides", "text": "Regular Languages – Examples\n...", "latex": ["L(M_1) = \\{w \\mid w \\text{ contains substring } 11\\} = A"], "diagram": "The state diagram of automaton M1: ..."}
```

| Field | Type | Notes |
|---|---|---|
| `frame` | string | PNG file name beside the labels |
| `kind` | string | lecture type: `slides`, `blackboard`, `whiteboard` or `handwriting` (on a tablet, or paper under a document camera, also over slides); the layout probe's kinds (architecture.md 6.2), so a label's kind is what the probe should answer |
| `text` | string | every piece of text a careful reader can make out at full resolution, in reading order, one line per line on the board; `""` for a blank board |
| `latex` | list of strings | each formula on the board, once, as LaTeX; `[]` when there is none |
| `diagram` | string or null | one line on any diagram, graph or figure; `null` when there is none |

**Conventions.**

- `text` keeps the spelling on the board ("independet" stays). Formulas are also in `text`, typed
  plainly (`x^2`, `∫`, `≤`, `lim_{x→-2}`).
- Words cut off at the edge of the frame are written as far as they are visible ("ukushima").
  Writing too small or smudged to read is left out, as is ghosted chalk from an erased board.
- Text inside a diagram (state names, axis labels, edge labels) belongs to `diagram`, not
  `text`.
- Not board content, so left out: the lecturer, the room and its signs, video overlays (a
  university logo, the presenter's name tag, a webcam column), and the drawing app's toolbar
  and status bar. Text printed on a slide, its footer included, is board content.
- No frame shows a student's face.

**Status.** The labels were drafted by an LLM (Claude) reading each frame at full resolution, and
are provisional until a human has checked them against the frames, as for `../held_out/`.

## Scoring

Per frame, `evaluate.py` computes:

- **CER**, the character error rate of the model's `text` against the label: edits (insert,
  delete, substitute) per label character after casefolding and collapsing whitespace, capped
  at 1. 0 is a perfect reading, 1 is nothing right.
- **Blank boards** (label `text` is `""`) have no CER, since any text would score 1, the same
  as misreading a full board. They count as read when the answer's text is empty or only
  `[unreadable]`, in a column of their own.
- **Formula recall**: both formulas are normalised (`$`, `\left`, `\right`, `\displaystyle` and
  spacing commands removed, `\dfrac`/`\tfrac` read as `\frac`) and split into tokens: a command,
  a number, or one other character. A labelled formula counts as found when some run of whole
  tokens in the model's `latex` is within one edit per four of its tokens. So a formula of three
  tokens or fewer must match exactly, and a labelled `x` is not found inside `\max`.

A frame the model could not read counts as an empty answer and is listed as failed. The report
gives the mean CER, blank boards read as blank and pooled formula recall per lecture type and in
total, then every frame, worst first, with the formulas it missed. `diagram` is not scored yet: it is there for a human
or an LLM judge to compare against.

## Frames

| Lecture type | Frames | From |
|---|---|---|
| `blackboard` | 9 | `mit_18_100b_l1` (5), `mit_6_042j_2010_l1` (4) |
| `handwriting` | 9 | `ttk4225_2026_09_14` (5), `math231_2020_r2` (4) |
| `slides` | 6 | `mit_18_404j_l1` |
| `whiteboard` | 6 | `uw_machine_intelligence_l13` |

They are keyframes the vision stage's own sampling chooses, written by
`scripts/extract_keyframes.py` (the stable-view detection of architecture.md 6.4), then
hand-picked for range: blank and full boards, wide shots with small writing, a lecturer in front
of the board, ghosted chalk, diagrams, tables and dense formulas. Re-run it on a recording to see
every candidate:

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

## Question links

`<recording-name>.links.jsonl` checks the `visual_ids` that link each question in `questions.json`
to board states (#135): one line per question, labelled with the board state on screen when it
was asked. `tests/test_export.py` runs the pipeline on the committed transcript and board states
(`../<recording-name>.visual.json`), so no video and no network, and checks each detected
question's `visual_ids` against the label.

```json
{"start": 97.27, "end": 98.57, "text": "Is it absolutely integrable?", "state_start": 93.0}
```

| Field | Type | Notes |
|---|---|---|
| `start`, `end`, `text` | number, number, string | copied from a `content` line of `../<recording-name>.labels.jsonl` |
| `state_start` | number | `start` of the board state in `../<recording-name>.visual.json` on screen when the question was asked; when the screen changes during the question, the state the question is about |

`ttk4225_2026_09_14` has 22 questions, spread over the lecture and biased towards questions near
a change of board state. **Status.** Drafted from the board states' times and descriptions, and
provisional until a human has checked each against the video.

## Sources and licences

Frames and labels are derived from the recording and carry its licence, not the repository's MIT
licence (same rule as `tests/data/README.md`). Only a recording whose licence allows
redistributing derivatives is used; a public YouTube upload alone is not enough.

| Slug | Recording | URL | Type | Licence |
|---|---|---|---|---|
| `mit_18_100b_l1` | MIT 18.100B Real Analysis, Lecture 1 | https://www.youtube.com/watch?v=BWvjy0a1Ffg | blackboard | CC BY-NC-SA 4.0, attribution: MIT OpenCourseWare, 18.100B Real Analysis |
| `mit_6_042j_2010_l1` | MIT 6.042J Mathematics for Computer Science, Fall 2010, Lecture 1, Tom Leighton | https://www.youtube.com/watch?v=L3LMbpZIKhQ | blackboard | CC BY-NC-SA, attribution: MIT OpenCourseWare, 6.042J Fall 2010. A clip in the video that OCW marks as not covered is not used. |
| `ttk4225_2026_09_14` | TTK4225, 2026-09-14, P03·12 | https://www.youtube.com/watch?v=ufmTACqiE78 | handwriting | the lecturer's own recording, committed with the lecturer's (Damiano's) permission |
| `math231_2020_r2` | Recitation 2 Recording for Math 231, Fall 2020, Linda Green | https://www.youtube.com/watch?v=8XYYUsFnl4A | handwriting | CC BY (YouTube's Creative Commons Attribution licence), attribution: Linda Green |
| `mit_18_404j_l1` | MIT 18.404J Theory of Computation, Fall 2020, Lecture 1, Michael Sipser | https://www.youtube.com/watch?v=9syvZr-9xwk | slides | CC BY-NC-SA, attribution: MIT OpenCourseWare, 18.404J Fall 2020. OCW notes that some content in the video is not covered; only the lecturer's own slides are used. |
| `uw_machine_intelligence_l13` | Machine Intelligence, Lecture 13 (Convolutional Neural Networks), Kimia Lab, University of Waterloo | https://www.youtube.com/watch?v=AVwS3VpMGlo | whiteboard | CC BY (YouTube's Creative Commons Attribution licence), attribution: Kimia Lab |

A Brian Douglas video (diagrams) was used locally in #52 for measuring and is never committed: it
has the standard YouTube licence.

## Prompt and spike results (#52)

| File | What it is |
|---|---|
| `prompt_v0.txt` | Captioning prompt v0. `{kind}` is the frame kind from the layout probe (slides, blackboard, whiteboard, handwriting, screencast, unknown); `{speech}` is the transcript overlapping the frame, as context only. |
| `prompt_v1.txt` | Prompt v0 with a stricter `latex` instruction (every equation, even a short one). It did not change how often the models fill `latex`, so v0 stays. |
| `prompt_v2.txt` | Prompt v0 with the previous board state (#137): its `{previous_description}`, `{previous_text}` and `{previous_latex}`, `{new_ink}` (the box around the new writing, or "not found"), and a rule never to copy what is no longer visible. Sent when a state follows another without an erase; otherwise v0 is sent. `scripts/compare_previous_state.py` reads the frames both ways, for #137. |
| `prompt_v0.schema.json` | The JSON the model must return: `description`, `text`, `latex` (or `null`), `confidence`, matching `VisualContext` in `models.py`. |

Raw answers of the model comparison, one line per call: `results/<run>/results.jsonl`, written by
`scripts/compare_vision.py` (frame, model, seconds, token counts, the answer or the error), over
the ten MIT 18.100B and TTK4225 frames. The OCR baseline is GLM-OCR: `results/glm-ocr/` and
`results/glm-ocr-q8-local/` hold its plain text, and `results/tesseract/` a classic OCR engine's,
for comparison. The comparisons and their caveats are on #52 and #143; the summary is in
[`docs/architecture.md`](../../../docs/architecture.md) 6.1.

| Run | Models | Frames |
|---|---|---|
| `auto`, `low` | OpenAI, image detail auto and low | MIT |
| `openai-ttk-360p` | OpenAI | TTK |
| `idun-mit`, `idun-ttk-360p`, `idun-ttk-720p` | Qwen3.8-27B, Mistral Medium 3.5, Kimi K2.6 instant, on Idun's hosted service | MIT, TTK at 640x360, TTK at 1280x720 (not committed) |
| `idun-mit-speech`, `idun-ttk-360p-speech`, `idun-ttk-720p-speech` | the same, with the words spoken within 20 s of each frame | as above |
| `idun-large-models` | Kimi K2.6, DeepSeek-V4-Flash-Vision-Exp, GLM-5.3-Flash | MIT and TTK at 360p |
| `idun-mistral-repeat` | Mistral Medium 3.5, three repeats | two TTK frames at 720p |
| `deepseek-173` | DeepSeek-V4.1-Flash-1017 on Idun with thinking off (`reasoning_effort` none), three repeat runs of `eval --score-vision` and `eval --detector llm` (`repeat-runs.txt`), plus the first full runs with thinking on and off (`first-run-*.txt`), which include Qwen3.8-27B and gpt-4o for comparison | 30 keyframes; the three labelled lectures |
| `idun-prompt-v1`, `openai-prompt-v1` | Qwen3.8-27B, Mistral Medium 3.5, Kimi K2.6 instant, GLM-5.3-Flash on Idun; gpt-5.6-luna and gpt-4.1-mini on OpenAI, all with `prompt_v1.txt` | MIT and TTK at 360p |
| `claude-hand-graded` | Claude Sonnet 5.5 reading each frame itself and answering in the prompt v0 format. **Not independent**: the same reader wrote the reference text the other models are scored against. | MIT and TTK |
| `claude-independent-agent` | A fresh Claude agent with no context, given only the prompt and the frames, told to open nothing else | MIT and TTK |
| `qwen3-vl-2b-instruct-local` | Qwen3-VL 2B through Ollama | MIT |
| `glm-ocr` | GLM-OCR (`zai-org/GLM-OCR`, 0.9B) on Idun's hosted service, prompt "Text Recognition:" | MIT and TTK at 360p |
| `glm-ocr-q8-local` | GLM-OCR `glm-ocr:q8_0` through Ollama on an 8 GB MacBook Air M2, same prompt; `ollama-ps.txt` holds the memory in use | MIT and TTK at 360p |

## Run the OCR comparison on your own computer

GLM-OCR and Qwen3-VL 2B on the 10 MIT 18.100B and TTK4225 frames, through a local Ollama: the
laptop measurement of #144. Works on macOS, Windows and Linux; about 10 minutes plus the downloads.
Results go to `spike-results/`, which Git ignores.

1. **Install Ollama** from [ollama.com](https://ollama.com) (on macOS also `brew install
   ollama`, on Windows also `winget install Ollama.Ollama`).
2. **Start it.** The app keeps running in the background (menu bar or tray). Check with
   `ollama ps`; if it cannot connect, run `ollama serve` in a terminal of its own and leave it open.
3. **Download the two models** (about 4-6 GB of disk):

   ```bash
   ollama pull glm-ocr:q8_0
   ollama pull qwen3-vl:2b-instruct
   ```

4. **Run them**, from the repository folder after `uv sync --dev`. The same commands work in
   Terminal, Git Bash and PowerShell; keep the quotes, the script expands the patterns itself:

   ```bash
   uv run python scripts/compare_vision.py "tests/data/vision/mit_18_100b_*.png" "tests/data/vision/ttk*.png" -m glm-ocr:q8_0 --ocr "Text Recognition:" --base-url http://localhost:11434/v1 -o spike-results/glm-ocr
   ollama ps
   uv run python scripts/compare_vision.py "tests/data/vision/mit_18_100b_*.png" -m qwen3-vl:2b-instruct --base-url http://localhost:11434/v1 --kind blackboard -o spike-results/qwen3-vl-2b
   uv run python scripts/compare_vision.py "tests/data/vision/ttk*.png" -m qwen3-vl:2b-instruct --base-url http://localhost:11434/v1 --kind tablet -o spike-results/qwen3-vl-2b
   ollama ps
   ```

5. **Score them** against the reference reading:

   ```bash
   uv run python scripts/score_vision.py spike-results/glm-ocr spike-results/qwen3-vl-2b
   ```

   The score is word-sequence similarity (0 to 1) against the reference reading: how many of the
   frame's words a reader returned, in order. It is not a character error rate or a formula
   recall, so do not compare it with those.

6. **Post on #144** the table, both `ollama ps` outputs (memory, and whether it ran on the GPU)
   and your computer (model, RAM, GPU).

Close heavy apps first: a model stays loaded for 5 minutes after use, and two at once can slow an
8 GB machine down (`ollama stop <model>` unloads one). The first frame of a run includes loading
the model. A second run appends to the same `results.jsonl`, so delete its folder before running
again.


# The QuestXtract Pipeline

How QuestXtract turns a lecture recording into questions with timestamps, answers and provenance.
Requirement ids (FR/NFR): [general_requirements.md](general_requirements.md). Why it is built this
way: [design_decisions.md](design_decisions.md). The audio-only prototype this package replaces is
[lecture-qa-extraction](https://github.com/matteopogetta/lecture-qa-extraction/tree/985c6f11594d0863274e13e2002b59bf17ad1c7f)
(commit `985c6f1`).

---

## 1. Overview

### 1.1 What the pipeline does

In: a lecture recording. Out: a JSON file of the questions asked in it, each with a timestamp, a
confidence, an answer span where the lecture has one, and the sentence ids it came from.

The package writes files: that JSON, a quiz made from it (12.3), and LMS and ExerPlaza exports of
the quiz (12.4, 12.5). Editors, quiz UIs and uploads live in a wrapper around it.

Most interrogative-sounding sentences in a lecture are not questions we want: rhetorical framing
("what happens if we double this?"), check-ins ("can everyone hear me?"), polls ("one or two?")
and tags ("right?"). Real questions are often answered several sentences later by the same
speaker. The input is ASR output, with unreliable sentence boundaries and punctuation and no
speaker labels. So the pipeline rebuilds timed sentences before detection and rejects false
positives after it.

### 1.2 The reference recordings

| Recording | Regime | What it stresses |
|---|---|---|
| [MIT 18.100B Real Analysis, L1](https://www.youtube.com/watch?v=BWvjy0a1Ffg) | chalk, sliding blackboards, dense LaTeX math, camera pans and zooms, few audible student questions | vision, LaTeX, ASR on math jargon |
| [TTK4225, 2026-09-14, P03·12, impulse responses and BIBO stability](https://www.youtube.com/watch?v=ufmTACqiE78) | digital tablet notes (OneNote/PDF sheet), single instructor, control-theory derivations with Socratic back-and-forth | implicit-cue and self-answered questions, ASR on control-theory notation |
| [TTK4225, 2026-09-15, P03·14, sinusoidal fidelity](https://www.youtube.com/watch?v=IQa2TSykWdc) | digital tablet notes, same course, frequent student Q&A | echo pairs (student question + instructor restatement), confirmation tags ("...right?", "...true?") |

- Video layouts vary (single camera, speaker plus screenshare, blackboard only, tablet notes,
  picture-in-picture), so section 6 asks a VLM for the layout.
- The TTK4225 instructor says a confirmation tag several times a minute;
  `docs/annotation_guidelines.md` 2 says how it is labelled.
- Students are often inaudible. The lecturer repeats the question, usually with its `?`. An
  off-mic question is not in the transcript, so no text rule can find it.

### 1.3 Stage order

```
                 .mp4 or .mp3 recording
                              |
  1 ingest                    |  validate, probe via PyAV, hash for cache key
                              v
  2 transcribe                |  audio -> temporary WAV -> faster-whisper
                              |  (VAD on, word_timestamps on) -> Word[] + AsrSegment[]
                              v
  3 sentences                 |  split on terminal [.!?] at word boundaries; run-ons
                              |  broken at pauses, then capped -> Sentence[]
                              v
  4 segments                  |  gap/duration/length breaks -> Segment[]
                              v
  5 vision (optional)         |  1 fps frames -> stable views -> VLM per keyframe
                              v
  6 detect                    |  lexical scoring + vetoes -> QuestionCandidate[]
                              v
  7 answer                    |  adjacent seed -> QAPair[]
                              v
  8 judge (optional)          |  --detector llm: an LLM reads the transcript and replaces
                              |  6 and 7 with the questions and answers it picks -> QAPair[]
                              v
  9 score                     |  confidence, threshold -> QAPair[]
                              v
 10 export                    |  questions.json (quiz.json is a separate command, 12.3)
```

Stages 5 and 8 are the only ones that call an LLM endpoint, and both are off by default.

`pipeline.STAGES` is rows 1 to 9, with vision split into `layout`, `vision` and `attach_visual`
(11.1). Export runs on the finished `Context` and is not a stage.

### 1.4 The rule path and the judge

By default, detection, the adjacent answer seed and the confidence are rules. With
`--detector llm`, the LLM detector (the judge, section 10) reads the transcript, picks the
questions, finds the sentences that answer them and may reword a question. A run with no endpoint
still produces questions.

The judge answers with sentence ids, and every id is checked against the transcript it was sent
(requirement 4.5). It may reword a question it found, with the spoken version kept beside it, but
never add one. Quiz generation writes new text and is a separate command (12.3).

### 1.5 Thresholds are measured, not assumed

Every threshold in sections 7 to 9 is swept by the evaluation harness (section 4) against the
labelled set. A new rule lands when a labelled failure needs it.

---

## 2. Front end: media to timed words

### 2.1 Ingest

`media.py` validates the path, opens the container with `av.open()` for duration and stream
layout, and hashes the file (`sha256` of the first and last 8 MiB plus the size). The hash keys
every later cache entry. PyAV bundles FFmpeg, so there is no system binary to install.

An `.mp4` or `.mp3` is accepted. A recording without a video track (any MP3, or an audio-only
MP4) is transcribed as usual and has no board to read. The faster-whisper backend calls
`media.extract_audio`, which decodes the first audio track into a temporary 16 kHz mono WAV and
deletes it after transcription. One recording per run, one clock.

### 2.2 Transcription

`faster-whisper`, batched, behind the `AsrBackend` interface:

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = BatchedInferencePipeline(WhisperModel(model_size, compute_type="int8"))
segments, info = model.transcribe(
    str(audio_path),  # 16 kHz mono WAV from media.extract_audio
    beam_size=5,
    language="en",  # English lectures (requirement 4.2)
    vad_filter=True,
    word_timestamps=True,
    initial_prompt=_PUNCTUATION_PROMPT,
)
```

- `vad_filter=True`: Whisper hallucinates over silence.
- `word_timestamps=True`: the only source of timing (2.3), and what per-word progress (11.3)
  uses.
- `initial_prompt` biases Whisper toward punctuation, question marks included (7.3). It only
  conditions the first window and is dropped when temperature fallback trips.

Default `model_size` is `turbo` (`large-v3-turbo`), usable on an `int8` CPU; `base` for quick
runs. `batch_size` is 4. Each `AsrSegment` keeps `avg_logprob`, `no_speech_prob` and
`compression_ratio`; nothing reads them yet.

The ASR engine is a class implementing `AsrBackend` (`transcribe(source, config, on_progress)`),
registered with `@register_backend` in `asr.py` and picked by `Config.asr_backend` (15.3).
`transcribe.py` never imports an engine.

On a CPU, `turbo` takes about as long as the recording, so the reference transcripts are
committed in `tests/data/` and work on sections 3 to 9 never runs Whisper.

### 2.3 Timing: word timestamps, no forced alignment

Whisper word timestamps (about ±100 to 200 ms) are the only timing source. There is no alignment
stage: the video side, sampled at 1 fps, is coarser than that (D28). Segment timestamps are not
used, since they drift by seconds.

### 2.4 Optional: vision-informed second ASR pass (not planned)

Re-transcribing with faster-whisper's `hotwords` set from board text would fix domain words
(*supremum*, *Cauchy*). Not planned (#62) until the eval shows vocabulary errors costing
questions.

### 2.5 Words are the atomic unit

There is no utterance layer. A sentence owns a half-open word index range, so its timings are
`words[i].start` and `words[j-1].end` and its provenance is the range. Timings are never
interpolated.

### 2.6 One speaker, no diarization

One main speaker is assumed (requirement 4.2). There is no diarization or speaker attribution
(D28).

---

## 3. Sentences and segments

### 3.1 Sentences

Whisper's chunks are not sentences. One walk over the word list asks each word whether it ends a
sentence. `_is_terminal` holds back three cases:

| Case | Example | Rule |
|---|---|---|
| abbreviation | `So Dr. Rudin defines it.` | a short suppression list; `Step 1.` still splits |
| single-letter initial | `for all n greater than capital N.` | only a capital counts; `u.` ends a sentence |
| ellipsis | `we take the limit... of the sequence` | not a stop |

A span that ends in terminal punctuation and fits in `max_sentence_words` (60) is a sentence. Any
other span is split at every silence of at least `force_split_min_gap_seconds` (0.6), then a piece
still over the limit is halved at its widest gap, ties toward the middle, until it fits (D29).

`stats["sentences_unpunctuated"]` counts sentences without terminal punctuation; the eval's
`question_mark_rate` (4.2) measures the same risk on the labelled set.

### 3.2 Segments

A segment is a run of sentences bounded by silence and length, not a topic. It closes on the
first of:

- a silence gap over `segment_max_gap_seconds` (2.0 s, strictly greater)
- a sentence that would push it past `segment_max_duration_seconds` (90 s)
- a sentence that would push it past `segment_max_chars` (800)

A single sentence already over a limit is its own segment. `stats["segments_closed_gap" |
"_duration" | "_chars"]` counts the reasons; caps closing more than a few segments means the
thresholds do not fit a recording (D30).

The board-state join (`attach_visual`, 11.1) uses segment intervals. Detection never reads
segments. The FR11 topic filter needs topics, which nothing produces (`topic=` in `filtering.py`
is experimental).

---

## 4. Evaluation harness

NFR2 asks for recall >= 50% on the labelled set, so the metric runs in CI.

### 4.1 The labelled set

One JSONL file per recording, `tests/data/<name>.labels.jsonl`:

```json
{"start": 1284.2, "end": 1288.9, "text": "why does a stable sort matter here", "kind": "content"}
```

next to its transcript, `tests/data/<name>.transcript.json`. A label's `text` quotes the
transcript and its `start`/`end` are snapped to the quoted words
(`scripts/scratch_to_labels.py --transcript`).

`kind` is `content` (a question we want), `rhetorical`, `check_in` or `procedural`.

Each transcript is labelled sentence by sentence with no seed detector, so every entry is
`blind`. A question the ASR never transcribed is not labelled. The MIT labels were drafted by an
LLM and reviewed by a human (#93). `tests/data/held_out/` holds recordings nothing is tuned on
(#92); `eval tests/data/` does not read them, `eval tests/data/held_out/` does. Totals and
conventions: `tests/data/README.md` and `docs/annotation_guidelines.md`.

### 4.2 Matching and metrics

A prediction matches a label when their time ranges overlap by at least half the shorter range.
Matching is one-to-one, highest overlap first.

```
recall    = matched_content_labels / content_labels        <- NFR2 gate, >= 0.50
precision = matched_content_labels / predictions
f1
false_positive_breakdown: by the kind of label a prediction wrongly matched, plus unmatched
question_mark_rate: fraction of content labels whose transcript span contains '?'
blind_slice_recall: recall over the entries labelled "blind" only
```

A falling `question_mark_rate` points at the sentence splitter (3.1).

### 4.3 Running it

```bash
questxtract eval tests/data/            # all labelled recordings, table to stdout
questxtract eval tests/data/ --json     # machine-readable, for CI
questxtract eval tests/data/ --board    # with the committed board states
```

`evaluate.py` runs stages 3 onward over the committed transcript with `min_confidence` at 0, then
gates at the default for the headline, so the threshold sweep covers every candidate. The report
lists every missed content question and every false positive with the label kind it hit.

`tests/test_evaluate.py` asserts pooled recall >= 0.50 over the reference recordings and
separately over `held_out/`.

`--board` puts the board states in `<name>.visual.json` beside a transcript into the run in place
of the vision stage's. The files are one vision run per recording (`scripts/board_states.py`); a
recording without one runs without the board.

### 4.4 Vision

The vision model is scored on labelled keyframes, `tests/data/vision/<name>.frames.jsonl`: one
line per frame with its lecture type (`slides`, `blackboard`, `whiteboard` or `handwriting`), the
board text, each formula as LaTeX and a line on any diagram. Sources, licences and conventions:
[`tests/data/vision/README.md`](../tests/data/vision/README.md).

```bash
questxtract eval --score-vision tests/data/vision/    # vision_model, per lecture type
```

Each frame is read by `vision.read_keyframe`, as a run reads it: `vision_model`
(`--vision-model`), or `vision_ocr_model` with `--vision-reader ocr`, prompt v0 with the label's
kind, no speech or previous state; cached per request. Per frame:

- character error rate of `text`: edits per label character, casefolded, whitespace collapsed,
  capped at 1
- formula recall: a labelled formula counts as found when some run of whole LaTeX tokens in the
  answer is within one edit per four tokens, after removing spacing and sizing commands; a short
  formula must match exactly
- a blank board has no error rate; it counts as read when the answer has no text but
  `[unreadable]`

A frame the model cannot read scores as an empty answer. The report gives all three per lecture
type and in total, then every frame, worst first. `diagram` is labelled but not scored (D20). CI
checks the labels (every type has at least 5 frames, every frame exists), not the scores.

---

## 5. Data model

All in `models.py`, except `Config` (`config.py`). Every type is
`@dataclass(frozen=True, kw_only=True)`, except `Context`.

| Type | What it is |
|---|---|
| `Word` | one transcribed word with `start`, `end`, `confidence` (2.5) |
| `AsrSegment` | a raw ASR decoding chunk; not a `Segment` |
| `Transcript` | the words, the ASR chunks, `language`, `duration`; `from_dict`, `from_file`, `to_dict` |
| `Sentence` | a half-open word range with its text and timings; `id` is a content hash |
| `Segment` | a run of sentences (3.2), its `close_reason`, and the board states that overlap it |
| `Layout` | where the content is in the frame (`content_region`) and its `kind`, one per recording (6.2) |
| `VisualContext` | one board state on its own clock: keyframe, description, text, `latex` (section 6) |
| `QuestionCandidate` | a detected question: sentence id, text, score, `question_type`, `reasons` |
| `QAPair` | a candidate with its answer span, confidence, timestamps and `verdict` |
| `Context` | the one mutable object every stage takes and returns (11.1) |

- `reasons` is append-only. Every rule that fires adds a stable string. A question the LLM
  detector keeps also gets `llm`.
- The export writes a fixed key set, `null` included.
- Board states live on `Context.visual` with their own time ranges. `Segment.visual` holds the
  states that overlap the segment, joined in the `attach_visual` stage.

---

## 6. Vision

Optional, off unless `vision_enabled`. The `layout` stage asks a VLM once where the content is and
what kind it is (6.2). The `vision` stage samples the recording inside that region (6.3), cuts it
into stable views with one keyframe each (6.4) and has a VLM read each keyframe (6.5). The ink
curve (6.4) and de-occlusion (6.3) are not planned.

### 6.1 The regime problem, and why not OCR

A VLM reads handwriting, diagrams and slides, emits LaTeX, and tells the board from the room, in
one call through the same optional client as the LLM detector (section 10). An OCR model reads
text and formulas only; it is the optional second reader (6.5). `vision` is one `STAGES` entry, so
tests replace it with a stub that loads pre-read frames.

**Model and prompt (#52, #153, #173, #179).** The default `vision_model` is DeepSeek-V4.1-Flash
(`deepseek-ai/DeepSeek-V4.1-Flash-1017`) on NTNU's LLM service with its thinking off
(`vision_reasoning_effort = "none"`), D22. Qwen3.8-27B on the same service needs
`vision_reasoning_effort = ""`. Of the hosted OpenAI models, `gpt-5.6-luna` is the cheapest that
stays under the customer's cost per lecture and fills `latex` on every formula frame. `detail: low`
is not used: it garbles handwriting. GLM-OCR (0.9B, open weights) reads less than the vision
models, gives no description and also reads the room and the app interface. Prompt v0 and its
schema are in [`tests/data/vision/`](../tests/data/vision/README.md); the raw answers are in
`tests/data/vision/results/`.

### 6.2 Layout probe

The `layout` stage sends three frames, at 20, 50 and 80 % of the recording and scaled to at most
640 pixels wide, to `vision_model` in one request, under a strict JSON schema:

```json
{"content_region": [0.32, 0.05, 0.66, 0.80], "kind": "blackboard"}
```

`content_region` is `[x0, y0, x1, y1]` in fractions of the frame from its top-left corner. `kind`
is `slides`, `blackboard`, `whiteboard`, `handwriting` (tablet or paper), `screencast` or
`unknown`. The answer becomes `Context.layout`: `vision` crops every sampled frame to the region
and puts the kind into the reading prompt (6.5).

The box is padded by 0.02 on each side within the frame, and a box under a tenth of the frame
keeps its kind but not its crop.

An endpoint error, a stall or invalid JSON twice, or a malformed answer (a region that is not four
fractions with `x0 < x1` and `y0 < y1`, an unknown kind) leaves the full frame and `unknown` with a
warning. A setting that cannot be used, such as an unreadable `.env`, stops the run.

The request, frames included, is cached per request (10.4), and only a valid answer is stored.
The views `vision` cuts (6.4) are cached in `views.json` keyed on the content region (11.2).

One answer holds for the whole recording: a layout that changes mid-lecture keeps the first one.

### 6.3 Frame sampling and de-occlusion

Sample at `vision_fps` (1) through PyAV, crop to `content_region` (6.2), grayscale, downscale to
about 480p.

**De-occlusion (not planned, #127).** A temporal median to remove a lecturer standing in front of
the board. The keyframe choice in 6.4 avoids most occluded frames.

### 6.4 Board states

**Static views.** Each sample is compared with the first sample of the current view: the share of
pixels whose grey level moved by more than 32 levels. Above `vision_change_fraction` (0.02), a new
view starts at that sample once `vision_change_hold` (3) samples in a row differ from the old view
and resemble each other. The last view ends at the recording's end. Each view's keyframe is its
sharpest sample (mean gradient) among those that still match it.

Over the board-state limit, the shortest view is merged into its shorter neighbour until the limit
holds. The limit is `vision_states_per_minute` (3) per minute and never fewer than
`vision_min_states` (80). Rate: #153.

A lecturer standing still in front of the board for the whole hold is a view of its own, with the
board partly hidden in its keyframe.

**Ink curve (not planned, #133).** Cutting at erase events found by counting ink pixels lost board
content on the lectures it was measured on. Its ink rule is used in 6.5.

### 6.5 Captioning

One VLM call per view: the prompt (`VISION_PROMPT_VERSION`, v2 below; v0, measured on #52, for a
new board), filled in with the layout probe's `kind` (6.2) and up to 1500 characters of what was
said while the view was on screen, as the system message, and the keyframe as a PNG data URL as
the user message. Out, under a strict JSON schema and checked again in code: `description`,
`text`, `latex` (or null) and `confidence`. The prompt asks for what is visible, with
`[unreadable]` for a word it cannot read. Cached per request, image included (10.4).

**Previous board state.** Each view is read with the view before it as its previous state: that
reading's `description`, `text` and `latex`, and the box around the new ink, as fractions of the
frame. This is prompt v2 (`tests/data/vision/prompt_v2.txt`), D27. Ink is a pixel more than 20
grey levels from the mean of the 15 px square around it, on the small grey copy (6.3). The board
is new, and v0 is sent, when less than 70 % of the previous keyframe's ink is still ink, for the
first view, after a failed reading and after a blank board. New ink is boxed by the 8 px cells
that are at least 20 % new ink. A copied reading (below) counts as a previous state. A partial
erase keeps over 70 % of the ink (`tests/data/vision/sequence/`), so the erased text goes with
the next view; the prompt's rule not to copy keeps it out. The OCR reader gets no prompt.

**A page that comes back is not read again** (D18). On `slides`, `handwriting` and `screencast`,
each keyframe is compared with the keyframes already read in this run; boards, and `unknown`, are
always read again. The earlier keyframe whose small grey copy differs least, by at most 1 %, is
the candidate. Its reading is copied only if, at full resolution, no 4 px square's mean brightness
moved by more than 20 levels, no 8 px square's by more than 11, and no 8 px square's mean colour
(Cb, Cr) by more than 12. The new board state keeps its own times, keyframe and id and counts in
`stats.vision_reused`. A failed reading is never copied. The limits come from the codec noise of
TTK4225 2026-09-14; a mark fainter than about 48 grey levels of contrast on a light page can go
unnoticed. Reuse works within the run, also with `--from-scratch`.

Each keyframe read is cached as `<source_hash>/keyframes/<content hash>.png`, and
`keyframe_path` holds that path relative to `cache_dir`. Frame times count from the start of the
audio stream. A keyframe whose bytes no longer match its hash name is sampled again. Writing
`questions.json` copies the images beside it (12.1).

Captioning is never required: an endpoint error, a stall or invalid JSON twice, or a malformed
field leaves that keyframe out with a warning, counted in `stats.vision_failed`, and nothing is
cached. When the first three readings fail and none has succeeded, the rest are not sent. A
setting that cannot be used, such as an unreadable `.env`, stops the run.

**The OCR reader.** `vision_reader = "ocr"` (`--vision-reader ocr`) reads each keyframe with
`vision_ocr_model` (GLM-OCR, `zai-org/GLM-OCR` on NTNU's service): the image and the prompt
`Text Recognition:` in one user message, with token probabilities asked for, through vision's
endpoint and key (10.1). It answers in plain text with formulas inline as `$...$`, which
`_from_ocr` turns into the same fields: `text` the words, formulas left out and labels written
plainly; `latex` the formula segments one per line, leaving out labels (a circled number, a number
with a unit, a single symbol); `description` empty; `confidence` the mean token probability, or
0.5 when the server sends none. The speech and the kind are not sent. The layout probe still asks
`vision_model`; where none answers, the probe falls back to the full frame (6.2).
`questions.json` names the OCR model in `config.vision_ocr_model` (12.1).

### 6.6 Consumers

- Detection (7.3): `board_overlap_delta` when a question shares a content word with the `text` of
  a board state on screen while it was asked.
- Search (`filtering.py`, `questxtract filter`): a text query matches every word against what was
  said and the board states in a question's `visual_ids`, as whole words, ignoring case.
- The quiz (12.3): the linked board states go into each request.

Offline use points `vision_base_url` or `base_url` at a local OpenAI-compatible server.

---

## 7. Question detection

Lexical and orthographic rules in `detect.py`; every decision leaves a reason code. Whether a
question is about the content is the LLM detector's call (section 10, D31).

### 7.1 Extraction

One candidate per sentence (3.1), scored in 7.3.

**The interrogative head** is the first word past leading discourse markers (`so`, `and`, `but`,
`now`, `then`, `okay`, ...), question preambles (*"my question is (that)"*, *"the question
is/was"*, *"I'd like to ask"*, *"I was wondering (if)"*), and a preposition right before a wh-word
(*"For which age ...?"*). For a sentence ending in `?`, later comma clauses count too: the last
clause with any head (*"If you have the data, does that mean ...?"*), an earlier one only with an
auxiliary (*"This area here, is it finite, infinite, ...?"*). Without the `?` the clause rule is
off. Contractions stay whole, so negated auxiliaries (*"Don't we ...?"*) open questions; a wh-word
with a contracted verb (*"What's"*) counts as the wh-word.

**The answer seed** is described in section 8.

Implicit cues without a `?` (#48) and missing-`?` recovery have no labelled failure behind them
and are not built.

### 7.2 Vetoes

Hard rejects before scoring: empty text, and text over `max_question_chars` (350).

### 7.3 Scoring

Additive, clamped to [0, 1], with `framing` subtracted after clamping:

```
  +0.60  mark      ends in '?'
  +0.40  wh        interrogative head is a wh-word
  +0.40  aux       interrogative head is an auxiliary, negated ones included
  -0.40  tag       the sentence, or its closing clause, is only a confirmation tag:
                   "Okay?", "True, you see?", "..., right?" - unless what precedes the tag
                   has its own interrogative head ("Why is this not a field, right?")
  -0.40  check_in  classroom management: "who here", "want to tell me", "your name", ...,
                   checks of the room ("do you see it", "is everything clear", "are you
                   following", "how many of you", a bare "Everyone?"), and "is that clear",
                   "make sense", "do you see" unless content follows them
                   ("Why does it make sense to ...?", "Do you see why ...?" are questions)
  -0.40  framing   a marked interrogative with no content word after stopword removal:
                   "So how do we do that?", "So what did it do?"
```

Gate: `min_question_score` (0.45). With the confidence of 9.2 and the emission threshold of 0.40:

| Sentence | Score | Candidate | Unanswered | Answered by the seed |
|---|---|---|---|---|
| `?` and a head | 1.00 | yes | 0.55, kept | 1.00, kept |
| `?` only | 0.60 | yes | 0.33, dropped | 0.60, kept |
| check-in with `?` and a head | 0.60 | yes, reason kept | 0.33, dropped | never seeded |
| framing with `?` and a head | 0.60 | yes, reason kept | 0.33, dropped | never seeded |
| tag | 0.20 | no | | |
| head, no `?` | 0.40 | no | | |

The tag and check-in lists are English (requirement 4.2) and come from the labelled recordings.

Framing stopwords are wh-words, positive auxiliaries, discourse markers, prepositions, pronouns,
articles and a few fillers; negation, numbers and ordinary verbs count as content. Wh contractions
are normalized ("What's" to "what"). Tags and check-ins do not also get `framing`.

**Board context.** A candidate that shares a content word with the `text` of a board state whose
lifetime overlaps the sentence gets `board_overlap_delta` added, clamped, and the reason
`board_overlap`; its `source` is then `combined` (12.1). It applies after the gate, so it never
makes a candidate. Only `text` counts, not `description`. Default `0.0`, off (D19).

**Current results**, default config, `questxtract eval tests/data/`. Update this table in the PR
that changes the numbers.

| Recording | Content labels | TP | FP | FN | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| MIT 18.100B L1 | 28 | 28 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| TTK4225 2026-09-14 | 41 | 39 | 10 | 2 | 0.95 | 0.80 | 0.87 |
| TTK4225 2026-09-15 | 60 | 59 | 18 | 1 | 0.98 | 0.77 | 0.86 |
| **Pooled** | 129 | 126 | 28 | 3 | **0.98** | 0.82 | 0.89 |

---

## 8. Answer search

One rule, the adjacent seed (`answer.py`): the sentence right after the question's own, when the
silence before it is at most `answer_seed_max_gap_seconds` (1.0 s) and it is not itself a
question. Tags, check-ins and framing candidates get no seed. There is no speaker constraint.

Answers further away come from the LLM detector (section 10, D11). With no endpoint, a question
keeps its seed or has no answer.

---

## 9. Scoring

### 9.1 Answer scoring (not planned)

An answer the adjacent seed found counts as found; better answers come from the LLM detector
(10.2).

### 9.2 Confidence

```
confidence = 0.55 * question_score    no answer found
           = 1.00 * question_score    the adjacent seed found one

label:  >= 0.75 high | >= 0.45 medium | else low
```

A question with no answer is still exported (FR6, FR7). Gate: `min_confidence` (0.40). With the
LLM detector, the confidence is the model's own (10.2), with no answer weighting.

### 9.3 Dedupe (not planned)

Echo pairs and near duplicates stay in the output. The LLM detector labels a fragment echo
`procedural` (annotation_guidelines.md 2).

---

## 10. The LLM layer

One client, three jobs, each off unless asked for: the LLM detector (stage 8, `--detector llm`),
vision (section 6) and the quiz (12.3). Each keeps its own prompt, schema and output checks.
Requests go through `endpoint.py`, which is not a stage: it builds the client and sends a request
with a deadline on the whole call, one resend after a stall or invalid JSON, the per-request cache
of 10.4, and the key hidden in every error. The `openai` SDK is in the optional extra
`questxtract[llm]` and imported inside functions.

### 10.1 One client, any endpoint

```python
OpenAI(api_key=ntnu_key, base_url="https://llm.hpc.ntnu.no/v1")  # NTNU's LLM service, the default
OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")  # OpenRouter
OpenAI(api_key="local", base_url="http://localhost:11434/v1")  # a local server: Ollama
```

By default all three jobs go to NTNU's LLM service: Qwen judges and writes the quiz, DeepSeek
reads the board (D25). Vision follows `base_url` unless `vision_base_url` is set
(`endpoint.route`). Both take a URL or a short name, any case, which `Config` turns into the URL
(`config.ENDPOINTS`): `ntnu`, `openrouter`, `openai`, `ollama` (`http://localhost:11434/v1`). The
key goes with the endpoint (`config.key_names`, D35):

| Endpoint | Key |
|---|---|
| NTNU's LLM service, host `llm.hpc.ntnu.no` (the default `base_url`) | `IDUN_API_KEY` (also read: `QUESTXTRACT_NTNU_API_KEY`) |
| OpenAI, host `api.openai.com` (or `base_url = None`) | `OPENAI_API_KEY` |
| OpenRouter, host `openrouter.ai` | `OPENROUTER_API_KEY` |
| a `vision_base_url` on any other host | `QUESTXTRACT_VISION_API_KEY` |
| any other host, a local server included | `QUESTXTRACT_LLM_API_KEY`, which may be unset |

The service is told by the URL's host, so each key goes to its own service only. Each key is read
from the environment, then the working directory's `.env`, then the user's key file (cli.md "API
keys and .env"); never from `questxtract.toml`. The CLI checks the key of every endpoint a run will
use before anything slow starts; NTNU's, OpenAI's and OpenRouter's need one and are refused over
plain `http://`. A request without a key carries the placeholder `"local"`, since the SDK would
otherwise read `OPENAI_API_KEY` itself. The CLI also refuses a model name from another service:
NTNU's and OpenRouter's models are named `organisation/model`, OpenAI's have no `/`, and OpenRouter
names NTNU's default models differently.

| Config key | Default | Role |
|---|---|---|
| `judge_model` | `Qwen/Qwen3.8-27B-FP8` | the LLM detector; one request per group of nearby candidates |
| `vision_model` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | reads the board (6.5) and finds it (6.2) |
| `quiz_model` | `Qwen/Qwen3.8-27B-FP8` | writes the quiz (12.3); one request per question |

Any OpenAI-compatible server (Ollama, llama.cpp, vLLM, LM Studio) works by `base_url` and its own
model names (D12).

`reasoning_effort` (`--llm-reasoning`) goes with every judge and quiz request; its default `""`
(`server`) sends no setting. Vision sends `vision_reasoning_effort` (default `"none"`, which turns
DeepSeek's thinking off), or `reasoning_effort` when it is `None`; only the settings file sets it.
The OCR model gets no setting. A level (`low`, `medium`, `high`) raises the per-request deadline
to 10 minutes. Ollama cuts prompts past its context length (4k below 24 GB of VRAM); a request
(10.4) stays within 8k, and [cli.md](cli.md) starts Ollama with `OLLAMA_CONTEXT_LENGTH=8192`.

ASR stays local.

### 10.2 What the LLM detector does

With `detector = "llm"`, the judge (stage 8, `llm.py`) decides for every candidate whether it is
a real question, and replaces the candidates and pairs of stages 6 and 7 with the ones it keeps.

**Candidates** are every sentence ending in `?` plus the rules' own candidates (7.3). A sentence
both keep keeps the rules' `reasons` and `question_type`.

**Windows.** A candidate is sent with `judge_window` (10) sentences on each side. A candidate
within `judge_window` sentences of the previous one joins its request, up to `judge_batch_size`
(8) candidates; any other starts a new request.

**Request.** One line per sentence, candidates marked `>>`:

```
   s_4a01bc 1279s So we sort by the first key and then by the second.
>> s_1b88c0 1284s so why does a stable sort matter here
>> s_77e2d0 1301s [said 23x in this lecture] True?
```

The repeat count is shown from three repeats on. The prompt (`JUDGE_PROMPT_VERSION`) holds the
four kinds of [annotation_guidelines.md](annotation_guidelines.md) 2. The response is an object
keyed by candidate id, its shape fixed by a JSON schema sent as `response_format` (D13):

```json
{
  "s_1b88c0": {
    "reason": "Asks why stable sorting matters for sorting by several keys.",
    "kind": "content",
    "question": "Why does a stable sort matter for multi-key sorting?",
    "answer_sentence_ids": ["s_3f2a91", "s_7c04de"],
    "confidence": 0.82
  }
}
```

Only `content` is kept; other kinds are counted in `stats.judge_rejected`. Mapping to section 12:

- `confidence` is the question's confidence; the gate of 9.2 applies, the answer weighting does
  not.
- A non-empty `question` becomes the question's `text`, with verdict `revise` and the transcript
  sentence as `original_text`; an empty one keeps the sentence, verdict `keep`.
- The answer is the cited sentences' own text.
- The question gains the reason `llm`, its pair `llm_answer` or `no_answer_found`.

### 10.3 Constraints on the LLM detector

1. It judges only the candidates the code chose, by id; a judgement for anything else, or a
   malformed one, is dropped and counted in `stats.judge_dropped` (requirement 4.5).
2. An answer is later sentences of the excerpt. The schema lists those ids per candidate; any
   other cited id is dropped.
3. Output is checked three times: constrained decoding on the server, `json.loads` with one
   repeat of the request, then rules 1 and 2.
4. The schema requires every candidate id. Candidates a server skips are asked for once more;
   any still missing are counted in `stats.judge_skipped`.
5. Once asked for, it is required: an unreachable endpoint, a bad key, an unknown model, invalid
   JSON twice, or no answer within `timeout_seconds` (120) twice ends the run with one `error:`
   line. The deadline covers the whole request. Answered requests stay cached (10.4), so a re-run
   resumes.

### 10.4 Requests, cached per request

A lecture makes about 15 to 25 judge requests. `judge_parallel` (1) sends several at once; a
local server queues them, and queue time counts against the deadline.

Each request is cached under `.questxtract/<caller>/<sha256 of the request>.json`, keyed on the
endpoint, the model, the prompt, the schema and the content (transcript lines, a keyframe, or the
layout probe's frames). Any change to what the model would see asks again; nothing else does. A
`.json` transcript input is cached too. Only a valid answer is stored. An unreadable entry is
asked again; an unwritable cache only loses the saving.

`questions.json`'s `stats` records what the calls cost, per caller: `judge_*`, `layout_*` and
`vision_*`, each with `_requests` (repeats included), `_input_tokens` and `_output_tokens` (as the
API reports them, 0 when it reports none) and `_ms`. A run served from the cache shows 0 for all
four.

---

## 11. Orchestration: stages, caching, progress

### 11.1 The stage contract

Every stage is a function `Context -> Context`.

| Want to change | Seam | What it costs |
|---|---|---|
| Whisper checkpoint, VLM, judge model | a `Config` string | nothing, no code |
| Hosted endpoint to a local one | `base_url` (10.1) | nothing, no code |
| The ASR engine | an `AsrBackend` class (2.2) | one class |
| The implementation of a stage, or a new step | replace or insert one entry in `STAGES` | one function |

A stage is tagged with a decorator carrying its name and cache policy:

```python
@stage("transcribe", cache="transcript", uses=("model_size", "language", "beam_size", ...))
def transcribe(ctx: Context) -> Context: ...
```

`cache` names the one `Context` field the stage produces (cached as `<stage name>.json`; `None`
means always recompute), and `uses` lists the `Config` fields in the cache header (11.2).

`pipeline.STAGES` is the diagram of 1.3, and `run(ctx, stages)` walks it; `process(path, config)`
builds the `Context` and runs it. Passing another `stages=` list swaps, drops or adds stages.
`tests/test_pipeline.py` checks that `STAGES` matches 1.3.

**Four rules:**

1. Stages never import each other. They communicate only through `Context`.
2. Stages read config only through `ctx.config`; no environment reads outside `config.py`.
3. `layout`, `vision` and `judge` return `ctx` unchanged when disabled.
4. A cached stage produces exactly one `Context` field of JSON-serialisable dataclasses.
   `transcribe` is the one stage the runner caches; `vision` caches its views and requests itself
   (11.2), and `attach_visual` (an interval join, always recomputed) is a stage of its own.

**Config.** One flat frozen dataclass in `config.py` with every default. For `process`,
`transcribe`, `eval` and `quiz`, explicit flags override a flat TOML file, which overrides the
defaults. `--config PATH` selects the only file to read; otherwise `questxtract.toml` in the
working directory is read if present. `load_config` checks keys against the fields and values
against their annotations, and rejects secret-like keys. `Config.api_key` and
`Config.vision_api_key` read the key for their endpoint (10.1). `Config()` alone does not read
TOML.

`STAGES`, `run()`, `process()`, the cache and progress live in `pipeline.py`. The `stage`
decorator lives in `models.py` to avoid an import cycle.

### 11.2 One cache

```
.questxtract/
  <source_hash>/transcribe.json   the transcript
  <source_hash>/views.json        the stable views of the recording (6.4)
  <source_hash>/keyframes/        their keyframe images, named by content hash
  layout/<request hash>.json      one per layout probe request (6.2, 10.4)
  vision/<request hash>.json      one per keyframe request (10.4)
  judge/<request hash>.json       one per LLM detector request (10.4)
  quiz/<request hash>.json        one per quiz request (12.3)
```

A stage entry has a header with the values of its `uses` fields; the runner reuses it when the
header matches. `--from-scratch` bypasses every cache. Sentences, segments, detection, answers
and scoring always recompute. A run with an empty `ctx.source_hash` skips the stage cache.

The judge, the layout probe and `vision` declare no `cache=`: their input includes the transcript
or may fail, so they cache per request (10.4), and `vision` caches its views in `views.json`,
keyed on the content region and sampling settings.

A stage that reads a config field missing from `uses` serves stale results.

### 11.3 Progress (FR9)

```python
Progress = Callable[[str, float], None]
```

Stage name plus fraction complete. The CLI draws a terminal bar; the library default is `None`.
The runner emits one call per stage. Transcription reports `word.end / info.duration` per word
through the backend's `on_progress`, capped at 0.99. Vision reports sampling as its first half
and keyframe reading as its second.

---

## 12. Output

### 12.1 questions.json

Schema 2.4: a fixed key set, `null` included, plus a keyframe folder beside the file. Formal
description: [`questions.schema.json`](../src/questxtract/questions.schema.json), shipped in the
package; the tests validate every export shape against it. The example has vision and the LLM
detector on; with the rules and no vision, `visuals` is empty, `verdict` is `unjudged` and the
judge and vision settings are `null`:

```json
{
  "schema_version": "2.4",
  "source": {"path": "lecture.mp4", "duration_seconds": 4812.3, "hash": "9f2a..."},
  "config": {"model_size": "turbo", "language": "en", "min_confidence": 0.4,
             "detector": "llm", "judge_model": "Qwen/Qwen3.8-27B-FP8",
             "judge_prompt_version": "v5", "judge_reasoning_effort": "",
             "vision_model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
             "vision_reasoning_effort": "none", "vision_ocr_model": null},
  "stats": {"sentences": 1840, "sentences_unpunctuated": 12, "segments": 96,
            "segments_closed_gap": 92, "segments_closed_duration": 1, "segments_closed_chars": 3,
            "board_states": 34, "candidates": 61, "answered": 30, "emitted": 38},
  "segments": [
    {"id": 37, "start": 1262.0, "end": 1340.5, "topic": null, "visual_ids": ["v_4c1e09"]}
  ],
  "visuals": [
    {
      "id": "v_4c1e09",
      "start": 1240.0,
      "end": 1402.5,
      "keyframe_path": "lecture.keyframes/v_4c1e09.png",
      "description": "two-column comparison of stable and unstable sort passes",
      "text": "stable: (3,a) (3,b) preserved",
      "latex": null,
      "confidence": 0.9
    }
  ],
  "questions": [
    {
      "id": "q_0007",
      "text": "Why does a stable sort matter for multi-key sorting?",
      "original_text": "so why does a stable sort matter here",
      "start": 1284.2,
      "end": 1288.9,
      "start_hms": "21:24",
      "end_hms": "21:29",
      "confidence": 0.82,
      "confidence_label": "high",
      "question_type": "why",
      "source": "audio",
      "answer": {
        "text": "Because the second pass preserves the order of the first.",
        "start": 1291.0,
        "end": 1296.4,
        "sentence_ids": ["s_3f2a91", "s_7c04de"]
      },
      "segment_id": 37,
      "sentence_id": "s_1b88c0",
      "visual_ids": ["v_4c1e09"],
      "verdict": "revise",
      "reasons": ["mark", "wh", "llm", "llm_answer"],
      "review": {"status": "pending", "text": null, "note": null, "metadata": {}}
    }
  ]
}
```

`segments` and `visuals` (board states, 6.4) are top-level lists; a question points into them
with `segment_id` and `visual_ids`.

- `visual_ids` holds the board states on screen while the question was asked, falling back to
  every state joined to its segment; empty when vision is off. A segment's `visual_ids` holds
  every state joined to it. `segment_id` is `null` when the sentence belongs to no segment.
- `answer` is `null` when no answer was found. Its `start`/`end` span its own sentences.
- `original_text` is set when the LLM detector reworded the question, else `null`.
- `source` is `audio`, or `combined` when the board raised the score (`board_overlap`, 7.3).
  `visual` is reserved for a question read from the board alone.
- `topic` on a segment is `null`; nothing produces topics.
- `keyframe_path` is relative to the JSON file: writing it copies each keyframe from the cache
  into a folder named after it (`lecture.questions.json` gets
  `lecture.keyframes/<visual id>.<ext>`). With `-o -` nothing is copied and the path is relative
  to the cache directory.
- `config.min_confidence` is the threshold the file was cut at; `config.judge_model` and
  `config.vision_model` are `null` when that stage was off. `config.vision_ocr_model` names the
  OCR model when it read the keyframes (6.5), else `null`.
- `config.detector`, `judge_prompt_version` and `judge_reasoning_effort` are `rules` or `llm`,
  the judge prompt's version, and the `--llm-reasoning` level (`""` for `server`); the last two
  are `null` with the rules. `vision_reasoning_effort` is `null` with vision off. Other settings
  are not recorded.

**Versioning.** `schema_version` is `MAJOR.MINOR`. A minor version adds keys, or changes what a
value means without changing its type, so a reader must ignore unknown keys; a major version moves
or removes something, and `read_questions` refuses a file of another major version.

### 12.2 Editing (FR5)

Each question has one block that belongs to people, `review`. The pipeline writes its defaults
and never reads or overwrites it; everything else is regenerated on every run.

| `review` field | Meaning |
|---|---|
| `status` | `pending` (untouched), `accepted`, `deleted`, or `false_positive` (not a question at all) |
| `text` | The edited question text; `null` while unedited. The question's own `text` stays the original. |
| `note` | Free text from the reviewer. |
| `metadata` | Any extra key-value data, such as a topic tag or a quiz difficulty. |

A re-run does not keep the edits; copying `review` across is not planned (#15). Question and
sentence ids are content hashes (section 5), so a wrapper can carry edits across by id.

### 12.3 quiz.json

`questxtract quiz questions.json`, or `process --quiz` after `questions.json` is written
(`quiz.py`, not a stage). It is the only part that writes text that was not spoken
(requirement 4.5).

**Items.** `quiz_options` (3, from 3 to 5) options per item, one correct. A stem is `spoken` only
when it is the question as it was said; the rest is `generated`. Every item starts `pending`.

**Selection.** Rejected questions and check-ins are left out. `quiz_strictness` `lenient` (the
default) also takes unanswered questions, whose answer the model writes from what was said around
them; `strict` keeps only answered ones. The model's first field, `quiz_material`, can turn a
question down (addressed to the room, rhetorical, too vague); it is then left out.

**One request per question** to `quiz_model` through the shared endpoint (10.1), cached under
`.questxtract/quiz/`: the question, the lecture's answer, the sentences around it (four before,
two after, the question marked), and the `description`, `text` and `latex` of its linked board
states. `quiz` takes `--transcript` or reads the cached transcript by the recording's hash
(`pipeline.cached_transcript`). The code checks that every field is filled and every option
differs. A question that fails or stalls twice is left out with a warning; an unreachable
endpoint or a bad key ends the run.

**Deterministic.** Items follow `questions.json`, item ids derive from question ids, the correct
option's place from a hash of the question id, and nothing records the time of the run.

Schema 1.0, formally [`quiz.schema.json`](../src/questxtract/quiz.schema.json), shipped in the
package (D15). Items are single choice: at least three options, exactly one correct, a `solution`
with an explanation and one `option_explanations` entry per option. Every text is `{text,
origin}`. `provenance` ties an item to its source question (`source_question_id`, `start`, `end`),
and the item id is `"quiz_"` plus that id. Every item has a `review` block (`pending`, `accepted`,
`rejected`). No run timestamps, no absolute paths.

`validate_quiz` (`quiz_schema.py`) checks what the schema cannot: the correct option exists,
explanations match the options one to one, option and item ids are unique, item ids are derived,
`end >= start`, and, given the source `questions.json`, the source question exists and a `spoken`
stem is the question as it was said (`original_text` when the judge reworded it, else `text`).
`read_quiz` refuses another major version, then validates. Every written file passes
`validate_quiz`. `jsonschema` is used in the tests only.

### 12.4 QTI packages

`questxtract qti quiz.json -o quiz.qti.zip` turns a quiz into a 1EdTech QTI content package: QTI
3.0 by default, or QTI 2.1 with `--qti-version 2.1` (`export_qti(..., version="2.1")`) for an LMS
that does not import 3.0, such as Canvas New Quizzes (D16, D23). Both hold the same items,
scoring, feedback and instructions. The file is read through `read_quiz`; `qti.py` checks every
field it uses and names its path.

- `qti.py`: the version-independent model and the mapping from `quiz.json`.
- `qti_v3.py`, `qti_v2.py`: one writer per version, producing the items, the test and the
  manifest in ZIP order.
- `qti_package.py`: `export_qti` checks the version, dispatches to that version's writer through
  a table, adds the JSON sidecar and writes the ZIP.

```text
items/<item id>.xml            one assessment item per quiz item, in document order
assessmentTest.xml             one linear test part and section referencing every item
imsmanifest.xml                the content package manifest
metadata/questxtract.json      provenance, review status and the origin of every text
```

Element names below are QTI 3.0's; the QTI 2.1 bullet gives 2.1's spelling.

- Selection: every item except those whose `review.status` is `rejected`.
- Items: a `qti-choice-interaction` with `max-choices="1"` and one `qti-simple-choice` per
  option, identified by the quiz's item and option ids. `RESPONSE` is a single identifier whose
  correct response is the correct option. Any other `interaction_type`, `NaN` or an infinite
  number fails with its path.
- Scoring: `SCORE` (float, default `0.0`) and `MAXSCORE` (the item's points). With points
  above zero, `SCORE` carries `normal-maximum` and `normal-minimum`. Explicit response processing
  from the Core subset: a match sets `SCORE` to `MAXSCORE`, otherwise `0.0`.
- Feedback: response processing sets `FEEDBACK` to the chosen option. Each option has a
  `qti-modal-feedback` with its explanation, then the general one, both from `solution`.
- Instructions: `show_feedback` becomes `show-feedback` on the test part's
  `qti-item-session-control`, `shuffle_options` the interaction's `shuffle`,
  `shuffle_questions` a `qti-ordering shuffle="true"` in the section, and `text` a candidate
  `qti-rubric-block`.
- XML, QTI 3.0: items and the test use the default namespace `imsqtiasi_v3p0` with an
  `xsi:schemaLocation` to the 3.0.1 XSD, no prefixes. The manifest uses the QTI 3.0
  content-package namespace and `schemaversion` `3.0.0`. Every file but the manifest is a
  resource; the JSON sidecar is `type="extension"`.
- XML, QTI 2.1: camelCase elements and attributes without the `qti-` prefix
  (`assessmentItem`, `choiceInteraction`, `maxChoices`, `showHide`). `modalFeedback` and
  `rubricBlock` hold their paragraphs directly. Default namespace `imsqti_v2p1` with an
  `xsi:schemaLocation` to the ASI 2.1.2 XSD. The manifest follows the QTI 2.1 profile of content
  packaging 1.2: namespace `imscp_v1p1`, `schema` `QTIv2.1 Package`, `schemaversion` `1.0.0`,
  resource types `imsqti_test_xmlv2p1` and `imsqti_item_xmlv2p1`; the sidecar is `webcontent`
  with `format` `questxtract-qti-2.1` (3.0 writes `questxtract-qti-3.0`). `responseProcessing`
  also names the `match_correct` template beside the rules (information model 8.1, 8.2); an LMS
  that supports only templates scores 1 and shows no per-option feedback, so a 2.1 export warns
  about items worth other than 1 point. No 2.1 package has been imported into a real LMS yet.
- Validation: the tests validate every package against the official XSDs of its version and
  check what an XSD cannot: the correct response and every feedback name an option, every `href`
  and dependency resolves, identifiers are unique, and a 2.1 package holds what the 3.0 package
  of the same quiz holds. `scripts/fetch_qti_xsd.py` downloads both schema trees (17 files for
  3.0, 37 for 2.1, pinned by SHA-256) into `.cache/qti-xsd/`; the 2.1 imports named over `http`
  are fetched over `https`. Without that copy the tests are skipped, except in CI, where they
  fail. The package does not depend on the XSDs.
- Determinism: entries are written in a fixed order with the date 1980-01-01, deflate, Unix
  permissions `0644` and no clock or platform data.

### 12.5 ExerPlaza TeX

`questxtract tex quiz.json -o quiz.tex --content-unit NAME` turns a quiz into the
`IndexedQuestion` format that ExerPlaza imports at `/questions/upload/tex`, as read by its parser
`exerplaza/backend/quiz/latex_parser.py` v2.1 (D24). A teacher uploads the file; `tex` does not
connect to ExerPlaza. There is no preamble.

Texts are written without LaTeX escaping, since ExerPlaza shows them with MathJax (`$...$`,
`processEscapes`).

**Mapping.** UTF-8 without a BOM, LF line endings. A header of four comment lines (quiz id, schema
version, title), then one block per item, each starting with `% questxtract: <quiz id>/<item id>`.

| TeX | From |
|---|---|
| `\QuestionContentUnits` | every `--content-unit`, in order, the same for every item |
| `\QuestionType` | always `multiple choice` |
| `\QuestionLanguage` | the language's primary subtag: `en` is `english`, `it` is `italian`; any other fails |
| `\QuestionDisclosability`, `\SolutionDisclosability` | `review.status`: `pending` is `only me`, `accepted` is `everybody` |
| `\QuestionBody` | the stem, CRLF as LF |
| `\QuestionPotentialAnswers` | one `\correctanswer text` or `\answer text` line per option, in order, whitespace collapsed |
| `\QuestionNotesForTheTeachers` | the reference, the source question, its times (`m:ss`, `h:mm:ss` from one hour on), the stem's origin, and the model and prompt version (the method when there is no model) |
| `\SolutionBody` | the explanation, a blank line, then `- <option>[ (correct)] — <explanation>` per option |

Points, instructions, the description, the source and the review note are not written. ExerPlaza
parses `\QuestionFeedbackForTheStudents` but does not save it, so option explanations go into
`\SolutionBody`.

**Guarantees.** The parser reads these mistakes wrong without reporting them, so the writer fails,
naming the text's path (`questions[1].options[2].text: unbalanced braces`), and writes nothing:

- answers are `\answer text` with a space and no braces, one per line;
- braces are balanced in every text, counting those of `\{` and `\}`;
- no text holds a command of the format (`\begin{Indexed`, `\end{Indexed`, `\Question`,
  `\Solution`, `\answer`, `\correctanswer`, `\penalizableanswer`); the title only gets this check;
- exactly one `\correctanswer` per block;
- `IndexedSolution` sits inside `IndexedQuestion`, after `\QuestionContentUnits`.

A quiz with no item left fails. A content unit must be non-empty, on one line and free of braces.

`tex.py` takes its items from `quiz_to_qti` (12.4), so selection, order and field checks match
QTI. Its errors are re-raised as `TexExportError`. The review status, the stem's origin and
`generation` come from the model's untyped `metadata`, and `tex.py` checks each value it reads.

### 12.6 Direct export to ExerPlaza

`questxtract exerplaza quiz.json --url URL --course-session ID --content-unit NAME` sends a quiz
(section 12.3) to ExerPlaza's import endpoint, so it arrives as a quiz in a course session with no
file in between. `tex` (12.5) stays the route that needs no change on ExerPlaza's side.
`exerplaza.py` is the only module that talks to ExerPlaza.

**Login.** ExerPlaza authenticates with Flask sessions only, so `exerplaza` logs in through its
web form as a browser does. `GET /login` gives the form's `csrf_token` and the session cookie it
belongs to; `POST /login` sends them with the email and password from
`QUESTXTRACT_EXERPLAZA_EMAIL` and `QUESTXTRACT_EXERPLAZA_PASSWORD`, read from the environment or
`.env` by `config.py`, like the API keys. A redirect anywhere but `/login` means logged in, back to
`/login` means refused, and the form again means the form was refused. ExerPlaza checks CSRF per
form, not for the whole site, so the import needs only the session cookie, which lives in memory
for that one call. The login depends on the form's field names (`email`, `password`,
`csrf_token`); `tests/test_exerplaza.py` holds a copy of the form as ExerPlaza sends it.

**Import.** One `POST /api/v1/quizzes/import?course_session=<id>&content_unit=<name>...`, whose
body is the `quiz.json` without its rejected items and is still a valid `quiz.json`. The items
are those of `quiz_to_qti` (12.4), so `qti`, `tex` and `exerplaza` send the same ones. ExerPlaza
saves the quiz in one transaction, with no change to its database schema: one multiple-choice
question per item, with the disclosability, solution and notes for the teachers that `tex` writes
(12.5) and `created_by_llm` set when any text of the item is `generated`, in the course session's
quiz with the quiz's `title`, or in a new one (two such quizzes are a 409). A pending item's
question, visible only to the teacher, stays out of the quiz, since ExerPlaza shows every
question of a quiz to whoever opens it; an import that sends the item accepted adds it. It
recognises a re-import by text, as its TeX upload does: an item whose stem, in the quiz's
language, is already the text of a question of that quiz, of a question the teacher is an author
of, or of an earlier item, is `skipped` with a reason, and that question is in the quiz unchanged
unless the item is pending. It answers `created` or `skipped` per item, with `<quiz id>/<item id>`
as the item's `ref` and a `reason` for a skipped item or a pending one left out of the quiz
(`pending: not added to the quiz`); `exerplaza` prints the reasons of skipped items only. The
contract keeps `updated` for an import that updates, which this one never does. Nothing is
updated or deleted, so students' answers never change, and an item edited after its import is
changed in ExerPlaza, not by sending it again. An error is an RFC 9457 problem whose `errors`
list names each item's path, printed as ExerPlaza sent it.

**Safety.** Every check that needs no network comes first: the URL, the course session, the
content units, `read_quiz` and `quiz_to_qti`, then the credentials. Plain `http` is refused except
for `localhost`, `127.0.0.1` and `::1`, because the password goes with the login, and an address
with credentials, a query or a fragment is refused. A local ExerPlaza is never reached through a
proxy from the environment, so the login stays on the machine. Redirects are never followed. Each
request times out after 30 s. The password is never printed, logged or stored, and the server's text is
printed with its control characters replaced.

The contract (requests, answer, errors, mapping) is in
[issue #37](https://github.com/mad201802/QuestXtract/issues/37). `tests/test_exerplaza.py` holds
the example request and answer, and ExerPlaza's tests hold a copy. The endpoint is ExerPlaza's;
until a server has it, `exerplaza` fails with HTTP 404 there and `tex` is the way in.

### 12.7 The questions as a LaTeX document

`questxtract questions-tex questions.json` writes the questions of a `questions.json` (12.1) as a
standalone `article` that compiles with pdfLaTeX, for reading or printing; it is a view of the
file, not a format anything reads back. One entry per question, in document order: its time, its
confidence, the question as reviewed (`review.text`, else `text`), the spoken wording when that
differs (`original_text`, or `text` under a teacher's edit), the lecturer's answer with its time,
and the text and description of the board states it points to. Rejected questions are left out,
as in the quiz exports; no question left is an error.

Every text is escaped. Characters beyond Latin-1, common on a math board, become LaTeX commands
where one exists (Greek letters, arrows, relations, subscripts, circled numbers) and `?`
otherwise, so `inputenc` never meets one it cannot set. The board's `latex` field goes into a
verbatim block, not typeset: a formula the vision model got wrong cannot break the compile. The
same file gives the same bytes. The app's Questions (.tex) box runs this command after a
Questions run.

---

## 13. Module layout and dependencies

### 13.1 Layout

```
src/questxtract/
  __init__.py      public API: process(), run(), STAGES, read/write, filters,
                   export_qti() for QTI 2.1 and 3.0, export_tex(), export_exerplaza()
  models.py        Word, AsrSegment, Transcript, Sentence, Segment, VisualContext, Layout,
                   QuestionCandidate, QAPair, Context, SchemaError, the @stage decorator
  config.py        one flat frozen Config dataclass, every default
  media.py         1 ingest: probe, source hash, audio extraction
  asr.py           the AsrBackend protocol and its registry (15.3)
  transcribe.py    2 transcribe, through the AsrBackend registry in asr.py
  sentences.py     3 sentences: word walk, abbreviation guard, force-split
  segments.py      4 segments: structural segmentation
  vision.py        5 layout, the probe; vision: sampling, views, captioning; attach_visual,
                   the interval join
  detect.py        6 detect: extraction, vetoes, scoring, cue tables
  answer.py        7 answer: adjacent seed
  llm.py           8 judge: the LLM detector
  endpoint.py      the OpenAI-compatible client and requests, shared by judge, vision and quiz
  score.py         9 score: confidence, threshold
  export.py        questions.json writer and reader, with questions.schema.json
  quiz_schema.py   quiz.json reader and validator, with quiz.schema.json
  qti.py           QTI model and the quiz.json mapping
  qti_package.py   export_qti(): version dispatch, JSON sidecar, deterministic ZIP
  qti_v2.py        QTI 2.1 XML
  qti_v3.py        QTI 3.0 XML
  tex.py           ExerPlaza TeX from the QTI model
  exerplaza.py     export_exerplaza(): ExerPlaza's login form, then quiz.json to its import
  questions_tex.py export_questions_tex(): questions.json as a LaTeX document that compiles
  filtering.py     time and text filters, experimental topic filter
  evaluate.py      labelled-set metrics
  schema.py        Question, a standalone shape the pipeline does not produce
  pipeline.py      STAGES, run(), process(), cache, progress
  quiz.py          quiz.json from questions.json: selection, one request per question
  cli.py           process / transcribe / filter / eval / quiz / qti / tex / exerplaza /
                   questions-tex / gui
  gui/             the graphical app, a run launcher over cli.execute (14):
    options.py     one table of the CLI options the app shows, checked against the CLI
    run.py         form values -> the CLI's argv -> cli.execute
    state.py       what the form holds, what Run needs, progress fractions (no tkinter)
    keys.py        the NTNU key: test, save, remove, where it is read from (no tkinter)
    key_dialog.py  the key window: a masked field, Test, Save, Remove
    window.py      the tkinter window, classic ttk look; drag and drop via tkinterdnd2
    i18n.py        the app's words in English, Norsk and Italiano

tests/            flat, one test module per source module, plus conftest.py
```

Tests sit flat beside the modules they cover; pytest runs with `--import-mode=importlib`.
`tests/data/` holds the labelled set (4.1).

Each stage module exposes only its `@stage`-decorated functions as its public surface (one each,
`vision.py` three: `layout`, `vision`, `attach_visual`). `tests/test_pipeline.py` checks that
`grep '@stage'` over the stage modules returns the contents of `STAGES`.

### 13.2 Dependencies

**Base** (`pyproject.toml`): `av` (container probe and audio extraction), `faster-whisper` (ASR),
`numpy` and `onnxruntime`, plus `tomli` on Python 3.10. `ctranslate2`, `tokenizers`,
`huggingface-hub` and `tqdm` come through faster-whisper. No system binaries, no torch.

**Extras:** `[llm]` -> `openai` (the LLM detector, vision and the quiz). `[gui]` -> `tkinterdnd2`
(drag and drop; the window is tkinter from the standard library) and `openai`, since the window
starts with the LLM detector and vision on. The ExerPlaza export speaks HTTP with `urllib` from the standard library.

A neural sentence splitter (`wtpsplit-lite`, 3.1) would add 428 MB of weights and no new
dependency (D29).

### 13.3 Status

What the code does today, and the open issue for each missing piece.

| Module | Stage | State | Issue |
|---|---|---|---|
| `models.py`, `config.py`, `pipeline.py` | the contract itself | done | |
| `media.py` | 1 ingest | done: probe, source hash, audio extraction to a temporary WAV | |
| `transcribe.py`, `asr.py` | 2 transcribe | done: faster-whisper behind the `AsrBackend` registry; the vision-informed second pass (2.4) is not planned | |
| `sentences.py` | 3 sentences | done | |
| `segments.py` | 4 segments | done | |
| `vision.py` | 5 layout, vision, attach_visual | done: the layout probe (6.2), frame sampling (6.3), stable views with one keyframe each (6.4, static), captioning with prompt v2 (6.5), with the previous board state unless the board is new, or the OCR reader (6.5), reusing the reading of a page that comes back (6.5) and the interval join; the ink curve (6.4) and de-occlusion (6.3) are not planned | |
| `detect.py` | 6 detect | done (7.1 to 7.3); implicit cues and missing-`?` recovery wait for a labelled miss (7.1) | |
| `answer.py` | 7 answer | adjacent seed done; answers further away come from the LLM detector | |
| `llm.py` | 8 judge | done: the LLM detector (10) | |
| `endpoint.py` | not a stage | done: client and requests for the judge, vision and the quiz (10) | |
| `quiz.py` | not a stage | done: quiz items from questions.json (12.3), checked by `quiz_schema.py` | |
| `score.py` | 9 score | done: confidence and threshold | |
| `export.py` | 10 export | done; keeping review edits across re-runs is not planned | |
| `quiz_schema.py` | not a stage | done: the `quiz.json` reader and validator (12.3); `quiz.py` writes it (12.3) |  |
| `filtering.py` | not a stage | time and text filters done; `topic=` is experimental, no stage produces topics | |
| `evaluate.py` | not a stage | done; NFR2 gated in pytest; the vision model scored with `eval --score-vision` (4.4) | |
| `qti.py`, `qti_package.py`, `qti_v2.py`, `qti_v3.py` | not a stage | done: the QTI 3.0 and 2.1 packages of a `quiz.json` (12.4) | |
| `tex.py` | not a stage | done: the ExerPlaza TeX of a `quiz.json` (12.5) | |
| `questions_tex.py` | not a stage | done: a `questions.json` as a LaTeX document (12.7) | |
| `exerplaza.py` | not a stage | done: a `quiz.json` sent to ExerPlaza's import endpoint (12.6); the endpoint itself is ExerPlaza's work | |
| `cli.py` | not a stage | `process` (with `--quiz`), `transcribe`, `filter`, `eval`, `quiz`, `qti`, `tex` and `exerplaza` work ([cli.md](cli.md)); `process --vision --detector llm --quiz` runs every stage and the quiz in one run, tested offline | |
| `gui/` | not a stage | done: the window, its run launcher and the API key window (`questxtract gui`, 13.1); QTI and TeX run from a recording to the export file in one run; it runs the CLI's code and does not review or edit questions (14) | |

What lands when: [GitHub milestones](https://github.com/mad201802/QuestXtract/milestones).

---

## 14. Non-goals

- The teacher review/edit UI is a separate application; this package's part is that its JSON
  round-trips (12.2). The graphical app (13.1) only starts runs and shows the file written.
- ExerPlaza's side: `tex` writes a file for ExerPlaza's TeX upload (12.5) and `exerplaza`
  sends a quiz to its import endpoint (12.6). The endpoint, the questions once saved and their
  review belong to ExerPlaza: `questxtract` deletes nothing there and uses no page of ExerPlaza's
  but the login form.
- No LMS upload; `qti` only writes a ZIP (12.4).
- The labelled dataset is test material under `tests/data/`, not a subpackage; the harness
  that reads it is `evaluate.py` (section 4).
- YouTube URLs: the CLI takes a local file.
- Speaker identification by name and diarization (2.6).
- Manim animations of the board are outside the core delivery
  ([design_decisions.md](design_decisions.md), D21).

## 15. Python API

```python
from pathlib import Path

from questxtract import STAGES, Config, Context, Transcript, run, write_questions

ctx = Context(
    source=Path("lecture.mp4"),
    config=Config(min_confidence=0.4),
    transcript=Transcript.from_file("lecture.transcript.json"),  # or let stage 2 produce it
)
ctx = run(ctx, STAGES[2:])  # skip ingest + transcribe: the transcript is already in hand

for pair in ctx.pairs:
    print(f"[{pair.start:7.2f}s] ({pair.confidence:.0%}) {pair.question.text}")

write_questions(ctx, "questions.json")
```

Pass your own `stages=` list to swap the detector, drop a stage or add a step:

```python
run(ctx, [my_detector if s.__name__ == "detect" else s for s in STAGES[2:]])
```

### 15.1 Filtering questions

`filter_questions` narrows a finished `Context`, `filter_document` a `questions.json` document.
Both return a new value:

```python
from questxtract import filter_document, filter_questions, process, read_questions

ctx = process("lecture.mp4")
window = filter_questions(ctx, start=300, end=600)  # new Context, ctx untouched

data = read_questions("lecture.questions.json")
window = filter_document(data, start=300, end=600)  # new dict, same schema
board = filter_document(data, text="eigenvalue")  # said, or on the board on screen
```

A question is kept when `start <= question.start < end`, in seconds; leave out a bound to open
that side. Only the question list shrinks, so `segment_id` and `visual_ids` still resolve. An
invalid range raises `ValueError`. `text=` keeps a question when every word appears, as a whole
word and ignoring case, in the question, its answer, its `review` text, or a board state in its
`visual_ids` (as `questxtract filter`, [cli.md](cli.md)). `topic=` is experimental: no stage
produces topics, so it warns and matches nothing.

### 15.2 Reading a quiz.json

`read_quiz` reads a `quiz.json` ([schema](../src/questxtract/quiz.schema.json)) and checks the
rules its schema cannot express; `validate_quiz` does the same for a document in memory. Pass the
source `questions.json` to also check that each item's question exists and that a `spoken` stem
is the question as it was said:

```python
from questxtract import read_questions, read_quiz

quiz = read_quiz("lecture.quiz.json", read_questions("lecture.questions.json"))
```

Every problem is listed in one `SchemaError`, one line each, with its path.

### 15.3 Swapping the ASR engine

The ASR engine is a backend picked by `Config.asr_backend`. A second engine is one class:

```python
from questxtract import AsrBackend, Config, register_backend


@register_backend("my_engine")
class MyBackend:
    def transcribe(self, source, config, on_progress=None):
        # return raw segments: [{"text": ..., "start": ..., "end": ...}, ...]
        # on_progress(fraction) must be accepted; calling it is optional.
        ...


ctx = Context(source=Path("lecture.mp4"), config=Config(asr_backend="my_engine"))
```

`faster_whisper` is the default, registered the same way in `asr.py`.

# The QuestXtract Pipeline

How QuestXtract turns a lecture recording into a set of questions with timestamps, answers and
provenance.

Related: [general_requirements.md](general_requirements.md) for the requirement ids (FR/NFR)
referenced throughout. The audio-only prototype this package replaces is
[lecture-qa-extraction](https://github.com/matteopogetta/lecture-qa-extraction/tree/985c6f11594d0863274e13e2002b59bf17ad1c7f)
(commit `985c6f1`); its own pipeline document lives there.

---

## 1. Overview

### 1.1 What the pipeline does

In: a lecture video. Out: a JSON file of the questions that were actually asked, each one with a
timestamp, a confidence, an answer span where the lecture contains one, and enough provenance to
jump back to the exact seconds in the recording.

The package's job stops at files: that JSON, a quiz made from it (section 12.3) and an LMS
package of that quiz (section 12.4).
Everything downstream - the editor, the quiz UI, the upload and the rest of the LMS glue - lives
in a wrapper around it. That boundary is why the package may reach the network at all
(section 10): the caller decides whether to configure an endpoint, not us.

The central difficulty, which every design decision below follows from: **a lecture is not a
dialogue corpus.** Most interrogative-sounding utterances are not questions we want. They are
rhetorical framing ("what happens if we double this?"), classroom check-ins ("can everyone hear
me?"), polls ("one or two?") and discourse tags ("right?"). Of the real questions, many are
answered several sentences later by the same speaker after a digression. And the input is ASR
output: unreliable sentence boundaries, unreliable punctuation, no speaker labels.

So the pipeline spends most of its effort on two things: rebuilding timed sentence structure
before detection, and rejecting false positives after it.

### 1.2 The reference recordings

The design is calibrated against deliberately different lectures. None is grading material;
all are what we develop and measure against.

| Recording | Regime | What it stresses |
|---|---|---|
| [MIT 18.100B Real Analysis, L1](https://www.youtube.com/watch?v=BWvjy0a1Ffg) | chalk, sliding blackboards, dense LaTeX math, **camera pans and zooms to follow the lecturer**, few audible student questions | vision, LaTeX, ASR on math jargon |
| [TTK4225, 2026-09-14, P03·12 — impulse responses and BIBO stability](https://www.youtube.com/watch?v=ufmTACqiE78) | digital tablet notes (OneNote/PDF sheet), single instructor, control-theory derivations with heavy Socratic back-and-forth | question detection on implicit-cue and self-answered questions, ASR on control-theory notation |
| [TTK4225, 2026-09-15, P03·14 — sinusoidal fidelity](https://www.youtube.com/watch?v=IQa2TSykWdc) | digital tablet notes (OneNote/PDF sheet), same course, frequent student Q&A | echo pairs (student question + instructor restatement), habitual confirmation-tag handling ("...right?"/"...true?") |

Three consequences that show up repeatedly below:

- **There is no single video layout.** Merged single-camera, separate speaker plus screenshare,
  blackboard-only, digital tablet notes, picture-in-picture. Section 6 uses a VLM layout probe
  rather than a classifier.
- **The TTK4225 instructor uses a habitual confirmation tag** ("...right?", "...true?") several
  times a minute; `docs/annotation_guidelines.md` section 2 says how it is labelled.
- **In a classroom the students are often inaudible.** What reaches the audio is the lecturer
  repeating the question, usually with its `?`. An off-mic student question is not in the
  transcript at all, so no text rule can find it.

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

Every other stage is pure Python plus one local model. Stages 5 and 8 are the only ones that touch
an LLM endpoint, and both are off unless asked for. Expensive stages are cached and report
progress (section 11).

The `STAGES` list in `pipeline.py` is this sequence, rows 1 to 9, with vision split into `layout`,
`vision` and `attach_visual` (11.1). Row 10 is not a stage: export runs on the finished `Context`.

### 1.4 The rule path and the judge

Detection, the adjacent answer seed and the confidence are cheap, inspectable rules, and the
default. **The LLM detector** (the judge, section 10) is the alternative a user picks per run with
`--detector llm`: a text LLM reads the transcript, picks the questions actually asked, finds the
sentences that answer them and rewrites a messy question into a clean one. The rule path is the
offline one: a run with no endpoint still produces questions, answered only where the adjacent
seed found one.

**Hard constraint (requirement 4.5, "the model shall not create questions on their own").** The
LLM detector answers with sentence ids, and every id is checked against the transcript it was
sent. It may reword a question it found, with the spoken version kept beside it, but never add
one. Quiz generation is generative and lives in a separate command (12.3), so the boundary is
visible in the code and in the output, not only in a prompt.

### 1.5 Thresholds are measured, not assumed

Every threshold in sections 7, 8 and 9 is a starting value, swept by the evaluation harness
(section 4) against the labelled set. A new rule lands when a labelled failure demands it, and the
commit says which.

---

## 2. Front end: media to timed words

### 2.1 Ingest

Validate the path, open the container with `av.open()` for duration and stream layout, and
compute a content hash (`sha256` of the first and last 8 MiB plus the file size - full hashing a
2 GB lecture on every run is wasted I/O). The hash is the cache key for every later stage.

PyAV bundles the FFmpeg libraries, so this is not a subprocess and there is nothing for the user
to install. That matters for NFR1 (Linux, macOS **and Windows**): "install ffmpeg and put it on
PATH" is the single most common install failure in this class of tool, and we do not have that
step.

An `.mp4` or an `.mp3` is accepted, and what the file holds decides the rest: a recording without
a video track (any MP3, or an audio-only MP4) is transcribed as usual, and the vision stage finds
no board to read in it. There is no separate audio-extraction stage: the faster-whisper
backend (`asr.py`) calls `media.extract_audio`, which decodes the first audio track with PyAV into
a temporary 16 kHz mono WAV and deletes it when transcription ends. The WAV never reaches the
cache, and there is no system binary in the dependency list.

One recording, one clock. No multi-input sessions, no session-absolute clock.

> **Decision.** `questxtract/media.py`. Config: none.

### 2.2 Transcription

`faster-whisper`, batched, on the WAV from 2.1, behind the `AsrBackend` interface (below).

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

Three settings are load-bearing:

- **`vad_filter=True`**: Whisper hallucinates confidently over silence, and a lecture is full of it.
- **`word_timestamps=True`**: the only source of timing in the pipeline (2.3), and what the
  per-word progress reporting (11.3) rides on.
- **`initial_prompt`** biases Whisper toward punctuation, question marks included, the strongest
  detection signal we have (7.3). It is a weak lever: it conditions the first decoding window and
  is dropped whenever temperature fallback trips, which happens on exactly the noisy stretches.
  For punctuation quality, model size comes first, then the prompt, then a neural splitter (3.1).

**Model size.** Default `turbo` (`large-v3-turbo`, 809M parameters, roughly 8x faster than
`large-v3` at close to `large-v2` quality), usable on an `int8` CPU. `base` for quick runs,
`small` when memory is tight.

**Swapping the engine.** The ASR *checkpoint* is a config string (`model_size`). The ASR *engine*
(a hosted endpoint, WhisperX, Parakeet) is a class implementing `AsrBackend`
(`transcribe(source, config, on_progress)`), registered by name with `@register_backend` in
`asr.py` and selected by `Config.asr_backend`. `transcribe.py` never imports a specific engine.

**Dev workflow.** No dev machine has a GPU, and `turbo` on a CPU takes about as long as the
recording. The reference transcripts are committed in `tests/data/`, so work on sections 3 to 9
never runs Whisper.

> **Decision.** `questxtract/transcribe.py`, `questxtract/asr.py`. Config: `asr_backend`
> (`"faster_whisper"`), `model_size` (`"turbo"`), `language` (`"en"`), `beam_size` (5),
> `vad_filter` (True), `compute_type` (`"int8"`), `batch_size` (4: faster-whisper yields results
> per finished batch, so a smaller batch gives a smoother progress bar for about 9% more time),
> `initial_prompt`. Each `AsrSegment` keeps `avg_logprob`, `no_speech_prob` and
> `compression_ratio`, the cheapest hallucination signals; nothing reads them yet.

### 2.3 Timing: word timestamps, no forced alignment

Whisper **word** timestamps are the single source of timing, requested unconditionally in stage 2.
There is no alignment stage, because forced alignment is not more accurate *enough* to earn its
dependency. The error budget of the thing the timings are actually for - joining a speech interval
to a board-state interval (section 6):

| Side | Precision |
|---|---|
| Video, sampled at 1 fps | ±500-1000 ms, and "when was this written" is inherently fuzzy - a line takes ~20 s to write |
| Speech, Whisper word timestamps (cross-attention DTW) | ~±100-200 ms |
| Speech, WhisperX wav2vec2 forced alignment | ~±30-50 ms |

The video side is three to six times coarser. Tightening the audio side from 200 ms to 50 ms buys
nothing any consumer can use. If the A/V merge ever measures badly, the fix is sampling video at
2-4 fps, not aligning audio harder.

What *would* wreck the merge is different, and is already handled: Whisper **segment** timestamps
(as opposed to word timestamps) drift by multiple seconds, and hallucinated segments over silence
invent time ranges outright. `word_timestamps=True` plus `vad_filter=True` kill both.

Dropping alignment also keeps the install small: `whisperx` pulls torch and the CUDA stack,
`faster-whisper` runs on standalone C++ CTranslate2. **There is no torch in the base install**
(13.2).

### 2.4 Optional: vision-informed second ASR pass (not planned)

Whisper mangles domain vocabulary (*supremum*, *Cauchy*, *epsilon*), and the board has those words
written on it. The idea: collect terms from the board states (section 6) and re-transcribe with
faster-whisper's `hotwords`. Not planned (#62): it is worth building only once the eval harness
shows vocabulary errors costing questions.

### 2.5 Words are the atomic unit

There is no utterance layer. A sentence owns a half-open word index range, so its timings are
`words[i].start` and `words[j-1].end`, both real measurements, and its provenance is the range
itself. A timing is never interpolated, and a partition that is a slice cannot silently stop being
a partition, so no provenance validator is needed.

### 2.6 One speaker, no diarization

Speaker identification by name is out of scope (requirements, 1. Purpose & Scope) and a single
main speaker is assumed (4.2). There is no diarization, no speaker attribution and no acoustic
speaker check. In a classroom the students are often off-mic, so there is frequently no second
voice to diarize: the lecturer's repetition of the question is the signal, and that is a lexical
problem, not an acoustic one.

---

## 3. Sentences and segments

### 3.1 Sentences

ASR segments are not sentences. Whisper emits chunks bounded by its decoding window and by
silence, so one chunk routinely holds two questions or half of one. Detection needs units that
read on their own.

A `turbo`-sized Whisper punctuates well, so one walk over the word list asks each word whether it
ends a sentence: a split on terminal punctuation, with the word-index snapping built in.

**Not every full stop is a boundary.** `_is_terminal` holds back three cases the plain
`[.!?]` test gets wrong, all common in lecture ASR:

| Case | Example | Why |
|---|---|---|
| abbreviation | `So Dr. Rudin defines it.` | a short suppression list; `Step 1.` still splits, digits are not abbreviations |
| single-letter initial | `for all n greater than capital N.` | the MIT 18.100B regime is nothing but these; only a capital counts, a lower-case letter (`u.`) is a variable ending its sentence |
| ellipsis | `we take the limit... of the sequence` | a trailing-off, not a stop |

It is a suppression list rather than a classifier because the two error directions are not
symmetric: a missed boundary merges two sentences, which detection still reads, while a false
boundary leaves two fragments that read as neither. Merging is the safer error for a stage whose
job is standalone units.

**Run-ons.** A span that both ends in real terminal punctuation and fits in `max_sentence_words`
(60) is a sentence and is left alone. Anything else is a run-on the ASR failed to punctuate, and is
split at every silence of at least `force_split_min_gap_seconds` (0.6), then capped: a piece still
over the limit is halved at its widest gap, ties broken toward the middle, until it fits.

Both halves of that rule are load-bearing. The gate protects the common case - lecturers pause
mid-sentence at the board constantly, so silence alone must not be a boundary. The cap is what
makes `max_sentence_words` a cap rather than a hint: without it a gapless run of 200 words stays
one "sentence", detection vetoes it on `max_question_chars`, and every question inside it is lost
with no reason code to show for it.

> **Decision.** `questxtract/sentences.py`. Config: `max_sentence_words` (60),
> `force_split_min_gap_seconds` (0.6, the pause threshold generally).
>
> **The bet:** this relies on Whisper emitting question marks. The eval harness measures it as
> `question_mark_rate` (4.2); on an unlabelled recording, `stats["sentences_unpunctuated"]` is the
> proxy, and a count that climbs means the splitter is guessing from silence. The fix, in order of
> cost: a better `initial_prompt`, a larger model, then a neural splitter (`wtpsplit-lite`: no new
> dependency, but 428 MB of weights, 13.2).

### 3.2 Segments

A segment is a contiguous run of sentences bounded by silence and by length. **Not a topic**:
nothing here measures what the run is about.

One mode, structural, closing on the first of:

- a silence gap over `segment_max_gap_seconds` (2.0 s), strictly: a pause of exactly 2.0 s does
  not break
- a sentence that would push the run past `segment_max_duration_seconds` (90 s)
- a sentence that would push it past `segment_max_chars` (800)

Both length rules count the candidate sentence in, so a closed segment is inside its budget,
except a single sentence already over on its own, which 3.1 bounds.

**Consumers.** The board-state join (`attach_visual`, 11.1) wants time intervals, which this
gives. The FR11 topic filter wants topic labels, which nothing produces yet (`topic=` in
`filtering.py` is experimental). Detection never reads segments, so better boundaries would not
move recall. If topics are ever needed, add a second, coarser layer over these (lexical cohesion
first, TextTiling-style, stdlib only) rather than loosening the thresholds.

> **Decision.** `questxtract/segments.py`. Thresholds 2.0 s, 90 s and 800 characters, measured on
> the three reference recordings (MIT / TTK 09-14 / TTK 09-15,
> [#41](https://github.com/mad201802/QuestXtract/issues/41)):
>
> | `segment_max_gap_seconds` | 2 s | 3 s | 4 s | 6 s | 8 s |
> |---|---|---|---|---|---|
> | segments closed by a pause (%) | 96 / 98 / 95 | 81 / 91 / 75 | 74 / 81 / 64 | 48 / 66 / 45 | 36 / 50 / 35 |
> | median segment (s) | 22 / 15 / 16 | 28 / 24 / 38 | 40 / 29 / 46 | 65 / 49 / 59 | 75 / 64 / 61 |
>
> At 2.0 s the length caps stay rare backstops; a looser gap hands the cutting to the caps, which
> cut mid-thought. `stats["segments_closed_gap" | "_duration" | "_chars"]` is the check on a new
> recording: caps closing more than a few segments means these numbers do not fit it.

---

## 4. Evaluation harness

NFR2 gives a number to hit (recall >= 50% on the team's labelled set), so the metric runs in CI
and every threshold in sections 7 to 9 is swept against it.

### 4.1 The labelled set

One JSONL file per labelled recording, `tests/data/<name>.labels.jsonl`:

```json
{"start": 1284.2, "end": 1288.9, "text": "why does a stable sort matter here", "kind": "content"}
```

next to the transcript it is timed against, `tests/data/<name>.transcript.json`. A label's `text`
quotes that transcript and its `start`/`end` are snapped to the quoted words
(`scripts/scratch_to_labels.py --transcript`), so labels and predictions share one clock.

`kind` is `content` (a real question we want), `rhetorical`, `check_in` or `procedural`. Labelling
the negatives is what makes the false-positive rate measurable, not only recall.

**How the committed set is labelled.** Each transcript sentence by sentence, from scratch, with
no seed detector, so every entry is `blind` and precision is measurable: an unlabelled prediction
is a real false positive. The ground truth is the transcript, not the audio: a question the ASR
never transcribed (an off-mic student) is not labelled. The MIT labels were drafted by an LLM and
wait for human review. Totals, conventions and the cheaper seed-assisted workflow for long
recordings: `tests/data/README.md` and `docs/annotation_guidelines.md`.

### 4.2 Matching and metrics

A prediction matches a label when their time ranges overlap and the overlap covers at least half
the shorter range. Time-based matching, not text matching, because ASR text will differ from what
a human typed and we are not evaluating the ASR here. Overlap is measured against the shorter
span, not as IoU, because a prediction is a whole sentence that may carry a long preamble
around a short labelled question. Matching is one-to-one, highest overlap first: otherwise one
run-on sentence satisfies two labels, or two predictions both count as hits on one.

```
recall    = matched_content_labels / content_labels        <- NFR2 gate, >= 0.50
precision = matched_content_labels / predictions
f1
false_positive_breakdown: by the kind of label a prediction wrongly matched, plus unmatched
question_mark_rate: fraction of content labels whose transcript span contains '?'
blind_slice_recall: recall over the entries labelled "blind" only
```

`question_mark_rate` keeps 3.1's bet honest. If it drops, the sentence splitter is the first
suspect. `blind_slice_recall` keeps the labelled set honest; if it sits far below headline recall,
the set is measuring the seed detector.

### 4.3 Running it

```bash
questxtract eval tests/data/            # all labelled recordings, table to stdout
questxtract eval tests/data/ --json     # machine-readable, for CI
```

What is scored is the pipeline's output: stages 3 onward over the committed transcript, with
`min_confidence` at 0 so every detection candidate comes back, then gated at the default for the
headline. The sweep over the gate therefore covers both ends: at 0 it scores every candidate,
at 0.40 what a user gets. The text report lists every missed content question and every false
positive with the label kind it hit; `--json` carries the same lists.

CI runs it through pytest (`tests/test_evaluate.py`), which asserts pooled recall >= 0.50 over
the reference recordings, in well under a second.

`--board` scores the run with the board read: the board states in `<name>.visual.json` beside a
transcript go into the run in place of the vision stage's, which has no video to read here. The
files are one vision run per recording (`scripts/board_states.py`), committed so the comparison
needs no video and no model; a recording without one runs without the board.

```bash
questxtract eval tests/data/ --board    # the same, with the committed board states
```

> **Decision.** `questxtract/evaluate.py`. The report includes a threshold sweep over
> `min_confidence`, so the NFR2 tradeoff is visible rather than assumed.

### 4.4 Vision

The vision model (section 6) is scored the same way, on labelled keyframes instead of labelled
questions: `tests/data/vision/<name>.frames.jsonl`, one line per frame with its lecture type
(`slides`, `blackboard`, `whiteboard` or `handwriting`, as the layout probe of 6.2 names them),
the text on the board, each formula as LaTeX and a line on any diagram. The set spans several lecturers per type, because QuestXtract has to
read any teacher's board, not only the reference recordings'. Sources, licences and labelling
conventions: [`tests/data/vision/README.md`](../tests/data/vision/README.md).

```bash
questxtract eval --score-vision tests/data/vision/    # vision_model, per lecture type
```

Each frame is read by `vision.read_keyframe`, the function the vision stage uses (6.5), so eval
scores what a run sends: `vision_model` (`--vision-model`), prompt v0 with the label's kind, as
when the layout probe gets the kind right, and no speech; cached per request. Per frame: the
**character error rate** of `text` (edits per label character, casefolded and with whitespace
collapsed, capped at 1) and **formula recall**, a labelled formula counting as found when some
run of whole LaTeX tokens in the answer is within one edit per four of its tokens, after removing
spacing and sizing commands; a short formula must match exactly. A blank board has no error
rate, since any text would score 1; it counts as read when the answer has no text but
`[unreadable]`. A frame the model cannot read scores as an empty answer, a blank one as not read. The report gives all
three per lecture type and in total, then every frame, worst first.

CI checks the labels (every type has at least 5 frames, every frame exists), not the scores: they
need a real model.

> **Decision.** Character error rate, the standard measure for reading text from images, written
> out in a few lines rather than added as a dependency. Formulas are matched loosely, because two
> correct LaTeX spellings of one formula differ in spacing and braces. `diagram` is labelled but
> not scored: comparing two descriptions needs a judge, not an edit distance.

---

## 5. Data model

All in `models.py`, except `Config` (`config.py`). Every type is a frozen, keyword-only dataclass
(`@dataclass(frozen=True, kw_only=True)`), except `Context`. Keyword-only construction stops a
positional call from silently putting a value into a newly added field. Fields and docstrings:
the code.

| Type | What it is |
|---|---|
| `Word` | one transcribed word with `start`, `end`, `confidence`: the atomic unit (2.5) |
| `AsrSegment` | a raw ASR decoding chunk; not a `Segment` |
| `Transcript` | the words, the ASR chunks, `language`, `duration`; `from_dict`, `from_file`, `to_dict` |
| `Sentence` | a half-open word range with its text and timings; `id` is a content hash, so it survives a re-transcription |
| `Segment` | a run of sentences (3.2), its `close_reason`, and the board states that overlap it |
| `Layout` | where the content is in the frame (`content_region`) and its `kind`, one per recording (6.2) |
| `VisualContext` | one board state on its own clock: keyframe, description, text, `latex` (section 6) |
| `QuestionCandidate` | a detected question: sentence id, text, score, `question_type`, `reasons` |
| `QAPair` | a candidate with its answer span, confidence, timestamps and `verdict` |
| `Context` | the one mutable object: what every stage takes and returns (11.1) |

Three rules:

**`reasons` is never dropped.** Every rule that fires, positive or negative, appends a stable
string. This is what makes review possible and makes a wrong result debuggable without a
debugger. A question the LLM detector picks keeps the rules' codes for its sentence and gains `llm`.

**Do not drop `None` on serialisation.** The export writes a fixed key set, `null` included, so
consumers never have to defensively `.get()` a missing field.

**`VisualContext` owns its own time range.** A board state's lifetime is not a speech segment's
span: the lecturer writes for two minutes and talks about it for six. Board states live on
`Context.visual` on their own clock, and `Segment.visual` holds the states that *overlap* the
segment, joined by interval intersection in the `attach_visual` stage. Nailing one state to one
segment at capture time would bake in a wrong join, and would make vision uncacheable on its own.

---

## 6. Vision

Optional enrichment of the audio with what is on the board or the slides, off unless
`vision_enabled`. The `layout` stage asks a VLM once where the content is and what kind it is
(6.2). The `vision` stage samples the recording inside that region (6.3), cuts it into stable views
with one keyframe each (6.4) and has a VLM read each keyframe (6.5). The ink curve (6.4) is not
built yet; de-occlusion (6.3) is not planned.

### 6.1 The regime problem, and why not OCR

Slides and screencasts are easy (near-duplicate frame dedupe, then text). Chalkboards, whiteboards
and handwriting are hard, and FR4 plus the MIT reference recording put us there. An OCR model
returns the text and formulas it sees and nothing more: it cannot describe a diagram, does not
tell the board from the room or the app around it, and does not say what it could not read. A VLM
reads handwriting, diagrams and slides and emits LaTeX in one call, through the same optional
client as the LLM detector (section 10), so it adds no dependency. `vision` is one `STAGES` entry,
so a local OCR pass or a stub that loads pre-captioned frames in tests replaces that one entry.

**Model and prompt (spike #52, #153, #173).** The default `vision_model` is DeepSeek-V4.1-Flash
(`deepseek-ai/DeepSeek-V4.1-Flash-1017`) on NTNU's LLM service, the same model and endpoint as
the LLM detector (10.1), with its thinking off (`reasoning_effort = "none"`). On the labelled
keyframes (`eval --score-vision`, 4.4) it reads the text about as well as Qwen3.8-27B and fills
`latex` on more of the formulas, at no cost per lecture; with its thinking on it is far slower
for nearly the same reading. Qwen3.8-27B on the same service reads the board well too and is
one setting away (it rejects `reasoning_effort = "none"`, so it needs `--llm-reasoning server`).
The service needs the NTNU network or VPN; the customer may move hosting later, which is a
`Config` change (10.1). Of the hosted OpenAI models, `gpt-5.6-luna` is the cheapest that
stays under the customer's cost per lecture, fills `latex` on every formula frame, and reads about
as well as models that cost an order of magnitude more (`gpt-4o`, `gpt-4o-mini`).
`detail: low` is not used: it garbles handwriting. The OCR baseline is GLM-OCR (0.9B, open weights,
on Idun's hosted service and on an 8 GB laptop through Ollama): it reads most of the text and
formulas on chalk and printed slides, less than the vision models, and answers in plain text with
formulas inline as `$...$`, with no description and no confidence, and it also reads the room and
the app interface (an exit sign, the tablet app's menu). It is the second reader (6.5): faster
and lighter, for the text and formulas alone. Printed slides read as well at 640x360 as
at 720p, at half the image tokens; chalk was only tried at 720p. An open model reaches the same
endpoint (10.1); a 2B model on a laptop reads worse than Qwen3.8-27B and leaves `latex` empty. Prompt v0 and its output
schema, matching `VisualContext`, are in [`tests/data/vision/`](../tests/data/vision/README.md); the
comparison, its caveats (a few frames, one run per model) and the raw answers are on #52 and in
`tests/data/vision/results/`.

### 6.2 Layout probe

Recordings mix layouts (single camera, speaker plus screenshare, blackboard only,
picture-in-picture), and a layout classifier would be a permanent bug farm. Instead, the `layout`
stage sends three frames, taken at 20, 50 and 80 % of the recording and scaled to at most 640
pixels wide, to `vision_model` in one request, under a strict JSON schema:

```json
{"content_region": [0.32, 0.05, 0.66, 0.80], "kind": "blackboard"}
```

`content_region` is `[x0, y0, x1, y1]` in fractions of the frame from its top-left corner. `kind`
is `slides`, `blackboard`, `whiteboard`, `handwriting` (tablet or paper), `screencast` or `unknown`.
The answer becomes `Context.layout`: `vision` crops every sampled frame to the region, so the
change detection never sees the speaker's camera and the keyframe holds only the content, and puts
the kind into the captioning prompt (6.5).

VLMs are loose at box regression: GPT-4o and Claude fall well behind detection models on COCO
boxes ([arXiv 2507.01955](https://arxiv.org/abs/2507.01955)). A lecture only needs a coarse box,
so the code guards it rather than trusting it: the box is padded by 0.02 on each side within the
frame, and a box under a tenth of the frame keeps its kind but not its crop.

**Never required.** An endpoint error, a stall or invalid JSON twice, or a malformed answer (a
region that is not four fractions with `x0 < x1` and `y0 < y1`, a kind outside the list) leaves the
full frame and `unknown` with a warning, and the run goes on. A setting that cannot be used, such
as an unreadable `.env`, stops the run, as it does for captioning.

**Cached per request, not per stage.** The request, frames included, is cached like a caption
(10.4), and only a valid answer is, so a probe that failed is asked again next run while a probe
that worked costs nothing. The three frames are decoded again each run, a few seeks. The views
`vision` cuts (6.4) are cached in `views.json` keyed on the content region (11.2), so views cut
from the full-frame fallback are not reused once the probe answers.

**Known limitations.** One answer holds for the whole recording: a layout that changes
mid-lecture (the camera switches to a screenshare) keeps the first answer.

### 6.3 Frame sampling and de-occlusion

Sample at `vision_fps` (1) through PyAV, crop to the layout probe's `content_region` (6.2),
grayscale, downscale to about 480p.

**De-occlusion (not planned, #127).** A per-pixel temporal median over a window of frames would
remove a lecturer standing in front of a static board, and a motion check would pick the
lowest-motion frame where the camera pans (MIT 18.100B). Neither is built. The keyframe choice of
6.4 passes over frames with someone in front of the board; its known gaps are listed there.

### 6.4 Board states

**Static views (M2).** Each sample is compared with the first sample of the current view: the
share of pixels whose grey level moved by more than the codec noise (32 levels). Above
`vision_change_fraction` (0.02), a new view starts at that sample, but only once
`vision_change_hold` (3) samples in a row, 3 s at the default `vision_fps`, differ from the old
view and resemble each other. A lecturer walking past changes from one second to the next, so it
never holds; a new slide does, and is found within one sample. The last view ends at the
recording's end. Each view's keyframe is its sharpest sample (mean gradient) among
those that still match it, so a frame with someone in front of the board is not picked. Over
the board-state limit, the shortest view is merged into its shorter neighbour until the limit
holds. The limit grows with the recording: `vision_states_per_minute` (3) per minute, and never
fewer than `vision_min_states` (80), so a 45-minute lecture keeps 135 states and a 90-minute one
270, while a fixed cap would merge more and more of a long lecture away. The number of model
calls still has a known ceiling for the longest lecture. The rate was chosen on #153: with the
free models on NTNU's service, a call costs time rather than money, and 3 per minute gets nearly
the reading quality of no limit at about two thirds of its calls.
On a tablet lecture the views are page switches and annotations added to the same slide; a
panning chalkboard camera adds a view per pan, which the cap absorbs. A lecturer standing still in
front of the board for the whole hold is a view of its own, with the board partly hidden in its
keyframe (6.3).

**Ink curve (planned, #133).** Count ink pixels per sampled frame after an adaptive threshold. A
rising curve means writing, a sharp drop is an erase, and the local maximum before a drop is the
completed board: the keyframe. The recording is cut into **board states** between erase events,
roughly 20 to 60 per lecture. On slides the same curve works unchanged: a slide change is a large
ink delta.

### 6.5 Captioning

One VLM call per view, sent as prompt v0 was measured on #52: the prompt
(`VISION_PROMPT_VERSION`), filled in with the layout probe's `kind` (6.2) and what was said
while the view was on screen (up to 1500 characters), as the system message, and the keyframe
alone, as a PNG data URL, as the user message. The previous state's description and the ink
difference are to join it (planned, #137). Out, under a strict JSON schema and checked again in
code: `description`, `text`, `latex` (or null) and `confidence`. The prompt asks for what is
visible only, with `[unreadable]` for a word it cannot read. Cached per request, the image
included (10.4), so no keyframe is sent twice. `latex` is there because prose does not compile,
and the quiz and `filter --text` use the formulas.

**A page that comes back is not read again.** Lecturers return to earlier slides and pages, and
each return is a view of its own. The request cache cannot catch it: two captures of one page
differ by codec noise, and the speech sent with them differs. So on `slides`, `handwriting` and
`screencast` (the layout probe's kind, 6.2), each keyframe is first compared with the keyframes
already read in this run; boards are left to the ink curve (6.4), and `unknown` may be a board. The
comparison has two steps, because a share of changed pixels misses a small change: a new digit is
a few hundredths of a percent of the page, and a change of colour at the same brightness is none.
The earlier keyframe whose small grey copy (6.3) differs least, by at most 1 %, is the candidate.
Its reading is copied only if, at full resolution, no 4 px square's mean brightness moved by more
than 20 levels, no 8 px square's by more than 11, and no 8 px square's mean colour (Cb, Cr) by more
than 12; the squares overlap, and the last rows and columns are compared too. The new board state
keeps its own times, keyframe and id, so questions asked during it still link to it, and counts in
`stats.vision_reused`. A reading that failed is never copied. The limits are set from the codec
noise of TTK4225 2026-09-14, between frames of one page and between its pages coming back after
H.264 encoding, so anything unsure is read again: a pen cursor that moved, like a new decimal
point, gives a new reading, and so does a page whose codec noise exceeds the limits, as it can in
a recording compressed harder than a screen capture. A mark fainter than about twice that noise, roughly 48
grey levels of contrast on a light page, can go unnoticed. The reuse lives within the run, also
with `--from-scratch`; only the small copies are kept, and full-resolution pictures are decoded
while their view is handled.

Each keyframe that was read is written to the cache as `<source_hash>/keyframes/<content
hash>.png`, and `keyframe_path` holds that path relative to `cache_dir`: a cached result stays
valid from another working directory, and the board-state id does not depend on where the cache
lives. Frame times count from the start of the audio stream, the transcript's zero, so a file
whose streams do not start at zero keeps its board states beside the right speech; of what was on
screen before the audio starts, only the board visible at its start is kept. A keyframe whose
bytes no longer match its content-hash name, such as an interrupted write, is sampled again.
Writing `questions.json` copies the images beside it (12.1).

Captioning is **never required**: an endpoint error, a stall or invalid JSON twice, or a
malformed field leaves that one keyframe out with a warning, counted in `stats.vision_failed`, and
the run goes on. None of these is cached, so the next run asks again. When the
first three readings fail and none has succeeded, the rest are not sent: a wrong model name, key
or endpoint would fail every call the same way, and each one costs. The judge does
the opposite (10.3, rule 5) because the user asked for it; a board description only enriches
questions the pipeline already found. A setting that cannot be used, such as an unreadable `.env`,
is not a bad keyframe: it would drop every one the same way, so it stops the run as it does for the
judge.

**The OCR reader.** `vision_reader = "ocr"` (`--vision-reader ocr`) reads each keyframe with
`vision_ocr_model` (GLM-OCR, `zai-org/GLM-OCR` on NTNU's service) instead: the image and the task
prompt `Text Recognition:` in one user message, with the token probabilities asked for, through
vision's endpoint and key (10.1). It answers in plain text with formulas inline as `$...$`, which
`_from_ocr` turns into the same fields: `text` without the `$` delimiters, `latex` the formula
segments one per line, leaving out labels (a circled number, a number with a unit, a single
symbol), `description` empty, and `confidence` the mean probability of the answer's tokens, or 0.5
when the server sends none, which means "not measured". The converted answer is what is cached,
and failures follow the rules above. The speech and the kind do not go to the OCR model. The
layout probe still asks `vision_model`, so the frame is cropped to the board before OCR reads it;
where no vision model answers, the probe falls back to the full frame (6.2). `questions.json`
names the OCR model in `config.vision_ocr_model` (12.1), beside `vision_model`, which found the
board.

> **Decision.** The vision model stays the default: OCR gives no description, and the 19.10
> comparison (#153) put Qwen3.8-27B ahead on the board's words. OCR is the option for a faster
> run, or for a local machine (an 8 GB laptop runs GLM-OCR through Ollama, #144).

### 6.6 Consumers

- **Detection** (7.3): a small positive delta when a question shares a content word with the
  text of a board state on screen while it was asked.
- **Search** (`filtering.py`, `questxtract filter`): a text query matches every word against
  what was said and the board states in a question's `visual_ids` together, as whole words and
  ignoring case, so "what if this is zero?" is found by a word only the board shows.
- **A second, VLM pass** over the question/answer pairs (not built): the frames on screen go into
  the prompt, where they matter most: "what does this term mean?" is unanswerable from audio
  alone.

> **Decision.** `questxtract/vision.py`. Config: `vision_enabled` (False), `vision_fps` (1.0),
> `vision_model`, `vision_states_per_minute` (3.0), `vision_min_states` (80),
> `vision_change_fraction` (0.02),
> `vision_change_hold` (3 samples), `vision_base_url` (`None`: vision follows `base_url`, 10.1),
> `vision_reasoning_effort` (`None`: `reasoning_effort`, 10.1). Offline use points `vision_base_url` or `base_url` at a local
> OpenAI-compatible server; worse quality there is accepted.

---

## 7. Question detection

Purely lexical and orthographic. No classifier, no prosody. The rules are inspectable and every
decision leaves a reason code, which is what makes the review workflow possible.

**Why rules, not a model.** Deciding whether a sentence is interrogative is the easy half and is
lexical: in meeting speech, lexico-syntactic features carry question detection and pitch only
adds to them (Boakye, Favre, Hakkani-Tür, ASRU 2009). Deciding whether it is a *content*
question is the hard half, and trained models do not solve it either: a teacher-question
detector trained on classroom ASR reaches weighted F1 0.59 (Blanchard et al., SIGDIAL 2016), and
rhetorical questions are identified at F1 0.54 only once the neighbouring utterances are
features (Bhattasali et al., ACL 2015). With a few hundred labels there is nothing to train on
anyway. So the rules do the lexical half and keep the known noise out; the content call is the
LLM detector's (section 10), which sees the context those papers found necessary.

### 7.1 Extraction

One candidate per sentence (3.1), scored in 7.3.

**The interrogative head.** A question's first word is rarely the sentence's first word in
lecture speech: *"So what does it mean ...?"*, *"Now, what is the challenge ...?"*. The head is
the first word past leading discourse markers (`so`, `and`, `but`, `now`, `then`, `okay`, ...),
and past question preambles (*"my question is (that)"*, *"the question is/was"*, *"I'd like to
ask"*, *"I was wondering (if)"*), which announce a question without opening it, and past a
preposition right before a wh-word (*"For which age ...?"*). For a sentence ending in `?` the
first words of later comma clauses count too, because ASR runs the preamble into the question:
the last clause with any head (*"If you have the data, does that mean ...?"*), an earlier one
only with an auxiliary (*"This area here, is it finite, infinite, ...?"*), since a wh-word there
usually opens a relative clause. Without the `?` the clause rule stays off, or every *", which
is ..."* would read as a wh-question. Contractions stay whole, so negated auxiliaries (*"Don't
we ...?"*) open questions; a wh-word with a contracted verb (*"What's"*, *"Who'd"*, *"How'll"*)
counts as the wh-word.

**The answer seed.** The sentence right after the question, when the silence before it is at most
`answer_seed_max_gap_seconds` (1.0 s) and it is not itself a question; tags, check-ins and framing
questions get no seed. On the reference recordings about 4 in 5 seeds of labelled content
questions are real answers up to 1.0 s, about half beyond it. Details in section 8.

**Not built, and why.** Two paths have no labelled failure behind them:

- *Implicit cue without a `?`* (*"the question is whether ..."*, #48): a lecturer's restatement
  usually reaches the transcript with its `?`, and none of the current misses has this shape.
- *Missing `?` recovery* (a question word **and** a local answer slot): Whisper `turbo` with the
  punctuation prompt marked every labelled content question in the transcript but one, a cut-off
  *"Can this one be..."*.

Each lands when a labelled question is lost to it.

### 7.2 Vetoes

Hard rejects, before scoring: empty text, and text over `max_question_chars` (350, untuned).

Declarative "what" (*"what we call ..."*) and a causal clause (*"Because the cache is cold."*)
need no veto: without a `?`, a question word alone stays under the gate (7.3).

### 7.3 Scoring

Additive, clamped to [0, 1], with `framing` subtracted from the bounded score:

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

The tag and check-in lists are English (requirement 4.2), grown from the labelled recordings
rather than written up front, and deliberately short: they only keep the obvious noise out, and
the content call is the LLM detector's. Tags follow the annotation guidelines, which label them `rhetorical`: a
penalty with a reason code, not a veto.

**Board context.** A candidate that shares a content word with the `text` of a board state on
screen while it was asked (the board state's lifetime overlaps the sentence) gets
`board_overlap_delta` added, clamped, and the reason `board_overlap`; its `source` is then
`combined` (12.1). Off by default (`0.0`): see the results below. Content words are the sentence's words minus the framing stopwords below. It
comes after the gate: the board raises a candidate's confidence, it never makes one. A lecturer
asking while pointing at the board is asking about the board. Only the `text` counts, not the
model's `description`, which paraphrases and would match anything.

Framing examines the whole sentence, so "The field, what is it?" keeps its content word.
The stopwords are wh-words, positive auxiliaries, discourse markers, prepositions,
pronouns, articles and a few fillers; negation (`not`, "can't"), numbers and ordinary
verbs stay as content signals. "What could happen?", "Can it be?" and "How much?" remain
questions. Wh contractions are normalized ("What's" to "what"). Tags and check-ins do not
also receive `framing`. Applying this penalty after bounding the score ensures a sentence
with both `wh` and `aux` signals is still penalised.

**Current results**, default config, `questxtract eval tests/data/`. Update this table in the PR
that changes the numbers; issues link here instead of copying them.

| Recording | Content labels | TP | FP | FN | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| MIT 18.100B L1 | 28 | 28 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| TTK4225 2026-09-14 | 41 | 39 | 10 | 2 | 0.95 | 0.80 | 0.87 |
| TTK4225 2026-09-15 | 60 | 59 | 18 | 1 | 0.98 | 0.77 | 0.86 |
| **Pooled** | 129 | 126 | 28 | 3 | **0.98** | 0.82 | 0.89 |

With the board (`eval --board`, `board_overlap_delta = 0.10`) every row is the same.
`board_overlap` fires on 38 of the 230 candidates, but 36 of them already score 1.0, a `?` and an
interrogative head, so the clamped delta moves two: a check-in (*"How much are we together on
this, people 1, 2, 3?"*, 0.33 to 0.39) and a framing question, neither near the 0.40 gate. The
sweep moves only off the default: precision 0.82 to 0.81 at 0.35, recall 0.44 to 0.45 at 0.65
and 0.70. A delta on a saturated score has nothing to add, so `board_overlap_delta` stays 0.0.

NFR2 holds with room, and `question_mark_rate` is 0.99, so 3.1's bet holds. The sweep: recall
0.99 / precision 0.56 up to a threshold of 0.30, 0.98 / 0.82 from 0.35 to 0.55, 0.44 / 0.86 from
0.65 up.

*Missed (3).* The `?`-less cut-off above, and two content questions that score 0.60 and stay
unanswered (0.33): a check-in wrapping a content question (*"..., are we together that this
area is going to be infinite?"*, `check_in` penalty) and a subject poll with no interrogative
head (*"Unbounded, bounded, ..., or you do not know?"*).

*False positives (28).* 18 hit `rhetorical` labels, including questions with ordinary verbs
that the conservative framing rule keeps (*"What does it mean?"*, *"What am I doing?"*).
8 hit `check_in` ones the list does not hold, such as elicitations (*"What would you say?"*).
Every check-in form on the list comes from one lecturer's two recordings, so it is re-checked
on another (#92). 2 hit
`procedural` ones. None is a confirmation tag, so the tag penalty does its job; the content call
beyond that is the LLM detector's (section 10).

> **Decision.** `questxtract/detect.py`, cue tables inline as module constants.
> `board_overlap_delta` (0.0, off).

---

## 8. Answer search

One rule, the **adjacent seed**: the sentence immediately after the question's own, at a
near-zero gap (`answer_seed_max_gap_seconds`, 1.0 s) and not itself a question. The Socratic
case, and the highest-precision answer shape we have. Stage 3 splits at the `?`, so a sentence
never holds its own answer; the seed is the next sentence.

Tags, check-ins and framing candidates get no seed: the following sentence is the lecture
continuing rather than an answer to a content question.

Every answer further away, after a digression or several sentences later, comes from the LLM
detector (section 10), not from rules. With no endpoint configured, a question keeps its seed or has no
answer.

There is **no speaker constraint on pairing**, and not only because we have no speaker layer:
lectures are full of self-answered questions, and requiring a speaker change would delete the
single most common QA shape in the corpus - which with off-mic students is *the entire shape*,
since the lecturer both repeats and answers the student's question. The LLM detector follows
the same rule.

> **Decision.** `questxtract/answer.py`, the adjacent seed only. Why the rules stop there:
> [design_decisions.md](design_decisions.md), D11.

---

## 9. Scoring

### 9.1 Answer scoring (not planned)

No rule-based answer scoring. An answer the adjacent seed found counts as found; finding a better
one is the LLM detector's job (10.2), for the same reason as section 8.

### 9.2 Confidence

```
confidence = 0.55 * question_score    no answer found
           = 1.00 * question_score    the adjacent seed found one

label:  >= 0.75 high | >= 0.45 medium | else low
```

A question with no answer keeps `0.55 * question_score` and an empty answer span. It is still
exported: FR6 and FR7 ask for questions with timestamps, not for answered questions.

Gate: `min_confidence` (0.40), swept in the eval report. With the LLM detector, the confidence is
the model's own (10.2), with no answer weighting: the model saw the answer when it labelled the
question, so weighing it again would count it twice. The gate is the same.

One number the user can reason about (FR10), plus a verdict. No parallel quality score, no risk
band, no profile-scoped emission gates.

### 9.3 Dedupe (not planned)

No rule-based dedupe. Echo pairs (a student asks, the lecturer restates, both get detected) and
near duplicates stay in the output. The LLM detector labels a fragment echo `procedural`
(annotation_guidelines.md 2) and sees a stretch of lecture at once (10.4).

> **Decision.** `questxtract/score.py`: confidence and the threshold.

---

## 10. The LLM layer

One client, two jobs, each off unless asked for: the LLM detector (stage 8, `--detector llm`),
which decides which candidate sentences are real questions, and the vision of section 6 (the
layout probe and captioning). Each stage keeps its own prompt, schema and output checks. What both send goes through
`endpoint.py`, which is not a stage, so neither stage imports the other (11.1, rule 1): it builds
the client and sends a request with a deadline on the whole call, one resend after a stall or
invalid JSON, the per-request cache of 10.4, and the key hidden in every error. The pipeline
produces a complete result without either job. A second LLM run, a VLM
that brings the frames on screen into the question/answer pairs, belongs with the vision stage and
is not built.

### 10.1 One client, two endpoints

The `openai` SDK with a configurable endpoint, which is what makes the offline story work
without a second code path:

```python
OpenAI(api_key=ntnu_key, base_url="https://llm.hpc.ntnu.no/v1")  # NTNU's LLM service, the default
OpenAI(api_key=openai_key, base_url="https://api.openai.com/v1")  # OpenAI
OpenAI(api_key="local", base_url="http://localhost:11434/v1")  # local device
```

By default the judge, the quiz and vision all go to NTNU's LLM service with one model. Vision
follows `base_url` unless `vision_base_url` sends it elsewhere (`endpoint.route`). The key goes
with the endpoint, not with the caller:

| Endpoint | Key |
|---|---|
| NTNU's LLM service, host `llm.hpc.ntnu.no` (the default `base_url`) | `QUESTXTRACT_NTNU_API_KEY` |
| a `vision_base_url` on any other host than NTNU's or OpenAI's (a hosted vision model) | `QUESTXTRACT_VISION_API_KEY` |
| any other: OpenAI (`https://api.openai.com/v1`, or `base_url = None`), OpenRouter, a local server | `OPENAI_API_KEY` (a local server needs none) |

The service is told by the URL's host alone. So the OpenAI key is never sent to NTNU, the NTNU
key never leaves it, and neither goes to a host set up for vision alone. Each key is read from
its variable, in the environment or the working directory's `.env`, never from
`questxtract.toml`, and hidden in error messages. The CLI checks the key of every endpoint a run
will use before anything slow starts, and refuses a model name from the other service: every
model on NTNU's is named `organisation/model`, and no OpenAI model has a `/`.

Any local OpenAI-compatible server (Ollama, llama.cpp's server, vLLM, LM Studio) plugs in with no
new code. Offline results may be worse, but the path must be *available* on a local device.

| Config key | Default | Role |
|---|---|---|
| `judge_model` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | the LLM detector; one request per group of nearby candidates |
| `vision_model` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | reads the board (6.5) and finds it (6.2); tens of board-state calls per lecture |
| `quiz_model` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | writes the quiz (12.3); one request per question |

A local server takes its own model names (`qwen3.5:9b` in Ollama). Swapping a model or pointing at
a local server is a `Config` change, no code. No abstraction library sits on top (litellm,
pydantic-ai, instructor): `base_url` already reaches every server we target, and the output checks
of 10.3 are ours to write either way ([design_decisions.md](design_decisions.md), D12).

`reasoning_effort` (`--llm-reasoning`) sets how long a thinking model may think first. The judge
and the quiz send it with every request, and vision with every request to `vision_model` (the
layout probe and its readings) unless `vision_reasoning_effort` sets vision apart, for a vision
model other than the judge's. The OCR model gets no setting: it does not think. The default,
`"none"`, switches DeepSeek's thinking off: it thinks unless told so, and on NTNU's service
`"none"` is the only value that stops it (`"low"` still thinks). `""` (`--llm-reasoning server`)
sends no setting, for a model that has none (gpt-4o) or rejects `"none"` (Qwen3.8-27B).
Thinking costs minutes per request instead of seconds, and it is what lets a small model such as
Qwen3.5 9B judge well, so the CLI raises the per-request deadline to 10 minutes when a level
(`low`, `medium`, `high`) is asked for. Ollama's context length cannot be set per
request and is 4k tokens below 24 GB of VRAM, cutting a longer prompt silently; a request (10.4)
stays within 8k, and [cli.md](cli.md) starts Ollama with `OLLAMA_CONTEXT_LENGTH=8192`.

ASR stays local: `faster-whisper` gives word timestamps, costs nothing and needs no network.

### 10.2 What the LLM detector does

With `detector = "llm"`, the judge (stage 8) decides, for every candidate, whether it is a real
question, and replaces the candidates and pairs of stages 6 and 7 with the ones it keeps.

**Candidates** are every sentence ending in `?` plus the rules' own candidates (7.3), so a
confirmation tag the rules score under their gate is still judged. A question with neither is out
of reach; the eval report's `question_mark_rate` tracks how often that could happen. A sentence
both keep keeps the rules' `reasons` and `question_type`.

**Windows.** A candidate is sent with `judge_window` sentences of lecture on each side. One within
`judge_window` sentences of the previous candidate joins its request, up to `judge_batch_size`
candidates, so the excerpt stays one piece of lecture; any other starts a new request. Excerpts
overlap where candidates are close, so no candidate loses the lecture around it.

**Request.** One line per sentence, candidates marked `>>`:

```
   s_4a01bc 1279s So we sort by the first key and then by the second.
>> s_1b88c0 1284s so why does a stable sort matter here
>> s_77e2d0 1301s [said 23x in this lecture] True?
```

The repeat count appears from three repeats on: on the labelled set no content question repeats
that often, while the lecturer's habitual tags do, and the count adapts to each speaker without a
list. The prompt (`JUDGE_PROMPT_VERSION`) holds the four kinds of
[annotation_guidelines.md](annotation_guidelines.md) 2, defined by what a question asks about:
`content` asks about the subject, whoever answers it, since lecturers answer most of their own
questions. The response is an object keyed by candidate id, its shape fixed by a JSON schema sent
as `response_format`:

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

`reason` comes first, so a model without a reasoning mode states its case before it labels. Only
`content` is kept; the other kinds are counted in `stats.judge_rejected`. A judgement maps onto the
output of section 12 without a schema change:

- `confidence` is the question's confidence as it is; the gate of 9.2 applies, the answer
  weighting does not.
- A non-empty `question` is the question's `text`, with verdict `revise` and the transcript
  sentence as `original_text`; an empty one keeps the sentence, verdict `keep`.
- The answer is the cited sentences' own text: extractive, no generated answer text.
- The question gains the reason `llm`, its pair `llm_answer` or `no_answer_found`.

Why one decision per candidate rather than a list per stretch of transcript: small models judge a
given sentence far better than they enumerate a long one, and the schema turns completeness into
something the decoder enforces ([design_decisions.md](design_decisions.md), D13).

### 10.3 Constraints on the LLM detector

1. **It may not add questions.** It judges only the candidates the code chose, by id; a judgement
   for anything else, or a malformed one, is dropped and counted in `stats.judge_dropped`. This
   enforces requirement 4.5 in code, not only in the prompt.
2. **An answer is later sentences of the excerpt.** The schema lists exactly those ids per
   candidate, and any other cited id is dropped.
3. **Three layers of output checks**, because servers differ in how faithfully they apply a
   schema: constrained decoding on the server, then `json.loads` with one repeat of the request,
   then rules 1 and 2. None trusts the layer before it.
4. **Every candidate comes back with a kind.** The schema requires each candidate id, so a
   constrained decoder cannot skip one or stop early. A server that ignores the schema can, so the
   skipped ones are asked for once more; any still missing are counted in `stats.judge_skipped`.
5. **It is required once asked for.** The user chose the LLM, so an unreachable endpoint, a bad
   key, an unknown model, invalid JSON twice, or no answer within `timeout_seconds` twice ends the
   run with one `error:` line; there is no silent fallback to the rules. The deadline covers the
   whole request: a server may keep the connection alive while the model works (OpenRouter does,
   with no limit of its own), which a per-read timeout never catches. A stall is usually the
   provider, so a request is sent once more before it counts as failed. Requests already answered
   stay cached (10.4), so a re-run resumes.

### 10.4 Requests, cached per request

A lecture makes one request per group of nearby candidates, about 15 to 25 for a lecture of an
hour or more, each a few thousand characters. `judge_parallel` (1) sends several at once. It stays
1 by default: a local server queues parallel requests, and the time in the queue would count
against the deadline; a hosted run can raise it.

Each request is cached under `.questxtract/judge/<sha256 of the request>.json`, keyed on the
endpoint, the model, the prompt, the schema and the transcript lines. Any change to what the model
would see asks again, and nothing else does: another `-t` or an `eval` re-run costs zero calls,
and a `.json` transcript input, which has no source hash, is cached too. A vision request (6.5) is
cached the same way under `.questxtract/vision/`, keyed on the keyframe image instead of transcript
lines, and a layout probe (6.2) under `.questxtract/layout/`, keyed on its three frames. Only a valid JSON answer is cached; a failed request is asked again on the next run. A cache entry that cannot be read
is asked again, and a cache that cannot be written costs only the saving: the answer is returned.

What the calls cost lands in `questions.json`'s `stats`, per caller, whenever that caller
runs: `judge_*` for the LLM detector, `layout_*` for the layout probe (6.2) and `vision_*` for
the board readings (6.5). Each caller has `_requests` (every request sent, a repeat included),
`_input_tokens` and `_output_tokens` (as the API reports them in `usage`; 0 from a server that
reports none) and `_ms`. A run served from the cache shows 0 for all four, which is how to tell
it made no calls. Prices are not applied: they depend on the provider and the date.

> **Decision.** `questxtract/llm.py` (the judge), `questxtract/endpoint.py` (client and requests,
> shared with vision). Config: `detector` (`rules`), `base_url` (NTNU's LLM service), `api_key`
> (the key for `base_url`, 10.1), `judge_model`, `judge_window` (10), `judge_batch_size` (8),
> `judge_parallel` (1), `reasoning_effort` (`"none"`), `timeout_seconds` (120). Prompt and
> schema are module-level, so a prompt change is a reviewable diff. Dependency: `openai`, in the
> optional extra `questxtract[llm]`, imported inside the functions.

---

## 11. Orchestration: stages, caching, progress

### 11.1 The stage contract

Every stage is a function with the same signature, `Context -> Context`. That sameness is the
whole extension mechanism.

| Want to change | Seam | What it costs |
|---|---|---|
| Whisper checkpoint, VLM, judge model | a `Config` string | nothing, no code |
| Hosted endpoint to a local one | `base_url` (10.1) | nothing, no code |
| The ASR engine | an `AsrBackend` class (2.2) | one class |
| The implementation of a stage, or a new step | replace or insert one entry in `STAGES` | one function |

**Stages.** A stage is tagged with a decorator carrying its name and cache policy:

```python
@stage("transcribe", cache="transcript", uses=("model_size", "language", "beam_size", ...))
def transcribe(ctx: Context) -> Context: ...
```

`cache` names the one `Context` field the stage produces (cached as `<stage name>.json`; `None`
means always recompute), and `uses` lists the `Config` fields that form the cache header (11.2).

**The pipeline is a list.** `pipeline.STAGES` is the diagram of 1.3, and `run(ctx, stages)` walks
it; `process(path, config)` builds the `Context` and runs it. Passing a different `stages=` list is
the entire plugin system: no plugin discovery, no `entry_points`, no DAG resolver. List order is
execution order, and a stage that needs something an earlier one did not produce fails on the spot
with a `None`. `tests/test_pipeline.py` checks that `STAGES` matches 1.3.

**Four rules hold the contract up:**

1. **Stages never import each other.** They communicate only through `Context`, which is what
   makes a stage replaceable in isolation.
2. **Stages read config only through `ctx.config`.** No module constants that are really
   settings, no environment reads outside `config.py`. A stage whose behaviour is not a function
   of `(ctx.config, ctx)` is not cacheable and not testable.
3. **Optional stages no-op themselves.** `layout`, `vision` and `judge` return `ctx` unchanged
   when disabled; the runner stays dumb.
4. **A cached stage produces exactly one `Context` field** of JSON-serialisable dataclasses. This
   is why `vision` (expensive, cached) and `attach_visual` (an interval join, always recomputed)
   are two stages.

**Config.** One flat frozen dataclass in `config.py` with every default, built from CLI flags or by
a library caller. Flat is what makes `uses=(...)` a tuple of names rather than path expressions.
For `process`, `transcribe` and `eval`, explicit flags override a flat TOML file, which overrides
these defaults. `--config PATH` selects the only file to read; otherwise `questxtract.toml` in
the working directory is read if present. `load_config` checks keys against the dataclass fields
and values against their annotations. The CLI validates the resulting choices and ranges before
running stages or writing output. Relative paths remain relative to the working directory.
`Config.api_key` and `Config.vision_api_key` read the key for their endpoint (10.1) from the
environment first, then from that directory's UTF-8 `.env` file. It never stores the key in the dataclass or expands shell expressions.
Secret-like TOML keys are rejected. Library callers can use `load_config` from `config.py`
explicitly; constructing `Config()` alone does not read TOML.

> **Decision.** `STAGES`, `run()`, `process()`, the cache and the progress plumbing live in
> `pipeline.py`, written once in the runner instead of in every expensive stage. The `stage`
> decorator lives in `models.py` next to `Context`, because `pipeline.py` imports every stage
> module and defining it there would be an import cycle.

### 11.2 One cache

```
.questxtract/
  <source_hash>/transcribe.json   the transcript
  <source_hash>/views.json        the stable views of the recording (6.4)
  <source_hash>/keyframes/        their keyframe images, named by content hash
  layout/<request hash>.json      one per layout probe request (6.2, 10.4)
  vision/<request hash>.json      one per keyframe request (10.4)
  judge/<request hash>.json       one per LLM detector request (10.4)
```

Each entry is written with a header holding the values of the stage's `uses` config fields. The
runner reuses an entry when the header matches and recomputes when it does not; `--from-scratch`
bypasses reuse for every stage. Sentences, segments, detection, answers and scoring declare no
`cache` and always recompute: caching a 20 ms stage buys nothing and costs an invalidation bug.

**No source hash, no cache.** The directory name *is* the identity of the recording, so a run with
an empty `ctx.source_hash` (a caller seeding a transcript directly, say) skips the cache instead of
sharing one bucket, which would replay one recording's results into another's.

**The LLM detector is the exception to the `uses` rule.** Its input is the transcript, which a
config-only header cannot key, so it declares no `cache=` and caches each request itself, keyed on
the request (10.4). `--from-scratch` bypasses it too. The layout probe does the same (6.2), so a
failed probe is not stored and is asked again.

**Vision is the second exception.** Each reading depends on the transcript (the speech is in the
prompt) and may fail, and a stage-level entry would replay both a stale reading and a failed
one. So `vision` declares no `cache=`: it caches the sampled views itself in `views.json`, keyed
on the content region and the sampling settings, and each reading per request, only once the
answer is valid (6.5).

A stage that reads a config field and forgets to list it in `uses` serves stale results: the one
bug class this design still has. Keep `uses` next to the signature so the two are reviewed
together.

### 11.3 Progress (FR9)

```python
Progress = Callable[[str, float], None]
```

Stage name plus fraction complete. The CLI renders a redrawn terminal bar from it; the library
default is `None`. The runner emits one call per stage from the decorator's name, so a new stage
reports progress for free. Long stages report mid-run: transcription reports
`word.end / info.duration` per decoded word through the `on_progress` callback every `AsrBackend`
accepts, capped at 0.99, since the runner, not the backend, decides when a stage is finished.
Vision reports sampling as its first half and each keyframe read as its second.

> **Decision.** A callable, not a progress framework, and not a logging handler.

---

## 12. Output

### 12.1 questions.json

Schema 2.4: a fixed key set, `null` included, in one file, plus a keyframe folder beside it.
The formal description is [`questions.schema.json`](../src/questxtract/questions.schema.json),
which ships inside the package; the tests validate every export shape against it. The example shows one entry of each
list, with vision and the LLM detector on; with the rules and no vision, `visuals` is
empty, `verdict` is `unjudged` and the judge and vision settings are `null`:

```json
{
  "schema_version": "2.4",
  "source": {"path": "lecture.mp4", "duration_seconds": 4812.3, "hash": "9f2a..."},
  "config": {"model_size": "turbo", "language": "en", "min_confidence": 0.4,
             "detector": "llm", "judge_model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
             "judge_prompt_version": "v5", "judge_reasoning_effort": "none",
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

**Shared things are listed once and referenced by id.** `segments` and `visuals` (board states,
6.4) are top-level lists; a question points into them with `segment_id` and `visual_ids`, so a
board state that ten questions discuss is written once. The vision milestones grow into this
shape without a breaking change: M3's board timeline is `visuals` in time order.

- `visual_ids` holds the board states on screen while the question was asked, falling back to
  every state joined to its segment when none was; empty when vision is off. A segment's own
  `visual_ids` holds every state joined to it. `segment_id` is `null` when the question's
  sentence belongs to no segment.
- `answer` is `null` when no answer was found. Its `start`/`end` span its own sentences
  (`sentence_ids`), not the question.
- `original_text` is set when the LLM detector reworded the question, `null` otherwise.
- `source` says what the question was read from: `audio`, or `combined` when the board on
  screen raised its score (`board_overlap`, 7.3). `visual` is reserved for a question read from
  the board alone.
- `topic` on a segment stays `null` until something produces topics.
- `keyframe_path` is relative to the JSON file: writing it copies each keyframe from the cache
  into a folder named after it (`lecture.questions.json` gets `lecture.keyframes/<visual id>.<ext>`),
  so the file and that folder move together and lectures in one folder keep their images apart. Printed
  to the terminal (`-o -`) there is no folder, nothing is copied, and it is relative to the cache
  directory, where the vision stage keeps keyframes.
- `config.min_confidence` is the threshold the file was cut at; `config.judge_model` and
  `config.vision_model` are `null` when that stage was off. `config.vision_ocr_model` names the
  OCR model when it read the keyframes (6.5), else `null`; `vision_model` then only found the
  board.
- `config.detector`, `judge_prompt_version` and `judge_reasoning_effort` tell two runs apart:
  `rules` or `llm`, the judge prompt's version, and the `--llm-reasoning` level
  (`reasoning_effort`, `""` for `server`). The last two are `null` with the rules.
  `vision_reasoning_effort` is the setting vision's model requests carried, `null` with vision
  off. The file is not a run log; other settings are not recorded.

**Versioning.** `schema_version` is `MAJOR.MINOR`. A minor version adds keys, or changes what a
value means without changing its type (as `keyframe_path` being relative to the file), so a
reader must ignore keys it does not know; a major version moves or removes something, and
`read_questions` refuses a file of another major version.

### 12.2 Editing (FR5)

The output JSON is the edit surface, and the editor lives in the wrapper, not in this package.
Each question carries one block that belongs to people, `review`. The pipeline writes its
defaults and never reads or overwrites it; everything else in the file is regenerated on every
run.

| `review` field | Meaning |
|---|---|
| `status` | `pending` (untouched), `accepted`, `deleted`, or `false_positive` (not a question at all) |
| `text` | The edited question text; `null` while unedited. The question's own `text` stays the original. |
| `note` | Free text from the reviewer. |
| `metadata` | Any extra key-value data, such as a topic tag or a quiz difficulty. |

A re-run does not keep the edits: copying `review` across by question id is not planned (#15).
Question and sentence ids are content hashes (section 5) rather than indices, so they survive a
re-run and the wrapper can carry edits across by id.

### 12.3 quiz.json

A different command producing a different file: `questxtract quiz questions.json`, or
`process --quiz` after `questions.json` is written. It is the only part of the system allowed to
write text that was not spoken, and keeping it out of the extraction pipeline is what keeps
requirement 4.5 checkable.

**Items.** `quiz_options` options per item, one correct. A stem is `spoken` only when it is
the question as it was said (rule 7 below); the rest is `generated`. Every item starts with
`review.status` `pending`.

**Selection.** Questions a reviewer rejected and check-ins are never quiz material. `lenient` (the
default) also takes questions the lecture did not answer, whose answer the model writes from what
was said around them; `strict` keeps only questions the lecture answered. The model's first field, `quiz_material`, can still turn a
question down (addressed to the room, rhetorical, too vague on its own): it is left out like a
check-in, and the answer is cached like any other. Without that way out, a model writes an item
for every question it is given, and invents a topic for a vague one.

**One request per question** through the shared endpoint (10.1), cached under
`.questxtract/quiz/`: the question, the lecture's answer, the sentences around it (four before,
two after, the question marked), and the `description`, `text` and `latex` of the board states
linked to it (12.1), so formulas on the board reach the options. The speech is what makes a
pointing question ("this area here, is it finite?") usable, and what lets the model tell a
misheard word from what was meant. `questions.json` does not hold the transcript, so `quiz` takes
`--transcript` or reads the cached one by the recording's hash (`pipeline.cached_transcript`);
`process --quiz` has its sentences in hand. The
response schema names every field, and the code checks that every field is filled and every
option differs. A question that fails twice, or stalls twice, is left out with a warning; an
unreachable endpoint or a bad key ends the run, as the same error would hit every question.

**Deterministic.** Items follow `questions.json`, item ids derive from the question ids, the
correct option's place from a hash of the question id, and nothing records the time of the run.

> **Decision.** `questxtract/quiz.py`, not a stage. Config: `quiz_model`
> (`deepseek-ai/DeepSeek-V4.1-Flash-1017`), `quiz_strictness` (`lenient`), `quiz_options` (3,
> from 3 to 5); `base_url` and `reasoning_effort` are shared with the judge. Every written file passes `validate_quiz`.

Schema 1.0, formally [`quiz.schema.json`](../src/questxtract/quiz.schema.json), which ships in
the package. Items are single choice: at least three options, exactly one of them correct, a
`solution` with an explanation and one `option_explanations` entry per option. Every text
(stem, options, explanations) is `{text, origin}`: `spoken` only when it is verbatim from the
lecture, `generated` for anything reworded or newly written. `provenance` ties an item to its
source question (`source_question_id`, `start`, `end`), and the item id is `"quiz_"` plus that
id. Every item has a `review` block (`pending`, `accepted`, `rejected`) that the teacher owns.
The file holds no run timestamps and no absolute paths, so the same input gives the same bytes;
exporters build feedback from `solution`.

`validate_quiz` (in `quiz_schema.py`) checks what the schema cannot: the correct option exists,
explanations match the options one to one, option ids are unique, item ids are derived and
unique, `end >= start`, and, given the source `questions.json`, the source question exists and a
`spoken` stem is that question as it was said (`original_text` when the judge reworded it, else
`text`). `read_quiz` refuses another major version, then validates. Spoken options and
explanations cannot be checked without the transcript, so they stay a rule for the writer. The
validator is plain Python: `jsonschema` checks the schema in the tests only.

### 12.4 QTI 3.0 package

`questxtract qti quiz.json -o quiz.qti.zip` turns a quiz (section 12.3) into a 1EdTech QTI 3.0
content package an LMS can import: `qti.py` maps the quiz to a model, `qti_v3.py` writes the XML
and the ZIP. The file is read through `read_quiz`, which refuses another major version and the
errors across fields; the structure is not checked at runtime, so `qti.py` checks every field it
uses and names its path.

```text
items/<item id>.xml            one qti-assessment-item per quiz item, in document order
assessmentTest.xml             one linear test part and section referencing every item
imsmanifest.xml                the content package manifest
metadata/questxtract.json      provenance, review status and the origin of every text
```

- **Selection.** Every item except those whose `review.status` is `rejected`.
- **Items.** A `qti-choice-interaction` with `max-choices="1"` and one `qti-simple-choice` per
  option. Identifiers are the quiz's item and option ids, which are already XML identifiers.
  `RESPONSE` is a single identifier whose correct response is the correct option. Any other
  `interaction_type`, which a newer 1.x quiz may add, fails with its path instead of being
  exported as single choice; so do `NaN` and infinite numbers, which `json.loads` accepts.
- **Scoring.** `SCORE` (float, default `0.0`) and `MAXSCORE` (the item's points). With points
  above zero, `SCORE` carries `normal-maximum` and `normal-minimum`. Explicit response
  processing from the Core subset: a match against the correct response sets `SCORE` to
  `MAXSCORE`, otherwise to `0.0`. The `match_correct` template is not used, because it always
  scores 1.
- **Feedback.** Response processing sets the `FEEDBACK` outcome to the chosen option. Each option
  has a `qti-modal-feedback` with its own explanation, then the general one, both from
  `solution`. No response, no feedback.
- **Instructions.** `show_feedback` becomes `show-feedback` on the test part's
  `qti-item-session-control`, `shuffle_options` the interaction's `shuffle`,
  `shuffle_questions` a `qti-ordering shuffle="true"` in the section, and `text` a candidate
  `qti-rubric-block`.
- **XML.** Items and the test use the default namespace `imsqtiasi_v3p0` with an
  `xsi:schemaLocation` to the 3.0.1 XSD, as every official example does, never prefixes. The
  manifest uses the QTI 3.0 content-package namespace and `schemaversion` `3.0.0`, the only value
  the implementation guide and the 3.0.0 CP XSD allow. Every file but the manifest is a resource;
  the JSON sidecar is `type="extension"`.
- **Validation.** The tests validate every package they write against the official XSDs (items
  and test against ASI 3.0.1, the manifest against the content-package XSD), then check what an
  XSD cannot: the correct response and every feedback name an option, every `href` and
  dependency resolves, identifiers are unique. `scripts/fetch_qti_xsd.py` downloads the 17
  schema files once, pinned by SHA-256, into `.cache/qti-xsd/`, and the tests compile them from
  there with network access off. Without that copy they are skipped, except in CI, where they
  fail. The package itself does not depend on the XSDs.
- **Determinism.** The same input gives the same bytes: entries are written in a fixed order with
  the date 1980-01-01, deflate, Unix permissions `0644` and no clock or platform data.

### 12.5 ExerPlaza TeX

`questxtract tex quiz.json -o quiz.tex --content-unit NAME` turns a quiz (section 12.3) into the
`IndexedQuestion` format that ExerPlaza imports at `/questions/upload/tex`, as read by its parser
`exerplaza/backend/quiz/latex_parser.py` v2.1. A quiz reaches ExerPlaza with no change on
ExerPlaza's side and no network access. The file is a transport format for that parser, not a
document to compile: there is no preamble.

**Verbatim texts.** ExerPlaza stores every text as parsed and shows it with MathJax (`$...$`,
`processEscapes`), so texts are written without LaTeX escaping; a `\%` would show up on the page
as `\%`.

**Mapping.** UTF-8 without a BOM, LF line endings. A header of four comment lines (quiz id, schema
version, title), then one block per item, each starting with `% questxtract: <quiz id>/<item id>`
so an import can recognise a re-import.

| TeX | From |
|---|---|
| `\QuestionContentUnits` | every `--content-unit`, in order, the same for every item |
| `\QuestionType` | always `multiple choice` |
| `\QuestionLanguage` | the language's primary subtag: `en` is `english`, `it` is `italian`; any other fails |
| `\QuestionDisclosability`, `\SolutionDisclosability` | `review.status`: `pending` is `only me`, `accepted` is `everybody` |
| `\QuestionBody` | the stem, verbatim, CRLF as LF |
| `\QuestionPotentialAnswers` | one `\correctanswer text` or `\answer text` line per option, in order, whitespace collapsed |
| `\QuestionNotesForTheTeachers` | the reference, the source question, its times (`m:ss`, `h:mm:ss` from one hour on), the stem's origin, and the model and prompt version (the method when there is no model) |
| `\SolutionBody` | the explanation, a blank line, then `- <option>[ (correct)] — <explanation>` per option |

Points, instructions, the description, the source and the review note have no place in
ExerPlaza and are not written. ExerPlaza parses `\QuestionFeedbackForTheStudents` but does not
save it, which is why the option explanations go into `\SolutionBody`.

**Guarantees.** The parser reports none of these mistakes; it reads the item wrong instead. So
the writer guarantees them, and fails naming the text's path (for example
`questions[1].options[2].text: unbalanced braces`) without writing anything:

- answers are `\answer text` with a space and no braces, one per line;
- braces are balanced in every text, counting those of `\{` and `\}`: one `}` too many cuts the
  field short, one `{` too many makes the upload preview fail;
- no text holds a command of the format (`\begin{Indexed`, `\end{Indexed`, `\Question`,
  `\Solution`, `\answer`, `\correctanswer`, `\penalizableanswer`); the title only gets this check,
  as it sits in a comment;
- exactly one `\correctanswer` per block;
- `IndexedSolution` sits inside `IndexedQuestion`, after `\QuestionContentUnits`.

A content unit must be non-empty, on one line and free of braces. ExerPlaza links it to an existing
unit only on an exact name and language match.

**Reuse of the QTI model.** `tex.py` takes its items from `quiz_to_qti` (12.4), so selection,
order, single choice only, one correct option and the field checks are the same as for QTI, and a
quiz exports the same items in both formats. Its errors are re-raised as `TexExportError`. The
review status, the stem's origin and `generation` come from the model's untyped `metadata`, and
`tex.py` checks each value it reads. The XML writers are not used.

---

## 13. Module layout and dependencies

### 13.1 Layout

```
src/questxtract/
  __init__.py      public API: process(), run(), STAGES, read/write, filters, export_qti(),
                   export_tex()
  models.py        Word, AsrSegment, Transcript, Sentence, Segment, VisualContext,
                   QuestionCandidate, QAPair, Context, SchemaError, the @stage decorator
  config.py        one flat frozen Config dataclass, every default
  media.py         1 ingest: probe, source hash, audio extraction
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
  qti_v3.py        QTI 3.0 XML and the ZIP package
  tex.py           ExerPlaza TeX from the QTI model
  filtering.py     time and text filters, experimental topic filter
  evaluate.py      labelled-set metrics
  schema.py        Question, a standalone shape the pipeline does not produce
  pipeline.py      STAGES, run(), process(), cache, progress
  quiz.py          quiz.json from questions.json: selection, one request per question
  cli.py           process / transcribe / filter / eval / quiz / qti / tex

tests/            flat, one test module per source module, plus conftest.py
```

Tests sit flat beside the modules they cover, and pytest runs with `--import-mode=importlib` so
same-named files never collide. `tests/data/` holds the labelled set (4.1), not code.

Each stage module exposes exactly one `@stage`-decorated function as its public surface;
everything else in it is private. That is the modularity claim, and `tests/test_pipeline.py`
checks it: `grep '@stage'` over the stage modules returns the contents of `STAGES`.

### 13.2 Dependencies

**Base** (`pyproject.toml`): `av` (container probe and audio extraction), `faster-whisper` (ASR),
`numpy` and `onnxruntime`, plus `tomli` on Python 3.10 for the config file (3.11 has `tomllib`).
`ctranslate2`, `tokenizers`, `huggingface-hub` and `tqdm` arrive transitively through
faster-whisper.

**Extras:** `[llm]` -> `openai` (the LLM detector, vision and the quiz: one dependency; vision has
its own endpoint and key, 10.1). The CLI's
progress bar is a few lines of `print`, not `tqdm`.

**No system binaries** (no ffmpeg) and **no torch** (so no CUDA wheels, transformers or pyannote).
The install is a couple of hundred MB, which keeps NFR1's Windows leg easy.

**A neural sentence splitter** (3.1's last rung) would not change that: `wtpsplit-lite` runs on
ONNX with four dependencies we already have through faster-whisper. Its real cost is 428 MB of
weights for `sat-3l-sm`, against a stage that costs about 20 ms today; decide on that number once
the eval harness shows the splitter losing questions.

### 13.3 Status

What the code does today, and the open issue for each missing piece. "Stub" means the module, its
name, its signature and its `@stage` registration are real; only the algorithm is missing.

| Module | Stage | State | Issue |
|---|---|---|---|
| `models.py`, `config.py`, `pipeline.py` | the contract itself | done | |
| `media.py` | 1 ingest | done: probe, source hash, audio extraction to a temporary WAV | |
| `transcribe.py`, `asr.py` | 2 transcribe | done: faster-whisper behind the `AsrBackend` registry; the vision-informed second pass (2.4) is not planned | |
| `sentences.py` | 3 sentences | done | |
| `segments.py` | 4 segments | done | |
| `vision.py` | 5 layout, vision, attach_visual | done: the layout probe (6.2), frame sampling (6.3), stable views with one keyframe each (6.4, static), captioning with prompt v0 (6.5) or the OCR reader (6.5), reusing the reading of a page that comes back (6.5) and the interval join; the ink curve (6.4) is not built, de-occlusion (6.3) is not planned | #133 |
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
| `qti.py`, `qti_v3.py` | not a stage | done: the QTI 3.0 package of a `quiz.json` (12.4) | |
| `tex.py` | not a stage | done: the ExerPlaza TeX of a `quiz.json` (12.5); the upload itself is a separate issue | |
| `cli.py` | not a stage | `process` (with `--quiz`), `transcribe`, `filter`, `eval`, `quiz`, `qti` and `tex` work ([cli.md](cli.md)); `process --vision --detector llm --quiz` runs every stage and the quiz in one run, tested offline | |

Every stub carries a `ponytail:` comment naming its issue, so `grep -rn 'ponytail:' src/` is the
running list of what this document describes and the code does not yet do.

What lands when: [GitHub milestones](https://github.com/mad201802/QuestXtract/milestones).

---

## 14. Non-goals

Explicit boundaries, so the package does not grow into them by accident.

- **The teacher review/edit UI** ([issue #15](https://github.com/mad201802/QuestXtract/issues/15))
  is a separate application. This package's part is that its JSON round-trips (section 12.2).
- **ExerPlaza upload.** `tex` writes a file for ExerPlaza's TeX upload (section 12.5); nothing in
  `questxtract` talks to ExerPlaza.
- **No LMS upload**; `qti` only writes a ZIP (section 12.4).
- **The labelled dataset** is test
  material under `tests/data/`, not an importable subpackage. The *harness* that reads it is
  `evaluate.py` (section 4), because NFR2 is a CI gate and the threshold sweep is library code.
- **YouTube URLs** are out of scope: the CLI takes a local file. If URL download is ever added it
  belongs in `media.py`.
- **Speaker identification by name** and diarization (section 2.6).
- **Manim animations** of the board are optional and outside the core delivery
  ([design_decisions.md](design_decisions.md), D21). The package has no Manim code or dependency.

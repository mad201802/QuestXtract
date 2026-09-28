# The QuestXtract Pipeline

How QuestXtract turns a lecture recording into a set of questions with timestamps, answers and
provenance.

Related: [general_requirements.md](general_requirements.md) for the requirement ids (FR/NFR)
referenced throughout.

---

## 1. Overview

### 1.1 What the pipeline does

In: a lecture video. Out: a JSON file of the questions that were actually asked, each one with a
timestamp, a confidence, an answer span where the lecture contains one, and enough provenance to
jump back to the exact seconds in the recording.

The package's job stops there. Everything downstream - the editor, the quiz UI, the LMS glue -
lives in a wrapper around it. That boundary is why the package may reach the network at all
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
| [MIT 18.100B Real Analysis, L1](https://www.youtube.com/watch?v=BWvjy0a1Ffg) | chalk, sliding blackboards, dense LaTeX math, **camera pans and zooms to follow the lecturer**, few audible student questions | vision, LaTeX, manim, ASR on math jargon |
| [TTK4225, 2026-09-14, P03·12 — impulse responses and BIBO stability](https://www.youtube.com/watch?v=ufmTACqiE78) | digital tablet notes (OneNote/PDF sheet), single instructor, control-theory derivations with heavy Socratic back-and-forth | question detection on implicit-cue and self-answered questions, ASR on control-theory notation |
| [TTK4225, 2026-09-15, P03·14 — sinusoidal fidelity](https://www.youtube.com/watch?v=IQa2TSykWdc) | digital tablet notes (OneNote/PDF sheet), same course, frequent student Q&A | echo pairs (student question + instructor restatement), habitual confirmation-tag handling ("...right?"/"...true?") |

Two consequences that show up repeatedly below:

- **There is no single video layout.** Merged single-camera, separate speaker plus screenshare,
  blackboard-only, digital tablet notes, picture-in-picture. Section 6 uses a VLM layout probe
  rather than a classifier.
- **The TTK4225 instructor uses a habitual confirmation tag** ("...right?"/"...true?") very
  frequently across both recordings — see `docs/annotation_guidelines.md` section 2 for how this is labelled
  consistently rather than case by case.
- **In a classroom the students are often inaudible.** What *is* in the audio is the lecturer
  repeating the question - "so the question is whether...". That makes the implicit-cue path
  (section 7.1b) a detection path on classroom recordings, not a nice-to-have.

### 1.3 Stage order

```
                      .mp4 recording
                              |
  1 ingest                    |  validate, probe via PyAV, hash for cache key
                              v
  2 transcribe                |  faster-whisper decodes the container directly
                              |  (VAD on, word_timestamps on) -> Word[] + AsrSegment[]
                              v
  3 sentences                 |  split on terminal [.!?] at word boundaries; run-ons
                              |  broken at pauses, then capped -> Sentence[]
                              v
  4 segments                  |  gap/duration/length breaks -> Segment[]   (FR11 topic filter)
                              v
  5 vision (optional)         |  layout probe -> 1 fps frames -> board states -> VLM
                              v
  6 detect                    |  lexical scoring + vetoes -> QuestionCandidate[]
                              v
  7 answer                    |  forward window + far-window rescue -> AnswerCandidate[]
                              v
  8 rank + score              |  rule ranking, confidence, dedupe -> QAPair[]
                              v
  9 judge (optional)          |  one LLM pass: keep/revise/reject, refine, draft answer
                              v
 10 export                    |  questions.json (+ quiz.json, separate generative stage)
```

Stages 1-8 are pure Python plus one local model. Stages 5 and 9 are the only stages that touch an
LLM endpoint, and both are off unless one is configured. Every expensive stage is cached
(section 11) and reports progress (FR9).

This diagram is not a description of `pipeline.py`, it **is** `pipeline.py`: the `STAGES` list in
section 11.1 is this sequence, rows 1 to 9, plus the one split noted there. Row 10 is not a stage
- export runs on the finished `Context`.

### 1.4 The rule path and the judge

Detection, answer search, scoring and dedupe are cheap, inspectable rules. On top of them sits
**one optional LLM judge pass** (section 10) that reviews the surviving candidates together. The
rule path is the offline fallback: a run with no endpoint configured still produces candidates.

**Hard constraint (requirement 4.5, "the model shall not create questions on their own").** The
judge may **reject, rescope or rewrite** a candidate, and it must cite the sentence ids it is
working from. It may never add a candidate. Quiz generation is generative and lives in a
separate, separately named stage (section 12.3), so the boundary is visible in the code and in
the output, not only in a prompt.

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

Only `.mp4` is accepted today. There is no separate audio-extraction stage: the faster-whisper
backend (`asr.py`) calls `media.extract_audio`, which decodes the first audio track with PyAV into
a temporary 16 kHz mono WAV and deletes it when transcription ends. The WAV never reaches the
cache, and there is no system binary in the dependency list.

One recording, one clock. No multi-input sessions, no session-absolute clock.

> **Decision.** `questxtract/media.py`. Config: none.

### 2.2 Transcription

`faster-whisper`, one backend, no abstraction layer over it.

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = BatchedInferencePipeline(WhisperModel(model_size, compute_type="int8"))
segments, info = model.transcribe(
    str(audio_path),  # 16 kHz mono WAV from media.extract_audio
    beam_size=5,
    language="en",  # FR: English lectures (4.2); None to auto-detect
    vad_filter=True,  # skip silence: fewer hallucinated segments, faster
    word_timestamps=True,  # the timings everything downstream is built on
    initial_prompt=_PUNCTUATION_PROMPT,
)
```

Three things here are load-bearing.

**`vad_filter=True`** is the cheapest quality win available: Whisper hallucinates confidently over
silence, and a lecture recording is full of it.

**`word_timestamps=True`** is the only source of timing in the pipeline (see 2.3). It also gives
us per-word confidences for free - and, since issue #14, is what per-word progress reporting
(11.3) rides on, not faster-whisper's own `log_progress` flag (that only writes to Python's
`logging` module, which isn't otherwise observable).

**`initial_prompt`** biases Whisper toward emitting punctuation, including question marks, which
is the strongest single detection signal we have (+0.40 in the scoring table). It is a weak lever,
though: `initial_prompt` conditions the first decoding window, after that it survives only through
`condition_on_previous_text=True`, and `prompt_reset_on_temperature=0.5` discards it whenever
temperature fallback trips - which happens on exactly the noisy stretches where punctuation is
worst. So the lever order for punctuation quality is:

```
1. model size          (turbo punctuates far better than small)
2. initial_prompt      (helps, decays)
3. a real splitter     (wtpsplit, only if 1 and 2 measurably fail)
```

`hotwords` is a vocabulary parameter, not a punctuation parameter; it belongs to 2.4, not here.

**Model size.** Default `large-v3-turbo` (faster-whisper alias `turbo`), 809M parameters, four
decoder layers, roughly 8x faster than `large-v3` at close to `large-v2` quality. On `int8` CPU
it is usable; on the server it is trivial. `base` for CI smoke tests, `small` if turbo's memory
footprint turns out to be a problem on a laptop.

> **Decision.** `questxtract/transcribe.py`. Config: `asr_backend` (`"faster_whisper"`),
> `model_size` (`"turbo"`), `language` (`"en"`), `beam_size` (5), `vad_filter` (True),
> `compute_type` (`"int8"`), `batch_size` (4 - lowered from 8 for issue #14's terminal progress
> bar: faster-whisper's batched pipeline only yields results once a whole batch finishes, so a
> smaller batch means more frequent, smaller bursts - measured roughly every 2.5s instead of
> every 8-9s - at the cost of ~9% more transcription time; worth reverting if a future backend's
> progress signal stops depending on batch granularity), `initial_prompt`. Per-segment `avg_logprob`,
> `no_speech_prob` and `compression_ratio` are kept on the segment: they are the cheapest
> hallucination detector we have and the eval harness uses them.
>
> **Swapping the engine.** Shipped with issue #8, revised from this section's original plan:
> the ASR *checkpoint* is still a config string (`model_size`), but the ASR *engine* - a hosted
> transcribe endpoint, WhisperX, Parakeet - is a class implementing `AsrBackend` (one method,
> `transcribe(source, config) -> list[dict]`), registered under a name with `@register_backend`
> in `asr.py` and selected by the `asr_backend` config string above. `resolve_backend` only
> turns that name into an instance; `transcribe.py` and the rest of the pipeline never import a
> specific engine. A string registry, not a `STAGES` swap, because the engine is what #14
> (progress) and #11/#12 (evaluation) need to vary independently of everything else in the
> stage.
>
> **Dev workflow.** No dev machine has a GPU, and `turbo` on CPU runs a 90-minute lecture in
> roughly the same wall-clock time again. So the cached `transcribe.json` for the reference
> recording is committed as `tests/data/<slug>.transcript.json` (#12), trimmed to the words and
> chunk spans the pipeline reads; the quality metrics above are dropped there, since nothing
> downstream of stage 2 reads them yet. Full ASR runs on the server or overnight; the inner loop
> for everyone working on sections 3 and 6 to 9 should never touch Whisper.

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

Dropping alignment is also what keeps the install small: `whisperx` pulls torch, torchvision,
triton, the CUDA wheels, pyannote and transformers; `faster-whisper` runs on standalone C++
CTranslate2. **There is no torch anywhere in our base install** (section 13.2).

### 2.4 Optional: vision-informed second ASR pass

Whisper mangles domain vocabulary, and a real-analysis lecture is nothing but domain vocabulary:
*supremum*, *Cauchy*, *epsilon*, *for all n greater than capital N*. The board has those words
written on it.

So: run section 6 on ~10 keyframes, collect the terms, re-transcribe with
`hotwords="supremum, Cauchy, epsilon, ..."`. One extra ASR run for a measurable win on MIT-style
lectures.

> **Decision.** Off by default, behind a flag, and enabled only once the eval harness shows ASR
> vocabulary errors actually costing us questions. It is the one place where the vision stage
> feeds *backwards* into audio.

### 2.5 Words are the atomic unit

There is no utterance layer. A sentence owns a half-open word index range, so its timings are
`words[i].start` and `words[j-1].end`, both real measurements, and its provenance is the range
itself. A timing is never interpolated, and a partition that is a slice cannot silently stop being
a partition, so no provenance validator is needed.

### 2.6 One speaker, no diarization

Speaker identification by name is out of scope (requirements, 1. Purpose & Scope) and a single
main speaker is assumed (4.2). There is no diarization, no speaker attribution and no acoustic
speaker check. In a classroom the students are often off-mic, so there is frequently no second
voice on the recording to diarize in the first place - the lecturer's repetition of the question is the signal,
and that is a lexical problem, not an acoustic one.

---

## 3. Sentences and segments

### 3.1 Sentences

ASR segments are not sentences. Whisper emits chunks bounded by its decoding window and by
silence, so one chunk routinely holds two questions or half of one. Detection needs units that
read on their own.

We attack the input rather than the output: a `turbo`-sized Whisper punctuates well, and a
punctuated transcript splits on its terminal punctuation. Rather than a regex over the joined text
plus a second pass to snap character offsets back to word indices, one walk over the word list
asks each word whether it ends a sentence, which is the same split with the snapping built in.

**Not every full stop is a boundary.** `_is_terminal` holds back three cases the plain
`[.!?]` test gets wrong, all common in lecture ASR:

| Case | Example | Why |
|---|---|---|
| abbreviation | `So Dr. Rudin defines it.` | a short suppression list; `Step 1.` still splits, digits are not abbreviations |
| single-letter initial | `for all n greater than capital N.` | the MIT 18.100B regime is nothing but these |
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
> `force_split_min_gap_seconds` (0.6) - which is the pause threshold generally, not only a
> force-split one.
>
> **Risk, stated plainly:** this bets on Whisper emitting question marks, and 2.2 already admits
> `initial_prompt` decays. The eval harness measures the bet directly via `question_mark_rate`
> (section 4.3; 1.00 on the reference recording, 7.3); between labelled runs, the stage's own
> `stats["sentences_unpunctuated"]` is the standing proxy, and a count that climbs means the
> splitter is guessing from silence alone. If it drops,
> the fix in order of cost is: a better `initial_prompt`, then a larger Whisper model, then a
> neural splitter as one function behind the same interface.
>
> **What that third rung actually costs.** Not a dependency - `sat-3l-sm` weights, 428 MB (13.2).
> Worth re-checking against the measured `question_mark_rate`, not against a guess, because the
> published advantage of neural segmenters is measured on ASR text with punctuation *and casing*
> stripped, which is not our input.

### 3.2 Segments

A segment is a contiguous run of sentences bounded by silence and by length. **Not a topic** -
nothing here measures what the run is about, and the code should not be read as if it did.

One mode, structural, closing on the first of:

- silence gap over `segment_max_gap_seconds` (2.0 s), strictly - a pause of exactly the
  threshold does not break, where 3.1's pause rule is inclusive; each matches its own wording
- a sentence that would push the run past `segment_max_duration_seconds` (90 s)
- a sentence that would push it past `segment_max_chars` (800)

Both length rules are measured with the candidate sentence *in*, so a closed segment is inside
its budget. The exception is a single sentence already over on its own, which this stage cannot
split - that is 3.1's job, and `max_sentence_words` is what bounds it.

**Why structural is enough, for now.** The three consumers this section used to claim need
different things, and only one of them needs topicality:

| Consumer | Wants | State |
|---|---|---|
| visual join (6.5) | time intervals | works; joins by interval intersection, no topic involved |
| QA scoring's segment-relation term (9.1) | topical relatedness | dropped before it was built |
| FR11 filter (#16) | a topic *label*, not a boundary | time filter done (`filtering.py`); topic is an experimental `topic=` hook that warns, since `topic` is `null` in the schema |

A semantic segmenter would improve exactly the two that need topicality, and neither exists: the
scoring term was dropped, and FR11's topic filter is a hook with no labels to match. It also
cannot move NFR2: `detect.py` reads `Sentence`, never `segment_index`, so better boundaries
change no detection number. The scoring term is dropped (9.1): segments are cut *on gaps*, so "same
segment" restates the gap and distance terms already in 9.1's table. That collinearity is a
property of structural segmentation, not a bug in it.

**If that changes, the first rung is not embeddings.** Classical lexical cohesion - token
overlap between adjacent windows, TextTiling-style - is stdlib-only and needs no dependency,
unlike sentence embeddings. Try that before adding weights.

**The granularity this section is papering over.** One 90 s unit is being asked to be two
things at once: a fine time window for the board join (a board state lives two minutes while the
lecturer talks about it for six) and a coarse topic for FR11 (a lecture has perhaps 5 to 15).
The third it used to be asked for, a local-relatedness signal for scoring, is dropped (9.1). If
FR11's topic filter ships, the answer is a second, coarser layer over these, not a retuned
threshold on them. Building that layer now would be building for consumers that do not exist.

> **Decision.** `questxtract/segments.py`. Thresholds kept at 2.0 s, 90 s and 800 characters,
> measured against the three transcribed recordings of #12 (`mit_18_100b_l1`,
> `ttk4225_2026_09_14`, `ttk4225_2026_09_15`): `build_segments` over each one's sentences at
> the current 3.1 config, force-split included, varying only `segment_max_gap_seconds`
> ([#41](https://github.com/mad201802/QuestXtract/issues/41)).
>
> | `segment_max_gap_seconds` | 2 s | 3 s | 4 s | 6 s | 8 s |
> |---|---|---|---|---|---|
> | segments closed by a pause (%) | 96 / 98 / 95 | 81 / 91 / 75 | 74 / 81 / 64 | 48 / 66 / 45 | 36 / 50 / 35 |
> | median segment (s) | 22 / 15 / 16 | 28 / 24 / 38 | 40 / 29 / 46 | 65 / 49 / 59 | 75 / 64 / 61 |
>
> At 2.0 s that is 123 / 66 / 86 segments, closed by gap 117 / 64 / 81, by duration 3 / 0 / 0
> and by chars 2 / 1 / 4; the longest is 85 s.
>
> The gap count does dwarf the other two, which this note used to read as "too tight". Measured,
> it reads the other way: a looser threshold finds no second, natural boundary. It hands the
> cutting to the length caps, which close at an arbitrary 90 s or 800 characters, mid-thought,
> and by 6 s they close about half of all segments. At 2.0 s the caps are the rare backstops
> they were meant to be. The one working consumer, the visual join (6.5), wants short
> intervals, and 15 to 22 s is well inside a board state's lifetime. A coarser unit for FR11 is
> still the second layer argued for above, not a looser threshold here. The chalkboard worry is
> real - writing pauses do end segments - but it costs nothing until a consumer needs segments
> to mean something.
>
> `stats["segments_closed_gap" | "_duration" | "_chars"]` stay as the check on a new recording:
> length caps closing more than a few segments is the sign it does not fit these numbers.

---

## 4. Evaluation harness

NFR2 gives a number to hit (recall >= 50% on the team's labelled set), so ground truth comes
first and the metric runs in CI. Sections 6 through 9 are blind without it.

### 4.1 The labelled set

One JSONL file per labelled recording, `tests/data/<name>.labels.jsonl`:

```json
{"start": 1284.2, "end": 1288.9, "text": "why does a stable sort matter here", "kind": "content"}
```

next to the transcript it is timed against, `tests/data/<name>.transcript.json`. A label's `text`
quotes that transcript and its `start`/`end` are snapped to the quoted words
(`scripts/scratch_to_labels.py --transcript`), so labels and predictions share one clock. The
first, hand-timed pass (#11) sat 3 to 55 s off it, which no overlap rule survives.

`kind` is `content` (a real question we want), `rhetorical`, `check_in` or `procedural`. Labelling
the negatives is what lets us measure the thing that is actually hard - the false positive rate -
rather than only recall.

Two properties of the set matter (the committed set is labelled differently, see below):

- **Pre-annotated.** Labellers correct a high-recall detector dump (`?` OR wh-head OR aux-head, no
  scoring, no vetoes) instead of scrubbing 80 minutes of video. Roughly 5x faster.
- **Blind control slice.** Two 10-minute windows per recording labelled from scratch, with no
  seed. Pre-annotation cannot surface what the detector never proposed, so without a blind slice
  the recall number measures the seed, not the system.

Coverage: the reference recording, 28 `content` questions, 34 entries in all.

**What the committed set actually is.** The transcript was labelled exhaustively, sentence by
sentence, with no seed, so every entry is `blind` and `blind_slice_recall` equals recall; the
seed workflow above stays the cheap path for longer recordings. Exhaustive labelling is what
makes precision measurable: an unlabelled prediction is a real false positive. The ground truth
is the transcript, not the audio - a question the ASR never transcribed (an off-mic student)
is not labelled, because no text rule can reach it; `tests/data/README.md` lists the
ones dropped for that reason. The MIT labels were LLM-drafted from the transcript and are
pending human review.

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
blind_slice_recall: recall restricted to the control windows from 4.1
```

`question_mark_rate` keeps section 3.1's bet honest. If it drops, the regex splitter is the first
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
the reference recording. It runs in well under a second, so there is no separate short
recording.

> **Decision.** `questxtract/evaluate.py`. The report includes a threshold sweep over
> `min_confidence`, so the NFR2 tradeoff is visible rather than assumed.

---

## 5. Data model

Five dataclasses plus the visual ones, in `models.py`, and the two that make the pipeline
pluggable - `Config` and `Context` - in `config.py` and `models.py` respectively (section 11.1).

Every frozen type is `kw_only=True` as well. Keyword-only construction removes a whole class of
silent bugs when fields are added: positional `Question(...)` calls once stored a reasons tuple in
a newly added field without error.

```python
@dataclass(frozen=True, kw_only=True)
class Word:
    text: str
    start: float
    end: float
    confidence: float


@dataclass(frozen=True, kw_only=True)
class AsrSegment:
    """A raw ASR decoding chunk. NOT a Segment - that is a run of sentences (3.2)."""

    text: str
    start: float
    end: float
    # the cheapest hallucination detector we have; the eval harness reads these (2.2)
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


@dataclass(frozen=True, kw_only=True)
class Transcript:
    words: tuple[Word, ...]  # the atomic unit, section 2.5
    asr_segments: tuple[AsrSegment, ...]
    language: str
    duration: float


@dataclass(frozen=True, kw_only=True)
class Sentence:
    id: str  # content hash, not an index: ids survive a re-transcription,
    # which is what lets human edits merge back in (FR5)
    text: str
    word_start: int  # half-open range into Transcript.words
    word_end: int
    start: float
    end: float
    segment_index: int


@dataclass(frozen=True, kw_only=True)
class Segment:
    sentence_start: int
    sentence_end: int
    start: float
    end: float
    text: str
    visual: tuple[VisualContext, ...] = ()  # board states overlapping this segment


@dataclass(frozen=True, kw_only=True)
class VisualContext:
    start: float  # board-state lifetime, NOT a segment's span
    end: float
    keyframe_path: str
    description: str  # what the VLM saw: diagram, derivation, slide
    text: str  # transcribed board/slide text
    latex: str | None  # formulas, when present. manim generation reads this.
    confidence: float


@dataclass(frozen=True, kw_only=True)
class QuestionCandidate:
    sentence_id: str
    text: str  # the refined focus, may be a clause of the sentence
    score: float
    question_type: str  # what | why | how | yes_no | definition | other
    reasons: tuple[str, ...]  # every rule that fired, positive and negative


@dataclass(frozen=True, kw_only=True)
class QAPair:
    question: QuestionCandidate
    answer_sentence_ids: tuple[str, ...]  # empty when no answer was found
    answer_text: str
    confidence: float
    start: float  # question start, the FR7 timestamp
    end: float
    verdict: str = "unjudged"  # unjudged | keep | revise | reject
    reasons: tuple[str, ...] = ()
```

And the one mutable object in the system, the value every stage takes and returns:

```python
@dataclass
class Context:
    source: Path
    config: Config
    transcript: Transcript | None = None
    sentences: tuple[Sentence, ...] = ()
    segments: tuple[Segment, ...] = ()
    visual: tuple[VisualContext, ...] = ()  # board states; joined to segments in a later stage
    candidates: tuple[QuestionCandidate, ...] = ()
    pairs: tuple[QAPair, ...] = ()
    stats: dict[str, int] = field(default_factory=dict)
    progress: Progress | None = None  # section 11.3; mid-stage reporting
    source_hash: str = ""  # set by ingest; the cache key for every later stage
    duration: float = 0.0
```

Three rules:

**`reasons` is never dropped.** Every rule that fires, positive or negative, appends a stable
string. This is what makes review possible and what makes a wrong result debuggable without a
debugger. It is also the judge's input.

**Do not drop `None` on serialisation.** We emit a fixed key set, always, `null` included, so
consumers never have to defensively `.get()` a missing `answer_text`.

**`VisualContext` owns its own time range.** A board state's lifetime is not a speech segment's
span - the lecturer writes for two minutes and talks about it for six. Board states therefore
live on `Context.visual` on their own clock, and `Segment.visual` is the tuple of states that
*overlap* that segment, computed by interval intersection in the `attach_visual` stage. Nailing
one `VisualContext` to one `Segment` at capture time would bake in a wrong join that every
consumer inherits, and would make the vision stage uncacheable on its own (section 11.1).

---

## 6. Vision

Optional enrichment of the audio with what is on the board. On unless no endpoint is configured.

### 6.1 The regime problem, and why not OCR

Two different worlds hide under "read the board":

| Regime | Hard part | What works |
|---|---|---|
| slides, screencast, beamer | essentially nothing | near-duplicate frame dedupe, then OCR |
| chalkboard, whiteboard, handwriting | everything | see below |

FR4 says "animation/drawing" and the reference recording is MIT chalk, so we are in
regime 2 and must also survive regime 1.

Tesseract, PaddleOCR and EasyOCR are printed-text models. On chalk they are close to useless, and
more importantly **they cannot represent a diagram at all** - an OCR engine handed a commutative
diagram or a state-transition graph returns a bag of stray glyphs.

A VLM reads handwriting, reads diagrams, reads slides, and emits LaTeX, in one call. We already
accept an optional LLM dependency for the judge (section 10), so the marginal dependency cost of
using it for vision too is **zero**, and it replaces OCR, layout analysis and formula recognition
in a single move.

`vision` is one entry in the `STAGES` list with the same `Context -> Context` signature as every
other stage, so a different board reader - a local OCR pass on the slides regime, a hosted
document-AI endpoint, a stub that loads pre-captioned frames in tests - replaces that one entry.

### 6.2 Layout probe

Merged single-camera, separate speaker plus screenshare, blackboard-only, picture-in-picture -
and whatever the next recording does. There is no layout classifier; a heuristic here would be a
permanent bug farm across mixed sources.

Sample three frames spread across the recording, send them in one VLM call:

```json
{"content_region": [0.32, 0.05, 0.66, 0.80], "kind": "blackboard"}
```

Normalised bbox plus a regime label, cached by source hash. Fallback on any failure: the full
frame, `kind: "unknown"`. One call per video, and it handles layouts we have not seen.

### 6.3 Frame sampling and de-occlusion

Sample at 1 fps through PyAV - already a dependency, no ffmpeg subprocess - crop to
`content_region`, grayscale, downscale to ~480p.

The lecturer stands in front of the board. Where the camera is **static**, a per-pixel temporal
median over a window of samples removes anything that moves and leaves the board:

```python
board = np.median(np.stack(window_frames), axis=0).astype(np.uint8)
```

Ten lines of numpy, and nothing downstream reads well without it.

Where the camera **pans** - which MIT 18.100B does, since an operator follows the lecturer - the
median is meaningless. So gate it:

```
camera motion = median absolute frame-diff over sampled frames
  low   -> temporal median de-occlusion
  high  -> skip it, pick the lowest-motion frame per board state, hand the raw frame to the VLM
```

VLMs cope with a person standing at a board; they just cannot read what is behind them. No
homography stabilisation, no OpenCV.

### 6.4 Board states

Adaptive-threshold the (de-occluded) board and count ink pixels per sample. The curve says
everything:

```
monotone rise    -> the lecturer is writing
sharp drop       -> erase event, the board state ended
local maximum    -> the completed board, this is the keyframe
```

Cut the recording into **board states** between erase events. A 90-minute lecture gives roughly
20-60 of them. That, not a scene cut and not a speech segment, is the real visual unit.

On slide decks the same curve works unchanged: a slide transition is a large ink delta in both
directions, so "board state" and "slide" fall out of one implementation.

Whether a long derivation should split into several states mid-board is open; 1 fps sampling with
erase-event segmentation is the starting point.

### 6.5 Captioning

One VLM call per board state: the keyframe, plus the transcript text overlapping its interval,
plus the previous state's description and the ink-delta mask.

```
image + local transcript + previous description + delta crop
    -> description, text, latex, what_changed
```

The delta is a numpy diff of two masks we already have, and the "derivation timeline" it produces
is what manim generation consumes. That is why `latex` is in the schema (section 5): prose
descriptions do not compile.

Cached per keyframe hash, so re-running costs nothing.

### 6.6 Consumers

- **Detection** (section 7): a question whose overlapping board state has text sharing content
  tokens with it gets a small positive context delta. A lecturer pointing at the board while
  asking is asking about the board.
- **The judge** (section 10): `description`, `text` and `latex` go into the prompt as segment
  context, which is where they are worth the most - "what does this term mean?" is unanswerable
  from audio alone and trivial once you can read the board.

> **Decision.** `questxtract/vision.py`. Config: `vision_enabled` (False unless an endpoint is
> configured), `vision_fps` (1.0), `vision_model`, `vision_max_states` (80).
>
> **Offline.** `vision_enabled` works on a local device by pointing `base_url` at a local
> OpenAI-compatible server (section 10.1). Quality is expected to be noticeably worse than the
> hosted path, and that is an accepted tradeoff, not a bug. There is no local OCR engine.

---

## 7. Question detection

Purely lexical and orthographic. No classifier, no prosody. The rules are inspectable and every
decision leaves a reason code, which is what makes both the review workflow and the LLM judge's
job possible.

**Why rules, not a model.** Deciding whether a sentence is interrogative is the easy half and is
lexical: in meeting speech, lexico-syntactic features carry question detection and pitch only
adds to them (Boakye, Favre, Hakkani-Tür, ASRU 2009). Deciding whether it is a *content*
question is the hard half, and trained models do not solve it either: a teacher-question
detector trained on classroom ASR reaches weighted F1 0.59 (Blanchard et al., SIGDIAL 2016), and
rhetorical questions are identified at F1 0.54 only once the neighbouring utterances are
features (Bhattasali et al., ACL 2015). With 42 labels there is nothing to train on anyway. So
the rules do the lexical half and keep the known noise out; the content call is the judge's
(section 10), which sees the context those papers found necessary.

### 7.1 Extraction

One candidate per sentence (3.1), scored in 7.3.

**The interrogative head.** A question's first word is rarely the sentence's first word in
lecture speech: *"So what does it mean ...?"*, *"Now, what is the challenge ...?"*. The head is
the first word past leading discourse markers (`so`, `and`, `but`, `now`, `then`, `okay`, ...),
and past question preambles (*"my question is (that)"*, *"the question is/was"*, *"I'd like to
ask"*, *"I was wondering (if)"*), which announce a question without opening it,
and for a sentence ending in `?` also the first word of its last comma clause, because ASR runs
the preamble into the question: *"If you have the data, does that mean ...?"*. Without the `?`
the clause rule stays off, or every *", which is ..."* would read as a wh-question. Contractions
stay whole, so negated auxiliaries (*"Don't we ...?"*) open questions.

**The answer seed.** Specified as "text *after* the `?`", which contradicts section 3.1: stage 3
splits *at* terminal punctuation, so a sentence never holds text after its own `?`.
It is redefined as the immediately following sentence, taken when the silence before it is at
most `answer_seed_max_gap_seconds` (1.0 s) and it is not itself a question; tags and check-ins
get no seed ([issue #38](https://github.com/mad201802/QuestXtract/issues/38)). Measured on the
three reference recordings: about 4 in 5 seeds of labelled content questions are real answers up
to 1.0 s, falling to about half beyond it.

**Not implemented, and why.** Two paths from the original design have no labelled failure
behind them yet:

- *Implicit cue, no `?`* (*"the question is whether ..."*). Expected on classroom recordings
  where students are off-mic. A lecturer's restatement usually reaches the ASR with its `?`,
  and an off-mic student question is not in the transcript at all, so no text rule reaches it.
- *Missing `?` recovery* (an interrogative head **and** a local answer slot). Whisper `turbo`
  with the punctuation prompt marked every labelled content question that is in the transcript;
  the 35 `?`-less head-word sentences on the measured recordings were all statements.

Either lands when a labelled question is lost to it; on the reference recordings the harness
(7.3) shows one, the `?`-less cut-off *"Can this one be..."*. The rules are follow-up work, not
part of #12.

### 7.2 Vetoes

Hard rejects, before scoring: empty text, and text over `max_question_chars` (350 - raised from
the planned 220 as a quick, untuned experiment against real lecture audio).

The two vetoes of the original design both only fire on sentences *without* a `?` -
declarative "what" (*"what we call ..."*) and a causal clause after a question (*"Because the
cache is cold."*). A head word alone now stays under the gate (7.3), which rejects both shapes
without a pattern list, so neither is implemented.

### 7.3 Scoring

Additive, clamped to [0, 1]:

```
  +0.60  mark      ends in '?'
  +0.40  wh        interrogative head is a wh-word
  +0.40  aux       interrogative head is an auxiliary, negated ones included
  -0.40  tag       the sentence, or its closing clause, is only a confirmation tag:
                   "Okay?", "True, you see?", "..., right?" - unless what precedes the tag
                   has its own interrogative head ("Why is this not a field, right?")
  -0.40  check_in  classroom management: "who here", "want to tell me", "your name", ...,
                   and "is that clear", "make sense" unless content follows them
                   ("Why does it make sense to ...?" is a question)
```

Gate: `min_question_score` (0.45). What that does with the unanswered confidence of 9.2
(`0.55 x score`, emission gate 0.40):

| Sentence | Score | Candidate | Emitted |
|---|---|---|---|
| `?` and a head | 1.00 | yes | yes, 0.55 |
| `?` only | 0.60 | yes | no, 0.33 |
| check-in with `?` and a head | 0.60 | yes, reason kept | no, 0.33 |
| tag | 0.20 | no | no |
| head, no `?` | 0.40 | no | no |

The tag and check-in lists are English (requirement 4.2), grown from the labelled recordings
rather than written up front, and deliberately short: they only keep the obvious noise out of
the judge's input. Tags follow the annotation guidelines, which label them `rhetorical`: a
penalty with a reason code, not a veto.

**Measured at #12** with the harness (section 4), default config, `questxtract eval tests/data/`:

| Recording | Content labels | TP | FP | FN | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| MIT 18.100B L1 | 28 | 28 | 1 | 0 | 1.00 | 0.97 | 0.98 |
| TTK4225 2026-09-14 | 39 | 33 | 21 | 6 | 0.85 | 0.61 | 0.71 |
| TTK4225 2026-09-15 | 60 | 52 | 42 | 8 | 0.87 | 0.55 | 0.68 |
| **Pooled** | 127 | 113 | 64 | 14 | **0.89** | 0.64 | 0.74 |

NFR2 holds with room. `question_mark_rate` is 0.99: Whisper `turbo` with the punctuation prompt
put a `?` on every labelled content question but one, a cut-off *"Can this one be..."*, so 3.1's
bet holds on these three. The sweep is flat from 0 to 0.30 (recall 0.99, precision 0.53) and
from 0.35 to 0.55 (0.89 / 0.64), and empty above 0.55, the unanswered ceiling of 9.2: the gate
only ever separates `?`-with-head (1.00) from `?`-only (0.60) sentences.

*Missed (14, all TTK4225).* Thirteen have their `?` but no head where 7.1 looks for one, so they
score 0.33 and fall under the gate. Six put the wh-word at the end as an answer slot - *"And
this is going to be what?"*, *"The integral of this violet here is going to be what?"*. Three
carry the head in an earlier comma clause than the last - *"The area of an exponential in time,
this area here, is it finite, infinite, or you do not know?"*. Two open on a preposition that 7.1's head
search does not skip (*"For which age are we guaranteed ...?"*), and two were merged by 3.1 with a preceding
fragment (*"Unit step function, u. Is it absolutely integrable?"*, *"Find z. How do you find
z?"*). The fourteenth is the `?`-less cut-off above. An end-slot *what* is the obvious next rule.

*False positives (64).* 43 hit `check_in` labels, nearly all this lecturer's own check-in forms
(*"Do you see it, people?"*, *"How much are we together, people?"*, *"Are they clear?"*) that the
check-in list, grown from the earlier recordings, does not yet hold. 18 hit `rhetorical` ones:
concept-free framing (*"So how do we do that?"*, *"What does it mean?"*) and rhetorical
confirmations (*"Doesn't it go like this?"*). 3 hit `procedural` ones (*"Shall we already watch
the video of the kid?"*, *"Am I recording?"*). None hit an unlabelled sentence.

The confirmation tags (MIT's *"..., right?"*, TTK4225's *"True?"* and *"Okay?"*, several a
minute) all stay out: none of the 64 is a tag, so the tag penalty does its job. Growing the
check-in list with the TTK4225 forms is the cheapest precision win left
([#70](https://github.com/mad201802/QuestXtract/issues/70)).

*After the answer seed (#38).* An answered question now carries its full score, so confidence is
graded and the sweep is no longer empty above 0.55:

| Recording | Content labels | TP | FP | FN | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| MIT 18.100B L1 | 28 | 28 | 1 | 0 | 1.00 | 0.97 | 0.98 |
| TTK4225 2026-09-14 | 39 | 34 | 26 | 5 | 0.87 | 0.57 | 0.69 |
| TTK4225 2026-09-15 | 60 | 59 | 49 | 1 | 0.98 | 0.55 | 0.70 |
| **Pooled** | 127 | 121 | 76 | 6 | **0.95** | 0.61 | 0.75 |

The sweep holds 0.95 / 0.61 from 0.35 to 0.55, then 0.49 / 0.62 at 0.60 and 0.42 / 0.68 from
0.65 up. Eight more content questions pass the gate; so do eleven more check-ins, whose next
sentence the seed takes as an answer because they are missing from the check-in list (#70).

> **Decision.** `questxtract/detect.py`, cue tables inline as module constants.

---

## 8. Answer search

Three channels, composing rather than competing.

**Adjacent seed.** The sentence immediately after the question's own, at a near-zero gap
(`answer_seed_max_gap_seconds`, 1.0 s) and not itself a question.
Distance 0. The Socratic case, and the highest-precision answer shape we have. Was specified as
"text after the `?` in the question's own sentence", which 3.1's splitter makes impossible; see
7.1(a) and [issue #38](https://github.com/mad201802/QuestXtract/issues/38).

**Forward window.** Walk forward 1 to `answer_window_sentences` (3), emitting the growing
consecutive span at each distance. At the window edge, append one continuation sentence when the
answer clearly runs on (previous text ends in `,;:` or next text starts with a connective or
lowercase). Stop with an explicit, exported reason: `window_exhausted`, `competing_question`,
`gap_limit` (> 45 s), `span_limit` (> 3 sentences), `duration_limit` (span > 45 s).

**Far-window rescue.** For questions answered much later, after a digression. Deliberately hard
to trigger:

- question score at least `deferred_min_question_score` (0.62)
- rhetorical, poll, tag and fragment questions excluded outright
- no strong local answer (a local answer scoring >= 0.52 within 10 s with shared keywords blocks
  the search entirely)
- a forward scan over the next `deferred_window_sentences` (28) finds a sentence sharing at least
  two content tokens with the question

Survivors are filtered on the principle that the further away an answer is, the more lexical
evidence it must carry: gap over 120 s needs 3 shared keywords or a shared number; gap over 60 s
needs a signal score of 0.28; distance over 12 needs 2 keywords.

There is **no speaker constraint on pairing**, and not only because we have no speaker layer:
lectures are full of self-answered questions, and requiring a speaker change would delete the
single most common QA shape in the corpus - which with off-mic students is *the entire shape*,
since the lecturer both repeats and answers the student's question.

> **Decision.** `questxtract/answer.py`. Three functions called in sequence behind the one
> `answer` stage entry (section 11.1): swapping the whole search replaces that entry, swapping one
> channel is an edit inside the module.

---

## 9. Scoring, ranking, dedupe

### 9.1 Answer scoring

Additive, and every term is recorded in `reasons`, so a final score decomposes after the fact.

```
Proximity      +0.18 base
               +0.28 / +0.20 / +0.12 / +0.06   distance 0 / 1 / 2 / beyond
               +0.12 / +0.06 / -0.08           gap < 4 s / < 10 s / beyond

Lexical        +up to 0.30  answer cue rules ("the answer is", "because", "so")
               +up to 0.15  shared content tokens
               +0.12        shared numbers
               -0.28        quantity question, answer contains no number
               -0.10        nothing shared at all

Span quality   +0.08        3 to 80 tokens
               -0.10        premise-only answer
               -0.08        span over 80 tokens
               -0.20        answer contains a '?'  (-0.35 if structurally a question)

Quality gate   -0.34        answer is a poll or backchannel
               -0.28        answer echoes the question
               -0.26        answer is boilerplate ("that is a good question")
               -0.24        answer is a continuation of the question
               -0.12        low-information answer
               +0.08        legitimate short Socratic answer
```

Answer text is trimmed before scoring: a meta opening ("that's a good question"), a trailing tag
("..., right?"), and an interior check-in clause all come off.

**No segment-relation term** ([#42](https://github.com/mad201802/QuestXtract/issues/42)). The
table once carried `+0.10 / +0.05 / -0.05` for same / adjacent / distant segment. It is dropped
before it was built, on 3.2's measurement:

- It double-counts proximity. A pause closes 95 to 98 % of segments, so "same segment" means "no
  pause over 2 s", which the gap term already prices and the distance term prices again.
- A pause after a question is not a signal of its own. The sentence after a labelled content
  question follows a pause over 2 s 29 / 18 / 13 % of the time, against 27 / 19 / 15 % after
  any sentence.
- The conditional variant, scoring only boundaries a length cap closed, would fire at 2 to 5 %
  of boundaries, and those are the arbitrary cuts.
- As a tie-break it decides nothing: the forward window's candidates already differ in
  distance.

If segments ever become topical (3.2), the term would carry a signal of its own; reopen it then.

### 9.2 Confidence

```
confidence = clamp(0.55 * question_score + 0.45 * answer_score
                   - 0.10 if a competing question was nearby
                   - 0.24 if the answer is itself a question)

label:  >= 0.75 high | >= 0.45 medium | else low
```

A question with no answer keeps `0.55 * question_score` and an empty answer span. It is still
exported: FR6 and FR7 ask for questions with timestamps, not for answered questions.

Gate: `min_confidence` (0.40), swept in the eval report.

One number the user can reason about (FR10), plus a judge verdict. No parallel quality score, no
risk band, no profile-scoped emission gates.

### 9.3 Dedupe

One overlap metric over stopword-filtered content tokens:

```
overlap(A, B) = |tokens(A) & tokens(B)| / min(|tokens(A)|, |tokens(B)|)
```

Two passes, both only over candidates starting within 90 seconds of each other:

- **Echo pairs.** A student asks and the lecturer restates - with off-mic students this is the
  *normal* case, not an edge case, and both directions get emitted. Detected by answer overlap >= 0.45 plus one
  cross pairing (left question vs right answer) >= 0.55. Keep whichever question echoes the other
  less.
- **Near duplicates.** Question overlap >= 0.72 with answer overlap >= 0.50, or question overlap
  >= 0.86 with answer overlap >= 0.35. Keep the higher confidence.

Losers keep a reason code rather than vanishing.

> **Decision.** `questxtract/score.py`, including dedupe.

---

## 10. The LLM layer

One module, one client, two jobs: the vision captioning of section 6 and the judge pass here.
Off unless an endpoint is configured; the pipeline produces a complete result without it.

### 10.1 One client, two endpoints

The `openai` SDK with a configurable `base_url`, which is what makes the offline story work
without a second code path:

```python
# server / default
OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # base_url=None
# offline / local device
OpenAI(api_key="local", base_url="http://localhost:11434/v1")
```

Any local OpenAI-compatible server - vLLM, llama.cpp's server, Ollama - serving a small VLM plugs
in with zero new code. That is requirement-level: offline results are allowed to be significantly
worse, but the visual path must be *available* on a local device, and a small local VLM behind
the same interface is how.

| Config key | Default | Role |
|---|---|---|
| `vision_model` | `gpt-5.6-luna` | cost-optimised; tens of board-state calls per lecture |
| `judge_model` | `gpt-5.6-terra` | balanced; one call per run |

Both are config values, not constants, and both get re-verified against the provider's model list
when the integration is wired - model ids move.

**This is the model seam** (section 11.1, rows 1 and 2). Swapping the VLM, swapping the judge
model, or pointing either at a local server is a `Config` change and no code. A provider that is
*not* OpenAI-compatible is an `if` on a `provider` field inside `llm.py`, in the two places that
build a request.

ASR stays local. `gpt-4o-transcribe` and friends are not used: `faster-whisper` gives word
timestamps, costs nothing, needs no network, and is the foundation of section 2.3.

### 10.2 What the judge does

One batched call per run (not per candidate). Input: every surviving candidate with its question
text, answer text, surrounding transcript window, overlapping board-state context where present,
and its `reasons` list. Output, per candidate:

```json
{
  "id": "q_0007",
  "verdict": "keep",
  "question_text": "Why does a stable sort matter for multi-key sorting?",
  "answer_text": "Because the second pass preserves the order of the first.",
  "answer_sentence_ids": ["s_3f2a91", "s_7c04de"],
  "confidence": 0.82,
  "reason": "genuine content question, answered two sentences later by the lecturer"
}
```

`verdict` is `keep`, `revise` or `reject`. `revise` means the judge rewrote the question text or
rescoped the answer span; the original is kept alongside it in the output so a human reviewer
(FR5, in the wrapper) can see what changed.

### 10.3 Constraints on the judge

1. **It may not add candidates.** The response is validated against the input id set; unknown ids
   are dropped and counted. This enforces requirement 4.5 in code, not only in the prompt.
2. **Answer spans must cite sentence ids that exist**, and the cited sentences must overlap the
   text it returned. A rewritten answer that does not trace back to the transcript is rejected
   and the candidate keeps its rule-based answer.
3. **It adjusts confidence within a band.** The final confidence is
   `clamp(0.5 * rule_confidence + 0.5 * judge_confidence)`, so a judge failure or a bad batch
   cannot push a candidate from 0.2 to 0.95.
4. **It is never required.** A timeout, a rate limit, a malformed response or a missing key logs
   a warning, marks every candidate `unjudged`, and the run completes on rule scores alone.

### 10.4 Why one call, not per candidate

A lecture yields tens of candidates, not thousands. One call sees them together, which is what
makes near-duplicate and echo judgement possible at all, costs one round trip, and caches as a
single unit keyed on the candidate set hash. Per-candidate calls would be ~40x the latency for a
narrower view.

> **Decision.** `questxtract/llm.py`. Config: `base_url`, `api_key` (env), `judge_enabled` (auto:
> true when an endpoint is reachable), `judge_model`, `vision_model`, `judge_max_candidates` (60,
> chunked above that), `timeout_seconds` (120). Prompts are module-level constants,
> version-tagged, so a prompt change is a reviewable diff.
>
> Dependency: `openai`, in an optional extra `questxtract[llm]`, imported inside the functions so
> the base install stays dependency-light.

---

## 11. Orchestration: stages, caching, progress

### 11.1 The stage contract

Everything in this document is one of ten functions with the same signature. That sameness is the
whole extension mechanism.

There are exactly **three seams**, ranked by how often anyone actually reaches for them:

| Want to change | Seam | What it costs |
|---|---|---|
| Whisper checkpoint, VLM, judge model | a `Config` string | nothing, no code |
| Hosted endpoint to a local one | `base_url` (section 10.1) | nothing, no code |
| The implementation of a stage | replace one entry in `STAGES` | one function |
| Add or remove a step | insert or drop one entry in `STAGES` | one function |

Rows three and four are the same seam used two ways.

**Stages.** A stage is a plain function `Context -> Context`, tagged with a decorator that carries
its name and its cache policy:

```python
Stage = Callable[[Context], Context]


def stage(name: str, *, cache: str | None = None, uses: tuple[str, ...] = ()) -> ...:
    """Tag a function as a pipeline stage.

    cache: the Context field this stage produces, cached as <stage name>.json.
           None = always recompute.
    uses:  Config field names whose values form the cache header (section 11.2).
    """


@stage(
    "transcribe",
    cache="transcript",
    uses=(
        "model_size",
        "language",
        "beam_size",
        "vad_filter",
        "compute_type",
        "batch_size",
        "initial_prompt",
    ),
)
def transcribe(ctx: Context) -> Context: ...
```

**The pipeline is a list.** Reading it is reading section 1.3's diagram:

```python
STAGES: tuple[Stage, ...] = (
    ingest,
    transcribe,
    sentences,
    segments,
    vision,
    attach_visual,
    detect,
    answer,
    score,
    judge,
)


def run(
    ctx: Context, stages: Sequence[Stage] = STAGES, progress: Progress | None = None
) -> Context:
    for fn in stages:
        if progress:
            progress(fn.stage.name, 0.0)
        ctx = _cached(fn, ctx) if fn.stage.cache else fn(ctx)
    return ctx


def process(path, config=None, *, stages=STAGES, progress=None) -> Context: ...
```

That `stages=` keyword is the entire plugin system. A caller who wants a different ASR engine, a
custom detector or an extra enrichment step passes a list. No plugin discovery, no
`entry_points`, no DAG resolver - list order is execution order, and if a stage needs something an
earlier stage did not produce, it fails on the spot with an attribute that is `None`.

**Four rules hold the contract up:**

1. **Stages never import each other.** They communicate only through `Context`. This is what makes
   the list reorderable and a stage replaceable in isolation.
2. **Stages read config only through `ctx.config`.** No module-level constants that are really
   settings, no environment reads outside `config.py`. A stage whose behaviour is not a function
   of `(ctx.config, ctx)` is not cacheable and not testable.
3. **Optional stages no-op themselves.** `vision` returns `ctx` unchanged when
   `config.vision_enabled` is false; `judge` does the same without an endpoint. The runner stays
   dumb - one `if` at the top of the stage beats filtering logic in `run()`.
4. **A cached stage produces exactly one `Context` field**, and that field is JSON-serialisable
   dataclasses. This is why `vision` and `attach_visual` are two stages: capture is expensive and
   caches as `visual.json`, the interval join is eight lines and always recomputes.

**Config.** One flat frozen dataclass in `config.py`, every default in one place, loaded from CLI
flags or constructed by a library caller. No per-stage config classes and no nesting: a flat
record is what makes `uses=(...)` a tuple of names rather than a path expression.

> **Decision.** `STAGES`, `run()` and `process()` live in `pipeline.py`; `Context` in `models.py`,
> `Config` in `config.py`. The cache and progress plumbing is written **once, in the runner**,
> instead of being repeated in every expensive stage.
>
> The `stage` decorator itself sits in `models.py` next to `Stage` and `StageInfo`, and
> `pipeline.py` re-exports it: the stage modules need it, and `pipeline.py` imports every stage
> module to build `STAGES`, so defining it there would be an import cycle.

### 11.2 One cache

```
.questxtract/<source_hash>/
  transcribe.json
  layout.json
  frames/<t>.jpg
  vision.json
  judge.json
```

Each entry is written with a small header holding the values of the stage's `uses` config fields
(section 11.1). The runner reuses an entry when the header matches, recomputes when it does not,
and `--from-scratch` bypasses reuse for every stage. Every layer verifies its config, including
transcription. Sentences, segments, detection, answers and scoring declare no `cache`, so they
always recompute: caching a 20 ms stage buys nothing and costs an invalidation bug.

**No source hash, no cache.** The directory name *is* the identity of the recording, so a run
whose `ctx.source_hash` is empty - a caller seeding a transcript directly, say - skips the cache
entirely rather than sharing one `unknown/` bucket. Otherwise one recording's results replay into
another's, which is a correctness bug, not a performance one.

**The judge is the exception to the `uses` rule.** Section 10.4 keys its entry on the *candidate
set* hash, which a config-only header cannot express: caching it on the source hash alone would
serve a stale verdict after detection changed. Its `cache=` is therefore wired at the same time as
the candidate-set key, not before.

Because the header is derived from `uses` rather than hand-written per stage, a new stage gets
correct invalidation by naming the config fields it reads, and a stage that gains a config field
and forgets to list it is the one bug class this design still has. Keep `uses` next to the
signature so the two are reviewed together.

The reference recording's transcript is committed next to its labels (section 2.2),
which is what lets CI run the eval gate in seconds with no ASR and no network.

### 11.3 Progress (FR9)

```python
Progress = Callable[[str, float], None]
```

Stage name plus fraction complete. The CLI renders its own `\r`-redrawn terminal bar from this
(cli.py, issue #14); the library default is `None`. The runner emits one call per stage from the
name on the decorator, so a stage added to `STAGES` reports progress without doing anything. Long
stages call `ctx.progress` again mid-run: transcription reports `word.end / info.duration` as each
word decodes, capped at 0.99 since the caller - not the backend - decides when a stage is actually
finished, through an `on_progress` callback every `AsrBackend` accepts (asr.py) rather than
faster-whisper's own `log_progress` flag. Vision will report per board state once that stage is
real - the only two stages long enough to need it.

> **Decision.** A callable, not a progress framework, and not a logging handler.

---

## 12. Output

### 12.1 questions.json

Schema 2.0: a fixed key set, `null` included, in one file. The formal description is
[`questions.schema.json`](../src/questxtract/questions.schema.json), which ships inside the
package; the tests validate every export shape against it. One entry of each list shown:

```json
{
  "schema_version": "2.0",
  "source": {"path": "lecture.mp4", "duration_seconds": 4812.3, "hash": "9f2a..."},
  "config": {"model_size": "turbo", "language": "en", "min_confidence": 0.4,
             "judge_model": "gpt-5.6-terra", "vision_model": "gpt-5.6-luna"},
  "stats": {"sentences": 1840, "board_states": 34, "candidates": 61,
            "emitted": 38, "judged": 38, "rejected": 23},
  "segments": [
    {"id": 37, "start": 1262.0, "end": 1340.5, "topic": null, "visual_ids": ["v_4c1e09"]}
  ],
  "visuals": [
    {
      "id": "v_4c1e09",
      "start": 1240.0,
      "end": 1402.5,
      "keyframe_path": ".questxtract/9f2a.../frames/1388.jpg",
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
      "reasons": ["mark", "wh", "answer_cue", "visual_overlap", "judge_revised"],
      "review": {"status": "pending", "text": null, "note": null, "metadata": {}}
    }
  ]
}
```

**Shared things are listed once and referenced by id.** `segments` and `visuals` (board states,
section 6.4) are top-level lists; a question points into them with `segment_id` and
`visual_ids`, so a board state that ten questions discuss is written once. This is the shape the
vision milestones grow into without a breaking change: M3's board timeline is the `visuals` list
in time order, and M4's animations become one more top-level list pointing at visual ids.

- `visual_ids` holds the board states on screen while the question was asked (overlapping its
  `start`/`end`), falling back to every state joined to its segment when none was; empty when
  vision is off. A segment's own `visual_ids` always holds every state joined to it.
  `segment_id` is `null` when the question's sentence belongs to no segment.
- `answer` is `null` when no answer was found. Its `start`/`end` span its own sentences
  (`sentence_ids`), not the question.
- `original_text` is present whenever the judge revised, `null` otherwise.
- `source` says what the question was read from. Only `audio` is produced until M2/M3;
  `visual` and `combined` are reserved.
- `topic` on a segment is reserved for topic filtering (#16) and `null` until topics exist.
- `config.min_confidence` is the threshold the file was cut at.

**Versioning.** `schema_version` is `MAJOR.MINOR`. A minor version only adds keys, so a reader
must ignore keys it does not know; a major version moves or removes something, and
`read_questions` refuses a file of another major version. 1.x to 2.0 moved the board state out of
each question (`visual` became the top-level `visuals` plus `visual_ids`) and added `segments`,
`source`, `review` and `config.min_confidence`.

Two consumers, two exporters: this file and the eval harness.

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

Merging edits after a re-run (#15) is therefore copying `review` across by question id, which is
why question and sentence ids are content hashes (section 5) rather than indices.

### 12.3 quiz.json (generative, separate)

A different command producing a different file: `questxtract quiz questions.json -o quiz.json`.
It is the only part of the system allowed to write text that was not spoken, and keeping it out
of the extraction pipeline is what keeps requirement 4.5 checkable.

---

## 13. Module layout and dependencies

### 13.1 Layout

```
src/questxtract/
  __init__.py      public API: process(), run(), STAGES, read/write
  models.py        Word, AsrSegment, Transcript, Sentence, Segment, VisualContext,
                   QuestionCandidate, QAPair, Context, SchemaError, the @stage decorator
  config.py        one flat frozen Config dataclass, every default
  media.py         av.open probe, source hash
  transcribe.py    faster-whisper, batched
  sentences.py     regex split, word-index snapping, force-split
  segments.py      structural segmentation
  vision.py        layout probe, sampling, board states, VLM captioning, attach
  detect.py        question extraction, vetoes, scoring, cue tables
  answer.py        adjacent seed, forward window, far-window rescue
  score.py         answer scoring, confidence, dedupe
  llm.py           OpenAI-compatible client, judge pass, vision calls
  evaluate.py      labelled-set metrics
  export.py        questions.json writer and reader (section 12.1)
  pipeline.py      STAGES, run(), process(), cache, progress
  cli.py           process / eval / quiz subcommands

tests/            flat, one test module per source module, plus conftest.py
```

Tests sit flat beside the modules they cover, and pytest runs with `--import-mode=importlib` so
same-named files never collide as the suite grows. `tests/data/` holds the labelled set
(section 4.1), not code.

Each of `transcribe.py`, `sentences.py`, `segments.py`, `vision.py`, `detect.py`, `answer.py`,
`score.py` and the judge half of `llm.py` exposes exactly one `@stage`-decorated
`Context -> Context` function as its public surface; everything else in the module is private to
it. That is the modularity claim, and it is checkable: `grep '@stage'` over the stage modules
should return the contents of `STAGES`. `tests/test_pipeline.py` asserts exactly that, so the
claim cannot quietly stop being true.

### 13.2 Dependencies

**Base** (`pyproject.toml`): `av` (container probe and audio extraction), `faster-whisper` (ASR),
`numpy` and `onnxruntime`. `ctranslate2`, `tokenizers`, `huggingface-hub` and `tqdm` arrive
transitively through faster-whisper.

**Extras:** `[llm]` -> `openai` (judge and vision, one key, one dependency). The CLI needs no
extra: its progress bar is a few lines of `print` in `cli.py`, not `tqdm`.

**No system binaries.** No ffmpeg, no ffprobe. **No torch**, and therefore no torchvision, no
triton, no CUDA wheels, no transformers, no pyannote, no speechbrain, no sentence-transformers,
no openpyxl. The install is a couple of hundred MB, and NFR1's Windows leg stops being the hard
one.

**`wtpsplit` is not on that list.** A neural splitter (3.1's third rung) does not need torch:
`wtpsplit-lite` (MIT) runs the same
SaT models on ONNX with exactly four dependencies - `huggingface-hub`, `numpy`, `onnxruntime` and
`tokenizers` - **all four of which we already install** transitively through `faster-whisper`. So
the code cost of that rung is zero new packages. Its real cost is the weights: 428 MB for
`sat-3l-sm`, against a stage that today costs ~20 ms and nothing on disk. That is the number the
decision should be made against, once the eval harness shows the splitter losing questions.

### 13.3 Status

What the code does today, and the open issue for each missing piece. "Stub" means the module, its
name, its signature and its `@stage` registration are real; only the algorithm is missing.

| Module | Stage | State | Issue |
|---|---|---|---|
| `models.py`, `config.py`, `pipeline.py` | the contract itself | done | |
| `media.py` | 1 ingest | done: probe, source hash, audio extraction to a temporary WAV | |
| `transcribe.py`, `asr.py` | 2 transcribe | done: faster-whisper behind the `AsrBackend` registry; the vision-informed second pass (2.4) is not built | #62 |
| `sentences.py` | 3 sentences | done | |
| `segments.py` | 4 segments | done | |
| `vision.py` | 5 vision, attach_visual | capture is a stub that raises when `vision_enabled`; the interval join is done | #59 |
| `detect.py` | 6 detect | done (7.1 to 7.3); implicit cues and missing-`?` recovery are not built | #48, #78 |
| `answer.py` | 7 answer | adjacent seed done; forward window and far-window rescue not built | |
| `score.py` | 8 score | confidence and threshold only; answer scoring (9.1) and dedupe (9.3) not built | |
| `llm.py` | 9 judge | stub: no-op when disabled, raises when `judge_enabled` | #46 |
| `export.py` | 10 export | done; keeping review edits across re-runs is not built | #15 |
| `filtering.py` | not a stage | time filter done; `topic=` is experimental, no stage produces topics | |
| `evaluate.py` | not a stage | done; NFR2 gated in pytest | |
| `cli.py` | not a stage | `process`, `transcribe` and `eval` work ([cli.md](cli.md)); `quiz` exits 1 | |

Every stub carries a `ponytail:` comment naming its issue, so `grep -rn 'ponytail:' src/` is the
running list of what this document describes and the code does not yet do.

What lands when: [GitHub milestones](https://github.com/mad201802/QuestXtract/milestones).

---

## 14. Non-goals

Explicit boundaries, so the package does not grow into them by accident.

- **The teacher review/edit UI** ([issue #15](https://github.com/mad201802/QuestXtract/issues/15))
  is a separate application. This package's part is that its JSON round-trips (section 12.2).
- **exerplaza.com upload** is likewise a separate consumer of the exported JSON. Nothing in
  `questxtract` talks to exerplaza directly.
- **The labelled dataset** ([#11](https://github.com/mad201802/QuestXtract/issues/11)) is test
  material under `tests/data/`, not an importable subpackage. The *harness* that reads it is
  `evaluate.py` (section 4), because NFR2 is a CI gate and the threshold sweep is library code.
- **YouTube URLs** are out of scope: the CLI takes a local file. If URL download is ever added it
  belongs in `media.py`.
- **Speaker identification by name** and diarization (section 2.6).

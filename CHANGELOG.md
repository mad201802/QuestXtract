# Changelog

## Unreleased

The package is restructured around the pipeline described in
[docs/architecture.md](docs/architecture.md), which is now the design of record. The package-per-stage
layout introduced earlier in this cycle is reversed: the `Context -> Context` stage contract, not
the directory, is the unit of ownership and replacement. `docs/architecture.md` is deleted, its
still-current content folded into architecture.md sections 13.1, 13.3 and 14;
`docs/architecture_review.md` keeps the decision history and records the reversal in D1 and D2.

### Changed (breaking)

- Modules moved back to one file per stage: `models/schemas.py` → `models.py`,
  `detection/questions.py` → `detect.py`, `audio/transcripts.py` → the runner's cache,
  `output/serialize.py` → `export.py`. The `video/`, `audio/`, `detection/`, `models/` and
  `output/` packages are gone.
- Schema renamed and widened: `TranscriptSegment` → `AsrSegment` (a raw ASR chunk) plus the new
  `Transcript` that owns the word list; `Question` → `QuestionCandidate` (detection) and `QAPair`
  (a question with its answer span); `LectureResult` → the `questions.json` envelope of
  architecture.md 12.1. `Question.text` is `QuestionCandidate.text`; `Question.source` is dropped,
  since `reasons` already records what fired.
- Export format replaced. Was `{"lecture", "questions": [...]}`; now a fixed key set with
  `schema_version`, `source`, `config`, `stats` and richer question objects carrying `answer`,
  `visual`, `verdict`, `question_type`, `confidence_label` and `original_text` — `null` included,
  never omitted.
- `write_result` / `result_to_json` / `read_result` → `write_questions` / `result_to_json` /
  `read_questions`, which take a finished `Context`. `write_segments` / `read_segments` /
  `segments_to_json` are removed: the transcript is a cache entry, written by the runner.
- `detect_questions` is no longer the public entry point; `process()` and `run()` are. The
  heuristic is still there, now over `Sentence` rather than `TranscriptSegment`.
- CLI takes a subcommand: `questxtract transcript.json` → `questxtract process transcript.json`.
  `--threshold` now gates the final confidence (architecture.md 9.2), not the raw detection score.
- `questxtract process` writes to a file next to its input (`lecture.questions.json`) instead of
  stdout. Pass `-o -` for the old behaviour.
- `AsrBackend.transcribe` takes a third argument, `on_progress: Callable[[float], None] | None`.
  An out-of-tree backend must accept it, even if it ignores it. It is called with fractions in
  `[0, 1)` only: the caller, never the backend, decides when the stage is finished.
- Export `schema_version` 1.0 → 1.1: every question object carries `start_hms` and `end_hms`
  alongside `start`/`end`. Added keys only, so a 1.0 consumer still reads a 1.1 file.
- Export `schema_version` 1.1 → 2.0 (#13). Board states move out of each question into a
  top-level `visuals` list, referenced by `visual_ids` (every overlapping state, not only the
  first). A top-level `segments` list (id, span, a reserved `topic`, `visual_ids`) makes
  `segment_id` resolvable. Every question gains `source` and a `review` block that belongs to
  people; `config` gains `min_confidence`. `read_questions` refuses another major version.
- `Config.max_question_chars` 220 → 350 and `Config.batch_size` 8 → 4 (architecture.md 2.2, 7.3);
  both are untuned experiments against real lecture audio, revisited once #11's labelled set
  exists. `batch_size` is part of the transcribe cache key, so the change re-transcribes once.

### Added

- `filter_questions(ctx, start=, end=)` and `filter_document(data, start=, end=)` filter a
  finished `Context` or a `read_questions()` document by time range, `start <= question.start <
  end` in seconds, either side optional (#16, FR11). They return a new value; only the question
  list shrinks. `topic=` is experimental: it matches the question's segment topic, and warns
  since no topic labels exist yet.
- `questions.schema.json`, the formal JSON Schema of `questions.json` 2.x, ships inside the
  package; the tests validate every export shape against it (#13). `VisualContext.id` gives a
  board state a stable id derived from its lifetime and keyframe.
- Automatic releases (#45): pushing a tag `vX.Y.Z` publishes the package to PyPI and creates the GitHub Release with the matching CHANGELOG section as notes. `scripts/release_notes.py` fails the release first if the tag, `pyproject.toml` and `CHANGELOG.md` disagree.
- A terminal progress bar on stderr, one redrawn line per stage, rendered by the CLI from the
  existing `Progress` callback (#14). The transcribe stage reports inside itself, per decoded
  word, through the backend's `on_progress`.
- `questxtract process lecture.mp4` runs every stage on a recording; a `.json` input still skips
  transcription (#17). `questxtract transcribe lecture.mp4` writes the transcript only, sharing
  the transcribe cache with `process`. `-m/--model` picks the Whisper model. Bad inputs fail
  with one `error:` line before anything slow runs; Ctrl+C exits 130. Documented in
  [docs/cli.md](docs/cli.md).
- `Transcript.to_dict()`, the inverse of `from_dict`.
- `scripts/dropped_questions.py` writes the candidates the threshold dropped next to the ones it
  kept, for inspecting false negatives.

- The stage contract: `@stage(name, cache=..., uses=...)`, the `STAGES` list, `run(ctx, stages=)`
  and `process()`. Passing a different `stages=` list is the entire plugin system.
- One cache, written by the runner, under `.questxtract/<source hash>/<stage>.json`, keyed on the
  config fields each stage declares in `uses`. `--from-scratch` bypasses it.
- Progress reporting: `Progress = Callable[[str, float], None]`, emitted once per stage from the
  name on the decorator, so a new stage reports progress without doing anything.
- `Config`: one flat frozen record holding every documented default, with `replace()` that rejects
  unknown field names.
- `Transcript.from_dict` accepts a bare list of chunks or a `{"segments": [...]}` object, Whisper's
  `word`/`probability` keys included, and synthesises evenly spaced words when an engine provides
  none, so downstream stages always have a word list to slice.
- Sentence rebuilding (`sentences.py`) with word-index snapping and force-split at the widest
  internal gap, and structural segmentation (`segments.py`) on gap, duration and length.
- `questxtract quiz` subcommand, which exits 1 with a one-line error until the generative stage
  lands.
- The evaluation harness (#12): `questxtract eval <dir|file>...` scores the pipeline against
  every `*.labels.jsonl` that has a `*.transcript.json` beside it - content recall, precision,
  f1, false positives by label kind, `question_mark_rate`, `blind_slice_recall`, a sweep over
  `min_confidence`, and every missed question and false positive. `--json` for machines.
  Matching is one-to-one by time overlap. `tests/test_evaluate.py` gates NFR2 (pooled recall
  >= 0.50) in CI; today it is 0.89, precision 0.64.
- `tests/data/<slug>.transcript.json` for the reference recordings, and exhaustive,
  transcript-anchored labels replacing the hand-timed #11 set (127 content questions, 314
  entries). `scripts/scratch_to_labels.py --transcript` snaps a label's times to the words it
  quotes.
- Optional `questxtract[llm]` extra for the judge and vision calls, imported inside the functions
  so the base install stays stdlib-only.
- `tests/test_pipeline.py` asserts that `STAGES` matches the stage order architecture.md 1.3
  specifies, so the document cannot quietly stop describing the code.
- `AsrBackend`, a pluggable ASR interface (`asr.py`): `register_backend`/`resolve_backend`
  let `Config.asr_backend` select the engine, so `transcribe.py` never imports a specific
  one. `FasterWhisperBackend` is the default, reusing `media.extract_audio`.
- `stats["sentences_unpunctuated"]`: the count of sentences that did not end in terminal
  punctuation. The standing proxy for architecture.md 3.1's punctuation bet until
  `question_mark_rate` lands with #12. Exported in the `stats` block; no schema change.
- `Segment.close_reason` (`gap`, `duration`, `chars`, or `end` for the last one) and
  `stats["segments_closed_gap" | "_duration" | "_chars"]`: which rule is actually doing the
  cutting. On the three labelled recordings a pause closes 95-98% of segments, so the
  thresholds are kept (architecture.md 3.2). Exported in the `stats` block; no schema change.
- Reference recordings (#33): two TTK4225 lectures (2026-09-14 P03·12, 2026-09-15 P03·14)
  join MIT 18.100B, each labelled from its committed transcript per
  `docs/annotation_guidelines.md` (114 and 166 entries, every one blind). Ground truth for the
  NFR2 recall gate in `evaluate.py` (#12).

### Fixed

- The same-sentence answer seed never fired: it looked for text after the `?` in the question's
  own sentence, and stage 3 splits at the `?` (#38). It is now the adjacent seed, the next
  sentence across at most `answer_seed_max_gap_seconds` (1.0 s), not itself a question, and never
  for tags or check-ins. Its reason code stays `intra_sentence_qa`. An answered question's
  confidence rises from 0.55 x score to 1.0 x score. Pooled recall at the default gate goes
  from 0.89 to 0.95 and precision from 0.64 to 0.61 (architecture.md 7.3, #70).
- A question's `visual_ids` list the board states on screen while it was asked, falling back to
  every state of its segment when none was. They listed the whole segment's, so a question could
  point at a board written up to 90 s later (#40).
- An answer's `start`/`end` in `questions.json` span its own sentences; they repeated the
  question's span (#13).
- Question detection only looked at a sentence's first word, so "So what ...?", "Then how ...?"
  and "If you have the data, does that mean ...?" scored on the `?` alone (0.60), which the
  unanswered confidence (0.55 x score) put under the 0.40 emission gate. The head is now found
  past discourse markers and, for a `?` sentence, in its last comma clause; negated auxiliaries
  count and `any` no longer does. On the reference recordings content questions found went from
  2/10 to 7/10 (MIT 18.100B), architecture.md 7.3.
- A question opened by a preamble ("My question is that doesn't ...?", "The question is, why
  ...?", "I was wondering if ...") scored on the `?` alone, since only single-word discourse
  markers were skipped. The head is now found past these preambles too (#47). No sentence score
  changes on the three reference recordings.
- Confirmation tags ("Okay?", "True, you see?", "..., right?") scored like real questions, and
  check-ins ("Is that clear?") above them. New reasons `tag` and `check_in` (-0.40 each) keep
  tags out of the candidates and check-ins out of the emitted questions.
  Check-in phrases match whole words, and the comprehension checks ("make sense", "is that
  clear") not when content follows them, so a content question that contains one is kept.
- A wh- or aux-head without a `?` passed the detection gate on its own; on three lectures all 35
  such candidates were statements ("What I mean is ..."). Head weights are 0.4 (was 0.5), under
  the gate alone, 1.0 with the `?`. Emitted questions are unaffected, the candidate count drops.
- A cached stage with no source hash wrote into a shared `unknown/` bucket, so one recording's
  results could be replayed into another's. A run without a content hash now skips the cache.
- `judge` declared a cache keyed only on config, which would have served a stale verdict after
  detection changed. Its entry is keyed on the candidate set instead (architecture.md 10.4), and is
  wired when that key is.
- The sentence splitter treated every word ending in `.` as a sentence end, so abbreviations
  (`So Dr. Rudin defines it.`), single-letter initials (`greater than capital N.`) and ellipses
  (`the limit... of the sequence`) each split one sentence into two fragments. `_is_terminal` holds
  those back; digits still split, so `Step 1.` is unaffected (architecture.md 3.1).
- `max_sentence_words` was a trigger, not a cap. An unpunctuated run with no silence over
  `force_split_min_gap_seconds` was returned whole however long it got, and a run *under*
  `max_sentence_words` was never split even at a multi-second silence. Both lost every question
  inside the resulting blob, since `detect` vetoes text over `max_question_chars`. A run-on is now
  split at every qualifying silence and then capped; a correctly punctuated sentence containing a
  long pause is still one sentence.
- `Config.force_split_min_gap_seconds` is now the pause threshold generally, not only a
  force-split one. Same name, same default, wider effect.

## Earlier in this cycle

### Changed (breaking)

- `Question.text` is now `Question.question`; the exported JSON key changed with it.
- `write_questions` / `questions_to_json` are replaced by `write_result` / `result_to_json`,
  which take a `LectureResult` and write a `{"lecture", "questions"}` envelope instead of a
  bare list.
- Schema dataclasses are keyword-only (`kw_only=True`).
- Modules moved: `models.py` → `models/schemas.py`, `detect.py` → `detection/questions.py`,
  transcript JSON I/O → `audio/transcripts.py`, result JSON I/O → `output/serialize.py`.

### Added

- `Question.id` (stable per recording), `Question.source`, `TranscriptSegment.words`
  (per-word timestamps), `Word`, `LectureResult`, `SchemaError`.
- `read_result`, `write_segments`, `segments_to_json`; `from_dict` / `to_dict` on every
  schema type.
- The CLI reports a missing or malformed transcript as a one-line `error: …` with exit code 1
  instead of a traceback.
- `docs/architecture.md`.

### Fixed

- `score_sentence` no longer skips a non-ASCII first word, and always returns a float.

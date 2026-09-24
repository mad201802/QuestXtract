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
- `AsrBackend.transcribe` takes a third argument, `on_progress: Callable[[float], None] | None`.
  An out-of-tree backend must accept it, even if it ignores it. It is called with fractions in
  `[0, 1)` only: the caller, never the backend, decides when the stage is finished.
- Export `schema_version` 1.0 → 1.1: every question object carries `start_hms` and `end_hms`
  alongside `start`/`end`. Added keys only, so a 1.0 consumer still reads a 1.1 file.
- `Config.max_question_chars` 220 → 350 and `Config.batch_size` 8 → 4 (architecture.md 2.2, 7.3);
  both are untuned experiments against real lecture audio, revisited once #11's labelled set
  exists. `batch_size` is part of the transcribe cache key, so the change re-transcribes once.

### Added

- Automatic releases (#45): pushing a tag `vX.Y.Z` publishes the package to PyPI and creates the GitHub Release with the matching CHANGELOG section as notes. `scripts/release_notes.py` fails the release first if the tag, `pyproject.toml` and `CHANGELOG.md` disagree.
- A terminal progress bar on stderr, one redrawn line per stage, rendered by the CLI from the
  existing `Progress` callback (#14). The transcribe stage reports inside itself, per decoded
  word, through the backend's `on_progress`.
- `scripts/`: `transcribe.py` and `extract_questions.py` run a raw recording end to end until
  the CLI accepts one (#17); `dropped_questions.py` writes the candidates the threshold dropped
  next to the ones it kept, for inspecting false negatives.

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
- `questxtract eval` and `questxtract quiz` subcommands, which exit 1 with a one-line error until
  `evaluate.py` (#12) and the generative stage land.
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
  `stats["segments_closed_gap" | "_duration" | "_chars"]`. The segment thresholds are untuned
  against a real recording and 3.1's force-split changed the sentence stream underneath them;
  this is what says which rule is actually doing the cutting. Exported in the `stats` block;
  no schema change.

### Fixed

- Question detection only looked at a sentence's first word, so "So what ...?", "Then how ...?"
  and "If you have the data, does that mean ...?" scored on the `?` alone (0.60), which the
  unanswered confidence (0.55 x score) put under the 0.40 emission gate. The head is now found
  past discourse markers and, for a `?` sentence, in its last comma clause; negated auxiliaries
  count and `any` no longer does. On the reference recordings content questions found went from
  2/10 to 7/10 (MIT 18.100B) and 5/17 to 9/17 (CS234), architecture.md 7.3.
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

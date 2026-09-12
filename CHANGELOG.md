# Changelog

## Unreleased

The package is restructured around the pipeline described in
[docs/pipeline.md](docs/pipeline.md), which is now the design of record. The package-per-stage
layout introduced earlier in this cycle is reversed: the `Context -> Context` stage contract, not
the directory, is the unit of ownership and replacement. `docs/architecture.md` is deleted, its
still-current content folded into pipeline.md sections 13.1, 13.3 and 14;
`docs/architecture_review.md` keeps the decision history and records the reversal in D1 and D2.

### Changed (breaking)

- Modules moved back to one file per stage: `models/schemas.py` → `models.py`,
  `detection/questions.py` → `detect.py`, `audio/transcripts.py` → the runner's cache,
  `output/serialize.py` → `export.py`. The `video/`, `audio/`, `detection/`, `models/` and
  `output/` packages are gone.
- Schema renamed and widened: `TranscriptSegment` → `AsrSegment` (a raw ASR chunk) plus the new
  `Transcript` that owns the word list; `Question` → `QuestionCandidate` (detection) and `QAPair`
  (a question with its answer span); `LectureResult` → the `questions.json` envelope of
  pipeline.md 12.1. `Question.text` is `QuestionCandidate.text`; `Question.source` is dropped,
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
  `--threshold` now gates the final confidence (pipeline.md 9.2), not the raw detection score.

### Added

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
- `tests/test_pipeline.py` asserts that `STAGES` matches the stage order pipeline.md 1.3
  specifies, so the document cannot quietly stop describing the code.

### Fixed

- A cached stage with no source hash wrote into a shared `unknown/` bucket, so one recording's
  results could be replayed into another's. A run without a content hash now skips the cache.
- `judge` declared a cache keyed only on config, which would have served a stale verdict after
  detection changed. Its entry is keyed on the candidate set instead (pipeline.md 10.4), and is
  wired when that key is.

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

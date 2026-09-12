# Changelog

## Unreleased

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

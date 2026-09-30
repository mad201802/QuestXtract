# Changelog

## Unreleased

### Fixed

- `questxtract process` rejects a `--threshold` outside 0 to 1 with a one-line error instead of
  writing a file that fails its own schema. (#85)

## 0.1.0

First release. QuestXtract turns a lecture recording (`.mp4`) or an ASR transcript (`.json`) into
`questions.json`: the questions asked in the lecture, with timestamps, a confidence, and the answer
where the next sentence gives it.

### What is in it

- **CLI** ([docs/cli.md](docs/cli.md)): `questxtract process` runs every stage on a recording, or
  skips transcription for a `.json` transcript; `transcribe` writes the transcript only; `eval`
  scores the pipeline against labelled recordings. A progress bar on stderr, one-line errors, and
  a transcription cache (`--from-scratch` bypasses it).
- **Pipeline**: a list of `Context -> Context` stages (`STAGES`, `run`, `process`); passing a
  different list is the plugin system. Audio extraction with PyAV (no system ffmpeg),
  faster-whisper transcription behind the pluggable `AsrBackend` interface, sentence rebuilding,
  structural segmentation, rule-based question detection with reason codes, the adjacent-sentence
  answer seed, and a confidence with a threshold.
- **Output**: `questions.json` schema 2.0, a fixed key set with `null` included, content-hash ids
  that survive a re-run, a `review` block for teacher edits, and the formal
  `questions.schema.json` inside the package. `read_questions`, `write_questions`, and
  `filter_questions` / `filter_document` by time range (`topic=` is experimental).
- **Configuration**: one flat frozen `Config` holding every default; `replace()` rejects unknown
  field names.
- **Evaluation**: three labelled reference recordings in `tests/data/` (127 content questions) and
  `questxtract eval`, which CI uses to gate recall >= 0.50. At the default threshold, pooled
  recall is 0.95 and precision 0.61.

### Not in it yet

- Reading the board and slides (`vision_enabled`, #59) and the LLM pass that filters and refines
  questions and finds answers beyond the next sentence (`judge_enabled`, #46): both raise
  `NotImplementedError` when enabled.
- `questxtract quiz` exits 1 (#79).
- Re-running over an edited `questions.json` does not keep the `review` edits yet (#15).
- Only `.mp4` recordings, English lectures.

### Install

```bash
pip install git+https://github.com/mad201802/QuestXtract@v0.1.0
```

Python 3.10+ on Linux, macOS or Windows.

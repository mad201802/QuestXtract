# Changelog

## Unreleased

### Added

- `questxtract quiz lecture.questions.json` writes multiple-choice quiz items to
  `lecture.quiz.json`, and `process --quiz` writes them after the questions in the same run. Each
  item has a stem, `--options` options (3 by default, one correct), an explanation and why each
  option is right or wrong; every text says whether it was spoken or generated, and every item
  starts as pending review. The model sees the question, the lecture's answer, what was said
  around it (from `--transcript` or the cached transcript) and, with vision, the board text and
  LaTeX. By default only questions the lecture answered are used;
  `--strictness lenient` adds the others. Rejected questions, check-ins and questions the model
  judges not to be quiz material are left out, a question the model fails on is skipped with a
  warning, and answers are cached. Settings: `quiz_model`, `quiz_strictness`, `quiz_options`. (#79)
- Vision: with `vision_enabled = true` (in `questxtract.toml`) and a recording with a video
  track, the slides or board are read. Each stable view's keyframe goes to `vision_model`,
  together with what was said while it was on screen, and comes back as a board state in
  `visuals` (text, LaTeX, a description and a confidence), linked to its segments through
  `visual_ids`, with the keyframe copied beside the file. Readings and keyframes are cached, so a
  re-run sends nothing; a reading that failed is asked again. A failed reading leaves its view
  out with a warning and is counted in `stats.vision_failed`; when the first three fail, the rest
  are not sent. Needs the `llm` extra and an API key, or `base_url` for a local server, checked
  before transcription starts. (#59)
- `process` and `transcribe` accept `.mp3` recordings as well as `.mp4`. A recording without a
  video track, an MP3 or an audio-only MP4, is transcribed as usual and has no board to read.
  (#117)
- README: setting up the API key safely, the same way on macOS and Windows: the key in the
  system's password store through `keyring`, and a git-ignored `.env` that only refers to it.
- `process`, `transcribe` and `eval` read settings from `questxtract.toml` in the working
  directory, or from `--config PATH`. Explicit flags override the file, and omitted values
  retain their defaults. Unknown keys, wrong types and secret-like keys fail with readable
  errors. `OPENAI_API_KEY` can also come from a local `.env`; the environment takes precedence
  and the key is never stored in Config, exports or caches. (#99)
- `--detector llm` on `process` and `eval`: a language model judges every sentence ending in
  "?" (and every question the rules found) in the lecture around it, keeps the real questions,
  cites the sentences that answer them and rewrites a messy question, keeping the spoken version
  as `original_text`. It only judges sentences the code chose, by id, so it never adds a
  question. Any OpenAI-compatible endpoint works: OpenAI with `OPENAI_API_KEY`, or a local server
  such as Ollama with `--llm-url` and `--llm-model`. Requests are cached in
  `.questxtract/judge/`. Needs the `llm` extra. (#88, #89)
- `--llm-reasoning` lets a thinking model think before it judges (`low`, `medium`, `high`); small
  models such as Qwen3.5 9B need it. `server` sends no setting, for a model without one such as
  gpt-4o-mini. Any level but `none` allows each request 10 minutes. (#89, #97)

### Changed

- `questions.json` schema 2.1: `keyframe_path` is relative to the JSON file. Writing the file
  copies each keyframe into a folder beside it named after the file (`lecture.keyframes/` for
  `lecture.questions.json`), so the two can be moved or shared together and lectures written to
  one folder keep their images apart; `-o -` copies nothing and keeps the cache path. (#64)

### Changed (breaking)

- `Config.judge_enabled` and `Config.judge_max_candidates` are gone. `Config(detector="llm")`
  turns the LLM on; `judge_window`, `judge_batch_size`, `judge_parallel` and
  `judge_reasoning_effort` tune it. The `judge` stage runs before `score`, so the threshold
  applies to what the model kept, and to the model's own confidence, without the rules' answer
  weighting. `evaluate()` takes an optional `Config`, and its report names the `detector`.
  (#88, #89)
- A question's `id` in `questions.json` comes from the sentence it was spoken in, not from its
  text, so it stays the same when `--detector llm` rewords the question or you switch detectors.
  Every `id` changes once with this release. (#94)

### Fixed

- The rules detector penalises framing questions without content words ("So how do we do
  that?") and excludes their adjacent answer seeds, so they stay below the default output
  threshold. Questions naming a concept and questions with negation remain eligible. (#49)
- The rules detector counts a contracted wh-word as a question word: "What's ...?", "Where's ...?",
  "How's ...?", "Who's ...?" (straight or curly apostrophe) now score and are typed like
  "What is ...?", instead of being dropped when unanswered. (#109)
- Provider connection errors mask the active API key if the service echoes it, including
  in the displayed traceback. (#99)
- Detection finds the question word after a leading preposition ("For which ...?"), in an earlier
  comma clause ("This area here, is it finite, ...?"), and after a lower-case single letter that
  ends a sentence ("Unit step function, u. Is it absolutely integrable?"). (#78)
- `questxtract process` rejects a `--threshold` outside 0 to 1 with a one-line error instead of
  writing a file that fails its own schema. (#85)
- Detection marks more classroom check-ins as `check_in` ("Do you see it?", "Is everything
  clear?", "Are you following?", "How many of you ...?", a bare "Everyone?"), so the rules
  detector stops emitting most of them. "Do you see why ...?" stays a question. (#70)

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

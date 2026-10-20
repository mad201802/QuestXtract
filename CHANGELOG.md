# Changelog

## Unreleased

### Added

- `questxtract filter lecture.questions.json --text WORDS --from TIME --to TIME` keeps the
  questions where every word appears, said or written on the board on screen while they were
  asked, and that fall in the time range, without running the pipeline again. Times are seconds,
  `M:SS` or `H:MM:SS`. It prints the result, or writes it with `-o`. `filter_questions` and
  `filter_document` take the same `text=`. (#136)
- Detection can read the board: with vision on and `board_overlap_delta` set, a question that
  shares a word with the text of the board on screen while it was asked gets that boost, the
  reason `board_overlap`, and `source: "combined"` in `questions.json`. Off by default (0.0): at
  0.10 it changed nothing on the labelled recordings. `eval --board` scores them with the board
  states committed beside their transcripts. (#60)
- Vision reads a slide or page that comes back only once per run. On slides, handwriting and
  screencasts, a keyframe that shows the same content as one read earlier in the run copies that
  reading instead of asking the model again; it stays a board state of its own, with its own
  times, and is counted in `stats.vision_reused`. A page whose changes exceed the limits of that
  comparison, such as new ink, a new decimal point or a moved pen cursor, is read again; a very
  faint mark can go unnoticed. Blackboards, whiteboards and frames the layout probe could not name
  are always read. (#138)
- Vision finds the slides or board first: three frames from across the recording go to
  `vision_model` in one request, which answers where the content is and what kind it is
  (`slides`, `blackboard`, `whiteboard`, `handwriting`, `screencast` or `unknown`). Frames are
  cropped to that region, so the lecturer's camera or a screenshare frame no longer starts new
  board states, and the kind goes into the captioning prompt. One answer holds for the whole
  recording. When the probe fails, vision reads the full frame as before, with a warning. (#57)
- `process --vision` / `--no-vision` and `--vision-model` switch reading the board on or off and
  pick its model for one run, over `questxtract.toml`; `--llm-url` now also serves vision. With
  vision on, a recording without video (an `.mp3`, an audio-only `.mp4`) or a transcript gets a
  note that the board is not read, and runs without vision. (#61)
- `quiz.json` schema 1.0 (`quiz.schema.json`, shipped in the package), the format the coming
  `quiz` command writes and the QTI and Exerplaza exporters read: single-choice items with at
  least three options, every text marked `spoken` or `generated`, a solution with one explanation
  per option, the source question's id and times, and a teacher `review` block. `read_quiz` and
  `validate_quiz` check a file against the rules the schema cannot express and list every
  problem with its path. (#146)
- `questxtract qti quiz.json -o quiz.qti.zip` (and `export_qti()`) writes a `quiz.json` as a
  QTI 3.0 package an LMS can import: one single-choice item per quiz item, worth its points and
  scored against its correct option, with each option's explanation as feedback. The quiz's
  instructions set shuffling, whether feedback is shown and the instructions text. Items a
  reviewer marked `rejected` are left out. The same quiz gives the same ZIP, byte for byte, and
  every package the tests write is validated against the official 1EdTech XSDs. (#67)
- Vision: with `vision_enabled = true` (in `questxtract.toml`) and a recording with a video
  track, the slides or board are read. Each stable view's keyframe goes to `vision_model`,
  together with what was said while it was on screen, and comes back as a board state in
  `visuals` (text, LaTeX, a description and a confidence), linked to its segments through
  `visual_ids`, with the keyframe copied beside the file. Readings and keyframes are cached, so a
  re-run sends nothing; a reading that failed is asked again. A failed reading leaves its view
  out with a warning and is counted in `stats.vision_failed`; when the first three fail, the rest
  are not sent. Needs the `llm` extra and an API key, or `base_url` for a local server, checked
  before transcription starts. (#59)
- `questions.json`'s `stats` records what the model calls cost, per caller: `judge_*` with
  `--detector llm`, `layout_*` and `vision_*` with vision on. Each has `_requests`,
  `_input_tokens`, `_output_tokens` and `_ms`, all 0 when the answers came from the cache.
  Multiply by your provider's prices for the cost of a lecture. Not a schema change: `stats`
  takes any integer key. (#123)
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

- `questions.json` schema 2.2: `config` records how the file was made, so two saved runs can be
  told apart: `detector` (`rules` or `llm`), `judge_prompt_version` (e.g. `v5`) and
  `judge_reasoning_effort` (the `--llm-reasoning` level, `""` for `server`). The last two are
  `null` with the rules detector. Files from schema 2.0 and 2.1 still read. (#114)

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

# Changelog

## Unreleased

### Changed

- The folder the app saves the NTNU key in (`questxtract/` in your user config folder) is made
  readable by you only on Linux and macOS, also when it exists from an earlier version. The key
  file itself already was. (#199)
### Added

- `questxtract exerplaza quiz.json --url URL --course-session ID --content-unit NAME` (and
  `export_exerplaza()`) sends a quiz straight to ExerPlaza: it logs in with your ExerPlaza email
  and password from `QUESTXTRACT_EXERPLAZA_EMAIL` and `QUESTXTRACT_EXERPLAZA_PASSWORD`
  (environment or `.env`), and ExerPlaza saves, in one go, a quiz in the course session with one
  multiple-choice question per item, marked as generated. Rejected items are not sent; pending
  ones are visible only to you. Sending the same quiz again adds its new items to the same quiz
  and skips, unchanged, those whose text ExerPlaza already has: ExerPlaza's database schema does
  not change. The address must be `https` (plain `http` for `localhost`
  only), and the password is never shown. Needs ExerPlaza's import endpoint on the server.
  (#37)

## 0.3.0

### Changed

- With vision on, each board state is read with the one before it: its text, LaTeX and
  description, and where new writing appeared (prompt v2). After an erase, a camera cut or a page
  flip, when less than 70 % of the earlier ink is left, the state is read on its own as before (prompt v0).
  The first run after updating reads the board states that follow another one again. The OCR
  reader is unchanged. (#137)

## 0.2.0

### Added

- `questxtract exerplaza quiz.json --url URL --course-session ID --content-unit NAME` (and
  `export_exerplaza()`) sends a quiz straight to ExerPlaza: it logs in with your ExerPlaza email
  and password from `QUESTXTRACT_EXERPLAZA_EMAIL` and `QUESTXTRACT_EXERPLAZA_PASSWORD`
  (environment or `.env`), and ExerPlaza saves, in one go, a quiz in the course session with one
  multiple-choice question per item, marked as generated. Rejected items are not sent; pending
  ones are visible only to you. Sending the same quiz again adds its new items to the same quiz
  and skips, unchanged, those whose text ExerPlaza already has: ExerPlaza's database schema does
  not change. The address must be `https` (plain `http` for `localhost`
  only), and the password is never shown. Needs ExerPlaza's import endpoint on the server.
  (#37)
- The app saves the NTNU key for you: the **NTNU key...** button on the start view opens a window
  where you paste the key and press Test (NTNU's service is asked once), Save or Remove. The key is
  saved as plain text in `questxtract/.env` in your user config folder (`%APPDATA%` on Windows,
  `~/Library/Application Support` on macOS, `~/.config` elsewhere), so every run finds it, from
  any folder. A key NTNU refuses is not saved; a key that could not be tested (off the NTNU
  network or VPN) is saved with a note. The CLI reads the same file, after the environment and
  `.env`. (#189)
- `questxtract gui` opens a window for people who do not use a terminal: drop or browse a
  recording (`.mp4`, `.mp3`) or a transcript (`.json`), choose questions, transcript, quiz or QTI
  package, set the options (the CLI's own, with its defaults) and press Run. It runs the CLI's code,
  shows progress, Cancel (between stages), the written path and an Open folder button. Install with
  the `[gui]` extra; drag and drop needs `tkinterdnd2`, and where it cannot start the
  window says so and Browse still works. The start view has only the file, the output and Run;
  the options are behind Advanced settings, except Read the board (vision), which stays on the start view. Without a `questxtract.toml` the window starts with the LLM detector and Read the board on (both need the NTNU key; the CLI's defaults are unchanged); with one, it starts from the file. The judge's model is greyed out with the rules detector. The window is in English, Norsk or Italiano (the
  switch at the top right). Needs a Python with tkinter. (#124)
- `questxtract tex quiz.json -o quiz.tex --content-unit NAME` (and `export_tex()`) writes a
  `quiz.json` as TeX for ExerPlaza's upload at `/questions/upload/tex`: one `IndexedQuestion`
  block per item, in English or Italian, with the given content units, its options and the
  correct one, a solution with every option's explanation, and notes for the teachers on where
  the question was asked and which model wrote the text. Pending items are visible only to their
  author, accepted ones to everybody, and rejected ones are left out: the same items as `qti`.
  Texts are written verbatim for MathJax. A text with unbalanced braces or a command of the
  format fails, naming the text, and nothing is written; so does a quiz with no item left. The
  same quiz gives the same bytes.
  You upload the file in ExerPlaza yourself. (#178)
- `--vision-reader ocr` (`vision_reader = "ocr"`) reads the board with GLM-OCR
  (`vision_ocr_model`, `zai-org/GLM-OCR` on NTNU's service) instead of the vision model: the text
  and the formulas as LaTeX, labels such as circled numbers and units left out, no description,
  and a confidence from the token probabilities (0.5 when the server sends none). It goes through
  vision's endpoint and key, is cached and fails like the vision model's reading, and works with
  `eval --score-vision`. The layout probe still uses `vision_model`. (#144)
- `questions.json` schema 2.4: `config.vision_reasoning_effort` records the reasoning setting
  sent to the vision model, `null` with vision off. Files from schema 2.0 to 2.3 still read. (#173)
- `questions.json` schema 2.3: `config.vision_ocr_model` names the OCR model that read the
  keyframes, `null` when the vision model read them or vision was off. Files from schema 2.0 to
  2.2 still read. (#144)
- `questxtract quiz lecture.questions.json` writes multiple-choice quiz items to
  `lecture.quiz.json`, and `process --quiz` writes them after the questions in the same run. Each
  item has a stem, `--options` options (3 by default, one correct), an explanation and why each
  option is right or wrong; every text says whether it was spoken or generated, and every item
  starts as pending review. The model sees the question, the lecture's answer, what was said
  around it (from `--transcript` or the cached transcript) and, with vision, the board text and
  LaTeX. Questions the lecture did not answer are used too, with an answer the model writes;
  `--strictness strict` uses only the answered ones. Rejected questions, check-ins and questions
  the model judges not to be quiz material are left out, a question the model fails on is skipped
  with a warning, and answers are cached. The run ends with the quiz's model requests, tokens and
  time. `--quiz-model` picks the model for one run. Settings: `quiz_model`, `quiz_strictness`,
  `quiz_options`. (#79)
- `eval --score-vision`: scores `vision_model` (or `--vision-model`) on 30 labelled keyframes in
  `tests/data/vision/`, from six lectures over four lecture types (slides, blackboard,
  whiteboard, handwriting), each recording one whose licence or lecturer allows sharing. It
  reports the character error rate of the text, blank boards read as blank, and the share of
  formulas read, per lecture type and per frame. Answers are cached, so a second run makes no
  calls. (#58)
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
- `questxtract qti --qti-version 2.1` (and `export_qti(..., version="2.1")`) writes the quiz as a
  QTI 2.1 package instead of 3.0, for an LMS that does not import 3.0, such as Canvas New
  Quizzes. It holds the same items, scoring, feedback and instructions as the 3.0 package; each
  item also names the standard `match_correct` template, for an LMS that supports only templates,
  and `qti` warns when an item is worth other than 1 point, which that template ignores. Every 2.1
  package the tests write is validated against the official 1EdTech QTI 2.1 XSDs, but none has
  been imported into a real LMS yet. Without the flag, `qti` writes QTI 3.0, byte for byte as
  before. (#163)
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
- `quiz.json` schema 1.0 (`quiz.schema.json`, shipped in the package), the format the `quiz`
  command writes and the QTI and Exerplaza exporters read: single-choice items with at
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

- NTNU's LLM service (the default endpoint) reads its key from `IDUN_API_KEY`, in the environment
  or `.env`. `QUESTXTRACT_NTNU_API_KEY`, its former name, is still read when `IDUN_API_KEY` is not
  set, so nothing has to be renamed. (#124)
- The LLM detector and the quiz send no reasoning setting by default (`reasoning_effort = ""`,
  the same as `--llm-reasoning server`). Only `--llm-reasoning low`, `medium` or `high` now raise
  the request timeout to 10 minutes; `server` keeps 2 minutes, so set `timeout_seconds` for a
  server that thinks on its own. (#164)
- `questions.json` schema 2.1: `keyframe_path` is relative to the JSON file. Writing the file
  copies each keyframe into a folder beside it named after the file (`lecture.keyframes/` for
  `lecture.questions.json`), so the two can be moved or shared together and lectures written to
  one folder keep their images apart; `-o -` copies nothing and keeps the cache path. (#64)

### Changed (breaking)

- The LLM detector and the quiz use `Qwen/Qwen3.8-27B-FP8` on NTNU's LLM service by default
  (`judge_model`, `quiz_model`) and send no reasoning setting (`reasoning_effort = ""`, the
  same as `--llm-reasoning server`), since Qwen rejects `none`. `--llm-reasoning` sets the judge
  and the quiz only. To judge with DeepSeek as before, pass
  `--llm-model deepseek-ai/DeepSeek-V4.1-Flash-1017 --llm-reasoning none`. (#179)
- Vision reads the board with `deepseek-ai/DeepSeek-V4.1-Flash-1017` on NTNU's LLM service by
  default, with its thinking off (`vision_reasoning_effort = "none"`). The key goes with the
  endpoint: `IDUN_API_KEY` for NTNU's service, `QUESTXTRACT_VISION_API_KEY` only for a
  `vision_base_url` on another host than NTNU's or OpenAI's, `OPENAI_API_KEY` for any other; an
  NTNU key stored as `QUESTXTRACT_VISION_API_KEY` must be renamed, and vision on OpenAI now reads
  `OPENAI_API_KEY`. A setup that only has `OPENAI_API_KEY` now stops before anything slow with one
  line naming both ways; for OpenAI, pass `--llm-url https://api.openai.com/v1 --llm-model gpt-4o
  --llm-reasoning server`, and for vision on OpenAI also `vision_reasoning_effort = ""` in
  `questxtract.toml`. `judge_reasoning_effort` is now `reasoning_effort` (the old name still
  works in `questxtract.toml`) and is sent with every judge and quiz request, not to the OCR
  model. A model that rejects `none` or takes no setting needs `reasoning_effort = ""`. The
  `questions.json` key `config.judge_reasoning_effort` is unchanged. (#173)
- `hotwords_from_vision` is removed from the settings: nothing read it, and the second
  transcription pass it was for is not planned (#62). A `questxtract.toml` that still sets it
  stops with an unknown-key error; delete the line. (#169)
- Vision has an endpoint and a key of its own, so the board can be read on another service than
  the judge's:
  `vision_base_url` in `questxtract.toml` and `QUESTXTRACT_VISION_API_KEY` in the environment or
  `.env`. The LLM detector stays on `base_url` with its own key, so the two
  run together and neither key reaches the other service. A setup that sets only `base_url` (or
  `--llm-url`) still sends vision there with the same key. To keep reading the board on OpenAI,
  set `vision_model = "gpt-5.6-luna"` and `vision_base_url = "https://api.openai.com/v1"`, and put
  the OpenAI key in `QUESTXTRACT_VISION_API_KEY`. The NTNU service needs the NTNU network or VPN.
  (#162)
- The limit on board states, and so on vision calls, grows with the lecture: `vision_max_states`
  is replaced by `vision_states_per_minute` (3.0) and `vision_min_states` (80). A 90-minute
  lecture can have 270 board states instead of 80; a lecture of 26 minutes or less keeps 80. A
  `questxtract.toml` that still sets `vision_max_states` is rejected as an unknown key. (#130)
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

- The app installs and runs with one command: the `[gui]` extra also installs `openai`, which
  the app needs since it starts with the LLM detector and reading the board on. The README and
  docs/cli.md install lines now point at this release
  (`pip install "questxtract[gui] @ git+https://github.com/mad201802/QuestXtract@v0.2.0"`), and a
  Download button at the top of the README opens the latest GitHub Release, whose wheel installs
  with `pip install "questxtract-0.2.0-py3-none-any.whl[gui]"`. (#228)
- The quiz no longer drops an item because two options differ only by a sign (`-2` and `2`) or a
  decimal point (`1.5` and `15`): the sameness check ignores only case, math delimiters, runs of
  spaces and punctuation at the end, and keeps everything inside an option. The quiz prompt (now `v1`) leaves out check-ins on the class, names the
  real expression from the board, and makes the model work the answer out instead of copying a
  lecturer's answer that does not fit the stem. (#124)
- The transcription bar starts moving while the recording is decoded to audio (the first tenth of
  the bar) instead of sitting at 0 % until the first words arrive. (#124)
- Installing no longer picks PyAV 19, with which faster-whisper 1.2.1 cannot open a recording
  (`av>=14.0,<19`). (#124)
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

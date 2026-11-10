# Command line

Every command and option, and recipes for common jobs. Library API:
[architecture.md 15](architecture.md#15-python-api); pipeline internals:
[architecture.md](architecture.md).

## Quick start

```bash
questxtract process lecture.mp4                # video -> lecture.questions.json
questxtract transcribe lecture.mp4             # transcription only -> lecture.transcript.json
questxtract process lecture.transcript.json    # questions from an existing transcript
questxtract filter lecture.questions.json --text eigenvalue   # the questions about a word
```

`questxtract --help` and `questxtract <command> --help` show the same in the terminal. From a
source checkout, prefix `uv run`.

## What decides how much runs

The input file. There are no flags for picking stages.

| You pass | What runs |
|---|---|
| an `.mp4` or `.mp3` recording | everything: check the recording, transcribe it, find the questions; without a video track (any `.mp3`, or an audio-only `.mp4`) there is no board to read |
| a `.json` transcript | everything after transcription |
| anything else | nothing: `error: ...: expected an .mp4 recording or an .mp3 recording or a .json transcript` |

`process` also takes a transcript from another ASR engine: a list of segments, or an object with
a `segments` key:

```json
{
  "segments": [
    { "text": "Today we cover sorting. What is a stable sort?", "start": 0.0, "end": 6.0 },
    { "text": "Does anyone know why that matters", "start": 9.0, "end": 12.0 }
  ]
}
```

## Commands

Defaults apply when neither a config file nor a flag sets a value.

### `questxtract process <input>`

Finds the questions in a recording or transcript and writes a questions file.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | next to the input | Where to write. `-o -` prints instead. Missing folders are created. |
| `-t`, `--threshold` | a number, 0 to 1 | `0.4` | Keeps questions with at least this confidence. |
| `-m`, `--model` | a model name | `turbo` | The Whisper model. Only for `.mp4` or `.mp3` input. See [Choosing a model](#choosing-a-model). |
| `--from-scratch` | nothing | off | Ignores cached results, transcription included. |
| `--detector` | `rules` or `llm` | `rules` | Offline rules, or a language model. The default is the [offline fallback](../README.md#offline-fallback). See [Finding questions with an LLM](#finding-questions-with-an-llm). |
| `--llm-model` | a model name | `Qwen/Qwen3.8-27B-FP8` | The model for `--detector llm`, as the endpoint names it (`qwen3.5:9b` in Ollama). |
| `--llm-url` | a URL, or `ntnu`, `openrouter`, `openai`, `ollama` | `ntnu` | OpenAI-compatible endpoint for the LLM detector, the quiz and vision: any URL, or a short name for NTNU's LLM service (`https://llm.hpc.ntnu.no/v1`, NTNU network or VPN), OpenRouter, OpenAI or Ollama on this computer (`http://localhost:11434/v1`). The key goes with the endpoint ([API keys](#api-keys-and-env)). Vision goes there too unless `vision_base_url` is set. |
| `--llm-reasoning` | `none`, `low`, `medium`, `high`, `server` | `server` | How long the judge and the quiz model may think. `server` sends no setting (Qwen rejects `none`; `gpt-4o` has no setting); DeepSeek needs `none`; `qwen3.5:9b` judges better with `low`. A level allows each request 10 minutes instead of 2. Vision uses `vision_reasoning_effort` from the settings file (default `"none"`). |
| `--vision`, `--no-vision` | nothing | off | Also reads the slides or board with a vision model (default DeepSeek on NTNU's service). Needs `questxtract[llm]`. `--no-vision` overrides `vision_enabled` for one run. Without video there is no board (a note says so). See [Reading the board](#reading-the-board). |
| `--vision-model` | a model name | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | The model that reads the board. It must take images. |
| `--vision-reader` | `vlm` or `ocr` | `vlm` | `vlm`: the vision model (text, formulas, description). `ocr`: `vision_ocr_model` (GLM-OCR: text and formulas, faster). The layout probe always uses `--vision-model`. |
| `--quiz` | nothing | off | Also writes a quiz next to the questions file (`lecture.quiz.json`), as [`questxtract quiz`](#questxtract-quiz-questionsjson). Not with `-o -`. If the quiz fails, the questions file is kept and the exit code is 1. |
| `--quiz-model` | a model name | `Qwen/Qwen3.8-27B-FP8` | The quiz model. Only with `--quiz`. |
| `--strictness`, `--options` | | | As for `questxtract quiz`. Only with `--quiz`. |
| `-h`, `--help` | nothing | | Shows the options and examples. |

`vision_fps`, `vision_states_per_minute`, `vision_min_states`, `vision_change_fraction` and
`vision_change_hold` are set in the TOML file only.

With the rules, an unanswered question scores at most `0.55`, so `-t` above `0.55` keeps only
answered questions. With `--detector llm` the confidence is the model's own.

### `questxtract quiz <questions.json>`

Writes multiple-choice items for the questions in a questions file: a stem, options with one
correct, an explanation, and why each option is right or wrong. It is the only command that
writes text that was not spoken: every text carries `origin` (`spoken` only for the question as
asked), and every item starts as pending review.

**The items are drafts: a teacher must review every item before students see it.** The model can
mark a wrong answer as correct, write a distractor that is also right, or give an explanation that
does not hold. Check each item's stem, correct option, distractors and explanation, then set its
`review.status` to `accepted` or `rejected`. `qti`, `tex` and `exerplaza` leave out rejected
items but export pending ones, so an unreviewed item reaches the export.

The model sees the question, the lecture's answer, the sentences around it, and the board states
on screen when vision ran. The speech comes from `--transcript`, else from the cached transcript
of the recording; without either, a `note:` says so and items are fewer and weaker.
`process --quiz` always has the speech. Rejected questions (`review.status` `deleted` or
`false_positive`), check-ins, and questions the model declines (addressed to the room,
rhetorical, too vague) are left out. A question the model fails on is skipped with a `warning:`;
an unreachable endpoint or a bad key ends the run. Answers are cached, so a re-run sends nothing
and gives the same file. The last line gives requests, tokens and time.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | next to the input | `lecture.questions.json` gives `lecture.quiz.json`. `-o -` prints it. |
| `--strictness` | `strict` or `lenient` | `lenient` | `lenient` also uses unanswered questions, whose answer the model writes from what was said around them; `strict` only answered ones. |
| `--options` | a number, 3 to 5 | `3` | Options per item, one correct. |
| `--transcript` | a path | the cached transcript | The lecture's transcript. |
| `--quiz-model` | a model name | `Qwen/Qwen3.8-27B-FP8` | The quiz model (`quiz_model`). |
| `--llm-url` | a URL, or `ntnu`, `openrouter`, `openai`, `ollama` | `ntnu` | As for `process`. |
| `--llm-reasoning` | `none`, `low`, `medium`, `high`, `server` | `server` | As for `process`. |
| `--config` | a path | `questxtract.toml` | The settings file; `quiz_model`, `quiz_strictness`, `quiz_options` can be set there. |
| `-h`, `--help` | nothing | | Shows the options and examples. |

### `questxtract transcribe <recording>`

Transcribes an `.mp4` or `.mp3` into a transcript file, which `process` and
`scripts/dropped_questions.py` accept.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | next to the recording | Where to write. `-o -` prints it. |
| `-m`, `--model` | a model name | `turbo` | The Whisper model. |
| `--from-scratch` | nothing | off | Transcribes again even if a cached transcript exists. |
| `-h`, `--help` | nothing | | Shows the options and examples. |

### `questxtract filter <questions.json>`

Keeps the questions that mention some words or fall in a time range, without re-running the
pipeline. Prints the result unless `-o` is given; the result is a questions file.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `--text` | words | | Keeps questions where every word appears, whole and ignoring case, in the question, its sentence, its answer, its `review` text, or a board state on screen while it was asked. The words may come from different places. `eigenvalue` does not find `eigenvalues`. |
| `--from` | a time | start | Questions at or after this time: `300`, `5:00` or `1:05:00`. |
| `--to` | a time | end | Questions before this time, same forms. |
| `-o`, `--output` | a path | the terminal | Where to write. Keyframe paths are rewritten to still resolve. |
| `-h`, `--help` | nothing | | Shows the options and examples. |

Give at least one of `--text`, `--from` and `--to`; a question must match all given.

### `questxtract qti <quiz.json>`

Exports a quiz as a QTI ZIP package: QTI 3.0 by default, or 2.1 for an LMS that does not import
3.0. Each item is single choice, scored against its correct option, with each option's
explanation as feedback; the quiz's instructions set shuffling and feedback. Items marked
`rejected` are left out. The same quiz and version always give the same bytes.

QTI 2.1 is for Canvas New Quizzes (NTNU's LMS) and Blackboard, which import 2.x but not 3.0. No
package has been imported into a real LMS yet. Each 2.1 item has its own scoring rules and also
names the `match_correct` template; an LMS that supports only templates scores every item 1 and
shows no per-option feedback, so `qti` warns about items worth other than 1 point. Some LMS skip
questions whose response processing they do not support, so check that every question arrived.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | required | Where to write the ZIP. Its folder must exist. |
| `--qti-version` | `2.1` or `3.0` | `3.0` | The QTI version. |
| `-h`, `--help` | nothing | | Shows the options. |

### `questxtract tex <quiz.json>`

Exports a quiz as TeX for ExerPlaza's upload page, `/questions/upload/tex`. Each item becomes an
`IndexedQuestion` block with its options, the correct one marked, a solution with the
explanations, and teacher notes on where the question was asked and which model wrote the text.
Pending items are visible only to you in ExerPlaza, accepted ones to everybody; rejected ones are
left out. The quiz must be in English or Italian. Texts are not LaTeX-escaped (ExerPlaza uses
MathJax). A text with unbalanced braces or a format command (such as `\answer`) fails, naming the
text, and nothing is written; so does a quiz with no items left. The same quiz always gives the
same bytes. You upload the file yourself.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | required | Where to write the `.tex` file. Its folder must exist. |
| `--content-unit` | a name | required | An ExerPlaza content unit for every question; repeat for more. Spell it exactly as in ExerPlaza, or a new unlinked unit is created. No braces, one line. |
| `-h`, `--help` | nothing | | Shows the options. |

### `questxtract exerplaza <quiz.json>`

Sends a quiz file straight to ExerPlaza, with no file to upload by hand. It logs in with your
ExerPlaza email and password, from `QUESTXTRACT_EXERPLAZA_EMAIL` and
`QUESTXTRACT_EXERPLAZA_PASSWORD` (see [API keys and .env](#api-keys-and-env)), and ExerPlaza
saves the quiz in one go: a quiz in your course session, one multiple-choice question per item,
with its options, a solution explaining every option, notes for the teachers on where the
question was asked and which model wrote the text, and the mark that the text was generated.
Accepted items are visible to everybody; rejected ones are not sent. A pending item becomes a
question only you can see, and it stays out of the quiz, since ExerPlaza shows every question of
a quiz to whoever opens it: accept it in the quiz file and send the quiz again to add it. Either
every question is saved or none is.

Sending the same quiz again adds to the course session's quiz with the same title instead of
making a new one. ExerPlaza recognises a question by its text, as its TeX upload does: an item
whose text, in the quiz's language, is already a question of that quiz or one of yours is not
saved again but listed as skipped, and that question is in the quiz as it was. So new items are
added, but an item you edited after the first import keeps its old version in ExerPlaza, options,
solution and visibility included (a pending item you accept later joins the quiz, but its
question stays marked as visible only to you): change it in ExerPlaza. Nothing is ever updated or
deleted in ExerPlaza, so no question students have answered ever changes. A quiz renamed in
ExerPlaza, or a new title, makes the next import a new quiz; if the course session has two
quizzes with the title, ExerPlaza refuses the import.

On success it prints `ExerPlaza quiz <id> (created): 3 created, 0 updated, 0 skipped`, then
one line per skipped question with ExerPlaza's reason; the quiz is `(updated)` when it already
existed. A pending item counts as created although it is not in the quiz; sent again while still
pending, it is skipped with `pending: not added to the quiz` in its reason. The quiz is checked
before anything is sent, and a broken item fails naming the field, as with `qti` and `tex`. The
password is never shown. The ExerPlaza server needs the import endpoint; without it, the command
fails with HTTP 404 and `tex` is the way in.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `--url` | an address | required | ExerPlaza's address, such as `https://exerplaza.example.org`. It must be `https`, because your password goes with the login; plain `http` only for `localhost`. |
| `--course-session` | a number | required | The course session the quiz belongs to: the number at the end of its page's address in ExerPlaza, `/course/session/<number>`. You must be a teacher of it. |
| `--content-unit` | a name | required | An ExerPlaza content unit for every question. Repeat it for more than one. Spell it exactly as in ExerPlaza: any other spelling creates a new, unlinked unit. One line. |
| `-h`, `--help` | nothing | | Shows the options. |

### `questxtract questions-tex <questions.json>`

Writes the questions of a `questions.json` as a LaTeX document that compiles with `pdflatex`, to
read or print. Each question gets its time, its confidence, the question (your reviewed text if
you edited it, with the spoken wording in a note), the lecturer's answer, and the text on the
board when it was asked. Rejected questions are left out. Texts are escaped; symbols from the
board such as Greek letters and arrows are set as math, and the board's own LaTeX is shown as it
was read, not typeset. Prints `Questions TeX, <n> questions -> <file>`.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `-o`, `--output` | a path | `<name>.questions.tex` next to the input | Where to write the `.tex` file. |
| `-h`, `--help` | nothing | | Shows the options. |

### Everywhere

| Option | What it does |
|---|---|
| `--version` | Prints the installed version. |
| `-h`, `--help` | Without a command, lists the commands; after one, that command's options. |

`questxtract eval <files|dirs> [--json] [--board]` scores the pipeline against the labelled set:
recall, precision, false positives by kind, a threshold sweep, and every miss
([tests/data/README.md](../tests/data/README.md)). `--board` adds the board states committed beside
each transcript (`<name>.visual.json`). It takes `--detector`, `--llm-model` and `--llm-url` to
compare rules and models; the first line names the detector, model and prompt version.
`--score-vision` scores the vision model (`--vision-model`, or the OCR model with
`--vision-reader ocr`) on the labelled keyframes in `tests/data/vision/`: character error rate,
blank boards read as blank and formula recall per lecture type, then every frame, worst first
([tests/data/vision/README.md](../tests/data/vision/README.md)). One cached call per frame.

## Graphical app

`questxtract gui` opens a window that runs the CLI's own code, so a run writes the same file as the
command with the same options. Install with the extra:
`pip install "questxtract[gui] @ git+https://github.com/mad201802/QuestXtract@v0.4.0"`; it
includes what the LLM detector and reading the board need, since the window starts with both on.

- The file: drop an `.mp4`, `.mp3` or `.json` on the box, or click Browse. One file at a time.
  A `.json` transcript does not offer the Transcript output.
- The output: Questions (`process`), Transcript (`transcribe`), Quiz (`process --quiz`), QTI
  package (`qti`) or TeX (ExerPlaza) (`tex`), named `<name>.questions.json`,
  `<name>.transcript.json`, `<name>.quiz.json`, `<name>.qti.zip` and `<name>.tex`, next to the
  input unless an output file is given.
- QTI package and TeX (ExerPlaza) take a finished `quiz.json`, or a recording or transcript: then
  one run makes the quiz (`process --quiz`, writing `<name>.questions.json` and `<name>.quiz.json`
  beside the input) and exports it. Cancel between the two keeps the quiz and writes no export.
  The quiz is not reviewed first: its items are pending, which `tex` writes as "only me" in
  ExerPlaza, and a QTI import lands in the platform's question bank for review. TeX asks for the
  ExerPlaza content units on the start view, separated by `;`, and Run waits for one. The app
  writes QTI 3.0; `--qti-version 2.1` is on the command line.
- The options are behind Advanced settings, below Run; the list scrolls when the screen is too
  small for it, and an export of a finished `quiz.json` says it has no options. Each control is
  one option of `process`, `transcribe` or `tex`, with its starting value in the label.
- Questions (.tex), the box below Questions, also writes `<name>.questions.tex` beside the
  questions file (`questions-tex`). Without a `questxtract.toml`
  ([Configuration files](#configuration-files)) the window starts with the LLM detector and
  reading the board on; with one, it starts from the file. A control left at the CLI's default is
  not passed, so the file still applies. LLM and quiz controls appear only for outputs that use
  them. Changing the output clears the output file. Run stays disabled, with the reason beside it,
  while a value is refused or the file is the wrong kind.
- Run, Cancel, progress: the bar follows the pipeline's stages. Cancel stops before the next
  stage, and no partial file is written. An error opens in a pop-up as the CLI's one line.
- Open folder opens the folder of the file written.
- Keys: read as the CLI reads them ([API keys](#api-keys-and-env)). The API key... button opens
  a window for the key of the endpoint in LLM endpoint (Advanced settings; a URL, or `ntnu`,
  `openrouter`, `openai` or `ollama` from its list): Test asks that endpoint once, Save tests and
  saves (a refused key is not saved; an untested one is saved with a note), Remove deletes it. The
  key is stored as plain text in the file named in the window, under the endpoint's variable,
  beside the other endpoints' keys, and is never shown, logged or put on a command line.

The window opens in English; the switch at the top right changes to Norsk or Italiano and keeps
what was entered. CLI messages stay in English.

Drag and drop uses `tkinterdnd2`, part of the extra; where it cannot start, the window says so and
Browse still works. The app needs a Python with tkinter (`python3-tk` on Linux); without it, or
without the extra, `questxtract gui` prints one `error:` line.

`quiz`, `filter`, `eval` and every flag not listed above stay on the command line.

## Configuration files

`process`, `transcribe`, `eval` and `quiz` accept `--config PATH`. Without it, they read
`questxtract.toml` in the current working directory if it exists. A missing `--config` file is an
error. With `--config`, only that file is read.

Precedence: explicit flag > file value > `Config` default.

```toml
# questxtract.toml: one flat table, with the names from config.py
model_size = "small"
min_confidence = 0.6
detector = "rules"
judge_window = 10
cache_dir = ".questxtract"
```

```bash
questxtract process lecture.mp4                    # uses small and threshold 0.6
questxtract process lecture.mp4 --threshold 0.3    # overrides only the threshold
questxtract transcribe lecture.mp4 --config other.toml
questxtract eval tests/data/ --config other.toml
```

Every `Config` field can be set. Flag names map to fields: `--model` is `model_size`,
`--threshold` `min_confidence`, `--llm-model` `judge_model`, `--llm-url` `base_url`,
`--llm-reasoning` `reasoning_effort` (`judge_reasoning_effort` is also read). `--llm-reasoning`
does not set vision: `vision_reasoning_effort` does, `"none"` by default. `reasoning_effort = ""`
equals `--llm-reasoning server`. A level (`"low"`, `"medium"`, `"high"`) raises the request
timeout to at least 600 seconds for `process` and `eval`.

The file must be UTF-8; a leading BOM is accepted. No nested sections or `[tool.questxtract]`.
Integer fields need integers; float fields also take integers. Leave a key out to keep its
default. Typos and wrong types are errors. The threshold must be between 0 and 1 and the detector
`rules` or `llm`. A relative `cache_dir` is resolved from the working directory.

### API keys and .env

A TOML key containing `key`, `token`, `secret` or `password` (any case) is rejected without
echoing its value. A hosted LLM needs `questxtract[llm]` and a key.

The key goes with the endpoint, told by the URL's host:

| Endpoint | Key |
|---|---|
| NTNU's LLM service (`ntnu`, the default) | `IDUN_API_KEY` (also `QUESTXTRACT_NTNU_API_KEY`) |
| OpenAI (`openai`) | `OPENAI_API_KEY` |
| OpenRouter (`openrouter`) | `OPENROUTER_API_KEY` |
| a `vision_base_url` on any other host | `QUESTXTRACT_VISION_API_KEY` |
| any other host, a local server such as Ollama included | `QUESTXTRACT_LLM_API_KEY`, or none |

No key is sent to another service. NTNU's, OpenAI's and OpenRouter's endpoints need their key and
are refused over plain `http://`, where it would travel unencrypted; any other host runs without a
key when its variable is unset. Each name is read from the environment, else from `.env` in the
working directory, else from the app's key file (below). `.env` takes `NAME=VALUE` lines with
optional quotes, blank lines and `#` comments; other variables are ignored and parent directories
are not searched. The key is never stored in `Config`, logs, cache entries or `questions.json`.

The key file: the app's API key... button writes the endpoint's variable (`IDUN_API_KEY="..."`,
`OPENROUTER_API_KEY="..."`) to `questxtract/.env` in your user config folder (`%APPDATA%` on
Windows, `~/Library/Application Support` on macOS, `$XDG_CONFIG_HOME` or `~/.config` elsewhere),
keeping its other lines. On Linux and macOS the file and its folder are readable by you only; on
Windows they are as private as your profile folder. The key is plain text there; for a safer
store, use [Setting up the API key](#setting-up-the-api-key).

The `.env` reader does not run commands or expand variables. Load a key from a password store
into the environment before running QuestXtract; the environment wins.

**The ExerPlaza login.** `exerplaza` reads your ExerPlaza email and password from
`QUESTXTRACT_EXERPLAZA_EMAIL` and `QUESTXTRACT_EXERPLAZA_PASSWORD`, in the environment or in
`.env` in the working directory, the same way: never from the command line, where the shell
history would keep it, and never from the app's key file. In `.env`, put the password in single
quotes, `QUESTXTRACT_EXERPLAZA_PASSWORD='...'`: unquoted, a `#` starts a comment and a backslash
is dropped, so ExerPlaza gets another password. A password containing `$`, a backtick or a `'`
cannot go in `.env`; set it in the environment.

### Setting up the API key

A hosted model needs a key; a local server (`--llm-url ollama`) does not. The key goes with the
endpoint (the table above). NTNU's service needs the NTNU network or VPN, and NTNU asks that heavy
use wait until after 18:00 or the weekend. There is no `--api-key` flag. These steps keep the NTNU
key in the system password store (Keychain on macOS, Credential Manager on Windows) and let a
git-ignored `.env` refer to it; for another endpoint, use its name for `ntnu` and its variable for
`IDUN_API_KEY` (`openrouter` and `OPENROUTER_API_KEY`). Run them in Terminal on macOS or Git Bash
on Windows, from the repository folder.

1. Save the key. Copy it, then store it from the clipboard:

   ```bash
   pbpaste | tr -d '[:space:]' | uvx keyring set questxtract-llm ntnu             # macOS
   cat /dev/clipboard | tr -d '[:space:]' | uvx keyring set questxtract-llm ntnu  # Windows
   ```

   Then empty the clipboard (`pbcopy < /dev/null` on macOS, copy other text on Windows). macOS's
   `security add-generic-password -w` prompt cuts a key at 128 characters, so pipe it in.

2. Check it. This prints the last three characters:

   ```bash
   uvx keyring get questxtract-llm ntnu | tail -c 4
   ```

3. Point `.env` at it, once:

   ```bash
   echo 'export IDUN_API_KEY="$(uvx keyring get questxtract-llm ntnu)"' >> .env
   ```

4. Load it in every new terminal:

   ```bash
   source .env
   ```

   On macOS, choose *Always Allow* if Python asks for the Keychain item. In PowerShell:
   `$env:IDUN_API_KEY = (uvx keyring get questxtract-llm ntnu)`.

5. Test it (listing models costs nothing):

   ```bash
   uv run python -c "import os, openai; print(len(openai.OpenAI(base_url='https://llm.hpc.ntnu.no/v1', api_key=os.environ['IDUN_API_KEY']).models.list().data), 'models visible - the key works')"
   ```

| Error | Cause | Fix |
|---|---|---|
| `--detector llm goes to NTNU's LLM service ..., so it needs IDUN_API_KEY` | The key is not loaded in this terminal | `source .env` |
| `... Connection error` with NTNU's service | Not on the NTNU network | Connect the NTNU VPN |
| `401 ... invalid_api_key` | The stored key is incomplete or revoked | Compare its end with the original (step 2), save it again (step 1), or ask for a new key |
| `400 ... reasoning_effort` | The judge or quiz model rejects `none` (Qwen) or has no setting (`gpt-4o`); or the vision model has no setting but vision sends `none` | Judge: `--llm-reasoning server`. Vision: `vision_reasoning_effort = ""` in `questxtract.toml` |

## Where files go

Without `-o`, output lands next to the input:

| Input | `process` writes | `transcribe` writes |
|---|---|---|
| `lecture.mp4` | `lecture.questions.json` | `lecture.transcript.json` |
| `lecture.transcript.json` | `lecture.questions.json` | (needs an `.mp4` or `.mp3`) |

`filter` prints unless `-o` names a file.

With vision, keyframe images are copied next to the questions file (`lecture.keyframes/` for
`lecture.questions.json`) and referenced by relative path, so move the two together. With `-o -`
nothing is copied and the file points into the cache.

A second run overwrites the file. Format: [architecture.md 12.1](architecture.md#121-questionsjson).

## Recipes

Get the questions from a lecture video:

```bash
questxtract process lecture.mp4
```

Quick test run on a laptop (smaller model, faster, less accurate):

```bash
questxtract process lecture.mp4 -m base
```

Another threshold without transcribing again (the transcript is cached):

```bash
questxtract process lecture.mp4 -t 0.4
questxtract process lecture.mp4 -t 0.3 -o lecture.questions-0.3.json
```

Keep the transcript:

```bash
questxtract transcribe lecture.mp4                 # writes lecture.transcript.json
questxtract process lecture.transcript.json        # writes lecture.questions.json
```

`transcribe` reuses the transcript cached by `process`, and the other way round.

Print the result instead of writing a file:

```bash
questxtract process lecture.transcript.json -o -
```

Everything in one run: the board, the LLM detector and a quiz, all on NTNU's service with
`IDUN_API_KEY`. Writes `lecture.questions.json`, `lecture.keyframes/` and `lecture.quiz.json`;
`qti` then packs the quiz:

```bash
questxtract process lecture.mp4 --vision --detector llm --quiz
questxtract qti lecture.quiz.json -o lecture.qti.zip
```

Export for an LMS:

```bash
questxtract qti lecture.quiz.json -o lecture.qti.zip   # QTI 3.0, every item but the rejected ones
questxtract qti lecture.quiz.json -o lecture.qti21.zip --qti-version 2.1   # QTI 2.1, see the caveat
```

Export for ExerPlaza:

```bash
questxtract tex lecture.quiz.json -o lecture.tex --content-unit "Improper integrals"
```

Upload `lecture.tex` at `/questions/upload/tex`, check the preview and save. On a re-upload,
ExerPlaza asks about every question it already has: choose Cancel, because Overwrite deletes the
question and the students' answers to it.

The questions as a document to read or print:

```bash
questxtract questions-tex lecture.questions.json   # -> lecture.questions.tex
pdflatex lecture.questions.tex
```

Send a quiz straight to ExerPlaza:

```bash
export QUESTXTRACT_EXERPLAZA_EMAIL=you@example.org
read -rs QUESTXTRACT_EXERPLAZA_PASSWORD && export QUESTXTRACT_EXERPLAZA_PASSWORD
questxtract exerplaza lecture.quiz.json --url https://exerplaza.example.org \
    --course-session 42 --content-unit "Improper integrals"
```

The number 42 comes from the course session's page, `/course/session/42`. Pending items stay out
of the quiz. Run it again after adding items or accepting pending ones: the new and the accepted
ones join the quiz, and those ExerPlaza already has are skipped and left as they are.

See which questions the threshold dropped (a development script):

```bash
questxtract transcribe lecture.mp4
uv run python scripts/dropped_questions.py lecture.transcript.json -o lecture.questions.json
# writes lecture.questions.json (kept) and lecture.questions.dropped.json (dropped)
```

Find the questions about a word, said or on the board:

```bash
questxtract filter lecture.questions.json --text eigenvalue
questxtract filter lecture.questions.json --text "transfer function" --from 5:00 --to 20:00 -o tf.json
```

Start over after changing the code or on a suspected stale result:

```bash
questxtract process lecture.mp4 --from-scratch
```

## Finding questions with an LLM

`--detector llm` sends every sentence ending in `?`, and every question the rules found, to a
language model with the lecture around it. It decides which are real questions, finds the
sentences that answer them, and may reword a messy question; the spoken version stays as
`original_text`. It judges only the sentences it was given, by id, so it cannot add a question.
The rules run first; their `reasons` stay on the questions the model keeps, plus `llm`.

Needs the `llm` extra: `pip install "questxtract[llm] @ git+https://github.com/mad201802/QuestXtract"`.

NTNU's LLM service (the default): `Qwen/Qwen3.8-27B-FP8`, free, on the NTNU network or VPN, with
the key in `IDUN_API_KEY` ([Setting up the API key](#setting-up-the-api-key)). To judge with
DeepSeek: `--llm-model deepseek-ai/DeepSeek-V4.1-Flash-1017 --llm-reasoning none`.

```bash
questxtract process lecture.mp4 --detector llm
```

OpenAI: key in `OPENAI_API_KEY`, `--llm-url` pointing at OpenAI. `gpt-4o` takes no reasoning
setting (`server`); a thinking model such as `gpt-5.6-terra` takes a level. OpenAI shuts the
gpt-4o snapshots down on 14 January 2027.

```bash
questxtract process lecture.mp4 --detector llm --llm-url openai \
    --llm-model gpt-4o --llm-reasoning server
questxtract process lecture.mp4 --detector llm --llm-url openai \
    --llm-model gpt-5.6-terra --llm-reasoning low
```

OpenRouter: its key in `OPENROUTER_API_KEY` (stored the same way, or saved in the app),
`--llm-url openrouter` and OpenRouter's model names, which differ from NTNU's
(`deepseek/deepseek-v4.1-flash`, not `deepseek-ai/DeepSeek-V4.1-Flash-1017`). DeepSeek needs
`--llm-reasoning none`.

```bash
questxtract process lecture.mp4 --detector llm \
    --llm-url openrouter --llm-model deepseek/deepseek-v4.1-flash --llm-reasoning none
```

Another hosted provider: its key in `QUESTXTRACT_LLM_API_KEY`, its URL and model name.

Local: any OpenAI-compatible server (Ollama, llama.cpp's `llama-server`, vLLM, LM Studio), by URL
or `ollama` for `http://localhost:11434/v1`. It needs no key and gets none of the hosted ones. Give
[Ollama](https://ollama.com) an 8k context: below 24 GB of graphics memory it defaults to 4k and
cuts longer prompts.

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve        # in one terminal
ollama pull qwen3.6:35b                        # once; needs about 24 GB
questxtract process lecture.mp4 --detector llm --llm-url ollama --llm-model qwen3.6:35b
```

On a smaller machine, `qwen3.5:9b` with thinking is slow but good:

```bash
questxtract process lecture.mp4 --detector llm --llm-reasoning low \
    --llm-url http://localhost:11434/v1 --llm-model qwen3.5:9b
```

Without thinking a request takes seconds, with it minutes; a lecture makes 15 to 25. Compare
models with `eval`:

```bash
questxtract eval tests/data/ --detector llm --llm-url http://localhost:11434/v1 --llm-model qwen3.5:9b
```

Caching: every request is cached in `.questxtract/judge/`, keyed on what was sent. Another `-t` or
a second `eval` makes no calls; a different model, endpoint, transcript or prompt asks again.
`--from-scratch` ignores the cache.

Cost: the `stats` block of `questions.json` has `judge_requests`, `judge_input_tokens`,
`judge_output_tokens` and `judge_ms`, and the same with `layout_` and `vision_` when vision is on.
Tokens are as the endpoint reports them (0 when it reports none); a cached run shows 0.

When the endpoint fails, the run stops with one `error:` line
([When something goes wrong](#when-something-goes-wrong)); there is no fallback to the rules.
Answered requests stay cached, so the same command resumes.

## Reading the board

A vision model reads the slides or board. Each stable view becomes a board state with its text,
formulas as LaTeX, a short description and a confidence, linked to the questions asked while it
was on screen. Needs the `llm` extra and a model that takes images. Default: DeepSeek-V4.1-Flash
on NTNU's service, with the LLM detector's key (`IDUN_API_KEY`) and its thinking off. One run:

```bash
questxtract process lecture.mp4 --vision        # DeepSeek on NTNU
questxtract process lecture.mp4 --vision --llm-url http://localhost:11434/v1 \
    --vision-model qwen3-vl:2b-instruct   # a local model in Ollama; see vision_reasoning_effort below
```

Every run, in `questxtract.toml` (`--no-vision` turns it off for one run):

```toml
vision_enabled = true
vision_model = "deepseek-ai/DeepSeek-V4.1-Flash-1017"   # the default, on NTNU's LLM service
# vision_base_url = "http://localhost:8000/v1"         # vision elsewhere; the judge stays on base_url
# vision_reasoning_effort = ""                         # for a vision model with no setting, such as qwen3-vl
```

Without `vision_base_url`, vision goes to `base_url` (`--llm-url`) with that endpoint's key.

`--vision-reader ocr` (`vision_reader = "ocr"`) reads keyframes with GLM-OCR through the same
endpoint: faster, text and formulas, no description. The vision model still finds the board once
per lecture, so OCR reads only the board. GLM-OCR runs on an 8 GB laptop through Ollama; give it a
small vision model for finding the board, or every run warns that it reads the full frame:

```bash
ollama pull glm-ocr:q8_0
ollama pull qwen3-vl:2b-instruct
```

```toml
vision_enabled = true
vision_reader = "ocr"
vision_base_url = "http://localhost:11434/v1"
vision_ocr_model = "glm-ocr:q8_0"
vision_model = "qwen3-vl:2b-instruct"   # finds the board; leave it out to read the full frame
vision_reasoning_effort = ""            # send no reasoning setting to the local models
```

A server that returns no token probabilities still works; its readings get confidence 0.5 ("not
measured").

A lecture costs one call to find the board and at most 3 per minute to read it, at least 80, so up
to 270 for 90 minutes (`vision_states_per_minute`, `vision_min_states`), plus resends. A slide or
page that comes back unchanged is not read again. Readings and keyframes are cached. A failed
reading leaves its view out with a warning. Example output:
[demo/](demo/mit_18_100b_l1.questions.json), four questions from MIT 18.100B Lecture 1 (MIT
OpenCourseWare, CC BY-NC-SA) with their boards; its `stats` count the whole lecture. Internals:
[architecture.md 6](architecture.md#6-vision).

## Choosing a model

`-m` picks the Whisper model size.

| Name | Download | When to use it |
|---|---|---|
| `tiny`, `base` | ~140 MB for `base` | quick test runs, CI |
| `small`, `medium` | in between | when `turbo` is too slow or too big |
| `turbo` (default) | ~1.5 GB | real lectures: close to `large-v2` quality, about 8x faster than `large-v3` |
| `large-v3` | largest | most accurate, slowest |

Any other faster-whisper name works, such as `small.en`; an unknown name fails with the valid
list. The first run downloads the model into `~/.cache/huggingface`; meanwhile the bar sits at
`Transcribing 0%`.

## The cache

Results are cached in `.questxtract/` in the folder you run the command from.

| Changing this... | ...transcribes again? |
|---|---|
| `-t`, `-o`, `--detector`, `--llm-model`, `--llm-url`, `--llm-reasoning` | no |
| `-m` | yes |
| `--vision-model`, `--vision-reader`, a `vision_*` setting, or `--llm-url` / `base_url` with vision on | no, but the board is read again; keyframes already read with the same model are not resent |
| `--from-scratch` | yes, and everything else is recomputed |
| the video file itself | yes: the cache is keyed on the file's content |

A `.json` transcript input caches only the model requests.

Within one run, a slide or page that comes back unchanged is copied, also with `--from-scratch`,
and counted in `stats.vision_reused` ([architecture.md 6.5](architecture.md#65-captioning)).

## When something goes wrong

| Exit code | Meaning |
|---|---|
| `0` | success |
| `1` | an error, explained in one line starting with `error:` |
| `2` | the command itself was wrong: a missing argument, or `-t abc` |
| `130` | stopped with Ctrl+C |

| Message | What to do |
|---|---|
| `...: no such file` | Check the path. |
| `...: expected an .mp4 recording or an .mp3 recording or a .json transcript` | Convert the recording to MP4 or MP3 first, e.g. with ffmpeg. |
| `...: recording contains no audio track` | The file has video only. |
| `...: invalid or unreadable recording` | The file is damaged, or not really an MP4 or MP3. |
| `Invalid model size '...', expected one of: ...` | Use a listed name. |
| `...: not valid JSON` / `expected a list of segments...` | The `.json` file is not a transcript. |
| `--detector llm goes to NTNU's LLM service (...), so it needs IDUN_API_KEY ...` | Make the NTNU key available, or pass `--llm-url` for OpenAI or a local server. |
| `--detector llm goes to OpenAI, so it needs OPENAI_API_KEY ...` | Make the OpenAI key available, or pass `--llm-url` for another endpoint. |
| `...: http://... would send the key unencrypted; use https://` | Use `https://` for NTNU's, OpenAI's or OpenRouter's endpoint, or its short name. |
| `--detector llm goes to OpenRouter, so it needs OPENROUTER_API_KEY ...` | Put the key in `OPENROUTER_API_KEY` (`OPENAI_API_KEY` goes to OpenAI only), or save it in the app. |
| `<file>: invalid TOML at line N` | Fix the TOML at that line. |
| `<file>: <key>: expected float, got string` | Use the expected type; do not quote numbers. |
| `<file>: unknown config field(s): ...` | Use a field name from `config.py`. |
| `<file>: <key>: secrets belong in the endpoint's key variable, ...` | Move the secret to the environment or `.env`. |
| `min_confidence / --threshold must be between 0 and 1` | Fix the threshold in the file or command. |
| `detector must be rules or llm` | Fix the detector name. |
| `.env: line N: expansion is not supported; ...` | Load the key into the environment from your shell or password store. |
| `--detector llm needs the LLM extra: ...` | Install `questxtract[llm]`. |
| `--vision / vision_enabled goes to NTNU's LLM service (...), so it needs IDUN_API_KEY ...` | Make the NTNU key available, set `vision_base_url` or `--llm-url`, or use `--no-vision`. |
| `...: <model> is not a model on NTNU's LLM service ...` / `...: <model> is not an OpenAI model ...` / `...: <model> is not OpenRouter's name for a model ...` | The model belongs to another service (NTNU's and OpenRouter's are `organisation/model`, OpenAI's have no `/`, OpenRouter names NTNU's defaults differently). Change the model or the endpoint as the message says. |
| `--vision / vision_enabled needs the LLM extra: ...` | Install `questxtract[llm]`, or use `--no-vision`. |
| `note: <file> has no video, so the board is not read` | Not an error: no video means no board. |
| `judge: <endpoint>: ...` | The endpoint failed: not running, wrong key, or unknown model; the rest of the message says which. |
| `judge: <endpoint>: ... reasoning_effort` (HTTP 400) | The model has no reasoning setting (`gpt-4o`) or rejects `none` (Qwen). Pass `--llm-reasoning server`. |
| `judge: <endpoint>: no answer within 120 s, twice` | The model did not answer in time, twice. DeepSeek: pass `--llm-reasoning none`. Another thinking model: pass a level, which allows 10 minutes. Otherwise the provider is overloaded. |
| `judge: <model> did not return the expected JSON, twice` | Try a larger model, or rerun. |
| `exerplaza: set QUESTXTRACT_EXERPLAZA_EMAIL ...` / `... PASSWORD ...` | Put your ExerPlaza email and password in the environment or `.env` (see [API keys and .env](#api-keys-and-env)). |
| `exerplaza: url: http://... is not encrypted ...` | Use the `https://` address; plain `http` is for a local ExerPlaza only. |
| `exerplaza: POST .../login: ExerPlaza refused the login ...` | Wrong email or password, or a disabled account: try them on ExerPlaza's login page. A password in `.env` with a `#` or a backslash needs single quotes ([API keys and .env](#api-keys-and-env)). |
| `exerplaza: POST .../api/v1/quizzes/import: ... HTTP 403 ...` / `... HTTP 404 Not Found: course session ...` | You do not teach that course session, or the number is wrong: check it in the page's address. |
| `exerplaza: ... HTTP 404, not a JSON answer ...` | This ExerPlaza has no import endpoint yet, or `--url` points elsewhere. Use `tex` meanwhile. |
| `exerplaza: ... HTTP 409 Conflict: course session ... has 2 quizzes named ...` | ExerPlaza cannot tell which quiz to add to: rename all but one in ExerPlaza, or give the quiz another `title`. Nothing was saved. |
| `exerplaza: ... HTTP 422 ...` with `questions[...]` lines | ExerPlaza refused the quiz; each line names the item and field, as ExerPlaza sent them. Nothing was saved. |
| `exerplaza: ...: cannot reach ExerPlaza (...)` / `no answer within 30 s` | Check the address and your network or VPN; try again later. |
| `exerplaza: POST .../api/v1/quizzes/import: no answer within 30 s. ExerPlaza may still have saved the quiz ...` | A big quiz can take ExerPlaza longer than the wait. Run the same command again: items ExerPlaza already has are skipped, so nothing is saved twice. |
| `judge: <model> judged none of the N candidates; ...` | The endpoint answered in another shape. Check that the server supports `response_format`; try another model or server. |

A missing file, a wrong file type, and `--detector llm` without a key or the extra are caught
before anything runs; a damaged recording or one without audio by the first stage. A bad input
never costs a transcription.

## Not available yet

- `--language`: every lecture is transcribed as English, and detection matches English words
  (requirement 4.2), so no flag is planned. `Config(language=...)` or `language` in the TOML file
  exists for experiments.
- YouTube URLs: download the video first.

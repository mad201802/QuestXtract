<div align="center">

# 🎓 QuestXtract

**Turn lecture recordings into timestamped questions and quizzes.**

Feed it a lecture video, get back every question that was actually asked in it, with timestamps,
confidence scores and a clean JSON export.

[![CI](https://github.com/mad201802/QuestXtract/actions/workflows/ci.yml/badge.svg)](https://github.com/mad201802/QuestXtract/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#roadmap)

</div>

---

## Why

Lectures are full of questions the lecturer asks the room, and they are exactly the questions
students should be able to answer afterwards. QuestXtract finds them, keeps their timestamps, and
turns them into study material. It **extracts** questions, it never invents them.

## Features

| | |
|---|---|
| 🎥 **Video or audio in** | `.mp4` or `.mp3` recordings, audio track extracted automatically (no system ffmpeg) |
| 🗣️ **Transcription** | Speech-to-text with faster-whisper, or any ASR engine behind a small interface |
| 🖼️ **Board and slides** | Optional: a vision model finds the slides or board in the frame, whatever the recording style, then reads each view into text, LaTeX and a description, linked to the questions asked while it was on screen |
| ❓ **Question detection** | Fast offline rules, or an LLM you pick (local in Ollama, or hosted) that reads the transcript; a confidence score and the reasons for every question |
| ⏱️ **Timestamps** | Every question carries its position in the recording |
| 🔎 **Search** | Find the questions by words said or written on the board while they were asked, and by time |
| ✏️ **Human in the loop** | A `review` block per question for teacher edits, never overwritten by the pipeline |
| 📦 **JSON export** | `questions.json` with a fixed key set and a formal JSON Schema |
| 🧩 **Quizzes** | Multiple-choice items from the questions asked, with explanations; every generated text is marked and starts as pending review |
| 🎓 **LMS export** | `questxtract qti` writes a `quiz.json` as a QTI 3.0 or 2.1 package for an LMS to import |

What is built today:
[architecture.md 13.3](docs/architecture.md#133-status).

## Install

```bash
# with uv (recommended)
uv add git+https://github.com/mad201802/QuestXtract@v0.1.0

# or with pip
pip install git+https://github.com/mad201802/QuestXtract@v0.1.0
```

Drop the `@v0.1.0` for the latest `main`. Releases: [GitHub Releases](https://github.com/mad201802/QuestXtract/releases).

Requires Python 3.10+ on Linux, macOS or Windows.

## Quickstart

### CLI

```bash
questxtract process lecture.mp4                # video -> lecture.questions.json
questxtract transcribe lecture.mp4             # transcription only -> lecture.transcript.json
questxtract process lecture.transcript.json    # questions from an existing transcript
questxtract quiz lecture.questions.json        # multiple-choice quiz -> lecture.quiz.json
questxtract filter lecture.questions.json --text eigenvalue   # the questions about a word
```

The input decides how much runs: an `.mp4` or `.mp3` goes through every stage (an audio-only
recording simply has no board to read), while a `.json` transcript skips transcription. A progress bar renders on stderr, and transcripts are cached, so running
`process` again with another threshold takes seconds. Every option, the cache, error messages and
recipes: [docs/cli.md](docs/cli.md).

Save settings in `questxtract.toml`, or choose `--config PATH`; explicit command flags win. See [configuration files](docs/cli.md#configuration-files).

`process` also takes a transcript made by any other ASR engine: a list of segments, or an object
with a `segments` key, which is what most engines emit:

```json
{
  "segments": [
    { "text": "Today we cover sorting. What is a stable sort?", "start": 0.0, "end": 6.0 },
    { "text": "Does anyone know why that matters", "start": 9.0, "end": 12.0 }
  ]
}
```

### With an LLM

The rules are fast and offline, but they miss what they were not written for. `--detector llm`
lets a language model decide instead: it judges every sentence ending in `?` in the lecture around
it, keeps the real questions, finds the sentences that answer them, and rewrites a messy question
into a clean one, keeping the spoken version beside it. It never adds a question that was not
spoken. It needs the `llm` extra
(`pip install "questxtract[llm] @ git+https://github.com/mad201802/QuestXtract"`) and a model:

```bash
# the default: DeepSeek-V4.1-Flash on NTNU's LLM service, key from QUESTXTRACT_NTNU_API_KEY
questxtract process lecture.mp4 --detector llm

# OpenAI instead, key from OPENAI_API_KEY; gpt-4o takes no reasoning setting, hence server
questxtract process lecture.mp4 --detector llm --llm-url https://api.openai.com/v1 \
    --llm-model gpt-4o --llm-reasoning server

# local: any OpenAI-compatible server, here Ollama with a small Qwen that thinks first
questxtract process lecture.mp4 --detector llm --llm-reasoning low \
    --llm-url http://localhost:11434/v1 --llm-model qwen3.5:9b
```

Answers are cached, so another `-t` costs no calls. Setting up Ollama, choosing a model, and what
happens when the endpoint fails: [docs/cli.md](docs/cli.md#finding-questions-with-an-llm).

To measure the detector against the labelled reference recordings in `tests/data/`
([how they were made](tests/data/README.md)):

```bash
questxtract eval tests/data/          # table, sweep, missed questions and false positives
questxtract eval tests/data/ --json   # the same report, machine-readable
questxtract eval tests/data/ --board  # with the board states committed beside the transcripts
questxtract eval tests/data/ --detector llm --llm-url http://localhost:11434/v1 --llm-model qwen3.5:9b
questxtract eval --score-vision tests/data/vision/   # how well vision_model reads slides and boards
```

#### Setting up the API key

A hosted model needs a key; a local server (`--llm-url`) does not. The key goes with the
endpoint: NTNU's LLM service, the default, reads `QUESTXTRACT_NTNU_API_KEY`; OpenAI and any other
endpoint read `OPENAI_API_KEY`, so neither key is ever sent to the other service. NTNU's service
needs the NTNU network or VPN, and NTNU asks that heavy use wait until after 18:00 or the
weekend. QuestXtract reads each key from the environment, or from a local `.env` (see
[docs/cli.md](docs/cli.md#api-keys-and-env)); there is no `--api-key` flag and TOML files
reject secrets, so a key never sits in the repository or in your shell history. The steps below
use the NTNU key; for an OpenAI key, use `openai` and `OPENAI_API_KEY` in its place. The setup
below keeps it in your system's password store (the Keychain on macOS, Credential Manager on
Windows) and lets a git-ignored `.env` file *refer* to it, so no file anywhere holds the key
itself. The environment wins over `.env`, which is why step 4 loads it into the shell. It is the same on
macOS and Windows: run it in Terminal on macOS, in **Git Bash** on Windows (it comes with Git for
Windows), from the repository folder.

1. **Save the key.** Copy the whole key, then store it from the clipboard:

   ```bash
   pbpaste | tr -d '[:space:]' | uvx keyring set questxtract-llm ntnu             # macOS
   cat /dev/clipboard | tr -d '[:space:]' | uvx keyring set questxtract-llm ntnu  # Windows
   ```

   Then empty the clipboard (`pbcopy < /dev/null` on macOS, copy any other text on Windows).
   Piping the key in avoids typing it into a hidden prompt: macOS's own
   `security add-generic-password -w` prompt silently cuts a key at 128 characters, and keys
   can be longer.

2. **Check it.** This prints only the last three characters; they must match your key's:

   ```bash
   uvx keyring get questxtract-llm ntnu | tail -c 4
   ```

3. **Point `.env` at it.** One line per key, written once; `.env` is in `.gitignore` and holds
   no key:

   ```bash
   echo 'export QUESTXTRACT_NTNU_API_KEY="$(uvx keyring get questxtract-llm ntnu)"' >> .env
   ```

4. **Load it in every new terminal**, before running `questxtract`:

   ```bash
   source .env
   ```

   On macOS the first read may ask whether Python may use the Keychain item: choose *Always
   Allow*. In PowerShell instead of Git Bash, step 4 is
   `$env:QUESTXTRACT_NTNU_API_KEY = (uvx keyring get questxtract-llm ntnu)`.

5. **Test it.** Listing the models costs nothing:

   ```bash
   uv run python -c "import os, openai; print(len(openai.OpenAI(base_url='https://llm.hpc.ntnu.no/v1', api_key=os.environ['QUESTXTRACT_NTNU_API_KEY']).models.list().data), 'models visible - the key works')"
   ```

| Error | Cause | Fix |
|---|---|---|
| `--detector llm goes to NTNU's LLM service ..., so it needs QUESTXTRACT_NTNU_API_KEY` | The key is not loaded in this terminal | `source .env` |
| `... Connection error` with NTNU's service | Not on the NTNU network | Connect the NTNU VPN |
| `401 ... invalid_api_key` | The stored key is incomplete, or was revoked | Compare its end with the original (step 2) and save it again (step 1); if they match, ask for a new key |
| `400 ... reasoning_effort` | The model has no reasoning setting (`gpt-4o`), or rejects `none` (Qwen); the default sends `none`, which DeepSeek needs | Pass `--llm-reasoning server` |

**For the project team.** We use NTNU's LLM service with our NTNU key, and the customer's
pay-as-you-go OpenAI key only for the OpenAI fallback, with `gpt-4o`. Never
commit the key, paste it into a chat, issue or pull request, or write it into any file; once it is
saved, delete the message you received it in. Ask the customer before using another model with it.
If a key leaks, tell its owner at once so it can be revoked.

### Reading the board

A vision model can also read the slides or the board. Each stable view of the recording becomes a
board state: its text, its formulas as LaTeX, a short description and a confidence, linked to the
questions asked while it was on screen. It needs the `llm` extra and a model that takes images.
By default it is the same model as the LLM detector, DeepSeek-V4.1-Flash on NTNU's LLM service,
with the same key (`QUESTXTRACT_NTNU_API_KEY`, set up as above) and its thinking off: with it on,
DeepSeek reads a frame far slower for nearly the same reading. Switch vision on for
one run with `--vision`:

```bash
questxtract process lecture.mp4 --vision        # DeepSeek on NTNU
questxtract process lecture.mp4 --vision --llm-url http://localhost:11434/v1 \
    --vision-model qwen3-vl:2b-instruct --llm-reasoning server   # a local model in Ollama
```

or for every run in `questxtract.toml` (`--no-vision` turns it off again for one run):

```toml
vision_enabled = true
vision_model = "deepseek-ai/DeepSeek-V4.1-Flash-1017"   # the default, on NTNU's LLM service
# vision_base_url = "http://localhost:8000/v1"         # vision elsewhere; the judge stays on base_url
```

Without `vision_base_url`, vision goes where `base_url` (`--llm-url`) points, NTNU's service by
default, with that endpoint's key.

`--vision-reader ocr` (or `vision_reader = "ocr"`) reads the keyframes with GLM-OCR instead,
through the same endpoint: faster and lighter, with the text and formulas but no description.
The vision model still finds the board first, once per lecture, so the room and the app around
it are cropped away before OCR reads the frame. GLM-OCR also runs on an 8 GB laptop through
Ollama. Give it a small vision model for finding the board too, or every run warns that it reads
the full frame, room included:

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

A server that does not return token probabilities still works; its readings get confidence 0.5,
which means "not measured".

A lecture costs one model call to find the slides or board and at most 3 per minute to read
them, at least 80 allowed, so up to 270 for a 90-minute lecture (`vision_states_per_minute`,
`vision_min_states`), plus a resend when a call fails. A slide or page that comes back later in the
lecture with nothing changed is not read again: its earlier reading is copied. Readings and
keyframe images are cached, so running it again sends nothing. A reading that fails leaves its
view out with a warning, and the run goes on. Example output:
[docs/demo/](docs/demo/mit_18_100b_l1.questions.json), four questions from MIT 18.100B Lecture 1
(MIT OpenCourseWare, CC BY-NC-SA) with the boards they were asked about; `stats` there still
count the whole lecture. How it works:
[architecture.md 6](docs/architecture.md#6-vision).

### Everything in one run

Reading the board, the LLM detector and the quiz run together in one command, all on
DeepSeek at NTNU's service with `QUESTXTRACT_NTNU_API_KEY`; the key is checked before anything
slow starts:

```bash
questxtract process lecture.mp4 --vision --detector llm --quiz
# -> lecture.questions.json, lecture.keyframes/ and lecture.quiz.json
questxtract qti lecture.quiz.json -o lecture.qti.zip   # the quiz as a QTI 3.0 package for an LMS
```

### Output

`questions.json` lists every question with its text, `start`/`end` in seconds, `confidence`, the
`answer` where one was found, the `reasons` (rules) that fired, and a `review` block for teacher
edits. Its `config` block records how it was made (the detector, and with `--detector llm` the
model, prompt version and reasoning level), so two saved runs can be told apart. Segments and
board states are listed once and referenced by id. Every key is always
present, `null` included. Example and field reference:
[architecture.md 12.1](docs/architecture.md#121-questionsjson); formal schema:
[questions.schema.json](src/questxtract/questions.schema.json).

`--threshold` gates the final **confidence**. With the rules, an unanswered question tops out at
0.55, and one the next sentence answers carries its full question score; with `--detector llm`,
it is the model's own confidence.

### Python API

```python
from pathlib import Path

from questxtract import STAGES, Config, Context, Transcript, run, write_questions

ctx = Context(
    source=Path("lecture.mp4"),
    config=Config(min_confidence=0.4),
    transcript=Transcript.from_file("lecture.transcript.json"),  # or let stage 2 produce it
)
ctx = run(ctx, STAGES[2:])  # skip ingest + transcribe: the transcript is already in hand

for pair in ctx.pairs:
    print(f"[{pair.start:7.2f}s] ({pair.confidence:.0%}) {pair.question.text}")

write_questions(ctx, "questions.json")
```

The pipeline is a list of `Context -> Context` functions. Passing your own `stages=` list is the
whole plugin system: swap the detector, drop a stage, add an enrichment step:

```python
run(ctx, [my_detector if s.__name__ == "detect" else s for s in STAGES[2:]])
```

### Filtering questions

`filter_questions` narrows a finished `Context`, `filter_document` a `questions.json` document
(edited or not). Both return a new value and leave the input untouched:

```python
from questxtract import filter_document, filter_questions, process, read_questions

ctx = process("lecture.mp4")
window = filter_questions(ctx, start=300, end=600)  # new Context, ctx untouched

data = read_questions("lecture.questions.json")
window = filter_document(data, start=300, end=600)  # new dict, same schema
board = filter_document(data, text="eigenvalue")  # said, or on the board on screen
```

A question is kept when `start <= question.start < end`, in seconds; leave out either bound to
open that side. Only the question list shrinks, so `segment_id` and `visual_ids` still resolve.
An invalid range raises `ValueError`. `text=` keeps a question when every word of it appears, as
a whole word and ignoring case, in the question, its answer, its `review` text, or a board state
in its `visual_ids`; `questxtract filter` does the same from the command line
([docs/cli.md](docs/cli.md)). `topic=` is **experimental**: no stage produces topics yet,
so it warns and matches nothing.

### Reading a quiz.json

`read_quiz` reads a `quiz.json` ([schema](src/questxtract/quiz.schema.json)) and checks the rules
its schema cannot express; `validate_quiz` does the same for a document already in memory. Pass
the source `questions.json` to also check that each item's question exists and that a stem
marked `spoken` is the question as it was said:

```python
from questxtract import read_questions, read_quiz

quiz = read_quiz("lecture.quiz.json", read_questions("lecture.questions.json"))
```

Every problem is listed in one `SchemaError`, one line each, with its path.

### Swapping the ASR engine

The ASR engine is a pluggable backend behind `Config.asr_backend`, not a hardcoded import.
A second engine is one class:

```python
from questxtract import AsrBackend, Config, register_backend


@register_backend("my_engine")
class MyBackend:
    def transcribe(self, source, config, on_progress=None):
        # return raw segments: [{"text": ..., "start": ..., "end": ...}, ...]
        # on_progress(fraction) is optional to call, but required in the signature -
        # accept and ignore it if this engine can't estimate progress.
        ...


ctx = Context(source=Path("lecture.mp4"), config=Config(asr_backend="my_engine"))
```

`faster_whisper` is the default (`Config.asr_backend = "faster_whisper"`), registered the same
way in `asr.py`.


## Development

```bash
git clone git@github.com:mad201802/QuestXtract.git
cd QuestXtract
uv sync --dev

scripts/ci-check.sh        # what CI runs: ruff (auto-fix), ruff format, mypy --strict, pytest (coverage gate 70%)
uv run pytest tests/test_detect.py -q   # the inner loop
```

`ci-check.sh` first downloads, once, the QTI 2.1 and 3.0 XSDs the tests validate packages against
(`scripts/fetch_qti_xsd.py`, into `.cache/qti-xsd/`). Offline, it skips those tests with a
warning; CI always runs them.

How the pipeline works and where each stage lives: [docs/architecture.md](docs/architecture.md).
Why it looks that way: [docs/design_decisions.md](docs/design_decisions.md). Conventions for
people and coding agents: [CLAUDE.md](CLAUDE.md).

CI runs on Linux against Python 3.10 for every pull request; green CI
is required before merging into `main`.

## Releasing

Releases are cut from `main` by pushing a version tag; [release.yml](.github/workflows/release.yml) does the rest.

1. Bump `version` in `pyproject.toml` and rename `## Unreleased` in `CHANGELOG.md` to that version,
   e.g. `## 0.2.0`. Merge this into `main`.
2. Tag the merge commit and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The workflow checks that the tag matches `pyproject.toml` and that `CHANGELOG.md` has a section for it, runs lint, mypy and the tests, builds the sdist and wheel, and creates the GitHub Release with that CHANGELOG section as its notes and the built files attached. A Release already created by hand for the tag gets the same files and notes. Nothing is released if any of those steps fails. Check a tag locally first with `python3 scripts/release_notes.py v0.2.0`.

PyPI publishing is off until the repository variable `PUBLISH_TO_PYPI` is set to `true` (Settings, Secrets and variables, Actions, Variables). It then needs a PyPI API token scoped to `questxtract`, stored as the `PYPI_API_TOKEN` secret of the repository's `pypi` environment, or this workflow registered as a [trusted publisher](https://docs.pypi.org/trusted-publishers/) on PyPI with the secret unset. A failed upload stops the GitHub Release.

## Roadmap

Scope, due dates and progress: [GitHub milestones](https://github.com/mad201802/QuestXtract/milestones).
The customer's requirements behind them: [general_requirements.md](docs/general_requirements.md#general-requirements).

## Contributing

Branch off `main` as `<issue>-<slug>`, keep `scripts/ci-check.sh` green, open a PR that closes one
issue. Public API functions need docstrings, type hints and tests.

An issue enters a sprint only when it has real acceptance criteria, an owner and a milestone. At
most two issues per person are "In progress" on the project board at a time. At the end of each
sprint and before each release, run the `/doc-drift` check (Claude Code) and fix what it finds.

## License

[MIT](LICENSE)

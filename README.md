<div align="center">

# QuestXtract

**Turn lecture recordings into timestamped questions and quizzes.**

A lecture recording in, every question asked in it out, with timestamps, confidence scores and a
JSON export.

[![Download](https://img.shields.io/github/v/release/mad201802/QuestXtract?label=Download&logo=github&style=for-the-badge&color=2ea44f)](https://github.com/mad201802/QuestXtract/releases/latest)

[![CI](https://github.com/mad201802/QuestXtract/actions/workflows/ci.yml/badge.svg)](https://github.com/mad201802/QuestXtract/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#roadmap)

</div>

---

## Why

The questions a lecturer asks the room are the questions students should be able to answer
afterwards. QuestXtract finds them, keeps their timestamps, and turns them into quizzes. It only
extracts questions that were spoken.

## Features

| | |
|---|---|
| Video or audio in | `.mp4` or `.mp3` recordings, audio track extracted automatically (no system ffmpeg) |
| Transcription | Speech-to-text with faster-whisper, or any ASR engine behind a small interface |
| Board and slides | Optional: a vision model finds the slides or board in the frame, whatever the recording style, then reads each view into text, LaTeX and a description, linked to the questions asked while it was on screen |
| Question detection | Fast offline rules, or an LLM you pick (local in Ollama, or hosted) that reads the transcript; a confidence score and the reasons for every question |
| Timestamps | Every question carries its position in the recording |
| Search | Find the questions by words said or written on the board while they were asked, and by time |
| Human in the loop | A `review` block per question for teacher edits, never overwritten by the pipeline |
| JSON export | `questions.json` with a fixed key set and a formal JSON Schema |
| Quizzes | Multiple-choice items from the questions asked, with explanations; every generated text is marked and starts as pending review; the items are drafts, so a teacher checks each one before students see it |
| LMS export | `questxtract qti` writes a `quiz.json` as a QTI 3.0 or 2.1 package for an LMS to import |
| ExerPlaza TeX | `questxtract tex` writes a `quiz.json` as a TeX file to upload to ExerPlaza |
| Straight into ExerPlaza | `questxtract exerplaza` sends a `quiz.json` to an ExerPlaza course session, logged in as you; sending it again adds only the new items |

What is built: [architecture.md 13.3](docs/architecture.md#133-status).

Not included: board animations with [Manim](https://www.manim.community/), which would need
Python 3.11+, LaTeX and a teacher's check of every clip ([#185](https://github.com/mad201802/QuestXtract/issues/185),
[D21](docs/design_decisions.md#d21-manim-animations-are-optional-outside-the-core-delivery)).

## Install

```bash
# with pip, the graphical app included
pip install "questxtract[gui] @ git+https://github.com/mad201802/QuestXtract@v0.4.0"

# or with uv
uv add "questxtract[gui] @ git+https://github.com/mad201802/QuestXtract@v0.4.0"
```

Or take the `.whl` file from the release (Download, above) and install it: `pip install "questxtract-0.4.0-py3-none-any.whl[gui]"`.

`[gui]` adds the graphical app plus what the LLM detector and board reading need; `[llm]` adds
only the latter. Without an extra you get the command line with the offline rules. Drop `@v0.4.0`
for the latest `main`. Releases: [GitHub Releases](https://github.com/mad201802/QuestXtract/releases).

Requires Python 3.10+ on Linux, macOS or Windows.

## Quickstart

```bash
questxtract process lecture.mp4                # video -> lecture.questions.json
questxtract transcribe lecture.mp4             # transcription only -> lecture.transcript.json
questxtract process lecture.transcript.json    # questions from an existing transcript
questxtract quiz lecture.questions.json        # multiple-choice quiz -> lecture.quiz.json
questxtract filter lecture.questions.json --text eigenvalue   # the questions about a word
```

**Review every quiz item before students see it.** A language model writes the quiz, and it can
be wrong: a marked answer that is not correct, a distractor that is also right, an explanation
that does not hold. Check each item and set its review status to `accepted` or `rejected`;
`qti`, `tex` and `exerplaza` leave out rejected items but export pending ones.

An `.mp4` or `.mp3` goes through every stage; a `.json` transcript, from QuestXtract or
[any other ASR engine](docs/cli.md#what-decides-how-much-runs), skips transcription.

Graphical app: install with `[gui]`, run `questxtract gui`, drop a recording or transcript on the
window and press Run ([details](docs/cli.md#graphical-app)).

### Offline fallback

The default run needs no network and no language model: local Whisper transcription, the rules
detector, no vision. Use it when no language model is available (no key, off the NTNU network or
VPN, a model retired). It is offline, not model-free, since Whisper is a local model. It gives up:

- **Precision.** The rules find questions by wording; the LLM detector also judges whether a
  question is about the content.
- **The board.** No slide or board text, so no link from a question to what was on screen.
- **The quiz.** `quiz` writes its items with a language model.

Whisper downloads its model the first time a recording is transcribed. A `.json` transcript needs
no download, and a recording runs offline once the model is on the machine.

For the board without the full vision model, `--vision-reader ocr` reads it with GLM-OCR
(`zai-org/GLM-OCR`, as served on IDUN, NTNU's service, or from a local Ollama). That is not part of
the offline default: it still calls a model server, and a vision model finds the board first
([details](docs/cli.md#reading-the-board)).

### With an LLM and the board

```bash
questxtract process lecture.mp4 --detector llm     # a language model judges the questions
questxtract process lecture.mp4 --vision           # also read the slides or the board

# everything in one run: board, LLM detector and quiz, then export the quiz
questxtract process lecture.mp4 --vision --detector llm --quiz
questxtract qti lecture.quiz.json -o lecture.qti.zip   # QTI 3.0 package for an LMS
questxtract tex lecture.quiz.json -o lecture.tex --content-unit "Sorting"   # TeX for ExerPlaza
questxtract exerplaza lecture.quiz.json --url https://exerplaza.example.org \
    --course-session 42 --content-unit "Sorting"   # straight into an ExerPlaza course session
```

Both run on NTNU's LLM service by default and need `IDUN_API_KEY` and the NTNU network or VPN.

`exerplaza` logs in with `QUESTXTRACT_EXERPLAZA_EMAIL` and `QUESTXTRACT_EXERPLAZA_PASSWORD` from
the environment or `.env`; see [docs/cli.md](docs/cli.md#questxtract-exerplaza-quizjson).

### Switching the model and the server

One setting for the model, one for the server, as a flag or in
[`questxtract.toml`](docs/cli.md#configuration-files):

| | Model | Server |
|---|---|---|
| Text: LLM detector | `--llm-model` / `judge_model` | `--llm-url` / `base_url` |
| Text: quiz | `--quiz-model` / `quiz_model` | `--llm-url` / `base_url` |
| Vision: the board | `--vision-model` / `vision_model` | `vision_base_url`, else `--llm-url` |

A server is `ntnu` (the default), `openai`, `openrouter`, `ollama` (local, no key) or any
OpenAI-compatible URL, each with [its own key](docs/cli.md#api-keys-and-env). The app's API key...
button saves the key; a safer setup: [Setting up the API key](docs/cli.md#setting-up-the-api-key).

```bash
# NTNU's LLM service (the default), IDUN_API_KEY
questxtract process lecture.mp4 --detector llm --vision

# OpenAI, OPENAI_API_KEY
questxtract process lecture.mp4 --detector llm --vision --llm-url openai \
    --llm-model gpt-4o --llm-reasoning server --vision-model gpt-4o

# Ollama on this machine, no key
ollama pull qwen3.5:9b && ollama pull qwen3-vl:2b-instruct    # once
questxtract process lecture.mp4 --detector llm --vision --llm-url ollama \
    --llm-model qwen3.5:9b --llm-reasoning low --vision-model qwen3-vl:2b-instruct
```

For vision on OpenAI or Ollama, also put `vision_reasoning_effort = ""` in `questxtract.toml`:
their vision models take no reasoning setting. More models, OpenRouter and other servers:
[Finding questions with an LLM](docs/cli.md#finding-questions-with-an-llm) and
[Reading the board](docs/cli.md#reading-the-board).

### What leaves your computer

| Setting | What leaves the computer | Where it goes |
|---|---|---|
| Command line default (rules detector, no vision) | No lecture content. Whisper downloads its model on first use | Hugging Face |
| `--detector llm` | Transcript excerpts around each candidate question | The LLM server (`--llm-url`) |
| `--quiz` | Each question, what was said around it and the board text linked to it | The LLM server (`--llm-url`) |
| `--vision` | Images of the slides or board, with what was said while each was shown | The vision server (`vision_base_url`, else `--llm-url`) |
| The app, with no `questxtract.toml` in the folder it starts in | It starts with the LLM detector and board reading on: transcript excerpts, images and speech | NTNU's service |

Where it ends up:

- `ntnu` (the default): NTNU's own servers.
- `openai`: OpenAI, a company outside NTNU.
- `ollama` or another server on this computer: it stays on the computer.

Recordings can contain students' voices and names. Process only recordings you have the rights
to, and check with your university before you send a real lecture to a hosted service. Details:
[design decisions, section 6](docs/design_decisions.md#6-security-at-the-design-level--report-ch-9-compendium-touchpoints-2-5-6).

## Documentation

- [docs/cli.md](docs/cli.md): every command and option, configuration files, LLM and vision setup, troubleshooting
- [docs/architecture.md](docs/architecture.md): how the pipeline works, the
  [`questions.json` format](docs/architecture.md#121-questionsjson) and the
  [Python API](docs/architecture.md#15-python-api)
- [docs/design_decisions.md](docs/design_decisions.md): why it looks that way
- [CONTRIBUTING.md](CONTRIBUTING.md): development setup, workflow and releasing

## Roadmap

Scope, due dates and progress: [GitHub milestones](https://github.com/mad201802/QuestXtract/milestones).
The customer's requirements behind them: [general_requirements.md](docs/general_requirements.md#general-requirements).

## License

[MIT](LICENSE)

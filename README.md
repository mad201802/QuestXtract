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
| 📝 **ExerPlaza TeX** | `questxtract tex` writes a `quiz.json` as a TeX file to upload to ExerPlaza |

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
Extras: `[llm]` for the LLM detector, reading the board and quizzes, `[gui]` for the graphical
app, e.g. `pip install "questxtract[llm] @ git+https://github.com/mad201802/QuestXtract"`.

## Quickstart

```bash
questxtract process lecture.mp4                # video -> lecture.questions.json
questxtract transcribe lecture.mp4             # transcription only -> lecture.transcript.json
questxtract process lecture.transcript.json    # questions from an existing transcript
questxtract quiz lecture.questions.json        # multiple-choice quiz -> lecture.quiz.json
questxtract filter lecture.questions.json --text eigenvalue   # the questions about a word
```

An `.mp4` or `.mp3` goes through every stage; a `.json` transcript, from QuestXtract or
[any other ASR engine](docs/cli.md#what-decides-how-much-runs), skips transcription.

**Graphical app.** `pip install "questxtract[gui]"`, then `questxtract gui`: drop a recording or
transcript on the window and press Run ([details](docs/cli.md#graphical-app)).

### With an LLM and the board

```bash
questxtract process lecture.mp4 --detector llm     # a language model judges the questions
questxtract process lecture.mp4 --vision           # also read the slides or the board

# everything in one run: board, LLM detector and quiz, then export the quiz
questxtract process lecture.mp4 --vision --detector llm --quiz
questxtract qti lecture.quiz.json -o lecture.qti.zip   # QTI 3.0 package for an LMS
questxtract tex lecture.quiz.json -o lecture.tex --content-unit "Sorting"   # TeX for ExerPlaza
```

By default both run on NTNU's LLM service: they need `IDUN_API_KEY` and the NTNU network or VPN.
OpenAI or a local Ollama work through `--llm-url`. Store the key safely:
[Setting up the API key](docs/cli.md#setting-up-the-api-key).

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

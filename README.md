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
| 🎥 **Video in** | Common video containers (`.mp4`, …), audio track extracted automatically |
| 🗣️ **Transcription** | Speech-to-text via an ASR model |
| ❓ **Question detection** | Question sentences found in the transcript, with a confidence score |
| ⏱️ **Timestamps** | Every question carries its position in the recording |
| 🖼️ **Blackboard content** | Drawings and animations from the video enrich the audio context |
| ✏️ **Human in the loop** | Detected questions stay editable before export |
| 📦 **JSON export** | Structured, machine-readable output |

Shipped in `0.1`: question detection, timestamps, confidence, JSON export, CLI.
The video, ASR and blackboard stages are on the [roadmap](#roadmap).

## Install

```bash
# with uv (recommended)
uv add git+https://github.com/mad201802/QuestXtract

# or with pip
pip install git+https://github.com/mad201802/QuestXtract
```

Requires Python 3.10+ on Linux, macOS or Windows.

## Quickstart

### CLI

```bash
questxtract transcript.json -o questions.json
questxtract transcript.json --threshold 0.8      # stricter detection, prints to stdout
```

`transcript.json` is either a list of segments or an object with a `segments` key, which is what
most ASR engines emit:

```json
{
  "segments": [
    { "text": "Today we cover sorting. What is a stable sort?", "start": 0.0, "end": 6.0 },
    { "text": "Does anyone know why that matters", "start": 9.0, "end": 12.0 }
  ]
}
```

Output:

```json
{
  "questions": [
    {
      "text": "What is a stable sort?",
      "start": 0.0,
      "end": 6.0,
      "confidence": 1.0,
      "reasons": ["mark", "wh"]
    },
    {
      "text": "Does anyone know why that matters",
      "start": 9.0,
      "end": 12.0,
      "confidence": 0.5,
      "reasons": ["aux"]
    }
  ]
}
```

`reasons` says which heuristics fired: `mark` (ends in `?`), `wh` (starts with a wh-word),
`aux` (starts with an auxiliary verb). Useful when reviewing borderline detections — ASR output
often drops the question mark.

### Python API

```python
from questxtract import TranscriptSegment, detect_questions, read_segments, write_questions

segments = read_segments("transcript.json")
# ... or build them yourself
segments = [TranscriptSegment(text="What is a stable sort?", start=0.0, end=6.0)]

questions = detect_questions(segments, threshold=0.5)
for q in questions:
    print(f"[{q.start:7.2f}s] ({q.confidence:.0%}) {q.text}")

write_questions(questions, "questions.json")
```

## Development

```bash
git clone git@github.com:mad201802/QuestXtract.git
cd QuestXtract
uv sync --dev

uv run pytest              # tests + coverage (gate: 70%)
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy                # type check
```

The package uses a `src/` layout and is built with [uv](https://github.com/astral-sh/uv):

```
src/questxtract/
├── __init__.py   public API
├── models.py     TranscriptSegment, Question, JSON read/write
├── detect.py     question detection
└── cli.py        questxtract command
tests/            pytest suite
```

CI runs lint, format check and the test suite on Linux, macOS and Windows against Python 3.10 and
3.13 for every pull request. Green CI is required before merging into `main`.

## Roadmap

- [x] **M0** — installable package, question detection over transcripts, JSON export, CI
- [ ] **M1** — refactor the audio-only codebase in: video → audio → ASR → questions, end to end
- [ ] **M2** — static image understanding, enriching the audio information
- [ ] **M3** — flow-based image understanding across the whole recording
- [ ] **M4** — [manim](https://github.com/manimCommunity/manim) animations generated from the image flow

Detailed scope in [general_requirements.md](general_requirements.md).

## Contributing

Branch off `main`, keep `uv run pytest` and `uv run ruff check .` green, open a PR. Public API
functions need docstrings, type hints and tests.

## License

[MIT](LICENSE)

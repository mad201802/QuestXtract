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
| ✏️ **Human in the loop** | Exported JSON reads back unchanged, so a separate review UI can edit and re-export |
| 📦 **JSON export** | Structured, machine-readable output |

Shipped in `0.1`: the ten-stage pipeline contract, MP4 validation and audio extraction, ASR
through a pluggable backend interface (faster-whisper by default), sentence and segment
rebuilding, question detection, timestamps, confidence, JSON export, CLI. The blackboard/vision
stage is still a stub behind that contract — see the [roadmap](#roadmap) and
[architecture.md 13.3](docs/architecture.md#133-status).

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
questxtract process transcript.json -o questions.json
questxtract process transcript.json --threshold 0.8   # stricter, prints to stdout
```

The ASR stage is implemented as a pluggable backend (see below), but the CLI's `process`
command still only accepts a pre-made transcript; wiring it to accept a raw recording
end-to-end is issue #17. Until then, `process` runs the pipeline from sentence splitting
onward.

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

Output ([docs/architecture.md](docs/architecture.md#121-questionsjson) 12.1) — a fixed key set, `null`
included, so a consumer never has to defensively check for a missing field:

```json
{
  "schema_version": "1.0",
  "source": { "path": "transcript.json", "duration_seconds": 12.0, "hash": null },
  "config": { "model_size": "turbo", "language": "en",
              "judge_model": "gpt-5.6-terra", "vision_model": "gpt-5.6-luna" },
  "stats": { "sentences": 3, "segments": 1, "candidates": 2, "answered": 0, "emitted": 1 },
  "questions": [
    {
      "id": "q_ca0d72",
      "text": "What is a stable sort?",
      "original_text": null,
      "start": 2.67,
      "end": 6.0,
      "confidence": 0.55,
      "confidence_label": "medium",
      "question_type": "what",
      "answer": null,
      "segment_id": 0,
      "sentence_id": "s_ca0d72",
      "visual": null,
      "verdict": "unjudged",
      "reasons": ["mark", "wh", "no_answer_found"]
    }
  ]
}
```

`id` and `sentence_id` identify a question across re-runs and edits; `sentence_id` is a content
hash, so it survives a re-transcription. `reasons` says which rules fired: `mark` (ends in `?`),
`wh` (starts with a wh-word), `aux` (starts with an auxiliary verb). Useful when reviewing
borderline detections — ASR output often drops the question mark.

`--threshold` gates the final **confidence**, not the raw detection score. An unanswered question
carries `0.55 x question_score` (architecture.md 9.2), so it tops out at 0.55 until answer search
lands.

### Python API

```python
from pathlib import Path

from questxtract import STAGES, Config, Context, Transcript, run, write_questions

ctx = Context(
    source=Path("lecture.mp4"),
    config=Config(min_confidence=0.4),
    transcript=Transcript.from_dict({"segments": [...]}),  # or let stage 2 produce it
)
ctx = run(ctx, STAGES[2:])  # skip ingest + transcribe: the transcript is already in hand

for pair in ctx.pairs:
    print(f"[{pair.start:7.2f}s] ({pair.confidence:.0%}) {pair.question.text}")

write_questions(ctx, "questions.json")
```

The pipeline is a list of `Context -> Context` functions. Passing your own `stages=` list is the
whole plugin system — swap the detector, drop a stage, add an enrichment step:

```python
run(ctx, [*STAGES[2:6], my_detector, *STAGES[7:]])
```

### Swapping the ASR engine

The ASR engine is a pluggable backend behind `Config.asr_backend`, not a hardcoded import.
A second engine is one class:

```python
from questxtract import AsrBackend, Config, register_backend


@register_backend("my_engine")
class MyBackend:
    def transcribe(self, source, config):
        # return raw segments: [{"text": ..., "start": ..., "end": ...}, ...]
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

uv run pytest              # tests + coverage (gate: 70%)
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy                # type check
```

The package uses a `src/` layout and is built with [uv](https://github.com/astral-sh/uv). One
module per pipeline stage, each exposing a single `@stage`-decorated `Context -> Context`
function. See [docs/architecture.md](docs/architecture.md) for what each one does and why, and
[docs/architecture_review.md](docs/architecture_review.md) for the decisions behind it:

```
src/questxtract/
├── __init__.py   public API: process, run, STAGES, the schemas
├── models.py     Word, Transcript, Sentence, Segment, QAPair, Context, @stage
├── config.py     one flat frozen Config, every default
├── pipeline.py   STAGES, run(), process(), the cache, progress
├── media.py      1 ingest        ├── detect.py    6 detect
├── transcribe.py 2 transcribe    ├── answer.py    7 answer
├── sentences.py  3 sentences     ├── score.py     8 score
├── segments.py   4 segments      ├── llm.py       9 judge
├── vision.py     5 vision        ├── export.py   10 export
├── evaluate.py   labelled-set metrics
└── cli.py        process / eval / quiz
tests/            flat, one module per source module
```

Which stages are real today and which are stubs: [architecture.md 13.3](docs/architecture.md#133-status).
`grep -rn 'ponytail:' src/` lists every shortcut the code still owes that document.

CI runs lint, format check and the test suite on Linux, macOS and Windows against Python 3.10 and
3.13 for every pull request. Green CI is required before merging into `main`.

## Roadmap

- [x] **M0** — installable package, question detection over transcripts, JSON export, CI
- [ ] **M1** — refactor the audio-only codebase in: video → audio → ASR → questions, end to end
- [ ] **M2** — static image understanding, enriching the audio information
- [ ] **M3** — flow-based image understanding across the whole recording
- [ ] **M4** — [manim](https://github.com/manimCommunity/manim) animations generated from the image flow

Detailed scope in [general_requirements.md](docs/general_requirements.md#general-requirements).

## Contributing

Branch off `main`, keep `uv run pytest` and `uv run ruff check .` green, open a PR. Public API
functions need docstrings, type hints and tests.

## License

[MIT](LICENSE)

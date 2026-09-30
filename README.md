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
| 🎥 **Video in** | `.mp4` recordings, audio track extracted automatically (no system ffmpeg) |
| 🗣️ **Transcription** | Speech-to-text with faster-whisper, or any ASR engine behind a small interface |
| ❓ **Question detection** | Fast offline rules, or an LLM you pick (local in Ollama, or hosted) that reads the transcript; a confidence score and the reasons for every question |
| ⏱️ **Timestamps** | Every question carries its position in the recording |
| ✏️ **Human in the loop** | A `review` block per question for teacher edits, never overwritten by the pipeline |
| 📦 **JSON export** | `questions.json` with a fixed key set and a formal JSON Schema |

Planned: reading the board and slides (#59) and quiz generation (#79). What is built today:
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
```

The input decides how much runs: an `.mp4` goes through every stage, while a `.json` transcript
skips transcription. A progress bar renders on stderr, and transcripts are cached, so running
`process` again with another threshold takes seconds. Every option, the cache, error messages and
recipes: [docs/cli.md](docs/cli.md).

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
lets a language model read the transcript instead: it picks the questions actually asked, finds
the sentences that answer them, and rewrites a messy question into a clean one, keeping the spoken
version beside it. It never adds a question that was not spoken. It needs the `llm` extra
(`pip install "questxtract[llm] @ git+https://github.com/mad201802/QuestXtract"`) and a model:

```bash
# hosted: OpenAI, key from the environment
OPENAI_API_KEY=sk-... questxtract process lecture.mp4 --detector llm

# local: any OpenAI-compatible server, here Ollama with a small Qwen
questxtract process lecture.mp4 --detector llm \
    --llm-url http://localhost:11434/v1 --llm-model qwen3.5:9b
```

Answers are cached, so another `-t` costs no calls. Setting up Ollama, choosing a model, and what
happens when the endpoint fails: [docs/cli.md](docs/cli.md#finding-questions-with-an-llm).

To measure the detector against the labelled reference recordings in `tests/data/`
([how they were made](tests/data/README.md)):

```bash
questxtract eval tests/data/          # table, sweep, missed questions and false positives
questxtract eval tests/data/ --json   # the same report, machine-readable
questxtract eval tests/data/ --detector llm --llm-url http://localhost:11434/v1 --llm-model qwen3.5:9b
```

### Output

`questions.json` lists every question with its text, `start`/`end` in seconds, `confidence`, the
`answer` where one was found, the `reasons` (rules) that fired, and a `review` block for teacher
edits. Segments and board states are listed once and referenced by id. Every key is always
present, `null` included. Example and field reference:
[architecture.md 12.1](docs/architecture.md#121-questionsjson); formal schema:
[questions.schema.json](src/questxtract/questions.schema.json).

`--threshold` gates the final **confidence**: an unanswered question tops out at 0.55, one the
next sentence answers carries its full question score.

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
run(ctx, [*STAGES[2:6], my_detector, *STAGES[7:]])
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
```

A question is kept when `start <= question.start < end`, in seconds; leave out either bound to
open that side. Only the question list shrinks, so `segment_id` and `visual_ids` still resolve.
An invalid range raises `ValueError`. `topic=` is **experimental**: no stage produces topics yet,
so it warns and matches nothing.

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

How the pipeline works and where each stage lives: [docs/architecture.md](docs/architecture.md).
Why it looks that way: [docs/design_decisions.md](docs/design_decisions.md). Conventions for
people and coding agents: [CLAUDE.md](CLAUDE.md).

CI runs on Linux, macOS and Windows against Python 3.10 and 3.13 for every pull request; green CI
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

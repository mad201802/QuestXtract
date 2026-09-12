# Package architecture

Defines the module boundaries for `questxtract` and tracks which parts exist versus which
issue delivers them, so the team can see at a glance what each M1 issue (#6–#18) adds and
where the later milestones (M2–M4) and the customer's remaining deliverables plug in.
The reasoning behind these choices — alternatives, trade-offs, review outcome — is in
[architecture_review.md](architecture_review.md).

## Goals

- One package per pipeline stage. The pipeline is a fork, not a line:

  ```
  video/ ──► audio/ ──► detection/ ──► extraction/ ──► builder/ ──► output/
     │                       ▲
     └──► visual/ ── enrich ─┘
  ```

  Recording in; audio track (and later frames) out. Speech becomes transcript segments,
  detection finds the questions, visual understanding (M2/M3) *enriches* those questions,
  solution extraction and question building turn them into quiz items, export writes JSON.
- Every module has a single responsibility and a small public surface.
- The package ends at structured quiz material on disk (a `LectureResult` JSON file).
  Reviewing, editing and uploading that material are separate applications — see
  [Non-goals](#non-goals-explicit-boundaries).

## Package layout

```
src/questxtract/
├── __init__.py            # public API re-exports
├── py.typed
├── video/                 # recording input: container decoding (issue #7; frames in M2)
│   └── __init__.py
├── audio/                 # speech-to-text
│   ├── __init__.py
│   └── transcripts.py     # read/write transcript segments as JSON
├── detection/
│   ├── __init__.py
│   └── questions.py       # heuristic question scoring
├── models/
│   ├── __init__.py
│   └── schemas.py         # Word, TranscriptSegment, Question, LectureResult, SchemaError
├── output/
│   ├── __init__.py
│   └── serialize.py       # LectureResult JSON export/import
└── cli.py                 # questxtract command

tests/                     # mirrors src/; conftest.py holds shared fixtures
├── audio/
├── detection/
├── models/
└── output/
```

## Module responsibilities

Status describes the *module*: "M0 placeholder" code runs end to end on transcripts but
does not close its issue — the Issue column says who finishes it.

| Module | Responsibility | Status | Issue |
|---|---|---|---|
| `models/schemas.py` | `Word`, `TranscriptSegment` (ASR output), `Question`, `LectureResult` (export unit), `SchemaError` | M0 placeholder | #9 |
| `detection/questions.py` | Heuristic question scoring over transcript segments; pluggable for a future trained classifier | M0 placeholder | #10 |
| `output/serialize.py` | Export a `LectureResult` to JSON and read it back, edited or not | M0 placeholder | #13 |
| `cli.py` | Argument parsing and process wiring only; takes a transcript until #7/#8 land | M0 placeholder | #17 |
| `audio/transcripts.py` | Read/write transcript JSON so ASR output is persisted, not recomputed | done | #8 |
| `video/audio.py` | `.mp4` in → audio track out; input validation, temp files | planned | #7 |
| `audio/transcription.py` | ASR interface + implementation(s), emitting `TranscriptSegment`s with `words` | planned | #8 |
| `progress.py` | Callback protocol (`report(stage, fraction)`); every stage entry point accepts an optional reporter | planned | #14 |
| `pipeline.py` | Orchestrates video → audio → transcript → detect → enrich → export, threading progress through | planned | #6 |
| `config.py` | Typed pipeline settings (threshold, ASR backend, …) | planned | #17 |
| `LectureResult.filter()` | Filter questions by time range / topic — a query on the result, not an output step | planned | #16 |
| `video/frames.py` | Key-frame extraction from the recording | planned (M2) | — |
| `visual/` | Slide/board OCR (M2), board-writing flow (M3); fills `visual_context` on existing questions | planned (M2/M3) | — |
| `extraction/` | The teacher's solution/explanation span following each question → `answer` | planned | — |
| `builder/` | LLM clean-up of question–answer pairs into MCQ / numeric items (LaTeX where needed) | planned | — |

Planned modules aren't scaffolded; the owning issue creates them against these boundaries.
`extraction/`, `builder/`, `visual/` and `video/frames.py` have no issue yet: they are the
customer's Milestone 2/3 deliverables ("extract the question and the teacher's solution",
"a question builder that produces multiple choice or numeric questions") and the README's
M2/M3, recorded here so the schema and pipeline leave room for them. The constraint from
`general_requirements.md` still holds: the builder formats questions the lecturer actually
asked, it never invents new ones.

## Public API

`import questxtract` exposes:

```python
from questxtract import (
    LectureResult,
    Question,
    SchemaError,
    TranscriptSegment,
    Word,
    detect_questions,
    score_sentence,
    read_segments,
    segments_to_json,
    write_segments,
    read_result,
    result_to_json,
    write_result,
)
```

**Breaking change against the M0 placeholder** (unreleased, alpha): `Question.text` is now
`Question.question` and the JSON key changed with it; `write_questions`/`questions_to_json`
became `write_result`/`result_to_json` over a `LectureResult`. See [CHANGELOG.md](../CHANGELOG.md).

Naming convention: input types use `text` for raw transcript strings (`Word.text`,
`TranscriptSegment.text`); output types name the field after what it holds
(`Question.question`, later `answer`). All schema dataclasses are frozen and keyword-only.

## Schemas

Exported JSON (`write_result`), one file per recording:

```json
{
  "lecture": "lecture.mp4",
  "questions": [
    {
      "id": "b257bd671c04",
      "question": "What is a stable sort?",
      "start": 0.0,
      "end": 6.0,
      "confidence": 1.0,
      "source": "audio",
      "reasons": ["mark", "wh"]
    }
  ]
}
```

- `lecture` + `id` are the identity. `id` is derived from `start` and `question`, so it is
  stable across re-runs of the same recording and survives edits made in a review UI
  (`read_result` accepts the file back and keeps ids as stored). "Update the quiz for this
  video" on an upload target should key on them.
- `source` is detector provenance: `"audio"` for transcript-based detection. M2 does **not**
  add a second detector — visual understanding enriches existing questions by filling
  `visual_context`, so no merge/dedup step is needed.
- Reserved, not yet present (issue #9): `visual_context` (M2), `topic` (#16), `answer`
  (solution extraction).
- Transcript JSON (`write_segments`): `{"segments": [{"text", "start", "end", "words"}]}`.
  `words` is optional and carries per-word timestamps when the ASR engine provides them;
  detection will use them for sentence-accurate `start`/`end` (today a question inherits its
  segment's span).

## Non-goals (explicit boundaries)

- **Teacher review/edit UI** ([issue #15](https://github.com/mad201802/QuestXtract/issues/15))
  is a separate application. The package's part is that its JSON round-trips: `read_result`
  loads an edited file back into a `LectureResult`.
- **exerplaza.com upload** (from the customer's project brief) is likewise a separate consumer
  of the exported JSON. Nothing in `questxtract` talks to exerplaza directly.
- **Evaluation / labelled dataset work** ([#11](https://github.com/mad201802/QuestXtract/issues/11),
  [#12](https://github.com/mad201802/QuestXtract/issues/12)) lives under `tests/` as fixtures
  and scripts, not as an importable subpackage.
- **YouTube URLs** are out of scope for M1: the CLI takes a local `.mp4` (issue #7). If URL
  download is ever added it belongs in `video/`.

## Testing layout

Tests mirror `src/` one directory per package (`tests/audio/`, `tests/detection/`,
`tests/models/`, `tests/output/`); the CLI is exercised from `tests/output/` as the export
path. Shared fixtures live in `tests/conftest.py`. pytest runs with
`--import-mode=importlib`, so same-named test files in different directories don't collide.
`tests/integration/` arrives with `pipeline.py` (issue #6).

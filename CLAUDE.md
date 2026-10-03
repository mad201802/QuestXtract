# QuestXtract

Python package: a lecture recording (`.mp4`) or an ASR transcript (`.json`) in, `questions.json`
out, with every question actually asked in the lecture, its timestamp and a confidence.
Five-person course project; milestones, priorities and sprints live on GitHub.

## Commands

- Setup once: `uv sync --dev`. Run everything through `uv run`.
- Before every commit: `scripts/ci-check.sh` (ruff with auto-fix, ruff format, mypy strict,
  pytest with the 70% coverage gate). CI runs the same on Linux, macOS and Windows, Python 3.10
  and 3.13.
- Inner loop: `uv run pytest tests/test_<module>.py -q`.
- Detection quality: `uv run questxtract eval tests/data/`. `tests/test_evaluate.py` fails if
  pooled recall drops below 0.50 (NFR2).
- Never run ASR or call a network API in tests or the inner loop. Use the committed
  transcripts: `uv run questxtract process tests/data/<slug>.transcript.json -o -`.

## Layout

- `src/questxtract/`: one module per pipeline stage, each exposing one `@stage` function
  `Context -> Context`. `pipeline.STAGES` is the run order. `models.py` holds the data types,
  `config.py` every default, `export.py` the `questions.json` writer and reader, `cli.py` the CLI.
- `tests/test_<module>.py`: one flat test file per module. `tests/data/`: the labelled
  recordings (data, not code).
- `scripts/`: dev tooling (labelling, release notes), not shipped.
- `docs/`: `architecture.md` (how the pipeline works), `cli.md`, `annotation_guidelines.md`,
  `general_requirements.md` (the customer's FR/NFR ids), `design_decisions.md` (decision log
  for the course report).

## Pipeline contract

1. Stages never import each other. They talk only through `Context`.
2. Stages read settings only from `ctx.config`. Every default lives in `config.py`, one flat
   frozen dataclass.
3. An optional stage checks its own flag and returns `ctx` unchanged when off.
4. A cached stage fills exactly one `Context` field and lists every `Config` field it reads in
   `uses=(...)`. A missing name means stale cache hits.
5. Data types are `@dataclass(frozen=True, kw_only=True)`.
6. `reasons` codes are stable strings, only ever appended. Never drop one.
7. `questions.json` has a fixed key set, `null` included. Adding keys is a minor
   `schema_version` bump, moving or removing one is major; update `questions.schema.json` and
   CHANGELOG.md in the same PR. The pipeline never writes into a question's `review` block.
8. The pipeline never invents a question (requirement 4.5). Only a separate, generative command
   may write text that was not spoken.

## Unbuilt features

- `grep -rn 'ponytail:' src` lists every stub and shortcut. Format:
  `# ponytail: #<issue>, <what is missing>` (issue first, so one grep line shows it). No issue,
  no stub: open the issue first. A Config field nothing reads yet gets `# not read yet, #<issue>`.
- Build only what the current issue asks for. No Config fields, schema keys, stages or flags
  "for later".
- When a stub gets implemented: delete its marker, update the Status table in
  `docs/architecture.md`, and README.md / docs/cli.md if users see the change.

## Docs

- `architecture.md` describes the code as it is. A planned part gets "(planned, #N)" in its
  heading and stays short: the decision and its reason; build detail belongs in the issue.
- No history in docs ("since #14", "raised from 220", "this is a correction"). That goes in
  CHANGELOG.md, `design_decisions.md` and commit messages.
- `architecture.md` section numbers are stable: code and issues cite them ("architecture.md
  9.2"). Never renumber; shorten or mark a section instead. Every cited number must resolve
  (`/doc-drift` checks it). No lettered sub-references like "7.1(a)".
- Metrics come from `questxtract eval`. Paste them in the PR; never copy them into issues or
  docs. architecture.md keeps one current results table.
- Update an existing doc instead of adding a new file.

## Code

- Python >= 3.10: nothing newer (no `typing.Self`, `except*`). For TOML, use
  `tomllib` only behind a Python >= 3.11 version check, with `tomli` on Python 3.10.
- Windows is supported: `pathlib` for paths, `encoding="utf-8"` on every open.
- Public API (`__all__` in `__init__.py`) needs docstrings, type hints and tests.

## Git and GitHub

- One issue per branch and PR: branch `<issue>-<slug>` off `main`, small PRs.
- PR title = squash commit title, Conventional Commits: `feat: <summary> (#N)`, `fix: ...`,
  `docs: ...`. The body follows `.github/PULL_REQUEST_TEMPLATE.md` and says `Closes #N`.
- User-visible change: an entry in CHANGELOG.md under `## Unreleased`.
- GitHub milestones are the only milestone definition; priority and status live on the
  project board.
- Ask before pushing, opening a PR, or changing issues, labels or milestones.

## Skills

`/work-issue <N or idea>` (idea or issue to PR), `/eval-detection` (main vs branch metrics),
`/label-recording <video> <slug>` (extend `tests/data/`), `/doc-drift` (docs vs code check).

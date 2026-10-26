# Contributing

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

CI runs on Linux against Python 3.10 for every pull request, and installs the built wheel into a
fresh environment on Python 3.10 and 3.13 to run it on a committed transcript; green CI
is required before merging into `main`.

## Workflow

Branch off `main` as `<issue>-<slug>`, keep `scripts/ci-check.sh` green, open a PR that closes one
issue. Public API functions need docstrings, type hints and tests.

An issue enters a sprint only when it has real acceptance criteria, an owner and a milestone. At
most two issues per person are "In progress" on the project board at a time. At the end of each
sprint and before each release, run the `/doc-drift` check (Claude Code) and fix what it finds.

## API keys

We use NTNU's LLM service with our NTNU key, and the customer's pay-as-you-go OpenAI key only for
the OpenAI fallback, with `gpt-4o`. Never commit the key, paste it into a chat, issue or pull
request, or write it into any file; once it is saved
([docs/cli.md](docs/cli.md#setting-up-the-api-key)), delete the message you received it in. Ask
the customer before using another model with it. If a key leaks, tell its owner at once so it
can be revoked.

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

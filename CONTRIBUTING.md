# Contributing

## Development

```bash
git clone git@github.com:mad201802/QuestXtract.git
cd QuestXtract
uv sync --dev

scripts/ci-check.sh        # what CI runs: ruff (auto-fix), ruff format, mypy --strict, pytest (coverage gate 70%)
uv run pytest tests/test_detect.py -q   # the inner loop
```

`ci-check.sh` downloads the QTI 2.1 and 3.0 XSDs once (`scripts/fetch_qti_xsd.py`, into
`.cache/qti-xsd/`). Offline, it skips the XSD tests with a warning; CI always runs them.

Pipeline and stages: [docs/architecture.md](docs/architecture.md). Decisions:
[docs/design_decisions.md](docs/design_decisions.md).

CI runs on Linux with Python 3.10 for every pull request, and installs the built wheel into a fresh
environment on Python 3.10 and 3.13 to run it on a committed transcript. Merging into `main` needs
green CI.

## Workflow

Branch off `main` as `<issue>-<slug>`, keep `scripts/ci-check.sh` green, open a PR that closes one
issue. Public API functions need docstrings, type hints and tests.

An issue enters a sprint only with acceptance criteria, an owner and a milestone. At most two
issues per person are "In progress" on the project board. At the end of each sprint and before
each release, review the docs against the code and fix what is out of date.

## API keys

We use NTNU's LLM service with our NTNU key, and the customer's pay-as-you-go OpenAI key only for
the OpenAI fallback with `gpt-4o`. Never commit a key or paste it into a chat, issue, pull request
or file; once it is saved ([docs/cli.md](docs/cli.md#setting-up-the-api-key)), delete the message
you received it in. Ask the customer before using another model with it. If a key leaks, tell its
owner at once so it can be revoked.

## Releasing

Releases are cut from `main` by pushing a version tag; [release.yml](.github/workflows/release.yml)
does the rest.

1. Bump `version` in `pyproject.toml` and rename `## Unreleased` in `CHANGELOG.md` to that
   version, e.g. `## 0.2.0`. Merge into `main`.
2. Tag the merge commit and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The workflow checks that the tag matches `pyproject.toml` and that `CHANGELOG.md` has its section,
runs the CI workflow (lint, mypy, tests, the installed-wheel check on 3.10 and 3.13, the dependency
audit), builds the sdist and wheel, and creates the GitHub Release with that section as notes and
the files attached; a Release created by hand for the tag gets the same. Any failed step stops the
release. Check a tag locally with `python3 scripts/release_notes.py v0.2.0`.

PyPI publishing is off until the repository variable `PUBLISH_TO_PYPI` is `true` (Settings,
Secrets and variables, Actions, Variables). It then needs a PyPI API token scoped to `questxtract`
as the `PYPI_API_TOKEN` secret of the `pypi` environment, or this workflow registered as a
[trusted publisher](https://docs.pypi.org/trusted-publishers/) with the secret unset. A failed
upload stops the GitHub Release.

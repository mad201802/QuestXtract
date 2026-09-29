# What

<!-- What does this change, and why? Link the issue: Closes #123
     The PR title becomes the squash commit: Conventional Commits, e.g. "feat: find answers with the LLM pass (#46)".
     Keep it to one issue; split a PR that grows past a few hundred lines. -->

# Checklist

- [ ] `uv run pytest` green (coverage gate 70%)
- [ ] `uv run ruff check .` and `uv run ruff format --check .` green
- [ ] `uv run mypy` green
- [ ] Public API changes have docstrings, type hints and tests
- [ ] README / docs updated if behaviour or CLI flags changed
- [ ] CHANGELOG.md `## Unreleased` entry if users can see the change
- [ ] Touched detection, answers or scoring: the `/eval-detection` table (main vs this branch) is pasted below

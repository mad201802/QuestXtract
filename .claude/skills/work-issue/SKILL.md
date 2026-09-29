---
name: work-issue
description: Work a QuestXtract GitHub issue end to end - read the issue and its acceptance criteria, branch, implement, test, update CHANGELOG and docs, prepare the PR. Use when asked to implement, fix or work on an issue number ("work on #70", "/work-issue 70").
argument-hint: <issue-number>
---

# Work an issue

Issue: $ARGUMENTS

1. **Read it.** `gh issue view $ARGUMENTS --comments`. If the acceptance criteria are missing or
   still the template placeholders ("Given ..., when ..., then ..."), stop and ask. Don't invent scope.
2. **Branch.** `git switch main && git pull --ff-only && git switch -c <N>-<short-slug>`
   (e.g. `70-ttk4225-check-ins`).
3. **Read before writing.** The module the issue touches, its `tests/test_<module>.py`, and the
   matching part of `docs/architecture.md` (find it by heading with grep; section numbers quoted
   in issues go stale).
4. **Implement the smallest change that meets the AC**, following the pipeline contract in
   CLAUDE.md. Nothing for later issues: no extra Config fields, schema keys or stages.
5. **Test.** One test per AC line in `tests/test_<module>.py`. Offline only: committed
   transcripts in `tests/data/`, never ASR or network.
6. **Touched detect, answer, score, sentences or segments?** Run `/eval-detection` and keep its
   table for the PR.
7. **Check.** `scripts/ci-check.sh` must be green (it auto-fixes ruff first).
8. **Update only what the change touches:**
   - `CHANGELOG.md` under `## Unreleased`, in `### Added`, `### Changed (breaking)` or
     `### Fixed`, ending with `(#N)`, when users can see the change.
   - A stub landed: delete its `ponytail:` marker, update the Status table in
     `docs/architecture.md`, drop "(planned, #N)" from its heading.
   - CLI flags changed: the option tables in README.md and docs/cli.md.
   - `questions.json` shape changed: `questions.schema.json` and `schema_version` (minor = added
     keys, major = moved or removed).
9. **Commit** with a Conventional Commits message (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
10. **PR.** Ask before pushing. Title = the squash commit title, `<type>: <summary> (#N)`. Body
    follows `.github/PULL_REQUEST_TEMPLATE.md`, starts with `Closes #N`, says how each AC line is
    met, and includes the eval table from step 6 if there is one.

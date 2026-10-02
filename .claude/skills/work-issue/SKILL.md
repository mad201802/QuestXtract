---
name: work-issue
description: Take a QuestXtract feature from idea or issue to PR - interview the developer until nothing is left to guess, create or refine the GitHub issue, branch, implement, test, update CHANGELOG and docs, prepare the PR. Use when asked for a new feature or fix ("I want X", "/work-issue add a quiz export") or to work on an issue number ("work on #70", "/work-issue 70").
argument-hint: <issue-number | feature idea>
---

# Work an issue

Input: $ARGUMENTS

1. **Read it.** A number: `gh issue view <N> --comments`, plus the issues and PRs it links. Free
   text: a new idea with no issue yet; first search for a duplicate with
   `gh issue list --state all --search "<keywords>"` and switch to that issue if one fits.
2. **Research before asking.** Read the module the work touches, its `tests/test_<module>.py`,
   `config.py`, and the matching part of `docs/architecture.md` (find it by heading with grep;
   section numbers quoted in issues go stale). Answer from code whatever code can answer.
3. **Find gaps, then interview.** A gap is anything you would otherwise guess. A new idea always
   gets an interview; an existing issue only if it has gaps. Check:
   - Behaviour: AC missing or still the template ("Given ..., when ..., then ..."); exact input
     and output per AC line; error and edge cases; an example where none is given.
   - Interfaces: CLI flag names and help text, Config field names and defaults,
     `questions.json` keys (`schema_version` bump), new `reasons` codes.
   - Placement: which stage or module, new stage vs existing, optional flag, cached `uses=(...)`.
   - Tests and metrics: which `tests/data/` recording proves it, acceptable recall/precision
     tradeoff (NFR2 floor 0.50).

   No gaps: say so in one line and go on. Gaps: ask with AskUserQuestion, up to 4 questions per
   round, with concrete options from what you read (recommended first, citing `file:line`).
   Repeat rounds until you can name every file, flag, field, default and test without guessing.
   Never fill a gap with your own default silently. Don't invent scope.

4. **Create or update the issue.** Draft the body in `.github/ISSUE_TEMPLATE/task.yml` shape
   (Description, Scope in/out, Acceptance criteria one line each, Definition of done copied from
   the template, Dependencies and risks), a plain title, one `type:` label plus `comp: code`, and
   a milestone (`gh api repos/{owner}/{repo}/milestones --jq '.[].title'`). Show it all and ask.
   On OK: new idea `gh issue create --title ... --body-file <file> --label ... --milestone ...`,
   existing issue `gh issue edit <N> --body-file <file>`. Without OK, keep the draft local; a new
   idea then stops here (no issue, no branch). Remind the developer to set Priority and Size on
   the project board. From here on N is the issue number.
5. **Branch.** `git switch main && git pull --ff-only && git switch -c <N>-<short-slug>`
   (e.g. `70-ttk4225-check-ins`).
6. **Implement the smallest change that meets the AC**, following the pipeline contract in
   CLAUDE.md. Nothing for later issues: no extra Config fields, schema keys or stages.
7. **Test.** One test per AC line in `tests/test_<module>.py`. Offline only: committed
   transcripts in `tests/data/`, never ASR or network.
8. **Touched detect, answer, score, sentences or segments?** Run `/eval-detection` and keep its
   table for the PR.
9. **Check.** `scripts/ci-check.sh` must be green (it auto-fixes ruff first).
10. **Update only what the change touches:**
   - `CHANGELOG.md` under `## Unreleased`, in `### Added`, `### Changed (breaking)` or
     `### Fixed`, ending with `(#N)`, when users can see the change.
   - A stub landed: delete its `ponytail:` marker, update the Status table in
     `docs/architecture.md`, drop "(planned, #N)" from its heading.
   - CLI flags changed: the option tables in README.md and docs/cli.md.
   - `questions.json` shape changed: `questions.schema.json` and `schema_version` (minor = added
     keys, major = moved or removed).
11. **Commit** with a Conventional Commits message (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
12. **PR.** Ask before pushing. Title = the squash commit title, `<type>: <summary> (#N)`. Body
    follows `.github/PULL_REQUEST_TEMPLATE.md`, starts with `Closes #N`, says how each AC line is
    met, and includes the eval table from step 8 if there is one.

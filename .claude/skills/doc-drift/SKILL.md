---
name: doc-drift
description: Read-only check that QuestXtract's docs and tracker still match the code - stub status, config fields nothing reads, dependencies, CLI options, dead doc references, version, overdue milestones. Use at sprint end, before a release, after a stage lands, or when asked "are the docs up to date", "check doc drift".
---

# Doc drift check

Report only; change nothing unless asked. Finish with one list,
`file:line - what is wrong - suggested fix`, most misleading first.

1. **Stubs vs status.** `grep -rn 'ponytail:\|NotImplementedError' src/` against the Status table
   in docs/architecture.md and README.md's status lines. Markers without their issue first:
   `grep -rn 'ponytail:' src | grep -v 'ponytail: #[0-9]'` (must be empty).
2. **Config fields nothing reads.** Each must carry a `# not read yet, #N` comment in config.py.
   ```bash
   uv run python -c "import dataclasses, pathlib, re; from questxtract.config import Config; src = ' '.join(p.read_text(encoding='utf-8') for p in pathlib.Path('src/questxtract').glob('*.py') if p.name != 'config.py'); print('unused:', [f.name for f in dataclasses.fields(Config) if not re.search(r'\b' + f.name + r'\b', src)])"
   ```
3. **Dependencies.** `[project] dependencies` in pyproject.toml vs the dependency section of
   docs/architecture.md.
4. **CLI.** `uv run questxtract <command> --help` for each subcommand vs the option tables in
   README.md and docs/cli.md.
5. **Dead references.**
   ```bash
   grep -rn 'pipeline\.md' src tests scripts docs README.md
   grep -rhoE '(architecture|pipeline)\.md,? (section )?[0-9]+(\.[0-9]+)?' src tests README.md | grep -oE '[0-9]+(\.[0-9]+)?$' | sort -u | while read n; do grep -qE "^#+ ${n}[. ]" docs/architecture.md || echo "architecture.md $n: no such heading"; done
   ```
   Also markdown links in docs/ and README.md that point to files that don't exist.
6. **Version.** `version` in pyproject.toml vs `__version__` in src/questxtract/__init__.py (fine
   if it reads `importlib.metadata`), and the newest `## x.y.z` in CHANGELOG.md vs `git tag -l`.
7. **Tracker.**
   ```bash
   gh api 'repos/{owner}/{repo}/milestones?state=open' --jq '.[] | select(.due_on != null and (.due_on | fromdateiso8601) < now) | "overdue: \(.title) (\(.open_issues) open)"'
   gh issue list --state open --search 'no:milestone' --json number,title --jq '.[] | "no milestone: #\(.number) \(.title)"'
   ```

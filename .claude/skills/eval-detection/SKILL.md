---
name: eval-detection
description: Measure how a change moves question-detection recall, precision and F1 on QuestXtract's labelled recordings, main vs the working tree, as a table for the PR. Use for any change to detect.py, answer.py, score.py, sentences.py, segments.py or their Config defaults, or when asked "did this improve detection", "run the eval", "compare with main".
---

# Evaluate a detection change

The eval runs stages 3 onward over the committed transcripts in `tests/data/`: no ASR, no
network, about a second. Both runs use this tree's labels, so only the code differs.

Run in one shell call (shell state does not carry over between calls):

```bash
set -e
out=$(mktemp -d)
uv run questxtract eval tests/data/ --json > "$out/branch.json"
git worktree add --detach "$out/main" main > /dev/null
PYTHONPATH="$out/main/src" uv run python -c "import sys; from questxtract.cli import main; sys.exit(main(['eval', 'tests/data/', '--json']))" > "$out/main.json"
git worktree remove "$out/main"
uv run python - "$out/main.json" "$out/branch.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
keys = ("tp", "fp", "fn", "recall", "precision", "f1")
print("| recording | " + " | ".join(keys) + " |")
print("|---" * (len(keys) + 1) + "|")
for name in [*b["recordings"], "total"]:
    old = a["total"] if name == "total" else a["recordings"].get(name, {})
    new = b["total"] if name == "total" else b["recordings"][name]
    cells = [str(new[k]) if old.get(k) == new[k] else f"{old.get(k, '-')} -> {new[k]}" for k in keys]
    print(f"| {name} | " + " | ".join(cells) + " |")
EOF
```

Then:

1. `uv run questxtract eval tests/data/` (text report) lists missed content questions and false
   positives. Name a few the change fixed or introduced.
2. Pooled recall must stay >= 0.50 (NFR2, `tests/test_evaluate.py`). A precision gain that costs
   content recall needs one sentence of justification.
3. Paste the table in the PR description. Don't copy numbers into issues or docs; update the one
   results table in `docs/architecture.md` only in the PR that changes the numbers.

If the change is to the labels rather than the code, compare `main`'s labels instead: run the
same eval from inside the worktree (`cd "$out/main" && uv run questxtract eval tests/data/ --json`).

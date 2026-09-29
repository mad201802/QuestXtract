---
name: label-recording
description: Add a lecture recording to QuestXtract's labelled evaluation set in tests/data/ - license check, transcribe, label per the annotation guidelines, snap labels to the transcript, validate, document, re-run the eval. Use when asked to label a lecture, add a recording to the dataset, or extend tests/data.
argument-hint: <video-path> <slug>
---

# Label a recording

Arguments: $ARGUMENTS (the video, then a short stable slug such as `ttk4225_2026_09_22`).

1. **License first.** Only commit derivatives of a recording whose license allows it
   (tests/data/README.md, "License"). A public YouTube upload alone does not. Unclear: stop and ask.
2. **Transcribe once, default config.** `turbo` on CPU takes about as long as the recording.
   `uv run questxtract transcribe <video> -o tests/data/<slug>.transcript.json`
   Re-transcribing later moves the word clock, and every label must be re-snapped. The committed
   transcripts are trimmed to what the pipeline reads (tests/data/README.md, "Transcripts"); no
   script does that yet, so check the file size before committing.
3. **Read the rules**: docs/annotation_guidelines.md (the four kinds, vetoes, blind slice,
   timestamps) and the conventions in tests/data/README.md.
4. **Label into a scratch file**, one line per question-shaped sentence, the text quoted verbatim
   from the transcript: `712.0 | Is h of t absolutely integrable? | content`.
   Kinds: `content`, `rhetorical`, `check_in`, `procedural`.
5. **Snap to the transcript's clock.** Whole transcript labelled from scratch: tag everything
   blind with one window covering the recording.
   `uv run python scripts/scratch_to_labels.py <scratch.txt> -o tests/data/<slug>.labels.jsonl --transcript tests/data/<slug>.transcript.json --blind 0 <duration-seconds>`
   Seed-assisted labelling: one `--blind START END` per exhaustively labelled 10-minute window.
6. **Validate**: `uv run python scripts/validate_labels.py tests/data/<slug>.labels.jsonl` must exit 0.
7. **Document** in tests/data/README.md: a row under "Reference recordings" and the totals line.
   Labels drafted by an LLM are marked provisional until a human has checked them against the audio.
8. **Measure**: `uv run questxtract eval tests/data/` and `uv run pytest tests/test_evaluate.py`.
   A new recording may lower pooled recall; report its row in the PR rather than tuning rules to
   hide it.

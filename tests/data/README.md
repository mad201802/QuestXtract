# Labelled evaluation dataset

Issue #11. Ground truth for `evaluate.py` (#12) and the NFR2 recall gate. Read
[`docs/annotation_guidelines.md`](../../docs/annotation_guidelines.md) before labelling
anything — this file only covers structure and naming, not what a label means.

## Layout

```
tests/data/
  <recording-name>.labels.jsonl       one file per labelled recording
  <recording-name>.transcript.json    the ASR transcript the labels are timed against
  README.md                           this file
```

One JSONL file per recording. File stem should be a short, stable slug for the recording, e.g.:

```
tests/data/mit_18_100b_l1.labels.jsonl
tests/data/ttk4225_2026_09_14.labels.jsonl
tests/data/ttk4225_2026_09_15.labels.jsonl
```

Record the source recording's title, uploader, and URL in a comment-style first line is not
valid JSONL, so instead: keep that provenance in this README's table below, not inside the
data file.

## Reference recordings

| Slug | Recording | URL | Regime |
|---|---|---|---|
| `mit_18_100b_l1` | MIT 18.100B Real Analysis, Lecture 1 | https://www.youtube.com/watch?v=BWvjy0a1Ffg | chalkboard, few audible questions |
| `ttk4225_2026_09_14` | TTK4225, 2026-09-14, P03·12 — impulse responses and BIBO stability | https://www.youtube.com/watch?v=ufmTACqiE78 | digital tablet notes (OneNote/PDF sheet), heavy Socratic back-and-forth |
| `ttk4225_2026_09_15` | TTK4225, 2026-09-15, P03·14 — sinusoidal fidelity | https://www.youtube.com/watch?v=IQa2TSykWdc | digital tablet notes (OneNote/PDF sheet), frequent student Q&A |

These three are the calibration recordings the design is built around (architecture.md 1.2).
Additional recordings can be added the same way; add a row here for each.

**Transcripts.** `<slug>.transcript.json` is `scripts/transcribe.py`'s output with the default
config (faster-whisper `turbo`, punctuation prompt), trimmed to what the pipeline reads: per ASR
chunk `text`/`start`/`end` and per word `word`/`start`/`end`. Re-transcribing changes the word
clock, so a new transcript means re-snapping the labels. The MIT recording is MIT
OpenCourseWare, CC BY-NC-SA 4.0; its transcript is kept here for non-commercial evaluation only.

**Labelling method** (issue #12, replacing the hand-timed first pass of #11). The full
transcript of each recording was read sentence by sentence, with no seed detector, and every
question-shaped utterance got a `kind` per the guidelines, including `?`-less ones. Each entry's
`text` quotes the transcript verbatim, trimmed to the question, one sentence per entry, and
`start`/`end` are snapped to its first and last word, so labels and predictions share one clock.
Because the whole recording is labelled from scratch, every entry is `"split": "blind"`.

**The ground truth is the transcript, not the audio.** A question that never reached the
transcript cannot be found by any text rule, so it is not labelled.

**TTK4225 conventions.** The instructor uses a habitual confirmation tag ("True?", "Okay?",
"...right?") several times a minute. Per `docs/annotation_guidelines.md` section 2, every
standalone tag or tag closing a statement is labelled `rhetorical`; a "True." the ASR
transcribed as a statement is not labelled. Bare framing questions that name no concept
("What am I doing?", "What does it mean?") are `rhetorical`; the same question naming its
subject ("If I am times 0.5, what am I doing?") is `content`. Prompts with no question mark
("Yes, no, I do not know.", "Together.", "Discuss.") are not labelled; spoken poll options
that did get a `?` ("One, two, three, four?") are `check_in`.

Totals: MIT 34 entries (28 content), TTK4225 2026-09-14 114 entries (39 content),
TTK4225 2026-09-15 166 entries (60 content).

## Schema

One JSON object per line:

```json
{"start": 1284.2, "end": 1288.9, "text": "why does a stable sort matter here", "kind": "content", "split": "seed"}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `start` | number | yes | seconds from the recording's start, >= 0 |
| `end` | number | yes | seconds, strictly greater than `start` |
| `text` | string | yes | human-readable transcription of the span, non-empty |
| `kind` | string | yes | one of `content`, `rhetorical`, `check_in`, `procedural` — see the guidelines |
| `split` | string | no, defaults to `"seed"` | `"seed"` or `"blind"` — see below |

Entries must not overlap in time within one file.

## Train/test separation

There is no separate train/test *file* split. Instead, every entry is tagged by how it was
found:

- **`"split": "seed"`** (the default) — found via the pre-annotation workflow: a high-recall
  seed detector proposes candidates, a human labels or discards each one.
- **`"split": "blind"`** — found by labelling a 10-minute window from scratch, with no seed
  detector run first. Two such windows per recording, per the guidelines.

`evaluate.py`'s headline recall/precision/f1 run over the whole file. `blind_slice_recall`
runs only over `"split": "blind"` entries — that is the metric that is actually a fair test of
the detector, since the seed-assisted entries share the seed detector's own blind spots by
construction.

If you are looking for a conventional "held-out test set": the blind slice *is* the held-out
set, in the sense that it was not generated by anything the detector itself contributes to.

## Validating a file before committing it

```bash
uv run python scripts/validate_labels.py tests/data/ttk4225_2026_09_14.labels.jsonl
# or, everything at once:
uv run python scripts/validate_labels.py tests/data/
```

Checks: valid JSON per line, all required fields present and correctly typed, `kind` and
`split` are from the allowed sets, `end > start`, and no two entries in the same file overlap
in time. This does not check labelling *correctness* — only structural validity. Two labellers
reviewing each other's files (per the guidelines' disagreement-resolution process) is what
catches actual labelling mistakes.

## Adding a new recording

1. Pick a public lecture recording (YouTube URL is fine — the video itself is never committed,
   only the labels).
2. Get a transcript with timestamps. Either watch while jotting candidate timestamp + question
   text, or work from a full auto-caption transcript and extract candidates directly; classify
   each into a `kind` afterward. See `scripts/scratch_to_labels.py --help` for the plain-text
   scratch format this converts into JSONL. (Once a standalone seed-detector tool exists over
   `detect.py`'s rules, that becomes the faster first pass per `docs/annotation_guidelines.md`
   section 5 — until then, either manual method above is the practical equivalent.)
3. Label every candidate per `docs/annotation_guidelines.md` section 2-4. In particular, note
   section 2's rule for habitual confirmation tags ("...right?", "...true?") — some
   instructors use these constantly; decide once and apply consistently through the whole
   recording rather than re-deciding per instance.
4. Pick two 10-minute windows, label them exhaustively (every candidate, not just obvious
   ones), tag those entries `"split": "blind"` via `scratch_to_labels.py --blind START END`.
5. Run the validator (above). Fix anything it flags.
6. Add a row to the table in this file with the recording's slug, title and URL.
7. Commit the `.labels.jsonl` file and this README's updated table together.
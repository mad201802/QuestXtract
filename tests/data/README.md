# Labelled evaluation dataset

Issue #11. Ground truth for `evaluate.py` (#12) and the NFR2 recall gate. Read
[`docs/annotation_guidelines.md`](../../docs/annotation_guidelines.md) before labelling
anything — this file only covers structure and naming, not what a label means.

## Layout

```
tests/data/
  <recording-name>.labels.jsonl       one file per labelled recording
  <recording-name>.transcript.json    the ASR transcript the labels are timed against
  held_out/                           the same pair per held-out recording (below)
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

Labelled keyframes for scoring the vision model, from these and other lectures, are in
[`vision/`](vision/README.md), with their own sources and licences.

### Held-out recordings

| Slug | Recording | URL | Regime |
|---|---|---|---|
| `held_out/mit_6_003_l8` | MIT 6.003 Signals and Systems (Fall 2011), Lecture 8: Convolution, Dennis Freeman | https://www.youtube.com/watch?v=w1Z2FX8rQc0 | chalkboard and slides, live class with in-class polls |

`held_out/` holds recordings from lecturers the rules and the LLM prompt were never written from
(#92). **Nothing is tuned on them:** no rule, prompt or threshold change may be chosen by reading
their misses. A change is justified by unit tests and the calibration recordings; its PR reports
the held-out numbers before and after, separately. `questxtract eval tests/data/` reads only the
top folder, so the default eval never shows them; `questxtract eval tests/data/held_out/`
measures them.

The held-out labels were drafted by an LLM (Claude) from the transcript text, the whole
transcript labelled from scratch (every entry blind), and are provisional until a human has
checked them against the audio. Three are in-class exercises phrased as instructions, without a
`?` ("Figure out what is the third response of this system to that signal."); they are
`content`.

**Transcripts.** `<slug>.transcript.json` is a transcript (`questxtract transcribe`) with the default
config (faster-whisper `turbo`, punctuation prompt), trimmed to what the pipeline reads: per ASR
chunk `text`/`start`/`end` and per word `word`/`start`/`end`. Committed so `evaluate.py` runs
in CI without an ASR run. Re-transcribing changes the word clock, so a new transcript means
re-snapping the labels (`scratch_to_labels.py --transcript`).

**License.** Transcripts and labels are derived from the recording, so they carry its license,
not the repository's MIT license: the MIT recordings are MIT OpenCourseWare, so
`mit_18_100b_l1.*` are CC BY-NC-SA 4.0 (attribution: MIT OpenCourseWare, 18.100B Real
Analysis) and `held_out/mit_6_003_l8.*` are CC BY-NC-SA (attribution: MIT OpenCourseWare,
6.003 Signals and Systems, Fall 2011, Dennis Freeman). The TTK4225 recordings are the
lecturer's own, and `ttk4225_*` are committed with the lecturer's (Damiano's) permission. Only commit a transcript of a recording whose license
allows redistributing derivatives; a public YouTube upload alone does not.

**Labelling method** (issue #12, replacing the hand-timed first pass of #11). The full
transcript of each recording was read sentence by sentence, with no seed detector, and every
question-shaped utterance got a `kind` per the guidelines, including `?`-less ones. Each entry's
`text` quotes the transcript verbatim, trimmed to the question, one sentence per entry, and
`start`/`end` are snapped to its first and last word by `scratch_to_labels.py --transcript`, so
labels and predictions share one clock. Because the whole recording is labelled from scratch,
every entry is `"split": "blind"`.

The MIT labels were drafted by an LLM (Claude) from the transcript text. Agnes Ulltveit-Moe
reviewed them on 2026-10-05 (#93): every `content` label, and every transcript sentence ending
in `?` (each one carries a label), checked against the guidelines; all 28 `content` labels kept.

**TTK4225 conventions.** The instructor uses a habitual confirmation tag ("True?", "Okay?",
"...right?") several times a minute. Per `docs/annotation_guidelines.md` section 2, every
standalone tag or tag closing a statement is labelled `rhetorical`; a "True." the ASR
transcribed as a statement is not labelled. Bare framing questions that name no concept
("What am I doing?", "What does it mean?") are `rhetorical`; the same question naming its
subject ("If I am times 0.5, what am I doing?") is `content`. Prompts with no question mark
("Yes, no, I do not know.", "Together.", "Discuss.") are not labelled; spoken poll options
that did get a `?` are `check_in` when they name no subject ("One, two, three, four?") and
`content` when they are answers about the material ("Unbounded, bounded, ..., or you do not
know?").

Totals: MIT 34 entries (28 content), TTK4225 2026-09-14 114 entries (41 content),
TTK4225 2026-09-15 167 entries (60 content). Held out: MIT 6.003 L8 53 entries (31 content).

**The ground truth is the transcript, not the audio.** A question that never reached the
transcript cannot be found by any text rule, so it is not labelled. Entries of the #11
hand-labelled pass that are therefore gone:

| Recording | Old label | Why dropped |
|---|---|---|
| MIT | 1046 s "...this is equal to another quotient, right?" (rhetorical) | transcribed as a statement, no tag |
| MIT | 3363 s "...how do the ordering interacts with addition?" (content) | transcribed as a statement: "this here is how the ordering interacts with addition" |
| MIT | 3375 s "...how do it act interacts with multiplication?" (content) | same sentence, same reason |

Every other old label is in the new files, re-timed. The #11 TTK4225 labels were hand-timed
up to several minutes off: 2026-09-14's "Is h of t absolutely integrable?", noted at 424 s, is
spoken at 712 s.

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
2. Transcribe it (`questxtract transcribe <recording>.mp4`), trim it like the committed ones and save it as
   `<slug>.transcript.json`. Read the transcript and note each question-shaped utterance as
   `start | verbatim transcript quote | kind` (`scripts/scratch_to_labels.py --help`); pass
   `--transcript` so the times come from the transcript's words, not from typing.
3. Label every candidate per `docs/annotation_guidelines.md` section 2-4. In particular, note
   section 2's rule for habitual confirmation tags ("...right?", "...true?") — some
   instructors use these constantly; decide once and apply consistently through the whole
   recording rather than re-deciding per instance.
4. Pick two 10-minute windows, label them exhaustively (every candidate, not just obvious
   ones), tag those entries `"split": "blind"` via `scratch_to_labels.py --blind START END`.
5. Run the validator (above). Fix anything it flags.
6. Add a row to the table in this file with the recording's slug, title and URL.
7. Commit the `.labels.jsonl` file and this README's updated table together.
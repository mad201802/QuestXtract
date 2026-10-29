# Labelled evaluation dataset

Ground truth for `evaluate.py` and the NFR2 recall gate (#11). What a label means:
[`docs/annotation_guidelines.md`](../../docs/annotation_guidelines.md). This file covers structure,
naming and provenance.

## Layout

```
tests/data/
  <recording-name>.labels.jsonl       one file per labelled recording
  <recording-name>.transcript.json    the ASR transcript the labels are timed against
  <recording-name>.visual.json        board states read from its video, for `eval --board`
  held_out/                           the same pair per held-out recording (below)
  README.md                           this file
```

File stems are short, stable slugs (`mit_18_100b_l1`, `ttk4225_2026_09_14`). A recording's title,
uploader and URL go in the tables below, not in the data file.

## Reference recordings

| Slug | Recording | URL | Regime |
|---|---|---|---|
| `mit_18_100b_l1` | MIT 18.100B Real Analysis, Lecture 1 | https://www.youtube.com/watch?v=BWvjy0a1Ffg | chalkboard, few audible questions |
| `ttk4225_2026_09_14` | TTK4225, 2026-09-14, P03·12, impulse responses and BIBO stability | https://www.youtube.com/watch?v=ufmTACqiE78 | digital tablet notes (OneNote/PDF sheet), Socratic back-and-forth |
| `ttk4225_2026_09_15` | TTK4225, 2026-09-15, P03·14, sinusoidal fidelity | https://www.youtube.com/watch?v=IQa2TSykWdc | digital tablet notes (OneNote/PDF sheet), frequent student Q&A |

The calibration recordings (architecture.md 1.2). Add a row for each new recording. Labelled
keyframes for scoring the vision model are in [`vision/`](vision/README.md).

### Held-out recordings

| Slug | Recording | URL | Regime |
|---|---|---|---|
| `held_out/mit_6_003_l8` | MIT 6.003 Signals and Systems (Fall 2011), Lecture 8: Convolution, Dennis Freeman | https://www.youtube.com/watch?v=w1Z2FX8rQc0 | chalkboard and slides, live class with in-class polls |

Recordings from lecturers the rules and the LLM prompt were not written from (#92). Nothing is
tuned on them: no rule, prompt or threshold change may be chosen by reading their misses. A PR
that changes detection reports the held-out numbers before and after, separately.
`questxtract eval tests/data/` reads only the top folder; `questxtract eval tests/data/held_out/`
measures these.

The held-out labels were drafted by an LLM (Claude) from the transcript, every entry blind, and
are provisional until a human checks them against the audio. Three are in-class exercises phrased
as instructions without a `?` ("Figure out what is the third response of this system to that
signal."); they are `content`.

**Transcripts.** `<slug>.transcript.json` comes from `questxtract transcribe` with the default
config, trimmed to per-chunk `text`/`start`/`end` and per-word `word`/`start`/`end`. Committed so
CI needs no ASR run. A new transcript changes the word clock, so the labels must be re-snapped
(`scratch_to_labels.py --transcript`).

**Board states.** `<slug>.visual.json` holds what the vision stage read from the recording's video:
per board state `start`, `end`, `description`, `text`, `latex` and `confidence`, plus the
`vision_model` and `vision_prompt_version`. No keyframes are committed. Written by
`scripts/board_states.py` from the 720p video against the committed transcript; regenerate with
the script, never edit by hand. The committed files were read on the full frame; regenerating them
crops to the layout probe's region (#57).

**License.** Transcripts, board states and labels derive from the recording and carry its license,
not the repository's MIT license. `mit_18_100b_l1.*` are CC BY-NC-SA 4.0 (attribution: MIT
OpenCourseWare, 18.100B Real Analysis) and `held_out/mit_6_003_l8.*` are CC BY-NC-SA (attribution:
MIT OpenCourseWare, 6.003 Signals and Systems, Fall 2011, Dennis Freeman). `ttk4225_*` are
committed with the lecturer's (Damiano's) permission. Only commit a transcript of a recording whose
license allows redistributing derivatives; a public YouTube upload alone does not.

**Labelling method** (#12). Each transcript was read sentence by sentence with no seed detector,
and every question-shaped sentence got a `kind`, including ones without `?`. Each entry's `text`
quotes the transcript, trimmed to the question, one sentence per entry, and `start`/`end` are
snapped to its first and last word by `scratch_to_labels.py --transcript`. Every entry is
`"split": "blind"`.

The MIT labels were drafted by an LLM (Claude). Agnes Ulltveit-Moe reviewed them on 2026-10-05
(#93): every `content` label and every transcript sentence ending in `?`; all 28 `content` labels
kept.

**TTK4225 conventions.** The instructor says a confirmation tag ("True?", "Okay?", "...right?")
several times a minute. Every standalone tag or tag closing a statement is `rhetorical`; a "True."
transcribed as a statement is not labelled. Bare framing questions that name no concept ("What am
I doing?", "What does it mean?") are `rhetorical`; the same question naming its subject ("If I am
times 0.5, what am I doing?") is `content`. Prompts without a `?` ("Yes, no, I do not know.",
"Together.", "Discuss.") are not labelled. Poll options with a `?` are `check_in` when they name no
subject ("One, two, three, four?") and `content` when they are answers about the material
("Unbounded, bounded, ..., or you do not know?").

Totals: MIT 34 entries (28 content), TTK4225 2026-09-14 114 entries (41 content), TTK4225
2026-09-15 167 entries (60 content). Held out: MIT 6.003 L8 53 entries (31 content).

The ground truth is the transcript, not the audio: a question that never reached the transcript is
not labelled.

## Schema

One JSON object per line:

```json
{"start": 1284.2, "end": 1288.9, "text": "why does a stable sort matter here", "kind": "content", "split": "seed"}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `start` | number | yes | seconds from the recording's start, >= 0 |
| `end` | number | yes | seconds, strictly greater than `start` |
| `text` | string | yes | transcription of the span, non-empty |
| `kind` | string | yes | `content`, `rhetorical`, `check_in` or `procedural` (see the guidelines) |
| `split` | string | no, defaults to `"seed"` | `"seed"` or `"blind"` (below) |

Entries in one file must not overlap in time.

## Seed and blind entries

Every entry is tagged by how it was found:

- `"split": "seed"` (default): a seed detector proposed it and a labeller kept it.
- `"split": "blind"`: labelled from scratch, with no seed run.

Headline recall, precision and F1 run over the whole file. `blind_slice_recall` runs over `blind`
entries only, which do not share the seed detector's blind spots.

## Validating a file before committing it

```bash
uv run python scripts/validate_labels.py tests/data/ttk4225_2026_09_14.labels.jsonl
# or, everything at once:
uv run python scripts/validate_labels.py tests/data/
```

Checks valid JSON per line, required fields and types, allowed `kind` and `split`, `end > start`,
and no overlapping entries. It checks structure, not whether a label is right; that is what review
by a second labeller is for.

## Adding a new recording

1. Pick a public lecture recording whose license allows derivatives. The video is never committed.
2. Transcribe it (`questxtract transcribe <recording>.mp4`), trim it like the committed ones and
   save it as `<slug>.transcript.json`. Note each question-shaped sentence as
   `start | verbatim transcript quote | kind` (`scripts/scratch_to_labels.py --help`) and pass
   `--transcript` so the times come from the transcript's words.
3. Label every candidate per `docs/annotation_guidelines.md` sections 2 to 4. Apply the
   confirmation-tag rule the same way through the whole recording.
4. For a seed-assisted recording, label two 10-minute windows exhaustively and tag them
   `"split": "blind"` with `scratch_to_labels.py --blind START END`.
5. Run the validator and fix what it flags.
6. Add a row to the table above and commit it with the `.labels.jsonl` file.

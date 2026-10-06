# Annotation Guidelines — Labelled Evaluation Dataset

Issue #11. This document is what a labeller reads before touching a transcript, and what a
reviewer checks a labelled file against. If a labelling decision isn't covered here, the
commit that adds the fix should update this file too — the guideline is meant to converge,
not to be re-derived from memory each time.

## 1. What we are labelling, and why

The pipeline's detector (`detect.py`) proposes candidate questions. NFR2 requires recall
>= 50% on our own labelled set, measured by `evaluate.py` (#12). This dataset is that ground
truth. See `architecture.md` sections 1.1 and 4 for the design this implements.

**The central difficulty, restated for labellers:** a lecture is not a dialogue corpus. Most
interrogative-*sounding* utterances are not questions we want extracted. Getting the
false-positive categories right is at least as important as finding real questions — a
labeller who only marks obvious content questions and calls everything else "not a question"
throws away the signal `evaluate.py` needs to report a false-positive breakdown.

## 2. The four labels

Every question-shaped utterance gets exactly one `kind`. Non-questions are not labelled at
all — see section 4 (negative sampling).

### `content` — a real question we want extracted

A genuine request for information that a student should be able to answer after the lecture.
This is the only category the detector is scored on finding (NFR2's recall target).

> "Why does a stable sort matter for multi-key sorting?"
> "So the question is: what happens to the derivative at that point?"
> "Does anyone know why that matters?"

Includes the **Socratic case**: the lecturer asks and answers within the same breath. Still
`content` — it is still a real question, just also self-answered.

### `rhetorical` — asked for effect, not for an answer

The lecturer is not expecting or wanting a spoken response. Three common shapes:

- **Framing rhetoric**, used to introduce a topic: "What happens if we double this? Well..."
- **Rhetorical confirmation**: "Isn't that interesting?" "Makes sense, right?"
- **The habitual confirmation tag** — "...right?", "...true?", "...you know?" — tacked onto
  the end of a declarative sentence. Label `rhetorical`, not a veto and not `check_in`, even
  when it reads as a bare verbal tic. Some instructors use this constantly (multiple times a
  minute); apply the same rule every time it occurs rather than re-deciding case by case. The
  boundary that still matters: label only the tag itself as `rhetorical` when it stands alone
  or closes an otherwise-declarative sentence. If it closes off a **real content question**
  ("...so why is this not a field? It's for multiplication, right?"), label the content
  question as `content` and do not also create a second entry for the trailing tag.

A framing question that only announces the next step and names no concept of its own ("So how
do we do that?", "So why is this popular?") is rhetorical; one that asks about a concept a student
should be able to answer afterwards ("What is a field?", "How does a Markov chain work?") is
`content`, even when the lecturer answers it straight away.

For the first two shapes, the test: if a student answered out loud, would it derail the
lecture rather than fit into it? If yes, `rhetorical`.

### `check_in` — a classroom-management question, not a content question

Not about the material at all.

> "Can everyone hear me?"
> "Is this too fast for anyone?"
> "Are we ready to move on?"

Includes polls that name no subject matter ("one or two? show of hands?", "One, two, three,
four?") — these are `check_in`, not `content`, even though a real answer is expected, because
the answer is not itself instructional content.

A **subject poll**, whose options are themselves answers about the material ("Do I go outside,
inside, or you do not know?", "Unbounded, bounded, or you do not know?"), is `content`: it asks
the question the options answer. A repeat of the options as its own sentence is `content` too,
like any repeated content question.

### `procedural` — logistics, not lecture content

> "Can you pass me the marker?"
> "Is this thing recording?"
> "Can you turn the lights down?"

Includes managing the exchange with a student rather than its content: asking for a name
("Could you say your name first, please?"), calling on someone ("Yeah?"), and **clarification
echoes** - the lecturer repeating a fragment of a student's question back to check it ("Okay,
sorting and what?", "It's faster, but only on average?"). A full restatement of the
student's question ("So the question was, ..., does that still hold for negative
numbers?") is not an echo: label it `content`.

### One sentence, two utterances

A label covers one transcript sentence, and `evaluate.py` matches one prediction to one label,
so a sentence never gets two entries. When one ASR sentence holds a content question and a
check-in, tag or procedural aside ("If I do the integral of this from minus infinity to plus
infinity, are we together that this area is going to be infinite?"), label the whole sentence
`content`, the same rule as a tag closing a content question. A check-in that only asks for
agreement on an answer just given ("Finite. True. Are we together that it's finite?") holds no
content question and stays `check_in`.

## 3. Vetoes — things that look like questions and are not

These get **no label at all** — they are not entered into the dataset as questions, positive
or negative. They exist as a labeller checklist, not a fifth category, because the detector
treats them as hard rejects (architecture.md 7.2), not low-scored candidates.

- **Declarative "what".** "That's what I mean." / "What we call a stable sort is..." — a
  statement using the word "what," not a question.
- **A causal clause following a real question.** "...? Because the cache is cold." — this is
  the *answer*, immediately after the question. Label the question itself normally; do not
  also create an entry for the causal clause.

**Note:** habitual confirmation tags ("...right?", "...true?", "...you know?") are **not** a
veto — they are labelled `rhetorical`. See section 2.

## 4. Negative sampling (`NOT_QUESTION` equivalent)

We do not exhaustively label every non-question sentence in a 90-minute lecture — that is
unnecessary and slow. Negatives come from two sources instead:

1. **Every sentence the seed detector proposed but a human rejected** (see section 5) is
   implicitly a negative — it is not entered as a positive `content`/`rhetorical`/
   `check_in`/`procedural` label, so `evaluate.py` treats any detector match against that
   span as a false positive by omission.
2. **The blind slice** (section 6) is labelled exhaustively within its 10-minute window —
   every interrogative-shaped sentence in that window gets a `kind`, whether or not the seed
   detector proposed it, because the blind slice exists specifically to catch what the seed
   never surfaced.

## 5. Pre-annotation workflow

Labelling 80 minutes of raw video from scratch is slow. Instead:

1. Run the **high-recall seed detector** over the transcript: any sentence containing `?`,
   OR starting with a wh-word, OR starting with an auxiliary verb. No scoring, no vetoes —
   deliberately over-inclusive.
2. A human labeller goes through every seed hit, in order, and assigns it a `kind` from
   section 2, or discards it per section 3/4 if it is not a real question at all.
3. This is roughly 5x faster than scrubbing the full recording, per architecture.md 4.1.

**The seed only proposes; it never decides.** A labeller who accepts every seed hit at face
value is not labelling, they are copying the detector's own bias into the ground truth.

## 6. The blind control slice — non-negotiable

Pre-annotation cannot surface a question the seed detector never proposed in the first place
(e.g. one with no `?`, no wh-word, no auxiliary verb — an implicit-cue question like "so the
question is whether... "). Without a check on this, our recall number would measure how good
the *seed* is, not how good the *pipeline's detector* is.

For every recording:

- Pick **two 10-minute windows**, spread apart (e.g. one from the first third, one from the
  last third of the recording).
- Label every interrogative-shaped sentence in those windows **from scratch**, with no seed
  detector run first.
- Mark every entry from these windows with `"split": "blind"` (see the schema doc). Every
  other entry is `"split": "seed"`.

`evaluate.py`'s `blind_slice_recall` metric is computed only over `"split": "blind"` entries.
If that number sits far below the headline recall, the dataset itself is suspect — it is
measuring the seed, not the system — and that is a data problem, not a detector problem.

## 7. Timestamps

- `start` / `end` are the transcript span's boundaries in seconds, matching the source
  recording's own clock (not a clipped excerpt's clock).
- Use the ASR transcript's own segment boundaries where they align with the spoken question;
  if the ASR segment includes extra material before/after the question, trim the label's
  `start`/`end` to the question itself, not the whole ASR chunk.
- `text` is a plain transcription of the labelled span, for human readability during review.
  It does not need to match the pipeline's own ASR output verbatim — `evaluate.py` matches by
  time overlap, not text (architecture.md 4.2), specifically because ASR text will differ from
  what a human typed.

## 8. Worked example

Source: MIT 18.100B Real Analysis, Lecture 1. The other reference recordings are TTK4225,
2026-09-14 (P03·12) and 2026-09-15 (P03·14) — see `tests/data/README.md` for the full titles
and URLs.

```jsonl
{"start": 142.0, "end": 145.3, "text": "what is a supremum", "kind": "content", "split": "seed"}
{"start": 310.5, "end": 312.0, "text": "does that make sense to everyone", "kind": "check_in", "split": "seed"}
{"start": 604.2, "end": 606.8, "text": "so why should we care about this", "kind": "rhetorical", "split": "seed"}
{"start": 1802.1, "end": 1805.6, "text": "so the question we should ask is whether this bound is tight", "kind": "content", "split": "blind"}
```

The fourth entry has no `?` and no wh-word start — the seed detector would never have proposed
it. It only exists because the 1800-1900s window fell inside a blind slice and was labelled
from scratch.

**Note on the TTK4225 instructor's habitual tag.** Both TTK4225 recordings use "...right?"/
"...true?" very frequently as a confirmation tag. Per section 2, every instance is labelled
`rhetorical` unless it trails a real content question, in which case only the content question
gets an entry:

```jsonl
{"start": 1185.51, "end": 1187.79, "text": "This area here is going to be infinite, true?", "kind": "rhetorical", "split": "blind"}
{"start": 3014.0, "end": 3017.0, "text": "and why is this not a field? it's for multiplication, right?", "kind": "content", "split": "blind"}
```

Both TTK4225 files apply this to every standalone tag and every tag closing a statement; see
`tests/data/README.md`'s TTK4225 conventions for the related framing and poll rules.

## 9. Disagreements

Two labellers reviewing the same recording occasionally disagree. Preferred resolution order:
(1) re-read section 2's test for the relevant category against the actual audio, not just the
transcript text — tone carries information text does not; (2) if still ambiguous, default to
the *lower*-stakes label (a real question wrongly dropped costs less than a rhetorical
question wrongly counted as content, since recall is the graded metric and false positives
are the harder problem to over-credit); (3) flag it in the PR description rather than silently
picking one — a genuinely ambiguous case is itself useful information about the label
boundary.
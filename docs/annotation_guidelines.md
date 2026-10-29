# Annotation guidelines: labelled evaluation dataset

What a labeller reads before labelling a transcript, and what a reviewer checks a labelled file
against (#11). When a labelling decision is not covered here, the commit that settles it updates
this file.

## 1. What we are labelling, and why

NFR2 requires recall >= 50% on our labelled set, measured by `evaluate.py`. This dataset is that
ground truth (architecture.md 4).

Most question-shaped sentences in a lecture are not questions we want. Label the false-positive
kinds as carefully as the real questions: `evaluate.py` reports false positives by kind.

## 2. The four labels

Every question-shaped sentence gets exactly one `kind`. Non-questions are not labelled (section 4).

### `content`: a question we want extracted

A request for information a student should be able to answer after the lecture. The only kind the
detector is scored on finding.

> "Why does a stable sort matter for multi-key sorting?"
> "So the question is: what happens to the derivative at that point?"
> "Does anyone know why that matters?"

A question the lecturer asks and answers at once (the Socratic case) is still `content`.

### `rhetorical`: asked for effect

The lecturer expects no spoken answer. Three shapes:

- Framing, to introduce a topic: "What happens if we double this? Well..."
- Rhetorical confirmation: "Isn't that interesting?" "Makes sense, right?"
- The habitual confirmation tag ("...right?", "...true?", "...you know?") at the end of a
  declarative sentence. Label it `rhetorical` every time, even when it is a verbal tic. When it
  closes a content question ("...so why is this not a field? It's for multiplication, right?"),
  label the question `content` and add no entry for the tag.

A framing question that only announces the next step and names no concept ("So how do we do
that?", "So why is this popular?") is `rhetorical`; one that asks about a concept ("What is a
field?", "How does a Markov chain work?") is `content`, even when answered straight away.

For the first two shapes: if a student answered out loud, would it derail the lecture? If yes,
`rhetorical`.

### `check_in`: classroom management

> "Can everyone hear me?"
> "Is this too fast for anyone?"
> "Are we ready to move on?"

Polls that name no subject matter ("one or two? show of hands?", "One, two, three, four?") are
`check_in`.

A subject poll, whose options are answers about the material ("Do I go outside, inside, or you do
not know?", "Unbounded, bounded, or you do not know?"), is `content`, and so is a repeat of its
options as a sentence of its own.

### `procedural`: logistics

> "Can you pass me the marker?"
> "Is this thing recording?"
> "Can you turn the lights down?"

Also managing the exchange with a student: asking for a name ("Could you say your name first,
please?"), calling on someone ("Yeah?"), and clarification echoes, where the lecturer repeats a
fragment of a student's question to check it ("Okay, sorting and what?", "It's faster, but only
on average?"). A full restatement of the student's question ("So the question was, ..., does that
still hold for negative numbers?") is `content`.

### One sentence, two utterances

A label covers one transcript sentence, and `evaluate.py` matches one prediction to one label, so
a sentence gets one entry. A sentence holding a content question and a check-in, tag or aside
("If I do the integral of this from minus infinity to plus infinity, are we together that this
area is going to be infinite?") is `content`. A check-in that only asks for agreement on an answer
just given ("Finite. True. Are we together that it's finite?") stays `check_in`.

## 3. Not questions

These get no label:

- Declarative "what": "That's what I mean.", "What we call a stable sort is..."
- A causal clause right after a question: "...? Because the cache is cold." Label the question;
  the clause is its answer.

Confirmation tags are labelled `rhetorical` (section 2), not skipped.

## 4. Negative sampling

Not every non-question sentence is labelled. Negatives come from two places:

1. A sentence the seed detector proposed and a labeller rejected (section 5) has no label, so a
   detector match on it counts as a false positive.
2. The blind slices (section 6) are labelled exhaustively: every question-shaped sentence in them
   gets a `kind`.

## 5. Pre-annotation workflow

For long recordings:

1. Run the seed detector over the transcript: every sentence containing `?`, starting with a
   wh-word, or starting with an auxiliary verb. No scoring, no vetoes.
2. Go through every seed hit in order and give it a `kind` (section 2), or no label (sections 3
   and 4).

Judge every hit yourself; accepting them as proposed copies the seed's bias into the ground truth.

## 6. The blind control slice

The seed cannot propose a question with no `?`, no wh-word and no auxiliary ("so the question is
whether..."). Without a check, recall would measure the seed.

For every seed-assisted recording:

- Pick two 10-minute windows, one from the first third and one from the last third.
- Label every question-shaped sentence in them from scratch, with no seed run.
- Mark those entries `"split": "blind"`; every other entry is `"split": "seed"`.

`blind_slice_recall` is computed over `blind` entries only. If it sits far below the headline
recall, the dataset is measuring the seed.

## 7. Timestamps

- `start` / `end` are in seconds on the source recording's clock.
- Trim them to the question itself, not the whole ASR chunk.
- `text` is a plain transcription of the span. `evaluate.py` matches by time overlap, not text
  (architecture.md 4.2).

## 8. Worked example

Source: MIT 18.100B Real Analysis, Lecture 1. The other reference recordings, TTK4225 2026-09-14
(P03·12) and 2026-09-15 (P03·14), are listed in `tests/data/README.md`.

```jsonl
{"start": 142.0, "end": 145.3, "text": "what is a supremum", "kind": "content", "split": "seed"}
{"start": 310.5, "end": 312.0, "text": "does that make sense to everyone", "kind": "check_in", "split": "seed"}
{"start": 604.2, "end": 606.8, "text": "so why should we care about this", "kind": "rhetorical", "split": "seed"}
{"start": 1802.1, "end": 1805.6, "text": "so the question we should ask is whether this bound is tight", "kind": "content", "split": "blind"}
```

The fourth entry has no `?` and no wh-word start, so the seed would not propose it; it exists
because 1800 to 1900 s fell inside a blind slice.

The TTK4225 instructor says "...right?" and "...true?" very often. Each is `rhetorical` unless it
closes a content question, which then gets the only entry:

```jsonl
{"start": 1185.51, "end": 1187.79, "text": "This area here is going to be infinite, true?", "kind": "rhetorical", "split": "blind"}
{"start": 3014.0, "end": 3017.0, "text": "and why is this not a field? it's for multiplication, right?", "kind": "content", "split": "blind"}
```

`tests/data/README.md`'s TTK4225 conventions cover the related framing and poll rules.

## 9. Disagreements

When two labellers disagree: (1) re-check section 2 against the audio, not only the transcript,
since tone matters; (2) if still unclear, pick the label that does not count as `content`, since
over-crediting a rhetorical question inflates recall; (3) note the case in the PR description.

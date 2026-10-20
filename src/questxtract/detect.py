"""Stage 6: find the sentences that are questions.

Purely lexical and orthographic. No classifier, no prosody. Every decision leaves a reason
code, which is what makes both the review workflow and the LLM judge's job possible
(architecture.md 7).

Sentences are only extracted, never invented: every candidate's text is a span of the
transcript. Board states, when vision ran, only add to the score of a candidate the rules
already found (architecture.md 7.3); they never make one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import Context, QuestionCandidate, Sentence, VisualContext, overlapping, stage

_WH_WORDS = frozenset({"who", "what", "when", "where", "why", "how", "which", "whose", "whom"})
# A wh-word with a contracted verb is still that wh-word: "What's ...?", "Who'd ...?".
_WH_CONTRACTIONS = frozenset({"s", "re", "d", "ll", "ve"})
_AUX_WORDS = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "am",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "have",
        "has",
        "had",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "don't",
        "doesn't",
        "didn't",
        "can't",
        "couldn't",
        "won't",
        "wouldn't",
        "shouldn't",
        "haven't",
        "hasn't",
    }
)
# Openers that sit in front of a question without being part of it: "So what ...?",
# "And then how ...?".
_DISCOURSE_MARKERS = frozenset(
    {"so", "and", "but", "or", "now", "then", "well", "okay", "ok", "alright", "also", "anyway"}
)
# Multi-word openers that announce a question without being part of it: "My question is
# that doesn't ...?". Only phrases that are preambles by construction, longest first so an
# optional trailing "that"/"if" is taken with its phrase.
_PREAMBLES = tuple(
    sorted(
        (
            tuple(p.split())
            for p in (
                "my question is",
                "my question is that",
                "the question is",
                "the question is that",
                "the question was",
                "i'd like to ask",
                "i would like to ask",
                "i was wondering",
                "i was wondering if",
                "i was wondering whether",
                "i'm wondering",
            )
        ),
        key=len,
        reverse=True,
    )
)

# A wh-word can follow a preposition that belongs to the question: "For which age ...?",
# "In what sense ...?". Skipped only right before a wh-word, so "In the limit ..." is untouched.
_PREPOSITIONS = frozenset(
    "about after against among as at before between by during for from in into like of on "
    "over per since than through to toward towards under until upon via with within without".split()
)

_TYPES = {
    "what": "what",
    "why": "why",
    "how": "how",
    "who": "what",
    "when": "what",
    "where": "what",
    "which": "what",
    "whose": "what",
    "whom": "what",
}

# A sentence, or its closing clause, made only of these is a confirmation tag: "Okay?",
# "True, you see?", "..., right?". Labelled rhetorical (annotation_guidelines.md 2).
_TAG_WORDS = frozenset({"okay", "ok", "right", "true", "yes", "yeah", "correct", "you", "see"})

# Classroom management, not content: labelled check_in or procedural. Matched as whole words
# of the lower-cased sentence, trailing vocatives allowed ("How much are we together, people,
# on this?"). The comprehension checks do not count when content follows them: "Does that
# make sense?" is a check-in, "Why does it make sense to sort first?" is a question, and
# "Do you see it?" is one while "Do you see why ...?" and "What do you see?" are not. A bare
# vocative ("Everyone?") is a check-in; a trailing one is not ("What do I get, people?" is
# content).
# English, grown from the labelled recordings: the LLM detector (#88) is the real content
# filter, this only keeps the obvious ones out.
_CHECK_INS = re.compile(
    r"\bmakes? sense\b(?! to (?!everyone\b|everybody\b|you\b)| of\b| for\b| in\b| here\b)"
    r"|\b(?:is|are) (?:that|it|this|they|these|those|the question|everything) clear\b"
    r"(?! (?:why|what|how|that|whether|if|when|where)\b)"
    r"|\bhow clear (?:have i been|is it|was it)\b"
    r"|(?:(?:^|, |how much )do you see\b|^you see(?= \w))"
    r"(?! (?:any|why|how|whether|if|which|where|when)\b)"
    r"|\bare you following\b|\b(?:get|getting|got) the point\b"
    r"|\bwe are together\b|\bhow are we(?=[,?.!]|$)"
    r"|\bdo (?:we|you) remember (?:this|that|it)\b|\b(?:have|keep) (?:that|this|it) in mind\b"
    r"|\bhow many of you\b|^\W*(?:people|everyone|everybody|guys|folks)\W*$"
    r"|\b(?:are we together|you see the point|any (?:other )?questions|who here|want to share"
    r"|want to tell me|your name)\b"
)

# Only function words: ordinary verbs, negation and numbers remain content signals.
# In particular, "be", "one" and "much" stay out: "Can it be?", "This one, what
# is it?" and "How much?" are content questions in the reference recordings.
_FRAMING_STOPWORDS = (
    _WH_WORDS
    | frozenset(word for word in _AUX_WORDS if "'" not in word)
    | _DISCOURSE_MARKERS
    | _PREPOSITIONS
    | frozenset(
        "a an the i me my mine myself we us our ours ourselves you your yours yourself "
        "yourselves he him his himself she her hers herself it its itself they them their "
        "theirs themselves this that these those if whether exactly right".split()
    )
)

# A head word without a '?' stays under the 0.45 gate: on the reference recordings every such
# sentence was a statement ("What I mean is ...", "So when we ...").
# Recovering a question the ASR left unpunctuated needs a second, independent gate (a local
# answer slot, architecture.md 7.1); add it when a labelled question is lost to it.
_SCORES = {
    "mark": 0.6,
    "wh": 0.4,
    "aux": 0.4,
    "tag": -0.4,
    "check_in": -0.4,
    "framing": -0.4,
}


def score_sentence(sentence: str) -> tuple[float, tuple[str, ...]]:
    """Score how likely a sentence is a question.

    Returns:
        The score in ``[0, 1]`` and the names of the heuristics that fired.

    """
    reasons: list[str] = []
    if sentence.rstrip().endswith("?"):
        reasons.append("mark")
    heads = _heads(sentence)
    if any(h in _WH_WORDS for h in heads):
        reasons.append("wh")
    if any(h in _AUX_WORDS for h in heads):
        reasons.append("aux")
    if _is_tag(sentence):
        reasons.append("tag")
    if _CHECK_INS.search(sentence.casefold().replace("\u2019", "'")):
        reasons.append("check_in")
    score = min(1.0, max(0.0, sum((_SCORES[r] for r in reasons), 0.0)))
    if (
        "mark" in reasons
        and ("wh" in reasons or "aux" in reasons)
        and not {"tag", "check_in"} & set(reasons)
        and all(_uncontract(word) in _FRAMING_STOPWORDS for word in _words(sentence))
    ):
        reasons.append("framing")
        # Penalise the bounded score even when both wh and aux signals fired.
        score = max(0.0, score + _SCORES["framing"])
    return round(score, 2), tuple(reasons)


def question_type(sentence: str) -> str:
    """Classify a question by its head word."""
    for head in _heads(sentence):
        if head in _TYPES:
            return _TYPES[head]
        if head in _AUX_WORDS:
            return "yes_no"
    return "other"


def _words(text: str) -> list[str]:
    """Lower-cased words, contractions kept whole so "don't" is one token."""
    return re.findall(r"\w+(?:'\w+)?", text.casefold().replace("\u2019", "'"))


def _is_tag(sentence: str) -> bool:
    """Whether ``sentence`` is, or ends in, a confirmation tag rather than a question.

    A tag after a real question ("Why is it not a field, right?") does not count: the
    question's own head wins.
    """
    if not sentence.rstrip().endswith("?"):
        return False
    tail = _words(sentence.rsplit(",", 1)[-1])
    if not tail or not set(tail) <= _TAG_WORDS:
        return False
    return not any(h in _WH_WORDS or h in _AUX_WORDS for h in _heads(sentence.rsplit(",", 1)[0]))


def _heads(sentence: str) -> tuple[str, ...]:
    """Return the words a question in ``sentence`` can open with, past discourse markers.

    For a sentence ending in ``?`` whose opener is not interrogative, later comma clauses
    count too, since ASR runs a preamble into the question: "If you have the data, does
    that mean ...?", "This area here, is it finite, infinite, or you do not know?". The last
    clause counts with any head; an earlier one only with an auxiliary, because a wh-word
    there usually opens a relative clause (", what would you ..., people?"). Without the
    ``?`` none of them count, or every ", which is ..." would read as a wh-question; nor after
    an interrogative opener, where the clauses are part of the question: "Is it a function
    from X to Y, where X is compact?" is yes/no.
    """
    heads: list[str] = []
    opener = _opener(sentence)
    if opener:
        heads.append(opener)
    if sentence.rstrip().endswith("?") and opener not in _WH_WORDS | _AUX_WORDS:
        *earlier, last = sentence.split(",")
        clause_heads = [h for h in map(_opener, earlier[1:]) if h in _AUX_WORDS]
        for head in [*clause_heads, _opener(last)]:
            if head and head not in heads:
                heads.append(head)
    return tuple(heads)


def _opener(clause: str) -> str:
    """Return the first word of ``clause`` past discourse markers and preambles, or ``""``.

    A preposition right before a wh-word is skipped too: "For which age ...?" opens on "which".
    """
    words = _words(clause)
    while words:
        if words[0] in _DISCOURSE_MARKERS:
            words = words[1:]
            continue
        preamble = next((p for p in _PREAMBLES if tuple(words[: len(p)]) == p), None)
        if preamble is None:
            break
        words = words[len(preamble) :]
    if len(words) > 1 and words[0] in _PREPOSITIONS and _uncontract(words[1]) in _WH_WORDS:
        return _uncontract(words[1])
    return _uncontract(words[0]) if words else ""


def _uncontract(word: str) -> str:
    """Return the wh-word in a contraction ("what's" -> "what"), or ``word`` unchanged."""
    base, _, suffix = word.partition("'")
    return base if base in _WH_WORDS and suffix in _WH_CONTRACTIONS else word


def _content_words(text: str) -> frozenset[str]:
    """Return the words of ``text`` that carry content: lower-cased, minus framing stopwords."""
    return frozenset(_uncontract(word) for word in _words(text)) - _FRAMING_STOPWORDS


def detect_questions(
    sentences: Sequence[Sentence],
    *,
    threshold: float = 0.45,
    max_chars: int = 350,
    visual: Sequence[VisualContext] = (),
    board_delta: float = 0.0,
) -> tuple[QuestionCandidate, ...]:
    """Find the question candidates among ``sentences``.

    Args:
        sentences: The transcript's sentences, in recording order.
        threshold: Minimum score for a sentence to be reported.
        max_chars: A sentence longer than this is rejected outright (architecture.md 7.3).
        visual: Board states, empty when vision is off.
        board_delta: Added to a candidate's score, with the reason ``board_overlap``, when it
            shares a content word with the ``text`` of a board state on screen while it was
            asked. Applied after the gate, so it never makes a candidate.

    Returns:
        The candidates, in recording order.

    """
    out: list[QuestionCandidate] = []
    for sentence in sentences:
        text = sentence.text.strip()
        if not text or len(text) > max_chars:
            continue
        score, reasons = score_sentence(text)
        if score >= threshold:
            if board_delta > 0 and _on_the_board(
                text, overlapping(visual, sentence.start, sentence.end)
            ):
                score = min(1.0, score + board_delta)
                reasons = (*reasons, "board_overlap")
            out.append(
                QuestionCandidate(
                    sentence_id=sentence.id,
                    text=text,
                    score=round(score, 2),
                    question_type=question_type(text),
                    reasons=reasons,
                )
            )
    return tuple(out)


def _on_the_board(text: str, states: Sequence[VisualContext]) -> bool:
    """Whether ``text`` shares a content word with the text of any of ``states``."""
    words = _content_words(text)
    return any(words & _content_words(state.text) for state in states)


@stage("detect")
def detect(ctx: Context) -> Context:
    """Score ``ctx.sentences`` into ``ctx.candidates``, with ``ctx.visual`` as context."""
    ctx.candidates = detect_questions(
        ctx.sentences,
        threshold=ctx.config.min_question_score,
        max_chars=ctx.config.max_question_chars,
        visual=ctx.visual,
        board_delta=ctx.config.board_overlap_delta,
    )
    ctx.stats["candidates"] = len(ctx.candidates)
    return ctx

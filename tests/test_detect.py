import pytest

from questxtract.detect import detect_questions, question_type, score_sentence


def test_an_unmarked_head_word_is_not_a_candidate(sentences):
    unmarked = type(sentences[0])(
        text="Does anyone know why that matters", word_start=15, word_end=20, start=9.0, end=12.0
    )
    assert [c.text for c in detect_questions([*sentences, unmarked])] == ["What is a stable sort?"]
    assert score_sentence(unmarked.text) == (0.4, ("aux",))


def test_statements_are_not_questions(sentences):
    assert detect_questions([sentences[0]]) == ()


def test_threshold_filters_low_scores(sentences):
    unmarked = type(sentences[0])(
        text="Does anyone know why that matters", word_start=15, word_end=20, start=9.0, end=12.0
    )
    assert len(detect_questions([*sentences, unmarked], threshold=0.3)) == 2
    assert [c.text for c in detect_questions([*sentences, unmarked], threshold=0.45)] == [
        "What is a stable sort?"
    ]
    assert detect_questions([*sentences, unmarked], threshold=1.1) == ()


def test_over_long_sentences_are_rejected(sentences):
    assert detect_questions(sentences, max_chars=5) == ()


def test_non_ascii_first_word_is_not_skipped():
    assert score_sentence("Кто is here") == (0.0, ())
    assert score_sentence("Äh is that right") == (0.0, ())


def test_score_is_always_a_float():
    score, _ = score_sentence("hello there")
    assert score == 0.0 and isinstance(score, float)


def test_question_type_from_the_head_word():
    assert question_type("Why does it matter?") == "why"
    assert question_type("How does it work?") == "how"
    assert question_type("Who wrote this?") == "what"
    assert question_type("Is it stable?") == "yes_no"
    assert question_type("...") == "other"
    assert question_type("Sorting matters") == "other"


def test_head_word_found_past_discourse_markers():
    assert score_sentence("So what does it mean that a field is ordered?")[1] == ("mark", "wh")
    assert score_sentence("And then, how do you add them?")[1] == ("mark", "wh")
    assert score_sentence("Now is the state Markov?")[1] == ("mark", "aux")
    assert question_type("So why is this popular?") == "why"


def test_head_word_found_past_question_preambles():
    sentence = "My question is that doesn't a Markov assumption make this harder?"
    assert score_sentence(sentence) == (1.0, ("mark", "aux"))
    assert score_sentence("The question is, why does it converge?")[1] == ("mark", "wh")
    assert score_sentence("So the question was how do we sample it?")[1] == ("mark", "wh")
    assert score_sentence("I'd like to ask is the policy stationary?")[1] == ("mark", "aux")
    assert score_sentence("I was wondering whether is it bounded?")[1] == ("mark", "aux")
    assert question_type("My question is what happens at zero?") == "what"
    # A preamble alone gives no head: the statement stays under the gate.
    assert score_sentence("The question is important.") == (0.0, ())


def test_head_word_found_in_the_last_clause_of_a_question():
    reasons = score_sentence("If you have the data, does that mean it is better?")[1]
    assert reasons == ("mark", "aux")
    assert question_type("For AlphaTensor, is that fast?") == "yes_no"
    # Without the question mark a trailing clause is a relative clause, not a question.
    assert score_sentence("We take the limit, which is zero.") == (0.0, ())


def test_negated_auxiliaries_open_questions():
    assert score_sentence("Don't we have a different state space?")[1] == ("mark", "aux")
    assert score_sentence("Doesn’t the assumption make it harder?")[1] == ("mark", "aux")


def test_a_contracted_wh_word_opens_a_question():
    assert score_sentence("What's a stable sort?") == score_sentence("What is a stable sort?")
    assert score_sentence("What's a stable sort?") == (1.0, ("mark", "wh"))
    assert question_type("What's a stable sort?") == "what"


def test_each_contracted_wh_word_keeps_its_own_type():
    for sentence, kind in [
        ("Where's the pole?", "what"),
        ("How's it going to grow?", "how"),
        ("Who's seen a Bode plot?", "what"),
        ("Why'd the integral diverge?", "why"),
        ("What're the two cases?", "what"),
        ("So how'll we bound it?", "how"),
        ("From where'd the factor come?", "what"),
    ]:
        assert "wh" in score_sentence(sentence)[1], sentence
        assert question_type(sentence) == kind, sentence


def test_a_curly_apostrophe_contracts_like_a_straight_one():
    assert score_sentence("What’s a stable sort?") == score_sentence("What's a stable sort?")
    assert question_type("What’s a stable sort?") == "what"


def test_a_contraction_outside_a_question_is_decided_as_before(sentences):
    assert "wh" not in score_sentence("That's what I mean?")[1]
    # Without the '?' a wh-contraction stays under the gate, like the bare wh-word.
    statement = type(sentences[0])(
        text="What's more, it is stable.", word_start=15, word_end=20, start=9.0, end=12.0
    )
    assert score_sentence(statement.text) == score_sentence("What is more, it is stable.")
    assert statement.text not in [c.text for c in detect_questions([*sentences, statement])]


def test_any_is_not_an_auxiliary():
    assert "aux" not in score_sentence("Any other questions?")[1]


def test_confirmation_tags_fall_below_the_gate():
    for tag in ("Okay?", "True, you see?", "It's for multiplication, right?"):
        score, reasons = score_sentence(tag)
        assert "tag" in reasons and score < 0.45, tag


def test_a_tag_after_a_real_question_does_not_count():
    assert score_sentence("And why is this not a field, right?") == (1.0, ("mark", "wh"))
    assert "tag" not in score_sentence("Is it right?")[1]


def test_check_ins_score_below_content_questions():
    check_in, reasons = score_sentence("Is that clear?")
    content, _ = score_sentence("Is the state Markov?")
    assert "check_in" in reasons and check_in < content
    # Still a candidate, but its unanswered confidence (0.55 x score) stays under 0.40.
    assert 0.45 <= check_in and 0.55 * check_in < 0.40


def test_classroom_management_is_a_check_in():
    for text in ("Does someone want to tell me what they are?", "Could you say your name?"):
        assert "check_in" in score_sentence(text)[1], text


def test_a_trailing_clause_does_not_override_an_interrogative_opener():
    text = "Is it a function from X to Y, where X is compact?"
    assert score_sentence(text) == (1.0, ("mark", "aux"))
    assert question_type(text) == "yes_no"
    assert question_type("The derivative is 2x, what do you get?") == "what"


def test_comprehension_checks_do_not_swallow_content_questions():
    for text in (
        "Does that make sense?",
        "Makes sense to everyone?",
        "Is that clear?",
        "How much are we together, people, on this?",
        "Does this plot make sense, people?",
    ):
        assert "check_in" in score_sentence(text)[1], text
    assert score_sentence("Why does it make sense to sort first?") == (1.0, ("mark", "wh"))
    assert score_sentence("Is that clear why the limit exists?") == (1.0, ("mark", "aux"))


def test_checking_the_room_is_a_check_in():
    """#70: seeing, clarity, following, being together, remembering, polls, a bare vocative."""
    for text in (
        "Do you see it?",
        "Okay, do you see the difference?",
        "You see the pole?",
        "Is everything clear?",
        "Are they clear, everyone?",
        "How clear have I been?",
        "Are you following?",
        "Did you get the point?",
        "We are together?",
        "So, how are we?",
        "Do you remember this?",
        "Do you have that in mind?",
        "How many of you have seen a Bode plot before?",
        "Everyone?",
    ):
        assert "check_in" in score_sentence(text)[1], text


def test_checking_the_room_leaves_content_alone():
    """#70: content after the verb, a wh-word before it, or a vocative alone keep a question."""
    for text in (
        "Do you see why the area is finite?",
        "Do you see any pole on the imaginary axis?",
        "What do you see here?",
        "How are we going to compute this?",
        "Is it clear that f is continuous?",
        "What do I get, everyone?",
    ):
        assert "check_in" not in score_sentence(text)[1], text
    assert score_sentence("Right, you see?") == (0.2, ("mark", "tag"))


def test_check_in_phrases_match_whole_words():
    assert "check_in" not in score_sentence("Is your namespace global?")[1]


def test_scores_are_rounded():
    assert score_sentence("Okay?") == (0.2, ("mark", "tag"))


def test_head_word_found_past_a_preposition_before_a_wh_word():
    """#78: "For which age ...?" opens on "which"."""
    sentence = "For which age are we guaranteed that starting here we end up here?"
    assert score_sentence(sentence) == (1.0, ("mark", "wh"))
    assert question_type("In what sense is it stable?") == "what"
    assert score_sentence("In the limit it is stable?")[1] == ("mark",)  # no wh-word after it


def test_head_word_found_in_an_earlier_comma_clause():
    """#78: an inverted auxiliary mid-sentence opens the question."""
    sentence = "This area here, is it going to be finite, infinite, or you do not know?"
    assert score_sentence(sentence) == (1.0, ("mark", "aux"))
    assert question_type(sentence) == "yes_no"


def test_an_earlier_clause_needs_the_mark_and_an_auxiliary():
    # without the '?' the clause walk stays off
    assert score_sentence("This area here, is it finite, or not.")[1] == ()
    # a wh-word mid-sentence usually opens a relative clause, so only the last clause counts
    assert score_sentence("The answer here, what would you answer, people?")[1] == ("mark",)


@pytest.mark.parametrize(
    "text",
    (
        "So how do we do that?",
        "So what did it do?",
        "So exactly what, was that right?",
        "What's that?",
        "What’s that?",
        "This, is it that, or what?",
    ),
)
def test_framing_questions_without_content_words_are_penalised(text: str) -> None:
    score, reasons = score_sentence(text)
    assert "framing" in reasons
    assert "mark" in reasons
    assert score == 0.6


@pytest.mark.parametrize(
    "text",
    (
        "What is a field?",
        "What could happen?",
        "Can it do that or can it not do that?",
        "Why is this not a field, right?",
        "The field, what is it?",
        "If I am times 0.5, what am I doing?",
        "This area here, is it going to be finite, infinite, or you do not know?",
        "Can it be?",
        "This one, what is it?",
        "How much?",
        "What do I get?",
        "Can't it do that?",
        "Can’t it do that?",
    ),
)
def test_framing_penalty_leaves_content_questions_alone(text: str) -> None:
    score, reasons = score_sentence(text)
    assert "framing" not in reasons
    assert score == 1.0

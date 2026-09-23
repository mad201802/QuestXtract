from questxtract.detect import detect_questions, question_type, score_sentence


def test_detects_marked_and_unmarked_questions(sentences):
    extra = list(sentences) + [
        type(sentences[0])(
            text="Does anyone know why that matters",
            word_start=15,
            word_end=20,
            start=9.0,
            end=12.0,
        )
    ]
    found = detect_questions(extra)
    assert [c.text for c in found] == [
        "What is a stable sort?",
        "Does anyone know why that matters",
    ]
    assert found[1].score < found[0].score


def test_statements_are_not_questions(sentences):
    assert detect_questions([sentences[0]]) == ()


def test_threshold_filters_low_scores(sentences):
    unmarked = type(sentences[0])(
        text="Does anyone know why that matters", word_start=15, word_end=20, start=9.0, end=12.0
    )
    assert len(detect_questions([*sentences, unmarked], threshold=0.45)) == 2
    assert [c.text for c in detect_questions([*sentences, unmarked], threshold=0.6)] == [
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


def test_head_word_found_in_the_last_clause_of_a_question():
    reasons = score_sentence("If you have the data, does that mean it is better?")[1]
    assert reasons == ("mark", "aux")
    assert question_type("For AlphaTensor, is that fast?") == "yes_no"
    # Without the question mark a trailing clause is a relative clause, not a question.
    assert score_sentence("We take the limit, which is zero.") == (0.0, ())


def test_negated_auxiliaries_open_questions():
    assert score_sentence("Don't we have a different state space?")[1] == ("mark", "aux")
    assert score_sentence("Doesn’t the assumption make it harder?")[1] == ("mark", "aux")


def test_any_is_not_an_auxiliary():
    assert score_sentence("Any other questions?")[1] == ("mark",)

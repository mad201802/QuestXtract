from questxtract import Sentence
from questxtract.answer import adjacent_seed, pair_answers
from questxtract.detect import detect_questions


def _sentence(text: str, start: float, end: float, i: int) -> Sentence:
    return Sentence(text=text, word_start=i, word_end=i + 1, start=start, end=end)


def _socratic() -> tuple[Sentence, ...]:
    return (
        _sentence("Today we cover sorting.", 0.0, 2.0, 0),
        _sentence("What is a stable sort?", 2.2, 4.0, 1),
        _sentence("One that keeps equal elements in order.", 4.3, 7.0, 2),
    )


def test_the_socratic_answer_is_seeded():
    sentences = _socratic()
    (pair,) = pair_answers(detect_questions(sentences), sentences)
    assert pair.answer_text == "One that keeps equal elements in order."
    assert pair.answer_sentence_ids == (sentences[2].id,)
    assert pair.reasons == ("adjacent_seed",)
    assert (pair.start, pair.end) == (2.2, 4.0)  # the question's timestamps, not the answer's


def test_a_long_silence_is_not_a_seed(sentences):
    # The shared fixture pauses 2 s after its question.
    (pair,) = pair_answers(detect_questions(sentences), sentences)
    assert pair.answer_text == "" and pair.reasons == ("no_answer_found",)
    assert pair_answers(detect_questions(sentences), sentences, max_seed_gap=2.0)[0].answer_text


def test_a_following_question_is_not_a_seed():
    question, other = _socratic()[1], _sentence("Or is it about memory?", 4.3, 6.0, 2)
    assert adjacent_seed(question, other) is None


def test_the_last_sentence_has_no_seed():
    question = _socratic()[1]
    (pair,) = pair_answers(detect_questions([question]), [question])
    assert pair.answer_text == ""


def test_check_ins_and_tags_are_not_seeded():
    for text in ("Is that clear?", "Okay?"):
        sentences = (
            _sentence(text, 0.0, 1.0, 0),
            _sentence("So the next step is merging.", 1.2, 3.0, 1),
        )
        pairs = pair_answers(detect_questions(sentences), sentences)
        assert all(p.answer_text == "" for p in pairs), text

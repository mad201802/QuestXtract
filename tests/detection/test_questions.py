from questxtract import detect_questions, score_sentence


def test_detects_marked_and_unmarked_questions(segments):
    questions = detect_questions(segments)
    assert [q.question for q in questions] == [
        "What is a stable sort?",
        "Does anyone know why that matters",
    ]
    assert questions[0].start == 0.0
    assert questions[1].confidence < questions[0].confidence


def test_statements_are_not_questions(segments):
    assert detect_questions([segments[1]]) == []


def test_threshold_filters_low_confidence(segments):
    assert len(detect_questions(segments, threshold=0.9)) == 1


def test_non_ascii_first_word_is_not_skipped():
    assert score_sentence("Кто is here") == (0.0, ())
    assert score_sentence("Äh, is that right") == (0.0, ())


def test_score_is_always_a_float():
    confidence, _ = score_sentence("hello there")
    assert confidence == 0.0 and isinstance(confidence, float)

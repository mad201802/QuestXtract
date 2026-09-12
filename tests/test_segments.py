from questxtract import Sentence
from questxtract.segments import build_segments


def _s(text, start, end):
    return Sentence(text=text, word_start=0, word_end=1, start=start, end=end)


def test_silence_gap_closes_a_segment():
    out = build_segments([_s("a.", 0.0, 1.0), _s("b.", 5.0, 6.0)], max_gap=2.0)
    assert [(s.sentence_start, s.sentence_end) for s in out] == [(0, 1), (1, 2)]


def test_duration_closes_a_segment():
    out = build_segments([_s("a.", 0.0, 1.0), _s("b.", 1.0, 95.0)], max_duration=90.0)
    assert len(out) == 2


def test_char_budget_closes_a_segment():
    long = _s("x" * 500 + ".", 0.0, 1.0)
    out = build_segments([long, _s("b.", 1.0, 2.0), _s("c.", 2.0, 3.0)], max_chars=400)
    assert len(out) == 2


def test_one_segment_when_nothing_breaks(sentences):
    out = build_segments(sentences)
    assert len(out) == 1
    assert out[0].start == sentences[0].start and out[0].end == sentences[-1].end


def test_segment_index_is_stamped_back_onto_sentences(context):
    from questxtract.segments import segments
    from questxtract.sentences import sentences as sentences_stage

    ctx = segments(sentences_stage(context))
    assert {s.segment_index for s in ctx.sentences} == {0}
    assert ctx.stats["segments"] == 1


def test_no_sentences_yields_no_segments():
    assert build_segments([]) == ()

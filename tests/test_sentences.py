from questxtract import Transcript, Word
from questxtract.sentences import split_sentences


def _t(spec):
    return Transcript(words=tuple(Word(text=t, start=s, end=e) for t, s, e in spec))


def test_splits_on_terminal_punctuation(sentences):
    assert [s.text for s in sentences] == [
        "Today we cover sorting.",
        "What is a stable sort?",
        "It keeps equal elements in order.",
    ]


def test_trailing_run_on_without_punctuation_is_kept(transcript):
    spec = [("does", 0.0, 0.3), ("anyone", 0.3, 0.8), ("know", 0.8, 1.2)]
    assert [s.text for s in split_sentences(_t(spec))] == ["does anyone know"]


def test_force_splits_a_long_run_at_the_widest_gap():
    spec = [(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(8)]
    spec[4] = ("w4", 3.0, 3.05)  # a 2.6 s hole after w3
    spec[5:] = [(f"w{i}", 3.0 + (i - 4) * 0.1, 3.05 + (i - 4) * 0.1) for i in range(5, 8)]
    out = split_sentences(_t(spec), max_words=5, min_gap=0.6)
    assert len(out) == 2
    assert out[0].text.endswith("w3") and out[1].text.startswith("w4")


def test_no_force_split_without_a_wide_enough_gap():
    spec = [(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(8)]
    assert len(split_sentences(_t(spec), max_words=5, min_gap=0.6)) == 1


def test_empty_transcript_yields_no_sentences():
    assert split_sentences(Transcript()) == ()

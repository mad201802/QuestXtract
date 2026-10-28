import dataclasses

import pytest

from questxtract import Config, Context, QAPair, QuestionCandidate
from questxtract.score import confidence_label, score, score_pairs


def _pair(score: float, answer: str = "", **kwargs) -> QAPair:
    question = QuestionCandidate(sentence_id="s_1", text="Why?", score=score)
    return QAPair(
        question=question, answer_text=answer, confidence=0.0, start=1.0, end=2.0, **kwargs
    )


@pytest.mark.parametrize(
    ("confidence", "label"),
    [(1.0, "high"), (0.75, "high"), (0.74, "medium"), (0.45, "medium"), (0.44, "low"), (0, "low")],
)
def test_confidence_label_buckets_at_the_documented_edges(confidence, label):
    assert confidence_label(confidence) == label


def test_an_answered_pair_keeps_the_question_score():
    (pair,) = score_pairs([_pair(0.8, answer="Because.")])
    assert pair.confidence == 0.8


def test_an_unanswered_pair_is_weighed_down_but_still_emitted():
    (pair,) = score_pairs([_pair(0.9)])
    assert pair.confidence == 0.5  # round(0.9 * 0.55, 2)
    assert pair.answer_sentence_ids == ()


def test_without_weighing_answers_the_question_score_is_the_confidence():
    (pair,) = score_pairs([_pair(0.9)], weigh_answers=False)
    assert pair.confidence == 0.9


def test_pairs_below_min_confidence_are_dropped_and_the_threshold_is_inclusive():
    kept = score_pairs([_pair(0.4, answer="a"), _pair(0.39, answer="b")], min_confidence=0.4)
    assert [p.answer_text for p in kept] == ["a"]


@pytest.mark.parametrize(("raw", "clamped"), [(1.7, 1.0), (-0.3, 0.0)])
def test_confidence_is_clamped_to_zero_and_one(raw, clamped):
    (pair,) = score_pairs([_pair(raw, answer="a")], min_confidence=0.0)
    assert pair.confidence == clamped


def test_rescoring_keeps_every_other_field():
    before = _pair(0.8, answer="a", answer_sentence_ids=("s_2",), verdict="keep", reasons=("r",))
    (after,) = score_pairs([before])
    assert after == dataclasses.replace(before, confidence=0.8)


@pytest.mark.parametrize(("detector", "confidence"), [("rules", 0.5), ("llm", 0.9)])
def test_the_stage_weighs_answers_only_for_the_rules_detector(tmp_path, detector, confidence):
    ctx = Context(
        source=tmp_path / "lecture.mp4",
        config=Config(detector=detector, min_confidence=0.0),
        pairs=(_pair(0.9),),
    )
    out = score(ctx)
    assert [p.confidence for p in out.pairs] == [confidence]
    assert out.stats["emitted"] == 1


def test_the_stage_reads_min_confidence_from_the_config(tmp_path):
    config = Config(min_confidence=0.6)
    ctx = Context(source=tmp_path / "lecture.mp4", config=config, pairs=(_pair(0.9),))
    assert score(ctx).pairs == ()
    assert ctx.stats["emitted"] == 0

"""Tests for the evaluation harness (issue #12)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from questxtract import Config, Transcript
from questxtract.cli import main
from questxtract.evaluate import evaluate, match, overlap_ratio, overlaps
from questxtract.llm import JUDGE_PROMPT_VERSION
from questxtract.sentences import split_sentences


def test_overlap_is_measured_against_the_shorter_span() -> None:
    # A 3 s question inside a 30 s run-on sentence is fully covered.
    assert overlap_ratio((0.0, 30.0), (26.0, 29.0)) == 1.0
    assert overlaps((0.0, 2.0), (1.0, 5.0))  # 1 s of the shorter 2 s
    assert not overlaps((0.0, 2.0), (1.5, 5.0))
    assert not overlaps((0.0, 1.0), (1.0, 2.0))  # touching is not overlapping


def test_match_is_one_to_one_best_overlap_first() -> None:
    # One long prediction covers two labels: only the better-covered one is matched.
    assert match([(0.0, 10.0)], [(1.0, 3.0), (8.0, 12.0)]) == {0: 0}
    # Two predictions on one label: the one covering more of it wins.
    assert match([(0.0, 1.5), (1.0, 3.0)], [(1.0, 3.0)]) == {1: 0}


def _write(tmp_path: Path, labels: list[dict[str, object]]) -> Path:
    words = [
        ("What", 0.0, 0.5),
        ("is", 0.5, 0.7),
        ("a", 0.7, 0.8),
        ("field?", 0.8, 1.5),
        ("Okay,", 3.0, 3.4),
        ("so", 3.4, 3.6),
        ("why", 3.6, 3.9),
        ("do", 3.9, 4.0),
        ("we", 4.0, 4.2),
        ("make", 4.2, 4.4),
        ("that?", 4.4, 5.0),
        ("Is", 7.0, 7.2),
        ("it", 7.2, 7.4),
        ("closed", 7.4, 8.0),
        ("under", 8.0, 8.3),
        ("division", 8.3, 9.0),
        ("A", 11.0, 11.2),
        ("field", 11.2, 11.8),
        ("has", 11.8, 12.1),
        ("inverses.", 12.1, 13.0),
    ]
    transcript = {
        "segments": [
            {
                "text": " ".join(w for w, _, _ in words),
                "start": 0.0,
                "end": 13.0,
                "words": [{"word": w, "start": s, "end": e} for w, s, e in words],
            }
        ]
    }
    (tmp_path / "lec.transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    (tmp_path / "lec.labels.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in labels), encoding="utf-8"
    )
    return tmp_path


def test_evaluate_scores_content_and_breaks_down_false_positives(tmp_path: Path) -> None:
    labels = [
        {"start": 0.0, "end": 1.5, "text": "What is a field?", "kind": "content", "split": "blind"},
        {"start": 3.6, "end": 5.0, "text": "why do we make that?", "kind": "rhetorical"},
        {"start": 7.0, "end": 9.0, "text": "Is it closed under division", "kind": "content"},
    ]
    report = evaluate([_write(tmp_path, labels)])
    total = report["total"]
    assert (total["tp"], total["fp"], total["fn"]) == (1, 1, 1)
    assert total["recall"] == 0.5 and total["precision"] == 0.5
    assert total["false_positive_breakdown"] == {"rhetorical": 1}
    assert total["question_mark_rate"] == 0.5  # the unpunctuated one has no '?'
    assert total["blind_slice_recall"] == 1.0
    assert [x["text"] for x in total["false_negatives"]] == ["Is it closed under division"]
    assert report["recordings"]["lec"]["recall"] == 0.5
    assert report["sweep"][0]["threshold"] == 0.0 and report["sweep"][-1]["recall"] == 0.0


def test_evaluate_skips_labels_without_a_transcript(tmp_path: Path) -> None:
    _write(tmp_path, [{"start": 0.0, "end": 1.5, "text": "What is a field?", "kind": "content"}])
    (tmp_path / "other.labels.jsonl").write_text("", encoding="utf-8")
    report = evaluate([tmp_path])
    assert list(report["recordings"]) == ["lec"]
    assert report["skipped"] == [str(tmp_path / "other.labels.jsonl")]


def test_evaluate_rejects_a_directory_with_nothing_to_score(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no \\*.labels.jsonl"):
        evaluate([tmp_path])


def test_cli_eval_prints_table_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path, [{"start": 0.0, "end": 1.5, "text": "What is a field?", "kind": "content"}])
    assert main(["eval", str(tmp_path)]) == 0
    assert "TOTAL" in capsys.readouterr().out
    assert main(["eval", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total"]["recall"] == 1.0


def test_eval_scores_the_llm_detector_and_names_it(
    tmp_path: Path,
    fake_llm: Callable[..., Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#88: `eval --detector llm` runs the same harness over what the model picked."""
    monkeypatch.chdir(tmp_path)  # the LLM cache is cwd-relative
    labels = [{"start": 0.0, "end": 1.5, "text": "What is a field?", "kind": "content"}]
    transcript = Transcript.from_file(_write(tmp_path, labels) / "lec.transcript.json")
    question = split_sentences(transcript)[0]
    # The model calls the labelled question rhetorical, so recall drops to 0.
    verdict = {"reason": "", "kind": "rhetorical", "question": "", "answer_sentence_ids": []}
    client = fake_llm({question.id: {**verdict, "confidence": 0.9}})

    url = ["--llm-url", "http://localhost:11434/v1"]
    assert main(["eval", str(_write(tmp_path, labels)), "--detector", "llm", *url, "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["detector"] == f"llm {Config().judge_model} (prompt {JUDGE_PROMPT_VERSION})"
    assert report["total"]["recall"] == 0.0 and client.requests[0]["model"] == Config().judge_model
    assert evaluate([tmp_path])["detector"] == "rules"


def test_nfr2_recall_on_the_reference_recordings() -> None:
    # NFR2: question detection recall >= 50% on the team's labelled set, at the default
    # emission threshold. Runs on the committed transcripts, so no ASR and no network.
    report = evaluate([Path(__file__).parent / "data"])
    assert set(report["recordings"]) == {
        "mit_18_100b_l1",
        "ttk4225_2026_09_14",
        "ttk4225_2026_09_15",
    }
    assert report["total"]["recall"] >= 0.5


def test_nfr2_recall_on_the_held_out_recordings() -> None:
    # #92: the same gate on lecturers the rules were never written from. Reported, never tuned
    # on (tests/data/README.md).
    report = evaluate([Path(__file__).parent / "data" / "held_out"])
    assert set(report["recordings"]) == {"mit_6_003_l8"}
    assert report["total"]["recall"] >= 0.5

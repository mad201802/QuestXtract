"""Tests for the evaluation harness (issue #12)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from questxtract import Config, Transcript, cli
from questxtract.cli import main
from questxtract.evaluate import (
    KINDS,
    _check_frame,
    character_error_rate,
    evaluate,
    evaluate_vision,
    format_vision_report,
    formula_found,
    match,
    overlap_ratio,
    overlaps,
)
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


VISION = Path(__file__).parent / "data" / "vision"


@pytest.mark.parametrize(
    ("predicted", "reference", "rate"),
    [
        ("Stable sort", "Stable sort", 0.0),
        ("stable   SORT\n", "Stable sort", 0.0),  # case and spacing are not misreadings
        ("Stable sart", "Stable sort", 1 / 11),
        ("", "Stable sort", 1.0),
        ("Stable sort, and much more invented text", "Stable sort", 1.0),  # capped at 1
        ("", "", 0.0),
        ("x", "", 1.0),
    ],
)
def test_character_error_rate(predicted: str, reference: str, rate: float) -> None:
    assert character_error_rate(predicted, reference) == pytest.approx(rate)


@pytest.mark.parametrize(
    ("formula", "latex", "found"),
    [
        (r"\frac{1}{1+t^2}", r"\left( \dfrac{1}{1 + t^2} \right)", True),
        (r"\int_0^1 f \, dx", r"$\int_0^1 f\,dx$", True),
        (r"y = \sum w_i x_i", "a = b\ny = \\sum w_i x_i\nc = d", True),  # among others
        (r"y = \sum w_i x_i", r"y = \sum w_i x_{i}", True),  # a brace more is not a misreading
        (r"y = \sum w_i x_i", r"y = \prod v_j z_j", False),
        (r"x(t) = \sin(t)", None, False),
        (r"\leftarrow", r"\rightarrow", False),  # \left inside a command name is not spacing
        (r"\quad", None, True),  # nothing left to find once spacing is removed
        # #58 review: a short formula is matched exactly and never inside a longer token.
        ("x", r"\max_{i} a_i", False),
        ("n", r"x \in Q", False),  # the space after \in keeps it one token, apart from Q
        ("0", "100", False),
        ("x", r"a = x + 1", True),
        ("x^2", "x^3", False),
        (r"r_n \in F", r"r_n \in F", True),
    ],
)
def test_formula_found(formula: str, latex: str | None, found: bool) -> None:
    assert formula_found(formula, latex) is found


def _frames(folder: Path, name: str, labels: list[dict[str, Any]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (folder / label["frame"]).write_bytes(b"\x89PNG " + label["frame"].encode())
    path = folder / f"{name}.frames.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in labels), encoding="utf-8")
    return path


def _frame(frame: str, kind: str, text: str, latex: list[str]) -> dict[str, Any]:
    return {"frame": frame, "kind": kind, "text": text, "latex": latex, "diagram": None}


def _board(text: str, latex: str | None) -> dict[str, Any]:
    return {"description": "A board.", "text": text, "latex": latex, "confidence": 0.9}


def test_vision_accuracy_is_reported_per_lecture_type(
    tmp_path: Path, fake_llm: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #58: scored against a model, the labelled set gives accuracy per lecture type."""
    monkeypatch.chdir(tmp_path)  # the vision cache is cwd-relative
    _frames(
        tmp_path / "set",
        "lec",
        [
            _frame("lec_1.png", "slides", "Stable sort", ["x^2"]),
            _frame("lec_2.png", "blackboard", "f(c) = 0", ["f(c) = 0", "a < c < b"]),
            _frame("lec_3.png", "blackboard", "Not cont.", []),
            _frame("lec_4.png", "blackboard", "", []),  # blank boards
            _frame("lec_5.png", "blackboard", "", []),
            _frame("lec_6.png", "blackboard", "", []),
        ],
    )
    client = fake_llm(
        _board("Stable sort", "x^2"),
        _board("f(c) = O", "f(c) = 0"),
        TimeoutError("no answer"),  # the third frame cannot be read: an empty answer
        TimeoutError("no answer"),
        _board(" [unreadable] ", None),  # read as blank: unsure, but no invented text
        _board("Q.E.D.", None),  # not read as blank
        TimeoutError("no answer"),  # a failed reading is not a blank one
        TimeoutError("no answer"),
    )

    with pytest.warns(UserWarning, match="vision: keyframe left out"):
        report = evaluate_vision([tmp_path / "set"], Config(vision_model="qwen3-vl"))

    assert report["model"] == "qwen3-vl" and report["prompt"] == "v0"
    assert list(report["kinds"]) == ["slides", "blackboard"]  # in KINDS order
    assert report["kinds"]["slides"] == {
        "frames": 1,
        "failed": 0,
        "cer": 0.0,
        "blank": 0,
        "blank_read": 0,
        "formulas": 1,
        "formulas_found": 1,
        "formula_recall": 1.0,
    }
    board = report["kinds"]["blackboard"]
    assert board["frames"] == 5 and board["failed"] == 2
    # Blank boards are out of the CER mean, and counted on their own.
    assert board["cer"] == pytest.approx((0.125 + 1.0) / 2, abs=1e-3)
    assert (board["blank"], board["blank_read"]) == (3, 1)
    assert (board["formulas"], board["formulas_found"], board["formula_recall"]) == (2, 1, 0.5)
    assert report["total"]["frames"] == 6 and report["total"]["formula_recall"] == 0.667
    assert report["frames"][1]["missed"] == ["a < c < b"]
    assert report["frames"][3]["cer"] is None and report["frames"][3]["read_blank"] is True
    # The frame went to the model as the vision stage sends a keyframe when the layout probe
    # got its kind right: the label's kind in the prompt, no speech.
    system, user = client.requests[0]["messages"]
    assert "Frame kind: slides" in system["content"] and "-" in system["content"]
    assert "Frame kind: blackboard" in client.requests[1]["messages"][0]["content"]
    assert user["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")

    table = format_vision_report(report)
    assert "slides" in table and "blackboard" in table and "TOTAL" in table
    assert "FAILED" in table and "missed: a < c < b" in table
    assert "1/3" in table  # blank boards read as blank
    assert table.index("NOT blank") < table.index("lec_3.png")  # an invented reading first


def test_vision_eval_reads_each_frame_once(
    tmp_path: Path, fake_llm: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _frames(tmp_path, "lec", [_frame("lec_1.png", "handwriting", "h(t)", [])])
    client = fake_llm(_board("h(t)", None))
    first = evaluate_vision([tmp_path])
    assert evaluate_vision([tmp_path / "lec.frames.jsonl"]) == first
    assert len(client.requests) == 1  # the second run is answered from the cache


@pytest.mark.parametrize(
    ("label", "problem"),
    [
        ([], "not a JSON object"),
        (_frame("../x.png", "slides", "", []), "frame must be the name of a .png file"),
        (_frame("", "slides", "", []), "frame must be the name of a .png file"),
        (_frame("..", "slides", "", []), "frame must be the name of a .png file"),
        (_frame("a.png", "lecture", "", []), "kind must be one of"),
        ({**_frame("a.png", "slides", "", []), "text": None}, "text must be"),
        (_frame("a.png", "slides", "", "x^2"), "latex must be a list"),  # type: ignore[arg-type]
        (_frame("a.png", "slides", "", [" "]), "latex must be a list"),
        ({**_frame("a.png", "slides", "", []), "diagram": 3}, "diagram must be"),
    ],
)
def test_a_malformed_frame_label_names_its_line(label: Any, problem: str) -> None:
    with pytest.raises(ValueError, match=f"lec.frames.jsonl:2: {problem}"):
        _check_frame(label, "lec.frames.jsonl:2")


def test_vision_eval_rejects_a_missing_frame_or_an_empty_set(tmp_path: Path) -> None:
    path = _frames(tmp_path, "lec", [_frame("lec_1.png", "slides", "", [])])
    (tmp_path / "lec_1.png").unlink()
    with pytest.raises(FileNotFoundError, match="lec.frames.jsonl:1: no frame lec_1.png"):
        evaluate_vision([path])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no labelled frame"):
        evaluate_vision([tmp_path / "empty"])


def test_cli_eval_score_vision_prints_the_table_and_json(
    tmp_path: Path,
    fake_llm: Callable[..., Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _frames(tmp_path, "lec", [_frame("lec_1.png", "whiteboard", "MLPs", [])])
    client = fake_llm(_board("MLPs", None))
    flags = ["--llm-url", "http://localhost:11434/v1", "--vision-model", "qwen3-vl"]
    assert main(["eval", "--score-vision", str(tmp_path), *flags]) == 0
    out = capsys.readouterr().out
    assert "whiteboard" in out and "vision model qwen3-vl" in out
    assert main(["eval", "--score-vision", str(tmp_path), *flags, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["kinds"]["whiteboard"]["cer"] == 0.0
    # #58 review: --vision-model picks the model scored, not --llm-model (the judge's).
    assert client.requests[0]["model"] == "qwen3-vl"


def test_cli_eval_score_vision_needs_a_model_to_ask(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "--score-vision", str(tmp_path)]) == 1
    assert "eval --score-vision needs OPENAI_API_KEY" in capsys.readouterr().err


def test_eval_keeps_vision_enabled_from_the_settings_file(tmp_path: Path) -> None:
    """#58 review: the eval flag must not reach _config's mapping of --vision."""
    settings = tmp_path / "questxtract.toml"
    settings.write_text("vision_enabled = true\n", encoding="utf-8")
    for extra in ([], ["--score-vision"]):
        args = cli.build_parser().parse_args(["eval", "x", "--config", str(settings), *extra])
        assert cli._config(args).vision_enabled is True


def test_the_labelled_keyframes_cover_every_lecture_type() -> None:
    """AC #58: every keyframe is tagged, and every lecture type has at least 5 frames."""
    counts = dict.fromkeys(KINDS, 0)
    for path in sorted(VISION.glob("*.frames.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            label = json.loads(line)
            _check_frame(label, f"{path.name}:{number}")
            assert (VISION / label["frame"]).is_file(), label["frame"]
            assert label["frame"].startswith(path.name.removesuffix(".frames.jsonl") + "_")
            counts[label["kind"]] += 1
    assert all(n >= 5 for n in counts.values()), counts


def test_every_labelled_recording_has_its_source_in_the_readme() -> None:
    """AC #58: only shareable recordings, each listed with its source and licence."""
    readme = (VISION / "README.md").read_text(encoding="utf-8")
    for path in VISION.glob("*.frames.jsonl"):
        assert f"`{path.name.removesuffix('.frames.jsonl')}`" in readme, path.name

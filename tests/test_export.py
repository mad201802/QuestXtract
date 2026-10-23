import dataclasses
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from questxtract import (
    Config,
    Context,
    SchemaError,
    Transcript,
    VisualContext,
    read_questions,
    write_questions,
)
from questxtract.cli import main
from questxtract.evaluate import match
from questxtract.export import SCHEMA_PATH, _format_hms, result_to_dict, result_to_json
from questxtract.llm import JUDGE_PROMPT_VERSION
from questxtract.pipeline import STAGES, run
from questxtract.vision import attach_visual

QUESTION_KEYS = {
    "id",
    "text",
    "original_text",
    "start",
    "end",
    "start_hms",
    "end_hms",
    "confidence",
    "confidence_label",
    "question_type",
    "source",
    "answer",
    "segment_id",
    "sentence_id",
    "visual_ids",
    "verdict",
    "reasons",
    "review",
}

DATA = Path(__file__).parent / "data"

BOARDS = (
    VisualContext(start=0.0, end=5.0, keyframe_path="frames/1.jpg", description="stable sort"),
    VisualContext(start=5.0, end=20.0, keyframe_path="frames/2.jpg", latex=r"O(n \log n)"),
)


@pytest.fixture
def finished(context):
    return run(context, STAGES[2:])


def test_export_emits_a_fixed_key_set_with_nulls_kept(finished):
    data = result_to_dict(finished)
    assert set(data) == {
        "schema_version",
        "source",
        "config",
        "stats",
        "segments",
        "visuals",
        "questions",
    }
    assert data["config"]["min_confidence"] == finished.config.min_confidence
    # the rules detect and vision is off by default, so no model is named as if it ran
    assert data["config"]["judge_model"] is None and data["config"]["vision_model"] is None
    judged = dataclasses.replace(finished, config=finished.config.replace(detector="llm"))
    assert result_to_dict(judged)["config"]["judge_model"] == finished.config.judge_model
    question = data["questions"][0]
    assert set(question) == QUESTION_KEYS
    assert question["answer"] is None
    assert question["visual_ids"] == []
    assert question["original_text"] is None
    assert question["verdict"] == "unjudged"
    assert question["source"] == "audio"
    assert question["review"] == {"status": "pending", "text": None, "note": None, "metadata": {}}


def test_a_question_raised_by_the_board_is_combined(context):
    context.config = context.config.replace(board_overlap_delta=0.1)
    context.visual = (VisualContext(start=0.0, end=5.0, text="Stable sort: keeps equal keys"),)
    ctx = run(context, STAGES[2:])
    (question,) = result_to_dict(ctx)["questions"]
    assert "board_overlap" in question["reasons"]
    assert question["source"] == "combined"


def test_a_question_the_board_did_not_raise_stays_audio(context):
    context.visual = (VisualContext(start=0.0, end=5.0, text="Heaps"),)
    (question,) = result_to_dict(run(context, STAGES[2:]))["questions"]
    assert question["source"] == "audio"


def test_the_config_block_names_the_detector_and_nothing_that_did_not_run(finished):
    config = result_to_dict(finished)["config"]
    assert config["detector"] == "rules"
    assert config["judge_prompt_version"] is None and config["judge_reasoning_effort"] is None
    assert config["vision_reasoning_effort"] is None


def test_the_config_block_names_visions_reasoning_without_the_judge(finished):
    """#173 review: vision's requests carry a reasoning setting of their own."""
    config = finished.config.replace(vision_enabled=True, vision_reasoning_effort="low")
    exported = result_to_dict(dataclasses.replace(finished, config=config))["config"]
    assert (exported["judge_reasoning_effort"], exported["vision_reasoning_effort"]) == (
        None,
        "low",
    )


@pytest.mark.parametrize(("effort", "expected"), [("low", "low"), ("", "")])
def test_the_config_block_names_the_llm_prompt_and_reasoning(finished, effort, expected):
    config = finished.config.replace(detector="llm", reasoning_effort=effort)
    exported = result_to_dict(dataclasses.replace(finished, config=config))["config"]
    assert exported["detector"] == "llm"
    assert exported["judge_prompt_version"] == JUDGE_PROMPT_VERSION
    assert exported["judge_reasoning_effort"] == expected


def test_export_carries_the_timestamp_and_the_label(finished):
    question = result_to_dict(finished)["questions"][0]
    assert question["text"] == "What is a stable sort?"
    assert question["start"] == 2.2 and question["end"] == 4.0
    assert question["start_hms"] == "0:02" and question["end_hms"] == "0:04"
    assert question["confidence_label"] in {"high", "medium", "low"}
    assert question["question_type"] == "what"


def test_questions_point_into_the_exported_segments(finished):
    data = result_to_dict(finished)
    segment = data["segments"][data["questions"][0]["segment_id"]]
    assert segment == {"id": 0, "start": 0.0, "end": 9.0, "topic": None, "visual_ids": []}


def test_a_question_without_a_known_segment_points_at_none(finished):
    pair = finished.pairs[0]
    orphan = dataclasses.replace(pair.question, sentence_id="s_gone")
    finished.pairs = (dataclasses.replace(pair, question=orphan),)
    question = result_to_dict(finished)["questions"][0]
    assert question["segment_id"] is None and question["visual_ids"] == []


def test_board_states_are_listed_once_and_referenced_by_id(context):
    context.visual = BOARDS
    data = result_to_dict(run(context, STAGES[2:]))
    ids = [board.id for board in BOARDS]
    assert [v["id"] for v in data["visuals"]] == ids
    assert data["visuals"][1]["latex"] == r"O(n \log n)"
    assert data["segments"][0]["visual_ids"] == ids
    # The question (2.2-4.0 s) was asked while only the first board was up.
    assert data["questions"][0]["visual_ids"] == [BOARDS[0].id]


def test_a_question_points_at_the_board_on_screen_not_its_whole_segment(context):
    early = VisualContext(start=0.0, end=2.0, description="agenda")
    current = VisualContext(start=2.0, end=20.0, description="stable sort")
    context.visual = (early, current)
    data = result_to_dict(run(context, STAGES[2:]))
    assert data["segments"][0]["visual_ids"] == [early.id, current.id]
    assert data["questions"][0]["visual_ids"] == [current.id]


def test_a_question_with_no_board_on_screen_falls_back_to_its_segments(context):
    later = VisualContext(start=6.0, end=9.0, description="written after the question")
    context.visual = (later,)
    data = result_to_dict(run(context, STAGES[2:]))
    assert data["questions"][0]["visual_ids"] == [later.id]


def _jsonl(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_questions_link_to_the_drafted_board_state():
    # #135: questions of a real lecture, each with the board state on screen when it was asked
    # (tests/data/vision/README.md), against the committed board states: no video, no network.
    # The links were drafted from the board states' times and stay provisional until checked
    # against the video, so this pins the linking rule; it cannot catch a board state cut late.
    name = "ttk4225_2026_09_14"
    visual = json.loads((DATA / f"{name}.visual.json").read_text(encoding="utf-8"))
    states = tuple(VisualContext(**state) for state in visual["board_states"])
    by_start = {state.start: state for state in states}
    links = _jsonl(DATA / "vision" / f"{name}.links.jsonl")
    labels = _jsonl(DATA / f"{name}.labels.jsonl")
    spoken = {(x["start"], x["end"], x["text"]) for x in labels if x["kind"] == "content"}
    assert all(link["state_start"] in by_start for link in links)
    assert all((x["start"], x["end"], x["text"]) in spoken for x in links)
    transcript = Transcript.from_file(DATA / f"{name}.transcript.json")
    ctx = Context(
        source=Path(name), config=Config(min_confidence=0.0), transcript=transcript, visual=states
    )
    ctx.duration = transcript.duration
    questions = result_to_dict(run(ctx, STAGES[2:]))["questions"]
    spans = [(q["start"], q["end"]) for q in questions]
    pairs = match(spans, [(x["start"], x["end"]) for x in links])
    right = only = later = none = 0
    for i, j in pairs.items():
        ids, end = questions[i]["visual_ids"], questions[i]["end"]
        wanted = by_start[links[j]["state_start"]].id
        right += wanted in ids
        only += ids == [wanted]
        later += any(state.id in ids and state.start >= end for state in states)
        none += not ids
    print(
        f"{name}: {len(pairs)} of {len(links)} labelled questions detected; of those, "
        f"right state {right} (only that one {only}), also a later state {later}, none {none}"
    )
    assert len(pairs) >= 18  # of the 22 linked questions; fewer means detection or matching broke
    assert right == len(pairs)


def _answered_by_last_sentence(ctx):
    """Give the first question an answer spanning the transcript's last sentence."""
    later = ctx.sentences[-1]
    ctx.pairs = (
        dataclasses.replace(ctx.pairs[0], answer_text=later.text, answer_sentence_ids=(later.id,)),
    )
    return later


def test_an_answer_carries_the_span_of_its_own_sentences(finished):
    later = _answered_by_last_sentence(finished)
    question = result_to_dict(finished)["questions"][0]
    assert (question["start"], question["end"]) == (2.2, 4.0)
    assert (question["answer"]["start"], question["answer"]["end"]) == (later.start, later.end)
    assert question["answer"]["sentence_ids"] == [later.id]


def test_a_recording_with_no_questions_still_exports_the_full_envelope(context, tmp_path):
    """#13 AC5: no questions is a valid result, not an error or an empty file."""
    context.transcript = Transcript.from_dict(
        {
            "segments": [
                {"text": "Today we cover sorting. It keeps order.", "start": 0.0, "end": 5.0}
            ]
        }
    )
    out = tmp_path / "q.json"
    write_questions(run(context, STAGES[2:]), out)
    data = read_questions(out)
    assert data["questions"] == [] and data["stats"]["emitted"] == 0
    assert data["segments"]  # the lecture is still described


@pytest.fixture
def validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_every_export_shape_matches_the_published_schema(finished, validator):
    """#13 AC2 as a CI check: plain, with an answer and board states, and empty."""
    validator.validate(result_to_dict(finished))
    finished.visual = BOARDS
    finished = attach_visual(finished)
    _answered_by_last_sentence(finished)
    validator.validate(result_to_dict(finished))
    finished.pairs = ()
    validator.validate(result_to_dict(finished))
    judged = finished.config.replace(detector="llm", reasoning_effort="")
    validator.validate(result_to_dict(dataclasses.replace(finished, config=judged)))


def test_the_schema_requires_the_detector_settings(finished, validator):
    for key in ("detector", "judge_prompt_version", "judge_reasoning_effort"):
        data = result_to_dict(finished)
        del data["config"][key]
        with pytest.raises(ValidationError, match=key):
            validator.validate(data)


def test_the_schema_rejects_what_the_format_does_not_allow(finished, validator):
    data = result_to_dict(finished)
    data["questions"][0]["review"]["status"] = "maybe"
    with pytest.raises(ValidationError):
        validator.validate(data)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (2.2, "0:02"), (59.6, "1:00"), (90, "1:30"), (3661, "1:01:01")],
)
def test_format_hms_switches_to_hours_past_an_hour(seconds, expected):
    assert _format_hms(seconds) == expected


def test_write_and_read_questions(tmp_path, finished):
    out = tmp_path / "q.json"
    write_questions(finished, out)
    assert read_questions(out) == json.loads(out.read_text(encoding="utf-8"))


def _boards_in_cache(finished):
    """Give ``finished`` the two BOARDS, with real keyframe files in the cache under the cwd.

    Their ``keyframe_path`` is relative to the cache directory, as the vision stage stores it.
    """
    cache = Path(finished.config.cache_dir)
    boards = []
    for board in BOARDS:
        keyframe = Path("abc123", "frames", Path(board.keyframe_path).name)
        (cache / keyframe).parent.mkdir(parents=True, exist_ok=True)
        (cache / keyframe).write_bytes(f"image of {board.description or board.latex}".encode())
        boards.append(dataclasses.replace(board, keyframe_path=keyframe.as_posix()))
    finished.visual = tuple(boards)
    return attach_visual(finished)


def test_write_questions_copies_keyframes_next_to_the_file(tmp_path, finished, monkeypatch):
    """#64 AC: every keyframe is under out/, and its relative path opens from another cwd."""
    monkeypatch.chdir(tmp_path)
    finished = _boards_in_cache(finished)
    out = tmp_path / "out" / "questions.json"
    out.parent.mkdir()
    write_questions(finished, out)
    monkeypatch.chdir(tmp_path / ".questxtract")
    data = read_questions(out)
    for state, written in zip(finished.visual, data["visuals"], strict=True):
        assert written["keyframe_path"] == f"questions.keyframes/{state.id}.jpg"
        assert (out.parent / written["keyframe_path"]).read_bytes() == (
            tmp_path / ".questxtract" / state.keyframe_path
        ).read_bytes()
    # the ids stay the states' own, so segments and questions still point at them
    assert [v["id"] for v in data["visuals"]] == data["segments"][0]["visual_ids"]


def test_keyframes_are_found_through_the_cache_from_any_working_directory(
    tmp_path, finished, monkeypatch
):
    """#59: keyframe_path is relative to the cache, so a run reused from elsewhere still copies."""
    finished.config = finished.config.replace(cache_dir=str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    finished = _boards_in_cache(finished)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    out = tmp_path / "out" / "questions.json"
    out.parent.mkdir()
    write_questions(finished, out)
    assert all((out.parent / v["keyframe_path"]).is_file() for v in read_questions(out)["visuals"])


def test_two_lectures_in_one_folder_keep_their_keyframes_apart(tmp_path, finished, monkeypatch):
    """Each output gets its own keyframe folder, so equal image names never overwrite."""
    monkeypatch.chdir(tmp_path)
    finished = _boards_in_cache(finished)
    write_questions(finished, tmp_path / "monday.questions.json")
    write_questions(finished, tmp_path / "tuesday.json")
    monday = read_questions(tmp_path / "monday.questions.json")["visuals"][0]["keyframe_path"]
    tuesday = read_questions(tmp_path / "tuesday.json")["visuals"][0]["keyframe_path"]
    assert monday.startswith("monday.keyframes/") and tuesday.startswith("tuesday.keyframes/")
    assert (tmp_path / monday).is_file() and (tmp_path / tuesday).is_file()


def test_board_states_survive_a_read_edit_write_round_trip(tmp_path, finished, monkeypatch):
    """#64 AC: editing a question's text and writing the file again leaves `visuals` alone."""
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "questions.json"
    write_questions(_boards_in_cache(finished), out)
    before = read_questions(out)
    edited = read_questions(out)
    edited["questions"][0]["text"] = "What does a stable sort keep in order?"
    out.write_text(json.dumps(edited, indent=2), encoding="utf-8")
    after = read_questions(out)
    assert after["visuals"] == before["visuals"]
    assert after["questions"][0]["visual_ids"] == before["questions"][0]["visual_ids"]


def test_written_board_states_match_the_schema(tmp_path, finished, monkeypatch, validator):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "questions.json"
    write_questions(_boards_in_cache(finished), out)
    validator.validate(read_questions(out))


def test_without_vision_the_written_file_is_the_plain_export(tmp_path, finished):
    """#64 AC: with vision off nothing is copied and the file is today's export."""
    out = tmp_path / "questions.json"
    write_questions(finished, out)
    assert out.read_text(encoding="utf-8") == result_to_json(finished)
    assert not (tmp_path / "keyframes").exists()


def test_printing_keeps_cache_paths_and_copies_nothing(tmp_path, finished, monkeypatch):
    """#64 AC: `-o -` has no folder for keyframes, so they keep their cache paths."""
    monkeypatch.chdir(tmp_path)
    finished = _boards_in_cache(finished)
    data = json.loads(result_to_json(finished))
    assert [v["keyframe_path"] for v in data["visuals"]] == [
        s.keyframe_path for s in finished.visual
    ]
    assert not (tmp_path / "keyframes").exists()


def test_a_missing_keyframe_fails_before_the_file_is_written(tmp_path, finished, monkeypatch):
    monkeypatch.chdir(tmp_path)
    finished = _boards_in_cache(finished)
    (Path(finished.config.cache_dir) / finished.visual[1].keyframe_path).unlink()
    out = tmp_path / "questions.json"
    with pytest.raises(FileNotFoundError, match=finished.visual[1].id):
        write_questions(finished, out)
    assert not out.exists()


def test_read_questions_accepts_a_newer_minor_and_refuses_another_major(tmp_path, finished):
    out = tmp_path / "q.json"
    data = result_to_dict(finished)
    data["schema_version"] = "2.7"
    out.write_text(json.dumps(data), encoding="utf-8")
    assert read_questions(out)["schema_version"] == "2.7"
    data["schema_version"] = "1.1"
    out.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SchemaError, match="reads 2.x"):
        read_questions(out)


def test_read_questions_accepts_an_older_minor_without_the_detector_settings(tmp_path, finished):
    out = tmp_path / "q.json"
    data = result_to_dict(finished)
    data["schema_version"] = "2.1"
    for key in ("detector", "judge_prompt_version", "judge_reasoning_effort"):
        del data["config"][key]
    out.write_text(json.dumps(data), encoding="utf-8")
    assert read_questions(out)["config"]["judge_model"] is None


@pytest.mark.parametrize(
    "content",
    ["not json", '{"questions": null}', "[]", '{"schema_version": "2.0", "questions": []}'],
)
def test_malformed_questions_file_raises_schema_error(tmp_path, content):
    path = tmp_path / "q.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SchemaError):
        read_questions(path)


def _transcript_file(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "text": "Today we cover sorting. What is a stable sort?",
                        "start": 0.0,
                        "end": 6.0,
                    },
                    {"text": "It keeps equal elements in order.", "start": 6.0, "end": 9.0},
                    {"text": "Does anyone know why that matters", "start": 9.0, "end": 12.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_roundtrip(tmp_path):
    out = tmp_path / "q.json"
    assert main(["process", str(_transcript_file(tmp_path)), "-o", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "2.4"
    # An aux head without a '?' scores 0.4, under the 0.45 detection gate, so it is never a
    # candidate (architecture.md 7.3).
    assert [q["text"] for q in data["questions"]] == ["What is a stable sort?"]
    assert data["stats"]["candidates"] == 1
    assert data["stats"]["sentences"] == 4


def test_cli_writes_to_stdout_with_a_dash_output(tmp_path, capsys):
    assert main(["process", str(_transcript_file(tmp_path)), "-o", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["questions"]


def test_cli_threshold_filters(tmp_path, capsys):
    assert main(["process", str(_transcript_file(tmp_path)), "-t", "0.2", "-o", "-"]) == 0
    assert len(json.loads(capsys.readouterr().out)["questions"]) == 1
    # Answered with no pause, so the full question score carries: 1.0.
    assert main(["process", str(_transcript_file(tmp_path)), "-t", "0.9", "-o", "-"]) == 0
    assert len(json.loads(capsys.readouterr().out)["questions"]) == 1


def test_cli_reports_malformed_transcript_readably(tmp_path, capsys):
    path = tmp_path / "t.json"
    path.write_text('{"chunks": []}', encoding="utf-8")
    assert main(["process", str(path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err


def test_cli_reports_missing_file(tmp_path, capsys):
    assert main(["process", str(tmp_path / "missing.json")]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_unimplemented_subcommands_fail_readably(tmp_path, capsys):
    assert main(["eval", str(tmp_path)]) == 1
    assert capsys.readouterr().err.startswith("error: ")
    assert main(["quiz", str(tmp_path / "q.json")]) == 1
    assert capsys.readouterr().err.startswith("error: ")


@pytest.mark.parametrize("threshold", ["1.1", "-0.1", "nan"])
def test_cli_rejects_a_threshold_outside_0_to_1(tmp_path, capsys, threshold):
    """#85 AC1: exit 1, one line naming the range, no file written."""
    out = tmp_path / "out" / "q.json"
    args = ["process", str(_transcript_file(tmp_path)), "-t", threshold, "-o", str(out)]
    assert main(args) == 1
    err = capsys.readouterr().err
    assert "between 0 and 1" in err and len(err.strip().splitlines()) == 1
    assert not out.parent.exists()


@pytest.mark.parametrize("threshold", ["0", "1"])
def test_cli_accepts_the_threshold_bounds(tmp_path, validator, threshold):
    """#85 AC2."""
    out = tmp_path / "q.json"
    assert main(["process", str(_transcript_file(tmp_path)), "-t", threshold, "-o", str(out)]) == 0
    validator.validate(read_questions(out))

import dataclasses
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import openai
import pytest
from jsonschema import Draft202012Validator

from questxtract import endpoint
from questxtract.cli import main
from questxtract.config import Config
from questxtract.models import Sentence, Transcript
from questxtract.quiz import QUIZ_PROMPT_VERSION, build_quiz, quiz_to_json
from questxtract.quiz_schema import QUIZ_SCHEMA_PATH, read_quiz, validate_quiz

DATA = Path(__file__).parent / "data"


def _answer(
    options: int = 3, *, stem: str = "Is sin(t) absolutely integrable?", **over: str
) -> dict[str, Any]:
    """A usable model answer for an item with ``options`` options."""
    answer: dict[str, Any] = {
        "quiz_material": True,
        "stem": stem,
        "correct_answer": "No, the integral diverges",
        "correct_explanation": "Every period adds the same area.",
        "explanation": "Absolute integrability needs a finite area under |x(t)|.",
    }
    for n in range(1, options):
        answer[f"distractor_{n}"] = f"Wrong option {n}"
        answer[f"why_wrong_{n}"] = f"Misconception {n}."
    answer.update(over)
    return answer


def _question(
    qid: str, text: str = "Is sin(t) absolutely integrable?", **over: Any
) -> dict[str, Any]:
    question: dict[str, Any] = {
        "id": qid,
        "text": text,
        "original_text": None,
        "start": 10.0,
        "end": 12.5,
        "answer": {"text": "No, it is not.", "start": 13.0, "end": 14.0, "sentence_ids": ["s_1"]},
        "visual_ids": [],
        "reasons": ["mark", "wh"],
        "review": {"status": "pending", "text": None, "note": None, "metadata": {}},
    }
    question.update(over)
    return question


def _document(
    *questions: dict[str, Any], visuals: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": "2.1",
        "config": {"language": "en"},
        "segments": [],
        "visuals": visuals or [],
        "questions": list(questions),
    }


def _build(document: dict[str, Any], config: Config | None = None) -> Any:
    return build_quiz(
        document,
        config or Config(),
        questions_name="lecture.questions.json",
        questions_hash="ab" * 32,
    )


def _assert_valid(quiz: dict[str, Any], questions: dict[str, Any]) -> None:
    """The quiz passes the shipped JSON Schema and every cross-field rule of ``validate_quiz``."""
    schema = json.loads(QUIZ_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = [e.message for e in Draft202012Validator(schema).iter_errors(quiz)]
    assert errors == []
    validate_quiz(quiz, questions)


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


@pytest.fixture
def questions_file(tmp_path: Path) -> Path:
    """A questions.json the real pipeline wrote from a committed transcript."""
    out = tmp_path / "lecture.questions.json"
    assert main(["process", str(DATA / "ttk4225_2026_09_14.transcript.json"), "-o", str(out)]) == 0
    return out


def test_lenient_quiz_has_one_item_per_question_that_is_quiz_material(
    questions_file: Path, fake_llm: Callable[..., Any], key: None
) -> None:
    data = json.loads(questions_file.read_text(encoding="utf-8"))
    data["questions"][0]["review"]["status"] = "deleted"
    data["questions"][1]["reasons"].append("check_in")
    material = [q["id"] for q in data["questions"][2:] if "check_in" not in q["reasons"]]
    fake_llm(_answer())

    result = _build(data, Config(quiz_strictness="lenient"))

    quiz = result.quiz
    assert [item["provenance"]["source_question_id"] for item in quiz["questions"]] == material
    _assert_valid(quiz, data)
    assert set(quiz) == {
        "schema_version",
        "id",
        "title",
        "language",
        "status",
        "source",
        "instructions",
        "generation",
        "questions",
    }
    assert set(quiz["questions"][0]) == {
        "id",
        "prompt",
        "interaction_type",
        "points",
        "options",
        "correct_response",
        "solution",
        "provenance",
        "review",
    }
    assert quiz["generation"] == {
        "method": "llm",
        "model": "gpt-4o",
        "prompt_version": QUIZ_PROMPT_VERSION,
    }
    assert data["questions"][0]["id"] in result.skipped and result.failed == ()


def test_strict_keeps_only_answered_questions_and_never_more_than_lenient(
    fake_llm: Callable[..., Any], key: None
) -> None:
    document = _document(_question("q_a"), _question("q_b", answer=None), _question("q_c"))
    fake_llm(_answer())

    strict = _build(document, Config(quiz_strictness="strict")).quiz["questions"]
    lenient = _build(document).quiz["questions"]  # lenient is the default

    assert [i["provenance"]["source_question_id"] for i in strict] == ["q_a", "q_c"]
    assert len(strict) <= len(lenient) == 3


def test_every_item_has_the_options_one_correct_and_an_explanation_each(
    fake_llm: Callable[..., Any], key: None
) -> None:
    fake_llm(_answer(4))

    items = _build(_document(_question("q_a"), _question("q_b")), Config(quiz_options=4)).quiz[
        "questions"
    ]

    for item in items:
        ids = [option["id"] for option in item["options"]]
        assert len(ids) == 4 == len(set(ids))
        assert len(item["correct_response"]["option_ids"]) == 1
        assert item["correct_response"]["option_ids"][0] in ids
        assert set(item["solution"]["option_explanations"]) == set(ids)
        correct = next(
            o for o in item["options"] if o["id"] == item["correct_response"]["option_ids"][0]
        )
        assert correct["text"]["text"] == "No, the integral diverges"


@pytest.mark.parametrize("options", ["2", "6"])
def test_options_out_of_range_are_rejected_before_any_request(
    options: str,
    questions_file: Path,
    fake_llm: Callable[..., Any],
    key: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = fake_llm(_answer())

    assert main(["quiz", str(questions_file), "--options", options]) == 1

    assert "--options must be between 3 and 5" in capsys.readouterr().err and client.requests == []


def test_the_board_on_screen_goes_into_the_request(fake_llm: Callable[..., Any], key: None) -> None:
    board = {
        "id": "v_1",
        "description": "a sine wave",
        "text": "x(t) = sin(t)",
        "latex": r"\int |\sin t|\,dt",
    }
    client = fake_llm(_answer())

    _build(_document(_question("q_a", visual_ids=["v_1"]), _question("q_b"), visuals=[board]))

    with_board, without = (r["messages"][1]["content"] for r in client.requests)
    assert "x(t) = sin(t)" in with_board and r"\int |\sin t|\,dt" in with_board
    assert "Board: nothing was linked" in without and "sin(t)" in without  # the question itself


def test_generated_text_is_marked_and_every_item_is_pending(
    fake_llm: Callable[..., Any], key: None
) -> None:
    fake_llm(_answer(), _answer(stem="Is the signal sin(t) absolutely integrable over all t?"))

    document = _document(_question("q_a"), _question("q_b", text="Is sin(t) integrable?"))
    same, reworded = _build(document).quiz["questions"]

    assert same["prompt"] == {"text": "Is sin(t) absolutely integrable?", "origin": "spoken"}
    assert reworded["prompt"]["origin"] == "generated"
    for item in (same, reworded):
        assert {o["text"]["origin"] for o in item["options"]} == {"generated"}
        assert item["solution"]["explanation"]["origin"] == "generated"
        assert item["review"] == {"status": "pending", "note": None, "metadata": {}}


def test_a_stem_that_only_lost_its_question_mark_is_generated(
    fake_llm: Callable[..., Any], key: None
) -> None:
    fake_llm(
        _answer(stem="is sin(t)  absolutely integrable"),
        _answer(stem="IS COS(T) ABSOLUTELY INTEGRABLE?"),
    )
    document = _document(
        _question("q_a"), _question("q_b", text="Is cos(t) absolutely integrable?")
    )

    quiz = _build(document).quiz
    lost, case = quiz["questions"]

    assert lost["prompt"]["origin"] == "generated"  # punctuation counts, as in validate_quiz
    assert case["prompt"]["origin"] == "spoken"  # case and spacing do not
    _assert_valid(quiz, document)


def test_a_question_reworded_by_the_llm_detector_is_not_spoken(
    fake_llm: Callable[..., Any], key: None
) -> None:
    fake_llm(_answer(stem="Is sin(t) absolutely integrable?"))
    question = _question("q_a", original_text="so is sine absolutely integrable")

    (item,) = _build(_document(question)).quiz["questions"]

    assert item["prompt"]["origin"] == "generated"  # the spoken words were different


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps(_answer(distractor_1="No, the integral diverges.")),  # same as the correct one
        json.dumps({k: v for k, v in _answer().items() if k != "why_wrong_2"}),  # missing field
        TimeoutError("no answer"),  # a stall, twice
    ],
)
def test_one_bad_answer_leaves_that_item_out_and_keeps_the_rest(
    bad: Any, fake_llm: Callable[..., Any], key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(endpoint, "_create", _create_or_raise)
    fake_llm(bad, bad, _answer())

    result = _build(_document(_question("q_a"), _question("q_b"), _question("q_c")))

    assert [i["provenance"]["source_question_id"] for i in result.quiz["questions"]] == [
        "q_b",
        "q_c",
    ]
    assert [question_id for question_id, _ in result.failed] == ["q_a"]


def _create_or_raise(llm: Any, request: dict[str, Any], seconds: float) -> Any:
    """``endpoint._create`` without the deadline thread: a fake reply that is an exception is
    raised as is, so a TimeoutError counts as a stall."""
    return llm.chat.completions.create(**request)


def test_an_unreachable_endpoint_ends_the_quiz(fake_llm: Callable[..., Any], key: None) -> None:
    fake_llm(openai.OpenAIError("connection refused"))

    with pytest.raises(ConnectionError, match="connection refused"):
        _build(_document(_question("q_a"), _question("q_b")))


def test_the_same_input_gives_the_same_bytes_and_a_second_run_asks_nothing(
    fake_llm: Callable[..., Any], key: None
) -> None:
    client = fake_llm(_answer())
    document = _document(*(_question(f"q_{n}", text=f"Question number {n}?") for n in "abc"))

    first = quiz_to_json(_build(document).quiz)
    asked = len(client.requests)
    second = quiz_to_json(_build(document).quiz)

    assert first == second and asked == 3 and len(client.requests) == asked
    places = {i["correct_response"]["option_ids"][0] for i in json.loads(first)["questions"]}
    assert places <= {"A", "B", "C"}


def test_quiz_writes_next_to_the_input_and_can_print(
    questions_file: Path,
    fake_llm: Callable[..., Any],
    key: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_llm(_answer())

    assert main(["quiz", str(questions_file)]) == 0
    written = questions_file.with_name("lecture.quiz.json")
    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    assert read_quiz(written, questions)["title"] == "lecture"
    capsys.readouterr()

    assert main(["quiz", str(questions_file), "-o", "-"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(written.read_text(encoding="utf-8"))


def test_quiz_without_a_key_or_url_fails_before_any_request(
    questions_file: Path, fake_llm: Callable[..., Any], capsys: pytest.CaptureFixture[str]
) -> None:
    client = fake_llm(_answer())

    assert main(["quiz", str(questions_file)]) == 1

    assert "OPENAI_API_KEY" in capsys.readouterr().err and client.requests == []


def test_process_with_quiz_writes_both_files(
    tmp_path: Path, fake_llm: Callable[..., Any], key: None
) -> None:
    source = tmp_path / "lecture.transcript.json"
    source.write_bytes((DATA / "ttk4225_2026_09_14.transcript.json").read_bytes())
    fake_llm(_answer(4))

    assert main(["process", str(source), "--quiz", "--options", "4"]) == 0

    questions = json.loads((tmp_path / "lecture.questions.json").read_text(encoding="utf-8"))
    quiz = read_quiz(tmp_path / "lecture.quiz.json", questions)
    _assert_valid(quiz, questions)
    assert quiz["questions"] and all(len(i["options"]) == 4 for i in quiz["questions"])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--quiz", "-o", "-"], "drop -o -"),
        (["--strictness", "strict"], "need --quiz"),
        (["--options", "4"], "need --quiz"),
        (["--quiz-model", "x"], "need --quiz"),
    ],
)
def test_process_quiz_flags_fail_before_anything_runs(
    argv: list[str],
    message: str,
    tmp_path: Path,
    fake_llm: Callable[..., Any],
    key: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = fake_llm(_answer())
    source = tmp_path / "lecture.transcript.json"
    source.write_bytes((DATA / "ttk4225_2026_09_14.transcript.json").read_bytes())

    assert main(["process", str(source), *argv]) == 1

    assert message in capsys.readouterr().err
    assert not (tmp_path / "lecture.questions.json").exists() and client.requests == []


def test_process_quiz_keeps_the_questions_when_the_quiz_endpoint_fails(
    tmp_path: Path, fake_llm: Callable[..., Any], key: None, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "lecture.transcript.json"
    source.write_bytes((DATA / "ttk4225_2026_09_14.transcript.json").read_bytes())
    fake_llm(openai.OpenAIError("connection refused"))

    assert main(["process", str(source), "--quiz"]) == 1

    assert (tmp_path / "lecture.questions.json").is_file()
    assert not (tmp_path / "lecture.quiz.json").exists()
    assert "connection refused" in capsys.readouterr().err


@pytest.mark.parametrize("config", [Config(quiz_strictness="medium"), Config(quiz_options=6)])
def test_settings_out_of_range_are_rejected_by_the_library(config: Config) -> None:
    with pytest.raises(ValueError, match="quiz_"):
        _build(_document(_question("q_a")), config)


def test_a_bad_strictness_in_the_settings_file_fails_before_any_request(
    questions_file: Path,
    tmp_path: Path,
    fake_llm: Callable[..., Any],
    key: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "questxtract.toml").write_text('quiz_strictness = "medium"\n', encoding="utf-8")
    client = fake_llm(_answer())

    assert main(["quiz", str(questions_file)]) == 1

    assert "quiz_strictness" in capsys.readouterr().err and client.requests == []


def test_quiz_without_the_llm_extra_says_how_to_install_it(
    questions_file: Path,
    key: None,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from questxtract import cli

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)

    assert main(["quiz", str(questions_file)]) == 1
    assert "pip install 'questxtract[llm]'" in capsys.readouterr().err


def test_the_cli_names_each_question_left_out(
    questions_file: Path,
    fake_llm: Callable[..., Any],
    key: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_llm("not json", "not json", _answer())

    assert main(["quiz", str(questions_file)]) == 0

    err = capsys.readouterr().err
    assert err.count("warning: quiz: ") == 1 and "1 failed" in err


def test_a_board_id_missing_from_visuals_adds_nothing(
    fake_llm: Callable[..., Any], key: None
) -> None:
    client = fake_llm(_answer())

    _build(_document(_question("q_a", visual_ids=["v_gone"])))

    assert "Board: nothing was linked" in client.requests[0]["messages"][1]["content"]


def test_a_question_the_model_turns_down_is_left_out_and_not_asked_again(
    fake_llm: Callable[..., Any], key: None
) -> None:
    declined = {k: ("" if isinstance(v, str) else False) for k, v in _answer().items()}
    client = fake_llm(declined, _answer())
    document = _document(
        _question("q_room", text="What would you answer, people?"), _question("q_a")
    )

    first = _build(document)
    second = _build(document)

    assert [i["provenance"]["source_question_id"] for i in first.quiz["questions"]] == ["q_a"]
    assert "q_room" in first.skipped and first.failed == ()
    assert len(client.requests) == 2 and second.quiz == first.quiz  # the decline is cached


def test_an_answer_without_the_quiz_material_flag_is_not_usable(
    fake_llm: Callable[..., Any], key: None
) -> None:
    fake_llm({k: v for k, v in _answer().items() if k != "quiz_material"})

    result = _build(_document(_question("q_a")))

    assert result.quiz["questions"] == [] and [q for q, _ in result.failed] == ["q_a"]


def test_lenient_is_the_default_and_strict_keeps_only_answered_questions(
    questions_file: Path, fake_llm: Callable[..., Any], key: None
) -> None:
    fake_llm(_answer())
    data = json.loads(questions_file.read_text(encoding="utf-8"))
    material = {q["id"] for q in data["questions"] if "check_in" not in q["reasons"]}
    answered = {q["id"] for q in data["questions"] if q["answer"] and q["id"] in material}
    written = questions_file.with_name("lecture.quiz.json")

    assert main(["quiz", str(questions_file)]) == 0
    default = {
        i["provenance"]["source_question_id"]
        for i in json.loads(written.read_text(encoding="utf-8"))["questions"]
    }
    assert main(["quiz", str(questions_file), "--strictness", "strict"]) == 0
    strict = {
        i["provenance"]["source_question_id"]
        for i in json.loads(written.read_text(encoding="utf-8"))["questions"]
    }

    assert default == material and strict == answered and answered < material


def _sentence(n: int, text: str | None = None) -> Sentence:
    return Sentence(
        id=f"s_{n}",
        text=text or f"Sentence {n}.",
        word_start=n,
        word_end=n + 1,
        start=n,
        end=n + 1.0,
    )


def test_what_was_said_around_the_question_goes_into_the_request(
    fake_llm: Callable[..., Any], key: None
) -> None:
    client = fake_llm(_answer())
    lecture = [_sentence(n) for n in range(10)]

    build_quiz(
        _document(_question("q_a", sentence_id="s_5")),
        Config(),
        questions_name="l.questions.json",
        questions_hash="ab" * 32,
        sentences=lecture,
    )

    content = client.requests[0]["messages"][1]["content"]
    assert ">> Sentence 5." in content
    assert all(f"Sentence {n}." in content for n in (1, 2, 3, 4, 6, 7))
    assert "Sentence 0." not in content and "Sentence 8." not in content


def test_long_speech_is_cut_far_from_the_question_never_the_question(
    fake_llm: Callable[..., Any], key: None
) -> None:
    client = fake_llm(_answer())
    lecture = [_sentence(n, f"Long sentence {n} " + "x" * 600) for n in range(10)]

    build_quiz(
        _document(_question("q_a", sentence_id="s_5")),
        Config(),
        questions_name="l.questions.json",
        questions_hash="ab" * 32,
        sentences=lecture,
    )

    content = client.requests[0]["messages"][1]["content"]
    assert ">> Long sentence 5 " in content and "Long sentence 1 " not in content


def _copy(source: Path, tmp_path: Path) -> Path:
    path = tmp_path / "lecture.transcript.json"
    path.write_bytes(source.read_bytes())
    return path


def test_quiz_takes_the_transcript_for_the_speech_around_each_question(
    tmp_path: Path, fake_llm: Callable[..., Any], key: None
) -> None:
    transcript = _copy(DATA / "ttk4225_2026_09_14.transcript.json", tmp_path)
    assert main(["process", str(transcript)]) == 0
    client = fake_llm(_answer())

    assert (
        main(["quiz", str(tmp_path / "lecture.questions.json"), "--transcript", str(transcript)])
        == 0
    )

    assert client.requests and all(">> " in r["messages"][1]["content"] for r in client.requests)


def test_without_a_transcript_the_quiz_says_so_and_goes_on(
    questions_file: Path,
    fake_llm: Callable[..., Any],
    key: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = fake_llm(_answer())

    assert main(["quiz", str(questions_file)]) == 0

    assert "pass --transcript" in capsys.readouterr().err
    assert "no transcript was given" in client.requests[0]["messages"][1]["content"]


def test_quiz_finds_the_cached_transcript_of_the_recording(
    questions_file: Path, tmp_path: Path, fake_llm: Callable[..., Any], key: None
) -> None:
    transcript = Transcript.from_file(DATA / "ttk4225_2026_09_14.transcript.json")
    entry = tmp_path / ".questxtract" / "f00d" / "transcribe.json"
    entry.parent.mkdir(parents=True)
    entry.write_text(
        json.dumps({"config": {}, "data": dataclasses.asdict(transcript)}), encoding="utf-8"
    )
    data = json.loads(questions_file.read_text(encoding="utf-8"))
    data["source"]["hash"] = "f00d"
    questions_file.write_text(json.dumps(data), encoding="utf-8")
    client = fake_llm(_answer())

    assert main(["quiz", str(questions_file)]) == 0

    assert all(">> " in r["messages"][1]["content"] for r in client.requests)


def test_quiz_with_a_missing_transcript_file_fails_before_any_request(
    questions_file: Path,
    fake_llm: Callable[..., Any],
    key: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = fake_llm(_answer())

    assert main(["quiz", str(questions_file), "--transcript", "nope.json"]) == 1

    assert "nope.json: no such file" in capsys.readouterr().err and client.requests == []


def test_process_quiz_sends_the_speech_it_already_has(
    tmp_path: Path, fake_llm: Callable[..., Any], key: None
) -> None:
    transcript = _copy(DATA / "ttk4225_2026_09_14.transcript.json", tmp_path)
    client = fake_llm(_answer())

    assert main(["process", str(transcript), "--quiz"]) == 0

    assert client.requests and all(">> " in r["messages"][1]["content"] for r in client.requests)


def test_a_written_quiz_exports_to_qti(
    tmp_path: Path, fake_llm: Callable[..., Any], key: None
) -> None:
    source = tmp_path / "lecture.transcript.json"
    source.write_bytes((DATA / "ttk4225_2026_09_14.transcript.json").read_bytes())
    fake_llm(_answer())
    assert main(["process", str(source), "--quiz"]) == 0
    quiz = json.loads((tmp_path / "lecture.quiz.json").read_text(encoding="utf-8"))

    package = tmp_path / "lecture.qti.zip"
    assert main(["qti", str(tmp_path / "lecture.quiz.json"), "-o", str(package)]) == 0

    with zipfile.ZipFile(package) as zipped:
        items = [n for n in zipped.namelist() if n.startswith("items/")]
    assert len(items) == len(quiz["questions"]) > 0


def test_a_transcript_of_another_recording_gives_a_note_and_no_context(
    questions_file: Path,
    fake_llm: Callable[..., Any],
    key: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = fake_llm(_answer())
    other = DATA / "ttk4225_2026_09_15.transcript.json"

    assert main(["quiz", str(questions_file), "--transcript", str(other)]) == 0

    assert "does not contain the sentences" in capsys.readouterr().err
    assert "no transcript was given" in client.requests[0]["messages"][1]["content"]


def test_the_quiz_counts_and_prints_its_model_usage(
    questions_file: Path,
    fake_llm: Callable[..., Any],
    key: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = fake_llm(_answer())
    data = json.loads(questions_file.read_text(encoding="utf-8"))

    result = _build(data)
    capsys.readouterr()
    assert main(["quiz", str(questions_file)]) == 0  # every answer is cached by now

    assert result.usage["quiz_requests"] == len(client.requests) > 0
    err = capsys.readouterr().err
    assert "left out" in err and "model requests" not in err  # cached answers cost nothing


def test_the_quiz_run_prints_its_model_requests(
    questions_file: Path,
    fake_llm: Callable[..., Any],
    key: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = fake_llm(_answer())

    assert main(["quiz", str(questions_file)]) == 0

    assert f"quiz: {len(client.requests)} model requests" in capsys.readouterr().err


def test_a_bad_quiz_setting_does_not_stop_a_run_without_the_quiz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "questxtract.toml").write_text("quiz_options = 6\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    transcript = _copy(DATA / "ttk4225_2026_09_14.transcript.json", tmp_path)

    assert main(["process", str(transcript)]) == 0
    capsys.readouterr()
    assert main(["process", str(transcript), "--quiz"]) == 1
    assert "quiz_options" in capsys.readouterr().err


def test_by_default_the_quiz_asks_gpt_4o_with_no_reasoning_setting(
    questions_file: Path, fake_llm: Callable[..., Any], key: None
) -> None:
    """#164 AC2: gpt-4o takes no reasoning_effort, so the default sends none."""
    client = fake_llm(_answer())

    assert main(["quiz", str(questions_file)]) == 0

    assert client.requests
    assert all(r["model"] == "gpt-4o" and "reasoning_effort" not in r for r in client.requests)


def test_a_reasoning_level_still_reaches_the_quiz_model(
    questions_file: Path, fake_llm: Callable[..., Any], key: None
) -> None:
    client = fake_llm(_answer())

    assert main(["quiz", str(questions_file), "--llm-reasoning", "low"]) == 0

    assert client.requests and all(r["reasoning_effort"] == "low" for r in client.requests)

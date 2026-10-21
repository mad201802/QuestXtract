"""Tests for the CLI: the progress bar (issue #14) and the commands themselves (issue #17).

The command tests never load a real Whisper model: `fake_asr` swaps a canned transcript in
for faster-whisper, while ingest still probes a real (tiny) MP4.
"""

from __future__ import annotations

import dataclasses
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import openai
import pytest
from jsonschema import Draft202012Validator

from questxtract import Transcript, asr, cli
from questxtract.cli import main
from questxtract.config import Config
from questxtract.export import SCHEMA_PATH
from questxtract.pipeline import STAGES
from questxtract.sentences import split_sentences

_SEGMENTS = [{"text": "Today we cover sorting. What is a stable sort?", "start": 0.0, "end": 3.0}]


@pytest.fixture
def fake_asr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Config]:
    """Swap faster-whisper for a canned transcript, and run inside ``tmp_path`` so the
    cwd-relative cache (``.questxtract/``) lands there rather than in the repo.

    Returns the configs the backend was called with: one entry per real transcription.
    """
    calls: list[Config] = []

    class _Backend:
        def transcribe(
            self,
            source: Path,
            config: Config,
            on_progress: Callable[[float], None] | None = None,
        ) -> list[dict[str, Any]]:
            del source
            calls.append(config)
            if on_progress:
                on_progress(0.5)
            return _SEGMENTS

    monkeypatch.setitem(asr._REGISTRY, "faster_whisper", _Backend)
    monkeypatch.chdir(tmp_path)
    return calls


def test_draw_renders_fraction_as_filled_blocks(capsys: pytest.CaptureFixture[str]) -> None:
    cli._draw(1, 3, "score", 0.7)
    err = capsys.readouterr().err
    assert err.startswith("\r[1/3]")
    assert cli._FULL * 7 + cli._EMPTY * 3 in err
    assert "70%" in err


def test_progress_bar_falls_back_to_a_title_cased_label_for_an_unknown_stage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = cli._progress_bar(1)
    report("some_new_stage", 0.0)
    assert "Some new stage" in capsys.readouterr().err


def test_progress_bar_completes_each_stage_before_advancing_to_the_next(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = cli._progress_bar(2)
    report("sentences", 0.0)
    report("sentences", 0.5)
    report("segments", 0.0)
    report("segments", 1.0)
    report("done", 1.0)

    err = capsys.readouterr().err
    first = err.index("[1/2] Splitting sentences")
    hundred = err.index("100%", first)
    second = err.index("[2/2] Grouping segments")
    # the first stage is redrawn to 100% before the bar ever shows the second stage's line
    assert first < hundred < second
    assert err.endswith("\n")  # "done" closes the redrawing line


def _transcript_file(tmp_path: Path) -> Path:
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps({"segments": [{"text": "Hi?", "start": 0.0, "end": 1.0}]}), encoding="utf-8"
    )
    return path


def test_process_closes_the_bar_line_before_an_error_propagates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage failing mid-run must not leave the bar's `\\r`-redrawn line dangling in
    front of the traceback."""

    def _boom(ctx: Any, stages: Any, progress: Callable[[str, float], None]) -> Any:
        del ctx, stages
        progress("sentences", 0.0)
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        main(["process", str(_transcript_file(tmp_path))])

    err = capsys.readouterr().err
    # `process` skips ingest and transcribe: the transcript is already in hand.
    assert f"[1/{len(STAGES) - 2}]" in err and "Splitting sentences" in err
    assert err.endswith("\n")


# --- the commands (issue #17) -------------------------------------------------


def _questions(path: Path) -> list[str]:
    return [q["text"] for q in json.loads(path.read_text(encoding="utf-8"))["questions"]]


def test_process_runs_every_stage_on_a_recording_and_writes_next_to_it(
    tmp_path: Path,
    make_mp4: Callable[..., Path],
    fake_asr: list[Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC1-3: a video in, the complete pipeline, questions.json out - with no flags at all."""
    video = make_mp4(tmp_path / "lecture.mp4")

    assert main(["process", str(video)]) == 0

    output = tmp_path / "lecture.questions.json"
    assert _questions(output) == ["What is a stable sort?"]
    assert json.loads(output.read_text(encoding="utf-8"))["source"]["hash"]  # ingest ran
    err = capsys.readouterr().err
    assert f"[{len(STAGES)}/{len(STAGES)}]" in err and "Transcribing" in err


def test_transcribe_writes_a_transcript_that_process_reads_back(
    tmp_path: Path, make_mp4: Callable[..., Path], fake_asr: list[Config]
) -> None:
    """The two halves of `process lecture.mp4`: only the ASR, then only the rest."""
    video = make_mp4(tmp_path / "lecture.mp4")

    assert main(["transcribe", str(video)]) == 0
    assert main(["process", str(tmp_path / "lecture.transcript.json")]) == 0

    # lecture.transcript.json defaults to lecture.questions.json, not ...transcript.questions...
    assert _questions(tmp_path / "lecture.questions.json") == ["What is a stable sort?"]


def test_transcribe_and_process_share_the_asr_cache(
    tmp_path: Path, make_mp4: Callable[..., Path], fake_asr: list[Config]
) -> None:
    """Whichever runs second on a recording skips transcription; so does a new threshold."""
    video = make_mp4(tmp_path / "lecture.mp4")

    assert main(["process", str(video)]) == 0
    assert main(["transcribe", str(video)]) == 0
    assert main(["process", str(video), "-t", "0.2"]) == 0

    assert len(fake_asr) == 1


def test_model_and_from_scratch_each_force_a_new_transcription(
    tmp_path: Path, make_mp4: Callable[..., Path], fake_asr: list[Config]
) -> None:
    video = make_mp4(tmp_path / "lecture.mp4")

    assert main(["transcribe", str(video)]) == 0
    assert main(["transcribe", str(video), "-m", "base"]) == 0
    assert main(["transcribe", str(video), "-m", "base", "--from-scratch"]) == 0

    assert [c.model_size for c in fake_asr] == [Config().model_size, "base", "base"]


def test_dash_prints_to_stdout_and_a_nested_output_directory_is_created(
    tmp_path: Path,
    make_mp4: Callable[..., Path],
    fake_asr: list[Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = make_mp4(tmp_path / "lecture.mp4")

    assert main(["transcribe", str(video), "-o", "-"]) == 0
    assert "asr_segments" in json.loads(capsys.readouterr().out)

    nested = tmp_path / "out" / "questions.json"
    assert main(["process", str(video), "-o", str(nested)]) == 0
    assert _questions(nested) == ["What is a stable sort?"]


@pytest.mark.parametrize(
    ("command", "name", "exists", "message"),
    [
        ("process", "lecture.mp4", False, "no such file"),
        (
            "process",
            "lecture.mkv",
            True,
            "expected an .mp4 recording or an .mp3 recording or a .json transcript",
        ),
        ("transcribe", "lecture.json", True, "expected an .mp4 recording"),
    ],
)
def test_bad_inputs_are_rejected_readably_before_any_stage_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    name: str,
    exists: bool,
    message: str,
) -> None:
    """AC4: one readable line, no traceback, and no half-drawn progress bar."""
    path = tmp_path / name
    if exists:
        path.write_text("{}", encoding="utf-8")

    assert main([command, str(path)]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ") and message in err
    assert "[1/" not in err


def test_a_recording_without_audio_fails_readably(
    tmp_path: Path, make_mp4: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    video = make_mp4(tmp_path / "lecture.mp4", audio=False)

    assert main(["process", str(video)]) == 1

    err = capsys.readouterr().err
    assert "\nerror: " in err and "no audio track" in err and "Traceback" not in err


def test_ctrl_c_closes_the_bar_line_and_exits_130(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _interrupt(ctx: Any, stages: Any, progress: Callable[[str, float], None]) -> Any:
        del ctx, stages
        progress("sentences", 0.3)
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run", _interrupt)

    assert main(["process", str(_transcript_file(tmp_path))]) == 130
    assert capsys.readouterr().err.endswith("\ninterrupted\n")


def test_help_documents_the_options_with_examples(capsys: pytest.CaptureFixture[str]) -> None:
    """AC5."""
    with pytest.raises(SystemExit) as top:
        main(["--help"])
    assert top.value.code == 0
    out = capsys.readouterr().out
    assert "questxtract transcribe lecture.mp4" in out
    assert "questxtract filter lecture.questions.json" in out

    with pytest.raises(SystemExit):
        main(["process", "--help"])
    out = capsys.readouterr().out
    assert "--threshold" in out and "--model" in out and "--from-scratch" in out
    assert "-o -" in out


# --- the LLM detector (#88) ---------------------------------------------------


def test_detector_llm_without_a_key_or_url_fails_before_transcription(
    tmp_path: Path,
    make_mp4: Callable[..., Path],
    fake_asr: list[Config],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    video = make_mp4(tmp_path / "lecture.mp4")

    assert main(["process", str(video), "--detector", "llm"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ") and "OPENAI_API_KEY" in err and "--llm-url" in err
    assert fake_asr == [] and "[1/" not in err


def test_vision_without_the_extra_fails_before_transcription(
    tmp_path: Path,
    make_mp4: Callable[..., Path],
    fake_asr: list[Config],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Jesper on PR #128: not a traceback after an hour of transcription and sampling."""
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    Path("questxtract.toml").write_text("vision_enabled = true", encoding="utf-8")
    video = make_mp4(tmp_path / "lecture.mp4", audio=False)  # has a video track

    assert main(["process", str(video)]) == 1

    err = capsys.readouterr().err
    assert "vision_enabled needs the LLM extra" in err and fake_asr == []


def test_vision_without_a_key_or_url_fails_before_transcription(
    tmp_path: Path,
    make_mp4: Callable[..., Path],
    fake_asr: list[Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    Path("questxtract.toml").write_text("vision_enabled = true", encoding="utf-8")
    video = make_mp4(tmp_path / "lecture.mp4", audio=False)

    assert main(["process", str(video)]) == 1

    err = capsys.readouterr().err
    assert "QUESTXTRACT_VISION_API_KEY" in err and "vision_base_url" in err and fake_asr == []
    assert "OPENAI_API_KEY" not in err  # vision does not need the judge's key


def test_vision_on_an_mp3_needs_neither_the_extra_nor_a_key(
    tmp_path: Path,
    make_mp3: Callable[..., Path],
    fake_asr: list[Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No video track, no board to read: the setting in the file is not an error here."""
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    Path("questxtract.toml").write_text("vision_enabled = true", encoding="utf-8")

    assert main(["process", str(make_mp3(tmp_path / "lecture.mp3"))]) == 0
    assert len(fake_asr) == 1


def test_detector_llm_without_the_extra_says_how_to_install_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)

    assert main(["process", str(_transcript_file(tmp_path)), "--detector", "llm"]) == 1
    assert "pip install 'questxtract[llm]'" in capsys.readouterr().err


def test_detector_llm_uses_the_model_and_endpoint_given(
    tmp_path: Path, fake_llm: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # the LLM cache is cwd-relative
    source = _transcript_file(tmp_path)
    (sentence,) = split_sentences(Transcript.from_file(source))
    judged = {
        "reason": "",
        "kind": "content",
        "question": "",
        "answer_sentence_ids": [],
        "confidence": 1.0,
    }
    client = fake_llm({sentence.id: judged})
    url = "http://localhost:11434/v1"

    args = ["process", str(source), "--detector", "llm", "--llm-model", "qwen3.5:9b"]
    assert main([*args, "--llm-url", url]) == 0

    data = json.loads((tmp_path / "t.questions.json").read_text(encoding="utf-8"))
    assert data["config"]["judge_model"] == "qwen3.5:9b"
    assert [q["verdict"] for q in data["questions"]] == ["keep"]
    assert client.requests[0]["model"] == "qwen3.5:9b"


@pytest.mark.parametrize("key_source", ["environment", "dotenv"])
def test_detector_llm_error_does_not_reveal_api_key(
    tmp_path, fake_llm, capsys, monkeypatch, key_source
):
    monkeypatch.chdir(tmp_path)
    secret = "sk-fake-cli-error-secret"
    if key_source == "environment":
        monkeypatch.setenv("OPENAI_API_KEY", secret)
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        (tmp_path / ".env").write_text(f"OPENAI_API_KEY={secret}\n", encoding="utf-8")
    fake_llm(openai.OpenAIError(f"Invalid API key: {secret}"))
    source = _transcript_file(tmp_path)
    assert main(["process", str(source), "--detector", "llm"]) == 1
    output = capsys.readouterr()
    assert secret not in output.out + output.err
    assert "Invalid API key: [REDACTED]" in output.err
    assert not (tmp_path / "t.questions.json").exists()


def test_detector_llm_endpoint_failure_is_one_error_line(
    tmp_path: Path,
    fake_llm: Callable[..., Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No silent fallback to the rules, and no traceback."""
    monkeypatch.chdir(tmp_path)
    fake_llm(openai.OpenAIError("Connection error."))
    source = _transcript_file(tmp_path)
    url = "http://localhost:11434/v1"

    assert main(["process", str(source), "--detector", "llm", "--llm-url", url]) == 1

    err = capsys.readouterr().err
    assert f"\nerror: judge: {url}: Connection error." in err and "Traceback" not in err
    assert not (tmp_path / "t.questions.json").exists()


def test_llm_reasoning_reaches_the_request_and_allows_a_longer_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A thinking model takes minutes per request: any level but none raises the deadline."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    source = str(_transcript_file(tmp_path))

    def config(*flags: str) -> Config:
        args = cli.build_parser().parse_args(["process", source, "--detector", "llm", *flags])
        return cli._with_llm(cli._config(args))

    assert config().judge_reasoning_effort == ""  # gpt-4o takes no setting (#164)
    assert config().timeout_seconds == Config().timeout_seconds == 120
    assert config("--llm-reasoning", "none").timeout_seconds == 120
    thinking = config("--llm-reasoning", "low")
    assert thinking.judge_reasoning_effort == "low"
    with pytest.raises(SystemExit):  # a typo fails at once, not at the provider after ASR
        config("--llm-reasoning", "hgih")
    assert thinking.timeout_seconds == 600
    # gpt-4o rejects any reasoning_effort; "" sends none (tests/test_llm.py). Only a level
    # asked for gets the long timeout, so a stalled request is noticed after 2 minutes.
    server = config("--llm-reasoning", "server")
    assert server.judge_reasoning_effort == ""
    assert server.timeout_seconds == 120


# --- config files (#99) -------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "file_value", "flag", "override", "expected_file", "expected_flag"),
    [
        ("model_size", '"small"', "--model", "base", "small", "base"),
        ("min_confidence", "0.6", "--threshold", "0.2", 0.6, 0.2),
        ("detector", '"llm"', "--detector", "rules", "llm", "rules"),
        ("judge_model", '"file-model"', "--llm-model", "flag-model", "file-model", "flag-model"),
        (
            "base_url",
            '"http://file/v1"',
            "--llm-url",
            "http://flag/v1",
            "http://file/v1",
            "http://flag/v1",
        ),
        ("judge_reasoning_effort", '"low"', "--llm-reasoning", "server", "low", ""),
        ("from_scratch", "false", "--from-scratch", None, False, True),
    ],
)
def test_each_explicit_flag_wins_and_omitted_flags_keep_file_values(
    tmp_path, monkeypatch, field, file_value, flag, override, expected_file, expected_flag
):
    monkeypatch.chdir(tmp_path)
    Path("questxtract.toml").write_text(f"{field} = {file_value}", encoding="utf-8")
    parser = cli.build_parser()
    args = ["process", "lecture.mp4"]
    assert getattr(cli._config(parser.parse_args(args)), field) == expected_file
    flags = [flag] if override is None else [flag, override]
    assert getattr(cli._config(parser.parse_args([*args, *flags])), field) == expected_flag


@pytest.mark.parametrize("command", ["process", "transcribe", "eval"])
def test_all_three_commands_accept_an_explicit_config_path(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    Path("questxtract.toml").write_text('model_size = "small"', encoding="utf-8")
    Path("other.toml").write_text("judge_window = 7\nfrom_scratch = true", encoding="utf-8")
    args = cli.build_parser().parse_args([command, "input.mp4", "--config", "other.toml"])
    config = cli._config(args)
    assert config.judge_window == 7 and config.from_scratch
    assert config.model_size == Config().model_size


@pytest.mark.parametrize("command", ["process", "transcribe", "eval"])
@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("min_confidance = 0.6", "min_confidance"),
        ('min_confidence = "high"', "expected float, got string"),
        ("min_confidence = 1.5", "between 0 and 1"),
        ("min_confidence = nan", "between 0 and 1"),
        ('detector = "typo"', "detector must be rules or llm"),
        ('judge_reasoning_effort = "typo"', "judge_reasoning_effort must be"),
        ('api_key = "sk-do-not-echo"', "OPENAI_API_KEY"),
        ("model_size = @", "invalid TOML at line 1"),
    ],
)
def test_bad_file_fails_before_any_command_writes_output(
    tmp_path, monkeypatch, capsys, command, contents, message
):
    monkeypatch.chdir(tmp_path)
    Path("questxtract.toml").write_text(contents, encoding="utf-8")
    assert main([command, "missing.mp4"]) == 1
    err = capsys.readouterr().err
    assert message in err and len(err.splitlines()) == 1
    assert "sk-do-not-echo" not in err and "Traceback" not in err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["questxtract.toml"]


@pytest.mark.parametrize("command", ["process", "transcribe", "eval"])
def test_explicit_missing_config_reports_path(tmp_path, monkeypatch, capsys, command):
    monkeypatch.chdir(tmp_path)
    assert main([command, "input.mp4", "--config", "missing.toml"]) == 1
    err = capsys.readouterr().err
    assert "missing.toml" in err and len(err.splitlines()) == 1


def test_file_model_invalidates_asr_cache_and_file_from_scratch_is_honoured(
    tmp_path, make_mp4, fake_asr
):
    video = make_mp4(tmp_path / "lecture.mp4")
    path = Path("questxtract.toml")
    path.write_text('model_size = "small"', encoding="utf-8")
    assert main(["transcribe", str(video)]) == 0
    assert main(["process", str(video)]) == 0
    assert len(fake_asr) == 1
    path.write_text('model_size = "base"', encoding="utf-8")
    assert main(["transcribe", str(video)]) == 0
    path.write_text('model_size = "base"\nfrom_scratch = true', encoding="utf-8")
    assert main(["transcribe", str(video)]) == 0
    assert [c.model_size for c in fake_asr] == ["small", "base", "base"]


def test_file_enables_llm_and_flag_can_switch_it_off(tmp_path, monkeypatch, fake_llm):
    monkeypatch.chdir(tmp_path)
    source = _transcript_file(tmp_path)
    (sentence,) = split_sentences(Transcript.from_file(source))
    client = fake_llm(
        {
            sentence.id: {
                "reason": "",
                "kind": "content",
                "question": "",
                "answer_sentence_ids": [],
                "confidence": 1.0,
            }
        }
    )
    Path("questxtract.toml").write_text(
        'detector = "llm"\nbase_url = "http://localhost:11434/v1"\njudge_model = "file-model"',
        encoding="utf-8",
    )
    assert main(["process", str(source)]) == 0
    assert client.requests[0]["model"] == "file-model"
    client.requests.clear()
    assert main(["process", str(source), "--detector", "rules"]) == 0
    assert client.requests == []


@pytest.mark.parametrize("reasoning", ["low", "", "none"])
def test_eval_receives_file_settings_and_reasoning_timeout(tmp_path, monkeypatch, reasoning):
    monkeypatch.chdir(tmp_path)
    Path("questxtract.toml").write_text(
        f'judge_window = 7\njudge_reasoning_effort = "{reasoning}"', encoding="utf-8"
    )
    seen = []

    def evaluate(paths, config, board=False):
        seen.append(config)
        return {}

    monkeypatch.setattr(cli, "evaluate", evaluate)
    assert main(["eval", "unused", "--json"]) == 0
    assert seen[0].judge_window == 7
    assert seen[0].timeout_seconds == (600 if reasoning == "low" else 120)


@pytest.mark.parametrize("environment_key", [None, "sk-environment"])
def test_dotenv_key_reaches_client_but_not_output_or_cache(
    tmp_path, monkeypatch, fake_llm, environment_key
):
    from questxtract import endpoint

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if environment_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", environment_key)
    Path(".env").write_text('OPENAI_API_KEY="sk-dotenv"', encoding="utf-8")
    source = _transcript_file(tmp_path)
    (sentence,) = split_sentences(Transcript.from_file(source))
    real_client = endpoint.client
    fake = fake_llm(
        {
            sentence.id: {
                "reason": "",
                "kind": "content",
                "question": "",
                "answer_sentence_ids": [],
                "confidence": 1.0,
            }
        }
    )
    received = []

    def openai_client(**kwargs):
        received.append(kwargs["api_key"])
        return fake

    monkeypatch.setattr(openai, "OpenAI", openai_client)
    monkeypatch.setattr(endpoint, "client", real_client)
    assert main(["process", str(source), "--detector", "llm"]) == 0
    assert received == [environment_key or "sk-dotenv"]
    for path in tmp_path.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "sk-dotenv" not in text and "sk-environment" not in text


def test_process_and_transcribe_take_an_mp3(
    tmp_path: Path, make_mp3: Callable[..., Path], fake_asr: list[Config]
) -> None:
    """#117 AC: an audio-only lecture runs every stage; vision finds no board, no error."""
    audio = make_mp3(tmp_path / "lecture.mp3")

    assert main(["process", str(audio)]) == 0
    data = json.loads((tmp_path / "lecture.questions.json").read_text(encoding="utf-8"))
    assert [q["text"] for q in data["questions"]] == ["What is a stable sort?"]
    assert data["visuals"] == [] and data["source"]["path"].endswith("lecture.mp3")

    assert main(["transcribe", str(audio), "--from-scratch"]) == 0
    assert (tmp_path / "lecture.transcript.json").is_file()


def test_a_file_that_is_not_a_recording_gets_one_line_and_no_output(
    tmp_path: Path, fake_asr: list[Config], capsys: pytest.CaptureFixture[str]
) -> None:
    """#117 AC: not a media file at all: one readable line, nothing written."""
    fake = tmp_path / "lecture.mp3"
    fake.write_bytes(b"not really an mp3")
    assert main(["process", str(fake)]) == 1
    err = capsys.readouterr().err.strip()
    errors = [line for line in err.splitlines() if line.startswith("error:")]
    assert len(errors) == 1 and "invalid or unreadable recording" in errors[0]
    assert not (tmp_path / "lecture.questions.json").exists()


# --- vision flags (#61) -------------------------------------------------------

_BOARD = {
    "description": "A slide on sorting.",
    "text": "Stable sort",
    "latex": None,
    "confidence": 0.8,
}


def test_vision_flag_without_a_key_or_url_fails_before_transcription(
    tmp_path: Path,
    make_mp4: Callable[..., Path],
    fake_asr: list[Config],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    video = make_mp4(tmp_path / "lecture.mp4", video=True)

    assert main(["process", str(video), "--vision"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ") and "QUESTXTRACT_VISION_API_KEY" in err and "--llm-url" in err
    assert err.count("\n") == 1 and fake_asr == [] and "[1/" not in err


def test_vision_without_the_extra_says_how_to_install_it(
    tmp_path: Path,
    make_mp4: Callable[..., Path],
    fake_asr: list[Config],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    video = make_mp4(tmp_path / "lecture.mp4", video=True)

    assert main(["process", str(video), "--vision"]) == 1
    assert "pip install 'questxtract[llm]'" in capsys.readouterr().err and fake_asr == []


def test_vision_reads_the_board_with_the_model_given(
    tmp_path: Path,
    make_mp4: Callable[..., Path],
    fake_asr: list[Config],
    fake_llm: Callable[..., Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUESTXTRACT_VISION_API_KEY", "ntnu-test")
    client = fake_llm(_BOARD)
    video = make_mp4(tmp_path / "lecture.mp4", video=True)

    assert main(["process", str(video), "--vision", "--vision-model", "qwen3.8-27b"]) == 0

    assert "Reading the board" in capsys.readouterr().err
    data = json.loads((tmp_path / "lecture.questions.json").read_text(encoding="utf-8"))
    assert [v["text"] for v in data["visuals"]] == ["Stable sort"]
    assert data["questions"] and all(q["visual_ids"] for q in data["questions"])
    assert data["config"]["vision_model"] == "qwen3.8-27b"
    assert {request["model"] for request in client.requests} == {"qwen3.8-27b"}


def test_no_vision_turns_off_vision_from_the_settings_file(
    tmp_path: Path,
    make_mp4: Callable[..., Path],
    fake_asr: list[Config],
    fake_llm: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # off, so no key is needed
    Path("questxtract.toml").write_text("vision_enabled = true", encoding="utf-8")
    client = fake_llm(_BOARD)
    video = make_mp4(tmp_path / "lecture.mp4", video=True)

    assert main(["process", str(video), "--no-vision"]) == 0

    data = json.loads((tmp_path / "lecture.questions.json").read_text(encoding="utf-8"))
    assert data["visuals"] == [] and client.requests == []


@pytest.mark.parametrize("kind", ["mp3", "transcript"])
def test_vision_on_an_input_without_video_says_so_and_goes_on(
    tmp_path: Path,
    make_mp3: Callable[..., Path],
    fake_asr: list[Config],
    fake_llm: Callable[..., Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # no board to read, so no key needed
    client = fake_llm(_BOARD)
    source = make_mp3(tmp_path / "t.mp3") if kind == "mp3" else _transcript_file(tmp_path)

    assert main(["process", str(source), "--vision"]) == 0

    notes = [line for line in capsys.readouterr().err.splitlines() if line.startswith("note:")]
    assert notes == [f"note: {source.name} has no video, so the board is not read"]
    data = json.loads((tmp_path / "t.questions.json").read_text(encoding="utf-8"))
    assert data["config"]["vision_model"] is None and client.requests == []


def test_a_missing_input_file_is_reported_before_the_vision_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert main(["process", str(tmp_path / "typo.mp4"), "--vision"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_without_a_vision_flag_or_setting_vision_stays_off() -> None:
    config = cli._config(cli.build_parser().parse_args(["process", "lecture.mp4"]))
    assert not config.vision_enabled and config.vision_model == Config().vision_model


# Every vision setting has a decision: a `process` flag, or the settings file only. A new
# vision_* field in config.py fails this test until it is added to one of the two (and, for a
# flag, to the parser, cli._config, cli._with_vision if it needs checking, docs/cli.md and
# README.md). Each flag maps to the field it is listed with, and to a value it can set.
_VISION_FLAGS: dict[str, tuple[list[str], Any]] = {
    "vision_enabled": (["--vision"], True),
    "vision_model": (["--vision-model", "some-model"], "some-model"),
}
_VISION_SETTINGS_FILE_ONLY = {
    "vision_fps",
    "vision_max_states",
    "vision_change_fraction",
    "vision_change_hold",
    "vision_base_url",  # an endpoint is set once per machine; --llm-url still moves vision too
}


def test_every_vision_setting_has_a_flag_decision() -> None:
    vision_fields = {f.name for f in dataclasses.fields(Config) if f.name.startswith("vision_")}
    assert vision_fields == set(_VISION_FLAGS) | _VISION_SETTINGS_FILE_ONLY

    for field, (flag, value) in _VISION_FLAGS.items():
        config = cli._config(cli.build_parser().parse_args(["process", "lecture.mp4", *flag]))
        assert getattr(config, field) == value, flag


# --- filter (#136) --------------------------------------------------------------

_TTK = Path(__file__).parent / "data" / "ttk4225_2026_09_14.transcript.json"


@pytest.fixture
def questions_file(tmp_path: Path) -> Path:
    """The TTK4225 questions, with a board state (and its keyframe) on the first question."""
    path = tmp_path / "lecture.questions.json"
    assert main(["process", str(_TTK), "-o", str(path)]) == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    first = data["questions"][0]
    data["visuals"] = [
        {
            "id": "v_board",
            "start": first["start"],
            "end": first["end"],
            "keyframe_path": "lecture.keyframes/v_board.png",
            "description": "",
            "text": "Zyzzyva",
            "latex": None,
            "confidence": 0.9,
        }
    ]
    first["visual_ids"] = ["v_board"]
    (tmp_path / "lecture.keyframes").mkdir()
    (tmp_path / "lecture.keyframes" / "v_board.png").write_bytes(b"png")
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _filtered(capsys: pytest.CaptureFixture[str], *args: str) -> dict[str, Any]:
    assert main(["filter", *args]) == 0
    data: dict[str, Any] = json.loads(capsys.readouterr().out)
    return data


def test_filter_prints_the_questions_matching_the_board_by_default(
    questions_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    before = sorted(questions_file.parent.iterdir())
    data = _filtered(capsys, str(questions_file), "--text", "ZYZZYVA")
    first = json.loads(questions_file.read_text(encoding="utf-8"))["questions"][0]
    assert data["questions"] == [first]
    assert data["visuals"][0]["keyframe_path"] == "lecture.keyframes/v_board.png"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(data)
    assert sorted(questions_file.parent.iterdir()) == before  # nothing written


def test_filter_reads_times_as_seconds_or_minutes(
    questions_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    by_seconds = _filtered(capsys, str(questions_file), "--from", "300", "--to", "1200.5")
    by_clock = _filtered(capsys, str(questions_file), "--from", "5:00", "--to", "0:20:00.5")
    starts = [q["start"] for q in by_clock["questions"]]
    assert by_clock == by_seconds and starts
    assert all(300 <= start < 1200.5 for start in starts)


@pytest.mark.parametrize("bad", ["5x", "1:2:3:4", "5:60", "5:-1", ""])
def test_filter_rejects_a_time_it_cannot_read(
    questions_file: Path, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["filter", str(questions_file), "--from", bad])
    assert exit_info.value.code == 2
    assert "expected seconds, M:SS or H:MM:SS" in capsys.readouterr().err


def test_filter_needs_text_or_a_time(
    questions_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["filter", str(questions_file)]) == 1
    assert capsys.readouterr().err == "error: give --text, --from or --to\n"


def test_filter_into_another_folder_keeps_the_keyframes_found(
    questions_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out" / "board.json"
    assert main(["filter", str(questions_file), "--text", "zyzzyva", "-o", str(out)]) == 0
    total = len(json.loads(questions_file.read_text(encoding="utf-8"))["questions"])
    assert capsys.readouterr().err.endswith(f"1 of {total} questions -> {out}\n")
    keyframe = json.loads(out.read_text(encoding="utf-8"))["visuals"][0]["keyframe_path"]
    assert keyframe == "../lecture.keyframes/v_board.png"
    assert (out.parent / keyframe).is_file()

    beside = tmp_path / "board.json"
    assert main(["filter", str(questions_file), "--text", "zyzzyva", "-o", str(beside)]) == 0
    keyframe = json.loads(beside.read_text(encoding="utf-8"))["visuals"][0]["keyframe_path"]
    assert keyframe == "lecture.keyframes/v_board.png"


def test_filter_onto_another_drive_points_at_the_keyframe_absolutely(
    questions_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def other_drive(path: object, start: object = None) -> str:
        raise ValueError("path is on mount 'C:', start on mount 'D:'")

    monkeypatch.setattr(cli.os.path, "relpath", other_drive)
    out = tmp_path / "out" / "board.json"
    assert main(["filter", str(questions_file), "--text", "zyzzyva", "-o", str(out)]) == 0
    keyframe = Path(json.loads(out.read_text(encoding="utf-8"))["visuals"][0]["keyframe_path"])
    assert keyframe.is_absolute() and keyframe.is_file()


# --- qti (#67) ----------------------------------------------------------------


def test_qti_exports_a_quiz_and_leaves_rejected_items_out(
    tmp_path: Path, make_quiz: Callable[..., dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    quiz = make_quiz("q_aaa", "q_bbb")
    quiz["questions"][1]["review"]["status"] = "rejected"
    (tmp_path / "lecture.quiz.json").write_text(json.dumps(quiz), encoding="utf-8")

    assert main(["qti", "lecture.quiz.json", "-o", "lecture.zip"]) == 0

    assert capsys.readouterr().err == "QTI 3.0 package -> lecture.zip\n"
    with zipfile.ZipFile(tmp_path / "lecture.zip") as package:
        items = [name for name in package.namelist() if name.startswith("items/")]
        assert items == ["items/quiz_q_aaa.xml"]
        assert b"qti-choice-interaction" in package.read(items[0])


@pytest.mark.parametrize("content", [None, "{", '{"schema_version": "2.0", "questions": []}'])
def test_qti_bad_input_gets_one_line_and_no_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None
) -> None:
    source = tmp_path / "quiz.json"
    if content is not None:
        source.write_text(content, encoding="utf-8")
    output = tmp_path / "out.zip"

    assert main(["qti", str(source), "-o", str(output)]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: qti: ") and err.count("\n") == 1
    assert not output.exists()


def test_qti_has_no_version_or_filter_flags(capsys: pytest.CaptureFixture[str]) -> None:
    for flag in ("--version", "--only-keep"):
        with pytest.raises(SystemExit):
            main(["qti", "quiz.json", "-o", "out.zip", flag])
        assert f"unrecognized arguments: {flag}" in capsys.readouterr().err

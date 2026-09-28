"""Tests for the CLI: the progress bar (issue #14) and the commands themselves (issue #17).

The command tests never load a real Whisper model: `fake_asr` swaps a canned transcript in
for faster-whisper, while ingest still probes a real (tiny) MP4.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from questxtract import asr, cli
from questxtract.cli import main
from questxtract.config import Config
from questxtract.pipeline import STAGES

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
        ("process", "lecture.mkv", True, "expected an .mp4 recording or a .json transcript"),
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
    assert "questxtract transcribe lecture.mp4" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(["process", "--help"])
    out = capsys.readouterr().out
    assert "--threshold" in out and "--model" in out and "--from-scratch" in out
    assert "-o -" in out

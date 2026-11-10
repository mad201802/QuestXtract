"""CLI: ``process``, ``transcribe``, ``filter``, ``eval``, ``quiz`` and the three exports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NoReturn, TextIO

from . import __version__
from .config import (
    DEEPSEEK,
    EXERPLAZA_EMAIL_ENV,
    EXERPLAZA_PASSWORD_ENV,
    NTNU_KEY_ENV,
    NTNU_LLM_URL,
    OPENAI_KEY_ENV,
    OPENROUTER_KEY_ENV,
    QWEN,
    Config,
    is_ntnu,
    is_openai,
    is_openrouter,
    load_config,
    sends_key_unencrypted,
)
from .evaluate import evaluate, evaluate_vision, format_report, format_vision_report
from .exerplaza import ExerPlazaExportError, export_exerplaza
from .export import read_questions, result_to_json, write_questions
from .filtering import filter_document
from .models import Context, Progress, Sentence, Stage, Transcript
from .pipeline import STAGES, Cancelled, cached_transcript, run, stage_info, stoppable
from .qti import QtiExportError
from .qti_package import DEFAULT_QTI_VERSION, QTI_VERSIONS, export_qti
from .questions_tex import QuestionsTexError, export_questions_tex
from .quiz import (
    MAX_OPTIONS,
    MIN_OPTIONS,
    STRICTNESS,
    QuizResult,
    build_quiz,
    file_hash,
    quiz_to_json,
)
from .tex import TexExportError, export_tex
from .vision import has_video

PROCESS_INPUTS = (".mp4", ".mp3", ".json")
TRANSCRIBE_INPUTS = (".mp4", ".mp3")

FLAG_FIELDS: tuple[tuple[str, str], ...] = (
    ("model", "model_size"),
    ("from_scratch", "from_scratch"),
    ("threshold", "min_confidence"),
    ("detector", "detector"),
    ("llm_model", "judge_model"),
    ("llm_url", "base_url"),
    ("llm_reasoning", "reasoning_effort"),
    ("vision", "vision_enabled"),
    ("vision_model", "vision_model"),
    ("vision_reader", "vision_reader"),
    ("quiz_model", "quiz_model"),
    ("strictness", "quiz_strictness"),
    ("options", "quiz_options"),
)

_AFTER_TRANSCRIBE = 2
_MAX_WARNINGS = 5

_BAR_WIDTH = 10
_THINKING_TIMEOUT_SECONDS = 600
# A redirected stderr on Windows is cp1252, which cannot encode the block glyphs.
_FULL, _EMPTY = ("█", "░") if "utf" in (sys.stderr.encoding or "").lower() else ("#", "-")
_STAGE_LABELS = {
    "ingest": "Checking recording",
    "transcribe": "Transcribing",
    "sentences": "Splitting sentences",
    "segments": "Grouping segments",
    "layout": "Reading layout",
    "vision": "Reading the board",
    "attach_visual": "Joining visuals",
    "detect": "Detecting questions",
    "answer": "Finding answers",
    "score": "Scoring",
    "judge": "LLM detection",
    "quiz": "Writing the quiz",
}

_INPUT_KINDS = {
    ".mp4": "an .mp4 recording",
    ".mp3": "an .mp3 recording",
    ".json": "a .json transcript",
}

_EXAMPLES = """\
examples:
  questxtract process lecture.mp4               video -> lecture.questions.json
  questxtract process lecture.mp3               audio only, no board -> lecture.questions.json
  questxtract transcribe lecture.mp4            transcript only -> lecture.transcript.json
  questxtract process lecture.transcript.json   questions from an existing transcript
  questxtract filter lecture.questions.json --text eigenvalue
                                                the questions about a word, said or on the board

every command and option, with recipes:
  https://github.com/mad201802/QuestXtract/blob/main/docs/cli.md
"""

_PROCESS_EXAMPLES = """\
examples:
  questxtract process lecture.mp4                        writes lecture.questions.json
  questxtract process lecture.mp4 -o out/questions.json  choose where it goes
  questxtract process lecture.mp4 -t 0.3                 more questions, less strict
  questxtract process lecture.mp4 -m base                quick run with a small model
  questxtract process lecture.transcript.json -o -       existing transcript, print it
  questxtract process lecture.mp4 --detector llm         an LLM picks the questions
  questxtract process lecture.mp4 --detector llm         a local model in Ollama:
      --llm-url http://localhost:11434/v1 --llm-model qwen3.5:9b --llm-reasoning low
  questxtract process lecture.mp4 --vision               also read the slides or the board
                                                         (DeepSeek on NTNU's LLM service)
"""

_QUIZ_EXAMPLES = """\
examples:
  questxtract quiz lecture.questions.json                writes lecture.quiz.json
  questxtract quiz lecture.questions.json --strictness strict --options 4
  questxtract process lecture.mp4 --quiz                 questions and quiz in one run
"""

_TRANSCRIBE_EXAMPLES = """\
examples:
  questxtract transcribe lecture.mp4                     writes lecture.transcript.json
  questxtract transcribe lecture.mp4 -m small -o t.json  smaller model, custom path
"""

_EXERPLAZA_EXAMPLES = f"""\
examples:
  questxtract exerplaza lecture.quiz.json --url https://exerplaza.example.org \\
      --course-session 42 --content-unit "Improper integrals"
                                                         a quiz in course session 42
  questxtract exerplaza lecture.quiz.json --url http://localhost:54321 \\
      --course-session 1 --content-unit Sorting          a local ExerPlaza

the login: {EXERPLAZA_EMAIL_ENV} and {EXERPLAZA_PASSWORD_ENV}, in the environment or .env
"""

_FILTER_EXAMPLES = """\
examples:
  questxtract filter lecture.questions.json --text eigenvalue      print the matching questions
  questxtract filter lecture.questions.json --text "transfer function" -o tf.json
                                                         every word, spoken or on the board
  questxtract filter lecture.questions.json --from 5:00 --to 20:00   asked in that window
"""


def _progress_bar(total: int) -> Callable[[str, float], None]:
    index = 0
    current = ""
    last = ""

    def draw(name: str, fraction: float) -> None:
        nonlocal last
        line = _render(index, total, name, fraction)
        if line != last:
            last = line
            _draw(index, total, name, fraction)

    def report(name: str, fraction: float) -> None:
        nonlocal index, current
        if name == "done":
            if current:
                draw(current, 1.0)
            print(file=sys.stderr)
            return
        if name != current:
            if current:
                draw(current, 1.0)
                print(file=sys.stderr)
            index += 1
            current = name
        draw(name, fraction)

    return report


def _render(index: int, total: int, name: str, fraction: float) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * _BAR_WIDTH)
    bar = _FULL * filled + _EMPTY * (_BAR_WIDTH - filled)
    label = _STAGE_LABELS.get(name, name.replace("_", " ").capitalize())
    return f"[{index}/{total}] {label:<20} {bar} {fraction:>4.0%}"


def _draw(index: int, total: int, name: str, fraction: float) -> None:
    print(f"\r{_render(index, total, name, fraction)}", end="", file=sys.stderr, flush=True)


def check_input(path: Path, allowed: Sequence[str]) -> None:
    """Reject a bad input up front, not minutes into a transcription."""
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such file")
    if path.suffix.lower() not in allowed:
        expected = " or ".join(_INPUT_KINDS[suffix] for suffix in allowed)
        raise ValueError(f"{path}: expected {expected}")


def _output_path(output: str | None, source: Path, kind: str) -> Path | None:
    if output == "-":
        return None
    if output:
        path = Path(output)
    else:
        base = source.name
        for suffix in (".transcript.json", ".questions.json"):
            base = base.removesuffix(suffix)
        if base == source.name:
            base = source.stem
        path = source.with_name(f"{base}.{kind}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write(text: str, path: Path | None, summary: str) -> None:
    if path is None:
        print(text)
        return
    path.write_text(text, encoding="utf-8")
    print(f"{summary} -> {path}", file=sys.stderr)


def _config(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    changes = {
        field: value
        for flag, field in FLAG_FIELDS
        if (value := getattr(args, flag, None)) is not None
    }
    if changes.get("reasoning_effort") == "server":
        changes["reasoning_effort"] = ""
    config = config.replace(**changes)
    if not 0.0 <= config.min_confidence <= 1.0:
        raise ValueError("min_confidence / --threshold must be between 0 and 1")
    if config.detector not in ("rules", "llm"):
        raise ValueError("detector must be rules or llm")
    if config.vision_reader not in ("vlm", "ocr"):
        raise ValueError("vision_reader / --vision-reader must be vlm or ocr")
    efforts = ("", "none", "low", "medium", "high")
    if config.reasoning_effort not in efforts:
        raise ValueError("reasoning_effort must be empty, none, low, medium or high")
    if config.vision_reasoning_effort not in (None, *efforts):
        raise ValueError("vision_reasoning_effort must be empty, none, low, medium or high")
    return config


def _with_quiz(config: Config, what: str) -> None:
    if config.quiz_strictness not in STRICTNESS:
        raise ValueError("quiz_strictness / --strictness must be strict or lenient")
    if not MIN_OPTIONS <= config.quiz_options <= MAX_OPTIONS:
        raise ValueError(
            f"quiz_options / --options must be between {MIN_OPTIONS} and {MAX_OPTIONS}"
        )
    _check_endpoint(config, what, model_flag="--quiz-model")


def _with_llm(config: Config) -> Config:
    if {config.reasoning_effort, config.vision_effort} & {"low", "medium", "high"}:
        config = config.replace(
            timeout_seconds=max(config.timeout_seconds, _THINKING_TIMEOUT_SECONDS)
        )
    if config.detector == "llm":
        _check_endpoint(config, "--detector llm", model_flag="--llm-model")
    return config


_MODEL_FLAGS = {
    "--llm-model": "judge_model",
    "--quiz-model": "quiz_model",
    "--vision-model": "vision_model",
}


def _check_endpoint(config: Config, what: str, *, model_flag: str, vision: bool = False) -> None:
    if importlib.util.find_spec("openai") is None:
        raise ValueError(f"{what} needs the LLM extra: pip install 'questxtract[llm]'")
    url = config.vision_url if vision else config.base_url
    if sends_key_unencrypted(url):
        raise ValueError(f"{what}: {url} would send the key unencrypted; use https://")
    if vision and config.vision_base_url is not None:
        mover = "vision_base_url in questxtract.toml"
        to_openai = 'set vision_base_url = "openai" in questxtract.toml and pass'
    else:
        mover = "--llm-url (base_url)"
        to_openai = "pass --llm-url openai"
    reasoning = (
        ' and vision_reasoning_effort = "" in questxtract.toml'
        if vision
        else " --llm-reasoning server"
    )
    openai_hint = (
        f"for OpenAI instead, {to_openai} {model_flag} gpt-4o{reasoning}, with {OPENAI_KEY_ENV}"
    )
    model = getattr(config, _MODEL_FLAGS[model_flag])
    if is_ntnu(url) and "/" not in model:
        raise ValueError(
            f"{what}: {model} is not a model on NTNU's LLM service ({NTNU_LLM_URL}); {openai_hint}"
        )
    if is_openai(url) and "/" in model:
        raise ValueError(
            f"{what}: {model} is not an OpenAI model; pass {model_flag} with one, such as gpt-4o, "
            f"or {mover} for another endpoint"
        )
    if is_openrouter(url) and ("/" not in model or model in (QWEN, DEEPSEEK)):
        raise ValueError(
            f"{what}: {model} is not OpenRouter's name for a model; pass {model_flag} with one "
            "from https://openrouter.ai/models, such as deepseek/deepseek-v4.1-flash"
        )
    if config.vision_api_key if vision else config.api_key:
        return
    if is_ntnu(url):
        raise ValueError(
            f"{what} goes to NTNU's LLM service ({NTNU_LLM_URL}, NTNU network or VPN), so "
            f"it needs {NTNU_KEY_ENV} in the environment, in .env or saved in the app; "
            f"{openai_hint}"
        )
    if is_openai(url):
        raise ValueError(
            f"{what} goes to OpenAI, so it needs {OPENAI_KEY_ENV} in the environment, in .env "
            f"or saved in the app, or {mover} for another endpoint"
        )
    if is_openrouter(url):
        raise ValueError(
            f"{what} goes to OpenRouter, so it needs {OPENROUTER_KEY_ENV} in the environment, "
            f"in .env or saved in the app ({OPENAI_KEY_ENV} goes to OpenAI only), or {mover} "
            "for another endpoint"
        )


def _quiz_text(result: QuizResult) -> str:
    return quiz_to_json(result.quiz).rstrip("\n")


def _write_quiz(
    questions: Path,
    output: Path | None,
    config: Config,
    sentences: Sequence[Sentence] = (),
    document: dict[str, object] | None = None,
) -> None:
    bar = _progress_bar(1)
    result = _quiz_result(questions, config, sentences, bar, document)
    bar("done", 1.0)
    for question_id, reason in result.failed:
        print(f"warning: quiz: {question_id} left out: {reason}", file=sys.stderr)
    items = len(result.quiz["questions"])
    left_out = f" ({len(result.skipped)} left out, {len(result.failed)} failed)"
    _write(_quiz_text(result), output, f"{items} quiz items{left_out}")
    usage = result.usage
    if usage.get("quiz_requests"):
        print(
            f"quiz: {usage['quiz_requests']} model requests, "
            f"{usage.get('quiz_input_tokens', 0)} input and "
            f"{usage.get('quiz_output_tokens', 0)} output tokens, "
            f"{usage.get('quiz_ms', 0) / 1000:.1f} s",
            file=sys.stderr,
        )


def _quiz_sentences(
    document: dict[str, object], transcript: Path | None, config: Config
) -> tuple[Sentence, ...]:
    if transcript is not None:
        found: Transcript | None = Transcript.from_file(transcript)
    else:
        source = document.get("source")
        digest = source.get("hash") if isinstance(source, dict) else None
        found = cached_transcript(config, digest) if isinstance(digest, str) and digest else None
    if found is None:
        print(
            "note: no transcript for this lecture in the cache, so the quiz sees only the "
            "question, the answer and the board; pass --transcript for what was said around it",
            file=sys.stderr,
        )
        return ()
    ctx = Context(source=Path(""), config=config, transcript=found)
    sentences = run(ctx, STAGES[_AFTER_TRANSCRIBE : _AFTER_TRANSCRIBE + 1]).sentences
    ids = {s.id for s in sentences}
    asked = [q.get("sentence_id") for q in _questions(document)]
    if asked and not any(sentence_id in ids for sentence_id in asked):
        print(
            "note: the transcript does not contain the sentences of these questions (another "
            "recording, or transcribed with other settings), so the quiz sees only the question, "
            "the answer and the board",
            file=sys.stderr,
        )
        return ()
    return sentences


def _questions(document: dict[str, object]) -> list[dict[str, object]]:
    questions = document.get("questions")
    return [q for q in questions if isinstance(q, dict)] if isinstance(questions, list) else []


def _with_vision(config: Config, source: Path) -> Config:
    if not config.vision_enabled:
        return config
    if not has_video(source):
        print(f"note: {source.name} has no video, so the board is not read", file=sys.stderr)
        return config.replace(vision_enabled=False)
    _check_endpoint(config, "--vision / vision_enabled", model_flag="--vision-model", vision=True)
    return config


def _switched_on(stages: Sequence[Stage], config: Config) -> tuple[Stage, ...]:
    """Leave out the optional stages this run turns off, so the bar shows only real steps.

    Each stage still checks its own flag; this only keeps an offline run from listing
    "Reading the board" and "LLM detection".
    """
    off = set() if config.vision_enabled else {"layout", "vision", "attach_visual"}
    if config.detector != "llm":
        off.add("judge")
    return tuple(fn for fn in stages if stage_info(fn).name not in off)


def _run(
    ctx: Context,
    stages: Sequence[Stage],
    progress: Progress | None = None,
    stop: Callable[[], bool] | None = None,
) -> Context:
    if progress is not None:
        return run(ctx, stages, progress=progress, stop=stop)
    try:
        return run(ctx, stages, progress=_progress_bar(len(stages)), stop=stop)
    except BaseException:
        print(file=sys.stderr)
        raise


@dataclass(frozen=True, kw_only=True)
class RunResult:
    """What a finished ``process`` or ``transcribe`` run produced (issue #124)."""

    path: Path | None
    summary: str
    text: str | None = None
    warnings: tuple[str, ...] = ()


def _quiz_result(
    questions: Path,
    config: Config,
    sentences: Sequence[Sentence],
    progress: Progress | None = None,
    document: dict[str, object] | None = None,
) -> QuizResult:
    return build_quiz(
        document if document is not None else read_questions(questions),
        config,
        questions_name=questions.name,
        questions_hash=file_hash(questions),
        sentences=sentences,
        progress=progress,
    )


def _process_core(
    args: argparse.Namespace,
    progress: Progress | None,
    stop: Callable[[], bool] | None,
) -> tuple[Context, Config, Path | None]:
    config = _with_llm(_config(args))
    source = Path(args.input)
    check_input(source, PROCESS_INPUTS)
    if not args.quiz and any(
        getattr(args, flag) is not None for flag in ("quiz_model", "strictness", "options")
    ):
        raise ValueError("--quiz-model, --strictness and --options need --quiz")
    if args.quiz:
        if args.output == "-":
            raise ValueError("--quiz writes a file next to questions.json, so drop -o -")
        _with_quiz(config, "--quiz")
    output = _output_path(args.output, source, "questions")
    config = _with_vision(config, source)

    if source.suffix.lower() == ".json":
        transcript = Transcript.from_file(source)
        ctx = Context(source=source, config=config, transcript=transcript)
        ctx.duration = transcript.duration
        ctx = _run(ctx, _switched_on(STAGES[_AFTER_TRANSCRIBE:], config), progress, stop)
    else:
        ctx = _run(
            Context(source=source, config=config), _switched_on(STAGES, config), progress, stop
        )

    if stop is not None and stop():
        raise Cancelled("stopped before writing the questions")
    if output is not None:
        write_questions(ctx, output)
    return ctx, config, output


def _process(args: argparse.Namespace) -> int:
    ctx, config, output = _process_core(args, None, None)
    if output is None:
        print(result_to_json(ctx))
    else:
        print(f"{len(ctx.pairs)} questions -> {output}", file=sys.stderr)
        if args.quiz:
            _write_quiz(output, _output_path(None, output, "quiz"), config, ctx.sentences)
    return 0


def _process_run(
    args: argparse.Namespace,
    progress: Progress | None = None,
    stop: Callable[[], bool] | None = None,
) -> RunResult:
    ctx, config, output = _process_core(args, progress, stop)
    summary = f"{len(ctx.pairs)} questions"
    if output is None:
        return RunResult(path=None, summary=summary, text=result_to_json(ctx))
    if not args.quiz:
        return RunResult(path=output, summary=summary)
    if stop is not None:
        progress = stoppable(progress, stop)  # also stops between two quiz items
        if stop():
            raise Cancelled("stopped before the quiz")
    result = _quiz_result(output, config, ctx.sentences, progress)
    quiz_path = _output_path(None, output, "quiz")
    assert quiz_path is not None  # noqa: S101 - a file name is given, never stdout
    quiz_path.write_text(_quiz_text(result), encoding="utf-8")
    items = len(result.quiz["questions"])
    left_out = f" ({len(result.skipped)} left out, {len(result.failed)} failed)"
    warnings = tuple(f"quiz: {qid} left out: {reason}" for qid, reason in result.failed)
    return RunResult(path=quiz_path, summary=f"{items} quiz items{left_out}", warnings=warnings)


def _transcribe_core(
    args: argparse.Namespace,
    progress: Progress | None,
    stop: Callable[[], bool] | None,
) -> tuple[Context, Path | None, str]:
    config = _config(args)
    source = Path(args.recording)
    check_input(source, TRANSCRIBE_INPUTS)
    output = _output_path(args.output, source, "transcript")
    ctx = _run(Context(source=source, config=config), STAGES[:_AFTER_TRANSCRIBE], progress, stop)
    assert ctx.transcript is not None  # noqa: S101 - the transcribe stage always sets it
    text = json.dumps(ctx.transcript.to_dict(), indent=2, ensure_ascii=False)
    if stop is not None and stop():
        raise Cancelled("stopped before writing the transcript")
    if output is not None:
        output.write_text(text, encoding="utf-8")
    return ctx, output, text


def _transcribe(args: argparse.Namespace) -> int:
    ctx, output, text = _transcribe_core(args, None, None)
    assert ctx.transcript is not None  # noqa: S101 - set by the transcribe stage
    if output is None:
        print(text)
    else:
        print(f"{len(ctx.transcript.asr_segments)} segments -> {output}", file=sys.stderr)
    return 0


def _transcribe_run(
    args: argparse.Namespace,
    progress: Progress | None = None,
    stop: Callable[[], bool] | None = None,
) -> RunResult:
    ctx, output, text = _transcribe_core(args, progress, stop)
    assert ctx.transcript is not None  # noqa: S101 - set by the transcribe stage
    summary = f"{len(ctx.transcript.asr_segments)} segments"
    if output is None:
        return RunResult(path=None, summary=summary, text=text)
    return RunResult(path=output, summary=summary)


def execute(
    argv: Sequence[str],
    *,
    progress: Progress | None = None,
    stop: Callable[[], bool] | None = None,
) -> RunResult:
    """Run ``process``, ``transcribe`` or an export from its arguments, unprinted.

    Warnings the run raises, such as keyframes vision could not read, come back in
    :attr:`RunResult.warnings` instead of going to stderr.
    """
    args = parse_arguments(argv)
    # catch_warnings swaps process-wide state: fine, as one thread at a time runs a pipeline.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _execute(args, progress, stop)
    noted = tuple(dict.fromkeys(str(warning.message) for warning in caught))
    if not noted:
        return result
    if len(noted) > _MAX_WARNINGS:  # the app shows them under the result, one per line
        more = len(noted) - _MAX_WARNINGS
        noted = (*noted[:_MAX_WARNINGS], f"... and {more} more warning{'s' * (more > 1)}")
    return replace(result, warnings=(*result.warnings, *noted))


def _execute(
    args: argparse.Namespace,
    progress: Progress | None,
    stop: Callable[[], bool] | None,
) -> RunResult:
    if args.command == "process":
        return _process_run(args, progress, stop)
    if args.command == "transcribe":
        return _transcribe_run(args, progress, stop)
    if args.command in ("qti", "tex", "questions-tex"):
        return _export(args)
    raise ValueError(
        f"the app runs process, transcribe, qti, tex or questions-tex, not {args.command}"
    )


def _export(args: argparse.Namespace) -> RunResult:
    """Run ``qti``, ``tex`` or ``questions-tex``; QTI's warnings come back in the result."""
    if args.command == "qti":
        output = _export_path(args.output, Path(args.quiz), ".qti.zip")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                export_qti(args.quiz, output, version=args.qti_version)
            except QtiExportError as exc:
                raise ValueError(f"qti: {exc}") from None
        notes = tuple(f"qti: {warning.message}" for warning in caught)
        return RunResult(path=output, summary=f"QTI {args.qti_version} package", warnings=notes)
    if args.command == "tex":
        output = _export_path(args.output, Path(args.quiz), ".tex")
        try:
            count = export_tex(args.quiz, output, content_units=args.content_units)
        except TexExportError as exc:
            raise ValueError(f"tex: {exc}") from None
        return RunResult(path=output, summary=f"ExerPlaza TeX, {_questions_count(count)}")
    output = _questions_tex_path(args.output, Path(args.questions))
    try:
        count = export_questions_tex(args.questions, output)
    except QuestionsTexError as exc:
        raise ValueError(f"questions-tex: {exc}") from None
    return RunResult(path=output, summary=f"Questions TeX, {_questions_count(count)}")


def _questions_count(count: int) -> str:
    return f"{count} question" if count == 1 else f"{count} questions"


def _export_path(output: str | None, quiz: Path, suffix: str) -> Path:
    """Return ``-o``, else the file next to the quiz: ``lecture.quiz.json`` -> ``lecture<suffix>``.

    The app names its export files by the same rule (``gui.state.default_output``).
    """
    if output:
        return Path(output)
    base = quiz.name.removesuffix(".quiz.json")
    return quiz.with_name((quiz.stem if base == quiz.name else base) + suffix)


def _exported(args: argparse.Namespace) -> int:
    result = _export(args)
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"{result.summary} -> {result.path}", file=sys.stderr)
    return 0


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments without printing: a bad one raises ``ValueError``."""
    return build_parser(quiet=True).parse_args(list(argv))


def check_arguments(argv: Sequence[str]) -> None:
    """Raise the ``ValueError`` a ``process`` or ``transcribe`` run would raise, up front."""
    args = parse_arguments(argv)
    if args.command in ("process", "transcribe"):
        _config(args)


def _gui(args: argparse.Namespace) -> int:
    del args
    try:
        import tkinter  # noqa: F401 - imported to see whether this Python has Tk at all
    except ImportError:
        print(
            "error: this Python has no tkinter, so the graphical app cannot open; "
            "use a Python build with Tk (the python.org installers have it)",
            file=sys.stderr,
        )
        return 1
    try:
        import tkinterdnd2  # noqa: F401 - part of the gui extra, for drag and drop
    except ImportError:
        print(
            'error: the graphical app needs its extra: pip install "questxtract[gui]"',
            file=sys.stderr,
        )
        return 1
    from .gui.window import run_app

    return run_app()


def _time(value: str) -> float:
    try:
        parts = [float(part) for part in value.split(":")]
    except ValueError:
        parts = []
    if not 1 <= len(parts) <= 3 or not all(0 <= part < 60 for part in parts[1:]):
        raise argparse.ArgumentTypeError(f"expected seconds, M:SS or H:MM:SS, got {value!r}")
    return sum(part * 60.0**power for power, part in enumerate(reversed(parts)))


def _repoint_keyframes(data: dict[str, object], source: Path, output: Path) -> None:
    visuals = data.get("visuals")
    for visual in visuals if isinstance(visuals, list) else ():
        path = visual.get("keyframe_path") if isinstance(visual, dict) else None
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            continue
        image = source.parent / path
        try:
            visual["keyframe_path"] = Path(os.path.relpath(image, output.parent)).as_posix()
        except ValueError:
            visual["keyframe_path"] = str(image.resolve())


def _filter(args: argparse.Namespace) -> int:
    if args.text is None and args.start is None and args.end is None:
        raise ValueError("give --text, --from or --to")
    source = Path(args.questions)
    data = read_questions(source)
    kept = filter_document(data, start=args.start, end=args.end, text=args.text)
    output = None if args.output is None else _output_path(args.output, source, "questions")
    if output is not None:
        _repoint_keyframes(kept, source, output)
    summary = f"{len(kept['questions'])} of {len(data['questions'])} questions"
    _write(json.dumps(kept, indent=2, ensure_ascii=False), output, summary)
    return 0


def _eval(args: argparse.Namespace) -> int:
    try:
        config = _with_llm(_config(args))
        if args.score_vision:
            _check_endpoint(config, "eval --score-vision", model_flag="--vision-model", vision=True)
            report = evaluate_vision(args.paths, config)
        else:
            report = evaluate(args.paths, config, board=args.board)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_vision_report(report) if args.score_vision else format_report(report))
    return 0


def _quiz(args: argparse.Namespace) -> int:
    config = _with_llm(_config(args))
    questions = Path(args.questions)
    if not questions.is_file():
        raise FileNotFoundError(f"{questions}: no such file")
    if questions.suffix.lower() != ".json":
        raise ValueError(f"{questions}: expected a questions.json written by `process`")
    _with_quiz(config, "quiz")
    if args.transcript is not None and not Path(args.transcript).is_file():
        raise FileNotFoundError(f"{args.transcript}: no such file")
    document = read_questions(questions)
    sentences = _quiz_sentences(document, args.transcript, config)
    output = _output_path(args.output, questions, "quiz")
    _write_quiz(questions, output, config, sentences, document)
    return 0


def _questions_tex_path(output: str | None, questions: Path) -> Path:
    if output:
        return Path(output)
    name = questions.name
    return questions.with_name(
        f"{name.removesuffix('.json')}.tex" if name.endswith(".json") else f"{name}.tex"
    )


def _exerplaza(args: argparse.Namespace) -> int:
    """Send quiz.json to ExerPlaza's import endpoint."""
    try:
        result = export_exerplaza(
            args.quiz,
            url=args.url,
            course_session=args.course_session,
            content_units=args.content_units,
        )
    except ExerPlazaExportError as exc:
        print(f"error: exerplaza: {exc}", file=sys.stderr)
        return 1
    outcomes = [item.outcome for item in result.items]
    print(
        f"ExerPlaza quiz {result.quiz_uid} ({result.quiz_outcome}): "
        f"{outcomes.count('created')} created, {outcomes.count('updated')} updated, "
        f"{outcomes.count('skipped')} skipped",
        file=sys.stderr,
    )
    for item in result.items:
        if item.outcome == "skipped":
            print(f"  skipped {item.ref}: {item.reason or 'no reason given'}", file=sys.stderr)
    return 0


class _QuietParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


def build_parser(*, quiet: bool = False) -> argparse.ArgumentParser:
    """Build the argument parser; with ``quiet`` its errors raise ``ValueError``."""
    defaults = Config()
    raw = argparse.RawDescriptionHelpFormatter
    parser = (_QuietParser if quiet else argparse.ArgumentParser)(
        prog="questxtract",
        description="Find the questions asked in a lecture recording.",
        epilog=_EXAMPLES,
        formatter_class=raw,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subs = parser.add_subparsers(dest="command", required=True)

    configured = argparse.ArgumentParser(add_help=False)
    configured.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help="settings file (default: questxtract.toml in the current directory, if present)",
    )

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="where to write (default: next to the input; '-' prints to the terminal)",
    )
    shared.add_argument(
        "-m",
        "--model",
        default=None,
        metavar="NAME",
        help="Whisper model: tiny, base, small, medium, large-v3 or turbo (default: "
        f"{defaults.model_size}); smaller is faster and less accurate. Only used when transcribing",
    )
    shared.add_argument(
        "--from-scratch",
        action="store_true",
        default=None,
        help="ignore cached results and recompute everything, transcription included",
    )

    llm = argparse.ArgumentParser(add_help=False)
    llm.add_argument(
        "--detector",
        choices=("rules", "llm"),
        default=None,
        help="how questions are found: rules (fast, offline) or llm (a language model reads the "
        f"transcript; needs the questxtract[llm] extra) (default: {defaults.detector})",
    )
    llm.add_argument(
        "--llm-model",
        default=None,
        metavar="NAME",
        help="the model for --detector llm, e.g. qwen3.5:9b in Ollama "
        f"(default: {defaults.judge_model})",
    )
    endpoint = argparse.ArgumentParser(add_help=False)
    endpoint.add_argument(
        "--llm-url",
        metavar="URL",
        help="the OpenAI-compatible endpoint of the LLM detector, vision and the quiz: a URL, "
        f"or ntnu (key from {NTNU_KEY_ENV}), openrouter (key from {OPENROUTER_KEY_ENV}), "
        f"openai (key from {OPENAI_KEY_ENV}) or ollama (no key) (default: ntnu)",
    )
    endpoint.add_argument(
        "--llm-reasoning",
        choices=("none", "low", "medium", "high", "server"),
        default=None,
        metavar="LEVEL",
        help="how long a thinking model may think before it answers: none, low, medium or high, "
        "or server to send no setting, for a model that takes none such as gpt-4o or rejects "
        f"none such as Qwen (default: {defaults.reasoning_effort or 'server'}). It sets the judge "
        "and the quiz; vision has vision_reasoning_effort in the settings file (default "
        f"{defaults.vision_reasoning_effort}, which DeepSeek needs). Small models such as "
        "qwen3.5:9b judge much better with low; low, medium and high also allow each request "
        "10 minutes instead of 2",
    )

    quizzing = argparse.ArgumentParser(add_help=False)
    quizzing.add_argument(
        "--strictness",
        choices=STRICTNESS,
        default=None,
        help="which questions become quiz items: strict (only questions the lecture answered) "
        f"or lenient (also the others; the model writes their answer) "
        f"(default: {defaults.quiz_strictness})",
    )
    quizzing.add_argument(
        "--options",
        type=int,
        default=None,
        metavar="N",
        help=f"options per quiz item, one of them correct: {MIN_OPTIONS} to {MAX_OPTIONS} "
        f"(default: {defaults.quiz_options})",
    )
    quizzing.add_argument(
        "--quiz-model",
        dest="quiz_model",
        default=None,
        metavar="NAME",
        help=f"the model that writes the quiz (default: {defaults.quiz_model})",
    )

    vision_model = argparse.ArgumentParser(add_help=False)
    vision_model.add_argument(
        "--vision-model",
        default=None,
        metavar="NAME",
        help=f"the model that reads the board, as the endpoint names it (default: "
        f"{defaults.vision_model})",
    )
    vision_model.add_argument(
        "--vision-reader",
        choices=("vlm", "ocr"),
        default=None,
        help="how keyframes are read: vlm (the vision model: text, formulas and a description) "
        f"or ocr ({defaults.vision_ocr_model}: text and formulas only, faster) (default: "
        f"{defaults.vision_reader})",
    )

    process = subs.add_parser(
        "process",
        parents=[configured, shared, llm, endpoint, vision_model, quizzing],
        help="extract questions from a recording or a transcript",
        description="Extract the questions asked in a recording, or in an existing transcript.",
        epilog=_PROCESS_EXAMPLES,
        formatter_class=raw,
    )
    process.add_argument(
        "input",
        help="an .mp4 or .mp3 recording (runs everything) or a .json transcript (skips "
        "transcription)",
    )
    process.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        metavar="0-1",
        help=f"keep questions with at least this confidence (default: {defaults.min_confidence})",
    )
    process.add_argument(
        "--quiz",
        action="store_true",
        help="also write a multiple-choice quiz next to the questions (lecture.quiz.json); "
        "needs the questxtract[llm] extra and a key or --llm-url",
    )
    process.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="also read the slides or the board with a vision model, by default DeepSeek on "
        f"NTNU's LLM service (needs the questxtract[llm] extra and {NTNU_KEY_ENV}, or "
        "vision_base_url / --llm-url for another endpoint); --no-vision turns off "
        "vision_enabled from the settings file for one run",
    )
    process.set_defaults(handler=_process)

    transcribe = subs.add_parser(
        "transcribe",
        parents=[configured, shared],
        help="transcribe a recording only, into a file `process` reads back",
        description="Transcribe a recording into a transcript file, without extracting questions.",
        epilog=_TRANSCRIBE_EXAMPLES,
        formatter_class=raw,
    )
    transcribe.add_argument("recording", help="an .mp4 or .mp3 recording")
    transcribe.set_defaults(handler=_transcribe)

    filtering = subs.add_parser(
        "filter",
        help="keep the questions that mention words, spoken or on the board, or fall in a time "
        "range",
        description="Keep the questions of a questions.json that match every given word, said "
        "or written on a board state linked to them, and fall in the time range. Prints the "
        "result unless -o is given.",
        epilog=_FILTER_EXAMPLES,
        formatter_class=raw,
    )
    filtering.add_argument("questions", help="a questions.json written by `process`")
    filtering.add_argument(
        "--text",
        metavar="WORDS",
        help="keep questions where every word appears, as a whole word and ignoring case, in "
        "the question, its answer, or the board on screen while it was asked",
    )
    filtering.add_argument(
        "--from",
        dest="start",
        type=_time,
        metavar="TIME",
        help="keep questions asked at or after TIME: seconds, M:SS or H:MM:SS",
    )
    filtering.add_argument(
        "--to",
        dest="end",
        type=_time,
        metavar="TIME",
        help="keep questions asked before TIME: seconds, M:SS or H:MM:SS",
    )
    filtering.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="write the result here (default: print it to the terminal)",
    )
    filtering.set_defaults(handler=_filter)

    evaluation = subs.add_parser(
        "eval",
        parents=[configured, llm, endpoint, vision_model],
        help="score against the labelled set",
    )
    evaluation.add_argument(
        "paths",
        nargs="+",
        help="*.labels.jsonl files, or directories holding them (with --score-vision: "
        "*.frames.jsonl)",
    )
    evaluation.add_argument(
        "--score-vision",
        action="store_true",
        dest="score_vision",
        help="score the vision model (--vision-model, at vision's endpoint) on the labelled "
        "keyframes instead of question detection; calls the model once per frame, then caches",
    )
    evaluation.add_argument(
        "--board",
        action="store_true",
        help="use the board states in <name>.visual.json beside each transcript, as if "
        "--vision had read them",
    )
    evaluation.add_argument("--json", action="store_true", help="machine-readable, for CI")
    evaluation.set_defaults(handler=_eval)

    quiz = subs.add_parser(
        "quiz",
        parents=[configured, quizzing, endpoint],
        help="write a multiple-choice quiz from a questions.json",
        description="Write multiple-choice quiz items from the questions of a lecture. The only "
        "command that writes text that was not spoken: every generated text is marked, and every "
        "item starts as pending review.",
        epilog=_QUIZ_EXAMPLES,
        formatter_class=raw,
    )
    quiz.add_argument("questions", help="a questions.json written by `process`")
    quiz.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="where to write (default: next to the input, lecture.quiz.json; '-' prints to the "
        "terminal)",
    )
    quiz.add_argument(
        "--transcript",
        type=Path,
        metavar="PATH",
        help="the lecture's transcript, so the quiz sees what was said around each question "
        "(default: the cached transcript of the recording, when there is one)",
    )
    quiz.set_defaults(handler=_quiz)

    qti = subs.add_parser("qti", help="export quiz.json as a QTI 2.1 or 3.0 package")
    qti.add_argument("quiz", help="a quiz.json")
    qti.add_argument(
        "-o",
        "--output",
        help="write the QTI ZIP package here (default: next to the quiz, lecture.qti.zip)",
    )
    qti.add_argument(
        "--qti-version",
        choices=QTI_VERSIONS,
        default=DEFAULT_QTI_VERSION,
        help=f"QTI version of the package: 2.1 or 3.0 (default: {DEFAULT_QTI_VERSION})",
    )
    qti.set_defaults(handler=_exported)

    gui = subs.add_parser(
        "gui",
        help="open the graphical app (needs questxtract[gui])",
    )
    gui.set_defaults(handler=_gui)
    tex = subs.add_parser(
        "tex", help="export quiz.json as TeX for ExerPlaza's upload at /questions/upload/tex"
    )
    tex.add_argument("quiz", help="a quiz.json")
    tex.add_argument(
        "-o", "--output", help="write the .tex file here (default: next to the quiz, lecture.tex)"
    )
    tex.add_argument(
        "--content-unit",
        action="append",
        required=True,
        dest="content_units",
        metavar="NAME",
        help="an ExerPlaza content unit for every question; repeat for more. Spell it exactly as "
        "in ExerPlaza: any other spelling creates a new, unlinked unit",
    )
    tex.set_defaults(handler=_exported)

    questions_tex = subs.add_parser(
        "questions-tex",
        help="write questions.json as a LaTeX document that compiles",
        description="Write the questions of a questions.json as a LaTeX document: each "
        "question with its time, confidence, the lecturer's answer and the board text. "
        "Questions reviewed as deleted or false_positive are left out; texts are escaped; "
        "compile it with pdflatex.",
    )
    questions_tex.add_argument("questions", help="a questions.json")
    questions_tex.add_argument(
        "-o",
        "--output",
        help="write the .tex file here (default: next to the input, <name>.questions.tex)",
    )
    questions_tex.set_defaults(handler=_exported)

    exerplaza = subs.add_parser(
        "exerplaza",
        help="send quiz.json to ExerPlaza, as a quiz in a course session",
        description="Send a quiz to ExerPlaza's import endpoint, logged in with your ExerPlaza "
        f"email and password from {EXERPLAZA_EMAIL_ENV} and {EXERPLAZA_PASSWORD_ENV}. "
        "Rejected items are not sent; pending ones become questions only you can see, outside "
        "the quiz until you send them again accepted. "
        "Sending the same quiz again adds its new items to the same quiz; items whose text "
        "ExerPlaza already has are skipped and left as they are.",
        epilog=_EXERPLAZA_EXAMPLES,
        formatter_class=raw,
    )
    exerplaza.add_argument("quiz", help="a quiz.json")
    exerplaza.add_argument(
        "--url",
        required=True,
        metavar="URL",
        help="ExerPlaza's address, e.g. https://exerplaza.example.org (http for localhost only)",
    )
    exerplaza.add_argument(
        "--course-session",
        required=True,
        type=int,
        metavar="ID",
        help="the course session the quiz belongs to: the number at the end of its page's "
        "address, /course/session/ID. You must teach it",
    )
    exerplaza.add_argument(
        "--content-unit",
        action="append",
        required=True,
        dest="content_units",
        metavar="NAME",
        help="an ExerPlaza content unit for every question; repeat for more. Spell it exactly as "
        "in ExerPlaza: any other spelling creates a new, unlinked unit",
    )
    exerplaza.set_defaults(handler=_exerplaza)

    return parser


def _print_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """Show a warning as one ``warning:`` line of its own, not inside the progress bar's line."""
    del category, filename, lineno, file, line
    print(f"\nwarning: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    handler: object = args.handler
    assert callable(handler)  # noqa: S101 - set_defaults guarantees it
    try:
        with warnings.catch_warnings():
            warnings.showwarning = _print_warning
            return int(handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line interface: ``process``, ``transcribe``, ``filter``, ``eval``, ``quiz`` and ``qti``.

Every command and option, with recipes: docs/cli.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from . import __version__
from .config import (
    NTNU_KEY_ENV,
    NTNU_LLM_URL,
    OPENAI_KEY_ENV,
    OPENAI_URL,
    Config,
    is_ntnu,
    is_openai,
    load_config,
)
from .evaluate import evaluate, evaluate_vision, format_report, format_vision_report
from .export import read_questions, result_to_json, write_questions
from .filtering import filter_document
from .models import Context, Sentence, Stage, Transcript
from .pipeline import STAGES, cached_transcript, run
from .qti import QtiExportError
from .qti_v3 import export_qti
from .quiz import MAX_OPTIONS, MIN_OPTIONS, STRICTNESS, build_quiz, file_hash, quiz_to_json
from .vision import has_video

_AFTER_TRANSCRIBE = 2  # STAGES[:2] is ingest + transcribe; a transcript input skips both

_BAR_WIDTH = 10
# A thinking model takes minutes per request, where one that answers straight away takes
# seconds: the per-request deadline follows the reasoning level.
_THINKING_TIMEOUT_SECONDS = 600
# A redirected stderr on Windows is cp1252, not the UTF-8 console, and the block glyphs
# raise UnicodeEncodeError there - a crash in the middle of a real run, over decoration.
_FULL, _EMPTY = ("█", "░") if "utf" in (sys.stderr.encoding or "").lower() else ("#", "-")
# Friendlier than the raw stage name; anything not listed falls back to a title-cased
# version of its name, so a new stage added later still renders something reasonable.
_STAGE_LABELS = {
    "ingest": "Checking recording",
    "transcribe": "Transcribing",
    "sentences": "Splitting sentences",
    "segments": "Grouping segments",
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

every command and option, with recipes: docs/cli.md
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
                                                         (Qwen on NTNU's LLM service)
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

_FILTER_EXAMPLES = """\
examples:
  questxtract filter lecture.questions.json --text eigenvalue      print the matching questions
  questxtract filter lecture.questions.json --text "transfer function" -o tf.json
                                                         every word, spoken or on the board
  questxtract filter lecture.questions.json --from 5:00 --to 20:00   asked in that window
"""


def _progress_bar(total: int) -> Callable[[str, float], None]:
    """Build a ``Progress`` callback that redraws one line per stage on stderr (issue #14).

    Only reacts to a stage name changing (or ``"done"``) or an intermediate fraction from
    a stage that reports one - it never assumes 1.0 on its own except when a later stage's
    0.0 call confirms the previous one finished.
    """
    index = 0
    current = ""
    last = ""

    def draw(name: str, fraction: float) -> None:
        """Draw only when the rendered line actually changes.

        The ASR backend reports once per word (asr.py): a 1% step is ~100 words, so the
        great majority of calls would otherwise reflush a byte-identical line to stderr -
        thousands of writes per lecture, and as many junk lines in a redirected log.
        """
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
    """Build one stage's bar line, without the leading carriage return."""
    filled = round(max(0.0, min(1.0, fraction)) * _BAR_WIDTH)
    bar = _FULL * filled + _EMPTY * (_BAR_WIDTH - filled)
    label = _STAGE_LABELS.get(name, name.replace("_", " ").capitalize())
    return f"[{index}/{total}] {label:<20} {bar} {fraction:>4.0%}"


def _draw(index: int, total: int, name: str, fraction: float) -> None:
    print(f"\r{_render(index, total, name, fraction)}", end="", file=sys.stderr, flush=True)


def _check_input(path: Path, allowed: Sequence[str]) -> None:
    """Reject a bad input up front, not minutes into a transcription.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If its suffix is not one of ``allowed``.

    """
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such file")
    if path.suffix.lower() not in allowed:
        expected = " or ".join(_INPUT_KINDS[suffix] for suffix in allowed)
        raise ValueError(f"{path}: expected {expected}")


def _output_path(output: str | None, source: Path, kind: str) -> Path | None:
    """Resolve ``-o``: ``None`` means stdout (``-o -``); omitted means next to the input.

    ``lecture.mp4`` and ``lecture.transcript.json`` both default to ``lecture.<kind>.json``.
    The parent directory is created now, so a typo'd path fails before the slow part.
    """
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
    """Write ``text`` to ``path``, or print it when ``path`` is ``None``."""
    if path is None:
        print(text)
        return
    path.write_text(text, encoding="utf-8")
    print(f"{summary} -> {path}", file=sys.stderr)


def _config(args: argparse.Namespace) -> Config:
    """Apply explicit flags over the selected file, which overlays the defaults."""
    config = load_config(args.config)
    changes = {
        field: value
        for flag, field in (
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
    if config.reasoning_effort not in ("", "none", "low", "medium", "high"):
        raise ValueError("reasoning_effort must be empty, none, low, medium or high")
    return config


def _with_quiz(config: Config, what: str) -> None:
    """Check the quiz settings and its endpoint before anything slow runs.

    Only for a run that writes a quiz, so a bad quiz setting in the settings file does not stop
    ``transcribe``, ``eval`` or ``process`` without ``--quiz``.

    Raises:
        ValueError: If ``quiz_strictness`` or ``quiz_options`` is out of range, or the endpoint
            cannot be asked (see :func:`_check_endpoint`).

    """
    if config.quiz_strictness not in STRICTNESS:
        raise ValueError("quiz_strictness / --strictness must be strict or lenient")
    if not MIN_OPTIONS <= config.quiz_options <= MAX_OPTIONS:
        raise ValueError(
            f"quiz_options / --options must be between {MIN_OPTIONS} and {MAX_OPTIONS}"
        )
    _check_endpoint(config, what)


def _with_llm(config: Config) -> Config:
    """Check an LLM run can start before anything slow runs, and allow thinking time.

    Only a thinking level asked for (low, medium or high) gets the long timeout; with no setting
    sent, a stalled request is noticed after ``timeout_seconds``.

    Raises:
        ValueError: If ``--detector llm`` lacks the ``[llm]`` extra, or has neither an API
            key nor a local endpoint.

    """
    if config.reasoning_effort in ("low", "medium", "high"):
        config = config.replace(
            timeout_seconds=max(config.timeout_seconds, _THINKING_TIMEOUT_SECONDS)
        )
    if config.detector == "llm":
        _check_endpoint(config, "--detector llm")
    return config


def _check_endpoint(config: Config, what: str, *, vision: bool = False) -> None:
    """Raise ``ValueError`` if ``what`` cannot reach its model: no extra, or no key for it.

    The key goes with the endpoint (:func:`config.key_env`): NTNU's LLM service, the default,
    needs ``QUESTXTRACT_NTNU_API_KEY``; OpenAI needs ``OPENAI_API_KEY``. Any other endpoint may
    be a local server without a key.
    """
    if importlib.util.find_spec("openai") is None:
        raise ValueError(f"{what} needs the LLM extra: pip install 'questxtract[llm]'")
    url = config.vision_url if vision else config.base_url
    if (is_ntnu(url) or is_openai(url)) and not (
        config.vision_api_key if vision else config.api_key
    ):
        if is_ntnu(url):
            raise ValueError(
                f"{what} goes to NTNU's LLM service ({NTNU_LLM_URL}, NTNU network or VPN), so "
                f"it needs {NTNU_KEY_ENV} in the environment or .env; for OpenAI instead, pass "
                f"--llm-url {OPENAI_URL} with {OPENAI_KEY_ENV} and --llm-reasoning server"
            )
        raise ValueError(
            f"{what} goes to OpenAI, so it needs {OPENAI_KEY_ENV} in the environment or .env, "
            "or --llm-url (base_url) for another endpoint"
        )


def _write_quiz(
    questions: Path,
    output: Path | None,
    config: Config,
    sentences: Sequence[Sentence] = (),
    document: dict[str, object] | None = None,
) -> None:
    """Generate the quiz for a questions file and write it (``None`` prints it).

    A question the model fails on is left out with a warning line; the others are written.
    """
    result = build_quiz(
        document if document is not None else read_questions(questions),
        config,
        questions_name=questions.name,
        questions_hash=file_hash(questions),
        sentences=sentences,
        progress=(bar := _progress_bar(1)),
    )
    bar("done", 1.0)
    for question_id, reason in result.failed:
        print(f"warning: quiz: {question_id} left out: {reason}", file=sys.stderr)
    items = len(result.quiz["questions"])
    left_out = f" ({len(result.skipped)} left out, {len(result.failed)} failed)"
    _write(quiz_to_json(result.quiz).rstrip("\n"), output, f"{items} quiz items{left_out}")
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
    """Find the lecture's sentences for the quiz's spoken context.

    From ``--transcript``, else from the cache of the recording ``questions.json`` names, else
    none, with a note.
    """
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
    # Sentence ids hash each sentence's start and text, so a transcript of another run or
    # recording finds none of the questions: say so instead of sending no context silently.
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
    """Return the question objects of a ``questions.json`` document."""
    questions = document.get("questions")
    return [q for q in questions if isinstance(q, dict)] if isinstance(questions, list) else []


# Vision flags (#61). The vision design is still moving in M3 (ink curve #133). When vision
# gains a setting, decide whether it gets a flag here or stays TOML-only, and update together:
# `_config`, this check, the `process` parser, the "Reading the board" section in README.md,
# the option and error tables in docs/cli.md, and
# test_every_vision_setting_has_a_flag_decision in tests/test_cli.py, which fails until you do.
def _with_vision(config: Config, source: Path) -> Config:
    """Check that vision can read the board before transcription and sampling run.

    Applies whether vision was turned on by ``--vision`` or by ``vision_enabled`` in the
    settings file; :func:`_with_llm` does the same for ``--detector llm``. A source without a
    video track (an MP3, an audio-only MP4, or a transcript) has no board: a note says so and
    the run goes on without vision, so it needs neither the extra nor a key, and
    ``questions.json`` does not name a vision model that read nothing.

    Returns:
        The settings to run with.

    Raises:
        ValueError: If vision would run without the ``[llm]`` extra, or on NTNU's service
            without its key (:func:`_check_endpoint`).

    """
    if not config.vision_enabled:
        return config
    if not has_video(source):
        print(f"note: {source.name} has no video, so the board is not read", file=sys.stderr)
        return config.replace(vision_enabled=False)
    _check_endpoint(config, "--vision / vision_enabled", vision=True)
    return config


def _run(ctx: Context, stages: Sequence[Stage]) -> Context:
    """Run ``stages`` under the progress bar, closing its line if anything interrupts it."""
    try:
        return run(ctx, stages, progress=_progress_bar(len(stages)))
    except BaseException:
        print(file=sys.stderr)  # close the redrawing bar line before the error/traceback
        raise


def _process(args: argparse.Namespace) -> int:
    """Extract questions from a recording (every stage) or a transcript (skips ASR)."""
    config = _with_llm(_config(args))
    source = Path(args.input)
    _check_input(source, (".mp4", ".mp3", ".json"))
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
        # The documented `stages=` seam (architecture.md 11.1): the transcript is in hand.
        ctx = _run(ctx, STAGES[_AFTER_TRANSCRIBE:])
    else:
        ctx = _run(Context(source=source, config=config), STAGES)

    if output is None:  # stdout has no folder for keyframes, so they keep their cache paths
        print(result_to_json(ctx))
    else:
        write_questions(ctx, output)
        print(f"{len(ctx.pairs)} questions -> {output}", file=sys.stderr)
        if args.quiz:  # questions.json is on disk first: a failing quiz never loses it
            _write_quiz(output, _output_path(None, output, "quiz"), config, ctx.sentences)
    return 0


def _transcribe(args: argparse.Namespace) -> int:
    """Transcribe a recording only, into a file ``process`` reads back.

    Runs the real ingest and transcribe stages, so it shares their cache with ``process``:
    whichever of the two runs second on the same recording skips the ASR.
    """
    config = _config(args)
    source = Path(args.recording)
    _check_input(source, (".mp4", ".mp3"))
    output = _output_path(args.output, source, "transcript")
    ctx = _run(Context(source=source, config=config), STAGES[:_AFTER_TRANSCRIBE])
    assert ctx.transcript is not None  # noqa: S101 - the transcribe stage always sets it
    text = json.dumps(ctx.transcript.to_dict(), indent=2, ensure_ascii=False)
    _write(text, output, f"{len(ctx.transcript.asr_segments)} segments")
    return 0


def _time(value: str) -> float:
    """Parse ``--from`` / ``--to``: seconds (``300``, ``12.5``) or ``M:SS`` / ``H:MM:SS``.

    A negative, infinite or reversed range is left to :func:`filter_document`'s checks.
    """
    try:
        parts = [float(part) for part in value.split(":")]
    except ValueError:
        parts = []
    if not 1 <= len(parts) <= 3 or not all(0 <= part < 60 for part in parts[1:]):
        raise argparse.ArgumentTypeError(f"expected seconds, M:SS or H:MM:SS, got {value!r}")
    return sum(part * 60.0**power for power, part in enumerate(reversed(parts)))


def _repoint_keyframes(data: dict[str, object], source: Path, output: Path) -> None:
    """Keep each relative ``keyframe_path`` pointing at its image from ``output``'s folder.

    A written ``questions.json`` names keyframes relative to itself (architecture.md 12.1), so a
    copy in another folder would lose them. Absolute when no relative path exists (another
    drive on Windows).
    """
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
    """Keep the questions of a ``questions.json`` that match the words and the time range."""
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
    """Score the pipeline, or with ``--score-vision`` the vision model, on the labelled set."""
    try:
        config = _with_llm(_config(args))
        if args.score_vision:
            _check_endpoint(config, "eval --score-vision", vision=True)
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
    """Turn questions.json into quiz.json. Generative, and deliberately separate."""
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


def _qti(args: argparse.Namespace) -> int:
    """Export quiz.json as a QTI 3.0 package."""
    try:
        export_qti(args.quiz, args.output)
    except QtiExportError as exc:
        print(f"error: qti: {exc}", file=sys.stderr)
        return 1
    print(f"QTI 3.0 package -> {args.output}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    defaults = Config()
    raw = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
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

    # Flags every command that can transcribe shares, so both --help pages read the same.
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

    # How questions are found: `process` and `eval` share these.
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
    llm.add_argument(
        "--llm-url",
        metavar="URL",
        help="an OpenAI-compatible endpoint for --detector llm, --vision and the quiz, e.g. "
        f"{OPENAI_URL} (key from {OPENAI_KEY_ENV}) or http://localhost:11434/v1 for Ollama "
        f"(default: NTNU's LLM service, key from {NTNU_KEY_ENV})",
    )
    llm.add_argument(
        "--llm-reasoning",
        choices=("none", "low", "medium", "high", "server"),
        default=None,
        metavar="LEVEL",
        help="how long a thinking model may think before it answers, for every request: none, "
        "low, medium or high, or server to send no setting, for a model that takes none such "
        f"as gpt-4o or rejects none such as Qwen (default: {defaults.reasoning_effort or 'server'})"
        ". Small models such as qwen3.5:9b judge much better with low; "
        "low, medium and high also allow each request 10 minutes instead of 2",
    )

    # How the quiz is made: `process --quiz` and `quiz` share these.
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

    # The vision model: `process` reads the board with it, `eval --score-vision` scores it.
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
        parents=[configured, shared, llm, vision_model, quizzing],
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
    # See the note above _with_vision before adding a vision flag.
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
        "eval", parents=[configured, llm, vision_model], help="score against the labelled set"
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
        dest="score_vision",  # not `vision`: that is vision_enabled, read by _config
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
        parents=[configured, quizzing],
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
    quiz.add_argument(
        "--llm-url",
        metavar="URL",
        help=f"an OpenAI-compatible endpoint, e.g. {OPENAI_URL} (key from {OPENAI_KEY_ENV}) "
        f"or http://localhost:11434/v1 for Ollama (default: NTNU's LLM service, key from "
        f"{NTNU_KEY_ENV})",
    )
    quiz.add_argument(
        "--llm-reasoning",
        choices=("none", "low", "medium", "high", "server"),
        default=None,
        metavar="LEVEL",
        help="how long a thinking model may think before it answers: none, low, medium or high, "
        f"or server to send no setting (default: {defaults.reasoning_effort or 'server'})",
    )
    quiz.set_defaults(handler=_quiz)

    qti = subs.add_parser("qti", help="export quiz.json as a QTI 3.0 package")
    qti.add_argument("quiz", help="a quiz.json")
    qti.add_argument("-o", "--output", required=True, help="write the QTI ZIP package here")
    qti.set_defaults(handler=_qti)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    handler: object = args.handler
    assert callable(handler)  # noqa: S101 - set_defaults guarantees it
    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        # Everything a user can cause - a missing or unreadable file, a non-media file, a recording
        # with no audio, a malformed transcript, an unknown model name - arrives as one of
        # these with a readable message; a traceback would only bury it (issue #17).
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

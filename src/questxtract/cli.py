"""Command line interface: ``process``, ``transcribe``, ``eval`` and ``quiz``.

Every command and option, with recipes: docs/cli.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from . import __version__
from .config import Config
from .evaluate import evaluate, format_report
from .export import result_to_json
from .models import Context, Stage, Transcript
from .pipeline import STAGES, run

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
}

_INPUT_KINDS = {".mp4": "an .mp4 recording", ".json": "a .json transcript"}

_EXAMPLES = """\
examples:
  questxtract process lecture.mp4               video -> lecture.questions.json
  questxtract transcribe lecture.mp4            transcript only -> lecture.transcript.json
  questxtract process lecture.transcript.json   questions from an existing transcript

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
"""

_TRANSCRIBE_EXAMPLES = """\
examples:
  questxtract transcribe lecture.mp4                     writes lecture.transcript.json
  questxtract transcribe lecture.mp4 -m small -o t.json  smaller model, custom path
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
        base = source.name.removesuffix(".transcript.json")
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
    """Apply the flags every command shares over the defaults."""
    return Config().replace(model_size=args.model, from_scratch=args.from_scratch)


def _with_llm(config: Config, args: argparse.Namespace) -> Config:
    """Apply the LLM flags, and check an LLM run can start before anything slow runs.

    Raises:
        ValueError: If ``--detector llm`` lacks the ``[llm]`` extra, or has neither an API
            key nor a local endpoint.

    """
    config = config.replace(
        detector=args.detector,
        judge_model=args.llm_model,
        base_url=args.llm_url,
        judge_reasoning_effort=args.llm_reasoning,
    )
    if args.llm_reasoning not in ("none", ""):
        config = config.replace(
            timeout_seconds=max(config.timeout_seconds, _THINKING_TIMEOUT_SECONDS)
        )
    if config.detector == "llm":
        if importlib.util.find_spec("openai") is None:
            raise ValueError("--detector llm needs the LLM extra: pip install 'questxtract[llm]'")
        if config.base_url is None and not config.api_key:
            raise ValueError("--detector llm needs OPENAI_API_KEY, or --llm-url for a local server")
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
    if not 0.0 <= args.threshold <= 1.0:  # also catches nan
        raise ValueError(f"--threshold must be between 0 and 1, got {args.threshold}")
    source = Path(args.input)
    _check_input(source, (".mp4", ".json"))
    output = _output_path(args.output, source, "questions")
    config = _with_llm(_config(args), args).replace(min_confidence=args.threshold)

    if source.suffix.lower() == ".json":
        transcript = Transcript.from_file(source)
        ctx = Context(source=source, config=config, transcript=transcript)
        ctx.duration = transcript.duration
        # The documented `stages=` seam (architecture.md 11.1): the transcript is in hand.
        ctx = _run(ctx, STAGES[_AFTER_TRANSCRIBE:])
    else:
        ctx = _run(Context(source=source, config=config), STAGES)

    _write(result_to_json(ctx), output, f"{len(ctx.pairs)} questions")
    return 0


def _transcribe(args: argparse.Namespace) -> int:
    """Transcribe a recording only, into a file ``process`` reads back.

    Runs the real ingest and transcribe stages, so it shares their cache with ``process``:
    whichever of the two runs second on the same recording skips the ASR.
    """
    source = Path(args.recording)
    _check_input(source, (".mp4",))
    output = _output_path(args.output, source, "transcript")
    ctx = _run(Context(source=source, config=_config(args)), STAGES[:_AFTER_TRANSCRIBE])
    assert ctx.transcript is not None  # noqa: S101 - the transcribe stage always sets it
    text = json.dumps(ctx.transcript.to_dict(), indent=2, ensure_ascii=False)
    _write(text, output, f"{len(ctx.transcript.asr_segments)} segments")
    return 0


def _eval(args: argparse.Namespace) -> int:
    """Score the pipeline against the labelled set."""
    try:
        report = evaluate(args.paths, _with_llm(Config(), args))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0


def _quiz(args: argparse.Namespace) -> int:
    """Turn questions.json into quiz.json. Generative, and deliberately separate."""
    del args
    print("error: quiz is not implemented yet (#79)", file=sys.stderr)
    return 1


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
        default=defaults.model_size,
        metavar="NAME",
        help="Whisper model: tiny, base, small, medium, large-v3 or turbo (default: "
        "%(default)s); smaller is faster and less accurate. Only used when transcribing",
    )
    shared.add_argument(
        "--from-scratch",
        action="store_true",
        help="ignore cached results and recompute everything, transcription included",
    )

    # How questions are found: `process` and `eval` share these.
    llm = argparse.ArgumentParser(add_help=False)
    llm.add_argument(
        "--detector",
        choices=("rules", "llm"),
        default=defaults.detector,
        help="how questions are found: rules (fast, offline) or llm (a language model reads the "
        "transcript; needs the questxtract[llm] extra) (default: %(default)s)",
    )
    llm.add_argument(
        "--llm-model",
        default=defaults.judge_model,
        metavar="NAME",
        help="the model for --detector llm, e.g. qwen3.5:9b in Ollama (default: %(default)s)",
    )
    llm.add_argument(
        "--llm-url",
        metavar="URL",
        help="an OpenAI-compatible endpoint for --detector llm, e.g. http://localhost:11434/v1 "
        "for Ollama (default: OpenAI, with the key from OPENAI_API_KEY)",
    )
    llm.add_argument(
        "--llm-reasoning",
        default=defaults.judge_reasoning_effort,
        metavar="LEVEL",
        help="how long a thinking model may think before it answers: none, low, medium or high "
        "(default: %(default)s). Small models such as qwen3.5:9b judge much better with low; "
        "anything but none also allows each request 10 minutes instead of 2",
    )

    process = subs.add_parser(
        "process",
        parents=[shared, llm],
        help="extract questions from a recording or a transcript",
        description="Extract the questions asked in a recording, or in an existing transcript.",
        epilog=_PROCESS_EXAMPLES,
        formatter_class=raw,
    )
    process.add_argument(
        "input",
        help="an .mp4 recording (runs everything) or a .json transcript (skips transcription)",
    )
    process.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=defaults.min_confidence,
        metavar="0-1",
        help="keep questions with at least this confidence (default: %(default)s)",
    )
    process.set_defaults(handler=_process)

    transcribe = subs.add_parser(
        "transcribe",
        parents=[shared],
        help="transcribe a recording only, into a file `process` reads back",
        description="Transcribe a recording into a transcript file, without extracting questions.",
        epilog=_TRANSCRIBE_EXAMPLES,
        formatter_class=raw,
    )
    transcribe.add_argument("recording", help="an .mp4 recording")
    transcribe.set_defaults(handler=_transcribe)

    evaluation = subs.add_parser("eval", parents=[llm], help="score against the labelled set")
    evaluation.add_argument(
        "paths", nargs="+", help="*.labels.jsonl files, or directories holding them"
    )
    evaluation.add_argument("--json", action="store_true", help="machine-readable, for CI")
    evaluation.set_defaults(handler=_eval)

    quiz = subs.add_parser(
        "quiz", help="generate quiz.json from questions.json (not implemented yet)"
    )
    quiz.add_argument("questions", help="a questions.json written by `process`")
    quiz.add_argument("-o", "--output", help="write the quiz here")
    quiz.set_defaults(handler=_quiz)

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
        # Everything a user can cause - a missing or unreadable file, a non-mp4, a recording
        # with no audio, a malformed transcript, an unknown model name - arrives as one of
        # these with a readable message; a traceback would only bury it (issue #17).
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

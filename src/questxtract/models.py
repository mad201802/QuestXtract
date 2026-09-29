"""Core data types for QuestXtract (architecture.md section 5).

``Context`` is the one mutable object in the system: the value every stage takes and
returns. Everything else is a frozen, keyword-only dataclass.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from .config import Config  # imported for real: the cache resolves Context's annotations

#: Stage name plus fraction complete (architecture.md 11.3).
Progress = Callable[[str, float], None]

#: Every stage has this signature. That sameness is the whole extension mechanism.
Stage = Callable[["Context"], "Context"]

_F = TypeVar("_F", bound=Stage)


class SchemaError(ValueError):
    """Raised when JSON input does not match the expected schema."""


@dataclass(frozen=True, kw_only=True)
class Word:
    """A single recognised word with its own timestamps.

    Args:
        text: The word as transcribed.
        start: Start time in seconds from the beginning of the recording.
        end: End time in seconds from the beginning of the recording.
        confidence: Per-word confidence from the ASR engine, in ``[0, 1]``.

    """

    text: str
    start: float
    end: float
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Word:
        """Build a word from a plain dict; accepts Whisper's ``word`` key or ``text``."""
        try:
            text = data["word"] if "word" in data else data["text"]
            return cls(
                text=str(text).strip(),
                start=float(data["start"]),
                end=float(data["end"]),
                confidence=float(data.get("probability", data.get("confidence", 1.0))),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"invalid word object: {exc}") from exc


@dataclass(frozen=True, kw_only=True)
class AsrSegment:
    """A raw ASR decoding chunk, as emitted by the engine.

    Not a :class:`Segment`: that is a run of sentences (architecture.md 3.2). This is the
    engine's own chunking, kept only for its quality metrics, which are the cheapest
    hallucination detector available and which the eval harness reads (architecture.md 2.2).

    Args:
        text: The transcribed text of the chunk.
        start: Start time in seconds from the beginning of the recording.
        end: End time in seconds from the beginning of the recording.
        avg_logprob: Mean token log-probability over the chunk.
        no_speech_prob: Probability the engine assigned to "no speech here".
        compression_ratio: gzip compression ratio of the text; high means repetition.

    """

    text: str
    start: float
    end: float
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AsrSegment:
        """Build a chunk from an ASR engine's JSON output."""
        try:
            return cls(
                text=str(data["text"]),
                start=float(data["start"]),
                end=float(data["end"]),
                avg_logprob=float(data.get("avg_logprob", 0.0)),
                no_speech_prob=float(data.get("no_speech_prob", 0.0)),
                compression_ratio=float(data.get("compression_ratio", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"invalid asr segment: {exc}") from exc


@dataclass(frozen=True, kw_only=True)
class Transcript:
    """The output of stage 2: timed words plus the engine's own chunking.

    Words are the atomic unit (architecture.md 2.5); every later timing is a slice of
    ``words``, never an interpolation.

    Args:
        words: Every recognised word, in recording order.
        asr_segments: The engine's chunks, kept for their quality metrics.
        language: Language code the engine decoded with.
        duration: Recording duration in seconds.

    """

    words: tuple[Word, ...] = ()
    asr_segments: tuple[AsrSegment, ...] = ()
    language: str = "en"
    duration: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        """Build a transcript from JSON.

        Accepts either this type's own serialisation or an ASR engine's output: a bare
        list of chunks, or an object with a ``segments`` key holding one. When a chunk
        carries no ``words``, the chunk text is spread evenly across its own span so
        downstream stages always have a word list to slice.

        Raises:
            SchemaError: If the shape is neither of the accepted forms.

        """
        if isinstance(data, list):
            data = {"segments": data}
        if not isinstance(data, dict):
            raise SchemaError("expected a list of segments or an object")
        raw = data.get("asr_segments", data.get("segments"))
        if raw is None:
            raise SchemaError("expected a list of segments or a 'segments' key holding one")
        if not isinstance(raw, list) or any(not isinstance(s, dict) for s in raw):
            raise SchemaError("'segments' must be a list of objects")

        chunks = tuple(AsrSegment.from_dict(s) for s in raw)
        if "words" in data:
            words = tuple(Word.from_dict(w) for w in data["words"])
        else:
            words = tuple(w for s, c in zip(raw, chunks, strict=True) for w in _chunk_words(s, c))
        try:
            duration = float(data["duration"])
        except (KeyError, TypeError, ValueError):
            duration = chunks[-1].end if chunks else 0.0
        return cls(
            words=words,
            asr_segments=chunks,
            language=str(data.get("language", "en")),
            duration=duration,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Transcript:
        """Read a transcript JSON file, in any shape :meth:`from_dict` accepts.

        Raises:
            SchemaError: If the file is not JSON or is not a recognised transcript shape.

        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise SchemaError(f"{path}: not valid JSON ({exc})") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain JSON types, in the shape :meth:`from_dict` reads back unchanged.

        Lists, not ``asdict``'s tuples: ``from_dict`` rejects a segments value that is not a list.
        """
        return {
            "words": [asdict(w) for w in self.words],
            "asr_segments": [asdict(s) for s in self.asr_segments],
            "language": self.language,
            "duration": self.duration,
        }


def _chunk_words(raw: dict[str, Any], chunk: AsrSegment) -> tuple[Word, ...]:
    """Words of one ASR chunk, synthesising evenly spaced ones when the engine gave none."""
    given = raw.get("words")
    if given:
        if not isinstance(given, list):
            raise SchemaError("'words' must be a list")
        return tuple(Word.from_dict(w) for w in given)
    tokens = chunk.text.split()
    if not tokens:
        return ()
    step = (chunk.end - chunk.start) / len(tokens)
    return tuple(
        Word(text=t, start=chunk.start + i * step, end=chunk.start + (i + 1) * step)
        for i, t in enumerate(tokens)
    )


@dataclass(frozen=True, kw_only=True)
class Sentence:
    """One sentence of the transcript, owning a half-open range of words.

    Args:
        id: Content hash, not an index: ids survive a re-transcription, which is what
            lets human edits merge back in (FR5).
        text: The sentence as transcribed.
        word_start: Inclusive start index into ``Transcript.words``.
        word_end: Exclusive end index into ``Transcript.words``.
        start: Start time in seconds, equal to ``words[word_start].start``.
        end: End time in seconds, equal to ``words[word_end - 1].end``.
        segment_index: Index of the :class:`Segment` this sentence belongs to.

    """

    id: str = ""
    text: str
    word_start: int
    word_end: int
    start: float
    end: float
    segment_index: int = -1

    def __post_init__(self) -> None:
        """Fill in the content-hash ``id`` when none was given."""
        if not self.id:
            digest = hashlib.sha1(f"{self.start:.3f}|{self.text}".encode()).hexdigest()
            object.__setattr__(self, "id", f"s_{digest[:6]}")


@dataclass(frozen=True, kw_only=True)
class Segment:
    """A contiguous run of sentences, bounded by silence and by length (architecture.md 3.2).

    Structural, not topical: nothing measures what the run is about. Distinct from
    :class:`AsrSegment`, which is the engine's own decoding chunk.

    Args:
        sentence_start: Inclusive start index into ``Context.sentences``.
        sentence_end: Exclusive end index into ``Context.sentences``.
        start: Start time in seconds.
        end: End time in seconds.
        text: The joined text of the run.
        visual: Board states overlapping this segment, filled by ``attach_visual``.
        close_reason: Which rule ended the run - ``gap``, ``duration``, ``chars``, or
            ``end`` for the last segment, which nothing closed.

    """

    sentence_start: int
    sentence_end: int
    start: float
    end: float
    text: str
    visual: tuple[VisualContext, ...] = ()
    close_reason: str = ""


@dataclass(frozen=True, kw_only=True)
class VisualContext:
    """What the board showed over one board-state lifetime (architecture.md 6.4).

    The lifetime is the board state's own, not a segment's span: the lecturer writes for
    two minutes and talks about it for six.

    Args:
        start: Start of the board-state lifetime in seconds.
        end: End of the board-state lifetime in seconds.
        keyframe_path: Path to the captured keyframe.
        description: What the VLM saw: diagram, derivation, slide.
        text: Transcribed board or slide text.
        latex: Formulas when present; manim generation reads this.
        confidence: Confidence in the reading, in ``[0, 1]``.

    """

    start: float
    end: float
    keyframe_path: str = ""
    description: str = ""
    text: str = ""
    latex: str | None = None
    confidence: float = 0.0

    @property
    def id(self) -> str:
        """Stable identifier from the lifetime and keyframe; a new caption keeps it."""
        key = f"{self.start:.3f}|{self.end:.3f}|{self.keyframe_path}"
        return f"v_{hashlib.sha1(key.encode()).hexdigest()[:6]}"


@dataclass(frozen=True, kw_only=True)
class QuestionCandidate:
    """A sentence that detection believes is a question (architecture.md 7).

    Args:
        sentence_id: Id of the :class:`Sentence` it came from.
        text: The refined focus; may be a clause of the sentence rather than all of it.
        score: Detection score in ``[0, 1]``.
        question_type: ``what``, ``why``, ``how``, ``yes_no``, ``definition`` or ``other``.
        reasons: Every rule that fired, positive and negative. Never dropped.

    """

    sentence_id: str
    text: str
    score: float
    question_type: str = "other"
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class QAPair:
    """A question with whatever answer the pipeline found for it, if any.

    Args:
        question: The candidate this pair was built from.
        answer_sentence_ids: Sentences the answer span covers; empty when none was found.
        answer_text: The answer span's text; empty when none was found.
        confidence: Combined confidence in ``[0, 1]`` (architecture.md 9.2).
        start: Question start in seconds, the FR7 timestamp.
        end: Question end in seconds.
        verdict: ``unjudged``, ``keep``, ``revise`` or ``reject``.
        reasons: Every scoring term that fired.

    """

    question: QuestionCandidate
    answer_sentence_ids: tuple[str, ...] = ()
    answer_text: str = ""
    confidence: float
    start: float
    end: float
    verdict: str = "unjudged"
    reasons: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """Stable identifier, derived from the question's start and text."""
        digest = hashlib.sha1(f"{self.start:.3f}|{self.question.text}".encode()).hexdigest()
        return f"q_{digest[:6]}"


@dataclass
class Context:
    """The value every stage takes and returns (architecture.md 11.1).

    A stage reads what earlier stages produced and sets its own field. If a stage needs
    something an earlier stage did not produce, it fails on the spot with a ``None``.
    """

    source: Path
    config: Config
    transcript: Transcript | None = None
    sentences: tuple[Sentence, ...] = ()
    segments: tuple[Segment, ...] = ()
    visual: tuple[VisualContext, ...] = ()  # board states; joined to segments by attach_visual
    candidates: tuple[QuestionCandidate, ...] = ()
    pairs: tuple[QAPair, ...] = ()
    stats: dict[str, int] = field(default_factory=dict)
    progress: Progress | None = None  # section 11.3; mid-stage reporting
    source_hash: str = ""
    duration: float = 0.0


@dataclass(frozen=True)
class StageInfo:
    """What the ``@stage`` decorator attaches to a stage function."""

    name: str
    cache: str | None
    uses: tuple[str, ...]


def stage(name: str, *, cache: str | None = None, uses: tuple[str, ...] = ()) -> Callable[[_F], _F]:
    """Tag a function as a pipeline stage.

    Lives here rather than in ``pipeline`` so the stage modules can import it without a
    cycle; ``questxtract.pipeline`` re-exports it, which is where architecture.md 11.1 puts it.

    Args:
        name: The stage name, used for progress reporting and the cache file.
        cache: The ``Context`` field this stage produces, cached as ``<name>.json``.
            ``None`` means always recompute.
        uses: ``Config`` field names whose values form the cache header (architecture.md 11.2).
            Keep this next to the signature: a stage that gains a config field and forgets
            to list it here is the one invalidation bug this design still has.

    """

    def decorate(fn: _F) -> _F:
        fn.stage = StageInfo(name=name, cache=cache, uses=uses)  # type: ignore[attr-defined]
        return fn

    return decorate


def stage_info(fn: Stage) -> StageInfo:
    """Return a stage function's :class:`StageInfo`.

    Raises:
        TypeError: If ``fn`` is not decorated with ``@stage``.

    """
    info = getattr(fn, "stage", None)
    if not isinstance(info, StageInfo):
        raise TypeError(f"{getattr(fn, '__name__', fn)!r} is not decorated with @stage")
    return info

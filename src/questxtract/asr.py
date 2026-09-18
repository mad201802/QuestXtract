"""Issue: a model-independent ASR interface.

Defines ``AsrBackend`` - the one seam every ASR engine implements - and a small registry
that turns ``Config.asr_backend`` (a plain string) into a concrete instance. The rest of
the pipeline (``transcribe.py``, ``pipeline.py``, ``cli.py``) never imports a specific
engine: they only ever see ``resolve_backend``.

Acceptance criteria this satisfies:
    1. ASR model can be called through a QuestXtract interface -> ``AsrBackend`` +
       ``resolve_backend``
    2. transcript segments contain timestamps -> the wire shape is ``{text, start, end}``,
       matching what ``Transcript.from_dict`` (models.py) already parses
    3. model configuration is not hard-coded -> every tunable is a ``Config`` field,
       backends read only ``config``
    4. ASR implementation can be replaced later -> ``register_backend`` / ``Config.asr_backend``
    5. unit/integration tests exist -> see ``test_asr.py``
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .config import Config
from .media import extract_audio


@runtime_checkable
class AsrBackend(Protocol):
    """The interface every ASR engine implements.

    A backend's only job is turning a recording into raw segments in the wire shape this
    issue proposes. It never needs to know about :class:`~questxtract.models.Transcript`:
    ``Transcript.from_dict`` already handles turning this into the full typed shape,
    including synthesising word timings when a backend gives none.
    """

    def transcribe(self, source: Path, config: Config) -> list[dict[str, Any]]:
        """Decode ``source`` and return raw ASR segments.

        Each at least::

            {"text": str, "start": float, "end": float}

        A backend may also include a ``"words"`` list per segment
        (``{"word", "start", "end", "probability"}``) when it can produce word-level
        timing; when it cannot, downstream stages synthesise evenly-spaced words instead.
        """
        ...


_REGISTRY: dict[str, type[AsrBackend]] = {}


def register_backend(name: str) -> Callable[[type[AsrBackend]], type[AsrBackend]]:
    """Class decorator: register a backend under ``name`` for :func:`resolve_backend`.

    A second ASR engine (a hosted API, WhisperX, Parakeet) is one class carrying this
    decorator - no changes anywhere else in the pipeline.
    """

    def decorate(cls: type[AsrBackend]) -> type[AsrBackend]:
        _REGISTRY[name] = cls
        return cls

    return decorate


def resolve_backend(name: str) -> AsrBackend:
    """Look up and instantiate a registered backend by name.

    Raises:
        ValueError: If ``name`` is not registered. Lists what is, so a typo'd
            ``Config.asr_backend`` fails loudly instead of silently falling back to a
            default the caller did not ask for.

    """
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        # _REGISTRY is never empty here: FasterWhisperBackend registers itself at import
        # time, before any caller could reach this function.
        raise ValueError(
            f"unknown ASR backend {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from exc
    return cls()


def _segment_to_dict(segment: Any) -> dict[str, Any]:
    """Convert one faster-whisper segment (plus its words) to the wire shape."""
    return {
        "text": segment.text,
        "start": segment.start,
        "end": segment.end,
        "avg_logprob": segment.avg_logprob,
        "no_speech_prob": segment.no_speech_prob,
        "compression_ratio": segment.compression_ratio,
        "words": [
            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
            for w in (segment.words or [])
        ],
    }


#: Loaded models, keyed on the two fields that determine which checkpoint gets loaded
#: (``model_size``, ``compute_type``) - not the full config, since options like ``language``
#: or ``beam_size`` are passed per call and never change which model file is read. Process
#: lifetime only: loading is the expensive part, so every call with the same key reuses it
#: instead of reading the checkpoint off disk again.
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _load_model(model_size: str, compute_type: str) -> Any:
    """Return the batched model for ``(model_size, compute_type)``, loading it once."""
    from faster_whisper import BatchedInferencePipeline, WhisperModel  # noqa: PLC0415

    key = (model_size, compute_type)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = BatchedInferencePipeline(
            WhisperModel(model_size, compute_type=compute_type)
        )
    return _MODEL_CACHE[key]


@register_backend("faster_whisper")
class FasterWhisperBackend:
    """The default backend: faster-whisper, batched (pipeline.md 2.2).

    Runs on audio extracted via :func:`~questxtract.media.extract_audio` rather than
    handing the raw container straight to faster-whisper: ``media.py`` already contains
    working, tested container and stream-selection logic (issue #7), so this backend
    reuses it instead of duplicating that handling here.
    """

    def transcribe(self, source: Path, config: Config) -> list[dict[str, Any]]:
        """Decode ``source`` into raw segments using faster-whisper.

        Raises:
            RuntimeError: If ``faster-whisper`` is not importable. It is a base
                dependency (pipeline.md 13.2), so this should not happen in a normal
                install; the import is still guarded so a broken environment fails with
                a clear message rather than an opaque traceback.

        """
        try:
            model = _load_model(config.model_size, config.compute_type)
        except ImportError as exc:  # pragma: no cover - base dependency, should be present
            raise RuntimeError(
                "faster-whisper is not installed; reinstall questxtract to restore it"
            ) from exc

        with extract_audio(source) as audio_path:
            segments, _info = model.transcribe(
                str(audio_path),
                beam_size=config.beam_size,
                language=config.language,
                vad_filter=config.vad_filter,
                word_timestamps=True,
                batch_size=config.batch_size,
                initial_prompt=config.initial_prompt,
            )
            return [_segment_to_dict(s) for s in segments]

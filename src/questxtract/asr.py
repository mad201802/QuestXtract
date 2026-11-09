"""The ASR backend interface and its registry."""

from __future__ import annotations

import warnings
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

    def transcribe(
        self,
        source: Path,
        config: Config,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Decode ``source`` and return raw ASR segments.

        Each at least::

            {"text": str, "start": float, "end": float}

        A backend may also include a ``"words"`` list per segment
        (``{"word", "start", "end", "probability"}``) when it can produce word-level
        timing; when it cannot, downstream stages synthesise evenly-spaced words instead.

        Args:
            source: The recording to decode.
            config: Pipeline settings; a backend reads only the fields it declared.
            on_progress: Called with a fraction in ``[0, 1)`` as segments complete, when
                the backend can estimate one (issue #14). Optional: a backend that cannot
                estimate progress may ignore it. Never called with ``1.0`` - the caller,
                not the backend, decides when the stage is actually finished.

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
        raise ValueError(
            f"unknown ASR backend {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from exc
    return cls()


def _segment_to_dict(segment: Any) -> dict[str, Any]:
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


_EXTRACT_SHARE = 0.1

# ctranslate2's error when a CUDA library such as cublas64_12.dll cannot load. It is raised
# at the first encode, not when the model loads.
_CUDA_LIBRARY_MISSING = "is not found or cannot be loaded"

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _load_model(model_size: str, compute_type: str, *, cpu: bool = False) -> Any:
    """Return the cached model; ``cpu=True`` replaces it with one that runs on the CPU."""
    from faster_whisper import BatchedInferencePipeline, WhisperModel  # noqa: PLC0415

    key = (model_size, compute_type)
    if cpu or key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = BatchedInferencePipeline(
            WhisperModel(model_size, device="cpu" if cpu else "auto", compute_type=compute_type)
        )
    return _MODEL_CACHE[key]


def _decode(
    model: Any,
    audio_path: Path,
    config: Config,
    on_progress: Callable[[float], None] | None,
) -> list[dict[str, Any]]:
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=config.beam_size,
        language=config.language,
        vad_filter=config.vad_filter,
        word_timestamps=True,
        batch_size=config.batch_size,
        initial_prompt=config.initial_prompt,
    )
    collected: list[dict[str, Any]] = []
    for s in segments:
        collected.append(_segment_to_dict(s))
        if on_progress and info.duration > 0:
            words = s.words or ()
            ends = (w.end for w in words) if words else (s.end,)
            for end in ends:
                on_progress(min(0.99, _EXTRACT_SHARE + (1 - _EXTRACT_SHARE) * end / info.duration))
    return collected


@register_backend("faster_whisper")
class FasterWhisperBackend:
    """The default backend: faster-whisper, batched (architecture.md 2.2)."""

    def transcribe(
        self,
        source: Path,
        config: Config,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Decode ``source`` into raw segments using faster-whisper."""
        try:
            model = _load_model(config.model_size, config.compute_type)
        except ImportError as exc:  # pragma: no cover - base dependency, should be present
            raise RuntimeError(
                "faster-whisper is not installed; reinstall questxtract to restore it"
            ) from exc

        def extracted(fraction: float) -> None:
            if on_progress:
                on_progress(_EXTRACT_SHARE * min(fraction, 0.99))

        with extract_audio(source, progress=extracted) as audio_path:
            try:
                return _decode(model, audio_path, config, on_progress)
            except RuntimeError as exc:
                if _CUDA_LIBRARY_MISSING not in str(exc):
                    raise
                warnings.warn(
                    f"transcribe: {exc}, transcribing on the CPU", UserWarning, stacklevel=2
                )
                model = _load_model(config.model_size, config.compute_type, cpu=True)
                return _decode(model, audio_path, config, on_progress)

"""Stage 2: decode the source into timed words through a pluggable ASR backend.

The pipeline no longer imports a specific ASR engine here: this stage resolves
``config.asr_backend`` through :mod:`questxtract.asr` and converts whatever raw segments
the backend returns into a :class:`~questxtract.models.Transcript` via
``Transcript.from_dict``, which already handles word synthesis when a backend gives none.

Swapping the *engine* is a ``Config.asr_backend`` string plus one ``@register_backend``
class in ``asr.py`` - no change here. Swapping the *checkpoint* (e.g. a different Whisper
size) is already a ``Config.model_size`` string, unchanged by this issue.
"""

from __future__ import annotations

from .asr import resolve_backend
from .models import Context, Transcript, stage


@stage(
    "transcribe",
    cache="transcript",
    uses=(
        "asr_backend",
        "model_size",
        "language",
        "beam_size",
        "vad_filter",
        "compute_type",
        "batch_size",
        "initial_prompt",
    ),
)
def transcribe(ctx: Context) -> Context:
    """Decode ``ctx.source`` into a :class:`~questxtract.models.Transcript`.

    Raises:
        ValueError: If ``ctx.config.asr_backend`` names a backend that is not
            registered (see :func:`questxtract.asr.resolve_backend`).

    """

    def report(fraction: float) -> None:
        if ctx.progress:
            ctx.progress("transcribe", fraction)

    backend = resolve_backend(ctx.config.asr_backend)
    raw_segments = backend.transcribe(ctx.source, ctx.config, on_progress=report)
    ctx.transcript = Transcript.from_dict(
        {"segments": raw_segments, "language": ctx.config.language}
    )
    return ctx

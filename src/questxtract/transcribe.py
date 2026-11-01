"""Stage 2: decode the source into timed words through a pluggable ASR backend."""

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
    """Decode ``ctx.source`` into a :class:`~questxtract.models.Transcript`."""

    def report(fraction: float) -> None:
        if ctx.progress:
            ctx.progress("transcribe", fraction)

    backend = resolve_backend(ctx.config.asr_backend)
    raw_segments = backend.transcribe(ctx.source, ctx.config, on_progress=report)
    ctx.transcript = Transcript.from_dict(
        {"segments": raw_segments, "language": ctx.config.language}
    )
    return ctx

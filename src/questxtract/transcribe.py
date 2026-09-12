"""Stage 2: faster-whisper decodes the container directly into timed words.

One backend, no abstraction layer over it. Word timestamps are the only source of timing
in the pipeline (pipeline.md 2.3), so they are requested unconditionally.

Swapping the engine is a second ``Context -> Context`` function replacing this entry in
``STAGES``; swapping the *checkpoint* is a config string.
"""

from __future__ import annotations

from .models import Context, stage


@stage(
    "transcribe",
    cache="transcript",
    uses=(
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
    """Decode the source into a :class:`~questxtract.models.Transcript`.

    Raises:
        NotImplementedError: The ASR engine arrives with issue #8. Until then a caller
            seeds ``ctx.transcript`` from a transcript file and runs ``STAGES[2:]``,
            which is what the ``stages=`` seam is for (pipeline.md 11.1).

    """
    # ponytail: faster-whisper + BatchedInferencePipeline per pipeline.md 2.2; issue #8.
    raise NotImplementedError(
        "transcribe: ASR lands with issue #8; pass a transcript file to `questxtract process`"
    )

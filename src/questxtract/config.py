"""Every pipeline setting in one flat frozen record (pipeline.md 11.1).

Flat, not nested: that is what lets a stage's ``uses=(...)`` be a tuple of names rather
than a path expression. Stages read config only through ``ctx.config``; nothing else in
the package reads the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any

_PUNCTUATION_PROMPT = (
    "The following is a university lecture. Transcribe it with full punctuation, "
    "including question marks on every question asked."
)


@dataclass(frozen=True, kw_only=True)
class Config:
    """Pipeline settings, with every documented default in one place."""

    # --- 2.2 transcription ---
    asr_backend: str = "faster_whisper"
    model_size: str = "turbo"
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = True
    compute_type: str = "int8"
    batch_size: int = 8
    initial_prompt: str = _PUNCTUATION_PROMPT

    # --- 2.4 vision-informed second ASR pass ---
    hotwords_from_vision: bool = False

    # --- 3.1 sentences ---
    max_sentence_words: int = 60
    force_split_min_gap_seconds: float = 0.6

    # --- 3.2 segments ---
    segment_max_gap_seconds: float = 2.0
    segment_max_duration_seconds: float = 90.0
    segment_max_chars: int = 800

    # --- 6 vision ---
    vision_enabled: bool = False  # unless an endpoint is configured
    vision_fps: float = 1.0
    vision_model: str = "gpt-5.6-luna"
    vision_max_states: int = 80

    # --- 7 detection ---
    min_question_score: float = 0.45
    max_question_chars: int = 220

    # --- 8 answer search ---
    answer_window_sentences: int = 3
    deferred_min_question_score: float = 0.62
    deferred_window_sentences: int = 28

    # --- 9 scoring ---
    min_confidence: float = 0.40

    # --- 10 the LLM layer ---
    base_url: str | None = None
    judge_enabled: bool = False  # auto: true when an endpoint is reachable
    judge_model: str = "gpt-5.6-terra"
    judge_max_candidates: int = 60
    timeout_seconds: int = 120

    # --- 11.2 cache ---
    cache_dir: str = ".questxtract"
    from_scratch: bool = False

    @property
    def api_key(self) -> str | None:
        """The LLM API key, read from the environment and never stored in the record."""
        return os.environ.get("OPENAI_API_KEY")

    def replace(self, **changes: Any) -> Config:
        """Return a copy with ``changes`` applied.

        Raises:
            TypeError: If a name is not a config field, so a typo fails loudly rather
                than silently keeping the default.

        """
        known = {f.name for f in fields(self)}
        unknown = set(changes) - known
        if unknown:
            raise TypeError(f"unknown config field(s): {', '.join(sorted(unknown))}")
        return Config(**{**{f.name: getattr(self, f.name) for f in fields(self)}, **changes})

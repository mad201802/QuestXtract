"""Stage 9 and the vision calls: one module, one client, two jobs.

Off unless an endpoint is configured; the pipeline produces a complete result without it.
The default ``Config`` makes zero outbound calls.

The hard constraint from the requirements (4.5, "the model shall not create questions on
their own"): the judge may reject, rescope or rewrite a candidate, and it must cite the
sentence ids it is working from. It may never add one. Quiz generation is generative and
lives in a separate, separately named stage (pipeline.md 12.3), so the boundary is visible
in the code and in the output, not only in a prompt.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from .config import Config
from .models import Context, stage

# Prompts are module-level constants, version-tagged, so a prompt change is a reviewable diff.
JUDGE_PROMPT_VERSION = "v0"
VISION_PROMPT_VERSION = "v0"

VISION_PROMPT = (
    "You are shown one frame of a lecture recording: a slide, a blackboard or a whiteboard. "
    "Reply with a JSON object with exactly these keys: "
    '"description" (one or two sentences on what the board shows: a diagram, a derivation, '
    'a slide), "text" (the words written on it, verbatim), "latex" (its formulas as LaTeX, '
    'or null when there are none) and "confidence" (0 to 1, how legible the board was). '
    "What the lecturer said while it was on screen is given for context only; describe what "
    "is written, not what was said."
)


def client(config: Config) -> Any:
    """Build the OpenAI-compatible client.

    ``base_url`` is what makes the offline story work without a second code path: any local
    OpenAI-compatible server (vLLM, llama.cpp, Ollama) plugs in with zero new code.

    Raises:
        RuntimeError: If the optional ``questxtract[llm]`` extra is not installed.

    """
    try:
        from openai import OpenAI  # noqa: PLC0415 - imported lazily so the base install is light
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise RuntimeError("the LLM layer needs `pip install questxtract[llm]`") from exc
    return OpenAI(api_key=config.api_key or "local", base_url=config.base_url)


def describe_board(config: Config, image: bytes, transcript: str = "") -> dict[str, Any]:
    """Caption one board keyframe with the vision model (pipeline.md 6.5).

    This is the only place the vision stage talks to a model, so it is also the seam tests
    replace: ``tests/conftest.py`` monkeypatches it with a stub that returns canned JSON,
    which is how the vision tests run with no network and no key.

    Args:
        config: Pipeline settings; reads ``vision_model``, ``timeout_seconds`` and the
            endpoint fields ``client()`` reads.
        image: The keyframe, PNG-encoded.
        transcript: What was said while the board was on screen, as context.

    Returns:
        The model's reply: ``description``, ``text``, ``latex`` and ``confidence``. Keys are
        not validated here; the caller reads them defensively.

    Raises:
        ValueError: If the model replies with something other than a JSON object.

    """
    url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    response = client(config).chat.completions.create(
        model=config.vision_model,
        timeout=config.timeout_seconds,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VISION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Said while on screen: {transcript or '(nothing)'}"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
        ],
    )
    reply = json.loads(response.choices[0].message.content or "null")
    if not isinstance(reply, dict):
        raise ValueError("vision: the model did not reply with a JSON object")
    return reply


# ponytail: no `cache=` yet. Pipeline.md 10.4 keys the judge's entry on the *candidate set*
# hash, which the config-only `uses` header cannot express; caching on the source hash alone
# would replay a stale verdict after detection changed. Wire both together in one change.
@stage("judge", uses=("judge_model", "judge_max_candidates", "timeout_seconds"))
def judge(ctx: Context) -> Context:
    """Review the surviving candidates together in one batched call.

    A no-op unless ``config.judge_enabled``. The judge is never required: a timeout, a rate
    limit, a malformed response or a missing key logs a warning, marks every candidate
    ``unjudged``, and the run completes on rule scores alone (pipeline.md 10.3).
    """
    if not ctx.config.judge_enabled:
        return ctx
    # ponytail: one batched call, id-set validation, sentence-id citation check and the
    # confidence band of pipeline.md 10.2-10.4.
    raise NotImplementedError("judge: the LLM review pass is not wired up yet")

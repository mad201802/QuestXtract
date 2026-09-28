"""Stage 9 and the vision calls: one module, one client, two jobs.

Off unless an endpoint is configured; the pipeline produces a complete result without it.
The default ``Config`` makes zero outbound calls.

The hard constraint from the requirements (4.5, "the model shall not create questions on
their own"): the judge may reject, rescope or rewrite a candidate, and it must cite the
sentence ids it is working from. It may never add one. Quiz generation is generative and
lives in a separate, separately named stage (architecture.md 12.3), so the boundary is visible
in the code and in the output, not only in a prompt.
"""

from __future__ import annotations

from typing import Any

from .config import Config
from .models import Context, stage

# Prompts are module-level constants, version-tagged, so a prompt change is a reviewable diff.
JUDGE_PROMPT_VERSION = "v0"


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


# ponytail: no `cache=` yet. architecture.md 10.4 keys the judge's entry on the *candidate set*
# hash, which the config-only `uses` header cannot express; caching on the source hash alone
# would replay a stale verdict after detection changed. Wire both together in one change.
@stage("judge", uses=("judge_model", "judge_max_candidates", "timeout_seconds"))
def judge(ctx: Context) -> Context:
    """Review the surviving candidates together in one batched call.

    A no-op unless ``config.judge_enabled``. The judge is never required: a timeout, a rate
    limit, a malformed response or a missing key logs a warning, marks every candidate
    ``unjudged``, and the run completes on rule scores alone (architecture.md 10.3).
    """
    if not ctx.config.judge_enabled:
        return ctx
    # ponytail: one batched call, id-set validation, sentence-id citation check and the
    # confidence band of architecture.md 10.2-10.4.
    raise NotImplementedError("judge: the LLM review pass is not wired up yet")

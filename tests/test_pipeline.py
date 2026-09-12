import re
from pathlib import Path

import pytest

from questxtract import STAGES, Context, run, stage
from questxtract.pipeline import stage_info

SRC = Path(__file__).resolve().parents[1] / "src" / "questxtract"

# pipeline.md 1.3, rows 1 to 9, plus the vision/attach_visual split noted in 11.1 rule 4.
DOC_ORDER = [
    "ingest",
    "transcribe",
    "sentences",
    "segments",
    "vision",
    "attach_visual",
    "detect",
    "answer",
    "score",
    "judge",
]


def test_stages_match_the_documented_order():
    assert [stage_info(fn).name for fn in STAGES] == DOC_ORDER


def test_grep_at_stage_returns_the_contents_of_stages():
    """pipeline.md 13.1 makes this claim about the code; assert it rather than trust it."""
    names = [
        m
        for path in sorted(SRC.glob("*.py"))
        if path.name not in {"models.py", "pipeline.py"}
        for m in re.findall(r'@stage\(\s*"([a-z_]+)"', path.read_text(encoding="utf-8"))
    ]
    assert sorted(names) == sorted(DOC_ORDER)


def test_run_accepts_a_custom_stage_list(context):
    """The `stages=` keyword is the entire plugin system (pipeline.md 11.1)."""

    @stage("shout")
    def shout(ctx: Context) -> Context:
        ctx.stats["shouted"] = 1
        return ctx

    assert run(context, [shout]).stats == {"shouted": 1}


def test_progress_fires_once_per_stage(context):
    seen: list[str] = []
    run(context, STAGES[2:5], progress=lambda name, _: seen.append(name))
    assert seen == ["sentences", "segments", "vision", "done"]


def test_a_stage_needing_missing_input_fails_on_the_spot(context):
    context.transcript = None
    with pytest.raises(ValueError, match="no transcript"):
        run(context, STAGES[2:3])


def test_undecorated_functions_are_rejected(context):
    with pytest.raises(TypeError, match="not decorated"):
        run(context, [lambda ctx: ctx])


def test_optional_stages_no_op_themselves(context):
    """pipeline.md 11.1 rule 3: the runner stays dumb."""
    from questxtract.llm import judge
    from questxtract.vision import vision

    assert vision(context) is context
    assert judge(context) is context


def test_cache_reuses_on_a_matching_header_and_recomputes_on_a_changed_one(tmp_path, context):
    calls = []

    @stage("counted", cache="visual", uses=("vision_fps",))
    def counted(ctx: Context) -> Context:
        calls.append(ctx.config.vision_fps)
        ctx.visual = ()
        return ctx

    context.source_hash = "abc123"
    context.config = context.config.replace(cache_dir=str(tmp_path))
    run(context, [counted])
    run(context, [counted])
    assert calls == [1.0], "second run should have hit the cache"

    context.config = context.config.replace(vision_fps=2.0)
    run(context, [counted])
    assert calls == [1.0, 2.0], "a changed `uses` value must invalidate"


def test_from_scratch_bypasses_the_cache(tmp_path, context):
    calls = []

    @stage("counted2", cache="visual", uses=())
    def counted2(ctx: Context) -> Context:
        calls.append(1)
        ctx.visual = ()
        return ctx

    context.source_hash = "abc123"
    context.config = context.config.replace(cache_dir=str(tmp_path), from_scratch=True)
    run(context, [counted2])
    run(context, [counted2])
    assert calls == [1, 1]


def test_cache_roundtrips_dataclasses(tmp_path, context):
    from questxtract import VisualContext

    state = VisualContext(start=1.0, end=2.0, description="a board", latex=None)

    @stage("captured", cache="visual", uses=())
    def captured(ctx: Context) -> Context:
        ctx.visual = (state,)
        return ctx

    context.source_hash = "abc123"
    context.config = context.config.replace(cache_dir=str(tmp_path))
    run(context, [captured])

    context.visual = ()
    assert run(context, [captured]).visual == (state,)


def test_unimplemented_stages_say_which_issue_owns_them(context):
    from questxtract.transcribe import transcribe

    with pytest.raises(NotImplementedError, match="#8"):
        transcribe(context)

"""Stage 5: optional enrichment of the audio with what is on the board.

Three stages. ``layout`` asks the model once where the content is and what it is (6.2), so
``vision`` can crop to it and tell the captioning prompt which kind of frame it reads.

Capture and the join are two stages, not one, because of architecture.md 11.1 rule 4:
capture is expensive and caches its views in ``views.json`` and each reading per request,
while the interval join is eight lines and always recomputes. Nailing one board state to one
segment at capture time would bake in a wrong join that every consumer inherits.

All three no-op themselves when vision is off, rather than being filtered out by the runner
(rule 3).
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import av
import numpy as np

from . import endpoint
from .config import Config
from .models import (
    Context,
    Layout,
    Progress,
    Segment,
    Sentence,
    VisualContext,
    overlapping,
    stage,
)

_COMPARE_HEIGHT = 480  # architecture.md 6.3: change detection compares frames at about 480p
_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # ITU-R BT.601 luma weights
_NOISE = 32  # grey levels a pixel must move by to count as changed, above codec noise

# A page that comes back reuses its earlier reading (architecture.md 6.5). The detail limits are
# about twice the codec noise between frames of one page in TTK4225 2026-09-14 (#138).
_REUSE_KINDS = ("slides", "handwriting", "screencast")  # boards are #133's; unknown may be one
_REUSE_CANDIDATE = 0.01  # share of the small copies' pixels that may differ for a candidate
# The detail check compares the mean brightness of squares at full resolution: (side in px, step
# in px between square starts, most the mean may move on the same page). The small squares find a
# decimal point, the large ones a broad faint mark; codec noise is larger in the small ones.
_DETAIL_SQUARES = ((4, 2, 20.0), (8, 4, 11.0))
_DETAIL_COLOUR = 12.0  # most an 8 px square's mean colour (Cb, Cr) may move on the same page
_YCBCR = np.array(  # ITU-R BT.601 RGB to Y, Cb and Cr, without offsets: only differences matter
    [[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]], dtype=np.float32
)

# The previous board state goes with each reading (architecture.md 6.5, #137). Ink is #133's rule,
# on the small grey copy: a pixel that differs from the mean of the square around it.
_INK_WINDOW = 15  # px of the small copy: the side of that square
_INK_CONTRAST = 20  # grey levels from that mean that make a pixel ink
# Share of the previous keyframe's ink still there, or the board is new: an erase, a camera cut or
# a page flip. On the #58 frames one board filling up keeps 0.99, different boards 0.14 to 0.46,
# and two pages under one app toolbar 0.62; a missed previous state only costs the context. On
# consecutive views (tests/data/vision/sequence/, #230) a board filling up keeps 0.94 to 0.97 and
# a partial erase 0.79 to 0.94, so no limit tells them apart; the prompt's rule not to copy does.
_INK_KEPT = 0.7
_NEW_INK_CELL = 8  # px: new ink is boxed by the cells of this side where at least
_NEW_INK_SHARE = 0.2  # this share of pixels is new ink, so codec noise and a cursor make no box

# The prompt and the schema are module-level, version-tagged, so a prompt change is a reviewable
# diff. Both are part of every request, so a change to either asks again (endpoint.ask's cache).
# tests/data/vision/prompt_v2.txt: v0 (chosen in #52) with the previous board state. A state
# without one is sent v0 word for word.
VISION_PROMPT_VERSION = "v2"
_SPEECH_CHARS = 1500  # the speech sent with one keyframe: its own view's words, kept short
_GIVE_UP_AFTER = 3  # failed readings in a row, before any success, that stop the captioning

_FIELDS = ("description", "text", "latex", "confidence")

# The OCR reader (architecture.md 6.5): GLM-OCR's task prompt, as measured in #52 and #143.
_OCR_PROMPT = "Text Recognition:"
_OCR_UNMEASURED = 0.5  # confidence when the server returns no token probabilities
# ``$$...$$`` may span lines; a single ``$...$`` stays on one line, opens before and closes
# after a non-space, and does not close before a digit, so a price ("$5 and $10") is no formula.
_OCR_FORMULA = re.compile(r"\$\$(.+?)\$\$|\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)", re.DOTALL)
_OCR_CIRCLED = re.compile(r"\\textcircled\{([^}]*)\}")
_OCR_COMMAND = re.compile(r"\\(?:mathrm|text|mathbf|mathit)\{([^}]*)\}|\\([A-Za-z]+)")
# Endpoints that refused ``logprobs`` in this process: asked without it from then on.
_NO_LOGPROBS: set[str | None] = set()
# Formula segments that are labels, not formulas: a circled number, a number with a unit,
# or a single symbol (a variable named in a sentence).
_OCR_LABEL = re.compile(r"\\textcircled\{[^}]*\}|[\d.,\s]*\\mathrm\{[^}]*\}|\\[A-Za-z]+|\S")

_PROMPT = """\
You read one frame from a lecture recording: a slide, a blackboard, a whiteboard or tablet notes.
Describe what is visible on the board or slide, for someone who cannot see it.

Frame kind: {kind}

What the lecturer said while this frame was on screen (context only):
{speech}

Rules:
- Report only what is visible in the image. The speech helps you read unclear writing, but never
  copy words, formulas or claims from it that are not on the board.
- "text": every piece of text on the board or slide, handwritten or printed, in reading order,
  one line per line on the board. Keep the wording and spelling you see. Mark a word you cannot
  read as [unreadable]. Empty string when there is no text.
- "latex": the formulas on the board as LaTeX, one formula per line. null when there is no
  formula. Do not write LaTeX for plain words.
- "description": one or two sentences on what the frame shows: the slide or board topic, and any
  diagram, graph or figure (its parts and how they connect). No text transcription here.
- "confidence": a number from 0 to 1 for how sure you are of "text" and "latex" together. Use a
  low value for blur, glare, a lecturer blocking the board, or handwriting you had to guess.
- Ignore the lecturer, the room, the camera and any on-screen interface.
- An empty or blank board is a valid answer: empty "text", null "latex", a short "description"
  saying so, confidence 1 if you are sure it is blank.

Answer with one JSON object and nothing else: \
{"description": ..., "text": ..., "latex": ..., "confidence": ...}\
"""

# Prompt v2 is v0 with these two parts added, so the diff to v0 is only them.
_PREVIOUS = """\
The board state before this one, as read from its own frame (context only):
Description: {previous_description}
Text:
{previous_text}
LaTeX:
{previous_latex}
Where new writing appeared since then: {new_ink}

"""
_PREVIOUS_RULE = """\
- The previous board state helps you keep names and notation the same and see what is new. It is
  not this frame: report everything visible in this frame, also what was there before, and never
  copy anything from the previous state that is no longer visible.
"""
_V0_FIRST_RULE = "  copy words, formulas or claims from it that are not on the board.\n"
_PROMPT_V2 = _PROMPT.replace("Rules:\n", _PREVIOUS + "Rules:\n").replace(
    _V0_FIRST_RULE, _V0_FIRST_RULE + _PREVIOUS_RULE
)
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")  # not the JSON example: its braces hold quotes

_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "board",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "text": {"type": "string"},
                "latex": {"type": ["string", "null"]},
                "confidence": {"type": "number"},
            },
            "required": list(_FIELDS),
            "additionalProperties": False,
        },
    },
}

Frame = np.ndarray[Any, np.dtype[np.uint8]]
Ink = np.ndarray[Any, np.dtype[np.bool_]]  # where a small grey copy has ink (_ink)
Crop = tuple[float, float, float, float]  # normalised (x0, y0, x1, y1), from the layout probe

_FULL_FRAME: Crop = (0.0, 0.0, 1.0, 1.0)
_KINDS = ("slides", "blackboard", "whiteboard", "handwriting", "screencast", "unknown")
_PROBE_AT = (0.2, 0.5, 0.8)  # where in the recording the probe's three frames are taken
_PROBE_WIDTH = 640  # the region is in fractions, so a small frame shows it as well
_REGION_PAD = 0.02  # VLM boxes are loose; a margin keeps the board's edges in the crop
_REGION_MIN_AREA = 0.1  # a smaller box is more likely a misreading than the content

_LAYOUT_PROMPT = """\
You see three frames taken from across one lecture recording. Find where the lecture content is:
the slides, the board or the written notes.

Rules:
- "content_region": the box around the content as [x0, y0, x1, y1], fractions of the frame's
  width and height measured from its top-left corner, each from 0 to 1. Leave out the lecturer's
  camera, the room around the board, and on-screen interface such as toolbars, a video-call
  window or a taskbar. Give one box that holds the content in all three frames. If the content
  fills the frame, answer [0, 0, 1, 1].
- "kind": what the content is.
  "slides": prepared slides with printed text and figures.
  "blackboard": chalk on a blackboard.
  "whiteboard": marker on a whiteboard.
  "handwriting": writing on a tablet or on paper under a camera, also over slides.
  "screencast": a computer screen with code, a terminal or an application.
  "unknown": none of these, or you cannot tell.

Answer with one JSON object and nothing else: {"content_region": [x0, y0, x1, y1], "kind": ...}\
"""

_LAYOUT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "layout",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "content_region": {"type": "array", "items": {"type": "number"}},
                "kind": {"type": "string", "enum": list(_KINDS)},
            },
            "required": ["content_region", "kind"],
            "additionalProperties": False,
        },
    },
}


@stage("layout")
def layout(ctx: Context) -> Context:
    """Find where the content is and what kind it is, once per recording (architecture.md 6.2).

    A no-op unless ``config.vision_enabled``, and for a source with no video track. Three
    frames from across the recording go to ``vision_model`` in one request; the answer sets
    ``ctx.layout``. Any failure of the endpoint, the model or the recording leaves the full
    frame and ``unknown`` with a warning: the probe only sharpens vision, so it never ends the
    run. Not stage-cached: the request is cached by :func:`endpoint.ask`, which keeps only a
    valid answer, so a probe that failed is asked again on the next run.

    Raises:
        RuntimeError: If the optional ``questxtract[llm]`` extra is not installed.
        ValueError: If the settings cannot be used, such as an unreadable ``.env``.

    """
    if not ctx.config.vision_enabled or not has_video(ctx.source):
        return ctx
    try:
        frames = _probe_frames(ctx.source)
    except (av.error.FFmpegError, OSError) as exc:
        return _fallback(ctx, f"the recording could not be read: {exc}")
    if not frames:
        return _fallback(ctx, "the recording has no frame to show")
    images = [{"type": "image_url", "image_url": {"url": _data_url(png)}} for png in frames]
    request: dict[str, Any] = {
        "model": ctx.config.vision_model,
        "messages": [
            {"role": "system", "content": _LAYOUT_PROMPT},
            {"role": "user", "content": images},
        ],
        "response_format": _LAYOUT_SCHEMA,
    }
    if ctx.config.vision_effort:
        request["reasoning_effort"] = ctx.config.vision_effort
    usage: Counter[str] = Counter()
    ctx.stats.update(dict.fromkeys(_cost_keys("layout"), 0))
    try:
        answer = endpoint.ask(ctx.config, request, name="layout", usage=usage, vision=True)
    except (ConnectionError, endpoint.InvalidAnswer) as exc:
        return _fallback(ctx, str(exc))
    finally:
        ctx.stats.update(usage)
    found = _layout_from(answer)
    if found is None:
        return _fallback(ctx, f"{ctx.config.vision_model} answered with a malformed layout")
    ctx.layout = found
    return ctx


def _fallback(ctx: Context, why: str) -> Context:
    warnings.warn(
        f"vision: layout probe failed, using the full frame: {why}", UserWarning, stacklevel=3
    )
    ctx.layout = Layout()
    return ctx


def _probe_frames(path: Path) -> list[bytes]:
    """Take the frames at ``_PROBE_AT`` of the recording, as PNG, ``_PROBE_WIDTH`` wide at most.

    Seeks to each one rather than decoding the whole recording. Without a known duration,
    only the first frame.
    """
    pngs: list[bytes] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = container.duration / av.time_base if container.duration else 0.0
        for fraction in _PROBE_AT if duration else (0.0,):
            target = duration * fraction
            container.seek(int(target * av.time_base))  # to the keyframe at or before it
            for frame in container.decode(stream):
                if frame.time is not None and frame.time + 1e-6 < target:
                    continue
                width = min(frame.width, _PROBE_WIDTH)
                height = round(frame.height * width / frame.width / 2) * 2 or 2
                picture: Frame = frame.reformat(width=width, height=height).to_ndarray(
                    format="rgb24"
                )
                pngs.append(_png(picture))
                break
    return pngs


def _layout_from(answer: dict[str, Any]) -> Layout | None:
    """Return the :class:`Layout` an answer gives, or ``None`` when it is malformed.

    The box is padded by ``_REGION_PAD`` on each side, within the frame. A box smaller than
    ``_REGION_MIN_AREA`` of the frame is not trusted: the kind is kept, the crop is not.
    """
    region, kind = answer.get("content_region"), answer.get("kind")
    if kind not in _KINDS or not isinstance(region, list) or len(region) != 4:
        return None
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in region
    ):
        return None
    x0, y0, x1, y1 = (float(v) for v in region)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        return None
    if (x1 - x0) * (y1 - y0) < _REGION_MIN_AREA:
        return Layout(kind=str(kind))
    padded = (
        max(0.0, x0 - _REGION_PAD),
        max(0.0, y0 - _REGION_PAD),
        min(1.0, x1 + _REGION_PAD),
        min(1.0, y1 + _REGION_PAD),
    )
    return Layout(content_region=padded, kind=str(kind))


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


@stage("vision")
def vision(ctx: Context) -> Context:
    """Capture board states from the recording into ``ctx.visual`` (architecture.md 6.3 to 6.5).

    A no-op unless ``config.vision_enabled``, and for a source with no video track (an MP3, or
    a transcript): there is no board to read. Otherwise the recording is sampled at
    ``vision_fps`` and cut into stable views, one keyframe each; each keyframe is read by
    ``vision_model`` together with what was said while its view was on screen (or, with
    ``vision_reader = "ocr"``, by ``vision_ocr_model`` alone: :func:`_read_ocr`), and stored in
    the cache. A reading that fails leaves that one view out with a warning; when the first
    readings all fail, the rest are not sent. Frames are cropped to the ``layout`` stage's
    ``content_region`` and its ``kind`` goes into the prompt; one answer holds for the whole
    recording. On slides, handwriting and screencasts, a page that comes back with the same
    content reuses its earlier reading in this run, with no model call (:func:`_earlier`),
    counted in ``stats.vision_reused``. Each reading gets the view before it, read or reused,
    as its previous board state, unless the board is new (:func:`_previous`). De-occlusion is
    not built.
    """
    if not ctx.config.vision_enabled or not has_video(ctx.source):
        return ctx
    config, report = ctx.config, ctx.progress
    views = _cut(ctx)
    reuse = ctx.layout.kind in _REUSE_KINDS
    carry = config.vision_reader != "ocr"  # the OCR reader has no prompt for a previous state
    readings: list[_Reading] = []
    visual: list[VisualContext] = []
    before: tuple[Ink, dict[str, Any]] | None = None  # the view just before: its ink, reading
    failed = reused = 0
    usage: Counter[str] = Counter()  # what the readings cost (#123)
    for index, view in enumerate(views):
        if report:
            report("vision", 0.5 + 0.5 * index / len(views))
        picture = _decoded(view.keyframe) if reuse or carry else None
        small = None if picture is None else _for_comparison(picture)
        ink = _ink(small) if carry and small is not None else None
        fields = _earlier(picture, small, readings, views) if reuse else None
        if fields is not None:
            reused += 1
        else:
            speech = _speech(ctx.sentences, view.start, view.end)
            fields = read_keyframe(
                config,
                view.keyframe,
                speech,
                kind=ctx.layout.kind,
                usage=usage,
                previous=_previous(before, ink),
            )
            if fields is None:
                failed += 1
                before = None
                if not visual and failed >= _GIVE_UP_AFTER:
                    warnings.warn(
                        f"vision: the first {failed} keyframes could not be read, so the rest "
                        "are not sent; check vision_model or vision_ocr_model, the API key and "
                        "the endpoint",
                        UserWarning,
                        stacklevel=2,
                    )
                    break
                continue
            if reuse and small is not None:
                readings.append(_Reading(small=small, view=index, fields=fields))
        before = None if ink is None else (ink, fields)
        path = _store(view.keyframe, config, ctx.source_hash)
        visual.append(VisualContext(start=view.start, end=view.end, keyframe_path=path, **fields))
    ctx.visual = tuple(visual)
    ctx.stats["vision_failed"] = len(views) - len(visual)
    ctx.stats["vision_reused"] = reused
    ctx.stats.update({**dict.fromkeys(_cost_keys("vision"), 0), **usage})
    return ctx


def _cost_keys(name: str) -> tuple[str, ...]:
    """Return the ``stats`` keys :func:`endpoint.ask` counts for ``name``, even when 0."""
    return tuple(f"{name}_{key}" for key in ("requests", "input_tokens", "output_tokens", "ms"))


@dataclass(frozen=True, kw_only=True)
class _Reading:
    """A reading that worked in this run, kept so that a page coming back can reuse it.

    Args:
        small: The small grey copy of its keyframe (:func:`_for_comparison`); the full
            keyframe is not held here, it stays in the views.
        view: Which view it read, by index.
        fields: What the model read: the :class:`VisualContext` fields.

    """

    small: Frame
    view: int
    fields: dict[str, Any]


def _earlier(
    picture: Frame | None,
    small: Frame | None,
    readings: Sequence[_Reading],
    views: Sequence[_View],
) -> dict[str, Any] | None:
    """Return the reading of an earlier keyframe that shows the same page, or ``None``.

    Two steps, because the views' measure alone misses a small change: a new digit or a
    decimal point is a few hundredths of a percent of the pixels, and a change of colour at the
    same brightness is none. The reading whose small copy differs least, and by at most
    ``_REUSE_CANDIDATE``, is the candidate; it is reused only when :func:`_same_page` passes at
    full resolution, in colour. An earlier keyframe that cannot be decoded is not reused.
    """
    if picture is None or small is None:
        return None
    candidates = [r for r in readings if r.small.shape == small.shape]
    if not candidates:
        return None
    closest = min(candidates, key=lambda r: _changed(r.small, small))
    if _changed(closest.small, small) > _REUSE_CANDIDATE:
        return None
    before = _decoded(views[closest.view].keyframe)
    if before is None or before.shape != picture.shape or not _same_page(before, picture):
        return None
    return closest.fields


def _same_page(before: Frame, after: Frame) -> bool:
    """Whether no square of the picture changed its mean brightness or colour beyond codec noise.

    Each size in ``_DETAIL_SQUARES`` has its own limit, and its squares overlap, so a mark lies
    whole in at least one: a decimal point in a small one, a broad faint mark in a large one. A
    mean, rather than single pixels, averages the codec's ringing at the edges away. Colour (Cb,
    Cr) has its own limit, so red turned to green of the same brightness is a change. The last
    rows and columns are repeated up to a whole square, so a change at the border is compared
    too. A mark fainter than the codec noise is not told apart: the codec mostly erases it.
    """
    change = (before.astype(np.float32) - after.astype(np.float32)) @ _YCBCR.T
    largest = max(side for side, _, _ in _DETAIL_SQUARES)
    border = ((0, -change.shape[0] % largest), (0, -change.shape[1] % largest), (0, 0))
    change = np.pad(change, border, mode="edge")
    for side, step, limit in _DETAIL_SQUARES:
        means = _square_means(change, side, step)
        if means[..., 0].max() > limit:
            return False
    return bool(_square_means(change[..., 1:], largest, largest // 2).max() <= _DETAIL_COLOUR)


def _square_means(change: np.ndarray[Any, Any], side: int, step: int) -> np.ndarray[Any, Any]:
    """Absolute mean of every ``side`` px square starting on a multiple of ``step`` px.

    ``change`` is height x width x channels, its height and width multiples of ``side``.
    Summed from ``step`` px cells, so the overlapping squares cost little.
    """
    height, width, channels = change.shape[0] // step, change.shape[1] // step, change.shape[2]
    cells = change.reshape(height, step, width, step, channels).sum(axis=(1, 3))
    span = side // step
    rows, columns = height - span + 1, width - span + 1
    squares = sum(cells[i : i + rows, j : j + columns] for i in range(span) for j in range(span))
    means: np.ndarray[Any, Any] = np.abs(np.asarray(squares)) / side**2
    return means


def _decoded(png: bytes) -> Frame | None:
    """Return the RGB picture in a keyframe PNG, or ``None`` when it cannot be decoded."""
    try:
        codec = av.CodecContext.create("png", "r")
        frames = [frame for packet in codec.parse(png) for frame in codec.decode(packet)]
    except (av.error.FFmpegError, ValueError):
        return None
    if not frames or not isinstance(frames[0], av.VideoFrame):
        return None
    return np.asarray(frames[0].to_ndarray(format="rgb24"), dtype=np.uint8)  # rgb24 is uint8


@dataclass(frozen=True, kw_only=True)
class PreviousState:
    """The board state before the one being read, as the captioning prompt gets it (6.5).

    Args:
        description: Its reading's ``description``.
        text: Its reading's ``text``.
        latex: Its reading's ``latex``, or ``None``.
        new_ink: The box around the ink that is new since, as fractions ``(x0, y0, x1, y1)`` of
            the keyframe's width and height from its top-left corner; ``None`` when none was
            found.

    """

    description: str
    text: str
    latex: str | None
    new_ink: Crop | None = None


def _previous(before: tuple[Ink, dict[str, Any]] | None, now: Ink | None) -> PreviousState | None:
    """Return the previous board state to send with a keyframe, or ``None`` when the board is new.

    ``before`` is the view just before: the ink of its small grey copy (:func:`_ink`) and its
    reading, ``None`` when it was not read; ``now`` is this keyframe's ink. The board is new when
    less than ``_INK_KEPT`` of the earlier keyframe's ink is still ink in this one
    (:func:`_ink_kept`): an erase, a camera cut or a page flip. A blank earlier board has nothing
    to carry over, and copies of different sizes cannot be compared, so neither gives a previous
    state.
    """
    if before is None or now is None:
        return None
    was, fields = before
    if _ink_kept(was, now) < _INK_KEPT:
        return None
    return PreviousState(
        description=fields["description"],
        text=fields["text"],
        latex=fields["latex"],
        new_ink=_new_ink(now & ~was),
    )


def _ink_kept(was: Ink, now: Ink) -> float:
    """Return the share of the earlier ink ``was`` that is still ink ``now``.

    0 for a blank earlier board, and for copies of different sizes, which cannot be compared.
    """
    if was.shape != now.shape or not was.any():
        return 0.0
    return float((was & now).sum() / was.sum())


def _ink(small: Frame) -> Ink:
    """Where a small grey copy has ink: pixels more than ``_INK_CONTRAST`` from their local mean.

    The mean is over the ``_INK_WINDOW`` square around each pixel, the edges repeated, from
    summed areas, so the window costs the same at any size.
    """
    side, height, width = _INK_WINDOW, small.shape[0], small.shape[1]
    gray = small.astype(np.float64)
    sums = np.pad(np.pad(gray, side // 2, mode="edge").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    window = (
        sums[side : side + height, side : side + width]
        - sums[:height, side : side + width]
        - sums[side : side + height, :width]
        + sums[:height, :width]
    )
    ink: Ink = np.abs(gray - window / side**2) > _INK_CONTRAST
    return ink


def _new_ink(new: Ink) -> Crop | None:
    """Return the box around the cells mostly of new ink, as fractions, or ``None`` when none.

    The cells are ``_NEW_INK_CELL`` px squares; one counts when at least ``_NEW_INK_SHARE`` of
    its pixels are new ink, so scattered codec noise draws no box. Rounded to two decimals, so
    the request, and with it the cache key, does not move with noise.
    """
    cell = _NEW_INK_CELL
    height, width = new.shape[0] // cell * cell, new.shape[1] // cell * cell
    share = new[:height, :width].reshape(height // cell, cell, width // cell, cell).mean((1, 3))
    rows, columns = np.nonzero(share >= _NEW_INK_SHARE)
    if not len(rows):
        return None
    return (
        round(int(columns.min()) * cell / new.shape[1], 2),
        round(int(rows.min()) * cell / new.shape[0], 2),
        round((int(columns.max()) + 1) * cell / new.shape[1], 2),
        round((int(rows.max()) + 1) * cell / new.shape[0], 2),
    )


def _cut(ctx: Context) -> list[_View]:
    """Sample the recording and cut it into views, cached on the sampling settings.

    Frames are cropped to the ``layout`` stage's ``content_region``. Sampling decodes the
    whole video, minutes for a lecture, while the views depend only on the recording, the
    crop and these settings, not on the transcript or the model. Entries live in
    ``cache_dir/<source_hash>/views.json``, keyframes beside them (:func:`_store`); without a
    source hash nothing is cached, as for the runner's own cache (architecture.md 11.2).
    """
    config, report = ctx.config, ctx.progress
    region = ctx.layout.content_region
    crop = None if region == _FULL_FRAME else region
    limit = state_limit(config, ctx.duration)
    header = {
        "content_region": list(region),
        "vision_fps": config.vision_fps,
        "vision_change_fraction": config.vision_change_fraction,
        "vision_change_hold": config.vision_change_hold,
        "max_views": limit,
        "duration": ctx.duration,
    }
    path = Path(config.cache_dir) / ctx.source_hash / "views.json"
    if ctx.source_hash and not config.from_scratch:
        cached = _load_views(path, header, Path(config.cache_dir))
        if cached is not None:
            return cached

    def sampling(_: str, fraction: float) -> None:
        if report:
            report("vision", fraction / 2)  # sampling is the first half of the stage

    samples = _sample_frames(ctx.source, config.vision_fps, crop=crop, progress=sampling)
    views = _views(
        _from_audio_start(samples, _audio_start(ctx.source)),
        change=config.vision_change_fraction,
        hold=config.vision_change_hold,
        max_views=limit,
        end=ctx.duration or None,
    )
    if ctx.source_hash:
        entries = []
        for view in views:
            stored = _store(view.keyframe, config, ctx.source_hash)
            if not stored:
                return views  # the cache cannot be written: keep the views, cache nothing
            entries.append(
                {
                    "start": view.start,
                    "end": view.end,
                    "keyframe": stored,
                    "keyframe_time": view.keyframe_time,
                }
            )
        try:
            path.write_text(json.dumps({"config": header, "views": entries}), encoding="utf-8")
        except OSError:
            pass
    return views


def _from_audio_start(
    samples: Iterable[tuple[float, Frame]], origin: float
) -> Iterator[tuple[float, Frame]]:
    """Shift sample times so that zero is the audio start, the transcript's clock.

    What was on screen before the audio starts is never spoken about, so those samples are
    dropped, except the last one: the board visible when the audio starts, at time zero. Every
    other sample keeps its own, later time, so no view can end where it starts.
    """
    before: Frame | None = None
    for time, frame in samples:
        shifted = time - origin
        if shifted < 0:
            before = frame
            continue
        if before is not None:
            if shifted > 0:
                yield 0.0, before
            before = None
        yield shifted, frame
    if before is not None:  # the picture ends before the audio starts
        yield 0.0, before


def state_limit(config: Config, duration: float) -> int:
    """Return how many board states a recording of ``duration`` seconds may have.

    ``vision_states_per_minute`` per minute, and never fewer than ``vision_min_states``, so the
    number of model calls grows with the lecture instead of a fixed cap merging more and more
    of a long lecture away, while a short one keeps the old 80 (architecture.md 6.4). An
    unknown duration (0) gets the minimum.

    Raises:
        ValueError: If ``vision_states_per_minute`` or ``vision_min_states`` is not positive.

    """
    if config.vision_states_per_minute <= 0 or config.vision_min_states < 1:
        raise ValueError("vision: vision_states_per_minute and vision_min_states must be positive")
    scaled = math.ceil(duration / 60 * config.vision_states_per_minute - 1e-9)
    return max(config.vision_min_states, scaled)


def _load_views(path: Path, header: dict[str, Any], cache: Path) -> list[_View] | None:
    """Read cached views, or ``None`` when they are missing, stale or damaged.

    A keyframe is named by the start of its content hash (:func:`_store`), so one whose bytes
    no longer match its name, such as a write cut short, makes the whole entry stale.
    """
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("config") != header:
            return None
        views = []
        for entry in stored["views"]:
            keyframe = (cache / entry["keyframe"]).read_bytes()
            if _content_name(keyframe) != PurePosixPath(entry["keyframe"]).name:
                return None
            views.append(
                _View(
                    start=entry["start"],
                    end=entry["end"],
                    keyframe=keyframe,
                    keyframe_time=entry["keyframe_time"],
                )
            )
        return views
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _audio_start(path: Path) -> float:
    """When the audio starts on the container's clock, in seconds: the transcript's zero.

    Word times count from the first audio sample, while frame times are the video stream's
    own, so a file whose streams do not start at zero (a trimmed recording) would put board
    states beside the wrong speech. Subtracting the audio start keeps a real offset between
    sound and picture, which zeroing the video stream alone would remove.
    """
    try:
        with av.open(str(path)) as container:
            audio = container.streams.audio
            if not audio or audio[0].start_time is None or audio[0].time_base is None:
                return 0.0
            return float(audio[0].start_time * audio[0].time_base)
    except (av.error.FFmpegError, OSError):
        return 0.0


def _speech(sentences: Sequence[Sentence], start: float, end: float) -> str:
    """Return what was said while a view was on screen, cut to ``_SPEECH_CHARS`` at a word."""
    text = " ".join(s.text.strip() for s in sentences if s.start < end and s.end > start)
    if len(text) <= _SPEECH_CHARS:
        return text
    return text[:_SPEECH_CHARS].rsplit(" ", 1)[0] + " ..."


def _store(png: bytes, config: Config, source_hash: str) -> str:
    """Write a keyframe into the cache; return its path relative to ``cache_dir`` (#59).

    Named by its content, so a re-run writes nothing new, and a file whose bytes do not match
    its name is written again. Written through a temporary file, so an interrupted write
    leaves no half image under the real name. Relative to the cache, so a cached result still
    finds its images from another working directory and board-state ids do not depend on where
    the cache lives. ``""`` when the cache cannot be written: the reading is kept, only the
    image is missing.
    """
    name = _content_name(png)
    relative = PurePosixPath(source_hash or "unhashed", "keyframes", name)
    path = Path(config.cache_dir) / relative
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or _content_name(path.read_bytes()) != name:
            partial = path.with_suffix(".tmp")
            partial.write_bytes(png)
            partial.replace(path)
    except OSError:
        return ""
    return relative.as_posix()


def _content_name(png: bytes) -> str:
    """Return a keyframe's file name: the start of its SHA-256, so the name checks the content."""
    return f"{hashlib.sha256(png).hexdigest()[:16]}.png"


def has_video(path: Path) -> bool:
    """Whether ``path`` is a recording with a video track; ``False`` for anything else."""
    try:
        with av.open(str(path)) as container:
            return bool(container.streams.video)
    except (av.error.FFmpegError, OSError):
        return False


def _sample_frames(
    path: Path, fps: float, *, crop: Crop | None = None, progress: Progress | None = None
) -> Iterator[tuple[float, Frame]]:
    """Yield ``(timestamp, frame)`` at ``fps`` from the first video stream (architecture.md 6.3).

    Each frame is the full-colour RGB picture, cropped to ``crop`` when given, so a frame that
    ends up a keyframe can go to the model as is; :func:`_for_comparison` makes the small copy.
    Streams: one decoded frame at a time, never the whole video. Every frame is decoded and
    only the sampled ones are converted, since seeking saves nothing at one sample a second.
    A sample is the first frame at or after each multiple of ``1 / fps``, so timestamps do
    not drift. Close the generator if you stop early: it holds the file open.

    Raises:
        ValueError: If ``fps`` is not positive, ``crop`` is not inside the frame, or the
            recording has no video stream (on the first ``next()``).

    """
    if fps <= 0:
        raise ValueError("vision: vision_fps must be positive")
    if crop is not None:
        x0, y0, x1, y1 = crop
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"vision: crop {crop} is not a box inside the frame")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: recording contains no video track")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # decoding is the cost; let FFmpeg use every core
        duration = container.duration / av.time_base if container.duration else 0.0
        due = 0
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-6 < due / fps:
                continue
            due = int(frame.time * fps + 1e-6) + 1
            picture: Frame = frame.to_ndarray(format="rgb24")
            if crop is not None:
                height, width = picture.shape[:2]
                picture = picture[
                    round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)
                ]
            if progress and duration:
                progress("vision", min(1.0, frame.time / duration))
            yield frame.time, picture


def _for_comparison(frame: Frame) -> Frame:
    """Grayscale, downscaled by a whole factor to about 480 rows, for change detection.

    Block averaging rather than dropping pixels also smooths codec noise between frames.
    """
    gray = frame @ _GRAY
    factor = max(1, round(gray.shape[0] / _COMPARE_HEIGHT))
    height, width = gray.shape[0] // factor * factor, gray.shape[1] // factor * factor
    blocks = gray[:height, :width].reshape(height // factor, factor, width // factor, factor)
    small: Frame = blocks.mean(axis=(1, 3)).round().astype(np.uint8)
    return small


@dataclass(frozen=True, kw_only=True)
class _View:
    """One stable view: a slide or board as it stood, with the keyframe that shows it.

    Args:
        start: First sample of the view, in seconds.
        end: Where the next view starts, or the end of the last sample.
        keyframe: The view's sharpest sample, PNG-encoded at full resolution.
        keyframe_time: When that sample was taken, in seconds.

    """

    start: float
    end: float
    keyframe: bytes
    keyframe_time: float


def _views(
    samples: Iterable[tuple[float, Frame]],
    *,
    change: float,
    hold: int,
    max_views: int,
    end: float | None = None,
) -> list[_View]:
    """Cut sampled frames into stable views, one keyframe each (architecture.md 6.4).

    A sample belongs to the current view while at most ``change`` of its pixels differ from
    the view's first sample. A new view starts at the first differing sample, but only once
    ``hold`` samples in a row differ from the old view and resemble each other: a person
    walking past never resembles themselves a second later, so they start nothing. Of the
    samples that match a view, the sharpest is its keyframe, so a frame with someone in
    front of the board is not chosen. Only a few full frames are held at a time. More than
    ``max_views`` views are merged, shortest first.

    ``hold`` counts samples, so its length in seconds depends on the sampling rate: 3 samples
    are 3 s at 1 fps but 0.6 s at 5 fps. The last view ends at ``end``, the recording's end
    time; without it, one sample interval after the last sample (one second when there is a
    single sample), so no view is ever empty.

    Raises:
        ValueError: If ``change`` is not in ``(0, 1)``, or ``hold`` or ``max_views`` is not
            positive.

    """
    if not 0 < change < 1 or hold < 1 or max_views < 1:
        raise ValueError(
            "vision: vision_change_fraction must be in (0, 1), and vision_change_hold and "
            "the board-state limit positive"
        )
    views: list[_View] = []
    reference: Frame | None = None  # the current view's first sample, small
    best: tuple[float, float, Frame] = (0.0, 0.0, np.zeros((1, 1, 3), np.uint8))
    pending: list[tuple[float, Frame, Frame]] = []  # (time, small, full): a new view forming
    start = previous = last = 0.0
    seen = 0
    for time, frame in samples:
        small = _for_comparison(frame)
        previous, last, seen = last, time, seen + 1
        if reference is None:
            reference, start, best = small, time, (_sharpness(small), time, frame)
        elif _changed(small, reference) <= change:
            pending = []
            if (score := _sharpness(small)) > best[0]:
                best = (score, time, frame)
        else:
            if pending and _changed(small, pending[-1][1]) > change:
                pending = []  # still moving: not a view yet
            pending.append((time, small, frame))
            if len(pending) >= hold:
                views.append(_close(start, pending[0][0], best))
                start, reference = pending[0][0], pending[0][1]
                best = max(((_sharpness(s), t, f) for t, s, f in pending), key=lambda b: b[0])
                pending = []
    if reference is None:
        return []
    step = last - previous if seen > 1 else 1.0
    views.append(_close(start, max(end, last) if end is not None else last + step, best))
    return _merge_shortest(views, max_views)


def _changed(before: Frame, after: Frame) -> float:
    """Share of pixels whose grey level moved by more than the codec noise."""
    return float((np.abs(before.astype(np.int16) - after.astype(np.int16)) > _NOISE).mean())


def _sharpness(gray: Frame) -> float:
    """Mean gradient: higher for crisp text and lines, lower for blur and empty board."""
    values = gray.astype(np.float32)
    return float(np.abs(np.diff(values, axis=0)).mean() + np.abs(np.diff(values, axis=1)).mean())


def _close(start: float, end: float, best: tuple[float, float, Frame]) -> _View:
    """Finish a view, encoding its keyframe only now that it is final."""
    return _View(start=start, end=end, keyframe=_png(best[2]), keyframe_time=best[1])


def _png(frame: Frame) -> bytes:
    """Encode an RGB frame as PNG, lossless so text stays readable for the model."""
    picture = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
    codec = av.CodecContext.create("png", "w")
    codec.width, codec.height, codec.pix_fmt = picture.width, picture.height, "rgb24"
    return b"".join(bytes(packet) for packet in [*codec.encode(picture), *codec.encode(None)])


def _merge_shortest(views: list[_View], max_views: int) -> list[_View]:
    """Merge the shortest view into its shorter neighbour until at most ``max_views`` remain.

    The merged view spans both and keeps the keyframe of the longer one.
    """
    views = list(views)
    while len(views) > max_views:
        i = min(range(len(views)), key=lambda k: _length(views[k]))
        if i == len(views) - 1 or i > 0 and _length(views[i - 1]) <= _length(views[i + 1]):
            i -= 1  # merge with the left neighbour
        left, right = views[i], views[i + 1]
        kept = left if _length(left) >= _length(right) else right
        views[i : i + 2] = [
            _View(
                start=left.start,
                end=right.end,
                keyframe=kept.keyframe,
                keyframe_time=kept.keyframe_time,
            )
        ]
    return views


def _length(view: _View) -> float:
    return view.end - view.start


def read_keyframe(
    config: Config,
    png: bytes,
    speech: str = "",
    *,
    kind: str = "unknown",
    usage: Counter[str] | None = None,
    previous: PreviousState | None = None,
) -> dict[str, Any] | None:
    """Read one keyframe with the vision model (architecture.md 6.5).

    The one place a keyframe is read: the ``vision`` stage and ``eval --score-vision``
    (architecture.md 4.4) both call it, so what eval scores is what a run sends. ``png`` is the
    keyframe as :class:`_View` holds it; ``speech`` is what was said while its view was on
    screen; ``kind`` is what the layout probe says the frame shows; ``usage`` collects what the
    call cost (:func:`endpoint.ask`). ``previous`` is the board state before this one: with it,
    the prompt is v2, which carries its reading and where new ink appeared; without it, v0. The
    OCR model has no prompt, so it gets none. Returns the
    :class:`VisualContext` fields the model reads, or ``None`` with a warning when the call
    fails or a field is malformed: a board description only enriches questions the pipeline
    already found, so unlike the judge, vision never ends the run. Cached on the request,
    image included (:func:`endpoint.ask`). Only what the endpoint or the model did wrong is
    tolerated; a setting that cannot be used, such as an unreadable ``.env``, is a user error
    that would drop every keyframe the same way, so it is raised.

    Raises:
        RuntimeError: If the optional ``questxtract[llm]`` extra is not installed.
        ValueError: If the settings cannot be used, such as an unreadable ``.env``.

    """
    image = _data_url(png)
    if config.vision_reader == "ocr":
        return _read_ocr(config, image, usage)
    # As prompt v0 was measured in #52: the prompt, with its kind and speech filled in, as the
    # system message, and the image alone as the user message.
    values = {"kind": kind, "speech": speech.strip() or "-"}
    if previous is not None:
        box = previous.new_ink
        values |= {
            "previous_description": previous.description.strip() or "-",
            "previous_text": previous.text.strip() or "-",
            "previous_latex": (previous.latex or "").strip() or "-",
            "new_ink": "not found"
            if box is None
            else f"[{', '.join(f'{v:.2f}' for v in box)}] (x0, y0, x1, y1: fractions of this "
            "frame's width and height from its top-left corner)",
        }
    template = _PROMPT if previous is None else _PROMPT_V2
    # One pass, so a placeholder inside the speech or a reading is never filled in.
    prompt = _PLACEHOLDER.sub(lambda m: values.get(m[1], m[0]), template)
    request: dict[str, Any] = {
        "model": config.vision_model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]},
        ],
        "response_format": _SCHEMA,
    }
    if config.vision_effort:
        request["reasoning_effort"] = config.vision_effort
    try:
        answer = endpoint.ask(
            config, request, name="vision", valid=_readable, usage=usage, vision=True
        )
    except (ConnectionError, endpoint.InvalidAnswer) as exc:
        warnings.warn(f"vision: keyframe left out: {exc}", UserWarning, stacklevel=2)
        return None
    description, text, latex, confidence = (answer[k] for k in _FIELDS)
    return {
        "description": description.strip(),
        "text": text.strip(),
        "latex": (latex or "").strip() or None,
        "confidence": min(1.0, max(0.0, float(confidence))),
    }


def _read_ocr(config: Config, image: str, usage: Counter[str] | None) -> dict[str, Any] | None:
    """Read one keyframe with the OCR model: its text and formulas, no description.

    The image and the task prompt go in one user message, as GLM-OCR expects, with the token
    probabilities asked for (:func:`_from_ocr`). A server that refuses them (HTTP 400) is asked
    again without, then and for the rest of the process; its confidence is "not measured".
    Cached, retried and tolerated exactly as the vision model's reading (:func:`read_keyframe`).
    """
    request: dict[str, Any] = {
        "model": config.vision_ocr_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": _OCR_PROMPT},
                ],
            }
        ],
    }
    url = config.vision_url
    if url not in _NO_LOGPROBS:
        request["logprobs"] = True
    try:
        try:
            answer = _ask_ocr(config, request, usage)
        except endpoint.Refused:
            if "logprobs" not in request:
                raise
            # Some servers do not take logprobs: ask without, and confidence is "not measured".
            _NO_LOGPROBS.add(url)
            del request["logprobs"]
            answer = _ask_ocr(config, request, usage)
    except (ConnectionError, endpoint.InvalidAnswer) as exc:
        warnings.warn(f"vision: keyframe left out: {exc}", UserWarning, stacklevel=3)
        return None
    return {key: answer[key] for key in _FIELDS}


def _ask_ocr(config: Config, request: dict[str, Any], usage: Counter[str] | None) -> dict[str, Any]:
    return endpoint.ask(
        config,
        request,
        name="vision",
        valid=_readable,
        usage=usage,
        vision=True,
        convert=_from_ocr,
    )


def _from_ocr(choice: Any) -> dict[str, Any] | None:
    r"""Turn an OCR answer (plain text, formulas inline as ``$...$``) into the board fields.

    ``latex`` holds the formula segments that are not labels (:func:`ocr_formulas`), one per
    line. ``text`` is the words: formulas are left out, since they are in ``latex``, and labels
    are written plainly (``\textcircled{2}`` as ``(2)``, ``1\mathrm{Hz}`` as ``1 Hz``), so
    it reads like the vision model's ``text``. ``description`` stays empty, since an OCR model
    does not describe. ``confidence`` is the mean probability of the answer's tokens
    when the server returns them, else 0.5, which means "not measured". ``None`` when the
    answer is not text.
    """
    content = getattr(choice.message, "content", None)
    if not isinstance(content, str):
        return None
    formulas = ocr_formulas(content)
    text = _OCR_FORMULA.sub(lambda m: _plain_label(m[1] or m[2]), content)
    text = "\n".join(" ".join(line.split()) for line in text.splitlines())
    return {
        "description": "",
        "text": text.strip(),
        "latex": "\n".join(formulas) or None,
        "confidence": _token_confidence(choice),
    }


def _plain_label(segment: str) -> str:
    """Return a label segment as plain text, or ``""`` for a formula (it is in ``latex``)."""
    segment = segment.strip()
    if not _OCR_LABEL.fullmatch(segment):
        return ""
    segment = _OCR_CIRCLED.sub(r"(\1)", segment)
    return _OCR_COMMAND.sub(lambda m: f" {m[1]}" if m[1] is not None else m[2], segment).strip()


def ocr_formulas(text: str) -> list[str]:
    r"""Return the formulas in OCR text: its ``$...$`` and ``$$...$$`` segments, labels left out.

    A label is a segment that is only a circled number (``\textcircled{4}``), a number with
    a unit (``1\mathrm{Hz}``), or a single symbol (``X``, ``\pi``).
    """
    formulas: list[str] = []
    for match in _OCR_FORMULA.finditer(text):
        formula = (match[1] or match[2]).strip()
        if formula and not _OCR_LABEL.fullmatch(formula):
            formulas.append(formula)
    return formulas


def _token_confidence(choice: Any) -> float:
    """Mean probability of the answer's tokens, or 0.5 when the server sent none."""
    tokens = getattr(getattr(choice, "logprobs", None), "content", None) or ()
    values = [t.logprob for t in tokens if isinstance(getattr(t, "logprob", None), (int, float))]
    if not values:
        return _OCR_UNMEASURED
    return round(min(1.0, max(0.0, sum(math.exp(v) for v in values) / len(values))), 3)


def _readable(answer: dict[str, Any]) -> bool:
    """Whether an answer has every board field with a usable type; nothing else is cached."""
    if not all(key in answer for key in _FIELDS):
        return False  # a missing "latex" is not null: the schema asks for every key
    description, text, latex, confidence = (answer[k] for k in _FIELDS)
    return (
        isinstance(description, str)
        and isinstance(text, str)
        and (latex is None or isinstance(latex, str))
        and not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and math.isfinite(confidence)
    )


@stage("attach_visual")
def attach_visual(ctx: Context) -> Context:
    """Join board states to the segments they overlap, by interval intersection."""
    if not ctx.visual:
        return ctx
    ctx.segments = tuple(
        _with_visual(s, overlapping(ctx.visual, s.start, s.end)) for s in ctx.segments
    )
    ctx.stats["board_states"] = len(ctx.visual)
    return ctx


def _with_visual(segment: Segment, visual: tuple[VisualContext, ...]) -> Segment:
    """Return ``segment`` with its board states attached; it is frozen, so copy."""
    return Segment(
        sentence_start=segment.sentence_start,
        sentence_end=segment.sentence_end,
        start=segment.start,
        end=segment.end,
        text=segment.text,
        visual=visual,
        close_reason=segment.close_reason,
    )

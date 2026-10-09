#!/usr/bin/env python3
r"""Run prompt v0 over test frames with several vision models and record what each one says.

Spike tooling for #52, not shipped: one request per frame and model against an
OpenAI-compatible endpoint (hosted, or a local server through ``--base-url``), the answer
checked against ``prompt_v0.schema.json``, with latency and token counts. Nothing here is
cached and nothing touches the pipeline.

Usage: uv run python scripts/compare_vision.py tests/data/vision/*.png \\
           -m gpt-4o -m gpt-5.6-luna@none -o spike-results/
The key is read from OPENAI_API_KEY, or from the variable ``--key-env`` names (for an
Idun key: ``--key-env IDUN_API_KEY --base-url https://llm.hpc.ntnu.no/v1``). A model is
``name`` or ``name@reasoning`` (the reasoning_effort to send, e.g. ``none``); without ``@``,
no setting is sent, which models without a reasoning mode such as gpt-4o need. Writes
DIR/results.jsonl, one line per call. ``--transcript T.json`` adds the words spoken within
``--window`` seconds of each frame (its second is the number in the file name).
``--price name=IN,OUT`` (USD per million tokens) adds a cost per call.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_VISION = Path(__file__).resolve().parent.parent / "tests" / "data" / "vision"
_KINDS = ("slides", "blackboard", "whiteboard", "tablet", "unknown")


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Return ``schema`` without the keywords OpenAI's strict mode rejects (checked locally)."""
    props = {
        name: {k: v for k, v in spec.items() if k not in ("minimum", "maximum")}
        for name, spec in schema["properties"].items()
    }
    return {**schema, "properties": props}


def _check(answer: Any) -> str | None:
    """Return why ``answer`` breaks the schema, or ``None`` when it fits."""
    if not isinstance(answer, dict):
        return "not a JSON object"
    for key, ok in (
        ("description", lambda v: isinstance(v, str)),
        ("text", lambda v: isinstance(v, str)),
        ("latex", lambda v: v is None or isinstance(v, str)),
        (
            "confidence",
            lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1,
        ),
    ):
        if key not in answer or not ok(answer[key]):
            return f"bad or missing {key}"
    return None


def _speech(transcript: dict[str, Any], seconds: float, window: float) -> str:
    """Return the words spoken within ``window`` seconds either side of ``seconds``."""
    words = [
        w["word"]
        for segment in transcript["segments"]
        for w in segment["words"]
        if seconds - window <= w["start"] <= seconds + window
    ]
    return " ".join("".join(words).split()) or "-"


def _request(
    model: str, reasoning: str | None, prompt: str, png: bytes, detail: str | None, schema: Any
) -> dict[str, Any]:
    """Build the chat request for one frame."""
    image: dict[str, Any] = {"url": "data:image/png;base64," + base64.b64encode(png).decode()}
    if detail:
        image["detail"] = detail
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "image_url", "image_url": image}]},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "board", "strict": True, "schema": _strict(schema)},
        },
    }
    if reasoning is not None:
        request["reasoning_effort"] = reasoning
    return request


def main(argv: list[str] | None = None) -> int:
    """Send every frame to every model and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path, help="PNG frames")
    parser.add_argument("-m", "--model", action="append", required=True, help="name[@reasoning]")
    parser.add_argument("-o", "--output", type=Path, default=Path("spike-results"))
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    parser.add_argument(
        "--prompt", default="prompt_v0.txt", help="prompt file in tests/data/vision"
    )
    parser.add_argument(
        "--key-env", default="OPENAI_API_KEY", help="environment variable holding the key"
    )
    parser.add_argument("--kind", choices=_KINDS, default="blackboard", help="frame kind")
    parser.add_argument(
        "--transcript", type=Path, default=None, help="transcript JSON: send the speech too"
    )
    parser.add_argument(
        "--window", type=float, default=20.0, help="seconds of speech either side of a frame"
    )
    parser.add_argument("--detail", choices=("low", "high", "auto"), default=None)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per call")
    parser.add_argument("--price", action="append", default=[], help="name=IN,OUT per 1M tokens")
    args = parser.parse_args(argv)

    key = os.environ.get(args.key_env) or ("local" if args.base_url else "")
    if not key:
        print(f"error: set {args.key_env} in the environment", file=sys.stderr)
        return 1
    try:
        from openai import OpenAI  # noqa: PLC0415 - the optional questxtract[llm] extra
    except ImportError:
        print("error: needs the openai package (`uv sync --dev`)", file=sys.stderr)
        return 1

    template = (_VISION / args.prompt).read_text(encoding="utf-8").replace("{kind}", args.kind)
    transcript = (
        json.loads(args.transcript.read_text(encoding="utf-8")) if args.transcript else None
    )
    schema = json.loads((_VISION / "prompt_v0.schema.json").read_text(encoding="utf-8"))
    prices = {}
    for spec in args.price:
        name, _, rates = spec.partition("=")
        low, _, high = rates.partition(",")
        prices[name] = (float(low), float(high))

    client = OpenAI(api_key=key, base_url=args.base_url, timeout=args.timeout, max_retries=0)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "results.jsonl"
    with out.open("a", encoding="utf-8") as sink:
        for frame in args.frames:
            png = frame.read_bytes()
            speech = "-"
            if transcript:
                found = re.search(r"_(\d+(?:\.\d+)?)\.png$", frame.name)
                if not found:
                    print(f"error: no second in the name of {frame.name}", file=sys.stderr)
                    return 1
                speech = _speech(transcript, float(found.group(1)), args.window)
            prompt = template.replace("{speech}", speech)
            for spec in args.model:
                model, sep, reasoning = spec.partition("@")
                request = _request(
                    model, reasoning if sep else None, prompt, png, args.detail, schema
                )
                row: dict[str, Any] = {
                    "frame": frame.name,
                    "model": spec,
                    "detail": args.detail,
                    "speech": speech if transcript else None,
                }
                started = time.monotonic()
                try:
                    response = client.chat.completions.create(**request)
                except Exception as exc:  # record any failure and carry on with the next call
                    row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}")
                else:
                    row["seconds"] = round(time.monotonic() - started, 2)
                    usage = response.usage
                    row["tokens_in"] = usage.prompt_tokens if usage else None
                    row["tokens_out"] = usage.completion_tokens if usage else None
                    content = response.choices[0].message.content or ""
                    try:
                        answer = json.loads(content)
                    except json.JSONDecodeError:
                        answer = None
                    problem = _check(answer)
                    row.update(ok=problem is None, answer=answer if problem is None else content)
                    if problem:
                        row["error"] = problem
                    if usage and model in prices:
                        low, high = prices[model]
                        row["usd"] = round(
                            (usage.prompt_tokens * low + usage.completion_tokens * high) / 1e6, 6
                        )
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                status = "ok" if row["ok"] else f"FAILED ({row.get('error', '')[:60]})"
                print(f"{frame.name} {spec}: {status} {row.get('seconds', '')}s", file=sys.stderr)
    print(f"-> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

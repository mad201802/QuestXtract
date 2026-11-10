"""The questions of a ``questions.json`` as a LaTeX document that compiles (12.7).

One entry per question in document order: its time, its confidence, the question as a teacher
would read it (the reviewed text when there is one), the spoken wording when that differs, the
lecturer's answer, and what was on the board. Questions a reviewer rejected are left out.

Every text is escaped, since spoken words hold ``%``, ``&`` and ``_``. Characters outside
Latin-1 that pdfLaTeX cannot typeset with ``inputenc`` (Greek letters and math symbols read off
the board) become LaTeX commands where a common one exists, else ``?``. The board's own ``latex``
field is shown in a verbatim block, not typeset, so a wrong formula cannot break the compile.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .export import read_questions
from .models import SchemaError
from .tex import _write_atomic

_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "%": r"\%",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
# Common characters beyond Latin-1, as LaTeX; anything else outside Latin-1 becomes "?".
_TEXT = {
    "‘": "`", "’": "'", "“": "``", "”": "''", "–": "--", "—": "---", "…": r"\ldots{}",
    "‹": r"\guilsinglleft{}", "›": r"\guilsinglright{}", "•": r"\textbullet{}",
}  # fmt: skip
_MATH = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon",
    "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "κ": r"\kappa", "λ": r"\lambda",
    "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma",
    "τ": r"\tau", "φ": r"\phi", "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Π": r"\Pi",
    "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    "−": "-", "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow", "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow", "≤": r"\leq", "≥": r"\geq", "≠": r"\neq",
    "≈": r"\approx", "∞": r"\infty", "∑": r"\sum", "∏": r"\prod", "∫": r"\int",
    "√": r"\surd", "∂": r"\partial", "∇": r"\nabla", "∈": r"\in", "∉": r"\notin",
    "⊂": r"\subset", "⊆": r"\subseteq", "∪": r"\cup", "∩": r"\cap", "∀": r"\forall",
    "∃": r"\exists", "∠": r"\angle", "ℜ": r"\Re", "ℑ": r"\Im", "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}", "ℝ": r"\mathbb{R}", "ℂ": r"\mathbb{C}",
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4", "₅": "_5", "₆": "_6",
    "₇": "_7", "₈": "_8", "₉": "_9", "₋": "_-", "⁻": "^-", "ᵗ": "^t",
}  # fmt: skip
_CIRCLED = {chr(0x2460 + n): f"({n + 1})" for n in range(20)}  # ① to ⑳
_IGNORED = {"̄", "̅", "​", "﻿"}  # combining bars, zero-width marks

_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amssymb}
\usepackage[margin=2.5cm]{geometry}
\setlength{\parindent}{0pt}
"""


class QuestionsTexError(SchemaError):
    """Raised when a ``questions.json`` cannot be read or written as a LaTeX document."""


def export_questions_tex(questions_path: str | Path, output_path: str | Path) -> int:
    """Read a ``questions.json`` and write its questions as a LaTeX document that compiles.

    Questions a reviewer marked ``rejected`` are left out. The file is UTF-8 with LF line
    endings, the same input gives the same bytes, and nothing is written if a check fails.

    Args:
        questions_path: A ``questions.json`` from ``process``, edited or not.
        output_path: Where to write the ``.tex`` file. Its folder must exist.

    Returns:
        The number of questions written.

    Raises:
        QuestionsTexError: If the file cannot be read or is not a ``questions.json``, if no
            question is left after the rejected ones, or if the file cannot be written.

    """
    try:
        document = read_questions(questions_path)
    except OSError as exc:
        raise QuestionsTexError(f"{questions_path}: {exc.strerror or exc}") from exc
    except SchemaError as exc:
        raise QuestionsTexError(str(exc)) from exc
    text, count = _document(document)
    try:
        _write_atomic(Path(output_path), text.encode("utf-8"))
    except OSError as exc:
        raise QuestionsTexError(
            f"{output_path}: cannot write the TeX file ({exc.strerror or exc})"
        ) from exc
    return count


def questions_to_tex(document: Mapping[str, Any]) -> str:
    """Return a ``questions.json`` document as LaTeX, as :func:`export_questions_tex` writes it."""
    return _document(document)[0]


def _document(document: Mapping[str, Any]) -> tuple[str, int]:
    questions = [
        question
        for question in document.get("questions") or []
        if isinstance(question, Mapping) and _review(question).get("status") != "rejected"
    ]
    if not questions:
        raise QuestionsTexError(
            "no question to write: every question is rejected or none was found"
        )
    visuals = {
        visual["id"]: visual
        for visual in document.get("visuals") or []
        if isinstance(visual, Mapping) and isinstance(visual.get("id"), str)
    }
    source = document.get("source")
    path = source.get("path") if isinstance(source, Mapping) else None
    name = Path(path).name if isinstance(path, str) and path else ""
    title = f"Questions asked in {name}" if name else "Questions asked in the lecture"
    noun = "question" if len(questions) == 1 else "questions"
    lines = [
        _PREAMBLE.rstrip("\n"),
        rf"\title{{{escape(title)}}}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        "",
        f"{len(questions)} {noun}, found by QuestXtract. Times are from the start of the "
        "recording.",
        "",
        r"\begin{enumerate}",
    ]
    for question in questions:
        lines += _entry(question, visuals)
    lines += [r"\end{enumerate}", r"\end{document}"]
    return "\n".join(lines) + "\n", len(questions)


def _entry(question: Mapping[str, Any], visuals: Mapping[str, Mapping[str, Any]]) -> list[str]:
    review = _review(question)
    spoken = _string(question.get("text"))
    reviewed = _string(review.get("text"))
    shown = reviewed or spoken
    wording = _string(question.get("original_text")) or (spoken if reviewed else "")
    when = f"{_string(question.get('start_hms'))}--{_string(question.get('end_hms'))}"
    head = rf"\item \textbf{{[{escape(when)}]}}"
    confidence = question.get("confidence")
    label = _string(question.get("confidence_label"))
    if _number(confidence):
        words = " ".join(part for part in ("confidence", escape(label)) if part)
        head += rf" \quad {words} ({float(confidence):.2f})"  # type: ignore[arg-type]
    lines = [head + r"\par", escape(shown) + r"\par"]
    if wording and wording != shown:
        lines.append(rf"\textit{{Spoken:}} {escape(wording)}\par")
    answer = question.get("answer")
    if isinstance(answer, Mapping) and _string(answer.get("text")):
        lines.append(
            rf"\textit{{Answer}} [{_span(answer)}]: {escape(_string(answer.get('text')))}\par"
        )
    for visual_id in question.get("visual_ids") or []:
        visual = visuals.get(visual_id) if isinstance(visual_id, str) else None
        if visual is None:
            continue
        said = ". ".join(
            part
            for part in (_string(visual.get("text")), _string(visual.get("description")))
            if part
        )
        if said:
            lines.append(rf"\textit{{Board}} [{_span(visual)}]: {escape(said)}\par")
        formula = _string(visual.get("latex"), keep_lines=True)
        if formula:
            lines += [r"\begin{verbatim}", _verbatim(formula), r"\end{verbatim}"]
    return lines


def escape(text: str) -> str:
    """Return ``text`` as LaTeX that pdfLaTeX typesets as written, on one line."""
    out: list[str] = []
    for char in " ".join(text.split()):
        if char in _ESCAPES:
            out.append(_ESCAPES[char])
        elif ord(char) <= 0xFF:
            out.append(char)
        elif char in _TEXT:
            out.append(_TEXT[char])
        elif char in _MATH:
            out.append(rf"\ensuremath{{{_MATH[char]}}}")
        elif char in _CIRCLED:
            out.append(_CIRCLED[char])
        elif char not in _IGNORED:
            out.append("?")
    return "".join(out)


def _verbatim(text: str) -> str:
    """Return ``text`` for a verbatim block: no closing tag inside, Latin-1 only."""
    safe = "".join(char if ord(char) <= 0xFF else "?" for char in text)
    return safe.replace(r"\end{verbatim}", r"\end {verbatim}")


def _span(part: Mapping[str, Any]) -> str:
    start, end = part.get("start"), part.get("end")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return ""
    if not (_number(start) and _number(end)):
        return ""
    return f"{_hms(float(start))}--{_hms(float(end))}"


def _hms(seconds: float) -> str:
    hours, rest = divmod(math.floor(max(seconds, 0.0)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _review(question: Mapping[str, Any]) -> Mapping[str, Any]:
    review = question.get("review")
    return review if isinstance(review, Mapping) else {}


def _string(value: object, *, keep_lines: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip("\n") if keep_lines else " ".join(value.split())

"""QTI package: ``export_qti``, the version dispatch, the deterministic ZIP and the JSON sidecar.

A QTI version is one writer module, which turns an assessment test into the package's XML
entries, plus one entry in ``_WRITERS``. This module adds the sidecar and writes the ZIP. The
package is deterministic: the same assessment and version give the same bytes, so every entry
gets a fixed timestamp and fixed attributes instead of the clock and the platform's.
"""

from __future__ import annotations

import io
import json
import stat
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path

from . import qti_v2, qti_v3
from .models import SchemaError
from .qti import METADATA_PATH, AssessmentTest, QtiExportError, quiz_to_qti
from .quiz_schema import read_quiz

_WRITERS: dict[str, Callable[[AssessmentTest], list[tuple[str, bytes]]]] = {
    "2.1": qti_v2.package_entries,
    "3.0": qti_v3.package_entries,
}
QTI_VERSIONS = tuple(_WRITERS)  # ("2.1", "3.0"), the choices of `questxtract qti --qti-version`
DEFAULT_QTI_VERSION = "3.0"  # export_qti, write_qti_package and the CLI flag all default to it
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # the earliest date a ZIP entry can carry


def export_qti(
    quiz_path: str | Path, output_path: str | Path, *, version: str = DEFAULT_QTI_VERSION
) -> None:
    """Read a ``quiz.json`` and write it as a QTI package of the given version.

    Every item becomes a single-choice item scored against its correct option, with each
    option's explanation and the general one as feedback. Items a reviewer marked ``rejected``
    are left out. The same input and version give a byte-identical ZIP.

    Args:
        quiz_path: A ``quiz.json``, schema version 1.x.
        output_path: Where to write the ZIP. Its folder must exist.
        version: The QTI version, one of :data:`QTI_VERSIONS`: ``"2.1"`` or ``"3.0"``, by
            default :data:`DEFAULT_QTI_VERSION`.

    Raises:
        QtiExportError: If ``version`` is not supported (before anything is read), if the quiz
            file cannot be read or breaks a rule of ``quiz.json``, if an item cannot be
            represented in QTI, or if the package cannot be written.

    """
    _check_version(version)
    try:
        document = read_quiz(quiz_path)
    except OSError as exc:
        raise QtiExportError(f"{quiz_path}: {exc.strerror or exc}") from exc
    except SchemaError as exc:
        raise QtiExportError(str(exc)) from exc
    write_qti_package(quiz_to_qti(document), output_path, version=version)


def write_qti_package(
    assessment: AssessmentTest, path: str | Path, *, version: str = DEFAULT_QTI_VERSION
) -> None:
    """Write an assessment test as a QTI ZIP package of the given version.

    For 2.1, items worth other than 1 point get a ``UserWarning``: an LMS that scores with the
    ``match_correct`` template instead of the item's own rules gives them 1 point and no feedback
    per option (architecture.md 12.4).

    Raises:
        QtiExportError: If ``version`` is not supported, or if the file cannot be written.

    """
    _check_version(version)
    if version == "2.1":
        _warn_template_points(assessment)
    data = _package_bytes(assessment, version)
    try:
        Path(path).write_bytes(data)
    except OSError as exc:
        raise QtiExportError(
            f"{path}: cannot write the QTI package ({exc.strerror or exc})"
        ) from exc


def _warn_template_points(assessment: AssessmentTest) -> None:
    other = [item.identifier for item in assessment.items if item.points != 1]
    if other:
        warnings.warn(
            f"QTI 2.1: {len(other)} item(s) are worth other than 1 point ({', '.join(other[:3])}"
            f"{', ...' if len(other) > 3 else ''}); an LMS that scores with the match_correct "
            "template gives them 1 point and no feedback per option",
            UserWarning,
            stacklevel=3,
        )


def _check_version(version: str) -> None:
    if version not in _WRITERS:
        raise QtiExportError(
            f"QTI version {version!r} is not supported; choose one of {', '.join(QTI_VERSIONS)}"
        )


def _package_bytes(assessment: AssessmentTest, version: str) -> bytes:
    entries = _WRITERS[version](assessment)
    entries.append((METADATA_PATH, _metadata_json(assessment, version)))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        for name, data in entries:
            _write_entry(package, name, data)
    return buffer.getvalue()


def _write_entry(package: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED  # ZipInfo defaults to STORED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.create_system = 3  # Unix, also on Windows, where ZipInfo would say 0
    package.writestr(info, data)


def _metadata_json(assessment: AssessmentTest, version: str) -> bytes:
    items = {
        item.identifier: {
            "source_question_id": item.provenance.source_question_id,
            "start": item.provenance.start,
            "end": item.provenance.end,
            **item.metadata,
        }
        for item in assessment.items
    }
    payload = {"format": f"questxtract-qti-{version}", "quiz": assessment.metadata, "items": items}
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

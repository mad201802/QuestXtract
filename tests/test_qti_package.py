"""Tests for the QTI package: the version dispatch, the deterministic ZIP and the file (#163).

What each version's XML holds, and its validation against the XSDs, is in test_qti_v2.py and
test_qti_v3.py.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from questxtract import qti_v2, qti_v3
from questxtract.qti import METADATA_PATH, QtiExportError, quiz_to_qti
from questxtract.qti_package import (
    DEFAULT_QTI_VERSION,
    QTI_VERSIONS,
    export_qti,
    write_qti_package,
)

MakeQuiz = Callable[..., dict[str, Any]]
_WRITERS: dict[str, ModuleType] = {"2.1": qti_v2, "3.0": qti_v3}


def _export(
    tmp_path: Path, quiz: dict[str, Any], name: str = "out.zip", version: str | None = None
) -> dict[str, bytes]:
    source = tmp_path / "quiz.json"
    source.write_text(json.dumps(quiz), encoding="utf-8")
    if version is None:
        export_qti(source, tmp_path / name)
    else:
        export_qti(source, tmp_path / name, version=version)
    with zipfile.ZipFile(tmp_path / name) as package:
        return {entry: package.read(entry) for entry in package.namelist()}


def test_the_versions_are_2_1_and_3_0_and_3_0_is_the_default() -> None:
    assert QTI_VERSIONS == ("2.1", "3.0")
    assert DEFAULT_QTI_VERSION == "3.0"


@pytest.mark.parametrize("version", QTI_VERSIONS)
def test_each_version_reaches_its_writer(tmp_path: Path, make_quiz: MakeQuiz, version: str) -> None:
    quiz = make_quiz("q_aaa", "q_bbb")
    package = _export(tmp_path, quiz, version=version)
    entries = _WRITERS[version].package_entries(quiz_to_qti(quiz))
    assert list(package) == [name for name, _ in entries] + [METADATA_PATH]
    assert [package[name] for name, _ in entries] == [data for _, data in entries]
    assert json.loads(package[METADATA_PATH])["format"] == f"questxtract-qti-{version}"


def test_without_a_version_the_package_is_qti_3_0(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    quiz = make_quiz("q_aaa")
    _export(tmp_path, quiz, "default.zip")
    _export(tmp_path, quiz, "explicit.zip", version="3.0")
    assert (tmp_path / "default.zip").read_bytes() == (tmp_path / "explicit.zip").read_bytes()


@pytest.mark.parametrize("version", ["2.2", "1.2", "3", "v3.0", "3.0 ", ""])
def test_an_unknown_version_fails_naming_the_supported_ones_before_the_quiz_is_read(
    tmp_path: Path, version: str
) -> None:
    output = tmp_path / "out.zip"
    # The quiz file does not exist either: the version error comes first.
    with pytest.raises(QtiExportError, match=r"choose one of 2\.1, 3\.0$") as error:
        export_qti(tmp_path / "missing.json", output, version=version)
    assert f"QTI version {version!r} is not supported" in str(error.value)
    assert not output.exists()


def test_writing_with_an_unknown_version_fails_and_writes_nothing(
    tmp_path: Path, make_quiz: MakeQuiz
) -> None:
    output = tmp_path / "out.zip"
    with pytest.raises(QtiExportError, match=r"'2\.2' is not supported; choose one of 2\.1, 3\.0"):
        write_qti_package(quiz_to_qti(make_quiz("q_aaa")), output, version="2.2")
    assert not output.exists()


@pytest.mark.parametrize("version", QTI_VERSIONS)
def test_every_entry_has_the_fixed_date_and_is_deflated(
    tmp_path: Path, make_quiz: MakeQuiz, version: str
) -> None:
    _export(tmp_path, make_quiz("q_aaa", "q_bbb"), version=version)
    with zipfile.ZipFile(tmp_path / "out.zip") as package:
        infos = package.infolist()
    assert [info.filename for info in infos] == [
        "items/quiz_q_aaa.xml",
        "items/quiz_q_bbb.xml",
        "assessmentTest.xml",
        "imsmanifest.xml",
        "metadata/questxtract.json",
    ]
    for info in infos:
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.compress_type == zipfile.ZIP_DEFLATED
        assert info.create_system == 3
        assert info.external_attr >> 16 == 0o100644


@pytest.mark.parametrize("version", QTI_VERSIONS)
def test_every_xml_entry_declares_utf_8(tmp_path: Path, make_quiz: MakeQuiz, version: str) -> None:
    package = _export(tmp_path, make_quiz("q_aaa"), version=version)
    for name, data in package.items():
        if name.endswith(".xml"):
            assert data.startswith(b"<?xml version='1.0' encoding='utf-8'?>\n<"), name


def test_an_output_in_a_missing_folder_fails_readably(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    with pytest.raises(QtiExportError, match="cannot write the QTI package"):
        _export(tmp_path, make_quiz("q_aaa"), "missing/out.zip")


def test_an_unreadable_or_invalid_quiz_fails_readably(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    with pytest.raises(QtiExportError, match="missing.json"):
        export_qti(tmp_path / "missing.json", tmp_path / "out.zip")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(QtiExportError, match="not valid JSON"):
        export_qti(bad, tmp_path / "out.zip")
    with pytest.raises(QtiExportError, match=r"schema version 2\.0"):
        _export(tmp_path, make_quiz("q_aaa", schema_version="2.0"))
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["correct_response"]["option_ids"] = ["d"]
    with pytest.raises(QtiExportError, match=r"quiz\.json: questions\[0\]\.correct_response"):
        _export(tmp_path, quiz)
    assert not (tmp_path / "out.zip").exists()


def test_a_2_1_item_worth_other_than_1_point_warns(tmp_path: Path, make_quiz: MakeQuiz) -> None:
    """Review of #175: a template-only LMS would score it 1 and drop the per-option feedback."""
    quiz = make_quiz("q_aaa", "q_bbb")  # the fixture's items are worth 2
    quiz["questions"][0]["points"] = 1
    with pytest.warns(
        UserWarning, match=r"1 item\(s\) are worth other than 1 point \(quiz_q_bbb\)"
    ):
        _export(tmp_path, quiz, version="2.1")


@pytest.mark.parametrize(("version", "points"), [("3.0", 2), ("2.1", 1)])
def test_no_points_warning_for_3_0_or_one_point_items(
    tmp_path: Path, make_quiz: MakeQuiz, version: str, points: int, recwarn: Any
) -> None:
    quiz = make_quiz("q_aaa")
    quiz["questions"][0]["points"] = points
    _export(tmp_path, quiz, version=version)
    assert not [w for w in recwarn if "worth other than 1 point" in str(w.message)]

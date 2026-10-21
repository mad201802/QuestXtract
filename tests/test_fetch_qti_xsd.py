"""Tests for scripts/fetch_qti_xsd.py (issues #67, #163). No network: urlopen is replaced."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from scripts import fetch_qti_xsd
from scripts.fetch_qti_xsd import (
    ASI_XSD,
    ASI_XSD_V2P1,
    PINNED,
    local_path,
    main,
    missing,
    uri_mapper,
)

_XML_XSD = "https://purl.imsglobal.org/spec/w3/2001/schema/xsd/xml.xsd"
_MATHML2_XSD = "http://www.w3.org/Math/XMLSchema/mathml2/mathml2.xsd"  # a 2.1 file, over http
_content = {url: url.encode() for url in PINNED}  # a stand-in body per file
# The pinned URL of each address a file is downloaded from: always https.
_pinned_at = {"https://" + url.split("://", 1)[1]: url for url in PINNED}


def _serve(monkeypatch: pytest.MonkeyPatch, body: bytes | None = None) -> list[str]:
    """Answer every download with ``body``, or else with that file's stand-in body."""
    requested: list[str] = []

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        requested.append(url)
        return io.BytesIO(body if body is not None else _content[_pinned_at[url]])

    monkeypatch.setattr(fetch_qti_xsd.urllib.request, "urlopen", urlopen)
    return requested


def test_files_are_mirrored_by_host_and_path(tmp_path: Path) -> None:
    assert local_path(_XML_XSD, tmp_path) == (
        tmp_path / "purl.imsglobal.org" / "spec" / "w3" / "2001" / "schema" / "xsd" / "xml.xsd"
    )
    assert local_path(_MATHML2_XSD, tmp_path) == (
        tmp_path / "www.w3.org" / "Math" / "XMLSchema" / "mathml2" / "mathml2.xsd"
    )


def test_no_two_pinned_files_share_a_local_path(tmp_path: Path) -> None:
    assert len({local_path(url, tmp_path) for url in PINNED}) == len(PINNED)


def test_only_pinned_urls_are_mapped_to_local_files(tmp_path: Path) -> None:
    mapper = uri_mapper(tmp_path)
    assert mapper(ASI_XSD) == local_path(ASI_XSD, tmp_path).resolve().as_uri()
    assert mapper(ASI_XSD_V2P1) == local_path(ASI_XSD_V2P1, tmp_path).resolve().as_uri()
    assert mapper("https://example.org/other.xsd") == "https://example.org/other.xsd"
    # A 2.1 file is pinned under the http URL its schema names, not under its https twin.
    twin = ASI_XSD_V2P1.replace("http://", "https://", 1)
    assert mapper(twin) == twin


def test_a_file_with_another_hash_counts_as_missing(tmp_path: Path) -> None:
    assert len(missing(tmp_path)) == len(PINNED)
    path = local_path(_XML_XSD, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"tampered")
    assert _XML_XSD in missing(tmp_path)


def test_a_download_with_the_wrong_hash_is_refused_and_not_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _serve(monkeypatch, b"<xs:schema/>")
    assert main(["--dir", str(tmp_path)]) == 1
    assert "SHA-256" in capsys.readouterr().err
    assert not any(tmp_path.rglob("*.xsd"))


def test_files_in_place_are_not_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fetch_qti_xsd,
        "PINNED",
        {url: hashlib.sha256(_content[url]).hexdigest() for url in PINNED},
    )
    requested = _serve(monkeypatch)
    assert main(["--dir", str(tmp_path)]) == 0
    assert sorted(requested) == sorted(_pinned_at)
    assert local_path(_XML_XSD, tmp_path).read_bytes() == _content[_XML_XSD]

    requested.clear()
    assert main(["--dir", str(tmp_path)]) == 0
    assert requested == []


def test_files_the_schemas_name_over_http_are_downloaded_over_https(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        fetch_qti_xsd,
        "PINNED",
        {url: hashlib.sha256(_content[url]).hexdigest() for url in PINNED},
    )
    requested = _serve(monkeypatch)
    assert main(["--dir", str(tmp_path)]) == 0
    assert "https://www.w3.org/Math/XMLSchema/mathml2/mathml2.xsd" in requested
    assert not [url for url in requested if not url.startswith("https://")]
    assert local_path(_MATHML2_XSD, tmp_path).read_bytes() == _content[_MATHML2_XSD]
    assert "QTI 2.1 and 3.0 XSDs" in capsys.readouterr().err


def test_the_folder_can_be_set_in_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QUESTXTRACT_QTI_XSD_DIR", str(tmp_path))
    assert fetch_qti_xsd.xsd_dir() == tmp_path

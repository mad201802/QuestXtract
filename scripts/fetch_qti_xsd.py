"""Download the 1EdTech QTI 3.0 XSDs once, so the tests validate QTI packages offline.

`tests/test_qti_v3.py` (issue #67) validates every item, test and manifest it writes against
the official schemas. The item schema alone is 18 MB and imports MathML, SSML and XInclude
over `https`, which libxml2 cannot fetch and the tests must not, so this script downloads the
raw bytes of every file in the import tree to a local mirror, `<host>/<path>`, and checks each
against a pinned SHA-256. The tests then compile from the mirror with `uri_mapper()`, which
points every pinned URL at its local copy. Files are saved as served: one of them is UTF-16,
which `xmlschema.export` would not keep intact. Stdlib only.

Usage:
    uv run python scripts/fetch_qti_xsd.py              # into .cache/qti-xsd
    uv run python scripts/fetch_qti_xsd.py --dir DIR    # elsewhere

The folder is `.cache/qti-xsd` in the repository (git-ignored) unless `QUESTXTRACT_QTI_XSD_DIR`
or `--dir` names another. Files already there with the right hash are not downloaded again.
Exit code 0 when every file is in place, 1 with a message on stderr otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_QTI = "https://purl.imsglobal.org/spec/qti/v3p0/schema/xsd/"
ASI_XSD = _QTI + "imsqti_asiv3p0p1_v1p0.xsd"  # items and tests, QTI 3.0.1
CP_XSD = _QTI + "imsqtiv3p0_imscpv1p2_v1p0.xsd"  # imsmanifest.xml

# Every file the two schemas import, directly or not, with the SHA-256 of its bytes as served.
PINNED = {
    ASI_XSD: "bd82f6df3c7acec951fffc78548325a01e9e9bc91d5768d7415bdcbdc018341d",
    CP_XSD: "30be3dad4fe0ee103295268317ee0ff2b1826b8b55023db974a993edf3ce9101",
    _QTI + "imsqti_metadatav3p0_v1p0.xsd": (
        "448c18d65607695d2a60ce96c42003df98f28630fb044ca14ccc609bad384c8b"
    ),
    _QTI + "imsqtiv3p0_afa3p0drd_v1p0.xsd": (
        "dd3822a3fe0f42af9a25b33be56d11d1a1f3c54ea791bcc355965ceb8f41b064"
    ),
    _QTI + "imsqtiv3p0_cpextv1p2_v1p0.xsd": (
        "3f8deb01dd5ac3ad4b81522e159650c01b3dfe199cae6a6be7e87487e790ad85"
    ),
    _QTI + "imsqtiv3p0_csmv1p1_v1p0.xsd": (
        "caa7ddf900afa66fd5efb212a8834b42480f675de092ae927ce0c95b74443d2e"
    ),
    "https://purl.imsglobal.org/spec/md/v1p3/schema/xsd/imsmd_loose_v1p3p2.xsd": (
        "1ae9e77aba63b6789574631213c97fd00a3d43aeffb58050ac7befb665fba9f6"
    ),
    "https://purl.imsglobal.org/spec/mathml/v3p0/schema/xsd/mathml3.xsd": (
        "af49edc2c29c4976d6a9d8860388c026b94ca75edfbd67bbaf67448da9990656"
    ),
    "https://purl.imsglobal.org/spec/mathml/v3p0/schema/xsd/mathml3-common.xsd": (
        "94007301de9c8ed29ffed4a5eba6ca6b1e2e7511eda8d2b39e01b97d7d7e5d06"
    ),
    "https://purl.imsglobal.org/spec/mathml/v3p0/schema/xsd/mathml3-content.xsd": (
        "922b4082d00559df86b0389842b1d1ba9537db0050ab229d64ad313ddfb44098"
    ),
    "https://purl.imsglobal.org/spec/mathml/v3p0/schema/xsd/mathml3-presentation.xsd": (
        "74d605b09528b7c55a4ae803debc429c27a2afa6cd34607d28df79bcc7f3575b"
    ),
    "https://purl.imsglobal.org/spec/mathml/v3p0/schema/xsd/mathml3-strict-content.xsd": (
        "cd0da554afbe2d1570c4e548a29ee5eb0b25aea1d6455bacf664080cb5982640"
    ),
    "https://purl.imsglobal.org/spec/ssml/v1p1/schema/xsd/ssmlv1p1-core.xsd": (
        "f5e6d5ca575722c5ee1de50aefbe483bb8c88d2c46e65f4d7eecb713b8c1f485"
    ),
    "https://purl.imsglobal.org/spec/ssml/v1p1/schema/xsd/synthesis-nonamespace.xsd": (
        "10608b0b9ad809f78be41e84c655d210fcf27ccc84996ef97cd6dc80b00ace5c"
    ),
    "https://purl.imsglobal.org/spec/w3/1999/schema/xsd/xlink.xsd": (
        "e49074eafa608cbf3bc01c2de3406781d466f6ae6adbb90f26cc450d5bf02a1c"
    ),
    "https://purl.imsglobal.org/spec/w3/2001/schema/xsd/XInclude.xsd": (
        "b72e1d2a0ab90aca502b827ce12572bfb4ef787fee89b4dcc1ce474dd3c00df0"
    ),
    "https://purl.imsglobal.org/spec/w3/2001/schema/xsd/xml.xsd": (
        "4c637d0381d988dbf24b680becde6e1c8301b4c4ceef8bfa45a1ee130e77a682"
    ),
}


def xsd_dir() -> Path:
    """Return the mirror folder: `QUESTXTRACT_QTI_XSD_DIR`, else `.cache/qti-xsd`."""
    configured = os.environ.get("QUESTXTRACT_QTI_XSD_DIR")
    return Path(configured) if configured else _ROOT / ".cache" / "qti-xsd"


def local_path(url: str, root: Path) -> Path:
    """Return where `url` is kept under `root`: `<host>/<path>`."""
    return root.joinpath(*url.removeprefix("https://").split("/"))


def missing(root: Path) -> list[str]:
    """Return the pinned URLs with no file under `root`, or one with another hash."""
    return [url for url, digest in PINNED.items() if _sha256(local_path(url, root)) != digest]


def uri_mapper(root: Path) -> Callable[[str], str]:
    """Return a function that maps each pinned URL to its file under `root`, as a file URI.

    Any other URI is returned unchanged, so a schema that imports something unpinned fails
    when it is compiled with network access off, instead of downloading it.
    """

    def mapper(uri: str) -> str:
        return local_path(uri, root).resolve().as_uri() if uri in PINNED else uri

    return mapper


def fetch(root: Path) -> None:
    """Download every pinned file missing under `root` and check its hash.

    Raises:
        OSError: If a download fails or a file has another hash than the pinned one.

    """
    for url in missing(root):
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - pinned https
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != PINNED[url]:
            raise OSError(f"{url}: SHA-256 {digest}, expected {PINNED[url]}")
        path = local_path(url, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, help="mirror folder (default: .cache/qti-xsd)")
    args = parser.parse_args(argv)
    root = args.dir or xsd_dir()
    try:
        fetch(root)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"QTI 3.0 XSDs ({len(PINNED)} files) in {root}", file=sys.stderr)
    return 0


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


if __name__ == "__main__":
    sys.exit(main())

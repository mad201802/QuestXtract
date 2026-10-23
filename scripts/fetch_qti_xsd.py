"""Download the 1EdTech QTI 2.1 and 3.0 XSDs once, so the tests validate QTI packages offline.

`tests/test_qti_v2.py` and `tests/test_qti_v3.py` validate every item, test and manifest they
write against the official schemas. The 3.0 item schema alone is 18 MB and imports MathML, SSML
and XInclude over `https`, which libxml2 cannot fetch, and the 2.1 schemas import MathML 2, the
APIP extension and more over `http`. The tests must not reach the network, so this script
downloads the raw bytes of every file in both import trees, 54 files (17 for 3.0, 37 for 2.1),
to a local mirror, `<host>/<path>`, and checks each against a pinned SHA-256. The 2.1 files are
pinned under the `http` URL the schemas name, which is the one `xmlschema` asks for, and
downloaded over `https`, which serves the same bytes. The tests then compile from the mirror
with `uri_mapper()`, which points every pinned URL at its local copy. Files are saved as served:
XInclude.xsd is UTF-16, which `xmlschema.export` would not keep intact. Stdlib only.

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
_QTI_V2P1 = "http://www.imsglobal.org/xsd/qti/qtiv2p1/"
ASI_XSD_V2P1 = _QTI_V2P1 + "imsqti_v2p1p2.xsd"  # items and tests, QTI 2.1.2
CP_XSD_V2P1 = _QTI_V2P1 + "qtiv2p1_imscpv1p2_v1p0.xsd"  # imsmanifest.xml, the QTI 2.1 profile
_MATHML2 = "http://www.w3.org/Math/XMLSchema/mathml2/"

# Every file the four schemas import, directly or not, with the SHA-256 of its bytes as served.
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
    # QTI 2.1: keyed by the http URL the schemas name, downloaded over https.
    ASI_XSD_V2P1: "512d67e02f051177a4502700b0fd9970702d771b8f598c9893ad79102fc01337",
    CP_XSD_V2P1: "0c230903d0edabfbbe7bfb84b5cb98b5e3b4d6dd4c9a8839c961bf8fe4021b62",
    "http://www.imsglobal.org/xsd/w3/2001/xml.xsd": (
        "1d8a08f612250dcb59b9e4a9568afd764bb5acf0fc52fc5837ae22379f75f630"
    ),
    "http://www.imsglobal.org/xsd/w3/2001/XInclude.xsd": (
        "b72e1d2a0ab90aca502b827ce12572bfb4ef787fee89b4dcc1ce474dd3c00df0"
    ),
    _MATHML2 + "mathml2.xsd": "ec7caf6358de24b2011aa7144a1ab4e6e9d00a25ed163a13a03c6dfbb6b80f3e",
    "http://www.imsglobal.org/profile/apip/apipv1p0/apipv1p0_qtiextv2p1_v1p0.xsd": (
        "9c4c35f4a4a6901281152a46d4704d990212ac7c4926113f450403012709df5c"
    ),
    "http://www.imsglobal.org/xsd/imsmd_loose_v1p3p2.xsd": (
        "1ae9e77aba63b6789574631213c97fd00a3d43aeffb58050ac7befb665fba9f6"
    ),
    _QTI_V2P1 + "imsqti_metadata_v2p1p1.xsd": (
        "31ddb2ab45f58f033750eec5da3f2310634984387a0babdc4dad859e9724f9dc"
    ),
    _MATHML2 + "common/math.xsd": (
        "8b5c502d74c3df7f4008dea98d4e876841721a69e492e0a2909ccc324c0c3739"
    ),
    _MATHML2 + "common/common-attribs.xsd": (
        "4d5de0a3e1be3b0848f7b9cde6474dde4e91c19f691d3c3b989fff37e5544c12"
    ),
    _MATHML2 + "presentation/common-types.xsd": (
        "35038ae8d324d7fb012925a8bc9717dd3a9d62c90d93bcaf36f31ee1c917a719"
    ),
    _MATHML2 + "presentation/common-attribs.xsd": (
        "1620530c7d2c69c02b03d2caaad55abe70f9e9cc96a735871a81e5aaca799c3d"
    ),
    _MATHML2 + "presentation/characters.xsd": (
        "ab3cdba7c60668ed538f5091d66b29ffbceb61c47611f8bdcc8b56a2bfbe56b3"
    ),
    _MATHML2 + "presentation/tokens.xsd": (
        "e61ddae3b2e16d7fcb5906798ed71891782fe176b284b3d0f2a5cc9db0829ca3"
    ),
    _MATHML2 + "presentation/scripts.xsd": (
        "b4a4b3a1671393b16085898a10c81cdfc853cb7ba6ad465f9f7a537780fd4d28"
    ),
    _MATHML2 + "presentation/space.xsd": (
        "3fa2b55a75e9b2721dd25ae372e1cd0027b15f029d76dc080feca3b7a3ad6aec"
    ),
    _MATHML2 + "presentation/layout.xsd": (
        "533693d7b1758904f88518bbf612ad165dd8a9b39eced28c74f71dd476f62bcf"
    ),
    _MATHML2 + "presentation/table.xsd": (
        "c2d3d5c92e4e2ed9af7c4ed29d86a83c4c01f1c6d5ec167483b5f4d5f8aa7982"
    ),
    _MATHML2 + "presentation/style.xsd": (
        "76ad5dc2f60304b5d80e7a2028bede255e5cf3b84f01095dd39e897d26276e7a"
    ),
    _MATHML2 + "presentation/error.xsd": (
        "f90f2bc05f50d295c82a2b08dc85d382c42a88a5a61ba9c14613f952fb0de200"
    ),
    _MATHML2 + "presentation/action.xsd": (
        "9e1dfc3d0f9368f2247ef2346cd060e897954da86408a0aa780da302cacec54e"
    ),
    _MATHML2 + "content/common-attrib.xsd": (
        "1d1698b91f2d425d522dbdd06b0735a81924e0fda8180902ee01ba5950fe3f8d"
    ),
    _MATHML2 + "content/tokens.xsd": (
        "33a36e2df4af73a6401359efa4f325894217f0448cf91f78f5fb7991798d145d"
    ),
    _MATHML2 + "content/arith.xsd": (
        "b203e803db7010550cd76474f81c4a9316b7588d7567263921ce5e206b334541"
    ),
    _MATHML2 + "content/functions.xsd": (
        "1fab2b4008fd9ec111620649aea7155608a87566a7e82ce83982714f1eaae8c7"
    ),
    _MATHML2 + "content/logic.xsd": (
        "704c70b9fcdb3ed2cdf8e42da7b759c1d88882bf170cda08449590da7fdd0184"
    ),
    _MATHML2 + "content/constructs.xsd": (
        "6b66cde43783dcfa5133cbada91b8cdae7eed11065665d3eaa5fea9ed9086780"
    ),
    _MATHML2 + "content/constants.xsd": (
        "2c9535bd0b0e4f4221dbc64e00fcc6df760711a8fbef405d308a8f839126ba16"
    ),
    _MATHML2 + "content/elementary-functions.xsd": (
        "f21e5cf7bf5a4233f6207e8b7d600da428a4acb96c10c42122c8930e38bc2ea3"
    ),
    _MATHML2 + "content/relations.xsd": (
        "32c1d030ff75ea3a01731ce5069ad834d9b637a5fdbcab69a7e95af648c85a9e"
    ),
    _MATHML2 + "content/semantics.xsd": (
        "6e4f7c1aa901f31f09039cc9a8bee0de04dfbb0e6769c696ed4c4ee5605edb25"
    ),
    _MATHML2 + "content/sets.xsd": (
        "2bc963ad8455308046c8c5ff4201a8f730a7c3f50345d3083948cd16b8715d0c"
    ),
    _MATHML2 + "content/linear-algebra.xsd": (
        "944dc4cbfd7a2de204e994e06dcdf3e078a5dd92d11dc941814b10484ab675e1"
    ),
    _MATHML2 + "content/calculus.xsd": (
        "839003dc4f19234aae570f351bd5217acdbf30ac2f05a55db32c86208a61be66"
    ),
    _MATHML2 + "content/vector-calculus.xsd": (
        "ca678f3018432009f111c78df09be7627652d5f89b40f5095ad9ee710436866b"
    ),
    _MATHML2 + "content/statistics.xsd": (
        "e48517a6b60bd07a9699a62f47dc08eea1ef2573da3ef5dbabb5f368d5c5d2a1"
    ),
    _MATHML2 + "common/xlink-href.xsd": (
        "08c9410c87a16bb9e3623f74d818277ec92308c85e61fda1b3857c778b787ecc"
    ),
}


def xsd_dir() -> Path:
    """Return the mirror folder: `QUESTXTRACT_QTI_XSD_DIR`, else `.cache/qti-xsd`."""
    configured = os.environ.get("QUESTXTRACT_QTI_XSD_DIR")
    return Path(configured) if configured else _ROOT / ".cache" / "qti-xsd"


def local_path(url: str, root: Path) -> Path:
    """Return where `url` is kept under `root`: `<host>/<path>`, for http and https alike."""
    return root.joinpath(*url.removeprefix("https://").removeprefix("http://").split("/"))


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
        with urllib.request.urlopen(_https(url), timeout=60) as response:  # noqa: S310 - pinned
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
    print(f"QTI 2.1 and 3.0 XSDs ({len(PINNED)} files) in {root}", file=sys.stderr)
    return 0


def _https(url: str) -> str:
    """Return the address `url` is downloaded from: always https, which serves the same bytes."""
    return "https://" + url.removeprefix("http://") if url.startswith("http://") else url


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


if __name__ == "__main__":
    sys.exit(main())

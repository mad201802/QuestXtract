from importlib.metadata import metadata, requires

import questxtract


def test_distribution_metadata_declares_supported_installation() -> None:
    package_metadata = metadata("questxtract")
    dependencies = requires("questxtract") or []

    assert package_metadata["Name"] == "questxtract"
    assert package_metadata["Requires-Python"] == ">=3.10"
    assert "av>=14.0" in dependencies
    assert questxtract.__version__ == package_metadata["Version"]

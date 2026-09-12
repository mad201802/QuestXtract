import pytest

from questxtract.media import _HASH_BYTES, ingest, source_hash


def test_hash_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"lecture")
    assert source_hash(a) == source_hash(a)

    b = tmp_path / "b.bin"
    b.write_bytes(b"lecturf")
    assert source_hash(a) != source_hash(b)


def test_hash_reads_both_ends_of_a_large_file(tmp_path):
    """Only the first and last 8 MiB are read, so a tail-only change must still register."""
    head = b"\x01" * _HASH_BYTES
    tail = b"\x02" * _HASH_BYTES
    big = tmp_path / "big.bin"
    big.write_bytes(head + b"\x00" * 1024 + tail)
    other = tmp_path / "other.bin"
    other.write_bytes(head + b"\x00" * 1024 + tail[:-1] + b"\x03")
    assert source_hash(big) != source_hash(other)


def test_size_alone_changes_the_hash(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"x" * 10)
    b = tmp_path / "b.bin"
    b.write_bytes(b"x" * 11)
    assert source_hash(a) != source_hash(b)


def test_ingest_rejects_a_missing_source(context):
    with pytest.raises(FileNotFoundError):
        ingest(context)


def test_ingest_hashes_before_it_gives_up(tmp_path, context):
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"not really an mp4")
    context.source = media
    with pytest.raises(NotImplementedError, match="#7"):
        ingest(context)
    assert context.source_hash == source_hash(media)

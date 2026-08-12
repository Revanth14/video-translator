import tracemalloc
from io import BytesIO
from pathlib import Path

import pytest

from dub_mvp.upload import UploadError, parse_multipart_stream

BOUNDARY = "test-boundary"
CONTENT_TYPE = f"multipart/form-data; boundary={BOUNDARY}"


class ChunkedStream:
    """A stream that returns small reads, as a real socket does.

    Delimiters then land across read boundaries, which is where naive
    multipart parsers truncate or corrupt the payload.
    """

    def __init__(self, data: bytes, chunk: int) -> None:
        self._data = data
        self._chunk = chunk
        self._position = 0

    def read(self, size: int) -> bytes:
        take = min(size, self._chunk, len(self._data) - self._position)
        data = self._data[self._position : self._position + take]
        self._position += take
        return data


def build_body(video: bytes, *, language: bytes = b"hi") -> bytes:
    return (
        f"--{BOUNDARY}\r\n".encode()
        + b'Content-Disposition: form-data; name="language"\r\n\r\n'
        + language
        + f"\r\n--{BOUNDARY}\r\n".encode()
        + b'Content-Disposition: form-data; name="video"; '
        + b'filename="clip.mp4"\r\n'
        + b"Content-Type: video/mp4\r\n\r\n"
        + video
        + f"\r\n--{BOUNDARY}--\r\n".encode()
    )


def parse(body: bytes, *, destination: Path | None = None, chunk: int = 1 << 20):
    return parse_multipart_stream(
        ChunkedStream(body, chunk),
        content_type=CONTENT_TYPE,
        content_length=len(body),
        file_sink=(lambda name, filename: destination if name == "video" else None),
    )


def test_streams_the_file_part_to_disk(tmp_path: Path) -> None:
    destination = tmp_path / "staged.upload"
    video = bytes(range(256)) * 64

    parts = parse(build_body(video), destination=destination)

    assert parts["video"].path == destination
    assert destination.read_bytes() == video
    assert parts["video"].size == len(video)
    assert parts["video"].filename == "clip.mp4"
    assert parts["language"].content == b"hi"


@pytest.mark.parametrize("chunk", [1, 3, 7, 64, 4096])
def test_payload_survives_delimiters_split_across_reads(
    tmp_path: Path,
    chunk: int,
) -> None:
    destination = tmp_path / f"staged-{chunk}.upload"
    # Content that contains near-misses of the delimiter must not truncate.
    video = b"\r\n--test-boundar" + b"\x00\xff" * 512 + b"\r\n--not-it\r\n"

    parse(build_body(video), destination=destination, chunk=chunk)

    assert destination.read_bytes() == video


def test_memory_stays_flat_for_a_large_upload(tmp_path: Path) -> None:
    destination = tmp_path / "large.upload"
    video = b"\0" * (16 * 1024 * 1024)
    body = build_body(video)

    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        parse(body, destination=destination)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Buffering would cost at least the payload; streaming stays near one chunk.
    assert peak - base < 8 * 1024 * 1024
    assert destination.stat().st_size == len(video)


def test_buffered_field_limit_is_enforced() -> None:
    body = build_body(b"video", language=b"x" * 4096)

    with pytest.raises(UploadError, match="exceeds"):
        parse_multipart_stream(
            BytesIO(body),
            content_type=CONTENT_TYPE,
            content_length=len(body),
            max_field_bytes=1024,
        )


def test_truncated_upload_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "partial.upload"
    body = build_body(b"video-bytes")[:-10]

    with pytest.raises(UploadError, match="terminated"):
        parse_multipart_stream(
            BytesIO(body),
            content_type=CONTENT_TYPE,
            content_length=len(body),
            file_sink=lambda name, filename: destination,
        )

    # A rejected upload must not leave a partial file behind.
    assert not destination.exists()


def test_missing_boundary_and_empty_body_are_rejected() -> None:
    with pytest.raises(UploadError, match="multipart"):
        parse_multipart_stream(
            BytesIO(b"body"),
            content_type="application/json",
            content_length=4,
        )

    with pytest.raises(UploadError, match="empty"):
        parse_multipart_stream(
            BytesIO(b""),
            content_type=CONTENT_TYPE,
            content_length=0,
        )

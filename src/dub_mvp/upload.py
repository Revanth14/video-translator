from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

CHUNK_BYTES = 1024 * 1024
MAX_FIELD_BYTES = 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024


class UploadError(RuntimeError):
    pass


@dataclass
class StreamedPart:
    """One multipart field. Large files live on disk, small fields in memory."""

    name: str
    filename: str | None = None
    content: bytes | None = None
    path: Path | None = None
    size: int = 0


def parse_multipart_stream(
    stream: BinaryIO,
    *,
    content_type: str,
    content_length: int,
    file_sink: Callable[[str, str | None], Path | None] | None = None,
    max_field_bytes: int = MAX_FIELD_BYTES,
) -> dict[str, StreamedPart]:
    """Parse a multipart body without holding it in memory.

    A creator video is gigabytes; reading the whole body and splitting it costs
    several times its size in RAM and cannot work on a worker that also holds
    model weights. `file_sink` returns a destination path for the parts that
    should stream to disk, and any other field is buffered under
    `max_field_bytes`.
    """
    if content_length <= 0:
        raise UploadError("Upload is empty.")

    delimiter = b"--" + _boundary(content_type)
    reader = _Reader(stream, content_length)
    if not reader.discard_until(delimiter):
        raise UploadError("Upload did not contain the multipart boundary.")

    terminator = b"\r\n" + delimiter
    parts: dict[str, StreamedPart] = {}

    while True:
        marker = reader.fill(2)[:2]
        if marker == b"--":
            break
        if marker != b"\r\n":
            raise UploadError("Malformed multipart delimiter.")
        reader.take(2)

        headers = _read_headers(reader)
        disposition = headers.get("content-disposition", "")
        name = _disposition_value(disposition, "name")
        filename = _disposition_value(disposition, "filename")

        if not name:
            _consume(reader, terminator)
            continue

        destination = file_sink(name, filename) if file_sink else None
        if destination is None:
            content, size = _buffer_part(reader, terminator, name, max_field_bytes)
            parts[name] = StreamedPart(
                name=name,
                filename=filename,
                content=content,
                size=size,
            )
            continue

        size = _stream_part_to_disk(reader, terminator, destination)
        parts[name] = StreamedPart(
            name=name,
            filename=filename,
            path=destination,
            size=size,
        )

    return parts


class _Reader:
    """A bounded, buffered view over the request body."""

    def __init__(self, stream: BinaryIO, remaining: int) -> None:
        self._stream = stream
        self._remaining = remaining
        self._buffer = b""

    @property
    def buffer(self) -> bytes:
        return self._buffer

    @property
    def exhausted(self) -> bool:
        return self._remaining <= 0

    def fill(self, size: int) -> bytes:
        while len(self._buffer) < size and self._remaining > 0:
            chunk = self._stream.read(min(CHUNK_BYTES, self._remaining))
            if not chunk:
                self._remaining = 0
                break
            self._remaining -= len(chunk)
            self._buffer += chunk
        return self._buffer

    def take(self, size: int) -> bytes:
        data = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return data

    def discard_until(self, needle: bytes) -> bool:
        while True:
            self.fill(CHUNK_BYTES)
            index = self._buffer.find(needle)
            if index >= 0:
                self.take(index + len(needle))
                return True
            if self.exhausted:
                return False
            # Keep enough tail to match a needle split across two reads.
            keep = len(needle) - 1
            if len(self._buffer) > keep:
                self._buffer = self._buffer[-keep:]


def _read_part_body(
    reader: _Reader,
    terminator: bytes,
    write: Callable[[bytes], int | None],
) -> int:
    """Feed a part's bytes to `write` until the next boundary."""
    written = 0
    keep = len(terminator) - 1
    while True:
        reader.fill(CHUNK_BYTES + len(terminator))
        index = reader.buffer.find(terminator)
        if index >= 0:
            written += _emit(write, reader.take(index))
            reader.take(len(terminator))
            return written
        if reader.exhausted:
            raise UploadError("Upload ended before the part was terminated.")
        # Hold back a possible partial terminator before flushing the rest.
        available = len(reader.buffer) - keep
        if available > 0:
            written += _emit(write, reader.take(available))


def _emit(write: Callable[[bytes], int | None], data: bytes) -> int:
    if not data:
        return 0
    write(data)
    return len(data)


def _consume(reader: _Reader, terminator: bytes) -> None:
    _read_part_body(reader, terminator, lambda data: len(data))


def _buffer_part(
    reader: _Reader,
    terminator: bytes,
    name: str,
    max_field_bytes: int,
) -> tuple[bytes, int]:
    chunks: list[bytes] = []
    total = 0

    def write(data: bytes) -> int:
        nonlocal total
        total += len(data)
        if total > max_field_bytes:
            raise UploadError(
                f"Form field '{name}' exceeds {max_field_bytes} bytes."
            )
        chunks.append(data)
        return len(data)

    size = _read_part_body(reader, terminator, write)
    return b"".join(chunks), size


def _stream_part_to_disk(
    reader: _Reader,
    terminator: bytes,
    destination: Path,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("wb") as handle:
            size = _read_part_body(reader, terminator, handle.write)
            handle.flush()
            os.fsync(handle.fileno())
    except UploadError:
        destination.unlink(missing_ok=True)
        raise
    return size


def _read_headers(reader: _Reader) -> dict[str, str]:
    while b"\r\n\r\n" not in reader.buffer:
        if len(reader.buffer) > MAX_HEADER_BYTES:
            raise UploadError("Multipart part headers are too large.")
        if reader.exhausted:
            raise UploadError("Upload ended inside part headers.")
        reader.fill(len(reader.buffer) + CHUNK_BYTES)

    blob, _, _ = reader.buffer.partition(b"\r\n\r\n")
    reader.take(len(blob) + 4)

    headers: dict[str, str] = {}
    for line in blob.decode("utf-8", errors="replace").split("\r\n"):
        key, _, value = line.partition(":")
        if key and value:
            headers[key.strip().lower()] = value.strip()
    return headers


def _boundary(content_type: str) -> bytes:
    marker = "boundary="
    if marker not in content_type:
        raise UploadError("Expected a multipart form upload.")
    boundary = content_type.split(marker, 1)[1].split(";", 1)[0].strip()
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]
    if not boundary:
        raise UploadError("Multipart boundary is empty.")
    return boundary.encode("utf-8")


def _disposition_value(disposition: str, key: str) -> str | None:
    for item in disposition.split(";"):
        name, _, value = item.strip().partition("=")
        if name == key:
            return value.strip().strip('"') or None
    return None

"""Transport interface: framed bytes in, framed bytes out."""

from __future__ import annotations

from easy_nn import protocol


class Transport:
    def send_frame(self, ftype: int, payload) -> None:
        raise NotImplementedError

    def recv_frame(self) -> tuple[int, bytes]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class StreamTransport(Transport):
    """Framing over any pair of binary streams."""

    def __init__(self, reader, writer):
        self._reader = reader
        self._writer = writer

    def send_frame(self, ftype: int, payload) -> None:
        view = memoryview(payload) if not isinstance(payload, memoryview) else payload
        self._writer.write(protocol.pack_header(ftype, len(view)))
        if len(view):
            self._writer.write(view)
        self._writer.flush()

    def recv_frame(self) -> tuple[int, bytes]:
        head = self._read_exactly(protocol.HEADER_SIZE)
        ftype, length = protocol.unpack_header(head)
        if length > protocol.MAX_FRAME:
            raise protocol.ProtocolError(f"frame of {length} bytes is out of bounds")
        return ftype, self._read_exactly(length)

    def _read_exactly(self, n: int) -> bytes:
        if n == 0:
            return b""
        chunks = []
        remaining = n
        while remaining:
            chunk = self._reader.read(remaining)
            if not chunk:
                raise EOFError("connection closed mid-frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks) if len(chunks) > 1 else chunks[0]

    def close(self) -> None:
        for stream in (self._writer, self._reader):
            try:
                stream.close()
            except Exception:
                pass

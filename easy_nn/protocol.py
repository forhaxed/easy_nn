"""
Wire protocol.

A frame is ``[type u8][len u64][payload]``.  A *message* is one header frame
optionally followed by raw part frames -- the header says how many to expect.

There is deliberately no multiplexing.  TCP is already full duplex and each
direction has exactly one writer, so messages are strictly sequential per
direction: the client can send a control message while the server streams logs
back, and neither has to interleave.
"""

from __future__ import annotations

import pickle
import struct
import threading

PROTO_VERSION = 3

# Message types (header frames)
HELLO = 1
WELCOME = 2
JOB = 3
ACCEPTED = 4
ERROR = 5
BLOB = 6
STREAM_END = 8
CTRL = 9
CREDIT = 10
LOG = 11
PRINT = 12
PROGRESS = 13
CHECKPOINT = 14
DONE = 15
PING = 16
PONG = 17
ARTIFACT = 18
SETUP = 19
READY = 20

# Raw payload frame, always preceded by a header frame that announces it.
PART = 0xF0

NAMES = {
    v: k
    for k, v in list(globals().items())
    if isinstance(v, int) and k.isupper() and k != "PROTO_VERSION"
}

_HEADER = struct.Struct("!BQ")
HEADER_SIZE = _HEADER.size

MAX_FRAME = 8 << 30  # 8 GiB, a sanity bound rather than a real limit


def pack_header(ftype: int, length: int) -> bytes:
    return _HEADER.pack(ftype, length)


def unpack_header(raw: bytes) -> tuple[int, int]:
    return _HEADER.unpack(raw)


class Message:
    """A decoded header frame plus its raw parts."""

    __slots__ = ("type", "meta", "header", "parts")

    def __init__(self, mtype: int, meta: dict, header: bytes | None = None, parts=None):
        self.type = mtype
        self.meta = meta
        self.header = header  # codec structure, when the message carries a body
        self.parts = parts or []

    @property
    def body(self):
        """Decode the carried object, or None if the message has no body."""
        if self.header is None:
            return None
        from easy_nn import codec

        return codec.decode(self.header, self.parts)

    def __repr__(self):
        return (
            f"<Message {NAMES.get(self.type, self.type)} "
            f"meta={self.meta} parts={len(self.parts)}>"
        )


class Channel:
    """Message-level view of a transport.  Sends are serialized by a lock."""

    def __init__(self, transport):
        self.transport = transport
        self._send_lock = threading.Lock()

    # -- sending ---------------------------------------------------------
    def send(self, mtype: int, meta: dict | None = None, body=None, on_progress=None):
        """Send one message.  ``body`` is an ``Encoded`` from the codec."""
        meta = meta or {}
        if body is None:
            head = {"meta": meta, "n_parts": None}
            with self._send_lock:
                self.transport.send_frame(mtype, pickle.dumps(head))
            return

        head = {
            "meta": meta,
            "n_parts": len(body.parts),
            "header_len": len(body.header),
        }
        with self._send_lock:
            self.transport.send_frame(mtype, pickle.dumps(head))
            self.transport.send_frame(PART, body.header)
            sent = 0
            for part in body.parts:
                self.transport.send_frame(PART, part)
                sent += len(part)
                if on_progress is not None:
                    on_progress(sent)

    # -- receiving -------------------------------------------------------
    def recv(self) -> Message:
        """Read one whole message, blocking.  Raises EOFError on clean close."""
        mtype, payload = self.transport.recv_frame()
        if mtype == PART:
            raise ProtocolError("unexpected part frame outside a message")
        head = pickle.loads(payload)
        n_parts = head["n_parts"]
        if n_parts is None:
            return Message(mtype, head["meta"])

        ftype, header = self.transport.recv_frame()
        if ftype != PART:
            raise ProtocolError("expected body header frame")
        parts = []
        for _ in range(n_parts):
            ftype, part = self.transport.recv_frame()
            if ftype != PART:
                raise ProtocolError("expected body part frame")
            parts.append(part)
        return Message(mtype, head["meta"], header, parts)

    def close(self):
        self.transport.close()


class ProtocolError(Exception):
    pass

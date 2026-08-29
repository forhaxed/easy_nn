import socket
import threading

import pytest
import torch

from easy_nn import codec, protocol
from easy_nn.transport.base import StreamTransport


def make_pair():
    a, b = socket.socketpair()
    ta = StreamTransport(a.makefile("rb"), a.makefile("wb"))
    tb = StreamTransport(b.makefile("rb"), b.makefile("wb"))
    return protocol.Channel(ta), protocol.Channel(tb)


def test_message_without_body():
    left, right = make_pair()
    left.send(protocol.CTRL, {"action": "pause"})
    msg = right.recv()
    assert msg.type == protocol.CTRL
    assert msg.meta == {"action": "pause"}
    assert msg.body is None


def test_message_with_tensor_body():
    left, right = make_pair()
    payload = {"latents": torch.randn(4, 8), "captions": ["a", "b"]}
    left.send(protocol.BLOB, {"seq": 7, "kind": "train"}, codec.encode(payload))

    msg = right.recv()
    assert msg.type == protocol.BLOB
    assert msg.meta == {"seq": 7, "kind": "train"}
    assert len(msg.parts) == 1
    out = msg.body
    assert torch.equal(out["latents"], payload["latents"])
    assert out["captions"] == ["a", "b"]


def test_progress_is_reported_per_part():
    left, right = make_pair()
    body = codec.encode([torch.zeros(1000) for _ in range(4)])
    seen = []

    reader = threading.Thread(target=right.recv)
    reader.start()
    left.send(protocol.JOB, {}, body, on_progress=seen.append)
    reader.join(timeout=10)

    assert len(seen) == 4
    assert seen == sorted(seen)
    assert seen[-1] == sum(len(p) for p in body.parts)


def test_messages_keep_order_and_are_independent():
    left, right = make_pair()
    left.send(protocol.PRINT, {"text": "first"})
    left.send(protocol.LOG, {"step": 1, "values": {"loss": 0.5}})
    left.send(protocol.DONE)

    assert right.recv().meta["text"] == "first"
    assert right.recv().meta["values"] == {"loss": 0.5}
    assert right.recv().type == protocol.DONE


def test_closed_connection_raises_eof():
    left, right = make_pair()
    left.close()
    with pytest.raises(EOFError):
        right.recv()


def test_oversized_frame_is_rejected():
    left, right = make_pair()
    left.transport._writer.write(protocol.pack_header(protocol.BLOB, protocol.MAX_FRAME + 1))
    left.transport._writer.flush()
    with pytest.raises(protocol.ProtocolError, match="out of bounds"):
        right.recv()


def test_byte_counters_track_both_directions():
    left, right = make_pair()
    assert left.transport.bytes_sent == 0
    assert right.transport.bytes_received == 0

    body = codec.encode({"weights": torch.zeros(5000)})
    left.send(protocol.JOB, {"job": 1}, body)
    right.recv()

    sent = left.transport.bytes_sent
    received = right.transport.bytes_received
    assert sent == received, "every byte written was read"
    # Header frames plus the tensor itself; 5000 floats is 20000 bytes.
    assert sent > 20000


def test_counters_include_framing_overhead():
    left, right = make_pair()
    left.send(protocol.DONE)
    right.recv()
    # A bodyless message is still a header frame plus a pickled meta payload.
    assert left.transport.bytes_sent > protocol.HEADER_SIZE
    assert right.transport.bytes_received == left.transport.bytes_sent

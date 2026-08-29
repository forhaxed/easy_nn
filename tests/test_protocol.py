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


def test_output_before_acceptance_is_printed_not_fatal():
    """A cold pod installs requirements before it can accept the job, and says
    so as it goes. Local() never installs anything, so only a real pod hits
    this path -- it has to be covered here instead."""
    from easy_nn.client.session import _await_accepted
    from easy_nn.client.sinks import ConsoleSink

    left, right = make_pair()
    console = ConsoleSink(enabled=False)

    left.send(protocol.PRINT, {"text": "Installing 5 packages: diffusers==0.37.1\n"})
    left.send(protocol.PRINT, {"text": "Successfully installed diffusers\n"})
    left.send(protocol.ACCEPTED, {"job_id": "abc"})

    accepted = _await_accepted(right, console)
    assert accepted.meta["job_id"] == "abc"


def test_a_failure_during_setup_surfaces_instead_of_hanging():
    from easy_nn.client.session import RemoteError, _await_accepted
    from easy_nn.client.sinks import ConsoleSink

    left, right = make_pair()
    left.send(protocol.PRINT, {"text": "Installing 5 packages\n"})
    left.send(protocol.ERROR, {"message": "pip install failed (1)"})

    with pytest.raises(RemoteError, match="pip install failed"):
        _await_accepted(right, ConsoleSink(enabled=False))


def test_requirements_are_installed_before_the_weights_are_sent():
    """A version mismatch is fatal on the executor. Discovering it after a
    multi-gigabyte upload wastes the upload, so SETUP goes first."""
    from easy_nn.client.session import _install_requirements
    from easy_nn.client.sinks import ConsoleSink

    class Executor:
        installs_requirements = True

    class Job:
        requirements = ["diffusers==0.37.1", "peft==0.18.1"]

    left, right = make_pair()

    def server():
        request = right.recv()
        assert request.type == protocol.SETUP
        assert request.meta["requirements"] == Job.requirements
        right.send(protocol.PRINT, {"text": "Installing 2 packages\n"})
        right.send(protocol.READY, {})

    thread = threading.Thread(target=server)
    thread.start()
    _install_requirements(Job(), Executor(), left, ConsoleSink(enabled=False))
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_a_local_executor_still_completes_the_setup_step():
    """Local installs nothing, but the exchange has to happen anyway so both
    sides stay in lockstep."""
    from easy_nn.client.session import _install_requirements
    from easy_nn.client.sinks import ConsoleSink

    class Executor:
        installs_requirements = False

    class Job:
        requirements = ["diffusers==0.37.1"]

    left, right = make_pair()
    seen = []

    def server():
        request = right.recv()
        seen.append(request.meta["requirements"])
        right.send(protocol.READY, {})

    thread = threading.Thread(target=server)
    thread.start()
    _install_requirements(Job(), Executor(), left, ConsoleSink(enabled=False))
    thread.join(timeout=10)

    assert seen == [[]], "nothing to install locally"


def test_a_torch_mismatch_is_refused_before_the_upload():
    """cloudpickle records submodules it thinks the code needs -- an attribute
    called `accelerator` is enough to pull in torch.accelerator, which exists
    only from torch 2.6. Unpickling then dies, but only after the whole model
    has been sent. So the versions get compared first."""
    from easy_nn.client.session import RemoteError, _check_compatibility

    with pytest.raises(RemoteError, match="torch mismatch"):
        _check_compatibility(None, "2.4.1+cu124")

    message = None
    try:
        _check_compatibility(None, "2.4.1+cu124")
    except RemoteError as exc:
        message = str(exc)
    assert "2.4.1+cu124" in message
    assert torch.__version__ in message
    assert "before easy-nn-server" in message


def test_the_same_torch_series_is_accepted():
    from easy_nn.client.session import _check_compatibility

    series = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    _check_compatibility(None, f"{series}.99+cpu")   # patch differences are fine
    _check_compatibility(None, torch.__version__)
    _check_compatibility(None, None)                 # an old executor says nothing


def test_the_mismatch_check_can_be_overridden():
    from easy_nn.client.session import _check_compatibility

    _check_compatibility("3.7.0", "1.13.0", allow_mismatch=True)


def test_setup_names_the_modules_the_executor_must_import():
    """Version numbers are not enough: a torchvision built against a different
    torch installs cleanly, imports as a package, then dies on its first
    operator -- and transformers imports it for you."""
    from easy_nn.client.session import _imports_to_verify

    class T:
        verify_imports = None

    names = _imports_to_verify(T(), ["diffusers==0.37.1", "transformers==5.2.0"])
    assert names[0] == "torch"
    assert "diffusers" in names and "transformers" in names
    assert "torchvision" in names, "this side has it, so the executor's must work"
    assert len(names) == len(set(names)), "no duplicates"


def test_verify_imports_can_be_set_explicitly():
    from easy_nn.client.session import _imports_to_verify

    class T:
        verify_imports = ["torch", "cv2"]

    assert _imports_to_verify(T(), ["opencv-python"]) == ["torch", "cv2"]


def test_a_broken_executor_environment_is_caught_before_the_upload():
    from easy_nn.server.runner import Runner

    left, right = make_pair()
    runner = Runner(right)

    with pytest.raises(RuntimeError, match="cannot import 'definitely_not_a_module'"):
        runner._install([], ["definitely_not_a_module"])


def test_a_healthy_environment_reports_what_it_verified():
    from easy_nn.server.runner import Runner

    left, right = make_pair()
    runner = Runner(right)
    runner._install([], ["torch", "json"])

    message = left.recv()
    assert message.type == protocol.PRINT
    assert "Verified imports: torch, json" in message.meta["text"]


def test_a_python_mismatch_is_refused_before_the_upload():
    """cloudpickle ships methods as bytecode, and bytecode is tied to the
    interpreter that produced it. Running 3.11 bytecode under 3.12 does not
    fail cleanly -- it misexecutes, and an innocent `assert x == 1` reports
    'too many values to unpack'. That has to be caught before the upload."""
    import platform

    from easy_nn.client.session import RemoteError, _check_compatibility

    ours = platform.python_version()
    major, minor = ours.split(".")[:2]
    theirs = f"{major}.{int(minor) + 1}.0"

    with pytest.raises(RemoteError, match="Python mismatch") as excinfo:
        _check_compatibility(theirs, torch.__version__)

    message = str(excinfo.value)
    assert theirs in message and ours in message
    assert f"Python {major}.{minor}" in message, "must name the version to use"


def test_a_matching_python_passes():
    import platform

    from easy_nn.client.session import _check_compatibility

    major, minor = platform.python_version().split(".")[:2]
    # A different patch release is fine; bytecode is stable within a minor.
    _check_compatibility(f"{major}.{minor}.99", torch.__version__)


def test_python_is_checked_before_torch():
    """Both are wrong here; the report has to name the fatal one."""
    from easy_nn.client.session import RemoteError, _check_compatibility

    with pytest.raises(RemoteError, match="Python mismatch"):
        _check_compatibility("2.7.0", "1.13.0")

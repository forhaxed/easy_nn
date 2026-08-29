"""
The RunPod executor, exercised over a loopback socket.

This is the same code path a pod uses -- the TCP transport, the token check and
the full protocol -- with the pod replaced by a server on localhost.  It cannot
prove RunPod's port mapping works, but everything downstream of it is covered.
"""

import os
import socket
import subprocess
import sys
import time

import pytest

from easy_nn import RunPod
from easy_nn.client.session import RemoteError
from tests import toy

TOKEN = "test-token"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_port(port, timeout=120):
    """Poke the port until it answers.

    The poke opens and drops a connection without saying HELLO, so it doubles
    as a check that a health probe or a port scan does not take the executor
    down -- the pod's port is reachable from the internet.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@pytest.fixture
def server(tmp_path):
    """An easy-nn-server on loopback, in a directory we can inspect afterwards."""
    port = free_port()
    workdir = tmp_path / "executor"
    workdir.mkdir()

    environment = dict(os.environ)
    import easy_nn

    root = os.path.dirname(os.path.dirname(os.path.abspath(easy_nn.__file__)))
    environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")

    process = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "easy_nn.server.app",
            "--host", "127.0.0.1", "--port", str(port),
            "--token", TOKEN, "--once",
        ],
        cwd=str(workdir),
        env=environment,
    )
    try:
        assert wait_for_port(port), "server never came up"
        yield port, workdir
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)


def test_trains_over_tcp_without_touching_the_server_directory(server, tmp_path):
    port, workdir = server

    trainer = toy.build(tmp_path, save_checkpoint_every_steps=16)
    result = trainer.train(on=RunPod(host="127.0.0.1", port=port, token=TOKEN))

    assert result["global_step"] == 32
    assert (tmp_path / "output" / "checkpoints" / "step_16").is_dir()
    assert list(workdir.iterdir()) == [], "the executor wrote to its own disk"


def test_a_wrong_token_is_refused(server, tmp_path):
    port, _ = server

    trainer = toy.build(tmp_path)
    with pytest.raises(RemoteError, match="bad token"):
        trainer.train(on=RunPod(host="127.0.0.1", port=port, token="wrong"))


def test_an_unreachable_executor_fails_fast_with_the_address(tmp_path):
    trainer = toy.build(tmp_path)
    executor = RunPod(
        host="127.0.0.1", port=free_port(), connect_timeout=1.0, retry_every=0.2
    )
    with pytest.raises(ConnectionError, match=r"could not reach easy-nn-server"):
        trainer.train(on=executor)

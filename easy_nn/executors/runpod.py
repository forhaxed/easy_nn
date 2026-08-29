"""
The RunPod executor.

Start ``easy-nn-server`` on a pod once (see ``deploy/start_command.txt``), note
the address it prints, and point this at it.  Nothing else about the pod needs
setting up: the job carries its own code, its own weights and its own list of
packages to install.
"""

from __future__ import annotations

import time

from easy_nn.executors.base import Executor
from easy_nn.transport.tcp import TcpTransport
from easy_nn import protocol


class RunPod(Executor):
    """Connect to an ``easy-nn-server`` over direct TCP.

    ``host`` and ``port`` are the *external* pair RunPod maps for you --
    ``RUNPOD_PUBLIC_IP`` and ``RUNPOD_TCP_PORT_<internal>`` inside the
    container.  The server prints the ready-made call on startup.
    """

    installs_requirements = True

    def __init__(
        self,
        host: str,
        port: int,
        token: str | None = None,
        connect_timeout: float = 120.0,
        retry_every: float = 3.0,
    ):
        self.host = host
        self.port = int(port)
        self.token = token
        self.connect_timeout = connect_timeout
        self.retry_every = retry_every
        self.transport = None

    def connect(self):
        deadline = time.time() + self.connect_timeout
        while True:
            try:
                self.transport = TcpTransport.connect(self.host, self.port)
                return protocol.Channel(self.transport)
            except OSError as exc:
                if time.time() >= deadline:
                    raise ConnectionError(
                        f"could not reach easy-nn-server at "
                        f"{self.host}:{self.port} within "
                        f"{self.connect_timeout:.0f}s: {exc}"
                    ) from exc
                time.sleep(self.retry_every)

    def close(self):
        if self.transport is not None:
            self.transport.close()
            self.transport = None

    def describe(self) -> str:
        return f"RunPod({self.host}:{self.port})"

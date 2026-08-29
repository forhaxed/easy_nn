"""
TCP transport.

On RunPod a pod with a public IP gets its container port mapped to an external
one; the container is told both through ``RUNPOD_PUBLIC_IP`` and
``RUNPOD_TCP_PORT_<port>``, which is what the server prints on startup.

The proxy at ``https://<id>-<port>.proxy.runpod.net`` is not used here.  It
would work and needs no public IP, but it is slower for multi-gigabyte weight
uploads.  Adding it later means another Transport, not another protocol.
"""

from __future__ import annotations

import socket

from easy_nn import protocol
from easy_nn.transport.base import StreamTransport

#: Chosen to keep the syscall count low on large tensor frames.
_BUFFER = 1 << 20


class TcpTransport(StreamTransport):
    def __init__(self, sock: socket.socket):
        self.socket = sock
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        super().__init__(sock.makefile("rb", _BUFFER), sock.makefile("wb", _BUFFER))

    @classmethod
    def connect(cls, host: str, port: int, timeout: float = 30.0):
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(None)  # blocking for the life of the job
        return cls(sock)

    def close(self):
        super().close()
        try:
            self.socket.close()
        except OSError:
            pass


class TcpListener:
    """Accepts one client at a time and checks its token."""

    def __init__(self, host: str, port: int, token: str | None = None, backlog=4):
        self.token = token
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.listen(backlog)

    @property
    def port(self) -> int:
        return self.socket.getsockname()[1]

    def accept(self) -> protocol.Channel:
        sock, _ = self.socket.accept()
        return protocol.Channel(TcpTransport(sock))

    def close(self):
        try:
            self.socket.close()
        except OSError:
            pass

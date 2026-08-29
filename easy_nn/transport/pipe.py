"""Transport over a subprocess' stdin/stdout.  Used by the Local executor."""

from __future__ import annotations

from easy_nn.transport.base import StreamTransport


class PipeTransport(StreamTransport):
    """Framed messages over two binary pipes."""

    @classmethod
    def from_popen(cls, proc):
        return cls(proc.stdout, proc.stdin)

    @classmethod
    def stdio(cls):
        """The executor side: talk over this process' own stdin/stdout."""
        import sys

        return cls(sys.stdin.buffer, sys.stdout.buffer)

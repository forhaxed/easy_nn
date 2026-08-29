"""
The executor application.

Upload this once to a pod and leave it running; it will accept whatever job
turns up.  ``--stdio`` is the same program speaking over pipes, which is how
the Local executor runs -- identical protocol, identical serialization, so a
job that works locally works on a pod for the same reasons.
"""

from __future__ import annotations

import argparse
import os
import sys

from easy_nn import protocol
from easy_nn.server.runner import Runner


def _binary_stdio():
    """Take over stdout for the protocol and keep stray prints off it.

    User code prints.  If any of it landed on stdout it would corrupt the
    frame stream, so stdout is claimed here and the text-level ``sys.stdout``
    is pointed at stderr instead.
    """
    if os.name == "nt":
        import msvcrt

        for stream in (sys.stdin, sys.stdout):
            msvcrt.setmode(stream.fileno(), os.O_BINARY)

    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    return reader, writer


def serve_stdio():
    from easy_nn.transport.base import StreamTransport

    reader, writer = _binary_stdio()
    channel = protocol.Channel(StreamTransport(reader, writer))
    Runner(channel).serve_one_job()

    # The reader thread is parked on stdin and will never come back; letting
    # the interpreter finalize around it aborts the process with a lock error.
    # The job is over and the last frame is flushed, so leave immediately.
    writer.flush()
    sys.stderr.flush()
    os._exit(0)


def serve_tcp(host: str, port: int, token: str | None, once: bool = False):
    from easy_nn.transport.tcp import TcpListener

    listener = TcpListener(host, port, token=token)
    _announce(port)
    while True:
        channel = listener.accept()
        served = False
        try:
            served = Runner(channel, token=token).serve_one_job()
        except Exception as exc:  # noqa: BLE001 - one bad client must not end the server
            print(f"connection failed: {exc!r}", file=sys.stderr)
        finally:
            channel.close()
        if once and served:
            return


def _announce(port: int):
    """Print the address a client should dial, including RunPod's mapping."""
    import platform

    import torch

    import easy_nn

    public_ip = os.environ.get("RUNPOD_PUBLIC_IP")
    mapped = os.environ.get(f"RUNPOD_TCP_PORT_{port}")
    print(f"easy_nn executor listening on port {port}", file=sys.stderr)
    # Which image did this pod actually get? Jobs are refused outright when the
    # torch series differs from the client's, so put it where the log shows it.
    print(
        f"  torch {torch.__version__} | python {platform.python_version()} "
        f"| easy_nn {easy_nn.__version__}",
        file=sys.stderr,
    )
    if public_ip and mapped:
        print(
            f"  connect with: RunPod(host={public_ip!r}, port={mapped})",
            file=sys.stderr,
        )
    sys.stderr.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="easy-nn-server")
    parser.add_argument(
        "--stdio", action="store_true", help="serve one job over stdin/stdout"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("EASY_NN_PORT", 7654)))
    parser.add_argument("--token", default=os.environ.get("EASY_NN_TOKEN"))
    parser.add_argument(
        "--once", action="store_true", help="exit after the first job"
    )
    args = parser.parse_args(argv)

    if args.stdio:
        serve_stdio()
    else:
        serve_tcp(args.host, args.port, args.token, once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

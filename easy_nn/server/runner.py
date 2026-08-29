"""
Runs one job.

The executor knows nothing until a job arrives: the trainer object carries its
own classes, so this module never imports the user's code.  It only wires the
trainer to a blob feed and a link, then gets out of the way.
"""

from __future__ import annotations

import threading
import traceback

from easy_nn import protocol
from easy_nn.server.blobq import WorkQueue
from easy_nn.server.link import Link


class Runner:
    def __init__(self, channel: protocol.Channel, token: str | None = None):
        self.channel = channel
        self.token = token
        self.feed = None
        self.link = None
        self._reader = None
        self._stopped = threading.Event()

    # ------------------------------------------------------------------
    def serve_one_job(self) -> bool:
        """Handshake, take a job, run it.

        Returns True only if a job was actually taken.  A connection that goes
        away without saying anything -- a health check, a port scan, a client
        that changed its mind -- is not an error and must not take the server
        down with it.
        """
        try:
            hello = self.channel.recv()
        except (EOFError, OSError):
            return False

        if hello.type != protocol.HELLO:
            self.channel.send(protocol.ERROR, {"message": "expected HELLO"})
            return False
        if hello.meta.get("proto") != protocol.PROTO_VERSION:
            self.channel.send(
                protocol.ERROR,
                {
                    "message": (
                        f"protocol mismatch: client speaks "
                        f"{hello.meta.get('proto')}, executor speaks "
                        f"{protocol.PROTO_VERSION}"
                    )
                },
            )
            return False
        if self.token is not None and hello.meta.get("token") != self.token:
            self.channel.send(protocol.ERROR, {"message": "bad token"})
            return False
        self.channel.send(
            protocol.WELCOME, {"proto": protocol.PROTO_VERSION, **_environment()}
        )

        # Dependencies first, before the client spends its uplink on weights:
        # a job that cannot be unpickled here should fail in seconds, not after
        # a multi-gigabyte upload.
        try:
            setup = self.channel.recv()
        except (EOFError, OSError):
            return False
        if setup.type != protocol.SETUP:
            self.channel.send(protocol.ERROR, {"message": "expected SETUP"})
            return False
        try:
            self._install(
                setup.meta.get("requirements") or [],
                setup.meta.get("verify_imports") or (),
            )
        except BaseException:
            self.channel.send(protocol.ERROR, {"traceback": traceback.format_exc()})
            return False
        self.channel.send(protocol.READY, {})

        try:
            job = self.channel.recv()
        except (EOFError, OSError):
            return False
        if job.type != protocol.JOB:
            self.channel.send(protocol.ERROR, {"message": "expected JOB"})
            return False

        try:
            self._run_job(job)
        except BaseException:
            self.channel.send(protocol.ERROR, {"traceback": traceback.format_exc()})
        finally:
            self._stopped.set()
        return True

    # ------------------------------------------------------------------
    def _install(self, requirements, verify_imports=()):
        say = lambda text: self.channel.send(protocol.PRINT, {"text": text})

        if requirements:
            from easy_nn.server.deps import ensure

            ensure(requirements, report=say)

        # Installing is not the same as working. A torchvision built against a
        # different torch imports fine as a package and then dies on its first
        # operator; transformers pulls it in on import. Find that out here,
        # while the client has sent nothing but a list of names.
        import importlib

        for name in verify_imports:
            try:
                importlib.import_module(name)
            except BaseException as exc:
                raise RuntimeError(
                    f"the executor cannot import {name!r}: {type(exc).__name__}: {exc}\n"
                    "Its environment is broken or mismatched -- fix the pod "
                    "image or its start command before sending a job."
                ) from exc

        if verify_imports:
            say(f"Verified imports: {', '.join(verify_imports)}\n")

    # ------------------------------------------------------------------
    def _run_job(self, job):
        trainer = job.body

        self.link = Link(self.channel)
        self.feed = WorkQueue(
            on_consume=lambda n: self.channel.send(protocol.CREDIT, {"n": n}),
            eval_memory_limit=job.meta.get("eval_memory_limit", 2 << 30),
        )
        trainer._link = self.link
        trainer._feed = self.feed

        resume = None
        if job.meta.get("has_resume"):
            message = self.channel.recv()
            if message.type != protocol.CHECKPOINT:
                raise protocol.ProtocolError("expected resume state")
            resume = message.body

        self.channel.send(protocol.ACCEPTED, {"job_id": job.meta.get("job_id")})

        self._reader = threading.Thread(
            target=self._read_loop, name="easy-nn-reader", daemon=True
        )
        self._reader.start()


        trainer.init()
        if resume is not None:
            trainer.load_checkpoint(resume)
            trainer.print(f"Resumed at step {trainer.global_step}.")

        trainer.training_loop()
        self.channel.send(protocol.DONE, {"global_step": trainer.global_step})

    # ------------------------------------------------------------------
    def _read_loop(self):
        try:
            while not self._stopped.is_set():
                message = self.channel.recv()
                self._dispatch(message)
        except (EOFError, OSError) as exc:
            if not self._stopped.is_set():
                self.feed.fail(ConnectionError(f"client went away: {exc}"))
        except BaseException as exc:  # noqa: BLE001 - surfaced to the loop
            if not self._stopped.is_set():
                self.feed.fail(exc)

    def _dispatch(self, message):
        if message.type == protocol.BLOB:
            if message.meta.get("kind") == "eval":
                self.feed.put_eval(message.body, _nbytes(message))
            else:
                self.feed.put(message.body, int(message.meta.get("units", 1)))
        elif message.type == protocol.STREAM_END:
            if message.meta.get("kind") == "eval":
                self.feed.eval_done()
            else:
                self.feed.put_end()
        elif message.type == protocol.CTRL:
            self.link.push_control(message.meta["action"])
        elif message.type == protocol.PING:
            self.channel.send(protocol.PONG, {})


def _nbytes(message):
    return sum(len(p) for p in message.parts)


def _environment():
    import platform

    import torch

    import easy_nn

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "easy_nn": easy_nn.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
    }

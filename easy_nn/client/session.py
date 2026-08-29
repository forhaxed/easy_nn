"""
The local brain.

It ships the job, feeds data as fast as the executor has room for, and writes
down everything that comes back.  The executor decides *when* to train; this
side decides *what* exists -- the data, the logs, the checkpoints.
"""

from __future__ import annotations

import itertools
import os
import threading
import uuid

from colorama import Fore, Style
from tqdm.auto import tqdm

from easy_nn import codec, job, protocol
from easy_nn.client.locks import LockWatcher
from easy_nn.client.sinks import CheckpointSink, ConsoleSink, TensorBoardSink


class RemoteError(RuntimeError):
    """The executor raised.  The traceback below happened over there."""


class _Compressor:
    """Decides once whether compressing payloads is worth it, then sticks."""

    def __init__(self, mode: str):
        self.mode = mode if codec.has_zstd() else "off"
        self._decided = mode != "auto" or not codec.has_zstd()
        self._on = self.mode == "zstd"

    def encode(self, obj):
        if self._decided:
            return codec.encode(obj, compress=self._on)

        plain = codec.encode(obj, compress=False)
        packed = codec.encode(obj, compress=True)
        self._on = packed.nbytes < plain.nbytes * 0.9
        self._decided = True
        return packed if self._on else plain


def run_job(trainer, executor):
    """Ship ``trainer`` to ``executor`` and run it to completion."""
    data = trainer.data
    if data is None:
        raise ValueError("trainer.data must be a DataSource")

    data.prepare()
    trainer.dataset_size = len(data)
    if trainer.eval_data is not None:
        trainer.eval_data.prepare()

    os.makedirs(trainer.output_dir, exist_ok=True)
    console = ConsoleSink()
    board = TensorBoardSink(trainer.output_dir)
    checkpoints = CheckpointSink(trainer.output_dir)

    channel = executor.connect()
    state = _State(trainer, channel)
    watcher = None
    feeder = None

    try:
        _handshake(channel, executor, console)
        _send_job(trainer, executor, channel, console)

        message = channel.recv()
        if message.type == protocol.ERROR:
            raise RemoteError(_error_text(message))
        if message.type != protocol.ACCEPTED:
            raise protocol.ProtocolError(f"unexpected reply {message!r}")

        watcher = LockWatcher(trainer.output_dir, state.send_control)
        watcher.start()

        feeder = threading.Thread(
            target=state.feed, name="easy-nn-feeder", daemon=True
        )
        feeder.start()

        return _pump(state, console, board, checkpoints)
    finally:
        state.stop.set()
        state.credits.release(1 << 20)  # unblock a feeder waiting on credit
        if watcher is not None:
            watcher.stop()
        if feeder is not None:
            feeder.join(timeout=5)
        console.close()
        board.close()
        try:
            channel.close()
        finally:
            executor.close()


# ======================================================================
#  Setup
# ======================================================================
def _handshake(channel, executor, console):
    import easy_nn

    channel.send(
        protocol.HELLO,
        {
            "proto": protocol.PROTO_VERSION,
            "token": getattr(executor, "token", None),
            "easy_nn": easy_nn.__version__,
        },
    )
    reply = channel.recv()
    if reply.type == protocol.ERROR:
        raise RemoteError(_error_text(reply))
    if reply.type != protocol.WELCOME:
        raise protocol.ProtocolError(f"unexpected greeting {reply!r}")

    console.text(
        f"{Fore.BLUE}Executor:{Style.RESET_ALL} {executor.describe()} "
        f"| python {reply.meta.get('python')} "
        f"| torch {reply.meta.get('torch')} "
        f"| easy_nn {reply.meta.get('easy_nn', '?')} "
        f"| {reply.meta.get('device')}\n"
    )

    # The executor runs its own copy of easy_nn, updated on its own schedule.
    # A mismatch is not fatal, but it explains a whole class of odd failures.
    theirs = reply.meta.get("easy_nn")
    if theirs is not None and theirs != easy_nn.__version__:
        console.text(
            f"{Fore.YELLOW}Warning: executor runs easy_nn {theirs}, you have "
            f"{easy_nn.__version__}. Restart the pod to pick up your changes."
            f"{Style.RESET_ALL}\n"
        )


def _send_job(trainer, executor, channel, console):
    requirements = list(trainer.requirements) if executor.installs_requirements else []
    resume = _read_resume(trainer.resume_from)
    if resume is not None:
        # Both halves have to agree on where the run stands: the executor
        # restores optimizer state from this, and the feeder here uses it to
        # skip the batches the previous run already consumed.
        _apply_resume_position(trainer, resume)

    # Project code travels by value; installed libraries are the executor's job.
    shipped = job.register_local_modules(getattr(trainer, "ship_modules", ()))
    if shipped:
        console.text(f"Shipping project modules: {', '.join(shipped)}\n")

    # Weights are float noise and do not compress; only an explicit "zstd"
    # asks for it here.  "auto" probes on the first data blob instead, where
    # the answer actually varies with what pack() produces.
    body = codec.encode(trainer, compress=trainer.compression == "zstd")
    bar = tqdm(
        total=body.nbytes,
        unit="B",
        unit_scale=True,
        desc="Uploading job",
        leave=False,
    )
    header = len(body.header)
    bar.update(header)
    channel.send(
        protocol.JOB,
        {
            "job_id": uuid.uuid4().hex,
            "requirements": requirements,
            "has_resume": resume is not None,
            "eval_memory_limit": getattr(trainer, "eval_memory_limit", 2 << 30),
        },
        body,
        on_progress=lambda sent: bar.update(header + sent - bar.n),
    )
    bar.close()

    if resume is not None:
        channel.send(protocol.CHECKPOINT, {"name": "resume"}, codec.encode(resume))
        console.text(f"Resuming from {trainer.resume_from}\n")


def _apply_resume_position(trainer, payload):
    raw = payload.get("trainer_metadata.json")
    if not raw:
        return
    import json

    meta = json.loads(bytes(raw))
    trainer.global_step = meta.get("global_step", 0)
    trainer.epochs_trained = meta.get("epochs_trained", 0)
    trainer.steps_in_epoch = meta.get("steps_in_epoch", 0)


def _read_resume(directory):
    if not directory:
        return None
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"resume_from directory not found: {directory}")
    payload = {}
    for root, _, files in os.walk(directory):
        for filename in files:
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, directory).replace(os.sep, "/")
            with open(path, "rb") as handle:
                payload[rel] = handle.read()
    return payload


# ======================================================================
#  Running
# ======================================================================
class _State:
    def __init__(self, trainer, channel):
        self.trainer = trainer
        self.channel = channel
        self.credits = threading.Semaphore(0)
        self.stop = threading.Event()
        self.compressor = _Compressor(trainer.compression)
        self.error = None

    def send_control(self, action):
        if not self.stop.is_set():
            try:
                self.channel.send(protocol.CTRL, {"action": action})
            except (OSError, ValueError):
                pass

    # -- the feeder thread ------------------------------------------------
    def feed(self):
        try:
            self._feed_eval()
            self._feed_train()
        except (OSError, ValueError, EOFError) as exc:
            # The executor stopped listening; the main loop reports why.
            self.error = exc
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
            self.send_control("stop")

    def _feed_eval(self):
        source = self.trainer.eval_data
        if source is not None:
            for blob in _grouped(source, self.trainer.precache_size):
                if self.stop.is_set():
                    break
                self.channel.send(
                    protocol.BLOB, {"kind": "eval"}, self.compressor.encode(blob)
                )
        self.channel.send(protocol.STREAM_END, {"kind": "eval"})

    def _feed_train(self):
        trainer = self.trainer
        skip = 0
        if trainer.steps_in_epoch > 0:
            skip = (
                trainer.steps_in_epoch * trainer.gradient_accumulation_steps
            ) // max(1, trainer.repeats)

        for epoch in range(trainer.epochs_trained, trainer.epochs):
            for blob in _grouped(trainer.data, trainer.precache_size, skip=skip):
                if not self._await_credit():
                    return
                self.channel.send(
                    protocol.BLOB,
                    {"kind": "train", "epoch": epoch},
                    self.compressor.encode(blob),
                )
            skip = 0
            self.channel.send(protocol.EPOCH_END, {"epoch": epoch})
            if self.stop.is_set():
                return
        self.channel.send(protocol.STREAM_END, {"kind": "train"})

    def _await_credit(self) -> bool:
        while not self.stop.is_set():
            if self.credits.acquire(timeout=0.25):
                return not self.stop.is_set()
        return False


def _grouped(source, size, skip=0):
    """Batches from a DataSource, packed ``size`` at a time."""
    stream = source.stream()
    if skip:
        stream = itertools.islice(stream, skip, None)
    buffer = []
    for batch in stream:
        buffer.append(batch)
        if len(buffer) >= size:
            yield source.pack(buffer)
            buffer = []
    if buffer:
        yield source.pack(buffer)


def _pump(state, console, board, checkpoints):
    """Read events from the executor until the run ends."""
    saved = []
    while True:
        try:
            message = state.channel.recv()
        except EOFError:
            if state.error is not None:
                raise RemoteError(
                    f"executor closed the connection while sending data: {state.error}"
                ) from state.error
            raise RemoteError("executor closed the connection unexpectedly")

        kind = message.type
        if kind == protocol.CREDIT:
            state.credits.release(int(message.meta.get("n", 1)))
        elif kind == protocol.LOG:
            board.log(message.meta["values"], message.meta["step"])
        elif kind == protocol.PRINT:
            console.text(message.meta["text"])
        elif kind == protocol.PROGRESS:
            console.progress(**message.meta)
        elif kind == protocol.CHECKPOINT:
            path = checkpoints.save(message.meta["name"], message.body)
            saved.append(path)
            console.text(f"Saved checkpoint to {path}\n")
        elif kind == protocol.DONE:
            return {
                "global_step": message.meta.get("global_step"),
                "checkpoints": saved,
                "log_dir": board.log_dir,
            }
        elif kind == protocol.ERROR:
            raise RemoteError(_error_text(message))


def _error_text(message):
    return message.meta.get("traceback") or message.meta.get("message", "unknown error")

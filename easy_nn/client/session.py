"""
The local brain.

It ships the job, keeps the executor's queue topped up, and writes down
everything that comes back.  The executor decides *when* to train; this side
decides *what* exists -- the data, the logs, the checkpoints.
"""

from __future__ import annotations

import collections
import importlib.util
import itertools
import json
import os
import re
import threading
import time
import uuid

from colorama import Fore, Style
from tqdm.auto import tqdm

from easy_nn import codec, job, protocol
from easy_nn.client.locks import LockWatcher
from easy_nn.client.reporter import Reporter
from easy_nn.client.sinks import CheckpointSink, ConsoleSink, TensorBoardSink


class RemoteError(RuntimeError):
    """The executor raised.  The traceback below happened over there."""


class ExecutorRestarted(RuntimeError):
    """The executor built the environment the job needs and restarted into it.

    Not a failure: nothing was uploaded, and the environment is cached. Run the
    script again.
    """


def local_env_spec(trainer=None) -> dict:
    """Describe the environment the executor has to reproduce.

    Python has to match to the minor version because the trainer's methods
    travel as bytecode; torch has to match because cloudpickle records torch
    submodules that only exist in some releases; torchvision has to match torch
    or it dies registering its operators, and transformers imports it for you.
    """
    import platform

    import torch

    spec = {
        "python": ".".join(platform.python_version().split(".")[:2]),
        "torch": torch.__version__.split("+")[0],
        "requirements": list(getattr(trainer, "requirements", []) or []),
    }

    build = torch.__version__.split("+")[1] if "+" in torch.__version__ else ""
    if build:
        spec["torch_index"] = f"https://download.pytorch.org/whl/{build}"

    try:
        import torchvision

        spec["torchvision"] = torchvision.__version__.split("+")[0]
    except Exception:
        pass

    return spec


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
    reporter = Reporter(board, console)

    channel = executor.connect()
    state = _State(trainer, channel)
    watcher = None
    feeder = None

    try:
        _handshake(channel, executor, console, trainer)
        _install_requirements(trainer, executor, channel, console)
        _send_job(trainer, channel, console)

        _await_accepted(channel, console)

        watcher = LockWatcher(trainer.output_dir, state.send_control)
        watcher.start()

        feeder = threading.Thread(target=state.feed, name="easy-nn-feeder", daemon=True)
        feeder.start()

        return _pump(state, console, board, checkpoints, reporter)
    finally:
        state.stop.set()
        with state.queue_changed:
            state.queue_changed.notify_all()
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
def _handshake(channel, executor, console, trainer=None):
    import easy_nn

    channel.send(
        protocol.HELLO,
        {
            "proto": protocol.PROTO_VERSION,
            "token": getattr(executor, "token", None),
            "easy_nn": easy_nn.__version__,
            # The executor builds this if it does not already have it.
            "env": local_env_spec(trainer),
        },
    )
    # Building an environment can take minutes and reports as it goes.
    reply = _await(channel, console, protocol.WELCOME)

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

    _check_compatibility(
        reply.meta.get("python"),
        reply.meta.get("torch"),
        allow_mismatch=getattr(trainer, "allow_version_mismatch", False),
    )


def _series(version):
    return tuple(version.split("+")[0].split(".")[:2])


def _check_compatibility(their_python, their_torch, allow_mismatch=False):
    """Refuse to upload into an environment that cannot run what we send.

    Both checks guard the same failure shape: the job is only unpickled once
    the whole model has arrived, so an incompatibility discovered there costs
    the entire upload. A second spent here saves it.
    """
    import platform

    import torch

    if allow_mismatch:
        return

    # Python first, and it is the strict one. cloudpickle ships your methods as
    # code objects -- raw bytecode -- and bytecode is specific to the
    # interpreter that produced it. Run 3.11 bytecode on 3.12 and the opcodes
    # no longer mean what they meant: you get nonsense errors from innocent
    # lines, like an assert reporting "too many values to unpack".
    ours = platform.python_version()
    if their_python and _series(their_python) != _series(ours):
        raise RemoteError(
            f"Python mismatch: this machine runs {ours}, the executor runs "
            f"{their_python}.\n"
            "Your trainer's methods travel as bytecode, which only the same "
            "Python minor version can execute. A mismatch does not fail "
            "cleanly -- it misexecutes, and the errors make no sense.\n"
            f"Use a pod image built on Python {'.'.join(_series(ours))}.\n"
            "Set trainer.allow_version_mismatch = True to try anyway."
        )

    ours = torch.__version__
    if their_torch and _series(their_torch) != _series(ours):
        raise RemoteError(
            f"torch mismatch: this machine has {ours}, the executor has "
            f"{their_torch}.\n"
            "The job is a pickled object graph, and cloudpickle records the "
            "torch submodules it thinks your code needs -- some of which only "
            "exist in newer releases.\n"
            f"Install torch=={ours.split('+')[0]} (with the matching "
            "torchvision) in the pod's start command, before easy-nn-server "
            "starts: installing it later is too late, the server has already "
            "imported torch.\n"
            "Set trainer.allow_version_mismatch = True to try anyway."
        )


def _await(channel, console, expected):
    """Wait for one message type, printing whatever the executor says meanwhile.

    Installing packages on a cold pod takes minutes and reports as it goes, so
    output legitimately arrives ahead of the reply we are waiting for.
    """
    while True:
        message = channel.recv()
        if message.type == expected:
            return message
        if message.type == protocol.PRINT:
            console.text(message.meta["text"])
            continue
        if message.type == protocol.RESTART:
            raise ExecutorRestarted(message.meta.get("message", "executor restarted"))
        if message.type == protocol.ERROR:
            raise RemoteError(_error_text(message))
        raise protocol.ProtocolError(f"unexpected reply {message!r}")


def _await_accepted(channel, console):
    return _await(channel, console, protocol.ACCEPTED)


def _install_requirements(trainer, executor, channel, console):
    """Get the executor's environment right before sending anything large.

    The trainer arrives there as a pickled object built by these libraries, so
    a version mismatch is fatal -- and finding that out after a multi-gigabyte
    upload wastes the upload. Ask first, send later.
    """
    requirements = list(trainer.requirements) if executor.installs_requirements else []
    channel.send(
        protocol.SETUP,
        {
            "requirements": requirements,
            "verify_imports": _imports_to_verify(trainer, requirements),
        },
    )
    _await(channel, console, protocol.READY)


def _imports_to_verify(trainer, requirements) -> list[str]:
    """Modules the executor must actually import before we send it anything.

    Version numbers are not enough: a torchvision built against a different
    torch installs cleanly, imports as a package, and then dies registering an
    operator -- and transformers imports it for you. Naming the modules and
    letting the executor try is the only check that catches that.
    """
    explicit = getattr(trainer, "verify_imports", None)
    if explicit is not None:
        return list(explicit)

    names = ["torch"]
    for requirement in requirements:
        name = re.split(r"[<>=!~\[;\s]", requirement.strip())[0]
        if name:
            names.append(name.replace("-", "_"))

    # Only if this side has it: the executor's copy has to survive the same
    # torch it was paired with here.
    if importlib.util.find_spec("torchvision") is not None:
        names.append("torchvision")

    return list(dict.fromkeys(names))


def _send_job(trainer, channel, console):
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
    # asks for it here.  "auto" probes on the first work batch instead, where
    # the answer actually varies with what pack() produces.
    body = codec.encode(trainer, compress=trainer.compression == "zstd")
    bar = tqdm(
        total=body.nbytes, unit="B", unit_scale=True, desc="Uploading job", leave=False
    )
    header = len(body.header)
    bar.update(header)
    channel.send(
        protocol.JOB,
        {
            "job_id": uuid.uuid4().hex,
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
        self.stop = threading.Event()
        self.compressor = _Compressor(trainer.compression)
        self.error = None

        #: Units of work believed to be sitting on the executor.
        self.outstanding = 0
        self.queue_changed = threading.Condition()
        self.pack_seconds = 0.0

    def send_control(self, action):
        if not self.stop.is_set():
            try:
                self.channel.send(protocol.CTRL, {"action": action})
            except (OSError, ValueError):
                pass

    def credit(self, n: int):
        with self.queue_changed:
            self.outstanding = max(0, self.outstanding - n)
            self.queue_changed.notify_all()

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
            for group in _grouped(source, self.trainer.blob_size_prepare):
                if self.stop.is_set():
                    break
                self.channel.send(
                    protocol.BLOB,
                    {"kind": "eval", "units": len(group)},
                    self.compressor.encode(source.pack(group)),
                )
        self.channel.send(protocol.STREAM_END, {"kind": "eval"})

    def _feed_train(self):
        """Top the executor's queue back up whenever it drops below blob_size.

        One unbroken stream: epoch boundaries never flush a partial batch, so a
        one-sample dataset simply fills the queue with itself instead of
        dribbling out a single unit per epoch.
        """
        trainer = self.trainer
        prepare = max(1, trainer.blob_size_prepare)

        for group in _grouped(_units(trainer), prepare):
            if not self._await_room():
                return
            payload = self.compressor.encode(trainer.data.pack(group))
            self.channel.send(
                protocol.BLOB, {"kind": "train", "units": len(group)}, payload
            )
            with self.queue_changed:
                self.outstanding += len(group)

        self.channel.send(protocol.STREAM_END, {"kind": "train"})

    def _await_room(self) -> bool:
        with self.queue_changed:
            while not self.stop.is_set() and self.outstanding >= self.trainer.blob_size:
                self.queue_changed.wait(timeout=0.25)
            return not self.stop.is_set()


def _units(trainer):
    """Every training sample the run needs, as one continuous stream."""
    skip = 0
    if trainer.steps_in_epoch > 0 and trainer.allow_skip_batches_on_resume:
        skip = (
            trainer.steps_in_epoch * trainer.gradient_accumulation_steps
        ) // max(1, trainer.repeats)

    for _ in range(trainer.epochs_trained, trainer.epochs):
        stream = trainer.data.stream()
        if skip:
            stream = itertools.islice(stream, skip, None)
            skip = 0
        yield from stream


def _grouped(source_or_iter, size):
    """Chunk an iterable into lists of at most ``size``."""
    iterator = (
        source_or_iter.stream()
        if hasattr(source_or_iter, "stream")
        else source_or_iter
    )
    group = []
    for item in iterator:
        group.append(item)
        if len(group) >= size:
            yield group
            group = []
    if group:
        yield group


def _pump(state, console, board, checkpoints, reporter):
    """Read events from the executor until the run ends."""
    saved = []
    transport = getattr(state.channel, "transport", None)
    net = _NetworkMeter(transport)

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
            state.credit(int(message.meta.get("n", 1)))
        elif kind == protocol.LOG:
            step = message.meta["step"]
            board.log(message.meta["values"], step)
            extra = net.sample()
            extra["queue/outstanding"] = float(state.outstanding)
            board.log(extra, step)
            console.set_extra(
                up=f"{extra['net/up_MBps']:.1f}MB/s",
                q=int(state.outstanding),
            )
        elif kind == protocol.PRINT:
            console.text(message.meta["text"])
        elif kind == protocol.PROGRESS:
            console.progress(**message.meta)
        elif kind == protocol.ARTIFACT:
            state.trainer.on_artifact(
                message.meta["name"], message.body, message.meta["step"], reporter
            )
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


class _NetworkMeter:
    """Turns the transport's byte counters into rates.

    Measured over a window rather than between samples: work is sent in
    batches, so one step in eight carries a whole upload and the rest carry
    nothing. Instantaneous rates would read zero almost always and spike
    absurdly once -- useless for telling whether the link is the bottleneck.
    """

    def __init__(self, transport, window=20.0):
        self.transport = transport
        self.window = window
        self._history = collections.deque()
        self._record()

    def _read(self, attribute):
        return float(getattr(self.transport, attribute, 0) or 0)

    def _record(self):
        now = time.perf_counter()
        self._history.append(
            (now, self._read("bytes_sent"), self._read("bytes_received"))
        )
        while len(self._history) > 2 and now - self._history[0][0] > self.window:
            self._history.popleft()
        return self._history[-1]

    def sample(self) -> dict:
        now, sent, received = self._record()
        then, was_sent, was_received = self._history[0]
        span = max(now - then, 1e-6)
        return {
            "net/up_MBps": (sent - was_sent) / span / 1e6,
            "net/down_MBps": (received - was_received) / span / 1e6,
            "net/up_total_GB": sent / (1 << 30),
            "net/down_total_GB": received / (1 << 30),
        }


def _error_text(message):
    return message.meta.get("traceback") or message.meta.get("message", "unknown error")

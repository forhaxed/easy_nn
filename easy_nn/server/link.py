"""
The executor's line home.

Everything the training loop would normally write to disk or print to a
terminal goes through here instead and comes out on the local machine.
"""

from __future__ import annotations

import collections
import threading

import torch

from easy_nn import codec, protocol


class Link:
    def __init__(self, channel: protocol.Channel):
        self.channel = channel
        self._controls = collections.deque()
        self._lock = threading.Lock()

    # -- called by the training loop --------------------------------------
    def log(self, values: dict, step: int):
        self.channel.send(
            protocol.LOG,
            {"step": int(step), "values": {k: _scalar(v) for k, v in values.items()}},
        )

    def print(self, text: str):
        self.channel.send(protocol.PRINT, {"text": text})

    def progress(self, **kw):
        self.channel.send(protocol.PROGRESS, kw)

    def checkpoint(self, name: str, payload: dict):
        self.channel.send(protocol.CHECKPOINT, {"name": name}, codec.encode(payload))

    def artifact(self, name: str, payload, step: int, compress: bool = False):
        """Send anything tensor-bearing home for local post-processing.

        Used for work the executor should not finish itself -- validation
        latents, for instance, which the local side decodes with its own VAE.
        """
        self.channel.send(
            protocol.ARTIFACT,
            {"name": name, "step": int(step)},
            codec.encode(payload, compress=compress),
        )

    def take_control(self):
        with self._lock:
            drained = list(self._controls)
            self._controls.clear()
        return drained

    # -- called by the reader thread --------------------------------------
    def push_control(self, command: str):
        with self._lock:
            self._controls.append(command)


def gpu_stats() -> dict:
    """VRAM figures for the log. Empty when there is no CUDA device."""
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    giga = float(1 << 30)
    return {
        "gpu/allocated_GB": torch.cuda.memory_allocated() / giga,
        "gpu/reserved_GB": torch.cuda.memory_reserved() / giga,
        "gpu/peak_GB": torch.cuda.max_memory_allocated() / giga,
        "gpu/free_GB": free / giga,
        "gpu/total_GB": total / giga,
    }


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().float().mean())
    return float(value)

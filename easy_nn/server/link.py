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
        self.channel.send(
            protocol.CHECKPOINT, {"name": name}, codec.encode(payload)
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


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().float().mean())
    return float(value)

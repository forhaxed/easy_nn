"""
Where everything the executor produces actually lands: your machine.

The executor computes; nothing it makes touches its own disk.  TensorBoard
events, checkpoints and terminal output are all written here.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import torch
from tqdm.auto import tqdm


class ConsoleSink:
    """Mirrors the executor's stdout and draws its progress bar locally."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.bar = None

    def text(self, text: str):
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.write(text.rstrip("\n"))
        else:
            sys.stdout.write(text)
            sys.stdout.flush()

    def progress(self, step=None, total=None, postfix=None, start=False):
        if not self.enabled:
            return
        if start or self.bar is None:
            if self.bar is not None:
                self.bar.close()
            self.bar = tqdm(total=total, initial=step or 0, desc="Steps")
            return
        if step is not None:
            self.bar.update(max(0, step - self.bar.n))
        if postfix:
            self.bar.set_postfix(**postfix)

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class TensorBoardSink:
    """The only TensorBoard writer in the system, and it is local."""

    def __init__(self, output_dir: str, run_name: str | None = None):
        stamp = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(output_dir, "logs", stamp)
        self._writer = None

    @property
    def writer(self):
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.log_dir, exist_ok=True)
            self._writer = SummaryWriter(self.log_dir)
        return self._writer

    def log(self, values: dict, step: int):
        for key, value in values.items():
            self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class CheckpointSink:
    """Writes whatever ``save_checkpoint`` returned into output_dir."""

    def __init__(self, output_dir: str):
        self.root = os.path.join(output_dir, "checkpoints")

    def save(self, name: str, payload: dict) -> str:
        directory = os.path.join(self.root, name)
        os.makedirs(directory, exist_ok=True)
        for rel, value in payload.items():
            path = os.path.join(directory, *str(rel).split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if isinstance(value, (bytes, bytearray, memoryview)):
                with open(path, "wb") as handle:
                    handle.write(value)
            else:
                # Tensors and tensor-bearing objects, e.g. an adapter state dict.
                torch.save(value, path)
        return directory

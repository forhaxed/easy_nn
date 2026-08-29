"""
What local code gets handed when the executor sends something home.

``Trainer.on_artifact`` receives one of these: a narrow view of the local sinks
so it can write a decoded image or an extra scalar without knowing anything
about how the session is wired.
"""

from __future__ import annotations


class Reporter:
    def __init__(self, board, console):
        self._board = board
        self._console = console

    def log(self, values: dict, step: int):
        self._board.log(values, step)

    def log_image(self, tag: str, image, step: int):
        """``image`` is HWC uint8, or CHW -- whatever TensorBoard accepts."""
        self._board.log_image(tag, image, step)

    def print(self, text: str):
        self._console.text(text if text.endswith("\n") else text + "\n")

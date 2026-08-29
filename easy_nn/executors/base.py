"""Executor interface: hand back a Channel, tear it down afterwards."""

from __future__ import annotations


class Executor:
    #: Whether the executor should pip-install ``trainer.requirements``.
    installs_requirements = True

    def connect(self):
        """Return a ``protocol.Channel`` talking to a running executor."""
        raise NotImplementedError

    def close(self):
        pass

    def describe(self) -> str:
        return type(self).__name__

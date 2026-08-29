"""
The local executor.

It is a subprocess, not an in-process call, and it speaks the same protocol
through the same codec as a pod does.  That is the point: if a job runs here it
runs there, and anything that fails to serialize fails on your own machine
rather than an hour into a rented GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from easy_nn import protocol
from easy_nn.executors.base import Executor
from easy_nn.transport.pipe import PipeTransport


class Local(Executor):
    #: The local environment already has everything; installing would be noise.
    installs_requirements = False

    def __init__(self, python: str | None = None, workdir: str | None = None, env=None):
        self.python = python or sys.executable
        self.workdir = workdir
        self.env = env
        self.process = None
        self._tempdir = None

    def connect(self):
        if self.workdir is None:
            self._tempdir = tempfile.mkdtemp(prefix="easy_nn_exec_")
            workdir = self._tempdir
        else:
            workdir = self.workdir

        environment = dict(os.environ)
        # Let the child import easy_nn even when it is not pip-installed.
        import easy_nn

        package_root = os.path.dirname(os.path.dirname(os.path.abspath(easy_nn.__file__)))
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_root if not existing else package_root + os.pathsep + existing
        )
        if self.env:
            environment.update(self.env)

        self.process = subprocess.Popen(
            [self.python, "-u", "-m", "easy_nn.server.app", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # tracebacks and stray prints land on your terminal
            cwd=workdir,
            env=environment,
        )
        return protocol.Channel(PipeTransport.from_popen(self.process))

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self._tempdir is not None:
            import shutil

            shutil.rmtree(self._tempdir, ignore_errors=True)
            self._tempdir = None

    def describe(self) -> str:
        return f"Local(python={os.path.basename(self.python)})"

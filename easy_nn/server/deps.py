"""
On-the-fly dependency installation.

A pod starts out knowing nothing about the job it will run, so whatever
``trainer.requirements`` lists gets installed before the trainer is unpickled.
Installs are keyed by the hash of the requirement list, so re-running the same
job on a warm pod costs nothing.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

STAMP_DIR = os.environ.get(
    "EASY_NN_DEPS_CACHE", os.path.join(os.path.expanduser("~"), ".easy_nn", "deps")
)


def _key(requirements, interpreter=None) -> str:
    """Identify an install by its packages *and* the environment they go into.

    The same pod can run the server from the image's Python one day and from a
    venv the next; $HOME does not change, so a stamp keyed on the package list
    alone would claim the venv already has them.
    """
    fingerprint = "\n".join([interpreter or sys.executable, *sorted(requirements)])
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def mark_installed(requirements, interpreter) -> None:
    """Record that ``interpreter`` already has these, so it skips reinstalling.

    The bootstrap installs a job's packages while building the venv; the server
    that later runs inside it should not repeat the work.
    """
    if not requirements:
        return
    os.makedirs(STAMP_DIR, exist_ok=True)
    with open(os.path.join(STAMP_DIR, _key(requirements, interpreter)), "w") as handle:
        handle.write("\n".join(sorted(requirements)))


def ensure(requirements, report=None, force=False) -> bool:
    """Install ``requirements`` unless this exact set was installed before.

    Returns True if pip actually ran.
    """
    if not requirements:
        return False

    say = report or (lambda text: None)
    stamp = os.path.join(STAMP_DIR, _key(requirements))
    if os.path.exists(stamp) and not force:
        say(f"Dependencies already installed ({len(requirements)} packages).\n")
        return False

    say(f"Installing {len(requirements)} packages: {', '.join(requirements)}\n")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-input", *requirements],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pip install failed ({result.returncode}):\n{result.stdout}"
        )

    say(result.stdout.strip().splitlines()[-1] + "\n" if result.stdout.strip() else "")
    os.makedirs(STAMP_DIR, exist_ok=True)
    with open(stamp, "w") as handle:
        handle.write("\n".join(sorted(requirements)))
    return True

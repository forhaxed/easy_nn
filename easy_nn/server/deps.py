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


def _key(requirements) -> str:
    joined = "\n".join(sorted(requirements))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def ensure(requirements, report=None) -> bool:
    """Install ``requirements`` unless this exact set was installed before.

    Returns True if pip actually ran.
    """
    if not requirements:
        return False

    say = report or (lambda text: None)
    stamp = os.path.join(STAMP_DIR, _key(requirements))
    if os.path.exists(stamp):
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

"""
Building the environment a job asks for.

The pod's image is whatever was free when you rented it, and it will not match
your machine. That matters more than it sounds: your trainer's methods travel
as bytecode, so the executor needs the same Python *minor* version or it
misexecutes them; and cloudpickle records torch submodules that only exist in
some releases, so torch has to match too.

Rather than making you hunt for an image, the server builds what the job asks
for: a venv on the right Python, with the right torch, cached by the exact
specification so the second run costs nothing. ``uv`` does the work -- it
downloads a standalone CPython when the image has no suitable one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

ENV_ROOT = os.environ.get(
    "EASY_NN_ENV_ROOT", os.path.join(os.path.expanduser("~"), ".easy_nn", "envs")
)

#: What easy_nn itself needs to run a job. tensorboard is deliberately absent:
#: logs are written on the client, never here.
SERVER_DEPS = ["cloudpickle", "tqdm", "colorama", "accelerate", "zstandard"]

_READY = ".easy_nn_ready"


def spec_key(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def python_series(version: str) -> str:
    return ".".join(version.split("+")[0].split(".")[:2])


def satisfied_by_current(spec: dict) -> bool:
    """True when the interpreter already running is good enough."""
    import platform

    if python_series(platform.python_version()) != spec.get("python"):
        return False

    wanted = spec.get("torch")
    if wanted:
        try:
            import torch
        except ImportError:
            return False
        if torch.__version__.split("+")[0] != wanted.split("+")[0]:
            return False

    wanted_vision = spec.get("torchvision")
    if wanted_vision:
        try:
            import torchvision
        except Exception:
            return False
        if torchvision.__version__.split("+")[0] != wanted_vision.split("+")[0]:
            return False

    return True


def ensure_interpreter(spec: dict, say=None) -> str:
    """Return a python executable that satisfies ``spec``, building one if needed."""
    say = say or (lambda text: None)

    if satisfied_by_current(spec):
        say(f"Executor environment already matches (python {spec.get('python')}, "
            f"torch {spec.get('torch')}).\n")
        return sys.executable

    directory = os.path.join(ENV_ROOT, spec_key(spec))
    python = _venv_python(directory)

    if os.path.exists(os.path.join(directory, _READY)) and os.path.exists(python):
        say(f"Reusing cached environment {directory}.\n")
        _install_easy_nn(python, say)  # keep it in step with this server
        return python

    say(
        f"Building an environment for python {spec.get('python')} / "
        f"torch {spec.get('torch')} -- this happens once per specification.\n"
    )
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(os.path.dirname(directory) or ".", exist_ok=True)

    _ensure_uv(say)
    _run(
        [sys.executable, "-m", "uv", "venv", "--python", spec["python"], directory],
        say,
        "creating the venv (uv downloads a standalone CPython if the image lacks one)",
    )

    torch_packages = [f"torch=={spec['torch']}"] if spec.get("torch") else []
    if spec.get("torchvision"):
        torch_packages.append(f"torchvision=={spec['torchvision']}")
    if torch_packages:
        # The pytorch index carries only torch's own wheels, so it gets its own
        # install step -- pointing everything at it would fail to find the rest.
        command = [sys.executable, "-m", "uv", "pip", "install", "--python", python]
        command += torch_packages
        if spec.get("torch_index"):
            command += ["--index-url", spec["torch_index"]]
        _run(command, say, "installing torch")

    extras = list(SERVER_DEPS) + list(spec.get("requirements") or [])
    _run(
        [sys.executable, "-m", "uv", "pip", "install", "--python", python, *extras],
        say,
        "installing the job's packages",
    )

    _install_easy_nn(python, say)

    # The server that runs inside this venv must not reinstall what we just put
    # there, so record the work against that interpreter.
    from easy_nn.server import deps

    deps.mark_installed(list(spec.get("requirements") or []), python)

    # uv makes this directory, but the marker is ours and must not depend on
    # exactly how it did so.
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, _READY), "w") as handle:
        json.dump(spec, handle, indent=2, default=str)

    say("Environment ready.\n")
    return python


# ----------------------------------------------------------------------
def _venv_python(directory: str) -> str:
    if os.name == "nt":
        return os.path.join(directory, "Scripts", "python.exe")
    return os.path.join(directory, "bin", "python")


def _ensure_uv(say):
    try:
        subprocess.run(
            [sys.executable, "-m", "uv", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return
    except (subprocess.CalledProcessError, OSError):
        pass
    _run([sys.executable, "-m", "pip", "install", "--no-input", "uv"], say, "installing uv")


def _install_easy_nn(python: str, say):
    """Copy this server's own easy_nn into the venv.

    Copied rather than installed from an index: the venv must run exactly the
    code this process is running, whatever branch or checkout that came from.
    """
    import easy_nn

    source = os.path.dirname(os.path.abspath(easy_nn.__file__))
    target_root = _site_packages(python)
    target = os.path.join(target_root, "easy_nn")
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )


def _site_packages(python: str) -> str:
    result = subprocess.run(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _run(command, say, what):
    say(f"  {what}...\n")
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed while {what} (exit {result.returncode}):\n{result.stdout}"
        )
    tail = [line for line in (result.stdout or "").splitlines() if line.strip()][-1:]
    if tail:
        say(f"  {tail[0]}\n")

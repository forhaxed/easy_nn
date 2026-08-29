"""
Building the environment a job asks for.

The expensive part -- uv creating a venv and downloading torch -- is not
exercised here; the subprocess calls are recorded instead. What is checked is
the decision logic, because getting that wrong is what wasted whole uploads:
saying "close enough" to a Python that cannot run the bytecode we send.
"""

import platform
import sys

import pytest
import torch

from easy_nn.client.session import local_env_spec
from easy_nn.server import bootstrap, deps


def spec(**overrides):
    base = local_env_spec(None)
    base.update(overrides)
    return base


def test_the_local_environment_describes_itself():
    described = local_env_spec(None)
    assert described["python"] == ".".join(platform.python_version().split(".")[:2])
    assert described["torch"] == torch.__version__.split("+")[0]
    if "+" in torch.__version__:
        assert torch.__version__.split("+")[1] in described["torch_index"]


def test_requirements_travel_with_the_spec():
    class T:
        requirements = ["diffusers==0.37.1"]

    assert local_env_spec(T())["requirements"] == ["diffusers==0.37.1"]


def test_this_interpreter_satisfies_its_own_spec():
    assert bootstrap.satisfied_by_current(spec())


def test_a_different_python_minor_is_not_good_enough():
    """The failure this guards against is silent: mismatched bytecode
    misexecutes rather than refusing to load."""
    major, minor = platform.python_version().split(".")[:2]
    assert not bootstrap.satisfied_by_current(
        spec(python=f"{major}.{int(minor) + 1}")
    )


def test_a_different_torch_is_not_good_enough():
    assert not bootstrap.satisfied_by_current(spec(torch="1.13.0"))


def test_a_different_torchvision_is_not_good_enough():
    assert not bootstrap.satisfied_by_current(spec(torchvision="0.1.0"))


def test_the_cache_key_follows_the_specification():
    assert bootstrap.spec_key(spec()) == bootstrap.spec_key(spec())
    assert bootstrap.spec_key(spec()) != bootstrap.spec_key(spec(python="3.99"))
    assert bootstrap.spec_key(spec()) != bootstrap.spec_key(
        spec(requirements=["diffusers"])
    )


def test_a_matching_environment_is_used_as_is(capsys):
    said = []
    assert bootstrap.ensure_interpreter(spec(), say=said.append) == sys.executable
    assert any("already matches" in line for line in said)


def test_the_venv_build_asks_uv_for_the_right_things(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ENV_ROOT", str(tmp_path / "envs"))
    monkeypatch.setattr(bootstrap, "_ensure_uv", lambda say: None)
    monkeypatch.setattr(bootstrap, "_install_easy_nn", lambda python, say: None)
    monkeypatch.setattr(deps, "mark_installed", lambda reqs, python: None)

    calls = []

    def fake_run(command, say, what):
        calls.append(command)

    monkeypatch.setattr(bootstrap, "_run", fake_run)

    wanted = spec(python="3.11", torch="2.10.0", torchvision="0.25.0",
                  requirements=["diffusers==0.37.1"])
    wanted["python"] = "3.99"  # force a mismatch so it builds
    python = bootstrap.ensure_interpreter(wanted, say=lambda t: None)

    assert str(tmp_path / "envs") in python

    venv_call = calls[0]
    assert "venv" in venv_call and "3.99" in venv_call

    torch_call = calls[1]
    assert "torch==2.10.0" in torch_call
    assert "torchvision==0.25.0" in torch_call
    # torch's own index carries only torch wheels, so it must not be used for
    # the rest of the packages.
    assert "--index-url" in torch_call

    packages_call = calls[2]
    assert "diffusers==0.37.1" in packages_call
    assert "cloudpickle" in packages_call
    assert "--index-url" not in packages_call
    assert "tensorboard" not in packages_call, "logs are written on the client"


def test_a_built_environment_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ENV_ROOT", str(tmp_path / "envs"))
    monkeypatch.setattr(bootstrap, "_ensure_uv", lambda say: None)
    monkeypatch.setattr(bootstrap, "_install_easy_nn", lambda python, say: None)
    monkeypatch.setattr(deps, "mark_installed", lambda reqs, python: None)

    builds = []
    monkeypatch.setattr(bootstrap, "_run", lambda c, s, w: builds.append(w))

    wanted = spec(python="3.99")
    python = bootstrap.ensure_interpreter(wanted, say=lambda t: None)

    # Pretend uv actually produced the interpreter.
    import os

    os.makedirs(os.path.dirname(python), exist_ok=True)
    open(python, "w").close()

    builds.clear()
    said = []
    again = bootstrap.ensure_interpreter(wanted, say=said.append)

    assert again == python
    assert builds == [], "a cached environment must not be rebuilt"
    assert any("Reusing cached" in line for line in said)


def test_installs_are_stamped_against_their_interpreter(tmp_path, monkeypatch):
    """A stamp from the image's Python must not tell a venv it is done."""
    monkeypatch.setattr(deps, "STAMP_DIR", str(tmp_path / "stamps"))
    requirements = ["diffusers==0.37.1"]

    deps.mark_installed(requirements, "/opt/easynn/bin/python")

    monkeypatch.setattr(sys, "executable", "/opt/easynn/bin/python")
    assert deps.ensure(requirements) is False, "stamped for this interpreter"

    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    ran = []
    monkeypatch.setattr(
        deps.subprocess, "run",
        lambda cmd, **kw: ran.append(cmd) or type("R", (), {"returncode": 0, "stdout": "ok"})(),
    )
    assert deps.ensure(requirements) is True, "a different interpreter must install"
    assert ran

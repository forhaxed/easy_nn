"""Dependency installation on a cold pod, without actually hitting the network."""

import pytest

from easy_nn.server import deps


@pytest.fixture
def fake_pip(tmp_path, monkeypatch):
    """Record pip invocations instead of running them, with a fresh cache."""
    monkeypatch.setattr(deps, "STAMP_DIR", str(tmp_path / "stamps"))
    calls = []

    class Result:
        returncode = 0
        stdout = "Successfully installed toy-1.0"

    def run(cmd, **kwargs):
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(deps.subprocess, "run", run)
    return calls


def test_nothing_to_install_is_not_an_install(fake_pip):
    assert deps.ensure([]) is False
    assert fake_pip == []


def test_installs_once_then_reuses_the_warm_pod(fake_pip):
    assert deps.ensure(["toy==1.0"]) is True
    assert deps.ensure(["toy==1.0"]) is False, "a warm pod must not reinstall"
    assert len(fake_pip) == 1
    assert "toy==1.0" in fake_pip[0]


def test_the_cache_key_is_the_set_not_the_order(fake_pip):
    deps.ensure(["a==1", "b==2"])
    assert deps.ensure(["b==2", "a==1"]) is False
    assert len(fake_pip) == 1


def test_a_different_set_installs_again(fake_pip):
    deps.ensure(["a==1"])
    deps.ensure(["a==2"])
    assert len(fake_pip) == 2


def test_pip_failure_is_reported_with_its_output(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "STAMP_DIR", str(tmp_path / "stamps"))

    class Result:
        returncode = 1
        stdout = "ERROR: No matching distribution found for nope"

    monkeypatch.setattr(deps.subprocess, "run", lambda cmd, **kw: Result())

    with pytest.raises(RuntimeError, match="No matching distribution"):
        deps.ensure(["nope"])

    # A failed install must not be remembered as done.
    monkeypatch.setattr(
        deps.subprocess,
        "run",
        lambda cmd, **kw: type("R", (), {"returncode": 0, "stdout": "ok"})(),
    )
    assert deps.ensure(["nope"]) is True


def test_progress_is_reported_to_the_client(fake_pip):
    said = []
    deps.ensure(["toy==1.0"], report=said.append)
    assert any("toy==1.0" in line for line in said)

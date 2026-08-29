"""Which modules travel by value, and -- more importantly -- which must not."""

import os
import sys

import torch

from easy_nn import job
from tests import toy


def decide(name):
    return job.is_project_module(
        name,
        sys.modules[name],
        job._project_roots(),
        job._library_roots(),
    )


def test_project_modules_ship():
    assert decide("tests.toy")
    assert os.path.abspath(toy.__file__).startswith(os.path.abspath(os.getcwd()))


def test_installed_libraries_do_not_ship():
    assert not decide("torch")
    assert not decide("cloudpickle")
    assert not decide("json")


def test_torch_pseudo_modules_do_not_ship():
    """torch.classes has a relative __file__ that resolves into the project
    directory; shipping torch internals by value breaks pickling of tensors."""
    for name in ("torch.classes", "torch.ops"):
        module = sys.modules.get(name)
        if module is None:
            continue
        assert not os.path.isabs(module.__file__), "assumption changed"
        assert not decide(name)


def test_easy_nn_itself_does_not_ship():
    # The executor has easy_nn installed; sending it by value would shadow it.
    assert not decide("easy_nn")
    assert not decide("easy_nn.trainer")


def test_main_module_is_left_to_cloudpickle():
    assert not job.is_project_module(
        "__main__", sys.modules["__main__"], job._project_roots(), job._library_roots()
    )


def test_registration_reports_only_project_modules():
    assert torch.__name__ in sys.modules  # torch must be loaded for this to mean anything
    registered = job.register_local_modules()
    assert "tests.toy" in registered
    assert not any(name.startswith("torch") for name in registered)
    assert not any(name.startswith("easy_nn") for name in registered)

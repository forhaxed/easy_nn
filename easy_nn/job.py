"""
Deciding which code travels.

cloudpickle sends classes defined in ``__main__`` by value and everything else
by reference.  That default is wrong here: a trainer living in
``myproject/trainers.py`` would arrive at the pod as a bare import that fails,
and it would fail *after* the weights finished uploading.

So before a job is packed, every module that belongs to your project -- as
opposed to an installed library -- is registered for pickling by value.  The
test is where the file lives: under your script or working directory means
yours; under site-packages or the standard library means the executor should
install it and import it itself.
"""

from __future__ import annotations

import os
import site
import sys
import sysconfig

import cloudpickle

#: Never ship these: easy_nn is installed on the executor, and the main module
#: is already pickled by value by cloudpickle itself.
_SKIP_NAMES = {"__main__", "__mp_main__"}
_SKIP_PREFIXES = ("easy_nn.", "easy_nn")


def _library_roots() -> set[str]:
    roots = set()
    for path in site.getsitepackages():
        roots.add(os.path.abspath(path))
    try:
        roots.add(os.path.abspath(site.getusersitepackages()))
    except Exception:
        pass
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.add(os.path.abspath(path))
    return roots


def _project_roots(extra=()) -> set[str]:
    roots = {os.path.abspath(os.getcwd())}
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    if main_file:
        roots.add(os.path.dirname(os.path.abspath(main_file)))
    for path in extra:
        roots.add(os.path.abspath(path))
    return roots


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def is_project_module(name, module, project, libraries) -> bool:
    """True if ``module`` is your code rather than an installed library."""
    if name in _SKIP_NAMES or name == "easy_nn" or name.startswith("easy_nn."):
        return False

    filename = getattr(module, "__file__", None)
    if not filename:
        return False

    # torch.classes and friends carry a bare relative __file__ that would
    # resolve against the current directory and masquerade as project code.
    # Shipping torch internals by value corrupts pickling of torch objects.
    if not os.path.isabs(filename):
        return False

    # A submodule of an installed package is library code even if its own file
    # sits somewhere odd; judge it by its top-level package.
    root_name = name.split(".")[0]
    root_module = sys.modules.get(root_name)
    root_file = getattr(root_module, "__file__", None) if root_module else None
    if root_file and os.path.isabs(root_file):
        root_path = os.path.abspath(root_file)
        if any(_under(root_path, root) for root in libraries):
            return False

    path = os.path.abspath(filename)
    if any(_under(path, root) for root in libraries):
        return False
    return any(_under(path, root) for root in project)


def register_local_modules(extra_roots=()) -> list[str]:
    """Mark project modules to be pickled by value.  Returns their names."""
    project = _project_roots(extra_roots)
    libraries = _library_roots()
    registered = []

    for name, module in list(sys.modules.items()):
        if not is_project_module(name, module, project, libraries):
            continue
        try:
            cloudpickle.register_pickle_by_value(module)
        except (ValueError, TypeError):
            continue
        registered.append(name)

    return sorted(registered)

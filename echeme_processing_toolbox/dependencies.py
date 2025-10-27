"""Centralised optional dependency guards for the toolbox."""
from __future__ import annotations

import sys
from types import ModuleType


def _import_numpy() -> ModuleType:
    try:
        import numpy as numpy_module  # type: ignore
    except ModuleNotFoundError as exc:
        executable = sys.executable or "python"
        raise ModuleNotFoundError(
            "NumPy is required by the Echeme Processing Toolbox but it was not "
            "found in the current Python environment.\n"
            "Install it into this interpreter with:\n"
            f"  {executable} -m pip install numpy\n"
            "If you use an IDE-managed virtual environment, ensure that NumPy is "
            "installed within that environment rather than a global interpreter."
        ) from exc
    return numpy_module


np = _import_numpy()

__all__ = ["np"]

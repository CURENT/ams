"""Capture runtime environment for a benchmark run.

Adapted from the removed ``ams/benchmarks.py``. ``pandapower`` is not
probed by default (step 2.4 scope decision, 2026-04-22); pass it
explicitly if needed.
"""
from __future__ import annotations

import datetime
import importlib.metadata as importlib_metadata
import os
import platform
import sys
from typing import Iterable

_DEFAULT_TOOLS: tuple[str, ...] = (
    "ltbams",
    "andes",
    "cvxpy",
    "gurobipy",
    "mosek",
    "piqp",
    "numba",
)


def capture_env(tools: Iterable[str] | None = None) -> dict:
    """Return a JSON-serializable dict describing the runtime stack."""
    tool_list = tuple(tools) if tools is not None else _DEFAULT_TOOLS

    versions: dict[str, str] = {}
    for tool in tool_list:
        try:
            versions[tool] = importlib_metadata.version(tool)
        except importlib_metadata.PackageNotFoundError:
            versions[tool] = "not installed"

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "tool_versions": versions,
    }

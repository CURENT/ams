"""Capture runtime environment for a benchmark run.

Adapted from the removed ``ams/benchmarks.py`` (commit ``45cd1bc8``).
``pandapower`` is not probed by default (step 2.4 scope decision,
2026-04-22); pass it explicitly if needed.
"""
from __future__ import annotations

import datetime
import importlib.metadata as importlib_metadata
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

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
    """Return a JSON-serializable dict describing the runtime stack.

    Covers software (Python, tool versions, BLAS backend, conda env),
    hardware (CPU brand + count, memory total/available), and
    provenance (git SHA + dirty flag, whether we're running inside
    pytest). Every field degrades gracefully to ``None`` / ``"unknown"``
    if it can't be determined — env capture must never raise.
    """
    tool_list = tuple(tools) if tools is not None else _DEFAULT_TOOLS

    versions: dict[str, str] = {}
    for tool in tool_list:
        try:
            versions[tool] = importlib_metadata.version(tool)
        except importlib_metadata.PackageNotFoundError:
            versions[tool] = "not installed"

    mem_total_gb, mem_avail_gb, mem_pct = _memory_snapshot()
    git_sha, git_dirty = _git_info()

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_brand": _cpu_brand(),
        "cpu_count": os.cpu_count(),
        "memory_total_gb": mem_total_gb,
        "memory_available_gb": mem_avail_gb,
        "memory_percent_used_at_start": mem_pct,
        "blas_backend": _blas_backend(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "running_under_pytest": "pytest" in sys.modules,
        "tool_versions": versions,
    }


def _memory_snapshot() -> tuple[float | None, float | None, float | None]:
    """Return (total_gb, available_gb, percent_used) via psutil if present."""
    try:
        import psutil
    except ImportError:
        return None, None, None
    try:
        vm = psutil.virtual_memory()
        return (
            round(vm.total / 1e9, 2),
            round(vm.available / 1e9, 2),
            vm.percent,
        )
    except Exception as exc:
        logger.warning("bench env: psutil.virtual_memory() failed: %s", exc)
        return None, None, None


def _cpu_brand() -> str:
    """Return a human-readable CPU model name, or ``platform.processor()`` fallback."""
    # Linux: /proc/cpuinfo
    proc_cpuinfo = Path("/proc/cpuinfo")
    if proc_cpuinfo.exists():
        try:
            for line in proc_cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception as exc:  # env capture must never raise
            logger.debug("bench env: /proc/cpuinfo read failed: %s", exc)
    # macOS: sysctl
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    # Fallback — Windows usually works here, macOS silicon returns "arm".
    return platform.processor() or "unknown"


def _git_info() -> tuple[str | None, bool | None]:
    """Return (short sha, dirty flag). Both ``None`` if not in a git repo."""
    try:
        sha_res = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if sha_res.returncode != 0 or not sha_res.stdout.strip():
            return None, None
        sha = sha_res.stdout.strip()

        dirty_res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        dirty = bool(dirty_res.stdout.strip()) if dirty_res.returncode == 0 else None
        return sha, dirty
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None, None


def _blas_backend() -> str | None:
    """Summarize numpy's BLAS backend name, or ``None`` if undetectable."""
    try:
        import numpy
    except ImportError:
        return None
    try:
        config = numpy.show_config(mode="dicts")
    except (TypeError, AttributeError):
        return None
    try:
        deps = config.get("Build Dependencies", {})
        blas = deps.get("blas") or {}
        name = blas.get("name")
        return name or None
    except Exception:
        return None

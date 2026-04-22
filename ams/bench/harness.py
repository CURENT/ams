"""Timer core for the AMS bench suite.

Zero non-stdlib deps. ``measure()`` wraps a callable with a configurable
warmup + repeat count and returns a ``Measurement`` summary. Exceptions
are caught per-rep so a single broken benchmark never aborts the suite.
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class Measurement:
    """Summary of one benchmarked callable.

    ``raw_s`` holds the per-rep wall-clock seconds; summary fields are
    ``None`` when the benchmark errored before producing any reps.
    """

    name: str
    group: str
    reps: int
    warmup_reps: int
    mean_s: float | None
    stdev_s: float | None
    min_s: float | None
    max_s: float | None
    raw_s: list[float]
    error: str | None = None


def measure(
    fn: Callable[[], float | None],
    *,
    name: str,
    group: str,
    reps: int = 5,
    warmup_reps: int = 1,
) -> Measurement:
    """Run ``fn`` ``warmup_reps`` times (discarded) then ``reps`` times measured.

    ``fn`` can self-time by returning a float (seconds), or return ``None``
    and let this helper wrap the call in ``perf_counter``. Self-timing lets
    benchmarks exclude setup/teardown from the measured window.

    On any exception, logs a warning and returns a ``Measurement`` with
    ``error`` set and ``None`` summary stats; the caller continues.
    """
    try:
        for _ in range(warmup_reps):
            fn()
    except Exception as exc:
        logger.warning("bench %s/%s: warmup failed: %s", group, name, exc)
        return _failed(name, group, reps, warmup_reps, str(exc))

    raw: list[float] = []
    for _ in range(reps):
        try:
            t0 = time.perf_counter()
            out = fn()
            dt = time.perf_counter() - t0
            raw.append(float(out) if isinstance(out, (int, float)) else dt)
        except Exception as exc:
            logger.warning("bench %s/%s: rep failed: %s", group, name, exc)
            return _failed(name, group, reps, warmup_reps, str(exc))

    return Measurement(
        name=name,
        group=group,
        reps=reps,
        warmup_reps=warmup_reps,
        mean_s=statistics.fmean(raw),
        stdev_s=statistics.stdev(raw) if len(raw) > 1 else 0.0,
        min_s=min(raw),
        max_s=max(raw),
        raw_s=raw,
    )


def _failed(name: str, group: str, reps: int, warmup_reps: int, msg: str) -> Measurement:
    return Measurement(
        name=name,
        group=group,
        reps=reps,
        warmup_reps=warmup_reps,
        mean_s=None,
        stdev_s=None,
        min_s=None,
        max_s=None,
        raw_s=[],
        error=msg,
    )

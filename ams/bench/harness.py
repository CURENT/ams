"""Timer core for the AMS bench suite.

Zero non-stdlib deps. ``measure()`` wraps a callable with a configurable
warmup + repeat count and returns a ``Measurement`` summary. Exceptions
are caught per-rep so a single broken benchmark never aborts the suite.
``summarize()`` builds the same structure from a pre-collected list of
per-rep timings — useful when one iteration produces multiple dimensions
(e.g., 5-phase routine-init breakdowns that share one system load).
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class Measurement:
    """Summary of one benchmarked callable.

    ``raw_s`` holds the per-rep wall-clock seconds; summary fields are
    ``None`` when the benchmark errored before producing any reps.
    Structured dimensions (``routine`` / ``case`` / ``solver`` / ``phase``)
    are optional tags that let downstream consumers (dashboards, cloud
    reporters) filter without parsing the ``name`` string.
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
    routine: str | None = None
    case: str | None = None
    solver: str | None = None
    phase: str | None = None
    metadata: dict = field(default_factory=dict)


def measure(
    fn: Callable[[], float | int | None],
    *,
    name: str,
    group: str,
    reps: int = 5,
    warmup_reps: int = 1,
    routine: str | None = None,
    case: str | None = None,
    solver: str | None = None,
    phase: str | None = None,
    metadata: dict | None = None,
) -> Measurement:
    """Run ``fn`` ``warmup_reps`` times (discarded) then ``reps`` times measured.

    ``fn`` can self-time by returning a number (seconds), or return ``None``
    and let this helper wrap the call in ``perf_counter``. Self-timing lets
    benchmarks exclude setup/teardown from the measured window.

    On a rep exception, logs a warning, stops iterating, and returns a
    ``Measurement`` built from whatever successful reps completed — with
    ``error`` set to the exception string so partial progress isn't hidden.
    Only when *no* rep succeeds are summary stats ``None``.
    """
    try:
        for _ in range(warmup_reps):
            fn()
    except Exception as exc:
        logger.warning("bench %s/%s: warmup failed: %s", group, name, exc)
        return summarize(
            [],
            name=name, group=group, reps=reps, warmup_reps=warmup_reps,
            error=str(exc),
            routine=routine, case=case, solver=solver, phase=phase,
            metadata=metadata,
        )

    raw: list[float] = []
    err: str | None = None
    for _ in range(reps):
        try:
            t0 = time.perf_counter()
            out = fn()
            dt = time.perf_counter() - t0
            raw.append(float(out) if isinstance(out, (int, float)) else dt)
        except Exception as exc:
            logger.warning("bench %s/%s: rep failed: %s", group, name, exc)
            err = str(exc)
            break

    return summarize(
        raw,
        name=name,
        group=group,
        reps=reps,
        warmup_reps=warmup_reps,
        error=err,
        routine=routine,
        case=case,
        solver=solver,
        phase=phase,
        metadata=metadata,
    )


def summarize(
    raw: list[float],
    *,
    name: str,
    group: str,
    reps: int,
    warmup_reps: int,
    error: str | None = None,
    routine: str | None = None,
    case: str | None = None,
    solver: str | None = None,
    phase: str | None = None,
    metadata: dict | None = None,
) -> Measurement:
    """Build a Measurement from a pre-collected list of per-rep timings.

    Use when you can't fit the work into a single ``fn`` call — e.g., a
    multi-phase init breakdown where one system load produces five
    timings, one per phase.
    """
    common = dict(
        name=name, group=group, reps=reps, warmup_reps=warmup_reps,
        routine=routine, case=case, solver=solver, phase=phase,
        metadata=metadata or {},
    )
    if not raw:
        return Measurement(
            mean_s=None, stdev_s=None, min_s=None, max_s=None, raw_s=[],
            error=error or "no successful reps", **common,
        )
    return Measurement(
        mean_s=statistics.fmean(raw),
        stdev_s=statistics.stdev(raw) if len(raw) > 1 else 0.0,
        min_s=min(raw),
        max_s=max(raw),
        raw_s=raw,
        error=error,
        **common,
    )



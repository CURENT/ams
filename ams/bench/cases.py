"""Case ladder for the AMS bench suite.

Each ``CaseSpec.path`` is an ``ams.get_case``-compatible relative
path. ``ROUTINE_CASES`` holds the subset that ships with the time-series
data needed for RTED / ED / UC-style routines.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseSpec:
    name: str
    path: str


# Full ladder for the case-load benchmark — spans small to stress scale.
CASE_LADDER: tuple[CaseSpec, ...] = (
    CaseSpec("case5", "matpower/case5.m"),
    CaseSpec("ieee14_uced", "ieee14/ieee14_uced.xlsx"),
    CaseSpec("ieee39_uced", "ieee39/ieee39_uced.xlsx"),
    CaseSpec("case300", "matpower/case300.m"),
    CaseSpec("npcc_uced", "npcc/npcc_uced.xlsx"),
)

# Cases where routine-level benchmarks run. Restricted to ``*_uced``
# variants because they ship the time-series / cost data that the
# scheduling routines expect.
ROUTINE_CASES: tuple[CaseSpec, ...] = (
    CaseSpec("ieee14_uced", "ieee14/ieee14_uced.xlsx"),
    CaseSpec("ieee39_uced", "ieee39/ieee39_uced.xlsx"),
)

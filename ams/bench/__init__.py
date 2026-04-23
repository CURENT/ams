"""AMS performance benchmark suite.

Homegrown, stdlib-only plus ``psutil`` (already an AMS runtime dep).
Runnable via ``python -m ams.bench``. See ``bench/README.md`` at the
repo root for usage and submission format.
"""
from ams.bench.env import capture_env
from ams.bench.harness import Measurement, measure, summarize
from ams.bench.routines import run_smoke, run_tier_a
from ams.bench.schema import SCHEMA_VERSION, build_report

__all__ = [
    "Measurement",
    "SCHEMA_VERSION",
    "build_report",
    "capture_env",
    "measure",
    "run_smoke",
    "run_tier_a",
    "summarize",
]

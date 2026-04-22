"""AMS performance benchmark suite.

Homegrown, zero non-stdlib deps. Runnable via ``python -m ams.bench``.
See ``bench/README.md`` at the repo root for usage and submission format.
"""
from ams.bench.env import capture_env
from ams.bench.harness import Measurement, measure
from ams.bench.schema import SCHEMA_VERSION, build_report

__all__ = [
    "Measurement",
    "SCHEMA_VERSION",
    "build_report",
    "capture_env",
    "measure",
]

"""Output schema for the AMS bench suite.

The suite emits one JSON document per run. ``SCHEMA_VERSION`` is
explicit so future consumers (cloud reporter, dashboard) can reject
incompatible runs instead of silently mis-parsing them.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from ams.bench.harness import Measurement

SCHEMA_VERSION = 1


def build_report(
    *,
    suite: str,
    environment: dict,
    results: Iterable[Measurement],
) -> dict:
    """Assemble a report dict ready for ``json.dumps()``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "environment": environment,
        "results": [asdict(m) for m in results],
    }

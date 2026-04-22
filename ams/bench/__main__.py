"""CLI entry point: ``python -m ams.bench``.

M1 scaffold — emits a report with the environment captured and an
empty ``results`` list. M2 wires in the Tier A benchmarks.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from ams.bench.env import capture_env
from ams.bench.harness import Measurement
from ams.bench.schema import build_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ams.bench",
        description="Run the AMS performance benchmark suite.",
    )
    parser.add_argument(
        "--suite",
        default="tier_a",
        help="Suite to run (default: tier_a).",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output path for the JSON report, or '-' for stdout (default).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Measured reps per benchmark (default: 5).",
    )
    parser.add_argument(
        "--warmup-reps",
        type=int,
        default=1,
        help="Warmup reps, discarded (default: 1).",
    )
    args = parser.parse_args(argv)

    env = capture_env()
    results: list[Measurement] = []  # M2 populates this.

    report = build_report(suite=args.suite, environment=env, results=results)
    payload = json.dumps(report, indent=2)

    if args.output == "-":
        print(payload)
    else:
        with open(args.output, "w") as fh:
            fh.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

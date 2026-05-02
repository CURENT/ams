"""CLI entry point: ``python -m ams.bench``.

Emits a JSON report with the ``environment`` captured and the results
of the selected ``--suite``. ``tier_a`` is the only suite that
produces real measurements today; M4 adds Tier B (co-sim, solver
sweep).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import Sequence

from ams.bench.env import capture_env
from ams.bench.harness import Measurement
from ams.bench.routines import run_smoke, run_tier_a
from ams.bench.schema import build_report

_SUITES = {
    "smoke": run_smoke,
    "tier_a": run_tier_a,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ams.bench",
        description="Run the AMS performance benchmark suite.",
    )
    parser.add_argument(
        "--suite",
        default="tier_a",
        choices=sorted(_SUITES.keys()),
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
    parser.add_argument(
        "--solver",
        default="CLARABEL",
        help="Solver to use for routine_solve benchmarks (default: CLARABEL). "
             "Per-result JSON records the solver used so outputs from different "
             "runs stay comparable. UC-family routines need a MIP solver and "
             "are not in Tier A.",
    )
    args = parser.parse_args(argv)

    env = capture_env()

    # Some benchmarked routines (notably ACOPF via PYPOWER) print to
    # stdout while solving. Redirect stdout to stderr during suite
    # execution so our JSON report on real stdout stays clean.
    with contextlib.redirect_stdout(sys.stderr):
        results: list[Measurement] = _SUITES[args.suite](
            reps=args.reps,
            warmup_reps=args.warmup_reps,
            solver=args.solver,
        )

    report = build_report(suite=args.suite, environment=env, results=results)
    payload = json.dumps(report, indent=2)

    if args.output == "-":
        print(payload)
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# AMS benchmark suite

Zero-dep performance benchmarking for AMS. This directory holds
reference baselines and user-submitted reports; the suite's **code**
lives in `ams/bench/` and ships in the wheel, so any pip-installed
user can run `python -m ams.bench` without cloning the repo.

## Running

```bash
pip install ltbams   # or work from a dev checkout
python -m ams.bench --suite tier_a > my_report.json
```

### Suites

| suite | approx. runtime | purpose |
|---|---|---|
| `smoke` | ~3 s | CI smoke test (1 case-load + 1 solve). Not a perf measurement. |
| `tier_a` | ~90 s | Baseline suite: 53 measurements across 5 cases + 4 routines. |

### CLI flags

```
--suite {smoke, tier_a}    suite to run (default: tier_a)
--output PATH | -          write JSON to PATH, or '-' for stdout (default)
--reps N                   measured reps per benchmark (default: 5)
--warmup-reps N            warmup reps, discarded (default: 1)
--solver SOLVER            solver for routine_solve (default: CLARABEL)
```

## Output (schema v1)

Each run emits one JSON document with `schema_version`, `suite`,
`environment`, and a `results` list. A single result looks like:

```json
{
  "name": "DCOPF/ieee14_uced/CLARABEL",
  "group": "routine_solve",
  "routine": "DCOPF",
  "case": "ieee14_uced",
  "solver": "CLARABEL",
  "phase": null,
  "reps": 5,
  "warmup_reps": 1,
  "mean_s": 0.00074,
  "stdev_s": 0.00001,
  "min_s": 0.00073,
  "max_s": 0.00076,
  "raw_s": [0.00073, 0.00074, 0.00074, 0.00076, 0.00073],
  "error": null,
  "metadata": {}
}
```

### Groups

- **`case_load`** — `ams.load(case)` wall-clock. File-parse + data-setup cost.
- **`routine_init_phase`** — 5 phases per routine per case
  (`mats`, `parse`, `evaluate`, `finalize`, `init`). Each rep starts
  with a fresh `ams.load(setup=True)` which already triggers first-time
  CVXPY compilation; the timed phase calls exercise **re-execution**
  cost, not cold start. Catches regressions in the interactive-iteration
  path. Cold-start timing is separate future work.
- **`routine_solve`** — `rtn.run(solver=...)` only. `rtn.init()` is
  called explicitly outside the timed loop so the first rep's number
  is insensitive to `--warmup-reps`.

### Environment fields

Reproducibility metadata captured with every run. Key fields:

- `git_commit`, `git_dirty` — code pinpoint. **`dirty=true` means the
  run included uncommitted changes; baselines must be captured on
  clean trees.**
- `blas_backend` — numpy's BLAS binding (`accelerate` on macOS,
  `openblas`, `mkl`, …). Strongly affects linear-algebra perf.
- `cpu_brand`, `cpu_count`, `memory_total_gb`,
  `memory_percent_used_at_start` — hardware profile + load at capture.
- `tool_versions` — `ltbams`, `andes`, `cvxpy`, `gurobipy`, `mosek`,
  `piqp`, `numba`.
- `running_under_pytest` — `true` if invoked from inside a pytest
  subprocess.

## Layout

```
bench/
├── README.md        # this file
├── baselines/       # maintainer-captured reference runs on clean develop
│   └── YYYY-MM-DD_<branch>.json
└── user_reports/    # optional: user-submitted bench outputs (created on demand)
```

`bench/` is tracked in git but **excluded from the wheel** — baselines
belong to the repo, not to pip-installed users. The wheel ships
`ams/bench/` (the runner) only.

## Submitting a user report

1. `python -m ams.bench --suite tier_a > my_report.json`
2. Attach `my_report.json` to a GitHub Discussion or Issue on
   `CURENT/ams`. One-line context (platform, any non-default
   configuration) helps the maintainer interpret.
3. Maintainer may drop your file into `bench/user_reports/` for
   longer-term reference if relevant.

## Caveats

- **Warm-path init** — `routine_init_phase` does not capture
  first-time CVXPY compile cost (see above).
- **UC / UC2 not in Tier A** — these are MIPs and the default
  `CLARABEL` can't solve them. Pass `--solver HIGHS` (or `GUROBI`,
  `MOSEK`) to bench UC-family routines; a Tier B suite is planned.
- **Quiet machine matters** — numbers on a busy laptop have higher
  stdev. Check `memory_percent_used_at_start` to gauge pressure;
  compare means ± stdev, not single reps.
- **Non-deterministic solvers** — solvers may exhibit small run-to-run
  variance. A baseline is a distribution, not a point.

# AMS benchmark suite

This directory holds benchmark baselines and user-submitted reports
for the AMS performance suite. The suite's **code** lives in
`ams/bench/` and ships in the wheel; this directory holds **data**
and is intentionally excluded from the wheel.

## Running a benchmark

```bash
python -m ams.bench --suite tier_a --output bench/baselines/YYYY-MM-DD_<branch>.json
```

Full usage and interpretation docs land with milestone M3 of step 2.4.

## Layout

```
bench/
├── README.md        # this file
├── baselines/       # maintainer-captured reference runs
└── user_reports/    # optional: user-submitted benchmark outputs
```

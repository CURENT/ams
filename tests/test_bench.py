"""Smoke tests for the AMS benchmark suite.

Kept fast so they can run as part of the regular test suite. Real
perf measurement isn't exercised here — that's what
``python -m ams.bench --suite tier_a`` is for.
"""
import json
import subprocess
import sys


def test_bench_public_exports_are_callable():
    """All public names from ams.bench import and are callable."""
    from ams.bench import (
        SCHEMA_VERSION, Measurement, build_report, capture_env,
        measure, run_smoke, run_tier_a, summarize,
    )
    assert SCHEMA_VERSION == 1
    assert Measurement.__name__ == "Measurement"
    for fn in (measure, summarize, capture_env, build_report, run_smoke, run_tier_a):
        assert callable(fn)


def test_bench_env_capture_has_expected_fields():
    """capture_env() always returns a dict with the documented fields."""
    from ams.bench import capture_env
    env = capture_env()
    assert isinstance(env, dict)
    for key in (
        "timestamp", "python", "platform", "machine", "cpu_brand",
        "cpu_count", "memory_total_gb", "memory_available_gb",
        "memory_percent_used_at_start", "blas_backend", "conda_env",
        "git_commit", "git_dirty", "running_under_pytest", "tool_versions",
    ):
        assert key in env, f"missing env field: {key}"
    assert env["python"].startswith("3.")
    assert env["running_under_pytest"] is True
    assert "ltbams" in env["tool_versions"]


def test_bench_build_report_shape():
    """build_report() wraps env + results in the versioned envelope."""
    from ams.bench import build_report, capture_env
    report = build_report(suite="tier_a", environment=capture_env(), results=[])
    assert report["schema_version"] == 1
    assert report["suite"] == "tier_a"
    assert report["environment"]["running_under_pytest"] is True
    assert report["results"] == []


def test_bench_measure_catches_exception_and_reports_error():
    """A fn that always raises lands as a Measurement with error set."""
    from ams.bench import measure

    def boom():
        raise RuntimeError("intentional test failure")

    m = measure(boom, name="boom", group="test", reps=3, warmup_reps=0)
    assert m.error is not None
    assert "intentional test failure" in m.error
    assert m.raw_s == []
    assert m.mean_s is None


def test_bench_cli_smoke_suite_emits_valid_schema_v1_report():
    """``python -m ams.bench --suite smoke`` runs end-to-end and emits valid JSON."""
    result = subprocess.run(
        [sys.executable, "-m", "ams.bench", "--suite", "smoke",
         "--reps", "1", "--warmup-reps", "0"],
        capture_output=True, text=True, check=True, timeout=60,
    )
    report = json.loads(result.stdout)

    assert report["schema_version"] == 1
    assert report["suite"] == "smoke"
    assert isinstance(report["results"], list)
    assert len(report["results"]) >= 1

    # Each result carries the schema v1 required fields.
    for res in report["results"]:
        for key in ("name", "group", "reps", "warmup_reps", "raw_s",
                    "routine", "case", "solver", "phase", "metadata"):
            assert key in res, f"result missing required field {key}: {res}"


def test_bench_cli_rejects_unknown_suite():
    """argparse's choices= should catch typos before we hit any ams.load."""
    result = subprocess.run(
        [sys.executable, "-m", "ams.bench", "--suite", "not_a_real_suite"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "not_a_real_suite" in result.stderr

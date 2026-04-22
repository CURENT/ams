"""Smoke test for the AMS benchmark suite scaffold.

Keeps ``ams/bench/`` honest: if a refactor breaks the CLI entry point
or the output schema, this test fails loudly. Full perf runs are not
exercised here — that's the suite's own job.
"""
import json
import subprocess
import sys


def test_bench_cli_emits_valid_schema_v1_report():
    """``python -m ams.bench`` emits a schema-v1 JSON with env captured."""
    out = subprocess.check_output(
        [sys.executable, "-m", "ams.bench", "--suite", "tier_a"],
        text=True,
    )
    report = json.loads(out)

    assert report["schema_version"] == 1
    assert report["suite"] == "tier_a"
    assert isinstance(report["results"], list)

    env = report["environment"]
    assert env["python"].startswith("3.")
    assert "ltbams" in env["tool_versions"]
    assert "andes" in env["tool_versions"]

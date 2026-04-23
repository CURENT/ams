#!/usr/bin/env bash
# ============================================================
# AMS × ANDES v2.0.0 Compatibility Audit — Phase 0.1
#
# Run from the AMS repo root in the andesre conda environment:
#
#   conda activate andesre
#   cd ~/work/ams
#   bash tests/run_compat_audit.sh
#
# Appends results to .claude/designs/step_0_1_results.txt
# ============================================================

AMS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AMS_ROOT"

RESULTS=".claude/designs/step_0_1_results.txt"
mkdir -p .claude/designs

echo "" | tee -a "$RESULTS"
echo "================================================================" | tee -a "$RESULTS"
echo "Phase 0.1 — Full test run (compat shim applied)" | tee -a "$RESULTS"
echo "Date: $(date)" | tee -a "$RESULTS"
echo "Branch: $(git branch --show-current)" | tee -a "$RESULTS"
echo "================================================================" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

echo "=== Environment ===" | tee -a "$RESULTS"
python -c "import sys; print('Python:', sys.version.split()[0])" 2>&1 | tee -a "$RESULTS"
python -c "import andes; print('ANDES:', andes.__version__)" 2>&1 | tee -a "$RESULTS"
python -c "import ams; print('AMS:', ams.__version__)" 2>&1 | tee -a "$RESULTS" || echo "AMS: import failed" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

echo "=== Full pytest run ===" | tee -a "$RESULTS"
python -m pytest tests/ -v --tb=short --no-header 2>&1 | tee -a "$RESULTS" || true

echo "" | tee -a "$RESULTS"
echo "=== One-line summary ===" | tee -a "$RESULTS"
python -m pytest tests/ -q --tb=no --no-header 2>&1 | tail -3 | tee -a "$RESULTS" || true

echo ""
echo "Results appended to: $RESULTS"

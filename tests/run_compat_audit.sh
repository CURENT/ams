#!/usr/bin/env bash
# ============================================================
# AMS × ANDES v2.0.0 Compatibility Audit — Phase 0.1
# Run this script in the andesre conda environment:
#
#   conda activate andesre
#   cd /path/to/ams
#   bash tests/run_compat_audit.sh
#
# Output is saved to .claude/designs/step_0_1_results.txt
# ============================================================

set -euo pipefail

RESULTS=".claude/designs/step_0_1_results.txt"
AMS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AMS_ROOT"

mkdir -p .claude/designs

echo "================================================================" | tee "$RESULTS"
echo "AMS × ANDES v2.0.0 Compatibility Audit — Phase 0.1" | tee -a "$RESULTS"
echo "Date: $(date)" | tee -a "$RESULTS"
echo "================================================================" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Environment info ---
echo "=== Environment ===" | tee -a "$RESULTS"
python -c "import sys; print('Python:', sys.version)" 2>&1 | tee -a "$RESULTS"
python -c "import andes; print('ANDES:', andes.__version__)" 2>&1 | tee -a "$RESULTS" || \
    echo "ANDES: IMPORT FAILED" | tee -a "$RESULTS"
python -c "import cvxpy; print('CVXPY:', cvxpy.__version__)" 2>&1 | tee -a "$RESULTS" || \
    echo "CVXPY: not installed" | tee -a "$RESULTS"
python -c "import pypower; print('PyPower: available')" 2>&1 | tee -a "$RESULTS" || \
    echo "PyPower: not installed" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- ANDES shared.py probe: identify which symbols are missing ---
echo "=== ANDES shared.py symbol probe ===" | tee -a "$RESULTS"
python - <<'PYEOF' 2>&1 | tee -a "$RESULTS"
symbols = ['NCPUS_PHYSICAL', 'NCPUS', 'coloredlogs', 'Pool', 'Process',
           'unittest', 'deg2rad', 'rad2deg', 'np', 'pd']
import importlib
m = importlib.import_module('andes.shared')
for sym in symbols:
    status = 'PRESENT' if hasattr(m, sym) else 'MISSING'
    print(f"  andes.shared.{sym}: {status}")
PYEOF
echo "" | tee -a "$RESULTS"

# --- ANDES system structure probe ---
echo "=== andes.system structure probe ===" | tee -a "$RESULTS"
python - <<'PYEOF' 2>&1 | tee -a "$RESULTS"
try:
    from andes.system import System, _config_numpy, load_config_rc
    print("  System, _config_numpy, load_config_rc: OK")
except ImportError as e:
    print(f"  FAILED: {e}")
PYEOF
echo "" | tee -a "$RESULTS"

# --- Step 1: Run pytest as-is (expect total failure) ---
echo "=== Pytest run 1: unpatched AMS against ANDES v2.0.0 ===" | tee -a "$RESULTS"
echo "(Expect: all tests fail with ImportError)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

python -m pytest tests/ -x \
    --tb=short \
    --no-header \
    -q \
    2>&1 | head -50 | tee -a "$RESULTS" || true

echo "" | tee -a "$RESULTS"

# --- Step 2: Apply minimal compat patch and re-run ---
echo "=== Applying minimal compat patch ===" | tee -a "$RESULTS"

# Patch ams/main.py
python - <<'PYEOF' 2>&1 | tee -a "$RESULTS"
import re

# --- Patch ams/main.py ---
with open('ams/main.py', 'r') as f:
    src = f.read()

old = "from andes.shared import Pool, Process, coloredlogs, unittest, NCPUS_PHYSICAL"
new = """from andes.shared import Pool, Process, unittest
try:
    from andes.shared import NCPUS_PHYSICAL
except ImportError:
    from andes.shared import NCPUS as NCPUS_PHYSICAL  # renamed in ANDES v2.0.0
try:
    from andes.shared import coloredlogs
except ImportError:
    coloredlogs = None  # removed in ANDES v2.0.0"""

if old in src:
    with open('ams/main.py', 'w') as f:
        f.write(src.replace(old, new))
    print("  ams/main.py: patched OK")
else:
    print("  ams/main.py: pattern not found (already patched?)")

# --- Patch ams/cli.py ---
with open('ams/cli.py', 'r') as f:
    src = f.read()

old_cli = "from andes.shared import NCPUS_PHYSICAL"
new_cli = """try:
    from andes.shared import NCPUS_PHYSICAL
except ImportError:
    from andes.shared import NCPUS as NCPUS_PHYSICAL  # renamed in ANDES v2.0.0"""

if old_cli in src:
    with open('ams/cli.py', 'w') as f:
        f.write(src.replace(old_cli, new_cli))
    print("  ams/cli.py: patched OK")
else:
    print("  ams/cli.py: pattern not found (already patched?)")
PYEOF

echo "" | tee -a "$RESULTS"

# --- Step 3: Verify import ams now works ---
echo "=== import ams probe (post-patch) ===" | tee -a "$RESULTS"
python -c "import ams; print('import ams: OK, version', ams.__version__)" 2>&1 | tee -a "$RESULTS" || \
    echo "import ams: STILL FAILING" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# --- Step 4: Full pytest run with patched AMS ---
echo "=== Pytest run 2: patched AMS against ANDES v2.0.0 (full suite) ===" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

python -m pytest tests/ \
    --tb=short \
    --no-header \
    -v \
    2>&1 | tee -a "$RESULTS" || true

echo "" | tee -a "$RESULTS"
echo "=== Summary ===" | tee -a "$RESULTS"
python -m pytest tests/ \
    --tb=no \
    --no-header \
    -q \
    2>&1 | tail -10 | tee -a "$RESULTS" || true

echo "" | tee -a "$RESULTS"
echo "Results written to: $RESULTS"
echo "================================================================"
echo "IMPORTANT: The patch above modifies ams/main.py and ams/cli.py."
echo "Review the diff before committing. These are temporary compat shims;"
echo "the proper fix is Phase 1.5 (logging) and 1.6 (CLI)."
echo "================================================================"

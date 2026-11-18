#!/bin/bash
# Create the Part-II virtual environment with the LATEST Qiskit stack, separate from Phase 1's ParOpt_PyEnv
# (which keeps qiskit 1.4.2 and is never modified).  One-time, from the Optimization folder:
#     bash hsf2_scripts/setup_hsf2_venv.sh                 # -> /eos/user/<u>/<user>/my_acts/hsf2_venv
#     bash hsf2_scripts/setup_hsf2_venv.sh <folder>        # elsewhere
# * Base interpreter: the system /usr/bin/python3.11 (else 3.12), NOT the LCG/ACTS python, so nothing of the ACTS
#   environment (PYTHONPATH, LD_LIBRARY_PATH, LCG site-packages with numpy 1.x) leaks into Part II.  PYTHON=... overrides.
# * Packages: the newest releases of hsf2_docs/requirements_hsf2.txt (qiskit, qiskit-aer, qiskit-ibm-runtime, numpy,
#   scipy, pandas, matplotlib, optuna, pytest) and the CPU build of PyTorch (~200 MB instead of the 2 GB CUDA build).
# * A .pth file makes the Optimization folder (hsf2/ and acts_qaoa/) importable from any working directory; it is
#   appended after site-packages, so Phase-1 top-level folders can never shadow an installed package.
# * The installed versions are written to hsf2_work/env/pip_freeze_<stamp>.txt (cited by every result's provenance).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)                    # the Optimization folder
HSF2_EOS_HOME=${HSF2_EOS_HOME:-/eos/user/$(id -un | cut -c1)/$(id -un)}   # your EOS home (id -un: also in batch jobs)
VENV=${1:-${HSF2_VENV:-$HSF2_EOS_HOME/my_acts/hsf2_venv}}
REQ="$ROOT/hsf2_docs/requirements_hsf2.txt"
[ -f "$REQ" ] || { echo "missing $REQ"; exit 2; }
[ -e "$VENV" ] && { echo "$VENV already exists: remove it first or give another folder"; exit 2; }
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_LIBRARY_PATH
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in /usr/bin/python3.11 /usr/bin/python3.12; do [ -x "$c" ] && { PY=$c; break; }; done
fi
[ -n "$PY" ] || { echo "no /usr/bin/python3.11 or python3.12: run with PYTHON=/path/to/python3.x"; exit 2; }
echo "base interpreter: $PY ($("$PY" -V 2>&1))  ->  venv $VENV"
"$PY" -m venv "$VENV"
P="$VENV/bin/python"
"$P" -m pip install --no-cache-dir --upgrade pip
"$P" -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
"$P" -m pip install --no-cache-dir -r "$REQ"
SITE=$("$P" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$ROOT" > "$SITE/hsf2_optimization_folder.pth"
mkdir -p "$ROOT/hsf2_work/env"
FREEZE="$ROOT/hsf2_work/env/pip_freeze_$(date +%d-%m-%Y-%H-%M-%S).txt"
"$P" -m pip freeze > "$FREEZE"
"$P" - <<'EOF'
import sys, numpy, scipy, pandas, torch, qiskit, qiskit_aer, qiskit_ibm_runtime, optuna, hsf2, acts_qaoa
print("OK  python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__, "pandas",
      pandas.__version__, "torch", torch.__version__)
print("    qiskit", qiskit.__version__, "qiskit-aer", qiskit_aer.__version__, "qiskit-ibm-runtime",
      qiskit_ibm_runtime.__version__, "optuna", optuna.__version__, "| hsf2", hsf2.__version__, "acts_qaoa",
      acts_qaoa.__version__)
EOF
echo "versions: $FREEZE"
echo "next: open a NEW shell (or 'deactivate'), then:  source $ROOT/hsf2_scripts/env_lxplus.sh"

#!/bin/bash
# HTCondor wrapper for HSF Part II:   run.sh <command ...>     or   run.sh script <hsf2_scripts file> [args]
#   e.g.  arguments = python -m hsf2 trackml-prep --workers 4
#         arguments = script trackml_download.sh
# Self-contained (getenv = False in the .sub files): it activates the Part-II venv with the latest Qiskit stack,
# ignores the ACTS / LCG environment, sets every HSF2_* path, and runs in hsf2_work/ so relative paths can never
# land in the Phase-1 Optimization folder.
set -euo pipefail
HSF2_EOS_HOME=${HSF2_EOS_HOME:-/eos/user/$(id -un | cut -c1)/$(id -un)}   # your EOS home (id -un: also in batch jobs)
ROOT=${HSF2_ROOT:-$HSF2_EOS_HOME/my_acts/acts/Examples/Scripts/Optimization}
VENV=${HSF2_VENV:-$HSF2_EOS_HOME/my_acts/hsf2_venv}
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
[ -x "$VENV/bin/python" ] || { echo "Part-II venv $VENV missing: run hsf2_scripts/setup_hsf2_venv.sh once"; exit 3; }
set +u; source "$VENV/bin/activate"; set -u
export HSF2_ROOT=$ROOT HSF2_VENV=$VENV HSF_PHASE1_DIR=$ROOT
export HSF2_WORK=$ROOT/hsf2_work
export HSF2_DATA=$HSF2_WORK/data HSF2_MODELS=$HSF2_WORK/models HSF2_RUNS=$HSF2_WORK/runs
export HSF2_FIGURES=$HSF2_WORK/figures HSF2_SCRIPTS=$ROOT/hsf2_scripts
[ -f "$ROOT/hsf2_docs/DEPLOYED_REV" ] && export HSF2_CODE_REV=$(head -1 "$ROOT/hsf2_docs/DEPLOYED_REV")
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HSF2_DATA" "$HSF2_MODELS" "$HSF2_RUNS" "$HSF2_FIGURES"
cd "$HSF2_WORK"
echo "host: $(hostname)   date: $(date '+%d-%m-%Y %H:%M:%S')   python: $(which python) ($(python -V 2>&1))"
echo "work: $HSF2_WORK   code: ${HSF2_CODE_REV:-?}   threads: $OMP_NUM_THREADS"
echo "command: $*"
python -c "import hsf2, torch, qiskit" || { echo "hsf2/torch/qiskit not importable in $VENV"; exit 3; }
if [ "${1:-}" = "script" ]; then
  shift
  name=$1; shift
  exec bash "$HSF2_SCRIPTS/$name" "$@"
fi
exec "$@"

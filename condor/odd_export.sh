#!/bin/bash
# HTCondor wrapper of HSF Part II for the ODD export (hsf2_scripts/odd_export.py): ACTS environment, NOT the Part-II venv.
#   arguments = odd_export.sh --events 100 --seed 1 --label odd_s1
# The ACTS environment (python bindings + ParOpt_PyEnv) is inherited from the submitting shell (getenv = True in the
# .sub file, as Phase 1's run_stage.sh), so submit from a shell in which  python -c "import acts"  works.
# Writes only inside hsf2_work/odd_export/; runs in hsf2_work/ so relative paths cannot land in the Phase-1 folder.
set -euo pipefail
HSF2_EOS_HOME=${HSF2_EOS_HOME:-/eos/user/$(id -un | cut -c1)/$(id -un)}   # your EOS home (id -un: also in batch jobs)
ROOT=${HSF2_ROOT:-$HSF2_EOS_HOME/my_acts/acts/Examples/Scripts/Optimization}
export HSF2_ROOT=$ROOT HSF_PHASE1_DIR=$ROOT HSF2_WORK=$ROOT/hsf2_work
export HSF2_DATA=$HSF2_WORK/data HSF2_MODELS=$HSF2_WORK/models HSF2_RUNS=$HSF2_WORK/runs
export PYTHONDONTWRITEBYTECODE=1
[ -f "$ROOT/hsf2_docs/DEPLOYED_REV" ] && export HSF2_CODE_REV=$(head -1 "$ROOT/hsf2_docs/DEPLOYED_REV")
mkdir -p "$HSF2_WORK/odd_export"
cd "$HSF2_WORK"
echo "host: $(hostname)   date: $(date '+%d-%m-%Y %H:%M:%S')   python: $(which python) ($(python -V 2>&1))"
echo "command: odd_export.py $*"
python -c "import acts" || { echo "ACTS python bindings not importable -- submit from the ACTS shell"; exit 3; }
exec python "$ROOT/hsf2_scripts/odd_export.py" "$@"

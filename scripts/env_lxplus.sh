# HSF Part II environment on lxplus -- SOURCE it (do not run it), preferably in a terminal used only for Part II:
#     source /eos/user/<u>/<user>/my_acts/acts/Examples/Scripts/Optimization/hsf2_scripts/env_lxplus.sh
# It leaves the ACTS / ParOpt_PyEnv environment of this shell (Part II has its own venv with the latest Qiskit,
# created once by hsf2_scripts/setup_hsf2_venv.sh) and changes into hsf2_work/, so relative paths can never land in
# the Phase-1 folder.  Part II code lives inside the Phase-1 Optimization folder, only under hsf2* names:
#     hsf2/ (package)  hsf2_condor/  hsf2_scripts/  hsf2_tests/  hsf2_docs/  hsf2_work/{data,models,runs,figures,env}
# HTCondor jobs do not need this file: hsf2_condor/run.sh sets up the same environment by itself.
export HSF2_EOS_HOME=${HSF2_EOS_HOME:-/eos/user/$(id -un | cut -c1)/$(id -un)}   # your EOS home (id -un: also in batch jobs)
export HSF2_ROOT=$HSF2_EOS_HOME/my_acts/acts/Examples/Scripts/Optimization
export HSF2_VENV=${HSF2_VENV:-$HSF2_EOS_HOME/my_acts/hsf2_venv}
if [ ! -x "$HSF2_VENV/bin/python" ]; then
  echo "no Part-II venv at $HSF2_VENV: run  bash $HSF2_ROOT/hsf2_scripts/setup_hsf2_venv.sh  once"; return 1
fi
type deactivate >/dev/null 2>&1 && deactivate               # leave ParOpt_PyEnv (Phase 1) if it is active
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH                 # nothing of the LCG / ACTS python environment
source "$HSF2_VENV/bin/activate"
export HSF_PHASE1_DIR=$HSF2_ROOT                            # acts_qaoa/ next to hsf2/
export HSF2_WORK=$HSF2_ROOT/hsf2_work
export HSF2_DATA=$HSF2_WORK/data
export HSF2_MODELS=$HSF2_WORK/models
export HSF2_RUNS=$HSF2_WORK/runs
export HSF2_FIGURES=$HSF2_WORK/figures
export HSF2_SCRIPTS=$HSF2_ROOT/hsf2_scripts
[ -f "$HSF2_ROOT/hsf2_docs/DEPLOYED_REV" ] && export HSF2_CODE_REV=$(head -1 "$HSF2_ROOT/hsf2_docs/DEPLOYED_REV")
export HSF2_QPU_LEDGER=${HSF2_QPU_LEDGER:-$HOME/.hsf2_part2_qpu_ledger.json}    # Part-II budget: $HSF2_QPU_BUDGET, default 400 s
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}               # one torch thread per HTCondor core
export PYTHONDONTWRITEBYTECODE=1                            # no new __pycache__ next to the Phase-1 code
mkdir -p "$HSF2_DATA" "$HSF2_MODELS" "$HSF2_RUNS" "$HSF2_FIGURES"
cd "$HSF2_WORK" || return
echo "Part II: venv $HSF2_VENV ($(python -V 2>&1))  work $HSF2_WORK  code ${HSF2_CODE_REV:-?}"
python -c "import qiskit, torch, hsf2, acts_qaoa; print('       qiskit', qiskit.__version__, 'torch', torch.__version__, 'hsf2', hsf2.__version__, 'acts_qaoa', acts_qaoa.__version__)"

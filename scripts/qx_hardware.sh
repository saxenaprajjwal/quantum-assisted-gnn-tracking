#!/bin/bash
# Part II QAOA on IBM hardware through the Phase-1 command (acts_qaoa stage_ibm) -- PREPARED, NOT RUN.
#
# The Phase-1 'ibm' command reads only problem.json + angles.json of a run folder, so it works on any Part-II
# closed-loop folder (runs/<stamp>_qx2_loop/seed<s>/<method>/ or runs/<landscape>/outputs/<stamp>_compare/...).
# Part II may use at most HSF2_QPU_BUDGET QPU seconds (default 400, the owner-approved total): this script keeps a
# SEPARATE ledger with that budget (one per IBM account, see below), in addition to the account-wide check that
# stage_ibm performs with the IBM service.
#
# usage:  bash hsf2_scripts/qx_hardware.sh status                             # account usage + calibration, 0 QPU s
#         bash hsf2_scripts/qx_hardware.sh <part-II run folder> dry-run      # transpile on a fake snapshot, 0 QPU s
#         bash hsf2_scripts/qx_hardware.sh <part-II run folder> dry-run-real # transpile on the REAL backend's
#                                     current target (IBM login, reads calibration; nothing submitted, 0 QPU s)
#         bash hsf2_scripts/qx_hardware.sh <part-II run folder> local-test   # Aer with the fake backend's noise
#         bash hsf2_scripts/qx_hardware.sh <part-II run folder> submit       # ONE budget-guarded job (needs approval)
# Backend: HSF2_IBM_BACKEND, default ibm_kingston -- the Open-Plan Heron r2 with the lowest median two-qubit error on
#          2 Nov 2026 (1.81e-3; ibm_fez 2.59e-3, ibm_marrakesh 2.62e-3).  'submit' shows today's calibration of every
#          backend of the account first, so the choice can be re-made on the day.
# Couplings kept: HSF2_TOP_K, default 24 (the Phase-1 hardware circuit); 0 = all couplings (full circuit).  The
#          sample tags carry it (..._top<K>_...), so jobs with different K never overwrite each other.
# Ledger: IBM's per-job usage can come back as 0 s (4 Nov 2026), so 'submit' reads the account usage before and
#          after the job and books the difference when the per-job figure is 0 or missing.
# Fake snapshot (dry-run, local-test): HSF2_FAKE_BACKEND, default the backend itself if this qiskit-ibm-runtime ships
#          its snapshot (Fake<Name>), else ibm_fez (FakeKingston is missing in some versions, e.g. 0.40.1).
# Environment: HSF_PHASE1_DIR = folder with DirectQAOAFeedback.py and acts_qaoa/ (env_lxplus.sh sets it),
#              IBM token in ~/ibm_creds.json (never in this folder), or another account's file in HSF2_IBM_CREDS
#              (only the path is passed on, as --ibm-creds; the ledger then is ~/.hsf2_part2_qpu_ledger_<file>.json),
#              HSF2_QPU_LEDGER = ledger file (wins), HSF2_QPU_BUDGET = budget in seconds (default 400).
set -euo pipefail
BACKEND=${HSF2_IBM_BACKEND:-ibm_kingston}
TOP_K=${HSF2_TOP_K:-24}
if [ "${1:-}" = "status" ]; then exec python -m hsf2 ibm-status ${HSF2_IBM_CREDS:+--creds "$HSF2_IBM_CREDS"}; fi
RUN_DIR=${1:?run folder (with problem.json and angles.json), or: status}
RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
RUNS=${HSF2_RUNS:-$(cd "$(dirname "$0")/.." && pwd -P)/hsf2_work/runs}
case "$RUN_DIR/" in
  "$(cd "$RUNS" 2>/dev/null && pwd -P)"/*) ;;
  *) echo "run folder must lie inside $RUNS (Part-II results only)"; exit 1 ;;
esac
MODE=${2:-dry-run}
HERE=$(cd "$(dirname "$0")/.." && pwd)                 # lxplus: the Optimization folder; PC: "HSF part II"
if [ -n "${HSF_PHASE1_DIR:-}" ]; then P1=$HSF_PHASE1_DIR
elif [ -f "$HERE/DirectQAOAFeedback.py" ]; then P1=$HERE
else P1=$(cd "$HERE/../HSF" && pwd); fi
[ -f "$P1/DirectQAOAFeedback.py" ] || { echo "Phase-1 DirectQAOAFeedback.py not found (set HSF_PHASE1_DIR)"; exit 1; }
export HSF_PHASE1_DIR=$P1                              # read by the book_usage helper below
# One ledger per IBM account: HSF2_QPU_LEDGER, else one per credentials file (HSF2_IBM_CREDS), else the shared default
# (same rule as hsf2.hw.config.default_ledger_path).  Budget: HSF2_QPU_BUDGET seconds per 28-day window, default 400.
if [ -n "${HSF2_QPU_LEDGER:-}" ]; then LEDGER=$HSF2_QPU_LEDGER
elif [ -n "${HSF2_IBM_CREDS:-}" ]; then
  CRED_BASE=$(basename "$HSF2_IBM_CREDS"); LEDGER=$HOME/.hsf2_part2_qpu_ledger_${CRED_BASE%.*}.json
else LEDGER=$HOME/.hsf2_part2_qpu_ledger.json; fi
BUDGET=${HSF2_QPU_BUDGET:-400}
COMMON=(--run-dir "$RUN_DIR" --ansatz standard --p 1 --sparsify-top-k "$TOP_K" --ledger "$LEDGER" --budget-seconds "$BUDGET"
        ${HSF2_IBM_CREDS:+--ibm-creds "$HSF2_IBM_CREDS"})
for f in problem.json angles.json; do
  [ -f "$RUN_DIR/$f" ] || { echo "missing $RUN_DIR/$f (run tune-loop / tune-compare first)"; exit 1; }
done
fake_backend() {   # the requested snapshot if this qiskit-ibm-runtime has it, else ibm_fez (stated, never silent)
  if [ -n "${HSF2_FAKE_BACKEND:-}" ]; then echo "$HSF2_FAKE_BACKEND"; return; fi
  python - "$BACKEND" <<'PY'
import sys
from qiskit_ibm_runtime import fake_provider as fp
name = sys.argv[1]
cls = "Fake" + name.replace("ibm_", "").capitalize()
if hasattr(fp, cls):
    print(name)
else:
    print(f"note: no {cls} snapshot in this qiskit-ibm-runtime; the offline modes use FakeFez", file=sys.stderr)
    print("ibm_fez")
PY
}
account_usage() {   # seconds used in the current window, from IBM (metadata only)
  python - <<'PY'
import os
from hsf2.tuning.phase1 import ibm_status
print((ibm_status(os.environ.get("HSF2_IBM_CREDS") or None).get("usage") or {}).get("usage_consumed_seconds", ""))
PY
}
book_usage() {      # $1 = usage before, $2 = usage after: book the difference if IBM's per-job figure is 0 or missing
  python - "$LEDGER" "$1" "$2" "$BUDGET" <<'PY'
import os, sys
sys.path.insert(0, os.environ["HSF_PHASE1_DIR"])
from acts_qaoa.ibm import QPULedger
led = QPULedger(sys.argv[1], budget_s=float(sys.argv[4]))
e = led.entries[-1]
try:
    before, after = float(sys.argv[2]), float(sys.argv[3])
except ValueError:
    sys.exit(f"account usage not available ({sys.argv[2]!r} -> {sys.argv[3]!r}): check the ledger by hand")
q = e.get("quantum_seconds")
if (q is None or q == 0) and after > before:
    led.set_usage(e["job_id"], after - before)
    print(f"ledger: {e['job_id']} booked {after - before:.0f} s (account {before:.0f} -> {after:.0f} s; "
          f"IBM's per-job figure was {q})")
else:
    print(f"ledger: {e['job_id']} {q} s (account {before:.0f} -> {after:.0f} s)")
print(led.table())
PY
}
case "$MODE" in
  dry-run)    FAKE=$(fake_backend)
              python "$P1/DirectQAOAFeedback.py" ibm "${COMMON[@]}" --dry-run --fake-backend "$FAKE" ;;
  dry-run-real)
    echo "Transpiling for the real $BACKEND (IBM login, calibration read; NOTHING is submitted, 0 QPU s)."
    python "$P1/DirectQAOAFeedback.py" ibm "${COMMON[@]}" --dry-run --ibm-backend "$BACKEND" ;;
  local-test) FAKE=$(fake_backend)
              python "$P1/DirectQAOAFeedback.py" ibm "${COMMON[@]}" --local-test --fake-backend "$FAKE" ;;
  submit)
    python -m hsf2 ibm-status ${HSF2_IBM_CREDS:+--creds "$HSF2_IBM_CREDS"} \
      || { echo "could not read the account status: not submitting"; exit 1; }
    echo "About to submit ONE job to $BACKEND for $RUN_DIR, top-$TOP_K couplings (0 = all)"
    echo "(Part-II ledger $LEDGER, budget $BUDGET s)."
    echo "Another backend: HSF2_IBM_BACKEND=<name> bash $0 $RUN_DIR submit"
    read -r -p "Type 'submit' to continue: " ok
    [ "$ok" = "submit" ] || { echo "aborted"; exit 1; }
    before=$(account_usage)
    python "$P1/DirectQAOAFeedback.py" ibm "${COMMON[@]}" --ibm-backend "$BACKEND"
    sleep 20                                            # IBM's usage counter lags the job by a few seconds
    book_usage "$before" "$(account_usage)" ;;
  *) echo "mode must be dry-run, dry-run-real, local-test or submit"; exit 1 ;;
esac

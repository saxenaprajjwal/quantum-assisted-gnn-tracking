#!/bin/bash
# HSF Part II, step 0: measure what this lxplus account can do (CPU/GPU, Python stack, ACTS writers,
# disk, network to Zenodo, HTCondor GPU access).  It only reads; it writes ONE report file
#     hsf2_work/discover_<DD-MM-YYYY-HH-MM>.txt   (never into the Phase-1 folder; paste it back or scp it).
#
# Run once from the ACTS shell (where  python -c "import acts"  works), e.g.
#     cd /eos/user/<u>/<user>/my_acts/acts/Examples/Scripts/Optimization      # activates ParOpt_PyEnv
#     bash hsf2_scripts/discover_lxplus.sh      (from the Optimization folder)

export HSF2_EOS_HOME=${HSF2_EOS_HOME:-/eos/user/$(id -un | cut -c1)/$(id -un)}   # your EOS home (id -un: also in batch jobs)
STAMP=$(date +%d-%m-%Y-%H-%M)
WORK=${HSF2_WORK:-$(cd "$(dirname "$0")/.." && pwd -P)/hsf2_work}
mkdir -p "$WORK"
OUT="$WORK/discover_${STAMP}.txt"
PROBE=$(mktemp /tmp/hsf2_probe_XXXXXX.py)
trap 'rm -f "$PROBE"' EXIT

sec() { printf '\n===== %s =====\n' "$*"; }
run() { echo "\$ $1"; timeout "${2:-60}" bash -c "$1" 2>&1 | head -n "${3:-40}"; }

cat > "$PROBE" <<'EOF'
import importlib, platform, sys, time
print("python", platform.python_version(), sys.executable)
for name in ["numpy", "scipy", "pandas", "pyarrow", "matplotlib", "sklearn", "torch", "torch_geometric",
             "torch_scatter", "networkx", "numba", "qiskit", "qiskit_aer", "qiskit_ibm_runtime", "optuna",
             "uproot", "awkward", "onnxruntime", "trackml", "pytest", "acts"]:
    t0 = time.time()
    try:
        m = importlib.import_module(name)
        print(f"  {name:20s} {getattr(m, '__version__', '?'):12s} ({time.time() - t0:.1f} s)")
    except Exception as exc:
        print(f"  {name:20s} MISSING ({type(exc).__name__}: {str(exc)[:60]})")
try:
    import torch
    print("torch: cuda", torch.cuda.is_available(), "| threads", torch.get_num_threads())
    x = torch.randn(2000, 2000); t0 = time.time(); (x @ x).sum().item()
    print(f"torch: 2000x2000 matmul {time.time() - t0:.3f} s")
except Exception as exc:
    print("torch test skipped:", exc)
try:
    import acts, acts.examples as ex
    keys = ("Csv", "Gnn", "ExaTrkX", "Torch", "Onnx", "Graph", "ModuleMap", "TrackFinding", "Root")
    names = sorted(n for n in dir(ex) if any(k in n for k in keys))
    print("acts", getattr(acts, "__version__", "?"), "| acts.examples:", " ".join(names))
    try:
        import acts.examples.reconstruction as rec
        print("acts.examples.reconstruction:", " ".join(n for n in dir(rec) if n.startswith("add")))
    except Exception as exc:
        print("acts.examples.reconstruction not importable:", exc)
except Exception as exc:
    print("acts import failed:", exc)
EOF

{
echo "HSF Part II discovery report  $STAMP"
sec "host"
run 'hostname; whoami; uname -r; grep PRETTY /etc/os-release; echo "cores: $(nproc)"; free -g | head -2'
sec "python stack of THIS shell"
run 'which python python3; echo "VIRTUAL_ENV=$VIRTUAL_ENV"; python -V'
run "python $PROBE" 240 80
sec "pip in this venv"
run 'python -m pip --version; python -m pip list 2>/dev/null | grep -i -E "^(torch|numpy|scipy|pandas|qiskit|optuna|scikit|matplotlib|pyarrow|uproot) "' 60 30
sec "GPU"
run 'command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "no nvidia-smi here (normal on lxplus login nodes)"'
run 'condor_status -gpus -compact 2>&1 | head -25' 90 30
run 'command -v haggis >/dev/null && haggis rights || echo "haggis not available"' 60 40
sec "LCG views (alternative Python stacks)"
run 'ls /cvmfs/sft.cern.ch/lcg/views/ | grep -E "^LCG_10[5-9]" | tr "\n" " "; echo'
for V in LCG_107/x86_64-el9-gcc13-opt LCG_107_cuda/x86_64-el9-gcc11-opt LCG_106/x86_64-el9-gcc13-opt; do
  S=/cvmfs/sft.cern.ch/lcg/views/$V/setup.sh
  if [ -f "$S" ]; then
    run "env -i HOME=\$HOME PATH=/usr/bin:/bin bash --noprofile --norc -c 'source $S >/dev/null 2>&1; python -c \"import sys,torch,numpy,scipy,pandas;print(\\\"$V\\\",sys.version.split()[0],\\\"torch\\\",torch.__version__,\\\"cuda\\\",torch.cuda.is_available())\"'" 180 5
  else
    echo "$V: not found"
  fi
done
sec "storage"
run 'df -h ~ 2>&1 | tail -1; command -v fs >/dev/null && fs lq ~'
run 'command -v eos >/dev/null && eos quota $HSF2_EOS_HOME 2>&1 | head -12 || df -h $HSF2_EOS_HOME 2>&1 | tail -1' 60 15
run 'ls -d $HSF2_EOS_HOME/my_acts/* 2>&1'
sec "network: Zenodo (TrackML record 4730167)"
run 'curl -s -L -m 60 -o /dev/null -w "detectors.csv: HTTP %{http_code}, %{size_download} bytes (expect 1942066), %{speed_download} B/s\n" https://zenodo.org/api/records/4730167/files/detectors.csv/content' 90
sec "HTCondor"
run 'condor_version | head -1; condor_q 2>&1 | tail -4' 90
sec "Phase 1 and ACTS examples on EOS"
run 'ls $HSF2_EOS_HOME/my_acts/acts/Examples/Scripts/Optimization/acts_qaoa | head -20; grep -m1 __version__ $HSF2_EOS_HOME/my_acts/acts/Examples/Scripts/Optimization/acts_qaoa/__init__.py'
run 'ls $HSF2_EOS_HOME/my_acts/acts/Examples/Scripts/Python | grep -i -E "gnn|exatrkx|csv|odd" ; cd $HSF2_EOS_HOME/my_acts/acts && git log -1 --format="ACTS checkout: %h %cd %s" 2>/dev/null'
echo
echo "report written to $PWD/$OUT"
} 2>&1 | tee "$OUT"

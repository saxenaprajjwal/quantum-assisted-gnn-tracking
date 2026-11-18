# quantum-assisted-gnn-tracking

Code of HSF Part II: the Python package `hsf2` for the second phase of the HSF project (Prajjwal Saxena, IISc):

* a graph-neural-network (GNN) track reconstruction in the style of Exa.TrkX / GNN4ITk, on TrackML events and on
  OpenDataDetector events exported from an ACTS chain, with a GNN-vs-CKF comparison on the same events;
* studies of quantum optimisation (QAOA, Qiskit) for small sub-problems of it — tuning the inference-time parameters
  through a surrogate → QUBO → QAOA loop, track-candidate and triplet selection, linear-system tracking — always
  compared with exact enumeration and classical heuristics;
* a runner for IBM Quantum hardware with error suppression, read-out correction, zero-noise extrapolation and a QPU
  budget ledger.

This repository holds **code only**. The documents (guide, report, preprint), the data, the trained models and all
results are kept outside it. No quantum advantage is assumed or claimed by the code or its studies.

Version 0.2.21 (17 November 2026).

## Requirements

* Python 3.10 or newer (developed on 3.10 on Windows, run on 3.11 on CERN lxplus).
* The packages in `requirements.txt`; PyTorch CPU is installed separately:

  ```bash
  python -m venv .venv
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu      # Windows: .venv\Scripts\pip
  .venv/bin/pip install -r requirements.txt
  ```
* **Phase 1** (`acts_qaoa`, tested with v2.0.23) for the tuning, QUBO/QAOA and hardware parts. It is not part of this
  repository; `hsf2/_phase1.py` finds it through `$HSF_PHASE1_DIR`, a folder `../HSF` next to this one, or the lxplus
  layout. The GNN chain (data, graph, model, tracking) works without it.
* Optional: `mthree` for read-out mitigation (the code and the tests fall back without it).

## Quick start

```bash
python -m hsf2 selftest                          # synthetic detector, no data needed
python -m pytest tests/test_core.py tests/test_metrics.py tests/test_odd_compare.py -q    # quick subset
python -m pytest tests -q                        # all 312 tests, about 30 min (Aer simulations)
```

Tests that need Phase 1 skip when `acts_qaoa` is not found; tests that compare with stored hardware results skip
when those results are not present (they are not in this repository).

## Layout

```
hsf2/            Python package  (python -m hsf2 <command>)
  data/          EventData; TrackML reader (local zip or Zenodo range requests); synthetic layered detector;
                 OpenDataDetector reader (simhit or measured positions; surface frames fitted on training events)
  graph/         loose "module-map-lite" graph + inference-time edge filters
  model/         Interaction Network edge classifier (pure PyTorch), training, quantum edge classifier (QX5)
  tracking/      track building (hysteresis, junction removal, walk-through), matching, metrics, cache,
                 GNN vs CKF comparison (same matching, optional same track cuts on a helix refit)
  tuning/        the tuning problem: 24-/16-bit spaces, grid evaluator, runner with the Phase-1 contract,
                 Phase-1 integration and closed loop, exhaustive landscape (QX1), optimiser comparison, baselines,
                 scaling study 16-32 qubits (QX6)
  qubo/          candidate-selection QUBO (QX3a), Bapst-style triplet QUBO (QX3b), linear-system / HHL / 1-bit
                 filter emulation (QX4)
  hw/            IBM hardware runner: suppression settings, QPU ledger and budget check, job manifests, read-out
                 calibration, zero-noise extrapolation by gate folding, tracking pieces, QX4 filter, noise analysis
  report.py      figures
  _phase1.py     locates the Phase-1 package acts_qaoa
scripts/         lxplus helpers: venv setup, environment, deploy, bundle, hardware wrapper, discovery,
                 TrackML download, OpenDataDetector export from the Phase-1 ACTS chain
condor/          HTCondor submit files and DAGs
tests/           pytest tests (synthetic events; no data needed)
```

## Commands

```
selftest  synth  trackml-prep  train  evaluate  scan  info            GNN chain
compare-evals  odd-prep  compare-ckf                                  hit levels (S0 / S0N / S1); GNN vs CKF on the ODD
tune-landscape  tune-compare  tune-loop  tune-objectives              QX1, QX2
select  triplet                                                       QX3
hhl  qgnn                                                             QX4, QX5 (emulation)
tune-scaling                                                          QX6 (16-32 qubits, Aer)
ibm-status  hw-ledger                                                 IBM account usage + calibration, QPU ledger (no QPU time)
hw-run  hw-ladder  hw-readout  hw-components  hw-zne  hw-qx4          hardware jobs (--mode dry / local / submit)
hw-retrieve                                                           rebuild the outputs of a submitted job from its id
hw-validate  hw-analyze  hw-components-analyze  hw-zne-analyze  hw-qx4-analyze     analysis of hardware samples
report                                                                figures
```
`python -m hsf2 <command> --help` lists the options. Outputs go to the work folder `$HSF2_WORK` (default: the
repository folder) under `data/`, `models/`, `runs/`, which git ignores.

## Data (not included)

* **TrackML**: Zenodo record 4730167 (DOI 10.5281/zenodo.4730167, CC-BY-4.0); `trackml-prep --zip zenodo:train_1.zip`
  reads the events by HTTP range requests, without the 16 GB download.
* **OpenDataDetector**: events of the Phase-1 ACTS chain, exported with `scripts/odd_export.py` (simhits,
  measurements, particles, CKF tracks) and read with `odd-prep`.

## CERN lxplus / HTCondor

`python scripts/make_bundle.py` builds a bundle that `scripts/deploy_lxplus.sh` installs into the Phase-1 folder
under `hsf2*` names only, without changing any Phase-1 file. Part II uses its own virtual environment
(`scripts/setup_hsf2_venv.sh`). The scripts default to folders under your own EOS home
(`/eos/user/<u>/<user>/my_acts/...`, from `id -un`); set `HSF2_EOS_HOME`, `HSF2_ROOT`, `HSF2_VENV`, `HSF2_DATA` and
`HSF_PHASE1_DIR` to use other locations.

## IBM Quantum hardware

Every hardware command has `--mode dry` (build and transpile only), `--mode local` (simulator) and `--mode submit`.
Submitting asks for a typed confirmation per job and checks the QPU budget in a local ledger. Credentials are never
stored here: pass the path of your credentials file with `--creds`; the code records only the file's name.

## Conventions

* Every command writes to a new folder named `DD-MM-YYYY-HH-MM_<label>`; results are never overwritten, and the
  configuration, command line and code version are saved next to every result.
* No credentials, data or results in the repository (`.gitignore`).

## Licence

MIT; see `LICENSE`. The TrackML data (CC-BY-4.0) and the Phase-1 package are not part of this repository.

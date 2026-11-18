"""Real-objective check of hardware samples (QX2 on IBM hardware).

The Phase-1 'ibm' command scores a hardware sample set on the exhaustive landscape of the QUBO.  This module takes
the next step that Phase 1 takes with ACTS (stage_validate): the best distinct states of each sample set by QUBO
energy are evaluated on the real GNN objective, and the best of each is confirmed on the held-out splits.

Every hardware sample set gets two classical controls with the same number of shots, saved next to it in the run
folder's samples/ (Phase-1 format, same problem fingerprint):
  uniform-<shots>-s<seed>       uniform random bit strings (what the hardware would give with F = 0)
  ideal-top<K>-<shots>-s<seed>  shots of the noiseless circuit that was run (same sparsified Hamiltonian and angles)
so the comparison hardware vs ideal vs uniform is at equal sample size.  The exact top states of the QUBO
(Phase-1 'exact') are evaluated too.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..utils import save_json
from .phase1 import P1

log = logging.getLogger(__name__)


def hardware_tags(run) -> List[str]:
    return sorted(t for t in run.samples() if t.startswith("ibm-") and not t.endswith("_control"))


def add_controls(run, hw_tag: str, seed: int = 0) -> List[str]:
    """Uniform and ideal-circuit sample sets with the shots of `hw_tag` (written once; deterministic in `seed`)."""
    from acts_qaoa.landscape import counts_from_probs, uniform_counts
    from acts_qaoa.qaoa_math import StatevectorQAOA
    samples = run.samples()
    hw = samples[hw_tag]
    meta = hw["meta"]
    shots = int(meta["shots"])
    top_k = int(hw_tag.split("_top")[1].split("_")[0]) if "_top" in hw_tag else None
    fit, qubo, ising, prob = run.load_problem()
    land = run.landscape(qubo)
    scale = float(prob["scale"])
    tags = []
    u_tag = f"uniform-{shots}-s{seed}"
    if u_tag not in samples:
        counts = uniform_counts(run.enc.n_qubits, shots, seed)
        run.save_samples(u_tag, counts, {"kind": "control", "shots": shots, "seed": seed}, land.sample_metrics(counts))
    tags.append(u_tag)
    i_tag = f"ideal-top{top_k or 'all'}-{shots}-s{seed}"
    if i_tag not in samples:
        ism = ising.sparsified(top_k) if top_k else ising
        hd = ((P1._sparse_energies(run, ism) if top_k else land.E) - ism.offset) / scale
        sv = StatevectorQAOA(hd, mixer="x", dtype=np.complex64 if run.enc.n_qubits >= 20 else np.complex128)
        p = np.asarray(sv.probs(meta["gammas"], meta["betas"]), dtype=np.float64)
        counts = counts_from_probs(p, shots, seed)
        m = land.sample_metrics(counts)
        m["exact_mean_energy"] = float(p @ land.E / p.sum())
        run.save_samples(i_tag, counts, {"kind": "ideal", "shots": shots, "seed": seed, "gammas": meta["gammas"],
                                         "betas": meta["betas"], "sparsify_top_k": top_k}, m)
    tags.append(i_tag)
    return tags


def validate_hardware(run, out_dir: Path, top_k: int = 10, min_distance: int = 2, seed: int = 0,
                      confirm_stride: Optional[int] = None, reference: Optional[Dict[str, float]] = None) -> dict:
    """Phase-1 stage_validate on the hardware sample sets and their controls, then the best pick of each method
    (and `reference`, normally the default configuration) on every confirm<k> split of the runner."""
    out_dir = Path(out_dir)
    hw = hardware_tags(run)
    if not hw:
        raise ValueError(f"no hardware sample set (samples/ibm-*.json) for the current problem in {run.dir}")
    methods = list(hw)
    for t in hw:
        methods += add_controls(run, t, seed)
    P1.stage_validate(run, top_k=top_k, methods=methods, include_exact=True, min_distance=min_distance)
    from ..utils import load_json
    vsum = load_json(run.p("validation_summary.json"))
    samples = run.samples()
    runner = run.runner
    picks = {m: d["best_params"] for m, d in vsum["methods"].items() if d.get("best_params")}
    if reference is not None:
        picks["default"] = dict(reference)
    splits = []
    if confirm_stride and hasattr(runner, "splits"):
        splits = sorted(int(k[len("confirm"):]) for k in runner.splits if k.startswith("confirm") and runner.splits[k])
    rows = {}
    for name, prm in picks.items():
        tune = runner.evaluate(prm, tag="hwv-pick")
        row = {"params": prm, "tune_objective": tune.get("objective"), "tune_metrics": _keep(tune.get("metrics")),
               "confirm": {}}
        for k in splits:
            rec = runner.evaluate(prm, tag="hwv-confirm", seed=k * confirm_stride)
            row["confirm"][f"confirm{k}"] = {"objective": rec.get("objective"), **_keep(rec.get("metrics"))}
        vals = [v["objective"] for v in row["confirm"].values() if v["objective"] is not None]
        row["confirm_mean"] = float(np.mean(vals)) if vals else None
        if name in vsum["methods"]:
            d = vsum["methods"][name]
            row.update({"best_of_k": d["best_objective"], "mean_of_k": d["mean_objective"],
                        "overlap_with_exact_topk": d["overlap_with_exact_topk"]})
            if name in samples:
                row["sample_metrics"] = samples[name]["metrics"]
        rows[name] = row
        log.info("[hw-validate] %-46s tune %s  confirm %s", name, _f(row["tune_objective"]), _f(row["confirm_mean"]))
    res = {"run_dir": str(run.dir), "top_k": top_k, "min_distance": min_distance, "seed": seed,
           "hardware_tags": hw, "methods": methods, "confirm_splits": [f"confirm{k}" for k in splits],
           "score": runner.score_cfg.describe() if hasattr(runner, "score_cfg") else None, "rows": rows}
    save_json(out_dir / "hw_validation.json", res)
    return res


def _keep(m: Optional[dict]) -> dict:
    keys = ("efficiency", "efficiency_dm", "fakerate", "duplicaterate", "time_proxy", "trackml_score")
    return {k: (m or {}).get(k) for k in keys}


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.5f}"

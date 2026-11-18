"""Controlled comparison of optimisers on the same objective with the same budget (QX1 on the exact landscape;
QX2 on the real GNN objective uses the same driver with a GNNRunner).

Per seed s, every method gets a budget of B distinct real-objective evaluations:
  qubo-<sampler>  Phase-1 Latin-hypercube warm start (lhs_seed = 1000 + s, shared by all samplers: a paired design),
                  then hsf2.tuning.phase1.closed_loop with that sampler: exact | sa | qaoa (ideal, p=1) |
                  qaoa-noise (white-noise model with the measured ibm_fez fidelity at the circuit's transpiled
                  two-qubit gate count) | qaoa-noise-top24 (the circuit with the 24 largest couplings, as run on
                  hardware in Phase 1) | random (uniform proposals ranked by the surrogate: control)
  random-grid, tpe-grid, sa-grid   black-box baselines (hsf2.tuning.baselines)
The result of a method is the configuration with the best MEASURED objective among its B evaluations; on the
exact landscape its true score and regret (true optimum - found) are exact.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..utils import save_json
from .baselines import BASELINES

log = logging.getLogger(__name__)
QUBO_SAMPLERS = ("exact", "sa", "qaoa", "qaoa-noise", "qaoa-noise-top24", "random")


def run_method(method: str, seed: int, base_runner, make_run_for: Callable[[str, object], object], enc, budget: int,
               n_ws: int, top_k: int, noise: Optional[dict], shots: int = 4096, min_distance: int = 2) -> dict:
    """One method, one seed; returns its trajectory (records in order of first evaluation)."""
    from acts_qaoa import pipeline as P1

    from .phase1 import TaggedView, closed_loop
    view = TaggedView(base_runner)
    t0 = time.time()
    detail: dict = {}
    if method.startswith("qubo-"):
        sampler = method[len("qubo-"):]
        run = make_run_for(method, view)
        P1.stage_warmstart(run, n_lhs=n_ws, lhs_seed=1000 + seed, noise_repeats=0)
        nz = None
        if sampler.startswith("qaoa-noise"):
            nz = (noise or {}).get("top24" if sampler.endswith("top24") else "full")
        detail = closed_loop(run, sampler, top_k=top_k, budget=budget, min_distance=min_distance, shots=shots,
                             seed=seed, noise=nz, tag=sampler)
        # keep per evaluation only what the surrogate-vs-measurement analysis needs
        detail["evaluations"] = [{k: e[k] for k in ("round", "index", "predicted", "objective", "qubo_energy",
                                                    "global_qubo_rank")} for e in detail.get("evaluations", [])]
    elif method in BASELINES:
        BASELINES[method](view, enc, budget, seed, method)
    else:
        raise ValueError(f"unknown method {method}")
    recs = view.records()[:budget]
    return {"method": method, "seed": seed, "budget": budget, "n_distinct": len(recs), "seconds": round(time.time() - t0, 1),
            "records": [{"index": int(enc.index_from_params(r["params"])), "tag": r["tag"], "objective": r["objective"],
                         "params": r["params"]} for r in recs], "detail": detail}


def summarize_runs(results: List[dict], truth: Callable[[int], float], best_true: float) -> Dict[str, dict]:
    """Per method: distribution over seeds of the true score of the pick and of the regret, P(optimum found)."""
    by: Dict[str, List[dict]] = {}
    for r in results:
        by.setdefault(r["method"], []).append(r)
    out = {}
    for m, rs in by.items():
        picks, regrets, found, curves = [], [], [], []
        for r in rs:
            recs = [x for x in r["records"] if x["objective"] is not None]
            if not recs:
                continue
            best = max(recs, key=lambda x: x["objective"])
            t = truth(best["index"])
            picks.append(t)
            regrets.append(best_true - t)
            found.append(any(abs(truth(x["index"]) - best_true) < 1e-12 for x in recs))
            cur, b = [], -np.inf
            for x in r["records"]:
                if x["objective"] is not None and x["objective"] > b:
                    b = x["objective"]
                    bt = truth(x["index"])
                cur.append(best_true - bt if np.isfinite(b) else np.nan)
            curves.append(cur)
        L = max(len(c) for c in curves) if curves else 0
        curv = np.array([c + [c[-1]] * (L - len(c)) for c in curves]) if curves else np.zeros((0, 0))
        out[m] = {"n_seeds": len(picks), "true_score_of_pick": _dist(picks), "regret": _dist(regrets),
                  "p_found_optimum": float(np.mean(found)) if found else None,
                  "regret_curve_median": np.nanmedian(curv, 0).tolist() if L else [],
                  "regret_curve_q25": np.nanquantile(curv, 0.25, 0).tolist() if L else [],
                  "regret_curve_q75": np.nanquantile(curv, 0.75, 0).tolist() if L else []}
    return out


def _dist(x: Sequence[float]) -> dict:
    x = np.asarray(x, float)
    if not len(x):
        return {}
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)), "min": float(x.min()),
            "max": float(x.max())}


def summarize_measured(results: List[dict]) -> Dict[str, dict]:
    """Per method, over seeds: the best MEASURED objective after B evaluations and the median best-so-far curve
    (for QX2, where the true optimum is unknown)."""
    by: Dict[str, List[dict]] = {}
    for r in results:
        by.setdefault(r["method"], []).append(r)
    out = {}
    for m, rs in by.items():
        best, curves = [], []
        for r in rs:
            cur, b = [], -np.inf
            for x in r["records"]:
                if x["objective"] is not None:
                    b = max(b, x["objective"])
                cur.append(b if np.isfinite(b) else np.nan)
            if cur:
                curves.append(cur)
                best.append(cur[-1])
        L = max((len(c) for c in curves), default=0)
        curv = np.array([c + [c[-1]] * (L - len(c)) for c in curves]) if curves else np.zeros((0, 0))
        out[m] = {"n_seeds": len(best), "best_measured": _dist(best),
                  "best_so_far_median": np.nanmedian(curv, 0).tolist() if L else []}
    return out


def paired_differences(results: List[dict], truth: Callable[[int], float], ref: str) -> Dict[str, dict]:
    """Per method, the paired (same seed) difference of the true score of the pick against method `ref`."""
    pick: Dict[str, Dict[int, float]] = {}
    for r in results:
        recs = [x for x in r["records"] if x["objective"] is not None]
        if recs:
            pick.setdefault(r["method"], {})[r["seed"]] = truth(max(recs, key=lambda x: x["objective"])["index"])
    out = {}
    for m, d in pick.items():
        if m == ref or ref not in pick:
            continue
        common = sorted(set(d) & set(pick[ref]))
        diff = np.array([d[s] - pick[ref][s] for s in common])
        if len(diff):
            out[m] = {"vs": ref, "n": len(diff), "mean": float(diff.mean()),
                      "sem": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else None,
                      "wins": int(np.sum(diff > 1e-12)), "ties": int(np.sum(np.abs(diff) <= 1e-12)),
                      "losses": int(np.sum(diff < -1e-12))}
    return out


def noise_model_for(run, n_ws: int, seed: int = 0) -> dict:
    """Measured-fidelity noise model for this problem size, from a Phase-1 Run (normally a TaggedView runner).

    The surrogate fitted on the warm start of `seed` gives a representative QUBO; its p=1 circuit is transpiled
    (fake ibm_fez, no hardware access) in full and with its 24 largest couplings, and F is read off the measured
    Phase-1 fidelity curve (hsf2.tuning.phase1.fidelity_from_2q).
    """
    from acts_qaoa import pipeline as P1

    from .phase1 import MEASURED_FIDELITY, transpiled_2q
    P1.stage_warmstart(run, n_lhs=n_ws, lhs_seed=1000 + seed, noise_repeats=0)
    P1.stage_build(run, surrogate="quadratic", fit_tags=("ws",), ansatz=("standard",), p_max=1)
    _, _, ising, prob = run.load_problem()
    full = transpiled_2q(ising, float(prob["scale"]))
    top = transpiled_2q(ising, float(prob["scale"]), top_k=24)
    return {"full": {"fidelity": full["fidelity_model"], **full}, "top24": {"fidelity": top["fidelity_model"], **top},
            "measured_points": [list(p) for p in MEASURED_FIDELITY],
            "source": "F(n_2q): log-linear interpolation of the Phase-1 ibm_fez measurements "
                      "[HSF/docs/QAOA_ACTS_Guide.tex]; white-noise model p = F p_ideal + (1 - F)/2^n"}


def run_comparison(out_dir: Path, base_runner, enc, reference: dict, n_events: int, seeds: Sequence[int],
                   methods: Sequence[str], budget: int, n_ws: int, top_k: int, shots: int = 4096,
                   min_distance: int = 2, noise: Optional[dict] = None, truth: Optional[Callable[[int], float]] = None,
                   best_true: Optional[float] = None, extra: Optional[dict] = None, score=None) -> dict:
    """All methods x seeds; results.json (trajectories) and comparison.json (summary) in out_dir.  `score`
    (ScoreConfig, default S) must be the one base_runner scores with; it is recorded in every Phase-1 run folder."""
    from ..config import ScoreConfig
    from .phase1 import make_run
    score = score or ScoreConfig()
    if getattr(base_runner, "score_cfg", score) != score:
        raise ValueError("run_comparison: `score` differs from the runner's score")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []
    t0 = time.time()
    for s in seeds:
        for m in methods:
            def mk(method, view, s=s):
                return make_run(out_dir / f"seed{s}" / method, enc.space, enc.bits, view, n_events, reference,
                                score=score)
            r = run_method(m, s, base_runner, mk, enc, budget, n_ws, top_k, noise, shots, min_distance)
            if truth is not None:
                recs = [x for x in r["records"] if x["objective"] is not None]
                r["true_score_of_pick"] = truth(max(recs, key=lambda x: x["objective"])["index"]) if recs else None
            results.append(r)
            log.info("[compare] seed %s %-22s best measured %.5f  (%d distinct, %.0f s)", s, m,
                     max((x["objective"] for x in r["records"] if x["objective"] is not None), default=float("nan")),
                     r["n_distinct"], r["seconds"])
            save_json(out_dir / "results.json", {"results": results}, keep_history=False)
    summary = {"budget": budget, "n_ws": n_ws, "top_k": top_k, "shots": shots, "min_distance": min_distance,
               "seeds": list(seeds), "methods": list(methods), "noise_model": noise, "seconds": round(time.time() - t0, 1),
               "score_definition": score.describe(), "score_config": score.to_dict(), **(extra or {})}
    summary["methods_measured"] = summarize_measured(results)
    if truth is not None:
        summary["best_true"] = best_true
        summary["methods_summary"] = summarize_runs(results, truth, best_true)
        summary["paired_vs_random_grid"] = paired_differences(results, truth, "random-grid")
        summary["paired_vs_qubo_exact"] = paired_differences(results, truth, "qubo-exact")
    save_json(out_dir / "comparison.json", summary)
    return summary

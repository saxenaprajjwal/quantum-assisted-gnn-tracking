"""QX3: selection of the final tracks among overlapping GNN track candidates as a QUBO.

Variables  x_i = 1 if candidate i (hsf2.tracking.build.build_candidates, steps 1-5) is kept.
Weights    w_i = n_hits_i * mean edge score_i  (> 0: the hits a candidate explains, weighted by the GNN's confidence)
Conflicts  (i, j) share at least one hit (hsf2.tracking.build.conflict_pairs).
QUBO       E(x) = - sum_i w_i x_i + P sum_{(i,j) conflict} x_i x_j,   P = penalty_factor * max_i w_i.

Penalty regime: if x selects a conflicting pair, removing its lighter member j changes E by
w_j - P * (#selected conflict partners of j) <= w_j - P < 0 for P > max w, so every minimum of E is
conflict-free and, among conflict-free x, E = -sum w x: the minimum is a MAXIMUM-WEIGHT INDEPENDENT SET of the
conflict graph (no hit is used twice).  penalty_factor > 1 is required (checked).

The conflict graph decomposes into connected components; each is an independent QUBO.  Components with <= max_vars
variables (default 24, the Phase-1 qubit count) are solved with exact enumeration, Phase-1 simulated annealing and
QAOA (Phase-1 closed-form p=1 angles + exact statevector + shots, ideal and with the measured-fidelity white-noise
model); larger components fall back to the greedy ambiguity resolution of build.py, and their fraction is reported.
The greedy resolution (merge_overlap = 0) is the classical reference.

This is a candidate-conflict formulation (one variable per complete track candidate), not Bapst et al.'s triplet
formulation (one variable per hit triplet; hsf2.qubo.triplet).  Small conflict components say nothing about
full-event quantum tracking at scale.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..tracking.build import Candidates, conflict_pairs, resolve_ambiguity


@dataclass
class SelectionProblem:
    n: int
    weights: np.ndarray
    ci: np.ndarray                    # conflict pairs (ci < cj)
    cj: np.ndarray
    components: List[np.ndarray] = field(default_factory=list)

    @property
    def n_conflicts(self) -> int:
        return len(self.ci)


def candidate_weights(cands: Candidates) -> np.ndarray:
    lens = np.array([len(t) for t in cands.tracks], dtype=float)
    return lens * np.asarray(cands.mean_score, float)


def build_problem(cands: Candidates, n_nodes: int) -> SelectionProblem:
    n = len(cands.tracks)
    w = candidate_weights(cands)
    ci, cj, _ = conflict_pairs(cands.tracks, n_nodes)
    if n == 0:
        return SelectionProblem(0, w, ci, cj, [])
    adj = coo_matrix((np.ones(len(ci)), (ci, cj)), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    order = np.argsort(lab, kind="stable")
    uk, start = np.unique(lab[order], return_index=True)
    comps = np.split(order, start[1:])
    return SelectionProblem(n, w, ci, cj, comps)


def component_qubo(prob: SelectionProblem, comp: np.ndarray, penalty_factor: float = 1.5):
    """Phase-1 QUBO container for one component (local variable k = candidate comp[k])."""
    from .._phase1 import ensure_phase1
    ensure_phase1()
    from acts_qaoa.surrogate import QUBO
    if not penalty_factor > 1.0:
        raise ValueError("penalty_factor must be > 1 (P > max w makes every minimum conflict-free)")
    m = len(comp)
    pos = {int(c): k for k, c in enumerate(comp)}
    w = prob.weights[comp]
    P = penalty_factor * float(w.max()) if m else 0.0
    W = np.zeros((m, m))
    for a, b in zip(prob.ci, prob.cj):
        if int(a) in pos and int(b) in pos:
            i, j = sorted((pos[int(a)], pos[int(b)]))
            W[i, j] = P
    return QUBO(n=m, const=0.0, L=-w.astype(float), W=W), P


def is_independent(sel_local: np.ndarray, qubo) -> bool:
    x = np.asarray(sel_local, float)
    return float(x @ qubo.W @ x) == 0.0


def bits_of(idx: int, n: int) -> np.ndarray:
    return ((int(idx) >> np.arange(n)) & 1).astype(np.int8)


def solve_exact(qubo) -> Tuple[int, float]:
    from acts_qaoa.landscape import enumerate_energies
    E = enumerate_energies(qubo)
    i = int(np.argmin(E))
    return i, float(E[i])


def solve_sa(qubo, restarts: int = 256, seed: int = 0) -> Tuple[int, float]:
    from acts_qaoa.landscape import simulated_annealing
    counts = simulated_annealing(qubo, restarts=restarts, seed=seed)
    idx = np.fromiter(counts, dtype=np.int64)
    e = qubo.energy_bits(((idx[:, None] >> np.arange(qubo.n)) & 1).astype(float))
    k = int(np.argmin(e))
    return int(idx[k]), float(e[k])


def solve_random(qubo, shots: int = 1024, seed: int = 0) -> Tuple[int, float]:
    """Control: best of `shots` uniformly random states (what best-of-shots gives without any quantum dynamics)."""
    from acts_qaoa.landscape import uniform_counts
    idx = np.fromiter(uniform_counts(qubo.n, shots, seed), dtype=np.int64)
    e = qubo.energy_bits(((idx[:, None] >> np.arange(qubo.n)) & 1).astype(float))
    k = int(np.argmin(e))
    return int(idx[k]), float(e[k])


def solve_qaoa(qubo, shots: int = 1024, seed: int = 0, fidelity: Optional[float] = None) -> Tuple[int, float, dict]:
    """Best-of-shots of p=1 QAOA (Phase-1 closed-form angles, exact statevector; white-noise model if fidelity)."""
    from acts_qaoa.landscape import counts_from_probs, enumerate_energies
    from acts_qaoa.qaoa_math import StatevectorQAOA, optimize_p1
    E = enumerate_energies(qubo)
    ising = qubo.to_ising()
    scale = ising.scale() if qubo.n > 0 else 1.0
    if not scale > 0:                           # no fields and no couplings: every state has the same energy
        return 0, float(E[0]), {"p_ground": 1.0}
    r = optimize_p1(ising.h / scale, ising.full_J() / scale, n_gamma=401, n_beta=181, n_refine=3)
    probs = StatevectorQAOA((E - ising.offset) / scale).probs([r["gamma"]], [r["beta"]])
    if fidelity is not None:
        probs = fidelity * probs + (1.0 - fidelity) / probs.size
    counts = counts_from_probs(probs, shots, seed)
    idx = np.fromiter(counts, dtype=np.int64)
    k = int(idx[np.argmin(E[idx])])
    ground = np.flatnonzero(E <= E.min() + 1e-12)
    return k, float(E[k]), {"gamma": r["gamma"], "beta": r["beta"], "p_ground": float(probs[ground].sum()),
                            "p_ground_uniform": len(ground) / E.size,
                            "mean_energy": float(probs @ E), "uniform_energy": float(E.mean())}


def greedy_select(cands: Candidates, n_nodes: int) -> np.ndarray:
    """Indices of the candidates kept by the greedy ambiguity resolution with merge_overlap = 0."""
    _, idx = resolve_ambiguity(cands, n_nodes, 0.0)
    return idx


def select(cands: Candidates, n_nodes: int, method: str, max_vars: int = 24, penalty_factor: float = 1.5,
           seed: int = 0, shots: int = 1024, fidelity_of=None) -> Dict[str, object]:
    """Selected candidate indices for one event with one solver, plus per-component bookkeeping."""
    prob = build_problem(cands, n_nodes)
    greedy = set(int(i) for i in greedy_select(cands, n_nodes))
    chosen: List[int] = []
    stats = {"n_candidates": prob.n, "n_conflicts": prob.n_conflicts, "n_components": len(prob.components),
             "component_sizes": [len(c) for c in prob.components], "n_fit": 0, "n_too_large": 0,
             "candidates_in_too_large": 0, "optimal": 0, "feasible": 0, "qubo_energy": 0.0,
             "exact_energy": 0.0, "seconds": 0.0, "p_ground": []}
    t0 = time.time()
    for k, comp in enumerate(prob.components):
        if len(comp) == 1:
            chosen.append(int(comp[0]))                # isolated candidate: w > 0, always kept
            stats["n_fit"] += 1
            stats["optimal"] += 1
            stats["feasible"] += 1
            e = -float(prob.weights[comp[0]])
            stats["qubo_energy"] += e
            stats["exact_energy"] += e
            continue
        if len(comp) > max_vars:
            stats["n_too_large"] += 1
            stats["candidates_in_too_large"] += len(comp)
            chosen += [int(c) for c in comp if int(c) in greedy]
            continue
        stats["n_fit"] += 1
        qubo, _ = component_qubo(prob, comp, penalty_factor)
        i_ex, e_ex = solve_exact(qubo)
        if method == "exact":
            i, e = i_ex, e_ex
        elif method == "sa":
            i, e = solve_sa(qubo, seed=seed + k)
        elif method == "random":
            i, e = solve_random(qubo, shots=shots, seed=seed + k)
        elif method in ("qaoa", "qaoa-noise"):
            F = fidelity_of(qubo) if (method == "qaoa-noise" and fidelity_of) else None
            i, e, info = solve_qaoa(qubo, shots=shots, seed=seed + k, fidelity=F)
            stats["p_ground"].append(info["p_ground"])
            # ground-state probability relative to uniform sampling (1 = no better than random guessing)
            stats.setdefault("ground_amplification", []).append(info["p_ground"] / info["p_ground_uniform"])
            stats.setdefault("component_n_for_amp", []).append(int(qubo.n))
        elif method == "greedy":
            loc = np.array([1 if int(c) in greedy else 0 for c in comp])
            i = int(np.sum(loc << np.arange(len(comp))))
            e = float(qubo.energy_bits(loc[None, :].astype(float))[0])
        else:
            raise ValueError(f"unknown selection method {method}")
        x = bits_of(i, len(comp))
        stats["qubo_energy"] += e
        stats["exact_energy"] += e_ex
        stats["optimal"] += int(abs(e - e_ex) < 1e-9)
        if is_independent(x, qubo):
            stats["feasible"] += 1
            chosen += [int(c) for c, b in zip(comp, x) if b]
        else:                                       # infeasible sample: keep a conflict-free subset (greedy repair)
            keep = [int(c) for c, b in zip(comp, x) if b]
            keep.sort(key=lambda c: -prob.weights[c])
            used: set = set()
            for c in keep:
                hits = set(cands.tracks[c].tolist())
                if not hits & used:
                    chosen.append(c)
                    used |= hits
    stats["seconds"] = round(time.time() - t0, 3)
    stats["selected_weight"] = float(prob.weights[chosen].sum()) if chosen else 0.0
    return {"selected": np.array(sorted(chosen), dtype=np.int64), "stats": stats}

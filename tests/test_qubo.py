"""QX3 candidate-conflict QUBO on hand-made candidates with analytically known optima."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsf2._phase1 import ensure_phase1, find_phase1  # noqa: E402

pytestmark = pytest.mark.skipif(find_phase1() is None, reason="Phase-1 package acts_qaoa not found")
if find_phase1() is not None:
    ensure_phase1()


def cands_of(tracks, scores=None):
    from hsf2.tracking.build import Candidates
    tracks = [np.asarray(t, dtype=np.int64) for t in tracks]
    ms = np.ones(len(tracks)) if scores is None else np.asarray(scores, float)
    return Candidates(tracks, np.array([len(t) for t in tracks]) + ms, ms)


def test_greedy_is_suboptimal_where_mwis_is_not():
    """A (6 hits) overlaps B (4) and C (5), B and C are disjoint: MWIS = {B, C} (9) > greedy {A} (6)."""
    from hsf2.qubo.selection import select
    c = cands_of([[0, 1, 2, 3, 4, 5], [0, 1, 2, 10], [3, 4, 5, 11, 12]])
    g = select(c, 20, "greedy")
    assert g["selected"].tolist() == [0] and g["stats"]["selected_weight"] == 6.0
    for m in ("exact", "sa", "qaoa", "random"):
        r = select(c, 20, m, seed=1)
        assert r["selected"].tolist() == [1, 2], m
        assert r["stats"]["selected_weight"] == 9.0 and r["stats"]["optimal"] == 1


def test_penalty_regime_makes_every_minimum_conflict_free():
    from acts_qaoa.landscape import enumerate_energies
    from hsf2.qubo.selection import build_problem, component_qubo
    rng = np.random.default_rng(3)
    for trial in range(20):
        tracks = [rng.choice(30, size=rng.integers(3, 7), replace=False) for _ in range(9)]
        c = cands_of(tracks, rng.uniform(0.3, 1.0, len(tracks)))
        prob = build_problem(c, 30)
        for comp in prob.components:
            if len(comp) < 2:
                continue
            q, P = component_qubo(prob, comp, 1.01)
            assert P > prob.weights[comp].max()
            E = enumerate_energies(q)
            best = int(np.argmin(E))
            x = ((best >> np.arange(q.n)) & 1).astype(float)
            assert x @ q.W @ x == 0.0                       # no selected conflict
            # brute-force maximum-weight independent set of the component agrees
            w = prob.weights[comp]
            confl = {(i, j) for i, j in zip(*np.nonzero(q.W))}
            bestw = max(sum(w[k] for k in S) for r in range(len(comp) + 1)
                        for S in itertools.combinations(range(len(comp)), r)
                        if not any((a, b) in confl or (b, a) in confl for a, b in itertools.combinations(S, 2)))
            assert abs(-E[best] - bestw) < 1e-9
    with pytest.raises(ValueError):
        component_qubo(prob, prob.components[0], 1.0)


def test_qaoa_amplifies_ground_state_over_uniform():
    from hsf2.qubo.selection import build_problem, component_qubo, solve_qaoa
    # a 6-candidate chain of conflicts: path graph, MWIS unique
    tracks = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 10], [10, 11, 12]]
    c = cands_of(tracks, [1.0, 0.9, 0.8, 0.95, 0.7, 1.0])
    prob = build_problem(c, 13)
    assert len(prob.components) == 1
    q, _ = component_qubo(prob, prob.components[0])
    _, _, info = solve_qaoa(q, shots=256, seed=0)
    assert info["p_ground"] > info["p_ground_uniform"]
    assert info["mean_energy"] < info["uniform_energy"]

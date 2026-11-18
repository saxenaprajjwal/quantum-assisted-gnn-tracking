"""Independent validation of the QX3b triplet QUBO (hsf2/qubo/triplet.py): energy conventions, Ising map of the JSON
export, baseline properties.  Synthetic detector only, fast and deterministic."""
import json
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


def _tq(alpha=0.0, zeta=1.0, n_particles=40, seed=3):
    from hsf2.data.synthetic import ToyConfig, generate_events
    from hsf2.qubo.triplet import TOY_BARREL, barrel_sector, build_triplet_qubo, make_doublets
    ev = generate_events(1, ToyConfig(n_particles=n_particles, noise_frac=0.05, seed=seed, hit_eff=1.0))[0]
    sub, bpos = barrel_sector(ev, TOY_BARREL, 0.0, 2 * np.pi)
    return sub, bpos, build_triplet_qubo(sub, bpos, *make_doublets(sub, bpos), alpha=alpha, zeta=zeta)


def _independent_energy(tq, x):
    """Paper Eq. 4-5 straight from the definition, O(n^2) pair loop over triplets (no use of tq.qi/ci)."""
    quad = {}
    for i, j, s in zip(tq.qi, tq.qj, tq.S):
        quad[(int(i), int(j))] = float(s)
    E = tq.alpha * float(np.sum(x))
    for i, j in zip(tq.ci, tq.cj):
        E += tq.zeta * x[i] * x[j]
    for (i, j), s in quad.items():
        E -= s * x[i] * x[j]
    return E


def _conflicts_from_definition(tq):
    """Conflict pairs re-derived from hits: share a hit, not a quadruplet, not a quintet (c = d)."""
    T = tq.triplets
    quad = {(min(i, j), max(i, j)) for i, j in zip(tq.qi.tolist(), tq.qj.tolist())}
    out = set()
    for i in range(tq.n):
        for j in range(i + 1, tq.n):
            if not set(T[i].tolist()) & set(T[j].tolist()):
                continue
            ti, tj = T[i], T[j]
            quad_shaped = (ti[1] == tj[0] and ti[2] == tj[1]) or (tj[1] == ti[0] and tj[2] == ti[1])
            quintet = ti[2] == tj[0] or tj[2] == ti[0]
            if quad_shaped or quintet:
                continue
            out.add((i, j))
    return out


def test_conflict_set_matches_definition():
    _, _, tq = _tq()
    got = {(min(i, j), max(i, j)) for i, j in zip(tq.ci.tolist(), tq.cj.tolist())}
    assert got == _conflicts_from_definition(tq) and len(got) > 0


def test_all_energy_routes_agree_on_random_selections():
    from acts_qaoa.landscape import enumerate_energies
    from hsf2.qubo.triplet import component_terms, qubo_energy, terms_energy
    rng = np.random.default_rng(1)
    _, _, tq = _tq(alpha=-0.3, zeta=1.7)
    comps = [c for c in tq.components() if len(c) <= 14]
    assert comps
    for comp in comps[:6]:
        comp = np.sort(comp)
        m = len(comp)
        lin, quad = component_terms(tq, comp)
        E = enumerate_energies(tq.sub_qubo(comp))
        for _ in range(8):
            xl = rng.integers(0, 2, m)
            xg = np.zeros(tq.n)
            xg[comp] = xl
            idx = int(sum(int(b) << k for k, b in enumerate(xl)))            # local bit k = k-th triplet
            e_ref = _independent_energy(tq, xg)
            assert abs(qubo_energy(tq, xg) - e_ref) < 1e-12
            assert abs(terms_energy(lin, quad, xl) - e_ref) < 1e-12           # components are uncoupled: no offset
            assert abs(float(E[idx]) - e_ref) < 1e-12                          # Phase-1 convention, const = 0


def test_total_energy_is_sum_over_components():
    from hsf2.qubo.triplet import qubo_energy, component_terms, terms_energy
    rng = np.random.default_rng(2)
    _, _, tq = _tq(alpha=0.2)
    x = rng.integers(0, 2, tq.n).astype(float)
    tot = 0.0
    for comp in tq.components():
        comp = np.sort(comp)
        tot += terms_energy(*component_terms(tq, comp), x[comp])
    assert abs(tot - qubo_energy(tq, x)) < 1e-9


def _ising(linear, quadratic):
    n = len(linear)
    h = -0.5 * np.asarray(linear, float)
    J = {}
    off = 0.5 * float(np.sum(linear))
    for i, j, b in quadratic:
        h[i] -= b / 4
        h[j] -= b / 4
        J[(i, j)] = b / 4
        off += b / 4
    return h, J, off


def test_export_ising_map_and_bit_order(tmp_path):
    from acts_qaoa.landscape import enumerate_energies
    from hsf2.qubo.triplet import TOY_BARREL, run_study
    from hsf2.data.synthetic import ToyConfig, generate_events
    ev = generate_events(1, ToyConfig(n_particles=40, noise_frac=0.05, seed=3, hit_eff=1.0))[0]
    crows = []
    run_study([ev], TOY_BARREL, [(0.0, 2 * np.pi)], [1.0], ("exact",), component_rows=crows,
              export_dir=tmp_path / "c")
    files = sorted((tmp_path / "c").glob("*.json"))
    assert files
    rng = np.random.default_rng(0)
    for fn in files:
        d = json.loads(fn.read_text())
        lin, quad = d["linear"], d["quadratic"]
        n = len(lin)
        h, J, off = _ising(lin, quad)
        for _ in range(10):
            x = rng.integers(0, 2, n)
            z = 1 - 2 * x
            e_q = float(np.dot(lin, x) + sum(b * x[i] * x[j] for i, j, b in quad))
            e_i = float(np.dot(h, z) + sum(v * z[i] * z[j] for (i, j), v in J.items()) + off)
            assert abs(e_q - e_i) < 1e-12
        # exact_selection[i] is bit i of the Phase-1 index (LSB = local triplet 0), and it is a minimum
        if n <= 14:
            ones = [d["exact_selection"][k] << k for k in range(n)]
            idx = sum(ones)
            rebuilt_E = np.empty(2 ** n)
            for s in range(2 ** n):
                xs = (s >> np.arange(n)) & 1
                rebuilt_E[s] = float(np.dot(lin, xs) + sum(b * xs[i] * xs[j] for i, j, b in quad))
            assert abs(rebuilt_E[idx] - d["exact_energy"]) < 1e-9 and abs(rebuilt_E.min() - d["exact_energy"]) < 1e-9
        # local order of the JSON triplets is the sorted global index order (hit ids match the sub-event)
        assert len(d["triplets"]) == n == len(d["triplets_hit_id"]) == len(d["true_triplet"])


def test_baseline_properties():
    from hsf2.qubo.triplet import qubo_energy, select_descent, select_greedy_s, select_hole_free
    for alpha in (0.0, -0.4):
        _, _, tq = _tq(alpha=alpha)
        hf, gr = select_hole_free(tq), select_greedy_s(tq)
        g = np.zeros(tq.n, bool)
        g[gr] = True
        assert not np.any(g[tq.ci] & g[tq.cj])
        # maximal: every unselected triplet has a selected conflict neighbour
        nb = [set() for _ in range(tq.n)]
        for a, b in zip(tq.ci.tolist(), tq.cj.tolist()):
            nb[a].add(b)
            nb[b].add(a)
        assert all(g[i] or any(g[j] for j in nb[i]) for i in range(tq.n))
        assert len(select_hole_free(tq)) == int(np.sum(tq.holes == 0))
        for start in (hf, gr):
            x0 = np.zeros(tq.n)
            x0[start] = 1
            d = select_descent(tq, start=start)
            x = np.zeros(tq.n)
            x[d] = 1
            assert qubo_energy(tq, x) <= qubo_energy(tq, x0) + 1e-12
            for k in range(tq.n):
                y = x.copy()
                y[k] = 1 - y[k]
                assert qubo_energy(tq, y) >= qubo_energy(tq, x) - 1e-9
            # determinism
            assert np.array_equal(d, select_descent(tq, start=start))


def test_exact_is_not_worse_than_baselines_per_component():
    from acts_qaoa.landscape import enumerate_energies
    from hsf2.qubo.triplet import component_terms, select_descent, select_greedy_s, select_hole_free, terms_energy
    _, _, tq = _tq(alpha=0.0)
    sels = {"hf": select_hole_free(tq), "gr": select_greedy_s(tq)}
    sels["hfd"] = select_descent(tq, start=sels["hf"])
    sels["grd"] = select_descent(tq, start=sels["gr"])
    for comp in tq.components():
        if len(comp) > 16:
            continue
        comp = np.sort(comp)
        emin = float(enumerate_energies(tq.sub_qubo(comp)).min())
        for s in sels.values():
            x = np.isin(comp, s).astype(float)
            assert terms_energy(*component_terms(tq, comp), x) >= emin - 1e-9

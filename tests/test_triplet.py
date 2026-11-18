"""QX3b triplet QUBO (Bapst et al. 2020 / qallse Config1GeV + max-path pre-filter) on the synthetic detector."""
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


def _event(n_particles=40, noise=0.05, seed=3):
    from hsf2.data.synthetic import ToyConfig, generate_events
    return generate_events(1, ToyConfig(n_particles=n_particles, noise_frac=noise, seed=seed, hit_eff=1.0))[0]


def test_curvature_is_menger_curvature_in_inverse_mm():
    """pT from the curvature of three barrel hits of a particle reproduces its true pT (B = 2 T): the cuts are in
    mm^-1, CURV_MAX = 8e-4 <-> 0.75 GeV."""
    from hsf2.qubo.triplet import CURV_MAX, curvature, pt_of_curvature
    assert abs(pt_of_curvature(CURV_MAX) - 0.75) < 1e-12
    ev = _event(n_particles=60, noise=0.0)
    ratios = []
    for pid in np.unique(ev.particle_id[ev.particle_id > 0]):
        h = np.nonzero((ev.particle_id == pid) & np.isin(ev.layer, [4, 6, 8]))[0]
        if len(np.unique(ev.layer[h])) < 3:
            continue
        h = h[np.argsort(ev.layer[h])][:3]
        pt = float(ev.particles["pt"][ev.particle_rows(np.array([pid]))[0]])
        if 0.8 < pt < 5.0:
            ratios.append(float(pt_of_curvature(curvature(ev, h[0], h[1], h[2]))) / pt)
    assert len(ratios) >= 5 and abs(np.median(ratios) - 1.0) < 0.03


def test_prefilter_keeps_only_chains_of_two_quadruplets_and_tracks_are_found():
    from hsf2.qubo.triplet import (S_MIN, TOY_BARREL, barrel_sector, build_triplet_qubo, doublet_metrics,
                                   make_doublets, postprocess, solve)
    ev = _event()
    sub, bpos = barrel_sector(ev, TOY_BARREL, 0.0, 2 * np.pi)
    ds, dd = make_doublets(sub, bpos)
    tq = build_triplet_qubo(sub, bpos, ds, dd)
    loose = build_triplet_qubo(sub, bpos, ds, dd, min_qplet_path=1)
    assert 0 < tq.n <= loose.n <= tq.n_raw and len(tq.qi) <= len(loose.qi) <= tq.n_quad_raw
    assert np.all(tq.S > S_MIN) and np.all(tq.S <= 1.0)
    # every kept triplet is in a kept quadruplet, and every kept quadruplet has a neighbour quadruplet (chain >= 2)
    in_quad = np.zeros(tq.n, bool)
    in_quad[tq.qi] = in_quad[tq.qj] = True
    assert in_quad.all()
    pairs = [(int(i), int(j)) if tq.triplets[i][1] == tq.triplets[j][0] else (int(j), int(i))
             for i, j in zip(tq.qi, tq.qj)]                                    # (inner, outer) triplet
    inner = {o for _, o in pairs}
    outer = {i for i, _ in pairs}
    assert all(i in inner or o in outer for i, o in pairs)
    # no conflict between the two triplets of a quadruplet
    quad = set(map(tuple, np.sort(np.c_[tq.qi, tq.qj], 1).tolist()))
    assert not quad & set(map(tuple, np.sort(np.c_[tq.ci, tq.cj], 1).tolist()))
    r = solve(tq, "sa", seed=0)
    _, final = postprocess(sub, tq, r["selected"])
    m = doublet_metrics(sub, bpos, final)
    assert m["efficiency"] > 0.9 and m["purity"] > 0.9


def _study(tmp_path, n_particles=40, methods=("exact",)):
    from hsf2.qubo.triplet import TOY_BARREL, run_study
    ev = _event(n_particles=n_particles)
    crows = []
    rows = run_study([ev], TOY_BARREL, [(0.0, 2 * np.pi)], [1.0], methods, component_rows=crows,
                     export_dir=tmp_path / "components")
    return ev, rows, crows


def test_baselines_are_scored_and_well_formed(tmp_path):
    from hsf2.qubo.triplet import (TOY_BARREL, barrel_sector, build_triplet_qubo, make_doublets, qubo_energy,
                                   select_descent, select_greedy_s, select_hole_free)
    ev, rows, _ = _study(tmp_path)
    by = {r["method"]: r for r in rows}
    for name in ("all-kept-triplets", "hole-free", "greedy-S", "hole-free+descent", "greedy-S+descent"):
        assert 0.0 <= by[name]["efficiency"] <= 1.0 and 0.0 <= by[name]["purity"] <= 1.0
    sub, bpos = barrel_sector(ev, TOY_BARREL, 0.0, 2 * np.pi)
    tq = build_triplet_qubo(sub, bpos, *make_doublets(sub, bpos))
    hole_free = select_hole_free(tq)
    assert np.all(tq.holes[hole_free] == 0)
    g = np.zeros(tq.n, bool)
    g[select_greedy_s(tq)] = True
    assert not np.any(g[tq.ci] & g[tq.cj]) and g.any()
    h = np.zeros(tq.n)
    h[hole_free] = 1.0
    d = np.zeros(tq.n)
    d[select_descent(tq, start=hole_free)] = 1.0
    e = qubo_energy(tq, d)
    assert e <= qubo_energy(tq, h) + 1e-12
    for k in range(tq.n):
        y = d.copy()
        y[k] = 1.0 - y[k]
        assert qubo_energy(tq, y) >= e - 1e-9
    # with alpha = 0 the descent from the empty selection never moves; a negative alpha makes it select something
    tq2 = build_triplet_qubo(sub, bpos, *make_doublets(sub, bpos), alpha=-0.5)
    assert len(select_descent(tq2)) > 0


def test_exported_components_reproduce_exact_energy(tmp_path):
    import itertools
    import json
    _, _, crows = _study(tmp_path)
    files = sorted((tmp_path / "components").glob("*.json"))
    assert files and len(files) == sum(r["n_variables"] <= 24 for r in crows)
    n_checked_bf = 0
    for fn in files:
        doc = json.loads(fn.read_text())
        n = len(doc["triplets"])
        assert fn.name == "{}_s{}_f{:g}_c{}.json".format(doc["meta"]["event"], doc["meta"]["sector"],
                                                         doc["meta"]["frac"], doc["meta"]["component"])
        assert len(doc["linear"]) == len(doc["exact_selection"]) == len(doc["true_triplet"]) == n
        assert len(doc["triplets_hit_id"]) == n and all(i < j for i, j, _ in doc["quadratic"])

        def energy(x):
            return sum(a * xi for a, xi in zip(doc["linear"], x)) + sum(b * x[i] * x[j] for i, j, b in doc["quadratic"])
        assert abs(energy(doc["exact_selection"]) - doc["exact_energy"]) < 1e-9
        if n <= 12:
            best = min(energy(x) for x in itertools.product((0, 1), repeat=n))
            assert abs(best - doc["exact_energy"]) < 1e-9
            n_checked_bf += 1
    assert n_checked_bf > 0


def test_components_table_is_consistent(tmp_path):
    import json
    from hsf2.qubo.triplet import TOY_BARREL, barrel_sector, build_triplet_qubo, make_doublets
    ev, _, crows = _study(tmp_path)
    sub, bpos = barrel_sector(ev, TOY_BARREL, 0.0, 2 * np.pi)
    tq = build_triplet_qubo(sub, bpos, *make_doublets(sub, bpos))
    assert len(crows) == len(tq.components())
    assert sum(r["n_variables"] for r in crows) == tq.n
    assert sum(r["n_true_triplets"] + r["n_fake_triplets"] for r in crows) == tq.n
    assert sum(r["n_quadruplet_couplings"] for r in crows) == len(tq.qi)
    assert sum(r["n_conflicts"] for r in crows) == len(tq.ci)
    for r in crows:
        assert r["exact_energy"] is None or r["exact_energy"] <= min(
            r["holefree_energy"], r["greedy_energy"], r["descent_energy"]) + 1e-9
        if r["n_variables"] > 24:
            assert r["exact_energy"] is None and r["trivial"] is None
            continue
        doc = json.loads(next((tmp_path / "components").glob(f"*_c{r['component']}.json")).read_text())
        comp = np.sort(tq.components()[r["component"]])
        hf = (tq.holes[comp] == 0).astype(int).tolist()
        assert r["trivial"] == (doc["exact_selection"] == hf)
        if r["trivial"]:
            assert abs(r["holefree_energy"] - r["exact_energy"]) < 1e-9 and r["holefree_optimal"]

"""QX3b components on hardware (hsf2.hw.components, hsf2.hw.comp_analysis, CLI hw-components): selection and hard flag,
QUBO -> Ising, circuits and bit order, equal-shot jobs with one typed confirmation each, analysis (single-flip descent,
uniform / ideal / white-noise controls, aggregates), local modes, retrieval, CLI.  0 QPU s: fake backends, Aer, mocks; IBM is
never contacted.  Physics references are written independently here (numpy), not taken from hsf2."""
import io
import json
import math
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsf2._phase1 import ensure_phase1, find_phase1  # noqa: E402

pytestmark = pytest.mark.skipif(find_phase1() is None, reason="Phase-1 package acts_qaoa not found")
if find_phase1() is not None:
    ensure_phase1()

SHOTS = 400
AER_SHOTS = 500                # Aer cases: at most 10 variables and 500 shots
EVENT = "hand"


# ---------------------------------------------------------------------------- independent references
def brute_energies(n, quad, linear=None):
    """E[index] of the QUBO  linear.x + sum b_ij x_i x_j  (bit i of the index = variable i)."""
    x = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)
    E = np.zeros(1 << n) if linear is None else x @ np.asarray(linear, float)
    for i, j, b in quad:
        E += b * x[:, i] * x[:, j]
    return E


def ref_ising(linear, quad):
    """h_i = -a_i/2 - sum_j b_ij/4, J_ij = b_ij/4 (upper triangle), offset = sum a/2 + sum b/4 (the design note)."""
    n = len(linear)
    h = -0.5 * np.asarray(linear, float)
    J = np.zeros((n, n))
    off = 0.5 * float(np.sum(linear))
    for i, j, b in quad:
        h[i] -= b / 4
        h[j] -= b / 4
        J[i, j] = b / 4
        off += b / 4
    return h, J, off


def ref_probs(h, J, scale, gamma, beta):
    """Independent numpy p = 1 QAOA: |+>^n, exp(-i gamma H / scale), exp(-i beta sum X); index bit k = qubit k."""
    n = len(h)
    bits = (np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1
    z = 1.0 - 2.0 * bits
    ph = z @ (h / scale) + np.einsum("ij,ij->i", z @ (J / scale), z)
    psi = np.exp(-1j * gamma * ph) / math.sqrt(1 << n)
    c, s = math.cos(beta), -1j * math.sin(beta)
    for k in range(n):
        psi = psi.reshape(1 << (n - k - 1), 2, 1 << k)
        psi = np.stack([c * psi[:, 0] + s * psi[:, 1], s * psi[:, 0] + c * psi[:, 1]], axis=1).reshape(-1)
    return np.abs(psi) ** 2


def py_descent(E, n, start):
    """Steepest single-flip descent written from the energy table only: flip the bit with the lowest dE (ties: lowest bit),
    stop when no flip lowers E by more than 1e-12."""
    x = int(start)
    while True:
        d = [E[x ^ (1 << k)] - E[x] for k in range(n)]
        k = int(np.argmin(d))
        if d[k] >= -1e-12:
            return x
        x ^= 1 << k


def tq_of(n, quad, alpha=0.0, zeta=1.0):
    """hsf2 TripletQUBO with the couplings b < 0 as quadruplets (S = -b) and b > 0 as conflicts; no holes."""
    from hsf2.qubo.triplet import TripletQUBO
    q = [(i, j, -b) for i, j, b in quad if b < 0]
    c = [(i, j) for i, j, b in quad if b > 0]
    as_i = lambda v: np.array(v, dtype=np.int64)  # noqa: E731
    return TripletQUBO(np.zeros((n, 3), np.int64), np.zeros(n, np.int64), np.zeros(n), np.zeros(n),
                       as_i([i for i, _, _ in q]), as_i([j for _, j, _ in q]), np.array([s for _, _, s in q], float),
                       as_i([i for i, _ in c]), as_i([j for _, j in c]), float(alpha), float(zeta))


# ---------------------------------------------------------------------------- instances
# 6 variables where steepest descent from the greedy-S start AND from the all-selected start stops at E = -1 on a plateau,
# while the exact optimum {1, 4, 5} has E = -2 (it is two flips away; 20 of the 64 starts descend into it)
HARD6 = dict(n=6, quad=[(0, 1, 1.0), (1, 3, 1.0), (1, 4, -1.0), (2, 3, -1.0), (2, 5, 1.0), (4, 5, -1.0)],
             true=[False, True, False, True, True, True])
# ordinary pieces (both descents find the optimum) whose distribution is far from its bit reversal (TV 0.68 and 0.49), so
# that a reversed bit order would be noticed
EASY5 = dict(n=5, quad=[(0, 1, -1.0), (1, 2, -1.0), (2, 3, -0.5), (3, 4, 1.0)], true=[True] * 5)
ASYM8 = dict(n=8, quad=sorted([(0, 1, -1.0), (1, 2, -1.0), (2, 3, -0.8), (4, 5, -0.3), (6, 7, -0.2), (1, 4, 1.0), (2, 5, 1.0),
                               (3, 6, 1.0), (0, 7, 1.0), (3, 4, 1.0)]), true=[True] * 6 + [False] * 2)
HAND = {0: HARD6, 1: EASY5, 2: ASYM8, 3: HARD6, 4: HARD6, 5: HARD6}          # component id -> instance


def hand_component(cid, spec):
    """(components.csv row, JSON document) of a hand-made component; the baselines are the hsf2 ones (hole-free = every
    triplet, as there are no holes), the exact optimum is brute force."""
    from hsf2.qubo.triplet import qubo_energy, select_descent, select_greedy_s, select_hole_free
    n, quad = spec["n"], spec["quad"]
    E = brute_energies(n, quad)
    e0 = float(E.min())
    sel = [(int(np.argmin(E)) >> k) & 1 for k in range(n)]
    tq = tq_of(n, quad)
    hf, gr = select_hole_free(tq), select_greedy_s(tq)

    def en(s):
        x = np.zeros(n)
        x[s] = 1.0
        return qubo_energy(tq, x)
    row = {"event": EVENT, "sector": 0, "phi0": 0.0, "dphi": 6.283185307179586, "frac": 1.0, "component": cid,
           "n_variables": n, "n_quadruplet_couplings": sum(b < 0 for _, _, b in quad),
           "n_conflicts": sum(b > 0 for _, _, b in quad), "n_true_triplets": int(sum(spec["true"])),
           "n_fake_triplets": n - int(sum(spec["true"])), "exact_energy": e0, "holefree_energy": en(hf),
           "greedy_energy": en(gr), "descent_energy": en(select_descent(tq, start=hf)),
           "greedy_descent_energy": en(select_descent(tq, start=gr)), "trivial": False, "holefree_optimal": False}
    doc = {"meta": {"event": EVENT, "sector": 0, "phi0": 0.0, "dphi": 6.283185307179586, "frac": 1.0, "component": cid,
                    "alpha": 0.0, "zeta": 1.0},
           "triplets": [[k, k + 1, k + 2] for k in range(n)], "triplets_hit_id": [[10 * k, 10 * k + 1, 10 * k + 2] for k in range(n)],
           "linear": [0.0] * n, "quadratic": [[i, j, b] for i, j, b in quad], "exact_energy": e0, "exact_selection": sel,
           "true_triplet": list(spec["true"])}
    return row, doc


def write_run(path, specs):
    """A triplet-run-like folder (components.csv + components/<event>_s0_f1_c<id>.json) of hand-made components."""
    path = Path(path)
    (path / "components").mkdir(parents=True)
    rows = []
    for cid, spec in specs.items():
        row, doc = hand_component(cid, spec)
        (path / "components" / f"{EVENT}_s0_f1_c{cid}.json").write_text(json.dumps(doc))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path / "components.csv", index=False)
    return path


def make_triplet_run(path):
    """The run folder `python -m hsf2 triplet` writes, on the synthetic detector (21 components of 3-22 variables)."""
    from hsf2.data.synthetic import ToyConfig, generate_events
    from hsf2.qubo.triplet import TOY_BARREL, run_study
    from hsf2.utils import save_csv
    ev = generate_events(1, ToyConfig(n_particles=40, noise_frac=0.05, seed=3, hit_eff=1.0))[0]
    path = Path(path)
    (path / "components").mkdir(parents=True)
    rows = []
    run_study([ev], TOY_BARREL, [(0.0, 2 * np.pi)], [1.0], ("exact",), component_rows=rows, export_dir=path / "components")
    save_csv(pd.DataFrame(rows), path / "components.csv")
    return path


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory):
    return make_triplet_run(tmp_path_factory.mktemp("synth") / "tri")


@pytest.fixture(scope="module")
def hand_run(tmp_path_factory):
    return write_run(tmp_path_factory.mktemp("hand") / "tri", HAND)


def by_name(run, cid):
    from hsf2.hw.components import load_components
    return next(c for c in load_components(run) if c.name == f"{EVENT}_s0_f1_c{cid}")


def cfg_local(tmp_path, **kw):
    from hsf2.hw.config import HwConfig
    d = dict(mode="local", shots=SHOTS, transpile_seeds=1, ledger=str(tmp_path / "ledger.json"), budget_s=400.0)
    d.update(kw)
    return HwConfig(**d)


# ---------------------------------------------------------------------------- 1. QUBO -> Ising
def test_qubo_to_ising_energy_matches_every_exported_component(synth_run):
    from hsf2.hw.components import component_ising, component_qubo, ising_to_qubo, load_components
    comps = load_components(synth_run)
    assert len(comps) == 21 and max(c.n for c in comps) == 22
    rng = np.random.default_rng(0)
    for c in comps:
        lin, quad, n = c.linear, c.quad, c.n
        h, J, off = ref_ising(lin, quad)
        ising = component_ising(c.doc)                       # raises if its energies differ from the QUBO's
        assert np.allclose(ising.h, h, atol=1e-12) and np.allclose(np.triu(ising.J, 1), J, atol=1e-12)
        assert abs(ising.offset - off) < 1e-12
        for x in rng.integers(0, 2, size=(25, n)):
            z = 1 - 2 * x
            e_q = float(lin @ x + sum(b * x[i] * x[j] for i, j, b in quad))
            e_i = float(h @ z + sum(J[i, j] * z[i] * z[j] for i, j, _ in quad) + off)
            assert abs(e_q - e_i) < 1e-12
            assert abs(float(ising.energy_bits(x[None, :])[0]) - e_q) < 1e-12
        x = np.array(c.doc["exact_selection"])
        z = 1 - 2 * x
        assert abs(float(h @ z + sum(J[i, j] * z[i] * z[j] for i, j, _ in quad) + off) - c.doc["exact_energy"]) < 1e-9
        back = ising_to_qubo(ising)                           # exact inverse (used for the circuit Hamiltonian)
        q = component_qubo(c.doc)
        assert np.allclose(back.L, q.L, atol=1e-12) and np.allclose(back.W, q.W, atol=1e-12) and abs(back.const) < 1e-12


def test_ising_check_fails_loudly_on_a_wrong_export(hand_run, tmp_path):
    from hsf2.hw.components import component_ising, load_components
    doc = json.loads(json.dumps(by_name(hand_run, 2).doc))
    doc["exact_energy"] += 0.25
    with pytest.raises(ValueError, match="exact_energy"):
        component_ising(doc)
    # a JSON that disagrees with its components.csv row, a missing file and a missing column are errors, never defaults
    bad = tmp_path / "tri"
    shutil.copytree(hand_run, bad)
    f = bad / "components" / f"{EVENT}_s0_f1_c1.json"
    d = json.loads(f.read_text())
    d["exact_energy"] += 0.5
    f.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="exact"):
        load_components(bad)
    f.write_text(json.dumps(by_name(hand_run, 1).doc))
    (bad / "components" / f"{EVENT}_s0_f1_c2.json").unlink()
    with pytest.raises(FileNotFoundError, match="c2"):
        load_components(bad)
    df = pd.read_csv(hand_run / "components.csv").drop(columns=["greedy_descent_energy"])
    df.to_csv(bad / "components.csv", index=False)
    (bad / "components" / f"{EVENT}_s0_f1_c2.json").write_text(json.dumps(by_name(hand_run, 2).doc))
    with pytest.raises(ValueError, match="greedy_descent_energy"):
        load_components(bad)


# ---------------------------------------------------------------------------- 2. circuit, ideal distribution, bit order
def test_circuit_ideal_distribution_equals_independent_statevector_and_bit_order(hand_run):
    from acts_qaoa.circuits import bind, counts_to_index
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from hsf2.hw import fake as F
    from hsf2.hw.build import build_circuit
    from hsf2.hw.components import component_angles, component_ising, ideal_distribution, plan_component
    from hsf2.hw.settings import get_setting
    comp = by_name(hand_run, 2)                                # 8 variables, asymmetric couplings
    n = comp.n
    ising = component_ising(comp.doc)
    ism, scale, gammas, betas, info = component_angles(ising)
    h, J, _ = ref_ising(comp.linear, comp.quad)
    scale_ref = max(np.abs(h).max(), np.abs(J).max())
    assert scale == pytest.approx(scale_ref, rel=1e-12) and info["n_couplings"] == len(comp.quad)
    p_ref = ref_probs(h, J, scale_ref, gammas[0], betas[0])
    qc, gam, bet, _, _ = build_circuit(ism, scale)
    p_circ = Statevector(bind(qc, gam, bet, gammas, betas).remove_final_measurements(inplace=False)).probabilities()
    assert np.abs(p_circ - p_ref).max() < 1e-12                # the Qiskit circuit, bit order included
    assert np.abs(ideal_distribution(ism, scale, gammas[0], betas[0]) - p_ref).max() < 1e-12
    E = brute_energies(n, comp.quad, comp.linear)
    assert float(p_ref @ E) == pytest.approx(info["ideal_mean_energy"], abs=1e-9)      # closed form of the angle search
    # the transpiled circuit samples the same distribution: bit i of a counts index is variable i.  500 noiseless shots are
    # enough because the likelihood of the counts separates the right bit order from the reversed one by ~ 100 nats
    S = 500
    plan = plan_component(comp, 0, S, get_setting("a"), F.fake_backend("FakeFez"), 1)
    res = AerSimulator(method="statevector").run(F.bound_circuit(plan.pub.isa, plan.pub.values), shots=S, seed_simulator=1).result()
    counts = counts_to_index(res.get_counts())
    idx, cnt = np.array(list(counts)), np.array(list(counts.values()))
    assert cnt.sum() == S and idx.max() < 1 << n
    rev = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(1 << n)])
    tv = lambda a, b: 0.5 * float(np.abs(a - b).sum())  # noqa: E731
    assert tv(p_ref, p_ref[rev]) > 0.4                                    # the reference is far from its bit reversal
    ll = lambda q: float(cnt @ np.log(np.maximum(q[idx], 1e-300)))      # noqa: E731  log-likelihood of the counts under q
    assert ll(p_ref) - ll(p_ref[rev]) > 20.0
    assert abs(float(cnt @ E[idx]) / S - float(p_ref @ E)) < 0.25          # mean energy of the samples (std(E) 1.1, 5 sigma)


def test_layout_record_fallback_reads_the_layout_when_qiskits_helper_fails(hand_run, monkeypatch):
    from qiskit.transpiler.layout import TranspileLayout
    from hsf2.hw import fake as F
    from hsf2.hw.components import component_layout_record, plan_component
    from hsf2.hw.settings import get_setting
    plan = plan_component(by_name(hand_run, 1), 0, 10, get_setting("a"), F.fake_backend("FakeFez"), 1)
    isa, st, normal = plan.pub.isa, plan.pub.stats, plan.pub.layout

    def broken(self, filter_ancillas=False):               # what qiskit 2.1 does for some circuits and seeds
        raise IndexError("list assignment index out of range")
    monkeypatch.setattr(TranspileLayout, "initial_index_layout", broken)
    rec = component_layout_record(isa, 5, st)
    assert rec["initial"] == normal["initial"] and rec["measured"] == normal["measured"]
    assert rec["final"] == rec["measured"]["qubits"] and rec["edges"] == normal["edges"]


# ---------------------------------------------------------------------------- 3. selection
def test_hard_flag_and_size_range(hand_run, synth_run):
    from hsf2.hw.components import is_hard, load_components, select_components
    comps = load_components(hand_run)
    by = {c.name: c for c in comps}
    hard, easy, asym = by[f"{EVENT}_s0_f1_c0"], by[f"{EVENT}_s0_f1_c1"], by[f"{EVENT}_s0_f1_c2"]
    r = hard.row
    assert (r["exact_energy"], r["descent_energy"], r["greedy_descent_energy"]) == (-2.0, -1.0, -1.0)
    assert hard.hard and not easy.hard and not asym.hard
    # both descents must miss: one that finds the optimum, or ends within 1e-9 of it, makes the piece ordinary
    e0 = r["exact_energy"]
    assert not is_hard(dict(r, descent_energy=e0)) and not is_hard(dict(r, greedy_descent_energy=e0))
    assert not is_hard(dict(r, descent_energy=e0 + 5e-10, greedy_descent_energy=e0 + 5e-10))
    assert is_hard(dict(r, descent_energy=e0 + 2e-9, greedy_descent_energy=e0 + 2e-9))
    with pytest.raises(ValueError, match="no exact"):
        is_hard(dict(r, exact_energy=None))
    # the descents (written from the energy table) are stuck at E = -1 from the greedy and from the all-selected start
    E = brute_energies(6, HARD6["quad"])
    assert E[py_descent(E, 6, (1 << 6) - 1)] == -1.0 and E[py_descent(E, 6, 0b010110)] == -1.0 > E.min()
    # size range is inclusive; --hard keeps only the hard pieces, hard pieces first
    sel = select_components(comps, 5, 8, hard_only=True)
    assert [c.name for c in sel] == [f"{EVENT}_s0_f1_c{i}" for i in (0, 3, 4, 5)] and {c.group for c in sel} == {"hard"}
    assert [c.n for c in select_components(comps, 5, 6, n_benchmark=1)] == [6, 6, 6, 6, 5]      # 4 hard + the n=5 piece
    with pytest.raises(ValueError, match="no component selected"):
        select_components(comps, 7, 8)                       # the n=8 piece is ordinary, no benchmark was asked for
    with pytest.raises(ValueError, match="no component selected"):
        select_components(comps, 2, 4, n_benchmark=0)
    # no piece of the synthetic detector is hard; the flag is the documented formula on every row
    sc = load_components(synth_run)
    assert not any(c.hard for c in sc)
    assert all(c.hard == (min(c.row["descent_energy"], c.row["greedy_descent_energy"]) > c.row["exact_energy"] + 1e-9) for c in sc)


def test_benchmark_sample_is_seeded_reproducible_and_validated(synth_run, hand_run):
    from hsf2.hw.components import load_components, select_components
    comps = load_components(synth_run)
    pool = [c for c in comps if 10 <= c.n <= 13]
    assert len(pool) == 10
    a = select_components(comps, 10, 13, n_benchmark=4, seed=0)
    assert [c.name for c in a] == [c.name for c in select_components(comps, 10, 13, n_benchmark=4, seed=0)]
    b = select_components(comps, 10, 13, n_benchmark=4, seed=1)
    assert [c.name for c in a] != [c.name for c in b]
    for sel in (a, b):
        assert len(sel) == 4 and len({c.name for c in sel}) == 4 and all(10 <= c.n <= 13 for c in sel)
        assert {c.group for c in sel} == {"benchmark"} and not any(c.hard for c in sel)
        order = [c.name for c in pool]
        assert [c.name for c in sel] == sorted((c.name for c in sel), key=order.index)       # file order, not draw order
    assert len(select_components(comps, 10, 13, n_benchmark=10, seed=5)) == 10
    for kw, msg in ((dict(n_benchmark=11), "only 10 ordinary"), (dict(hard_only=True, n_benchmark=1), "cannot be combined"),
                    (dict(nmax=25), "nmax"), (dict(nmin=1), "nmin"), (dict(nmin=14, nmax=13), "nmax"),
                    (dict(n_benchmark=-1), ">= 0"), (dict(n_benchmark=3, max_total=2), "smaller than")):
        with pytest.raises(ValueError, match=msg):
            select_components(comps, **dict(dict(nmin=10, nmax=13), **kw))
    # --max keeps the benchmark sample and a seeded random subset of the hard pieces, hard pieces first
    hc = load_components(hand_run)
    full = select_components(hc, 5, 8, n_benchmark=1, seed=0)
    assert [c.group for c in full] == ["hard"] * 4 + ["benchmark"] and full[-1].n in (5, 8)
    part = select_components(hc, 5, 8, n_benchmark=1, seed=0, max_total=3)
    assert [c.group for c in part] == ["hard", "hard", "benchmark"] and part[-1].name == full[-1].name
    assert {c.name for c in part} <= {c.name for c in full}
    assert [c.name for c in part] == [c.name for c in select_components(hc, 5, 8, n_benchmark=1, seed=0, max_total=3)]
    assert len(select_components(hc, 5, 8, hard_only=True, max_total=2)) == 2


# ---------------------------------------------------------------------------- 4. jobs, confirmations, budget (mocked sampler)
class FakeService:
    def __init__(self, consumed=100.0, limit=600.0, step=5.0):
        self.consumed, self.limit, self.step = consumed, limit, step

    def usage(self):
        return {"usage_consumed_seconds": self.consumed, "usage_limit_seconds": self.limit}


class FakeJob:
    def __init__(self, job_id, results, charge=None):
        self._id, self._results, self.charge = job_id, results, charge

    def job_id(self):
        return self._id

    def result(self):
        return self._results

    def metrics(self):
        if self.charge is None:
            return {"usage": {"status": "pending"}}      # usage stays pending: the account difference is booked
        return {"usage": {"status": "finished", "qpu_charge_time_seconds": self.charge}}

    def properties(self):
        return None


class FakeSampler:
    """Records the PUBs of every run() and answers uniform counts of the right register size for each PUB."""

    def __init__(self, svc, charge=None):
        self.svc, self.calls, self.pubs_seen, self.options = svc, 0, [], SimpleNamespace()
        self.charge, self.jobs = charge, {}

    def run(self, pubs, **kw):
        self.calls += 1
        self.svc.consumed += self.svc.step
        self.pubs_seen.append([(p[0].num_clbits, p[2]) for p in pubs])
        rng = np.random.default_rng(self.calls)
        results = []
        for isa, values, shots in pubs:
            nb = isa.num_clbits
            c = rng.multinomial(shots, np.ones(1 << nb) / (1 << nb))
            counts = {format(i, f"0{nb}b"): int(v) for i, v in enumerate(c) if v}
            results.append(SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda c=counts: c))))
        job = FakeJob(f"fakejob{self.calls:04d}", results, self.charge)
        self.jobs[job.job_id()] = job
        return job


@pytest.fixture
def fast_polling(monkeypatch):
    from hsf2.hw import run as R
    for k, v in (("USAGE_TIMEOUT_S", 0.05), ("USAGE_INTERVAL_S", 0.005), ("ACCOUNT_TIMEOUT_S", 0.05),
                 ("ACCOUNT_INTERVAL_S", 0.005)):
        monkeypatch.setattr(R, k, v)


def mock_ibm(monkeypatch, charge=None):
    """Mock the IBM side of a submit: fake backend, fake service, fake sampler (nothing leaves the process)."""
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from hsf2.hw import run as R
    svc = FakeService()
    smp = FakeSampler(svc, charge)
    svc.job = lambda job_id: smp.jobs[job_id]
    monkeypatch.setattr(R, "get_job_backend", lambda cfg, setting, service=None: (FakeFez(), svc))
    monkeypatch.setattr(R, "make_sampler", lambda *a, **k: (smp, {"mock": True}))
    return svc, smp


def test_equal_shot_grouping_one_confirmation_per_job_and_budget_refusal(hand_run, tmp_path, monkeypatch, fast_polling):
    from acts_qaoa.ibm import QPULedger
    from hsf2.hw.components import load_components, run_components, select_components
    from hsf2.hw.ledger import BudgetError
    from hsf2.hw.run import SubmitAborted
    comps = select_components(load_components(hand_run), 5, 8, n_benchmark=2, seed=0, max_total=4)    # 2 hard + 2 ordinary
    assert [c.group for c in comps] == ["hard", "hard", "benchmark", "benchmark"]
    run = tmp_path / "tri"
    shutil.copytree(hand_run, run)
    svc, smp = mock_ibm(monkeypatch)
    cfg = cfg_local(tmp_path, mode="submit", backend="ibm_fake")
    asked = []
    res = run_components(run, comps, "d", cfg, per_job=3, command="test", out_dir=tmp_path / "out", confirm=asked.append)
    # 4 components, 3 per job: jobs of 3 and 1 PUBs, one confirmation and one sampler call each, equal shots everywhere
    assert [len(j["manifest"]["pubs"]) for j in res["jobs"]] == [3, 1] and len(asked) == 2 and smp.calls == 2
    assert all(s == SHOTS for seen in smp.pubs_seen for _, s in seen)
    assert [[nb for nb, _ in seen] for seen in smp.pubs_seen] == [[c.n for c in comps[:3]], [c.n for c in comps[3:]]]
    assert "submit ONE job" in asked[0] and "comp_d_1" in asked[0] and "3 PUBs x 400 shots" in asked[0]
    assert all(c.name in asked[0] for c in comps[:3]) and comps[3].name not in asked[0]    # the confirmation names the components
    assert [j["manifest"]["tag"] for j in res["jobs"]] == ["comp_d_1", "comp_d_2"]
    # PUB k of a job is the k-th component of its group, in components.json and in the manifest
    for j, grp in zip(res["jobs"], (comps[:3], comps[3:])):
        listed = json.loads((j["job_dir"] / "components.json").read_text())["pubs"]
        assert [(p["pub"], p["component"], p["group"]) for p in listed] == [(k, c.name, c.group) for k, c in enumerate(grp)]
        assert [p["component"] for p in j["manifest"]["pubs"]] == [c.name for c in grp]
        assert j["manifest"]["state"] == "done" and j["manifest"]["kind"] == "components" and j["manifest"]["step"] == "step3"
        assert j["manifest"]["billed"]["source"] == "account_diff" and j["manifest"]["billed"]["used"] == 5.0
        for k, c in enumerate(grp):
            cf = json.loads((j["job_dir"] / f"pub_{k}_counts.json").read_text())
            assert cf["component"] == c.name and sum(cf["counts"].values()) == SHOTS and cf["n"] == c.n
            assert (j["job_dir"] / f"pub_{k}_main.isa.qpy").exists() and (j["job_dir"] / f"pub_{k}.layout.json").exists()
        assert (j["job_dir"] / "snapshot_pre.json").exists() and (j["job_dir"] / "snapshot_post.json").exists()
    led = QPULedger(cfg.ledger_path(), budget_s=400.0)
    assert [e["job_id"] for e in led.entries] == ["fakejob0001", "fakejob0002"]
    assert [e["quantum_seconds"] for e in led.entries] == [5.0, 5.0]
    assert res["analysis"] is not None and len(res["analysis"]["records"]) == 4       # the analysis ran by itself
    assert len(pd.read_csv(tmp_path / "out" / "components_hw.csv")) == 4
    # a refused budget stops everything before the first confirmation and the first submission
    n_jobs_before, n_calls = len(list((run / "hw").iterdir())), smp.calls
    asked.clear()
    with pytest.raises(BudgetError, match="exceeds"):
        run_components(run, comps[:2], "d", cfg_local(tmp_path, mode="submit", backend="ibm_fake", budget_s=5.0,
                                                  ledger=str(tmp_path / "l2.json")), per_job=1, confirm=asked.append)
    assert smp.calls == n_calls and not asked and not (tmp_path / "l2.json").exists()
    assert len(list((run / "hw").iterdir())) == n_jobs_before
    # a refused confirmation of the second job: the first job is finished, analysed and booked, the second is not submitted
    seq = []

    def second_no(msg):
        seq.append(msg)
        if len(seq) == 2:
            raise SubmitAborted("no")
    cfg3 = cfg_local(tmp_path, mode="submit", backend="ibm_fake", ledger=str(tmp_path / "l3.json"))
    with pytest.raises(SubmitAborted):
        run_components(run, comps[:2], "d", cfg3, per_job=1, out_dir=tmp_path / "out3", confirm=second_no)
    assert len(seq) == 2 and smp.calls == n_calls + 1
    assert len(pd.read_csv(tmp_path / "out3" / "components_hw.csv")) == 1
    assert [e["job_id"] for e in QPULedger(cfg3.ledger_path(), budget_s=400.0).entries] == ["fakejob0003"]
    mans = [json.loads(p.read_text()) for p in (run / "hw").glob("*/manifest.json")]
    refused = [m for m in mans if m["state"] == "aborted"]               # the refused job's folder says why it has no counts
    assert len(refused) == 1 and refused[0]["job_id"] is None and "SubmitAborted" in refused[0]["aborted"]
    assert refused[0]["tag"] == "comp_d_2" and not [m for m in mans if m["state"] == "planned"]


def test_no_wait_submit_then_retrieve_resolves_and_books_the_billed_time(hand_run, tmp_path, monkeypatch, fast_polling):
    from acts_qaoa.ibm import QPULedger
    from hsf2.hw.components import load_components, retrieve_any, run_components, select_components
    run = tmp_path / "tri"
    shutil.copytree(hand_run, run)
    comps = select_components(load_components(run), 5, 8, hard_only=True, max_total=2)
    svc, smp = mock_ibm(monkeypatch, charge=6.5)               # IBM's own per-job figure is final
    cfg = cfg_local(tmp_path, mode="submit", backend="ibm_fake")
    asked = []
    res = run_components(run, comps, "d", cfg, per_job=2, wait=False, confirm=asked.append, out_dir=tmp_path / "out")
    job = res["jobs"][0]
    assert len(asked) == 1 and job["job_id"] == "fakejob0001" and job["counts"] is None and res["analysis"] is None
    man = json.loads((job["job_dir"] / "manifest.json").read_text())
    assert man["state"] == "submitted" and man["job_id"] == "fakejob0001" and not list(job["job_dir"].glob("*counts*"))
    assert QPULedger(cfg.ledger_path(), budget_s=400.0).entries[-1]["quantum_seconds"] is None      # only the estimate so far
    out = retrieve_any("fakejob0001", cfg, run_dir=run, service=svc)                              # the shell was lost
    man = json.loads((out["job_dir"] / "manifest.json").read_text())
    assert man["state"] == "done" and man["billed"]["source"] == "job" and man["billed"]["used"] == 6.5
    cfs = [json.loads((out["job_dir"] / f"pub_{k}_counts.json").read_text()) for k in (0, 1)]
    assert [cf["component"] for cf in cfs] == [c.name for c in comps]
    assert all(sum(cf["counts"].values()) == SHOTS for cf in cfs)
    e = QPULedger(cfg.ledger_path(), budget_s=400.0).entries[-1]
    assert e["job_id"] == "fakejob0001" and e["quantum_seconds"] == 6.5 and e["usage_source"] == "job"
    assert smp.calls == 1                                                                       # retrieval never resubmits


def test_planning_validation_and_unequal_shots(hand_run, tmp_path):
    from hsf2.hw.components import load_components, run_components, select_components
    comps = select_components(load_components(hand_run), 5, 6, hard_only=True, max_total=2)
    shutil.copytree(hand_run, tmp_path / "tri")
    res = run_components(tmp_path / "tri", comps, "d", cfg_local(tmp_path, mode="dry"), per_job=2)
    job = res["jobs"][0]["job"]
    assert job.shots == SHOTS and [p.spec.shots for p in job.pubs] == [SHOTS, SHOTS]
    job.plans[1].pub.spec.shots += 1
    with pytest.raises(ValueError, match="unequal shots"):
        job.shots
    with pytest.raises(ValueError, match="gauges"):
        run_components(tmp_path / "tri", comps, "f", cfg_local(tmp_path, mode="dry"))
    with pytest.raises(ValueError, match="per-job"):
        run_components(tmp_path / "tri", comps, "d", cfg_local(tmp_path, mode="dry"), per_job=0)
    with pytest.raises(ValueError, match="no-wait"):
        run_components(tmp_path / "tri", comps, "d", cfg_local(tmp_path, mode="dry"), per_job=1, wait=False)
    with pytest.raises(ValueError, match="sim_method"):
        run_components(tmp_path / "tri", comps, "d", cfg_local(tmp_path, mode="dry"), sim_method="gpu")
    assert res["jobs"][0]["manifest"]["state"] == "dry" and not list(res["jobs"][0]["job_dir"].glob("*counts*"))


def test_fractional_setting_isa_has_rzz_in_range_and_twirling_off(hand_run, tmp_path):
    from hsf2.hw.components import load_components, run_components
    from hsf2.hw.transpile import check_fractional_isa
    try:
        from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle  # noqa: F401
    except ImportError:
        pytest.skip("FoldRzzAngle not available in this qiskit-ibm-runtime")
    run = tmp_path / "tri"
    shutil.copytree(hand_run, run)
    comps = [c for c in load_components(run) if c.n == 5]
    res = run_components(run, comps, "e", cfg_local(tmp_path, mode="dry"), per_job=1)
    for p in res["jobs"][0]["job"].pubs:
        ang = check_fractional_isa(p.isa)
        assert ang and all(0.0 < a <= math.pi / 2 + 1e-9 for a in ang) and p.values is None
    assert res["jobs"][0]["manifest"]["setting"]["options"]["twirling.enable_gates"] is False


# ---------------------------------------------------------------------------- 5. analysis
def pub_of(comp, shots, top_k=None, k=0):
    """The manifest PUB entry that analyse_pub needs (angles from the Phase-1 closed form), and the circuit Ising."""
    from hsf2.hw.components import component_angles, component_ising
    ism, scale, g, b, info = component_angles(component_ising(comp.doc), top_k)
    return ({"k": k, "shots": shots, "gammas": g, "betas": b, "scale": scale, "ideal_mean_energy": info["ideal_mean_energy"],
             "n_couplings": info["n_couplings"], "n_2q": 100, "fidelity_est": 0.5}, ism)


def to_counts(c):
    return {int(i): int(v) for i, v in zip(np.nonzero(c)[0], c[c > 0])}


def test_white_noise_mixture_recovers_F_and_ideal_statistics(hand_run):
    from hsf2.hw.comp_analysis import analyse_pub
    comp = by_name(hand_run, 2)
    n, N = comp.n, 1 << comp.n
    S, F = 400000, 0.6
    pub, ism = pub_of(comp, S)
    h, J, _ = ref_ising(comp.linear, comp.quad)
    p = ref_probs(h, J, pub["scale"], pub["gammas"][0], pub["betas"][0])
    E = brute_energies(n, comp.quad, comp.linear)
    c = np.random.default_rng(2).multinomial(S, F * p + (1 - F) / N)
    rec = analyse_pub(comp, to_counts(c), pub, None, seed=0)
    E_unif, E_ideal = float(E.mean()), float(p @ E)
    idx = np.nonzero(c)[0]
    se_e = np.sqrt(np.var(np.repeat(E[idx], c[idx])) / S) / (E_unif - E_ideal)
    se_x = np.sqrt(np.var(np.repeat(p[idx], c[idx])) / S) / (float(p @ p) - 1.0 / N)
    hw = rec["hardware"]
    assert abs(hw["F_E"] - F) < 4 * se_e and abs(hw["F_XEB"] - F) < 4 * se_x
    ground = E <= E.min() + 1e-9
    assert rec["E0"] == pytest.approx(float(E.min())) and rec["ground_degeneracy"] == int(ground.sum())
    assert rec["p_unif_ground"] == pytest.approx(ground.sum() / N)
    assert rec["ideal"]["p_ground"] == pytest.approx(float(p[ground].sum()), abs=1e-12)
    assert rec["ideal"]["mean_energy"] == pytest.approx(E_ideal, abs=1e-12)
    assert hw["p_ground"] == pytest.approx(c[ground].sum() / S) and hw["mean_energy"] == pytest.approx(float(c @ E / S))
    assert rec["p_hw_over_p_unif"] == pytest.approx(hw["p_ground"] / rec["p_unif_ground"])
    assert hw["best_energy"] == pytest.approx(float(E[idx].min())) and hw["best_reaches_E0"] == bool(c[ground].sum() > 0)
    # the white-noise control is the same model with the measured F_E, N = S draws: its P(ground) agrees
    assert rec["controls"]["white_noise"]["p_ground"] == pytest.approx(hw["p_ground"], abs=0.02)


def test_batch_descent_reaches_only_local_minima_and_equals_triplet_select_descent():
    from hsf2.hw.comp_analysis import batch_descent, bits_of, coupling_matrix, index_of
    from hsf2.qubo.triplet import select_descent
    for ties in (False, True):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            n = int(rng.integers(4, 12))
            quad = []
            for i in range(n):
                for j in range(i + 1, n):
                    r = rng.random()
                    if r < 0.25:
                        quad.append((i, j, -float(rng.choice([0.5, 0.75, 1.0]) if ties else rng.uniform(0.2, 1.0))))
                    elif r < 0.6:
                        quad.append((i, j, 1.0))
            E = brute_energies(n, quad)
            B = coupling_matrix(n, quad)
            starts = rng.integers(0, 1 << n, size=40)
            fin = index_of(batch_descent(np.zeros(n), B, bits_of(starts, n)))
            tq = tq_of(n, quad)
            for s0, f in zip(starts, fin):
                assert E[f] <= E[s0] + 1e-12                                              # never uphill
                assert all(E[f ^ (1 << k)] >= E[f] - 1e-9 for k in range(n))              # a local minimum: no flip helps
                ref = select_descent(tq, start=np.nonzero((int(s0) >> np.arange(n)) & 1)[0])
                assert int(f) == int(sum(1 << int(k) for k in ref))                        # the hsf2 descent, state for state
    with pytest.raises(ValueError, match="shapes"):
        batch_descent(np.zeros(3), np.zeros((3, 3)), np.zeros((2, 4)))


def test_greedy_start_is_the_triplet_greedy_s_without_holes():
    from hsf2.hw.comp_analysis import greedy_start
    from hsf2.qubo.triplet import select_greedy_s
    for spec in (HARD6, EASY5, ASYM8):
        sel = np.nonzero(greedy_start(spec["n"], spec["quad"]))[0]
        assert np.array_equal(sel, select_greedy_s(tq_of(spec["n"], spec["quad"])))
    assert list(np.nonzero(greedy_start(6, HARD6["quad"]))[0]) == [1, 2, 4]


def test_hard_component_descent_statistics_tracking_and_controls(hand_run):
    from hsf2.hw.comp_analysis import analyse_pub, draw_samples, rng_for
    from hsf2.hw.components import ideal_distribution
    comp = by_name(hand_run, 0)
    n, N, S = comp.n, 1 << comp.n, 1000
    E = brute_energies(n, comp.quad)
    E0 = float(E.min())
    reach = np.array([py_descent(E, n, s) for s in range(N)])
    assert E0 == -2.0 and np.isclose(E[reach] , E0).sum() == 20                    # 20 of 64 starts descend into the optimum
    pub, ism = pub_of(comp, S)
    i_opt, i_greedy, i_pair = 0b110010, 0b010110, 0b010010                          # {1,4,5}, {1,2,4}, {1,4}
    assert E[i_opt] == E0 and E[i_greedy] == -1.0 and E[i_pair] == -1.0 and reach[i_pair] == i_opt and reach[i_greedy] == i_greedy

    def run(counts, seed=0):
        return analyse_pub(comp, counts, pub, None, seed=seed)

    # the optimum is sampled: P(ground) = 0.4, best-of-N is E0, and the descent of 40 % + 10 % of the shots ends there
    r = run({i_opt: 400, i_pair: 100, i_greedy: 300, 0: 200})
    hw = r["hardware"]
    assert hw["p_ground"] == 0.4 and hw["best_energy"] == E0 and hw["best_reaches_E0"] and hw["best_index"] == i_opt
    assert hw["desc_frac_E0"] == 0.5 and hw["desc_any_E0"]
    assert r["tracking"]["hw_best"]["is_exact_selection"] and r["tracking"]["hw_best"]["n_true"] == 3
    # the optimum is never sampled, but one descent finds it: best-of-N misses E0, "descent added" reaches it
    r = run({i_pair: 100, i_greedy: 300, 0: 600})
    hw = r["hardware"]
    assert hw["p_ground"] == 0.0 and not hw["best_reaches_E0"] and hw["best_energy"] == -1.0
    assert hw["desc_any_E0"] and hw["desc_frac_E0"] == 0.1
    assert hw["best_index"] == i_greedy                                              # tie in energy: the more frequent sample
    t = r["tracking"]["hw_best"]
    assert (t["n_selected"], t["n_true"], t["n_fake"], t["is_exact_selection"], t["is_ground_state"]) == (3, 2, 1, False, False)
    # nothing sampled leads to the optimum: the descent does not help
    r = run({i_greedy: 500, 0: 500})
    assert not r["hardware"]["desc_any_E0"] and r["hardware"]["desc_frac_E0"] == 0.0
    # greedy-S + descent is stuck at -1 (= components.csv), the exact optimum is the exact selection
    assert r["greedy_descent_matches_csv"] is True
    g, e = r["tracking"]["greedy_descent"], r["tracking"]["exact"]
    assert g["energy"] == -1.0 and not g["is_ground_state"] and (g["n_true"], g["n_fake"]) == (2, 1) and g["start_energy"] == -1.0
    assert e["is_exact_selection"] and e["energy"] == E0 and (e["n_true"], e["n_fake"]) == (3, 0)
    assert r["n_true"] == 4 and r["n_fake"] == 2 and r["group"] == "" and r["hard"] is True
    # controls: seeded and reproducible, independent of the component order, and exactly the documented statistics
    r3, r3b, r4 = run({i_greedy: 500, 0: 500}, seed=3), run({i_greedy: 500, 0: 500}, seed=3), run({i_greedy: 500, 0: 500}, seed=4)
    assert r3["controls"] == r3b["controls"] and r3["controls"]["uniform"] != r4["controls"]["uniform"]
    ui, uc = draw_samples(None, n, S, rng_for(3, comp.name, 1))
    u = r3["controls"]["uniform"]
    assert u["shots"] == S and u["p_ground"] == pytest.approx(uc[E[ui] <= E0 + 1e-9].sum() / S)
    assert u["desc_frac_E0"] == pytest.approx(uc[E[reach[ui]] <= E0 + 1e-9].sum() / S)             # multi-start descent
    assert u["desc_any_E0"] == bool((E[reach[ui]] <= E0 + 1e-9).any()) and u["best_energy"] == float(E[ui].min())
    p = ideal_distribution(ism, pub["scale"], pub["gammas"][0], pub["betas"][0])
    ii, ic = draw_samples(p, n, S, rng_for(3, comp.name, 2))
    assert r3["controls"]["ideal"]["desc_frac_E0"] == pytest.approx(ic[E[reach[ii]] <= E0 + 1e-9].sum() / S)
    F = min(max(r3["hardware"]["F_E"], 0.0), 1.0)
    wi, wc = draw_samples(F * p + (1 - F) / N, n, S, rng_for(3, comp.name, 3))
    assert r3["controls"]["white_noise"]["p_ground"] == pytest.approx(wc[E[wi] <= E0 + 1e-9].sum() / S)
    # the uniform + descent selection is the best of the N descents
    ut = r3["tracking"]["uniform_descent"]
    assert ut["energy"] == float(E[reach[ui]].min())
    # counts that do not belong to the PUB are refused
    with pytest.raises(ValueError, match="counted shots"):
        run({i_opt: 10})
    with pytest.raises(ValueError, match="outside"):
        run({1 << 6: S})


def test_degenerate_ground_states_count_and_tracking(tmp_path):
    """Two interchangeable optima, {0,1} and {2,3}: g = 2, P(ground) counts both, exact_selection is only one of them."""
    from hsf2.hw.comp_analysis import analyse_pub
    deg = dict(n=4, quad=[(0, 1, -1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (1, 3, 1.0), (2, 3, -1.0)], true=[True, True, False, False])
    run = write_run(tmp_path / "tri", {0: deg})
    comp = by_name(run, 0)
    assert comp.doc["exact_selection"] == [1, 1, 0, 0] and comp.row["exact_energy"] == -1.0 and not comp.hard
    pub, _ = pub_of(comp, 1000)
    rec = analyse_pub(comp, {0b1100: 600, 0b0011: 100, 0: 300}, pub, None, seed=0)
    assert rec["ground_degeneracy"] == 2 and rec["p_unif_ground"] == 2 / 16
    assert rec["hardware"]["p_ground"] == 0.7 and rec["p_hw_over_p_unif"] == pytest.approx(0.7 / (2 / 16))
    t = rec["tracking"]["hw_best"]                       # the more frequent optimum {2, 3}: a ground state, not exact_selection
    assert rec["hardware"]["best_index"] == 0b1100 and t["is_ground_state"] and not t["is_exact_selection"]
    assert (t["n_true"], t["n_fake"]) == (0, 2) and rec["tracking"]["exact"]["is_exact_selection"]
    assert rec["hardware"]["desc_frac_E0"] == 0.7        # the descent of the empty selection (300 shots) does not move


def test_group_aggregates_medians_fractions_and_bootstrap():
    from hsf2.hw.comp_analysis import SOURCES, boot_stat, group_summary, summarize

    def rec(group, n, ratio, fe, fx, reach_hw, desc_hw, frac_hw, reach_u=False, desc_u=True, ideal=True):
        st = lambda r, d, f: {"best_reaches_E0": r, "desc_any_E0": d, "desc_frac_E0": f}  # noqa: E731
        return {"group": group, "n": n, "p_hw_over_p_unif": ratio,
                "hardware": dict(st(reach_hw, desc_hw, frac_hw), F_E=fe, F_XEB=fx),
                "controls": {"uniform": st(reach_u, desc_u, 0.3), "ideal": st(ideal, True, 0.5) if ideal is not None else None,
                             "white_noise": st(reach_hw, desc_hw, 0.2)}}
    recs = [rec("hard", 14, 1.0, 0.4, 0.45, False, True, 0.1), rec("hard", 16, 2.0, 0.3, 0.35, True, True, 0.2),
            rec("hard", 18, 3.0, 0.2, 0.25, False, False, 0.0), rec("hard", 20, 100.0, None, None, True, True, 0.4, ideal=None),
            rec("benchmark", 12, 7.0, 0.5, 0.5, True, True, 0.6)]
    s = summarize(recs)
    assert set(s) == {"hard", "benchmark", "all"} and s["hard"]["n_components"] == 4 and s["all"]["n_components"] == 5
    h = s["hard"]
    assert h["sizes"] == [14, 16, 18, 20]
    assert h["p_hw_over_p_unif_median"]["value"] == 2.5                          # median of 1, 2, 3, 100
    assert h["F_E_median"]["value"] == pytest.approx(0.3) and h["F_E_median"]["n"] == 3        # a missing F_E is skipped
    assert h["best_of_N_reaches_E0"]["hardware"]["value"] == 0.5
    assert h["best_of_N_plus_descent_reaches_E0"]["hardware"]["value"] == 0.75
    assert h["best_of_N_reaches_E0"]["uniform"]["value"] == 0.0 and h["best_of_N_plus_descent_reaches_E0"]["uniform"]["value"] == 1.0
    assert h["best_of_N_reaches_E0"]["ideal"]["n"] == 3 and h["best_of_N_reaches_E0"]["ideal"]["value"] == 1.0   # n > 22: no ideal
    assert h["descent_start_success_median"]["hardware"]["value"] == pytest.approx(0.15)
    assert h["p_hw_over_p_unif_median"]["lo68"] <= h["p_hw_over_p_unif_median"]["hi68"]
    assert h["F_E_median"]["lo68"] <= 0.3 <= h["F_E_median"]["hi68"]                # 0.2, 0.3, 0.4: the interval spans them
    b = h["best_of_N_reaches_E0"]["hardware"]
    assert 0.0 <= b["lo68"] <= b["value"] <= b["hi68"] <= 1.0
    assert summarize(recs) == s                                                    # the bootstrap is seeded
    assert group_summary(recs[:4], "hard")["F_XEB_median"] == h["F_XEB_median"]
    one = s["benchmark"]["p_hw_over_p_unif_median"]
    assert one["value"] == 7.0 and one["lo68"] is None and one["hi68"] is None and one["n"] == 1   # one piece: no interval
    assert set(h["best_of_N_reaches_E0"]) == set(SOURCES) == {"hardware", "uniform", "ideal", "white_noise"}
    assert boot_stat([None, float("nan")], np.median, np.random.default_rng(0))["value"] is None


def test_components_above_22_variables_have_no_ideal_statistics(tmp_path):
    """23 variables: the exact optimum and the descent statistics are there, the ideal statevector and its derivatives are not."""
    from hsf2.hw.comp_analysis import analyse_pub, flat_record, summarize
    from hsf2.hw.components import load_components
    n = 23
    quad = [(i, i + 1, -0.5) for i in range(n - 1)]                      # attractive chain: the all-ones state is the unique optimum
    row = {"event": EVENT, "sector": 0, "phi0": 0.0, "dphi": 1.0, "frac": 1.0, "component": 0, "n_variables": n,
           "n_quadruplet_couplings": n - 1, "n_conflicts": 0, "n_true_triplets": n, "n_fake_triplets": 0,
           "exact_energy": -0.5 * (n - 1), "descent_energy": -0.5 * (n - 1), "greedy_descent_energy": -0.5 * (n - 1)}
    doc = {"meta": {"event": EVENT, "sector": 0, "component": 0, "alpha": 0.0, "zeta": 1.0}, "linear": [0.0] * n,
           "quadratic": [[i, j, b] for i, j, b in quad], "exact_energy": -0.5 * (n - 1), "exact_selection": [1] * n,
           "true_triplet": [True] * n}
    run = tmp_path / "tri"
    (run / "components").mkdir(parents=True)
    (run / "components" / f"{EVENT}_s0_f1_c0.json").write_text(json.dumps(doc))
    pd.DataFrame([row]).to_csv(run / "components.csv", index=False)
    comp = load_components(run)[0]
    assert comp.n == n and not comp.hard
    S = 2000
    c = np.random.default_rng(0).integers(0, 1 << n, size=S)
    idx, cnt = np.unique(c, return_counts=True)
    pub = {"k": 0, "shots": S, "gammas": [0.1], "betas": [0.2], "scale": 1.0, "ideal_mean_energy": 0.0, "n_couplings": n - 1}
    rec = analyse_pub(comp, {int(i): int(v) for i, v in zip(idx, cnt)}, pub, None, seed=0)
    assert rec["E0"] == -0.5 * (n - 1) and rec["ground_degeneracy"] == 1 and rec["p_unif_ground"] == 1.0 / (1 << n)
    assert rec["ideal"] is None and rec["hardware"]["F_E"] is None and rec["hardware"]["F_XEB"] is None
    assert rec["controls"]["ideal"] is None and rec["controls"]["white_noise"] is None and rec["controls"]["uniform"]["shots"] == S
    assert rec["p_ideal_over_p_unif"] is None and rec["tracking"]["exact"]["n_true"] == n
    flat = flat_record(rec)
    assert flat["F_E"] is None and flat["ideal_p_ground"] is None and flat["ideal_sample_best_energy"] is None and flat["n"] == n
    g = summarize([dict(rec, group="benchmark")])["benchmark"]
    assert g["best_of_N_reaches_E0"]["ideal"]["value"] is None and g["best_of_N_reaches_E0"]["uniform"]["n"] == 1
    assert 0.0 <= rec["hardware"]["desc_frac_E0"] <= 1.0 and 0.0 <= rec["controls"]["uniform"]["desc_frac_E0"] <= 1.0


def test_ideal_circuit_with_top_k_uses_the_sparsified_hamiltonian(hand_run):
    from hsf2.hw.comp_analysis import analyse_pub
    comp = by_name(hand_run, 2)
    pub, ism = pub_of(comp, 2000, top_k=5)
    assert pub["n_couplings"] == 5 and int(np.count_nonzero(np.triu(ism.J, 1))) == 5
    h, J, _ = ref_ising(comp.linear, comp.quad)
    Js = np.zeros_like(J)
    iu, ju = np.nonzero(np.triu(ism.J, 1))
    Js[iu, ju] = J[iu, ju]                                                         # the 5 kept couplings of the original
    p = ref_probs(h, Js, pub["scale"], pub["gammas"][0], pub["betas"][0])
    E = brute_energies(comp.n, comp.quad, comp.linear)
    assert float(p @ E) == pytest.approx(pub["ideal_mean_energy"], abs=1e-9)       # closed form with the full observable
    rec = analyse_pub(comp, to_counts(np.random.default_rng(0).multinomial(2000, p)), pub, 5, seed=0)
    assert rec["ideal"]["mean_energy"] == pytest.approx(float(p @ E), abs=1e-9)
    bad = dict(pub, ideal_mean_energy=pub["ideal_mean_energy"] + 0.1)
    with pytest.raises(ValueError, match="not the one that was run"):
        analyse_pub(comp, to_counts(np.random.default_rng(0).multinomial(2000, p)), bad, 5, seed=0)


# ---------------------------------------------------------------------------- 6. local modes, outputs, retrieval
def small_fake_name():
    from qiskit_ibm_runtime import fake_provider as fp
    for name in ("FakeLagosV2", "FakeNairobiV2", "FakePerth", "FakeJakartaV2", "FakeOslo"):
        if hasattr(fp, name):
            return name
    pytest.skip("no small fake backend in this qiskit-ibm-runtime")


@pytest.fixture(scope="module")
def aer_job(hand_run, tmp_path_factory):
    """One local job of the 5-variable chain simulated by Aer noise trajectories (500 shots) on a small fake device."""
    from hsf2.hw.components import run_components
    tmp = tmp_path_factory.mktemp("aerjob")
    run = tmp / "tri"
    shutil.copytree(hand_run, run)
    comps = [by_name(run, 1)]
    cfg = cfg_local(tmp, shots=AER_SHOTS, fake_backend=small_fake_name())
    res = run_components(run, comps, "d", cfg, per_job=1, command="test", out_dir=tmp / "out", sim_method="statevector", seed=7)
    return {"run": run, "res": res, "cfg": cfg, "tmp": tmp, "comp": comps[0]}


def test_local_aer_job_files_and_distribution(aer_job):
    from hsf2.hw.components import ideal_distribution
    res, comp = aer_job["res"], aer_job["comp"]
    j = res["jobs"][0]
    man, jd = j["manifest"], j["job_dir"]
    assert man["state"] == "done" and man["job_id"].startswith("local_") and man["billed"] == {"job_s": 0.0, "used": 0.0, "source": "local"}
    assert man["sim_methods"] == ["statevector"] and man["pubs"][0]["sim_method"] == "statevector"
    for f in ("manifest.json", "components.json", "snapshot_pre.json", "snapshot_post.json", "local_result.json",
              "pub_0_main.isa.qpy", "pub_0.layout.json", "pub_0_counts.json"):
        assert (jd / f).exists(), f
    stored = json.loads((jd / "local_result.json").read_text())
    assert stored["job_id"] == man["job_id"] and stored["sim_methods"] == ["statevector"]
    cf = json.loads((jd / "pub_0_counts.json").read_text())
    assert cf["counts"] == stored["counts"][0] and sum(cf["counts"].values()) == AER_SHOTS
    # the counts follow the circuit's distribution (small device noise) and not its bit reversal
    pub, ism = pub_of(comp, AER_SHOTS)
    p = ideal_distribution(ism, pub["scale"], pub["gammas"][0], pub["betas"][0])
    emp = np.zeros(32)
    for k, v in cf["counts"].items():
        emp[int(k)] = v / AER_SHOTS
    rev = np.array([int(format(i, "05b")[::-1], 2) for i in range(32)])
    tv = lambda a, b: 0.5 * float(np.abs(a - b).sum())  # noqa: E731
    assert tv(p, p[rev]) > 0.1 and tv(emp, p) < 0.3 and tv(emp, p) < tv(emp, p[rev])
    # the analysis ran and its files are in the command folder
    csv = pd.read_csv(aer_job["tmp"] / "out" / "components_hw.csv")
    assert len(csv) == 1 and csv.loc[0, "sim_method"] == "statevector" and csv.loc[0, "component"] == comp.name


def test_top_k_job_is_the_same_circuit_in_the_plan_and_in_the_analysis(hand_run, tmp_path):
    from hsf2.hw.components import run_components
    run = tmp_path / "tri"
    shutil.copytree(hand_run, run)
    comp = by_name(run, 1)                                       # 5 variables, 4 couplings
    cfg = cfg_local(tmp_path, shots=6000)                        # the white-noise model is cheap, so many shots
    full = run_components(run, [comp], "d", cfg_local(tmp_path, mode="dry"), per_job=1)["jobs"][0]
    res = run_components(run, [comp], "d", cfg, per_job=1, top_k=2, sim_method="white-noise", out_dir=tmp_path / "out")
    man = res["jobs"][0]["manifest"]
    assert man["top_k"] == 2 and man["pubs"][0]["n_couplings"] == 2 and full["manifest"]["top_k"] is None
    assert man["pubs"][0]["n_2q"] < full["manifest"]["pubs"][0]["n_2q"]                   # fewer couplings, fewer 2q gates
    assert man["pubs"][0]["gammas"] != full["manifest"]["pubs"][0]["gammas"]               # the angles were re-optimised
    rec = res["analysis"]["records"][0]                           # the ideal <E> of the sparsified circuit matches the plan's
    assert rec["ideal"]["mean_energy"] == pytest.approx(man["pubs"][0]["ideal_mean_energy"], abs=1e-6)
    assert rec["hardware"]["F_E"] == pytest.approx(man["pubs"][0]["fidelity_est"], abs=0.1)         # standard error 0.011


def test_white_noise_local_mode_and_auto_choice(hand_run, tmp_path):
    from hsf2.hw import components as C
    from hsf2.hw.comp_analysis import analyse_pub
    run = tmp_path / "tri"
    shutil.copytree(hand_run, run)
    comps = [by_name(run, 2), by_name(run, 1)]
    res = C.run_components(run, comps, "d", cfg_local(tmp_path, shots=20000), per_job=2, command="test", sim_method="white-noise",
                           seed=5, out_dir=tmp_path / "out")
    man = res["jobs"][0]["manifest"]
    assert man["sim_methods"] == ["white-noise", "white-noise"] and man["state"] == "done"
    # F_E measured on the model counts is the estimated circuit fidelity that generated them
    for rec, plan in zip(res["analysis"]["records"], res["jobs"][0]["job"].plans):
        assert rec["hardware"]["F_E"] == pytest.approx(plan.pub.stats["fidelity_est"], abs=0.08)
        assert rec["sim_method"] == "white-noise"
    # auto: Aer up to AUTO_MAX_QUBITS active qubits, the model above; every choice is explicit
    plan = res["jobs"][0]["job"].plans[0]
    assert C.sim_method_for("auto", plan) == "statevector" and C.sim_method_for("matrix_product_state", plan) == "matrix_product_state"
    plan.pub.stats = dict(plan.pub.stats, n_active_qubits=C.AUTO_MAX_QUBITS + 1)
    assert C.sim_method_for("auto", plan) == "white-noise" and C.sim_method_for("statevector", plan) == "statevector"
    with pytest.raises(ValueError, match="sim_method"):
        C.sim_method_for("fast", plan)
    # the model needs the exact ideal distribution
    plan.comp = C.replace(plan.comp, doc=dict(plan.comp.doc, linear=[0.0] * 23))
    with pytest.raises(ValueError, match="n <= 22"):
        C.white_noise_counts(plan, 0)


def test_retrieve_rebuilds_a_component_job_and_other_jobs_go_to_run_retrieve_job(aer_job, tmp_path, monkeypatch):
    from hsf2.hw import components as C
    from hsf2.hw import run as R
    run = tmp_path / "copy"
    shutil.copytree(aer_job["run"], run)                      # a moved folder: the run folder comes from the path
    j = aer_job["res"]["jobs"][0]
    jd = run / "hw" / j["job_dir"].name
    job_id = j["manifest"]["job_id"]
    before = json.loads((jd / "pub_0_counts.json").read_text())
    for f in ("pub_0_counts.json", "snapshot_post.json"):
        (jd / f).unlink()
    man = json.loads((jd / "manifest.json").read_text())
    man["state"] = "submitted"
    (jd / "manifest.json").write_text(json.dumps(man))
    res = C.retrieve_any(job_id, aer_job["cfg"], run_dir=run)
    assert res["job_dir"] == jd and json.loads((jd / "manifest.json").read_text())["state"] == "done"
    assert json.loads((jd / "pub_0_counts.json").read_text()) == before and (jd / "snapshot_post.json").exists()
    with pytest.raises(FileNotFoundError):
        C.retrieve_any("local_00000000", aer_job["cfg"], run_dir=run)
    # a job of another kind is served by run.retrieve_job, unchanged
    other = run / "hw" / "x_other"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"job_id": "local_cafe0001", "kind": None}))
    monkeypatch.setattr(R, "retrieve_job", lambda *a, **k: ("phase-A", a[0]))
    assert C.retrieve_any("local_cafe0001", aer_job["cfg"], run_dir=run) == ("phase-A", "local_cafe0001")
    with pytest.raises(ValueError, match="not a components job"):
        C.retrieve_components_job("local_cafe0001", aer_job["cfg"], run_dir=run)


def test_analysis_refuses_changed_components_and_unfinished_jobs(aer_job, tmp_path):
    from hsf2.hw.comp_analysis import analyze_jobs
    run = tmp_path / "copy"
    shutil.copytree(aer_job["run"], run)
    jd = run / "hw" / aer_job["res"]["jobs"][0]["job_dir"].name
    ok = analyze_jobs([jd], tmp_path / "ok")
    assert ok["records"][0]["component"] == aer_job["comp"].name and (tmp_path / "ok" / "components_hw_summary.json").exists()
    f = run / "components" / f"{aer_job['comp'].name}.json"
    f.write_text(f.read_text() + " ")                         # same content, different bytes: the plan no longer matches
    with pytest.raises(ValueError, match="changed since"):
        analyze_jobs([jd], tmp_path / "bad")
    f.write_text(f.read_text().rstrip(" "))
    man = json.loads((jd / "manifest.json").read_text())
    man["state"] = "submitted"
    (jd / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="not done"):
        analyze_jobs([jd], tmp_path / "bad2")
    with pytest.raises(ValueError, match="no job folder"):
        analyze_jobs([], tmp_path / "bad3")
    other = tmp_path / "elsewhere" / "hw" / "j"
    other.mkdir(parents=True)
    with pytest.raises(ValueError, match="different run folders"):
        analyze_jobs([jd, other], tmp_path / "bad4")


def test_manifest_records_only_the_credentials_basename(hand_run, tmp_path, monkeypatch):
    from hsf2.hw.components import run_components
    monkeypatch.delenv("HSF2_IBM_CREDS", raising=False)
    run = tmp_path / "tri"
    shutil.copytree(hand_run, run)
    secret = tmp_path / "private_dir" / "apikey_x.json"
    res = run_components(run, [by_name(run, 1)], "d", cfg_local(tmp_path, creds_file=str(secret)), sim_method="white-noise")
    man = res["jobs"][0]["manifest"]
    text = (res["jobs"][0]["job_dir"] / "manifest.json").read_text()
    assert man["creds"] == "apikey_x.json" and "private_dir" not in text and not secret.exists()
    assert not (tmp_path / "ledger.json").exists()           # a local job never touches the ledger


# ---------------------------------------------------------------------------- CLI
def snapshot_tree(root, skip):
    return {str(p.relative_to(root)) for p in root.rglob("*") if skip not in p.parents and p != skip}


def command_folders(run, command):
    """<run>/outputs/<stamp>_<command> folders (a second one in the same minute gets a _2, _3 suffix), oldest first."""
    pat = re.compile(r"^\d\d-\d\d-\d{4}-\d\d-\d\d_" + command + r"(_\d+)?$")
    return sorted((p for p in (Path(run) / "outputs").iterdir() if pat.match(p.name)), key=lambda p: p.stat().st_mtime)


def test_cli_list_dry_local_analyze_and_outputs_stay_inside_the_run_folder(synth_run, tmp_path, monkeypatch, capsys):
    from hsf2.cli import main
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))            # deployed next to Phase 1: run folders must lie in the work area
    monkeypatch.delenv("HSF2_IBM_CREDS", raising=False)
    run = tmp_path / "runs" / "tri"
    shutil.copytree(synth_run, run)
    outside = snapshot_tree(tmp_path, run)
    sel = ["--nmin", "10", "--nmax", "13", "--benchmark", "2", "--seed", "0"]
    # --list prints the selection and writes nothing
    capsys.readouterr()
    assert main(["hw-components", str(run), *sel, "--list"]) == 0
    out = capsys.readouterr().out
    assert "2 components: 0 hard, 2 benchmark (of 10 in the size range, 0 of them hard)" in out and "toy000000_s0_f1_c" in out
    assert not (run / "outputs").exists() and not (run / "hw").exists()
    # dry: plan files, estimate, no counts, no ledger
    ledger = tmp_path / "ledger.json"
    secret = tmp_path / "private_dir" / "apikey_cli.json"                    # never opened; only its name may be recorded
    assert main(["hw-components", str(run), *sel, "--mode", "dry", "--seeds", "1", "--ledger", str(ledger),
                 "--creds", str(secret)]) == 0
    jobs = sorted((run / "hw").glob("*_comp_d_*"))
    assert len(jobs) == 1 and not ledger.exists()
    man = json.loads((jobs[0] / "manifest.json").read_text())
    assert man["state"] == "dry" and man["step"] == "step3" and len(man["pubs"]) == 2 and man["estimate_s"] > 0
    assert man["creds"] == "apikey_cli.json" and "apikey_cli.json" in man["command"]
    assert "private_dir" not in (jobs[0] / "manifest.json").read_text()
    assert not any("private_dir" in p.read_text() for p in (run / "outputs").rglob("*.json*"))
    assert (jobs[0] / "snapshot_pre.json").exists() and not list(jobs[0].glob("*counts*"))
    cmd = command_folders(run, "hw_components")[0]
    assert (cmd / "commands.jsonl").exists() and (cmd / "config.json").exists() and (cmd / "selection.json").exists()
    # local (white-noise model): counts, then the automatic analysis in the command folder
    assert main(["hw-components", str(run), *sel, "--mode", "local", "--shots", "2000", "--seeds", "1", "--per-job", "1",
                 "--sim-method", "white-noise"]) == 0
    done = sorted(p for p in (run / "hw").glob("*_comp_d_*") if (p / "pub_0_counts.json").exists())
    assert len(done) == 2
    auto = [p for p in command_folders(run, "hw_components") if (p / "components_hw.csv").exists()]
    assert len(auto) == 1
    table = pd.read_csv(auto[0] / "components_hw.csv")
    assert len(table) == 2 and set(table["group"]) == {"benchmark"} and (table["shots"] == 2000).all()
    summ = json.loads((auto[0] / "components_hw_summary.json").read_text())
    assert summ["n_components"] == 2 and set(summ["groups"]) == {"benchmark", "all"} and len(summ["jobs"]) == 2
    # hw-components-analyze: the same analysis from the job folders, in its own command folder
    assert main(["hw-components-analyze", *map(str, done), "--seed", "1"]) == 0
    an = command_folders(run, "hw_components_analyze")
    assert len(an) == 1 and (an[0] / "commands.jsonl").exists()
    again = pd.read_csv(an[0] / "components_hw.csv")
    assert list(again["component"]) == list(table["component"]) and (again["E0"] == table["E0"]).all()
    # nothing was written outside the run folder
    assert snapshot_tree(tmp_path, run) == outside


def test_cli_errors_are_clean_and_hw_retrieve_serves_component_jobs(hand_run, tmp_path, monkeypatch, capsys):
    from hsf2.cli import main
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))
    monkeypatch.delenv("HSF2_IBM_CREDS", raising=False)
    run = tmp_path / "runs" / "tri"
    shutil.copytree(hand_run, run)
    for argv, msg in ((["--hard", "--benchmark", "2", "--nmin", "5", "--nmax", "8"], "cannot be combined"),
                      (["--nmin", "5", "--nmax", "30"], "nmax"), (["--nmin", "7", "--nmax", "8"], "no component selected"),
                      (["--nmin", "5", "--nmax", "8", "--benchmark", "9"], "only 2 ordinary")):
        with pytest.raises(SystemExit, match=msg):
            main(["hw-components", str(run), "--list", *argv])
    for argv, msg in ((["--per-job", "0"], "per-job"), (["--top-k", "-1"], "top-k"),
                      (["--no-wait", "--per-job", "2"], "single job")):             # 4 hard pieces: 2 jobs of 2
        with pytest.raises(SystemExit, match=msg):
            main(["hw-components", str(run), "--hard", "--nmin", "5", "--nmax", "8", "--mode", "dry", *argv])
    assert not (run / "outputs").exists() and not (run / "hw").exists()             # invalid options write nothing
    # a hard-only local run with per-job 3: jobs of 3 and 1 PUBs, hard group only
    assert main(["hw-components", str(run), "--hard", "--nmin", "5", "--nmax", "8", "--mode", "local", "--shots", "1000",
                 "--seeds", "1", "--per-job", "3", "--sim-method", "white-noise"]) == 0
    jobs = sorted((run / "hw").glob("*_comp_d_*"))
    assert [len(json.loads((j / "manifest.json").read_text())["pubs"]) for j in jobs] == [3, 1]
    csv = pd.read_csv([p for p in command_folders(run, "hw_components") if (p / "components_hw.csv").exists()][0] / "components_hw.csv")
    assert list(csv["hard"]) == [True] * 4 and set(csv["group"]) == {"hard"}
    # hw-retrieve of a local component job id (counts removed first): the component dispatch rebuilds them
    man = json.loads((jobs[0] / "manifest.json").read_text())
    (jobs[0] / "pub_0_counts.json").unlink()
    capsys.readouterr()
    assert main(["hw-retrieve", man["job_id"], "--run-dir", str(run), "--mode", "local"]) == 0
    assert (jobs[0] / "pub_0_counts.json").exists()
    with pytest.raises(FileNotFoundError):
        main(["hw-retrieve", "local_deadbeef", "--run-dir", str(run), "--mode", "local"])


def test_cli_submit_refused_by_budget_is_a_normal_stop(hand_run, tmp_path, monkeypatch, fast_polling):
    from hsf2.cli import main
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))
    monkeypatch.delenv("HSF2_IBM_CREDS", raising=False)
    run = tmp_path / "runs" / "tri"
    shutil.copytree(hand_run, run)
    svc, smp = mock_ibm(monkeypatch)
    ledger = tmp_path / "l.json"
    rc = main(["hw-components", str(run), "--hard", "--nmin", "5", "--nmax", "8", "--mode", "submit", "--seeds", "1",
               "--shots", "400", "--budget", "1", "--ledger", str(ledger)])
    assert rc == 1 and smp.calls == 0 and not ledger.exists()
    monkeypatch.setattr(sys, "stdin", io.StringIO("no" + chr(10)))          # anything but 'submit' stops, exit code 1
    rc = main(["hw-components", str(run), "--hard", "--nmin", "5", "--nmax", "8", "--mode", "submit", "--seeds", "1",
               "--shots", "400", "--ledger", str(ledger)])
    assert rc == 1 and smp.calls == 0

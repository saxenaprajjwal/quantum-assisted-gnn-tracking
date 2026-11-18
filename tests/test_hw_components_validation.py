"""Independent validation of the step-3 code (hsf2.hw.components, hsf2.hw.comp_analysis, CLI hw-components /
hw-components-analyze), written by the validation engineer, not by the implementer.

0 QPU s, IBM is never contacted: numpy / Aer references, fake backends, mocked service and sampler.  Everything physical is
re-derived here (QUBO -> Ising, p = 1 QAOA, steepest descent, greedy-S, estimator formula), not taken from hsf2.  The file is
self-contained (no import from the other test modules) so that it also runs as hsf2_tests/test_hsf2_hw_components_validation.py
on lxplus; every test that goes through the CLI sets HSF2_WORK to its temporary folder and keeps its run folders inside it.
Instances have at most 10 variables, Aer cases at most 500 shots.
"""
import builtins
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

EVENT = "val"
SHOTS = 400


# ---------------------------------------------------------------------------------------------- independent references
def bits_table(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)


def brute(n, linear, quad):
    """E[index] of  linear.x + sum b_ij x_i x_j, bit i of the index = variable i."""
    x = bits_table(n)
    E = x @ np.asarray(linear, float)
    for i, j, b in quad:
        E = E + b * x[:, i] * x[:, j]
    return E


def ref_descent(E, n, start):
    """Steepest single-flip descent from the energy table only: lowest dE, ties lowest bit, stop below 1e-12."""
    x = int(start)
    while True:
        d = [E[x ^ (1 << k)] - E[x] for k in range(n)]
        k = int(np.argmin(d))
        if d[k] >= -1e-12:
            return x
        x ^= 1 << k


def ref_greedy(n, quad):
    """Greedy-S without holes: weight = sum of -b over the quadruplet couplings (b < 0) of a triplet, decreasing, ties lowest
    index; add a triplet unless a selected one conflicts (b > 0)."""
    w = np.zeros(n)
    nbr = [set() for _ in range(n)]
    for i, j, b in quad:
        if b < 0:
            w[i] += -b
            w[j] += -b
        elif b > 0:
            nbr[i].add(j)
            nbr[j].add(i)
    chosen = np.zeros(n, bool)
    for i in sorted(range(n), key=lambda t: (-w[t], t)):
        if not any(chosen[j] for j in nbr[i]):
            chosen[i] = True
    return int(sum(1 << i for i in range(n) if chosen[i]))


def ref_ising(linear, quad):
    """z = 1 - 2x:  h_i = -a_i/2 - sum_j b_ij/4,  J_ij = b_ij/4 (i < j),  offset = sum a/2 + sum b/4."""
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
    """p = 1 QAOA: |+>^n, exp(-i gamma H / scale), exp(-i beta sum X); bit k of the index = qubit k."""
    n = len(h)
    z = 1.0 - 2.0 * bits_table(n)
    ph = z @ (h / scale) + np.einsum("ij,ij->i", z @ (J / scale), z)
    psi = np.exp(-1j * gamma * ph) / math.sqrt(1 << n)
    c, s = math.cos(beta), -1j * math.sin(beta)
    for k in range(n):
        psi = psi.reshape(1 << (n - k - 1), 2, 1 << k)
        psi = np.stack([c * psi[:, 0] + s * psi[:, 1], s * psi[:, 0] + c * psi[:, 1]], axis=1).reshape(-1)
    return np.abs(psi) ** 2


def reverse_index(n):
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(1 << n)])


def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def rand_instance(n, seed, linear=False, quad_p=0.3, conf_p=0.35):
    rng = np.random.default_rng(seed)
    quad = []
    for i in range(n):
        for j in range(i + 1, n):
            r = rng.random()
            if r < quad_p:
                quad.append((i, j, -round(float(rng.uniform(0.25, 1.0)), 3)))
            elif r < quad_p + conf_p:
                quad.append((i, j, 1.0))
    lin = [round(float(v), 3) for v in rng.normal(0, 0.2, n)] if linear else [0.0] * n
    true = [bool(v) for v in (rng.random(n) < 0.6)]
    return dict(n=n, linear=lin, quad=quad, true=true)


def classify(spec):
    """(exact energy, energy of descent from 'everything selected', energy of greedy-S + descent), independently."""
    n = spec["n"]
    E = brute(n, spec["linear"], spec["quad"])
    return (float(E.min()), float(E[ref_descent(E, n, (1 << n) - 1)]),
            float(E[ref_descent(E, n, ref_greedy(n, spec["quad"]))]))


def _search(count, sizes, pattern, start=0):
    """Deterministic search for instances whose two descents behave as `pattern`: 'hard' (both miss), 'easy' (both reach),
    'all_only' (only the descent from all-selected misses), 'greedy_only' (only greedy-S + descent misses)."""
    out, seed = [], start
    while len(out) < count:
        spec = rand_instance(sizes[seed % len(sizes)], seed)
        seed += 1
        if not any(b < 0 for *_, b in spec["quad"]):
            continue
        e0, d_all, d_gr = classify(spec)
        miss_all, miss_gr = d_all > e0 + 1e-9, d_gr > e0 + 1e-9
        if (miss_all, miss_gr) == {"hard": (True, True), "easy": (False, False), "all_only": (True, False),
                                   "greedy_only": (False, True)}[pattern]:
            out.append(spec)
    return out


HARD = _search(4, (6, 7, 8, 9), "hard")
EASY = _search(8, (5, 6, 7, 8, 9), "easy")
ONE_MISS = _search(1, (6, 7, 8, 9), "all_only") + _search(1, (6, 7, 8, 9), "greedy_only")


def write_run(path, specs, event=EVENT):
    """A triplet-run-like folder (components.csv + components/<event>_s0_f1_c<id>.json) from instance specs; every number of
    the CSV row is computed here (brute force, own descent, own greedy-S)."""
    path = Path(path)
    (path / "components").mkdir(parents=True)
    rows = []
    for cid, spec in enumerate(specs):
        n, lin, quad = spec["n"], spec["linear"], spec["quad"]
        E = brute(n, lin, quad)
        e0, k = float(E.min()), int(np.argmin(E))
        gs = ref_greedy(n, quad)
        rows.append({"event": event, "sector": 0, "phi0": 0.0, "dphi": 6.28, "frac": 1.0, "component": cid, "n_variables": n,
                     "n_quadruplet_couplings": sum(b < 0 for *_, b in quad), "n_conflicts": sum(b > 0 for *_, b in quad),
                     "n_true_triplets": int(sum(spec["true"])), "n_fake_triplets": n - int(sum(spec["true"])),
                     "exact_energy": e0, "holefree_energy": float(E[(1 << n) - 1]), "greedy_energy": float(E[gs]),
                     "descent_energy": float(E[ref_descent(E, n, (1 << n) - 1)]),
                     "greedy_descent_energy": float(E[ref_descent(E, n, gs)]), "trivial": False, "holefree_optimal": False})
        doc = {"meta": {"event": event, "sector": 0, "phi0": 0.0, "dphi": 6.28, "frac": 1.0, "component": cid, "alpha": 0.0,
                        "zeta": 1.0},
               "triplets": [[3 * t, 3 * t + 1, 3 * t + 2] for t in range(n)],
               "triplets_hit_id": [[t, t + 1, t + 2] for t in range(n)], "linear": list(lin),
               "quadratic": [[i, j, b] for i, j, b in quad], "exact_energy": e0,
               "exact_selection": [(k >> b) & 1 for b in range(n)], "true_triplet": list(spec["true"])}
        (path / "components" / f"{event}_s0_f1_c{cid}.json").write_text(json.dumps(doc))
    pd.DataFrame(rows).to_csv(path / "components.csv", index=False)
    return path


def cfg_of(tmp_path, **kw):
    from hsf2.hw.config import HwConfig
    d = dict(mode="local", shots=SHOTS, transpile_seeds=1, ledger=str(tmp_path / "ledger.json"), budget_s=400.0)
    d.update(kw)
    return HwConfig(**d)


def pub_of(comp, shots, top_k=None, k=0, F=0.5):
    """The manifest PUB entry that analyse_pub needs (Phase-1 closed-form angles of the component's own Ising)."""
    from hsf2.hw.components import component_angles, component_ising
    ism, scale, g, b, info = component_angles(component_ising(comp.doc), top_k)
    return {"k": k, "shots": shots, "gammas": g, "betas": b, "scale": scale, "ideal_mean_energy": info["ideal_mean_energy"],
            "n_couplings": info["n_couplings"], "n_2q": 100, "fidelity_est": F}


def to_counts(c):
    return {int(i): int(v) for i, v in zip(np.nonzero(c)[0], c[c > 0])}


def isa_probs(isa, values):
    """EXACT output distribution of a transpiled (ISA) circuit, from the circuit alone: the measure instructions give the
    qubit of every classical bit, the gates are applied by Aer's statevector, and save_probabilities returns the
    probabilities of those qubits, in classical-bit order (index bit k = classical bit k).  Returns (p, measured qubits)."""
    import qiskit_aer  # noqa: F401  (adds QuantumCircuit.save_probabilities)
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    bound = isa if values is None else isa.assign_parameters(dict(zip(isa.parameters, [float(v) for v in values])))
    pairs = sorted((bound.find_bit(c).index, bound.find_bit(q).index) for ins in bound.data
                   if ins.operation.name == "measure" for q, c in zip(ins.qubits, ins.clbits))
    assert [c for c, _ in pairs] == list(range(bound.num_clbits))
    mq = [q for _, q in pairs]
    body = QuantumCircuit(bound.num_qubits)
    for ins in bound.data:
        if ins.operation.name in ("measure", "barrier"):
            continue
        body.append(ins.operation, [bound.find_bit(q).index for q in ins.qubits])
    body.save_probabilities(mq, label="p")
    res = AerSimulator(method="statevector").run(body, shots=1).result()
    assert res.success
    return np.array(res.data(0)["p"]), mq


# ---------------------------------------------------------------------------------------------- mocks of the IBM side
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
            return {"usage": {"status": "pending"}}
        return {"usage": {"status": "finished", "qpu_charge_time_seconds": self.charge}}

    def properties(self):
        return None


def uniform_counts(isa, values, shots, rng):
    nb = isa.num_clbits
    c = rng.multinomial(shots, np.ones(1 << nb) / (1 << nb))
    return {format(i, f"0{nb}b"): int(v) for i, v in enumerate(c) if v}


def exact_counts(isa, values, shots, rng):
    """IBM-style bit strings (classical bit 0 = rightmost character) drawn from the exact distribution of the submitted ISA."""
    nb = isa.num_clbits
    p, _ = isa_probs(isa, values)
    c = rng.multinomial(shots, p / p.sum())
    return {format(i, f"0{nb}b"): int(v) for i, v in enumerate(c) if v}


class MockSampler:
    """Records every run(); answers each PUB with counts from `maker(isa, values, shots, rng)`."""

    def __init__(self, svc, charge=None, maker=uniform_counts):
        self.svc, self.calls, self.options = svc, 0, SimpleNamespace()
        self.pubs_seen, self.jobs, self.charge, self.maker, self.make_args = [], {}, charge, maker, []

    def run(self, pubs, **kw):
        self.calls += 1
        self.svc.consumed += self.svc.step
        self.pubs_seen.append([(p[0].num_clbits, p[2]) for p in pubs])
        rng = np.random.default_rng(self.calls)
        results = []
        for isa, values, shots in pubs:
            counts = self.maker(isa, values, shots, rng)
            results.append(SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda c=counts: c))))
        job = FakeJob(f"fakejob{self.calls:04d}", results, self.charge)
        self.jobs[job.job_id()] = job
        return job


def mock_ibm(monkeypatch, charge=None, maker=uniform_counts, via_session=None):
    """Nothing leaves the process.  via_session=list: the REAL run.get_job_backend runs, with session.get_service /
    get_backend replaced (the credentials path it receives is appended to the list); else get_job_backend itself is replaced."""
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from hsf2.hw import run as R
    from hsf2.hw import session
    svc = FakeService()
    smp = MockSampler(svc, charge, maker)
    svc.job = lambda job_id: smp.jobs[job_id]
    if via_session is None:
        monkeypatch.setattr(R, "get_job_backend", lambda cfg, setting, service=None: (FakeFez(), svc))
    else:
        monkeypatch.setattr(session, "get_service", lambda creds_file=None: (via_session.append(creds_file), svc)[1])
        monkeypatch.setattr(session, "get_backend", lambda service, name, fractional=False: FakeFez())

    def make_sampler(backend, primitive, setting, shots, max_exec, strict=True):
        smp.make_args.append({"shots": shots, "max_exec": max_exec})
        return smp, {"mock": True}
    monkeypatch.setattr(R, "make_sampler", make_sampler)
    return svc, smp


@pytest.fixture
def fast_polling(monkeypatch):
    from hsf2.hw import run as R
    for k, v in (("USAGE_TIMEOUT_S", 0.05), ("USAGE_INTERVAL_S", 0.005), ("ACCOUNT_TIMEOUT_S", 0.05),
                 ("ACCOUNT_INTERVAL_S", 0.005)):
        monkeypatch.setattr(R, k, v)


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """HSF2_WORK = tmp_path (the deployed code refuses run folders outside it), no credentials, a private HOME so that no
    default ledger of the user can be read or written."""
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))
    for k in ("HSF2_IBM_CREDS", "HSF2_QPU_LEDGER", "HSF2_QPU_BUDGET"):
        monkeypatch.delenv(k, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return tmp_path


def tree(root, skip=()):
    return {str(p.relative_to(root)) for p in Path(root).rglob("*") if not any(s in p.parents or p == s for s in skip)}


# ============================================================================================== 1. selection
def test_hard_flag_is_the_documented_formula_on_independent_rows_and_boundaries(tmp_path):
    from hsf2.hw import components as C
    assert len(HARD) == 4 and len(EASY) == 8 and len(ONE_MISS) == 2
    specs = HARD + EASY + ONE_MISS
    run = write_run(tmp_path / "tri", specs)
    comps = C.load_components(run)
    assert len(comps) == len(specs)
    for c, spec in zip(comps, specs):
        e0, d_all, d_gr = classify(spec)
        assert c.row["exact_energy"] == pytest.approx(e0, abs=1e-12)
        assert c.hard == (min(d_all, d_gr) > e0 + 1e-9)                 # both descents must miss the optimum
    assert [c.hard for c in comps] == [True] * 4 + [False] * 10        # a piece that ONE descent solves is ordinary
    # boundaries of the formula min(descent, greedy_descent) > exact + 1e-9
    r = dict(comps[0].row)
    e0 = r["exact_energy"]
    cases = [(e0, e0, False), (e0, e0 + 5.0, False), (e0 + 5.0, e0, False), (e0 + 1e-9, e0 + 1e-9, False),
             (e0 + 1.5e-9, e0 + 1.5e-9, True), (e0 + 1e-9, e0 + 3e-9, False), (e0 + 0.5, e0 + 0.7, True)]
    for d, g, want in cases:
        assert C.is_hard(dict(r, descent_energy=d, greedy_descent_energy=g)) is want, (d - e0, g - e0)
    for key in ("descent_energy", "greedy_descent_energy", "exact_energy"):
        with pytest.raises(ValueError, match="no exact / descent energy"):
            C.is_hard(dict(r, **{key: None}))
    # the size range is inclusive at both ends, and only exported components (n <= 24) are loaded
    hard_sizes = sorted({c.n for c in comps if c.hard})
    lo, hi = hard_sizes[0], hard_sizes[-1]
    sel = C.select_components(comps, lo, hi, hard_only=True)
    assert {c.name for c in sel} == {c.name for c in comps if c.hard and lo <= c.n <= hi} and {c.n for c in sel} >= {lo, hi}
    sel1 = C.select_components(comps, lo, lo, hard_only=True)
    assert sel1 and {c.n for c in sel1} == {lo} and all(c.hard for c in sel1)
    df = pd.read_csv(run / "components.csv")
    big = pd.concat([df, pd.DataFrame([dict(df.iloc[0], component=90 + k, n_variables=25 + 5 * k, exact_energy=np.nan,
                                            descent_energy=np.nan, greedy_descent_energy=np.nan) for k in range(2)])])
    big.to_csv(run / "components.csv", index=False)
    assert [c.name for c in C.load_components(run)] == [c.name for c in comps]    # n = 25, 30: not exported, not an error
    for kw in (dict(nmax=25), dict(nmin=1), dict(nmin=9, nmax=8)):
        with pytest.raises(ValueError):
            C.select_components(comps, **dict(dict(nmin=5, nmax=9), **kw))
    big.loc[big.component == 90, "n_variables"] = 24                              # a listed n <= 24 component needs its file
    big.to_csv(run / "components.csv", index=False)
    with pytest.raises(FileNotFoundError, match="c90"):
        C.load_components(run)


def test_benchmark_sample_is_seeded_reproducible_disjoint_from_hard_and_max_never_reduces_it(cli_env, capsys):
    from hsf2.cli import main
    from hsf2.hw import components as C
    run = write_run(cli_env / "runs" / "tri", HARD + EASY)
    comps = C.load_components(run)
    hard_names = {c.name for c in comps if c.hard}
    ordinary = [c.name for c in comps if not c.hard]
    assert len(hard_names) == 4 and len(ordinary) == 8
    seen, samples = set(), set()
    for seed in range(30):
        a = C.select_components(comps, 5, 9, n_benchmark=3, seed=seed)
        assert [c.name for c in a] == [c.name for c in C.select_components(comps, 5, 9, n_benchmark=3, seed=seed)]
        bench = [c for c in a if c.group == "benchmark"]
        assert [c.group for c in a] == ["hard"] * 4 + ["benchmark"] * 3 and {c.name for c in a if c.group == "hard"} == hard_names
        assert len({c.name for c in bench}) == 3 and {c.name for c in bench} <= set(ordinary)        # disjoint from the hard set
        assert not any(c.hard for c in bench) and [c.name for c in bench] == sorted((c.name for c in bench), key=ordinary.index)
        seen |= {c.name for c in bench}
        samples.add(tuple(c.name for c in bench))
        # --max keeps the whole benchmark sample (the same pieces) and a seeded subset of the hard pieces
        m = C.select_components(comps, 5, 9, n_benchmark=3, seed=seed, max_total=5)
        assert [c.name for c in m if c.group == "benchmark"] == [c.name for c in bench]
        mh = [c.name for c in m if c.group == "hard"]
        assert len(mh) == 2 and set(mh) <= hard_names
        assert mh == [c.name for c in C.select_components(comps, 5, 9, n_benchmark=3, seed=seed, max_total=5) if c.group == "hard"]
        # the benchmark never leaves the size range
        r = C.select_components(comps, 6, 8, n_benchmark=2, seed=seed)
        assert all(6 <= c.n <= 8 for c in r)
    assert seen == set(ordinary) and len(samples) >= 5                       # a random sample: every piece can be drawn
    assert len({tuple(c.name for c in C.select_components(comps, 5, 9, hard_only=True, max_total=2, seed=s)) for s in range(30)}) >= 3
    with pytest.raises(ValueError, match="smaller than"):
        C.select_components(comps, 5, 9, n_benchmark=3, max_total=2)
    with pytest.raises(ValueError, match="cannot be combined"):
        C.select_components(comps, 5, 9, hard_only=True, n_benchmark=1)
    # --hard refuses --benchmark at the command line as well, and --list writes nothing
    capsys.readouterr()
    with pytest.raises(SystemExit, match="cannot be combined"):
        main(["hw-components", str(run), "--hard", "--benchmark", "2", "--nmin", "5", "--nmax", "9", "--list"])
    assert main(["hw-components", str(run), "--nmin", "5", "--nmax", "9", "--benchmark", "2", "--max", "5", "--list"]) == 0
    out = capsys.readouterr().out
    assert "5 components: 3 hard, 2 benchmark (of 12 in the size range, 4 of them hard)" in out
    assert not (run / "outputs").exists() and not (run / "hw").exists()


def test_analysis_refuses_a_component_json_that_changed_after_planning(tmp_path):
    from hsf2.hw import components as C
    from hsf2.hw.comp_analysis import analyze_jobs
    run = write_run(tmp_path / "tri", [EASY[0]])
    comps = C.load_components(run)
    C.run_components(run, comps, "d", cfg_of(tmp_path), per_job=1, sim_method="white-noise")
    jd = next((run / "hw").iterdir())
    assert analyze_jobs([jd], tmp_path / "ok")["records"][0]["component"] == comps[0].name
    f = run / "components" / f"{comps[0].name}.json"
    good = f.read_bytes()
    f.write_bytes(good + b" ")                                                  # one byte more: not the planned component
    with pytest.raises(ValueError, match="changed since"):
        analyze_jobs([jd], tmp_path / "bad1")
    doc = json.loads(good)
    doc["quadratic"][0][2] *= 0.99                                              # a real change of one coupling
    f.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        analyze_jobs([jd], tmp_path / "bad2")                                   # sha1 (and the CSV cross-check) refuse it
    f.write_bytes(good)
    assert analyze_jobs([jd], tmp_path / "ok2")["records"][0]["component"] == comps[0].name


# ============================================================================================== 2. physics
@pytest.mark.parametrize("n,seed,linear", [(3, 1, False), (5, 2, True), (8, 3, True), (10, 4, False), (10, 5, True)])
def test_qubo_to_ising_energy_for_every_bit_string_including_offset(tmp_path, n, seed, linear):
    from hsf2.hw import components as C
    spec = rand_instance(n, 900 + seed, linear=linear, quad_p=0.4, conf_p=0.4)
    run = write_run(tmp_path / "tri", [spec])
    comp = C.load_components(run)[0]
    lin, quad = comp.linear, comp.quad
    E = brute(n, lin, quad)
    h, J, off = ref_ising(lin, quad)
    ising = C.component_ising(comp.doc)
    assert np.allclose(ising.h, h, atol=1e-12) and np.allclose(np.triu(ising.J, 1), J, atol=1e-12) and abs(ising.offset - off) < 1e-12
    assert abs(off - (0.5 * lin.sum() + 0.25 * sum(b for *_, b in quad))) < 1e-12
    X = bits_table(n)
    z = 1.0 - 2.0 * X
    e_ref_ising = z @ h + np.einsum("ij,ij->i", z @ J, z) + off
    assert np.abs(e_ref_ising - E).max() < 1e-12                               # my Ising reproduces the QUBO energy of every state
    assert np.abs(ising.energy_bits(X) - E).max() < 1e-12                      # and so does the hsf2 / Phase-1 Ising
    assert np.abs(C.component_energies(comp.doc) - E).max() < 1e-12            # hsf2's enumeration, index = sum x_i 2^i
    back = C.ising_to_qubo(ising)                                              # the inverse map (circuit Hamiltonian) is exact
    assert np.abs(C.component_energies(dict(comp.doc, linear=back.L.tolist(), quadratic=[
        [i, j, back.W[i, j]] for i, j in zip(*np.nonzero(back.W))])) + back.const - E).max() < 1e-11


def test_recorded_angles_are_the_p1_optimum_of_the_components_own_ising_and_scale_is_the_full_one(tmp_path):
    from hsf2.hw import components as C
    run = write_run(tmp_path / "tri", [rand_instance(6, 11), rand_instance(7, 12, linear=True)])
    for comp in C.load_components(run):
        n = comp.n
        h, J, _ = ref_ising(comp.linear, comp.quad)
        ism, scale, gam, bet, info = C.component_angles(C.component_ising(comp.doc))
        scale_ref = max(np.abs(h).max(), np.abs(J).max())
        assert scale == pytest.approx(scale_ref, rel=1e-12) and info["n_couplings"] == len(comp.quad)
        E = brute(n, comp.linear, comp.quad)
        e_rec = float(ref_probs(h, J, scale_ref, gam[0], bet[0]) @ E)
        assert e_rec == pytest.approx(info["ideal_mean_energy"], abs=1e-9)    # the closed form of the angle search is the statevector
        grid = min(float(ref_probs(h, J, scale_ref, g, b) @ E) for g in np.linspace(-math.pi, math.pi, 41)
                   for b in np.linspace(-math.pi / 2, math.pi / 2, 21, endpoint=False))
        assert e_rec <= grid + 1e-9                                            # no grid point beats the recorded optimum
        assert e_rec < E.mean() - 0.1                                          # and QAOA is better than uniform sampling


@pytest.fixture(scope="module")
def planned_job(tmp_path_factory):
    """One dry job of 5 components (6-10 variables, two with a linear term) on FakeFez, planned by run_components."""
    from hsf2.hw import components as C
    tmp = tmp_path_factory.mktemp("planned")
    specs = [rand_instance(6, 21), rand_instance(7, 22, linear=True), rand_instance(8, 23), rand_instance(9, 24, linear=True),
             rand_instance(10, 25)]
    run = write_run(tmp / "tri", specs)
    comps = C.load_components(run)
    res = C.run_components(run, comps, "d", cfg_of(tmp, mode="dry"), per_job=5)
    return {"run": run, "comps": comps, "res": res, "tmp": tmp, "specs": specs}


def test_isa_distribution_equals_independent_numpy_p1_for_every_pub_with_nontrivial_layouts(planned_job):
    job = planned_job["res"]["jobs"][0]["job"]
    man = planned_job["res"]["jobs"][0]["manifest"]
    assert [p.comp.name for p in job.plans] == [c.name for c in planned_job["comps"]] and len(man["pubs"]) == 5
    permuted = 0
    for plan, mp in zip(job.plans, man["pubs"]):
        comp, pub = plan.comp, plan.pub
        n = comp.n
        h, J, _ = ref_ising(comp.linear, comp.quad)
        scale_ref = max(np.abs(h).max(), np.abs(J).max())
        assert mp["scale"] == pytest.approx(scale_ref, rel=1e-12) and mp["component"] == comp.name and mp["n"] == n
        g, b = mp["gammas"][0], mp["betas"][0]                                  # the recorded angles ...
        p_ref = ref_probs(h, J, scale_ref, g, b)                                # ... through the independent numpy p = 1 QAOA
        p_isa, mq = isa_probs(pub.isa, pub.values)                              # ... and what the transpiled circuit really does
        assert abs(p_isa.sum() - 1.0) < 1e-9 and tv(p_isa, p_ref) < 1e-9, (comp.name, tv(p_isa, p_ref))
        # bit k of an outcome is variable k, whatever the layout: the reversed order is a clearly different distribution
        assert tv(p_ref, p_ref[reverse_index(n)]) > 0.05 and tv(p_isa, p_ref[reverse_index(n)]) > 0.05
        lay = pub.layout
        assert lay["measured"]["qubits"] == mq and lay["measured"]["clbits"] == list(range(n))      # layout file = the circuit
        assert mp["measured_qubits"] == mq and len(set(mq)) == n
        assert lay["final"] == mq                                               # the fallback of component_layout_record assumes this
        permuted += lay["initial"] != lay["final"]
        E = brute(n, comp.linear, comp.quad)
        assert float(p_ref @ E) == pytest.approx(mp["ideal_mean_energy"], abs=1e-9)
    assert permuted >= 1                                                        # routing really permuted qubits (not a trivial layout)


def test_fractional_setting_isa_distribution_equals_numpy_p1_and_rzz_angles_are_in_range(tmp_path):
    from hsf2.hw import components as C
    from hsf2.hw.transpile import check_fractional_isa
    try:
        from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle  # noqa: F401
    except ImportError:
        pytest.skip("FoldRzzAngle not available in this qiskit-ibm-runtime")
    run = write_run(tmp_path / "tri", [rand_instance(6, 71), rand_instance(7, 72, linear=True)])
    comps = C.load_components(run)
    res = C.run_components(run, comps, "e", cfg_of(tmp_path, mode="dry"), per_job=2)
    man = res["jobs"][0]["manifest"]
    assert man["setting"]["fractional"] and man["setting"]["options"]["twirling.enable_gates"] is False
    for plan, mp in zip(res["jobs"][0]["job"].plans, man["pubs"]):
        angles = check_fractional_isa(plan.pub.isa)                                  # rzz present, every angle in (0, pi/2]
        assert plan.pub.values is None and angles and all(0.0 < a <= math.pi / 2 + 1e-9 for a in angles)
        h, J, _ = ref_ising(plan.comp.linear, plan.comp.quad)
        p_ref = ref_probs(h, J, mp["scale"], mp["gammas"][0], mp["betas"][0])
        p_isa, mq = isa_probs(plan.pub.isa, None)                                    # the folded circuit still is the same QAOA
        assert tv(p_isa, p_ref) < 1e-9 and mq == plan.pub.layout["measured"]["qubits"]


def test_counts_to_variable_index_for_every_pub_of_a_multi_pub_submit_job(planned_job, tmp_path, monkeypatch, fast_polling):
    """The sampler mock answers each submitted ISA with IBM-style bit strings drawn from that circuit's exact distribution;
    the counts files and the analysis must see variable i at bit i in every PUB of the job."""
    from hsf2.hw import components as C
    run = tmp_path / "tri"
    shutil.copytree(planned_job["run"], run, ignore=shutil.ignore_patterns("hw", "outputs"))
    comps = C.load_components(run)
    svc, smp = mock_ibm(monkeypatch, maker=exact_counts)
    S = 3000
    res = C.run_components(run, comps, "d", cfg_of(tmp_path, mode="submit", backend="ibm_fake", shots=S), per_job=5,
                           confirm=lambda m: None, out_dir=tmp_path / "out")
    assert smp.calls == 1 and [s for _, s in smp.pubs_seen[0]] == [S] * 5 and [nb for nb, _ in smp.pubs_seen[0]] == [c.n for c in comps]
    jd = res["jobs"][0]["job_dir"]
    listed = json.loads((jd / "components.json").read_text())["pubs"]
    recs = {r["component"]: r for r in res["analysis"]["records"]}
    for k, comp in enumerate(comps):
        assert (listed[k]["pub"], listed[k]["component"]) == (k, comp.name)
        cf = json.loads((jd / f"pub_{k}_counts.json").read_text())
        assert cf["component"] == comp.name and cf["n"] == comp.n and cf["total"] == S
        n = comp.n
        h, J, _ = ref_ising(comp.linear, comp.quad)
        mp = res["jobs"][0]["manifest"]["pubs"][k]
        p = ref_probs(h, J, mp["scale"], mp["gammas"][0], mp["betas"][0])
        idx = np.array([int(i) for i in cf["counts"]])
        cnt = np.array(list(cf["counts"].values()))
        rev = reverse_index(n)
        ll = lambda q: float(cnt @ np.log(np.maximum(q[idx], 1e-300)))          # noqa: E731
        assert ll(p) - ll(p[rev]) > 20.0                                       # the counts follow p, not its bit reversal
        rec = recs[comp.name]
        E = brute(n, comp.linear, comp.quad)
        se_x = math.sqrt(float(np.var(np.repeat(p[idx], cnt))) / S) / (float(p @ p) - 1.0 / (1 << n))   # standard errors of the
        se_e = math.sqrt(float(np.var(np.repeat(E[idx], cnt))) / S) / (float(E.mean()) - float(p @ E))  # two fidelity estimators
        assert rec["pub"] == k and abs(rec["hardware"]["F_XEB"] - 1.0) < 5 * se_x + 1e-9 and se_x < 0.2
        assert abs(rec["hardware"]["F_E"] - 1.0) < 5 * se_e + 1e-9 and se_e < 0.2        # noiseless sampler: F = 1
        # the best sampled state, recomputed from the counts file with the brute-force energies, and its true / fake content
        E = brute(n, comp.linear, comp.quad)
        order = sorted(range(len(idx)), key=lambda t: (E[idx[t]], -cnt[t], idx[t]))
        best = int(idx[order[0]])
        x = [(best >> i) & 1 for i in range(n)]
        t = rec["tracking"]["hw_best"]
        assert rec["hardware"]["best_index"] == best and t["energy"] == pytest.approx(float(E[best]))
        assert t["n_true"] == sum(1 for i in range(n) if x[i] and comp.doc["true_triplet"][i])
        assert t["n_fake"] == sum(1 for i in range(n) if x[i] and not comp.doc["true_triplet"][i])


def test_local_aer_path_noiseless_counts_follow_the_ideal_distribution_for_every_pub(tmp_path, monkeypatch):
    from qiskit_aer import AerSimulator
    from hsf2.hw import components as C
    from hsf2.hw import fake as F
    monkeypatch.setattr(F, "local_simulator", lambda backend, method="matrix_product_state": AerSimulator(method="statevector"))
    specs = [rand_instance(6, 31), rand_instance(7, 32, linear=True), rand_instance(8, 33)]
    run = write_run(tmp_path / "tri", specs)
    comps = C.load_components(run)
    shots = 500
    res = C.run_components(run, comps, "d", cfg_of(tmp_path, shots=shots), per_job=3, sim_method="statevector", seed=5,
                           out_dir=tmp_path / "out")
    man = res["jobs"][0]["manifest"]
    assert man["sim_methods"] == ["statevector"] * 3
    for k, comp in enumerate(comps):
        cf = json.loads((res["jobs"][0]["job_dir"] / f"pub_{k}_counts.json").read_text())
        h, J, _ = ref_ising(comp.linear, comp.quad)
        p = ref_probs(h, J, man["pubs"][k]["scale"], man["pubs"][k]["gammas"][0], man["pubs"][k]["betas"][0])
        idx = np.array([int(i) for i in cf["counts"]])
        cnt = np.array(list(cf["counts"].values()))
        rev = reverse_index(comp.n)
        assert cnt.sum() == shots and idx.max() < 1 << comp.n
        ll = lambda q: float(cnt @ np.log(np.maximum(q[idx], 1e-300)))          # noqa: E731
        assert ll(p) - ll(p[rev]) > 10.0 and ll(p) - ll(np.full(p.size, 1.0 / p.size)) > 5.0
    assert all(r["hardware"]["F_XEB"] > 0.3 for r in res["analysis"]["records"])


def test_degenerate_ground_states_p_ground_reaches_e0_and_tolerance(tmp_path, caplog):
    """Three interchangeable optima {0,1}, {2,3}, {4,5} (E = -1): g = 3 and P(ground) is their total probability."""
    from hsf2.hw import components as C
    from hsf2.hw.comp_analysis import analyse_pub
    quad = [(0, 1, -1.0), (2, 3, -1.0), (4, 5, -1.0)] + [(i, j, 1.0) for i in range(6) for j in range(i + 1, 6) if i // 2 != j // 2]
    spec = dict(n=6, linear=[0.0] * 6, quad=sorted(quad), true=[True, True, False, False, True, False])
    run = write_run(tmp_path / "tri", [spec])
    comp = C.load_components(run)[0]
    E = brute(6, spec["linear"], spec["quad"])
    assert comp.doc["exact_selection"] == [1, 1, 0, 0, 0, 0] and (E <= E.min() + 1e-9).sum() == 3 and E.min() == -1.0
    S = 1000
    pub = pub_of(comp, S)
    ground = {0b000011: 50, 0b001100: 100, 0b110000: 300}
    counts = {**ground, 0b000001: 200, 0: 350}                                  # {0}: E = 0 ; empty selection: E = 0
    rec = analyse_pub(comp, counts, pub, None, seed=0)
    hw = rec["hardware"]
    assert rec["ground_degeneracy"] == 3 and rec["E0"] == -1.0 and rec["p_unif_ground"] == 3 / 64
    assert hw["p_ground"] == 0.45 and hw["best_reaches_E0"] and rec["p_hw_over_p_unif"] == pytest.approx(0.45 / (3 / 64))
    assert hw["best_index"] == 0b110000 and hw["best_energy"] == -1.0           # the most frequent of the three optima
    t = rec["tracking"]["hw_best"]
    assert t["is_ground_state"] and not t["is_exact_selection"] and (t["n_true"], t["n_fake"]) == (1, 1)
    assert rec["tracking"]["exact"]["is_exact_selection"]
    # the descent statistics, recomputed from the energy table: every start counted once per shot, g = 3 endpoints count
    reach = {s: float(E[ref_descent(E, 6, s)]) <= E.min() + 1e-9 for s in counts}
    assert hw["desc_frac_E0"] == pytest.approx(sum(v for s, v in counts.items() if reach[s]) / S)
    assert hw["desc_any_E0"] == any(reach.values())
    # a state 5e-10 above the optimum is still a ground state (tolerance 1e-9), one 2e-9 above is not
    for delta, g_expected in ((5e-10, 3), (2e-9, 2)):
        lin = [0.0] * 6
        lin[2] = delta                                                          # raises every state that selects variable 2
        run2 = write_run(tmp_path / f"tri_{g_expected}", [dict(spec, linear=lin)])
        comp2 = C.load_components(run2)[0]
        rec2 = analyse_pub(comp2, counts, pub_of(comp2, S), None, seed=0)
        assert rec2["ground_degeneracy"] == g_expected, delta
        assert rec2["hardware"]["p_ground"] == pytest.approx((450 if g_expected == 3 else 350) / S)
    # the greedy-S + descent rebuilt from the export is compared with components.csv, a mismatch is visible, not silent
    bad = C.Component(comp.name, comp.path, comp.sha1, comp.doc, dict(comp.row, greedy_descent_energy=comp.row["greedy_descent_energy"] + 0.5))
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert analyse_pub(bad, counts, pub, None, seed=0)["greedy_descent_matches_csv"] is False
    assert "components.csv has" in caplog.text
    assert rec["greedy_descent_matches_csv"] is True


# ============================================================================================== 3. analysis fairness
def test_batch_descent_equals_triplet_select_descent_start_by_start_with_float_ties():
    from hsf2.hw.comp_analysis import batch_descent, bits_of, coupling_matrix, index_of
    from hsf2.qubo.triplet import TripletQUBO, select_descent
    rng = np.random.default_rng(11)
    checked = 0
    for case in range(10):
        n = int(rng.integers(5, 9))
        alpha = float(rng.choice([0.0, 0.1, -0.15, 0.3]))
        zeta = float(rng.choice([1.0, 0.7, 0.3]))
        vals = rng.choice([0.1, 0.2, 0.3, 0.4, 0.7, 1.0 / 3, 0.55], size=(n, n))     # decimals: float sums depend on their order
        quad = []
        for i in range(n):
            for j in range(i + 1, n):
                r = rng.random()
                if r < 0.3:
                    quad.append((i, j, -float(vals[i, j])))
                elif r < 0.65:
                    quad.append((i, j, zeta))
        if not quad:
            continue
        q = [(i, j, -b) for i, j, b in quad if b < 0]
        c = [(i, j) for i, j, b in quad if b > 0]
        as_i = lambda v: np.array(v, dtype=np.int64)  # noqa: E731
        tq = TripletQUBO(np.zeros((n, 3), np.int64), np.zeros(n, np.int64), np.zeros(n), np.zeros(n), as_i([a for a, _, _ in q]),
                         as_i([b for _, b, _ in q]), np.array([s for _, _, s in q], float), as_i([a for a, _ in c]),
                         as_i([b for _, b in c]), alpha, zeta)
        starts = np.arange(1 << n)
        fin = index_of(batch_descent(np.full(n, alpha), coupling_matrix(n, quad), bits_of(starts, n)))
        for s0 in starts:
            ref = select_descent(tq, start=np.nonzero((int(s0) >> np.arange(n)) & 1)[0])
            assert int(fin[s0]) == int(sum(1 << int(k) for k in ref)), (case, s0)
            checked += 1
    assert checked > 600


def test_sample_statistics_match_an_independent_brute_force_on_random_instances(tmp_path):
    """P(ground), best sample, tracking content and the descent statistics of analyse_pub against numbers recomputed here from
    the energy table (own descent) for random instances with continuous couplings (no float ties) and random sample sets."""
    from hsf2.hw import components as C
    from hsf2.hw.comp_analysis import analyse_pub
    rng = np.random.default_rng(5)
    for case in range(12):
        n = int(rng.integers(5, 9))
        spec = rand_instance(n, 3000 + case, linear=bool(case % 2))
        spec["quad"] = [(i, j, b if b > 0 else -float(rng.uniform(0.25, 1.0))) for i, j, b in spec["quad"]]
        if not any(b < 0 for *_, b in spec["quad"]):
            continue
        run = write_run(tmp_path / f"tri{case}", [spec])
        comp = C.load_components(run)[0]
        E = brute(n, spec["linear"], spec["quad"])
        E0, N, S = float(E.min()), 1 << n, 500
        ground = E <= E0 + 1e-9
        w = rng.dirichlet(np.full(N, 0.15))                                         # a lumpy distribution: some states very likely
        c = rng.multinomial(S, w)
        if case % 3 == 0:                                                            # sometimes the optimum is among the samples
            c[int(np.argmin(E))] += 3
            c[int(np.argmax(c))] -= 3
        counts = to_counts(c)
        rec = analyse_pub(comp, counts, pub_of(comp, S), None, seed=case)
        hw = rec["hardware"]
        idx = np.array(sorted(counts))
        cnt = np.array([counts[i] for i in idx])
        fin = np.array([ref_descent(E, n, i) for i in idx])
        assert hw["distinct"] == len(idx) and hw["shots"] == S
        assert hw["p_ground"] == pytest.approx(cnt[ground[idx]].sum() / S) and hw["mean_energy"] == pytest.approx(float(cnt @ E[idx]) / S)
        best = sorted(range(len(idx)), key=lambda t: (E[idx[t]], -cnt[t], idx[t]))[0]
        assert hw["best_index"] == int(idx[best]) and hw["best_energy"] == pytest.approx(float(E[idx[best]]))
        assert hw["best_reaches_E0"] == bool(ground[idx].any())
        hit = E[fin] <= E0 + 1e-9
        assert hw["desc_frac_E0"] == pytest.approx(cnt[hit].sum() / S) and hw["desc_any_E0"] == bool(hit.any())
        assert hw["desc_best_energy"] == pytest.approx(float(E[fin].min())) and hw["desc_mean_energy"] == pytest.approx(float(cnt @ E[fin]) / S)
        t = rec["tracking"]["hw_best"]
        x = [(int(idx[best]) >> i) & 1 for i in range(n)]
        assert (t["n_true"], t["n_fake"], t["n_selected"]) == (sum(1 for i in range(n) if x[i] and spec["true"][i]),
                                                              sum(1 for i in range(n) if x[i] and not spec["true"][i]), sum(x))
        assert rec["ground_degeneracy"] == int(ground.sum()) and rec["p_unif_ground"] == pytest.approx(ground.sum() / N)
        # the greedy-S + descent of the export is the one of components.csv (own greedy, own descent)
        assert rec["greedy_descent_matches_csv"] is True
        assert rec["tracking"]["greedy_descent"]["energy"] == pytest.approx(float(E[ref_descent(E, n, ref_greedy(n, spec["quad"]))]))


def test_one_descent_routine_serves_the_hardware_samples_and_every_control(tmp_path, monkeypatch):
    from hsf2.hw import comp_analysis as CA
    from hsf2.hw import components as C
    spec = HARD[0]
    run = write_run(tmp_path / "tri", [spec])
    comp = C.load_components(run)[0]
    n, S = comp.n, 2000
    pub = pub_of(comp, S)
    seen = []
    real = CA.batch_descent
    monkeypatch.setattr(CA, "batch_descent", lambda lin, B, X: (seen.append((np.array(lin), np.array(B), X.shape[0])), real(lin, B, X))[1])
    counts = to_counts(np.random.default_rng(0).multinomial(S, np.ones(1 << n) / (1 << n)))
    rec = CA.analyse_pub(comp, counts, pub, None, seed=1)
    assert len(seen) == 5                                                       # hardware, uniform, ideal, white noise, greedy
    Bref = np.zeros((n, n))
    for i, j, b in spec["quad"]:
        Bref[i, j] = Bref[j, i] = b
    assert all(np.array_equal(lin, np.zeros(n)) and np.array_equal(B, Bref) for lin, B, _ in seen)
    assert seen[0][2] == len(counts) == rec["hardware"]["distinct"] and seen[4][2] == 1
    for (_, _, rows), name in zip(seen[1:4], ("uniform", "ideal", "white_noise")):
        assert rows == rec["controls"][name]["distinct"] and rec["controls"][name]["shots"] == S == rec["shots"]


def test_controls_have_the_same_N_a_fixed_seed_and_the_documented_distributions(tmp_path):
    from hsf2.hw import components as C
    from hsf2.hw.comp_analysis import analyse_pub
    spec = rand_instance(6, 41)
    other = rand_instance(5, 42)
    run = write_run(tmp_path / "tri", [spec, other])
    comp, comp_b = C.load_components(run)
    E = brute(6, spec["linear"], spec["quad"])
    ground = E <= E.min() + 1e-9
    g, N = int(ground.sum()), 64
    S = 20000
    pub = pub_of(comp, S)
    h, J, _ = ref_ising(comp.linear, comp.quad)
    p = ref_probs(h, J, pub["scale"], pub["gammas"][0], pub["betas"][0])
    F = 0.4
    counts = to_counts(np.random.default_rng(3).multinomial(S, F * p + (1 - F) / N))
    a = analyse_pub(comp, counts, pub, None, seed=7)
    analyse_pub(comp_b, {0: 100}, dict(pub_of(comp_b, 100), k=1), None, seed=7)         # another component in between
    b = analyse_pub(comp, counts, pub, None, seed=7)
    assert a["controls"] == b["controls"]                                       # a fixed seed: the same controls, whatever ran before
    assert analyse_pub(comp, counts, pub, None, seed=8)["controls"]["uniform"] != a["controls"]["uniform"]
    for name in ("uniform", "ideal", "white_noise"):
        assert a["controls"][name]["shots"] == S                                # the same N as the hardware PUB
    F_E = a["hardware"]["F_E"]
    assert F_E == pytest.approx(F, abs=0.03)
    sig = lambda q: math.sqrt(q * (1 - q) / S)                                  # noqa: E731
    P_id = float(p[ground].sum())
    for name, q in (("uniform", g / N), ("ideal", P_id), ("white_noise", F_E * P_id + (1 - F_E) * g / N)):
        assert abs(a["controls"][name]["p_ground"] - q) < 5 * sig(q), (name, a["controls"][name]["p_ground"], q)


def test_F_E_and_F_XEB_with_N_equal_2_pow_n_on_expected_white_noise_counts(tmp_path):
    from hsf2.hw import components as C
    from hsf2.hw.comp_analysis import analyse_pub
    for n, seed, F in ((6, 51, 0.37), (9, 52, 0.62)):
        spec = rand_instance(n, seed, linear=True)
        run = write_run(tmp_path / f"tri{n}", [spec])
        comp = C.load_components(run)[0]
        N, S = 1 << n, 2_000_000
        pub = pub_of(comp, S)
        h, J, _ = ref_ising(comp.linear, comp.quad)
        p = ref_probs(h, J, pub["scale"], pub["gammas"][0], pub["betas"][0])
        E = brute(n, comp.linear, comp.quad)
        mix = F * p + (1 - F) / N
        c = np.floor(mix * S).astype(np.int64)                                  # expected counts (up to rounding), S in total
        c[int(np.argmax(mix))] += S - int(c.sum())
        rec = analyse_pub(comp, to_counts(c), pub, None, seed=0)
        assert rec["hardware"]["F_E"] == pytest.approx(F, abs=2e-3) and rec["hardware"]["F_XEB"] == pytest.approx(F, abs=2e-3)
        F_E_ref = (E.mean() - float(c @ E) / S) / (E.mean() - float(p @ E))
        F_X_ref = (float(c @ p) / S - 1.0 / N) / (float(p @ p) - 1.0 / N)
        assert rec["hardware"]["F_E"] == pytest.approx(F_E_ref, abs=1e-12) and rec["hardware"]["F_XEB"] == pytest.approx(F_X_ref, abs=1e-12)
        ground = E <= E.min() + 1e-9
        assert rec["p_unif_ground"] == pytest.approx(ground.sum() / N) and rec["E_unif"] == pytest.approx(float(E.mean()))
        assert rec["ideal"]["p_ground"] == pytest.approx(float(p[ground].sum()), abs=1e-12)
        assert rec["hardware"]["p_ground"] == pytest.approx(float(c[ground].sum()) / S)


def _flat_case(tmp_path):
    """One analysed component (n = 8, linear term, white-noise counts F = 0.5): (record, its flat CSV row, exact ideal P(ground))."""
    from hsf2.hw import components as C
    from hsf2.hw.comp_analysis import analyse_pub, flat_record
    spec = rand_instance(8, 81, linear=True)
    run = write_run(tmp_path / "tri", [spec])
    comp = C.load_components(run)[0]
    S = 2000
    pub = pub_of(comp, S)
    h, J, _ = ref_ising(comp.linear, comp.quad)
    p = ref_probs(h, J, pub["scale"], pub["gammas"][0], pub["betas"][0])
    E = brute(comp.n, comp.linear, comp.quad)
    rec = analyse_pub(comp, to_counts(np.random.default_rng(0).multinomial(S, 0.5 * p + 0.5 / p.size)), pub, None, seed=0)
    return rec, flat_record(rec), float(p[E <= E.min() + 1e-9].sum())


def test_csv_row_carries_the_values_of_the_record(tmp_path):
    """components_hw.csv is what is read first: every column except ideal_p_ground (next test) equals the nested record."""
    from hsf2.hw.comp_analysis import COLUMN_PREFIX, SOURCES, STAT_KEYS, TRACK_KEYS
    rec, flat, _ = _flat_case(tmp_path)
    for s in SOURCES:
        st = rec["hardware"] if s == "hardware" else rec["controls"][s]
        for k in STAT_KEYS:
            col = f"{COLUMN_PREFIX[s]}_{k}"
            assert flat[col] == st[k], col
    for t, d in rec["tracking"].items():
        for k in TRACK_KEYS:
            assert flat[f"trk_{t}_{k}"] == d[k]
    hw = rec["hardware"]
    assert (flat["F_E"], flat["F_XEB"], flat["hw_distinct"], flat["hw_mean_energy"]) == (hw["F_E"], hw["F_XEB"], hw["distinct"], hw["mean_energy"])
    assert flat["ideal_mean_energy"] == rec["ideal"]["mean_energy"] and flat["p_ideal_over_p_unif"] == rec["p_ideal_over_p_unif"]
    assert flat["p_unif_ground"] == rec["p_unif_ground"] and flat["E0"] == rec["E0"] and flat["ground_degeneracy"] == rec["ground_degeneracy"]


def test_csv_column_ideal_p_ground_is_the_exact_statevector_probability(tmp_path):
    """D2 (fixed): the sampled ideal control is in ideal_sample_*, the exact statevector value stays in ideal_p_ground."""
    rec, flat, p_exact = _flat_case(tmp_path)
    assert rec["ideal"]["p_ground"] == pytest.approx(p_exact, abs=1e-12)                # the record is right ...
    assert flat["ideal_p_ground"] == pytest.approx(p_exact, abs=1e-12)                  # ... and so is the CSV column
    assert flat["ideal_p_ground"] == pytest.approx(flat["p_ideal_over_p_unif"] * flat["p_unif_ground"], rel=1e-9)


def test_tracking_content_reads_true_triplet_with_bit_i_equal_triplet_i(tmp_path):
    from hsf2.hw import components as C
    from hsf2.hw.comp_analysis import analyse_pub
    spec = dict(rand_instance(6, 61), true=[True, True, False, True, False, False])
    run = write_run(tmp_path / "tri", [spec])
    comp = C.load_components(run)[0]
    E = brute(6, spec["linear"], spec["quad"])
    pub = pub_of(comp, 100)
    # the state selecting triplets 0, 1 and 5 is the only sample with the lowest energy among these three
    cand = [0b100011, 0b000110, 0b111000]
    cand.sort(key=lambda s: E[s])
    best = cand[0]
    rec = analyse_pub(comp, {cand[0]: 40, cand[1]: 30, cand[2]: 30}, pub, None, seed=0)
    x = [(best >> i) & 1 for i in range(6)]
    t = rec["tracking"]["hw_best"]
    assert rec["hardware"]["best_index"] == best
    assert (t["n_selected"], t["n_true"], t["n_fake"]) == (sum(x), sum(1 for i in range(6) if x[i] and spec["true"][i]),
                                                          sum(1 for i in range(6) if x[i] and not spec["true"][i]))
    ex = rec["tracking"]["exact"]
    xs = comp.doc["exact_selection"]
    assert (ex["n_true"], ex["n_fake"]) == (sum(1 for i in range(6) if xs[i] and spec["true"][i]),
                                            sum(1 for i in range(6) if xs[i] and not spec["true"][i]))
    assert rec["n_true"] == 3 and rec["n_fake"] == 3
    # reading the flags in reverse order would give another number for at least one of these states (the test can tell)
    rv = spec["true"][::-1]
    assert any(sum(1 for i in range(6) if (s >> i) & 1 and spec["true"][i]) != sum(1 for i in range(6) if (s >> i) & 1 and rv[i])
               for s in cand + [int(sum(b << i for i, b in enumerate(xs)))])


def test_aggregates_and_bootstrap_over_components_on_a_known_answer():
    from hsf2.hw.comp_analysis import SOURCES, boot_stat, group_summary, summarize

    def rec(group, i, reach_hw, ratio):
        st = lambda r, d, f: {"best_reaches_E0": r, "desc_any_E0": d, "desc_frac_E0": f}  # noqa: E731
        return {"group": group, "n": 14 + i % 7, "p_hw_over_p_unif": ratio,
                "hardware": dict(st(reach_hw, True, 0.5), F_E=0.3, F_XEB=0.3),
                "controls": {"uniform": st(False, True, 0.1), "ideal": st(True, True, 0.9), "white_noise": st(reach_hw, True, 0.4)}}
    # 100 hard components: ratios 1..100 (median 50.5), best-of-N reaches E0 in exactly 30 of them
    hard = [rec("hard", i, i < 30, float(i + 1)) for i in range(100)]
    bench = [rec("benchmark", i, True, 7.0) for i in range(5)]
    s = summarize(hard + bench)
    assert set(s) == {"hard", "benchmark", "all"} and [s[g]["n_components"] for g in ("hard", "benchmark", "all")] == [100, 5, 105]
    h = s["hard"]
    assert h["p_hw_over_p_unif_median"]["value"] == 50.5 and h["best_of_N_reaches_E0"]["hardware"]["value"] == 0.3
    assert h["best_of_N_reaches_E0"]["uniform"]["value"] == 0.0 and h["best_of_N_reaches_E0"]["ideal"]["value"] == 1.0
    assert h["best_of_N_plus_descent_reaches_E0"]["uniform"]["value"] == 1.0
    # the bootstrap resamples the COMPONENTS: the 16 / 84 % interval of a fraction p over n components is p -/+ sqrt(p (1-p) / n),
    # the one of the median of 1..100 is 50.5 -/+ 1 / (2 f sqrt(n)) = 5 (uniform density f = 1/100)
    b = h["best_of_N_reaches_E0"]["hardware"]
    sd = math.sqrt(0.3 * 0.7 / 100)
    assert b["lo68"] == pytest.approx(0.3 - sd, abs=0.015) and b["hi68"] == pytest.approx(0.3 + sd, abs=0.015) and b["n"] == 100
    m = h["p_hw_over_p_unif_median"]
    assert m["lo68"] == pytest.approx(50.5 - 5.0, abs=2.0) and m["hi68"] == pytest.approx(50.5 + 5.0, abs=2.0)
    assert summarize(hard + bench) == s                                          # seeded: the same numbers again
    same = group_summary([rec("hard", i, True, 3.0) for i in range(20)], "hard")  # identical components: no spread at all
    assert same["p_hw_over_p_unif_median"]["lo68"] == same["p_hw_over_p_unif_median"]["hi68"] == 3.0
    assert same["best_of_N_reaches_E0"]["hardware"]["lo68"] == same["best_of_N_reaches_E0"]["hardware"]["hi68"] == 1.0
    assert boot_stat([None, float("nan"), 2.0], np.median, np.random.default_rng(0)) == {"value": 2.0, "lo68": None, "hi68": None, "n": 1}
    assert set(h["best_of_N_reaches_E0"]) == set(SOURCES)


# ============================================================================================== 4. safety and booking
def test_cli_submit_asks_one_typed_confirmation_per_job_after_the_budget_check_with_equal_shots(cli_env, monkeypatch, capsys,
                                                                                               fast_polling):
    from acts_qaoa.ibm import QPULedger
    from hsf2.cli import main
    run = write_run(cli_env / "runs" / "tri", HARD[:3] + [EASY[0], EASY[1], EASY[2]])
    svc, smp = mock_ibm(monkeypatch)
    ledger = cli_env / "ledgers" / "l.json"
    argv = ["hw-components", str(run), "--nmin", "5", "--nmax", "9", "--benchmark", "2", "--seed", "0", "--mode", "submit",
            "--shots", str(SHOTS), "--seeds", "1", "--per-job", "2", "--ledger", str(ledger), "--backend", "ibm_fake"]
    # (a) a budget below the estimate: refused before any prompt, any job folder and any submission
    monkeypatch.setattr(sys, "stdin", io.StringIO("submit\n" * 9))
    capsys.readouterr()
    assert main(argv + ["--budget", "5"]) == 1
    out = capsys.readouterr().out
    assert "Type 'submit'" not in out and smp.calls == 0 and not (run / "hw").exists() and not ledger.exists()
    # (b) 5 components, 2 per job: three jobs, the owner types 'submit' twice and then something else
    monkeypatch.setattr(sys, "stdin", io.StringIO("submit\nsubmit\nyes\n"))
    assert main(argv) == 1
    out = capsys.readouterr().out
    assert out.count("Type 'submit' to continue") == 3 and smp.calls == 2        # one prompt per job, nothing submitted for the third
    names = [c for c in sorted(p.stem for p in (run / "components").glob("*.json"))]
    sel = json.loads((next((run / "outputs").glob("*_hw_components")) / "selection.json").read_text())["components"]
    chunks = [sel[0:2], sel[2:4], sel[4:5]]
    prompts = out.split("Type 'submit' to continue")
    for chunk, text in zip(chunks, prompts):
        assert all(c["component"] in text for c in chunk) and "ONE job" in text and f"{len(chunk)} PUBs x {SHOTS} shots" in text
    assert [[s for _, s in seen] for seen in smp.pubs_seen] == [[SHOTS] * 2, [SHOTS] * 2]       # equal shots in every PUB
    assert [[nb for nb, _ in seen] for seen in smp.pubs_seen] == [[c["n"] for c in chunks[0]], [c["n"] for c in chunks[1]]]
    assert [a["shots"] for a in smp.make_args] == [SHOTS] * 3                    # the sampler object of job 3 is built, never run
    assert all(a["max_exec"] >= 30 for a in smp.make_args)
    led = QPULedger(ledger, budget_s=400.0)
    assert [e["job_id"] for e in led.entries] == ["fakejob0001", "fakejob0002"] and all(e["quantum_seconds"] == 5.0 for e in led.entries)
    mans = sorted(((json.loads(p.read_text()), p) for p in (run / "hw").glob("*/manifest.json")), key=lambda t: t[0]["tag"])
    assert [m["state"] for m, _ in mans] == ["done", "done", "aborted"] and mans[2][0]["job_id"] is None
    assert names and not list((cli_env / "home").glob("*"))                      # the user's default ledger was never involved


def test_no_wait_books_the_estimate_and_retrieve_upgrades_but_never_downgrades(tmp_path, monkeypatch, fast_polling):
    from acts_qaoa.ibm import QPULedger
    from hsf2.hw.components import load_components, retrieve_any, run_components
    run = write_run(tmp_path / "tri", HARD[:2])
    comps = load_components(run)
    svc, smp = mock_ibm(monkeypatch, charge=None)                                # IBM's per-job usage stays 'pending'
    cfg = cfg_of(tmp_path, mode="submit", backend="ibm_fake")
    res = run_components(run, comps, "d", cfg, per_job=2, wait=False, confirm=lambda m: None, out_dir=tmp_path / "out")
    jd, jid = res["jobs"][0]["job_dir"], "fakejob0001"
    est = res["jobs"][0]["manifest"]["estimate_s"]
    assert res["analysis"] is None and not list(jd.glob("*counts*")) and json.loads((jd / "manifest.json").read_text())["state"] == "submitted"
    led = lambda: QPULedger(cfg.ledger_path(), budget_s=400.0)                   # noqa: E731
    e = led().entries[-1]
    assert e["job_id"] == jid and e["quantum_seconds"] is None and led().used_seconds() == pytest.approx(est, abs=0.01)  # estimate booked
    # 1. retrieved while IBM's figure is still pending: nothing better than the estimate exists, it stays booked (as 'estimate')
    out = retrieve_any(jid, cfg, run_dir=run, service=svc)
    e = led().entries[-1]
    assert e["quantum_seconds"] == pytest.approx(est, abs=0.01) and e["usage_source"] == "estimate"
    assert json.loads((out["job_dir"] / "manifest.json").read_text())["state"] == "done" and len(list(jd.glob("pub_*_counts.json"))) == 2
    # 2. IBM's own figure arrives: it replaces the estimate
    smp.jobs[jid].charge = 6.5
    retrieve_any(jid, cfg, run_dir=run, service=svc)
    e = led().entries[-1]
    assert e["quantum_seconds"] == 6.5 and e["usage_source"] == "job"
    # 3. pending again (or an estimate-only retrieval): the real figure is never downgraded
    smp.jobs[jid].charge = None
    retrieve_any(jid, cfg, run_dir=run, service=svc)
    e = led().entries[-1]
    assert e["quantum_seconds"] == 6.5 and e["usage_source"] == "job"
    assert json.loads((jd / "manifest.json").read_text())["billed"]["used"] == 6.5
    # 4. an account-difference figure booked earlier survives a later pending retrieval as well
    from hsf2.hw.ledger import HwLedger
    HwLedger(cfg.ledger_path(), 400.0).book(jid, 5.0, "account_diff", "ibm_fake", str(run))
    retrieve_any(jid, cfg, run_dir=run, service=svc)
    e = led().entries[-1]
    assert e["quantum_seconds"] == 5.0 and e["usage_source"] == "account_diff" and smp.calls == 1      # and nothing was resubmitted
    assert len(led().entries) == 1


def test_cli_no_wait_submit_then_hw_retrieve_then_hw_components_analyze(cli_env, monkeypatch, fast_polling):
    from acts_qaoa.ibm import QPULedger
    from hsf2.cli import main
    run = write_run(cli_env / "runs" / "tri", HARD[:2])
    svc, smp = mock_ibm(monkeypatch, charge=None, via_session=[])
    ledger = cli_env / "ledgers" / "l.json"
    common = ["--ledger", str(ledger), "--backend", "ibm_fake"]
    sel = ["--hard", "--nmin", "5", "--nmax", "9", "--mode", "submit", "--shots", str(SHOTS), "--seeds", "1"]
    with pytest.raises(SystemExit, match="single job"):                           # two jobs cannot be booked without waiting
        main(["hw-components", str(run), *sel, "--per-job", "1", "--no-wait", *common])
    assert smp.calls == 0 and not (run / "hw").exists() and not ledger.exists()
    monkeypatch.setattr(sys, "stdin", io.StringIO("submit" + chr(10)))
    assert main(["hw-components", str(run), *sel, "--per-job", "2", "--no-wait", *common]) == 0
    jd = next((run / "hw").iterdir())
    man = json.loads((jd / "manifest.json").read_text())
    assert man["state"] == "submitted" and man["job_id"] == "fakejob0001" and smp.calls == 1 and not list(jd.glob("*counts*"))
    assert QPULedger(ledger, 400.0).entries[0]["quantum_seconds"] is None            # the estimate is booked
    smp.jobs["fakejob0001"].charge = 4.25                                              # IBM finalised its figure meanwhile
    assert main(["hw-retrieve", "fakejob0001", "--run-dir", str(run), "--mode", "submit", *common]) == 0
    man = json.loads((jd / "manifest.json").read_text())
    assert man["state"] == "done" and man["billed"]["source"] == "job" and man["billed"]["used"] == 4.25
    assert len(list(jd.glob("pub_*_counts.json"))) == 2 and smp.calls == 1
    assert [e["quantum_seconds"] for e in QPULedger(ledger, 400.0).entries] == [4.25]
    assert main(["hw-components-analyze", str(jd), "--seed", "2"]) == 0
    out = next((run / "outputs").glob("*_hw_components_analyze"))
    df = pd.read_csv(out / "components_hw.csv")
    assert len(df) == 2 and set(df["group"]) == {"hard"} and (df["shots"] == SHOTS).all()
    assert json.loads((out / "components_hw_summary.json").read_text())["config"]["seed"] == 2


def test_local_simulation_is_reproducible_for_a_fixed_sim_seed(planned_job, monkeypatch):
    from qiskit_aer import AerSimulator
    from hsf2.hw import components as C
    from hsf2.hw import fake as F
    monkeypatch.setattr(F, "local_simulator", lambda backend, method="matrix_product_state": AerSimulator(method="statevector"))
    job = planned_job["res"]["jobs"][0]["job"]                                      # 5 PUBs of 400 shots, planned once
    for method in ("white-noise", "statevector"):                                   # (Aer is noiseless here: speed)
        a, methods = C.run_local_counts(None, job, 3, method)
        b, _ = C.run_local_counts(None, job, 3, method)
        c, _ = C.run_local_counts(None, job, 4, method)
        assert a == b and a != c and methods == [method] * 5 and all(sum(x.values()) == SHOTS for x in a)


def test_outputs_stay_in_the_run_folder_and_credentials_are_passed_on_by_path_and_recorded_by_basename(cli_env, monkeypatch, fast_polling):
    import pathlib
    from hsf2.cli import main
    run = write_run(cli_env / "runs" / "tri", [HARD[0], EASY[0]])
    secret = cli_env / "private_dir_xyz" / "apikey_val.json"
    ledger = cli_env / "ledgers" / "l.json"
    real_open, real_path_open = builtins.open, pathlib.Path.open

    def is_secret(f):
        return "apikey_val" in str(f) or "private_dir_xyz" in str(f)

    def guard_open(file, *a, **k):
        assert not is_secret(file), f"credentials file opened: {file}"
        return real_open(file, *a, **k)

    def guard_path_open(self, *a, **k):
        assert not is_secret(self), f"credentials file opened: {self}"
        return real_path_open(self, *a, **k)
    monkeypatch.setattr(builtins, "open", guard_open)
    monkeypatch.setattr(pathlib.Path, "open", guard_path_open)
    base = ["hw-components", str(run), "--nmin", "5", "--nmax", "9", "--benchmark", "1", "--seeds", "1", "--shots", "400",
            "--per-job", "2", "--ledger", str(ledger), "--creds", str(secret)]
    before = tree(cli_env, skip=(run,))
    assert main(base + ["--mode", "dry"]) == 0                                    # dry: plan files only
    assert main(base + ["--mode", "local", "--sim-method", "white-noise"]) == 0   # local: counts and the analysis
    assert tree(cli_env, skip=(run,)) == before                                   # nothing outside the run folder, no ledger
    assert not ledger.exists() and not secret.exists() and not list((cli_env / "home").glob("*"))
    created = {p.relative_to(run).parts[0] for p in run.rglob("*") if p.is_file()} - {"components", "components.csv"}
    assert created == {"hw", "outputs"}
    text = "\n".join(p.read_text(errors="ignore") for p in run.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".txt", ".csv"))
    assert "private_dir_xyz" not in text and "apikey_val.json" in text            # the basename only, never the directory
    # submit (mocked service): the path is handed to the service factory (Phase 1 reads it), results record the basename only
    got = []
    mock_ibm(monkeypatch, via_session=got)
    monkeypatch.setattr(sys, "stdin", io.StringIO("submit\nsubmit\n"))
    assert main(base + ["--mode", "submit", "--backend", "ibm_fake"]) == 0
    assert got == [str(secret)]
    text = "\n".join(p.read_text(errors="ignore") for p in run.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".txt", ".csv"))
    assert "private_dir_xyz" not in text
    with real_open(str(ledger)) as fh:
        ledger_text = fh.read()
    assert "private_dir_xyz" not in ledger_text and "apikey_val" not in ledger_text


def test_static_the_component_code_has_no_file_opening_of_credentials():
    src = {f: (ROOT / "hsf2" / "hw" / f).read_text() for f in ("components.py", "comp_analysis.py")}
    for name, text in src.items():
        assert not re.search(r"\bopen\(", text), name
        assert not re.search(r"read_text\(|write_text\(", text), name
        assert "environ" not in text and "expanduser" not in text, name
    # the only uses of the credentials in the component code: the basename label, the pass-through of the path to the ledger and
    # to the service factory of Phase 1 (which reads the file); nothing else may mention them
    allowed = (r"import HwConfig, creds_label$", r'"creds": creds_label\(cfg\.creds_file\)', r"HwLedger\(cfg\.ledger_path\(\), cfg\.budget_s, creds_file=cfg\.creds_file\)$",
               r"session\.get_service\(cfg\.creds_file\)$")
    uses = [ln.strip() for ln in src["components.py"].splitlines() if "creds" in ln]
    assert uses and all(any(re.search(a, ln) for a in allowed) for ln in uses), uses
    assert "creds" not in src["comp_analysis.py"]


def test_counts_are_kept_even_when_a_pub_returns_another_number_of_shots(tmp_path, monkeypatch, fast_polling):
    """D1 (fixed): write_counts saves every PUB's counts first, then raises on the shot mismatch."""
    from hsf2.hw.components import load_components, run_components
    run = write_run(tmp_path / "tri", [EASY[0]])
    comps = load_components(run)

    def plus_eight(isa, values, shots, rng):
        return uniform_counts(isa, values, shots + 8, rng)
    mock_ibm(monkeypatch, charge=6.5, maker=plus_eight)
    try:
        run_components(run, comps, "d", cfg_of(tmp_path, mode="submit", backend="ibm_fake"), per_job=1, confirm=lambda m: None)
    except ValueError:
        pass
    jd = next((run / "hw").iterdir())
    assert list(jd.glob("pub_*_counts.json"))                                    # the raw counts are on disk for a later look


# ============================================================================================== 5. cost
def test_cost_estimate_is_the_documented_formula_with_the_longest_circuit_and_all_pubs(planned_job, tmp_path):
    """estimate = 1.5 x (3 s + PUBs x (0.25 + 0.05 x randomisations) + shots_total x (rep_delay + longest circuit + 5 us))."""
    from hsf2.hw import components as C
    res = planned_job["res"]["jobs"][0]
    man, job = res["manifest"], res["job"]
    durs = [p["duration_s"] for p in man["pubs"]]
    assert man["circuit_duration_s"] == pytest.approx(max(durs)) and max(durs) > min(durs)      # the longest one, for every PUB
    rep, P = man["rep_delay_s"], 5
    assert rep == pytest.approx(250e-6)
    expect = 1.5 * (3.0 + P * (0.25 + 0.05 * 16) + P * man["shots"] * (rep + max(durs) + 5e-6))
    assert man["estimate_s"] == pytest.approx(expect, rel=1e-12) and job.estimate_s == man["estimate_s"]
    # bare settings (no twirling) have one randomisation per PUB; 10 PUBs of 4000 shots grow linearly in the PUB count
    from hsf2.hw.config import HwConfig
    cfg = HwConfig(mode="dry", shots=4000, transpile_seeds=1, ledger=str(tmp_path / "l.json"))
    comps = planned_job["comps"]
    run = tmp_path / "tri"
    shutil.copytree(planned_job["run"], run, ignore=shutil.ignore_patterns("hw", "outputs"))
    a = C.run_components(run, comps[:2], "a", cfg, per_job=2)["jobs"][0]["manifest"]
    d = C.run_components(run, comps[:2], "d", cfg, per_job=2)["jobs"][0]["manifest"]
    assert a["setting"]["gate_twirl"] is False and d["setting"]["gate_twirl"] is True
    assert d["estimate_s"] - a["estimate_s"] == pytest.approx(1.5 * 2 * 0.05 * 15, rel=1e-9)  # same circuits, 16 vs 1 randomisations
    # 10 components of 14-20 variables at 4000 shots would be 1.5 x (3 + 10 x 1.05 + 40000 x (250 us + <= 30 us + 5 us)) = 37-38 s
    assert 1.5 * (3.0 + 10 * 1.05 + 40000 * (250e-6 + 28e-6 + 5e-6)) == pytest.approx(37.0, abs=1.0)


def test_planning_is_deterministic_same_inputs_same_angles_layouts_and_estimate(planned_job, tmp_path):
    from hsf2.hw import components as C
    run = tmp_path / "tri"
    shutil.copytree(planned_job["run"], run, ignore=shutil.ignore_patterns("hw", "outputs"))
    comps = C.load_components(run)[:2]
    again = C.run_components(run, comps, "d", cfg_of(tmp_path, mode="dry"), per_job=2)["jobs"][0]
    first = planned_job["res"]["jobs"][0]
    for pa, pb in zip(first["manifest"]["pubs"][:2], again["manifest"]["pubs"]):
        for key in ("gammas", "betas", "scale", "n_2q", "depth_2q", "fidelity_est", "transpile_seed", "measured_qubits", "initial_layout",
                    "component_sha1", "ideal_mean_energy"):
            assert pa[key] == pb[key], key
    for ia, ib in zip(first["job"].plans[:2], again["job"].plans):
        ops = lambda c: [(i.operation.name, tuple(c.find_bit(q).index for q in i.qubits), tuple(map(str, i.operation.params)))  # noqa: E731
                         for i in c.data]
        assert ops(ia.pub.isa) == ops(ib.pub.isa)


def test_max_that_leaves_no_room_for_the_hard_pieces_is_refused(tmp_path):
    """D4 (fixed): --max equal to --benchmark would silently drop every hard piece; it is refused instead."""
    from hsf2.hw.components import load_components, select_components
    run = write_run(tmp_path / "tri", [HARD[0], EASY[0], EASY[1]])
    comps = load_components(run)
    assert any(c.hard for c in comps) and sum(not c.hard for c in comps) >= 2
    with pytest.raises(ValueError, match="no room for the"):
        select_components(comps, 2, 24, n_benchmark=2, seed=0, max_total=2)
    sel = select_components(comps, 2, 24, n_benchmark=2, seed=0, max_total=3)
    assert sum(c.group == "hard" for c in sel) == 1 and sum(c.group == "benchmark" for c in sel) == 2


def test_retrieve_books_in_the_ledger_of_the_submission(tmp_path):
    """D6 (fixed): hw-retrieve uses the ledger recorded in the manifest (if its folder exists here) unless --ledger
    names another one explicitly."""
    from hsf2.hw.config import HwConfig
    from hsf2.hw.run import retrieve_ledger
    sub = tmp_path / "acct_a"
    sub.mkdir()
    man = {"config": {"ledger": str(sub / "ledger_a.json")}}
    assert retrieve_ledger(HwConfig.from_env(), man).path == sub / "ledger_a.json"
    other = tmp_path / "ledger_b.json"
    assert retrieve_ledger(HwConfig.from_env(ledger=str(other)), man).path == other
    gone = {"config": {"ledger": str(tmp_path / "no_such_folder" / "ledger_a.json")}}
    cfg = HwConfig.from_env()
    assert retrieve_ledger(cfg, gone).path == cfg.ledger_path()


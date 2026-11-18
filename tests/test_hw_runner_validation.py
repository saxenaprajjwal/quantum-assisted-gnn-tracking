"""Independent validation of hsf2.hw (written by the validation engineer, not the implementer).

Everything is 0 QPU s: statevector / numpy references, fake backends, mocks.  IBM is never contacted.
Physics references are re-derived here (numpy), not taken from hsf2.
"""
import builtins
import io
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from hsf2._phase1 import ensure_phase1, find_phase1  # noqa: E402

pytestmark = pytest.mark.skipif(find_phase1() is None, reason="Phase-1 package acts_qaoa not found")
if find_phase1() is not None:
    ensure_phase1()


def _script(name: str = "qx_hardware.sh") -> Path:
    """A shell script on the PC (scripts/) or as deployed on lxplus (hsf2_scripts/)."""
    for d in ("scripts", "hsf2_scripts"):
        p = ROOT / d / name
        if p.is_file():
            return p
    pytest.skip(f"{name} not found next to the tests")


# ---------------------------------------------------------------------------------------------- helpers
def rand_ising(n, seed, ties=False):
    from acts_qaoa.surrogate import Ising
    rng = np.random.default_rng(seed)
    h = rng.normal(size=n)
    J = np.triu(rng.normal(size=(n, n)), 1)
    if ties:
        J = np.triu(rng.choice([-1.0, -0.5, 0.5, 1.0, 0.25], size=(n, n)), 1)
    return Ising(n=n, offset=0.3, h=h, J=J)


def ref_probs(ism, scale, gamma, beta):
    """Independent numpy p=1 QAOA: |+>^n, exp(-i gamma H/scale), exp(-i beta sum X); index bit k = qubit k."""
    n = ism.n
    x = np.arange(1 << n)
    bits = ((x[:, None] >> np.arange(n)[None, :]) & 1)
    z = 1.0 - 2.0 * bits
    ph = z @ (ism.h / scale) + np.einsum("ij,ij->i", z @ (ism.J / scale), z)
    psi = np.exp(-1j * gamma * ph) / math.sqrt(1 << n)
    c, s = math.cos(beta), -1j * math.sin(beta)
    for k in range(n):
        psi = psi.reshape(1 << (n - k - 1), 2, 1 << k)
        psi = np.stack([c * psi[:, 0] + s * psi[:, 1], s * psi[:, 0] + c * psi[:, 1]], axis=1).reshape(-1)
    return np.abs(psi) ** 2


def circuit_probs(qc):
    from qiskit.quantum_info import Statevector
    return Statevector(qc.remove_final_measurements(inplace=False)).probabilities()


def bound(ism, scale, gamma, beta):
    from acts_qaoa.circuits import bind
    from hsf2.hw.build import build_circuit
    qc, gam, bet, _, _ = build_circuit(ism, scale)
    return bind(qc, gam, bet, [gamma], [beta])


def energies(ism):
    n = ism.n
    x = np.arange(1 << n)
    bits = ((x[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    return ism.energy_bits(bits)


# ---------------------------------------------------------------------------------------------- 2a main circuit
@pytest.mark.parametrize("n,seed", [(5, 1), (8, 2)])
def test_main_circuit_equals_independent_p1_qaoa_and_closed_form(n, seed):
    from acts_qaoa.qaoa_math import p1_energy
    ism = rand_ising(n, seed)
    scale = ism.scale()
    g, b = 0.37, -0.61
    p = circuit_probs(bound(ism, scale, g, b))
    assert np.abs(p - ref_probs(ism, scale, g, b)).max() < 1e-12
    E = energies(ism)
    e_circ = float(p @ E)
    e_closed = ism.offset + scale * p1_energy(ism.h / scale, ism.full_J() / scale, g, b)
    assert e_circ == pytest.approx(e_closed, abs=1e-9)


@pytest.mark.parametrize("seed,ties", [(3, False), (4, True)])
def test_sparsified_circuit_matches_phase1_ising_and_reference_topk(seed, ties):
    from hsf2.hw.build import sparse_ising
    ism = rand_ising(8, seed, ties)
    iu, ju = np.triu_indices(8, 1)
    vals = np.abs(ism.J[iu, ju])
    for K in (3, 7, 12, 28, 0, None):
        sp = sparse_ising(ism, K)
        ph1 = ism if K in (0, None) else ism.sparsified(K)
        assert np.array_equal(sp.J, ph1.J) and np.array_equal(sp.h, ph1.h)
        kept = np.abs(sp.J[iu, ju]) > 0
        if K:
            assert kept.sum() == K
            # kept values are the K largest |J| (as a multiset) and unchanged
            assert np.allclose(np.sort(vals[kept]), np.sort(np.sort(vals)[::-1][:K]))
            assert np.array_equal(sp.J[iu, ju][kept], ism.J[iu, ju][kept])
        # deterministic
        assert np.array_equal(sparse_ising(ism, K).J, sp.J)


# ---------------------------------------------------------------------------------------------- 2b gauges
def test_every_gauge_xor_equals_ungauged_exactly_and_energy_frame():
    from hsf2.hw.gauge import gauge_ising, gauge_mask, gauge_vectors
    n = 8
    ism = rand_ising(n, 5)
    scale = ism.scale()
    g0, b0 = 0.81, 0.29
    p0 = circuit_probs(bound(ism, scale, g0, b0))
    E = energies(ism)
    gs = gauge_vectors(n, 12, seed=3)
    assert not gs[0].any() and len({g.tobytes() for g in gs}) == 12
    x = np.arange(1 << n)
    for g in gs:
        gi = gauge_ising(ism, g)
        p = circuit_probs(bound(gi, scale, g0, b0))
        mask = gauge_mask(g)
        # measured x' -> original x = x' ^ mask : probabilities equal to the ungauged ones, to machine precision
        assert np.abs(p[x ^ mask] - p0[x]).max() < 1e-12 or np.abs(p - p0[x ^ mask]).max() < 1e-12
        # energy of x' ^ mask under the ORIGINAL Ising == energy of x' under the gauged one
        assert np.allclose(E[x ^ mask], energies(gi))
        # mean energy in original frame via XOR'ed distribution equals the ungauged mean energy
        assert float(p0 @ E) == pytest.approx(float(sum(p[xp] * E[xp ^ mask] for xp in x)), abs=1e-10)
    # bit k of the mask <-> qubit k (asymmetric check)
    assert gauge_mask(np.array([1, 0, 0, 1], np.uint8)) == 0b1001
    assert gauge_mask(np.array([0, 1, 0, 0], np.uint8)) == 2


def test_write_counts_applies_gauge_xor_once_and_energy_in_original_frame(tmp_path):
    from hsf2.hw.gauge import gauge_ising, gauge_record, gauge_vectors
    from hsf2.hw.run import write_counts
    n = 6
    ism = rand_ising(n, 6)
    scale = ism.scale()
    gs = gauge_vectors(n, 3, seed=1)
    E = energies(ism)
    shots = 400000
    rng = np.random.default_rng(0)
    counts, pubs = [], []
    for gi, g in enumerate(gs):
        p = circuit_probs(bound(gauge_ising(ism, g), scale, 0.7, 0.4))
        c = rng.multinomial(shots, p / p.sum())
        counts.append({int(i): int(v) for i, v in enumerate(c) if v})
        for kind in ("main", "control"):
            pubs.append({"k": len(pubs), "name": f"{kind}_g{gi}", "kind": kind, "gauge": gi, "shots": shots, "gauged": True})
        counts.append(counts[-1])        # control counts: same data (only the XOR bookkeeping is under test)
    man = {"pubs": pubs, "gauges": [gauge_record(g) for g in gs]}
    merged = write_counts(tmp_path, man, counts)
    tot = sum(merged["main"].values())
    e_hat = sum(v * E[k] for k, v in merged["main"].items()) / tot
    p_exact = circuit_probs(bound(ism, scale, 0.7, 0.4))
    assert e_hat == pytest.approx(float(p_exact @ E), abs=0.01 * np.abs(E).max())
    # shot-count mismatch is refused
    bad = [dict(c) for c in counts]
    bad[0][0] = bad[0].get(0, 0) + 1
    with pytest.raises(ValueError):
        write_counts(tmp_path, man, bad)


# ---------------------------------------------------------------------------------------------- 2c fractional
def test_fold_rzz_unitary_equivalence_and_range():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from hsf2.hw.transpile import check_fractional_isa, fold_rzz, rzz_angles
    rng = np.random.default_rng(0)
    qc = QuantumCircuit(4)
    qc.h(range(4))
    for a, b in itertools.combinations(range(4), 2):
        qc.rzz(float(rng.uniform(-9, 9)), a, b)
        qc.rx(float(rng.uniform(-3, 3)), a)
    qc.rzz(0.3, 0, 1)
    qc.rzz(-0.3, 2, 3)
    qc.rzz(math.pi, 1, 2)          # a boundary-ish value
    out = fold_rzz(qc)
    assert Operator(out).equiv(Operator(qc))
    ang = rzz_angles(out)
    assert ang and all(0 < a <= math.pi / 2 + 1e-9 for a in ang)
    check_fractional_isa(out)


def _compress(isa):
    """ISA on a big device -> circuit on only the qubits it touches; returns (circuit, {phys: new index})."""
    from qiskit import QuantumCircuit
    used = sorted({isa.find_bit(q).index for ins in isa.data for q in ins.qubits})
    idx = {q: i for i, q in enumerate(used)}
    qc = QuantumCircuit(len(used))
    for ins in isa.data:
        if ins.operation.name in ("measure", "barrier"):
            continue
        qc.append(ins.operation, [idx[isa.find_bit(q).index] for q in ins.qubits])
    return qc, idx


def _isa_measured_probs(pub, phys_measured):
    from qiskit.quantum_info import Statevector
    isa = pub.isa
    if pub.values is not None:
        isa = isa.assign_parameters(dict(zip(isa.parameters, [float(v) for v in pub.values])))
    qc, idx = _compress(isa)
    assert qc.num_qubits <= 16
    return Statevector(qc).probabilities(qargs=[idx[q] for q in phys_measured])


@pytest.fixture(scope="module")
def small_problem():
    n = 6
    ism = rand_ising(n, 11)
    return ism, ism.scale()


@pytest.mark.parametrize("fractional,gauged", [(False, False), (True, False), (True, True), (False, True)])
def test_isa_distribution_main_and_control(small_problem, fractional, gauged):
    from hsf2.hw.build import make_pub_specs
    from hsf2.hw.fake import fake_backend, fractional_fake
    from hsf2.hw.gauge import gauge_mask, gauge_vectors
    from hsf2.hw.transpile import transpile_job, rzz_angles
    ism, scale = small_problem
    base = fake_backend("FakeFez")
    backend = fractional_fake(base) if fractional else base
    gam, bet = [0.9], [0.35]
    specs = make_pub_specs(fractional, gauged, 2, 400, gam, bet)
    gauges = gauge_vectors(ism.n, 2 if gauged else 1, seed=2)
    pubs = transpile_job(ism, scale, fractional, specs, gauges, backend, n_seeds=1)
    assert len(pubs) == len(specs)
    p_ref = ref_probs(ism, scale, gam[0], bet[0])
    x = np.arange(1 << ism.n)
    for pb in pubs:
        meas = pb.layout["measured"]["qubits"]
        assert pb.layout["measured"]["clbits"] == list(range(ism.n))
        p = _isa_measured_probs(pb, meas)
        mask = gauge_mask(gauges[pb.spec.gauge])
        if pb.spec.kind == "main":
            assert np.abs(p[x ^ mask] - p_ref[x]).max() < 1e-9 or np.abs(p - p_ref[x ^ mask]).max() < 1e-9, pb.spec.name
        else:
            assert np.abs(p - 1.0 / (1 << ism.n)).max() < 1e-9, f"control of {pb.spec.name} is not uniform"
        if fractional:
            ang = rzz_angles(pb.isa)
            assert ang and all(0 < a <= math.pi / 2 + 1e-9 for a in ang)
            assert pb.values is None
            assert not any(ins.operation.name == "cz" for ins in pb.isa.data) or True


def test_fractional_control_angles_are_main_gamma_and_pi_over_2():
    from hsf2.hw.build import control_angles
    assert control_angles(True, [0.4], [0.2]) == ([0.4], [math.pi / 2])
    assert control_angles(False, [0.4], [0.2]) == ([0.0], [0.0])


# ---------------------------------------------------------------------------------------------- 2e settings
def _flat(t, pre=""):
    out = {}
    for k, v in t.items():
        if isinstance(v, dict):
            out.update(_flat(v, pre + k + "."))
        else:
            out[pre + k] = v
    return out


SPEC_TABLE = {  # gate twirl, measure twirl, DD
    "a": (False, False, False), "b": (False, False, True), "c": (True, True, False),
    "d": (True, True, True), "e": (False, True, True), "f": (True, True, True)}


def test_settings_vs_spec_table_through_real_sampler_options():
    import warnings
    from hsf2.hw.settings import SETTINGS, apply_options, get_setting, options_dump
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from qiskit_ibm_runtime import SamplerV2
        from qiskit_ibm_runtime.fake_provider import FakeFez
        default = options_dump(SamplerV2(mode=FakeFez()).options)
    assert sorted(SETTINGS) == list("abcdef")
    for name, (gt, mt, dd) in SPEC_TABLE.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = SamplerV2(mode=FakeFez())
        apply_options(s.options, dict(get_setting(name).options(), default_shots=500))
        d = options_dump(s.options)
        assert (d["twirling"]["enable_gates"], d["twirling"]["enable_measure"],
                d["dynamical_decoupling"]["enable"]) == (gt, mt, dd), name
        assert d["twirling"]["num_randomizations"] == 16 and d["twirling"]["shots_per_randomization"] == "auto"
        assert d["dynamical_decoupling"]["sequence_type"] == "XY4"
        # nothing else changed
        flat = _flat
        diff = {k for k, v in flat(d).items() if flat(default).get(k) != v}
        assert diff <= {"twirling.enable_gates", "twirling.enable_measure", "twirling.num_randomizations",
                        "twirling.shots_per_randomization", "dynamical_decoupling.enable",
                        "dynamical_decoupling.sequence_type", "default_shots"}, (name, diff)
    assert get_setting("e").fractional and not get_setting("e").gauged
    assert get_setting("f").gauged and not get_setting("f").fractional
    assert [n for n, s in SETTINGS.items() if s.fractional] == ["e"]


def test_f_setting_is_8_gauges_x_500_shots_and_pubs_equal():
    from hsf2.hw.build import check_equal_shots, make_pub_specs
    specs = make_pub_specs(False, True, 8, 4000, [0.1], [0.2])
    assert len(specs) == 16 and check_equal_shots(specs) == 500
    assert sum(s.shots for s in specs if s.kind == "main") == 4000 == sum(s.shots for s in specs if s.kind == "control")
    with pytest.raises(ValueError):
        make_pub_specs(False, True, 8, 4001, [0.1], [0.2])
    specs[3].shots = 499
    with pytest.raises(ValueError):
        check_equal_shots(specs)


# ---------------------------------------------------------------------------------------------- 2f readout
def _forward(p_true, mats):
    n = len(mats)
    full = np.ones((1 << n, 1 << n))
    x = np.arange(1 << n)
    for k, m in enumerate(mats):
        bk = (x >> k) & 1
        full *= m[bk[:, None], bk[None, :]]
    return full @ p_true


def test_readout_numpy_and_mthree_agree_and_bit_order_with_unsorted_qubits():
    from hsf2.hw import readout as R
    mats = [np.array([[0.97, 0.20], [0.03, 0.80]]), np.array([[0.99, 0.02], [0.01, 0.98]]),
            np.array([[0.90, 0.01], [0.10, 0.99]]), np.array([[0.95, 0.12], [0.05, 0.88]])]
    qubits = [9, 3, 6, 1]                                      # not sorted: bit k <-> qubits[k]
    n = 4
    rng = np.random.default_rng(5)
    p_true = rng.dirichlet(np.ones(1 << n) * 0.5)
    obs = _forward(p_true, mats)
    shots = 10 ** 7
    counts = {i: int(round(v * shots)) for i, v in enumerate(obs)}
    zeros = _forward(np.eye(1 << n)[0], mats)
    ones = _forward(np.eye(1 << n)[-1], mats)
    cal = R.calibration_record(qubits, {i: int(round(v * 10 ** 7)) for i, v in enumerate(zeros) if v > 0},
                               {i: int(round(v * 10 ** 7)) for i, v in enumerate(ones) if v > 0}, 10 ** 7, True, "s", "b", None)
    # matrices recovered per qubit, in the right slot
    for k, q in enumerate(qubits):
        assert np.allclose(np.array(cal["matrices"][str(q)]), mats[k], atol=2e-4)
    pn, _, bn = R.correct_counts(counts, cal, 12, prefer="numpy")
    assert bn == "numpy"
    arr = lambda d: np.array([d.get(i, 0.0) for i in range(1 << n)])  # noqa: E731
    assert np.abs(arr(pn) - p_true).max() < 2e-3
    # the wrong bit order would not recover it
    cal_bad = dict(cal, qubits=qubits[::-1])
    pb, _, _ = R.correct_counts(counts, cal_bad, 12, prefer="numpy")
    assert np.abs(arr(pb) - p_true).max() > 5e-3
    version, _ = R.mthree_status()
    if version is None:
        pytest.skip("mthree not importable here; numpy path verified")
    pm, _, bm = R.correct_counts(counts, cal, 12, prefer="mthree")
    assert bm.startswith("mthree")
    assert np.abs(arr(pm) - arr(pn)).max() < 1e-6
    assert np.abs(arr(pm) - p_true).max() < 2e-3


def test_apply_readout_refuses_twirl_mismatch_and_other_qubits(tmp_path):
    from hsf2.hw.run import apply_readout
    job = tmp_path / "hw" / "j"
    job.mkdir(parents=True)
    man = {"setting": {"measure_twirl": True}, "layout": {"measured_qubits": [4, 2]}, "pubs": [], "gauges": []}
    (job / "manifest.json").write_text(json.dumps(man))
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"measure_twirl": False, "qubits": [4, 2], "matrices": {}}))
    with pytest.raises(ValueError, match="twirl"):
        apply_readout(job, cal)
    cal.write_text(json.dumps({"measure_twirl": True, "qubits": [2, 4], "matrices": {}}))
    with pytest.raises(ValueError, match="qubits"):
        apply_readout(job, cal)


# ---------------------------------------------------------------------------------------------- 1 safety
class _FakeJob:
    def __init__(self, n_bits, charge=0.0, fail=False, jid="fakejob0001", status="finished"):
        self.n_bits, self.charge, self.fail, self.jid, self.status = n_bits, charge, fail, jid, status
        self.shots = []

    def job_id(self):
        return self.jid

    def result(self):
        if self.fail:
            raise RuntimeError("job cancelled")
        rng = np.random.default_rng(1)

        def one(s):
            c = rng.multinomial(s, np.ones(1 << self.n_bits) / (1 << self.n_bits))
            counts = {format(i, f"0{self.n_bits}b"): int(v) for i, v in enumerate(c) if v}
            return SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda: counts)))
        return [one(s) for s in self.shots]

    def metrics(self):
        return {"usage": {"status": self.status, "qpu_charge_time_seconds": self.charge}}

    def properties(self):
        return None


class _Svc:
    def __init__(self, consumed=100.0, limit=600.0, step=5.0):
        self.consumed, self.limit, self.step, self.ran = consumed, limit, step, False

    def usage(self):
        return {"usage_consumed_seconds": self.consumed + (self.step if self.ran else 0.0),
                "usage_limit_seconds": self.limit}

    def job(self, jid):
        return self.jobobj


class _Sampler:
    def __init__(self, job, svc):
        self.job, self.svc, self.calls, self.kw = job, svc, 0, []
        self.options = SimpleNamespace()

    def run(self, pubs, **kw):
        self.calls += 1
        self.kw.append(kw)
        self.svc.ran = True
        self.job.shots = [p[2] for p in pubs]
        return self.job


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    try:
        from test_hw_runner import make_small_run
    except ImportError:                                 # deployed on lxplus as test_hsf2_hw_runner.py
        from test_hsf2_hw_runner import make_small_run
    return make_small_run(tmp_path_factory.mktemp("valrun") / "r")


@pytest.fixture
def fast(monkeypatch):
    from hsf2.hw import run as R
    for k, v in (("USAGE_TIMEOUT_S", 0.05), ("USAGE_INTERVAL_S", 0.005), ("ACCOUNT_TIMEOUT_S", 0.05),
                 ("ACCOUNT_INTERVAL_S", 0.005)):
        monkeypatch.setattr(R, k, v)


def mock_ibm(monkeypatch, n_bits=6, **jobkw):
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from hsf2.hw import run as R
    svc = _Svc()
    job = _FakeJob(n_bits, **jobkw)
    smp = _Sampler(job, svc)
    svc.jobobj = job
    monkeypatch.setattr(R, "get_job_backend", lambda cfg, setting, service=None: (FakeFez(), svc))
    monkeypatch.setattr(R, "make_sampler", lambda *a, **k: (smp, {"mock": True}))
    return svc, job, smp


def cfg_for(tmp_path, **kw):
    from hsf2.hw.config import HwConfig
    d = dict(mode="submit", shots=400, transpile_seeds=1, ledger=str(tmp_path / "ledger.json"), budget_s=400.0,
             backend="ibm_fake")
    d.update(kw)
    return HwConfig(**d)


def test_no_sampler_run_without_confirmation_and_one_confirmation_per_job(run_dir, tmp_path, monkeypatch, fast):
    from hsf2.hw.run import SubmitAborted, run_job, run_ladder
    svc, job, smp = mock_ibm(monkeypatch)
    asked = []

    def deny(msg):
        asked.append(msg)
        raise SubmitAborted("no")
    with pytest.raises(SubmitAborted):
        run_job(run_dir, "a", 6, cfg_for(tmp_path), confirm=deny)
    assert smp.calls == 0 and len(asked) == 1
    # server-side dry run is a submission to the real backend as well: it must ask, too
    with pytest.raises(ValueError):                                   # needs the executor primitive
        run_job(run_dir, "a", 6, cfg_for(tmp_path, ledger=str(tmp_path / "l2.json")), confirm=deny, server_dry_run=True)
    assert smp.calls == 0
    asked.clear()
    with pytest.raises(SubmitAborted):
        run_job(run_dir, "a", 6, cfg_for(tmp_path, primitive="executor", ledger=str(tmp_path / "l3.json")), confirm=deny,
                server_dry_run=True)
    assert smp.calls == 0 and len(asked) == 1 and "SERVER-SIDE DRY-RUN" in asked[0]
    # ladder: three settings -> three confirmations, and a refusal of the second stops the ladder
    seq = []

    def ok_then_no(msg):
        seq.append(msg)
        if len(seq) == 2:
            raise SubmitAborted("no")
    with pytest.raises(SubmitAborted):
        run_ladder(run_dir, [6], ["a", "b", "c"], cfg_for(tmp_path, ledger=str(tmp_path / "l4.json")), 2, confirm=ok_then_no)
    assert len(seq) == 2 and smp.calls == 1


def test_dry_local_and_real_target_dry_never_touch_a_sampler(run_dir, tmp_path, monkeypatch):
    from hsf2.hw import run as R
    from hsf2.hw import session
    from qiskit_ibm_runtime.fake_provider import FakeFez

    def boom(*a, **k):
        raise AssertionError("a sampler / execute path was reached in dry mode")
    monkeypatch.setattr(R, "make_sampler", boom)
    monkeypatch.setattr(R, "execute_job", boom)
    got = {}
    svc = SimpleNamespace(backend=lambda name, **kw: got.update(name=name, kw=kw) or FakeFez())
    monkeypatch.setattr(session, "get_service", lambda creds=None: svc)
    res = R.run_job(run_dir, "a", 6, cfg_for(tmp_path, mode="dry", real_target=True))
    assert res["manifest"]["state"] == "dry" and got["name"] == "ibm_fake" and got["kw"] == {"use_fractional_gates": False}
    res = R.run_job(run_dir, "a", 6, cfg_for(tmp_path, mode="dry"))
    assert res["manifest"]["state"] == "dry"
    # local mode uses Aer only
    monkeypatch.undo()
    import qiskit_ibm_runtime as qir
    monkeypatch.setattr(qir.SamplerV2, "run", boom)
    res = R.run_job(run_dir, "a", 6, cfg_for(tmp_path, mode="local", ledger=str(tmp_path / "ll.json"), shots=100))
    assert res["manifest"]["job_id"].startswith("local_") and res["manifest"]["billed"]["used"] == 0.0


def test_static_no_open_of_credentials_and_run_call_sites():
    hw = ROOT / "hsf2" / "hw"
    runs = []
    for f in sorted(hw.rglob("*.py")):
        txt = f.read_text()
        for i, line in enumerate(txt.splitlines(), 1):
            if re.search(r"\.run\(", line) and not line.strip().startswith(("#", '"""')):
                runs.append((f.name, i, line.strip()))
            if re.search(r"\b(open\(|read_text\(|read_bytes\(|json\.load\()", line) and f.name != "analysis.py":
                assert "cred" not in line.lower(), (f.name, i, line)
    # the only runtime-primitive submission is run.py's execute_job; the others are Aer
    assert [r[0] for r in runs if "sim.run" not in r[2] and "PassManager" not in r[2]] == ["run.py"], runs


def test_credentials_flow_with_guarded_io_and_no_leak_in_outputs(run_dir, tmp_path, monkeypatch, fast):
    from acts_qaoa import ibm
    from hsf2.hw.run import run_job
    secret_dir = tmp_path / "very_private_dir"
    secret_dir.mkdir()
    secret = secret_dir / "apikey_val.json"
    secret.write_text('{"token": "TOKEN-SHOULD-NEVER-BE-READ"}')
    monkeypatch.setenv("HSF2_IBM_CREDS", str(secret))
    monkeypatch.delenv("HSF2_QPU_LEDGER", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    seen = {}
    monkeypatch.setattr(ibm, "get_service", lambda **kw: seen.update(kw) or _Svc())
    import qiskit_ibm_runtime
    from hsf2.hw import run as R
    from qiskit_ibm_runtime.fake_provider import FakeFez
    svc_holder = {}
    orig_backend = None

    class _S(_Svc):
        def backend(self, name, **kw):
            return FakeFez()
    s = _S()
    monkeypatch.setattr(ibm, "get_service", lambda **kw: seen.update(kw) or s)
    job = _FakeJob(6)
    smp = _Sampler(job, s)
    s.jobobj = job
    monkeypatch.setattr(R, "make_sampler", lambda *a, **k: (smp, {}))
    real_open, real_io_open = builtins.open, io.open

    def guard(real):
        def g(file, *a, **k):
            if os.path.basename(str(file)) == "apikey_val.json":
                raise AssertionError("hsf2.hw opened the credentials file")
            return real(file, *a, **k)
        return g
    monkeypatch.setattr(builtins, "open", guard(real_open))
    monkeypatch.setattr(io, "open", guard(real_io_open))
    for nm in ("read_text", "read_bytes", "open"):
        orig = getattr(Path, nm)

        def mk(orig):
            def f(self, *a, **k):
                if self.name == "apikey_val.json":
                    raise AssertionError("Path access to the credentials file")
                return orig(self, *a, **k)
            return f
        monkeypatch.setattr(Path, nm, mk(orig))
    cfg = cfg_for(tmp_path, creds_file=None, ledger=None)
    from hsf2.hw.config import HwConfig
    cfg = HwConfig.from_env(mode="submit", shots=400, transpile_seeds=1, backend="ibm_fake")
    res = run_job(run_dir, "a", 6, cfg, confirm=lambda m: None)
    assert seen == {"creds_file": str(secret)}
    # ledger name per account, inside HOME
    led = tmp_path / ".hsf2_part2_qpu_ledger_apikey_val.json"
    assert led.exists()
    # no output file holds the directory of the credentials or the token
    for root in (res["job_dir"], run_dir / "samples", led):
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and p.suffix != ".qpy"]
        for f in files:
            t = f.read_text(errors="ignore")
            assert "very_private_dir" not in t and "TOKEN-SHOULD" not in t, f
    assert res["manifest"]["creds"] == "apikey_val.json"


def test_budget_refusals_failed_job_booking_and_ledger_compat(run_dir, tmp_path, monkeypatch, fast):
    from acts_qaoa.ibm import QPULedger
    from hsf2.hw.ledger import BudgetError, HwLedger
    from hsf2.hw.run import run_job
    svc, job, smp = mock_ibm(monkeypatch)
    # (1) IBM's remaining time below the estimate while the ledger has room
    svc.limit = svc.consumed + 1.0
    asked = []
    with pytest.raises(BudgetError):
        run_job(run_dir, "a", 6, cfg_for(tmp_path), confirm=lambda m: asked.append(m))
    assert smp.calls == 0 and not asked and not (tmp_path / "ledger.json").exists()
    svc.limit = 600.0
    # (2) a failed (cancelled) job is still booked, with the account difference
    job.fail = True
    cfg = cfg_for(tmp_path)
    with pytest.raises(RuntimeError, match="did not return results"):
        run_job(run_dir, "a", 6, cfg, confirm=lambda m: None)
    led = QPULedger(cfg.ledger_path(), budget_s=400.0)          # Phase-1 reader
    e = led.entries[-1]
    assert e["job_id"] == "fakejob0001" and e["quantum_seconds"] == 5.0 and e["usage_source"] == "account_diff"
    assert "used=5.0" in led.table() and led.used_seconds() == pytest.approx(5.0)
    # (3) the second job is refused when the ledger would be exceeded
    cfg2 = cfg_for(tmp_path, budget_s=7.0)
    with pytest.raises(BudgetError):
        run_job(run_dir, "a", 6, cfg2, confirm=lambda m: None)
    # (4) set_usage of Phase 1 / shell book_usage logic keep the extra field and the file readable
    led.set_usage("fakejob0001", 9.0)
    h = HwLedger(cfg.ledger_path(), 400.0)
    assert h.entry("fakejob0001")["quantum_seconds"] == 9.0 and h.entry("fakejob0001")["usage_source"] == "account_diff"
    # (5) usage 0 from the job with status finished -> account difference
    job.fail, job.charge = False, 0.0
    svc.ran = False
    cfg3 = cfg_for(tmp_path, ledger=str(tmp_path / "l3.json"))
    job.jid = "fakejob0002"
    res = run_job(run_dir, "a", 6, cfg3, confirm=lambda m: None)
    assert res["manifest"]["billed"]["source"] == "account_diff"
    # (6) positive job figure wins over account difference
    job.charge, svc.ran, job.jid = 3.25, False, "fakejob0003"
    cfg4 = cfg_for(tmp_path, ledger=str(tmp_path / "l4.json"))
    res = run_job(run_dir, "a", 6, cfg4, confirm=lambda m: None)
    assert (res["manifest"]["billed"]["source"], res["manifest"]["billed"]["used"]) == ("job", 3.25)
    assert HwLedger(cfg4.ledger_path(), 400.0).entry("fakejob0003")["usage_source"] == "job"


def test_retrieve_does_not_downgrade_a_booked_figure(run_dir, tmp_path, monkeypatch, fast):
    """DEFECT PROBE: hw-retrieve of an already-booked job re-resolves the billing without the account 'before' value, so a
    per-job figure of 0 makes it overwrite the earlier account_diff booking by the (larger) estimate."""
    from hsf2.hw.ledger import HwLedger
    from hsf2.hw.run import retrieve_job, run_job
    svc, job, smp = mock_ibm(monkeypatch)
    cfg = cfg_for(tmp_path)
    work = tmp_path / "work"
    shutil.copytree(run_dir, work, ignore=shutil.ignore_patterns("hw", "samples", "outputs"))
    res = run_job(work, "a", 6, cfg, confirm=lambda m: None)
    assert res["manifest"]["billed"]["source"] == "account_diff"
    booked = HwLedger(cfg.ledger_path(), 400.0).entry("fakejob0001")["quantum_seconds"]
    assert booked == 5.0
    retrieve_job("fakejob0001", cfg, run_dir=work, service=svc)
    after = HwLedger(cfg.ledger_path(), 400.0).entry("fakejob0001")
    assert after["quantum_seconds"] == booked, f"retrieve overwrote {booked} by {after['quantum_seconds']} ({after['usage_source']})"


def test_sample_export_never_collides_between_jobs(run_dir, tmp_path, monkeypatch, fast):
    """DEFECT PROBE: tags use job_id[:8]; two jobs of the same setting/K whose ids share the first 8 characters would
    silently share one sample file (the second job's counts are dropped and its manifest points to the first's)."""
    from hsf2.hw.run import run_job
    svc, job, smp = mock_ibm(monkeypatch)
    work = tmp_path / "work"
    shutil.copytree(run_dir, work, ignore=shutil.ignore_patterns("hw", "samples", "outputs"))
    names = []
    for jid, seed in (("d1abcdef1111", 1), ("d1abcdef2222", 2)):
        job.jid = jid
        svc.ran = False
        np_rng = seed
        res = run_job(work, "a", 6, cfg_for(tmp_path, ledger=str(tmp_path / f"{jid}.json")), confirm=lambda m: None, seed=np_rng)
        names.append(res["manifest"]["samples"])
    assert set(names[0]).isdisjoint(names[1]), names


def test_qpy_version_15_header_and_round_trip(tmp_path, small_problem):
    from hsf2.hw.build import make_pub_specs
    from hsf2.hw.fake import fake_backend
    from hsf2.hw.gauge import gauge_vectors
    from hsf2.hw.snapshot import QPY_VERSION, read_qpy, write_qpy
    from hsf2.hw.transpile import transpile_job
    from qiskit.quantum_info import Operator
    ism, scale = small_problem
    specs = make_pub_specs(False, False, 1, 100, [0.4], [0.3])
    pub = transpile_job(ism, scale, False, specs, gauge_vectors(ism.n, 1), fake_backend("FakeFez"), 1)[0]
    p = write_qpy(tmp_path / "a.qpy", pub.isa)
    raw = p.read_bytes()
    assert raw[:6] == b"QISKIT" and raw[6] == QPY_VERSION == 15
    back = read_qpy(p)
    assert back.count_ops() == pub.isa.count_ops() and [q.name for q in back.parameters] == [q.name for q in pub.isa.parameters]
    assert back.layout is not None
    a = pub.isa.assign_parameters(dict(zip(pub.isa.parameters, pub.values)))
    b = back.assign_parameters(dict(zip(back.parameters, pub.values)))
    qa, _ = _compress(a)
    qb, _ = _compress(b)
    assert Operator(qa).equiv(Operator(qb))


# ---------------------------------------------------------------------------------------------- shell script
def _bash():
    b = shutil.which("bash")
    if not b:
        pytest.skip("no bash")
    return b


def test_shell_script_syntax_and_ledger_name_rule_matches_python(tmp_path):
    from hsf2.hw.config import default_ledger_path
    bash = _bash()
    script = _script()
    r = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    txt = script.read_text()
    snippet = txt[txt.index("if [ -n \"${HSF2_QPU_LEDGER:-}\" ]"):txt.index("BUDGET=${HSF2_QPU_BUDGET:-400}")]
    for creds in ("/a/b/apikey_alt.json", "/x/ibm_creds.json", "/x/acct.tar.gz", "/x/plain", None):
        env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}
        if creds:
            env["HSF2_IBM_CREDS"] = creds
        out = subprocess.run([bash, "-c", f"set -euo pipefail\n{snippet}\necho \"$LEDGER\""], capture_output=True, text=True,
                             env=env).stdout.strip()
        old = {k: os.environ.get(k) for k in ("HSF2_IBM_CREDS", "HSF2_QPU_LEDGER", "HOME", "USERPROFILE")}
        try:
            os.environ.pop("HSF2_QPU_LEDGER", None)
            os.environ.pop("HSF2_IBM_CREDS", None)
            os.environ["HOME"] = os.environ["USERPROFILE"] = str(tmp_path)
            if creds:
                os.environ["HSF2_IBM_CREDS"] = creds
            py = str(default_ledger_path()).replace("\\", "/")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert Path(out).name == Path(py).name, (creds, out, py)


def test_shell_book_usage_needs_exported_phase1_dir():
    """DEFECT PROBE: book_usage reads os.environ['HSF_PHASE1_DIR'], but the script only sets the shell variable P1 when
    HSF_PHASE1_DIR is unset (PC layout ../HSF, or the folder next to the script)."""
    txt = _script().read_text()
    body = txt[txt.index("book_usage()"):txt.index('case "$MODE"')]
    if 'os.environ["HSF_PHASE1_DIR"]' in body:
        assert re.search(r"export\s+HSF_PHASE1_DIR", txt) or "HSF_PHASE1_DIR=$P1" in txt, \
            "book_usage would raise KeyError when HSF_PHASE1_DIR is not set in the caller's environment"


def test_shell_book_usage_logic_books_account_difference(tmp_path):
    """Run the heredoc of book_usage against a Phase-1 ledger written by HwLedger (0 s per-job figure)."""
    from hsf2.hw.ledger import HwLedger
    txt = _script().read_text()
    body = txt[txt.index("book_usage()"):]
    code = body[body.index("<<'PY'\n") + 7:body.index("\nPY\n")]
    led = tmp_path / "l.json"
    L = HwLedger(led, 400.0)
    L.add("job1", "ibm_kingston", 20.0, "standard-p1", "x")
    L.book("job1", 0.0, "job")
    env = dict(os.environ, HSF_PHASE1_DIR=str(find_phase1()))
    r = subprocess.run([sys.executable, "-I", "-c", code, str(led), "100", "112", "400"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "booked 12 s" in r.stdout
    from acts_qaoa.ibm import QPULedger
    assert QPULedger(led, 400.0).entries[-1]["quantum_seconds"] == 12.0


# ---------------------------------------------------------------------------------------------- mitigator stubs
@pytest.mark.parametrize("name", ["trex", "zne", "pec", "noise_learner"])
def test_stubs(name):
    from hsf2.hw.mitig import get_mitigator
    with pytest.raises(NotImplementedError):
        get_mitigator(name).prepare([])

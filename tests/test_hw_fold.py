"""Folding diagnosis (hsf2.hw.fold, the fold mode of hsf2.hw.zne / hw-zne, hsf2.hw.noisesim): folded circuits are Operator-equivalent
for both gate sets and fractional lambda, the achieved lambda and the gate counts follow the target, the circuit stays ISA, the seed
fixes the random subsets, the noise simulation at zero error is the ideal distribution and its trajectory estimator is exact, the
extrapolation over fractional lambda recovers a known exponential, and the fold-mode jobs keep the runner's safeguards (manifest
and QPY before submission, ledger, budget check before the first prompt, one typed confirmation per job, deployed layout).
0 QPU s: fake backends, Aer, mocks; IBM is never contacted.  Shared helpers come from the runner / component / zne tests."""
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hsf2._phase1 import ensure_phase1, find_phase1  # noqa: E402

pytestmark = pytest.mark.skipif(find_phase1() is None, reason="Phase-1 package acts_qaoa not found")
if find_phase1() is not None:
    ensure_phase1()

SHOTS = 200


def shared(*names):
    for n in names:
        try:
            return importlib.import_module(n)
        except ImportError:
            continue
    pytest.skip(f"none of the shared test modules {names} can be imported")


def runner_tests():
    return shared("test_hw_runner", "test_hsf2_hw_runner")


def component_tests():
    return shared("test_hw_components", "test_hsf2_hw_components")


def cfg_for(tmp_path, **kw):
    from hsf2.hw.config import HwConfig
    d = dict(mode="dry", shots=SHOTS, transpile_seeds=1, ledger=str(tmp_path / "ledger.json"), budget_s=400.0)
    d.update(kw)
    return HwConfig(**d)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    return runner_tests().make_small_run(tmp_path_factory.mktemp("fold") / "r")


@pytest.fixture
def fast_polling(monkeypatch):
    from hsf2.hw import run as R
    for k, v in (("USAGE_TIMEOUT_S", 0.05), ("USAGE_INTERVAL_S", 0.005), ("ACCOUNT_TIMEOUT_S", 0.05), ("ACCOUNT_INTERVAL_S", 0.005)):
        monkeypatch.setattr(R, k, v)


def isa_circuit(n=5, seed=0, measure=False):
    """A bound ISA-native circuit (rz, sx, x, cz) on a line, optionally with measurements into one register."""
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, n if measure else 0)
    for layer in range(4):
        for q in range(n):
            qc.rz(float(rng.uniform(0, 6)), q)
            qc.sx(q)
        qc.x(layer % n)
        for q in range(layer % 2, n - 1, 2):
            qc.cz(q, q + 1)
    if measure:
        qc.barrier()
        qc.measure(range(n), range(n))
    return qc


# ---------------------------------------------------------------------------------------- folding
@pytest.mark.parametrize("gates", ["cz", "all"])
def test_folded_circuits_are_operator_equivalent_for_fractional_lambda(gates):
    from qiskit.quantum_info import Operator
    from hsf2.hw.fold import fold_gates
    qc = isa_circuit(5, 1)
    ref = Operator(qc)
    for lam in (1, 1.25, 1.5, 2, 2.5, 3, 4.5, 5):
        f, rec = fold_gates(qc, lam, gates, seed=3)
        assert Operator(f).equiv(ref), (gates, lam)
        assert f.num_qubits == qc.num_qubits and rec["lam_target"] == lam
    # SX^dagger = RZ(pi) SX RZ(pi): the folded single-qubit gate is the identity up to a phase
    from qiskit import QuantumCircuit
    one = QuantumCircuit(1)
    one.sx(0)
    f, _ = fold_gates(one, 3, "all", seed=0)
    assert [i.operation.name for i in f.data if i.operation.name != "barrier"] == ["sx", "rz", "sx", "rz", "sx"]
    assert Operator(f).equiv(Operator(one))


def test_achieved_lambda_matches_the_target_and_gate_counts_scale():
    from hsf2.hw.fold import fold_gates, folded_pairs
    qc = isa_circuit(6, 2)
    ops = qc.count_ops()
    for gates, names in (("cz", ("cz",)), ("all", ("cz", "sx", "x"))):
        for lam in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
            f, rec = fold_gates(qc, lam, gates, seed=1)
            fo = f.count_ops()
            for g in names:
                n, pr = ops[g], rec["per_gate"][g]
                assert pr["n"] == n and pr["pairs"] == folded_pairs(n, lam)
                assert abs(pr["achieved_lam"] - lam) <= 1.0 / n + 1e-12                # as close as the gate count allows
                assert pr["achieved_lam"] == pytest.approx(1 + 2 * pr["pairs"] / n)
                assert len(pr["folded_indices"]) == pr["pairs"] % n and pr["base_pairs"] == pr["pairs"] // n
            assert fo["cz"] == ops["cz"] + 2 * rec["per_gate"]["cz"]["pairs"]
            if gates == "cz":                                                          # sx, x, rz are untouched
                assert all(fo[g] == ops[g] for g in ("sx", "x", "rz"))
            else:
                assert fo["sx"] == ops["sx"] + 2 * rec["per_gate"]["sx"]["pairs"]
                assert fo["x"] == ops["x"] + 2 * rec["per_gate"]["x"]["pairs"]
                assert fo["rz"] == ops["rz"] + 2 * rec["per_gate"]["sx"]["pairs"]      # two virtual RZ(pi) per folded SX
            if lam % 2 == 1:                                                            # odd integer: every gate the same
                assert rec["per_gate"]["cz"]["pairs"] % ops["cz"] == 0
        # lam = 3 is exactly 3 x the CZ gates; lam = 1 is the circuit
    f, rec = fold_gates(qc, 3, "cz")
    assert f.count_ops()["cz"] == 3 * ops["cz"] and rec["lam_cz"] == 3.0 and rec["lam_eff"] == 3.0
    f1, rec1 = fold_gates(qc, 1, "all")
    assert f1.count_ops() == qc.count_ops() and rec1["lam_eff"] == 1.0
    with pytest.raises(ValueError, match=r"\[1, 9\]"):
        fold_gates(qc, 0.5, "cz")
    with pytest.raises(ValueError, match="finite|\\[1, 9\\]"):
        fold_gates(qc, float("nan"), "cz")
    with pytest.raises(ValueError, match="unknown gate set"):
        fold_gates(qc, 2, "rz")
    with pytest.raises(ValueError, match="only"):
        fold_gates(qc, 2, {"rz"})


def test_folded_circuit_stays_isa_keeps_measurements_and_layout_and_refuses_foreign_gates():
    from qiskit import QuantumCircuit
    from hsf2.hw.fold import ISA_NAMES, fold_gates
    qc = isa_circuit(5, 3, measure=True)
    f, rec = fold_gates(qc, 2.5, "all", seed=2)
    assert set(f.count_ops()) <= set(ISA_NAMES) and set(rec["ops"]) <= set(ISA_NAMES)
    pairs = lambda c: [(c.find_bit(q).index, c.find_bit(b).index) for i in c.data if i.operation.name == "measure"
                       for q, b in zip(i.qubits, i.clbits)]
    assert pairs(f) == pairs(qc) and f.num_clbits == qc.num_clbits
    # barriers keep the folded copies apart: two per inserted pair, plus the one that was there
    n_pairs = sum(p["pairs"] for p in rec["per_gate"].values())
    assert f.count_ops()["barrier"] == qc.count_ops()["barrier"] + 2 * n_pairs
    bad = QuantumCircuit(2)
    bad.cx(0, 1)
    with pytest.raises(ValueError, match="not in the Heron ISA"):
        fold_gates(bad, 2, "cz")
    nocz = QuantumCircuit(2)
    nocz.sx(0)
    with pytest.raises(ValueError, match="to fold"):
        fold_gates(nocz, 2, "cz")


def test_seed_makes_the_subsets_reproducible_and_a_different_seed_changes_them():
    from hsf2.hw.fold import fold_gates
    qc = isa_circuit(6, 4)
    a, ra = fold_gates(qc, 1.5, "all", seed=7)
    b, rb = fold_gates(qc, 1.5, "all", seed=7)
    c, rc = fold_gates(qc, 1.5, "all", seed=8)
    assert ra == rb and a == b
    assert ra["per_gate"]["cz"]["folded_indices"] != rc["per_gate"]["cz"]["folded_indices"] or \
        ra["per_gate"]["sx"]["folded_indices"] != rc["per_gate"]["sx"]["folded_indices"]
    assert ra["seed"] == 7 and rc["seed"] == 8
    # the subset is a subset of the gate indices, without repeats
    idx = ra["per_gate"]["sx"]["folded_indices"]
    assert len(set(idx)) == len(idx) and all(0 <= i < ra["per_gate"]["sx"]["n"] for i in idx)
    # integer lambda: every gate is folded, no randomness
    assert fold_gates(qc, 3, "all", seed=1)[0] == fold_gates(qc, 3, "all", seed=99)[0]


# ---------------------------------------------------------------------------------------- settings
def test_twirled_setting_with_32_randomisations_through_the_settings_mechanism():
    from hsf2.hw.settings import SETTINGS, get_setting, with_randomizations
    assert sorted(SETTINGS) == list("abcdef")                                          # the ladder is unchanged
    d = get_setting("d")
    assert d.options()["twirling.num_randomizations"] == 16 and d.randomizations == 16
    d32 = with_randomizations(d, 32)
    o = d32.options()
    assert d32.name == "d32" and o["twirling.num_randomizations"] == 32 and o["twirling.enable_gates"] and o["twirling.enable_measure"]
    assert o["dynamical_decoupling.enable"] is True and o["dynamical_decoupling.sequence_type"] == "XY4"
    assert with_randomizations(d, 16) is d
    assert get_setting("d32") == d32 and get_setting("d32").options() == o            # the name round-trips (hw-readout --from-job)
    with pytest.raises(ValueError, match="unknown setting"):
        get_setting("x32")
    with pytest.raises(ValueError, match="no twirling"):
        with_randomizations(get_setting("b"), 32)
    with pytest.raises(ValueError, match="positive"):
        with_randomizations(d, 0)


# ---------------------------------------------------------------------------------------- noise simulation
def zero_noise_device(n):
    from hsf2.hw.noisesim import DeviceNoise
    t = {q: 1e9 for q in range(n)}
    g1 = {(g, q): (0.0, 2.4e-8) for g in ("sx", "x") for q in range(n)}
    cz = {(q, q + 1): (0.0, 6.8e-8) for q in range(n - 1)}
    return DeviceNoise(dict(t), dict(t), {q: 0.0 for q in range(n)}, {q: 0.0 for q in range(n)}, g1, cz)


def sim_target(qc, dev, n):
    """A SimTarget of a bound ISA circuit with its exact ideal distribution (Statevector) and random energies."""
    from qiskit.quantum_info import Statevector
    from hsf2.hw import noisesim as NS
    bare = qc.remove_final_measurements(inplace=False) if qc.num_clbits else qc
    p = Statevector(bare).probabilities()
    rng = np.random.default_rng(5)
    E = rng.normal(size=1 << n)
    ctx = NS.make_context(E, p)
    return NS.SimTarget("T", "qx2", bare, None, list(range(n)), dev, ctx, {}, Path("."), NS.readout_matrices(dev, list(range(n)))), p


def test_noise_simulation_at_zero_error_reproduces_the_ideal_distribution_and_observables():
    from hsf2.hw import noisesim as NS
    from hsf2.hw.analysis import f_energy, f_xeb
    n = 5
    qc = isa_circuit(n, 6)
    dev = zero_noise_device(n)
    tgt, p = sim_target(qc, dev, n)
    ops, order = NS.isa_ops(tgt.isa, None, tgt.measured)
    loc = NS.fault_locations(ops, order, dev)
    assert loc.p.size == 0 and loc.p0 == 1.0                                              # no error: no fault location
    q = next(NS.simulate_probs([NS.build_circuit(ops, n, n)], n))
    assert np.allclose(q, p, atol=1e-12)                                                  # bit order and gates are right
    for lam, gates in ((1.0, "cz"), (2.5, "all"), (3.0, "cz")):                          # folded circuits are the same unitary
        r = NS.simulate_config(tgt, lam, gates, 0.0, False, trajectories=5, subsets=2, seed=1)
        assert r.p0 == 1.0 and r.trajectories == 0
        assert r.value[1] == pytest.approx(1.0) and r.value[2] == pytest.approx(1.0) and np.allclose(r.value, r.raw)
        assert r.lam_eff == pytest.approx(lam, abs=0.1)
    # the observables are those of zne.build_observables (the analysis estimators), in expectation
    from hsf2.hw.zne import build_observables
    ctx = tgt.ctx
    idx = np.arange(p.size)
    obs = build_observables(ctx.E, ctx.p, idx, ctx.masks, kmax=0)
    qq = np.random.default_rng(0).dirichlet(np.ones(p.size))
    assert list(ctx.observables(qq)) == pytest.approx(list(qq @ obs.values(idx)))
    assert ctx.observables(qq)[1] == pytest.approx(f_energy(ctx.E, idx, qq, float(ctx.E.mean()), float(ctx.p @ ctx.E)))
    assert ctx.observables(qq)[2] == pytest.approx(f_xeb(ctx.p, idx, qq))
    # a coherent ZZ phase changes the state; with twirling there is no coherent part (the fault-free state is the ideal one)
    r = NS.simulate_config(tgt, 1.0, "cz", 0.2, False, trajectories=0)
    assert r.value[2] != pytest.approx(1.0, abs=1e-3)


def test_trajectory_estimator_is_exact_for_a_pauli_channel_and_the_fault_count_law_is_right():
    """P0 q0 + (1 - P0) E[q | >= 1 fault] equals the density-matrix result of the same Pauli channels (one qubit, one gate), and the
    control-variate mean keeps it."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Kraus, SuperOp
    from hsf2.hw import noisesim as NS
    # two Pauli channels on a 2-qubit circuit: X error after sx on qubit 0, ZZ-type error after cz
    qc = QuantumCircuit(2)
    qc.sx(0)
    qc.sx(1)
    qc.cz(0, 1)
    qc.sx(0)
    ops, order = NS.isa_ops(qc, None, [0, 1])
    dev = zero_noise_device(2)
    dev.gate1[("sx", 0)] = (0.04, 2.4e-8)                  # 1q depolarising: p = 1.5 e = 0.06
    dev.cz[(0, 1)] = (0.08, 6.8e-8)                        # 2q depolarising: p = 1.25 e = 0.1
    loc = NS.fault_locations(ops, order, dev)
    assert list(loc.p) == pytest.approx([0.06, 0.1, 0.06], abs=1e-9) and list(loc.after) == [0, 2, 3]
    exact = DensityMatrix.from_label("00")
    for i, (name, qs, _) in enumerate(ops):
        sub = QuantumCircuit(2)
        getattr(sub, name)(*qs)
        exact = exact.evolve(sub)
        for j in np.nonzero(loc.after == i)[0]:
            ch = np.concatenate([[1.0 - loc.p[j]], loc.p[j] * loc.cond[j]])
            k = len(loc.qubits[j])
            paulis = ["I", "X", "Z", "Y"]
            from qiskit.quantum_info import Pauli
            mats = []
            for code in range(4 ** k):
                digits = [(code >> 2) & 3, code & 3] if k == 2 else [code]
                lab = ["I", "I"]
                for q, dgt in zip(loc.qubits[j], digits):
                    lab[1 - q] = paulis[dgt]                 # qiskit labels are little-endian
                mats.append(Pauli("".join(lab)).to_matrix())
            rho = sum(c * (m @ exact.data @ m.conj().T) for c, m in zip(ch, mats))
            exact = DensityMatrix(rho)
    p_exact = exact.probabilities()
    rng = np.random.default_rng(3)
    T = 3000
    pats = NS.sample_faults(loc, T, rng)
    qs = np.array(list(NS.simulate_probs([NS.build_circuit(ops, 2, 2, 0.0, pat) for pat in pats[:600]], 2)))
    q0 = next(NS.simulate_probs([NS.build_circuit(ops, 2, 2)], 2))
    est = loc.p0 * q0 + (1 - loc.p0) * qs.mean(axis=0)
    assert np.allclose(est, p_exact, atol=0.015)
    ks = np.array([pat[-2][0] for pat in pats])
    P = NS.fault_count_law(loc.p)
    assert P.sum() == pytest.approx(1.0) and P[0] == pytest.approx(loc.p0)
    cond = P[1:] / (1 - P[0])
    assert ks.mean() == pytest.approx(float(np.arange(1, cond.size + 1) @ cond), abs=0.05)
    assert NS.control_expectations(loc.p)[0] == pytest.approx(float(np.arange(1, cond.size + 1) @ cond))
    # control-variate mean: unchanged when the observable does not depend on the controls, exact for a linear dependence
    ctrl = NS.controls(ks)
    obs = np.column_stack([3.0 + 0.5 * ks])
    assert NS.cv_mean(obs, ctrl, NS.control_expectations(loc.p))[0] == pytest.approx(3.0 + 0.5 * float(np.arange(1, cond.size + 1) @ cond))


def test_relaxation_gate_errors_and_scheduling_follow_the_snapshot():
    from hsf2.hw import noisesim as NS
    p = NS.relaxation_pauli(1e-5, 1e-4, 8e-5)
    assert p.sum() == pytest.approx(1.0) and p[1] == p[3] == pytest.approx((1 - np.exp(-0.1)) / 4)
    assert p[2] == pytest.approx((1 - np.exp(-0.125)) / 2 - p[1])
    assert NS.relaxation_pauli(0.0, 1e-4, 1e-4)[0] == 1.0
    d = NS.depolarizing_pauli(0.3, 2)
    assert d.sum() == pytest.approx(1.0) and d[0] == pytest.approx(0.7) and np.allclose(d[1:], 0.02)
    c = NS.convolve(NS.depolarizing_pauli(0.2, 1), NS.depolarizing_pauli(0.2, 1))
    assert c.sum() == pytest.approx(1.0) and c[0] == pytest.approx(0.8 * 0.8 + 3 * (0.2 / 3) ** 2)
    # idle time counts: a qubit that waits for a cz of the other pair relaxes more than one that does not wait
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(3)
    qc.cz(0, 1)
    qc.cz(0, 1)
    qc.sx(2)
    ops, order = NS.isa_ops(qc, None, [0, 1, 2])
    dev = zero_noise_device(3)
    dev.t1 = {q: 1e-4 for q in range(3)}
    dev.t2 = {q: 1e-4 for q in range(3)}
    loc = NS.fault_locations(ops, order, dev, gate_errors=False)
    end = [j for j in range(loc.p.size) if loc.qubits[j] == (2,)]
    assert end and loc.p[end[-1]] > 0                          # qubit 2 idles from its sx until the end of the two cz
    assert NS.fault_locations(ops, order, dev, relaxation=False, gate_errors=False).p.size == 0
    # twirling turns a coherent ZZ phase into a Pauli ZZ error of probability sin^2(phi / 2) per cz
    loc_t = NS.fault_locations(ops, order, zero_noise_device(3), phi=0.4, twirled=True)
    assert loc_t.p.size == 2 and loc_t.p == pytest.approx([np.sin(0.2) ** 2] * 2)
    assert NS.fault_locations(ops, order, zero_noise_device(3), phi=0.4, twirled=False).p.size == 0


def test_device_noise_from_a_snapshot_refuses_missing_values():
    from hsf2.hw.noisesim import device_noise
    q = {"t1": 1e-4, "t2": 1e-4, "readout_error": 0.01, "prob_meas0_prep1": 0.02, "prob_meas1_prep0": 0.01,
         "measure": {"error": 0.01, "duration": 1e-6}, "sx": {"error": 2e-4, "duration": 2.4e-8}, "x": {"error": 2e-4, "duration": 2.4e-8}}
    snap = {"qubits": {"0": q, "1": dict(q)}, "edges": {"0-1": {"cz": {"error": 3e-3, "duration": 6.8e-8}}}}
    dev = device_noise(snap)
    assert dev.cz[(0, 1)] == (3e-3, 6.8e-8) and dev.p01[0] == 0.02 and dev.p10[1] == 0.01 and dev.gate1[("sx", 1)] == (2e-4, 2.4e-8)
    snap["qubits"]["1"] = dict(q, t2=None)
    with pytest.raises(ValueError, match="no t2"):
        device_noise(snap)


# ---------------------------------------------------------------------------------------- extrapolation over fractional lambda
def test_extrapolation_over_fractional_lambda_recovers_a_known_exponential_and_analysis_accepts_float_keys():
    from hsf2.hw.zne import analyze_counts, build_observables, extrapolate
    lam = np.array([1.0, 1.497, 2.006, 3.0])
    floor = np.array([0.0, 0.1])
    f = lambda l: np.stack([0.9 * np.exp(-0.45 * l) + 0.0, 0.8 * np.exp(-0.3 * l) + 0.1], axis=-1)
    Y = np.stack([f(l)[None, :] for l in lam])
    ex = extrapolate(lam, Y, floor)
    assert ex["exponential"][0] == pytest.approx([0.9, 0.9]) and ex["exp_k"][0] == pytest.approx([0.45, 0.3])
    assert abs(ex["linear"][0, 0] - 0.9) > 0.03
    # white noise F(lam) = f^lam at fractional lambda: the bootstrap analysis brackets the ideal value with the exponential model
    n = 8
    rng = np.random.default_rng(2)
    E = rng.normal(size=1 << n)
    p = np.exp(-2.5 * (E - E.min()))
    p /= p.sum()
    N, fq, shots = p.size, 0.7, 100_000
    counts = {}
    for l in (1, 1.5, 2, 3):
        q = fq ** l * p + (1 - fq ** l) / N
        c = rng.multinomial(shots, q)
        counts[l] = {int(i): int(c[i]) for i in np.nonzero(c)[0]}
    idx = np.unique(np.concatenate([np.fromiter(c, np.int64) for c in counts.values()]))
    obs = build_observables(E, p, idx, {}, kmax=0)
    res = analyze_counts(obs, counts, boot_n=200, seed=1, tag="frac")
    assert res["lambdas"] == [1, 1.5, 2, 3] and set(res["observables"]["F_XEB"]["points"]) == {"1", "1.5", "2", "3"}
    e = res["observables"]["F_XEB"]["models"]["exponential"]
    assert abs(e["value"] - 1.0) < 4 * e["err"] and e["k"] == pytest.approx(-np.log(fq), abs=0.03)
    assert res == analyze_counts(obs, counts, boot_n=200, seed=1, tag="frac")        # seeded
    # lambda = 1 and 1.5 do not share a bootstrap stream
    pts = res["observables"]["F_XEB"]["points"]
    assert pts["1"]["err"] > 0 and pts["1.5"]["err"] > 0


def test_extrapolate_arm_over_the_achieved_lambda_with_trajectory_bootstrap():
    from hsf2.hw import noisesim as NS
    rng = np.random.default_rng(1)
    n = 6
    E = rng.normal(size=1 << n)
    p = np.exp(-2.0 * (E - E.min()))
    p /= p.sum()
    ctx = NS.make_context(E, p)
    J = len(ctx.names)
    res = []
    for lam in (1.0, 1.497, 2.006, 3.0):
        T = 200
        obs_t = np.tile(np.exp(-0.4 * lam) * (ctx.ideal - ctx.floor) + ctx.floor, (T, 1)) + rng.normal(scale=0.01, size=(T, J))
        ks = rng.integers(1, 5, T)
        res.append(NS.ConfigResult(lam, lam, "cz", False, 0.0, 10, 5, 1.0, 0.0, T, ctx.ideal, obs_t, NS.controls(ks),
                                   np.array([2.5, 0.25, 0.25]), obs_t.mean(0), obs_t.mean(0)))
    rec = NS.extrapolate_arm(res, ctx, boot=100, seed=0)
    assert rec["lambdas"] == [1, 1.497, 2.006, 3]
    o = rec["observables"]["F_XEB"]
    assert o["models"]["exponential"]["value"] == pytest.approx(1.0, abs=0.03) and o["models"]["exponential"]["err"] < 0.05
    assert o["points"]["1"]["value"] == pytest.approx(np.exp(-0.4), abs=0.01)


# ---------------------------------------------------------------------------------------- jobs: dry run, from-job, submit safeguards
def test_dry_fold_job_has_both_arms_subsets_fold_records_and_keeps_isa_in_qpy(small_run, tmp_path):
    from hsf2.hw.snapshot import read_qpy
    from hsf2.hw.fold import ISA_NAMES
    from hsf2.hw.zne import run_zne
    run = shutil.copytree(small_run, tmp_path / "copy")
    res = run_zne(run, ["6:1,1.5,2,3"], [], "b", cfg_for(tmp_path, mode="dry", shots=4000), command="test", out_dir=tmp_path / "o",
                  fold_gates=["cz", "all"], subsets=2, per_job=21)
    man, jd = res["jobs"][0]["manifest"], res["jobs"][0]["job_dir"]
    assert man["state"] == "dry" and man["job_id"] is None and man["kind"] == "zne"
    pubs = man["pubs"]
    # lambda = 1 once (base), the three other lambdas for each of the two arms; every circuit twice (two subsets), 2 000 shots each
    assert [(p["arm"], p["lam"]) for p in pubs[:2]] == [("base", 1.0)] * 2
    assert len(pubs) == 2 * (1 + 3 + 3) and {p["shots"] for p in pubs} == {2000}
    assert [p["k"] for p in pubs] == list(range(14)) and man["selection"]["fold"]["arms"] == ["cz", "all"]
    base_n2q = pubs[0]["n_2q_base"]
    for p in pubs:
        f = p["fold"]
        assert p["lam_eff"] == pytest.approx(p["lam"], abs=0.2) and f["seed"] in (0, 1000) and f["lam_target"] == p["lam"]
        assert p["n_2q"] == base_n2q + 2 * f["per_gate"]["cz"]["pairs"] and p["subset"] in (0, 1)
        if p["arm"] == "cz":
            assert f["gates"] == ["cz"]
        if p["arm"] == "all":
            assert f["gates"] == ["cz", "sx", "x"]
        isa = read_qpy(jd / f"pub_{p['k']}_{p['name']}.isa.qpy")
        assert set(isa.count_ops()) <= set(ISA_NAMES) and isa.count_ops().get("cz", 0) == p["n_2q"]
        assert (jd / f"pub_{p['k']}.layout.json").exists()
    assert len({tuple(p["measured_qubits"]) for p in pubs}) == 1                          # one layout for every PUB
    sub = [p for p in pubs if p["arm"] == "cz" and p["lam"] == 1.5]
    assert len(sub) == 2 and sub[0]["fold"]["per_gate"]["cz"]["folded_indices"] != sub[1]["fold"]["per_gate"]["cz"]["folded_indices"]
    assert (jd / "snapshot_pre.json").exists() and not (tmp_path / "ledger.json").exists()
    plan = json.loads((tmp_path / "o" / "zne_plan.json").read_text())
    assert len(plan["pubs"]) == 14 and plan["pubs"][3]["arm"] == "cz"
    # subsets must divide the shots, lambda must contain 1
    with pytest.raises(ValueError, match="divide the shots"):
        run_zne(run, ["6"], [], "b", cfg_for(tmp_path, shots=201), fold_gates=["cz"], subsets=2)
    with pytest.raises(ValueError, match="including 1"):
        run_zne(run, ["6:1.5,2"], [], "b", cfg_for(tmp_path), fold_gates=["cz"])
    with pytest.raises(ValueError, match="between 1 and 30"):
        run_zne(run, ["6"], [], "b", cfg_for(tmp_path), fold_gates=["cz"], per_job=31)
    with pytest.raises(ValueError, match="between 1 and 10"):                                   # the legacy limit is unchanged
        run_zne(run, ["6"], [], "b", cfg_for(tmp_path), per_job=11)


def test_cli_dry_run_from_job_reuses_the_stored_circuits_and_a_twirled_job_has_32_randomisations(small_run, tmp_path, monkeypatch):
    """Deployed layout (HSF2_WORK): job 1 (setting b, both arms) and job 2 (setting d with 32 randomisations, CZ folding) from the
    stored lambda = 1 circuits of an earlier zne job: same QPY, same layouts, same angles."""
    from hsf2.cli import main
    from hsf2.hw.snapshot import read_qpy
    runner_tests()._script()                                                                     # the deployed scripts exist
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))
    monkeypatch.delenv("HSF2_IBM_CREDS", raising=False)
    run = tmp_path / "runs" / "r"
    shutil.copytree(small_run, run)
    led = tmp_path / "led.json"
    common = ["--seeds", "1", "--ledger", str(led), "--shots", "400"]
    assert main(["hw-zne", str(run), "--K", "6", "--lambdas", "1", "3"] + common) == 0            # the earlier zne job (legacy)
    earlier = sorted((run / "hw").glob("*_zne_b_*"))
    assert len(earlier) == 1
    em = json.loads((earlier[0] / "manifest.json").read_text())
    assert main(["hw-zne", str(run), "--K", "6", "--from-job", str(earlier[0]), "--fold-gates", "cz", "all", "--lambdas", "1", "1.5", "2", "3"]
                + common) == 0
    jobs = sorted(p for p in (run / "hw").glob("*_zne_b_*") if p not in earlier)
    assert len(jobs) == 1
    m1 = json.loads((jobs[0] / "manifest.json").read_text())
    assert m1["state"] == "dry" and len(m1["pubs"]) == 1 + 3 + 3 and m1["selection"]["fold"]["from_job"] == earlier[0].name
    assert m1["selection"]["fold"]["base_pubs"]["K6"]["qpy_sha1"] and {p["shots"] for p in m1["pubs"]} == {400}
    base = [p for p in em["pubs"] if p["lam"] == 1][0]
    for p in m1["pubs"]:
        assert p["measured_qubits"] == base["measured_qubits"] and p["initial_layout"] == base["initial_layout"]      # same layout
        assert p["gammas"] == base["gammas"] and p["betas"] == base["betas"] and p["fingerprint"] == base["fingerprint"]
    b0 = read_qpy(earlier[0] / f"pub_{base['k']}_{base['name']}.isa.qpy")
    n0 = read_qpy(jobs[0] / f"pub_{m1['pubs'][0]['k']}_{m1['pubs'][0]['name']}.isa.qpy")
    assert n0 == b0                                                                               # the lambda = 1 circuit is the stored one
    # job 2: twirled, CZ-only folding, 32 randomisations, 4 000 shots in total per circuit
    assert main(["hw-zne", str(run), "--K", "6", "--from-job", str(earlier[0]), "--fold-gates", "cz", "--lambdas", "1", "1.5", "2", "3",
                 "--setting", "d", "--randomizations", "32", "--seeds", "1", "--ledger", str(led), "--shots", "4000"]) == 0
    tw = [p for p in (run / "hw").glob("*_zne_d32_*")]
    assert len(tw) == 1
    m2 = json.loads((tw[0] / "manifest.json").read_text())
    assert len(m2["pubs"]) == 4 and m2["setting"]["gate_twirl"] and m2["setting"]["measure_twirl"] and m2["setting"]["dd"]
    assert m2["setting"]["options"]["twirling.num_randomizations"] == 32 and {p["shots"] for p in m2["pubs"]} == {4000}
    assert [p["arm"] for p in m2["pubs"]] == ["base", "cz", "cz", "cz"] and not led.exists()
    with pytest.raises(ValueError, match="of the run folder"):                                       # not a job folder
        main(["hw-zne", str(run), "--K", "6", "--from-job", str(run), "--fold-gates", "cz"])
    # a job of another run folder is refused
    other = tmp_path / "runs" / "other"
    shutil.copytree(small_run, other)
    with pytest.raises(ValueError, match="of the run folder"):
        main(["hw-zne", str(other), "--K", "6", "--from-job", str(earlier[0]), "--fold-gates", "cz"])


def test_fold_job_local_run_and_analysis_group_the_arms_over_the_achieved_lambda(small_run, tmp_path):
    from hsf2.hw.zne import analyze_zne, run_zne
    run = shutil.copytree(small_run, tmp_path / "r")
    res = run_zne(run, ["6:1,1.5,2,3"], [], "b", cfg_for(tmp_path, mode="local", shots=400), command="test",
                  sim_method="matrix_product_state", seed=3, fold_gates=["cz", "all"], subsets=2, per_job=14)
    jd, man = res["jobs"][0]["job_dir"], res["jobs"][0]["manifest"]
    assert man["state"] == "done" and man["billed"]["source"] == "local"
    assert not (run / "samples").exists() or not list((run / "samples").glob("*zne*.json"))   # fold mode exports no Phase-1 samples
    out = analyze_zne([jd], tmp_path / "an", boot_n=50)
    assert sorted(out["targets"]) == ["K6[all]", "K6[cz]"]
    for label in out["targets"]:
        t = out["targets"][label]
        assert t["lambdas"][0] == 1 and len(t["lambdas"]) == 4 and all(isinstance(x, (int, float)) for x in t["lambdas"])
        assert t["shots"][1] == 400                                                             # the two base PUBs (2 x 200) are merged
        assert t["shots"][t["lambdas"][1]] == 400
        assert set(t["observables"]["F_XEB"]["models"]) == {"linear", "richardson", "exponential"}
    pts = (tmp_path / "an" / "zne_points.csv").read_text().splitlines()
    assert len(pts) > 1 and "K6[cz]" in pts[1]


def test_fold_job_submit_keeps_the_runner_safeguards(small_run, tmp_path, monkeypatch, fast_polling):
    """Mocked IBM: manifest and QPY are on disk before the sampler is called, one typed confirmation per job (preceded by the PUBs),
    the budget is checked before the first prompt, every PUB has the same shots, the ledger books the job."""
    from acts_qaoa.ibm import QPULedger
    from hsf2.hw.ledger import BudgetError
    from hsf2.hw.run import SubmitAborted
    from hsf2.hw.zne import run_zne
    C = component_tests()
    run = tmp_path / "r"
    shutil.copytree(small_run, run)
    svc, smp = C.mock_ibm(monkeypatch)
    seen, on_disk = [], []
    orig = smp.run

    def spy(pubs, **kw):
        jobs = sorted((run / "hw").glob("*_zne_b_*"))
        man = json.loads((jobs[-1] / "manifest.json").read_text())
        on_disk.append((man["state"], len(list(jobs[-1].glob("*.isa.qpy"))), (jobs[-1] / "snapshot_pre.json").exists(), len(man["pubs"])))
        seen.append([(p[0], p[2]) for p in pubs])
        return orig(pubs, **kw)
    smp.run = spy
    cfg = cfg_for(tmp_path, mode="submit", backend="ibm_fake", shots=400)
    asked = []
    res = run_zne(run, ["6:1,1.5,2,3"], [], "b", cfg, command="test", confirm=asked.append, fold_gates=["cz", "all"], subsets=1, per_job=7)
    assert len(res["jobs"]) == 1 and len(asked) == 1 and smp.calls == 1
    assert on_disk == [("planned", 7, True, 7)]                                                   # before submission, state planned
    assert "submit ONE job" in asked[0] and "ZNE job 1 of 1" in asked[0] and "K6 lambda=1.5 cz s0" in asked[0] and "K6 lambda=3 all s0" in asked[0]
    assert all(s == 400 for sub in seen for _, s in sub) and len(seen[0]) == 7
    for (isa, _), p in zip(seen[0], res["jobs"][0]["manifest"]["pubs"]):
        assert isa.count_ops().get("cz", 0) == p["n_2q"] and set(isa.count_ops()) <= {"cz", "sx", "x", "rz", "measure", "barrier", "delay"}
    m = res["jobs"][0]["manifest"]
    assert m["state"] == "done" and m["billed"]["used"] == 5.0 and (res["jobs"][0]["job_dir"] / "snapshot_post.json").exists()
    assert [e["job_id"] for e in QPULedger(cfg.ledger_path(), budget_s=400.0).entries] == ["fakejob0001"]
    assert not list((run / "samples").glob("*zne*")) if (run / "samples").exists() else True
    # budget refusal before the first prompt and before any submission
    calls = smp.calls
    asked.clear()
    with pytest.raises(BudgetError, match="exceeds"):
        run_zne(run, ["6:1,1.5"], [], "b", cfg_for(tmp_path, mode="submit", backend="ibm_fake", budget_s=1.0, ledger=str(tmp_path / "l2.json")),
                confirm=asked.append, fold_gates=["cz"])
    assert smp.calls == calls and not asked and not (tmp_path / "l2.json").exists()
    # a refused confirmation submits nothing and says why
    def no(msg):
        raise SubmitAborted("no")
    with pytest.raises(SubmitAborted):
        run_zne(run, ["6:1,1.5"], [], "b", cfg_for(tmp_path, mode="submit", backend="ibm_fake", ledger=str(tmp_path / "l3.json")),
                confirm=no, fold_gates=["cz"])
    assert smp.calls == calls
    aborted = [json.loads(p.read_text()) for p in (run / "hw").glob("*/manifest.json")]
    assert [m["state"] for m in aborted].count("aborted") == 1

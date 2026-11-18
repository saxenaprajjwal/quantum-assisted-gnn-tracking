"""Independent validation of step 4 (zero-noise extrapolation, read-out mitigation inside the analysis).

Written by the validation engineer, not the implementer.  0 QPU s: Operator / numpy / brute-force references re-derived here,
FakeFez, mocks; IBM is never contacted.  Works in the repository layout (test_hw_*.py, scripts/) and as deployed on lxplus
(test_hsf2_hw_*.py, hsf2_scripts/): shared helpers are imported under either name.
"""
import importlib
import inspect
import itertools
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
    return runner_tests().make_small_run(tmp_path_factory.mktemp("val4") / "r")


@pytest.fixture
def fast_polling(monkeypatch):
    from hsf2.hw import run as R
    for k, v in (("USAGE_TIMEOUT_S", 0.05), ("USAGE_INTERVAL_S", 0.005), ("ACCOUNT_TIMEOUT_S", 0.05),
                 ("ACCOUNT_INTERVAL_S", 0.005)):
        monkeypatch.setattr(R, k, v)


# ======================================================================================== 1. folding on the real ISA
def _active_unitary(isa, values, max_active=10):
    """Operator of the bound ISA circuit restricted to the qubits that carry gates (measurements and barriers dropped)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from hsf2.hw.fake import bound_circuit
    c = bound_circuit(isa, values).remove_final_measurements(inplace=False)
    act = sorted({c.find_bit(q).index for i in c.data if i.operation.name != "barrier" for q in i.qubits})
    if len(act) > max_active:
        pytest.skip(f"{len(act)} active qubits: too many for an Operator")
    pos = {q: k for k, q in enumerate(act)}
    out = QuantumCircuit(len(act))
    for i in c.data:
        if i.operation.name in ("barrier", "delay"):
            continue
        out.append(i.operation, [pos[c.find_bit(q).index] for q in i.qubits])
    return Operator(out), act


def _signature_1q(c):
    return [(i.operation.name, tuple(str(p) for p in i.operation.params), tuple(c.find_bit(q).index for q in i.qubits))
            for i in c.data if len(i.qubits) == 1 and i.operation.name not in ("barrier", "measure", "delay")]


def _measure_map(c):
    return [(c.find_bit(q).index, c.find_bit(b).index) for i in c.data if i.operation.name == "measure"
            for q, b in zip(i.qubits, i.clbits)]


def _check_fold_of_base(tg, backend, lams=(3, 5)):
    from hsf2.hw.zne import ZnePubSpec, fold_pub
    base = tg.base
    u1, act = _active_unitary(base.isa, base.values)
    for lam in lams:
        spec = ZnePubSpec(0, "main", 0, list(tg.info["gammas"]), list(tg.info["betas"]), 100, False, tg.label, lam)
        f = fold_pub(base, spec, backend)
        uf, act_f = _active_unitary(f.isa, f.values)
        assert act_f == act and uf.equiv(u1), f"lambda {lam}: not the same unitary on the active qubits {act}"
        assert f.stats["n_2q"] == lam * base.stats["n_2q"]
        assert _signature_1q(f.isa) == _signature_1q(base.isa)                       # single-qubit gates unchanged, in order
        assert _measure_map(f.isa) == _measure_map(base.isa)                         # measurement map unchanged
        assert f.layout == base.layout and f.values is base.values or np.array_equal(f.values, base.values)
        assert [p.name for p in f.isa.parameters] == [p.name for p in base.isa.parameters]
        # every inserted G^dagger G pair is fenced by barriers on its own two qubits: per qubit, between two consecutive
        # two-qubit gates of the same pair that come from one folded gate there is a barrier
        last = {}
        for i in f.isa.data:
            nm = i.operation.name
            qs = tuple(f.isa.find_bit(q).index for q in i.qubits)
            if nm == "barrier":
                for q in qs:
                    last[q] = "barrier"
            elif len(qs) == 2:
                tok = ("2q", tuple(sorted(qs)))
                if all(last.get(q) == tok for q in qs):
                    raise AssertionError(f"two adjacent two-qubit gates on {qs} without a barrier: they could cancel")
                for q in qs:
                    last[q] = tok
            else:
                for q in qs:
                    last[q] = nm
        n_barrier = f.isa.count_ops().get("barrier", 0) - base.isa.count_ops().get("barrier", 0)
        assert n_barrier == (lam - 1) * base.stats["n_2q"]


def test_real_fakefez_isa_of_k12_folds_to_the_same_unitary_with_lambda_times_the_2q_gates(small_run, tmp_path):
    """6-variable problem, K = 12 of 15 couplings, setting b on FakeFez (the runner's own transpile path)."""
    from hsf2.hw import run as R
    from hsf2.hw.settings import get_setting
    from hsf2.hw.zne import plan_qx2_targets
    cfg = cfg_for(tmp_path)
    setting = get_setting("b")
    backend, _ = R.get_job_backend(cfg, setting, None)
    run = shutil.copytree(small_run, tmp_path / "copy")
    tg = plan_qx2_targets(run, ["12:1,3,5"], [1, 3, 5], setting, cfg, backend)[0]
    assert tg.base.stats["n_2q"] > 10
    _check_fold_of_base(tg, backend)


def test_real_fakefez_isa_of_a_component_folds_to_the_same_unitary(tmp_path):
    from hsf2.hw import run as R
    from hsf2.hw.settings import get_setting
    from hsf2.hw.zne import plan_component_targets
    C = component_tests()
    run = C.write_run(tmp_path / "tri", C.HAND)
    cfg = cfg_for(tmp_path)
    setting = get_setting("b")
    backend, _ = R.get_job_backend(cfg, setting, None)
    tg = plan_component_targets(run, [f"{C.EVENT}_s0_f1_c2:1,3,5"], [1, 3, 5], setting, cfg, backend)[0]     # 8 variables
    assert tg.info["n_vars"] == 8
    _check_fold_of_base(tg, backend)


def test_nothing_between_the_pub_and_the_sampler_can_cancel_the_folded_pairs():
    """The PUB tuples sent by the runner are (p.isa, values, shots) of the folded circuits; execute_job / make_sampler contain
    no transpile, optimisation or barrier removal, and the only circuit-modifying options of the setting are runtime DD /
    twirling (server side; inserted on idle qubits / around gates, never merging gates)."""
    from hsf2.hw import run as R
    from hsf2.hw import zne as Z
    src = inspect.getsource(R.execute_job) + inspect.getsource(R.make_sampler) + inspect.getsource(R.pub_tuples)
    for bad in ("transpile(", "optimization_level", "RemoveBarriers", "remove_barriers", "PassManager", "Optimize1q",
                "CommutativeCancellation", "InverseCancellation"):
        assert bad not in src, bad
    src_z = inspect.getsource(Z.fold_two_qubit_gates) + inspect.getsource(Z.fold_pub) + inspect.getsource(Z.plan_zne_jobs)
    for bad in ("transpile(", "optimization_level", "RemoveBarriers", "PassManager"):
        assert bad not in src_z, bad
    from hsf2.hw.settings import get_setting
    opts = get_setting("b").options()
    assert "dynamical_decoupling" in json.dumps(opts) and "twirling" in json.dumps(opts)
    assert not get_setting("b").gate_twirl


def test_sampler_receives_the_folded_pubs_with_their_barriers_lambda_times_the_cz(small_run, tmp_path, monkeypatch, fast_polling):
    from hsf2.hw.zne import run_zne
    C = component_tests()
    run = tmp_path / "r"
    shutil.copytree(small_run, run)
    svc, smp = C.mock_ibm(monkeypatch)
    seen = []
    orig = smp.run
    smp.run = lambda pubs, **kw: (seen.append(list(pubs)), orig(pubs, **kw))[1]
    asked = []
    cfg = cfg_for(tmp_path, mode="submit", backend="ibm_fake")
    res = run_zne(run, ["12:1,3,5"], [], "b", cfg, confirm=asked.append)
    assert len(asked) == 1 and len(seen) == 1 and len(seen[0]) == 3
    n2q = [p[0].count_ops().get("cz", 0) for p in seen[0]]
    assert n2q[1] == 3 * n2q[0] and n2q[2] == 5 * n2q[0]
    assert [p[2] for p in seen[0]] == [SHOTS] * 3
    assert [p[0].count_ops().get("barrier", 0) - seen[0][0][0].count_ops().get("barrier", 0) for p in seen[0]] == \
        [0, 2 * n2q[0], 4 * n2q[0]]
    assert [p["lam"] for p in res["jobs"][0]["manifest"]["pubs"]] == [1, 3, 5]
    names = sorted(x.name for x in (run / "samples").glob("*.json"))
    assert [n.rsplit("_", 1)[1] for n in names] == ["zne1.json", "zne3.json", "zne5.json"]     # lambda is in the sample tags


# ======================================================================================== 2. extrapolation maths
def test_extrapolators_agree_with_numpy_polyfit_and_the_log_linear_fit_on_random_data():
    from hsf2.hw.zne import extrapolate, richardson_weights
    rng = np.random.default_rng(0)
    for lam in ([1, 3], [1, 3, 5], [1, 3, 5, 7, 9], [1, 5]):
        lam = np.array(lam, float)
        Y = rng.uniform(0.3, 0.9, size=(lam.size, 4, 3))
        floor = np.array([0.0, 0.05, 0.1])
        ex = extrapolate(lam, Y, floor)
        for r in range(4):
            for j in range(3):
                y = Y[:, r, j]
                assert ex["linear"][r, j] == pytest.approx(np.polyfit(lam, y, 1)[1])
                assert ex["richardson"][r, j] == pytest.approx(np.polyfit(lam, y, lam.size - 1)[-1], abs=1e-8)
                a, b = np.polyfit(lam, np.log(y - floor[j]), 1)                    # log (y - B) = log A - k lambda
                assert ex["exponential"][r, j] == pytest.approx(floor[j] + np.exp(b))
                assert ex["exp_k"][r, j] == pytest.approx(-a)
        assert richardson_weights(lam) @ lam ** 0 == pytest.approx(1.0)
        if lam.size > 1:
            assert richardson_weights(lam) @ lam == pytest.approx(0.0, abs=1e-12)    # reproduces a line exactly


def test_exponential_with_two_lambdas_is_exact_and_floor_dependence_is_as_documented():
    from hsf2.hw.zne import extrapolate
    A, k, B = 0.7, 0.35, 0.04
    lam = np.array([1.0, 3.0])
    Y = (B + A * np.exp(-k * lam))[:, None, None]
    assert extrapolate(lam, Y, np.array([B]))["exponential"][0, 0] == pytest.approx(B + A)
    # a wrong floor changes the answer (it is fixed, not fitted): this is the model's single assumption
    assert abs(extrapolate(lam, Y, np.array([0.0]))["exponential"][0, 0] - (B + A)) > 1e-3
    # y below the floor for all lambda (e.g. an observable that decays to below uniform): sign-consistent, finite
    Yn = (B - 0.2 * np.exp(-k * lam))[:, None, None]
    assert extrapolate(lam, Yn, np.array([B]))["exponential"][0, 0] == pytest.approx(B - 0.2)


def _synthetic(n=7, seed=1):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=1 << n)
    p = np.exp(-2.0 * (E - E.min()))
    p /= p.sum()
    return E, p


def test_bootstrap_resamples_the_shots_independently_per_lambda_and_is_deterministic():
    from hsf2.hw.zne import analyze_counts, build_observables
    E, p = _synthetic()
    N = p.size
    rank = np.empty(N, np.int32)
    rank[np.argsort(E, kind="stable")] = np.arange(N, dtype=np.int32)
    masks = {"P_best_10pct": rank < int(round(0.1 * N))}
    rng = np.random.default_rng(5)
    S = 400
    d1 = rng.multinomial(S, 0.5 * p + 0.5 / N)
    d3 = rng.multinomial(S, 0.2 * p + 0.8 / N)
    cb = {1: {int(i): int(d1[i]) for i in np.nonzero(d1)[0]}, 3: {int(i): int(d3[i]) for i in np.nonzero(d3)[0]}}
    idx_all = np.unique(np.concatenate([np.fromiter(c, np.int64) for c in cb.values()]))
    obs = build_observables(E, p, idx_all, masks)
    a = analyze_counts(obs, cb, boot_n=2000, seed=3, tag="K12")
    b = analyze_counts(obs, cb, boot_n=2000, seed=3, tag="K12")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)                  # same seed -> same result
    c = analyze_counts(obs, cb, boot_n=2000, seed=4, tag="K12")
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)
    o = a["observables"]["P_best_10pct"]
    m = masks["P_best_10pct"]
    # point errors are binomial shot noise
    for lam, d in ((1, d1), (3, d3)):
        ph = d[m].sum() / S
        assert o["points"][str(lam)]["value"] == pytest.approx(ph)
        assert o["points"][str(lam)]["err"] == pytest.approx(np.sqrt(ph * (1 - ph) / S), rel=0.1)
    # independent resampling per lambda: Var(1.5 y1 - 0.5 y3) = 2.25 v1 + 0.25 v3 (a shared stream would give (1.5 e1 - 0.5 e3)^2)
    e1, e3 = o["points"]["1"]["err"], o["points"]["3"]["err"]
    lin = o["models"]["linear"]
    assert lin["value"] == pytest.approx(1.5 * o["points"]["1"]["value"] - 0.5 * o["points"]["3"]["value"])
    assert lin["err"] == pytest.approx(np.sqrt(2.25 * e1 ** 2 + 0.25 * e3 ** 2), rel=0.08)
    assert abs(lin["err"] - abs(1.5 * e1 - 0.5 * e3)) > 0.05 * lin["err"]
    # documented outputs: ideal value, bracket and z
    assert o["ideal"] == pytest.approx(float(p[m].sum())) and o["floor"] == pytest.approx(m.mean())
    for name, rec in a["observables"].items():
        for mod, r in rec["models"].items():
            if r["err"] is not None:
                assert r["brackets_ideal"] == bool(r["lo68"] <= rec["ideal"] <= r["hi68"]), (name, mod)
                if r["err"] > 0:
                    assert r["z"] == pytest.approx((r["value"] - rec["ideal"]) / r["err"])
    with pytest.raises(ValueError, match="at least two"):
        analyze_counts(obs, {1: cb[1]}, boot_n=10)


# ======================================================================================== 3. read-out
def _asym_mats(n, seed):
    rng = np.random.default_rng(seed)
    return [np.array([[1 - a, b], [a, 1 - b]]) for a, b in zip(rng.uniform(0.0, 0.15, n), rng.uniform(0.0, 0.3, n))]


def _kron_full(mats):
    """P(read | true) on all 2^n states, bit k of the index <-> mats[k]; independent construction by explicit enumeration."""
    n = len(mats)
    A = np.empty((1 << n, 1 << n))
    for r, t in itertools.product(range(1 << n), repeat=2):
        A[r, t] = np.prod([mats[k][(r >> k) & 1, (t >> k) & 1] for k in range(n)])
    return A


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_tensor_inverse_equals_the_explicit_linear_solve_with_asymmetric_matrices(n):
    from hsf2.hw.readout import tensor_inverse
    mats = _asym_mats(n, n)
    A = _kron_full(mats)
    assert np.allclose(A.sum(axis=0), 1.0)
    rng = np.random.default_rng(10 + n)
    counts = {int(i): int(v) for i, v in enumerate(rng.integers(0, 50, 1 << n)) if v}
    v = np.zeros(1 << n)
    for k, c in counts.items():
        v[k] = c
    v /= v.sum()
    q = tensor_inverse(counts, mats)
    assert q.shape == (1 << n,) and q.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(q, np.linalg.solve(A, v), atol=1e-10)
    assert np.allclose(A @ q, v, atol=1e-10)                                           # forward-model round trip
    if n >= 3:
        assert (q < 0).any()                                                           # quasi-probabilities do go negative


def test_tensor_inverse_bit_order_single_flip():
    """Bit 0 of the index is mats[0]: a qubit-0 flip probability must move weight between index 0 and 1, not 0 and 2."""
    from hsf2.hw.readout import tensor_inverse
    m0 = np.array([[0.9, 0.0], [0.1, 1.0]])               # qubit 0 reads 1 with 10 % when prepared 0
    m1 = np.eye(2)
    q = tensor_inverse({0: 90, 1: 10}, [m0, m1])
    assert q == pytest.approx([1.0, 0.0, 0.0, 0.0])
    q = tensor_inverse({0: 90, 2: 10}, [m1, m0])
    assert q == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_weighted_estimators_with_negative_quasi_probabilities_equal_dense_definitions():
    from hsf2.hw.analysis import correlation_fidelity, f_energy, f_xeb, tail
    from hsf2.hw.readout import tensor_inverse
    n = 5
    E, p = _synthetic(n, 3)
    N = 1 << n
    mats = _asym_mats(n, 8)
    rng = np.random.default_rng(1)
    S = 3000
    counts = {int(i): int(c) for i, c in enumerate(rng.multinomial(S, 0.5 * p + 0.5 / N)) if c}
    q = tensor_inverse(counts, mats)
    assert (q < 0).any()
    w, idx = q * S, np.arange(N)
    E_unif, e_ideal = float(E.mean()), float(p @ E)
    assert f_energy(E, idx, w, E_unif, e_ideal) == pytest.approx((E_unif - q @ E) / (E_unif - e_ideal))
    assert f_xeb(p, idx, w) == pytest.approx((q @ p - 1 / N) / (p @ p - 1 / N))
    rank = np.empty(N, np.int32)
    rank[np.argsort(E, kind="stable")] = np.arange(N, dtype=np.int32)
    for qq in (0.1, 0.25):
        top = rank < int(round(qq * N))
        t = tail(rank, p, idx, w, qq)
        assert t["p_hw"] == pytest.approx(q[top].sum()) and t["tail_F"] == pytest.approx((q[top].sum() - qq) / (p[top].sum() - qq))
    # s_k from the definition: c_S = sum_x p(x) (-1)^{S.x}, chat_S = sum_x q(x) (-1)^{S.x}, s_k = sum_{|S|=k} c chat / sum c^2
    H = np.array([[(-1) ** bin(s & x).count("1") for x in range(N)] for s in range(N)], float)
    c, ch = H @ p, H @ q
    wt = np.array([bin(s).count("1") for s in range(N)])
    ref = [float(c[wt == k] @ ch[wt == k] / (c[wt == k] @ c[wt == k])) for k in (1, 2, 3)]
    sk = correlation_fidelity(p, [(idx, w)], 3)[0]
    assert [s for s, _ in sk] == pytest.approx(ref)
    # linearity: mitigating then estimating == estimating the dense forward-model inverse (the estimators are linear in q)
    A = _kron_full(mats)
    v = np.zeros(N)
    for k, cn in counts.items():
        v[k] = cn / S
    assert f_energy(E, idx, np.linalg.solve(A, v) * S, E_unif, e_ideal) == pytest.approx(f_energy(E, idx, w, E_unif, e_ideal))


def test_mitigation_improves_a_known_noisy_distribution_in_f_e_f_xeb_and_the_tails():
    """Forward-model a known distribution with read-out noise (exact, no sampling noise): the mitigated estimators return the
    ideal values, the raw ones are below them."""
    from hsf2.hw.analysis import f_energy, f_xeb
    from hsf2.hw.readout import tensor_inverse
    n = 6
    E, p = _synthetic(n, 4)
    N = 1 << n
    mats = _asym_mats(n, 2)
    p_obs = _kron_full(mats) @ p
    E_unif, e_ideal = float(E.mean()), float(p @ E)
    idx = np.arange(N)
    q = tensor_inverse({int(i): float(v) * 1e9 for i, v in enumerate(p_obs)}, mats)
    assert f_energy(E, idx, q, E_unif, e_ideal) == pytest.approx(1.0, abs=1e-9)
    assert f_xeb(p, idx, q) == pytest.approx(1.0, abs=1e-9)
    assert f_energy(E, idx, p_obs, E_unif, e_ideal) < 1.0 - 1e-3 and f_xeb(p, idx, p_obs) < 1.0 - 1e-3


def test_calibration_for_checks_and_subset_slicing_in_the_requested_order():
    from hsf2.hw.readout import calibration_for
    mats = {str(q): [[1 - 0.01 * q, 0.02], [0.01 * q, 0.98]] for q in (3, 5, 9, 12)}
    cal = {"measure_twirl": False, "qubits": [3, 5, 9, 12], "matrices": mats, "backend": "ibm_fez"}
    assert calibration_for(cal, [3, 5, 9, 12], False, "ibm_fez") is cal
    for bad_args in (dict(qubits=[3, 5, 9, 12], measure_twirl=True), dict(qubits=[3, 5, 9, 12], measure_twirl=False, backend="ibm_x"),
                     dict(qubits=[5, 3, 9, 12], measure_twirl=False), dict(qubits=[3, 5, 9], measure_twirl=False),
                     dict(qubits=[3, 5, 9, 13], measure_twirl=False, subset=True)):
        with pytest.raises(ValueError):
            calibration_for(cal, **bad_args)
    sub = calibration_for(cal, [12, 3, 9], False, "ibm_fez", subset=True)
    assert sub["qubits"] == [12, 3, 9] and list(sub["matrices"]) == ["12", "3", "9"]
    assert sub["matrices"]["12"] == mats["12"] and sub["matrices"]["3"] == mats["3"]
    twirled = dict(cal, measure_twirl=True)
    with pytest.raises(ValueError, match="twirl"):
        calibration_for(twirled, [3, 5, 9, 12], False)
    assert calibration_for(twirled, [3, 5, 9, 12], True) is twirled


# ======================================================================================== 4. safety
def test_budget_of_all_jobs_is_checked_before_the_first_confirmation_even_when_the_first_job_fits(small_run, tmp_path, monkeypatch,
                                                                                               fast_polling):
    from hsf2.hw.ledger import BudgetError
    from hsf2.hw.zne import run_zne
    C = component_tests()
    run = tmp_path / "r"
    shutil.copytree(small_run, run)
    svc, smp = C.mock_ibm(monkeypatch)
    plan = run_zne(run, ["12:1,3,5"], [], "b", cfg_for(tmp_path, ledger=str(tmp_path / "dry.json")), per_job=2)
    e = [j["job"].estimate_s for j in plan["jobs"]]
    assert len(e) == 2 and plan["estimate_total_s"] == pytest.approx(sum(e))
    budget = (e[0] + sum(e)) / 2                                  # the first job fits, the two together do not
    assert e[0] < budget < sum(e)
    n_dirs = len(list((run / "hw").iterdir()))
    asked = []
    cfg = cfg_for(tmp_path, mode="submit", backend="ibm_fake", budget_s=budget, ledger=str(tmp_path / "tight.json"),
                  reserve_s=0.0) if "reserve_s" in __import__("hsf2.hw.config", fromlist=["HwConfig"]).HwConfig.__dataclass_fields__ \
        else cfg_for(tmp_path, mode="submit", backend="ibm_fake", budget_s=budget, ledger=str(tmp_path / "tight.json"))
    with pytest.raises(BudgetError):
        run_zne(run, ["12:1,3,5"], [], "b", cfg, per_job=2, confirm=asked.append)
    assert smp.calls == 0 and not asked and len(list((run / "hw").iterdir())) == n_dirs


def test_manifest_records_credentials_by_basename_only_and_at_most_ten_pubs_per_job(small_run, tmp_path):
    from hsf2.hw.zne import run_zne
    run = shutil.copytree(small_run, tmp_path / "copy")
    secret = tmp_path / "very_private_dir" / "my_key.json"                                  # never opened
    res = run_zne(run, ["12:1,3,5,7", "6:1,3,5,7", "4:1,3,5"], [], "b", cfg_for(tmp_path, creds_file=str(secret)))
    sizes = [len(j["manifest"]["pubs"]) for j in res["jobs"]]
    assert sizes == [10, 1] and sum(sizes) == 11
    for j in res["jobs"]:
        txt = (j["job_dir"] / "manifest.json").read_text()
        assert "very_private_dir" not in txt and j["manifest"]["creds"] == "my_key.json"
        assert {p["shots"] for p in j["manifest"]["pubs"]} == {SHOTS}
    assert not secret.parent.exists()

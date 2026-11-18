"""Zero-noise extrapolation (hsf2.hw.zne, CLI hw-zne / hw-zne-analyze): folded circuits (unitary, 2q count, barriers), equal-shot
jobs with one confirmation each and the budget refusal (mocked IBM), local end to end for a QX2 problem and for components,
extrapolators on known lambda dependences, bootstrap analysis.  0 QPU s: fake backends, Aer, mocks; IBM is never contacted.
Shared helpers come from the runner / component tests under either name (repository test_hw_*.py, lxplus test_hsf2_hw_*.py)."""
import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
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


def snapshot_tree(root, skip):
    return {str(p.relative_to(root)) for p in root.rglob("*") if skip not in p.parents and p != skip}


def cfg_for(tmp_path, **kw):
    from hsf2.hw.config import HwConfig
    d = dict(mode="dry", shots=SHOTS, transpile_seeds=1, ledger=str(tmp_path / "ledger.json"), budget_s=400.0)
    d.update(kw)
    return HwConfig(**d)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    return runner_tests().make_small_run(tmp_path_factory.mktemp("zne") / "r")


@pytest.fixture
def fast_polling(monkeypatch):
    from hsf2.hw import run as R
    for k, v in (("USAGE_TIMEOUT_S", 0.05), ("USAGE_INTERVAL_S", 0.005), ("ACCOUNT_TIMEOUT_S", 0.05),
                 ("ACCOUNT_INTERVAL_S", 0.005)):
        monkeypatch.setattr(R, k, v)


# ---------------------------------------------------------------------------------------- folding
def line_circuit(n=4, seed=0, kinds=("cz",)):
    """A bound n-qubit circuit of rz / sx / two-qubit gates on a line (no measurements)."""
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for layer in range(3):
        for q in range(n):
            qc.rz(float(rng.uniform(0, 6)), q)
            qc.sx(q)
        for q in range(layer % 2, n - 1, 2):
            kind = kinds[(q + layer) % len(kinds)]
            if kind == "cz":
                qc.cz(q, q + 1)
            elif kind == "ecr":
                qc.ecr(q, q + 1)
            else:
                qc.rzz(float(rng.uniform(0.1, 1.5)), q, q + 1)
    return qc


@pytest.mark.parametrize("kinds", [("cz",), ("ecr", "rzz", "cz")])
def test_folded_circuit_is_the_same_unitary_for_odd_lambda_with_lambda_times_the_2q_gates(kinds):
    from qiskit.quantum_info import Operator
    from hsf2.hw.zne import fold_two_qubit_gates
    qc = line_circuit(4, 1, kinds)
    n2q = sum(1 for i in qc.data if len(i.qubits) == 2)
    ref = Operator(qc)
    for lam in (1, 3, 5, 7):
        f = fold_two_qubit_gates(qc, lam)
        assert Operator(f).equiv(ref), lam
        assert sum(1 for i in f.data if len(i.qubits) == 2 and i.operation.name != "barrier") == lam * n2q
        assert f.count_ops().get("barrier", 0) == (lam - 1) * n2q            # two barriers per inserted G^dagger G pair
        assert f.num_qubits == qc.num_qubits and f.depth() >= qc.depth()
    # the folded gates sit next to their original, in the order G, barrier, G^dagger, barrier, G
    f = fold_two_qubit_gates(qc, 3)
    names = [i.operation.name for i in f.data]
    k = next(j for j, i in enumerate(f.data) if len(i.qubits) == 2)
    assert names[k:k + 5] == [qc.data[next(j for j, i in enumerate(qc.data) if len(i.qubits) == 2)].operation.name, "barrier",
                              f.data[k + 2].operation.name, "barrier", f.data[k].operation.name]
    assert fold_two_qubit_gates(qc, 1) is not qc and Operator(fold_two_qubit_gates(qc, 1)).equiv(ref)


def test_fold_keeps_parameters_measurements_registers_and_layout_and_refuses_bad_lambda():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from hsf2.hw.zne import check_lambda, fold_two_qubit_gates
    th = Parameter("theta")
    qc = QuantumCircuit(3)
    qc.rz(th, 0)
    qc.cz(0, 1)
    qc.cz(1, 2)
    qc.measure_all()
    f = fold_two_qubit_gates(qc, 3)
    assert [p.name for p in f.parameters] == ["theta"]
    assert [c.name for c in f.cregs] == [c.name for c in qc.cregs] and f.num_clbits == qc.num_clbits
    pairs = lambda c: [(c.find_bit(q).index, c.find_bit(b).index) for i in c.data if i.operation.name == "measure"
                       for q, b in zip(i.qubits, i.clbits)]
    assert pairs(f) == pairs(qc)
    assert f.count_ops()["barrier"] == qc.count_ops()["barrier"] + 4          # the measure_all barrier is still there
    for bad in (0, 2, 4, -1, 1.5, True):
        with pytest.raises(ValueError, match="odd positive"):
            check_lambda(bad)
    with pytest.raises(ValueError, match="odd positive"):
        fold_two_qubit_gates(qc, 2)
    from qiskit.circuit.library import CCXGate
    big = QuantumCircuit(3)
    big.append(CCXGate(), [0, 1, 2])
    with pytest.raises(ValueError, match="not in the ISA"):
        fold_two_qubit_gates(big, 3)


# ---------------------------------------------------------------------------------------- plan: PUBs, counts, barriers
def test_dry_plan_pubs_have_lambda_times_the_2q_gates_and_their_barriers_survive_qpy(small_run, tmp_path):
    from hsf2.hw.snapshot import read_qpy
    from hsf2.hw.zne import run_zne
    small_run = shutil.copytree(small_run, tmp_path / "copy")                           # the module's run folder stays clean
    res = run_zne(small_run, ["6:1,3,5", "4:1,3"], [], "b", cfg_for(tmp_path, mode="dry"), command="test", out_dir=tmp_path / "o")
    assert len(res["jobs"]) == 1 and res["jobs"][0]["manifest"]["state"] == "dry"
    man = res["jobs"][0]["manifest"]
    jd = res["jobs"][0]["job_dir"]
    assert man["kind"] == "zne" and man["step"] == "step4" and man["setting"]["name"] == "b" and not man["setting"]["gate_twirl"]
    assert man["setting"]["dd"] is True and man["job_id"] is None and man["estimate_s"] > 0
    pubs = man["pubs"]
    assert [(p["target"], p["lam"]) for p in pubs] == [("K6", 1), ("K6", 3), ("K6", 5), ("K4", 1), ("K4", 3)]
    assert [p["k"] for p in pubs] == list(range(5)) and {p["shots"] for p in pubs} == {SHOTS}      # equal shots in every PUB
    base = {p["target"]: p["n_2q_base"] for p in pubs}
    assert all(p["n_2q"] == p["lam"] * base[p["target"]] for p in pubs) and all(p["n_2q"] == res["plan"][i]["n_2q"] for i, p in enumerate(pubs))
    assert len({tuple(p["measured_qubits"]) for p in pubs if p["target"] == "K6"}) == 1      # same layout at every lambda
    for p in pubs:
        isa = read_qpy(jd / f"pub_{p['k']}_{p['target']}_lam{p['lam']}.isa.qpy")
        ops = isa.count_ops()
        assert sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx", "rzz")) == p["n_2q"]
        assert ops.get("barrier", 0) >= (p["lam"] - 1) * p["n_2q_base"]                 # survive the QPY round trip
        assert (jd / f"pub_{p['k']}.layout.json").exists()
    assert (jd / "snapshot_pre.json").exists() and not list(jd.glob("*counts*"))
    plan = json.loads((tmp_path / "o" / "zne_plan.json").read_text())
    assert [r["n_2q"] for r in plan["pubs"]] == [p["n_2q"] for p in pubs] and plan["estimate_total_s"] == man["estimate_s"]
    assert not (tmp_path / "ledger.json").exists()                                      # dry mode never books anything


def test_planning_validation_equal_shots_and_per_job_limit(small_run, tmp_path):
    from hsf2.hw.zne import ZnePubSpec, ZneJob, parse_target, run_zne
    small_run = shutil.copytree(small_run, tmp_path / "copy")
    cfg = cfg_for(tmp_path)
    with pytest.raises(ValueError, match="between 1 and 10"):
        run_zne(small_run, ["6"], [], "b", cfg, per_job=11)
    with pytest.raises(ValueError, match="odd positive"):
        run_zne(small_run, ["6:1,2"], [], "b", cfg)
    with pytest.raises(ValueError, match="either --K"):
        run_zne(small_run, ["6"], ["x"], "b", cfg)
    with pytest.raises(ValueError, match="either --K"):
        run_zne(small_run, [], [], "b", cfg)
    with pytest.raises(ValueError, match="no gauges"):
        run_zne(small_run, ["6"], [], "f", cfg)
    with pytest.raises(ValueError, match="positive integer"):
        run_zne(small_run, ["abc"], [], "b", cfg)
    assert parse_target("12:3,1,5", (1, 3)) == ("12", [1, 3, 5]) and parse_target("48", (1, 3)) == ("48", [1, 3])
    with pytest.raises(ValueError, match="distinct"):
        parse_target("12:1,1", (1,))
    res = run_zne(small_run, ["6:1,3"], [], "b", cfg)
    job = res["jobs"][0]["job"]
    assert job.shots == SHOTS
    job.pubs[1].spec.shots = SHOTS + 1                                                   # unequal shots are refused
    with pytest.raises(ValueError, match="unequal shots"):
        job.shots
    assert isinstance(job, ZneJob) and isinstance(job.pubs[0].spec, ZnePubSpec)


def test_component_names_resolve_with_or_without_the_event_prefix():
    from hsf2.hw.zne import resolve_component
    comps = [SimpleNamespace(name=n) for n in ("event000001335_s3_f1_c20", "event000001332_s2_f1_c31", "event000011335_s3_f1_c20")]
    assert resolve_component(comps, "1335_s3_f1_c20").name == "event000001335_s3_f1_c20"
    assert resolve_component(comps, "event000001332_s2_f1_c31").name == "event000001332_s2_f1_c31"
    with pytest.raises(ValueError, match="0 matches"):
        resolve_component(comps, "1331_s1_f1_c8")


# ---------------------------------------------------------------------------------------- mocked submit
def test_one_confirmation_per_job_equal_shots_barriers_in_the_submitted_pubs_and_budget_refusal(small_run, tmp_path, monkeypatch,
                                                                                               fast_polling):
    from acts_qaoa.ibm import QPULedger
    from hsf2.hw.ledger import BudgetError
    from hsf2.hw.run import SubmitAborted
    from hsf2.hw.zne import run_zne
    C = component_tests()
    run = tmp_path / "r"
    shutil.copytree(small_run, run)
    svc, smp = C.mock_ibm(monkeypatch)
    seen = []
    orig = smp.run
    smp.run = lambda pubs, **kw: (seen.append([(p[0], p[2]) for p in pubs]), orig(pubs, **kw))[1]
    cfg = cfg_for(tmp_path, mode="submit", backend="ibm_fake")
    asked = []
    res = run_zne(run, ["6:1,3,5,7", "5:1,3,5,7", "4:1,3"], [], "b", cfg, per_job=4, command="test", confirm=asked.append)
    # 10 PUBs, 4 per job: jobs of 4, 4 and 2 PUBs, one confirmation and one sampler call each, equal shots everywhere
    assert [len(j["manifest"]["pubs"]) for j in res["jobs"]] == [4, 4, 2] and len(asked) == 3 and smp.calls == 3
    assert all(s == SHOTS for sub in seen for _, s in sub)
    assert "submit ONE job" in asked[0] and "ZNE job 1 of 3" in asked[0] and "K6 lambda=1" in asked[0] and "K6 lambda=7" in asked[0]
    assert "K5" not in asked[0] and "K5 lambda=1" in asked[1] and "4 PUBs x 200 shots" in asked[0]
    assert [j["manifest"]["tag"] for j in res["jobs"]] == ["zne_b_1", "zne_b_2", "zne_b_3"]
    for sub, j in zip(seen, res["jobs"]):                                   # the circuits that reach the sampler are the folded ones
        for (isa, _), p in zip(sub, j["manifest"]["pubs"]):
            ops = isa.count_ops()
            assert ops.get("barrier", 0) >= (p["lam"] - 1) * p["n_2q_base"] and ops.get("cz", 0) == p["n_2q"]
        m = j["manifest"]
        assert m["state"] == "done" and m["billed"]["source"] == "account_diff" and m["billed"]["used"] == 5.0
        assert (j["job_dir"] / "snapshot_post.json").exists()
        assert all((j["job_dir"] / f"pub_{p['k']}_counts.json").exists() for p in m["pubs"])
    led = QPULedger(cfg.ledger_path(), budget_s=400.0)
    assert [e["job_id"] for e in led.entries] == ["fakejob0001", "fakejob0002", "fakejob0003"]
    names = sorted(p.name for p in (run / "samples").glob("*zne*.json"))
    assert len(names) == 10 and any(n.endswith("_p1_top6_" + "fakejob0001"[:12] + "_zne7.json") for n in names)
    assert not any(n.endswith("_main.json") for n in names)                 # hw-analyze needs a control: never picks these up
    # a refused budget stops everything before the first confirmation and submission
    calls, n_dirs = smp.calls, len(list((run / "hw").iterdir()))
    asked.clear()
    with pytest.raises(BudgetError, match="exceeds"):
        run_zne(run, ["6:1,3"], [], "b", cfg_for(tmp_path, mode="submit", backend="ibm_fake", budget_s=1.0,
                                                 ledger=str(tmp_path / "l2.json")), confirm=asked.append)
    assert smp.calls == calls and not asked and not (tmp_path / "l2.json").exists()
    assert len(list((run / "hw").iterdir())) == n_dirs
    # a refused confirmation of the second job: the first job is finished, the second is not submitted and says why
    seq = []

    def second_no(msg):
        seq.append(msg)
        if len(seq) == 2:
            raise SubmitAborted("no")
    with pytest.raises(SubmitAborted):
        run_zne(run, ["6:1,3,5"], [], "b", cfg_for(tmp_path, mode="submit", backend="ibm_fake", ledger=str(tmp_path / "l3.json")),
                per_job=2, confirm=second_no)
    assert len(seq) == 2 and smp.calls == calls + 1
    aborted = [json.loads(p.read_text()) for p in (run / "hw").glob("*/manifest.json")]
    aborted = [m for m in aborted if m["state"] == "aborted"]
    assert len(aborted) == 1 and aborted[0]["job_id"] is None and "SubmitAborted" in aborted[0]["aborted"]


def test_no_wait_zne_and_readout_jobs_then_retrieve_rebuilds_both(small_run, tmp_path, monkeypatch, fast_polling):
    """--no-wait: a zne job and the read-out calibration of its qubits go into the queue before either has run (the
    reference job only has to be planned); hw-retrieve rebuilds both later, books IBM's figure and never resubmits."""
    from acts_qaoa.ibm import QPULedger
    from hsf2.hw.components import retrieve_any
    from hsf2.hw.run import run_readout
    from hsf2.hw.zne import run_zne
    C = component_tests()
    run = tmp_path / "r"
    shutil.copytree(small_run, run)
    svc, smp = C.mock_ibm(monkeypatch, charge=4.0)
    cfg = cfg_for(tmp_path, mode="submit", backend="ibm_fake")
    asked = []
    res = run_zne(run, ["6:1,3"], [], "d", cfg, command="test", confirm=asked.append, wait=False)
    j = res["jobs"][0]
    man = json.loads((j["job_dir"] / "manifest.json").read_text())
    assert man["state"] == "submitted" and man["job_id"] == "fakejob0001" and j["counts"] is None
    assert not list(j["job_dir"].glob("pub_*_counts.json")) and not (j["job_dir"] / "snapshot_post.json").exists()
    cal = run_readout(run, j["job_dir"], cfg, command="test", confirm=asked.append, wait=False)
    rman = json.loads((cal["job_dir"] / "manifest.json").read_text())
    assert cal["cal"] is None and rman["state"] == "submitted" and rman["job_id"] == "fakejob0002"
    assert not (cal["job_dir"] / "readout_cal.json").exists() and len(asked) == 2 and smp.calls == 2
    led = lambda: QPULedger(cfg.ledger_path(), budget_s=400.0).entries                      # noqa: E731
    assert [e["quantum_seconds"] for e in led()] == [None, None]                            # only the estimates so far
    # later, from another shell: both jobs are rebuilt
    got = retrieve_any("fakejob0001", cfg, run_dir=run, service=svc)
    m = json.loads((got["job_dir"] / "manifest.json").read_text())
    assert got["job_dir"] == j["job_dir"] and m["state"] == "done" and m["billed"]["used"] == 4.0
    assert all(sum(json.loads((j["job_dir"] / f"pub_{k}_counts.json").read_text())["counts"].values()) == SHOTS for k in (0, 1))
    got = retrieve_any("fakejob0002", cfg, run_dir=run, service=svc)
    rc = json.loads((got["job_dir"] / "readout_cal.json").read_text())
    qubits = ([int(q) for q in man["layout"]["measured_qubits"]] if "layout" in man else
              sorted({int(q) for p in man["pubs"] for q in p["measured_qubits"]}))
    assert got["job_dir"] == cal["job_dir"] and rc["qubits"] == qubits and rc["measure_twirl"] is True
    assert rc["job_id"] == "fakejob0002" and rc["shots"] == SHOTS and rc["session"] == rman["session"]
    for k in (0, 1):
        assert sum(json.loads((got["job_dir"] / f"pub_{k}_counts.json").read_text())["counts"].values()) == SHOTS
    assert json.loads((got["job_dir"] / "manifest.json").read_text())["state"] == "done"
    assert smp.calls == 2 and [e["quantum_seconds"] for e in led()] == [4.0, 4.0]
    with pytest.raises(ValueError, match="not a read-out calibration job"):
        from hsf2.hw.run import retrieve_readout_job
        retrieve_readout_job("fakejob0001", cfg, run_dir=run, service=svc)


def test_readout_local_job_is_rebuilt_by_retrieve(small_run, tmp_path):
    """hw-retrieve of a local read-out calibration writes the same readout_cal.json as the run itself."""
    from hsf2.hw.components import retrieve_any
    from hsf2.hw.run import run_readout
    from hsf2.hw.zne import run_zne
    run = tmp_path / "r"
    shutil.copytree(small_run, run)
    cfg = cfg_for(tmp_path, mode="local")
    ref = run_zne(run, ["6:1"], [], "b", cfg, command="test", sim_method="matrix_product_state", seed=3)["jobs"][0]
    res = run_readout(run, ref["job_dir"], cfg, command="test")
    first = json.loads((res["job_dir"] / "readout_cal.json").read_text())
    (res["job_dir"] / "readout_cal.json").unlink()
    got = retrieve_any(res["manifest"]["job_id"], cfg, run_dir=run)
    assert got["job_dir"] == res["job_dir"] and json.loads((res["job_dir"] / "readout_cal.json").read_text()) == first


# ---------------------------------------------------------------------------------------- extrapolators
def test_extrapolators_recover_known_lambda_dependences():
    from hsf2.hw.zne import extrapolate, richardson_weights
    lam = np.array([1.0, 3.0, 5.0])
    assert richardson_weights(lam).sum() == pytest.approx(1.0) and richardson_weights([1, 3]) == pytest.approx([1.5, -0.5])
    floor = np.array([0.0, 0.3, 0.1])

    def Y(f):
        return np.stack([np.array([[f[j](l) for j in range(3)]]) for l in lam])     # (L, 1, J)
    # observable 0: a line; 1: a quadratic; 2: A exp(-k lambda) + B with B = 0.1
    lin, quad, expo = (lambda l: 0.9 - 0.12 * l), (lambda l: 1.0 - 0.3 * l + 0.02 * l * l), (lambda l: 0.1 + 0.8 * np.exp(-0.4 * l))
    ex = extrapolate(lam, Y([lin, quad, expo]), floor)
    assert ex["linear"][0, 0] == pytest.approx(0.9) and ex["richardson"][0, 0] == pytest.approx(0.9)
    assert ex["richardson"][0, 1] == pytest.approx(1.0) and abs(ex["linear"][0, 1] - 1.0) > 0.01          # exact only for quadratics
    assert ex["exponential"][0, 2] == pytest.approx(0.9) and ex["exp_A"][0, 2] == pytest.approx(0.8) and ex["exp_k"][0, 2] == pytest.approx(0.4)
    assert abs(ex["linear"][0, 2] - 0.9) > 0.01 and abs(ex["richardson"][0, 2] - 0.9) > 0.001
    # a decaying negative-going observable below its floor keeps the sign
    neg = lambda l: -0.2 + (-0.5) * np.exp(-0.3 * l)
    assert extrapolate(lam, Y([neg, neg, neg]), np.array([-0.2] * 3))["exponential"][0, 0] == pytest.approx(-0.7)
    # two points: line = Richardson, exponential through both; a sign change makes the exponential undefined (NaN), the others not
    ex2 = extrapolate([1, 3], Y([lin, quad, expo])[:2], floor)
    assert ex2["linear"][0, 0] == pytest.approx(ex2["richardson"][0, 0]) == pytest.approx(0.9)
    cross = np.array([[[0.5]], [[-0.1]], [[-0.3]]])
    exc = extrapolate(lam, cross, np.array([0.0]))
    assert np.isnan(exc["exponential"][0, 0]) and np.isfinite(exc["linear"][0, 0]) and np.isfinite(exc["richardson"][0, 0])
    with pytest.raises(ValueError, match="at least|shape"):
        extrapolate([1.0], Y([lin, quad, expo])[:1], floor)
    with pytest.raises(ValueError, match="shape"):
        extrapolate(lam, Y([lin, quad, expo])[:2], floor)


def synthetic_target(n=8, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=1 << n)
    p = np.exp(-2.5 * (E - E.min()))
    p /= p.sum()
    return E, p


def draw_counts(q, shots, rng):
    c = rng.multinomial(shots, q)
    idx = np.nonzero(c)[0]
    return {int(i): int(c[i]) for i in idx}


def test_observables_are_the_hw_analysis_estimators_as_shot_means():
    from hsf2.hw.analysis import correlation_fidelity, counts_arrays, f_energy, f_xeb, tail
    from hsf2.hw.zne import build_observables
    E, p = synthetic_target()
    rank = np.empty(E.size, np.int32)
    rank[np.argsort(E, kind="stable")] = np.arange(E.size, dtype=np.int32)
    masks = {"P_best_10pct": rank < int(round(0.1 * E.size))}
    counts = draw_counts(0.6 * p + 0.4 / p.size, 5000, np.random.default_rng(1))
    idx, c = counts_arrays({str(k): v for k, v in counts.items()})
    o = np.argsort(idx)
    idx, c = idx[o], c[o]
    obs = build_observables(E, p, idx, masks)
    assert obs.names == ["mean_energy", "F_E", "F_XEB", "P_best_10pct", "s1", "s2", "s3"]
    m = c @ obs.values(idx) / c.sum()
    E_unif, e_ideal = float(E.mean()), float(p @ E)
    assert m[0] == pytest.approx(float(c @ E[idx]) / c.sum()) and m[1] == pytest.approx(f_energy(E, idx, c, E_unif, e_ideal))
    assert m[2] == pytest.approx(f_xeb(p, idx, c)) and m[3] == pytest.approx(tail(rank, p, idx, c, 0.1)["p_hw"])
    assert list(m[4:]) == pytest.approx([s for s, _ in correlation_fidelity(p, [(idx, c)], 3)[0]])
    assert list(obs.ideal[:3]) == pytest.approx([e_ideal, 1.0, 1.0]) and obs.ideal[3] == pytest.approx(float(p[masks["P_best_10pct"]].sum()))
    assert list(obs.floor[:3]) == pytest.approx([E_unif, 0.0, 0.0]) and obs.floor[3] == pytest.approx(masks["P_best_10pct"].mean())


def test_bootstrap_analysis_recovers_the_zero_noise_value_with_the_exponential_model_and_reports_brackets():
    from hsf2.hw.zne import analyze_counts, build_observables
    E, p = synthetic_target(seed=2)
    N = p.size
    ground = E <= E.min() + 1e-9
    rank = np.empty(N, np.int32)
    rank[np.argsort(E, kind="stable")] = np.arange(N, dtype=np.int32)
    masks = {"P_ground": ground, "P_best_10pct": rank < int(round(0.1 * N))}
    f, shots = 0.62, 120_000
    rng = np.random.default_rng(4)
    lams = (1, 3, 5)
    counts = {lam: draw_counts(f ** lam * p + (1 - f ** lam) / N, shots, rng) for lam in lams}     # white noise: F(lam) = f^lam
    idx_all = np.unique(np.concatenate([np.fromiter(c, np.int64) for c in counts.values()]))
    obs = build_observables(E, p, idx_all, masks)
    res = analyze_counts(obs, counts, boot_n=300, seed=1, tag="synth")
    again = analyze_counts(obs, counts, boot_n=300, seed=1, tag="synth")
    assert res == again                                                                   # seeded: reproducible
    assert analyze_counts(obs, counts, boot_n=300, seed=2, tag="synth") != res
    assert res["lambdas"] == [1, 3, 5] and res["shots"] == {1: shots, 3: shots, 5: shots}
    o = res["observables"]
    for name in ("F_E", "F_XEB", "s1", "s2", "s3", "mean_energy", "P_ground", "P_best_10pct"):
        e = o[name]["models"]["exponential"]
        assert e["err"] > 0 and abs(e["value"] - o[name]["ideal"]) < 4 * e["err"], name       # the exact model recovers lambda = 0
        assert e["brackets_ideal"] is True or abs(e["z"]) < 4
    for name in ("F_E", "F_XEB", "s1"):
        assert o[name]["models"]["exponential"]["k"] == pytest.approx(-np.log(f), abs=0.03)
        assert o[name]["models"]["exponential"]["A"] == pytest.approx(1.0, abs=0.08)
        lin = o[name]["models"]["linear"]
        assert lin["brackets_ideal"] is False and lin["value"] < o[name]["ideal"] - 0.1      # a line through f^lambda undershoots
        pts = o[name]["points"]
        assert pts["1"]["value"] == pytest.approx(f, abs=0.05) and pts["3"]["value"] == pytest.approx(f ** 3, abs=0.05)
        assert pts["1"]["lo68"] <= pts["1"]["value"] <= pts["1"]["hi68"] and pts["1"]["err"] > 0
    assert o["F_E"]["floor"] == 0.0 and o["mean_energy"]["floor"] == pytest.approx(float(E.mean()))
    assert o["P_ground"]["floor"] == pytest.approx(ground.sum() / N)
    with pytest.raises(ValueError, match="at least two"):
        analyze_counts(obs, {1: counts[1]})


# ---------------------------------------------------------------------------------------- local end to end
def test_local_qx2_job_files_samples_retrieve_and_analysis(small_run, tmp_path):
    from hsf2.hw.analysis import main_tags
    from hsf2.hw.components import retrieve_any
    from hsf2.hw.zne import analyze_zne, run_zne
    run = tmp_path / "r"
    shutil.copytree(small_run, run)
    cfg = cfg_for(tmp_path, mode="local")
    res = run_zne(run, ["6:1,3"], [], "b", cfg, command="test", sim_method="matrix_product_state", seed=3)
    jd, man = res["jobs"][0]["job_dir"], res["jobs"][0]["manifest"]
    assert man["state"] == "done" and man["billed"]["source"] == "local" and man["job_id"].startswith("local_")
    assert not (tmp_path / "ledger.json").exists()
    for k, lam in enumerate((1, 3)):
        cf = json.loads((jd / f"pub_{k}_counts.json").read_text())
        assert cf["lam"] == lam and cf["target"] == "K6" and sum(cf["counts"].values()) == SHOTS
        assert (jd / f"pub_{k}_K6_lam{lam}.isa.qpy").exists()
    samples = sorted((run / "samples").glob("*zne*.json"))
    assert [s.name.rsplit("_", 1)[1] for s in samples] == ["zne1.json", "zne3.json"] and man["samples"] == [s.name for s in samples]
    meta = json.loads(samples[1].read_text())["meta"]
    assert meta["zne_lambda"] == 3 and meta["sparsify_top_k"] == 6 and meta["job_id"] == man["job_id"] and meta["kind"] == "local"
    assert main_tags(run / "samples") == []                                              # hw-analyze keeps working next to them
    # hw-retrieve rebuilds the counts of a local zne job
    (jd / "pub_0_counts.json").unlink()
    got = retrieve_any(man["job_id"], cfg, run_dir=run)
    assert got["job_dir"] == jd and (jd / "pub_0_counts.json").exists() and (jd / "snapshot_post.json").exists()
    # analysis
    out = analyze_zne([jd], tmp_path / "an", boot_n=100)
    t = out["targets"]["K6"]
    assert t["kind"] == "qx2" and t["lambdas"] == [1, 3] and t["n_2q"][3] == 3 * t["n_2q"][1]
    assert list(t["observables"]) == ["mean_energy", "F_E", "F_XEB", "P_best_1pct", "P_best_10pct", "s1", "s2", "s3"]
    assert t["observables"]["F_E"]["ideal"] == 1.0 and set(t["observables"]["F_E"]["models"]) == {"linear", "richardson", "exponential"}
    ex = pd.read_csv(tmp_path / "an" / "zne_extrapolation.csv")
    pts = pd.read_csv(tmp_path / "an" / "zne_points.csv")
    assert len(ex) == 8 * 3 and len(pts) == 8 * 2 and set(ex["model"]) == {"linear", "richardson", "exponential"}
    lin = ex[(ex["observable"] == "F_E") & (ex["model"] == "linear")].iloc[0]
    rich = ex[(ex["observable"] == "F_E") & (ex["model"] == "richardson")].iloc[0]
    assert lin["value"] == pytest.approx(rich["value"])                                  # two lambdas: the same line


def test_local_component_job_and_analysis(tmp_path):
    from hsf2.hw.zne import analyze_zne, run_zne
    C = component_tests()
    run = C.write_run(tmp_path / "tri", C.HAND)
    cfg = cfg_for(tmp_path, mode="local", fake_backend=C.small_fake_name())
    res = run_zne(run, [], [f"{C.EVENT}_s0_f1_c1:1,3"], "b", cfg, command="test", sim_method="statevector", seed=2)
    jd, man = res["jobs"][0]["job_dir"], res["jobs"][0]["manifest"]
    assert man["kind"] == "zne" and [(p["target"], p["lam"]) for p in man["pubs"]] == [(f"{C.EVENT}_s0_f1_c1", 1), (f"{C.EVENT}_s0_f1_c1", 3)]
    assert man["pubs"][0]["component_sha1"] and man["pubs"][1]["n_2q"] == 3 * man["pubs"][1]["n_2q_base"]
    assert not (run / "samples").exists()                                                # components export no Phase-1 samples
    out = analyze_zne([jd], tmp_path / "an", boot_n=100)
    t = out["targets"][f"{C.EVENT}_s0_f1_c1"]
    assert t["kind"] == "component" and list(t["observables"]) == ["mean_energy", "F_E", "F_XEB", "P_ground", "s1", "s2", "s3"]
    pg = t["observables"]["P_ground"]
    assert pg["ideal"] > pg["floor"] > 0 and pg["points"]["1"]["value"] >= 0
    # a changed component file is refused
    f = next((run / "components").glob("*_c1.json"))
    doc = json.loads(f.read_text())
    doc["meta"]["alpha"] = 0.5
    f.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="changed"):
        analyze_zne([jd], tmp_path / "an2", boot_n=10)


def test_cli_hw_zne_dry_local_analyze_in_deployed_layout(small_run, tmp_path, monkeypatch):
    from hsf2.cli import main
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))
    monkeypatch.delenv("HSF2_IBM_CREDS", raising=False)
    run = tmp_path / "runs" / "r"
    shutil.copytree(small_run, run)
    outside = snapshot_tree(tmp_path, run)
    secret = tmp_path / "private_dir" / "apikey_zne.json"                                # never opened; only its name is recorded
    with pytest.raises(SystemExit, match="either --K"):
        main(["hw-zne", str(run), "--K", "6", "--components", "x"])
    with pytest.raises(SystemExit, match="either --K"):
        main(["hw-zne", str(run)])
    with pytest.raises(ValueError, match="odd positive"):
        main(["hw-zne", str(run), "--K", "6", "--lambdas", "1", "2"])
    assert main(["hw-zne", str(run), "--K", "6:1,3,5", "4", "--lambdas", "1", "3", "--seeds", "1", "--ledger", str(tmp_path / "led.json"),
                 "--shots", str(SHOTS), "--creds", str(secret)]) == 0
    jobs = sorted((run / "hw").glob("*_zne_b_*"))
    assert len(jobs) == 1 and not (tmp_path / "led.json").exists()
    man = json.loads((jobs[0] / "manifest.json").read_text())
    assert man["state"] == "dry" and [(p["target"], p["lam"]) for p in man["pubs"]] == [("K6", 1), ("K6", 3), ("K6", 5), ("K4", 1), ("K4", 3)]
    assert man["creds"] == "apikey_zne.json" and "private_dir" not in (jobs[0] / "manifest.json").read_text()
    cmd = next(p for p in (run / "outputs").glob("*_hw_zne*") if (p / "zne_plan.json").exists())    # the refused call has a folder too
    assert (cmd / "commands.jsonl").exists() and (cmd / "config.json").exists() and (cmd / "zne_plan.json").exists()
    assert main(["hw-zne", str(run), "--K", "6:1,3", "--mode", "local", "--seeds", "1", "--shots", str(SHOTS),
                 "--sim-method", "matrix_product_state"]) == 0
    done = [p for p in (run / "hw").glob("*_zne_b_*") if (p / "pub_0_counts.json").exists()]
    assert len(done) == 1
    assert main(["hw-zne-analyze", str(done[0]), "--boot-n", "50"]) == 0
    an = [p for p in (run / "outputs").glob("*_hw_zne_analyze*") if (p / "zne_analysis.json").exists()]
    assert len(an) == 1 and (an[0] / "commands.jsonl").exists() and (an[0] / "zne_extrapolation.csv").exists()
    assert not any("private_dir" in p.read_text() for p in (run / "outputs").rglob("*.json*"))
    with pytest.raises(SystemExit, match="job folders"):
        main(["hw-zne-analyze", str(run)])
    assert snapshot_tree(tmp_path, run) == outside                                       # nothing was written outside the run folder


# ---------------------------------------------------------------------------- read-out correction before extrapolation (lead)
def test_readout_corrected_observables_recover_the_noiseless_values():
    """Counts of a known distribution passed through asymmetric read-out noise: with readout_mats every observable at every
    lambda returns to its noiseless value (F_E = F_XEB = s_k = 1, P(ground) = ideal) within shot noise, while the raw ones
    stay damped; an identity calibration changes nothing."""
    from hsf2.hw.zne import analyze_counts, build_observables
    n, S = 6, 2_000_000
    rng = np.random.default_rng(7)
    N = 1 << n
    p = rng.dirichlet(np.ones(N) * 0.4)
    E = -np.log(p)                                  # low energy where the ideal state is large: F_E well defined
    mats = []
    for _ in range(n):
        a, b = rng.uniform(0.01, 0.05), rng.uniform(0.02, 0.10)
        mats.append(np.array([[1 - a, b], [a, 1 - b]]))
    t = p.reshape([2] * n)
    for k, m in enumerate(mats):
        t = np.moveaxis(np.tensordot(m, t, axes=([1], [n - 1 - k])), 0, n - 1 - k)
    p_obs = t.reshape(-1)
    cnt = rng.multinomial(S, p_obs / p_obs.sum())
    counts = {lam: {i: int(c) for i, c in enumerate(cnt) if c} for lam in (1, 3)}
    idx_all = np.arange(N)
    masks = {"P_ground": E <= E.min() + 1e-12}
    raw = analyze_counts(build_observables(E, p, idx_all, masks, kmax=3), counts, boot_n=20, seed=0)["observables"]
    mit = analyze_counts(build_observables(E, p, idx_all, masks, kmax=3, readout_mats=mats), counts, boot_n=20,
                         seed=0)["observables"]
    for name in ("F_E", "F_XEB", "s1", "s2", "s3"):
        pt = mit[name]["points"]["1"]
        assert abs(pt["value"] - 1.0) < 4 * pt["err"] + 0.005, (name, pt)
        assert raw[name]["points"]["1"]["value"] < 0.99, (name, raw[name]["points"]["1"])
    assert abs(mit["P_ground"]["points"]["1"]["value"] - mit["P_ground"]["ideal"]) < 0.002
    ident = [np.eye(2)] * n
    same = analyze_counts(build_observables(E, p, idx_all, masks, kmax=3, readout_mats=ident), counts, boot_n=20,
                          seed=0)["observables"]
    for name in raw:
        assert abs(same[name]["points"]["1"]["value"] - raw[name]["points"]["1"]["value"]) < 1e-12, name

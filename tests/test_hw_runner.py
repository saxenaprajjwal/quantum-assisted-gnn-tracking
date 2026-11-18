"""Hardware runner (hsf2.hw): settings, gauges, fractional ISA, QPY, layout, snapshot, ledger, readout mitigation,
credentials, sparsification, retrieve.  0 QPU s: everything runs on a fake backend with Aer or on mocks; IBM is never
contacted.  A 6-qubit Phase-1 problem (3 knobs x 2 bits) is built once per module."""
import builtins
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


SHOTS = 400


# ---------------------------------------------------------------------------- fixtures
def make_small_run(path: Path) -> Path:
    """Phase-1 run folder of a 6-qubit tuning problem (quadratic surrogate -> QUBO -> Ising, 15 couplings)."""
    from acts_qaoa import pipeline as P1
    from acts_qaoa.config import ParamSpace
    from acts_qaoa.encoding import Encoding
    from hsf2.config import TRACKING_DEFAULTS
    from hsf2.tuning.phase1 import make_run
    from hsf2.tuning.runner import LookupRunner
    names = ["dphi_scale", "deta_scale", "z0_scale"]
    space = ParamSpace.from_ranges({"dphi_scale": (0.3, 1.0), "deta_scale": (0.05, 1.0), "z0_scale": (0.05, 1.0)}, names=names)
    enc = Encoding(space, 2)
    tgt = np.array([1, 2, 0]) / 3.0

    def metrics(idx):
        u = enc.unit_from_index(idx)
        eff = 1.0 - float(((u - tgt) ** 2).sum()) + 0.2 * float(u[0] * u[1])
        return {"efficiency": eff, "efficiency_dm": eff, "fakerate": 0.0, "duplicaterate": 0.0, "time_proxy": 0.0}

    run = make_run(path, space, 2, LookupRunner(enc, metrics, n_events=1), 1, {k: TRACKING_DEFAULTS[k] for k in names})
    P1.stage_warmstart(run, n_lhs=40, lhs_seed=3, noise_repeats=0)
    P1.stage_build(run, surrogate="quadratic", fit_tags=("ws",), ansatz=("standard",), p_max=1)
    return Path(path)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return make_small_run(tmp_path_factory.mktemp("hwrun") / "r")


def cfg_local(tmp_path, **kw):
    from hsf2.hw.config import HwConfig
    d = dict(mode="local", shots=SHOTS, transpile_seeds=1, ledger=str(tmp_path / "ledger.json"), budget_s=400.0)
    d.update(kw)
    return HwConfig(**d)


@pytest.fixture(scope="module")
def local_job(run_dir, tmp_path_factory):
    """One local job of setting d (K = 6 of 15 couplings), used by several tests."""
    from hsf2.hw.run import run_job
    return run_job(run_dir, "d", 6, cfg_local(tmp_path_factory.mktemp("cfg")), command="test")


# ---------------------------------------------------------------------------- 1. settings
def test_settings_option_dicts_and_independent_switches():
    from hsf2.hw.settings import SETTINGS, get_setting
    base = {"twirling.num_randomizations": 16, "twirling.shots_per_randomization": "auto",
            "dynamical_decoupling.sequence_type": "XY4"}

    def opts(gate, meas, dd):
        return dict(base, **{"twirling.enable_gates": gate, "twirling.enable_measure": meas,
                             "dynamical_decoupling.enable": dd})

    assert get_setting("a").options() == opts(False, False, False)
    assert get_setting("b").options() == opts(False, False, True)
    assert get_setting("c").options() == opts(True, True, False)
    assert get_setting("d").options() == opts(True, True, True)
    assert get_setting("e").options() == opts(False, True, True) and get_setting("e").fractional
    assert get_setting("f").options() == opts(True, True, True) and get_setting("f").gauged
    assert sorted(SETTINGS) == list("abcdef")
    # gate and measurement twirling are separate switches (Phase 1's --no-twirl switches both together)
    assert {(s.gate_twirl, s.measure_twirl) for s in SETTINGS.values()} >= {(True, True), (False, True), (False, False)}
    with pytest.raises(ValueError):
        get_setting("g")


def test_options_are_applied_strictly_to_the_runtime_options():
    from hsf2.hw.settings import apply_options, get_setting, options_dump
    from qiskit_ibm_runtime import SamplerV2
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from qiskit_ibm_runtime.fake_provider import FakeFez
        s = SamplerV2(mode=FakeFez())
    setting = get_setting("e")
    applied = apply_options(s.options, dict(setting.options(), default_shots=123))
    assert applied["twirling.enable_gates"] is False and applied["twirling.enable_measure"] is True
    d = options_dump(s.options)
    assert d["twirling"]["enable_gates"] is False and d["twirling"]["enable_measure"] is True
    assert d["twirling"]["num_randomizations"] == 16 and d["dynamical_decoupling"]["sequence_type"] == "XY4"
    assert d["default_shots"] == 123
    with pytest.raises(RuntimeError, match="cannot be set"):
        apply_options(s.options, {"twirling.no_such_option": 1})
    assert apply_options(s.options, {"twirling.no_such_option": 1}, strict=False) == {}


def test_executor_primitive_is_used_when_available_and_refused_clearly_otherwise():
    from hsf2.hw.run import make_sampler
    from hsf2.hw.settings import get_setting
    from qiskit_ibm_runtime.fake_provider import FakeFez
    try:
        import qiskit_ibm_runtime.executor_sampler  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="executor Sampler"):
            make_sampler(FakeFez(), "executor", get_setting("d"), 100, None)
        return
    sampler, applied = make_sampler(FakeFez(), "executor", get_setting("c"), 100, 40)
    assert applied["twirling.enable_measure"] is True and applied["dynamical_decoupling.enable"] is False
    assert applied["max_execution_time"] == 40


# ---------------------------------------------------------------------------- 2. equal shots
def test_equal_shots_enforced():
    from hsf2.hw.build import PubSpec, check_equal_shots, make_pub_specs
    ok = make_pub_specs(False, False, 1, 4000, [0.1], [0.2])
    assert [p.shots for p in ok] == [4000, 4000] and [p.kind for p in ok] == ["main", "control"]
    assert check_equal_shots(ok) == 4000
    bad = [PubSpec(0, "main", 0, [0.1], [0.2], 4000), PubSpec(1, "control", 0, [0.0], [0.0], 1000)]
    with pytest.raises(ValueError, match="unequal shots"):
        check_equal_shots(bad)
    with pytest.raises(ValueError, match="unequal shots"):
        check_equal_shots([(None, None, 100), (None, None, 200)])
    g = make_pub_specs(False, True, 8, 4000, [0.1], [0.2])
    assert len(g) == 16 and {p.shots for p in g} == {500}
    with pytest.raises(ValueError, match="do not divide"):
        make_pub_specs(False, True, 8, 4001, [0.1], [0.2])


def test_execute_job_refuses_unequal_pubs(run_dir, local_job, tmp_path):
    from hsf2.hw.run import execute_job
    from hsf2.hw.settings import get_setting
    import copy
    plan = local_job["plan"]
    pubs = copy.copy(plan.pubs)
    pubs[1] = copy.copy(pubs[1])
    pubs[1].spec = copy.copy(pubs[1].spec)
    pubs[1].spec.shots = SHOTS // 2
    with pytest.raises(ValueError, match="unequal shots"):
        execute_job(plan.backend, pubs, cfg_local(tmp_path), get_setting("d"), "x", 1.0, tmp_path, {"tag": "t"}, str(run_dir))


# ---------------------------------------------------------------------------- 3. gauges
def test_gauge_round_trip_and_identity(run_dir):
    from acts_qaoa.circuits import bind, build_qaoa_circuit
    from qiskit.quantum_info import Statevector
    from hsf2.hw.build import circuit_angles, load_problem, sparse_ising
    from hsf2.hw.gauge import gauge_ising, gauge_mask, gauge_vectors, xor_counts
    pb = load_problem(run_dir)
    ism = sparse_ising(pb.ising, 6)
    gam, bet, _ = circuit_angles(pb, ism, 6)
    gv = gauge_vectors(pb.n, 4, seed=3)
    assert not gv[0].any() and len({g.tobytes() for g in gv}) == 4
    ident = gauge_ising(ism, gv[0])
    assert np.array_equal(ident.h, ism.h) and np.array_equal(ident.J, ism.J) and gauge_mask(gv[0]) == 0

    def probs(m):
        h_t, j_t = m.terms(pb.scale)
        qc, g, b = build_qaoa_circuit(m.n, h_t, j_t, p=1, measure=False)
        return Statevector(bind(qc, g, b, gam, bet)).probabilities()

    p0 = probs(ism)
    E = pb.run.landscape(pb.qubo).E
    for g in gv[1:]:
        mask = gauge_mask(g)
        pg = probs(gauge_ising(ism, g))
        assert not np.allclose(pg, p0)                       # the gauged circuit really is a different circuit
        # measured x' in the gauge frame -> x = x' XOR g: the original distribution and its energies
        back = pg[np.arange(pg.size) ^ mask]                 # p_original[x] = p_gauged[x XOR g]
        assert np.allclose(back, p0, atol=1e-12)
        assert abs(float(pg @ E[np.arange(pg.size) ^ mask]) - float(p0 @ E)) < 1e-12
        assert xor_counts({3: 5, 0: 2}, mask) == {3 ^ mask: 5, 0 ^ mask: 2}


# ---------------------------------------------------------------------------- 4. fractional setting (e)
def test_fractional_isa_has_rzz_in_range_and_twirling_off(run_dir, tmp_path):
    from hsf2.hw.run import run_job
    from hsf2.hw.settings import get_setting
    from hsf2.hw.snapshot import read_qpy
    from hsf2.hw.transpile import check_fractional_isa, rzz_angles
    try:
        from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle  # noqa: F401
    except ImportError:
        pytest.skip("FoldRzzAngle not available in this qiskit-ibm-runtime")
    s = get_setting("e")
    assert s.options()["twirling.enable_gates"] is False and s.fractional
    res = run_job(run_dir, "e", 6, cfg_local(tmp_path), command="test")
    for p in res["plan"].pubs:
        ang = check_fractional_isa(p.isa)
        assert ang and all(0.0 < a <= math.pi / 2 + 1e-9 for a in ang)
        assert p.values is None                               # angles are bound in a fractional ISA
    main_isa = read_qpy(res["job_dir"] / "pub_0_main.isa.qpy")
    assert rzz_angles(main_isa) and "rzz" in main_isa.count_ops()
    # control keeps the gate content (same number of rzz) with beta = pi/2 so the transpiler cannot delete them
    ctrl = res["plan"].pubs[1]
    assert ctrl.spec.betas == [math.pi / 2] and len(rzz_angles(ctrl.isa)) == len(rzz_angles(res["plan"].pubs[0].isa))
    assert sum(json.loads((res["job_dir"] / "pub_0_counts.json").read_text())["counts"].values()) == SHOTS
    assert res["manifest"]["setting"]["options"]["twirling.enable_gates"] is False


def test_fractional_angle_outside_range_fails_loudly():
    from qiskit import QuantumCircuit
    from hsf2.hw.transpile import check_fractional_isa
    qc = QuantumCircuit(2)
    qc.rzz(2.0, 0, 1)
    with pytest.raises(ValueError, match="outside"):
        check_fractional_isa(qc)
    with pytest.raises(ValueError, match="no rzz"):
        check_fractional_isa(QuantumCircuit(2))
    from hsf2.hw.transpile import require_fractional_target
    from qiskit_ibm_runtime.fake_provider import FakeFez
    with pytest.raises(RuntimeError, match="no native rzz"):
        require_fractional_target(FakeFez())


# ---------------------------------------------------------------------------- 5. QPY
def test_qpy_round_trip_version_15(local_job, tmp_path):
    from qiskit import qpy
    from hsf2.hw.snapshot import QPY_VERSION, read_qpy, write_qpy
    isa = local_job["plan"].pubs[0].isa
    path = write_qpy(tmp_path / "x.isa.qpy", isa)
    assert QPY_VERSION == 15 and path.read_bytes()[:6] == b"QISKIT" and path.read_bytes()[6] == 15
    back = read_qpy(path)
    assert back == isa and list(back.parameters) == list(isa.parameters)
    with open(path, "rb") as f:                               # the file written into the job folder reads back too
        assert qpy.load(f)[0].count_ops() == isa.count_ops()
    assert read_qpy(local_job["job_dir"] / "pub_0_main.isa.qpy") == isa


# ---------------------------------------------------------------------------- 6. layout
def test_layout_json_consistent_with_isa_layout_and_stats(local_job):
    from acts_qaoa.circuits import circuit_stats
    plan = local_job["plan"]
    for p in plan.pubs:
        lay = json.loads((local_job["job_dir"] / f"pub_{p.spec.k}.layout.json").read_text())
        isa = p.isa
        assert lay["initial"] == [int(q) for q in isa.layout.initial_index_layout(filter_ancillas=True)]
        assert lay["final"] == [int(q) for q in isa.layout.final_index_layout()]
        # classical bit k is measured on the physical qubit that logical qubit k ends on
        assert lay["measured"]["clbits"] == list(range(plan.problem.n)) and lay["measured"]["qubits"] == lay["final"]
        st = circuit_stats(isa, plan.backend)
        assert lay["active_qubits"] == st["active_qubits"] and set(lay["measured"]["qubits"]) <= set(st["active_qubits"])
        assert lay["stats"]["n_2q"] == st["n_2q"] >= len(lay["edges"]) > 0
        cmap = {tuple(sorted(e)) for e in plan.backend.coupling_map.get_edges()}
        assert all(tuple(e) in cmap for e in lay["edges"])
    assert plan.pubs[0].layout["measured"] == plan.pubs[1].layout["measured"]      # main and control: same qubits


# ---------------------------------------------------------------------------- 7. snapshots
def test_snapshots_contain_every_used_qubit_edge_and_the_date(local_job):
    import datetime as dt
    plan = local_job["plan"]
    qs = sorted({q for p in plan.pubs for q in p.layout["active_qubits"]})
    es = sorted({tuple(e) for p in plan.pubs for e in p.layout["edges"]})
    for name in ("snapshot_pre.json", "snapshot_post.json"):
        snap = json.loads((local_job["job_dir"] / name).read_text())
        dt.datetime.fromisoformat(snap["date"])
        assert sorted(int(q) for q in snap["qubits"]) == qs
        assert sorted(tuple(int(x) for x in k.split("-")) for k in snap["edges"]) == es
        assert snap["calibration_id"]
        for q in snap["qubits"].values():
            assert set(q) >= {"t1", "t2", "readout_error", "prob_meas0_prep1", "prob_meas1_prep0", "measure", "sx"}
            assert q["t1"] is not None and q["measure"]["error"] is not None
        assert all("cz" in e and e["cz"]["error"] is not None for e in snap["edges"].values())


# ---------------------------------------------------------------------------- 8. ledger
def test_resolve_billed_prefers_job_then_account_difference_then_estimate():
    from hsf2.hw.run import resolve_billed
    assert resolve_billed(7.5, 100, 110, 20)["source"] == "job" and resolve_billed(7.5, 100, 110, 20)["used"] == 7.5
    b = resolve_billed(0, 100.0, 108.0, 20.0)                 # IBM's per-job figure is 0 while 'pending'
    assert (b["source"], b["used"], b["account_diff_s"]) == ("account_diff", 8.0, 8.0)
    assert resolve_billed(None, 100.0, 103.0, 20.0)["source"] == "account_diff"
    b = resolve_billed(0, 100.0, 100.0, 20.0)
    assert (b["source"], b["used"]) == ("estimate", 20.0)
    assert resolve_billed(None, None, None, 5.0)["source"] == "estimate"


class FakeJob:
    """A runtime job whose usage is 'pending' for `pending_polls` calls of metrics()."""

    def __init__(self, shots, n_bits, pending_polls=0, charge=6.5, fail=False):
        self.shots, self.n_bits, self.pending_polls, self.charge, self.fail = shots, n_bits, pending_polls, charge, fail
        self.polls = 0

    def job_id(self):
        return "fakejob0001"

    def result(self):
        if self.fail:
            raise RuntimeError("job failed")
        rng = np.random.default_rng(1)

        def one(shots):
            c = rng.multinomial(shots, np.ones(1 << self.n_bits) / (1 << self.n_bits))
            counts = {format(i, f"0{self.n_bits}b"): int(v) for i, v in enumerate(c) if v}
            return SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda: counts)))
        return [one(s) for s in self.shots]

    def metrics(self):
        self.polls += 1
        if self.polls <= self.pending_polls:
            return {"usage": {"status": "pending"}}
        return {"usage": {"status": "finished", "qpu_charge_time_seconds": self.charge}}

    def properties(self):
        return None


class FakeService:
    def __init__(self, consumed=100.0, limit=600.0, step=5.0):
        self.consumed, self.limit, self.step, self.ran = consumed, limit, step, False

    def usage(self):
        return {"usage_consumed_seconds": self.consumed + (self.step if self.ran else 0.0),
                "usage_limit_seconds": self.limit}


@pytest.fixture
def fast_polling(monkeypatch):
    from hsf2.hw import run as R
    for k, v in (("USAGE_TIMEOUT_S", 0.05), ("USAGE_INTERVAL_S", 0.005), ("ACCOUNT_TIMEOUT_S", 0.05),
                 ("ACCOUNT_INTERVAL_S", 0.005)):
        monkeypatch.setattr(R, k, v)


def test_usage_polling_and_booking_sources(fast_polling, tmp_path):
    from hsf2.hw.ledger import HwLedger
    from hsf2.hw.run import collect_job, poll_usage
    # pending, then final.  An explicit, generous timeout: three 5 ms sleeps can take 60 ms on Windows (timer
    # granularity 15.6 ms), which raced the 50 ms of fast_polling under load (seen in the full run of 16 Nov 2026)
    assert poll_usage(FakeJob([1], 1, pending_polls=3, charge=4.0), timeout_s=10.0) == 4.0
    assert poll_usage(FakeJob([1], 1, pending_polls=10 ** 6)) is None              # stays pending: no figure
    man = {"backend": "ibm_fake", "run_dir": "r"}
    led = HwLedger(tmp_path / "l.json", 400.0)
    # usage stays pending (job.usage() would say 0): the account difference is booked
    svc = FakeService()
    job = FakeJob([10], 3, pending_polls=10 ** 6)
    led.add("fakejob0001", "ibm_fake", 12.0, "hw-d", "r")
    svc.ran = True
    got = collect_job(job, tmp_path, man, svc, led, 12.0, 100.0, 1)
    assert got["billed"]["source"] == "account_diff" and got["billed"]["used"] == 5.0
    e = led.entry("fakejob0001")
    assert e["quantum_seconds"] == 5.0 and e["usage_source"] == "account_diff"
    # IBM's own figure when it is final
    led.add("fakejob0001", "ibm_fake", 12.0, "hw-d", "r")
    got = collect_job(FakeJob([10], 3, pending_polls=1, charge=6.5), tmp_path, man, svc, led, 12.0, 100.0, 1)
    assert got["billed"]["source"] == "job" and got["billed"]["used"] == 6.5
    # a failed job is booked too (it can still consume QPU time) and the failure is reported
    led.add("fakejob0001", "ibm_fake", 12.0, "hw-d", "r")
    with pytest.raises(RuntimeError, match="did not return results"):
        collect_job(FakeJob([10], 3, charge=2.0, fail=True), tmp_path, man, svc, led, 12.0, 100.0, 1)
    assert man["state"] == "failed" and led.entry("fakejob0001")["quantum_seconds"] == 2.0


def test_ledger_budget_refusal_warning_and_compatibility(tmp_path, caplog, monkeypatch):
    from acts_qaoa.ibm import QPULedger
    from hsf2.hw.ledger import BudgetError, HwLedger
    monkeypatch.delenv("HSF2_QPU_BUDGET", raising=False)
    path = tmp_path / "led.json"
    led = HwLedger(path)
    assert led.budget_s == 400.0
    led.add("j1", "ibm_fez", 30.0, "hw-d", "run")
    led.book("j1", 350.0, "account_diff")
    with pytest.raises(BudgetError, match="exceeds"):
        led.check(60.0)                                       # 350 + 60 > 400
    with pytest.raises(BudgetError):
        led.check(10.0, service_remaining_s=5.0)              # IBM says less is left than the ledger does
    with pytest.raises(BudgetError):
        led.check(30.0, reserve_s=30.0)                       # the reserve counts
    caplog.set_level("WARNING")
    assert led.check(20.0) == pytest.approx(50.0)
    assert "of the budget" in caplog.text                     # 370 s > 80 % of 400 s
    with pytest.raises(ValueError):
        led.book("j1", 1.0, "guess")
    # the file is Phase 1's format: QPULedger and the shell script read and update it unchanged
    old = QPULedger(path, budget_s=400.0)
    assert old.entries[-1]["job_id"] == "j1" and old.entries[-1]["quantum_seconds"] == 350.0
    old.set_usage("j1", 360.0)                                # what scripts/qx_hardware.sh book_usage does
    assert "j1" in old.table() and HwLedger(path).entry("j1")["quantum_seconds"] == 360.0
    assert HwLedger(path).used_seconds() == pytest.approx(360.0)
    monkeypatch.setenv("HSF2_QPU_BUDGET", "123")
    assert HwLedger(tmp_path / "other.json").budget_s == 123.0
    monkeypatch.setenv("HSF2_QPU_BUDGET", "abc")
    with pytest.raises(ValueError):
        HwLedger(tmp_path / "other.json")


def test_ledger_path_per_account(tmp_path, monkeypatch):
    from hsf2.hw.config import default_ledger_path
    monkeypatch.delenv("HSF2_QPU_LEDGER", raising=False)
    monkeypatch.delenv("HSF2_IBM_CREDS", raising=False)
    assert default_ledger_path() == Path.home() / ".hsf2_part2_qpu_ledger.json"
    monkeypatch.setenv("HSF2_IBM_CREDS", str(tmp_path / "acct2" / "ibm_creds2.json"))
    assert default_ledger_path() == Path.home() / ".hsf2_part2_qpu_ledger_ibm_creds2.json"
    monkeypatch.setenv("HSF2_QPU_LEDGER", str(tmp_path / "mine.json"))
    assert default_ledger_path() == tmp_path / "mine.json"


def test_shell_script_uses_creds_budget_and_ledger_env():
    txt = _script().read_text()
    assert '${HSF2_IBM_CREDS:+--ibm-creds "$HSF2_IBM_CREDS"}' in txt
    assert "HSF2_QPU_BUDGET:-400" in txt and "--budget-seconds 300" not in txt and "budget_s=300" not in txt
    assert "_qpu_ledger_${CRED_BASE%.*}.json" in txt
    import shutil as sh
    import subprocess
    bash = sh.which("bash")
    if bash:
        assert subprocess.run([bash, "-n", str(_script())]).returncode == 0


@dataclass
class FakeOptions:
    note: str = "fake"


class FakeSampler:
    def __init__(self, job, service):
        self.job, self.service, self.calls = job, service, 0
        self.options = FakeOptions()

    def run(self, pubs, **kw):
        self.calls += 1
        self.service.ran = True
        self.job.shots = [p[2] for p in pubs]
        return self.job


def submit_env(monkeypatch, n_bits, pending_polls=10 ** 6, consumed=100.0):
    """Mock the IBM side of run_job(mode='submit'): fake backend, fake service, fake sampler."""
    from hsf2.hw import run as R
    from qiskit_ibm_runtime.fake_provider import FakeFez
    svc = FakeService(consumed)
    job = FakeJob([], n_bits, pending_polls=pending_polls)
    sampler = FakeSampler(job, svc)
    monkeypatch.setattr(R, "get_job_backend", lambda cfg, setting, service=None: (FakeFez(), svc))
    monkeypatch.setattr(R, "make_sampler", lambda *a, **k: (sampler, {"mock": True}))
    return svc, job, sampler


def test_submit_end_to_end_with_mocks(run_dir, tmp_path, monkeypatch, fast_polling):
    from acts_qaoa.ibm import QPULedger
    from hsf2.hw.run import run_job
    svc, job, sampler = submit_env(monkeypatch, 6)
    cfg = cfg_local(tmp_path, mode="submit", backend="ibm_fake")
    asked = []
    res = run_job(run_dir, "a", 6, cfg, command="test", confirm=lambda msg: asked.append(msg))
    assert len(asked) == 1 and "submit ONE job" in asked[0] and sampler.calls == 1
    man = res["manifest"]
    assert man["job_id"] == "fakejob0001" and man["state"] == "done"
    assert man["billed"]["source"] == "account_diff" and man["billed"]["used"] == 5.0   # usage pending: account difference
    led = QPULedger(cfg.ledger_path(), budget_s=400.0)
    assert led.entries[-1]["job_id"] == "fakejob0001" and led.entries[-1]["quantum_seconds"] == 5.0
    assert (run_dir / "samples" / "ibm-fake_fez_a_p1_top6_fakejob0001_main.json").exists()   # 12-character job id
    assert json.loads((res["job_dir"] / "manifest.json").read_text())["job_id"] == "fakejob0001"
    assert man["sampler_options"] == {"note": "fake"}


def test_submit_refused_by_budget_or_missing_confirmation(run_dir, tmp_path, monkeypatch):
    from hsf2.hw.ledger import BudgetError, HwLedger
    from hsf2.hw.run import SubmitAborted, confirm_submit, run_job
    svc, job, sampler = submit_env(monkeypatch, 6)
    cfg = cfg_local(tmp_path, mode="submit", backend="ibm_fake", budget_s=5.0)
    asked = []
    with pytest.raises(BudgetError, match="exceeds"):               # the estimate (about 8 s) is above the 5 s budget
        run_job(run_dir, "a", 6, cfg, confirm=lambda m: asked.append(m))
    assert sampler.calls == 0 and not asked and not cfg.ledger_path().exists()
    cfg = cfg_local(tmp_path, mode="submit", backend="ibm_fake", ledger=str(tmp_path / "l2.json"))
    with pytest.raises(SubmitAborted):                              # anything but 'submit' aborts
        run_job(run_dir, "a", 6, cfg, confirm=lambda m: confirm_submit(m, input_fn=lambda p: "yes"))
    assert sampler.calls == 0 and not HwLedger(cfg.ledger_path()).entries
    with pytest.raises(SubmitAborted):                              # no terminal: no job
        def eof(prompt):
            raise EOFError
        confirm_submit("x", input_fn=eof)
    confirm_submit("x", input_fn=lambda p: " submit ")              # exact word accepted


# ---------------------------------------------------------------------------- 9. mthree / readout mitigation
def _forward(p_true, n_bits, mats):
    """Observed distribution = (tensor product of the bit matrices) p_true, written independently of the module."""
    obs = np.zeros(1 << n_bits)
    for x in range(1 << n_bits):
        for y in range(1 << n_bits):
            a = 1.0
            for k in range(n_bits):
                a *= mats[k][(x >> k) & 1, (y >> k) & 1]
            obs[x] += a * p_true[y]
    return obs


def _cal(qubits, mats, shots=10 ** 6):
    from hsf2.hw.readout import calibration_record
    n = len(qubits)
    zeros = _forward(np.eye(1 << n)[0], n, mats)
    ones = _forward(np.eye(1 << n)[(1 << n) - 1], n, mats)
    c0 = {i: int(round(v * shots)) for i, v in enumerate(zeros) if v > 0}
    c1 = {i: int(round(v * shots)) for i, v in enumerate(ones) if v > 0}
    return calibration_record(qubits, c0, c1, shots, True, "s", "fake", None)


MATS = [np.array([[0.97, 0.20], [0.03, 0.80]]),      # clbit 0: strong 1 -> 0 decay
        np.array([[0.995, 0.01], [0.005, 0.99]]),
        np.array([[0.90, 0.02], [0.10, 0.98]])]      # clbit 2: strong 0 -> 1 flips
QUBITS = [5, 2, 7]


@pytest.mark.parametrize("prefer", ["numpy", "mthree"])
def test_readout_correction_recovers_distribution_and_bit_order(prefer):
    from hsf2.hw import readout as R
    if prefer == "mthree" and R.mthree_status()[0] is None:
        pytest.skip(f"mthree not usable here: {R.mthree_status()[1]}")
    cal = _cal(QUBITS, MATS)
    # the matrices come back from the calibration counts, per physical qubit, in the order of the classical bits
    for q, m in zip(QUBITS, MATS):
        assert np.allclose(cal["matrices"][str(q)], m, atol=2e-3)
    p_true = np.zeros(8)
    p_true[1] = 0.7                                           # '001': only clbit 0 set
    p_true[6] = 0.3                                           # '110'
    obs = _forward(p_true, 3, MATS)
    counts = {i: int(round(v * 10 ** 6)) for i, v in enumerate(obs) if v > 0}
    probs, overhead, used = R.correct_counts(counts, cal, 8, prefer=prefer)
    got = np.array([probs.get(i, 0.0) for i in range(8)])
    assert np.allclose(got, p_true, atol=5e-3), (got, p_true)
    assert overhead >= 1.0 - 1e-9 and (used == "numpy") == (prefer == "numpy")
    # the asymmetric case would be wrong with reversed bit order: correcting with reversed matrices must NOT recover it
    wrong = dict(cal, matrices={str(q): m.tolist() for q, m in zip(QUBITS, MATS[::-1])})
    bad, _, _ = R.correct_counts(counts, wrong, 8, prefer=prefer)
    assert not np.allclose([bad.get(i, 0.0) for i in range(8)], p_true, atol=5e-2)


def test_readout_calibration_circuits_and_numpy_matches_mthree_when_present():
    from hsf2.hw import readout as R
    c0, c1 = R.calibration_circuits([5, 2, 7], 10)
    assert c0.count_ops().get("x", 0) == 0 and c1.count_ops()["x"] == 3 and c0.num_clbits == 3
    meas = [(c0.find_bit(i.qubits[0]).index, c0.find_bit(i.clbits[0]).index) for i in c0.data if i.operation.name == "measure"]
    assert meas == [(5, 0), (2, 1), (7, 2)]                   # clbit k <- qubits[k]
    with pytest.raises(ValueError):
        R.calibration_circuits([5, 5], 10)
    with pytest.raises(ValueError):
        R.calibration_circuits([5, 12], 10)
    if R.mthree_status()[0] is None:
        pytest.skip(f"mthree not usable here: {R.mthree_status()[1]}")
    cal = _cal(QUBITS, MATS)
    rng = np.random.default_rng(0)
    counts = {int(i): int(v) for i, v in enumerate(rng.multinomial(2000, _forward(rng.dirichlet(np.ones(8)), 3, MATS))) if v}
    a, _, _ = R.correct_counts(counts, cal, 8, prefer="numpy")
    b, _, _ = R.correct_counts(counts, cal, 8, prefer="mthree")
    assert np.allclose([a.get(i, 0) for i in range(8)], [b.get(i, 0) for i in range(8)], atol=1e-3)


def test_mthree_failure_is_explained_not_silent(monkeypatch):
    from hsf2.hw import readout as R
    monkeypatch.setitem(sys.modules, "mthree", None)           # import mthree -> ImportError
    v, why = R.mthree_status()
    assert v is None and "mthree" in why and "Error" in why
    assert R.mitigation_backend().startswith("numpy")
    with pytest.raises(RuntimeError, match="mthree requested"):
        R.correct_counts({0: 10, 1: 5}, _cal([0], MATS[:1]), 4, prefer="mthree")


def test_apply_readout_to_a_job_refuses_mixed_twirling_and_other_qubits(local_job, tmp_path):
    from hsf2.hw.run import apply_readout
    man = local_job["manifest"]
    mq = man["layout"]["measured_qubits"]
    from hsf2.hw.readout import calibration_record
    n = len(mq)
    zeros = {0: 5000}
    ones = {(1 << n) - 1: 5000}
    rec = calibration_record(mq, zeros, ones, 5000, True, "s", "fake", None)      # a perfect readout
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(rec))
    res = apply_readout(local_job["job_dir"], path, prefer="numpy", out_name="readout_mitigated_test.json")
    assert res["mitigation"] == "numpy" and set(res["mean_energy"]) == {"main_mitigated", "control_mitigated",
                                                                         "main_raw", "control_raw"}
    assert res["mean_energy"]["main_mitigated"] == pytest.approx(res["mean_energy"]["main_raw"], abs=1e-6)
    assert (local_job["job_dir"] / "readout_mitigated_test.json").exists()
    bare = dict(rec, measure_twirl=False)
    path.write_text(json.dumps(bare))
    with pytest.raises(ValueError, match="never be mixed"):
        apply_readout(local_job["job_dir"], path, with_energy=False)
    other = calibration_record(mq[::-1], zeros, ones, 5000, True, "s", "fake", None)
    path.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="differ"):
        apply_readout(local_job["job_dir"], path, with_energy=False)


def test_readout_calibration_job_local(run_dir, local_job, tmp_path):
    from hsf2.hw.run import run_readout
    res = run_readout(run_dir, local_job["job_dir"], cfg_local(tmp_path), command="test", shots=SHOTS)
    cal = json.loads((res["job_dir"] / "readout_cal.json").read_text())
    assert cal["qubits"] == local_job["manifest"]["layout"]["measured_qubits"] and cal["measure_twirl"] is True
    assert cal["shots"] == SHOTS and cal["session"] and set(cal["matrices"]) == {str(q) for q in cal["qubits"]}
    for m in cal["matrices"].values():                          # columns are probabilities
        assert np.allclose(np.array(m).sum(axis=0), 1.0)
    assert all(0.0 <= v < 0.25 for v in cal["p1_given_0"].values())


# ---------------------------------------------------------------------------- 10. credentials
def test_credentials_path_only_never_opened(monkeypatch, tmp_path):
    from acts_qaoa import ibm
    from hsf2.hw import session
    from hsf2.hw.config import HwConfig, creds_label
    secret = tmp_path / "acct" / "apikey_secret.json"
    monkeypatch.setenv("HSF2_IBM_CREDS", str(secret))
    seen = {}
    monkeypatch.setattr(ibm, "get_service", lambda **kw: seen.update(kw) or "SERVICE")
    real_open = builtins.open

    def guarded(file, *a, **k):
        if "apikey_secret" in str(file):
            raise AssertionError("hsf2.hw opened the credentials file")
        return real_open(file, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded)
    assert session.get_service() == "SERVICE" and seen == {"creds_file": str(secret)}
    seen.clear()
    assert session.get_service(str(tmp_path / "other.json")) == "SERVICE" and seen["creds_file"].endswith("other.json")
    cfg = HwConfig.from_env()
    assert cfg.creds_file == str(secret) and cfg.to_dict()["creds_file"] == "apikey_secret.json"
    d = HwConfig.from_env(creds_file=str(secret)).to_dict()
    d.pop("ledger")
    assert "acct" not in json.dumps(d)                        # only the basename is ever recorded
    assert creds_label(None) is None and creds_label(str(secret)) == "apikey_secret.json"


def test_manifest_records_only_the_credentials_basename(run_dir, tmp_path):
    from hsf2.hw.run import run_job
    secret = tmp_path / "private_dir" / "apikey_x.json"
    res = run_job(run_dir, "a", 6, cfg_local(tmp_path, creds_file=str(secret)), command="test")   # local: never opened
    text = (res["job_dir"] / "manifest.json").read_text()
    assert res["manifest"]["creds"] == "apikey_x.json" and "private_dir" not in text and not secret.exists()


# ---------------------------------------------------------------------------- 11. sparsification
def _reference_top_k(J, k):
    """The documented rule: the k largest |J_kl| of the upper triangle; among equal values the later pairs win."""
    iu, ju = np.triu_indices(J.shape[0], 1)
    vals = np.abs(J[iu, ju])
    order = sorted(range(len(vals)), key=lambda i: (-vals[i], -i))
    return {(int(iu[i]), int(ju[i])) for i in order[:k]}


@pytest.mark.parametrize("k", [12, 48])
def test_sparsification_identical_to_phase1_including_ties(k):
    from acts_qaoa.surrogate import Ising
    from hsf2.hw.build import sparse_ising
    rng = np.random.default_rng(5)
    n = 24
    J = np.triu(np.round(rng.normal(size=(n, n)), 1), 1)        # rounded: many tied |J|, also exact zeros
    ising = Ising(n=n, offset=0.3, h=rng.normal(size=n), J=J)
    sp = sparse_ising(ising, k)
    kept = {(int(i), int(j)) for i, j in zip(*np.nonzero(sp.J))}
    assert kept == _reference_top_k(J, k)
    assert np.array_equal(sp.J, ising.sparsified(k).J) and np.array_equal(sp.h, ising.h) and sp.offset == 0.3
    assert sparse_ising(ising, None) is ising and sparse_ising(ising, 0) is ising
    assert len(kept) == k
    with pytest.raises(ValueError):
        sparse_ising(ising, -1)


def test_sparsification_on_the_qx2_problem_keeps_the_largest_couplings():
    prob = ROOT / "results" / "final" / "runs" / "02-11-2026-14-42_qx2_loop" / "seed0" / "qubo-qaoa-noise-top24"
    if not (prob / "problem.json").exists():
        pytest.skip("QX2 problem not present")
    from acts_qaoa.surrogate import Ising
    from hsf2.hw.build import sparse_ising
    ising = Ising.from_dict(json.loads((prob / "problem.json").read_text())["ising"])
    for k in (12, 24, 48):
        sp = sparse_ising(ising, k)
        kept = {(int(i), int(j)) for i, j in zip(*np.nonzero(sp.J))}
        assert kept == _reference_top_k(ising.J, k) and sp.n_couplings(0.0) == k


def test_backend_selection_real_target_and_fractional_flag(monkeypatch, tmp_path):
    from hsf2.hw import run as R
    from hsf2.hw import session
    from hsf2.hw.settings import get_setting
    from qiskit_ibm_runtime.fake_provider import FakeFez
    calls = []

    class Svc:
        def backend(self, name, use_fractional_gates=False):
            calls.append((name, use_fractional_gates))
            return FakeFez()

    class OldSvc:
        def backend(self, name):
            return FakeFez()

    monkeypatch.setattr(session, "get_service", lambda creds=None: Svc())
    R.get_job_backend(cfg_local(tmp_path, mode="dry", real_target=True, backend="ibm_x"), get_setting("e"))
    R.get_job_backend(cfg_local(tmp_path, mode="submit", backend="ibm_x"), get_setting("d"))
    assert calls == [("ibm_x", True), ("ibm_x", False)]
    n = len(calls)
    R.get_job_backend(cfg_local(tmp_path, mode="dry"), get_setting("d"))          # plain dry / local: fake, no login
    R.get_job_backend(cfg_local(tmp_path, mode="local"), get_setting("e"))
    assert len(calls) == n
    with pytest.raises(RuntimeError, match="use_fractional_gates"):
        session.get_backend(OldSvc(), "ibm_x", fractional=True)
    assert session.get_backend(OldSvc(), "ibm_x", fractional=False) is not None


# ---------------------------------------------------------------------------- 12. retrieve, outputs, analysis
def test_retrieve_job_rebuilds_the_outputs_of_a_stored_local_job(local_job, tmp_path):
    from hsf2.hw.run import retrieve_job
    src_run = local_job["job_dir"].parents[1]
    run = tmp_path / "copy"
    shutil.copytree(src_run, run)                               # a moved folder: paths come from the folder, not the manifest
    jd = run / "hw" / local_job["job_dir"].name
    job_id = local_job["manifest"]["job_id"]
    short = job_id[len("local_"):]
    mine = lambda: sorted((run / "samples").glob(f"*{short}*.json"))  # noqa: E731
    before = {p.name: p.read_bytes() for p in mine()}
    assert len(before) == 2
    for f in list(jd.glob("pub_*_counts.json")) + [jd / "snapshot_post.json"] + mine():
        f.unlink()
    man = json.loads((jd / "manifest.json").read_text())
    man["state"], man["samples"] = "submitted", []
    (jd / "manifest.json").write_text(json.dumps(man))
    res = retrieve_job(job_id, cfg_local(tmp_path), run_dir=run)
    assert res["job_dir"] == jd and json.loads((jd / "manifest.json").read_text())["state"] == "done"
    assert (jd / "snapshot_post.json").exists() and len(list(jd.glob("pub_*_counts.json"))) == 2
    after = {p.name: p.read_bytes() for p in mine()}
    assert set(after) == set(before)
    for name in before:
        a, b = json.loads(before[name]), json.loads(after[name])
        assert a["counts"] == b["counts"] and a["tag"] == b["tag"]
    # sample sets that exist are kept, never replaced
    mtime = {p: p.stat().st_mtime_ns for p in mine()}
    retrieve_job(job_id, cfg_local(tmp_path), run_dir=run)
    assert mtime == {p: p.stat().st_mtime_ns for p in mine()}
    with pytest.raises(FileNotFoundError):
        retrieve_job("local_00000000", cfg_local(tmp_path), run_dir=run)


def test_local_job_outputs_and_phase1_sample_format(local_job):
    jd, man = local_job["job_dir"], local_job["manifest"]
    assert man["state"] == "done" and man["job_id"].startswith("local_") and man["billed"]["used"] == 0.0
    assert man["schema"] == 1 and man["versions"]["qiskit"] and man["code"]["hsf2"] and man["primitive"] == "sampler_v2"
    for f in ("manifest.json", "snapshot_pre.json", "snapshot_post.json", "local_result.json", "pub_0_main.isa.qpy",
              "pub_1_control.isa.qpy", "pub_0.layout.json", "pub_0_counts.json", "pub_1_counts.json"):
        assert (jd / f).exists(), f
    run = jd.parents[1]
    short = man["job_id"][len("local_"):]
    main = json.loads((run / "samples" / f"ibm-fake_fez_d_p1_top6_{short}_main.json").read_text())
    ctrl = json.loads((run / "samples" / f"ibm-fake_fez_d_p1_top6_{short}_control.json").read_text())
    assert sum(main["counts"].values()) == SHOTS == sum(ctrl["counts"].values())
    assert main["meta"]["problem_fingerprint"] == man["fingerprint"] and main["meta"]["gammas"] == man["gammas"]
    assert {"mean_energy", "F_measured", "F_lo68", "F_hi68", "best_rank"} <= set(main["metrics"])
    # the tag is the one hw-analyze parses
    from hsf2.hw.analysis import parse_tag
    assert parse_tag(main["tag"]) == (6, short)


def test_hw_analyze_reads_the_exported_samples(local_job, tmp_path):
    from hsf2.hw.analysis import analyze_run
    run = local_job["job_dir"].parents[1]
    res = analyze_run(run, tmp_path / "out", kmax=3)
    rows = [r for r in res["rows"] if "_d_p1_top6_" in r["tag"]]
    assert len(rows) == 1 and rows[0]["K"] == 6 and rows[0]["shots"] == SHOTS
    assert rows[0]["ideal_mean_check"][0] == pytest.approx(rows[0]["ideal_mean_check"][1], abs=1e-6)
    assert 0.0 < rows[0]["F_E"] < 1.5


def test_gauged_local_job(run_dir, tmp_path):
    from hsf2.hw.gauge import merge_counts, xor_counts
    from hsf2.hw.run import run_job
    cfg = cfg_local(tmp_path)
    res = run_job(run_dir, "f", 6, cfg, n_gauges=2, command="test")
    jd, man = res["job_dir"], res["manifest"]
    assert [p["shots"] for p in man["pubs"]] == [SHOTS // 2] * 4 and len(man["gauges"]) == 2
    assert man["gauges"][0]["mask"] == 0 and man["gauges"][1]["mask"] != 0
    parts = []
    for g in (0, 1):
        d = json.loads((jd / f"gauge_{g}_counts.json").read_text())
        frame = {int(k): v for k, v in d["main"]["gauge_frame"].items()}
        orig = {int(k): v for k, v in d["main"]["original_frame"].items()}
        assert orig == xor_counts(frame, d["gauge"]["mask"]) and sum(frame.values()) == SHOTS // 2
        parts.append(orig)
    short = man["job_id"][len("local_"):]
    sample = json.loads((jd.parents[1] / "samples" / f"ibm-fake_fez_f_p1_top6_{short}_main.json").read_text())
    assert {int(k): v for k, v in sample["counts"].items()} == merge_counts(parts)
    # one gauge-0 circuit layout for every gauge: same qubits
    lays = [json.loads(p.read_text())["measured"]["qubits"] for p in sorted(jd.glob("pub_*.layout.json"))]
    assert all(l == lays[0] for l in lays)


def test_dry_mode_writes_the_plan_only(run_dir, tmp_path):
    from hsf2.hw.run import run_job
    res = run_job(run_dir, "b", 6, cfg_local(tmp_path, mode="dry"), command="test")
    jd = res["job_dir"]
    man = json.loads((jd / "manifest.json").read_text())
    assert man["state"] == "dry" and man["job_id"] is None and (jd / "snapshot_pre.json").exists()
    assert (jd / "pub_0_main.isa.qpy").exists() and not (jd / "snapshot_post.json").exists()
    assert not list(jd.glob("*counts*")) and not (tmp_path / "ledger.json").exists()
    assert res["plan"].estimate_s > 0


# ---------------------------------------------------------------------------- CLI
def test_cli_commands_dry_ledger_and_retrieve(run_dir, local_job, tmp_path, capsys, monkeypatch):
    from hsf2.cli import main
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))       # deployed next to Phase 1, run folders must lie in the work area
    work = tmp_path / "run"
    shutil.copytree(run_dir, work, ignore=shutil.ignore_patterns("hw", "samples", "outputs"))
    ledger = tmp_path / "cli_ledger.json"
    monkeypatch.delenv("HSF2_IBM_CREDS", raising=False)
    rc = main(["hw-run", str(work), "--setting", "a", "--K", "6", "--mode", "dry", "--seeds", "1", "--ledger", str(ledger)])
    assert rc == 0
    out = [p for p in (work / "outputs").iterdir() if p.name.endswith("_hw_run")]
    assert len(out) == 1 and (out[0] / "commands.jsonl").exists() and (out[0] / "config.json").exists()
    assert "hw-run" in (out[0] / "commands.jsonl").read_text()
    assert len(list((work / "hw").glob("*_a_K6"))) == 1
    # ledger command prints the table and the budget
    capsys.readouterr()
    assert main(["hw-ledger", "--ledger", str(ledger), "--budget", "250"]) == 0
    assert "budget 250 s" in capsys.readouterr().out
    # a refused submit is a normal stop with exit code 1 (budget too small, nothing submitted)
    from hsf2.hw import run as R
    monkeypatch.setattr(R, "get_job_backend", lambda cfg, setting, service=None: (local_job["plan"].backend, FakeService()))
    rc = main(["hw-run", str(work), "--setting", "a", "--K", "6", "--mode", "submit", "--seeds", "1", "--budget", "1",
               "--ledger", str(ledger)])
    assert rc == 1 and not ledger.exists()
    # retrieve of an unknown job id fails loudly
    with pytest.raises(FileNotFoundError):
        main(["hw-retrieve", "local_deadbeef", "--run-dir", str(work), "--mode", "local"])


def test_cli_ladder_local_two_settings(run_dir, tmp_path, monkeypatch):
    from hsf2.cli import main
    from hsf2.hw import run as R
    calls = []
    real = R.transpile_job
    monkeypatch.setattr(R, "transpile_job", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))       # deployed next to Phase 1, run folders must lie in the work area
    work = tmp_path / "run"
    shutil.copytree(run_dir, work, ignore=shutil.ignore_patterns("hw", "samples", "outputs"))
    rc = main(["hw-ladder", str(work), "--K", "6", "--settings", "a", "d", "d", "--mode", "local", "--shots", str(SHOTS),
               "--seeds", "1", "--ledger", str(tmp_path / "l.json")])
    assert rc == 0
    out = [p for p in (work / "outputs").iterdir() if p.name.endswith("_hw_ladder")][0]
    rows = json.loads((out / "ladder_summary.json").read_text())["rows"]
    assert [r["setting"] for r in rows] == ["a", "d", "d"] and all(r["state"] == "done" for r in rows)
    assert [r["tag"] for r in rows] == ["a_K6", "d_K6_1", "d_K6_2"]       # the repeated d is told apart
    assert len({r["job_id"] for r in rows}) == 3 and len(list((work / "samples").glob("ibm-*_main.json"))) == 3
    assert not (tmp_path / "l.json").exists()                              # local jobs never touch the ledger
    assert len(calls) == 1                                                 # a, d, d: one transpilation, same circuits/qubits
    lays = [json.loads(next(Path(r["job_dir"]).glob("pub_0.layout.json")).read_text())["measured"]["qubits"] for r in rows]
    assert lays[0] == lays[1] == lays[2]


# ---------------------------------------------------------------------------- 13. step-4 stubs
@pytest.mark.parametrize("name", ["trex", "zne", "pec", "noise_learner"])
def test_mitigation_stubs_raise_not_implemented(name):
    from hsf2.hw.mitig import Mitigator, get_mitigator
    m = get_mitigator(name)
    assert isinstance(m, Mitigator) and m.name == name
    with pytest.raises(NotImplementedError, match=name):
        m.prepare([])
    with pytest.raises(NotImplementedError, match="not implemented"):
        m.postprocess(None)
    with pytest.raises(ValueError):
        get_mitigator("nope")

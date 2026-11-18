"""Step 5, QX4 on hardware (hsf2.hw.qx4hw, CLI hw-qx4 / hw-qx4-analyze): the 1-bit filter circuits against the eigen-emulation, folding,
bit order (ancilla last), the white-noise closed form, ratio zero-noise extrapolation on synthetic counts, manifest round trip, the
typed confirmation, the budget refusal and the dry run's gate counts on FakeFez.  0 QPU s: fake backends, Aer, mocks; IBM is never
contacted.  Shared helpers come from the component tests under either name (repository or lxplus)."""
import importlib
import io
import json
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

SHOTS = 300


def component_tests():
    for n in ("test_hw_components", "test_hsf2_hw_components"):
        try:
            return importlib.import_module(n)
        except ImportError:
            continue
    pytest.skip("the shared component tests cannot be imported")


def cfg_for(tmp_path, **kw):
    from hsf2.hw.config import HwConfig
    d = dict(mode="dry", shots=SHOTS, transpile_seeds=1, ledger=str(tmp_path / "ledger.json"), budget_s=400.0)
    d.update(kw)
    return HwConfig(**d)


@pytest.fixture
def fast_polling(monkeypatch):
    from hsf2.hw import run as R
    for k, v in (("USAGE_TIMEOUT_S", 0.05), ("USAGE_INTERVAL_S", 0.005), ("ACCOUNT_TIMEOUT_S", 0.05),
                 ("ACCOUNT_INTERVAL_S", 0.005)):
        monkeypatch.setattr(R, k, v)


@pytest.fixture(scope="module")
def dry_job(tmp_path_factory):
    """The headline dry run: 2x3 at 1/3/5, 2x5, HHL, 6 transpiler seeds."""
    from hsf2.hw.qx4hw import parse_events, run_qx4
    tmp = tmp_path_factory.mktemp("qx4dry")
    res = run_qx4(tmp / "run", parse_events("2x3,2x5"), "b", cfg_for(tmp, mode="dry", shots=4000, transpile_seeds=6), command="test",
                  out_dir=tmp / "out")
    res["tmp"] = tmp
    return res


# ---------------------------------------------------------------------------------------- circuits and ideal values
def test_statevector_conditional_distribution_equals_the_eigen_emulation_and_leakage_is_zero():
    from hsf2.hw.qx4hw import build_filter, eigen_reference, ideal_cond
    for npart, nlay in ((2, 3), (1, 3)):
        c = build_filter(npart, nlay)
        ide, eig = ideal_cond(c), eigen_reference(c)
        assert ide["cond"][:c.N] == pytest.approx(eig["cond"], abs=1e-12) and ide["p_post"] == pytest.approx(eig["p_post"], abs=1e-12)
        assert ide["leakage"] == pytest.approx(0.0, abs=1e-12) and abs(ide["cond"][c.N:]).sum() < 1e-12
    c = build_filter(2, 3)
    ide = ideal_cond(c)
    assert (c.N, c.q, c.n_bits, c.n_terms, c.label) == (8, 3, 4, 3, "filter8")
    assert ide["p_post"] == pytest.approx(0.125) and c.kept.sum() == 4
    assert ide["cond"][c.kept] == pytest.approx([0.25] * 4) and abs(ide["cond"][~c.kept]).max() < 1e-12
    assert c.truth.tolist() == c.kept[:c.N].tolist()                       # the connected segments are the true ones here
    c16 = build_filter(2, 5)                                                # Trotter error: the circuit differs slightly from exp(iAt)
    i16 = ideal_cond(c16)
    assert c16.N == 16 and c16.n_bits == 5 and i16["leakage"] == pytest.approx(0.0, abs=1e-12)
    assert i16["cond"] == pytest.approx(eigen_reference(c16)["cond"], abs=0.01)


def test_hhl_post_selection_probability_matches_the_qpe_emulation():
    from hsf2.hw.qx4hw import build_filter, eigen_reference, ideal_cond
    c = build_filter(1, 3, circuit="hhl", n_clock=2)
    ide = ideal_cond(c)
    assert (c.N, c.n_bits, c.anc_bit, c.anc_value, c.clock_bits) == (2, 4, 3, 1, [1, 2])
    assert ide["p_post"] == pytest.approx(eigen_reference(c)["p_post"], abs=1e-9)
    assert ide["cond"] == pytest.approx(eigen_reference(c)["cond"], abs=1e-9)
    with pytest.raises(ValueError, match="'filter' or 'hhl'"):
        build_filter(2, 3, circuit="x")


def test_event_and_lambda_parsing():
    from hsf2.hw.qx4hw import parse_events, parse_lams
    assert parse_events("2x3,2x5") == [(2, 3), (2, 5)] and parse_lams("1,3,5") == [1, 3, 5] and parse_lams("3,1.5,2") == [1.5, 2, 3]
    for bad in ("2x2", "x3", "2x3,2x3", ""):
        with pytest.raises(ValueError):
            parse_events(bad)
    for bad in ("0.5", "1,1", "", "nan"):
        with pytest.raises(ValueError):
            parse_lams(bad)


def test_folding_preserves_the_ideal_operator_and_output():
    from qiskit import transpile
    from qiskit.quantum_info import Operator
    from hsf2.hw.qx4hw import build_filter
    from hsf2.hw.zne import fold_two_qubit_gates
    c = build_filter(2, 3)
    tq = transpile(c.qc, basis_gates=["cz", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)
    ref = Operator(tq)
    n2q = tq.count_ops()["cz"]
    for lam in (3, 5):
        f = fold_two_qubit_gates(tq, lam)
        assert Operator(f).equiv(ref) and f.count_ops()["cz"] == lam * n2q


def noiseless_counts(isa, shots=20000, seed=1):
    from acts_qaoa.circuits import counts_to_index
    from qiskit_aer import AerSimulator
    res = AerSimulator(method="matrix_product_state").run(isa, shots=shots, seed_simulator=seed).result()
    return counts_to_index(res.get_counts())


def test_bit_order_ancilla_is_the_last_classical_bit_and_the_system_index_is_the_segment():
    """Noiseless sampling of the transpiled (and folded) circuit: P(ancilla bit = 0) = 1/8 and the conditional frequency of system
    index s is 1/4 on the true segments and exactly 0 on the isolated ones, with index = s + 2^q * ancilla."""
    from hsf2.hw import fake as F
    from hsf2.hw.qx4hw import build_filter, default_folder, ideal_cond, transpile_filter
    c = build_filter(2, 3)
    assert [cr.name for cr in c.qc_meas.cregs] == ["meas"] and c.anc_bit == c.n_bits - 1 == 3
    pairs = [(c.qc_meas.find_bit(q).index, c.qc_meas.find_bit(b).index) for i in c.qc_meas.data if i.operation.name == "measure"
             for q, b in zip(i.qubits, i.clbits)]
    assert pairs == [(k, k) for k in range(4)]
    pub = transpile_filter(c, F.fake_backend("FakeFez"), 1)
    assert pub.layout["measured"]["clbits"] == [0, 1, 2, 3]
    ide = ideal_cond(c)
    for lam in (1, 3):
        counts = noiseless_counts(default_folder(pub.isa, lam, 0))
        v = np.zeros(16)
        for k, n in counts.items():
            v[k] = n
        v /= v.sum()
        assert v.reshape(2, 8)[0].sum() == pytest.approx(0.125, abs=0.01)             # ancilla = bit 3 = the last bit
        cond = v[:8] / v[:8].sum()
        assert cond[~c.kept[:8]].sum() == 0 and cond[c.kept[:8]] == pytest.approx([0.25] * 4, abs=0.03)
        assert cond == pytest.approx(ide["cond"], abs=0.03)
        assert v[8:][~c.kept[:8]].sum() > 0.2                                          # ancilla = 1 carries the rejected segments


# ---------------------------------------------------------------------------------------- closed forms and analysis
def white_noise_vector(c, ide, F):
    return F * ide["probs"] + (1 - F) / (1 << c.n_bits)


def test_white_noise_counts_give_the_closed_form_leakage():
    from hsf2.hw.qx4hw import analyze_target, build_filter, ideal_cond, uniform_leakage, white_noise_leakage
    c = build_filter(2, 3)
    ide = ideal_cond(c)
    assert white_noise_leakage(c, 0.74, ide) == pytest.approx(0.2921348, abs=1e-6) and uniform_leakage(c) == 0.5
    assert white_noise_leakage(c, 1.0, ide) == 0.0 and white_noise_leakage(c, 0.0, ide) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="fidelity"):
        white_noise_leakage(c, 1.2)
    for F in (0.74, 0.3):
        p = white_noise_vector(c, ide, F)
        num, den = p[c.kept_post_mask].sum(), p[c.post_mask].sum()
        assert 1 - num / den == pytest.approx(white_noise_leakage(c, F, ide), abs=1e-12)
    rng = np.random.default_rng(3)
    p = white_noise_vector(c, ide, 0.74)
    cnt = rng.multinomial(400_000, p)
    counts = {int(i): int(n) for i, n in enumerate(cnt) if n}
    res = analyze_target(c, ide, {1: counts}, boot_n=200, seed=1, tag="wn")
    pt = res["variants"]["raw"]["points"]["1"]["L"]
    assert abs(pt["value"] - white_noise_leakage(c, 0.74, ide)) < 4 * pt["err"] and pt["err"] < 0.01
    assert pt["brackets_ideal"] is False and "extrapolation" not in res["variants"]["raw"]


def test_ratio_extrapolation_recovers_known_values_and_negative_values_are_not_clipped():
    from hsf2.hw.qx4hw import analyze_target, build_filter, ideal_cond, ratio_extrapolation
    c = build_filter(2, 3)
    ide = ideal_cond(c)
    f, shots, lams = 0.62, 200_000, (1, 3, 5)
    rng = np.random.default_rng(5)
    counts = {}
    for lam in lams:
        p = white_noise_vector(c, ide, f ** lam)
        cnt = rng.multinomial(shots, p)
        counts[lam] = {int(i): int(n) for i, n in enumerate(cnt) if n}
    res = analyze_target(c, ide, counts, boot_n=300, seed=2, tag="wn")["variants"]["raw"]
    ex = res["extrapolation"]
    e = ex["exponential"]
    assert e["num"]["value"] == pytest.approx(ide["p_kept_post"], abs=4 * e["num"]["err"])
    assert e["den"]["value"] == pytest.approx(ide["p_post"], abs=4 * e["den"]["err"])
    assert e["L"]["err"] > 0 and abs(e["L"]["value"]) < 2.5 * e["L"]["err"] and e["L"]["consistent_with_ideal_2sigma"] in (True, False)
    assert abs(e["L"]["value"]) < 0.05 and res["points"]["1"]["L"]["value"] > 0.15       # lambda = 1 is far from 0, the extrapolation is not
    assert ex["linear"]["L"]["value"] > e["L"]["value"] + 0.05                             # a line through f^lambda undershoots P(post)
    # a known non-zero ideal: num = a, den = b at lambda = 0, exponential decay to the uniform floor
    lam_a = np.array(lams, float)
    a, b, fn, fd = 0.10, 0.12, 0.04, 0.5
    num = np.stack([fn + (a - fn) * f ** l for l in lam_a])[:, None]
    den = np.stack([fd + (b - fd) * f ** l for l in lam_a])[:, None]
    out = ratio_extrapolation(lams, num, den, fn, fd, 1 - a / b)
    assert out["exponential"]["L"]["value"] == pytest.approx(1 - a / b, abs=1e-9)
    assert out["richardson"]["L"]["value"] != pytest.approx(1 - a / b, abs=1e-3)
    # an extrapolated numerator that exceeds the denominator gives a negative L, which is kept
    out = ratio_extrapolation(lams, np.stack([fn + (0.14 - fn) * f ** l for l in lam_a])[:, None], den, fn, fd, 0.0)
    assert out["exponential"]["L"]["value"] == pytest.approx(1 - 0.14 / b) and out["exponential"]["L"]["value"] < 0


def test_readout_variant_recovers_the_noiseless_conditional_distribution():
    from hsf2.hw.qx4hw import analyze_target, build_filter, ideal_cond
    from hsf2.hw.readout import tensor_apply
    c = build_filter(2, 3)
    ide = ideal_cond(c)
    rng = np.random.default_rng(8)
    mats = [np.array([[1 - a, b], [a, 1 - b]]) for a, b in zip(rng.uniform(0.01, 0.04, 4), rng.uniform(0.02, 0.08, 4))]
    p_obs = tensor_apply(ide["probs"], mats)
    cnt = rng.multinomial(2_000_000, p_obs / p_obs.sum())
    counts = {int(i): int(n) for i, n in enumerate(cnt) if n}
    res = analyze_target(c, ide, {1: counts}, mats, boot_n=50, seed=0, tag="ro")["variants"]
    raw, cor = res["raw"]["points"]["1"], res["readout"]["points"]["1"]
    assert raw["L"]["value"] > 0.02 and abs(cor["L"]["value"]) < 0.004 and cor["TV"]["value"] < raw["TV"]["value"]
    assert cor["P_post"]["value"] == pytest.approx(0.125, abs=0.002)
    assert cor["eff_classical"]["value"] == 1.0 and cor["pur_classical"]["value"] == 1.0


# ---------------------------------------------------------------------------------------- the dry run
def test_dry_plan_counts_manifest_round_trip_and_files(dry_job):
    from hsf2.hw.qx4hw import _manifest_circuit, build_filter, ideal_cond
    from hsf2.hw.snapshot import read_qpy
    man, jd = dry_job["manifest"], dry_job["job_dir"]
    assert man["kind"] == "qx4" and man["step"] == "step5" and man["state"] == "dry" and man["setting"]["name"] == "b"
    assert not man["setting"]["gate_twirl"] and man["setting"]["dd"] is True and man["job_id"] is None
    pubs = man["pubs"]
    assert [(p["target"], p["lam"]) for p in pubs] == [("filter8", 1), ("filter8", 3), ("filter8", 5), ("filter16", 1), ("hhl2", 1)]
    assert {p["shots"] for p in pubs} == {4000}
    base = {p["target"]: p["n_2q_base"] for p in pubs}
    assert 50 <= base["filter8"] <= 60 and 270 <= base["filter16"] <= 282 and 48 <= base["hhl2"] <= 58
    assert all(p["n_2q"] == p["lam"] * base[p["target"]] for p in pubs)
    assert [p["n_vars"] for p in pubs] == [4, 4, 4, 5, 4] and all(p["measured_qubits"] for p in pubs)
    assert len({tuple(p["measured_qubits"]) for p in pubs if p["target"] == "filter8"}) == 1     # same layout at every lambda
    for p in pubs:
        isa = read_qpy(jd / f"pub_{p['k']}_{p['target']}_lam{p['lam']}.isa.qpy")
        assert isa.count_ops().get("cz", 0) == p["n_2q"] and isa.count_ops().get("barrier", 0) >= (p["lam"] - 1) * p["n_2q_base"]
        assert (jd / f"pub_{p['k']}.layout.json").exists()
        assert abs(p["ideal_p_post"] - ideal_cond(build_filter(p["n_particles"], p["n_layers"], p["event_seed"], p["alpha"], p["beta"],
                                                              p["eps"], p["T"], p["circuit"], p["n_clock"] or 2))["p_post"]) < 1e-12
    assert (jd / "snapshot_pre.json").exists() and not list(jd.glob("*counts*"))
    # the manifest on disk is the manifest in memory, and rebuilds the circuits that were planned
    on_disk = json.loads((jd / "manifest.json").read_text())
    assert on_disk["pubs"] == json.loads(json.dumps(pubs)) and on_disk["selection"]["events"] == [[2, 3], [2, 5]]
    for label in ("filter8", "filter16", "hhl2"):
        p = next(x for x in on_disk["pubs"] if x["target"] == label)
        assert _manifest_circuit(p).meta()["A_sha1"] == p["A_sha1"]
    bad = dict(next(x for x in on_disk["pubs"] if x["target"] == "filter8"), A_sha1="0" * 40)
    with pytest.raises(ValueError, match="not the circuit that was run"):
        _manifest_circuit(bad)
    plan = json.loads((dry_job["tmp"] / "out" / "qx4_plan.json").read_text())
    assert [r["n_2q"] for r in plan["pubs"]] == [p["n_2q"] for p in pubs] and plan["estimate_total_s"] > plan["estimate_job_s"] > 0
    assert not (dry_job["tmp"] / "ledger.json").exists()                                            # dry mode never books anything
    ro = dry_job["readout"]["manifest"]
    assert ro["kind"] == "readout_calibration" and ro["state"] == "dry" and len(ro["qubits"]) >= 5


def test_fractional_factor_needs_a_folder_and_a_custom_folder_is_called_with_isa_lambda_seed(tmp_path):
    from hsf2.hw.qx4hw import parse_events, run_qx4
    cfg = cfg_for(tmp_path, mode="dry")
    with pytest.raises(ValueError, match="odd positive"):
        run_qx4(tmp_path / "r1", parse_events("2x3"), "b", cfg, lams=[1, 1.5], hhl=False, readout=False)
    calls = []

    def folder(isa, lam, seed):
        calls.append((lam, seed))
        return isa.copy()                                  # a stand-in: no folding at all
    res = run_qx4(tmp_path / "r2", parse_events("2x3"), "b", cfg, lams=[1, 1.5, 2], hhl=False, readout=False, folder=folder)
    assert [c[0] for c in calls] == [1, 1.5, 2] and calls[0][1] == 0
    assert [p["lam"] for p in res["manifest"]["pubs"]] == [1, 1.5, 2]
    with pytest.raises(ValueError, match="setting|no gauges"):
        run_qx4(tmp_path / "r3", parse_events("2x3"), "f", cfg)


# ---------------------------------------------------------------------------------------- mocked submit
def test_submit_confirms_job_and_calibration_budget_is_checked_first_and_no_confirmation_aborts(tmp_path, monkeypatch, fast_polling):
    from acts_qaoa.ibm import QPULedger
    from hsf2.hw import run as R
    from hsf2.hw.ledger import BudgetError
    from hsf2.hw.qx4hw import parse_events, run_qx4
    C = component_tests()
    svc, smp = C.mock_ibm(monkeypatch)
    cfg = cfg_for(tmp_path, mode="submit", backend="ibm_fake")
    run = tmp_path / "run"
    asked = []
    res = run_qx4(run, parse_events("2x3"), "b", cfg, lams=[1, 3], hhl=False, command="test", confirm=asked.append)
    assert len(asked) == 2 and smp.calls == 2                                             # the job and its read-out calibration
    assert "submit ONE job" in asked[0] and "filter8 lambda=3" in asked[0] and "QX4 job" in asked[0] and "readout" in asked[1]
    man = res["manifest"]
    assert man["state"] == "done" and man["billed"]["source"] == "account_diff" and man["job_id"] == "fakejob0001"
    assert (res["job_dir"] / "snapshot_post.json").exists() and (res["job_dir"] / "pub_1_counts.json").exists()
    assert [e["job_id"] for e in QPULedger(cfg.ledger_path(), budget_s=400.0).entries] == ["fakejob0001", "fakejob0002"]
    assert res["readout"]["cal"]["qubits"] == sorted({q for p in man["pubs"] for q in p["measured_qubits"]})
    # hw-retrieve of a qx4 job goes through retrieve_any
    from hsf2.hw.components import retrieve_any
    (res["job_dir"] / "pub_0_counts.json").unlink()
    got = retrieve_any("fakejob0001", cfg, run_dir=run, service=svc)
    assert got["job_dir"] == res["job_dir"] and (res["job_dir"] / "pub_0_counts.json").exists()
    # the budget of job + calibration is refused before the first prompt, the job folder is not even created
    n_dirs, calls = len(list((run / "hw").iterdir())), smp.calls
    asked.clear()
    with pytest.raises(BudgetError, match="exceeds"):
        run_qx4(run, parse_events("2x3"), "b", cfg_for(tmp_path, mode="submit", backend="ibm_fake", budget_s=1.0,
                                                       ledger=str(tmp_path / "l2.json")), hhl=False, confirm=asked.append)
    assert smp.calls == calls and not asked and not (tmp_path / "l2.json").exists() and len(list((run / "hw").iterdir())) == n_dirs
    # a job that costs more than the budget minus the calibration is refused as a whole (calibration is part of the check)
    one = run_qx4(tmp_path / "probe", parse_events("2x3"), "b", cfg_for(tmp_path), hhl=False)
    assert one["estimate_total_s"] > one["manifest"]["estimate_s"] + 1.0
    with pytest.raises(BudgetError):
        run_qx4(run, parse_events("2x3"), "b", cfg_for(tmp_path, mode="submit", backend="ibm_fake", ledger=str(tmp_path / "l3.json"),
                                                       budget_s=one["manifest"]["estimate_s"] + 0.5), hhl=False, confirm=asked.append)
    assert not asked and smp.calls == calls
    # the default confirmation reads standard input: EOF, or anything but 'submit', aborts before the sampler is called
    for text in ("", "yes\n"):
        monkeypatch.setattr(sys, "stdin", io.StringIO(text))
        with pytest.raises(R.SubmitAborted):
            run_qx4(run, parse_events("2x3"), "b", cfg_for(tmp_path, mode="submit", backend="ibm_fake", ledger=str(tmp_path / "l4.json")),
                    hhl=False)
        assert smp.calls == calls
    aborted = [json.loads(p.read_text()) for p in (run / "hw").glob("*/manifest.json")]
    aborted = [m for m in aborted if m.get("state") == "aborted"]
    assert len(aborted) == 2 and all(m["job_id"] is None and "SubmitAborted" in m["aborted"] for m in aborted)
    monkeypatch.setattr(sys, "stdin", io.StringIO("submit\nsubmit\n"))                      # what `yes submit |` delivers
    run_qx4(run, parse_events("2x3"), "b", cfg_for(tmp_path, mode="submit", backend="ibm_fake", ledger=str(tmp_path / "l5.json")),
            lams=[1], hhl=False)
    assert smp.calls == calls + 2


# ---------------------------------------------------------------------------------------- local end to end and the CLI
def test_cli_hw_qx4_dry_local_and_analyze_in_deployed_layout(tmp_path, monkeypatch):
    from hsf2.cli import main
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))
    monkeypatch.delenv("HSF2_IBM_CREDS", raising=False)
    secret = tmp_path / "private_dir" / "apikey_qx4.json"                                   # never opened; only its name is recorded
    common = ["--event", "2x3", "--lams", "1,3", "--no-hhl", "--seeds", "1", "--shots", str(SHOTS), "--ledger", str(tmp_path / "led.json")]
    assert main(["hw-qx4", *common, "--creds", str(secret)]) == 0
    runs = sorted((tmp_path / "runs").glob("*_qx4_hw*"))
    assert len(runs) == 1
    jobs = sorted((runs[0] / "hw").glob("??-??-????-??-??_qx4_b_1"))
    assert len(jobs) == 1 and not (tmp_path / "led.json").exists()
    man = json.loads((jobs[0] / "manifest.json").read_text())
    assert man["state"] == "dry" and man["creds"] == "apikey_qx4.json" and "private_dir" not in (jobs[0] / "manifest.json").read_text()
    cmd = next(p for p in (runs[0] / "outputs").glob("*_hw_qx4*") if (p / "qx4_plan.json").exists())
    assert (cmd / "commands.jsonl").exists() and (cmd / "config.json").exists()
    with pytest.raises(SystemExit, match="event"):
        main(["hw-qx4", *common[2:], "--event", "2x2", "--run-dir", str(runs[0])])
    with pytest.raises(ValueError, match="odd positive"):
        main(["hw-qx4", "--run-dir", str(runs[0]), "--event", "2x3", "--lams", "1,2", "--no-hhl", "--seeds", "1"])
    # local: simulate on FakeFez with Aer noise, read-out calibration, automatic analysis; then analyse again with the calibration
    assert main(["hw-qx4", *common, "--mode", "local", "--run-dir", str(runs[0]), "--boot-n", "50", "--hhl"]) == 0
    done = [p for p in (runs[0] / "hw").glob("??-??-????-??-??_qx4_b_1*") if (p / "pub_0_counts.json").exists()]
    assert len(done) == 1
    cal = next((runs[0] / "hw").glob("*readout_bare_qx4_b_1*/readout_cal.json"))
    auto = next(p for p in (runs[0] / "outputs").glob("*_hw_qx4*") if "analyze" not in p.name and (p / "qx4_analysis.json").exists())
    res = json.loads((auto / "qx4_analysis.json").read_text())
    assert set(res["targets"]) == {"filter8", "hhl2"} and set(res["targets"]["filter8"]["variants"]) == {"raw", "readout"}
    assert main(["hw-qx4-analyze", str(done[0]), "--readout", str(cal), "--boot-n", "40", "--seed", "3"]) == 0
    an = next(p for p in (runs[0] / "outputs").glob("*_hw_qx4_analyze*") if (p / "qx4_analysis.json").exists())
    out = json.loads((an / "qx4_analysis.json").read_text())
    f8 = out["targets"]["filter8"]
    L1 = f8["variants"]["raw"]["points"]["1"]["L"]
    assert f8["lambdas"] == [1, 3] and f8["n_2q"]["3"] == 3 * f8["n_2q"]["1"] and 0.0 <= L1["value"] <= 0.6 and L1["err"] > 0
    assert 0.0 < f8["variants"]["raw"]["points"]["1"]["P_post"]["value"] < 0.4
    assert f8["variants"]["readout"]["extrapolation"]["exponential"]["L"]["n_valid"] >= 0
    assert (an / "qx4_points.csv").exists() and (an / "qx4_extrapolation.csv").exists() and (an / "qx4_control.csv").exists()
    h = out["targets"]["hhl2"]
    assert h["circuit"] == "hhl" and h["ideal"]["p_post"] == pytest.approx(0.6565, abs=1e-3) and "extrapolation" not in h["variants"]["raw"]
    assert not any("private_dir" in p.read_text() for p in (runs[0] / "outputs").rglob("*.json*"))
    with pytest.raises(SystemExit, match="job folder"):
        main(["hw-qx4-analyze", str(runs[0])])

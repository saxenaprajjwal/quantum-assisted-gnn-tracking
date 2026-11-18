"""Readout mitigation inside the analysis (hw-analyze --readout, hw-components-analyze --readout): calibration checks, bit order
and gauge frames of the correction, raw and mitigated values side by side, tag selection, CLI.  0 QPU s: synthetic job
folders, the fake backend with Aer, white-noise local mode; IBM is never contacted.  Shared helpers are imported from the
runner / component tests under either name (repository: test_hw_*.py, lxplus: test_hsf2_hw_*.py)."""
import importlib
import json
import shutil
import sys
from pathlib import Path

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


def shared(*names):
    """The first importable test module of `names` (repository name, deployed name)."""
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


def noisy_cal(qubits, seed=0, twirl=True, backend="fakeback", scale=0.04, session="s1"):
    """A readout_cal.json record with asymmetric, qubit-dependent confusion matrices M[i, j] = P(read i | prepared j)."""
    rng = np.random.default_rng(seed)
    mats = {}
    for q in qubits:
        e01, e10 = scale * rng.uniform(0.2, 1.0), 1.7 * scale * rng.uniform(0.2, 1.0)      # P(1|0), P(0|1)
        mats[str(q)] = [[1 - e01, e10], [e01, 1 - e10]]
    return {"schema": 1, "session": session, "backend": backend, "job_id": "caljob", "shots": 4000,
            "measure_twirl": twirl, "qubits": list(qubits), "matrices": mats}


def apply_noise(p, cal):
    """The exact distribution that the confusion matrices of `cal` (classical-bit order) produce from the true p."""
    n = int(np.log2(p.size))
    t = p.reshape([2] * n)                              # axis a = bit n-1-a
    for k, q in enumerate(cal["qubits"]):
        axis = n - 1 - k
        t = np.moveaxis(np.tensordot(np.array(cal["matrices"][str(q)]), t, axes=([1], [axis])), 0, axis)
    return t.reshape(-1)


# ---------------------------------------------------------------------------------------- calibration checks
def test_calibration_for_refuses_mixing_wrong_qubits_wrong_backend_and_slices_supersets():
    from hsf2.hw.readout import calibration_for
    cal = noisy_cal([4, 7, 2, 9], twirl=True, backend="ibm_x")
    assert calibration_for(cal, [4, 7, 2, 9], True, "ibm_x") is cal
    with pytest.raises(ValueError, match="never be mixed"):
        calibration_for(cal, [4, 7, 2, 9], False)                       # a twirled calibration on bare counts
    with pytest.raises(ValueError, match="never be mixed"):
        calibration_for(dict(cal, measure_twirl=False), [4, 7, 2, 9], True)    # and the reverse
    with pytest.raises(ValueError, match="differ"):
        calibration_for(cal, [7, 4, 2, 9], True)                        # same qubits, other classical-bit order
    with pytest.raises(ValueError, match="backend"):
        calibration_for(cal, [4, 7, 2, 9], True, "ibm_y")
    sub = calibration_for(cal, [9, 4], True, "ibm_x", subset=True)       # a larger calibration, matrices in the PUB's order
    assert sub["qubits"] == [9, 4] and sub["matrices"] == {"9": cal["matrices"]["9"], "4": cal["matrices"]["4"]}
    with pytest.raises(ValueError, match="no matrix"):
        calibration_for(cal, [9, 5], True, subset=True)


# ---------------------------------------------------------------------------------------- bit order and gauge frames
def make_job_folder(run, p, cal, gauges, shots=1_000_000, qubits=None):
    """<run>/hw/<job> of a hw-run style manifest whose PUBs hold the counts that the confusion matrices of `cal` produce
    from the true distribution p in each PUB's gauge frame (exact expectation, rounded to `shots`)."""
    n = int(np.log2(p.size))
    qubits = qubits or cal["qubits"]
    job = run / "hw" / "01-01-2026-00-00_x"
    job.mkdir(parents=True)
    pubs, gl = [], []
    for g, mask in enumerate(gauges):
        gl.append({"mask": mask})
        q = apply_noise(p[np.arange(p.size) ^ mask], cal)             # the device runs the gauged circuit: p in the gauge frame
        c = np.round(q * shots).astype(np.int64)
        c[np.argmax(c)] += shots - c.sum()
        pubs.append({"k": g, "name": f"main_g{g}", "kind": "main", "gauge": g, "shots": shots})
        (job / f"pub_{g}_counts.json").write_text(json.dumps(
            {"counts": {str(i): int(v) for i, v in enumerate(c) if v}}))
    man = {"state": "done", "backend": cal["backend"], "job_id": "jobA", "setting": {"name": "f", "measure_twirl": cal["measure_twirl"]},
           "layout": {"measured_qubits": qubits}, "pubs": pubs, "gauges": gl}
    (job / "manifest.json").write_text(json.dumps(man))
    return job, man


def test_mitigation_recovers_the_true_distribution_in_bit_order_and_gauge_frame(tmp_path):
    from hsf2.hw.analysis import mitigated_weights
    n = 6
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(1 << n) * 0.5)
    cal = noisy_cal([11, 3, 8, 20, 5, 14], seed=1)
    gauges = [0, 0b010110, 0b101001]                                    # gauge 0 is the identity
    job, man = make_job_folder(tmp_path, p, cal, gauges)
    raw = {}
    from hsf2.hw.run import raw_counts
    raw = raw_counts(job, man)["main"]
    idx, w, info = mitigated_weights(tmp_path, {"job_id": "jobA"}, {str(k): v for k, v in raw.items()}, cal, prefer="numpy", method="m3")
    S = sum(raw.values())
    assert idx.size == (1 << n) or idx.size <= (1 << n)
    full = np.zeros(1 << n)
    full[idx] = w / S
    assert abs(w.sum() - S) < 1e-6 * S
    assert np.abs(full - p).max() < 2e-3                                 # the true p, not the raw noisy one
    assert np.abs(apply_noise(p, cal) - p).max() > 5e-3                  # the correction was worth something
    assert info["mitigation"] == "numpy" and info["backend"] == "fakeback" and info["n_pubs"] == 3 and info["overhead_mean"] >= 1.0
    # a wrong classical-bit order of the calibration is refused, never applied
    with pytest.raises(ValueError, match="differ"):
        mitigated_weights(tmp_path, {"job_id": "jobA"}, {str(k): v for k, v in raw.items()},
                          dict(cal, qubits=cal["qubits"][::-1]), prefer="numpy")
    # a bare calibration on twirled counts and counts that are not the job's are refused
    with pytest.raises(ValueError, match="never be mixed"):
        mitigated_weights(tmp_path, {"job_id": "jobA"}, {str(k): v for k, v in raw.items()}, dict(cal, measure_twirl=False))
    other = dict(raw)
    k0 = next(iter(other))
    other[k0] += 1
    with pytest.raises(ValueError, match="differ from the sample file"):
        mitigated_weights(tmp_path, {"job_id": "jobA"}, {str(k): v for k, v in other.items()}, cal, prefer="numpy", method="m3")
    with pytest.raises(FileNotFoundError):
        mitigated_weights(tmp_path, {"job_id": "nope"}, {str(k): v for k, v in raw.items()}, cal)


def test_weighted_estimators_accept_float_weights_and_the_identity_calibration_changes_nothing(tmp_path):
    from hsf2.hw.analysis import (bootstrap, correlation_fidelity, f_energy, f_xeb, mitigated_record, mitigated_weights, tail)
    from hsf2.hw.run import raw_counts
    n = 5
    rng = np.random.default_rng(5)
    E = rng.normal(size=1 << n)
    p = np.exp(-2.0 * (E - E.min()))
    p /= p.sum()
    ident = {"schema": 1, "session": "id", "backend": "fakeback", "measure_twirl": True, "qubits": list(range(n)),
             "matrices": {str(q): [[1.0, 0.0], [0.0, 1.0]] for q in range(n)}}
    job, man = make_job_folder(tmp_path, p, ident, [0], shots=2000)
    raw = raw_counts(job, man)["main"]
    idx, w, info = mitigated_weights(tmp_path, {"job_id": "jobA"}, {str(k): v for k, v in raw.items()}, ident, prefer="numpy", method="m3")
    ridx = np.array(sorted(raw), np.int64)
    rc = np.array([raw[int(i)] for i in ridx], np.int64)
    assert np.array_equal(idx, ridx) and np.allclose(w, rc)
    E_unif, e_ideal = float(E.mean()), float(p @ E)
    order = np.argsort(E, kind="stable")
    rank = np.empty(E.size, np.int32)
    rank[order] = np.arange(E.size, dtype=np.int32)
    sk_raw = correlation_fidelity(p, [(ridx, rc)], 3)[0]
    sk_mit = correlation_fidelity(p, [(idx, w)], 3)[0]
    rec = mitigated_record(E, p, rank, E_unif, e_ideal, idx, w, sk_mit, sk_raw, info, tmp_path / "cal.json", ident,
                           np.random.default_rng(0))
    assert rec["F_E"] == pytest.approx(f_energy(E, ridx, rc, E_unif, e_ideal), abs=1e-12)
    assert rec["F_XEB"] == pytest.approx(f_xeb(p, ridx, rc), abs=1e-12)
    assert [rec["s_k"][str(k)]["s"] for k in (1, 2, 3)] == pytest.approx([s for s, _ in sk_raw], abs=1e-12)
    assert [rec["s_k"][str(k)]["err"] for k in (1, 2, 3)] == [e for _, e in sk_raw]       # the standard error is the raw one
    assert rec["calibration_file"] == "cal.json" and rec["calibration_session"] == "id" and rec["calibration_backend"] == "fakeback"
    t_raw, t_mit = tail(rank, p, ridx, rc, 0.1), rec["tail"][2]
    assert t_mit["q"] == 0.1 and t_mit["p_hw"] == pytest.approx(t_raw["p_hw"], abs=1e-12)
    # a fractional tail weight is not truncated to an integer count
    w2 = rc.astype(float) * 0.5 + 0.25
    assert tail(rank, p, ridx, w2, 0.3)["p_hw"] == pytest.approx(w2[rank[ridx] < int(round(0.3 * E.size))].sum() / w2.sum())
    lo, hi = bootstrap(lambda i, x: f_xeb(p, i, x), idx, w, np.random.default_rng(1), 50)
    assert lo < hi


# ---------------------------------------------------------------------------------------- hw-analyze on a run folder
@pytest.fixture(scope="module")
def ladder_run(tmp_path_factory):
    """A 6-qubit Phase-1 run folder with one local job of setting d (K = 6) and a local twirled readout calibration."""
    from hsf2.hw.run import run_job, run_readout
    T = runner_tests()
    tmp = tmp_path_factory.mktemp("roa")
    run = T.make_small_run(tmp / "r")
    cfg = T.cfg_local(tmp, shots=T.SHOTS)
    job = run_job(run, "d", 6, cfg, command="test")
    cal = run_readout(run, job["job_dir"], cfg, command="test", shots=2000)
    return {"run": run, "job": job["job_dir"], "cal": cal["job_dir"] / "readout_cal.json", "tmp": tmp, "cfg": cfg}


def test_analyze_run_raw_unchanged_and_mitigated_next_to_it(ladder_run, tmp_path):
    from hsf2.hw.analysis import analyze_run, main_tags
    run, cal = ladder_run["run"], ladder_run["cal"]
    assert len(main_tags(run / "samples")) == 1
    plain = analyze_run(run, tmp_path / "plain")
    both = analyze_run(run, tmp_path / "both", readout=cal, prefer="numpy", readout_method="m3")
    assert "readout" not in plain["rows"][0] and plain["config"]["readout"] is None
    assert "F_E_mit" not in pd.read_csv(tmp_path / "plain" / "hw_analysis.csv").columns
    r0, r1 = plain["rows"][0], both["rows"][0]
    for k in ("F_E", "F_XEB", "E_hw", "s_k", "F_E_68", "tail", "cvar", "F_vs_control"):
        assert r0[k] == r1[k], k                                         # the raw values are exactly the ones without --readout
    ro = r1["readout"]
    assert ro["calibration_file"] == "readout_cal.json"
    assert ro["mitigation"] == "numpy" and ro["setting"] == "d" and ro["measure_twirl"] is True and ro["overhead_mean"] >= 1.0
    assert ro["calibration_session"] and ro["calibration_backend"] == ro["backend"] and ro["n_pubs"] == 1
    assert ro["F_E"] != r1["F_E"] and np.isfinite(ro["F_E"]) and np.isfinite(ro["F_XEB"])
    assert [ro["s_k"][str(k)]["err"] for k in (1, 2, 3)] == [r1["s_k"][str(k)]["err"] for k in (1, 2, 3)]
    assert sum(t["p_hw"] for t in ro["tail"]) > 0 and len(ro["tail"]) == len(r1["tail"])
    assert both["config"]["readout"]["tags"] == [r1["tag"]]
    csv = pd.read_csv(tmp_path / "both" / "hw_analysis.csv")
    for c in ("F_E", "F_E_mit", "F_XEB", "F_XEB_mit", "s1", "s1_mit", "s3_mit_err", "readout_calibration", "readout_mitigation",
              "E_hw_mit"):
        assert c in csv.columns, c
    assert csv["F_E_mit"][0] == pytest.approx(ro["F_E"]) and csv["F_E"][0] == pytest.approx(r1["F_E"])
    # the same correction as hw-readout --apply (run.apply_readout): the mean energy of the mitigated main samples
    from hsf2.hw.run import apply_readout
    res = apply_readout(ladder_run["job"], cal, "numpy", out_name="readout_check.json")
    assert ro["E_hw"] == pytest.approx(res["mean_energy"]["main_mitigated"], abs=1e-9)
    (ladder_run["job"] / "readout_check.json").unlink()


def test_readout_tags_select_and_fail_loudly(ladder_run, tmp_path):
    from hsf2.hw.analysis import analyze_run
    run, cal = ladder_run["run"], ladder_run["cal"]
    res = analyze_run(run, tmp_path / "a", readout=cal, readout_tags=r"_d_p1_top6_", prefer="numpy")
    assert "readout" in res["rows"][0] and res["config"]["readout"]["tags_regex"] == r"_d_p1_top6_"
    with pytest.raises(ValueError, match="matches none"):
        analyze_run(run, tmp_path / "b", readout=cal, readout_tags=r"_a_p1_")
    with pytest.raises(ValueError, match="needs --readout"):
        analyze_run(run, tmp_path / "c", readout_tags="x")
    with pytest.raises(ValueError, match="regular expression"):
        analyze_run(run, tmp_path / "d", readout=cal, readout_tags="(")
    with pytest.raises(FileNotFoundError):
        analyze_run(run, tmp_path / "e", readout=tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"qubits": [1]}))
    with pytest.raises(ValueError, match="not a readout calibration"):
        analyze_run(run, tmp_path / "f", readout=bad)
    bare = tmp_path / "bare.json"                                       # a bare calibration on twirled counts
    bare.write_text(json.dumps(dict(json.loads(cal.read_text()), measure_twirl=False)))
    with pytest.raises(ValueError, match="never be mixed"):
        analyze_run(run, tmp_path / "g", readout=bare)
    assert not (tmp_path / "g" / "hw_analysis.json").exists()


def test_cli_hw_analyze_readout_in_deployed_layout(ladder_run, tmp_path, monkeypatch):
    from hsf2.cli import main
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))
    run = tmp_path / "runs" / "r"
    shutil.copytree(ladder_run["run"], run)
    cal = next((run / "hw").glob("*readout*/readout_cal.json"))
    before = snapshot_tree(tmp_path, run)
    with pytest.raises(SystemExit, match="--readout-tags needs --readout"):
        main(["hw-analyze", "--run-dir", str(run), "--readout-tags", "x"])
    assert main(["hw-analyze", "--run-dir", str(run), "--readout", str(cal), "--prefer", "numpy"]) == 0
    outs = sorted((run / "outputs").glob("*_hw_analyze*"), key=lambda p: p.stat().st_mtime)
    new = [o for o in outs if (o / "hw_analysis.csv").exists() and "F_E_mit" in pd.read_csv(o / "hw_analysis.csv").columns]
    assert len(new) == 1 and (new[0] / "commands.jsonl").exists()
    assert "--readout" in json.loads((new[0] / "commands.jsonl").read_text().splitlines()[-1])["command"]
    assert snapshot_tree(tmp_path, run) == before                        # nothing outside the run folder


# ---------------------------------------------------------------------------------------- hw-components-analyze
@pytest.fixture(scope="module")
def comp_jobs(tmp_path_factory):
    """Two hand-made components (5 and 8 variables) as one local white-noise job of setting d."""
    from hsf2.hw.components import load_components, run_components
    C = component_tests()
    tmp = tmp_path_factory.mktemp("roc")
    run = C.write_run(tmp / "tri", C.HAND)
    comps = [c for c in load_components(run) if c.name in (f"{C.EVENT}_s0_f1_c1", f"{C.EVENT}_s0_f1_c2")]
    res = run_components(run, comps, "d", C.cfg_local(tmp, shots=2000), per_job=2, command="test", sim_method="white-noise", seed=3)
    return {"run": run, "job": res["jobs"][0]["job_dir"], "tmp": tmp, "names": [c.name for c in comps]}


def comp_cal(job_dir, extra_qubits=(150, 151), twirl=True):
    man = json.loads((job_dir / "manifest.json").read_text())
    qs = sorted({q for p in man["pubs"] for q in p["measured_qubits"]}) + list(extra_qubits)
    return noisy_cal(qs, seed=2, twirl=twirl, backend=man["backend"], scale=0.03)


def test_components_analysis_mitigated_next_to_raw_with_a_larger_calibration(comp_jobs, tmp_path):
    from hsf2.hw.comp_analysis import analyze_jobs
    job = comp_jobs["job"]
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(comp_cal(job)))
    plain = analyze_jobs([job], tmp_path / "p", seed=0, boot_n=50)
    mit = analyze_jobs([job], tmp_path / "m", seed=0, boot_n=50, readout=cal, prefer="numpy")
    assert all("readout" not in r for r in plain["records"])
    assert "F_E_mit" not in pd.read_csv(tmp_path / "p" / "components_hw.csv").columns
    for a, b in zip(plain["records"], mit["records"]):
        assert a["hardware"] == b["hardware"] and a["controls"] == b["controls"]      # raw values and controls unchanged
        ro, hm = b["readout"], b["hardware_mitigated"]
        assert ro["calibration_file"] == "cal.json" and ro["mitigation"] == "numpy" and ro["overhead"] >= 1.0
        assert ro["measured_qubits"] == json.loads((job / "manifest.json").read_text())["pubs"][b["pub"]]["measured_qubits"]
        assert set(ro["measured_qubits"]) <= set(ro["calibration_qubits"]) and len(ro["calibration_qubits"]) == len(set(ro["calibration_qubits"]))
        assert hm["shots"] == b["shots"] and np.isfinite(hm["F_E"]) and np.isfinite(hm["F_XEB"])
        assert hm["p_ground"] != b["hardware"]["p_ground"]
        assert b["p_hw_over_p_unif_mit"] == pytest.approx(hm["p_ground"] / b["p_unif_ground"])
    csv = pd.read_csv(tmp_path / "m" / "components_hw.csv")
    for c in ("F_E", "F_E_mit", "F_XEB_mit", "mit_p_ground", "p_hw_over_p_unif_mit", "readout_calibration", "readout_overhead"):
        assert c in csv.columns, c
    g = mit["summary"]["groups"]["all"]["readout_mitigated"]
    assert g["n_components"] == 2 and g["F_E_median"]["value"] is not None
    # a regex selects components; one that selects none, a twirl mismatch and a calibration without the qubits are refused
    only = analyze_jobs([job], tmp_path / "o", seed=0, boot_n=50, readout=cal, readout_tags=r"_c2$", prefer="numpy")
    assert [("readout" in r) for r in only["records"]] == [r["component"].endswith("_c2") for r in only["records"]]
    with pytest.raises(ValueError, match="matches none"):
        analyze_jobs([job], tmp_path / "x1", readout=cal, readout_tags="zzz")
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(comp_cal(job, twirl=False)))
    with pytest.raises(ValueError, match="never be mixed"):
        analyze_jobs([job], tmp_path / "x2", readout=bare)
    small = tmp_path / "small.json"
    c0 = comp_cal(job)
    drop = c0["qubits"][0]
    small.write_text(json.dumps(dict(c0, qubits=c0["qubits"][1:], matrices={k: v for k, v in c0["matrices"].items() if k != str(drop)})))
    with pytest.raises(ValueError, match="no matrix"):
        analyze_jobs([job], tmp_path / "x3", readout=small)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(dict(c0, backend="some_other_backend")))
    with pytest.raises(ValueError, match="backend"):
        analyze_jobs([job], tmp_path / "x4", readout=wrong)
    with pytest.raises(ValueError, match="needs --readout"):
        analyze_jobs([job], tmp_path / "x5", readout_tags="a")


def test_cli_components_analyze_readout_and_readout_calibration_of_a_components_job(comp_jobs, tmp_path, monkeypatch):
    from hsf2.cli import main
    from hsf2.hw.config import HwConfig
    from hsf2.hw.run import run_readout
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))
    run = tmp_path / "runs" / "tri"
    shutil.copytree(comp_jobs["run"], run)
    job = next((run / "hw").glob("*_comp_d_1"))
    # hw-readout --from-job works on a components job: its calibration covers the union of the PUBs' qubits
    cfg = HwConfig(mode="dry", shots=100, ledger=str(tmp_path / "l.json"))
    dry = run_readout(run, job, cfg, command="test")
    man = json.loads((job / "manifest.json").read_text())
    assert dry["manifest"]["qubits"] == sorted({q for p in man["pubs"] for q in p["measured_qubits"]})
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(comp_cal(job, extra_qubits=())))
    before = snapshot_tree(tmp_path, run)
    with pytest.raises(SystemExit, match="--readout-tags needs --readout"):
        main(["hw-components-analyze", str(job), "--readout-tags", "x"])
    assert main(["hw-components-analyze", str(job), "--readout", str(cal), "--prefer", "numpy", "--seed", "0"]) == 0
    outs = [o for o in (run / "outputs").glob("*_hw_components_analyze*") if (o / "components_hw.csv").exists()]
    assert len(outs) == 1 and "F_E_mit" in pd.read_csv(outs[0] / "components_hw.csv").columns
    assert snapshot_tree(tmp_path, run) == before                        # nothing outside the run folder


# ---------------------------------------------------------------------------- tensor-product inverse (lead, 12 Nov)
def _forward(p_true, mats):
    """p_obs = (M_{n-1} x ... x M_0) p_true with M_k[read, true] the matrix of bit k (bit k of the index)."""
    n = len(mats)
    t = np.asarray(p_true, float).reshape([2] * n)
    for k, m in enumerate(mats):
        a = n - 1 - k
        t = np.moveaxis(np.tensordot(m, t, axes=([1], [a])), 0, a)
    return t.reshape(-1)


def _mats(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a, b = rng.uniform(0.005, 0.06), rng.uniform(0.01, 0.12)        # P(1|0), P(0|1): asymmetric
        out.append(np.array([[1 - a, b], [a, 1 - b]]))
    return out


def test_tensor_inverse_recovers_the_true_distribution_exactly_and_in_bit_order():
    from hsf2.hw.readout import tensor_inverse
    n = 6
    mats = _mats(n, 1)
    p_true = np.random.default_rng(2).dirichlet(np.ones(1 << n) * 0.3)
    p_obs = _forward(p_true, mats)
    q = tensor_inverse({i: v * 1e9 for i, v in enumerate(p_obs)}, mats)
    assert abs(q.sum() - 1.0) < 1e-12 and np.abs(q - p_true).max() < 1e-12
    assert np.abs(tensor_inverse({i: v for i, v in enumerate(p_obs)}, mats[::-1]) - p_true).max() > 1e-4   # wrong order fails


def test_tensor_inverse_corrects_broad_samples_where_the_reduced_solve_cannot():
    """4 000 shots over 2^20 states (a broad sample, as on hardware): every corrected <Z_k> equals the one-qubit closed form
    (E_read - (b - a)) / (1 - a - b), while the reduced-subspace (M3-type) solve leaves <Z_k> at its read-out value,
    because almost no two observed strings are one flip apart."""
    from hsf2.hw.readout import correct_numpy, tensor_inverse
    n, S = 20, 4000
    mats = _mats(n, 3)
    rng = np.random.default_rng(4)
    bits = (rng.random((S, n)) < rng.uniform(0.2, 0.8, n)).astype(int)                        # true product state
    flip0, flip1 = np.array([m[1, 0] for m in mats]), np.array([m[0, 1] for m in mats])
    u = rng.random((S, n))
    read = np.where(bits == 0, (u < flip0).astype(int), (u >= flip1).astype(int))               # read-out errors
    idx = (read * (1 << np.arange(n))).sum(axis=1)
    counts = {int(k): int(v) for k, v in zip(*np.unique(idx, return_counts=True))}

    def z_means(w):
        t = np.asarray(w, float).reshape([2] * n)
        out = []
        for j in range(n):
            m = t.sum(axis=tuple(a for a in range(n) if a != n - 1 - j))
            out.append((m[0] - m[1]) / m.sum())
        return np.array(out)

    e_read = np.array([sum(c * (1 - 2 * ((k >> j) & 1)) for k, c in counts.items()) / S for j in range(n)])
    expected = (e_read - (flip1 - flip0)) / (1 - flip0 - flip1)
    q = tensor_inverse(counts, mats)
    assert np.abs(z_means(q) - expected).max() < 1e-10 and abs(q.sum() - 1) < 1e-12
    assert np.abs(expected - e_read).max() > 0.02                                             # the correction is visible
    m3, _ = correct_numpy(counts, mats)
    v = np.zeros(1 << n)
    for k, x in m3.items():
        v[k] = x
    frac = np.linalg.norm(z_means(v) - e_read) / np.linalg.norm(expected - e_read)
    assert frac < 0.1                       # reduced solve recovers only a few per cent of the correction (3.6 % here)

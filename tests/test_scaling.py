"""QX6 scaling study: mixed-bit QUBO, chunked exact scan, Aer QAOA against the Phase-1 statevector (synthetic)."""
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


def _fit(seed=0, n_points=300):
    """A quadratic surrogate of a smooth synthetic objective on the 8-knob unit box (Phase-1 fit_surrogate)."""
    from acts_qaoa.surrogate import fit_surrogate
    rng = np.random.default_rng(seed)
    U = rng.random((n_points, 8))
    c = rng.uniform(0.2, 0.8, 8)
    y = 0.9 - ((U - c) ** 2 * rng.uniform(0.05, 0.3, 8)).sum(1) + 0.05 * U[:, 0] * U[:, 1] - 0.03 * U[:, 2] * U[:, 5]
    return fit_surrogate(U, y + rng.normal(0, 1e-3, n_points), kind="quadratic", seed=0)


def test_bits_layout():
    from hsf2.tuning.scaling import bits_layout
    assert bits_layout(16) == [2] * 8 and bits_layout(24) == [3] * 8 and bits_layout(32) == [4] * 8
    assert bits_layout(20) == [3, 3, 3, 3, 2, 2, 2, 2]
    assert bits_layout(30) == [4, 4, 4, 4, 4, 3, 3, 4]       # max_degree, min_hits (knobs 5, 6) keep 3 bits
    for n in range(16, 33):
        assert sum(bits_layout(n)) == n
    with pytest.raises(ValueError):
        bits_layout(33)


def test_qubo_mixed_is_phase1_qubo_and_exact():
    from acts_qaoa.surrogate import qubo_from_surrogate
    from hsf2.tuning.scaling import bits_from_index, check_qubo, qubo_mixed, unit_from_index
    fit = _fit()
    for b in (2, 3):
        mine, p1 = qubo_mixed(fit, [b] * 8), qubo_from_surrogate(fit, b)
        assert mine.const == pytest.approx(p1.const, abs=1e-12)
        assert np.allclose(mine.L, p1.L, atol=1e-12) and np.allclose(mine.W, p1.W, atol=1e-12)
    bits = [2, 3, 4, 2, 3, 4, 2, 3]
    q = qubo_mixed(fit, bits)
    assert check_qubo(q, fit, bits) < 1e-10
    idx = np.array([0, (1 << 23) - 1, 12345, 999999])
    assert np.allclose(q.energy_bits(bits_from_index(idx, 23)), -fit.predict(unit_from_index(idx, bits)), atol=1e-10)
    # knob 1 (3 bits) sits on qubits 2..4: state 2^2 * 5 is level 5 of knob 1
    assert unit_from_index(5 << 2, bits)[1] == pytest.approx(5 / 7)


def test_chunked_scan_is_exact_enumeration():
    from acts_qaoa.landscape import enumerate_energies
    from acts_qaoa.surrogate import fit_surrogate
    from hsf2.tuning.scaling import ChunkedLandscape, qubo_mixed
    rng = np.random.default_rng(3)
    fit = fit_surrogate(rng.random((200, 7)), rng.random(200), kind="quadratic", seed=0)
    q = qubo_mixed(fit, [2, 2, 2, 2, 2, 2, 2])              # 14 qubits, random couplings
    E = enumerate_energies(q)
    land = ChunkedLandscape(q, chunk_bits=9)                 # 32 chunks of 512 states
    allE = np.concatenate([e for _, e in land.chunks()])
    assert np.allclose(allE, E, atol=1e-12)
    st = land.scan(k_low=50)
    assert st["argmin"] == int(np.argmin(E)) and st["e_min"] == pytest.approx(E.min(), abs=1e-12)
    assert st["e_max"] == pytest.approx(E.max(), abs=1e-12) and st["e_mean"] == pytest.approx(E.mean(), abs=1e-12)
    assert np.allclose(st["top_energy"], np.sort(E)[:50], atol=1e-12)
    v = [E[17], E[st["argmin"]], np.sort(E)[100]]
    assert land.count_below(v) == [int(np.sum(E < x - 1e-10)) for x in v]
    assert land.count_below([E.min()]) == [0]


def test_aer_qaoa_matches_phase1_statevector():
    pytest.importorskip("qiskit_aer")
    from acts_qaoa.landscape import enumerate_energies
    from acts_qaoa.qaoa_math import StatevectorQAOA, optimize_p1
    from acts_qaoa.surrogate import fit_surrogate
    from hsf2.tuning.scaling import aer_qaoa, qubo_mixed
    rng = np.random.default_rng(5)
    fit = fit_surrogate(rng.random((120, 5)), rng.random(120), kind="quadratic", seed=0)
    q = qubo_mixed(fit, [2] * 5)                              # 10 qubits
    ising = q.to_ising()
    scale = ising.scale()
    E = enumerate_energies(q)
    ang = optimize_p1(ising.h / scale, ising.full_J() / scale)
    p_ref = StatevectorQAOA((E - ising.offset) / scale, mixer="x").probs([ang["gamma"]], [ang["beta"]])
    a = aer_qaoa(ising, scale, ang["gamma"], ang["beta"], range(1 << 10), shots=4000, seed=1)
    assert np.allclose(a["p_watch"], p_ref, atol=2e-6)
    assert p_ref @ E == pytest.approx(ising.offset + scale * ang["energy"], abs=1e-9)    # closed form = statevector
    assert a["samples"].size == 4000 and a["samples"].max() < 1 << 10
    # Aer's samples follow the same distribution (loose chi-square-like check on the 20 likeliest states)
    top = np.argsort(p_ref)[::-1][:20]
    freq = np.array([np.mean(a["samples"] == i) for i in top])
    assert np.all(np.abs(freq - p_ref[top]) < 5 * np.sqrt(p_ref[top] / 4000) + 1e-3)


def test_run_size_and_study(tmp_path):
    pytest.importorskip("qiskit_aer")
    from hsf2.tuning.scaling import run_size, run_study
    fit = _fit(2)
    r = run_size(fit, 16, shots=256, seed=0, k_low=64, noise=False, noise_fidelity={"full": 0.5, "top": 0.8},
                 precision="double")
    assert r["bits"] == [2] * 8 and r["n_states"] == 1 << 16
    m = r["methods"]
    assert set(m) == {"random", "sa", "qaoa", "qaoa-top24", "qaoa-noise", "qaoa-top24-noise"}
    for d in m.values():
        assert 0 <= d["best_rank"] < r["n_states"] and 0.0 <= d["best_gap"] <= 1.0
        assert d["found_ground"] == (d["best_rank"] == 0)
    assert m["random"]["amplification"] == pytest.approx(1.0)
    q, qn = m["qaoa"], m["qaoa-noise"]
    assert qn["p_ground"] == pytest.approx(0.5 * q["p_ground"] + 0.5 / r["n_states"])
    assert q["approx_ratio"] > m["random"]["approx_ratio"]            # p=1 QAOA beats uniform on average
    # the study driver: resumable per size, a source of real evaluations is required
    src = tmp_path / "loop"
    src.mkdir()
    from hsf2.tuning.space import full_space
    from hsf2.utils import save_json
    rng = np.random.default_rng(0)
    sp = full_space()
    recs = []
    for _ in range(80):
        p = sp.from_unit(rng.random(8).tolist())
        recs.append({"params": p, "objective": float(-sum((v - 0.5) ** 2 for v in sp.to_unit(p)))})
    save_json(src / "results.json", {"results": [{"records": recs}]})
    out = tmp_path / "qx6"
    s = run_study(src, out, sizes=[16], shots=128, k_low=32, noise=False, precision="double")
    assert (out / "n16.json").exists() and (out / "sizes.csv").exists()
    assert s["surrogate"]["n_points"] == 80 and s["sizes"][16]["bits"] == [2] * 8
    t0 = (out / "n16.json").stat().st_mtime
    run_study(src, out, sizes=[16], shots=128, k_low=32, noise=False, precision="double")   # resumed
    assert (out / "n16.json").stat().st_mtime == t0
    with pytest.raises(RuntimeError):
        run_study(src, out, sizes=[16], shots=256, k_low=32, noise=False, precision="double")  # other shots


def test_single_precision_probe_runs_in_a_subprocess():
    pytest.importorskip("qiskit_aer")
    from hsf2.tuning.scaling import _PROBE, aer_memory_gb, single_precision_works
    _PROBE.clear()
    ok = single_precision_works()                 # False on the Windows build where single precision crashes
    assert isinstance(ok, bool) and _PROBE["single"] is ok
    assert aer_memory_gb(32, "single") == pytest.approx(34.36, abs=0.01)
    assert aer_memory_gb(32, "double") == pytest.approx(68.72, abs=0.01)


def test_resolve_source_qx2_skips_objective_variant_loops(tmp_path):
    """'qx2' must pick the newest loop scored with the standard S, never a newer t/20, DM or floor variant
    (QX6 n=32 of 9 Nov 2026 resolved 'qx2' at start time to the t/20 loop)."""
    import json
    import os
    from hsf2.tuning.scaling import resolve_source

    def loop(name, score_definition, mtime):
        d = tmp_path / name
        d.mkdir()
        cfg = {} if score_definition is None else {"score_definition": score_definition}
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (d / "comparison.json").write_text("{}", encoding="utf-8")
        os.utime(d / "comparison.json", (mtime, mtime))
        return d

    main = loop("02-11-2026-14-42_qx2_loop", None, 1_000)
    loop("04-11-2026-12-44_qx2_loop_t20", "eff - fake - dup/5 - t/20", 2_000)
    loop("04-11-2026-12-36_qx2_loop_dm", "eff_DM - fake - dup/5 - t/5", 3_000)
    assert resolve_source("qx2", tmp_path) == main
    newer = loop("05-11-2026-10-00_qx2_loop", "eff - fake - dup/5 - t/5", 4_000)
    assert resolve_source("qx2", tmp_path) == newer
    # an explicit folder is still accepted, whatever its objective
    (tmp_path / "04-11-2026-12-44_qx2_loop_t20" / "results.json").write_text("{}", encoding="utf-8")
    assert resolve_source(str(tmp_path / "04-11-2026-12-44_qx2_loop_t20"), tmp_path).name.endswith("_t20")

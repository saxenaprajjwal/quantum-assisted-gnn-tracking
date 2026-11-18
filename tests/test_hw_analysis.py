"""Hardware-sample analysis: Walsh transform, F_E / F_XEB / s_k estimators, CVaR bounds (synthetic, no run folder)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsf2.hw.analysis import (counts_arrays, correlation_fidelity, cvar, cvar_bracket, f_energy, f_xeb,  # noqa: E402
                              tail, walsh)

N_Q = 10


def _problem(n=N_Q, seed=0):
    """Random energies E and a peaked ideal distribution p (Boltzmann-like) on n qubits."""
    rng = np.random.default_rng(seed)
    N = 1 << n
    E = rng.normal(size=N)
    p = np.exp(-3.0 * (E - E.min()))
    p /= p.sum()
    return E, p


def _draw(p, shots, rng):
    c = rng.multinomial(shots, p)
    idx = np.nonzero(c)[0].astype(np.int64)
    return idx, c[idx].astype(np.int64)


def test_walsh_matches_hadamard_matrix_and_is_involution():
    n = 6
    H = np.array([[1.0]])
    for _ in range(n):
        H = np.block([[H, H], [H, -H]])
    v = np.random.default_rng(1).normal(size=1 << n)
    w = walsh(v.copy())
    assert np.allclose(w, H @ v)
    assert np.allclose(walsh(w.copy()), (1 << n) * v)
    with pytest.raises(ValueError):
        walsh(np.zeros(12))
    with pytest.raises(ValueError):
        walsh(np.zeros(8, np.float32))


def test_white_noise_recovers_fidelity():
    E, p = _problem()
    N = p.size
    F, shots = 0.6, 400_000
    mix = F * p + (1 - F) / N
    rng = np.random.default_rng(2)
    idx, c = _draw(mix, shots, rng)
    E_unif, E_ideal = float(E.mean()), float(p @ E)
    # standard error of F_E and F_XEB from the multinomial spread of the shots
    se_e = np.sqrt(np.var(np.repeat(E[idx], c)) / shots) / (E_unif - E_ideal)
    pv = np.repeat(p[idx], c)
    se_x = np.sqrt(np.var(pv) / shots) / (float(p @ p) - 1.0 / N)
    assert abs(f_energy(E, idx, c, E_unif, E_ideal) - F) < 4 * se_e
    assert abs(f_xeb(p, idx, c) - F) < 4 * se_x
    for k, (s, e) in enumerate(correlation_fidelity(p, [(idx, c)], 5)[0], start=1):
        assert e > 0 and abs(s - F) < 4 * e, (k, s, e)


def test_bit_flips_give_power_law_in_order():
    E, p = _problem(seed=3)
    eps, shots = 0.08, 600_000
    rng = np.random.default_rng(4)
    idx, c = _draw(p, shots, rng)
    x = np.repeat(idx, c)
    for b in range(N_Q):
        x = x ^ ((rng.random(x.size) < eps).astype(np.int64) << b)
    cnt = np.bincount(x, minlength=p.size)
    idx2 = np.nonzero(cnt)[0].astype(np.int64)
    for k, (s, e) in enumerate(correlation_fidelity(p, [(idx2, cnt[idx2].astype(np.int64))], 5)[0], start=1):
        assert abs(s - (1 - 2 * eps) ** k) < 4 * e, (k, s, e)


def test_cvar_values_and_bracket():
    E = np.array([3.0, 1.0, 2.0])
    c = np.array([1, 2, 1])                       # shots: 1,1,2,3
    assert cvar(E, c, 0.5) == pytest.approx(1.0)
    assert cvar(E, c, 0.75) == pytest.approx((1 + 1 + 2) / 3)
    assert cvar(E, c, 0.5, upper=True) == pytest.approx(2.5)
    assert cvar(E, c, 1.0) == pytest.approx(E @ c / c.sum())
    assert cvar_bracket(E, c, 1.3, 0.0)["inside"] is None
    with pytest.raises(ValueError):
        cvar(E, c, 0.0)


def test_cvar_bracket_contains_ideal_mean_for_white_noise():
    E, p = _problem(seed=5)
    N = p.size
    F = 0.5
    rng = np.random.default_rng(6)
    idx, c = _draw(F * p + (1 - F) / N, 200_000, rng)
    br = cvar_bracket(E[idx], c, F, float(p @ E))
    assert br["lower"] <= br["upper"] and br["inside"]


def test_tail_matches_definition():
    E, p = _problem(seed=7)
    order = np.argsort(E, kind="stable")
    rank = np.empty(p.size, np.int32)
    rank[order] = np.arange(p.size, dtype=np.int32)
    q, F = 0.05, 0.7
    idx, c = _draw(F * p + (1 - F) / p.size, 300_000, np.random.default_rng(8))
    t = tail(rank, p, idx, c, q)
    assert abs(t["tail_F"] - F) < 4 * t["tail_F_err"]


# ---------------------------------------------------------------------------- regression on the ibm_fez jobs
HW = ROOT / "results" / "hw_curve"
PROB = ROOT / "results" / "final" / "runs" / "02-11-2026-14-42_qx2_loop" / "seed0" / "qubo-qaoa-noise-top24"


@pytest.mark.skipif(not (HW / "noise_curve_analysis.json").exists() or not (HW / "samples").is_dir()
                    or not (PROB / "problem.json").exists(), reason="ibm_fez hw_curve results not present")
def test_regression_against_noise_curve_analysis(tmp_path):
    from hsf2._phase1 import find_phase1
    if find_phase1() is None:
        pytest.skip("Phase-1 package acts_qaoa not found")
    from hsf2.hw.analysis import analyze_run
    run = tmp_path / "run"
    shutil.copytree(HW / "samples", run / "samples")
    for name in ("run_config.json", "problem.json", "angles.json"):
        if (PROB / name).exists():
            shutil.copy2(PROB / name, run / name)
    res = analyze_run(run, tmp_path / "out", kmax=3)
    ref = {r["tag"]: r for r in json.loads((HW / "noise_curve_analysis.json").read_text())["rows"]}
    assert len(res["rows"]) == len(ref) == 5
    for r in res["rows"]:
        assert r["F_E"] == pytest.approx(ref[r["tag"]]["F_mean"], abs=1e-9)
        assert r["F_XEB"] == pytest.approx(ref[r["tag"]]["F_xeb"], abs=1e-9)
        assert set(r["s_k"]) == {"1", "2", "3"}
    assert (tmp_path / "out" / "hw_analysis.json").exists() and (tmp_path / "out" / "hw_analysis.csv").exists()

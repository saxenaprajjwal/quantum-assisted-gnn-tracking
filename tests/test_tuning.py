"""Tuning problem, Phase-1 integration and quantum conventions (synthetic data only)."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsf2._phase1 import ensure_phase1, find_phase1  # noqa: E402

pytestmark = pytest.mark.skipif(find_phase1() is None, reason="Phase-1 package acts_qaoa not found")
if find_phase1() is not None:
    ensure_phase1()                       # acts_qaoa importable even when a single test runs on its own


# ---------------------------------------------------------------------------- encoding
def test_full_encoding_roundtrip_and_default():
    from hsf2.config import TRACKING_DEFAULTS, TRACKING_RANGES
    from hsf2.tuning.space import default_levels, full_encoding, level_values
    enc = full_encoding()
    assert enc.n_qubits == 24 and all(enc.exact)
    rng = np.random.default_rng(0)
    for idx in rng.integers(0, 1 << 24, 500):
        p = enc.params_from_index(int(idx))
        assert enc.index_from_params(p) == idx
        assert isinstance(p["max_degree"], int) and isinstance(p["min_hits"], int)
    lv = level_values(enc)
    assert lv["max_degree"] == list(range(1, 9)) and lv["min_hits"] == list(range(3, 11))
    for n, (a, b) in TRACKING_RANGES.items():
        assert abs(lv[n][0] - a) < 1e-12 and abs(lv[n][-1] - b) < 1e-12
    assert default_levels(enc).tolist() == [7, 7, 7, 3, 7, 0, 0, 0]
    assert enc.params_from_levels(default_levels(enc)) == pytest.approx(TRACKING_DEFAULTS)


def test_bit_order_matches_phase1_and_qiskit():
    from hsf2.tuning.space import full_encoding
    enc = full_encoding()
    idx = int("101" + "000" * 6 + "011", 2)          # knob 7 = level 5, knob 0 = level 3 (lsb first)
    assert enc.levels_from_index(idx).tolist() == [3, 0, 0, 0, 0, 0, 0, 5]
    assert enc.key_to_index(enc.index_to_key(idx)) == idx
    bits = enc.bits_from_index(idx)
    assert bits[0] == 1 and bits[1] == 1 and bits[2] == 0 and bits[23] == 1     # qubit k = bit k of the index


def test_subgrid_is_embedded_in_full_grid():
    from hsf2.config import TRACKING_DEFAULTS
    from hsf2.tuning.phase1 import integer_levels_exact
    from hsf2.tuning.space import full_encoding, sub_encoding, sub_to_full_levels
    full = full_encoding()
    sub = sub_encoding(full)
    assert sub.n_qubits == 16 and integer_levels_exact(sub)
    for idx in range(0, 1 << 16, 97):
        p16 = sub.params_from_index(idx)
        p24 = full.params_from_levels(sub_to_full_levels(sub.levels_from_index(idx), full))
        for k in p16:
            assert abs(float(p16[k]) - float(p24[k])) < 1e-12
    assert sub.params_from_index(sub.index_from_params(TRACKING_DEFAULTS)) == pytest.approx(TRACKING_DEFAULTS)


def test_geometric_knobs_on_both_grids_and_in_run_config(tmp_path):
    """deta_scale and z0_scale (TRACKING_LOG) are spaced geometrically on the 24-bit grid and the QX1 sub-grid, with
    exact end points; the other knobs stay linear; the Phase-1 run_config.json records the spacing."""
    from hsf2.config import TRACKING_DEFAULTS, TRACKING_LOG, TRACKING_RANGES
    from hsf2.tuning.phase1 import make_run
    from hsf2.tuning.runner import LookupRunner
    from hsf2.tuning.space import KnobSpace, full_encoding, level_values, log_flags, sub_encoding
    from hsf2.utils import load_json
    full = full_encoding()
    sub = sub_encoding(full)
    lv, lv16 = level_values(full), level_values(sub)
    for n in TRACKING_LOG:
        a, b = TRACKING_RANGES[n]
        assert lv[n] == pytest.approx([a * (b / a) ** (l / 7) for l in range(8)], rel=1e-12)
        assert lv[n][0] == a and lv[n][-1] == b == TRACKING_DEFAULTS[n]
        assert lv16[n] == pytest.approx(lv[n][1::2], rel=1e-12) and lv16[n][-1] == b
    assert lv["dphi_scale"] == pytest.approx([0.3 + 0.1 * l for l in range(8)], abs=1e-12)
    u = np.linspace(0.05, 0.95, full.space.dim)                 # off the grid too: to_unit inverts from_unit
    back = full.space.to_unit(full.space.from_unit(u))
    assert all(abs(back[k] - u[k]) < 1e-12 for k in np.flatnonzero(log_flags(full.space)))
    assert KnobSpace.from_dict(full.space.to_dict()) == full.space
    make_run(tmp_path / "r", sub.space, sub.bits, LookupRunner(sub, lambda i: {}, n_events=1), 1, TRACKING_DEFAULTS)
    stored = load_json(tmp_path / "r" / "run_config.json")["space"]
    assert stored["log"] == [n in TRACKING_LOG for n in full.space.names]


def test_landscape_index_order():
    from hsf2.tuning.landscape import to_index_order
    from hsf2.tuning.space import full_encoding, sub_encoding
    enc = sub_encoding(full_encoding())
    t = np.zeros((4,) * 8 + (1,))
    for lv in itertools.product(range(4), repeat=8):
        t[lv] = enc.index_from_levels(np.array(lv))
    flat = to_index_order(t)[:, 0]
    assert np.array_equal(flat, np.arange(1 << 16))


# ---------------------------------------------------------------------------- objective variants (v0.2.4)
def test_score_variants_and_unchanged_default():
    """S keeps its pre-v0.2.4 dictionary (GNNRunner cache keys) and formula; S_DM, t/20 and the DM floor follow
    their definitions; the array form equals the scalar form."""
    from hsf2.config import ScoreConfig
    assert ScoreConfig().to_dict() == {"k_dup": 5.0, "k_time": 5.0, "time_mode": "proxy"}
    assert ScoreConfig().tag() == "" and ScoreConfig().is_default
    rng = np.random.default_rng(1)
    n = 200
    A = {"efficiency": rng.random(n), "efficiency_dm": rng.random(n), "fakerate": 0.1 * rng.random(n),
         "duplicaterate": 0.5 * rng.random(n), "time_proxy": rng.random(n)}
    variants = {"": ScoreConfig(), "dm": ScoreConfig(efficiency_key="efficiency_dm"),
                "t20": ScoreConfig(k_time=20.0), "floor0.9": ScoreConfig(dm_floor=0.9)}
    for tag, sc in variants.items():
        assert sc.tag() == tag
        phys, s = sc.score_arrays(A)
        for i in range(n):
            m = {k: float(v[i]) for k, v in A.items()}
            eff = m["efficiency_dm" if tag == "dm" else "efficiency"]
            ref = eff - m["fakerate"] - m["duplicaterate"] / 5 - m["time_proxy"] / (20 if tag == "t20" else 5)
            if tag == "floor0.9":
                ref -= 10.0 * max(0.0, 0.9 - m["efficiency_dm"])
            assert abs(sc.score(m) - ref) < 1e-12 and abs(s[i] - ref) < 1e-12
            assert abs(sc.physics(m) - phys[i]) < 1e-12
    with pytest.raises(ValueError):
        ScoreConfig(efficiency_key="trackml_score")
    assert ScoreConfig.from_dict(ScoreConfig(efficiency_key="efficiency_dm", dm_floor=0.99).to_dict()) == \
        ScoreConfig(efficiency_key="efficiency_dm", dm_floor=0.99)


def _fake_landscape(tmp_path):
    from hsf2.utils import save_json
    rng = np.random.default_rng(2)
    N = 1 << 16
    A = {"efficiency": 0.9 + 0.1 * rng.random(N), "fakerate": 0.01 * rng.random(N),
         "duplicaterate": 0.05 * rng.random(N), "time_proxy": rng.random(N), "trackml_score": rng.random(N),
         "n_tracks": rng.random(N)}
    A["efficiency_dm"] = A["efficiency"] - 0.1 * rng.random(N)
    A["physics"] = A["efficiency"] - A["fakerate"] - A["duplicaterate"] / 5
    A["score"] = A["physics"] - A["time_proxy"] / 5
    d = tmp_path / "land"
    d.mkdir()
    np.savez_compressed(d / "landscape.npz", **A)
    save_json(d / "spec.json", {"events": ["e1", "e2"]})
    return d, A


def test_landscape_rescored_and_objective_sensitivity(tmp_path):
    from hsf2.config import TRACKING_DEFAULTS, ScoreConfig
    from hsf2.tuning.landscape import LandscapeTable, objective_sensitivity
    from hsf2.tuning.runner import LookupRunner
    d, A = _fake_landscape(tmp_path)
    sc = ScoreConfig(efficiency_key="efficiency_dm", k_time=20.0)
    ref = A["efficiency_dm"] - A["fakerate"] - A["duplicaterate"] / 5 - A["time_proxy"] / 20
    tab = LandscapeTable(d, sc)
    assert np.allclose(tab.score, ref, atol=1e-15, rtol=0) and tab.best == ref.max()
    assert np.array_equal(LandscapeTable(d).score, A["score"])            # S: the stored array, untouched
    lr = LookupRunner(tab.enc, tab.metrics_of_index, 2, score=sc)
    i = int(np.argmax(ref))
    assert abs(lr.evaluate(tab.enc.params_from_index(i))["objective"] - ref[i]) < 1e-12
    res = objective_sensitivity(d)
    i_def = int(tab.enc.index_from_params(TRACKING_DEFAULTS))
    for name in ("S", "S_DM", "S, t/20", "S, DM floor 0.99"):
        o = res["objectives"][name]
        v = LandscapeTable(d, ScoreConfig.from_dict(o["score_config"])).score
        assert o["optimum"]["index"] == int(np.argmax(v)) and o["rank_of_default"] == int(np.sum(v > v[i_def]))
    ok = A["efficiency_dm"] >= A["efficiency_dm"][i_def] - 0.001
    assert abs(res["edge_reduction_at_no_physics_cost"]["efficiency_dm within 0.001"]["time_proxy"]
               - A["time_proxy"][ok].min()) < 1e-15


def test_comparison_records_and_checks_the_objective(tmp_path):
    from hsf2.config import TRACKING_DEFAULTS, ScoreConfig
    from hsf2.tuning.compare import run_comparison
    from hsf2.tuning.phase1 import TaggedView, make_run
    from hsf2.tuning.runner import LookupRunner
    from hsf2.utils import load_json
    d, _ = _fake_landscape(tmp_path)
    from hsf2.tuning.landscape import LandscapeTable
    sc = ScoreConfig(efficiency_key="efficiency_dm")
    tab = LandscapeTable(d, sc)
    base = LookupRunner(tab.enc, tab.metrics_of_index, 2, score=sc)
    with pytest.raises(ValueError):
        run_comparison(tmp_path / "c0", base, tab.enc, TRACKING_DEFAULTS, 2, [0], ["random-grid"], 5, 4, 2)
    s = run_comparison(tmp_path / "c", base, tab.enc, TRACKING_DEFAULTS, 2, [0], ["random-grid"], 5, 4, 2,
                       truth=lambda i: float(tab.score[i]), best_true=tab.best, score=sc)
    assert s["score_definition"] == sc.describe() and s["score_config"] == sc.to_dict()
    make_run(tmp_path / "r", tab.enc.space, tab.enc.bits, TaggedView(base), 2, TRACKING_DEFAULTS, score=sc)
    assert load_json(tmp_path / "r" / "hsf2_score.json")["score_config"] == sc.to_dict()


# ---------------------------------------------------------------------------- QUBO / Ising / QAOA conventions
def _random_qubo_from_surrogate(n_params=4, bits=2, seed=0):
    from acts_qaoa.surrogate import fit_surrogate, qubo_from_surrogate
    rng = np.random.default_rng(seed)
    U = rng.random((60, n_params))
    y = 1 - ((U - 0.3) ** 2).sum(1) + 0.3 * U[:, 0] * U[:, 1] + 0.01 * rng.normal(size=60)
    fit = fit_surrogate(U, y, kind="quadratic", repeats=2)
    return fit, qubo_from_surrogate(fit, bits)


def test_qubo_equals_minus_surrogate_and_ising():
    from acts_qaoa.encoding import Encoding
    from acts_qaoa.landscape import enumerate_energies
    from hsf2.tuning.phase1 import ising_to_qubo
    from hsf2.tuning.space import full_space
    fit, qubo = _random_qubo_from_surrogate()
    space = full_space()
    space4 = type(space)(space.names[:4], space.lo[:4], space.hi[:4], space.integer[:4])
    enc = Encoding(space4, 2)
    idx = np.arange(1 << qubo.n)
    E = enumerate_energies(qubo)
    assert np.allclose(E, -fit.predict(enc.unit_from_index(idx)), atol=1e-10)          # sign: minimise -score
    assert np.allclose(E, qubo.energy_bits(enc.bits_from_index(idx)), atol=1e-10)
    ising = qubo.to_ising()
    assert np.allclose(ising.energy_bits(enc.bits_from_index(idx)), E, atol=1e-10)
    back = ising_to_qubo(ising)
    assert np.allclose(enumerate_energies(back), E, atol=1e-10)
    # the sparsified model keeps the fields; the inverse is exact for it too
    sp = ising.sparsified(3)
    assert np.allclose(enumerate_energies(ising_to_qubo(sp)), sp.energy_bits(enc.bits_from_index(idx)), atol=1e-10)


def test_qaoa_closed_form_statevector_and_aer():
    from acts_qaoa.landscape import enumerate_energies
    from acts_qaoa.qaoa_math import StatevectorQAOA, optimize_p1, p1_energy
    _, qubo = _random_qubo_from_surrogate(n_params=3, bits=2, seed=1)
    ising = qubo.to_ising()
    scale = ising.scale()
    E = enumerate_energies(qubo)
    hd = (E - ising.offset) / scale
    r = optimize_p1(ising.h / scale, ising.full_J() / scale, n_gamma=201, n_beta=91)
    sv = StatevectorQAOA(hd)
    probs = sv.probs([r["gamma"]], [r["beta"]])
    assert abs(float(probs @ hd) - r["energy"]) < 1e-6
    assert abs(p1_energy(ising.h / scale, ising.full_J() / scale, r["gamma"], r["beta"]) - r["energy"]) < 1e-9
    # QAOA lowers the mean energy below uniform sampling (cost sign: low energy = high surrogate score)
    assert float(probs @ E) < float(E.mean())
    try:
        from acts_qaoa.circuits import aer_counts, bind, build_qaoa_circuit
    except ImportError:
        pytest.skip("qiskit not available")
    pytest.importorskip("qiskit_aer")
    h_t, j_t = ising.terms(scale)
    qc, gam, bet = build_qaoa_circuit(ising.n, h_t, j_t, p=1)
    shots = 20000
    counts = aer_counts(bind(qc, gam, bet, [r["gamma"]], [r["beta"]]), shots, seed=3)
    mu = float(probs @ E)
    sd = float(np.sqrt(probs @ (E - mu) ** 2))
    m_aer = sum(c * E[i] for i, c in counts.items()) / shots
    assert abs(m_aer - mu) / (sd / np.sqrt(shots)) < 5          # Qiskit key -> index convention and circuit agree


def test_fidelity_model_matches_measurements():
    from hsf2.tuning.phase1 import MEASURED_FIDELITY, fidelity_from_2q
    for n, f in MEASURED_FIDELITY:
        assert abs(fidelity_from_2q(n) - f) < 1e-12
    assert fidelity_from_2q(90) > fidelity_from_2q(500) > fidelity_from_2q(1164) > fidelity_from_2q(2000) > 0


def test_phase1_objective_scores_like_part2():
    from hsf2.config import ScoreConfig
    from hsf2.tuning.phase1 import objective_config
    rng = np.random.default_rng(0)
    sc = ScoreConfig()
    oc = objective_config(sc)
    for _ in range(50):
        m = {"efficiency": rng.random(), "fakerate": rng.random() * 0.1, "duplicaterate": rng.random() * 0.5,
             "time_proxy": rng.random()}
        m["runtime_reco_rel"] = m["time_proxy"]
        assert abs(oc.score(m) - sc.score(m)) < 1e-15 and abs(oc.physics(m) - sc.physics(m)) < 1e-15


# ---------------------------------------------------------------------------- closed loop on a known optimum
def test_closed_loop_finds_known_optimum_of_quadratic_objective(tmp_path):
    """A purely quadratic objective is fitted exactly, so the exact-QUBO loop must find its known optimum."""
    from hsf2.config import TRACKING_DEFAULTS
    from hsf2.tuning.phase1 import TaggedView, closed_loop, make_run
    from hsf2.tuning.runner import LookupRunner
    from hsf2.tuning.space import full_encoding, sub_encoding
    from acts_qaoa import pipeline as P1
    enc = sub_encoding(full_encoding())
    target = np.array([1, 2, 0, 3, 1, 2, 3, 0]) / 3.0          # optimum at these 2-bit levels

    def metrics(idx):
        u = enc.unit_from_index(idx)
        eff = 1.0 - float(((u - target) ** 2).sum()) + 0.2 * float(u[0] * u[1])
        return {"efficiency": eff, "fakerate": 0.0, "duplicaterate": 0.0, "time_proxy": 0.0}

    vals = np.array([metrics(i)["efficiency"] for i in range(1 << 16)])
    best = int(np.argmax(vals))
    base = LookupRunner(enc, metrics, n_events=1)
    view = TaggedView(base)
    run = make_run(tmp_path / "loop", enc.space, enc.bits, view, 1, TRACKING_DEFAULTS)
    P1.stage_warmstart(run, n_lhs=60, lhs_seed=5, noise_repeats=0)
    res = closed_loop(run, "exact", top_k=5, budget=70, seed=0)
    found = [enc.index_from_params(r["params"]) for r in view.records()]
    assert best in found and abs(res["best_objective"] - vals[best]) < 1e-12
    assert view.n_distinct == 70


def test_hardware_validation_with_controls(tmp_path):
    """hw-validate: uniform and ideal-circuit controls of the hardware sample size, the ideal mean energy equals
    the Phase-1 closed form for the sparsified circuit, and every method's best state is evaluated."""
    from acts_qaoa import pipeline as P1
    from acts_qaoa.qaoa_math import p1_energy
    from hsf2.config import TRACKING_DEFAULTS
    from hsf2.tuning.hardware import validate_hardware
    from hsf2.tuning.phase1 import make_run
    from hsf2.tuning.runner import LookupRunner
    from hsf2.tuning.space import full_encoding, sub_encoding
    from hsf2.utils import load_json
    enc = sub_encoding(full_encoding())
    target = np.array([1, 2, 0, 3, 1, 2, 3, 0]) / 3.0

    def metrics(idx):
        u = enc.unit_from_index(idx)
        eff = 1.0 - float(((u - target) ** 2).sum()) + 0.2 * float(u[0] * u[1])
        return {"efficiency": eff, "efficiency_dm": eff, "fakerate": 0.0, "duplicaterate": 0.0, "time_proxy": 0.0}

    base = LookupRunner(enc, metrics, n_events=1)
    run = make_run(tmp_path / "r", enc.space, enc.bits, base, 1, TRACKING_DEFAULTS)
    P1.stage_warmstart(run, n_lhs=60, lhs_seed=3, noise_repeats=0)
    P1.stage_build(run, surrogate="quadratic", fit_tags=("ws",), ansatz=("standard",), p_max=1)
    fit, qubo, ising, prob = run.load_problem()
    land = run.landscape(qubo)
    g, b = -0.4, 0.35
    hw = {int(i): 1 for i in np.random.default_rng(0).choice(1 << 16, 500, replace=False)}   # stand-in "hardware"
    run.save_samples("ibm-fake_standard_p1_top24_x_main", hw, {"shots": 500, "gammas": [g], "betas": [b]},
                     land.sample_metrics(hw))
    res = validate_hardware(run, tmp_path / "out", top_k=3, min_distance=0, reference=TRACKING_DEFAULTS)
    s = run.samples()
    assert {"uniform-500-s0", "ideal-top24-500-s0"} <= set(s)
    assert sum(s["uniform-500-s0"]["counts"].values()) == 500 == sum(s["ideal-top24-500-s0"]["counts"].values())
    scale = float(prob["scale"])
    ism = ising.sparsified(24)
    closed = ising.offset + scale * p1_energy(ism.h / scale, ism.full_J() / scale, g, b,
                                              h_obs=ising.h / scale, J_obs=ising.full_J() / scale)
    assert abs(s["ideal-top24-500-s0"]["metrics"]["exact_mean_energy"] - closed) < 1e-9 * max(1.0, abs(closed))
    rows = res["rows"]
    assert {"exact", "ibm-fake_standard_p1_top24_x_main", "uniform-500-s0", "ideal-top24-500-s0", "default"} <= set(rows)
    best_of_top3 = max(base.score({"status": "ok", "metrics": metrics(int(i))}) for i in land.top_k(3))
    assert abs(rows["exact"]["tune_objective"] - best_of_top3) < 1e-12
    assert load_json(tmp_path / "out" / "hw_validation.json")["hardware_tags"] == ["ibm-fake_standard_p1_top24_x_main"]


def test_baselines_respect_budget():
    from hsf2.tuning.baselines import BASELINES
    from hsf2.tuning.phase1 import TaggedView
    from hsf2.tuning.runner import LookupRunner
    from hsf2.tuning.space import full_encoding, sub_encoding
    enc = sub_encoding(full_encoding())
    base = LookupRunner(enc, lambda i: {"efficiency": (i % 977) / 977.0, "fakerate": 0.0, "duplicaterate": 0.0,
                                        "time_proxy": 0.0}, n_events=1)
    for name, fn in BASELINES.items():
        view = TaggedView(base)
        fn(view, enc, 40, 3, name)
        assert view.n_distinct == 40, name


# ---------------------------------------------------------------------------- real pipeline: grid evaluator, runner
@pytest.fixture(scope="module")
def toy_model(tmp_path_factory):
    import os
    from hsf2 import cli
    d = tmp_path_factory.mktemp("toy")
    old = {k: os.environ.get(k) for k in ("HSF2_DATA", "HSF2_MODELS")}
    os.environ["HSF2_DATA"] = str(d / "data")
    os.environ["HSF2_MODELS"] = str(d / "models")
    try:
        assert cli.main(["synth", "--n-events", "16", "--particles", "120", "--n-train", "8", "--n-val", "2",
                         "--n-tune", "2", "--n-confirm", "2", "--n-confirm-sets", "2", "--seed", "4"]) == 0
        assert cli.main(["train", "--data", "latest", "--level", "S1", "--epochs", "2", "--hidden", "16",
                         "--n-iter", "1", "--label", "t"]) == 0
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    yield next((d / "models").iterdir()), next((d / "data").iterdir())


def test_grid_evaluator_equals_reference(toy_model):
    from hsf2.config import TargetConfig, TrackingParams
    from hsf2.datasets import load_split
    from hsf2.model.train import EdgeModel
    from hsf2.tracking.metrics import COUNT_KEYS
    from hsf2.tracking.pipeline import GNNPipeline
    from hsf2.tuning.evaluator import POST_KNOBS, post_grid_counts
    model_dir, data_dir = toy_model
    pipe = GNNPipeline(EdgeModel.load(model_dir), TargetConfig())
    grid = {"cut_hi": [0.3, 0.7], "cut_lo_frac": [0.4, 1.0], "max_degree": [1, 4], "min_hits": [3, 6],
            "merge_overlap": [0.0, 0.4]}
    ev = load_split(data_dir, "tune", 1)[0]
    for filt in [(1.0, 1.0, 1.0), (0.5, 0.7, 0.6)]:
        base = TrackingParams(dphi_scale=filt[0], deta_scale=filt[1], z0_scale=filt[2])
        es, g = pipe.prepare(ev)
        keep, s, _ = pipe.scores(ev, base)
        fast = post_grid_counts(es, g, keep, s, pipe.targets, grid)
        for idx in itertools.product(*[range(2)] * 5):
            vals = {k: grid[k][i] for k, i in zip(POST_KNOBS, idx)}
            _, counts, _ = pipe.run_event(ev, TrackingParams(dphi_scale=filt[0], deta_scale=filt[1],
                                                             z0_scale=filt[2], **vals))
            assert np.array_equal(np.array([counts[k] for k in COUNT_KEYS], float), fast[idx]), (filt, vals)


def test_gnn_runner_cache_provenance_and_leakage(toy_model, tmp_path):
    from hsf2.config import TRACKING_DEFAULTS, TargetConfig
    from hsf2.tuning.runner import EVENT_SEED_STRIDE, GNNRunner, canonical_params
    from hsf2.tuning.space import full_encoding
    model_dir, data_dir = toy_model
    names = full_encoding().space.names
    r = GNNRunner(model_dir, data_dir, tmp_path / "run", names)
    a = r.evaluate(TRACKING_DEFAULTS, tag="a")
    b = r.evaluate(TRACKING_DEFAULTS, tag="b")
    assert b.get("cached") and a["objective"] == b["objective"] and a["key"] == b["key"]
    c = r.evaluate(TRACKING_DEFAULTS, tag="c", use_cache=False)
    assert c["objective"] == a["objective"]                     # deterministic re-evaluation
    conf = r.evaluate(TRACKING_DEFAULTS, tag="conf", seed=1 * EVENT_SEED_STRIDE)
    assert conf["split"] == "confirm1" and conf["key"] != a["key"]
    with pytest.raises(ValueError):
        r.evaluate(TRACKING_DEFAULTS, seed=12345)               # not an independent event-sample seed
    with pytest.raises(ValueError):
        canonical_params({k: v for k, v in TRACKING_DEFAULTS.items() if k != "cut_hi"}, names)
    with pytest.raises(ValueError):
        canonical_params({**TRACKING_DEFAULTS, "cut_high": 0.4}, names)
    # a different target definition must give a different key (no stale cache hits)
    r2 = GNNRunner(model_dir, data_dir, tmp_path / "run", names, targets=TargetConfig(pt_min=2.0))
    assert r2._key(canonical_params(TRACKING_DEFAULTS, names), 2, None) != a["key"]
    # the cache file is reloaded by a new runner and refused for another model
    r3 = GNNRunner(model_dir, data_dir, tmp_path / "run", names)
    assert a["key"] in r3.cache
    lines = (tmp_path / "run" / "eval_cache.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    rec["model_fp"] = "other"
    (tmp_path / "run2").mkdir()
    (tmp_path / "run2" / "eval_cache.jsonl").write_text(json.dumps(rec) + "\n")
    with pytest.raises(RuntimeError):
        GNNRunner(model_dir, data_dir, tmp_path / "run2", names)
    # tuning on training events is refused
    sp = json.loads((data_dir / "splits.json").read_text())
    bad = dict(sp, tune=sp["train"][:2])
    leak = tmp_path / "leakdata"
    leak.mkdir()
    for f in ("manifest.json",):
        (leak / f).write_text((data_dir / f).read_text())
    (leak / "splits.json").write_text(json.dumps(bad))
    (leak / "events").mkdir()
    for e in sp["train"][:2]:
        (leak / "events" / f"{e}.npz").write_bytes((data_dir / "events" / f"{e}.npz").read_bytes())
    with pytest.raises(RuntimeError):
        GNNRunner(model_dir, leak, tmp_path / "run3", names)


def test_training_is_deterministic(toy_model, tmp_path):
    from hsf2.config import GraphConfig, HitSelection, ModelConfig, TrainConfig
    from hsf2.data.event import select_hits
    from hsf2.datasets import load_split
    from hsf2.graph.construct import build_graph, learn_pair_map
    from hsf2.model.train import train_model
    _, data_dir = toy_model
    sel = HitSelection.preset("S1")
    tr = [select_hits(e, sel) for e in load_split(data_dir, "train", 3)]
    va = [select_hits(e, sel) for e in load_split(data_dir, "val", 1)]
    pm = learn_pair_map(tr, GraphConfig())
    out = []
    for _ in range(2):
        m, h = train_model([(e, build_graph(e, pm)) for e in tr], [(e, build_graph(e, pm)) for e in va],
                           ModelConfig(hidden=8, n_iter=1), TrainConfig(epochs=2, seed=7), pm, sel)
        out.append((m.net.state_dict(), h))
    for k in out[0][0]:
        assert np.array_equal(out[0][0][k].numpy(), out[1][0][k].numpy()), k
    assert [r["val_loss"] for r in out[0][1]] == [r["val_loss"] for r in out[1][1]]


def test_output_folders_are_unique(tmp_path):
    from hsf2.utils import new_run_dir, save_json
    a = new_run_dir(tmp_path, "x")
    b = new_run_dir(tmp_path, "x")
    assert a != b and a.exists() and b.exists()
    save_json(a / "r.json", {"v": 1})
    save_json(a / "r.json", {"v": 2})
    assert json.loads((a / "r.json").read_text())["v"] == 2 and len(list((a / "history").iterdir())) == 1


def test_worker_pool_equals_in_process(toy_model, tmp_path):
    """Event-parallel evaluation in spawned worker processes gives exactly the in-process result."""
    from hsf2.config import TRACKING_DEFAULTS
    from hsf2.tuning.runner import GNNRunner
    from hsf2.tuning.space import full_encoding
    model_dir, data_dir = toy_model
    names = full_encoding().space.names
    params = {**TRACKING_DEFAULTS, "max_degree": 3, "cut_hi": 0.6}
    one = GNNRunner(model_dir, data_dir, tmp_path / "a", names).evaluate(params)
    pool = GNNRunner(model_dir, data_dir, tmp_path / "b", names, workers=2)
    try:
        two = pool.evaluate(params)
    finally:
        pool.close()
    assert one["status"] == two["status"] == "ok"
    for k in ("efficiency", "fakerate", "duplicaterate", "time_proxy", "trackml_score"):
        assert one["metrics"][k] == two["metrics"][k], k


def test_deployed_layout_never_uses_phase1_folders(tmp_path, monkeypatch):
    """hsf2/ next to acts_qaoa/ (the lxplus Optimization folder): outputs go to hsf2_work/, Phase 1 is found."""
    import importlib
    import shutil
    import sys as _sys
    opt = tmp_path / "Optimization"
    (opt / "acts_qaoa").mkdir(parents=True)
    (opt / "DirectQAOAFeedback.py").write_text("")
    shutil.copytree(ROOT / "hsf2", opt / "hsf2", ignore=shutil.ignore_patterns("__pycache__"))
    for k in ("HSF2_WORK", "HSF2_DATA", "HSF2_MODELS", "HSF2_RUNS", "HSF2_FIGURES", "HSF_PHASE1_DIR"):
        monkeypatch.delenv(k, raising=False)
    code = ("import hsf2.datasets as d, hsf2._phase1 as p; print(d.data_root()); print(d.runs_root()); "
            "print(d.models_root()); print(d.figures_root()); print(p.candidates()[0])")
    import subprocess
    out = subprocess.run([_sys.executable, "-B", "-c", code], cwd=opt, capture_output=True, text=True, check=True)
    lines = [l.strip() for l in out.stdout.splitlines()]
    work = (opt / "hsf2_work").resolve()
    assert [Path(l).resolve() for l in lines[:4]] == [work / "data", work / "runs", work / "models", work / "figures"]
    assert Path(lines[4]).resolve() == opt.resolve()          # acts_qaoa next to hsf2 is found first


def test_paths_refuse_empty_relative_and_outside(tmp_path, monkeypatch):
    """Empty HSF2_* values fall back to the default; relative values and outputs outside hsf2_work are refused
    when deployed next to Phase 1 (they could land in the Phase-1 folder)."""
    import shutil
    import subprocess
    import sys as _sys
    opt = tmp_path / "Optimization"
    (opt / "acts_qaoa").mkdir(parents=True)
    (opt / "DirectQAOAFeedback.py").write_text("")
    shutil.copytree(ROOT / "hsf2", opt / "hsf2", ignore=shutil.ignore_patterns("__pycache__"))
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("HSF2_")}

    def run(code, **extra):
        return subprocess.run([_sys.executable, "-B", "-c", code], cwd=opt, capture_output=True, text=True,
                              env={**env, **extra})
    r = run("import hsf2.datasets as d; print(d.runs_root())", HSF2_RUNS="")
    assert r.returncode == 0 and Path(r.stdout.strip()).resolve() == (opt / "hsf2_work" / "runs").resolve()
    r = run("import hsf2.datasets as d; print(d.runs_root())", HSF2_RUNS="runs")
    assert r.returncode != 0 and "absolute" in r.stderr
    r = run("import hsf2.datasets as d; from pathlib import Path; d.ensure_inside_work(Path('runs/x'), '--out')")
    assert r.returncode != 0 and "outside" in r.stderr
    r = run("import hsf2.datasets as d; from pathlib import Path; print(d.ensure_inside_work(Path('hsf2_work/runs/x'), '--out'))")
    assert r.returncode == 0


def test_unique_dir_survives_eos_mkdir_race(tmp_path, monkeypatch):
    """Stage A eval_c1 (30-10-2026): 12 jobs created <model>/outputs at once; EOS answered EEXIST while this node's
    FUSE cache still reported the folder missing, so pathlib raised FileExistsError. Emulated here: another client
    creates the folder, this one is told EEXIST and sees it only on the third look."""
    import os
    import pathlib
    from hsf2.utils import unique_dir
    target = tmp_path / "outputs"
    real_mkdir, real_isdir = pathlib.Path.mkdir, os.path.isdir
    looks = {"n": 0}

    def racy_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if self == target:
            if not real_isdir(target):
                real_mkdir(target)                                  # the other client wins the race
            raise FileExistsError(17, "File exists", str(self))     # ... and pathlib's own check failed here
        return real_mkdir(self, mode, parents, exist_ok)

    def laggy_isdir(p):
        if Path(p) == target and looks["n"] < 2:
            looks["n"] += 1
            return False
        return real_isdir(p)
    monkeypatch.setattr(pathlib.Path, "mkdir", racy_mkdir)
    monkeypatch.setattr(os.path, "isdir", laggy_isdir)
    d = unique_dir(target, "evaluate_confirm1")
    assert d.is_dir() and d.parent == target and looks["n"] == 2

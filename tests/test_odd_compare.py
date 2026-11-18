"""ODD export -> EventData -> data folder -> train -> GNN versus CKF (hsf2.data.odd, hsf2.tracking.compare, CLI odd-prep and
compare-ckf, scripts/odd_export.py).  No ACTS here: the tests write a synthetic export in the format the exporter produces
(ACTS CSV writer layout) and check the reader, the geometry-id decoding, the CKF-state matching, the metric on a hand-made
case and on perfect tracks, the CLI chain on 5 tiny events, and the exporter's ACTS-facing code against fake ACTS modules.
Deployed-layout conventions as in test_hw_zne.py: HSF2_WORK is set, the script is found by _script() in scripts/ or
hsf2_scripts/."""
import ast
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsf2.data import odd  # noqa: E402
from hsf2.data.event import EventData  # noqa: E402


def _script(name: str = "odd_export.py") -> Path:
    """A script on the PC (scripts/) or as deployed on lxplus (hsf2_scripts/)."""
    for d in ("scripts", "hsf2_scripts"):
        p = ROOT / d / name
        if p.is_file():
            return p
    pytest.skip(f"{name} not found next to the tests")


# ---------------------------------------------------------------------------------------------- synthetic export
LAYERS = [(17, 2 * k) for k in range(1, 11)] + [(16, 2)]          # 10 barrel layers + one end-cap disc without hits
N_HI, N_LO = 25, 5


def build_export(root: Path, seed: int, n_events: int, with_surfaces: bool = True, export_id=None) -> dict:
    """Write an export folder in the layout of hsf2.data.odd; returns the truth it was built from (per event)."""
    rng = np.random.default_rng(1000 + seed)
    ex = Path(root)
    for d in (*odd.TRUTH_DIRS.values(), odd.TRACKS_DIR):
        (ex / d).mkdir(parents=True, exist_ok=True)
    if with_surfaces:
        with open(ex / odd.SURFACES_CSV, "w") as f:
            f.write("geometry_id,volume,layer,sensitive\n")
            for v, l in LAYERS:
                for s in range(1, 9):
                    f.write(f"{int(odd.geo_encode(volume=v, layer=l, sensitive=s))},{v},{l},{s}\n")
    truth = {}
    for nr in range(n_events):
        parts, hits = [], []
        for k in range(N_HI + N_LO):
            pt = rng.uniform(1.2, 6.0) if k < N_HI else rng.uniform(0.4, 0.9)
            pid = (1 << 52) | ((k + 1) << 28) | (nr + 1)                # > 2**32 like ACTS particle ids
            eta, phi0, q = rng.uniform(-2, 2), rng.uniform(-np.pi, np.pi), int(rng.choice([-1, 1]))
            parts.append((pid, pt, eta, phi0, q))
            radius = pt / 0.6 * 1000.0
            for (v, l) in LAYERS[:10]:
                r = 15.0 * l
                phi = phi0 + q * np.arcsin(r / (2 * radius))
                hits.append((pid, v, l, int(rng.integers(1, 9)), r * np.cos(phi), r * np.sin(phi), r * np.sinh(eta),
                             rng.uniform(-20, 20), rng.uniform(-20, 20)))
        n = len(hits)
        mid = rng.permutation(n)                                         # measurement id of generated hit j
        sidx = rng.permutation(n)                                        # simhit index of generated hit j
        with open(ex / odd.TRUTH_DIRS["simhits"] / f"event{nr:09d}-simhits.csv", "w") as f:
            f.write("particle_id,geometry_id,tx,ty,tz,tt,tpx,tpy,tpz,te,deltapx,deltapy,deltapz,deltae,index\n")
            for j in rng.permutation(n):                                 # file rows in arbitrary order
                pid, v, l, s, x, y, z, _, _ = hits[j]
                f.write(f"{pid},{int(odd.geo_encode(volume=v, layer=l, sensitive=s))},{x!r},{y!r},{z!r},0,1,1,1,1,0,0,0,0,{sidx[j]}\n")
        with open(ex / odd.TRUTH_DIRS["measurements"] / f"event{nr:09d}-measurements.csv", "w") as f:
            f.write("measurement_id,geometry_id,local_key,local0,local1,phi,theta,time\n")
            for j in rng.permutation(n):
                pid, v, l, s, x, y, z, a, b = hits[j]
                f.write(f"{mid[j]},{int(odd.geo_encode(volume=v, layer=l, sensitive=s))},3,{a!r},{b!r},0,0,0\n")
        with open(ex / odd.TRUTH_DIRS["measurements"] / f"event{nr:09d}-measurement-simhit-map.csv", "w") as f:
            f.write("measurement_id,hit_id\n")
            for j in range(n):
                f.write(f"{mid[j]},{sidx[j]}\n")
        with open(ex / odd.TRUTH_DIRS["particles"] / f"event{nr:09d}-particles.csv", "w") as f:
            f.write("particle_id,particle_type,process,vx,vy,vz,vt,px,py,pz,m,q\n")
            for pid, pt, eta, phi0, q in parts:
                f.write(f"{pid},13,0,0.0,0.0,0.5,0,{pt * np.cos(phi0)!r},{pt * np.sin(phi0)!r},{pt * np.sinh(eta)!r},0.105,{q}\n")
            for k in range(3):                                           # particles that left no hit
                f.write(f"{(1 << 52) | ((100 + k) << 28) | (nr + 1)},13,0,0,0,0,0,1,1,1,0.1,1\n")
        tracks = {}
        for j, h in enumerate(hits):
            tracks.setdefault(h[0], []).append((h[2], mid[j]))
        pt_of = {p[0]: p[1] for p in parts}
        good = [sorted(v) for pid, v in tracks.items() if pt_of[pid] >= 1.0]
        with open(ex / odd.TRACKS_DIR / f"event{nr:09d}-ckf_tracks.csv", "w") as f:
            f.write("track_id,measurement_id\n")
            for t, trk in enumerate(good):
                f.writelines(f"{t},{m}\n" for _, m in trk)
        truth[nr] = {"mid": mid, "x": np.array([h[4] for h in hits]), "y": np.array([h[5] for h in hits]),
                     "z": np.array([h[6] for h in hits]), "pid": np.array([h[0] for h in hits]),
                     "vol": np.array([h[1] for h in hits]), "layer": np.array([h[2] for h in hits]),
                     "sens": np.array([h[3] for h in hits]), "loc0": np.array([h[7] for h in hits]),
                     "loc1": np.array([h[8] for h in hits]), "parts": parts, "n_tracks": len(good)}
    (ex / odd.PERFORMANCE_JSON).write_text(json.dumps({"efficiency": 1.0, "fakerate": 0.0, "duplicaterate": 0.0}))
    (ex / odd.EXPORT_MANIFEST).write_text(json.dumps({
        "export_id": export_id or ex.name, "seed": seed, "events": list(range(n_events)), "status": "complete",
        "acts_version": "synthetic"}))
    return truth


@pytest.fixture(scope="module")
def export1(tmp_path_factory):
    d = tmp_path_factory.mktemp("exp") / "01-01-2026-00-00_s7"
    return d, build_export(d, 7, 3, with_surfaces=True)


# ---------------------------------------------------------------------------------------------- geometry id
def test_geometry_id_layout_and_roundtrip():
    masks = list(odd.GEO_MASKS.values())
    assert np.bitwise_or.reduce(np.array(masks, dtype=np.uint64)) == np.uint64(0xFFFFFFFFFFFFFFFF)
    for i, a in enumerate(masks):                                        # disjoint
        for b in masks[i + 1:]:
            assert a & b == 0
    g = odd.geo_encode(volume=17, layer=4, sensitive=12)
    assert int(g) == 0x1100004000000C00                                  # volume 17 << 56, layer 4 << 36, sensitive 12 << 8
    d = odd.geo_decode(g)
    assert (d["volume"][0], d["layer"][0], d["sensitive"][0], d["boundary"][0], d["approach"][0], d["extra"][0]) == \
        (17, 4, 12, 0, 0, 0)
    ids = np.array([odd.geo_encode(volume=v, layer=l, sensitive=s) for v, l, s in
                    [(2, 2, 1), (255, 4095, 1048575), (16, 14, 77)]], dtype=np.uint64)
    d = odd.geo_decode(ids)
    assert d["volume"].tolist() == [2, 255, 16] and d["layer"].tolist() == [2, 4095, 14]
    assert d["sensitive"].tolist() == [1, 1048575, 77]
    for bad in (dict(volume=256), dict(layer=4096), dict(sensitive=1 << 20), dict(layer=-1)):
        with pytest.raises(ValueError):
            odd.geo_encode(**bad)


def test_surfaces_table_overrides_bit_masks():
    g = odd.geo_encode(volume=17, layer=4, sensitive=12)
    tab = {"geometry_id": np.array([g], dtype=np.uint64), "volume": np.array([99]), "layer": np.array([98]),
           "sensitive": np.array([97])}                                  # ACTS says something else: the table wins
    d = odd.decode_geometry_ids(g, tab)
    assert (d["volume"][0], d["layer"][0], d["sensitive"][0]) == (99, 98, 97)
    with pytest.raises(KeyError):
        odd.decode_geometry_ids(odd.geo_encode(volume=3), tab)


# ---------------------------------------------------------------------------------------------- reader
def test_reader_roundtrip(export1):
    d, truth = export1
    for nr, t in truth.items():
        ev = odd.read_event(d, nr)
        assert ev.event_id == odd.event_id_of(7, nr) and ev.meta["geometry_decode"] == "surfaces.csv"
        order = np.argsort(t["mid"])
        h = ev.hits
        assert np.array_equal(h["hit_id"], t["mid"][order])
        for k in ("x", "y", "z"):
            assert np.allclose(h[k], t[k][order].astype(np.float32), rtol=0, atol=1e-6 * np.abs(t[k]).max())
        assert np.array_equal(h["particle_id"], t["pid"][order])
        assert np.array_equal(h["module"], t["sens"][order])
        assert np.all(h["weight"] == 0) and not np.any(h["particle_id"] == 0)         # no noise on the ODD
        index = odd.layer_index_from_surfaces(odd.read_surfaces(d))
        assert len(index) == len(LAYERS)
        assert np.array_equal(h["layer"], np.array([index[(v, l)] for v, l in zip(t["vol"], t["layer"])])[order])
        # particles: only those with hits; pT / eta / q / vertex as written; nhits = hits in the event
        p = ev.particles
        assert ev.n_particles == N_HI + N_LO
        for pid, pt, eta, phi0, q in t["parts"]:
            r = int(ev.particle_rows(np.array([pid]))[0])
            assert r >= 0
            assert abs(p["pt"][r] - pt) < 1e-5 and abs(p["eta"][r] - eta) < 1e-5 and p["q"][r] == q
            assert abs(p["vz"][r] - 0.5) < 1e-6 and p["nhits"][r] == 10
        # the CKF tracks reference measurement ids that exist in the event
        trk = odd.read_ckf_tracks(d, nr)
        assert len(trk) == t["n_tracks"] and all(len(x) == 10 for x in trk)
        assert set(np.concatenate(trk)) <= set(h["hit_id"].tolist())


def test_reader_fails_loudly(export1, tmp_path):
    d, _ = export1
    with pytest.raises(FileNotFoundError):
        odd.read_event(d, 99)
    with pytest.raises(FileNotFoundError):
        odd.load_manifest(tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        odd.read_event(d, 0, layer_index={(17, 2): 0})


def test_match_states_to_measurements(export1):
    d, truth = export1
    meas = odd.read_measurements(d, 0)
    surf = odd.read_surfaces(d)
    t = truth[0]
    pick = np.array([0, 5, 77, 140, 299])
    mid = t["mid"][pick]
    out = odd.match_states_to_measurements(meas, surf, t["vol"][pick], t["layer"][pick], t["sens"][pick],
                                           t["loc0"][pick] + 1e-6, t["loc1"][pick] - 1e-6)
    assert np.array_equal(out, mid)
    off = odd.match_states_to_measurements(meas, surf, t["vol"][pick], t["layer"][pick], t["sens"][pick],
                                           t["loc0"][pick] + 1.0, t["loc1"][pick])
    assert np.all(off == -1)
    nan1 = odd.match_states_to_measurements(meas, surf, t["vol"][pick], t["layer"][pick], t["sens"][pick],
                                            t["loc0"][pick], np.full(len(pick), np.nan))
    assert np.array_equal(nan1, mid)                                      # a NaN local coordinate is ignored
    wrong_surface = odd.match_states_to_measurements(meas, surf, t["vol"][pick], t["layer"][pick],
                                                     t["sens"][pick] + 20, t["loc0"][pick], t["loc1"][pick])
    assert np.all(wrong_surface == -1)


# ---------------------------------------------------------------------------------------------- metric
def hand_event():
    pids = [101, 102, 103, 104]
    pt = [3.0, 3.0, 3.0, 0.5]
    hits = {"x": np.arange(40, dtype=float), "y": np.zeros(40), "z": np.zeros(40), "layer": np.tile(np.arange(10), 4),
            "particle_id": np.repeat(pids, 10), "hit_id": np.arange(40)}
    par = {"particle_id": pids, "pt": pt, "eta": [0.1, 0.2, -0.3, 0.4], "nhits": [10] * 4}
    return EventData("hand", hits, par)


def test_metric_hand_made_case():
    from hsf2.tracking.compare import counts_for_tracks, ratios
    ev = hand_event()
    tracks = [np.arange(0, 10),                                           # A: p1, all hits           -> matched
              np.arange(0, 9),                                            # B: p1, 9 hits             -> matched, duplicate
              np.r_[10:14, 20:24, 30:32],                                 # C: 4 p2 + 4 p3 + 2 p4     -> purity 0.4: fake
              np.r_[10:16, 20:24],                                        # D: 6 p2 + 4 p3            -> purity 0.6: matched p2
              np.arange(30, 40)]                                          # E: p4 (pT 0.5, no target) -> matched
    c = counts_for_tracks(ev, tracks)
    assert c == {"n_tracks": 5, "n_matched": 4, "n_fake": 1, "n_duplicate": 1, "n_targets": 3, "n_targets_found": 2,
                 "n_matched_target": 3, "n_duplicate_target": 1,                     # A, B, D; B duplicates p1
                 "n_targets_found_dm": 2, "n_hits_target_tracks": 29}               # A (10/10), D (6 of p2's 10)
    r = ratios(c)
    assert r["efficiency"] == pytest.approx(2 / 3) and r["fakerate"] == pytest.approx(0.2)
    assert r["duplicaterate"] == pytest.approx(0.2) and r["duplicaterate_target"] == pytest.approx(1 / 3)
    assert r["efficiency_dm"] == pytest.approx(2 / 3)


def test_short_fragment_finds_a_target_but_not_by_double_majority():
    """A 3-hit pure fragment of p1 (10 hits) matches p1 (purity 1) but holds < half of its hits: efficiency counts
    it, the DM efficiency does not."""
    from hsf2.tracking.compare import counts_for_tracks, ratios
    ev = hand_event()
    c = counts_for_tracks(ev, [np.arange(0, 3), np.r_[10:16, 20:24]])             # p1 fragment, p2 majority track
    assert (c["n_targets_found"], c["n_targets_found_dm"], c["n_hits_target_tracks"]) == (2, 1, 13)
    assert ratios(c)["efficiency"] == pytest.approx(2 / 3) and ratios(c)["efficiency_dm"] == pytest.approx(1 / 3)
    with pytest.raises(IndexError):
        counts_for_tracks(ev, [np.array([0, 40])])


def test_duplicates_of_non_target_particles_count_only_in_the_all_track_rate():
    """A method that builds extra tracks of a pT < 1 GeV particle (p4) gets a higher ACTS duplicate rate, but its
    target duplicate rate is unchanged: the two methods are compared on the same particles."""
    from hsf2.tracking.compare import counts_for_tracks, ratios
    ev = hand_event()
    base = [np.arange(0, 10), np.r_[10:16, 20:24], np.arange(30, 40)]               # p1, p2, p4 once each
    loops = base + [np.arange(30, 39), np.arange(31, 40), np.arange(30, 38)]       # p4 three more times
    a, b = counts_for_tracks(ev, base), counts_for_tracks(ev, loops)
    assert (a["n_duplicate"], b["n_duplicate"]) == (0, 3)
    assert (a["n_duplicate_target"], b["n_duplicate_target"]) == (0, 0)
    assert a["n_matched_target"] == b["n_matched_target"] == 2
    assert ratios(b)["duplicaterate"] == pytest.approx(0.5) and ratios(b)["duplicaterate_target"] == 0.0


def test_metric_target_definition_is_phase1():
    """pT >= 1, |eta| < 3, >= 9 measurements: boundaries as the ACTS selector."""
    from hsf2.tracking.compare import phase1_targets
    ev = hand_event()
    ev.particles["pt"][:] = [1.0, 0.999, 3.0, 3.0]
    ev.particles["eta"][:] = [0.0, 0.0, 2.999, 3.0]
    assert phase1_targets(ev).tolist() == [True, False, True, False]
    h = hand_event()
    ev2 = EventData("h", {k: v[:-3] for k, v in h.hits.items()}, h.particles)       # p4 keeps 7 hits, p3 keeps 10 ...
    ev2.particles["pt"][:] = 3.0
    assert phase1_targets(ev2).tolist() == [True, True, True, False]               # ... p4 has < 9 measurements


def test_perfect_tracks_give_eff1_fake0_dup0(export1):
    from hsf2.tracking.compare import ckf_tracks_full, counts_for_tracks, ratios
    d, _ = export1
    ev = odd.read_event(d, 1)
    c = counts_for_tracks(ev, ckf_tracks_full(ev, d))
    assert c["n_targets"] == N_HI and c["n_tracks"] == N_HI
    r = ratios(c)
    assert (r["efficiency"], r["fakerate"], r["duplicaterate"]) == (1.0, 0.0, 0.0)


def test_statistics():
    from hsf2.tracking.compare import COUNT_COLS, paired_bootstrap, wilson
    lo, hi = wilson(0, 100)
    assert lo == 0.0 and 0.03 < hi < 0.04
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 0.01
    assert all(np.isnan(wilson(1, 0)))
    rng = np.random.default_rng(0)
    n_tr = rng.integers(40, 60, 30)
    a = np.zeros((30, len(COUNT_COLS)), dtype=np.int64)
    a[:, 0] = n_tr
    a[:, 1] = n_tr - 2
    a[:, 2] = 2
    a[:, 3] = 1
    a[:, 4] = 50
    a[:, 5] = 45
    a[:, COUNT_COLS.index("n_matched_target")] = 46
    a[:, COUNT_COLS.index("n_duplicate_target")] = 1
    a[:, COUNT_COLS.index("n_targets_found_dm")] = 44
    a[:, COUNT_COLS.index("n_hits_target_tracks")] = 460
    bt = paired_bootstrap(a, a.copy(), n_boot=300, seed=3)
    assert bt["duplicaterate_target"]["diff_a_minus_b"] == [0.0, 0.0]
    assert bt["efficiency"]["diff_a_minus_b"] == [0.0, 0.0]               # same events, same tracks: paired diff is zero
    lo, hi = bt["efficiency"]["a"]
    assert lo <= 0.9 <= hi
    b = a.copy()
    b[:, 5] = 25
    d = paired_bootstrap(a, b, n_boot=300, seed=3)["efficiency"]["diff_a_minus_b"]
    assert d[0] > 0.3                                                      # 0.9 - 0.5 = 0.4
    with pytest.raises(ValueError):
        paired_bootstrap(a, b[:5])
    assert paired_bootstrap(a, b, 200, 1) == paired_bootstrap(a, b, 200, 1)      # seeded


class _PerfectPipe:
    """Stands in for GNNPipeline: S0 hit selection, then one perfect track per particle of the reduced event, indices
    in the REDUCED event (as the real pipeline returns them); `shuffle` breaks the hit order of the reduced event."""
    def __init__(self, drop_low_pt_tracks=True):
        self.drop = drop_low_pt_tracks

    def run_event(self, ev, params):
        from hsf2.config import HitSelection
        from hsf2.data.event import select_hits
        es = select_hits(ev, HitSelection.preset("S0", 1.0))
        tracks = [np.nonzero(es.particle_id == p)[0] for p in np.unique(es.particle_id)]
        return tracks, {"t_gnn_s": 0.1, "t_build_s": 0.2, "t_graph_s": 0.3}, (es,)


def test_compare_events_maps_gnn_tracks_back_to_the_full_event(export1):
    from hsf2.tracking.compare import compare_events, export_map, summarize_comparison
    d, _ = export1
    events = [odd.read_event(d, nr) for nr in range(3)]
    res = compare_events(_PerfectPipe(), None, events, export_map([d]))
    s = summarize_comparison(res, n_boot=40, seed=1)
    for m in ("gnn", "ckf"):
        assert (s[m]["efficiency"], s[m]["fakerate"], s[m]["duplicaterate"]) == (1.0, 0.0, 0.0)
        assert s[m]["n_targets"] == 3 * N_HI and s[m]["n_events"] == 3
    assert s["bootstrap"]["efficiency"]["diff_a_minus_b"] == [0.0, 0.0]
    assert s["timing"]["gnn_infer_build_s_per_event"] == pytest.approx(0.3)
    # the S0 event really is smaller than the full one (the mapping is not an identity)
    assert _PerfectPipe().run_event(events[0], None)[2][0].n_hits < events[0].n_hits
    # a wrong event / export pairing is refused
    events[0].meta["export_id"] = "somewhere-else"
    with pytest.raises(KeyError, match="somewhere-else"):
        compare_events(_PerfectPipe(), None, events[:1], export_map([d]))


# ---------------------------------------------------------------------------------------------- CLI chain
def _last_line(capsys) -> str:
    return [l for l in capsys.readouterr().out.splitlines() if l.strip()][-1]


def test_odd_prep_train_and_compare(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HSF2_WORK", str(tmp_path))
    monkeypatch.setenv("HSF2_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("HSF2_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("HSF2_RUNS", str(tmp_path / "runs"))
    exp = tmp_path / "odd_export"
    e1, e2 = exp / "01-01-2026-00-00_s7", exp / "01-01-2026-00-01_s8"
    build_export(e1, 7, 3, with_surfaces=True)
    build_export(e2, 8, 2, with_surfaces=False)                              # bit-mask path, layer index from the events
    from hsf2.cli import main
    assert main(["odd-prep", "--export", str(e1), str(e2), "--label", "odd_t", "--n-train", "3", "--n-val", "1",
                 "--n-tune", "0", "--n-confirm", "1", "--n-confirm-sets", "1"]) == 0
    data = Path(_last_line(capsys))
    man = json.loads((data / "manifest.json").read_text())
    sp = json.loads((data / "splits.json").read_text())
    assert man["source"] == "odd" and man["n_events"] == 5 and man["position_source"] == "simhit_truth"
    assert [len(sp[k]) for k in ("train", "val", "tune", "confirm1")] == [3, 1, 0, 1]
    assert len(list((data / "events").glob("*.npz"))) == 5 and (data / "events.csv").is_file()
    assert sp["train"] == [odd.event_id_of(7, i) for i in range(3)]
    assert (data / "prep_config.json").is_file()
    with pytest.raises(SystemExit):                                          # more events than exported: loud
        main(["odd-prep", "--export", str(e1), "--n-train", "10", "--n-val", "0", "--n-tune", "0", "--n-confirm", "0"])
    capsys.readouterr()
    # the existing train command accepts the folder unchanged
    assert main(["train", "--data", str(data), "--epochs", "1", "--hidden", "8", "--n-iter", "1", "--label", "odd_t"]) == 0
    model = Path(_last_line(capsys))
    assert (model / "model.json").is_file()
    # GNN versus CKF on confirm1 (sorted ids: train = s7 e0-2, val = s8 e0, confirm1 = s8 e1, from the 2nd export)
    assert sp["val"] == [odd.event_id_of(8, 0)] and sp["confirm1"] == [odd.event_id_of(8, 1)]
    assert main(["compare-ckf", "--model", str(model), "--data", str(data), "--export", str(e1), str(e2),
                 "--split", "confirm1", "--n-boot", "50"]) == 0
    out = Path(_last_line(capsys))
    s = json.loads((out / "summary.json").read_text())
    assert s["ckf"]["efficiency"] == 1.0 and s["ckf"]["fakerate"] == 0.0 and s["ckf"]["duplicaterate"] == 0.0
    assert s["ckf"]["n_targets"] == s["gnn"]["n_targets"] == N_HI and s["ckf"]["n_tracks"] == N_HI
    assert 0.0 <= s["gnn"]["efficiency"] <= 1.0 and s["bootstrap"]["n_boot"] == 50
    assert any("truth-informed" in c for c in s["caveats"])                  # S0 model: stated
    cc = json.loads((out / "crosscheck.json").read_text())
    assert len(cc) == 2 and all(c["complete"] and abs(c["difference_python_minus_acts"]["efficiency"]) < 1e-12
                                and c["difference_python_minus_acts"]["fakerate"] == 0.0 for c in cc)
    assert (out / "config.json").is_file() and (out / "per_event.csv").is_file() and (out / "compare.txt").is_file()
    # a missing export for an event is an error, not a silent skip
    with pytest.raises(KeyError):
        main(["compare-ckf", "--model", str(model), "--data", str(data), "--export", str(e1), "--split", "confirm1",
              "--n-boot", "10", "--no-crosscheck"])


# ---------------------------------------------------------------------------------------------- exporter script
def _load_script():
    spec = importlib.util.spec_from_file_location("odd_export_under_test", _script())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_parses_and_help_lists_options():
    p = _script()
    ast.parse(p.read_text())
    r = subprocess.run([sys.executable, str(p), "--help"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    for opt in ("--events", "--seed", "--label", "--skip", "--check"):
        assert opt in r.stdout


def test_script_check_without_acts_fails_clearly():
    try:
        import acts  # noqa: F401
        pytest.skip("ACTS is installed here")
    except ImportError:
        pass
    r = subprocess.run([sys.executable, str(_script()), "--check"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "ACTS" in r.stdout


class _FakeSeq:
    def __init__(self, **kw):
        self.kw, self.added = kw, []


def _fake_acts(monkeypatch, writer_fields=None):
    acts = types.ModuleType("acts")
    ex = types.ModuleType("acts.examples")
    rec = types.ModuleType("acts.examples.reconstruction")
    ex.Sequencer = _FakeSeq
    calls = []
    rec.addCKFTracks = lambda s, *a, **kw: calls.append("ckf") or rec.kw_seen.append(kw) or "ckf-result"
    rec.kw_seen = []
    ex.reconstruction, acts.examples = rec, ex
    fields = writer_fields if writer_fields is not None else {
        "CsvSimHitWriter": ["inputSimHits", "outputDir", "outputStem", "level"],
        "CsvMeasurementWriter": ["inputMeasurements", "inputMeasurementSimHitsMap", "outputDir", "level"],
        "CsvParticleWriter": ["inputParticles", "outputDir", "outputStem", "outputPrecision", "level"]}
    for name, f in fields.items():
        cfg = type("Config", (), {k: None for k in f})
        setattr(ex, name, type(name, (), {"Config": cfg, "__init__": lambda self, **kw: setattr(self, "kw", kw)}))
    for n, m in (("acts", acts), ("acts.examples", ex), ("acts.examples.reconstruction", rec)):
        monkeypatch.setitem(sys.modules, n, m)
    return acts, ex, rec, calls


def test_exporter_writers_and_hooks_against_fake_acts(monkeypatch):
    mod = _load_script()
    acts, ex, rec, calls = _fake_acts(monkeypatch)
    mod.acts = acts
    # unknown optional fields are dropped, required ones kept
    w = mod.make_writer("CsvSimHitWriter", 3, inputSimHits="simhits", outputDir="o", outputStem="s", outputPrecision=12)
    assert w.kw == {"level": 3, "inputSimHits": "simhits", "outputDir": "o", "outputStem": "s"}
    # a changed field name fails loudly with the fields that exist
    acts2, ex2, rec2, _ = _fake_acts(monkeypatch, {"CsvSimHitWriter": ["input_sim_hits", "outputDir"]})
    with pytest.raises(mod.ExportError, match="input_sim_hits"):
        mod.make_writer("CsvSimHitWriter", 3, inputSimHits="simhits", outputDir="o")
    # a missing class is reported with the list of writers
    acts3, ex3, rec3, _ = _fake_acts(monkeypatch, {"RootParticleWriter": ["x"]})
    with pytest.raises(mod.ExportError, match="RootParticleWriter"):
        mod.make_writer("CsvParticleWriter", 3, inputParticles="p", outputDir="o")
    # hooks: skip is injected into the Sequencer, the writers are added after addCKFTracks, everything restored
    acts, ex, rec, calls = _fake_acts(monkeypatch)
    added = []
    restore = mod.install_hooks(4, lambda s: added.append((tuple(calls), s)), None)
    s = ex.Sequencer(events=2)
    assert s.kw == {"events": 2, "skip": 4}
    assert rec.addCKFTracks(s, "geo") == "ckf-result" and added == [(("ckf",), s)]
    assert rec.kw_seen == [{"writeTrackStates": True}]          # Phase 1 leaves it False: no trackstates_ckf.root
    restore()
    assert ex.Sequencer is _FakeSeq and ex.Sequencer(events=1).kw == {"events": 1}
    restore_noskip = mod.install_hooks(0, lambda s: None, None)
    assert "skip" not in ex.Sequencer(events=1).kw
    restore_noskip()


def test_exporter_collection_profiles(monkeypatch):
    mod = _load_script()
    profs = mod.collection_profiles({})
    assert profs[0] == {"initial": "particles_initial", "selected": "particles_selected",
                        "map": "measurement_simhits_map"} and len(profs) <= 6
    assert any(p["selected"] is None for p in profs)
    forced = mod.collection_profiles({"initial": "mine"})
    assert all(p["initial"] == "mine" for p in forced)


def test_exporter_geometry_check_detects_mismatch():
    mod = _load_script()
    g = np.array([odd.geo_encode(volume=17, layer=4, sensitive=12), odd.geo_encode(volume=8, layer=2, sensitive=5)],
                 dtype=np.uint64)
    tab = {"geometry_id": g, "volume": np.array([17, 8]), "layer": np.array([4, 2]), "sensitive": np.array([12, 5])}
    assert mod.check_geometry_bits(tab)["bit_layout_verified"] is True
    tab["layer"] = np.array([4, 3])
    res = mod.check_geometry_bits(tab)
    assert res["bit_layout_verified"] is False and res["mismatches"]["layer"] == 1


# ---------------------------------------------------------------------------------------------- ACTS versions
def _split_barcode_columns(path: Path) -> None:
    """Rewrite a simhit / particle CSV of the synthetic export in the layout of newer ACTS (lxplus build of 13 Nov
    2026): five barcode columns particle_id_pv, _sv, _part, _gen, _subpart instead of particle_id."""
    import pandas as pd
    df = pd.read_csv(path, dtype={"particle_id": np.uint64})
    pid = df.pop("particle_id").to_numpy(np.uint64)
    cols = {}
    for name, shift, bits in odd.BARCODE_FIELDS:
        cols[name] = ((pid >> np.uint64(shift)) & np.uint64((1 << bits) - 1)).astype(np.int64)
    pd.concat([pd.DataFrame(cols), df], axis=1).to_csv(path, index=False)


def test_split_barcode_columns_give_the_same_event(tmp_path):
    import shutil
    a = tmp_path / "01-01-2026-00-00_a"
    build_export(a, 11, 2)
    b = tmp_path / "01-01-2026-00-00_b"
    shutil.copytree(a, b)
    for sub in ("simhits", "particles"):
        for f in (b / odd.TRUTH_DIRS[sub]).glob("event*.csv"):
            _split_barcode_columns(f)
    head = (b / odd.TRUTH_DIRS["simhits"] / "event000000000-simhits.csv").read_text().splitlines()[0]
    assert head.startswith("particle_id_pv,particle_id_sv,particle_id_part,particle_id_gen,particle_id_subpart,")
    for nr in (0, 1):
        ea, eb = odd.read_event(a, nr), odd.read_event(b, nr)
        assert np.array_equal(ea.hits["particle_id"], eb.hits["particle_id"])
        assert np.array_equal(ea.hits["hit_id"], eb.hits["hit_id"])
        for k in ("particle_id", "pt", "eta", "nhits"):
            assert np.array_equal(ea.particles[k], eb.particles[k]), k
    # a component wider than its field cannot be packed
    import pandas as pd
    bad = pd.DataFrame({c: [1] for c, _, _ in odd.BARCODE_FIELDS})
    bad["particle_id_gen"] = [256]
    with pytest.raises(ValueError, match="particle_id_gen"):
        odd.particle_ids(bad, "x.csv")
    with pytest.raises(ValueError, match="no particle id"):
        odd.particle_ids(pd.DataFrame({"geometry_id": [1]}), "x.csv")


def test_incomplete_exports_are_refused(tmp_path):
    from hsf2.tracking import compare as cmp
    d = tmp_path / "01-01-2026-00-00_run"
    build_export(d, 12, 2)
    man = json.loads((d / odd.EXPORT_MANIFEST).read_text())
    assert odd.require_complete(d)["status"] == "complete" and cmp.export_map([d])
    man["status"] = "running"                                  # a job removed while it ran (13 Nov 2026)
    (d / odd.EXPORT_MANIFEST).write_text(json.dumps(man))
    with pytest.raises(ValueError, match="not 'complete'"):
        odd.prep_from_exports([d], tmp_path / "out", 2)
    with pytest.raises(ValueError, match="not 'complete'"):
        cmp.export_map([d])


def test_simhit_rows_in_container_order_with_trajectory_index(tmp_path):
    """Current ACTS: the measurement-simhit map's hit_id is the row of the simhit file (container order) and the
    `index` column is the hit's index along its particle's trajectory (not unique); found on lxplus, 13 Nov 2026."""
    import shutil
    import pandas as pd
    a = tmp_path / "01-01-2026-00-00_a"
    build_export(a, 13, 2)
    b = tmp_path / "01-01-2026-00-00_b"
    shutil.copytree(a, b)
    for f in (b / odd.TRUTH_DIRS["simhits"]).glob("event*.csv"):
        df = pd.read_csv(f, dtype={"particle_id": np.uint64})
        df = df.sort_values("index", kind="stable").reset_index(drop=True)      # row k = container index k
        df["index"] = df.groupby("particle_id").cumcount()                     # along-trajectory index
        df.to_csv(f, index=False)
    for nr in (0, 1):
        ea, eb = odd.read_event(a, nr), odd.read_event(b, nr)
        assert ea.meta["simhit_map"] == "index column" and eb.meta["simhit_map"] == "row order"
        for k in ("hit_id", "particle_id", "x", "y", "z", "layer"):
            assert np.array_equal(ea.hits[k], eb.hits[k]), k
        assert np.array_equal(ea.particles["particle_id"], eb.particles["particle_id"])
    # a map that fits neither the row order nor a unique index column is refused
    sh = pd.DataFrame({"geometry_id": np.array([5, 6, 7], dtype=np.uint64), "index": [0, 0, 1]})
    with pytest.raises(ValueError, match="does not fit"):
        odd.simhit_rows(sh, np.array([0, 1, 2]), np.array([6, 5, 7], dtype=np.uint64), 0)


def test_crosscheck_with_acts_selected_particle_counts(tmp_path):
    """Real exports record ACTS's selected-particle count per event under post.n_selected_particles; the cross-check
    must report it next to its own target count (it raised UnboundLocalError before v0.2.15)."""
    from hsf2.tracking import compare as cmp
    d = tmp_path / "01-01-2026-00-00_cc"
    build_export(d, 14, 2)
    man = json.loads((d / odd.EXPORT_MANIFEST).read_text())
    man["post"] = {"n_selected_particles": {"0": 25, "1": 24}}
    (d / odd.EXPORT_MANIFEST).write_text(json.dumps(man))
    cc = cmp.crosscheck_export(d)
    assert cc["n_selected_particles_acts"] == 49 and "n_targets_python" in cc
    assert cc["acts"]["efficiency"] == 1.0 and cc["complete"] is True
    assert abs(cc["difference_python_minus_acts"]["fakerate"]) < 1e-12


def _helix_points(pt, eta, d0=0.0, b=2.0, n=12, phi0=0.3, q=1, rmax=1000.0):
    """Points of a helix from near the beam line: radius R = pT / (0.2998 B) (mm), a d0 offset of the circle."""
    R = pt / (0.299792458e-3 * b)
    # circle centre at distance R + d0 from the origin, so the closest approach to the beam line is d0
    cx, cy = (R + d0) * np.cos(phi0 + q * np.pi / 2), (R + d0) * np.sin(phi0 + q * np.pi / 2)
    a0 = np.arctan2(-cy, -cx)
    out = []
    for k in range(1, n + 1):
        rho_target = rmax * k / n
        # walk along the circle until the transverse radius reaches rho_target
        lo, hi = 0.0, np.pi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            a = a0 - q * mid
            if np.hypot(cx + R * np.cos(a), cy + R * np.sin(a)) < rho_target:
                lo = mid
            else:
                hi = mid
        a = a0 - q * lo
        x, y = cx + R * np.cos(a), cy + R * np.sin(a)
        s = R * lo
        out.append((x, y, s * np.sinh(eta)))
    return np.array(out)


def test_helix_parameters_recover_pt_d0_eta():
    from hsf2.tracking.compare import helix_parameters
    for pt, eta, d0 in ((1.0, 0.0, 0.0), (2.5, 1.2, 1.5), (5.0, -2.0, 0.5), (1.2, 2.8, 3.0)):
        p = _helix_points(pt, eta, d0)
        h = helix_parameters(p[:, 0], p[:, 1], p[:, 2])
        assert h["pt"] == pytest.approx(pt, rel=1e-3) and h["d0"] == pytest.approx(d0, abs=1e-3)
        assert h["eta"] == pytest.approx(eta, abs=0.01)
    assert np.isnan(helix_parameters([1, 2], [1, 2], [0, 1])["pt"])


def test_select_tracks_applies_the_ckf_cuts_to_any_track_list():
    from hsf2.data.event import EventData
    from hsf2.tracking.compare import CKF_SELECTION, select_tracks
    good = _helix_points(2.0, 0.5)                       # passes
    low = _helix_points(0.6, 0.5)                        # pT < 1 GeV
    far = _helix_points(2.0, 0.5, d0=8.0)                # |d0| > 4 mm
    fwd = _helix_points(2.0, 3.3)                        # |eta| > 3
    pts = np.concatenate([good, low, far, fwd])
    n = len(pts)
    ev = EventData("h", {"hit_id": np.arange(n), "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
                         "layer": np.tile(np.arange(12), 4), "module": np.zeros(n, int),
                         "particle_id": np.repeat([1, 2, 3, 4], 12), "weight": np.zeros(n)},
                   {"particle_id": [1, 2, 3, 4], "pt": [2.0, 0.6, 2.0, 2.0], "eta": [0.5, 0.5, 0.5, 3.3],
                    "nhits": [12] * 4})
    tracks = [np.arange(k * 12, k * 12 + 12) for k in range(4)] + [np.arange(0, 6)]      # + a 6-hit piece of 'good'
    kept = select_tracks(ev, tracks, CKF_SELECTION)
    assert len(kept) == 1 and kept[0].tolist() == list(range(12))


def test_surface_frames_place_measurements_on_the_plane(tmp_path):
    """fit_surface_frames on a synthetic tilted pixel module (2-D, small smearing) and a 1-D strip module: positions
    O + loc0 u + loc1 v reproduce the simhits within the smearing; the 1-D module puts hits on its centre line."""
    from hsf2.data import odd
    rng = np.random.default_rng(1)
    rows = []
    for gid, two_d, s0, s1, L in ((101, True, 0.015, 0.015, 30.0), (202, False, 0.07, 0.0, 80.0)):
        n = np.array([np.cos(0.4), np.sin(0.4), 0.2]); n /= np.linalg.norm(n)
        u = np.cross([0, 0, 1.0], n); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        c = 300.0 * n + np.array([0, 0, 50.0 * gid / 100])
        for _ in range(60):
            a, b = rng.uniform(-20, 20), rng.uniform(-L, L)
            g = c + a * u + b * v
            rows.append((gid, a + rng.normal(0, s0), (b + rng.normal(0, s1)) if two_d else 0.0, s1 ** 2, g))
    # feed the private collector through a monkeypatched event reader
    import types
    pairs = {"geometry_id": np.array([r[0] for r in rows], dtype=np.uint64), "loc0": np.array([r[1] for r in rows]),
             "loc1": np.array([r[2] for r in rows]), "var1": np.array([r[3] for r in rows]),
             "g": np.array([r[4] for r in rows])}
    orig = odd._event_pairs
    odd._event_pairs = lambda ex, nr: pairs
    try:
        fr = odd.fit_surface_frames([(tmp_path, 0)])
    finally:
        odd._event_pairs = orig
    assert fr["geometry_id"].tolist() == [101, 202] and fr["two_d"].tolist() == [True, False]
    xyz, missing = odd.measured_positions(fr, pairs["geometry_id"], pairs["loc0"], pairs["loc1"])
    assert not missing.any()
    d = xyz - pairs["g"]
    pix = pairs["geometry_id"] == 101
    assert np.sqrt(np.mean(np.sum(d[pix] ** 2, axis=1))) < 0.05             # ~ sqrt(2) * 0.015 mm
    strip = ~pix
    n2 = np.cross(fr["u"][1], fr["v"][1])
    assert np.max(np.abs(d[strip] @ n2)) < 1e-6                             # on the plane
    assert np.sqrt(np.mean((d[strip] @ fr["u"][1]) ** 2)) < 0.15            # across the strip: the smearing
    assert np.max(np.abs(d[strip] @ fr["v"][1])) <= 80.0 + 1.0              # along the strip: the centre line
    _, miss = odd.measured_positions(fr, np.array([999], dtype=np.uint64), [0.0], [0.0])
    assert miss.tolist() == [True]


def test_select_tracks_fit_mask_leaves_out_unmeasured_coordinates():
    """A track whose outer hits are displaced along their strips (one-coordinate hits on the strip centre line) fails
    the cuts when they enter the refit and passes when the fit mask leaves them out; they still count as hits."""
    from hsf2.data.event import EventData
    from hsf2.tracking.compare import CKF_SELECTION, select_tracks
    pts = _helix_points(1.1, 0.5).copy()                  # pT just above the 1 GeV cut
    outer = np.arange(8, 12)
    rho = np.hypot(pts[outer, 0], pts[outer, 1])
    pts[outer, 0] *= (rho - 60.0) / rho                   # moved 60 mm inwards along a radial strip
    pts[outer, 1] *= (rho - 60.0) / rho
    n = len(pts)
    ev = EventData("m", {"hit_id": np.arange(n), "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
                         "layer": np.arange(n), "module": np.zeros(n, int), "particle_id": np.ones(n, int),
                         "weight": np.zeros(n)},
                   {"particle_id": [1], "pt": [1.1], "eta": [0.5], "nhits": [n]})
    t = [np.arange(n)]
    assert select_tracks(ev, t, CKF_SELECTION) == []
    mask = ~np.isin(ev.hits["layer"], outer)
    assert len(select_tracks(ev, t, CKF_SELECTION, fit_mask=mask)) == 1
    few = np.zeros(n, bool)
    few[:2] = True                                        # fewer than 3 hits to fit: not selected
    assert select_tracks(ev, t, CKF_SELECTION, fit_mask=few) == []


def test_one_coordinate_layers_from_the_frames():
    from hsf2.data import odd
    layers = [[16, 2], [16, 4], [28, 2]]
    gid = np.array([odd.geo_encode(volume=16, layer=2, sensitive=1), odd.geo_encode(volume=16, layer=2, sensitive=2),
                    odd.geo_encode(volume=16, layer=4, sensitive=1), odd.geo_encode(volume=28, layer=2, sensitive=1),
                    odd.geo_encode(volume=28, layer=2, sensitive=2)], dtype=np.uint64)
    frames = {"geometry_id": gid, "two_d": np.array([True, False, True, False, False]),
              "n_hits": np.array([50, 1, 40, 30, 0])}
    assert odd.one_coordinate_layers(frames, layers) == [2]   # one 1-hit surface does not make layer 0 one-coordinate

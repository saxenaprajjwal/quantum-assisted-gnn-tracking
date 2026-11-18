"""python -m hsf2 selftest: checks every stage on the synthetic detector (no data, no network, ~1 min)."""
from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np


def _toy(n_events: int, start: int = 0, particles: int = 300):
    from .data.synthetic import ToyConfig, generate_events
    return generate_events(n_events, ToyConfig(n_particles=particles), start=start)


def test_event_io() -> str:
    from .data.event import EventData
    ev = _toy(1)[0]
    with tempfile.TemporaryDirectory() as d:
        p = ev.save(Path(d) / "e.npz")
        e2 = EventData.load(p)
    assert e2.event_id == ev.event_id and e2.n_hits == ev.n_hits
    assert np.array_equal(e2.particle_id, ev.particle_id) and np.allclose(e2.x, ev.x)
    assert e2.meta["source"] == "synthetic"
    return f"{ev.n_hits} hits round-trip"


def test_graph_construction() -> str:
    from .config import GraphConfig, HitSelection
    from .data.event import select_hits
    from .graph.construct import build_graph, learn_pair_map
    sel = HitSelection.preset("S1")
    tr = [select_hits(e, sel) for e in _toy(20)]
    pm = learn_pair_map(tr, GraphConfig())
    ev = select_hits(_toy(1, start=900)[0], sel)
    g = build_graph(ev, pm, truth_pt_min=1.0)
    assert g.meta["truth_edge_eff"] > 0.98, g.meta
    assert np.all(ev.rho[g.src] <= ev.rho[g.dst]), "edges must point away from the beam"
    from .data.event import pair_keys
    assert len(np.unique(pair_keys(g.src, g.dst, ev.n_hits))) == g.n_edges, "duplicate edges"
    return f"{pm.n_pairs} layer pairs, truth-edge eff {g.meta['truth_edge_eff']:.4f}, purity {g.meta['edge_purity']:.3f}"


def test_perfect_scores() -> str:
    """Truth (consecutive-hit) labels as edge scores -> every target particle found, no fakes/duplicates."""
    from .config import GraphConfig, HitSelection, TargetConfig, TrackingParams
    from .data.event import select_hits
    from .graph.construct import build_graph, learn_pair_map
    from .tracking.build import build_tracks
    from .tracking.metrics import event_counts, summarize
    sel = HitSelection.preset("S1")
    pm = learn_pair_map([select_hits(e, sel) for e in _toy(20)], GraphConfig())
    rows = []
    for ev in _toy(3, start=900):
        es = select_hits(ev, sel)
        g = build_graph(es, pm)
        tracks, _ = build_tracks(es.n_hits, g.src, g.dst, g.y.astype(float), TrackingParams())
        rows.append(event_counts(es, tracks, TargetConfig()))
    m = summarize(rows)
    assert m["efficiency"] > 0.97 and m["fakerate"] == 0.0 and m["duplicaterate"] < 0.01, m
    return f"eff {m['efficiency']:.4f} fake {m['fakerate']:.4f} dup {m['duplicaterate']:.4f}"


def test_matching_rules() -> str:
    """Hand-made event: majority matching, fake, duplicate and double-majority definitions."""
    from .config import TargetConfig
    from .data.event import EventData
    from .tracking.metrics import event_counts, match_tracks
    pid = np.array([1] * 6 + [2] * 6 + [0] * 2)
    n = len(pid)
    hits = {"x": np.arange(n, dtype=float) + 1, "y": np.zeros(n), "z": np.zeros(n), "layer": np.arange(n),
            "particle_id": pid, "weight": np.where(pid > 0, 1.0 / 12, 0.0)}
    parts = {"particle_id": np.array([1, 2]), "pt": np.array([2.0, 2.0]), "eta": np.zeros(2), "nhits": np.array([6, 6])}
    ev = EventData("hand", hits, parts)
    tracks = [np.array([0, 1, 2, 3]), np.array([4, 5, 12]), np.array([6, 7, 8, 9, 10, 11]), np.array([12, 13, 0])]
    m = match_tracks(ev, tracks, 0.5)
    assert m["matched"].tolist() == [True, True, True, False], m
    assert m["dm"].tolist() == [True, False, True, False], m
    c = event_counts(ev, tracks, TargetConfig(min_hits=3))
    assert c["n_targets_found"] == 2 and c["n_fake"] == 1 and c["n_duplicate"] == 1, c
    assert abs(c["trackml_score_raw"] - 10 / 12) < 1e-6, c     # 4 + 6 hits of weight 1/12 (float32)
    return "matched/fake/duplicate/double-majority/TrackML score as defined"


def test_junction_removal() -> str:
    from .tracking.build import junction_mask
    rng = np.random.default_rng(1)
    src = rng.integers(0, 50, 400)
    dst = src + rng.integers(1, 5, 400)
    w = rng.random(400)
    for k in (1, 2, 3):
        keep = junction_mask(60, src, dst, w, k)
        assert np.bincount(src[keep], minlength=60).max() <= k and np.bincount(dst[keep], minlength=60).max() <= k
        # the best outgoing edge of a node survives if it is also among the best k incoming of its target
    return "degree <= k for k = 1, 2, 3"


def test_training_and_tracking() -> str:
    import torch
    from .config import GraphConfig, HitSelection, ModelConfig, TargetConfig, TrackingParams, TrainConfig
    from .data.event import select_hits
    from .graph.construct import build_graph, learn_pair_map
    from .model.train import EdgeModel, train_model
    from .tracking.pipeline import GNNPipeline
    torch.manual_seed(0)
    sel = HitSelection.preset("S1")
    tr = [select_hits(e, sel) for e in _toy(16)]
    va = [select_hits(e, sel) for e in _toy(2, start=500)]
    pm = learn_pair_map(tr, GraphConfig())
    model, hist = train_model([(e, build_graph(e, pm)) for e in tr], [(e, build_graph(e, pm)) for e in va],
                              ModelConfig(hidden=24, n_iter=2), TrainConfig(epochs=5, lr=3e-3), pm, sel)
    assert hist[-1]["auc"] > 0.95, hist[-1]
    with tempfile.TemporaryDirectory() as d:
        model.save(Path(d))
        m2 = EdgeModel.load(Path(d))
    pipe = GNNPipeline(m2, TargetConfig())
    test = _toy(3, start=1000)
    m = pipe.evaluate(test, TrackingParams())
    assert m["efficiency"] > 0.85, m
    m_loose = pipe.evaluate(test, TrackingParams(dphi_scale=0.5))
    assert m_loose["time_proxy"] < 0.9 and pipe.n_inference == 6
    pipe.evaluate(test, TrackingParams(cut_hi=0.7))         # same filters -> cached scores, no new inference
    assert pipe.n_inference == 6
    return f"AUC {hist[-1]['auc']:.4f}; tracks eff {m['efficiency']:.3f} fake {m['fakerate']:.3f} dup {m['duplicaterate']:.3f}"


def test_trackml_frames() -> str:
    import pandas as pd
    from .data.trackml import frames_to_event
    hits = pd.DataFrame({"hit_id": [1, 2, 3], "x": [32.0, 72.0, 116.0], "y": [0.0, 1.0, 2.0], "z": [1.0, 2.0, 3.0],
                         "volume_id": [8, 8, 8], "layer_id": [2, 4, 6], "module_id": [1, 2, 3]})
    truth = pd.DataFrame({"hit_id": [1, 2, 3], "particle_id": [11, 11, 0], "weight": [0.5, 0.5, 0.0]})
    parts = pd.DataFrame({"particle_id": [11], "vx": [0.0], "vy": [0.0], "vz": [0.0], "px": [1.0], "py": [1.0],
                          "pz": [0.0], "q": [1], "nhits": [2]})
    ev = frames_to_event("event000000001", hits, truth, parts)
    assert ev.layer.tolist() == [7, 8, 9] and abs(ev.particles["pt"][0] - np.sqrt(2)) < 1e-6
    return "hits/truth/particles -> EventData, layer index of volume 8 = 7..10"


TESTS: List[Tuple[str, Callable[[], str]]] = [
    ("event save/load", test_event_io),
    ("graph construction (module-map-lite)", test_graph_construction),
    ("perfect edge scores -> perfect tracks", test_perfect_scores),
    ("matching and metric definitions", test_matching_rules),
    ("junction removal", test_junction_removal),
    ("TrackML csv -> EventData", test_trackml_frames),
    ("IN training + track building + caches", test_training_and_tracking),
]


def run_selftest(quick: bool = False) -> bool:
    ok = True
    t_all = time.time()
    for name, fn in TESTS:
        if quick and fn is test_training_and_tracking:
            continue
        t0 = time.time()
        try:
            msg = fn()
            print(f"PASS  {name:42s} {time.time() - t0:5.1f} s  {msg}")
        except Exception as exc:
            ok = False
            print(f"FAIL  {name:42s} {time.time() - t0:5.1f} s  {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"{'ALL PASSED' if ok else 'SOME TESTS FAILED'} ({time.time() - t_all:.1f} s)")
    return ok

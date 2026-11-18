"""pytest tests for HSF Part II (synthetic data only):   python -m pytest tests -q"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsf2.selftest import TESTS  # noqa: E402


@pytest.mark.parametrize("name,fn", TESTS, ids=[t[0] for t in TESTS])
def test_selftest(name, fn):
    fn()


def test_walkthrough_and_ambiguity():
    from hsf2.config import TrackingParams
    from hsf2.tracking.build import build_candidates, conflict_pairs, resolve_ambiguity
    # 0 -> 1 -> 2 -> 3 -> 4, branch 2 -> 5 -> 6 (scores: main path 0.9, branch 0.8)
    src = np.array([0, 1, 2, 3, 2, 5])
    dst = np.array([1, 2, 3, 4, 5, 6])
    w = np.array([0.9, 0.9, 0.9, 0.9, 0.8, 0.8])
    # max_degree 1: junction removal drops the weaker branch -> one chain of 5 hits
    c1 = build_candidates(7, src, dst, w, TrackingParams(cut_hi=0.5, max_degree=1, min_hits=3))
    assert sorted(len(t) for t in c1.tracks) == [5]
    # max_degree 2, branching allowed (0.8 >= cut_hi): two paths sharing hits 0, 1, 2
    p = TrackingParams(cut_hi=0.7, max_degree=2, min_hits=3)
    c2 = build_candidates(7, src, dst, w, p)
    assert sorted(len(t) for t in c2.tracks) == [5, 5]
    i, j, n = conflict_pairs(c2.tracks, 7)
    assert list(n) == [3]
    acc0, _ = resolve_ambiguity(c2, 7, 0.0)
    acc1, _ = resolve_ambiguity(c2, 7, 0.6)
    assert len(acc0) == 1 and len(acc1) == 2
    # branch below cut_hi: only the best edge is followed -> one path
    c3 = build_candidates(7, src, dst, w, TrackingParams(cut_hi=0.85, cut_lo_frac=0.5, max_degree=2, min_hits=3))
    assert sorted(len(t) for t in c3.tracks) == [5]


def test_filter_mask_monotone():
    from hsf2.config import GraphConfig, HitSelection
    from hsf2.data.event import select_hits
    from hsf2.data.synthetic import ToyConfig, generate_events
    from hsf2.graph.construct import build_graph, filter_mask, learn_pair_map
    sel = HitSelection.preset("S1")
    evs = [select_hits(e, sel) for e in generate_events(12, ToyConfig(n_particles=200))]
    pm = learn_pair_map(evs[:10], GraphConfig())
    g = build_graph(evs[11], pm)
    n = [filter_mask(g, pm, s, s, s).sum() for s in (1.0, 0.8, 0.6, 0.4)]
    assert n[0] == g.n_edges and all(a >= b for a, b in zip(n, n[1:]))


def test_cli_end_to_end(tmp_path, monkeypatch):
    """synth -> train -> evaluate -> scan with the folder conventions (DD-MM-YYYY-HH-MM_<label>, outputs/)."""
    from hsf2 import cli
    monkeypatch.setenv("HSF2_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("HSF2_MODELS", str(tmp_path / "models"))
    assert cli.main(["synth", "--n-events", "14", "--particles", "150", "--n-train", "8", "--n-val", "2",
                     "--n-tune", "2", "--n-confirm", "1", "--n-confirm-sets", "2"]) == 0
    dd = list((tmp_path / "data").iterdir())
    assert len(dd) == 1 and dd[0].name.endswith("_toy") and (dd[0] / "manifest.json").exists()
    assert cli.main(["train", "--data", "latest", "--level", "S1", "--epochs", "2", "--hidden", "16",
                     "--n-iter", "1", "--label", "t"]) == 0
    md = list((tmp_path / "models").iterdir())
    assert len(md) == 1 and (md[0] / "model.json").exists() and (md[0] / "train_config.json").exists()
    assert cli.main(["evaluate", "--model", "latest", "--split", "tune", "--set", "cut_hi=0.4"]) == 0
    assert cli.main(["scan", "--model", "latest", "--split", "tune", "--param", "min_hits", "--values", "3,5"]) == 0
    outs = sorted(p.name for p in (md[0] / "outputs").iterdir())
    assert any(o.endswith("_evaluate_tune") for o in outs) and any(o.endswith("_scan_min_hits") for o in outs)
    ev_dir = [p for p in (md[0] / "outputs").iterdir() if p.name.endswith("_evaluate_tune")][0]
    cfg = json.loads((ev_dir / "config.json").read_text())
    assert cfg["tracking"]["cut_hi"] == 0.4 and (ev_dir / "metrics.json").exists()

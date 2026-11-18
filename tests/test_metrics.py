"""Metric definitions on hand-made events with answers computed by hand (no data needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsf2.config import ScoreConfig, SplitConfig, TargetConfig  # noqa: E402
from hsf2.data.event import EventData, pair_keys, select_hits, truth_edges  # noqa: E402
from hsf2.tracking.metrics import (event_counts, match_tracks, summarize, trackml_score,  # noqa: E402
                                   trackml_track_weights)


def hand_event():
    """Particles 1 (5 hits), 2 (4 hits, pT 0.5), 3 (6 hits) and 3 noise hits; weights 1 per real hit."""
    pid = np.array([1] * 5 + [2] * 4 + [3] * 6 + [0] * 3)
    n = len(pid)
    hits = {"x": np.arange(n, dtype=float) + 10, "y": np.zeros(n), "z": np.zeros(n), "layer": np.arange(n),
            "particle_id": pid, "weight": (pid > 0).astype(float)}
    parts = {"particle_id": np.array([3, 1, 2]), "pt": np.array([3.0, 2.0, 0.5]), "eta": np.array([0.1, 0.0, 0.0]),
             "nhits": np.array([6, 5, 4])}              # deliberately unsorted: EventData sorts by particle_id
    return EventData("hand", hits, parts)


def test_particle_table_sorted_and_rows():
    ev = hand_event()
    assert ev.particles["particle_id"].tolist() == [1, 2, 3]
    assert ev.particles["pt"].tolist() == [2.0, 0.5, 3.0]
    assert ev.particle_rows(np.array([3, 0, 1, 99])).tolist() == [2, -1, 0, -1]


def test_counts_by_hand():
    ev = hand_event()
    # t0: 4 of p1 (+1 noise) -> purity 0.8, matched, DM (4 > 2.5)
    # t1: 1 of p1 + 2 of p3 -> majority p3 with 2/3, matched; DM no (2 < 3)
    # t2: 3 of p3 + 3 noise -> purity 0.5 >= 0.5 matched (ACTS); DM no (purity not > 0.5)
    # t3: 2 of p2 + 2 of p3 -> tie 2/2: majority = smaller pid (2), purity 0.5 -> matched to p2
    # t4: 3 noise hits only -> fake
    tracks = [np.array([0, 1, 2, 3, 15]), np.array([4, 9, 10]), np.array([11, 12, 13, 15, 16, 17]),
              np.array([5, 6, 14, 9]), np.array([15, 16, 17])]
    m = match_tracks(ev, tracks, 0.5)
    assert m["major"].tolist() == [1, 3, 3, 2, 0]
    assert m["matched"].tolist() == [True, True, True, True, False]
    assert m["dm"].tolist() == [True, False, False, False, False]
    # targets: pT >= 1, >= 5 hits -> p1 and p3 (p2 has pT 0.5)
    c = event_counts(ev, tracks, TargetConfig(pt_min=1.0, min_hits=5))
    assert c["n_targets"] == 2 and c["n_targets_found"] == 2 and c["n_targets_found_dm"] == 1
    assert c["n_tracks"] == 5 and c["n_matched"] == 4 and c["n_fake"] == 1
    assert c["n_duplicate"] == 1            # p3 matched by t1 and t2
    s = summarize([c])
    assert s["efficiency"] == 1.0 and s["fakerate"] == 0.2 and s["duplicaterate"] == 0.2
    assert s["efficiency_dm"] == 0.5


def test_trackml_score_by_hand():
    ev = hand_event()
    # t0 owns 0..3 (p1, 4 of 5 > 2.5) -> weight 4; t1 = [4, 9, 10]: p3 has 2 of 3 owned hits but 2 < 6/2 -> 0;
    # t2 = [11..13, 15..17]: owns 3 of p3 + 3 noise (noise weight 0): 3 > 3? no -> 0
    tracks = [np.array([0, 1, 2, 3]), np.array([4, 9, 10]), np.array([11, 12, 13, 15, 16, 17])]
    w = trackml_track_weights(ev, tracks)
    assert w.tolist() == [4.0, 0.0, 0.0]
    assert trackml_score(ev, tracks) == 4.0
    # a hit shared with an earlier track belongs to the earlier one: t2 = [0, 1] owns nothing (both are t1's) -> 0
    tracks2 = [np.array([9, 10, 11, 12, 13, 14]), np.array([0, 1, 2, 3, 4]), np.array([0, 1])]
    assert trackml_track_weights(ev, tracks2).tolist() == [6.0, 5.0, 0.0]
    # prefix invariance: dropping later tracks never changes earlier contributions
    full = trackml_track_weights(ev, tracks2)
    for k in range(1, 4):
        assert trackml_track_weights(ev, tracks2[:k]).tolist() == full[:k].tolist()


def test_empty_and_degenerate_inputs():
    ev = hand_event()
    c = event_counts(ev, [], TargetConfig())
    assert c["n_tracks"] == 0 and c["n_targets_found"] == 0 and c["trackml_score_raw"] == 0.0
    s = summarize([c])
    assert s["fakerate"] == 0.0 and s["duplicaterate"] == 0.0 and s["efficiency"] == 0.0
    with pytest.raises(KeyError):
        EventData("bad", {"x": np.zeros(3), "y": np.zeros(3)}, {"particle_id": np.zeros(0)})
    with pytest.raises(ValueError):
        EventData("bad", {"x": np.zeros(3), "y": np.zeros(3), "z": np.zeros(2), "layer": np.zeros(3),
                          "particle_id": np.zeros(3)}, {"particle_id": np.zeros(0)})


def test_truth_edges_consecutive_and_selection():
    ev = hand_event()
    s, d = truth_edges(ev)
    # consecutive hits per particle: 4 + 3 + 5 edges, never noise, never across particles
    assert len(s) == 12 and np.all(ev.particle_id[s] == ev.particle_id[d]) and np.all(ev.particle_id[s] > 0)
    assert len(np.unique(pair_keys(s, d, ev.n_hits))) == 12
    s1, _ = truth_edges(ev, pt_min=1.0)
    assert len(s1) == 9                                   # p2 (pT 0.5) removed
    from hsf2.config import HitSelection
    e0 = select_hits(ev, HitSelection.preset("S0", 1.0))
    assert e0.n_hits == 11 and np.all(e0.particle_id > 0)
    e0n = select_hits(ev, HitSelection("S0N", 1.0, True))
    assert e0n.n_hits == 14 and int((e0n.particle_id == 0).sum()) == 3


def test_score_definition_and_errors():
    sc = ScoreConfig()
    m = {"efficiency": 0.9, "fakerate": 0.01, "duplicaterate": 0.1, "time_proxy": 0.5}
    assert abs(sc.score(m) - (0.9 - 0.01 - 0.02 - 0.1)) < 1e-15
    assert abs(sc.physics(m) - (0.9 - 0.01 - 0.02)) < 1e-15
    with pytest.raises(KeyError):
        sc.score({"efficiency": 0.9, "fakerate": 0.0, "duplicaterate": 0.0})     # no silent t = 0
    assert ScoreConfig(time_mode="none").score({"efficiency": 1, "fakerate": 0, "duplicaterate": 0}) == 1.0
    with pytest.raises(ValueError):
        ScoreConfig(time_mode="wallclock")


def test_splits_disjoint_and_ordered():
    ids = [f"event{i:09d}" for i in range(1000, 1200)]
    sp = SplitConfig(100, 10, 20, 25, 2).make(list(reversed(ids)))
    seen = set()
    for name, v in sp.items():
        assert not (seen & set(v)), name
        seen |= set(v)
    assert seen == set(ids)
    assert sp["train"][0] == "event000001000" and sp["tune"][0] == "event000001110"

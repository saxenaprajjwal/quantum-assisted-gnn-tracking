"""Exact evaluation of a whole grid of tracking-knob settings on one event (fast path of the QX1 landscape).

The reference is GNNPipeline.run_event (one knob setting at a time).  For a grid, every stage that does not
depend on a knob is shared:

  graph filter (dphi/deta/z0_scale)          -> GNN inference once per filter setting (level-1 cache)
  cut_hi, cut_lo_frac, max_degree            -> build_candidates once per triple  (level-2 cache: candidates)
  merge_overlap                              -> resolve_ambiguity once per value
  min_hits                                   -> all values from ONE accepted list:

The greedy ambiguity resolution visits candidates in order of quality = n_hits + mean edge score, and the mean
score of a candidate lies in [cut_lo, 1] with cut_lo = cut_lo_frac * cut_hi > 0.  So candidates are visited strictly
by length first: with min_hits = m the visited candidates (those with >= m hits) come first, in the same order,
and see the same set of used hits as without the shorter ones.  The tracks accepted for min_hits = m are therefore
exactly the tracks accepted for the smallest min_hits that have >= m hits -- a prefix of the acceptance order.
Track matching is per track, and a track's TrackML weight depends only on the tracks before it (shared hits
belong to the earlier track), so every count follows from per-track arrays and a prefix mask.  The prefix
property is asserted at run time; tests/test_tuning.py checks equality with GNNPipeline.run_event.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..config import TargetConfig, TrackingParams
from ..data.event import EventData, target_mask
from ..graph.construct import Graph
from ..tracking.build import build_candidates, resolve_ambiguity
from ..tracking.metrics import COUNT_KEYS, match_tracks, trackml_track_weights

FILTER_KNOBS = ("dphi_scale", "deta_scale", "z0_scale")
POST_KNOBS = ("cut_hi", "cut_lo_frac", "max_degree", "min_hits", "merge_overlap")
EDGE_KEYS = ("edges_loose", "edges_filtered", "true_edges_loose", "true_edges_filtered")


def accepted_track_counts(es: EventData, acc: List[np.ndarray], tpid: np.ndarray, targets: TargetConfig,
                          min_hits_values: Sequence[int]) -> np.ndarray:
    """COUNT_KEYS rows (one per min_hits value) from one accepted list (acceptance order, smallest min_hits)."""
    lens = np.array([len(t) for t in acc], dtype=np.int64)
    m = match_tracks(es, acc, targets.match_min)
    tw = trackml_track_weights(es, acc)
    w_total = float(es.weight.sum())
    out = np.zeros((len(min_hits_values), len(COUNT_KEYS)), np.float64)
    for j, mh in enumerate(min_hits_values):
        sel = lens >= int(mh)
        k = int(sel.sum())
        if k and not (sel[:k].all() and not sel[k:].any()):
            raise AssertionError("prefix property violated: accepted tracks are not ordered by length")
        major, matched, dm = m["major"][sel], m["matched"][sel], m["dm"][sel]
        mp = np.unique(major[matched])
        dmp = np.unique(major[dm])
        n_match = int(matched.sum())
        out[j] = (k, n_match, k - n_match, n_match - len(mp), len(tpid), int(np.isin(tpid, mp).sum()),
                  int(np.isin(tpid, dmp).sum()), int(dm.sum()), float(tw[sel].sum()), w_total, es.n_hits)
    return out


def post_grid_counts(es: EventData, g: Graph, keep: np.ndarray, scores: np.ndarray, targets: TargetConfig,
                     grid: Mapping[str, Sequence[float]], max_paths: int = 64) -> np.ndarray:
    """Counts for every post-GNN knob combination of `grid` on one event and one filter setting.

    Returns an array of shape (n_cut_hi, n_cut_lo_frac, n_max_degree, n_min_hits, n_merge_overlap, len(COUNT_KEYS)).
    """
    tpid = es.particles["particle_id"][target_mask(es, targets)]
    src, dst = g.src[keep], g.dst[keep]
    mh_values = [int(round(v)) for v in grid["min_hits"]]
    if sorted(mh_values) != mh_values:
        raise ValueError("min_hits grid values must be increasing")
    shape = tuple(len(grid[k]) for k in POST_KNOBS) + (len(COUNT_KEYS),)
    out = np.zeros(shape, np.float64)
    for (i, ch), (j, clf), (k, md) in itertools.product(enumerate(grid["cut_hi"]), enumerate(grid["cut_lo_frac"]),
                                                        enumerate(grid["max_degree"])):
        p = TrackingParams(cut_hi=ch, cut_lo_frac=clf, max_degree=md, min_hits=mh_values[0], merge_overlap=0.0,
                           max_paths=max_paths)
        if not p.cut_lo > 0:
            raise ValueError("the shared-min_hits evaluation needs cut_lo = cut_lo_frac * cut_hi > 0")
        cands = build_candidates(es.n_hits, src, dst, scores, p)
        for l, mo in enumerate(grid["merge_overlap"]):
            acc, _ = resolve_ambiguity(cands, es.n_hits, float(mo))
            out[i, j, k, :, l, :] = accepted_track_counts(es, acc, tpid, targets, mh_values)
    return out


def edge_counts(g: Graph, keep: np.ndarray) -> np.ndarray:
    return np.array([g.n_edges, int(keep.sum()), int(g.y.sum()), int(g.y[keep].sum())], np.float64)


def metrics_from_totals(counts: np.ndarray, edges: np.ndarray, n_events: int) -> Dict[str, float]:
    """The metrics of hsf2.tracking.pipeline.aggregate_rows from summed counts (no timing information)."""
    tot = dict(zip(COUNT_KEYS, (float(v) for v in counts)))
    nt = max(tot["n_tracks"], 1.0)
    e_loose, e_filt, t_loose, t_filt = (float(v) for v in edges)
    return {"n_events": n_events, **tot,
            "efficiency": tot["n_targets_found"] / max(tot["n_targets"], 1.0),
            "efficiency_dm": tot["n_targets_found_dm"] / max(tot["n_targets"], 1.0),
            "fakerate": tot["n_fake"] / nt if tot["n_tracks"] else 0.0,
            "duplicaterate": tot["n_duplicate"] / nt if tot["n_tracks"] else 0.0,
            "trackml_score": tot["trackml_score_raw"] / max(tot["trackml_weight_total"], 1e-12),
            "edges_loose_per_event": e_loose / n_events, "edges_filtered_per_event": e_filt / n_events,
            "true_edge_eff_filtered": t_filt / max(t_loose, 1.0), "time_proxy": e_filt / max(e_loose, 1.0)}


def vector_metrics(counts: np.ndarray, edges: np.ndarray) -> Dict[str, np.ndarray]:
    """metrics_from_totals for arrays of configurations: counts [..., len(COUNT_KEYS)], edges [..., 4]."""
    c = {k: counts[..., i] for i, k in enumerate(COUNT_KEYS)}
    nt = np.maximum(c["n_tracks"], 1.0)
    has = c["n_tracks"] > 0
    return {"efficiency": c["n_targets_found"] / np.maximum(c["n_targets"], 1.0),
            "efficiency_dm": c["n_targets_found_dm"] / np.maximum(c["n_targets"], 1.0),
            "fakerate": np.where(has, c["n_fake"] / nt, 0.0),
            "duplicaterate": np.where(has, c["n_duplicate"] / nt, 0.0),
            "trackml_score": c["trackml_score_raw"] / np.maximum(c["trackml_weight_total"], 1e-12),
            "time_proxy": edges[..., 1] / np.maximum(edges[..., 0], 1.0),
            "true_edge_eff_filtered": edges[..., 3] / np.maximum(edges[..., 2], 1.0),
            "n_tracks": c["n_tracks"]}

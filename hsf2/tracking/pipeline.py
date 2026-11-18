"""The GNN reconstruction chain for one parameter set: hit selection -> loose graph -> filters -> GNN ->
track building -> metrics, with the two caches that make tuning affordable:

level 1  edge scores per (event, graph-filter setting): the GNN runs once per filter setting;
level 2  everything after the GNN (track building, ambiguity resolution, matching) is cheap and redone.

Runtime: the measured CPU time per event (GNN inference + track building) is recorded, but tuning uses a
deterministic proxy -- the number of edges the GNN processes relative to the loose (default) graph -- so that
landscapes are free of timing noise (the Phase-1 lesson: on batch machines the time term mostly measured
the machine).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import TargetConfig, TrackingParams
from ..data.event import EventData, select_hits
from ..graph.construct import Graph, build_graph, filter_mask
from .build import build_tracks
from .metrics import event_counts, summarize


class GNNPipeline:
    def __init__(self, model, targets: Optional[TargetConfig] = None, score_cache_size: int = 2048) -> None:
        self.model = model
        self.targets = targets or TargetConfig()
        self._prep: Dict[str, Tuple[EventData, Graph]] = {}
        self._t_graph: Dict[str, float] = {}
        self._scores: "OrderedDict[tuple, tuple]" = OrderedDict()
        self.score_cache_size = int(score_cache_size)
        self.n_inference = 0

    # ------------------------------------------------------------------ stages
    def prepare(self, ev: EventData) -> Tuple[EventData, Graph]:
        if ev.event_id not in self._prep:
            t0 = time.process_time()
            es = select_hits(ev, self.model.hit_selection)
            g = build_graph(es, self.model.pmap, truth_pt_min=self.targets.pt_min)
            self._t_graph[ev.event_id] = time.process_time() - t0
            self._prep[ev.event_id] = (es, g)
        return self._prep[ev.event_id]

    def scores(self, ev: EventData, params: TrackingParams):
        es, g = self.prepare(ev)
        key = (ev.event_id,) + params.filter_key()
        hit = self._scores.get(key)
        if hit is not None:
            self._scores.move_to_end(key)
            return hit
        keep = filter_mask(g, self.model.pmap, params.dphi_scale, params.deta_scale, params.z0_scale)
        t0 = time.process_time()
        s = self.model.predict(es, g, keep)
        out = (keep, s, time.process_time() - t0)
        self.n_inference += 1
        self._scores[key] = out
        if len(self._scores) > self.score_cache_size:
            self._scores.popitem(last=False)
        return out

    def run_event(self, ev: EventData, params: TrackingParams):
        es, g = self.prepare(ev)
        keep, s, t_gnn = self.scores(ev, params)
        t0 = time.process_time()
        tracks, cands = build_tracks(es.n_hits, g.src[keep], g.dst[keep], s, params)
        t_build = time.process_time() - t0
        counts = event_counts(es, tracks, self.targets)
        counts.update({"edges_loose": int(g.n_edges), "edges_filtered": int(keep.sum()),
                       "true_edges_loose": int(g.y.sum()), "true_edges_filtered": int(g.y[keep].sum()),
                       "t_gnn_s": float(t_gnn), "t_build_s": float(t_build),
                       "t_graph_s": float(self._t_graph.get(ev.event_id, 0.0)),
                       "candidates": int(cands.stats.get("candidates", 0)),
                       "branching_components": int(cands.stats.get("branching_components", 0))})
        return tracks, counts, (es, g, keep, s, cands)

    def evaluate(self, events: Sequence[EventData], params: TrackingParams, per_event: bool = False) -> dict:
        rows = [self.run_event(ev, params)[1] for ev in events]
        out = aggregate_rows(rows)
        if per_event:
            out["per_event"] = rows
        return out


def aggregate_rows(rows: Sequence[dict]) -> dict:
    """Metrics of a set of events from their per-event counts (GNNPipeline.run_event rows).

    Ratios come from counts summed over the events (summarize); time_proxy = edges the GNN processed / edges of the
    unfiltered loose graphs, summed over the events (the deterministic runtime term of the tuning score).
    """
    if not rows:
        raise ValueError("no events to aggregate")
    n = len(rows)
    out = summarize(rows)
    e_loose = sum(r["edges_loose"] for r in rows)
    e_filt = sum(r["edges_filtered"] for r in rows)
    out["edges_loose_per_event"] = e_loose / n
    out["edges_filtered_per_event"] = e_filt / n
    out["true_edge_eff_filtered"] = sum(r["true_edges_filtered"] for r in rows) / max(sum(r["true_edges_loose"] for r in rows), 1)
    out["time_proxy"] = e_filt / max(e_loose, 1)
    out["t_gnn_per_event"] = sum(r.get("t_gnn_s", 0.0) for r in rows) / n
    out["t_build_per_event"] = sum(r.get("t_build_s", 0.0) for r in rows) / n
    out["t_per_event"] = out["t_gnn_per_event"] + out["t_build_per_event"]       # unchanged: graph build excluded
    out["t_graph_per_event"] = sum(r.get("t_graph_s", 0.0) for r in rows) / n
    out["edge_keep_fraction"] = e_filt / max(e_loose, 1)
    nf = sum(r.get("n_fake", 0) for r in rows)
    for k in ("n_fake_noise", "n_fake_mixed", "n_fake_wrong"):
        out[k] = int(sum(r.get(k, 0) for r in rows))
        out[k.replace("n_", "", 1) + "_frac"] = out[k] / nf if nf else 0.0
    out["candidates_per_event"] = sum(r.get("candidates", 0) for r in rows) / n
    return out

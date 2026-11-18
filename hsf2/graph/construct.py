"""Loose graph construction ("module-map-lite") and inference-time edge filters.

GNN4ITk builds its graph from a *module map*: the pairs of detector modules that consecutive hits of
simulated particles connect, each with cut windows learned from simulation.  We do the same at the level of
detector *layers* (enough for CPU-sized events):

1. learn_pair_map: from truth edges (consecutive hits of particles with pT >= pt_min_map) of training
   events, keep every layer pair that carries enough truth edges and learn per pair the windows
   |dphi| <= w_dphi, |deta| <= w_deta and, for pairs that are separated in radius, |z0| <= w_z0
   (z0 = intercept of the two-hit line with the beam axis);
2. build_graph: for every pair, all hit pairs inside the windows (k-d tree box query in (phi, eta), with
   phi wrap-around) become candidate edges, oriented from the hit closer to the origin to the farther one;
3. filter_mask: the tuning knobs dphi_scale / deta_scale / z0_scale tighten all windows at inference time
   (a boolean mask over the loose graph; the GNN is trained once on the loose graph).

Edge labels: y = the two hits are consecutive hits of one particle (ordered by distance from its
production vertex; what junction removal and the walk-through need), y_pid = same particle (Exa.TrkX "pid
truth", also true for edges that skip a hit of the particle).  Noise is never true.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..data.event import EventData, pair_keys, truth_edges

EDGE_FEATURES = ("dr", "dphi", "dz", "deta", "z0", "phi_slope")
Z0_BIG = 1.0e4
PHI_SLOPE_MAX = 0.01      # rad / mm (|dphi/dr| of a 1 GeV track in 2 T is ~3e-4)


def wrap_phi(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def edge_features(ev: EventData, src: np.ndarray, dst: np.ndarray) -> Dict[str, np.ndarray]:
    r, z, phi, eta = ev.r.astype(np.float64), ev.z.astype(np.float64), ev.phi.astype(np.float64), ev.eta
    dr = r[dst] - r[src]
    dz = z[dst] - z[src]
    dphi = wrap_phi(phi[dst] - phi[src])
    safe = np.abs(dr) > 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        z0 = np.where(safe, z[src] - r[src] * dz / np.where(safe, dr, 1.0), np.sign(dz + 1e-12) * Z0_BIG)
        slope = np.where(safe, dphi / np.where(safe, dr, 1.0), np.sign(dphi + 1e-12) * PHI_SLOPE_MAX)
    return {"dr": dr.astype(np.float32), "dphi": dphi.astype(np.float32), "dz": dz.astype(np.float32),
            "deta": (eta[dst] - eta[src]).astype(np.float32),
            "z0": np.clip(z0, -Z0_BIG, Z0_BIG).astype(np.float32),
            "phi_slope": np.clip(slope, -PHI_SLOPE_MAX, PHI_SLOPE_MAX).astype(np.float32)}


def orient(ev: EventData, a: np.ndarray, b: np.ndarray):
    """(src, dst) with src the hit closer to the origin (ties: lower index) -> every graph is a DAG."""
    rho = ev.rho
    swap = (rho[b] < rho[a]) | ((rho[b] == rho[a]) & (b < a))
    return np.where(swap, b, a), np.where(swap, a, b)


# ---------------------------------------------------------------------------
# Layer-pair map
# ---------------------------------------------------------------------------
@dataclass
class LayerPairMap:
    pairs: np.ndarray            # (P, 2) layer indices, a <= b
    w_dphi: np.ndarray
    w_deta: np.ndarray
    w_z0: np.ndarray             # inf where the pair has no z0 window
    counts: np.ndarray
    stats: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def to_dict(self) -> dict:
        return {"pairs": self.pairs.tolist(), "w_dphi": self.w_dphi.tolist(), "w_deta": self.w_deta.tolist(),
                "w_z0": [None if not math.isfinite(v) else float(v) for v in self.w_z0],
                "counts": self.counts.tolist(), "stats": self.stats, "config": self.config}

    @classmethod
    def from_dict(cls, d: dict) -> "LayerPairMap":
        return cls(np.asarray(d["pairs"], dtype=np.int64).reshape(-1, 2), np.asarray(d["w_dphi"], float),
                   np.asarray(d["w_deta"], float),
                   np.asarray([np.inf if v is None else v for v in d["w_z0"]], float),
                   np.asarray(d["counts"], np.int64), d.get("stats", {}), d.get("config", {}))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "LayerPairMap":
        return cls.from_dict(json.loads(Path(path).read_text()))


def learn_pair_map(events: Iterable[EventData], gcfg) -> LayerPairMap:
    """Layer pairs and windows from the truth edges of (already hit-selected) training events."""
    per_pair: Dict[tuple, List[np.ndarray]] = {}
    n_total, n_events = 0, 0
    for ev in events:
        n_events += 1
        s, d = truth_edges(ev, gcfg.pt_min_map)
        if len(s) == 0:
            continue
        s, d = orient(ev, s, d)
        f = edge_features(ev, s, d)
        la, lb = ev.layer[s].astype(np.int64), ev.layer[d].astype(np.int64)
        a, b = np.minimum(la, lb), np.maximum(la, lb)
        key = a * 10000 + b
        order = np.argsort(key, kind="stable")
        key = key[order]
        stacked = np.stack([np.abs(f["dphi"]), np.abs(f["deta"]), np.abs(f["z0"]), np.abs(f["dr"])], 1)[order]
        uk, start = np.unique(key, return_index=True)
        ends = np.append(start[1:], len(key))
        for k, i0, i1 in zip(uk, start, ends):
            per_pair.setdefault((int(k) // 10000, int(k) % 10000), []).append(stacked[i0:i1])
        n_total += len(s)
    if n_total == 0:
        raise ValueError("no truth edges in the training events (check the hit selection / pt_min_map)")
    min_count = max(int(gcfg.pair_min_count), int(math.ceil(gcfg.pair_min_frac * n_total)))
    pooled = {k: np.concatenate(v) for k, v in per_pair.items()}
    allv = np.concatenate(list(pooled.values()))
    cap_d, cap_e = gcfg.cap_margin * np.quantile(allv[:, :2], gcfg.cap_quantile, axis=0)
    zsel = np.concatenate([v[:, 2] for v in pooled.values() if np.median(v[:, 3]) >= gcfg.z0_min_dr] or [allv[:, 2]])
    cap_z = gcfg.cap_margin * float(np.quantile(zsel, gcfg.cap_quantile))
    rows = []
    for (a, b), v in sorted(pooled.items()):
        if len(v) < min_count:
            continue
        qd, qe, qz = np.quantile(v[:, :3], gcfg.quantile, axis=0)
        med_dr = float(np.median(v[:, 3]))
        wz = min(cap_z, max(gcfg.floor_z0, gcfg.margin * qz)) if med_dr >= gcfg.z0_min_dr else np.inf
        rows.append((a, b, min(cap_d, max(gcfg.floor_dphi, gcfg.margin * qd)),
                     min(cap_e, max(gcfg.floor_deta, gcfg.margin * qe)), wz, len(v)))
    pairs = np.array([[r[0], r[1]] for r in rows], dtype=np.int64).reshape(-1, 2)
    counts = np.array([r[5] for r in rows], dtype=np.int64)
    stats = {"n_events": n_events, "n_truth_edges": int(n_total), "n_pairs_seen": len(per_pair),
             "n_pairs_kept": len(rows), "truth_edges_in_kept_pairs": float(counts.sum() / n_total),
             "min_count": min_count, "cap_dphi": float(cap_d), "cap_deta": float(cap_e), "cap_z0": float(cap_z)}
    return LayerPairMap(pairs, np.array([r[2] for r in rows]), np.array([r[3] for r in rows]),
                        np.array([r[4] for r in rows], dtype=float), counts, stats, gcfg.to_dict())


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    event_id: str
    n_nodes: int
    src: np.ndarray
    dst: np.ndarray
    pair: np.ndarray             # index into LayerPairMap.pairs
    feats: Dict[str, np.ndarray]
    y: np.ndarray                # consecutive-hit label
    y_pid: np.ndarray            # same-particle label
    meta: dict = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def mask(self, keep: np.ndarray) -> "Graph":
        keep = np.asarray(keep, bool)
        return Graph(self.event_id, self.n_nodes, self.src[keep], self.dst[keep], self.pair[keep],
                     {k: v[keep] for k, v in self.feats.items()}, self.y[keep], self.y_pid[keep], dict(self.meta))

    def labels(self, kind: str = "consecutive") -> np.ndarray:
        return self.y if kind == "consecutive" else self.y_pid

    def save(self, path: Path) -> None:
        arrays = {"src": self.src, "dst": self.dst, "pair": self.pair, "y": self.y, "y_pid": self.y_pid,
                  "n_nodes": np.array(self.n_nodes), "event_id": np.array(self.event_id),
                  "meta": np.array(json.dumps(self.meta))}
        arrays.update({f"f_{k}": v for k, v in self.feats.items()})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Graph":
        with np.load(path, allow_pickle=False) as f:
            feats = {k[2:]: f[k] for k in f.files if k.startswith("f_")}
            return cls(str(f["event_id"]), int(f["n_nodes"]), f["src"], f["dst"], f["pair"], feats, f["y"],
                       f["y_pid"], json.loads(str(f["meta"])))


def _layer_index(layer: np.ndarray) -> Dict[int, np.ndarray]:
    order = np.argsort(layer, kind="stable")
    ls = layer[order]
    uk, start = np.unique(ls, return_index=True)
    ends = np.append(start[1:], len(ls))
    return {int(k): order[i0:i1] for k, i0, i1 in zip(uk, start, ends)}


def _box_pairs(pa: np.ndarray, ea: np.ndarray, pb: np.ndarray, eb: np.ndarray, wphi: float, weta: float):
    """Index pairs (i in A, j in B) with |dphi| <= wphi and |deta| <= weta (phi periodic)."""
    ext_i = np.arange(len(pa))
    hi = pa > np.pi - wphi
    lo = pa < -np.pi + wphi
    pa_ext = np.concatenate([pa, pa[hi] - 2 * np.pi, pa[lo] + 2 * np.pi])
    ea_ext = np.concatenate([ea, ea[hi], ea[lo]])
    ext_i = np.concatenate([ext_i, ext_i[hi], ext_i[lo]])
    ta = cKDTree(np.stack([pa_ext / wphi, ea_ext / weta], 1))
    tb = cKDTree(np.stack([pb / wphi, eb / weta], 1))
    m = ta.sparse_distance_matrix(tb, 1.0, p=np.inf, output_type="ndarray")
    return ext_i[m["i"]], m["j"]


def build_graph(ev: EventData, pmap: LayerPairMap, truth_pt_min: float = 0.0) -> Graph:
    """Loose graph of one (hit-selected) event (meta: construction statistics and wall time)."""
    t0 = time.perf_counter()
    by_layer = _layer_index(ev.layer.astype(np.int64))
    phi, eta = ev.phi.astype(np.float64), ev.eta.astype(np.float64)
    S, D, P = [], [], []
    for p, (a, b) in enumerate(pmap.pairs):
        ia, ib = by_layer.get(int(a)), by_layer.get(int(b))
        if ia is None or ib is None or len(ia) == 0 or len(ib) == 0:
            continue
        i, j = _box_pairs(phi[ia], eta[ia], phi[ib], eta[ib], float(pmap.w_dphi[p]), float(pmap.w_deta[p]))
        u, v = ia[i], ib[j]
        if a == b:                       # same layer: each unordered pair once, no self-loops
            keep = u < v
            u, v = u[keep], v[keep]
        if len(u) == 0:
            continue
        s, d = orient(ev, u, v)
        S.append(s); D.append(d); P.append(np.full(len(s), p, dtype=np.int16))
    if S:
        src, dst, pair = np.concatenate(S), np.concatenate(D), np.concatenate(P)
    else:
        src = dst = np.zeros(0, np.int64)
        pair = np.zeros(0, np.int16)
    # remove duplicates (a hit pair reachable through two wrap copies)
    if len(src):
        key = pair_keys(src, dst, ev.n_hits)
        _, first = np.unique(key, return_index=True)
        first.sort()
        src, dst, pair = src[first], dst[first], pair[first]
    feats = edge_features(ev, src, dst)
    keep = np.abs(feats["z0"]) <= pmap.w_z0[pair] if len(src) else np.zeros(0, bool)
    src, dst, pair = src[keep], dst[keep], pair[keep]
    feats = {k: v[keep] for k, v in feats.items()}
    pid = ev.particle_id
    y_pid = (pid[src] == pid[dst]) & (pid[src] != 0)
    ts, td = truth_edges(ev)
    y = np.isin(pair_keys(src, dst, ev.n_hits), pair_keys(ts, td, ev.n_hits)) if len(src) else np.zeros(0, bool)
    g = Graph(ev.event_id, ev.n_hits, src.astype(np.int64), dst.astype(np.int64), pair.astype(np.int16), feats,
              y, y_pid)
    g.meta.update(construction_stats(ev, g, truth_pt_min))
    g.meta["t_build_graph_s"] = round(time.perf_counter() - t0, 4)
    return g


def construction_stats(ev: EventData, g: Graph, pt_min: float = 0.0) -> dict:
    """Truth-edge efficiency (consecutive hits of particles with pT >= pt_min found in the graph) and purity."""
    s, d = truth_edges(ev, pt_min)
    tk = pair_keys(s, d, ev.n_hits)
    gk = pair_keys(g.src, g.dst, ev.n_hits)
    found = np.isin(tk, gk)
    return {"n_nodes": int(g.n_nodes), "n_edges": int(g.n_edges), "n_true_edges": int(g.y.sum()),
            "edge_purity": float(g.y.mean()) if g.n_edges else 0.0,
            "edge_purity_pid": float(g.y_pid.mean()) if g.n_edges else 0.0,
            "truth_edges": int(len(tk)), "truth_edge_eff": float(found.mean()) if len(tk) else float("nan"),
            "truth_pt_min": float(pt_min)}


def filter_mask(g: Graph, pmap: LayerPairMap, dphi_scale: float = 1.0, deta_scale: float = 1.0,
                z0_scale: float = 1.0) -> np.ndarray:
    """Inference-time tightening of every window of the loose graph (all scales 1 -> keep everything)."""
    p = g.pair.astype(np.int64)
    keep = np.abs(g.feats["dphi"]) <= dphi_scale * pmap.w_dphi[p]
    keep &= np.abs(g.feats["deta"]) <= deta_scale * pmap.w_deta[p]
    wz = pmap.w_z0[p]
    keep &= ~np.isfinite(wz) | (np.abs(g.feats["z0"]) <= z0_scale * np.where(np.isfinite(wz), wz, 0.0))
    return keep

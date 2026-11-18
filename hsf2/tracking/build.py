"""Track building from edge scores (all steps are inference-time knobs of TrackingParams).

1. usable edges: score >= cut_lo  (cut_lo = cut_lo_frac * cut_hi, GNN4ITk walk-through "thMin");
2. junction removal: every hit keeps only its max_degree best incoming and best outgoing edges
   (max_degree = 1 reproduces the connected components with junction removal of the ACTS GNN examples);
3. connected components; a component without branching is one track candidate;
4. walk-through of branching components (GNN4ITk): from every hit without incoming edge follow the best
   outgoing edge, and branch also along every other edge with score >= cut_hi ("thAdd"); every path is a
   candidate (at most max_paths per component);
5. candidates with fewer than min_hits hits are dropped;
6. ambiguity resolution: candidates sorted by (number of hits, mean edge score) are accepted greedily if
   at most merge_overlap of their hits are already used by accepted tracks.

Steps 4-6 produce overlapping candidates; the QUBO track selection (hsf2.qubo.selection) replaces step 6.
Edges must point away from the beam (graph.construct.orient), so the graph is a DAG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class Candidates:
    tracks: List[np.ndarray]          # hit indices of each candidate
    quality: np.ndarray               # n_hits + mean edge score (sort key of the ambiguity resolution)
    mean_score: np.ndarray
    stats: Dict[str, float] = field(default_factory=dict)


def junction_mask(n_nodes: int, src: np.ndarray, dst: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    """Keep an edge if it is among the k best outgoing edges of its source AND the k best incoming of its target."""
    if len(src) == 0:
        return np.zeros(0, bool)
    keep = np.ones(len(src), bool)
    for node in (src, dst):
        order = np.lexsort((-score, node))
        ns = node[order]
        start = np.searchsorted(ns, ns, side="left")
        rank = np.arange(len(ns)) - start
        r = np.empty(len(ns), np.int64)
        r[order] = rank
        keep &= r < k
    return keep


def _walk(nodes_local_out: Dict[int, List[Tuple[int, float]]], starts: List[int], cut_hi: float,
          max_paths: int) -> List[List[int]]:
    paths: List[List[int]] = []
    for s in starts:
        stack = [[s]]
        while stack and len(paths) < max_paths:
            path = stack.pop()
            nxt = nodes_local_out.get(path[-1], ())
            if not nxt:
                paths.append(path)
                continue
            follow = [nxt[0]] + [e for e in nxt[1:] if e[1] >= cut_hi]
            for v, _ in reversed(follow):          # best branch explored first
                stack.append(path + [v])
        if len(paths) >= max_paths:
            break
    return paths


def build_candidates(n_nodes: int, src: np.ndarray, dst: np.ndarray, score: np.ndarray, params) -> Candidates:
    """Steps 1-5: track candidates (possibly sharing hits) from scored edges of one event."""
    cut_hi, cut_lo = float(params.cut_hi), float(params.cut_lo)
    sel = score >= cut_lo
    s, d, w = src[sel], dst[sel], score[sel]
    jm = junction_mask(n_nodes, s, d, w, int(params.max_degree))
    s, d, w = s[jm], d[jm], w[jm]
    stats = {"edges_used": int(len(s))}
    if len(s) == 0:
        return Candidates([], np.zeros(0), np.zeros(0), stats)
    adj = coo_matrix((np.ones(len(s)), (s, d)), shape=(n_nodes, n_nodes))
    _, label = connected_components(adj, directed=False)
    deg_in = np.bincount(d, minlength=n_nodes)
    deg_out = np.bincount(s, minlength=n_nodes)
    involved = np.zeros(n_nodes, bool)
    involved[s] = True
    involved[d] = True
    branching_nodes = (deg_in > 1) | (deg_out > 1)
    comp_branch = np.zeros(label.max() + 1, bool)
    comp_branch[label[branching_nodes]] = True
    edge_comp = label[s]
    tracks: List[np.ndarray] = []
    means: List[float] = []
    # simple components: one candidate each (vectorised)
    simple_nodes = np.nonzero(involved & ~comp_branch[label])[0]
    if len(simple_nodes):
        lab = label[simple_nodes]
        order = np.argsort(lab, kind="stable")
        lab_s, nodes_s = lab[order], simple_nodes[order]
        uk, start = np.unique(lab_s, return_index=True)
        ends = np.append(start[1:], len(lab_s))
        esel = ~comp_branch[edge_comp]
        sums = np.bincount(edge_comp[esel], weights=w[esel], minlength=len(comp_branch))
        cnts = np.bincount(edge_comp[esel], minlength=len(comp_branch))
        for c, i0, i1 in zip(uk, start, ends):
            tracks.append(nodes_s[i0:i1])
            means.append(sums[c] / max(cnts[c], 1))
    # branching components: walk-through
    n_branch = int(comp_branch.sum())
    if n_branch:
        bsel = comp_branch[edge_comp]
        bs, bd, bw, bc = s[bsel], d[bsel], w[bsel], edge_comp[bsel]
        order = np.lexsort((-bw, bs))
        bs, bd, bw, bc = bs[order], bd[order], bw[order], bc[order]
        out: Dict[int, List[Tuple[int, float]]] = {}
        for a, b, x in zip(bs.tolist(), bd.tolist(), bw.tolist()):
            out.setdefault(a, []).append((b, x))
        edge_w = {(a, b): x for a, b, x in zip(bs.tolist(), bd.tolist(), bw.tolist())}
        comp_nodes: Dict[int, set] = {}
        for a, b, c in zip(bs.tolist(), bd.tolist(), bc.tolist()):
            comp_nodes.setdefault(c, set()).update((a, b))
        for c in sorted(comp_nodes):
            nodes = comp_nodes[c]
            starts = sorted(v for v in nodes if deg_in[v] == 0)
            for path in _walk(out, starts, cut_hi, int(params.max_paths)):
                if len(path) < 2:
                    continue
                tracks.append(np.asarray(path, dtype=np.int64))
                means.append(float(np.mean([edge_w[(path[i], path[i + 1])] for i in range(len(path) - 1)])))
    stats["components"] = int(len(np.unique(label[involved])))
    stats["branching_components"] = n_branch
    lengths = np.array([len(t) for t in tracks], dtype=np.int64)
    keep = lengths >= int(params.min_hits)
    tracks = [t for t, k in zip(tracks, keep) if k]
    means_a = np.asarray(means, float)[keep] if len(means) else np.zeros(0)
    quality = lengths[keep] + means_a
    stats["candidates"] = int(len(tracks))
    return Candidates(tracks, quality, means_a, stats)


def resolve_ambiguity(cands: Candidates, n_nodes: int, merge_overlap: float) -> Tuple[List[np.ndarray], np.ndarray]:
    """Step 6, greedy: returns the accepted tracks (in acceptance order) and the indices of the candidates."""
    order = np.argsort(-cands.quality, kind="stable")
    used = np.zeros(n_nodes, bool)
    acc, idx = [], []
    for i in order:
        t = cands.tracks[i]
        shared = int(used[t].sum())
        if shared <= merge_overlap * len(t) + 1e-12:
            acc.append(t)
            idx.append(int(i))
            used[t] = True
    return acc, np.asarray(idx, dtype=np.int64)


def build_tracks(n_nodes: int, src: np.ndarray, dst: np.ndarray, score: np.ndarray, params):
    """Steps 1-6 for one event: (accepted tracks, candidates)."""
    cands = build_candidates(n_nodes, src, dst, score, params)
    tracks, _ = resolve_ambiguity(cands, n_nodes, float(params.merge_overlap))
    return tracks, cands


def conflict_pairs(tracks: List[np.ndarray], n_nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(i, j, n_shared) for every pair of candidates sharing at least one hit (i < j)."""
    if not tracks:
        z = np.zeros(0, np.int64)
        return z, z, z
    tid = np.concatenate([np.full(len(t), k) for k, t in enumerate(tracks)])
    hit = np.concatenate(tracks)
    order = np.lexsort((tid, hit))
    hit, tid = hit[order], tid[order]
    uk, start, cnt = np.unique(hit, return_index=True, return_counts=True)
    pairs: Dict[Tuple[int, int], int] = {}
    for i0, c in zip(start[cnt > 1], cnt[cnt > 1]):
        ts = tid[i0:i0 + c]
        for a in range(c):
            for b in range(a + 1, c):
                key = (int(min(ts[a], ts[b])), int(max(ts[a], ts[b])))
                pairs[key] = pairs.get(key, 0) + 1
    if not pairs:
        z = np.zeros(0, np.int64)
        return z, z, z
    k = np.array(list(pairs.keys()), dtype=np.int64)
    return k[:, 0], k[:, 1], np.array(list(pairs.values()), dtype=np.int64)

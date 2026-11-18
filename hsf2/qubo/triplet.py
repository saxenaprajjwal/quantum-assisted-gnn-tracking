"""Bapst-style triplet QUBO (F. Bapst et al., "A pattern recognition algorithm for quantum annealers", Comput. Softw.
Big Sci. 4 (2020) 1, arXiv:1902.08324) on barrel sectors of TrackML or synthetic events.

As in the paper
  * barrel hits only, one hit per particle and layer (the one closest to the origin), noise kept;
  * triplets T = (a, b, c) of two doublets sharing hit b, ordered by radius, with holes H (skipped layers) <= 1,
    |curvature| <= CURV_MAX and delta-theta (difference of the doublets' polar angles in r-z) <= 0.1 rad;
  * T_i = (a, b, c) and T_j = (b, c, d) form a quadruplet if |delta curvature| <= DCURV_MAX and S(T_i, T_j) > 0.2,
    S = [1 - (|dcurv| / DCURV_MAX + max(dtheta_i, dtheta_j) / DTHETA_MAX) / 2] / (1 + H_q)^2    (paper Eq. 3,
    with the normalisation of the reference code; H_q = layers skipped by the four hits)
  * pre-filter: only quadruplets in a chain of >= MIN_QPLET_PATH quadruplets (>= 5 hits), and only triplets in such a
    quadruplet, enter the QUBO ("triplets in no quadruplet or whose longest potential track has fewer than five hits
    are dropped"; reference code: QallseMp, min_qplet_path = 2);
  * O = sum_i a T_i + sum_{i<j} b_ij T_i T_j, a = alpha, b_ij = -S (quadruplet), zeta (any other shared hit; a
    "quintet" c = d is neither), 0 otherwise                                               (paper Eqs. 4-5)
  * post-processing: selected triplets -> doublets; doublets with unresolved conflicts (a hit with two selected
    inner or two selected outer doublets) are removed; tracks = chains of >= 5 hits;
  * metrics on the final doublets: efficiency = D_matched / D_true, purity = D_matched / (D_rec - D_oa), with D_true the
    consecutive doublets of particles with pT > 1 GeV and >= 5 barrel hits, and D_oa the matched doublets of other
    particles (paper Eqs. 6-7); plus the TrackML score of the tracks.
Different from the paper (stated wherever results are reported)
  * initial doublets from geometric windows on consecutive / next-to-consecutive barrel layers (|dphi| <= k dr,
    |z0| <= 150 mm), not the ATLAS seeding code; a phi sector and a particle fraction keep the QUBO small;
  * curvature cuts: the paper prints 8e-4 and 1e-4 with the unit "GeV^-1"; its reference code (qallse, Config1GeV)
    applies them to the Menger curvature 1/R of the hits' x, y in mm, i.e. in mm^-1 (pT >~ 0.75 GeV at 2 T), as here.
    (v0.2.0 read them as MeV^-1 and had no pre-filter: those QX3b numbers are withdrawn.)
  * qbsolv / D-Wave are replaced by a decomposition into connected components of the coupling graph, solved by
    exact enumeration (<= 24 variables), Phase-1 simulated annealing (any size) and p=1 QAOA (exact statevector,
    components <= max_vars), so the comparison is exact where it can be.

Classical baselines on the same QUBO, scored with the same postprocess + doublet_metrics (run_study rows)
  * all-kept-triplets: every triplet of the QUBO.  A diagnostic only: it creates branches, postprocess drops every
    doublet with a conflict, so it reconstructs (almost) nothing;
  * hole-free: every kept triplet with holes == 0;
  * greedy-S: conflict-free greedy; weight w_i = sum of S over the kept quadruplets containing triplet i; visit in
    decreasing w (ties: fewer holes, then index), add a triplet if it has no conflict (zeta pair) with the selected ones;
  * hole-free+descent, greedy-S+descent: steepest-descent single-flip local search on the QUBO energy (deterministic,
    stops when no flip lowers the energy) started from the hole-free / greedy-S selection.  (From the empty selection
    it never moves when alpha = 0: every single flip has dE = 0.)
Energy convention (all solvers): E(x) = alpha sum_i x_i - sum_(quadruplets) S_ij x_i x_j + zeta sum_(conflicts) x_i x_j.

Component table and export (run_study(..., component_rows=list, export_dir=Path))
  * component_rows gets one dict per connected component of the coupling graph (columns of components.csv): event,
    sector, phi0, dphi, frac, component, n_variables, n_quadruplet_couplings, n_conflicts, n_true_triplets,
    n_fake_triplets (true = the three hits belong to one particle id > 0), exact_energy (n_variables <= 24, else
    None), holefree_energy, greedy_energy, descent_energy (hole-free+descent), greedy_descent_energy (energies of the
    baseline selections restricted to the component), trivial (exact optimum selection == hole-free selection; n <= 24 only), holefree_optimal (hole-free
    energy equals the exact optimum within 1e-9; n <= 24 only).
  * export_dir gets one JSON file per component with n_variables <= 24, named
    `<event>_s<sector index>_f<frac>_c<component id>.json` (frac as %g; existing files are never overwritten):
      {"meta": {"event", "sector", "phi0", "dphi", "frac", "component", "alpha", "zeta"},
       "triplets": [[a, b, c], ...]            hit indices in the sub-event (barrel_sector output),
       "triplets_hit_id": [[a, b, c], ...]     the corresponding global hit ids of the event,
       "linear": [a_i, ...]                    QUBO diagonal (= alpha), local order,
       "quadratic": [[i, j, b_ij], ...]        i < j local indices, b_ij = -S (quadruplet) or zeta (conflict),
       "exact_energy": float, "exact_selection": [0/1, ...], "true_triplet": [bool, ...]}
    so E(x) = sum_i linear[i] x_i + sum quadratic b_ij x_i x_j; local index k is the k-th triplet of the list.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..data.event import EventData

TRACKML_BARREL = (7, 8, 9, 10, 24, 25, 26, 27, 40, 41)     # (vol, layer) 8/2..8, 13/2..8, 17/2..4 in LAYER_INDEX
TOY_BARREL = tuple(range(10))
B_FIELD = 2.0                                                # T
CURV_MAX = 8e-4                                              # mm^-1: R >= 1.25 m, pT >~ 0.75 GeV at 2 T
DCURV_MAX = 1e-4                                             # mm^-1
DTHETA_MAX = 0.1                                             # rad
S_MIN = 0.2
MIN_QPLET_PATH = 2                                           # quadruplets per chain: 2 = five hits


def barrel_sector(ev: EventData, barrel: Sequence[int], phi0: float, dphi: float, frac: float = 1.0,
                  seed: int = 0) -> Tuple[EventData, np.ndarray]:
    """Barrel hits in phi0 <= phi < phi0 + dphi, one per (particle, layer), a fraction `frac` of the particles and of
    the noise; returns the sub-event and the barrel position (0..len(barrel)-1) of every hit."""
    pos_of = {int(l): k for k, l in enumerate(barrel)}
    lay = ev.layer.astype(np.int64)
    inb = np.isin(lay, list(pos_of))
    dp = (ev.phi - phi0) % (2 * np.pi)
    keep = inb & (dp < dphi)
    rng = np.random.default_rng(seed)
    pids = np.unique(ev.particle_id[keep & (ev.particle_id > 0)])
    chosen = set(pids[rng.random(len(pids)) < frac].tolist())
    noise_keep = rng.random(ev.n_hits) < frac
    keep &= np.where(ev.particle_id > 0, np.isin(ev.particle_id, list(chosen)), noise_keep)
    idx = np.nonzero(keep)[0]
    # one hit per particle and layer: the one closest to the origin (noise hits are all kept)
    order = idx[np.lexsort((ev.rho[idx], lay[idx], ev.particle_id[idx]))]
    pid, ly = ev.particle_id[order], lay[order]
    first = np.ones(len(order), bool)
    first[1:] = (pid[1:] != pid[:-1]) | (ly[1:] != ly[:-1])
    first |= pid == 0
    sub = ev.subset(np.sort(order[first]))
    return sub, np.array([pos_of[int(l)] for l in sub.layer], dtype=np.int64)


def make_doublets(ev: EventData, bpos: np.ndarray, k_dphi: float = 7e-4, z0_max: float = 150.0,
                  max_gap: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Doublets (inner, outer) between barrel positions p and p + 1 .. p + max_gap with |dphi| <= k_dphi * dr and
    |z0| <= z0_max (dphi / dr bound of a 1 GeV track in 2 T is ~3e-4 rad/mm)."""
    S, D = [], []
    r, z, phi = ev.r.astype(float), ev.z.astype(float), ev.phi.astype(float)
    by = [np.nonzero(bpos == p)[0] for p in range(int(bpos.max()) + 1)] if len(bpos) else []
    for p in range(len(by)):
        for g in range(1, max_gap + 1):
            if p + g >= len(by) or not len(by[p]) or not len(by[p + g]):
                continue
            a, b = by[p], by[p + g]
            A, B = np.repeat(a, len(b)), np.tile(b, len(a))
            dr = r[B] - r[A]
            dphi = np.abs((phi[B] - phi[A] + np.pi) % (2 * np.pi) - np.pi)
            z0 = z[A] - r[A] * (z[B] - z[A]) / np.where(np.abs(dr) > 1e-6, dr, 1e-6)
            ok = (dphi <= k_dphi * np.abs(dr)) & (np.abs(z0) <= z0_max)
            S.append(A[ok])
            D.append(B[ok])
    if not S:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(S).astype(np.int64), np.concatenate(D).astype(np.int64)


def pt_of_curvature(curv_per_mm, b_field: float = B_FIELD):
    """pT [GeV] of a circle of curvature 1/R [mm^-1] in a field of b_field [T]: pT = 0.3 B R."""
    return 0.3 * b_field * 1e-3 / np.maximum(np.abs(curv_per_mm), 1e-30)


def curvature(ev: EventData, a, b, c) -> np.ndarray:
    """Signed Menger curvature 1/R [mm^-1] of the circle through three hits in the transverse plane."""
    x, y = ev.x.astype(float), ev.y.astype(float)
    ax, ay, bx, by_, cx, cy = x[a], y[a], x[b], y[b], x[c], y[c]
    cross = (bx - ax) * (cy - by_) - (by_ - ay) * (cx - bx)
    d_ab = np.hypot(bx - ax, by_ - ay)
    d_bc = np.hypot(cx - bx, cy - by_)
    d_ac = np.hypot(cx - ax, cy - ay)
    return 2.0 * cross / np.maximum(d_ab * d_bc * d_ac, 1e-12)


@dataclass
class TripletQUBO:
    triplets: np.ndarray        # (n, 3) hit indices a, b, c
    holes: np.ndarray
    curv: np.ndarray            # signed curvature 1/R, mm^-1
    dtheta: np.ndarray
    qi: np.ndarray              # quadruplet pairs (i < j) and their strength S
    qj: np.ndarray
    S: np.ndarray
    ci: np.ndarray              # conflict pairs
    cj: np.ndarray
    alpha: float
    zeta: float
    n_raw: int = 0              # triplets passing the triplet cuts, before the pre-filter
    n_quad_raw: int = 0         # quadruplets passing the quadruplet cuts, before the pre-filter

    @property
    def n(self) -> int:
        return len(self.triplets)

    def components(self) -> List[np.ndarray]:
        if self.n == 0:
            return []
        i = np.concatenate([self.qi, self.ci])
        j = np.concatenate([self.qj, self.cj])
        _, lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(self.n, self.n)), directed=False)
        order = np.argsort(lab, kind="stable")
        _, start = np.unique(lab[order], return_index=True)
        return np.split(order, start[1:])

    def sub_qubo(self, comp: np.ndarray):
        """Phase-1 QUBO container of one component: L = alpha, W_ij = b_ij."""
        from acts_qaoa.surrogate import QUBO
        pos = {int(c): k for k, c in enumerate(comp)}
        m = len(comp)
        W = np.zeros((m, m))
        for arr_i, arr_j, val in ((self.qi, self.qj, -self.S), (self.ci, self.cj, np.full(len(self.ci), self.zeta))):
            for a, b, v in zip(arr_i, arr_j, val):
                if int(a) in pos and int(b) in pos:
                    k, l = sorted((pos[int(a)], pos[int(b)]))
                    W[k, l] += float(v)
        return QUBO(n=m, const=0.0, L=np.full(m, float(self.alpha)), W=W)


def build_triplet_qubo(ev: EventData, bpos: np.ndarray, ds: np.ndarray, dd: np.ndarray, alpha: float = 0.0,
                       zeta: float = 1.0, min_qplet_path: int = MIN_QPLET_PATH) -> TripletQUBO:
    r, z = ev.r.astype(float), ev.z.astype(float)
    # triplets: doublet (a, b) followed by doublet (b, c)
    order = np.argsort(ds, kind="stable")
    ds_s, dd_s = ds[order], dd[order]
    T = []
    for a, b in zip(ds, dd):
        lo, hi = np.searchsorted(ds_s, b), np.searchsorted(ds_s, b, side="right")
        for c in dd_s[lo:hi]:
            T.append((a, b, c))
    T = np.array(T, dtype=np.int64).reshape(-1, 3)
    if len(T):
        a, b, c = T.T
        holes = (bpos[b] - bpos[a] - 1) + (bpos[c] - bpos[b] - 1)
        curv = curvature(ev, a, b, c)
        th1 = np.arctan2(r[b] - r[a], z[b] - z[a])
        th2 = np.arctan2(r[c] - r[b], z[c] - z[b])
        dth = np.abs(th2 - th1)
        ok = (holes <= 1) & (np.abs(curv) <= CURV_MAX) & (dth <= DTHETA_MAX)
        T, holes, curv, dth = T[ok], holes[ok], curv[ok], dth[ok]
    else:
        holes = curv = dth = np.zeros(0)
    n_raw = len(T)
    # quadruplets (T_i, T_j) with T_i = (a, b, c), T_j = (b, c, d)
    starts: Dict[Tuple[int, int], List[int]] = {}
    for k, (a, b, _) in enumerate(T):
        starts.setdefault((int(a), int(b)), []).append(k)
    Q = []
    for i, (a, b, c) in enumerate(T):
        for j in starts.get((int(b), int(c)), []):
            dq = abs(curv[i] - curv[j])
            span = int(bpos[T[j][2]] - bpos[a]) - 3                       # layers skipped by the four hits
            s = (1.0 - 0.5 * (dq / DCURV_MAX + max(dth[i], dth[j]) / DTHETA_MAX)) / (1.0 + span) ** 2
            if dq <= DCURV_MAX and s > S_MIN:
                Q.append((i, j, s))
    n_quad_raw = len(Q)
    # pre-filter: longest chain of quadruplets through each quadruplet (the triplet graph is a DAG: outward in r)
    if Q:
        qi0 = np.array([q[0] for q in Q], np.int64)
        qj0 = np.array([q[1] for q in Q], np.int64)
        lin = np.zeros(n_raw, np.int64)                  # quadruplets in the longest chain ending at triplet t
        lout = np.zeros(n_raw, np.int64)                 # ... starting at triplet t
        by_first = np.argsort(bpos[T[:, 0]], kind="stable")
        rank = np.empty(n_raw, np.int64)
        rank[by_first] = np.arange(n_raw)
        e_in = np.argsort(rank[qi0], kind="stable")      # edges i -> j in increasing order of i
        for e in e_in:
            lin[qj0[e]] = max(lin[qj0[e]], lin[qi0[e]] + 1)
        for e in e_in[::-1]:
            lout[qi0[e]] = max(lout[qi0[e]], lout[qj0[e]] + 1)
        path = lin[qi0] + 1 + lout[qj0]
        keep_q = path >= min_qplet_path
        kept = np.unique(np.concatenate([qi0[keep_q], qj0[keep_q]]))
        new = -np.ones(n_raw, np.int64)
        new[kept] = np.arange(len(kept))
        Q = [(int(new[i]), int(new[j]), s) for (i, j, s), k in zip(Q, keep_q) if k]
        T, holes, curv, dth = T[kept], holes[kept], curv[kept], dth[kept]
    else:
        T, holes, curv, dth = T[:0], holes[:0], curv[:0], dth[:0]
    quad = {(min(i, j), max(i, j)): s for i, j, s in Q}
    qi, qj, Sv, ci, cj = [], [], [], [], []
    # pairs of kept triplets sharing at least one hit
    hit_to: Dict[int, List[int]] = {}
    for k, tr in enumerate(T):
        for h in tr:
            hit_to.setdefault(int(h), []).append(k)
    pairs = set()
    for lst in hit_to.values():
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                pairs.add((min(lst[x], lst[y]), max(lst[x], lst[y])))
    for i, j in sorted(pairs):
        ti, tj = T[i], T[j]
        if (ti[1] == tj[0] and ti[2] == tj[1]) or (tj[1] == ti[0] and tj[2] == ti[1]):
            if (i, j) in quad:                   # a kept quadruplet couples its triplets
                qi.append(i)
                qj.append(j)
                Sv.append(quad[(i, j)])
            continue                             # a quadruplet-shaped pair that failed the cuts is no conflict either
        if ti[2] == tj[0] or tj[2] == ti[0]:
            continue                             # quintet (c = d): neither coupled nor in conflict
        ci.append(i)
        cj.append(j)
    as_i = lambda v: np.array(v, dtype=np.int64)
    return TripletQUBO(T, holes, curv, dth, as_i(qi), as_i(qj), np.array(Sv, float), as_i(ci), as_i(cj),
                       float(alpha), float(zeta), int(n_raw), int(n_quad_raw))


def qubo_energy(tq: TripletQUBO, x: np.ndarray) -> float:
    """Energy of a 0/1 vector over all triplets (convention of the module docstring)."""
    x = np.asarray(x, dtype=float)
    return float(tq.alpha * x.sum() - (tq.S * x[tq.qi] * x[tq.qj]).sum() + tq.zeta * (x[tq.ci] * x[tq.cj]).sum())


def component_terms(tq: TripletQUBO, comp: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """Linear terms and quadratic terms (local i < j, b_ij) of one component; local order = sorted global index."""
    comp = np.sort(comp)
    pos = {int(c): k for k, c in enumerate(comp)}
    quad: Dict[Tuple[int, int], float] = {}
    for ai, aj, val in ((tq.qi, tq.qj, -tq.S), (tq.ci, tq.cj, np.full(len(tq.ci), tq.zeta))):
        for a, b, v in zip(ai, aj, val):
            if int(a) in pos and int(b) in pos:
                k, l = sorted((pos[int(a)], pos[int(b)]))
                quad[(k, l)] = quad.get((k, l), 0.0) + float(v)
    return np.full(len(comp), float(tq.alpha)), [(k, l, v) for (k, l), v in sorted(quad.items())]


def terms_energy(linear: np.ndarray, quad: Sequence[Tuple[int, int, float]], x: np.ndarray) -> float:
    return float(np.dot(linear, x) + sum(b * x[i] * x[j] for i, j, b in quad))


def select_hole_free(tq: TripletQUBO) -> np.ndarray:
    """Baseline: every kept triplet without holes."""
    return np.nonzero(tq.holes == 0)[0].astype(np.int64)


def select_greedy_s(tq: TripletQUBO) -> np.ndarray:
    """Baseline: conflict-free greedy in decreasing w_i = sum of the strengths S of the quadruplets containing i
    (ties: fewer holes, then index); a triplet is added if it has no conflict with the already selected ones."""
    w = np.zeros(tq.n)
    np.add.at(w, tq.qi, tq.S)
    np.add.at(w, tq.qj, tq.S)
    nbr: List[set] = [set() for _ in range(tq.n)]
    for a, b in zip(tq.ci, tq.cj):
        nbr[int(a)].add(int(b))
        nbr[int(b)].add(int(a))
    chosen = np.zeros(tq.n, bool)
    for i in np.lexsort((np.arange(tq.n), tq.holes, -w)):
        if not any(chosen[j] for j in nbr[int(i)]):
            chosen[i] = True
    return np.nonzero(chosen)[0].astype(np.int64)


def select_descent(tq: TripletQUBO, start: Optional[np.ndarray] = None) -> np.ndarray:
    """Baseline: steepest-descent single-flip local search on the QUBO energy from the empty selection (or `start`);
    deterministic (ties: lowest index), stops when no single flip lowers the energy by more than 1e-12."""
    from scipy.sparse import csc_matrix
    n = tq.n
    x = np.zeros(n)
    if start is not None:
        x[np.asarray(start, dtype=np.int64)] = 1.0
    if n == 0:
        return np.zeros(0, np.int64)
    i_all, j_all = np.concatenate([tq.qi, tq.ci]), np.concatenate([tq.qj, tq.cj])
    v_all = np.concatenate([-tq.S, np.full(len(tq.ci), tq.zeta)])
    B = csc_matrix((np.concatenate([v_all, v_all]), (np.concatenate([i_all, j_all]), np.concatenate([j_all, i_all]))),
                   shape=(n, n))
    field = tq.alpha + B @ x
    while True:
        dE = (1.0 - 2.0 * x) * field
        k = int(np.argmin(dE))
        if dE[k] >= -1e-12:
            break
        delta = 1.0 - 2.0 * x[k]
        x[k] += delta
        lo, hi = B.indptr[k], B.indptr[k + 1]
        field[B.indices[lo:hi]] += delta * B.data[lo:hi]
    return np.nonzero(x)[0].astype(np.int64)


def true_triplets(ev: EventData, tq: TripletQUBO) -> np.ndarray:
    """Triplets whose three hits belong to one particle id > 0."""
    if tq.n == 0:
        return np.zeros(0, bool)
    p = ev.particle_id[tq.triplets]
    return (p[:, 0] > 0) & (p[:, 0] == p[:, 1]) & (p[:, 1] == p[:, 2])


def sa_bits(q, restarts: int = 16, sweeps: int = 300, seed: int = 0) -> np.ndarray:
    """Single-flip Metropolis SA like acts_qaoa.landscape.simulated_annealing (same schedule: T from 2 max|coef| to
    1e-3 max|coef|), but with sparse columns and bit vectors, so it works for any number of variables (Phase 1
    packs states into int64 indices, i.e. <= 62 variables).  Returns the lowest-energy final state."""
    from scipy.sparse import csc_matrix
    rng = np.random.default_rng(seed)
    n = q.n
    Ws = csc_matrix(q.W + q.W.T)
    scale = float(max(np.abs(q.L).max(), np.abs(Ws.data).max() if Ws.nnz else 0.0, 1e-12))
    X = rng.integers(0, 2, size=(restarts, n)).astype(np.float64)
    for T in np.geomspace(2.0 * scale, 1e-3 * scale, sweeps):
        for k in rng.permutation(n):
            lo, hi = Ws.indptr[k], Ws.indptr[k + 1]
            field = q.L[k] + X[:, Ws.indices[lo:hi]] @ Ws.data[lo:hi]
            dE = (1.0 - 2.0 * X[:, k]) * field
            acc = (dE <= 0) | (rng.random(restarts) < np.exp(-np.clip(dE, 0, None) / T))
            X[acc, k] = 1.0 - X[acc, k]
    return X[int(np.argmin(q.energy_bits(X)))].astype(bool)


def solve(tq: TripletQUBO, method: str, max_vars: int = 20, seed: int = 0, shots: int = 1024) -> Dict[str, object]:
    """Selected triplets: exact (components <= 24), sa (any size), qaoa (components <= max_vars; SA beyond)."""
    from acts_qaoa.landscape import enumerate_energies
    from .selection import solve_qaoa
    sel: List[int] = []
    st = {"n_triplets": tq.n, "n_quadruplets": len(tq.qi), "n_conflicts": len(tq.ci), "n_components": 0,
          "component_sizes": [], "energy": 0.0, "solved_by": {}, "exact_energy_small": 0.0, "energy_small": 0.0}
    for k, comp in enumerate(tq.components()):
        st["n_components"] += 1
        st["component_sizes"].append(len(comp))
        q = tq.sub_qubo(comp)
        m = len(comp)
        if m <= 24:
            E = enumerate_energies(q)
            st["exact_energy_small"] += float(E.min())
        if method in ("exact", "qaoa", "sa") and m <= 24 and (method == "exact" or m == 1):
            x = ((int(np.argmin(E)) >> np.arange(m)) & 1).astype(bool)
            how = "exact"
        elif method == "qaoa" and m <= max_vars:
            i, _, _ = solve_qaoa(q, shots=shots, seed=seed + k)
            x = ((i >> np.arange(m)) & 1).astype(bool)
            how = "qaoa"
        elif method in ("sa", "exact", "qaoa"):
            x = sa_bits(q, seed=seed + k)          # exact: components > 24 are solved by SA (counted separately)
            how = "sa"
        else:
            raise ValueError(f"unknown method {method}")
        st["solved_by"][how] = st["solved_by"].get(how, 0) + 1
        e = float(q.energy_bits(x[None, :].astype(float))[0])
        st["energy"] += e
        if m <= 24:
            st["energy_small"] += e
        sel += [int(c) for c, b in zip(comp, x) if b]
    return {"selected": np.array(sorted(sel), dtype=np.int64), "stats": st}


def postprocess(ev: EventData, tq: TripletQUBO, selected: np.ndarray, min_hits: int = 5):
    """Selected triplets -> doublets -> drop doublets with unresolved conflicts -> chains of >= min_hits hits."""
    if len(selected) == 0:
        return [], np.zeros((0, 2), np.int64)
    T = tq.triplets[selected]
    dbl = np.unique(np.concatenate([T[:, [0, 1]], T[:, [1, 2]]]), axis=0)
    out_deg = np.bincount(dbl[:, 0], minlength=ev.n_hits)
    in_deg = np.bincount(dbl[:, 1], minlength=ev.n_hits)
    ok = (out_deg[dbl[:, 0]] <= 1) & (in_deg[dbl[:, 1]] <= 1)
    dbl = dbl[ok]
    if not len(dbl):
        return [], np.zeros((0, 2), np.int64)
    _, lab = connected_components(coo_matrix((np.ones(len(dbl)), (dbl[:, 0], dbl[:, 1])),
                                             shape=(ev.n_hits, ev.n_hits)), directed=False)
    hits = np.unique(dbl)
    tracks = []
    keep_lab = []
    for L in np.unique(lab[hits]):
        t = hits[lab[hits] == L]
        if len(t) >= min_hits:
            tracks.append(t)
            keep_lab.append(L)
    final = dbl[np.isin(lab[dbl[:, 0]], keep_lab)]
    return tracks, final


def doublet_metrics(ev: EventData, bpos: np.ndarray, final: np.ndarray, pt_min: float = 1.0,
                    min_hits: int = 5) -> Dict[str, float]:
    """Paper Eqs. 6-7 on the final doublets."""
    pid = ev.particle_id
    order = np.lexsort((ev.rho, pid))
    s = order[:-1][pid[order][1:] == pid[order][:-1]]
    d = order[1:][pid[order][1:] == pid[order][:-1]]
    real = pid[s] > 0
    s, d = s[real], d[real]
    nh = np.bincount(ev.particle_rows(pid[pid > 0]), minlength=ev.n_particles)
    rows = ev.particle_rows(pid[s])
    good = (ev.particles["pt"][rows] > pt_min) & (nh[rows] >= min_hits)
    key = lambda a, b: np.minimum(a, b) * ev.n_hits + np.maximum(a, b)
    true_all = key(s, d)
    true_good = true_all[good]
    rk = key(final[:, 0], final[:, 1]) if len(final) else np.zeros(0, np.int64)
    matched_good = np.isin(rk, true_good)
    matched_other = np.isin(rk, true_all[~good])
    n_rec = len(rk)
    return {"D_true": int(len(true_good)), "D_rec": n_rec, "D_matched": int(matched_good.sum()),
            "D_oa": int(matched_other.sum()),
            "efficiency": float(matched_good.sum() / max(len(true_good), 1)),
            "purity": float(matched_good.sum() / max(n_rec - matched_other.sum(), 1))}


def analyse_components(ev: EventData, tq: TripletQUBO, baselines: Dict[str, np.ndarray], meta: dict,
                       rows: Optional[list] = None, export_dir: Optional[Path] = None) -> Dict[str, float]:
    """Per connected component: energies of the baseline selections restricted to it (always), and, if `rows` /
    `export_dir` are given, the components.csv rows and the JSON files of the module docstring.  Returns the summed
    energy of each baseline over the components with <= 24 variables."""
    from acts_qaoa.landscape import enumerate_energies
    is_true = true_triplets(ev, tq)
    x_base = {k: np.isin(np.arange(tq.n), s).astype(float) for k, s in baselines.items()}
    small = {k: 0.0 for k in baselines}
    need_exact = rows is not None or export_dir is not None
    for cid, comp in enumerate(tq.components()):
        comp = np.sort(comp)
        m = len(comp)
        lin, quad = component_terms(tq, comp)
        e_base = {k: terms_energy(lin, quad, x[comp]) for k, x in x_base.items()}
        if m <= 24:
            for k, e in e_base.items():
                small[k] += e
        if not need_exact:
            continue
        in_comp = np.zeros(tq.n, bool)
        in_comp[comp] = True
        n_q = int(in_comp[tq.qi].sum())
        n_c = int(in_comp[tq.ci].sum())
        exact_e = x_ex = None
        if m <= 24:
            E = enumerate_energies(tq.sub_qubo(comp))
            x_ex = ((int(np.argmin(E)) >> np.arange(m)) & 1).astype(int)
            exact_e = float(E.min())
        hf = x_base["hole-free"][comp].astype(int) if "hole-free" in x_base else None
        if rows is not None:
            rows.append({**meta, "component": cid, "n_variables": m, "n_quadruplet_couplings": n_q, "n_conflicts": n_c,
                         "n_true_triplets": int(is_true[comp].sum()), "n_fake_triplets": int((~is_true[comp]).sum()),
                         "exact_energy": exact_e, "holefree_energy": e_base.get("hole-free"),
                         "greedy_energy": e_base.get("greedy-S"), "descent_energy": e_base.get("hole-free+descent"),
                         "greedy_descent_energy": e_base.get("greedy-S+descent"),
                         "trivial": None if x_ex is None else bool(np.array_equal(x_ex, hf)),
                         "holefree_optimal": None if x_ex is None else bool(e_base["hole-free"] <= exact_e + 1e-9)})
        if export_dir is not None and m <= 24:
            name = f"{meta['event']}_s{meta['sector']}_f{meta['frac']:g}_c{cid}.json".replace("/", "_").replace("\\", "_")
            tri = tq.triplets[comp]
            doc = {"meta": {**meta, "component": cid, "alpha": tq.alpha, "zeta": tq.zeta},
                   "triplets": tri.tolist(), "triplets_hit_id": ev.hit_id[tri].tolist(),
                   "linear": lin.tolist(), "quadratic": [[int(i), int(j), float(b)] for i, j, b in quad],
                   "exact_energy": exact_e, "exact_selection": x_ex.tolist(), "true_triplet": is_true[comp].tolist()}
            export_dir = Path(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            with open(export_dir / name, "x") as fh:                 # never overwrite
                json.dump(doc, fh)
    return small


def run_study(events, barrel: Sequence[int], sectors: Sequence[Tuple[float, float]], fracs: Sequence[float],
              methods: Sequence[str] = ("exact", "sa", "qaoa"), max_vars: int = 16, seed: int = 0,
              component_rows: Optional[list] = None, export_dir: Optional[Path] = None) -> List[dict]:
    """Every event x sector x particle fraction: QUBO size, component sizes, doublet efficiency / purity per solver
    and per classical baseline.  component_rows / export_dir: see the module docstring."""
    import time
    rows = []
    for ev in events:
        for s_idx, (phi0, dphi) in enumerate(sectors):
            for f in fracs:
                sub, bpos = barrel_sector(ev, barrel, phi0, dphi, f, seed)
                ds, dd = make_doublets(sub, bpos)
                tq = build_triplet_qubo(sub, bpos, ds, dd)
                sizes = np.array([len(c) for c in tq.components()]) if tq.n else np.zeros(0, int)
                base = {"event_id": ev.event_id, "phi0": phi0, "dphi": dphi, "frac": f, "n_hits": sub.n_hits,
                        "n_particles": int(len(np.unique(sub.particle_id[sub.particle_id > 0]))),
                        "n_doublets": int(len(ds)), "n_triplets_raw": tq.n_raw, "n_quadruplets_raw": tq.n_quad_raw,
                        "n_triplets": tq.n, "n_quadruplets": int(len(tq.qi)),
                        "n_conflicts": int(len(tq.ci)), "n_components": int(len(sizes)),
                        "max_component": int(sizes.max()) if len(sizes) else 0,
                        "triplets_in_components_le24": float(sizes[sizes <= 24].sum() / max(sizes.sum(), 1)),
                        "components_le24": float(np.mean(sizes <= 24)) if len(sizes) else None}
                # reference without a QUBO: every triplet of the QUBO (cuts and pre-filter passed)
                tr_all, fin_all = postprocess(sub, tq, np.arange(tq.n))
                rows.append({**base, "method": "all-kept-triplets", **doublet_metrics(sub, bpos, fin_all),
                             "n_tracks": len(tr_all), "seconds": 0.0})
                # classical baselines on the same QUBO, scored identically
                t0 = time.time()
                hole_free, greedy = select_hole_free(tq), select_greedy_s(tq)
                base_sel = {"hole-free": hole_free, "greedy-S": greedy,
                            "hole-free+descent": select_descent(tq, start=hole_free),
                            "greedy-S+descent": select_descent(tq, start=greedy)}
                secs = round(time.time() - t0, 2)
                meta = {"event": ev.event_id, "sector": s_idx, "phi0": phi0, "dphi": dphi, "frac": f}
                small = analyse_components(sub, tq, base_sel, meta, component_rows, export_dir)
                for name, sel in base_sel.items():
                    tracks, final = postprocess(sub, tq, sel)
                    rows.append({**base, "method": name, **doublet_metrics(sub, bpos, final), "n_tracks": len(tracks),
                                 "energy": qubo_energy(tq, np.isin(np.arange(tq.n), sel)), "energy_small": small[name],
                                 "seconds": secs})
                for m in methods:
                    t0 = time.time()
                    r = solve(tq, m, max_vars=max_vars, seed=seed)
                    tracks, final = postprocess(sub, tq, r["selected"])
                    rows.append({**base, "method": m, **doublet_metrics(sub, bpos, final), "n_tracks": len(tracks),
                                 "energy": r["stats"]["energy"], "energy_small": r["stats"]["energy_small"],
                                 "exact_energy_small": r["stats"]["exact_energy_small"],
                                 "solved_by": dict(r["stats"]["solved_by"]), "seconds": round(time.time() - t0, 2)})
    return rows

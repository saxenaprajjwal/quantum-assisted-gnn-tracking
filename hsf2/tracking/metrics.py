"""Track matching and performance metrics.

ACTS-style (same definitions as the CKF numbers of Phase 1, TrackFinderPerformanceWriter):
  a track is *matched* to its majority particle if that particle has >= match_min of the track's hits
  (noise hits count in the denominator, never as majority); otherwise it is *fake*;
  efficiency    = target particles with >= 1 matched track / target particles
  fake rate     = fake tracks / tracks
  duplicate rate = (matched tracks - distinct matched particles) / tracks
Double majority (Exa.TrkX / TrackML): additionally > 50 % of the particle's hits are in the track.
TrackML score (Amrouche et al. 2019): every hit in at most one track; a track counts if > 50 % of its hits
come from its majority particle and > 50 % of that particle's hits are in it; score = sum of the TrackML
weights of their common hits (normalised to the weight of the hits that entered the reconstruction).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from ..data.event import EventData, particle_hit_counts, target_mask

COUNT_KEYS = ("n_tracks", "n_matched", "n_fake", "n_duplicate", "n_targets", "n_targets_found",
              "n_targets_found_dm", "n_tracks_dm", "trackml_score_raw", "trackml_weight_total", "n_hits")


def match_tracks(ev: EventData, tracks: List[np.ndarray], match_min: float = 0.5) -> Dict[str, np.ndarray]:
    """Per-track majority particle, its number of hits in the track, track size, purity, matched, dm."""
    nt = len(tracks)
    if nt == 0:
        z = np.zeros(0)
        return {"major": z.astype(np.int64), "n_major": z, "n_hits": z, "purity": z, "matched": z.astype(bool),
                "dm": z.astype(bool), "n_noise": z}
    sizes = np.array([len(t) for t in tracks], dtype=np.int64)
    tid = np.repeat(np.arange(nt), sizes)
    pid = ev.particle_id[np.concatenate(tracks)]
    real = pid != 0
    major = np.zeros(nt, np.int64)
    n_major = np.zeros(nt, np.int64)
    if real.any():
        t_r, p_r = tid[real], pid[real]
        order = np.lexsort((p_r, t_r))
        t_r, p_r = t_r[order], p_r[order]
        brk = np.ones(len(t_r), bool)
        brk[1:] = (t_r[1:] != t_r[:-1]) | (p_r[1:] != p_r[:-1])
        gstart = np.nonzero(brk)[0]
        gcount = np.diff(np.append(gstart, len(t_r)))
        gt, gp = t_r[gstart], p_r[gstart]
        # per track: the group with the largest count (ties -> smallest particle id: groups are sorted by pid)
        order2 = np.lexsort((gp, -gcount, gt))
        gt2 = gt[order2]
        first = np.ones(len(gt2), bool)
        first[1:] = gt2[1:] != gt2[:-1]
        sel = order2[first]
        major[gt[sel]] = gp[sel]
        n_major[gt[sel]] = gcount[sel]
    n_noise = np.bincount(tid[~real], minlength=nt)
    purity = n_major / np.maximum(sizes, 1)
    matched = (major != 0) & (purity >= match_min)
    nh = particle_hit_counts(ev)
    rows = ev.particle_rows(major)
    p_hits = np.where(rows >= 0, nh[np.maximum(rows, 0)], 0)
    dm = (major != 0) & (purity > 0.5) & (n_major > 0.5 * np.maximum(p_hits, 1))
    return {"major": major, "n_major": n_major.astype(float), "n_hits": sizes.astype(float), "purity": purity,
            "matched": matched, "dm": dm, "n_noise": n_noise.astype(float)}


def trackml_score(ev: EventData, tracks: List[np.ndarray]) -> float:
    """Official TrackML score of an assignment (shared hits go to the first track that has them)."""
    return float(trackml_track_weights(ev, tracks).sum())


def trackml_track_weights(ev: EventData, tracks: List[np.ndarray]) -> np.ndarray:
    """Contribution of every track to the TrackML score (0 for tracks that do not count).

    A shared hit belongs to the first track that has it, so the contribution of track k depends only on tracks
    0..k: dropping tracks from the END of the list leaves the others unchanged (used by the grid evaluator).
    """
    n = ev.n_hits
    out = np.zeros(len(tracks), np.float64)
    owner = np.full(n, -1, np.int64)
    if tracks:
        # owner of a hit = the first (lowest-index) track that contains it
        hits = np.concatenate(tracks).astype(np.int64)
        tids = np.repeat(np.arange(len(tracks), dtype=np.int64), [len(t) for t in tracks])
        order = np.lexsort((tids, hits))
        uh, first = np.unique(hits[order], return_index=True)
        owner[uh] = tids[order][first]
    assigned = np.nonzero(owner >= 0)[0]
    if len(assigned) == 0:
        return out
    tid, pid, w = owner[assigned], ev.particle_id[assigned], ev.weight[assigned].astype(np.float64)
    order = np.lexsort((pid, tid))
    tid, pid, w = tid[order], pid[order], w[order]
    brk = np.ones(len(tid), bool)
    brk[1:] = (tid[1:] != tid[:-1]) | (pid[1:] != pid[:-1])
    gstart = np.nonzero(brk)[0]
    gcount = np.diff(np.append(gstart, len(tid)))
    gweight = np.add.reduceat(w, gstart)
    gt, gp = tid[gstart], pid[gstart]
    track_size = np.bincount(tid, minlength=len(tracks))
    order2 = np.lexsort((gp, -gcount, gt))
    first = np.ones(len(order2), bool)
    first[1:] = gt[order2][1:] != gt[order2][:-1]
    sel = order2[first]
    # number of hits of every particle among the event's hits (noise: all noise hits)
    upid, ucnt = np.unique(ev.particle_id, return_counts=True)
    p_tot = ucnt[np.searchsorted(upid, gp[sel])]
    good = (gcount[sel] > 0.5 * track_size[gt[sel]]) & (gcount[sel] > 0.5 * p_tot)
    out[gt[sel][good]] = gweight[sel][good]
    return out


FAKE_KEYS = ("n_fake_noise", "n_fake_mixed", "n_fake_wrong")


def fake_breakdown(m: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Split the fake tracks of match_tracks() (the definition of 'fake' is unchanged), by precedence:
    n_fake_noise  every hit is a noise hit; n_fake_mixed  real and noise hits, below match_min;
    n_fake_wrong  only real hits, no particle has match_min of them (wrong / split-between-particles majority).
    The three sum to n_fake."""
    fake = ~m["matched"]
    all_noise = fake & (m["major"] == 0)
    mixed = fake & ~all_noise & (m["n_noise"] > 0)
    wrong = fake & ~all_noise & ~mixed
    return {"n_fake_noise": int(all_noise.sum()), "n_fake_mixed": int(mixed.sum()), "n_fake_wrong": int(wrong.sum())}


def edge_pr_summary(y: np.ndarray, s: np.ndarray, recalls=(0.95, 0.99)) -> Dict[str, float]:
    """Edge-classifier precision-recall summary (average precision, purity at fixed recall) over all kept edges."""
    y = np.asarray(y, bool)
    s = np.asarray(s, np.float64)
    n_true = int(y.sum())
    out: Dict[str, float] = {"edge_pr_auc": float("nan")}
    for r in recalls:
        out[f"edge_purity_at_recall_{int(round(r * 100))}"] = float("nan")
    if n_true == 0 or len(y) == 0:
        return out
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / n_true
    out["edge_pr_auc"] = float((prec * ys).sum() / n_true)                 # average precision
    for r in recalls:
        k = int(np.searchsorted(rec, r - 1e-12))                           # first index with recall >= r
        out[f"edge_purity_at_recall_{int(round(r * 100))}"] = float(prec[min(k, len(prec) - 1)]) if k < len(rec) else float("nan")
    return out


def event_counts(ev: EventData, tracks: List[np.ndarray], tcfg) -> Dict[str, float]:
    m = match_tracks(ev, tracks, tcfg.match_min)
    targets = target_mask(ev, tcfg)
    tpid = ev.particles["particle_id"][targets]
    matched_pids = np.unique(m["major"][m["matched"]])
    dm_pids = np.unique(m["major"][m["dm"]])
    n_tr = len(tracks)
    n_match = int(m["matched"].sum())
    return {"n_tracks": n_tr, "n_matched": n_match, "n_fake": n_tr - n_match,
            "n_duplicate": n_match - len(matched_pids), "n_targets": int(targets.sum()),
            "n_targets_found": int(np.isin(tpid, matched_pids).sum()),
            "n_targets_found_dm": int(np.isin(tpid, dm_pids).sum()), "n_tracks_dm": int(m["dm"].sum()),
            "trackml_score_raw": trackml_score(ev, tracks), "trackml_weight_total": float(ev.weight.sum()),
            "n_hits": int(ev.n_hits), **fake_breakdown(m)}


def summarize(counts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Ratios from summed counts over events (events with many tracks weigh more, as in ACTS)."""
    tot = {k: 0.0 for k in COUNT_KEYS}
    n_ev = 0
    for c in counts:
        n_ev += 1
        for k in COUNT_KEYS:
            tot[k] += float(c.get(k, 0.0))
    nt = max(tot["n_tracks"], 1.0)
    out = {"n_events": n_ev, **{k: tot[k] for k in COUNT_KEYS},
           "efficiency": tot["n_targets_found"] / max(tot["n_targets"], 1.0),
           "efficiency_dm": tot["n_targets_found_dm"] / max(tot["n_targets"], 1.0),
           "fakerate": tot["n_fake"] / nt if tot["n_tracks"] else 0.0,
           "duplicaterate": tot["n_duplicate"] / nt if tot["n_tracks"] else 0.0,
           "trackml_score": tot["trackml_score_raw"] / max(tot["trackml_weight_total"], 1e-12)}
    return out


def efficiency_by(ev_list: List[EventData], tracks_list: List[List[np.ndarray]], tcfg, key: str,
                  bins: np.ndarray) -> Dict[str, list]:
    """Efficiency of target particles in bins of a particle property (pt, eta)."""
    found_all, val_all = [], []
    for ev, tracks in zip(ev_list, tracks_list):
        m = match_tracks(ev, tracks, tcfg.match_min)
        targets = target_mask(ev, tcfg)
        tpid = ev.particles["particle_id"][targets]
        found_all.append(np.isin(tpid, np.unique(m["major"][m["matched"]])))
        val_all.append(ev.particles[key][targets])
    found = np.concatenate(found_all) if found_all else np.zeros(0, bool)
    vals = np.concatenate(val_all) if val_all else np.zeros(0)
    idx = np.digitize(vals, bins) - 1
    n = np.bincount(idx[(idx >= 0) & (idx < len(bins) - 1)], minlength=len(bins) - 1)
    k = np.bincount(idx[(idx >= 0) & (idx < len(bins) - 1)], weights=found[(idx >= 0) & (idx < len(bins) - 1)],
                    minlength=len(bins) - 1)
    return {"lo": bins[:-1].tolist(), "hi": bins[1:].tolist(), "n": n.tolist(),
            "eff": [float(a / b) if b else float("nan") for a, b in zip(k, n)]}

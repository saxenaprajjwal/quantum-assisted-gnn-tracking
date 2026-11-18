"""GNN versus ACTS CKF on exactly the same ODD events: one matching and metric definition, applied to both.

The tracks of both methods are brought to the same form (arrays of measurement rows of the FULL exported event, no hit
selection) and go through the same function, counts_for_tracks(): majority-particle matching (metrics.match_tracks)
and the Phase-1 target definition.  So the numbers of the two methods differ only through the tracks.

Phase-1 target definition (ckf_odd.py run_ckf, addDigiParticleSelection): charged particles with pT >= 1 GeV,
|eta| < 3 and >= 9 measurements; efficiency = targets with >= 1 matched track / targets.

Differences between hsf2.tracking.metrics (evaluate, QX loops) and ACTS / Phase 1, line by line.  The ACTS side is
written from the ACTS sources (TrackTruthMatcher, TrackFinderPerformanceWriter, ParticleSelector) as I know them and is
NOT checkable on the PC (no ACTS here): compare_ckf's cross-check (Python numbers of the exported CKF tracks against
the CKF's own performance_finding_ckf.root, which Phase 1 used) is what verifies it on the real export.

 1. Majority particle.  ACTS: for every track the measurement states (outliers and holes excluded) are counted per
    contributing particle of the measurement-particle map; metrics: one particle per hit (the hit's truth).  Same
    on the ODD, where the digitiser smears single simhits (one particle per measurement, no merged clusters).
    Ties: ACTS keeps the first maximum of its map, metrics the smallest particle id (negligible).
 2. Matched.  ACTS: nMajorityHits / nTrackMeasurements >= truthMatchProbMin (0.5); metrics: purity >= match_min
    (0.5), noise counted in the denominator.  Same.  No noise on the ODD.
 3. Fake rate = fake tracks / all tracks; duplicate rate = (matched tracks - distinct matched particles) / all
    tracks; both are ratios of sums over events.  Same in metrics.summarize and in the ACTS writer (sums over the
    run).  Per-event ratios are reported here, but totals are never means of per-event ratios.
 4. Efficiency.  ACTS: particles of the selected collection with >= 1 matched track; metrics: the same, with the target
    set built by metrics.target_mask, which differs: pt > or >= pt_min (here >=, as the ACTS selector), |eta| <= eta_max
    (ACTS: < 3), default eta_max 4 and min_hits 5 (TrackML study) against 3 and 9 here, hits counted among the
    SELECTED hits (S0) against all measurements of the event here, and no charge test (neutrals have no hits).
    compare-ckf therefore does not use target_mask / event_counts but phase1_targets() below on the full event.
 5. Hit selection.  evaluate runs the GNN on S0 hits (hits of particles with truth pT >= 1 GeV) -- a truth-informed
    hit filter the CKF does not have; matching there happens inside the reduced event.  Here the GNN tracks are mapped
    back to the full event, and the report states the hit selection of the model; with S0 the GNN has an advantage.
 6. Tracks whose majority particle is not a target (pT < 1 GeV, < 9 hits, secondary): here counted as matched (not
    fake) for both methods.  ACTS counts them against the particles of its selected collection; the cross-check shows
    whether this matters (the CKF itself requires pT > 1 GeV).
 7. Particle momenta: pT, eta of the particle table exported by odd_export.py (initial state of the particle),
    which is what the ParticleSelector of Phase 1 sees if its input is the initial-state collection; the export
    manifest records the collection name that was used.
 8. Hit positions: GNN = simhit truth position, CKF = smeared measurement (see hsf2.data.odd).
 9. Timing: GNN = process time of inference + track building (graph construction listed separately), CKF = seeding +
    track finding time per event from the Sequencer's timing.csv; different code, same CPU class only if both ran on
    the same kind of node; quoted as indicative.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import TargetConfig
from ..data import odd
from ..data.event import EventData, particle_hit_counts
from .metrics import match_tracks

PHASE1_TARGETS = TargetConfig(pt_min=1.0, eta_max=3.0, min_hits=9, match_min=0.5)
# n_matched_target / n_duplicate_target: the tracks whose majority particle is a target, and their duplicates.  The
# all-track duplicate rate (ACTS's definition) also counts the tracks of non-target particles (pT < 1 GeV loopers,
# secondaries); a method that builds tracks for every particle (a GNN on all hits) has many more of them than the CKF,
# whose own pT cut keeps it near the targets, so only the target duplicate rate compares the two on the same particles.
# n_targets_found_dm: targets with a double-majority track (purity > 0.5 AND more than half of the particle's
# measurements on the track), so a short fragment does not count as finding the particle; n_hits_target_tracks: the
# measurements of the tracks matched to a target (their mean length).
COUNT_COLS = ("n_tracks", "n_matched", "n_fake", "n_duplicate", "n_targets", "n_targets_found", "n_matched_target",
              "n_duplicate_target", "n_targets_found_dm", "n_hits_target_tracks")


# ---------------------------------------------------------------------------
# the one definition
# ---------------------------------------------------------------------------
def phase1_targets(ev: EventData, tcfg: TargetConfig = PHASE1_TARGETS) -> np.ndarray:
    """Mask over the particle table: pT >= pt_min, |eta| < eta_max, >= min_hits measurements (full event)."""
    p = ev.particles
    return ((p["particle_id"] != 0) & (p["pt"] >= tcfg.pt_min) & (np.abs(p["eta"]) < tcfg.eta_max)
            & (particle_hit_counts(ev) >= tcfg.min_hits))


def counts_for_tracks(ev: EventData, tracks: List[np.ndarray], tcfg: TargetConfig = PHASE1_TARGETS) -> Dict[str, int]:
    """Counts of one event for tracks given as row indices into `ev` (the full event)."""
    for t in tracks:
        if len(t) and (int(np.min(t)) < 0 or int(np.max(t)) >= ev.n_hits):
            raise IndexError("track hit index outside the event")
    m = match_tracks(ev, tracks, tcfg.match_min)
    targets = phase1_targets(ev, tcfg)
    tpid = ev.particles["particle_id"][targets]
    matched_pids = np.unique(m["major"][m["matched"]])
    n_tr, n_match = len(tracks), int(m["matched"].sum())
    n_found = int(np.isin(tpid, matched_pids).sum())
    tmask = m["matched"] & np.isin(m["major"], tpid)
    n_match_t = int(tmask.sum())
    n_found_dm = int(np.isin(tpid, np.unique(m["major"][m["dm"]])).sum())
    return {"n_tracks": n_tr, "n_matched": n_match, "n_fake": n_tr - n_match,
            "n_duplicate": n_match - len(matched_pids), "n_targets": int(targets.sum()),
            "n_targets_found": n_found, "n_matched_target": n_match_t, "n_duplicate_target": n_match_t - n_found,
            "n_targets_found_dm": n_found_dm, "n_hits_target_tracks": int(m["n_hits"][tmask].sum())}


# ---------------------------------------------------------------------------
# one track selection for both methods (the CKF's own cuts, on a helix refit of the hits)
# ---------------------------------------------------------------------------
B_FIELD_T = 2.0              # ODD solenoid, as in Phase 1's ckf_odd.py (ConstantBField 2 T)
# Phase 1's ckf_odd.py, ckf-config "odd": TrackSelectorConfig(pt=(1 GeV, None), absEta=(None, 3), loc0=(-4, 4) mm,
# nMeasurementsMin=7); holes and outliers are CKF concepts without a GNN counterpart and are not applied.
CKF_SELECTION = {"pt_min": 1.0, "abs_eta_max": 3.0, "d0_max": 4.0, "n_meas_min": 7}


def helix_parameters(x, y, z, b_field: float = B_FIELD_T) -> Dict[str, float]:
    """pT (GeV), transverse impact parameter d0 (mm, to the beam line) and eta of a track from its hit positions:
    algebraic circle fit in x-y (radius R, pT = 0.2998 B R), then z linear in the transverse arc length
    s = 2R asin(rho / 2R) from the beam line.  Fewer than 3 hits or collinear hits: NaN."""
    x, y, z = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
    nan = {"pt": float("nan"), "d0": float("nan"), "eta": float("nan")}
    if len(x) < 3:
        return nan
    A = np.column_stack([x, y, np.ones(len(x))])
    rhs = -(x * x + y * y)
    try:
        (a, b, c), *_ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return nan
    xc, yc = -a / 2.0, -b / 2.0
    r2 = xc * xc + yc * yc - c
    if not np.isfinite(r2) or r2 <= 0:
        return nan
    R = float(np.sqrt(r2))
    d0 = abs(float(np.hypot(xc, yc)) - R)
    rho = np.hypot(x, y)
    s = 2.0 * R * np.arcsin(np.clip(rho / (2.0 * R), 0.0, 1.0))
    if np.ptp(s) <= 0:
        return nan
    cot = float(np.polyfit(s, z, 1)[0])
    return {"pt": 0.299792458e-3 * b_field * R, "d0": d0, "eta": float(np.arcsinh(cot))}


def select_tracks(ev: EventData, tracks: List[np.ndarray], sel: Optional[Dict[str, float]] = None,
                  b_field: float = B_FIELD_T, fit_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """The tracks (rows of `ev`) that pass the selection on a helix refit of their hits: pT >= pt_min,
    |eta| < abs_eta_max, |d0| < d0_max, at least n_meas_min hits.  The same function is applied to both methods.
    fit_mask (bool per row of `ev`): only these hits enter the refit -- on measured positions, the hits whose three
    coordinates are measured (a one-coordinate strip hit sits on the strip centre line, tens of mm from the true
    point along the strip); every hit still counts towards n_meas_min; fewer than 3 such hits: not selected."""
    sel = CKF_SELECTION if sel is None else sel
    x, y, z = ev.hits["x"], ev.hits["y"], ev.hits["z"]
    out = []
    for t in tracks:
        t = np.asarray(t, dtype=np.int64)
        if len(t) < sel["n_meas_min"]:
            continue
        f = t if fit_mask is None else t[fit_mask[t]]
        p = helix_parameters(x[f], y[f], z[f], b_field)
        if (np.isfinite(p["pt"]) and p["pt"] >= sel["pt_min"] and abs(p["eta"]) < sel["abs_eta_max"]
                and p["d0"] < sel["d0_max"]):
            out.append(t)
    return out


def rows_of_hit_ids(ev: EventData, hit_ids: np.ndarray) -> np.ndarray:
    """Rows of the full event for hit ids (EventData.hit_id); every id must exist."""
    ids = ev.hits["hit_id"]
    order = np.argsort(ids, kind="stable")
    pos = np.clip(np.searchsorted(ids[order], hit_ids), 0, max(len(ids) - 1, 0))
    if len(ids) == 0 or np.any(ids[order][pos] != hit_ids):
        raise KeyError(f"{ev.event_id}: track hits not among the event's hit ids")
    return order[pos].astype(np.int64)


def gnn_tracks_full(ev: EventData, es: EventData, tracks: List[np.ndarray]) -> List[np.ndarray]:
    """GNN tracks (rows of the hit-selected event es) -> rows of the full event."""
    return [rows_of_hit_ids(ev, es.hits["hit_id"][np.asarray(t, dtype=np.int64)]) for t in tracks]


def ckf_tracks_full(ev: EventData, export_dir: Path) -> List[np.ndarray]:
    nr = int(ev.meta["event_nr"])
    return [rows_of_hit_ids(ev, t) for t in odd.read_ckf_tracks(export_dir, nr)]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def ratios(c: Dict[str, float]) -> Dict[str, float]:
    nt, ng, nmt = c["n_tracks"], c["n_targets"], c.get("n_matched_target", 0)
    return {"efficiency": c["n_targets_found"] / ng if ng else float("nan"),
            "efficiency_dm": c.get("n_targets_found_dm", 0) / ng if ng else float("nan"),
            "fakerate": c["n_fake"] / nt if nt else float("nan"),
            "duplicaterate": c["n_duplicate"] / nt if nt else float("nan"),
            "duplicaterate_target": c.get("n_duplicate_target", 0) / nmt if nmt else float("nan")}


def wilson(k: float, n: float, z: float = 1.959964) -> List[float]:
    """Wilson score interval of a binomial proportion k/n (95 % by default)."""
    if n <= 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Per-event paired bootstrap of eff / fake / dup of two methods; a, b = (n_events, len(COUNT_COLS)) counts of the
    same events.  The SAME resampled events are used for both methods, so the interval of the difference (a - b)
    contains the event-to-event correlation.  Ratios are ratios of sums over the resampled events."""
    if a.shape != b.shape or a.shape[0] == 0:
        raise ValueError("bootstrap needs the same non-empty events for both methods")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.shape[0], size=(n_boot, a.shape[0]))
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    col = {c: i for i, c in enumerate(COUNT_COLS)}

    def ratio_matrix(x):
        s = x[idx].sum(axis=1).astype(np.float64)                      # (n_boot, ncols)
        with np.errstate(invalid="ignore", divide="ignore"):
            return {"efficiency": s[:, col["n_targets_found"]] / s[:, col["n_targets"]],
                    "efficiency_dm": s[:, col["n_targets_found_dm"]] / s[:, col["n_targets"]],
                    "fakerate": s[:, col["n_fake"]] / s[:, col["n_tracks"]],
                    "duplicaterate": s[:, col["n_duplicate"]] / s[:, col["n_tracks"]],
                    "duplicaterate_target": s[:, col["n_duplicate_target"]] / s[:, col["n_matched_target"]]}
    ra, rb = ratio_matrix(a), ratio_matrix(b)
    out = {}
    with warnings.catch_warnings():                       # a method without tracks has NaN fake / dup rates: stays NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in ra:
            d = ra[k] - rb[k]
            out[k] = {"a": [float(np.nanpercentile(ra[k], lo)), float(np.nanpercentile(ra[k], hi))],
                      "b": [float(np.nanpercentile(rb[k], lo)), float(np.nanpercentile(rb[k], hi))],
                      "diff_a_minus_b": [float(np.nanpercentile(d, lo)), float(np.nanpercentile(d, hi))]}
    return out


def total_counts(rows: Sequence[Dict[str, float]]) -> Dict[str, int]:
    return {c: int(sum(r[c] for r in rows)) for c in COUNT_COLS}


def method_summary(rows: Sequence[Dict[str, float]]) -> dict:
    """Totals, ratios and Wilson intervals (tracks for fake / dup, targets for efficiency; tracks of one event are
    correlated, so the bootstrap intervals are the ones to quote; Wilson is the optimistic binomial one)."""
    tot = total_counts(rows)
    r = ratios(tot)
    return {"n_events": len(rows), **tot, **r,
            "efficiency_wilson": wilson(tot["n_targets_found"], tot["n_targets"]),
            "efficiency_dm_wilson": wilson(tot["n_targets_found_dm"], tot["n_targets"]),
            "hits_per_target_track": tot["n_hits_target_tracks"] / tot["n_matched_target"] if tot["n_matched_target"]
            else float("nan"),
            "fakerate_wilson": wilson(tot["n_fake"], tot["n_tracks"]),
            "duplicaterate_wilson": wilson(tot["n_duplicate"], tot["n_tracks"]),
            "duplicaterate_target_wilson": wilson(tot["n_duplicate_target"], tot["n_matched_target"]),
            "tracks_per_event": tot["n_tracks"] / max(len(rows), 1),
            "target_tracks_per_event": tot["n_matched_target"] / max(len(rows), 1),
            "nontarget_tracks_per_event": (tot["n_matched"] - tot["n_matched_target"]) / max(len(rows), 1),
            "fakes_per_event": tot["n_fake"] / max(len(rows), 1)}


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def export_map(export_dirs: Sequence[Path]) -> Dict[str, Path]:
    """export_id -> folder (from the manifests)."""
    out = {}
    for d in export_dirs:
        man = odd.require_complete(d)
        out[man["export_id"]] = Path(d)
    return out


def ckf_timing(export_dir: Path) -> Optional[float]:
    """Seeding + track-finding time per event (s) of the exported CKF run: timing.csv of its Sequencer."""
    import pandas as pd
    for name, sep in (("timing.csv", ","), ("timing.tsv", "\t")):
        p = Path(export_dir) / "ckf" / name
        if p.is_file():
            df = pd.read_csv(p, sep=sep)
            if "identifier" in df.columns and "time_perevent_s" in df.columns:
                sel = df["identifier"].astype(str).str.contains(r"SeedingAlgorithm|TrackFindingAlgorithm", regex=True)
                if sel.any():
                    return float(df.loc[sel, "time_perevent_s"].sum())
    return None


def compare_events(pipe, params, events: Sequence[EventData], exports: Dict[str, Path],
                   tcfg: TargetConfig = PHASE1_TARGETS, selection: Optional[Dict[str, float]] = None,
                   fit_skip_layers: Sequence[int] = ()) -> dict:
    """GNN and CKF counts of the same events.  Returns {'gnn': rows, 'ckf': rows, 'timing': {...}}; with `selection`
    (select_tracks) also 'gnn_sel' and 'ckf_sel': the counts after the same selection on both methods' tracks.
    fit_skip_layers: layer indices whose hits do not enter the selection's refit (odd.one_coordinate_layers; for
    measured positions), the same for both methods."""
    skip = np.asarray(sorted(int(v) for v in fit_skip_layers), dtype=np.int64)
    g_rows, c_rows, gs_rows, cs_rows = [], [], [], []
    t_gnn = t_build = t_graph = 0.0
    for ev in events:
        if ev.meta.get("source") != "odd":
            raise ValueError(f"{ev.event_id}: not an ODD event (meta source {ev.meta.get('source')!r})")
        ex = exports.get(ev.meta.get("export_id"))
        if ex is None:
            raise KeyError(f"{ev.event_id} comes from export {ev.meta.get('export_id')!r}: pass its folder with "
                           f"--export (given: {sorted(exports)})")
        tracks, counts, (es, *_rest) = pipe.run_event(ev, params)
        t_gnn += counts.get("t_gnn_s", 0.0)
        t_build += counts.get("t_build_s", 0.0)
        t_graph += counts.get("t_graph_s", 0.0)
        gt, ct = gnn_tracks_full(ev, es, tracks), ckf_tracks_full(ev, ex)
        g = counts_for_tracks(ev, gt, tcfg)
        c = counts_for_tracks(ev, ct, tcfg)
        if g["n_targets"] != c["n_targets"]:
            raise AssertionError(f"{ev.event_id}: target sets differ ({g['n_targets']} vs {c['n_targets']})")
        g_rows.append({"event_id": ev.event_id, **g})
        c_rows.append({"event_id": ev.event_id, **c})
        if selection is not None:
            fm = ~np.isin(ev.hits["layer"], skip) if len(skip) else None
            for rows, trk in ((gs_rows, gt), (cs_rows, ct)):
                rows.append({"event_id": ev.event_id,
                             **counts_for_tracks(ev, select_tracks(ev, trk, selection, fit_mask=fm), tcfg)})
    n = max(len(events), 1)
    tc = {Path(p).name: ckf_timing(p) for p in exports.values()}
    out = {"gnn": g_rows, "ckf": c_rows,
           "timing": {"gnn_infer_build_s_per_event": (t_gnn + t_build) / n, "gnn_graph_s_per_event": t_graph / n,
                      "ckf_seed_find_s_per_event": tc}}
    if selection is not None:
        out.update(gnn_sel=gs_rows, ckf_sel=cs_rows,
                   selection={**dict(selection), "fit_skip_layers": [int(v) for v in skip]})
    return out


def summarize_comparison(res: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    def block(g, c):
        a = np.array([[r[k] for k in COUNT_COLS] for r in g], dtype=np.int64)
        b = np.array([[r[k] for k in COUNT_COLS] for r in c], dtype=np.int64)
        return {"gnn": method_summary(g), "ckf": method_summary(c),
                "bootstrap": {"n_boot": n_boot, "seed": seed, "level": 0.95, **paired_bootstrap(a, b, n_boot, seed)}}
    out = {**block(res["gnn"], res["ckf"]), "timing": res["timing"]}
    if "gnn_sel" in res:
        out["selected"] = {**block(res["gnn_sel"], res["ckf_sel"]), "selection": res["selection"],
                           "kept_fraction": {m: (sum(r["n_tracks"] for r in res[f"{m}_sel"]) /
                                                 max(sum(r["n_tracks"] for r in res[m]), 1)) for m in ("gnn", "ckf")}}
    return out


def crosscheck_export(export_dir: Path, tcfg: TargetConfig = PHASE1_TARGETS, max_events: int = 0) -> dict:
    """Python matching of the exported CKF tracks against the CKF's own performance_finding_ckf numbers
    (ckf_performance.json, written by the exporter with Phase 1's read_metrics).  Both cover ALL events of the export
    (the ACTS file has no per-event numbers), so max_events > 0 only makes the Python side a partial check."""
    man = odd.load_manifest(export_dir)
    events = [int(e) for e in man["events"]]
    if max_events:
        events = events[:max_events]
    rows = []
    for nr in events:
        ev = odd.read_event(export_dir, nr)
        rows.append(counts_for_tracks(ev, ckf_tracks_full(ev, export_dir), tcfg))
    py = method_summary(rows)
    out = {"export_id": man["export_id"], "n_events_python": len(events), "n_events_export": len(man["events"]),
           "python": {k: py[k] for k in ("efficiency", "fakerate", "duplicaterate", "n_tracks", "n_targets")}}
    sel = (man.get("post") or {}).get("n_selected_particles") or {}
    if sel:                                    # ACTS's own count of its selected particles (the efficiency denominator)
        out["n_selected_particles_acts"] = int(sum(int(sel.get(str(nr), 0)) for nr in events))
        out["n_targets_python"] = py["n_targets"]
    perf_path = Path(export_dir) / odd.PERFORMANCE_JSON
    if not perf_path.is_file():
        out["acts"] = None
        out["note"] = f"{perf_path.name} missing: the exporter could not read performance_finding_ckf.root"
        return out
    acts = json.loads(perf_path.read_text())
    out["acts"] = {k: acts.get(k) for k in ("efficiency", "fakerate", "duplicaterate")}
    out["difference_python_minus_acts"] = {k: (None if acts.get(k) is None else out["python"][k] - float(acts[k]))
                                          for k in ("efficiency", "fakerate", "duplicaterate")}
    out["complete"] = len(events) == len(man["events"])
    return out


def format_report(summary: dict, caveats: Sequence[str]) -> str:
    lines = _format_block(summary)
    if "selected" in summary:
        s = summary["selected"]
        sel = s["selection"]
        lines += ["", f"--- after the same track selection on both methods (helix refit of the hits: pT >= "
                      f"{sel['pt_min']:g} GeV, |eta| < {sel['abs_eta_max']:g}, |d0| < {sel['d0_max']:g} mm, >= "
                      f"{sel['n_meas_min']} measurements; the CKF's own cuts"
                      + (f"; refit without the hits of the {len(sel['fit_skip_layers'])} one-coordinate layers"
                         if sel.get("fit_skip_layers") else "")
                      + f"); tracks kept: GNN {s['kept_fraction']['gnn']:.3f}, CKF {s['kept_fraction']['ckf']:.3f}"]
        lines += _format_block(s)[1:]
    lines.append("dup. (targets) = (tracks matched to a target - targets found) / tracks matched to a target; "
                 "duplicate rate = ACTS's definition over all tracks; eff. (DM) = targets with a track of purity > 0.5 "
                 "holding more than half of the particle's measurements")
    lines += [f"NOTE: {x}" for x in caveats]
    return "\n".join(lines)


def _format_block(summary: dict) -> List[str]:
    g, c, bt = summary["gnn"], summary["ckf"], summary["bootstrap"]

    def ci(x):
        return f"[{x[0]:.4f}, {x[1]:.4f}]"
    lines = [f"events {g['n_events']}   targets {g['n_targets']}   (pT >= 1 GeV, |eta| < 3, >= 9 measurements)",
             f"{'':14s}{'GNN':>34s}{'CKF':>34s}{'GNN - CKF (95% paired boot.)':>40s}"]
    for k, lab in (("efficiency", "efficiency"), ("efficiency_dm", "eff. (DM)"), ("fakerate", "fake rate"),
                   ("duplicaterate", "duplicate rate"), ("duplicaterate_target", "dup. (targets)")):
        if k not in g or k not in bt:                     # summaries written before the target duplicate rate
            continue
        lines.append(f"{lab:14s}{g[k]:>10.4f} {ci(bt[k]['a']):>23s}{c[k]:>10.4f} {ci(bt[k]['b']):>23s}"
                     f"{g[k] - c[k]:>+12.4f} {ci(bt[k]['diff_a_minus_b']):>23s}")
    lines.append(f"binomial (Wilson, tracks): fake GNN {ci(g['fakerate_wilson'])} CKF {ci(c['fakerate_wilson'])};  "
                 f"dup GNN {ci(g['duplicaterate_wilson'])} CKF {ci(c['duplicaterate_wilson'])}")
    lines.append(f"tracks per event: GNN {g['tracks_per_event']:.1f}  CKF {c['tracks_per_event']:.1f}")
    if "target_tracks_per_event" in g:
        lines.append(f"  of which matched to a target: GNN {g['target_tracks_per_event']:.1f}  CKF "
                     f"{c['target_tracks_per_event']:.1f};  matched to a non-target particle: GNN "
                     f"{g['nontarget_tracks_per_event']:.1f}  CKF {c['nontarget_tracks_per_event']:.1f};  fakes: GNN "
                     f"{g['fakes_per_event']:.2f}  CKF {c['fakes_per_event']:.2f}")
    if "hits_per_target_track" in g:
        lines.append(f"  measurements per target track: GNN {g['hits_per_target_track']:.1f}  CKF "
                     f"{c['hits_per_target_track']:.1f}")
    return lines

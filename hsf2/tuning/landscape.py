"""QX1: the exhaustive landscape of the REAL objective on the 16-bit sub-grid (4^8 = 65,536 configurations).

compute_landscape()  every configuration on every event of a split (tuning events), in (event, filter-setting)
                     units on worker processes (hsf2.tuning.evaluator: exact, shares every stage that can be
                     shared); per-event count tables are saved as they complete (resumable);
summarize_landscape() score of every configuration in Phase-1 index order, the global optimum and all exact ties,
                     the default configuration, marginals per knob, and an event bootstrap of the optimum;
verify_landscape()   independent re-evaluation of the optimum, the default and random configurations with the
                     reference implementation (GNNPipeline.run_event, one configuration at a time).
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from concurrent.futures import as_completed
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..config import ScoreConfig, TargetConfig, TRACKING_DEFAULTS
from ..datasets import splits
from ..tracking.metrics import COUNT_KEYS
from ..utils import code_version, load_json, save_json, stamp
from .evaluator import EDGE_KEYS, FILTER_KNOBS, POST_KNOBS, vector_metrics
from .space import describe_space, full_encoding, level_values, sub_encoding

log = logging.getLogger(__name__)
METRIC_KEYS = ("score", "physics", "efficiency", "efficiency_dm", "fakerate", "duplicaterate", "time_proxy",
               "trackml_score", "n_tracks")


def to_index_order(t: np.ndarray, n_knobs: int = 8) -> np.ndarray:
    """[l_0, ..., l_7, ...] -> [index, ...] with index = sum_p l_p 4^p (Phase-1: knob 0 = least significant)."""
    axes = tuple(range(n_knobs - 1, -1, -1)) + tuple(range(n_knobs, t.ndim))
    return np.ascontiguousarray(t.transpose(axes)).reshape((-1,) + t.shape[n_knobs:])


def _event_file(out_dir: Path, eid: str) -> Path:
    return Path(out_dir) / "events" / f"{eid}.npz"


def compute_landscape(model_dir: Path, data_dir: Path, out_dir: Path, split: str = "tune", max_events: int = 0,
                      targets: Optional[TargetConfig] = None, workers: int = 4, threads: int = 1) -> dict:
    from ..model.train import EdgeModel
    from .runner import data_fingerprint, model_fingerprint
    from .workers import grid_unit, make_executor
    out_dir = Path(out_dir)
    enc = sub_encoding(full_encoding())
    vals = level_values(enc)
    ids = list(splits(data_dir)[split])[: max_events or None]
    model = EdgeModel.load(model_dir)
    targets = targets or TargetConfig(pt_min=max(model.hit_selection.pt_keep, 0.0) or 1.0)
    spec = {"split": split, "events": ids, "space": describe_space(enc), "targets": targets.to_dict(),
            "model_dir": str(model_dir), "model_fp": model_fingerprint(model_dir), "data_dir": str(data_dir),
            "data_fp": data_fingerprint(data_dir, ids), "count_keys": list(COUNT_KEYS), "edge_keys": list(EDGE_KEYS),
            "code": code_version()}
    spec_path = out_dir / "spec.json"
    if spec_path.exists():
        old = load_json(spec_path)
        for k in ("events", "model_fp", "data_fp", "targets", "space"):
            if json.dumps(old[k], sort_keys=True) != json.dumps(spec[k], sort_keys=True):
                raise RuntimeError(f"{out_dir} holds a landscape with a different {k}; use a new folder")
    else:
        save_json(spec_path, spec)
    grid = {k: vals[k] for k in POST_KNOBS}
    filters = list(itertools.product(*[vals[k] for k in FILTER_KNOBS]))
    todo = [e for e in ids if not _event_file(out_dir, e).exists()]
    log.info("QX1 landscape: %d events (%d to do) x %d filter settings x %d post-GNN settings = %d configurations "
             "per event, %d workers", len(ids), len(todo), len(filters), int(np.prod([len(v) for v in grid.values()])),
             len(filters) * int(np.prod([len(v) for v in grid.values()])), workers)
    t_start = time.time()
    cpu_s = 0.0
    if todo:
        shape_post = tuple(len(grid[k]) for k in POST_KNOBS)
        buf: Dict[str, dict] = {}
        ex = make_executor(model_dir, data_dir, targets.to_dict(), workers, threads, score_cache=4)
        try:
            futs = [ex.submit(grid_unit, e, f, grid) for e in todo for f in filters]
            done_units = 0
            for fut in as_completed(futs):
                eid, filt, counts, edges, secs = fut.result()
                cpu_s += secs
                b = buf.setdefault(eid, {"counts": np.zeros((4, 4, 4) + shape_post + (len(COUNT_KEYS),)),
                                         "edges": np.zeros((4, 4, 4, len(EDGE_KEYS))), "n": 0, "secs": 0.0})
                a, bb, c = (vals[k].index(v) for k, v in zip(FILTER_KNOBS, filt))
                b["counts"][a, bb, c] = counts
                b["edges"][a, bb, c] = edges
                b["n"] += 1
                b["secs"] += secs
                done_units += 1
                if b["n"] == len(filters):
                    path = _event_file(out_dir, eid)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(path.stem + ".tmp.npz")
                    np.savez_compressed(tmp, counts=b["counts"], edges=b["edges"], cpu_s=b["secs"])
                    tmp.replace(path)
                    del buf[eid]
                    el = time.time() - t_start
                    log.info("  event %s done (%.0f CPU s); %d/%d units, %.0f s elapsed, ETA %.0f s", eid, b["secs"],
                             done_units, len(futs), el, el / done_units * (len(futs) - done_units))
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    cost = {"wall_s_this_call": round(time.time() - t_start, 1), "cpu_s_this_call": round(cpu_s, 1),
            "configs_per_event": 65536, "n_events": len(ids), "objective_evaluations": 65536 * len(ids)}
    save_json(out_dir / "cost.json", cost)
    return assemble(out_dir)


def load_event_tables(out_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    spec = load_json(Path(out_dir) / "spec.json")
    out = {}
    for eid in spec["events"]:
        with np.load(_event_file(out_dir, eid)) as f:
            out[eid] = {"counts": to_index_order(f["counts"]),
                        "edges": to_index_order(np.broadcast_to(f["edges"][:, :, :, None, None, None, None, None, :],
                                                                f["counts"].shape[:8] + (len(EDGE_KEYS),))),
                        "cpu_s": float(f["cpu_s"])}
    return out


def metrics_table(counts: np.ndarray, edges: np.ndarray, score: Optional[ScoreConfig] = None) -> Dict[str, np.ndarray]:
    score = score or ScoreConfig()
    m = vector_metrics(counts, edges)
    m["physics"], m["score"] = score.score_arrays(m)
    return m


def assemble(out_dir: Path) -> dict:
    """Sum the per-event tables into landscape.npz (index order) and write summary.json."""
    out_dir = Path(out_dir)
    tables = load_event_tables(out_dir)
    counts = sum(t["counts"] for t in tables.values())
    edges = sum(t["edges"] for t in tables.values())
    m = metrics_table(counts, edges)
    np.savez_compressed(out_dir / "landscape.npz", counts=counts, edges=edges,
                        **{k: np.asarray(m[k], np.float64) for k in METRIC_KEYS})
    summ = summarize_landscape(out_dir, tables)
    save_json(out_dir / "summary.json", summ)
    return summ


class LandscapeTable:
    """landscape.npz in memory: metrics of every configuration index of the 16-bit encoding.

    `score` other than the specified S re-scores every configuration from the stored metrics (no new evaluation);
    "score" and "physics" then follow that objective everywhere, including metrics_of_index."""

    def __init__(self, out_dir: Path, score: Optional[ScoreConfig] = None) -> None:
        self.dir = Path(out_dir)
        self.spec = load_json(self.dir / "spec.json")
        with np.load(self.dir / "landscape.npz") as f:
            self.arrays = {k: f[k] for k in f.files}
        self.score_cfg = score or ScoreConfig()
        if not self.score_cfg.is_default:
            self.arrays["physics"], self.arrays["score"] = self.score_cfg.score_arrays(self.arrays)
        self.enc = sub_encoding(full_encoding())
        self.score = self.arrays["score"]
        self.n_events = len(self.spec["events"])
        self.best = float(self.score.max())
        self.order = np.argsort(-self.score, kind="stable")
        self._sorted = np.sort(self.score)

    def metrics_of_index(self, idx: int) -> Dict[str, float]:
        return {k: float(self.arrays[k][idx]) for k in METRIC_KEYS}

    def rank(self, idx: int) -> int:
        """0-based: number of configurations with a strictly higher score."""
        return int(len(self._sorted) - np.searchsorted(self._sorted, self.score[idx], side="right"))


def summarize_landscape(out_dir: Path, tables: Optional[dict] = None, n_boot: int = 500, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    tables = tables or load_event_tables(out_dir)
    spec = load_json(out_dir / "spec.json")
    enc = sub_encoding(full_encoding())
    eids = list(spec["events"])
    C = np.stack([tables[e]["counts"] for e in eids])            # [event, index, key]
    E = np.stack([tables[e]["edges"] for e in eids])
    m = metrics_table(C.sum(0), E.sum(0))
    s = m["score"]
    best = float(s.max())
    ties = np.nonzero(s >= best - 1e-12)[0]
    order = np.argsort(-s, kind="stable")
    i_def = int(enc.index_from_params(TRACKING_DEFAULTS))

    def row(i: int) -> dict:
        return {"index": int(i), "levels_2bit": enc.levels_from_index(int(i)).tolist(),
                "params": enc.params_from_index(int(i)), **{k: float(m[k][i]) for k in METRIC_KEYS}}

    marg = {}
    lv = enc.levels_from_index(np.arange(len(s)))
    for p, name in enumerate(enc.space.names):
        marg[name] = {"values": level_values(enc)[name],
                      "best_score": [float(s[lv[:, p] == l].max()) for l in range(enc.n_levels)],
                      "mean_score": [float(s[lv[:, p] == l].mean()) for l in range(enc.n_levels)]}
    # event bootstrap: how stable are the optimum and its gain over the default?
    rng = np.random.default_rng(seed)
    n_ev = len(eids)
    arg, gain, best_b = [], [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_ev, n_ev), minlength=n_ev).astype(float)
        mb = metrics_table(np.tensordot(w, C, 1), np.tensordot(w, E, 1))["score"]
        arg.append(int(np.argmax(mb)))
        gain.append(float(mb[order[0]] - mb[i_def]))
        best_b.append(float(mb.max()))
    arg = np.array(arg)
    uniq, cnt = np.unique(arg, return_counts=True)
    top_boot = [{"index": int(u), "fraction": float(c / n_boot), "params": enc.params_from_index(int(u))}
                for u, c in sorted(zip(uniq, cnt), key=lambda t: -t[1])[:10]]
    gain = np.array(gain)
    # per-event score of the optimum and the default (for paired statistics)
    per_ev = {k: [] for k in ("opt", "default")}
    for e in range(n_ev):
        me = metrics_table(C[e], E[e])["score"]
        per_ev["opt"].append(float(me[order[0]]))
        per_ev["default"].append(float(me[i_def]))
    cost = load_json(out_dir / "cost.json") if (out_dir / "cost.json").exists() else {}
    return {
        "created": stamp(), "n_configurations": int(len(s)), "n_events": n_ev, "events": eids,
        "space": describe_space(enc), "score_definition": "S = eff - fake - dup/5 - t/5, t = edges processed / "
                                                          "edges of the unfiltered graph (summed over events)",
        "optimum": row(int(order[0])), "n_exact_ties": int(len(ties)), "ties": [row(int(i)) for i in ties[:50]],
        "default": {**row(i_def), "rank": int(np.sum(s > s[i_def]))},
        "top20": [row(int(i)) for i in order[:20]],
        "distribution": {"min": float(s.min()), "max": best, "mean": float(s.mean()), "median": float(np.median(s)),
                         "std": float(s.std()), "q99": float(np.quantile(s, 0.99)), "q999": float(np.quantile(s, 0.999)),
                         "fraction_better_than_default": float(np.mean(s > s[i_def])),
                         "gap_best_to_second_distinct": float(best - s[s < best - 1e-12].max()) if np.any(s < best - 1e-12) else None},
        "marginals": marg,
        "bootstrap": {"n_boot": n_boot, "argmax_is_optimum_fraction": float(np.mean(arg == order[0])),
                      "argmax_in_top10_fraction": float(np.mean(np.isin(arg, order[:10]))),
                      "most_frequent_argmax": top_boot, "best_score_mean": float(np.mean(best_b)),
                      "best_score_std": float(np.std(best_b)),
                      "gain_opt_minus_default": {"mean": float(gain.mean()), "std": float(gain.std()),
                                                 "q05": float(np.quantile(gain, 0.05)),
                                                 "q95": float(np.quantile(gain, 0.95))}},
        "per_event_score": per_ev, "cost": cost,
        "cpu_s_total": float(sum(t["cpu_s"] for t in tables.values())),
    }


def verify_landscape(out_dir: Path, n_random: int = 25, workers: int = 4, seed: int = 1) -> dict:
    """Re-evaluate configurations one at a time with the reference implementation and compare every metric."""
    from ..config import ScoreConfig
    from ..tracking.pipeline import aggregate_rows
    from .workers import OwnedPool
    out_dir = Path(out_dir)
    tab = LandscapeTable(out_dir)
    spec = tab.spec
    enc = tab.enc
    summ = load_json(out_dir / "summary.json")
    rng = np.random.default_rng(seed)
    picks = [t["index"] for t in summ["ties"][:5]] + [summ["default"]["index"]] + \
        [int(i) for i in tab.order[1:4]] + [int(i) for i in rng.choice(len(tab.score), n_random, replace=False)]
    picks = list(dict.fromkeys(picks))
    pool = OwnedPool(Path(spec["model_dir"]), Path(spec["data_dir"]), spec["targets"], workers)
    sc = ScoreConfig()
    rows, worst = [], 0.0
    t0 = time.time()
    try:
        for i in picks:
            params = enc.params_from_index(int(i))
            ref = aggregate_rows(pool.eval_events(params, spec["events"]))
            ref["score"] = sc.score(ref)
            ref["physics"] = sc.physics(ref)
            diffs = {k: abs(float(ref[k]) - float(tab.arrays[k][i])) for k in METRIC_KEYS}
            worst = max(worst, max(diffs.values()))
            rows.append({"index": int(i), "params": params, "score_reference": ref["score"],
                         "score_landscape": float(tab.score[i]), "max_abs_diff": max(diffs.values())})
    finally:
        pool.close()
    res = {"n_checked": len(rows), "max_abs_difference": worst, "passed": worst < 1e-9, "seconds": round(time.time() - t0, 1),
           "rows": rows, "method": "GNNPipeline.run_event per configuration (reference), independent of the grid "
                                   "evaluator's shared stages"}
    save_json(out_dir / "verification.json", res)
    return res


# Objectives compared on one landscape (no new evaluations): the specified S and its sensitivity variants.
SENSITIVITY_SCORES: Dict[str, ScoreConfig] = {
    "S": ScoreConfig(),
    "S_DM": ScoreConfig(efficiency_key="efficiency_dm"),
    "S, t/10": ScoreConfig(k_time=10.0),
    "S, t/20": ScoreConfig(k_time=20.0),
    "S_DM, t/20": ScoreConfig(efficiency_key="efficiency_dm", k_time=20.0),
    "S, DM floor 0.99": ScoreConfig(dm_floor=0.99),
    "physics (no time term)": ScoreConfig(time_mode="none"),
}


def objective_sensitivity(out_dir: Path, scores: Optional[Mapping[str, ScoreConfig]] = None,
                          tolerances: Sequence[float] = (0.001, 0.005)) -> dict:
    """The optimum of every objective on the exact landscape, the ranks of the default and of the S optimum under
    it, and the smallest edge fraction t that keeps DM efficiency / TrackML score within `tolerances` of the
    default (the speed-up available at no physics cost)."""
    tab = LandscapeTable(out_dir)
    enc = tab.enc
    A = tab.arrays
    i_def = int(enc.index_from_params(TRACKING_DEFAULTS))
    keys = ("efficiency", "efficiency_dm", "fakerate", "duplicaterate", "time_proxy", "trackml_score")
    scores = dict(scores or SENSITIVITY_SCORES)
    vals = {name: sc.score_arrays(A)[1] for name, sc in scores.items()}
    i_s = int(np.argmax(vals["S"])) if "S" in vals else None

    def row(i: int) -> dict:
        return {"index": int(i), "params": enc.params_from_index(int(i)), **{k: float(A[k][i]) for k in keys},
                **{f"value[{n}]": float(v[i]) for n, v in vals.items()}}

    def rank(v: np.ndarray, i: int) -> int:
        return int(np.sum(v > v[i]))

    out = {"created": stamp(), "landscape": str(Path(out_dir)), "n_configurations": int(len(A["score"])),
           "default": row(i_def), "objectives": {}, "edge_reduction_at_no_physics_cost": {}}
    for name, v in vals.items():
        i = int(np.argmax(v))
        out["objectives"][name] = {
            "definition": scores[name].describe(), "score_config": scores[name].to_dict(), "optimum": row(i),
            "n_exact_ties": int(np.sum(v >= v[i] - 1e-12)), "rank_of_default": rank(v, i_def),
            "rank_of_S_optimum": rank(v, i_s) if i_s is not None else None}
    for key in ("efficiency_dm", "trackml_score"):
        for tol in tolerances:
            ok = np.nonzero(A[key] >= A[key][i_def] - tol)[0]
            i = int(ok[np.argmin(A["time_proxy"][ok])])
            out["edge_reduction_at_no_physics_cost"][f"{key} within {tol:g}"] = row(i)
    return out

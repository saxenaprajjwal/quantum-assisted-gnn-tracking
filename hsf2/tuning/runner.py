"""Runners with the Phase-1 CKFRunner contract, so the acts_qaoa stages run unchanged on the GNN problem.

Contract used by acts_qaoa.pipeline (warmstart / build / sample / validate / refine / baselines / confirm):
    evaluate(params, tag, n_events=None, seed=None, use_cache=True) -> record
    records() -> [record];  score(record) -> float | None;  check() -> [problem];  flags["seed"];
    _key(params_run, n_events, seed)
    record = {key, tag, params, n_events, seed, status ("ok" / "failed"), reason, metrics, objective, ...}

GNNRunner      the real objective: hit selection -> loose graph -> filters -> GNN -> track building -> metrics on a
               data split (GNNPipeline.run_event, the reference implementation).  seed=None -> the loop split
               ("tune"); seed = acts_qaoa.config.event_seed(k) = k * 10^6 -> split "confirm<k>" (Phase 1 uses the
               same seeds for its disjoint confirmation event samples).
LookupRunner   the same objective read from an exhaustive landscape (QX1): parameters are snapped to the grid.

Score (hsf2.config.ScoreConfig, the authoritative definition): S = eff - fake - dup/5 - t/5 with t the
deterministic time proxy.  Phase 1 scores the same record with ObjectiveConfig(runtime_mode="reco_relative"), whose
runtime slot "runtime_reco_rel" carries t (it is relative to the reference = default configuration by
construction); GNNRunner checks that both scores agree.

Cache: every evaluation is appended to <run>/eval_cache.jsonl with a key over the knob values, the split and its
event ids, and fingerprints of the model, the data folder, the target definition, the score and the pipeline
version.  Records from a different model / data folder in the same file are refused (stale results can never
contaminate a comparison).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import socket
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .. import __version__
from ..config import ScoreConfig, TargetConfig, TrackingParams
from ..datasets import event_path, splits
from ..utils import append_jsonl, read_jsonl

log = logging.getLogger(__name__)

# Bump when a change of the tracking chain changes results for the same model, data and knobs.
PIPELINE_VERSION = "1"
EVENT_SEED_STRIDE = 1_000_000       # = acts_qaoa.config.EVENT_SEED_STRIDE (checked in phase1.py)


def _sha(*parts: bytes) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p)
        h.update(b"\0")
    return h.hexdigest()[:16]


def model_fingerprint(model_dir: Path) -> str:
    d = Path(model_dir)
    return _sha((d / "model.pt").read_bytes(), (d / "model.json").read_bytes())


def data_fingerprint(data_dir: Path, event_ids: Sequence[str]) -> str:
    """Manifest + sizes of the event files used (event files are written once and never overwritten)."""
    d = Path(data_dir)
    sizes = ",".join(f"{e}:{event_path(d, e).stat().st_size}" for e in event_ids)
    return _sha((d / "manifest.json").read_bytes(), sizes.encode())


def canonical_params(params: Mapping[str, float], names: Sequence[str]) -> Dict[str, float]:
    """The knob values the pipeline is run with (integers rounded as TrackingParams does)."""
    missing = [n for n in names if n not in params]
    if missing:
        raise ValueError(f"missing tracking knobs {missing}")
    unknown = [n for n in params if n not in TrackingParams.__dataclass_fields__]
    if unknown:
        raise ValueError(f"unknown tracking knobs {unknown}")
    p = TrackingParams(**{n: float(params[n]) for n in names})
    out = {}
    for n in names:
        v = getattr(p, n)
        out[n] = int(v) if isinstance(v, (int, np.integer)) else float(v)
        if isinstance(out[n], float) and not math.isfinite(out[n]):
            raise ValueError(f"knob {n} is not finite")
    return out


def _param_key_text(params_run: Mapping[str, float], names: Sequence[str]) -> str:
    return "|".join(f"{n}={params_run[n]}" if isinstance(params_run[n], int) else f"{n}={float(params_run[n]):.9g}"
                    for n in names)


class GNNRunner:
    def __init__(self, model_dir: Path, data_dir: Path, run_dir: Path, names: Sequence[str],
                 loop_split: str = "tune", max_events: int = 0, targets: Optional[TargetConfig] = None,
                 score: Optional[ScoreConfig] = None, workers: int = 1, threads: int = 1) -> None:
        from ..model.train import EdgeModel
        self.model_dir, self.data_dir, self.run_dir = Path(model_dir), Path(data_dir), Path(run_dir)
        self.names = list(names)
        self.loop_split = loop_split
        self.max_events = int(max_events)
        self.score_cfg = score or ScoreConfig()
        self.model = EdgeModel.load(self.model_dir)
        hs = self.model.hit_selection
        self.targets = targets or TargetConfig(pt_min=max(hs.pt_keep, 0.0) or 1.0)
        self.splits = splits(self.data_dir)
        if loop_split not in self.splits or not self.splits[loop_split]:
            raise ValueError(f"split {loop_split!r} is empty or missing in {self.data_dir}/splits.json")
        self._check_disjoint()
        self.model_fp = model_fingerprint(self.model_dir)
        self.flags = {"seed": True, "nThreads": True}
        self.host = socket.gethostname()
        self.workers = max(int(workers), 1)
        self.threads = int(threads)
        self._pool = None
        self._pipe = None
        self._data_fp: Dict[str, str] = {}
        self.cache_path = self.run_dir / "eval_cache.jsonl"
        self.cache: Dict[str, dict] = {}
        for rec in read_jsonl(self.cache_path):
            if rec.get("model_fp") != self.model_fp:
                raise RuntimeError(f"{self.cache_path} holds results of another model ({rec.get('model_fp')} != "
                                   f"{self.model_fp}); use a new run folder")
            self.cache[rec["key"]] = rec
        self.n_new = 0
        self._retried: set = set()

    # ------------------------------------------------------------------ provenance
    def _check_disjoint(self) -> None:
        """Events the model was trained or selected on must never be used for tuning or confirmation."""
        from ..utils import load_json
        used = set(self.splits.get("train", [])) | set(self.splits.get("val", []))
        tc_path = self.model_dir / "train_config.json"
        if tc_path.exists():
            tc = load_json(tc_path)
            used |= set(tc.get("events", [])) | set(tc.get("extra", {}).get("val_events", []))
        else:
            log.warning("%s missing: the model's training events cannot be cross-checked", tc_path)
        for name, ids in self.splits.items():
            if name in ("train", "val", "unused"):
                continue
            both = used & set(ids)
            if both:
                raise RuntimeError(f"split {name} shares {len(both)} events with the model's training/validation "
                                   f"events (e.g. {sorted(both)[0]}): tuning on them would leak")

    def split_for_seed(self, seed: Optional[int]) -> str:
        if seed is None:
            return self.loop_split
        k, r = divmod(int(seed), EVENT_SEED_STRIDE)
        if r != 0 or k < 1:
            raise ValueError(f"seed {seed} is not an independent event-sample seed k * {EVENT_SEED_STRIDE}")
        name = f"confirm{k}"
        if name not in self.splits or not self.splits[name]:
            raise ValueError(f"event sample {k} needs split {name!r}, not in {self.data_dir}/splits.json")
        return name

    def events_for(self, n_events: Optional[int], seed: Optional[int]) -> List[str]:
        ids = list(self.splits[self.split_for_seed(seed)])
        n = int(n_events) if n_events else (self.max_events if seed is None and self.max_events else len(ids))
        return ids[:n]

    @property
    def loop_n_events(self) -> int:
        return len(self.events_for(None, None))

    def data_fp(self, event_ids: Sequence[str]) -> str:
        key = ",".join(event_ids)
        if key not in self._data_fp:
            self._data_fp[key] = data_fingerprint(self.data_dir, event_ids)
        return self._data_fp[key]

    def _key(self, params_run: Mapping[str, float], n_events: int, seed: Optional[int]) -> str:
        ids = self.events_for(n_events, seed)
        blob = "|".join([_param_key_text(params_run, self.names), f"split={self.split_for_seed(seed)}",
                         "events=" + ",".join(ids), f"model={self.model_fp}", f"data={self.data_fp(ids)}",
                         "targets=" + json.dumps(self.targets.to_dict(), sort_keys=True),
                         "score=" + json.dumps(self.score_cfg.to_dict(), sort_keys=True),
                         f"pipeline={PIPELINE_VERSION}"])
        return hashlib.sha1(blob.encode()).hexdigest()[:20]

    # ------------------------------------------------------------------ Phase-1 contract
    def check(self) -> List[str]:
        problems = []
        for eid in self.events_for(None, None):
            if not event_path(self.data_dir, eid).exists():
                problems.append(f"missing event file {event_path(self.data_dir, eid)}")
                break
        return problems

    def records(self) -> List[dict]:
        return list(self.cache.values())

    def score(self, rec: Mapping) -> Optional[float]:
        if rec.get("status") != "ok":
            return None
        try:
            s = self.score_cfg.score(rec["metrics"])
        except (KeyError, TypeError, ValueError):
            return None
        return s if math.isfinite(s) else None

    def _rows(self, params_run: dict, ids: List[str]) -> List[dict]:
        if self.workers > 1:
            if self._pool is None:
                from .workers import OwnedPool
                self._pool = OwnedPool(self.model_dir, self.data_dir, self.targets.to_dict(), self.workers,
                                       self.threads)
            return self._pool.eval_events(params_run, ids)
        if self._pipe is None:
            from ..tracking.pipeline import GNNPipeline
            self._pipe = GNNPipeline(self.model, self.targets, score_cache_size=256)
        from ..data.event import EventData
        p = TrackingParams(**params_run)
        rows = []
        for eid in ids:
            _, counts, _ = self._pipe.run_event(EventData.load(event_path(self.data_dir, eid)), p)
            counts["event_id"] = eid
            rows.append(counts)
        return rows

    def evaluate(self, params: Mapping[str, float], tag: str = "eval", n_events: Optional[int] = None,
                 seed: Optional[int] = None, use_cache: bool = True) -> dict:
        from ..tracking.pipeline import aggregate_rows
        params_run = canonical_params(params, self.names)
        ids = self.events_for(n_events, seed)
        key = self._key(params_run, len(ids), seed)
        if use_cache and key in self.cache:
            if self.cache[key].get("status") == "ok" or key in self._retried:
                rec = dict(self.cache[key])
                rec["cached"] = True
                rec["objective"] = self.score(rec)
                return rec
            self._retried.add(key)            # a failed evaluation is run again, once per process
            log.warning("  [GNN %s] re-running a configuration that failed before: %s", tag,
                        self.cache[key].get("reason", ""))
        t0 = time.time()
        status, reason, metrics, per_event = "ok", "", {}, []
        try:
            rows = self._rows(params_run, ids)
            metrics = aggregate_rows(rows)
            metrics["runtime_reco_rel"] = metrics["time_proxy"]        # Phase-1 runtime slot (see module doc)
            metrics["physics"] = self.score_cfg.physics(metrics)
            per_event = [{k: r[k] for k in ("event_id", "n_targets", "n_targets_found", "n_tracks", "n_fake",
                                            "n_duplicate", "edges_loose", "edges_filtered")} for r in rows]
        except BrokenProcessPool:                                     # a dead worker pool: stop, do not
            raise                                                     # turn every later evaluation into 'failed'
        except Exception as exc:                                      # recorded, never a silent number
            status, reason = "failed", f"{type(exc).__name__}: {exc}"
            log.exception("evaluation failed for %s", params_run)
        rec = {"key": key, "tag": tag, "params": params_run, "n_events": len(ids), "seed": seed,
               "split": self.split_for_seed(seed), "status": status, "reason": reason, "metrics": metrics,
               "per_event": per_event, "wall_s": round(time.time() - t0, 2), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "mock": False, "host": self.host, "model_fp": self.model_fp, "data_fp": self.data_fp(ids),
               "pipeline_version": PIPELINE_VERSION, "hsf2": __version__}
        rec["objective"] = self.score(rec)
        self.cache[key] = rec
        append_jsonl(self.cache_path, rec)
        self.n_new += 1
        if status == "ok":
            log.info("  [GNN %s] score=%.5f eff=%.4f fake=%.4f dup=%.4f t=%.3f (%d events, %.0f s)", tag,
                     rec["objective"], metrics["efficiency"], metrics["fakerate"], metrics["duplicaterate"],
                     metrics["time_proxy"], len(ids), rec["wall_s"])
        return rec

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None


class LookupRunner:
    """CKFRunner contract on an exhaustive landscape (QX1): every evaluation is an exact table look-up.

    `table` maps a configuration index of `enc` (the 16-bit encoding) to its metrics; parameters are snapped to the
    nearest grid point (Phase-1 Encoding.index_from_params) and the snapped values are recorded, so the surrogate
    is fitted on what was actually evaluated.  Distinct configurations are counted (`n_distinct`).
    """

    def __init__(self, enc, metrics_of_index, n_events: int, score: Optional[ScoreConfig] = None) -> None:
        self.enc = enc
        self.metrics_of_index = metrics_of_index
        self.n_events = int(n_events)              # recorded in every record (Phase-1 loop_records filters on it)
        self.score_cfg = score or ScoreConfig()
        self.flags = {"seed": False, "nThreads": False}
        self.cache: Dict[str, dict] = {}
        self.cache_path = None
        self.n_evaluations = 0

    def check(self) -> List[str]:
        return []

    def records(self) -> List[dict]:
        return list(self.cache.values())

    def score(self, rec: Mapping) -> Optional[float]:
        if rec.get("status") != "ok":
            return None
        return self.score_cfg.score(rec["metrics"])

    def _key(self, params_run: Mapping[str, float], n_events=None, seed=None) -> str:
        return str(self.enc.index_from_params(params_run))

    def evaluate(self, params: Mapping[str, float], tag: str = "eval", n_events: Optional[int] = None,
                 seed: Optional[int] = None, use_cache: bool = True) -> dict:
        idx = int(self.enc.index_from_params(params))
        key = str(idx)
        self.n_evaluations += 1
        if key in self.cache:
            rec = dict(self.cache[key])
            rec["cached"] = True
            return rec
        m = dict(self.metrics_of_index(idx))
        m["runtime_reco_rel"] = m["time_proxy"]
        rec = {"key": key, "tag": tag, "params": self.enc.params_from_index(idx), "index": idx,
               "n_events": self.n_events, "seed": None, "status": "ok", "reason": "", "metrics": m, "mock": False}
        rec["objective"] = self.score(rec)
        self.cache[key] = rec
        return rec

    @property
    def n_distinct(self) -> int:
        return len(self.cache)

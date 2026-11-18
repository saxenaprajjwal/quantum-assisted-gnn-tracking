"""Event-parallel evaluation in worker processes (spawn-safe: everything here is a top-level function).

A worker loads the model once and keeps a GNNPipeline (hit selection + loose graph per event, GNN scores per
(event, filter setting)).  Two pools:

  OwnedPool    W single-process executors, executor k owns the events k, k+W, ...  -> per-event caches stay
               warm across evaluations (GNNRunner: one knob setting, all events)
  grid_pool    one executor with W processes and dynamic scheduling of (event, filter) units (QX1 landscape)

torch runs single-threaded inside workers (threads=1 by default), so W workers use W cores.
"""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_STATE: Dict[str, object] = {}


def _init(model_dir: str, data_dir: str, targets: dict, threads: int, score_cache: int) -> None:
    import torch

    from ..config import TargetConfig
    from ..model.train import EdgeModel
    from ..tracking.pipeline import GNNPipeline
    if threads and threads > 0:
        torch.set_num_threads(int(threads))
    model = EdgeModel.load(Path(model_dir))
    _STATE.update(pipe=GNNPipeline(model, TargetConfig.from_dict(targets), score_cache_size=score_cache),
                  data_dir=Path(data_dir), events={})


def _event(eid: str):
    from ..data.event import EventData
    from ..datasets import event_path
    ev = _STATE["events"].get(eid)
    if ev is None:
        ev = EventData.load(event_path(_STATE["data_dir"], eid))
        _STATE["events"][eid] = ev
    return ev


def eval_events(params: dict, event_ids: Sequence[str]) -> List[dict]:
    """GNNPipeline.run_event counts (the reference evaluation) for one knob setting on the given events."""
    from ..config import TrackingParams
    pipe = _STATE["pipe"]
    p = TrackingParams(**params)
    rows = []
    for eid in event_ids:
        _, counts, _ = pipe.run_event(_event(eid), p)
        counts["event_id"] = eid
        rows.append(counts)
    return rows


def grid_unit(eid: str, filt: Tuple[float, float, float], grid: Mapping[str, Sequence[float]],
              max_paths: int = 64) -> Tuple[str, Tuple[float, float, float], np.ndarray, np.ndarray, float]:
    """All post-GNN knob combinations of `grid` for one event and one filter setting (QX1 landscape unit)."""
    import time

    from ..config import TrackingParams
    from .evaluator import edge_counts, post_grid_counts
    t0 = time.perf_counter()
    pipe = _STATE["pipe"]
    ev = _event(eid)
    es, g = pipe.prepare(ev)
    keep, s, _ = pipe.scores(ev, TrackingParams(dphi_scale=filt[0], deta_scale=filt[1], z0_scale=filt[2]))
    counts = post_grid_counts(es, g, keep, s, pipe.targets, grid, max_paths)
    return eid, tuple(filt), counts, edge_counts(g, keep), time.perf_counter() - t0


def make_executor(model_dir: Path, data_dir: Path, targets: dict, workers: int, threads: int = 1,
                  score_cache: int = 256) -> ProcessPoolExecutor:
    # spawn on every platform: fork after torch (and its threads) are loaded can deadlock on Linux
    return ProcessPoolExecutor(max_workers=max(int(workers), 1), initializer=_init,
                               mp_context=multiprocessing.get_context("spawn"),
                               initargs=(str(model_dir), str(data_dir), dict(targets), int(threads), int(score_cache)))


class OwnedPool:
    """W single-process executors; executor k evaluates the events k, k+W, ... of every request."""

    def __init__(self, model_dir: Path, data_dir: Path, targets: dict, workers: int, threads: int = 1,
                 score_cache: int = 256) -> None:
        self.workers = max(int(workers), 1)
        self.execs = [make_executor(model_dir, data_dir, targets, 1, threads, score_cache) for _ in range(self.workers)]

    def eval_events(self, params: dict, event_ids: Sequence[str]) -> List[dict]:
        futs = [ex.submit(eval_events, dict(params), list(event_ids[k::self.workers]))
                for k, ex in enumerate(self.execs) if event_ids[k::self.workers]]
        rows = [r for f in futs for r in f.result()]
        order = {e: i for i, e in enumerate(event_ids)}
        return sorted(rows, key=lambda r: order[r["event_id"]])

    def close(self) -> None:
        for ex in self.execs:
            ex.shutdown(wait=True, cancel_futures=True)


def default_workers() -> int:
    """Physical cores minus one (logical cpu_count / 2 on hyper-threaded PCs), at least 1."""
    return max(1, (os.cpu_count() or 2) // 2 - 1)

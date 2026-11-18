"""Data folders (data/<DD-MM-YYYY-HH-MM>_<label>/) and model folders (models/<DD-MM-YYYY-HH-MM>_<label>/).

data folder:   manifest.json (source, events, statistics, code version), splits.json, events/<event>.npz
model folder:  train_config.json (written first), model.pt + model.json (written at the end), history.csv,
               graph_stats.json, log.txt, outputs/<DD-MM-YYYY-HH-MM>_<command>/ for later evaluations
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .data.event import EventData
from .utils import load_json, resolve_dir

PART2_ROOT = Path(__file__).resolve().parents[1]


def deployed_in_phase1() -> bool:
    """hsf2/ was copied into the Phase-1 Optimization folder (next to acts_qaoa/): never use its data/models/runs."""
    return (PART2_ROOT / "acts_qaoa" / "__init__.py").is_file() or (PART2_ROOT / "DirectQAOAFeedback.py").is_file()


def _env_path(name: str) -> Optional[Path]:
    """$name if set to a non-empty value; when deployed next to Phase 1 it must be absolute (a relative value would
    resolve against the working directory, which may be the Phase-1 folder)."""
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    p = Path(v).expanduser()
    if not p.is_absolute() and deployed_in_phase1():
        raise ValueError(f"{name}={v!r} must be an absolute path")
    return p


def work_root() -> Path:
    """Parent of data/, models/, runs/, figures/: $HSF2_WORK; <Optimization>/hsf2_work when deployed next to Phase 1;
    otherwise the Part-II folder itself (standalone repository)."""
    env = _env_path("HSF2_WORK")
    if env:
        return env
    return PART2_ROOT / "hsf2_work" if deployed_in_phase1() else PART2_ROOT


def data_root() -> Path:
    return _env_path("HSF2_DATA") or work_root() / "data"


def models_root() -> Path:
    return _env_path("HSF2_MODELS") or work_root() / "models"


def runs_root() -> Path:
    return _env_path("HSF2_RUNS") or work_root() / "runs"


def figures_root() -> Path:
    """Figures of the guide: docs/figures in the standalone repository, <work>/figures when deployed."""
    env = _env_path("HSF2_FIGURES")
    if env:
        return env
    if deployed_in_phase1() or _env_path("HSF2_WORK"):
        return work_root() / "figures"
    return PART2_ROOT / "docs" / "figures"


def ensure_inside_work(path: Path, what: str) -> Path:
    """An output folder given on the command line must lie inside the work folder when deployed next to Phase 1."""
    p = Path(path).expanduser().resolve()
    if deployed_in_phase1():
        w = work_root().resolve()
        if p != w and w not in p.parents:
            raise SystemExit(f"{what} {p} is outside {w}: Part II never writes into the Phase-1 folder")
    return p


def resolve_data(arg: str) -> Path:
    return resolve_dir(arg, data_root(), "manifest.json")


def resolve_model(arg: str) -> Path:
    return resolve_dir(arg, models_root(), "model.json")


def splits(data_dir: Path) -> Dict[str, List[str]]:
    return load_json(Path(data_dir) / "splits.json")


def event_path(data_dir: Path, event_id: str) -> Path:
    return Path(data_dir) / "events" / f"{event_id}.npz"


def iter_split(data_dir: Path, split: str, max_events: int = 0) -> Iterator[EventData]:
    sp = splits(data_dir)
    if split not in sp:
        raise KeyError(f"split {split!r} not in {data_dir}/splits.json (have {list(sp)})")
    ids = sp[split][:max_events] if max_events and max_events > 0 else sp[split]
    for eid in ids:
        yield EventData.load(event_path(data_dir, eid))


def load_split(data_dir: Path, split: str, max_events: int = 0) -> List[EventData]:
    return list(iter_split(data_dir, split, max_events))

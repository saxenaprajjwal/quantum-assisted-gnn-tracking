"""TrackML (Kaggle 2018 challenge, Zenodo 4730167) -> EventData, read directly from the zip files.

Each event has hits.csv (hit_id, x, y, z, volume_id, layer_id, module_id), truth.csv (hit_id, particle_id,
tx, ty, tz, tpx, tpy, tpz, weight) and particles.csv (particle_id, vx, vy, vz, px, py, pz, q, nhits);
cells.csv is not used.  The 48 sensitive layers are numbered 0..47 in (volume_id, layer_id) order:
pixel end-cap 7 / barrel 8 / end-cap 9, short strips 12 / 13 / 14, long strips 16 / 17 / 18.
Reference: S. Amrouche et al., "The Tracking Machine Learning challenge: Accuracy phase", Springer (2019),
arXiv:1904.06778; data licence CC-BY-4.0.
"""
from __future__ import annotations

import io
import logging
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .event import EventData

log = logging.getLogger(__name__)

TRACKML_LAYERS: List[Tuple[int, int]] = (
    [(7, l) for l in range(2, 15, 2)] + [(8, l) for l in range(2, 9, 2)] + [(9, l) for l in range(2, 15, 2)]
    + [(12, l) for l in range(2, 13, 2)] + [(13, l) for l in range(2, 9, 2)] + [(14, l) for l in range(2, 13, 2)]
    + [(16, l) for l in range(2, 13, 2)] + [(17, l) for l in (2, 4)] + [(18, l) for l in range(2, 13, 2)])
LAYER_INDEX: Dict[Tuple[int, int], int] = {vl: i for i, vl in enumerate(TRACKML_LAYERS)}
MEMBER_RE = re.compile(r"(?:^|/)(event\d{9})-(hits|truth|particles|cells)\.csv(?:\.gz)?$")


def check_detectors(path: Path) -> None:
    """The (volume, layer) pairs of detectors.csv must be exactly the 48 layers assumed here."""
    import pandas as pd
    d = pd.read_csv(path, usecols=["volume_id", "layer_id"])
    found = sorted(set(map(tuple, d[["volume_id", "layer_id"]].drop_duplicates().to_numpy().tolist())))
    if found != sorted(TRACKML_LAYERS):
        raise ValueError(f"unexpected TrackML layer layout in {path}: {found}")


def open_zip(src) -> zipfile.ZipFile:
    """A local TrackML zip, or 'zenodo:train_1.zip' = read from Zenodo with HTTP range requests (no download)."""
    s = str(src)
    if s.startswith("zenodo:"):
        from .remote_zip import open_remote
        return zipfile.ZipFile(open_remote(s.split(":", 1)[1]))
    return zipfile.ZipFile(src)


def zip_index(src, zf: Optional[zipfile.ZipFile] = None) -> Dict[str, Dict[str, str]]:
    """event name -> {kind: member name} for the events of a TrackML zip (sorted by event name)."""
    out: Dict[str, Dict[str, str]] = {}
    own = zf is None
    zf = zf or open_zip(src)
    try:
        for name in zf.namelist():
            m = MEMBER_RE.search(name)
            if m:
                out.setdefault(m.group(1), {})[m.group(2)] = name
    finally:
        if own:
            zf.close()
    return dict(sorted((k, v) for k, v in out.items() if {"hits", "truth", "particles"} <= set(v)))


def _read_csv(zf: zipfile.ZipFile, member: str, usecols=None):
    import pandas as pd
    with zf.open(member) as f:
        data = f.read()
    comp = "gzip" if member.endswith(".gz") else None
    return pd.read_csv(io.BytesIO(data), usecols=usecols, compression=comp)


def frames_to_event(event_id: str, hits, truth, particles) -> EventData:
    h = hits.merge(truth[["hit_id", "particle_id", "weight"]], on="hit_id", how="left", validate="one_to_one")
    h["particle_id"] = h["particle_id"].fillna(0).astype(np.int64)
    h["weight"] = h["weight"].fillna(0.0)
    keys = list(zip(h["volume_id"].to_numpy().tolist(), h["layer_id"].to_numpy().tolist()))
    try:
        layer = np.fromiter((LAYER_INDEX[k] for k in keys), dtype=np.int64, count=len(keys))
    except KeyError as exc:
        raise ValueError(f"{event_id}: hit on unknown (volume, layer) {exc}") from None
    module = (h["volume_id"].to_numpy() * 1000000 + h["layer_id"].to_numpy() * 10000
              + h["module_id"].to_numpy()).astype(np.int64)
    hit = {"hit_id": h["hit_id"].to_numpy(), "x": h["x"].to_numpy(), "y": h["y"].to_numpy(), "z": h["z"].to_numpy(),
           "layer": layer, "module": module, "particle_id": h["particle_id"].to_numpy(),
           "weight": h["weight"].to_numpy()}
    p = particles
    pt = np.hypot(p["px"].to_numpy(), p["py"].to_numpy())
    par = {"particle_id": p["particle_id"].to_numpy(), "pt": pt,
           "eta": np.arcsinh(p["pz"].to_numpy() / np.maximum(pt, 1e-9)),
           "phi": np.arctan2(p["py"].to_numpy(), p["px"].to_numpy()), "vx": p["vx"].to_numpy(),
           "vy": p["vy"].to_numpy(), "vz": p["vz"].to_numpy(), "q": p["q"].to_numpy(), "nhits": p["nhits"].to_numpy()}
    meta = {"source": "trackml", "n_layers": len(TRACKML_LAYERS)}
    return EventData(event_id, hit, par, meta)


def iter_zip_events(zip_path, names: Optional[List[str]] = None) -> Iterator[EventData]:
    with open_zip(zip_path) as zf:
        index = zip_index(zip_path, zf)
        names = list(index) if names is None else names
        for name in names:
            mem = index[name]
            hits = _read_csv(zf, mem["hits"], ["hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id"])
            truth = _read_csv(zf, mem["truth"], ["hit_id", "particle_id", "weight"])
            parts = _read_csv(zf, mem["particles"])
            ev = frames_to_event(name, hits, truth, parts)
            ev.meta["zip"] = str(zip_path)
            yield ev


def _event_row(ev: EventData) -> dict:
    return {"event_id": ev.event_id, "n_hits": ev.n_hits, "n_particles": ev.n_particles,
            "n_noise": int((ev.particle_id == 0).sum()),
            "n_particles_pt1": int((ev.particles["pt"] >= 1.0).sum()),
            "n_hits_pt1": int((ev.hit_particle("pt") >= 1.0).sum())}


def prep_from_zip(zip_path: Path, out_dir: Path, n_events: int, skip: int = 0, workers: int = 1) -> List[dict]:
    """Convert the first n_events (after `skip`) events of a TrackML zip to out_dir/events/<event>.npz.

    workers > 1 reads events with that many independent zip handles in parallel threads (the remote reader is
    latency-bound: ~26 s per event with one stream from home); rows are returned sorted by event id, so the
    result does not depend on the number of workers.
    """
    with open_zip(zip_path) as zf:
        index = zip_index(zip_path, zf)
    names = list(index)[skip:skip + n_events]
    if not names:
        raise ValueError(f"no complete events in {zip_path}")
    log.info("%s: %d events in the zip, converting %d (%s ... %s) with %d worker(s)", zip_path, len(index),
             len(names), names[0], names[-1], max(int(workers), 1))
    rows: List[dict] = []
    t0 = time.time()

    def convert(chunk: List[str]) -> List[dict]:
        out = []
        for ev in iter_zip_events(zip_path, chunk):
            path = Path(out_dir) / "events" / f"{ev.event_id}.npz"
            if not path.exists():
                ev.save(path)
            out.append(_event_row(ev))
            rows.append(out[-1])
            k = len(rows)
            if k % 10 == 0 or k == len(names):
                log.info("  %d/%d events  (%.1f s/event)", k, len(names), (time.time() - t0) / k)
        return out

    if int(workers) <= 1:
        convert(names)
    else:
        from concurrent.futures import ThreadPoolExecutor
        chunks = [names[i::int(workers)] for i in range(int(workers))]
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            for fut in [pool.submit(convert, c) for c in chunks if c]:
                fut.result()
    return sorted(rows, key=lambda r: r["event_id"])

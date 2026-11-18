"""One event as plain numpy arrays -- the common format of TrackML, ODD-export and synthetic events.

Hits:      hit_id, x, y, z [mm], layer (detector-wide layer index), module, particle_id (0 = noise),
           weight (TrackML score weight; 0 for noise)
Particles: particle_id, pt [GeV], eta, phi (momentum direction), vx, vy, vz [mm], q, nhits
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

HIT_DTYPES = {"hit_id": np.int64, "x": np.float32, "y": np.float32, "z": np.float32, "layer": np.int16,
              "module": np.int32, "particle_id": np.int64, "weight": np.float32}
PARTICLE_DTYPES = {"particle_id": np.int64, "pt": np.float32, "eta": np.float32, "phi": np.float32,
                   "vx": np.float32, "vy": np.float32, "vz": np.float32, "q": np.int8, "nhits": np.int16}


@dataclass(eq=False)
class EventData:
    event_id: str
    hits: Dict[str, np.ndarray]
    particles: Dict[str, np.ndarray]
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.hits["x"])
        for k, dt in HIT_DTYPES.items():
            if k not in self.hits:
                if k == "hit_id":
                    self.hits[k] = np.arange(1, n + 1)
                elif k == "module":
                    self.hits[k] = np.full(n, -1)
                elif k == "weight":
                    self.hits[k] = np.zeros(n)
                else:
                    raise KeyError(f"hit field {k} missing")
            self.hits[k] = np.asarray(self.hits[k], dtype=dt)
            if len(self.hits[k]) != n:
                raise ValueError(f"hit field {k} has length {len(self.hits[k])}, expected {n}")
        m = len(self.particles["particle_id"])
        for k, dt in PARTICLE_DTYPES.items():
            if k not in self.particles:
                self.particles[k] = np.zeros(m)
            self.particles[k] = np.asarray(self.particles[k], dtype=dt)
        # the particle table is kept sorted by particle_id (particle_rows uses searchsorted)
        order = np.argsort(self.particles["particle_id"], kind="stable")
        if np.any(order != np.arange(m)):
            self.particles = {k: v[order] for k, v in self.particles.items()}
        self._cache = {}

    @property
    def n_hits(self) -> int:
        return len(self.hits["x"])

    @property
    def n_particles(self) -> int:
        return len(self.particles["particle_id"])

    def __getattr__(self, name):
        # ev.x, ev.layer, ev.particle_id ... are the hit columns
        if name in ("hits", "particles", "meta", "_cache"):
            raise AttributeError(name)
        hits = self.__dict__.get("hits", {})
        if name in hits:
            return hits[name]
        raise AttributeError(name)

    def _derived(self, key: str, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def r(self) -> np.ndarray:
        return self._derived("r", lambda: np.hypot(self.hits["x"], self.hits["y"]).astype(np.float32))

    @property
    def phi(self) -> np.ndarray:
        return self._derived("phi", lambda: np.arctan2(self.hits["y"], self.hits["x"]).astype(np.float32))

    @property
    def eta(self) -> np.ndarray:
        """Pseudorapidity of the hit position seen from the origin."""
        return self._derived("eta", lambda: np.arcsinh(self.hits["z"] / np.maximum(self.r, 1e-3)).astype(np.float32))

    @property
    def rho(self) -> np.ndarray:
        """Distance of the hit from the origin (orders hits along a track without truth)."""
        return self._derived("rho", lambda: np.sqrt(self.r.astype(np.float64) ** 2 + self.hits["z"].astype(np.float64) ** 2))

    def particle_rows(self, pids: np.ndarray) -> np.ndarray:
        """Row in the particle table for each particle id (-1 if unknown, e.g. noise)."""
        table = self.particles["particle_id"]
        pids = np.asarray(pids, dtype=np.int64)
        if len(table) == 0:
            return np.full(len(pids), -1, dtype=np.int64)
        pos = np.clip(np.searchsorted(table, pids), 0, len(table) - 1)
        return np.where(table[pos] == pids, pos, -1).astype(np.int64)

    def hit_particle(self, key: str, fill: float = 0.0) -> np.ndarray:
        """A particle property (pt, eta, vz ...) for every hit; `fill` for noise hits."""
        rows = self.particle_rows(self.hits["particle_id"])
        if self.n_particles == 0:
            return np.full(len(rows), fill, dtype=np.float64)
        vals = self.particles[key][np.maximum(rows, 0)].astype(np.float64)
        return np.where(rows >= 0, vals, fill)

    def subset(self, idx) -> "EventData":
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.nonzero(idx)[0]
        meta = dict(self.meta)
        return EventData(self.event_id, {k: v[idx] for k, v in self.hits.items()},
                         {k: v.copy() for k, v in self.particles.items()}, meta)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"hit_{k}": v for k, v in self.hits.items()}
        arrays.update({f"par_{k}": v for k, v in self.particles.items()})
        arrays["event_id"] = np.array(self.event_id)
        arrays["meta"] = np.array(json.dumps(self.meta))
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "EventData":
        with np.load(Path(path), allow_pickle=False) as f:
            hits = {k[4:]: f[k] for k in f.files if k.startswith("hit_")}
            parts = {k[4:]: f[k] for k in f.files if k.startswith("par_")}
            eid = str(f["event_id"])
            meta = json.loads(str(f["meta"])) if "meta" in f.files else {}
        return cls(eid, hits, parts, meta)


def select_hits(ev: EventData, sel) -> EventData:
    """Apply a HitSelection (hsf2.config): S0 = hits of particles with pT >= pt_keep only; S1 = all hits."""
    pid = ev.hits["particle_id"]
    noise = pid == 0
    keep = np.ones(ev.n_hits, dtype=bool)
    if sel.pt_keep > 0:
        keep &= ev.hit_particle("pt", fill=0.0) >= sel.pt_keep
    if sel.keep_noise:
        keep |= noise
    else:
        keep &= ~noise
    out = ev.subset(keep)
    out.meta["hit_selection"] = sel.to_dict()
    return out


def particle_hit_counts(ev: EventData) -> np.ndarray:
    """Number of hits of each particle-table row among the hits of this (possibly reduced) event."""
    rows = ev.particle_rows(ev.hits["particle_id"])
    rows = rows[rows >= 0]
    return np.bincount(rows, minlength=ev.n_particles)


def target_mask(ev: EventData, tcfg) -> np.ndarray:
    """Boolean mask over the particle table: reconstructable particles (efficiency denominator)."""
    p = ev.particles
    return ((p["particle_id"] != 0) & (p["pt"] >= tcfg.pt_min) & (np.abs(p["eta"]) <= tcfg.eta_max)
            & (particle_hit_counts(ev) >= tcfg.min_hits))


def truth_edges(ev: EventData, pt_min: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Consecutive hits of every particle (pT >= pt_min), ordered by distance from its production vertex.

    Returns (src, dst) hit indices; src is the hit closer to the vertex.
    """
    pid = ev.hits["particle_id"]
    keep = pid != 0
    if pt_min > 0:
        keep &= ev.hit_particle("pt", fill=0.0) >= pt_min
    idx = np.nonzero(keep)[0]
    if len(idx) < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rows = ev.particle_rows(pid[idx])
    vx = np.where(rows >= 0, ev.particles["vx"][np.maximum(rows, 0)], 0.0)
    vy = np.where(rows >= 0, ev.particles["vy"][np.maximum(rows, 0)], 0.0)
    vz = np.where(rows >= 0, ev.particles["vz"][np.maximum(rows, 0)], 0.0)
    d = np.sqrt((ev.hits["x"][idx] - vx) ** 2 + (ev.hits["y"][idx] - vy) ** 2 + (ev.hits["z"][idx] - vz) ** 2)
    order = np.lexsort((idx, d, pid[idx]))
    s = idx[order]
    p = pid[s]
    same = p[1:] == p[:-1]
    return s[:-1][same].astype(np.int64), s[1:][same].astype(np.int64)


def pair_keys(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Order-independent integer key of hit pairs (a, b) in an event with n hits."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return np.minimum(a, b) * np.int64(n) + np.maximum(a, b)

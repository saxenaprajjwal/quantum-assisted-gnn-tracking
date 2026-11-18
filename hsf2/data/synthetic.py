"""Synthetic layered detector (TrackML-like geometry, helical tracks in a solenoid, smearing, noise).

Used by the self-test and the unit tests (no data needed) and to exercise every stage of the pipeline on
events whose truth is known exactly.  Geometry: 10 barrel layers at the TrackML pixel/strip radii and 7
pixel-like end-cap disks on each side.  Tracks start on the beam line (vx = vy = 0) at vz ~ N(0, sigma_vz);
a particle of transverse momentum pT [GeV] in a field B [T] moves on a circle of radius R = pT / (0.3 B) [m]
through the origin, so it crosses a cylinder of radius r after turning angle t = 2 asin(r / 2R), at
z = vz + R t sinh(eta) and azimuth phi0 - q t / 2.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .event import EventData


@dataclass
class ToyDetector:
    # barrel: 4 pixel + 6 strip layers (mm); end caps: 7 disks per side
    barrel_r: Tuple[float, ...] = (32.0, 72.0, 116.0, 172.0, 260.0, 360.0, 500.0, 660.0, 820.0, 1020.0)
    barrel_halfz: Tuple[float, ...] = (450.0, 450.0, 450.0, 450.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0)
    disk_z: Tuple[float, ...] = (600.0, 700.0, 820.0, 960.0, 1100.0, 1300.0, 1500.0)
    disk_rmin: float = 30.0
    disk_rmax: float = 175.0
    pixel_rmax: float = 200.0
    sigma_rphi_pixel: float = 0.015
    sigma_z_pixel: float = 0.05
    sigma_rphi_strip: float = 0.04
    sigma_z_strip: float = 0.5
    sigma_r_disk: float = 0.05
    b_field: float = 2.0
    n_phi_modules: int = 32

    @property
    def n_barrel(self) -> int:
        return len(self.barrel_r)

    @property
    def n_layers(self) -> int:
        return len(self.barrel_r) + 2 * len(self.disk_z)

    def layer_table(self) -> List[dict]:
        """Index -> description: barrel 0..9, negative disks (by |z|), positive disks (by |z|)."""
        rows = [{"layer": k, "kind": "barrel", "r": r, "halfz": hz}
                for k, (r, hz) in enumerate(zip(self.barrel_r, self.barrel_halfz))]
        nb, nd = self.n_barrel, len(self.disk_z)
        rows += [{"layer": nb + k, "kind": "disk", "z": -z} for k, z in enumerate(self.disk_z)]
        rows += [{"layer": nb + nd + k, "kind": "disk", "z": z} for k, z in enumerate(self.disk_z)]
        return rows


@dataclass
class ToyConfig:
    n_particles: int = 150
    pt_min: float = 0.5           # GeV
    pt_max: float = 20.0
    pt_power: float = 2.0         # dN/dpT ~ pT^-pt_power
    eta_max: float = 2.5
    sigma_vz: float = 40.0        # mm
    hit_eff: float = 0.98
    noise_frac: float = 0.05      # noise hits as a fraction of the particle hits
    seed: int = 0


def _sample_pt(rng: np.random.Generator, n: int, cfg: ToyConfig) -> np.ndarray:
    a, b, p = cfg.pt_min, cfg.pt_max, cfg.pt_power
    u = rng.random(n)
    if abs(p - 1.0) < 1e-9:
        return a * (b / a) ** u
    e = 1.0 - p
    return (a ** e + u * (b ** e - a ** e)) ** (1.0 / e)


def generate_event(det: ToyDetector, cfg: ToyConfig, event_index: int = 0) -> EventData:
    rng = np.random.default_rng([cfg.seed, event_index])
    n = cfg.n_particles
    pt = _sample_pt(rng, n, cfg)
    eta = rng.uniform(-cfg.eta_max, cfg.eta_max, n)
    phi0 = rng.uniform(-math.pi, math.pi, n)
    q = rng.choice([-1, 1], n)
    vz = rng.normal(0.0, cfg.sigma_vz, n)
    R = pt / (0.3 * det.b_field) * 1000.0          # mm
    sh = np.sinh(eta)
    cols: Dict[str, list] = {k: [] for k in ("p", "layer", "t", "r", "phi", "z", "disk")}
    for k, (rl, hz) in enumerate(zip(det.barrel_r, det.barrel_halfz)):
        ok = rl < 2.0 * R
        t = 2.0 * np.arcsin(np.clip(rl / (2.0 * R), 0.0, 1.0))
        z = vz + R * t * sh
        ok &= np.abs(z) <= hz
        idx = np.nonzero(ok)[0]
        cols["p"].append(idx)
        cols["layer"].append(np.full(len(idx), k))
        cols["t"].append(t[idx])
        cols["r"].append(np.full(len(idx), rl))
        cols["phi"].append(phi0[idx] - q[idx] * t[idx] / 2.0)
        cols["z"].append(z[idx])
        cols["disk"].append(np.zeros(len(idx), bool))
    nb, nd = det.n_barrel, len(det.disk_z)
    for side, off in ((-1.0, nb), (1.0, nb + nd)):
        for k, zd in enumerate(det.disk_z):
            zs = side * zd
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (zs - vz) / (R * sh)
            ok = np.isfinite(t) & (t > 0) & (t <= math.pi)
            r = 2.0 * R * np.sin(np.clip(t, 0.0, math.pi) / 2.0)
            ok &= (r >= det.disk_rmin) & (r <= det.disk_rmax)
            idx = np.nonzero(ok)[0]
            cols["p"].append(idx)
            cols["layer"].append(np.full(len(idx), off + k))
            cols["t"].append(t[idx])
            cols["r"].append(r[idx])
            cols["phi"].append(phi0[idx] - q[idx] * t[idx] / 2.0)
            cols["z"].append(np.full(len(idx), zs))
            cols["disk"].append(np.ones(len(idx), bool))
    c = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    keep = rng.random(len(c["p"])) < cfg.hit_eff
    c = {k: v[keep] for k, v in c.items()}
    # smearing: pixel resolution on the disks and the inner barrel, strip resolution outside
    pix = c["disk"] | (c["r"] < det.pixel_rmax)
    s_rphi = np.where(pix, det.sigma_rphi_pixel, det.sigma_rphi_strip)
    s_z = np.where(pix, det.sigma_z_pixel, det.sigma_z_strip)
    r = c["r"] + np.where(c["disk"], rng.normal(0.0, det.sigma_r_disk, len(c["r"])), 0.0)
    phi = c["phi"] + rng.normal(0.0, 1.0, len(r)) * s_rphi / np.maximum(r, 1.0)
    z = c["z"] + np.where(c["disk"], 0.0, rng.normal(0.0, 1.0, len(r)) * s_z)
    layer = c["layer"].astype(np.int64)
    pid = c["p"].astype(np.int64) + 1
    n_noise = int(round(cfg.noise_frac * len(r)))
    if n_noise:
        table = det.layer_table()
        nl = rng.integers(0, det.n_layers, n_noise)
        nphi = rng.uniform(-math.pi, math.pi, n_noise)
        nr = np.empty(n_noise)
        nz = np.empty(n_noise)
        for i, L in enumerate(nl):
            row = table[L]
            if row["kind"] == "barrel":
                nr[i] = row["r"]
                nz[i] = rng.uniform(-row["halfz"], row["halfz"])
            else:
                nr[i] = math.sqrt(rng.uniform(det.disk_rmin ** 2, det.disk_rmax ** 2))
                nz[i] = row["z"]
        r = np.concatenate([r, nr])
        phi = np.concatenate([phi, nphi])
        z = np.concatenate([z, nz])
        layer = np.concatenate([layer, nl])
        pid = np.concatenate([pid, np.zeros(n_noise, np.int64)])
    phi = (phi + math.pi) % (2.0 * math.pi) - math.pi
    perm = rng.permutation(len(r))
    r, phi, z, layer, pid = r[perm], phi[perm], z[perm], layer[perm], pid[perm]
    x, y = r * np.cos(phi), r * np.sin(phi)
    is_disk = layer >= nb
    seg = np.where(is_disk, np.minimum(((r - det.disk_rmin) / (det.disk_rmax - det.disk_rmin) * 4).astype(int), 3),
                   np.clip(((z / 1000.0 + 1.0) * 4).astype(int), 0, 7))
    module = ((phi + math.pi) / (2 * math.pi) * det.n_phi_modules).astype(int) % det.n_phi_modules \
        + det.n_phi_modules * seg
    nh = np.bincount(pid, minlength=n + 1)[1:]
    # TrackML-like weights: uniform over the hits of particles with >= 4 hits, normalised to 1
    w = np.where(pid > 0, (nh[np.maximum(pid - 1, 0)] >= 4).astype(float), 0.0)
    w = w / w.sum() if w.sum() > 0 else w
    hits = {"hit_id": np.arange(1, len(r) + 1), "x": x, "y": y, "z": z, "layer": layer, "module": module,
            "particle_id": pid, "weight": w}
    parts = {"particle_id": np.arange(1, n + 1), "pt": pt, "eta": eta, "phi": phi0, "vx": np.zeros(n),
             "vy": np.zeros(n), "vz": vz, "q": q, "nhits": nh}
    meta = {"source": "synthetic", "event_index": int(event_index), "n_layers": det.n_layers,
            "detector": asdict(det), "generator": asdict(cfg)}
    return EventData(f"toy{event_index:06d}", hits, parts, meta)


def generate_events(n_events: int, cfg: Optional[ToyConfig] = None, det: Optional[ToyDetector] = None,
                    start: int = 0) -> List[EventData]:
    cfg = cfg or ToyConfig()
    det = det or ToyDetector()
    return [generate_event(det, cfg, start + i) for i in range(n_events)]


def write_dataset(out_dir: Path, n_events: int, cfg: Optional[ToyConfig] = None,
                  det: Optional[ToyDetector] = None) -> List[Path]:
    out = []
    for ev in generate_events(n_events, cfg, det):
        out.append(ev.save(Path(out_dir) / "events" / f"{ev.event_id}.npz"))
    return out

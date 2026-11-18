"""OpenDataDetector (ODD) export of the ACTS chain of Phase 1 -> EventData, for "GNN versus CKF on the same events".

The export is written by scripts/odd_export.py (hsf2_scripts/odd_export.py on lxplus), which runs Phase 1's
ckf_odd.run_ckf (Fatras -> digitisation -> seeding -> CKF) with the same RandomNumbers(seed) and the same particle
file, and adds CSV writers.  Folder layout of one export (hsf2_work/odd_export/<DD-MM-YYYY-HH-MM>_<label>/):

    export_manifest.json                      seed, events, ACTS version, collection names, geometry-id check, files
    surfaces.csv                              geometry_id, volume, layer, sensitive of every sensitive surface (ACTS-decoded)
    truth/simhits/event<N>-*.csv              ACTS CsvSimHitWriter: particle id, geometry_id, tx, ty, tz, ..., index
    truth/measurements/event<N>-measurements.csv            measurement_id, geometry_id, local0, local1, ...
    truth/measurements/event<N>-measurement-simhit-map.csv  measurement_id, hit_id
    truth/particles/event<N>-*.csv            ACTS CsvParticleWriter: particle id, vx, vy, vz, px, py, pz, q, ...

The particle id is one column `particle_id` in older ACTS versions and five barcode columns (particle_id_pv, _sv,
_part, _gen, _subpart) in newer ones, e.g. the build on lxplus (13 November 2026); particle_ids() packs the five into
ACTS's classic 64-bit barcode, so both layouts give the same ids.
    ckf_tracks/event<N>-ckf_tracks.csv        track_id, measurement_id  (the final CKF tracks; written by the exporter)
    ckf_performance.json                      efficiency / fakerate / duplicaterate of performance_finding_ckf.root
    ckf/                                      the raw ckf_odd output (ROOT files, timing.csv, run_settings.json)

A "hit" of EventData is one ACTS measurement (hit_id = measurement_id).  The ODD has no noise hits: every measurement
comes from a simulated hit of a particle.  Position of a hit = the truth position of its simhit (tx, ty, tz), i.e. the
GNN sees the unsmeared crossing point while the CKF fits the smeared measurement (digitisation smearing of the ODD:
tens of micrometres in the pixels, which is small against the module spacing; stated in every comparison report).

Geometry identifier (acts::GeometryIdentifier, 64 bit, as in the ACTS versions with an 8/8/12/8/20/8 layout):

    bits 56-63  volume      0xff00000000000000
    bits 48-55  boundary    0x00ff000000000000
    bits 36-47  layer       0x0000fff000000000
    bits 28-35  approach    0x0000000ff0000000
    bits  8-27  sensitive   0x000000000fffff00
    bits  0- 7  extra       0x00000000000000ff

If the export has a surfaces.csv, the exporter has compared this decoding with the ACTS bindings for every sensitive
surface (and written the ACTS values), and the reader uses that table instead of the bit masks, so a changed bit
layout in another ACTS version cannot silently produce wrong layers.  The detector-wide layer index of EventData is
the position of (volume, layer) in the sorted list of all (volume, layer) pairs of the sensitive surfaces.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .event import EventData

EXPORT_MANIFEST = "export_manifest.json"
SURFACES_CSV = "surfaces.csv"
TRACKS_DIR = "ckf_tracks"
PERFORMANCE_JSON = "ckf_performance.json"
TRUTH_DIRS = {"simhits": "truth/simhits", "measurements": "truth/measurements", "particles": "truth/particles"}

GEO_MASKS = {"volume": 0xFF00000000000000, "boundary": 0x00FF000000000000, "layer": 0x0000FFF000000000,
             "approach": 0x0000000FF0000000, "sensitive": 0x000000000FFFFF00, "extra": 0x00000000000000FF}
GEO_SHIFTS = {k: (m & -m).bit_length() - 1 for k, m in GEO_MASKS.items()}
STATE_TOL_MM = 2e-3                     # local-coordinate tolerance when matching CKF track states to measurements
# ACTS's classic 64-bit barcode: (column of the split layout, bit shift, bit width)
BARCODE_FIELDS = (("particle_id_pv", 52, 12), ("particle_id_sv", 40, 12), ("particle_id_part", 24, 16),
                  ("particle_id_gen", 16, 8), ("particle_id_subpart", 0, 16))


# ---------------------------------------------------------------------------
# geometry identifier
# ---------------------------------------------------------------------------
def geo_decode(geoid) -> Dict[str, np.ndarray]:
    """Fields of 64-bit geometry identifiers (array or scalar) by the bit layout in the module docstring."""
    g = np.atleast_1d(np.asarray(geoid, dtype=np.uint64))
    return {k: ((g & np.uint64(m)) >> np.uint64(GEO_SHIFTS[k])).astype(np.int64) for k, m in GEO_MASKS.items()}


def geo_encode(volume=0, boundary=0, layer=0, approach=0, sensitive=0, extra=0) -> np.uint64:
    """Inverse of geo_decode (every field must fit its bit range)."""
    vals = dict(volume=volume, boundary=boundary, layer=layer, approach=approach, sensitive=sensitive, extra=extra)
    out = 0
    for k, v in vals.items():
        v = int(v)
        if v < 0 or (v << GEO_SHIFTS[k]) & ~GEO_MASKS[k]:
            raise ValueError(f"geometry-id field {k}={v} does not fit its bits")
        out |= v << GEO_SHIFTS[k]
    return np.uint64(out)


def read_surfaces(export_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """surfaces.csv of an export (sorted by geometry id), or None if the exporter could not write it."""
    path = Path(export_dir) / SURFACES_CSV
    if not path.is_file():
        return None
    df = _read_csv(path, {"geometry_id": np.uint64}, ["geometry_id", "volume", "layer", "sensitive"])
    order = np.argsort(df["geometry_id"].to_numpy(np.uint64), kind="stable")
    return {k: df[k].to_numpy()[order] for k in ("geometry_id", "volume", "layer", "sensitive")}


def decode_geometry_ids(geoid, surfaces: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """volume, layer, sensitive of geometry ids: from the ACTS-decoded surfaces table if given, else by bit masks."""
    g = np.atleast_1d(np.asarray(geoid, dtype=np.uint64))
    if surfaces is None:
        d = geo_decode(g)
        return {k: d[k] for k in ("volume", "layer", "sensitive")}
    tab = surfaces["geometry_id"].astype(np.uint64)
    pos = np.clip(np.searchsorted(tab, g), 0, max(len(tab) - 1, 0))
    bad = (len(tab) == 0) | (tab[pos] != g)
    if np.any(bad):
        raise KeyError(f"{int(np.sum(bad))} geometry ids are not in surfaces.csv, e.g. {int(g[bad][0])}")
    return {k: surfaces[k][pos].astype(np.int64) for k in ("volume", "layer", "sensitive")}


def layer_index_from_surfaces(surfaces: Dict[str, np.ndarray]) -> Dict[Tuple[int, int], int]:
    """(volume, layer) -> detector-wide layer index (sorted order of the sensitive surfaces' pairs)."""
    pairs = sorted(set(zip(surfaces["volume"].astype(int).tolist(), surfaces["layer"].astype(int).tolist())))
    return {p: i for i, p in enumerate(pairs)}


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------
def _read_csv(path: Path, dtypes: dict, need: Sequence[str]):
    import pandas as pd
    head = [c.strip() for c in pd.read_csv(path, nrows=0).columns]
    df = pd.read_csv(path, dtype={k: v for k, v in dtypes.items() if k in head})   # dtypes only of existing columns
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: columns {missing} missing (found {list(df.columns)})")
    return df


def particle_ids(df, path="") -> np.ndarray:
    """int64 particle ids of a simhit or particle table: the column particle_id (older ACTS) or the five barcode
    columns of newer ACTS packed into the classic 64-bit barcode (vertex primary 12 bits, vertex secondary 12,
    particle 16, generation 8, sub-particle 16).  A component outside its bit width raises."""
    if "particle_id" in df.columns:
        return df["particle_id"].to_numpy(np.uint64).view(np.int64)
    if all(c in df.columns for c, _, _ in BARCODE_FIELDS):
        out = np.zeros(len(df), dtype=np.uint64)
        for c, shift, bits in BARCODE_FIELDS:
            v = df[c].to_numpy(np.int64)
            if len(v) and (int(v.min()) < 0 or int(v.max()) >= (1 << bits)):
                raise ValueError(f"{path}: {c} outside 0..{(1 << bits) - 1}; the barcode cannot be packed into 64 bits")
            out |= v.astype(np.uint64) << np.uint64(shift)
        return out.view(np.int64)
    raise ValueError(f"{path}: no particle id: neither particle_id nor the columns "
                     f"{[c for c, _, _ in BARCODE_FIELDS]} (found {list(df.columns)})")


def require_complete(export_dir: Path, man: Optional[dict] = None) -> dict:
    """The manifest of an export whose status is 'complete'; any other status (an export still running, killed or
    with failed post-processing) raises, so that a half-written export is never read."""
    man = man if man is not None else load_manifest(export_dir)
    status = man.get("status")
    if status != "complete":
        raise ValueError(f"{export_dir}: export status {status!r}, not 'complete' (killed, still running or failed "
                         "post-processing): use only complete exports")
    return man


def load_manifest(export_dir: Path) -> dict:
    path = Path(export_dir) / EXPORT_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: not an export folder of odd_export.py")
    man = json.loads(path.read_text())
    for k in ("seed", "events", "export_id"):
        if k not in man:
            raise ValueError(f"{path}: key {k!r} missing")
    return man


def event_file(folder: Path, event_nr: int, suffix: Optional[str] = None) -> Path:
    """The ACTS per-event file event<N>-<name>.csv of a folder (N zero-padded or not); suffix selects the name, e.g.
    'measurements'; without a suffix the folder must hold exactly one file of this event."""
    folder = Path(folder)
    found = []
    pat = re.compile(r"^event0*(\d+)-(.+)\.csv$")
    for p in folder.iterdir() if folder.is_dir() else []:
        m = pat.match(p.name)
        if m and int(m.group(1)) == int(event_nr) and (suffix is None or m.group(2) == suffix):
            found.append(p)
    if len(found) != 1:
        raise FileNotFoundError(f"{folder}: expected exactly one event {event_nr} file"
                                f"{'' if suffix is None else ' ' + repr(suffix)}, found {[p.name for p in found]}")
    return found[0]


def read_measurements(export_dir: Path, event_nr: int) -> Dict[str, np.ndarray]:
    """measurement_id, geometry_id (uint64), loc0, loc1 of one event, sorted by measurement_id."""
    p = event_file(Path(export_dir) / TRUTH_DIRS["measurements"], event_nr, "measurements")
    df = _read_csv(p, {"geometry_id": np.uint64, "measurement_id": np.int64},
                   ["measurement_id", "geometry_id", "local0", "local1"])
    df = df.sort_values("measurement_id", kind="stable")
    if df["measurement_id"].duplicated().any():
        raise ValueError(f"{p}: duplicate measurement_id")
    return {"measurement_id": df["measurement_id"].to_numpy(np.int64),
            "geometry_id": df["geometry_id"].to_numpy(np.uint64),
            "loc0": df["local0"].to_numpy(np.float64), "loc1": df["local1"].to_numpy(np.float64)}


def read_ckf_tracks(export_dir: Path, event_nr: int) -> List[np.ndarray]:
    """Final CKF tracks of one event as arrays of measurement ids (= EventData hit_id)."""
    folder = Path(export_dir) / TRACKS_DIR
    try:
        p = event_file(folder, event_nr, "ckf_tracks")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}; the exporter writes these files from trackstates_ckf.root "
                                "(see ckf_tracks_report.json of the export)") from None
    df = _read_csv(p, {"track_id": np.int64, "measurement_id": np.int64}, ["track_id", "measurement_id"])
    if len(df) == 0:
        return []
    out = []
    for _, g in df.groupby("track_id", sort=True):
        out.append(g["measurement_id"].to_numpy(np.int64))
    return out


def simhit_rows(sh, hit_idx: np.ndarray, meas_gid: np.ndarray, event_nr: int = -1) -> Tuple[np.ndarray, str]:
    """(row of the simhit file for every measurement, how it was found).

    hit_id of the measurement-simhit map is the simhit's position in ACTS's simhit container, and CsvSimHitWriter
    writes the container in order, so row k of the file is simhit k.  The file's `index` column is the hit's index
    along its particle's trajectory (not unique; found on lxplus, 13 November 2026), so it is used only as a fallback
    when it is unique and the row order does not fit.  A mapping is accepted only if every measurement's geometry id
    equals its simhit's."""
    gid = sh["geometry_id"].to_numpy(np.uint64)
    meas_gid = np.asarray(meas_gid, dtype=np.uint64)
    n = len(sh)
    tried = []
    if n and len(hit_idx) and int(hit_idx.min()) >= 0 and int(hit_idx.max()) < n:
        bad = int(np.sum(gid[hit_idx] != meas_gid))
        if bad == 0:
            return hit_idx.astype(np.int64), "row order"
        tried.append(f"row order: {bad} of {len(hit_idx)} geometry ids differ")
    else:
        tried.append(f"row order: hit ids outside 0..{n - 1}")
    if "index" in sh.columns:
        sidx = sh["index"].to_numpy(np.int64)
        if len(np.unique(sidx)) == n:
            order = np.argsort(sidx, kind="stable")
            p = np.clip(np.searchsorted(sidx[order], hit_idx), 0, max(n - 1, 0))
            if n and np.all(sidx[order][p] == hit_idx):
                srow = order[p]
                bad = int(np.sum(gid[srow] != meas_gid))
                if bad == 0:
                    return srow.astype(np.int64), "index column"
                tried.append(f"index column: {bad} geometry ids differ")
            else:
                tried.append("index column: hit ids missing")
        else:
            tried.append("index column not unique (hit index along the trajectory)")
    raise ValueError(f"event {event_nr}: the measurement-simhit map does not fit the simhit file ({'; '.join(tried)})")


def event_id_of(seed: int, event_nr: int) -> str:
    return f"odd_s{int(seed):09d}_e{int(event_nr):06d}"


# ---------------------------------------------------------------------------
# measured hit positions: local frames of the sensitive surfaces, fitted from (measurement, simhit) pairs
# ---------------------------------------------------------------------------
FRAMES_FILE = "surface_frames.npz"
FRAME_MIN_HITS = 3


def _event_pairs(export_dir: Path, event_nr: int) -> Dict[str, np.ndarray]:
    """Per measurement of one event: geometry id, loc0, loc1, var_local1 and the global position of its simhit."""
    ex = Path(export_dir)
    p = event_file(ex / TRUTH_DIRS["measurements"], event_nr, "measurements")
    df = _read_csv(p, {"geometry_id": np.uint64, "measurement_id": np.int64},
                   ["measurement_id", "geometry_id", "local0", "local1", "var_local1"])
    df = df.sort_values("measurement_id", kind="stable")
    sh_path = event_file(ex / TRUTH_DIRS["simhits"], event_nr)
    sh = _read_csv(sh_path, {"geometry_id": np.uint64}, ["geometry_id", "tx", "ty", "tz"])
    mm = _read_csv(event_file(ex / TRUTH_DIRS["measurements"], event_nr, "measurement-simhit-map"),
                   {"measurement_id": np.int64, "hit_id": np.int64}, ["measurement_id", "hit_id"])
    mm = mm.drop_duplicates("measurement_id", keep="first").sort_values("measurement_id", kind="stable")
    mid = df["measurement_id"].to_numpy(np.int64)
    ids = mm["measurement_id"].to_numpy(np.int64)
    pos = np.clip(np.searchsorted(ids, mid), 0, max(len(ids) - 1, 0))
    if len(ids) == 0 or np.any(ids[pos] != mid):
        raise ValueError(f"event {event_nr}: measurements without simhit in the measurement-simhit map")
    geo = df["geometry_id"].to_numpy(np.uint64)
    srow, _ = simhit_rows(sh, mm["hit_id"].to_numpy(np.int64)[pos], geo, event_nr)
    return {"geometry_id": geo, "loc0": df["local0"].to_numpy(np.float64), "loc1": df["local1"].to_numpy(np.float64),
            "var1": df["var_local1"].to_numpy(np.float64),
            "g": np.column_stack([sh[k].to_numpy(np.float64)[srow] for k in ("tx", "ty", "tz")])}


def fit_surface_frames(sources: Sequence[Tuple[Path, int]], log=None) -> Dict[str, np.ndarray]:
    """Local frames (origin O, axes u, v) of the sensitive surfaces, so that a measurement's global position is
    O + loc0 u + loc1 v.  The exports have no surface transforms (the CSV writer leaves global_x/y/z empty), so the
    frames are fitted from the events in `sources` ((export folder, event number) pairs; use training events only):

    * the plane from the simhit positions on the surface (their smallest principal direction is the normal);
    * u (and v for two-coordinate measurements) by regressing the MEASURED local coordinate on the in-plane simhit
      position: the measurement noise is then in the dependent variable, so the fit is unbiased (the reverse
      regression is biased by the strip smearing of 1.2 mm along short strips);
    * one-coordinate measurements (loc1 always 0 and var_local1 0: the long strips) get v = n x u and the origin on
      the centre line of the hit range along v, i.e. the strip centre: the position along the strip is unknown, as
      for the CKF, which uses only loc0 there.
    Surfaces with fewer than FRAME_MIN_HITS hits or collinear hits get no frame."""
    acc: Dict[int, List[np.ndarray]] = {}
    for k, (ex, nr) in enumerate(sources, 1):
        d = _event_pairs(ex, nr)
        rows = np.column_stack([d["loc0"], d["loc1"], d["var1"], d["g"]])
        gid = d["geometry_id"]
        order = np.argsort(gid, kind="stable")
        gs, rs = gid[order], rows[order]
        starts = np.nonzero(np.r_[True, gs[1:] != gs[:-1]])[0]
        for a, b in zip(starts, np.r_[starts[1:], len(gs)]):
            acc.setdefault(int(gs[a]), []).append(rs[a:b])
        if log and (k % 50 == 0 or k == len(sources)):
            log(f"  frames: {k}/{len(sources)} events read")
    keys = sorted(acc)
    out = {"geometry_id": np.array(keys, dtype=np.uint64), "origin": np.full((len(keys), 3), np.nan),
           "u": np.full((len(keys), 3), np.nan), "v": np.full((len(keys), 3), np.nan),
           "two_d": np.zeros(len(keys), bool), "n_hits": np.zeros(len(keys), np.int64),
           "scale_u": np.full(len(keys), np.nan), "rms_u": np.full(len(keys), np.nan)}
    for i, key in enumerate(keys):
        r = np.concatenate(acc[key])
        out["n_hits"][i] = len(r)
        if len(r) < FRAME_MIN_HITS:
            continue
        loc0, loc1, var1, g = r[:, 0], r[:, 1], r[:, 2], r[:, 3:6]
        p0 = g.mean(axis=0)
        x = g - p0
        _, sv, vt = np.linalg.svd(x, full_matrices=False)
        if len(sv) < 2 or sv[1] < 1e-6 * max(sv[0], 1e-12):
            continue                                            # collinear hits: no plane
        e = vt[:2]
        n = np.cross(e[0], e[1])
        xy = x @ e.T
        A = np.column_stack([xy, np.ones(len(r))])
        c0, *_ = np.linalg.lstsq(A, loc0, rcond=None)
        u_raw = c0[0] * e[0] + c0[1] * e[1]
        su = float(np.linalg.norm(u_raw))
        if su == 0:
            continue
        u = u_raw / su
        two_d = bool(np.any(var1 > 0) and np.ptp(loc1) > 0)
        if two_d:
            c1, *_ = np.linalg.lstsq(A, loc1, rcond=None)
            v = c1[0] * e[0] + c1[1] * e[1]
            v = v - (v @ u) * u                                 # orthogonal to u, sign from the fit
            v /= np.linalg.norm(v)
            b1 = float(np.mean(loc1 - x @ v))
        else:
            v = np.cross(n, u)
            t = x @ v
            b1 = -0.5 * float(t.min() + t.max())                # loc1 = 0 at the centre of the hit range
        b0 = float(np.mean(loc0 - x @ u))
        out["origin"][i] = p0 - b0 * u - b1 * v
        out["u"][i], out["v"][i], out["two_d"][i] = u, v, two_d
        out["scale_u"][i] = su
        out["rms_u"][i] = float(np.sqrt(np.mean((loc0 - (x @ u + b0)) ** 2)))
    return out


def save_frames(frames: Dict[str, np.ndarray], path: Path) -> None:
    np.savez_compressed(path, **frames)


def load_frames(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def one_coordinate_layers(frames: Dict[str, np.ndarray], layers: Sequence[Sequence[int]]) -> List[int]:
    """Detector-wide layer indices (position in `layers`, the data manifest's sorted (volume, layer) pairs) in which
    no surface with hits has a measured second coordinate: on measured positions their hits lie on the strip centre
    line.  A few surfaces of 2-D layers with only 1-2 training hits are also flagged 1-D; they do not make a layer 1-D."""
    index = {(int(v), int(l)): i for i, (v, l) in enumerate(layers)}
    dec = decode_geometry_ids(frames["geometry_id"])
    two_d: Dict[int, bool] = {}
    for v, l, t, n in zip(dec["volume"], dec["layer"], frames["two_d"], frames["n_hits"]):
        i = index.get((int(v), int(l)))
        if i is not None and n > 0:
            two_d[i] = two_d.get(i, False) or bool(t)
    return sorted(i for i, t in two_d.items() if not t)


def measured_positions(frames: Dict[str, np.ndarray], geometry_id, loc0, loc1) -> Tuple[np.ndarray, np.ndarray]:
    """Global positions O + loc0 u + loc1 v of measurements; second value: True where the surface has no frame."""
    g = np.asarray(geometry_id, dtype=np.uint64)
    tab = frames["geometry_id"].astype(np.uint64)
    pos = np.clip(np.searchsorted(tab, g), 0, max(len(tab) - 1, 0))
    known = (len(tab) > 0) & (tab[pos] == g)
    known &= np.isfinite(frames["origin"][pos, 0])
    l0 = np.asarray(loc0, dtype=np.float64)[:, None]
    l1 = np.where(frames["two_d"][pos], np.asarray(loc1, dtype=np.float64), 0.0)[:, None]
    xyz = frames["origin"][pos] + l0 * frames["u"][pos] + l1 * frames["v"][pos]
    return xyz, ~known


def read_event(export_dir: Path, event_nr: int, layer_index: Optional[Dict[Tuple[int, int], int]] = None,
               event_id: Optional[str] = None, frames: Optional[Dict[str, np.ndarray]] = None) -> EventData:
    """One exported event as EventData (all measurements as hits; truth from the simhits; no noise).  Hit positions:
    the simhit (truth) positions, or with `frames` (fit_surface_frames) the measured ones, O + loc0 u + loc1 v; a
    measurement on a surface without a frame keeps its simhit position and is counted in meta."""
    ex = Path(export_dir)
    man = load_manifest(ex)
    surfaces = read_surfaces(ex)
    meas = read_measurements(ex, event_nr)
    # simhits
    sh_path = event_file(ex / TRUTH_DIRS["simhits"], event_nr)
    sh = _read_csv(sh_path, {"particle_id": np.uint64, "geometry_id": np.uint64}, ["geometry_id", "tx", "ty", "tz"])
    sh_pid = particle_ids(sh, sh_path)
    # measurement -> simhit (first simhit of a measurement; merged measurements would have several)
    mm = _read_csv(event_file(ex / TRUTH_DIRS["measurements"], event_nr, "measurement-simhit-map"),
                   {"measurement_id": np.int64, "hit_id": np.int64}, ["measurement_id", "hit_id"])
    n_multi = int(mm["measurement_id"].duplicated().sum())
    mm = mm.drop_duplicates("measurement_id", keep="first")
    mid = meas["measurement_id"]
    mm_sorted = mm.sort_values("measurement_id", kind="stable")
    mm_ids = mm_sorted["measurement_id"].to_numpy(np.int64)
    pos = np.clip(np.searchsorted(mm_ids, mid), 0, max(len(mm_ids) - 1, 0))
    if len(mm_ids) == 0 or np.any(mm_ids[pos] != mid):
        raise ValueError(f"event {event_nr}: measurements without simhit in the measurement-simhit map")
    hit_idx = mm_sorted["hit_id"].to_numpy(np.int64)[pos]
    srow, simhit_map = simhit_rows(sh, hit_idx, meas["geometry_id"], event_nr)
    geo = decode_geometry_ids(meas["geometry_id"], surfaces)
    if layer_index is None:
        layer_index = (layer_index_from_surfaces(surfaces) if surfaces is not None else
                       {p: i for i, p in enumerate(sorted(set(zip(geo["volume"].tolist(), geo["layer"].tolist()))))})
    try:
        layer = np.fromiter((layer_index[(int(v), int(l))] for v, l in zip(geo["volume"], geo["layer"])),
                            dtype=np.int64, count=len(mid))
    except KeyError as exc:
        raise ValueError(f"event {event_nr}: hit on unknown (volume, layer) {exc}") from None
    pid = sh_pid[srow]
    # particles with at least one hit
    pt_path = event_file(ex / TRUTH_DIRS["particles"], event_nr)
    pt_ = _read_csv(pt_path, {"particle_id": np.uint64}, ["vx", "vy", "vz", "px", "py", "pz", "q"])
    ppid = particle_ids(pt_, pt_path)
    upid, cnt = np.unique(pid, return_counts=True)
    keep = np.isin(ppid, upid)
    px, py, pz = (pt_[k].to_numpy(np.float64)[keep] for k in ("px", "py", "pz"))
    ptm = np.hypot(px, py)
    ppid_k = ppid[keep]
    nhits = cnt[np.searchsorted(upid, ppid_k)]
    particles = {"particle_id": ppid_k, "pt": ptm, "eta": np.arcsinh(pz / np.maximum(ptm, 1e-9)),
                 "phi": np.arctan2(py, px), "vx": pt_["vx"].to_numpy(np.float64)[keep],
                 "vy": pt_["vy"].to_numpy(np.float64)[keep], "vz": pt_["vz"].to_numpy(np.float64)[keep],
                 "q": np.rint(pt_["q"].to_numpy(np.float64)[keep]), "nhits": nhits}
    xyz = np.column_stack([sh[k].to_numpy(np.float64)[srow] for k in ("tx", "ty", "tz")])
    n_no_frame = 0
    if frames is not None:
        mxyz, no_frame = measured_positions(frames, meas["geometry_id"], meas["loc0"], meas["loc1"])
        xyz = np.where(no_frame[:, None], xyz, mxyz)
        n_no_frame = int(no_frame.sum())
    hits = {"hit_id": mid, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "layer": layer, "module": geo["sensitive"],
            "particle_id": pid, "weight": np.zeros(len(mid))}
    meta = {"source": "odd", "export_id": man["export_id"], "seed": int(man["seed"]), "event_nr": int(event_nr),
            "position_source": "measured" if frames is not None else "simhit_truth",
            "n_positions_without_frame": n_no_frame, "n_layers": len(layer_index),
            "geometry_decode": "surfaces.csv" if surfaces is not None else "bit masks",
            "n_particles_with_hits": int(len(ppid_k)), "n_multi_simhit": n_multi, "simhit_map": simhit_map,
            "n_hits_particle_missing": int(np.sum(~np.isin(upid, ppid))) if len(upid) else 0}
    return EventData(event_id or event_id_of(man["seed"], event_nr), hits, particles, meta)


# ---------------------------------------------------------------------------
# CKF track states (trackstates_ckf.root) -> measurement ids
# ---------------------------------------------------------------------------
def match_states_to_measurements(meas: Dict[str, np.ndarray], surfaces: Optional[Dict[str, np.ndarray]],
                                 volume, layer, sensitive, lx, ly, tol: float = STATE_TOL_MM) -> np.ndarray:
    """Measurement id of every track state (-1 if none): same surface (volume, layer, sensitive) and local position
    (lx, ly) within `tol` mm of the measurement's (loc0, loc1).  A coordinate that is NaN in either (1-d strip
    measurements) is ignored.  States and measurements come from the same ACTS run, so the match is exact up to the
    float precision of the files."""
    volume, layer, sensitive = (np.asarray(a, dtype=np.int64) for a in (volume, layer, sensitive))
    lx, ly = np.asarray(lx, dtype=np.float64), np.asarray(ly, dtype=np.float64)
    gd = decode_geometry_ids(meas["geometry_id"], surfaces)
    mkey = (gd["volume"] << 40) | (gd["layer"] << 28) | gd["sensitive"]
    skey = (volume << 40) | (layer << 28) | sensitive
    out = np.full(len(skey), -1, dtype=np.int64)
    order = np.lexsort((meas["loc0"], mkey))
    mk, m0, m1, mi = mkey[order], meas["loc0"][order], meas["loc1"][order], meas["measurement_id"][order]
    for key in np.unique(skey):
        lo, hi = np.searchsorted(mk, key, "left"), np.searchsorted(mk, key, "right")
        if hi == lo:
            continue
        sel = np.nonzero(skey == key)[0]
        g0, g1, gi = m0[lo:hi], m1[lo:hi], mi[lo:hi]
        j = np.searchsorted(g0, lx[sel])
        best = np.full(len(sel), np.inf)
        arg = np.full(len(sel), -1, dtype=np.int64)
        for d in (-2, -1, 0, 1):                         # neighbours in loc0 order: ties in loc0 are resolved by loc1
            k = np.clip(j + d, 0, hi - lo - 1)
            d0 = np.abs(g0[k] - lx[sel])
            d1 = np.abs(g1[k] - ly[sel])
            dist = np.where(np.isnan(d0), 0.0, d0) + np.where(np.isnan(d1), 0.0, d1)
            better = dist < best
            best = np.where(better, dist, best)
            arg = np.where(better, k, arg)
        ok = best <= tol
        out[sel[ok]] = gi[arg[ok]]
    return out


STATE_ALIASES = {"event": ("event_nr",), "volume": ("volume_id",), "layer": ("layer_id",),
                 "sensitive": ("module_id", "sensitive_id"), "lx": ("l_x_hit", "lx_hit"), "ly": ("l_y_hit", "ly_hit")}


def convert_trackstates_root(export_dir: Path, root_file: Path, tree: Optional[str] = None) -> dict:
    """trackstates_ckf.root (ACTS RootTrackStatesWriter, one entry per final CKF track) -> ckf_tracks/event<N>-
    ckf_tracks.csv with the measurement ids of every track, matched by surface and local position.  Needs uproot
    (the ACTS environment of Phase 1 has it).  Fails with the list of branches if an expected one is missing.
    A state counts as a measurement if its local hit position is finite (holes carry no hit)."""
    try:
        import uproot
    except ImportError as exc:
        raise ImportError("uproot is required to read the CKF track states (pip install uproot)") from exc
    ex = Path(export_dir)
    man = load_manifest(ex)
    surfaces = read_surfaces(ex)
    wanted = {int(e) for e in man["events"]}
    out_dir = ex / TRACKS_DIR
    out_dir.mkdir(exist_ok=True)
    report = {"root_file": str(root_file), "events": {}, "n_states": 0, "n_states_unmatched": 0}
    with uproot.open(str(root_file)) as f:
        names = [k.split(";")[0] for k in f.keys()]
        tname = tree or ("trackstates" if "trackstates" in names else (names[0] if len(names) == 1 else None))
        if tname is None:
            raise KeyError(f"{root_file}: cannot tell which tree holds the track states; trees {names}")
        t = f[tname]
        branches = list(t.keys())
        pick = {}
        for key, aliases in STATE_ALIASES.items():
            hit = next((a for a in aliases if a in branches), None)
            if hit is None:
                raise KeyError(f"{root_file}:{tname} has none of the branches {aliases} for {key!r}; "
                               f"branches: {branches}")
            pick[key] = hit
        arrs = t.arrays(list(pick.values()), library="np")
    ev_nr = np.asarray(arrs[pick["event"]]).astype(np.int64)
    by_event: Dict[int, List[int]] = {}
    for i, e in enumerate(ev_nr.tolist()):
        by_event.setdefault(int(e), []).append(i)
    missing_events = sorted(wanted - set(by_event))
    for e in sorted(wanted):
        entries = by_event.get(e, [])
        meas = read_measurements(ex, e)
        rows_t, rows_m = [], []
        n_state = n_bad = 0
        for tid, i in enumerate(entries):
            vol, lay, sen, lx, ly = (np.asarray(arrs[pick[k]][i]) for k in ("volume", "layer", "sensitive", "lx", "ly"))
            ism = np.isfinite(lx.astype(np.float64))
            ids = match_states_to_measurements(meas, surfaces, vol[ism], lay[ism], sen[ism], lx[ism], ly[ism])
            n_state += len(ids)
            n_bad += int(np.sum(ids < 0))
            rows_t += [tid] * int(np.sum(ids >= 0))
            rows_m += ids[ids >= 0].tolist()
        path = out_dir / f"event{e:09d}-ckf_tracks.csv"
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="\n") as fh:
            fh.write("track_id,measurement_id\n")
            fh.writelines(f"{a},{b}\n" for a, b in zip(rows_t, rows_m))
        tmp.replace(path)
        report["events"][str(e)] = {"n_tracks": len(entries), "n_states": n_state, "n_unmatched": n_bad}
        report["n_states"] += n_state
        report["n_states_unmatched"] += n_bad
    report["events_without_tracks_in_root"] = missing_events
    report["branches_used"] = pick
    return report


# ---------------------------------------------------------------------------
# data folder (odd-prep)
# ---------------------------------------------------------------------------
def event_row(ev: EventData) -> dict:
    return {"event_id": ev.event_id, "n_hits": ev.n_hits, "n_particles": ev.n_particles, "n_noise": 0,
            "n_particles_pt1": int((ev.particles["pt"] >= 1.0).sum()),
            "n_hits_pt1": int((ev.hit_particle("pt") >= 1.0).sum()), "n_hits_particle_missing":
            int(ev.meta.get("n_hits_particle_missing", 0))}


def prep_from_exports(export_dirs: Sequence[Path], out_dir: Path, n_events: int, log=None,
                      positions: str = "simhit", n_frame_events: int = 0):
    """The first n_events events (sorted by seed, event number) of the exports -> out_dir/events/<event>.npz.

    Returns (rows, exports, layer_index, frame_info).  All events share one layer index (from surfaces.csv when the
    exports have it, else from the (volume, layer) pairs of all events read here).  positions 'measured': hit
    positions from the measurements, with surface frames fitted on the first `n_frame_events` events only (the
    training events), saved as out_dir/surface_frames.npz."""
    if positions not in ("simhit", "measured"):
        raise ValueError(f"positions must be 'simhit' or 'measured', got {positions!r}")
    exports = []
    for d in export_dirs:
        man = require_complete(d)
        exports.append({"path": str(Path(d).resolve()), "export_id": man["export_id"], "seed": int(man["seed"]),
                        "n_events": len(man["events"]), "events": [int(e) for e in man["events"]],
                        "acts_version": man.get("acts_version")})
    seeds = [e["seed"] for e in exports]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"two exports with the same seed {sorted(seeds)}: their events would be identical")
    todo = sorted((e["seed"], nr, e["path"]) for e in exports for nr in e["events"])
    if n_events > len(todo):
        raise SystemExit(f"{n_events} events requested but the exports hold {len(todo)} "
                         f"({', '.join(str(e['n_events']) for e in exports)} per export): export more events or "
                         "lower --n-train/--n-val/--n-confirm")
    todo = todo[:n_events]
    surf = [read_surfaces(d) for d in export_dirs]
    if all(s is not None for s in surf):
        merged = {k: np.concatenate([s[k] for s in surf]) for k in surf[0]}
        layer_index = layer_index_from_surfaces(merged)
    else:
        layer_index = None
        pairs = set()
        for _, nr, path in todo:
            m = read_measurements(path, nr)
            g = decode_geometry_ids(m["geometry_id"], read_surfaces(path))
            pairs |= set(zip(g["volume"].tolist(), g["layer"].tolist()))
        layer_index = {p: i for i, p in enumerate(sorted(pairs))}
    frames, frame_info = None, {"positions": positions}
    if positions == "measured":
        if not 0 < n_frame_events <= len(todo):
            raise ValueError(f"n_frame_events must be in 1..{len(todo)} (the training events), got {n_frame_events}")
        src = [(Path(path), nr) for _, nr, path in todo[:n_frame_events]]
        frames = fit_surface_frames(src, log)
        save_frames(frames, Path(out_dir) / FRAMES_FILE)
        ok = np.isfinite(frames["origin"][:, 0])
        frame_info.update({"frame_events": [event_id_of(s, nr) for s, nr, _ in todo[:n_frame_events]][:3] + ["..."],
                           "n_frame_events": n_frame_events, "n_surfaces_seen": int(len(ok)),
                           "n_surfaces_with_frame": int(ok.sum()), "n_two_d": int(frames["two_d"][ok].sum()),
                           "scale_u_range": [float(np.nanmin(frames["scale_u"])), float(np.nanmax(frames["scale_u"]))]})
    rows = []
    n_no_frame = 0
    for k, (seed, nr, path) in enumerate(todo, 1):
        ev = read_event(path, nr, layer_index, frames=frames)
        n_no_frame += int(ev.meta.get("n_positions_without_frame", 0))
        target = Path(out_dir) / "events" / f"{ev.event_id}.npz"
        if not target.exists():
            ev.save(target)
        rows.append(event_row(ev))
        if log and (k % 10 == 0 or k == len(todo)):
            log(f"  {k}/{len(todo)} events")
    frame_info["n_positions_without_frame"] = n_no_frame
    return rows, exports, layer_index, frame_info

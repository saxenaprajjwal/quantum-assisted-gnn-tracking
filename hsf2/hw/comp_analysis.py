"""Analysis of QX3b component jobs on hardware (hw-components-analyze; it also runs after every local or submitted job).

For every component (one PUB of a job folder written by hsf2.hw.components), with N = the shots of the PUB:
  instance   n, couplings, 2q gates and estimated fidelity F_est of the transpiled circuit, the exact ground energy E0 (full
             enumeration), the ground degeneracy g, the uniform P(ground) = g / 2^n
  ideal      the noiseless p = 1 distribution of the circuit that was run (exact statevector, n <= 22): P_ideal(ground), <E>
  hardware   P(ground), F_E and F_XEB (hsf2.hw.analysis), the best-of-N energy and whether it is E0, and a single-flip
             steepest descent from every distinct sample: the fraction of shots whose descent ends at E0, and whether any does
  controls   uniform samples, ideal-QAOA samples and white-noise samples (p = F_E p_ideal + (1 - F_E) / 2^n with the F_E of the
             hardware), N of each, seeded: the same best-of-N and multi-start-descent statistics
  tracking   true and fake triplets selected by the best hardware solution, by the exact optimum, by greedy-S + descent and
             by the best of N uniform-start descents, and whether the selection equals exact_selection
Readout mitigation (analyze_jobs(readout=<readout_cal.json>, readout_tags=<regex over component names>)): the counts of the
selected PUBs are corrected with hsf2.hw.readout.correct_counts (mthree at full distance, else the numpy solve) using the
matrices of the PUB's measured qubits in classical-bit order (the calibration may cover more qubits than a component; the
measurement-twirling switch and the backend must agree), the nearest probability distribution times the shots is used as
weights in the same statistics, and the record gets "hardware_mitigated" and "readout" next to the raw "hardware" values (the
controls stay raw).
Aggregates by group (hard / benchmark / all), each with a bootstrap over the components (16 / 84 % interval): the median
P_hw(ground) / P_unif(ground), the fraction of components whose best-of-N reaches E0 (hardware, uniform, ideal, white noise),
and the same fraction with the descent added.  Outputs: components_hw.csv and components_hw_summary.json.

Descent: steepest descent on E(x) = a.x + sum b_ij x_i x_j, as hsf2.qubo.triplet.select_descent (flip the bit with the lowest
dE, ties: lowest index, stop when no flip lowers E by more than 1e-12), vectorised over many starts and bit for bit the same
arithmetic.  Greedy-S is rebuilt from the exported couplings (weights = sum of the quadruplet strengths, a triplet is skipped
if a selected one conflicts with it, ties by index): the export has no hole counts, so its tie-break by holes is missing; the
energy of greedy-S + descent is compared with greedy_descent_energy of components.csv (greedy_descent_matches_csv).
"""
from __future__ import annotations

import logging
import math
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._phase1 import ensure_phase1
from ..utils import load_json, save_csv, save_json
from .analysis import counts_arrays, f_energy, f_xeb
from .build import sparse_ising
from .components import (ENERGY_TOL, IDEAL_MAX_N, KIND, Component, component_energies, component_ising, ideal_distribution,
                         load_components)
from .snapshot import utc_now

log = logging.getLogger(__name__)

SCHEMA = 1
TOL = ENERGY_TOL                  # a state is a ground state if E <= E0 + TOL
DESCENT_EPS = 1e-12               # a flip must lower the energy by more than this (as triplet.select_descent)
COMP_BOOT_N = 1000
COMP_BOOT_SEED = 0
SOURCES = ("hardware", "uniform", "ideal", "white_noise")
CONTROL_STREAM = {"uniform": 1, "ideal": 2, "white_noise": 3}


# ---------------------------------------------------------------------------
# bits, couplings, steepest descent
# ---------------------------------------------------------------------------
def coupling_matrix(n: int, quad: Sequence[Tuple[int, int, float]]) -> np.ndarray:
    """Symmetric (n, n) matrix of the quadratic terms: B[i, j] = B[j, i] = b_ij."""
    B = np.zeros((n, n))
    for i, j, b in quad:
        B[i, j] += b
        B[j, i] += b
    return B


def bits_of(idx: np.ndarray, n: int) -> np.ndarray:
    """(m, n) float 0/1 array: bit i of the sample index = variable i."""
    idx = np.asarray(idx, dtype=np.int64)
    return ((idx[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.float64)


def index_of(X: np.ndarray) -> np.ndarray:
    """Inverse of bits_of."""
    X = np.rint(np.asarray(X)).astype(np.int64)
    return (X << np.arange(X.shape[1], dtype=np.int64)).sum(axis=1)


def batch_descent(linear: np.ndarray, B: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Steepest single-flip descent of E(x) = linear.x + sum_{i<j} B_ij x_i x_j from every row of X (m, n); returns the
    local minima reached.  Same rule and same floating-point operations as hsf2.qubo.triplet.select_descent (the initial
    field is computed with the same sparse product), so a start gives the same end point."""
    from scipy.sparse import csc_matrix
    linear = np.asarray(linear, dtype=np.float64)
    X = np.array(X, dtype=np.float64, copy=True)
    if X.ndim != 2 or X.shape[1] != linear.size or B.shape != (linear.size, linear.size):
        raise ValueError(f"shapes: X {X.shape}, linear {linear.shape}, B {B.shape}")
    if X.shape[0] == 0:
        return X
    field = linear[None, :] + (csc_matrix(B) @ X.T).T
    rows = np.arange(X.shape[0])
    while rows.size:
        dE = (1.0 - 2.0 * X[rows]) * field[rows]
        k = dE.argmin(axis=1)
        go = dE[np.arange(rows.size), k] < -DESCENT_EPS
        rows, k = rows[go], k[go]
        if not rows.size:
            break
        delta = 1.0 - 2.0 * X[rows, k]
        X[rows, k] += delta
        field[rows] += delta[:, None] * B[k]
    return X


def descent_indices(linear: np.ndarray, B: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    """Sample indices of the local minima that the descent reaches from the states `idx`."""
    return index_of(batch_descent(linear, B, bits_of(idx, n)))


def greedy_start(n: int, quad: Sequence[Tuple[int, int, float]]) -> np.ndarray:
    """Conflict-free greedy selection (bool, n): triplets in decreasing w_i = sum of the quadruplet strengths S = -b of the
    couplings that contain i; a triplet is added unless a selected one conflicts with it (b > 0).  Ties by index (see the
    module docstring)."""
    w = np.zeros(n)
    for i, _, b in quad:                    # the order of np.add.at(w, qi, S); np.add.at(w, qj, S) of triplet.select_greedy_s
        if b < 0:
            w[i] += -b
    for _, j, b in quad:
        if b < 0:
            w[j] += -b
    nbr: List[set] = [set() for _ in range(n)]
    for i, j, b in quad:
        if b > 0:
            nbr[i].add(j)
            nbr[j].add(i)
    chosen = np.zeros(n, bool)
    for i in np.lexsort((np.arange(n), -w)):
        if not any(chosen[j] for j in nbr[int(i)]):
            chosen[i] = True
    return chosen


# ---------------------------------------------------------------------------
# samples and controls
# ---------------------------------------------------------------------------
def rng_for(seed: int, name: str, stream: int) -> np.random.Generator:
    """A generator that depends on the seed, the component and the stream only (not on the order of the components)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(name.encode()), int(stream)]))


def draw_samples(p: Optional[np.ndarray], n: int, shots: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """`shots` samples (indices, counts) from p over 2^n states, or uniformly if p is None."""
    if p is None:
        s = rng.integers(0, 1 << n, size=shots, dtype=np.int64)
    else:
        q = np.asarray(p, dtype=np.float64)
        s = rng.choice(1 << n, size=shots, p=q / q.sum())
    idx, cnt = np.unique(s, return_counts=True)
    return idx.astype(np.int64), cnt.astype(np.int64)


def sample_stats(E: np.ndarray, E0: float, idx: np.ndarray, cnt: np.ndarray, linear: np.ndarray, B: np.ndarray, n: int) -> dict:
    """Ground-state statistics of a sample set (indices `idx`, counts `cnt`): P(ground), the best energy and whether it is
    E0, and the descent from every distinct sample (fraction of shots whose descent ends at E0, whether any does)."""
    S = float(cnt.sum())
    Es = E[idx]
    ground = Es <= E0 + TOL
    fin = descent_indices(linear, B, idx, n)
    Ef = E[fin]
    hit = Ef <= E0 + TOL
    best = int(np.lexsort((idx, -cnt, Es))[0])           # lowest energy, then the most frequent, then the lowest index
    dbest = int(np.lexsort((fin, Ef))[0])
    return {"shots": int(round(S)), "distinct": int(idx.size), "p_ground": float(cnt[ground].sum() / S),
            "mean_energy": float(cnt @ Es / S), "best_energy": float(Es[best]), "best_reaches_E0": bool(ground.any()),
            "best_index": int(idx[best]), "desc_frac_E0": float(cnt[hit].sum() / S), "desc_any_E0": bool(hit.any()),
            "desc_mean_energy": float(cnt @ Ef / S), "desc_best_energy": float(Ef[dbest]), "desc_best_index": int(fin[dbest])}


def track_content(i: int, comp: Component, E: np.ndarray, E0: float, i_exact: int) -> dict:
    """True and fake triplets selected by the state with index i."""
    n = comp.n
    x = ((int(i) >> np.arange(n)) & 1).astype(bool)
    t = comp.true_triplet
    return {"energy": float(E[i]), "n_selected": int(x.sum()), "n_true": int((x & t).sum()), "n_fake": int((x & ~t).sum()),
            "is_exact_selection": bool(int(i) == i_exact), "is_ground_state": bool(E[i] <= E0 + TOL)}


# ---------------------------------------------------------------------------
# one PUB
# ---------------------------------------------------------------------------
def mitigated_pub(counts: Dict[int, int], pub: dict, man: dict, cal: dict, prefer: str = "auto"
                  ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """(indices, weights, info) of the readout-mitigated counts of one PUB: weights = shots x the nearest probability
    distribution of the correction (see the module docstring); raises if the calibration does not fit the PUB."""
    from . import readout as R
    mq = [int(q) for q in pub["measured_qubits"]]
    cal_used = R.calibration_for(cal, mq, man["setting"]["measure_twirl"], man["backend"], subset=True)
    pr, ov, used = R.correct_counts(counts, cal_used, max(mq) + 1, prefer=prefer)
    idx = np.array(sorted(pr), np.int64)
    w = np.array([pr[int(i)] for i in idx], np.float64)
    if not abs(w.sum() - 1.0) < 1e-6:
        raise ValueError(f"{pub['component']}: mitigated distribution sums to {w.sum()}, not 1")
    S = float(sum(counts.values()))
    info = {"calibration": str(cal.get("_path", "")), "calibration_file": cal.get("_file"),
            "calibration_session": cal.get("session"), "calibration_job_id": cal.get("job_id"),
            "calibration_backend": cal.get("backend"), "calibration_qubits": [int(q) for q in cal["qubits"]],
            "backend": man["backend"], "setting": man["setting"]["name"], "measure_twirl": bool(man["setting"]["measure_twirl"]),
            "mitigation": used, "mthree": R.mthree_status()[0], "overhead": float(ov), "measured_qubits": mq,
            "n_distinct_raw": len(counts), "n_distinct_mitigated": int((w > 0).sum())}
    return idx, w / w.sum() * S, info


def analyse_pub(comp: Component, counts: Dict[int, int], pub: dict, top_k: Optional[int], seed: int,
                job: str = "", job_id: Optional[str] = None,
                mitigated: Optional[Tuple[np.ndarray, np.ndarray, dict]] = None) -> dict:
    """The record of one component of a job (see the module docstring).  Raises if the counts, the export and the manifest
    do not describe the same problem.  `mitigated`: (indices, weights, info) of mitigated_pub, for the readout-mitigated
    statistics next to the raw ones."""
    ensure_phase1()
    n, linear = comp.n, comp.linear
    quad, B = comp.quad, coupling_matrix(comp.n, comp.quad)
    N = 1 << n
    E = component_energies(comp.doc)
    E0 = float(E.min())
    ground = E <= E0 + TOL
    g = int(ground.sum())
    if abs(E0 - float(comp.doc["exact_energy"])) > TOL:
        raise ValueError(f"{comp.name}: enumerated ground energy {E0} != exact_energy {comp.doc['exact_energy']}")
    i_exact = int(sum(int(b) << k for k, b in enumerate(comp.doc["exact_selection"])))
    if not ground[i_exact]:
        raise ValueError(f"{comp.name}: exact_selection is not a ground state")
    E_unif = float(E.mean())
    idx, cnt = counts_arrays(counts)
    S = int(cnt.sum())
    if S != int(pub["shots"]):
        raise ValueError(f"{comp.name}: {S} counted shots, the manifest says {pub['shots']}")
    if idx.size == 0 or idx.min() < 0 or idx.max() >= N:
        raise ValueError(f"{comp.name}: sample index outside 0..2^{n} - 1")
    rec: dict = {"job": job, "job_id": job_id, "sim_method": pub.get("sim_method"), "pub": int(pub["k"]),
                 "component": comp.name, "group": comp.group, "hard": bool(comp.hard), "n": n,
                 "n_couplings": pub.get("n_couplings"), "n_quadruplet_couplings": int(comp.row["n_quadruplet_couplings"]),
                 "n_conflicts": int(comp.row["n_conflicts"]), "n_true": int(comp.true_triplet.sum()),
                 "n_fake": int((~comp.true_triplet).sum()), "n_2q": pub.get("n_2q"), "fidelity_est": pub.get("fidelity_est"),
                 "shots": S, "E0": E0, "ground_degeneracy": g, "p_unif_ground": g / N, "E_unif": E_unif}
    # ideal circuit
    p = None
    if n <= IDEAL_MAX_N:
        ising = component_ising(comp.doc, check=False)
        scale = float(ising.scale())
        if abs(scale - float(pub["scale"])) > 1e-12 * max(1.0, abs(scale)):
            raise ValueError(f"{comp.name}: Ising scale {scale} != {pub['scale']} of the manifest")
        ism = sparse_ising(ising, top_k)
        diag = (E - ising.offset) / scale if not top_k else None
        p = ideal_distribution(ism, scale, float(pub["gammas"][0]), float(pub["betas"][0]), diag)
        e_ideal = float(p @ E)
        e_closed = float(pub["ideal_mean_energy"])
        if abs(e_ideal - e_closed) > 1e-6 * max(1.0, abs(e_closed)):
            raise ValueError(f"{comp.name}: ideal <E> {e_ideal} of the statevector != {e_closed} of the manifest: the "
                             "analysed circuit is not the one that was run")
        rec["ideal"] = {"p_ground": float(p[ground].sum()), "mean_energy": e_ideal}
    else:
        rec["ideal"] = None
    # hardware
    hw = sample_stats(E, E0, idx, cnt, linear, B, n)
    if p is not None:
        den = E_unif - rec["ideal"]["mean_energy"]
        hw["F_E"] = float(f_energy(E, idx, cnt, E_unif, rec["ideal"]["mean_energy"])) if abs(den) > 1e-12 else None
        hw["F_XEB"] = float(f_xeb(p, idx, cnt))
    else:
        hw["F_E"] = hw["F_XEB"] = None
    rec["hardware"] = hw
    if mitigated is not None:
        midx, mw, minfo = mitigated
        keep = mw > 0                                     # states that the correction removed carry no weight
        hm = sample_stats(E, E0, midx[keep], mw[keep], linear, B, n)
        if p is not None:
            hm["F_E"] = float(f_energy(E, midx, mw, E_unif, rec["ideal"]["mean_energy"])) if abs(den) > 1e-12 else None
            hm["F_XEB"] = float(f_xeb(p, midx, mw))
        else:
            hm["F_E"] = hm["F_XEB"] = None
        rec["hardware_mitigated"] = hm
        rec["readout"] = minfo
        rec["p_hw_over_p_unif_mit"] = hm["p_ground"] / rec["p_unif_ground"]
    rec["p_hw_over_p_unif"] = hw["p_ground"] / rec["p_unif_ground"]
    rec["p_ideal_over_p_unif"] = None if p is None else rec["ideal"]["p_ground"] / rec["p_unif_ground"]
    # controls at the same N
    ctrl: Dict[str, Optional[dict]] = {}
    models: Dict[str, Optional[np.ndarray]] = {"uniform": None}
    if p is not None:
        models["ideal"] = p
        F = hw["F_E"]
        models["white_noise"] = None if F is None else min(max(F, 0.0), 1.0) * p + (1.0 - min(max(F, 0.0), 1.0)) / N
    for name in ("uniform", "ideal", "white_noise"):
        if name != "uniform" and models.get(name) is None:
            ctrl[name] = None
            continue
        ci, cc = draw_samples(models[name], n, S, rng_for(seed, comp.name, CONTROL_STREAM[name]))
        ctrl[name] = sample_stats(E, E0, ci, cc, linear, B, n)
    rec["controls"] = ctrl
    # tracking content
    xg = greedy_start(n, quad)
    ig = int(index_of(xg[None, :].astype(float))[0])
    fg = int(descent_indices(linear, B, np.array([ig]), n)[0])
    csv_gd = comp.row.get("greedy_descent_energy")
    rec["greedy_descent_matches_csv"] = None if csv_gd is None else bool(abs(float(E[fg]) - float(csv_gd)) <= TOL)
    if rec["greedy_descent_matches_csv"] is False:
        log.warning("[hw-comp-analysis] %s: greedy-S + descent rebuilt from the export reaches %.9g, components.csv has "
                    "%.9g (ties of the greedy order are broken by the hole counts there)", comp.name, E[fg], csv_gd)
    rec["tracking"] = {"hw_best": track_content(hw["best_index"], comp, E, E0, i_exact),
                       "exact": track_content(i_exact, comp, E, E0, i_exact),
                       "greedy_descent": track_content(fg, comp, E, E0, i_exact),
                       "uniform_descent": track_content(ctrl["uniform"]["desc_best_index"], comp, E, E0, i_exact)}
    rec["tracking"]["greedy_descent"]["start_energy"] = float(E[ig])
    log.info("[hw-comp-analysis] %s (n=%d, %s): P(ground) %.4g vs uniform %.3g, best-of-%d %s E0, descent %s%s", comp.name, n,
             comp.group or "-", hw["p_ground"], rec["p_unif_ground"], S, "=" if hw["best_reaches_E0"] else ">",
             "reaches" if hw["desc_any_E0"] else "misses",
             "" if hw["F_E"] is None else f", F_E {hw['F_E']:.3f} F_XEB {hw['F_XEB']:.3f}")
    return rec


# ---------------------------------------------------------------------------
# aggregates
# ---------------------------------------------------------------------------
def boot_stat(values: Sequence, stat, rng: np.random.Generator, boot_n: int = COMP_BOOT_N) -> dict:
    """stat(values) with the 16 / 84 % bootstrap interval over the components (None entries are skipped; the interval is
    None for fewer than two components).  `stat` is np.median or np.mean."""
    v = np.array([float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return {"value": None, "lo68": None, "hi68": None, "n": 0}
    val = float(stat(v))
    if v.size < 2:
        return {"value": val, "lo68": None, "hi68": None, "n": int(v.size)}
    s = stat(v[rng.integers(0, v.size, size=(boot_n, v.size))], axis=1)
    lo, hi = np.percentile(s, [16, 84])
    return {"value": val, "lo68": float(lo), "hi68": float(hi), "n": int(v.size)}


def _src(rec: dict, source: str) -> Optional[dict]:
    return rec["hardware"] if source == "hardware" else rec["controls"][source]


def group_summary(recs: Sequence[dict], name: str, boot_n: int = COMP_BOOT_N) -> dict:
    """Aggregates of the records of one group (see the module docstring)."""
    rng = np.random.default_rng(np.random.SeedSequence([COMP_BOOT_SEED, zlib.crc32(name.encode())]))
    out = {"n_components": len(recs), "sizes": sorted(r["n"] for r in recs),
           "p_hw_over_p_unif_median": boot_stat([r["p_hw_over_p_unif"] for r in recs], np.median, rng, boot_n),
           "F_E_median": boot_stat([r["hardware"]["F_E"] for r in recs], np.median, rng, boot_n),
           "F_XEB_median": boot_stat([r["hardware"]["F_XEB"] for r in recs], np.median, rng, boot_n)}
    if any("hardware_mitigated" in r for r in recs):      # readout-mitigated statistics of the records that have them
        mit = [r for r in recs if "hardware_mitigated" in r]
        out["readout_mitigated"] = {
            "n_components": len(mit),
            "p_hw_over_p_unif_median": boot_stat([r["p_hw_over_p_unif_mit"] for r in mit], np.median, rng, boot_n),
            "p_hw_over_p_unif_median_raw_same_components": boot_stat([r["p_hw_over_p_unif"] for r in mit], np.median, rng,
                                                                     boot_n),
            "F_E_median": boot_stat([r["hardware_mitigated"]["F_E"] for r in mit], np.median, rng, boot_n),
            "F_XEB_median": boot_stat([r["hardware_mitigated"]["F_XEB"] for r in mit], np.median, rng, boot_n)}
    for key, field in (("best_of_N_reaches_E0", "best_reaches_E0"), ("best_of_N_plus_descent_reaches_E0", "desc_any_E0")):
        out[key] = {s: boot_stat([None if _src(r, s) is None else _src(r, s)[field] for r in recs], np.mean, rng, boot_n)
                    for s in SOURCES}
    out["descent_start_success_median"] = {
        s: boot_stat([None if _src(r, s) is None else _src(r, s)["desc_frac_E0"] for r in recs], np.median, rng, boot_n)
        for s in SOURCES}
    return out


def summarize(records: Sequence[dict], boot_n: int = COMP_BOOT_N) -> dict:
    groups = {}
    for g in ("hard", "benchmark", "all"):
        rs = [r for r in records if g == "all" or r["group"] == g]
        if rs:
            groups[g] = group_summary(rs, g, boot_n)
    return groups


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
TRACK_KEYS = ("energy", "n_true", "n_fake", "is_exact_selection", "is_ground_state")
STAT_KEYS = ("p_ground", "best_energy", "best_reaches_E0", "desc_frac_E0", "desc_any_E0")
# column prefix per sample source; the sampled ideal control is "ideal_sample", so that it never overwrites the
# exact statevector columns ideal_p_ground / ideal_mean_energy
COLUMN_PREFIX = {"hardware": "hw", "uniform": "uniform", "ideal": "ideal_sample", "white_noise": "white_noise"}


def flat_record(r: dict) -> dict:
    """One flat row of components_hw.csv."""
    row = {k: r[k] for k in ("job", "job_id", "sim_method", "pub", "component", "group", "hard", "n", "n_couplings",
                             "n_quadruplet_couplings", "n_conflicts", "n_true", "n_fake", "n_2q", "fidelity_est", "shots",
                             "E0", "ground_degeneracy", "p_unif_ground", "E_unif")}
    ideal = r["ideal"] or {}
    row.update(ideal_p_ground=ideal.get("p_ground"), ideal_mean_energy=ideal.get("mean_energy"),
               p_hw_over_p_unif=r["p_hw_over_p_unif"], p_ideal_over_p_unif=r["p_ideal_over_p_unif"])
    for s in SOURCES:
        st = _src(r, s) or {}
        pre = COLUMN_PREFIX[s]
        for k in STAT_KEYS:
            row[f"{pre}_{k}"] = st.get(k)
        if s == "hardware":
            row.update(hw_mean_energy=st["mean_energy"], hw_distinct=st["distinct"], F_E=st["F_E"], F_XEB=st["F_XEB"])
    for t, d in r["tracking"].items():
        for k in TRACK_KEYS:
            row[f"trk_{t}_{k}"] = d[k]
    row["greedy_descent_matches_csv"] = r["greedy_descent_matches_csv"]
    hm = r.get("hardware_mitigated")
    if hm is not None:                                   # mitigated values next to the raw ones above
        for k in STAT_KEYS:
            row[f"mit_{k}"] = hm[k]
        row.update(mit_mean_energy=hm["mean_energy"], F_E_mit=hm["F_E"], F_XEB_mit=hm["F_XEB"],
                   p_hw_over_p_unif_mit=r["p_hw_over_p_unif_mit"], readout_calibration=r["readout"]["calibration_file"],
                   readout_mitigation=r["readout"]["mitigation"], readout_overhead=r["readout"]["overhead"])
    return row


# ---------------------------------------------------------------------------
# job folders
# ---------------------------------------------------------------------------
def _run_dir_of(job_dirs: Sequence[Path]) -> Path:
    runs = set()
    for jd in job_dirs:
        if jd.parent.name != "hw":
            raise ValueError(f"{jd} is not a job folder <triplet run>/hw/<job>")
        runs.add(jd.parents[1])
    if len(runs) != 1:
        raise ValueError(f"the job folders belong to {len(runs)} different run folders: {sorted(map(str, runs))}")
    return runs.pop()


def analyze_jobs(job_dirs: Sequence[Path], out_dir: Path, seed: Optional[int] = None, boot_n: int = COMP_BOOT_N,
                 readout: Optional[Path] = None, readout_tags: Optional[str] = None, prefer: str = "auto") -> dict:
    """Analyse finished component jobs of one triplet run folder; writes components_hw.csv and components_hw_summary.json
    into `out_dir` and returns {'records', 'summary', 'csv', 'json'}.  `seed` of the controls: by default the selection
    seed recorded in the manifests, as in the analysis that runs after a job (0 if they record none or disagree)."""
    import pandas as pd
    job_dirs = [Path(d) for d in job_dirs]
    if not job_dirs:
        raise ValueError("no job folder to analyse")
    run_dir = _run_dir_of(job_dirs)
    mans = []
    for jd in job_dirs:
        man = load_json(jd / "manifest.json")
        if man.get("kind") != KIND:
            raise ValueError(f"{jd} is not a components job (kind {man.get('kind')!r}); hw-components writes them")
        if man.get("state") != "done":
            raise ValueError(f"{jd} is in state {man.get('state')!r}, not done: no counts to analyse (hw-retrieve a submitted job)")
        mans.append(man)
    if seed is None:
        seeds = {(m.get("selection") or {}).get("seed") for m in mans}
        seed = int(seeds.pop()) if len(seeds) == 1 and None not in seeds else 0
    comps = {c.name: c for c in load_components(run_dir, {p["component"] for m in mans for p in m["pubs"]})}
    if readout_tags and readout is None:
        raise ValueError("--readout-tags needs --readout <readout_cal.json>")
    cal, mit_names = None, []
    if readout is not None:
        from .analysis import load_calibration, select_readout_tags
        cal = dict(load_calibration(readout), _path=str(readout), _file=Path(readout).name)
        mit_names = select_readout_tags(sorted(comps), readout_tags)
    records: List[dict] = []
    jobs: List[dict] = []
    for jd, man in zip(job_dirs, mans):
        listed = load_json(jd / "components.json")["pubs"]
        if [(p["pub"], p["component"]) for p in listed] != [(p["k"], p["component"]) for p in man["pubs"]]:
            raise ValueError(f"{jd}: components.json and the manifest disagree on which component is which PUB")
        for pub in man["pubs"]:
            comp = comps[pub["component"]]
            if comp.sha1 != pub["component_sha1"]:
                raise ValueError(f"{comp.path} changed since job {jd.name} was planned (sha1 {comp.sha1} != "
                                 f"{pub['component_sha1']})")
            comp = Component(comp.name, comp.path, comp.sha1, comp.doc, comp.row, pub["group"])
            cf = load_json(jd / f"pub_{pub['k']}_counts.json")
            counts = {int(k): int(v) for k, v in cf["counts"].items()}
            mit = mitigated_pub(counts, pub, man, cal, prefer) if comp.name in mit_names else None
            records.append(analyse_pub(comp, counts, pub, man.get("top_k"), seed, jd.name, man.get("job_id"), mit))
        jobs.append({"job_dir": jd.name, "job_id": man.get("job_id"), "mode": man["mode"], "backend": man["backend"],
                     "setting": man["setting"]["name"], "primitive": man["primitive"], "shots": man["shots"],
                     "n_pubs": len(man["pubs"]), "estimate_s": man.get("estimate_s"), "billed": man.get("billed"),
                     "sim_methods": man.get("sim_methods"), "top_k": man.get("top_k")})
    groups = summarize(records, boot_n)
    summary = {"schema": SCHEMA, "created": utc_now(), "run_dir": str(run_dir), "jobs": jobs,
               "config": {"seed": seed, "tolerance": TOL, "ideal_max_n": IDEAL_MAX_N, "bootstrap": "components",
                          "bootstrap_n": boot_n, "bootstrap_seed": COMP_BOOT_SEED, "interval": "16/84 %",
                          "controls": "N = shots of the PUB; uniform, ideal and white noise (F_E of the hardware)",
                          "readout": None if cal is None else {"calibration": str(readout), "tags_regex": readout_tags,
                                                               "components": mit_names, "prefer": prefer}},
               "n_components": len(records), "groups": groups, "components": records}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = save_csv(pd.DataFrame([flat_record(r) for r in records]), out_dir / "components_hw.csv")
    js = save_json(out_dir / "components_hw_summary.json", summary)
    for g, d in groups.items():
        log.info("[hw-comp-analysis] group %-9s %2d components: median P_hw/P_unif(ground) %s | best-of-N reaches E0: %s | "
                 "with descent: %s", g, d["n_components"], _fmt(d["p_hw_over_p_unif_median"]),
                 " ".join(f"{s} {_fmt(d['best_of_N_reaches_E0'][s])}" for s in SOURCES),
                 " ".join(f"{s} {_fmt(d['best_of_N_plus_descent_reaches_E0'][s])}" for s in SOURCES))
    return {"records": records, "summary": summary, "csv": csv, "json": js}


def _fmt(b: dict) -> str:
    if b["value"] is None:
        return "-"
    iv = "" if b["lo68"] is None else f" [{b['lo68']:.3g}, {b['hi68']:.3g}]"
    return f"{b['value']:.3g}{iv}"

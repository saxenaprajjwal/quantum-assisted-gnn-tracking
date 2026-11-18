"""Noise analysis of hardware sample sets of a Phase-1 run folder (0 QPU s, no GNN needed).

For every hardware job (samples/ibm-<backend>_<ansatz>_p1_<topK|full>_<jobid8>_main.json, with its _control.json) the
ideal distribution p of the circuit that was run is computed with the Phase-1 StatevectorQAOA (the job's sparsified
Ising, its angles) and compared with the measured counts:
  F_E     mean-energy fidelity  (E_unif - <E>_hw) / (E_unif - <E>_ideal)       (white-noise model: p_hw = F p + (1-F)/N)
  F_XEB   (<p(x)>_hw - 1/N) / (sum p^2 - 1/N)                                   (global circuit fidelity)
  tail F  (p_hw(q) - q) / (p_ideal(q) - q) for the lowest-energy fraction q of the 2^n states
  s_k     correlation-order fidelity: with z = 1 - 2 bit and the ideal correlators c_S = <z_S>_ideal (Walsh coefficients
          of p), s_k = sum_{|S|=k} <z_S>_hw c_S / sum_{|S|=k} c_S^2.  White noise gives s_k = F for every k; independent
          bit flips with probability eps give (1 - 2 eps)^k.  Linear in the empirical distribution, hence computed per
          shot as g_k(x) = (W (mask_k c))(x) (W: Walsh-Hadamard transform) with a standard error from the shot spread.
  CVaR    bounds of the ideal mean energy (Barron et al., Nat. Comput. Sci. 2024): if p_hw >= F p_ideal, the ideal mean lies
          in [CVaR_F^lower, CVaR_F^upper], the mean energy of the lowest / highest F fraction of the shots.
The estimators are pure numpy; Phase 1 is imported only by ideal_probs / analyze_run.

Readout mitigation (analyze_run(readout=<readout_cal.json>, readout_tags=<regex>)): for the selected main sample sets the job
folder (hw/<stamp>_<tag>, found by the job id in the sample meta) is read, its PUBs are corrected in their gauge frames with the
calibration (hsf2.hw.readout: mthree at full distance, else the numpy solve; same measured qubits in classical-bit order and
same measurement-twirling switch, hsf2.hw.run.mitigate_job), the quasi-distribution is projected to the nearest probability
distribution and used as weights (S x probability) in the same estimators.  Every selected record keeps the raw values and gets
a "readout" dict with the mitigated ones next to them (backend, calibration file and session named).  For s_k the point
estimate is the weighted one and the standard error is the RAW one; the 68 % intervals of F_E / F_XEB of the mitigated set
resample the mitigated distribution with S draws, so they are a lower bound (the variance inflation of the correction, the
recorded overhead, is not propagated).  The controls stay raw.

Phase-1 Run is built on a temporary copy of run_config.json, problem.json and angles.json (analyze_run), so nothing is
ever written into the folder that holds the samples; the samples are only read.
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import load_json, save_csv, save_json

log = logging.getLogger(__name__)

ALPHAS = (0.5, 0.6, 0.7, 0.8, 0.9)
TAIL_QS = (1e-3, 1e-2, 0.1)
BOOT_N = 400
BOOT_SEED = 0
TAG_RE = re.compile(r"^ibm-(?P<backend>.+?)_(?P<ansatz>[a-z0-9]+)_p1_(?P<circ>top\d+|full)_(?P<job>[0-9a-z]+)_main$")


# ---------------------------------------------------------------------------
# pure estimators
# ---------------------------------------------------------------------------
def counts_arrays(counts: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """{'index': shots} (Phase-1 sample file) -> (indices int64, counts int64)."""
    idx = np.array([int(k) for k in counts], np.int64)
    c = np.array(list(counts.values()), np.int64)
    return idx, c


def ideal_probs(run, ising, top_k: Optional[int], gammas, betas, scale: float) -> np.ndarray:
    """Noiseless QAOA distribution (2^n, float64, sums to 1) of the circuit that was run: Phase-1 StatevectorQAOA with
    the Ising sparsified to its `top_k` largest couplings (None: all) and the angles of the job; `scale` is the
    problem's energy scale (problem.json)."""
    from .._phase1 import ensure_phase1
    ensure_phase1()
    from acts_qaoa import pipeline as P1
    from acts_qaoa.qaoa_math import StatevectorQAOA
    ism = ising if top_k is None else ising.sparsified(top_k)
    hd = (P1._sparse_energies(run, ism) - ism.offset) / scale
    p = np.asarray(StatevectorQAOA(hd, mixer="x", dtype=np.complex64).probs(gammas, betas), dtype=np.float64)
    return p / p.sum()


def f_energy(E: np.ndarray, idx: np.ndarray, counts: np.ndarray, E_unif: float, E_ideal: float) -> float:
    """Mean-energy fidelity (E_unif - <E>_hw) / (E_unif - E_ideal)."""
    return (E_unif - float(counts @ E[idx]) / counts.sum()) / (E_unif - E_ideal)


def f_xeb(p: np.ndarray, idx: np.ndarray, counts: np.ndarray) -> float:
    """Normalised linear cross entropy (<p(x)>_hw - 1/N) / (sum p^2 - 1/N)."""
    N = p.size
    return (float(counts @ p[idx]) / counts.sum() - 1.0 / N) / (float(p @ p) - 1.0 / N)


def tail(E_rank: np.ndarray, p: np.ndarray, idx: np.ndarray, counts: np.ndarray, q: float) -> dict:
    """Lowest-energy fraction q of the states (E_rank[x] = rank of x by energy, 0 = best): ideal and measured mass,
    tail fidelity (p_hw - q)/(p_ideal - q) and its binomial error."""
    top = E_rank < int(round(q * p.size))
    S = float(counts.sum())
    p_id = float(p[top].sum())
    n_hw = counts[top[idx]].sum()                       # weights of a readout-mitigated distribution are not integers
    ph = float(n_hw) / S
    return {"q": q, "p_ideal": p_id, "p_hw": ph, "shots": int(n_hw) if np.issubdtype(counts.dtype, np.integer) else float(n_hw),
            "tail_F": (ph - q) / (p_id - q), "tail_F_err": float(np.sqrt(ph * (1 - ph) / S) / (p_id - q))}


def walsh(v: np.ndarray) -> np.ndarray:
    """In-place (unnormalised) Walsh-Hadamard transform of a length-2^n float64 vector, v[S] -> sum_x (-1)^{S.x} v[x];
    n butterfly passes, no matrix.  Applying it twice multiplies by 2^n.  Returns v."""
    if v.dtype != np.float64 or v.ndim != 1 or not v.flags.c_contiguous:
        raise ValueError("walsh needs a contiguous 1-D float64 array")
    N = v.size
    if N < 1 or N & (N - 1):
        raise ValueError(f"walsh needs a power-of-two length, got {N}")
    h = 1
    while h < N:
        w = v.reshape(-1, 2, h)
        a, b = w[:, 0, :], w[:, 1, :]
        t = a - b
        a += b
        b[...] = t
        h *= 2
    return v


def hamming_weights(n: int) -> np.ndarray:
    """popcount of 0 .. 2^n - 1 as int8 (built by doubling, no int64 temporaries)."""
    w = np.zeros(1, np.int8)
    for _ in range(n):
        w = np.concatenate([w, w + np.int8(1)])
    return w


def correlation_fidelity(p: np.ndarray, samples: Sequence[Tuple[np.ndarray, np.ndarray]], kmax: int = 5
                         ) -> List[List[Tuple[float, float]]]:
    """s_k, k = 1..kmax, for each (idx, counts) in `samples`: [(s_k, standard error), ...] per sample set.
    Ideal correlators c = walsh(p); g_k = walsh(mask_k c); s_k = mean_shots g_k(x) / sum_{|S|=k} c_S^2.
    At most four 2^n float64 vectors are alive at a time (p is not modified)."""
    N = p.size
    n = N.bit_length() - 1
    if 1 << n != N:
        raise ValueError(f"p must have a power-of-two length, got {N}")
    if not 1 <= kmax <= n:
        raise ValueError(f"kmax must be in 1..{n}, got {kmax}")
    c = walsh(np.array(p, np.float64))
    w = hamming_weights(n)
    out: List[List[Tuple[float, float]]] = [[] for _ in samples]
    for k in range(1, kmax + 1):
        g = np.where(w == k, c, 0.0)
        denom = float(g @ g)
        walsh(g)
        for j, (idx, cnt) in enumerate(samples):
            S = float(cnt.sum())
            gv = g[idx]
            m = float(cnt @ gv) / S
            var = max(float(cnt @ (gv * gv)) / S - m * m, 0.0)
            out[j].append((m / denom, float(np.sqrt(var / S)) / denom) if denom > 0 else (float("nan"), float("nan")))
        del g
    return out


def cvar(E_vals: np.ndarray, counts: np.ndarray, alpha: float, upper: bool = False) -> float:
    """Mean energy of the lowest (upper: highest) alpha fraction of the shots, fractional weight at the boundary."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    order = np.argsort(-E_vals if upper else E_vals, kind="stable")
    e = E_vals[order].astype(np.float64)
    wgt = counts[order] / float(counts.sum())
    before = np.cumsum(wgt) - wgt
    take = np.clip(alpha - before, 0.0, wgt)
    return float(take @ e) / alpha


def cvar_bracket(E_vals: np.ndarray, counts: np.ndarray, alpha: float, E_ideal: float) -> dict:
    """[lower CVaR_alpha, upper CVaR_alpha] and whether the ideal mean lies inside (None when alpha is not in (0, 1])."""
    if not 0.0 < alpha <= 1.0:
        return {"alpha": alpha, "lower": None, "upper": None, "inside": None}
    lo, hi = cvar(E_vals, counts, alpha), cvar(E_vals, counts, alpha, upper=True)
    return {"alpha": alpha, "lower": lo, "upper": hi, "inside": bool(lo <= E_ideal <= hi)}


def bootstrap(stat, idx: np.ndarray, counts: np.ndarray, rng: np.random.Generator, n: int = BOOT_N) -> List[float]:
    """16/84 % interval of stat(idx, counts) under multinomial resampling of the shots."""
    S = int(round(float(counts.sum())))                 # float weights (readout-mitigated distribution): S shots drawn from them
    pr = counts / counts.sum()
    vals = [stat(idx, rng.multinomial(S, pr)) for _ in range(n)]
    return [float(np.percentile(vals, 16)), float(np.percentile(vals, 84))]


# ---------------------------------------------------------------------------
# run folder
# ---------------------------------------------------------------------------
def parse_tag(tag: str) -> Tuple[Optional[int], str]:
    """'ibm-ibm_fez_standard_p1_top24_<job>_main' -> (24, job id prefix); 'full' -> (None, ...)."""
    m = TAG_RE.match(tag)
    if m is None:
        raise ValueError(f"not a hardware main tag (ibm-<backend>_<ansatz>_p1_<topK|full>_<jobid8>_main): {tag!r}")
    circ = m["circ"]
    return (None if circ == "full" else int(circ[3:])), m["job"]


def main_tags(samples_dir: Path) -> List[str]:
    return sorted(f.name[:-len(".json")] for f in Path(samples_dir).glob("ibm-*_main.json"))


def load_calibration(path: Path) -> dict:
    """A readout_cal.json (hsf2.hw.readout.calibration_record), checked for the keys that the correction needs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"readout calibration {path} not found")
    cal = load_json(path)
    missing = [k for k in ("measure_twirl", "qubits", "matrices") if k not in cal]
    if missing:
        raise ValueError(f"{path} is not a readout calibration (readout_cal.json): no {missing}")
    return cal


def select_readout_tags(tags: Sequence[str], regex: Optional[str]) -> List[str]:
    """The tags (re.search) whose samples are readout-mitigated; all of them without a regex.  A regex that selects none of
    the analysed tags is an error."""
    if not regex:
        return list(tags)
    try:
        pat = re.compile(regex)
    except re.error as exc:
        raise ValueError(f"--readout-tags {regex!r} is not a regular expression: {exc}") from None
    sel = [t for t in tags if pat.search(t)]
    if not sel:
        raise ValueError(f"--readout-tags {regex!r} matches none of the analysed tags {list(tags)}")
    return sel


def mitigated_weights(run_dir: Path, meta: dict, raw_main: Dict[str, int], cal: dict, prefer: str = "auto",
                      method: str = "tensor") -> Tuple[np.ndarray, np.ndarray, dict]:
    """(indices, weights, info) of the readout-mitigated main sample set of the job named in the sample `meta`.
    method "tensor": the exact inverse of the uncorrelated read-out model on all 2^n states (readout.tensor_inverse;
    weights = S x quasi-probabilities, some negative); every estimator of this module is linear in the weights, so it
    applies unchanged (the resampled intervals and the CVaR bracket, which need a distribution, are skipped).
    method "m3": the reduced-subspace solve (mthree or numpy) projected to the nearest probability distribution; on a
    broad sample (24 qubits, 4 000 shots) its matrix is the identity and nothing changes.  The job folder is found by job
    id under <run_dir>/hw; its raw main counts must be those of the sample file."""
    if method not in ("tensor", "m3"):
        raise ValueError(f"readout method must be tensor or m3, got {method!r}")
    from . import readout as R
    from .run import find_job_dir, mitigate_job, raw_counts
    jid = meta.get("job_id")
    if not jid:
        raise ValueError("the sample set records no job id: cannot find the job folder for the readout correction")
    job_dir = find_job_dir(jid, run_dir)
    man = load_json(job_dir / "manifest.json")
    if man.get("state") != "done":
        raise ValueError(f"{job_dir} is in state {man.get('state')!r}, not done: no counts to correct")
    if "layout" not in man:
        raise ValueError(f"{job_dir} is not a hw-run / hw-ladder job (no layout of one circuit)")
    raw = raw_counts(job_dir, man)["main"]
    if {int(k): int(v) for k, v in raw_main.items()} != raw:
        raise ValueError(f"the raw main counts of {job_dir.name} differ from the sample file: not the same data")
    cal_used = R.calibration_for(cal, man["layout"]["measured_qubits"], man["setting"]["measure_twirl"], man["backend"])
    S = float(sum(raw.values()))
    if method == "tensor":
        q = R.tensor_inverse(raw, R.matrices_of(cal_used)[1])
        info = {"job_dir": job_dir.name, "job_id": jid, "backend": man["backend"], "setting": man["setting"]["name"],
                "measure_twirl": bool(man["setting"]["measure_twirl"]), "mitigation": "tensor-inverse",
                "mthree": R.mthree_status()[0], "overhead": [float(np.abs(q).sum())],
                "overhead_mean": float(np.abs(q).sum()), "n_distinct_raw": len(raw),
                "n_distinct_mitigated": int((q != 0).sum()), "n_pubs": 1}
        return np.arange(q.size, dtype=np.int64), q * S, info
    merged, overhead, used = mitigate_job(job_dir, man, cal_used, prefer, kinds=("main",))
    pr = merged["main"]
    idx = np.array(sorted(pr), np.int64)
    p = np.array([pr[int(i)] for i in idx], np.float64)
    if not abs(p.sum() - 1.0) < 1e-6:
        raise ValueError(f"mitigated distribution of {job_dir.name} sums to {p.sum()}, not 1")
    info = {"job_dir": job_dir.name, "job_id": jid, "backend": man["backend"], "setting": man["setting"]["name"],
            "measure_twirl": bool(man["setting"]["measure_twirl"]), "mitigation": used, "mthree": R.mthree_status()[0],
            "overhead": [float(o) for o in overhead], "overhead_mean": float(np.mean(overhead)),
            "n_distinct_raw": len(raw), "n_distinct_mitigated": int((p > 0).sum()), "n_pubs": len(overhead)}
    return idx, p / p.sum() * S, info


def mitigated_record(E: np.ndarray, p: np.ndarray, rank: np.ndarray, E_unif: float, e_ideal: float, idx: np.ndarray,
                     w: np.ndarray, sk: Sequence[Tuple[float, float]], sk_raw: Sequence[Tuple[float, float]], info: dict,
                     cal_path: Path, cal: dict, rng: np.random.Generator) -> dict:
    """The "readout" dict of one record: the estimators of the module on the mitigated weights `w` over the states `idx`
    (see the module docstring for the errors)."""
    def fe(i, x):
        return f_energy(E, i, x, E_unif, e_ideal)

    def fx(i, x):
        return f_xeb(p, i, x)

    F_E, F_X = fe(idx, w), fx(idx, w)
    Ev = E[idx]
    dist = bool((w >= 0).all())                 # the tensor inverse gives quasi-probabilities: no resampling, no CVaR
    return dict(info, calibration=str(cal_path), calibration_file=Path(cal_path).name, calibration_session=cal.get("session"),
                calibration_job_id=cal.get("job_id"), calibration_backend=cal.get("backend"),
                calibration_shots=cal.get("shots"),
                E_hw=float(w @ Ev) / float(w.sum()), F_E=F_E, F_E_68_resample=bootstrap(fe, idx, w, rng) if dist else None,
                F_XEB=F_X, F_XEB_68_resample=bootstrap(fx, idx, w, rng) if dist else None,
                s_k={str(k + 1): {"s": s, "err": e_raw} for k, ((s, _), (_, e_raw)) in enumerate(zip(sk, sk_raw))},
                tail=[tail(rank, p, idx, w, q) for q in TAIL_QS],
                cvar={"at_F_E": cvar_bracket(Ev, w, F_E, e_ideal), "at_F_XEB": cvar_bracket(Ev, w, F_X, e_ideal)} if dist
                else None,
                notes="s_k error is the raw one; F_E / F_XEB intervals resample the mitigated distribution (lower bound, the "
                      "overhead is not propagated; none for the quasi-probabilities of the tensor inverse); controls are raw")


def analyze_run(run_dir: Path, out_dir: Path, tags: Optional[Sequence[str]] = None, kmax: int = 5,
                readout: Optional[Path] = None, readout_tags: Optional[str] = None, prefer: str = "auto",
                readout_method: str = "tensor") -> dict:
    """Analyse the hardware main samples (and controls) of `run_dir`/samples; writes hw_analysis.json and
    hw_analysis.csv into `out_dir` and returns the result dict.  `readout`: a readout_cal.json applied to the main samples
    whose tag matches the regex `readout_tags` (default: all analysed ones), see the module docstring; `prefer`: auto /
    mthree / numpy as in hsf2.hw.readout.correct_counts."""
    from .._phase1 import ensure_phase1
    ensure_phase1()
    from acts_qaoa import pipeline as P1
    from acts_qaoa.config import RunConfig

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    sdir = run_dir / "samples"
    for need in ("run_config.json", "problem.json"):
        if not (run_dir / need).is_file():
            raise FileNotFoundError(f"{run_dir / need} not found")
    avail = main_tags(sdir)
    tags = list(tags) if tags else avail
    if not tags:
        raise ValueError(f"no hardware sample set (samples/ibm-*_main.json) in {run_dir}")
    missing = [t for t in tags if t not in avail]
    if missing:
        raise FileNotFoundError(f"no such hardware main sample in {sdir}: {missing}")
    if readout_tags and readout is None:
        raise ValueError("--readout-tags needs --readout <readout_cal.json>")
    cal = None if readout is None else load_calibration(readout)
    mit_tags = [] if cal is None else select_readout_tags(tags, readout_tags)

    with tempfile.TemporaryDirectory(prefix="hsf2_hwa_") as tmp:
        tmp = Path(tmp)
        for name in ("run_config.json", "problem.json", "angles.json"):
            if (run_dir / name).is_file():
                shutil.copy2(run_dir / name, tmp / name)
        cfg = RunConfig.from_dict(load_json(tmp / "run_config.json"))
        run = P1.Run(tmp, cfg, None)
        fit, qubo, ising, prob = run.load_problem()
        land = run.landscape(qubo)
        scale = float(prob["scale"])
        E = land.E.astype(np.float64)
        N = E.size
        n = N.bit_length() - 1
        order = np.argsort(E, kind="stable")
        rank = np.empty(N, np.int32)
        rank[order] = np.arange(N, dtype=np.int32)
        E_unif = float(E.mean())
        E_min = float(E[order[0]])
        rows = []
        for tag in tags:
            top_k, _ = parse_tag(tag)
            main = load_json(sdir / f"{tag}.json")
            ctrl = load_json(sdir / f"{tag[:-len('_main')]}_control.json")
            meta = main["meta"]
            if meta["problem_fingerprint"] != prob["fingerprint"]:
                raise ValueError(f"{tag}: problem fingerprint {meta['problem_fingerprint']} != {prob['fingerprint']}")
            g, b = meta["gammas"], meta["betas"]
            p = ideal_probs(run, ising, top_k, g, b, scale)
            e_ideal = float(p @ E)
            idx, c = counts_arrays(main["counts"])
            cidx, cc = counts_arrays(ctrl["counts"])
            S = int(c.sum())
            e_hw = float(c @ E[idx]) / S
            e_ctrl = float(cc @ E[cidx]) / cc.sum()
            rng = np.random.default_rng(BOOT_SEED)

            def fe(i, w):
                return f_energy(E, i, w, E_unif, e_ideal)

            def fx(i, w):
                return f_xeb(p, i, w)

            F_E, F_X = fe(idx, c), fx(idx, c)
            samples = [(idx, c), (cidx, cc)]
            mit = None
            if tag in mit_tags:
                midx, mw, minfo = mitigated_weights(run_dir, meta, main["counts"], cal, prefer, readout_method)
                samples.append((midx, mw))
            sk, sk_ctrl, *sk_mit = correlation_fidelity(p, samples, kmax)
            Ev = E[idx]
            cdf_ideal = np.cumsum(p[order])
            row = {"tag": tag, "K": top_k if top_k is not None else n * (n - 1) // 2, "shots": S, "gamma": g, "beta": b,
                   "ideal_mean_check": [e_ideal, meta.get("ideal_mean_energy")],
                   "E_hw": e_hw, "E_ideal": e_ideal, "E_unif": E_unif, "E_ctrl": e_ctrl,
                   "F_E": F_E, "F_E_68": bootstrap(fe, idx, c, rng), "F_E_logged": main["metrics"].get("F_measured"),
                   "F_vs_control": (e_ctrl - e_hw) / (E_unif - e_ideal),
                   "F_XEB": F_X, "F_XEB_68": bootstrap(fx, idx, c, rng), "F_XEB_control": fx(cidx, cc),
                   "s_k": {str(k + 1): {"s": s, "err": e} for k, (s, e) in enumerate(sk)},
                   "s_k_control": {str(k + 1): {"s": s, "err": e} for k, (s, e) in enumerate(sk_ctrl)},
                   "ideal_p_ground": float(p[order[0]]), "ideal_sum_p2_times_N": float(p @ p) * N,
                   "best_rank_hw": int(rank[idx].min()), "best_rank_ctrl": int(rank[cidx].min()),
                   "expected_best_rank_uniform": float(((1 - np.arange(1, N + 1) / N) ** S).sum()),
                   "expected_best_rank_ideal": float(((1 - cdf_ideal) ** S).sum()),
                   "tail": [], "cvar": {"alphas": [cvar_bracket(Ev, c, a, e_ideal) for a in ALPHAS],
                                        "at_F_E": cvar_bracket(Ev, c, F_E, e_ideal),
                                        "at_F_XEB": cvar_bracket(Ev, c, F_X, e_ideal)}}
            for q in TAIL_QS:
                t = tail(rank, p, idx, c, q)
                top = rank < int(round(q * N))
                t["ctrl"] = int(cc[top[cidx]].sum()) / cc.sum()
                row["tail"].append(t)
            if tag in mit_tags:
                row["readout"] = mitigated_record(E, p, rank, E_unif, e_ideal, midx, mw, sk_mit[0], sk, minfo, readout, cal, rng)
                mit = row["readout"]
                del midx, mw
            rows.append(row)
            log.info("[hw-analyze] %s: F_E %.3f  F_XEB %.3f  s_k %s", tag, F_E, F_X,
                     " ".join(f"{s:.3f}" for s, _ in sk))
            if mit is not None:
                log.info("[hw-analyze] %s readout-mitigated (%s, %s): F_E %.3f  F_XEB %.3f  s_k %s", tag, mit["mitigation"],
                         mit["calibration_file"], mit["F_E"], mit["F_XEB"],
                         " ".join(f"{v['s']:.3f}" for v in mit["s_k"].values()))
            del p, cdf_ideal
        rows.sort(key=lambda r: (r["K"], r["tag"]))
    res = {"run_dir": str(run_dir), "N": N, "n_qubits": n, "E_unif": E_unif, "E_min": E_min,
           "config": {"tags": tags, "kmax": kmax, "alphas": list(ALPHAS), "tail_q": list(TAIL_QS),
                      "bootstrap_n": BOOT_N, "bootstrap_seed": BOOT_SEED,
                      "readout": None if cal is None else {"calibration": str(readout), "tags_regex": readout_tags,
                                                           "tags": mit_tags, "prefer": prefer,
                                                           "method": readout_method}},
           "rows": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "hw_analysis.json", res)
    save_csv(_table(rows, kmax), out_dir / "hw_analysis.csv")
    return res


def _table(rows: List[dict], kmax: int):
    import pandas as pd
    recs = []
    for r in rows:
        d = {"tag": r["tag"], "K": r["K"], "shots": r["shots"], "F_E": r["F_E"], "F_XEB": r["F_XEB"],
             "F_vs_control": r["F_vs_control"]}
        for k in range(1, kmax + 1):
            d[f"s{k}"] = r["s_k"][str(k)]["s"]
            d[f"s{k}_err"] = r["s_k"][str(k)]["err"]
        d["cvar_F_E_lo"], d["cvar_F_E_hi"] = r["cvar"]["at_F_E"]["lower"], r["cvar"]["at_F_E"]["upper"]
        d["ideal_inside_F_E"] = r["cvar"]["at_F_E"]["inside"]
        d["cvar_F_XEB_lo"], d["cvar_F_XEB_hi"] = r["cvar"]["at_F_XEB"]["lower"], r["cvar"]["at_F_XEB"]["upper"]
        d["ideal_inside_F_XEB"] = r["cvar"]["at_F_XEB"]["inside"]
        d["E_ideal"] = r["E_ideal"]
        ro = r.get("readout")
        if ro is not None:                      # mitigated values next to the raw ones above
            d.update(readout_calibration=ro["calibration_file"], readout_session=ro.get("calibration_session"),
                     readout_mitigation=ro["mitigation"], readout_overhead=ro["overhead_mean"], E_hw_mit=ro["E_hw"],
                     F_E_mit=ro["F_E"], F_XEB_mit=ro["F_XEB"])
            for k in range(1, kmax + 1):
                d[f"s{k}_mit"] = ro["s_k"][str(k)]["s"]
                d[f"s{k}_mit_err"] = ro["s_k"][str(k)]["err"]
        recs.append(d)
    return pd.DataFrame(recs)

"""Integration layer: the Phase-1 package acts_qaoa runs the GNN tuning problem unchanged.

A Phase-1 study is a folder with run_config.json and a runner (normally CKFRunner, i.e. ACTS).  make_run() builds
acts_qaoa.pipeline.Run around a Part-II runner (GNNRunner or LookupRunner), so the Phase-1 stages --
stage_warmstart (Latin hypercube), stage_build (ridge surrogate with nested CV -> exact QUBO -> Ising, exactness
checks, exact 2^n landscape, classical QAOA angles), stage_sample, stage_validate, stage_refine, stage_baselines,
stage_confirm -- are used as they are.

closed_loop() is the sampler-driven loop of QX1/QX2: after the shared warm start, every round refits the Phase-1
surrogate on everything this loop has measured (stage_build with the loop's own tags), lets ONE sampler propose
states of the exact QUBO (exact enumeration, SA, ideal QAOA, QAOA with the measured-fidelity noise model, or uniform
sampling as a control), decodes the best distinct proposals (Phase-1 diverse_select) and evaluates them on the real
objective; the new records feed the next fit.  TaggedView gives every loop its own record list (and budget) while
all loops share one evaluation cache, so identical configurations are computed once but counted for each method.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .._phase1 import ensure_phase1
from ..config import ScoreConfig
from ..utils import load_json, save_json
from .runner import EVENT_SEED_STRIDE

ensure_phase1()
import acts_qaoa  # noqa: E402
from acts_qaoa import pipeline as P1  # noqa: E402
from acts_qaoa.config import EVENT_SEED_STRIDE as P1_STRIDE  # noqa: E402
from acts_qaoa.config import ObjectiveConfig, ParamSpace, RunConfig  # noqa: E402
from acts_qaoa.landscape import counts_from_probs, simulated_annealing, uniform_counts  # noqa: E402
from acts_qaoa.qaoa_math import StatevectorQAOA, optimize_p1  # noqa: E402

if P1_STRIDE != EVENT_SEED_STRIDE:
    raise ImportError(f"Phase-1 EVENT_SEED_STRIDE {P1_STRIDE} != Part-II {EVENT_SEED_STRIDE}")

log = logging.getLogger(__name__)

# Measured effective fidelity of Phase-1 QAOA circuits on ibm_fez (white-noise model <E>_hw = F <E>_ideal +
# (1 - F) <E>_uniform), versus two-qubit gates of the transpiled circuit: HSF/docs/QAOA_ACTS_Guide.tex
# ("fidelity 0.82 / 0.79 / 0.67 / 0.15 at 36 / 90 / 220 / 1164 two-qubit gates (GenericDetector)").
MEASURED_FIDELITY: Tuple[Tuple[int, float], ...] = ((36, 0.82), (90, 0.79), (220, 0.67), (1164, 0.15))


def phase1_info() -> dict:
    return {"acts_qaoa": getattr(acts_qaoa, "__version__", None), "path": str(Path(acts_qaoa.__file__).parent)}


def objective_config(score: ScoreConfig) -> ObjectiveConfig:
    """The Phase-1 ObjectiveConfig that scores a Part-II record exactly like ScoreConfig."""
    mode = {"proxy": "reco_relative", "none": "none"}.get(score.time_mode)
    if mode is None:
        raise ValueError("the Phase-1 objective supports time_mode proxy or none")
    return ObjectiveConfig(w_eff=1.0, w_fake=1.0, k_dup=score.k_dup, k_time=score.k_time, runtime_mode=mode)


def integer_levels_exact(enc) -> bool:
    """Every level of every integer knob is an integer (before rounding), e.g. the 16-bit sub-grid's steps of 2."""
    for k, name in enumerate(enc.space.names):
        if enc.space.integer[k]:
            span = enc.space.hi[k] - enc.space.lo[k]
            vals = enc.space.lo[k] + np.arange(enc.n_levels) / enc.mask * span
            if np.max(np.abs(vals - np.rint(vals))) > 1e-9:
                return False
    return True


class _ExactIntegerGridFilter(logging.Filter):
    """Phase-1 stage_build warns 'integer parameter whose range is not 2^b-1 wide' for every integer knob whose span is
    not 2^b - 1.  Its check is conservative: on the 16-bit sub-grid the span is 2 (2^b - 1) and every level is an
    exact integer (checked by integer_levels_exact before this filter is installed), so the QUBO is exact."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # also expected in a closed loop: every round refits, so the previous round's angles become stale
        return "is an integer parameter whose range is not" not in msg and "[build] the problem changed" not in msg


def make_run(run_dir: Path, space: ParamSpace, bits: int, runner, n_events: int,
             reference: Mapping[str, float], score: Optional[ScoreConfig] = None, extra: Optional[dict] = None):
    """acts_qaoa.pipeline.Run around a Part-II runner; run_config.json is frozen like in Phase 1."""
    score = score or ScoreConfig()
    from acts_qaoa.encoding import Encoding
    if integer_levels_exact(Encoding(space, bits)):
        lg = logging.getLogger("acts_qaoa.pipeline")
        if not any(isinstance(f, _ExactIntegerGridFilter) for f in lg.filters):
            lg.addFilter(_ExactIntegerGridFilter())
    else:
        raise ValueError("an integer knob has non-integer grid levels: the QUBO would not be exact")
    cfg = RunConfig(space=space, bits=bits, objective=objective_config(score), ckf_n_events=int(n_events),
                    ckf_seed=None, ckf_threads=None, mock_ckf=False, version=acts_qaoa.__version__,
                    ckf_script=None, preset="hsf2-gnn", reference_params=dict(reference))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_config.json"
    if path.exists():
        old = load_json(path)
        if old != cfg.to_dict():
            raise RuntimeError(f"{path} defines a different problem; use a new run folder")
    else:
        save_json(path, cfg.to_dict())
    if extra:
        save_json(run_dir / "hsf2_problem.json", extra)
    if not score.is_default:
        # Phase-1 ObjectiveConfig has no DM efficiency or floor: it stays metadata, the runner scores every record
        save_json(run_dir / "hsf2_score.json", {"definition": score.describe(), "score_config": score.to_dict(),
                                                "note": "every record is scored by the Part-II runner with this "
                                                        "definition; the objective in run_config.json is metadata"})
    run = P1.Run(run_dir, cfg, ckf=None)
    run._runner = runner
    return run


class TaggedView:
    """A loop's own view of a shared runner: own records and tags, shared computation and cache."""

    def __init__(self, base) -> None:
        self.base = base
        self.flags = getattr(base, "flags", {})
        self.cache: Dict[str, dict] = {}
        self.cache_path = None

    def check(self) -> List[str]:
        return self.base.check()

    def score(self, rec):
        return self.base.score(rec)

    def _key(self, params_run, n_events, seed):
        return self.base._key(params_run, n_events, seed)

    def records(self) -> List[dict]:
        return list(self.cache.values())

    def evaluate(self, params, tag: str = "eval", n_events=None, seed=None, use_cache: bool = True) -> dict:
        rec = dict(self.base.evaluate(params, tag=tag, n_events=n_events, seed=seed, use_cache=use_cache))
        rec["tag"] = tag
        if rec["key"] not in self.cache:
            self.cache[rec["key"]] = rec
        else:
            rec["cached"] = True
        return rec

    @property
    def n_distinct(self) -> int:
        return len(self.cache)


# ---------------------------------------------------------------------------
# Measured-fidelity noise model
# ---------------------------------------------------------------------------
def fidelity_from_2q(n_2q: float, points: Sequence[Tuple[int, float]] = MEASURED_FIDELITY) -> float:
    """Effective fidelity F(n_2q): log-linear interpolation of the measured points (extrapolated beyond them)."""
    xs = np.array([p[0] for p in points], float)
    ls = np.log(np.array([p[1] for p in points], float))
    n = float(n_2q)
    if n <= xs[0]:
        k = 0
    elif n >= xs[-1]:
        k = len(xs) - 2
    else:
        k = int(np.searchsorted(xs, n)) - 1
    slope = (ls[k + 1] - ls[k]) / (xs[k + 1] - xs[k])
    return float(min(1.0, np.exp(ls[k] + slope * (n - xs[k]))))


def transpiled_2q(ising, scale: float, top_k: Optional[int] = None, backend_name: str = "ibm_fez",
                  seeds: Sequence[int] = (0, 1)) -> dict:
    """Two-qubit gate count of the p=1 QAOA circuit of `ising` (optionally sparsified) on a FAKE backend.

    Only transpilation (no hardware access): acts_qaoa.circuits.build_qaoa_circuit + transpile_best on the
    Qiskit fake backend of `backend_name` (calibration snapshot shipped with qiskit-ibm-runtime).
    """
    from acts_qaoa.circuits import bind, build_qaoa_circuit, transpile_best
    from acts_qaoa.ibm import fake_backend
    ism = ising.sparsified(top_k) if top_k else ising
    h_t, j_t = ism.terms(scale)
    qc, gam, bet = build_qaoa_circuit(ism.n, h_t, j_t, p=1)
    qc = bind(qc, gam, bet, [0.3], [0.4])           # angles do not change the gate count
    backend = fake_backend(backend_name)
    t0 = time.time()
    _, st = transpile_best(qc, backend, seeds=seeds)
    return {"backend": f"fake {backend_name}", "top_k": top_k, "n_couplings": ism.n_couplings(1e-12),
            "n_2q": int(st["n_2q"]), "depth_2q": st.get("depth_2q"), "fidelity_est_calibration": st.get("fidelity_est"),
            "fidelity_model": fidelity_from_2q(st["n_2q"]), "transpile_s": round(time.time() - t0, 1)}


def ibm_status(creds_file: Optional[str] = None) -> dict:
    """Account usage and the calibration medians of every QPU the account can use (metadata only, no QPU time).

    Credentials are read by Phase 1 (acts_qaoa.ibm.get_service: ~/ibm_creds.json or the saved account); nothing
    of them is returned.  'best' is the operational backend with the lowest median two-qubit gate error.
    """
    from acts_qaoa import ibm
    service = ibm.get_service(creds_file=creds_file)
    out: dict = {"usage": None, "backends": []}
    try:
        u = service.usage()
        out["usage"] = {k: v for k, v in u.items() if any(w in k for w in ("second", "period"))}
    except Exception as exc:  # older runtimes / plans without a usage endpoint
        out["usage_error"] = type(exc).__name__
    for b in service.backends(simulator=False):
        row: dict = {"name": b.name, "n_qubits": int(b.num_qubits)}
        try:
            st, p = b.status(), b.properties()
            e2 = [x.value for g in p.gates if len(g.qubits) == 2 for x in g.parameters if x.name == "gate_error"]
            ro = [p.readout_error(q) for q in range(b.num_qubits)]
            row.update(operational=bool(st.operational), queue=int(st.pending_jobs),
                       median_2q_error=float(np.median(e2)), median_readout_error=float(np.median(ro)),
                       calibrated=str(getattr(p, "last_update_date", "")))
        except Exception as exc:
            row["error"] = type(exc).__name__
        out["backends"].append(row)
    ok = [r for r in out["backends"] if r.get("operational") and "median_2q_error" in r]
    out["best"] = min(ok, key=lambda r: r["median_2q_error"])["name"] if ok else None
    return out


# ---------------------------------------------------------------------------
# Samplers of an exact QUBO (all Phase-1 functions); each returns {index: count}
# ---------------------------------------------------------------------------
def _probs_standard_p1(run, ising, scale, land, sparse_top_k: Optional[int] = None):
    hd = (land.E - ising.offset) / scale            # = Phase-1 _hdiag: the circuit Hamiltonian on all states
    dtype = np.complex64 if run.enc.n_qubits >= 20 else np.complex128
    if sparse_top_k:
        ism = ising.sparsified(sparse_top_k)
        # the circuit evolves under the sparse H; angles re-optimised for the FULL cost (Phase-1 _circuit_angles)
        r = optimize_p1(ism.h / scale, ism.full_J() / scale, h_obs=ising.h / scale, J_obs=ising.full_J() / scale)
        from acts_qaoa.landscape import enumerate_energies
        hd_sparse = (enumerate_energies(ising_to_qubo(ism)) - ism.offset) / scale
        sv = StatevectorQAOA(hd_sparse, mixer="x", dtype=dtype)
        return sv.probs([r["gamma"]], [r["beta"]]), {"gamma": r["gamma"], "beta": r["beta"], "sparse_top_k": sparse_top_k}
    entry = P1._angles_entry(run, "standard", 1)
    sv = StatevectorQAOA(hd, mixer="x", dtype=dtype)
    return sv.probs(entry["gammas"], entry["betas"]), {"gamma": entry["gammas"][0], "beta": entry["betas"][0]}


def ising_to_qubo(ism):
    """Exact inverse of Phase-1 QUBO.to_ising (z_k = 1 - 2 x_k):  z_k z_l = 1 - 2x_k - 2x_l + 4 x_k x_l."""
    from acts_qaoa.surrogate import QUBO
    n = ism.n
    J = np.triu(ism.J, 1)
    Jfull = J + J.T
    L = -2.0 * ism.h - 2.0 * Jfull.sum(axis=1)
    return QUBO(n=n, const=float(ism.offset + ism.h.sum() + J.sum()), L=L, W=4.0 * J)


def sample_qubo(run, sampler: str, shots: int, seed: int, noise: Optional[dict] = None) -> Tuple[Dict[int, int], dict]:
    """Proposals of one sampler for the run's current problem.json (built by stage_build)."""
    fit, qubo, ising, prob = run.load_problem()
    land = run.landscape(qubo)
    scale = float(prob["scale"])
    meta = {"sampler": sampler, "shots": shots, "seed": seed}
    if sampler == "exact":
        top = land.top_k(min(land.n_states, 20000))
        return {int(i): 1 for i in top}, meta
    if sampler == "random":
        return uniform_counts(run.enc.n_qubits, shots, seed), meta
    if sampler == "sa":
        # one final state per restart: as many restarts as the other samplers draw shots (equal sample counts)
        return simulated_annealing(qubo, restarts=shots, seed=seed), meta
    if sampler in ("qaoa", "qaoa-noise", "qaoa-noise-top24"):
        top_k = 24 if sampler == "qaoa-noise-top24" else None
        probs, ang = _probs_standard_p1(run, ising, scale, land, top_k)
        meta.update(ang)
        if sampler != "qaoa":
            if noise is None or "fidelity" not in noise:
                raise ValueError(f"sampler {sampler} needs the noise model (fidelity)")
            F = float(noise["fidelity"])
            probs = F * probs + (1.0 - F) / probs.size        # Phase-1 'predicted' white-noise model
            meta.update({"fidelity": F, "noise_model": noise})
        meta["exact_mean_energy"] = float(probs @ land.E)
        meta["exact_p_ground"] = float(probs[land.argmin])
        return counts_from_probs(probs, shots, seed), meta
    raise ValueError(f"unknown sampler {sampler}")


# ---------------------------------------------------------------------------
# The closed loop
# ---------------------------------------------------------------------------
def closed_loop(run, sampler: str, top_k: int, budget: int, min_distance: int = 2, shots: int = 4096, seed: int = 0,
                surrogate: str = "quadratic", noise: Optional[dict] = None, tag: str = "", max_rounds: int = 100) -> dict:
    """Warm-start records (tag 'ws') must exist; rounds of <= top_k new real evaluations proposed by `sampler`
    until the loop has `budget` distinct evaluations (warm start included)."""
    tag = tag or sampler
    fit_tags = ("ws", f"{tag}-rf*")
    rows: List[dict] = []
    summary: List[dict] = []
    r = 0
    while len(run.runner.records()) < budget and r < max_rounds:
        r += 1
        want = min(top_k, budget - len(run.runner.records()))
        P1.stage_build(run, surrogate=surrogate, fit_tags=fit_tags, ansatz=("standard",), p_max=1)
        fit, qubo, ising, prob = run.load_problem()
        land = run.landscape(qubo)
        counts, meta = sample_qubo(run, sampler, shots, seed + 1000 * r, noise)
        order = sorted(counts, key=lambda i: land.E[i])
        # never propose a configuration this loop has measured already (the budget buys new information)
        seen = {run.enc.index_from_params(rec["params"]) for rec in run.runner.records()}
        order = [i for i in order if i not in seen]
        ids = P1.diverse_select(order, run.enc, want, min_distance)
        if len(ids) < want:       # too few diverse proposals: the sampler's own next-best ones, never the exact order
            ids += [i for i in order if i not in ids][:want - len(ids)]
            meta["filled_without_min_distance"] = True
        if len(ids) < want:       # the sampler proposed fewer new states than needed: uniform unseen states
            rng = np.random.default_rng(seed + 7 * r)
            while len(ids) < want:
                i = int(rng.integers(land.n_states))
                if i not in seen and i not in ids:
                    ids.append(i)
            meta["filled_uniform"] = True
        pred, meas = [], []
        for rank, i in enumerate(ids):
            params = run.enc.params_from_index(i)
            rec = run.runner.evaluate(params, tag=f"{tag}-rf{r}")
            p = float(fit.predict(run.enc.unit_from_index(i)[None, :])[0])
            rows.append({"round": r, "rank": rank + 1, "index": int(i), "qubo_energy": float(land.E[i]),
                         "global_qubo_rank": land.rank(i), "predicted": p, "objective": rec.get("objective"),
                         "status": rec.get("status"), **params})
            if rec.get("objective") is not None:
                pred.append(p)
                meas.append(rec["objective"])
        summary.append({"round": r, "cv_r2": fit.cv_r2, "cv_rmse": fit.cv_rmse, "train_r2": fit.train_r2,
                        "n_fit": fit.n_samples, "surrogate": fit.kind, "alpha": fit.alpha,
                        "qubo_e_min": land.e_min, "predicted_best": max(pred) if pred else None,
                        "measured_best": max(meas) if meas else None,
                        "mean_overprediction": float(np.mean(np.subtract(pred, meas))) if meas else None,
                        "n_proposed_distinct": len(counts), "sampler_meta": {k: v for k, v in meta.items()
                                                                              if k != "noise_model"}})
    recs = [r for r in run.runner.records() if r.get("objective") is not None]
    best = max(recs, key=lambda r: r["objective"]) if recs else None
    return {"sampler": sampler, "tag": tag, "rounds": summary, "evaluations": rows,
            "n_real_evaluations": len(run.runner.records()),
            "best_objective": best["objective"] if best else None, "best_params": best["params"] if best else None,
            "best_tag": best["tag"] if best else None}


def best_so_far(records: Sequence[dict]) -> List[float]:
    """Running maximum of the objective over records in evaluation order (failed records keep the maximum)."""
    out, b = [], -math.inf
    for r in records:
        if r.get("objective") is not None:
            b = max(b, r["objective"])
        out.append(b)
    return out

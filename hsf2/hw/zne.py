"""Step 4, part B: zero-noise extrapolation of sampled quantities (hw-zne, hw-zne-analyze).

Circuits.  The ISA circuit of a target is built once by the runner's own transpile path (setting b by default: DD, no gate
twirling): QX2 targets by run.plan_job (Phase-1 transpile_best), QX3b components by components.plan_component.  Every two-qubit
gate G is then folded locally, G -> G (G^dagger G)^m, so that the noise factor is lambda = 2m + 1 in {1, 3, 5, ...}; a barrier
on the gate's two qubits separates the copies, so that nothing cancels (CZ is its own inverse; other gates use .inverse()).
Single-qubit gates and the layout are unchanged: the unitary is that of the unfolded circuit and the 2q count is multiplied by
lambda.  One PUB per (circuit, lambda), equal shots in every PUB, at most 10 PUBs per job, one typed confirmation per job,
budget check before the first one, booking and snapshots by hsf2.hw.run.

Job folder  <run folder>/hw/<DD-MM-YYYY-HH-MM>_zne_<setting>_<k>/   (k = 1, 2, ...)
  manifest.json                 kind "zne"; per PUB: target, lambda, K or component, angles, 2q counts of base and folded circuit
  pub_<k>_<target>_lam<l>.isa.qpy, pub_<k>.layout.json, snapshot_pre.json / _post.json, pub_<k>_counts.json, local_result.json
  QX2 targets also write <run folder>/samples/ibm-<backend>_<setting>_p1_top<K>_<jobid>_zne<l>.json (Phase-1 format; the lambda
  is in the tag, and the name does not end in _main, so that hw-analyze, which needs a control, never picks them up).

Analysis (analyze_zne).  Observables per (target, lambda): mean energy, F_E, F_XEB, the weight of the lowest q = 1 % and 10 % of
the 2^n states (QX2) or of the ground states (QX3b), and s_1..s_3 (hsf2.hw.analysis estimators; every one is a mean over the shots
of a per-state function, so a bootstrap over the shots at each lambda is a matrix product).  Each is extrapolated to lambda = 0
with (i) a least-squares line, (ii) Richardson (the polynomial through all points), (iii) A exp(-k lambda) + B with B fixed at the
value of the uniform distribution (0 for F_E, F_XEB, s_k; E_unif; q or g / 2^n).  Errors: bootstrap over the shots (1000
replicates, 16 / 84 %); the ideal value (noiseless circuit) is reported with whether the interval brackets it.  No model is
preferred in advance.  The errors are those of the shot noise only.
"""
from __future__ import annotations

import logging
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._phase1 import ensure_phase1
from ..utils import code_version, load_json, save_csv, save_json
from . import fold as FD
from . import run as R
from .analysis import counts_arrays, hamming_weights, walsh
from .build import PubSpec, check_equal_shots
from .config import HwConfig, creds_label
from .ledger import HwLedger
from .settings import Setting, get_setting, with_randomizations
from .snapshot import calibration_snapshot, layout_files, used_qubits_edges, utc_now
from .transpile import IsaPub

log = logging.getLogger(__name__)

KIND = "zne"
STEP = "step4"
DEFAULT_LAMBDAS = (1, 3, 5)
MAX_PUBS_PER_JOB = 10
FOLD_MAX_PUBS_PER_JOB = 30            # folding diagnosis (fold mode): both arms of a circuit in one job, one confirmation
AUTO_MAX_QUBITS = 14                  # local mode: Aer noise trajectories up to this many active qubits, else matrix product
LOCAL_METHODS = ("auto", "statevector", "matrix_product_state")
MODELS = ("linear", "richardson", "exponential")
TAIL_QS = (0.01, 0.1)
KMAX = 3
BOOT_N = 1000
BOOT_SEED = 0
GROUND_TOL = 1e-9
IDEAL_MEAN_TOL = 1e-4                 # relative: statevector (complex64 for QX2) vs the closed form of the manifest


# ---------------------------------------------------------------------------
# folding
# ---------------------------------------------------------------------------
def check_lambda(lam) -> int:
    """lambda must be an odd positive integer (2m + 1 copies of every two-qubit gate)."""
    if isinstance(lam, bool) or int(lam) != lam or lam < 1 or int(lam) % 2 != 1:
        raise ValueError(f"the noise factor must be an odd positive integer (1, 3, 5, ...), got {lam!r}")
    return int(lam)


def fold_two_qubit_gates(circ, lam: int):
    """`circ` with every two-qubit gate G replaced by G (G^dagger G)^m, lambda = 2m + 1, and a barrier on the gate's qubits
    between the copies.  Registers, qubit order, parameters, measurements, single-qubit gates, the barriers that are already
    there and the layout are unchanged; lambda = 1 returns a copy."""
    from qiskit.circuit import Gate
    lam = check_lambda(lam)
    m = (lam - 1) // 2
    if m == 0:
        return circ.copy()
    out = circ.copy_empty_like()
    for ins in circ.data:
        op, qs, cs = ins.operation, ins.qubits, ins.clbits
        out.append(op, qs, cs)
        if isinstance(op, Gate) and len(qs) > 1:
            if len(qs) != 2:
                raise ValueError(f"cannot fold the {len(qs)}-qubit gate {op.name}: the circuit is not in the ISA")
            inv = op.inverse()
            for _ in range(m):
                out.barrier(*qs)
                out.append(inv, qs, cs)
                out.barrier(*qs)
                out.append(op, qs, cs)
    if getattr(circ, "_layout", None) is not None:
        out._layout = circ._layout
    return out


@dataclass
class ZnePubSpec(PubSpec):
    """A PUB of a zne job: the circuit `target` folded to noise factor `lam` (kind main, equal shots in every PUB)."""
    target: str = ""
    lam: float = 1
    arm: str = ""                   # fold mode (hsf2.hw.fold): "base" (unfolded), "cz" or "all" (the folded gate set); "" = hw-zne
    subset: int = 0                 # fold mode: index of the random fold subset (shots are split equally between the subsets)
    lam_eff: float = 0.0            # fold mode: the achieved noise factor (the extrapolation fits over this)
    fold: Optional[dict] = None     # fold mode: the fold record (gate counts, folded indices, achieved lambda per type, seed)

    @property
    def name(self) -> str:
        if self.arm:
            return f"{self.target}_lam{self.lam:g}_{self.arm}_s{self.subset}"
        return f"{self.target}_lam{self.lam}"

    def to_dict(self) -> dict:
        d = dict(super().to_dict(), target=self.target, lam=self.lam)
        if self.arm:
            d.update(arm=self.arm, subset=self.subset, lam_eff=self.lam_eff, fold=self.fold)
        return d


def fold_pub(base: IsaPub, spec: ZnePubSpec, backend) -> IsaPub:
    """The PUB of `base` folded to spec.lam: same parameter values and layout, stats recomputed (2q count = lam x base's)."""
    ensure_phase1()
    from acts_qaoa.circuits import circuit_stats
    isa = fold_two_qubit_gates(base.isa, spec.lam)
    if [p.name for p in isa.parameters] != [p.name for p in base.isa.parameters]:
        raise RuntimeError("folding changed the circuit's parameters: the parameter values would no longer match")
    st = circuit_stats(isa, backend)
    st["transpile_seed"] = base.stats.get("transpile_seed")
    st["n_2q_base"] = int(base.stats["n_2q"])
    if st["n_2q"] != spec.lam * st["n_2q_base"]:
        raise RuntimeError(f"folded circuit has {st['n_2q']} two-qubit gates, expected {spec.lam} x {st['n_2q_base']}")
    return IsaPub(spec, isa, base.values, st, base.layout)


# ---------------------------------------------------------------------------
# targets and jobs
# ---------------------------------------------------------------------------
def parse_target(item: str, default_lams: Sequence[int]) -> Tuple[str, List[int]]:
    """'12:1,3,5' -> ('12', [1, 3, 5]); '12' -> ('12', default_lams).  Every lambda is checked."""
    name, _, ls = str(item).partition(":")
    if not name:
        raise ValueError(f"empty target in {item!r}")
    lams = [int(x) for x in ls.split(",") if x] if ls else [int(x) for x in default_lams]
    if not lams or len(set(lams)) != len(lams):
        raise ValueError(f"target {item!r}: needs distinct noise factors, got {lams}")
    return name, sorted(check_lambda(x) for x in lams)


@dataclass
class ZneTarget:
    label: str                      # K<k> (QX2) or the component name
    kind: str                       # "qx2" | "component"
    lams: List[int]
    base: IsaPub                    # the transpiled, unfolded circuit (lambda = 1)
    info: dict                      # fields copied into every PUB record of the manifest


@dataclass
class ZneJob:
    k: int
    n_jobs: int
    tag: str
    pubs: List[IsaPub]
    infos: List[dict]               # per PUB: the target's info
    estimate_s: float
    rep_delay_s: float
    duration_s: Optional[float]

    @property
    def shots(self) -> int:
        return check_equal_shots(self.pubs)


def _backend_name(backend) -> str:
    name = getattr(backend, "name", None)
    return name() if callable(name) else str(name)


def plan_qx2_targets(run_dir: Path, items: Sequence[str], default_lams: Sequence[int], setting: Setting, cfg: HwConfig,
                     backend, parse: Callable = None) -> List[ZneTarget]:
    """One target per K of a QX2 run folder: the main circuit of run.plan_job (the runner's transpile path).  `parse`: the
    target parser (default parse_target; parse_target_fold in fold mode)."""
    parse = parse or parse_target
    from .build import load_problem
    problem = load_problem(run_dir)
    cache: dict = {}
    out = []
    for item in items:
        name, lams = parse(item, default_lams)
        if not name.isdigit() or int(name) < 1:
            raise ValueError(f"QX2 target {item!r}: K must be a positive integer (couplings kept)")
        K = int(name)
        plan = R.plan_job(run_dir, setting, K, cfg, backend, 1, 0, f"zne_K{K}", problem, cache)
        base = plan.pubs[0]
        st = base.stats
        log.info("[hw-zne] K=%d: %d 2q gates, F_est %.3g, physical qubits %s", K, st["n_2q"], st.get("fidelity_est", float("nan")),
                 base.layout["measured"]["qubits"])
        out.append(ZneTarget(f"K{K}", "qx2", lams, base, {
            "K": K, "top_k": K, "n_vars": problem.n, "gammas": plan.gammas, "betas": plan.betas, "scale": problem.scale,
            "ideal_mean_energy": plan.ideal_energy, "fingerprint": problem.fingerprint}))
    return out


def resolve_component(comps, name: str):
    """The component whose file stem is `name`, or whose stem without the event prefix and its zeros is (1335_s3_f1_c20 for
    event000001335_s3_f1_c20); exactly one must match."""
    hits = [c for c in comps if c.name == name]
    if not hits:
        hits = [c for c in comps if re.sub(r"^[A-Za-z]*0*", "", c.name) == name]
    if len(hits) != 1:
        raise ValueError(f"component {name!r}: {len(hits)} matches among the components of the run folder")
    return hits[0]


def plan_component_targets(run_dir: Path, items: Sequence[str], default_lams: Sequence[int], setting: Setting, cfg: HwConfig,
                           backend, top_k: Optional[int] = None, parse: Callable = None) -> List[ZneTarget]:
    """One target per component JSON (by name) of a triplet run folder: components.plan_component."""
    parse = parse or parse_target
    from . import components as C
    comps = C.load_components(run_dir)
    out = []
    seen = set()
    for item in items:
        name, lams = parse(item, default_lams)
        comp = resolve_component(comps, name)
        if comp.name in seen:
            raise ValueError(f"component {comp.name} listed twice")
        seen.add(comp.name)
        plan = C.plan_component(comp, 0, cfg.shots, setting, backend, cfg.transpile_seeds, top_k)
        out.append(ZneTarget(comp.name, "component", lams, plan.pub, {
            "component": comp.name, "component_sha1": comp.sha1, "n_vars": comp.n, "top_k": top_k or None,
            "gammas": plan.pub.spec.gammas, "betas": plan.pub.spec.betas, "scale": plan.scale,
            "ideal_mean_energy": plan.info["ideal_mean_energy"], "n_couplings": plan.info["n_couplings"]}))
    return out


def plan_zne_jobs(targets: Sequence[ZneTarget], setting: Setting, cfg: HwConfig, backend, per_job: int = MAX_PUBS_PER_JOB
                  ) -> List[ZneJob]:
    """The (target, lambda) PUBs, in order, `per_job` to a job (at most 10), equal shots; the estimated QPU seconds of a job
    come from Phase-1 estimate_qpu_seconds with the longest (most folded) circuit of the job."""
    ensure_phase1()
    from acts_qaoa import ibm
    if setting.gauged or setting.fractional:
        raise ValueError("hw-zne uses settings a-d (no gauges, no fractional gates)")
    if not 1 <= per_job <= MAX_PUBS_PER_JOB:
        raise ValueError(f"--per-job must be between 1 and {MAX_PUBS_PER_JOB}, got {per_job}")
    if not targets:
        raise ValueError("no target to plan")
    entries = [(t, lam) for t in targets for lam in t.lams]
    rep = ibm.rep_delay_seconds(backend)
    rnd = 16 if (setting.gate_twirl or setting.measure_twirl) else 1
    chunks = [entries[i:i + per_job] for i in range(0, len(entries), per_job)]
    jobs = []
    for j, chunk in enumerate(chunks, 1):
        pubs = [fold_pub(t.base, ZnePubSpec(k, "main", 0, list(t.info["gammas"]), list(t.info["betas"]), int(cfg.shots), False,
                                           t.label, lam), backend) for k, (t, lam) in enumerate(chunk)]
        dur = max((p.stats.get("duration_s") or 0.0) for p in pubs) or None
        est = ibm.estimate_qpu_seconds([cfg.shots] * len(pubs), dur, rep, rnd)
        jobs.append(ZneJob(j, len(chunks), f"zne_{setting.name}_{j}", pubs, [t.info for t, _ in chunk], est, rep, dur))
    return jobs


def plan_table(jobs: Sequence[ZneJob]) -> List[dict]:
    """One row per PUB: job, pub, target, lambda, 2q gates (folded and base), depth, F_est, duration."""
    rows = []
    for job in jobs:
        for p in job.pubs:
            s = p.stats
            rows.append({"job": job.tag, "pub": p.spec.k, "target": p.spec.target, "lam": p.spec.lam, "n_2q": s["n_2q"],
                         "n_2q_base": s["n_2q_base"], "depth_2q": s.get("depth_2q"), "fidelity_est": s.get("fidelity_est"),
                         "duration_s": s.get("duration_s"), "shots": p.spec.shots, "job_estimate_s": job.estimate_s})
            if getattr(p.spec, "arm", ""):
                rows[-1].update(arm=p.spec.arm, subset=p.spec.subset, lam_eff=p.spec.lam_eff, ops=s.get("ops"))
    return rows


# ---------------------------------------------------------------------------
# folding diagnosis: fractional lambda, gate sets, random subsets, stored circuits as the base (hsf2.hw.fold)
# ---------------------------------------------------------------------------
def parse_target_fold(item: str, default_lams: Sequence[float]) -> Tuple[str, List[float]]:
    """'48:1,1.5,2,3' -> ('48', [1.0, 1.5, 2.0, 3.0]); '48' -> ('48', default_lams).  Fractional lambdas in [1, 9]; lambda = 1
    must be there (it is the unfolded base circuit of the target, shared by every arm)."""
    name, _, ls = str(item).partition(":")
    if not name:
        raise ValueError(f"empty target in {item!r}")
    lams = [float(x) for x in ls.split(",") if x] if ls else [float(x) for x in default_lams]
    lams = sorted(FD.check_fold_lambda(x) for x in lams)
    if len(set(lams)) != len(lams) or 1.0 not in lams:
        raise ValueError(f"target {item!r}: needs distinct noise factors including 1, got {lams}")
    return name, lams


def fold_arms(fold_gates: Optional[Sequence[str]]) -> List[str]:
    """The arms of a fold-mode job: 'cz' and / or 'all' (default: cz), each once."""
    arms = list(dict.fromkeys(fold_gates or ["cz"]))
    for a in arms:
        if a not in FD.GATE_SETS:
            raise ValueError(f"unknown fold gate set {a!r}: use cz or all")
    return arms


def fold_pub_gates(base: IsaPub, spec: ZnePubSpec, backend, gates: str, seed: int) -> IsaPub:
    """The PUB of `base` folded with hsf2.hw.fold.fold_gates to spec.lam: same parameter values and layout; the fold record and
    the achieved lambda go into the spec; the 2q count is checked against the record."""
    ensure_phase1()
    from acts_qaoa.circuits import circuit_stats
    isa, rec = FD.fold_gates(base.isa, spec.lam, gates, seed)
    if [p.name for p in isa.parameters] != [p.name for p in base.isa.parameters]:
        raise RuntimeError("folding changed the circuit's parameters: the parameter values would no longer match")
    st = circuit_stats(isa, backend)
    st["transpile_seed"] = base.stats.get("transpile_seed")
    st["n_2q_base"] = int(base.stats["n_2q"])
    st["ops"] = rec["ops"]
    pz = rec["per_gate"].get("cz")
    want = st["n_2q_base"] + (2 * pz["pairs"] if pz else 0)
    if st["n_2q"] != want:
        raise RuntimeError(f"folded circuit has {st['n_2q']} two-qubit gates, expected {want}")
    spec.lam_eff, spec.fold = rec["lam_eff"], rec
    return IsaPub(spec, isa, base.values, st, base.layout)


def plan_fold_jobs(targets: Sequence[ZneTarget], setting: Setting, cfg: HwConfig, backend, arms: Sequence[str], subsets: int = 1,
                   fold_seed: int = 0, per_job: int = FOLD_MAX_PUBS_PER_JOB) -> List[ZneJob]:
    """Fold-mode PUBs, in order: per target and lambda, lambda = 1 once (the unfolded base, arm 'base', shared by every arm) and
    every other lambda once per arm; each circuit `subsets` times, with different random fold subsets (identical copies where
    the fold is deterministic: lambda = 1 and the odd integers), the shots of the target split equally between the subsets.  At
    most FOLD_MAX_PUBS_PER_JOB PUBs per job, equal shots in every PUB."""
    ensure_phase1()
    from acts_qaoa import ibm
    if setting.gauged or setting.fractional:
        raise ValueError("hw-zne uses settings a-d (no gauges, no fractional gates)")
    if not 1 <= per_job <= FOLD_MAX_PUBS_PER_JOB:
        raise ValueError(f"--per-job must be between 1 and {FOLD_MAX_PUBS_PER_JOB}, got {per_job}")
    subsets = int(subsets)
    if subsets < 1 or cfg.shots % subsets:
        raise ValueError(f"--subsets {subsets} must be positive and divide the shots ({cfg.shots}): equal shots in every PUB")
    shots = cfg.shots // subsets
    if not targets:
        raise ValueError("no target to plan")
    entries = []
    for t in targets:
        for lam in t.lams:
            for arm in (["base"] if float(lam) == 1.0 else list(arms)):
                for s in range(subsets):
                    entries.append((t, float(lam), arm, s))
    rep = ibm.rep_delay_seconds(backend)
    rnd = setting.randomizations if (setting.gate_twirl or setting.measure_twirl) else 1
    chunks = [entries[i:i + per_job] for i in range(0, len(entries), per_job)]
    jobs = []
    for j, chunk in enumerate(chunks, 1):
        pubs = []
        for k, (t, lam, arm, s) in enumerate(chunk):
            spec = ZnePubSpec(k, "main", 0, list(t.info["gammas"]), list(t.info["betas"]), shots, False, t.label, lam, arm, s)
            pubs.append(fold_pub_gates(t.base, spec, backend, "cz" if arm == "base" else arm, int(fold_seed) + 1000 * s))
        dur = max((p.stats.get("duration_s") or 0.0) for p in pubs) or None
        est = ibm.estimate_qpu_seconds([shots] * len(pubs), dur, rep, rnd)
        jobs.append(ZneJob(j, len(chunks), f"zne_{setting.name}_{j}", pubs, [t.info for t, _, _, _ in chunk], est, rep, dur))
    return jobs


_PUB_RUN_KEYS = ("k", "name", "kind", "gauge", "shots", "gauged", "lam", "target", "n_2q", "n_2q_base", "depth_2q", "fidelity_est",
                 "duration_s", "transpile_seed", "measured_qubits", "initial_layout", "arm", "subset", "lam_eff", "fold", "ops")


def plan_job_targets(job_dir: Path, items: Sequence[str], default_lams: Sequence[float], backend) -> Tuple[List[ZneTarget], dict]:
    """The targets of a stored zne job (--from-job): the lambda = 1 PUB of every selected target is the base circuit, loaded
    from its QPY with its layout and angles, so that the same transpiled circuit on the same qubits is folded again.  `items`
    select the targets as in --K / --components ('48', '12:1,1.5,2,3', '1332_s2_f1_c31'); returns (targets, provenance)."""
    import hashlib
    from types import SimpleNamespace
    from .snapshot import read_qpy
    job_dir = Path(job_dir)
    man = load_json(job_dir / "manifest.json")
    if man.get("kind") != KIND:
        raise ValueError(f"{job_dir} is not a zne job (kind {man.get('kind')!r})")
    bases = {}
    for p in man["pubs"]:
        if (not p.get("arm") and float(p["lam"]) == 1.0) or p.get("arm") == "base":
            bases.setdefault(p["target"], p)
    if not bases:
        raise ValueError(f"{job_dir} has no lambda = 1 PUB to use as the base circuit")
    out, prov = [], {"from_job": job_dir.name, "from_job_backend": man.get("backend"), "base_pubs": {}}
    for item in items:
        name, lams = parse_target_fold(item, default_lams)
        label = f"K{name}" if name.isdigit() else resolve_component([SimpleNamespace(name=x) for x in bases], name).name
        if label not in bases:
            raise ValueError(f"target {label} is not in {job_dir.name}: it has {sorted(bases)}")
        p = bases[label]
        qpy = job_dir / f"pub_{p['k']}_{p['name']}.isa.qpy"
        lay = load_json(job_dir / f"pub_{p['k']}.layout.json")
        isa = read_qpy(qpy)
        gam, bet = list(p["gammas"]), list(p["betas"])
        values = np.array([float((gam if x.name.startswith("gamma") else bet)[int(x.name.split("[")[1].rstrip("]"))])
                           for x in isa.parameters])
        stats = dict(lay["stats"], transpile_seed=lay.get("transpile_seed"))
        layout = {k: lay[k] for k in ("n_logical", "initial", "final", "measured", "active_qubits", "edges")}
        spec = ZnePubSpec(0, "main", 0, gam, bet, int(p["shots"]), False, label, 1, "base", 0)
        info = {k: v for k, v in p.items() if k not in _PUB_RUN_KEYS}
        out.append(ZneTarget(label, "qx2" if p.get("K") else "component", lams, IsaPub(spec, isa, values, stats, layout), info))
        prov["base_pubs"][label] = {"pub": p["k"], "qpy": qpy.name, "qpy_sha1": hashlib.sha1(qpy.read_bytes()).hexdigest(),
                                    "transpile_seed": lay.get("transpile_seed"), "measured_qubits": layout["measured"]["qubits"]}
        log.info("[hw-zne] %s: base circuit from %s (%d 2q gates, qubits %s)", label, qpy.name, stats["n_2q"],
                 layout["measured"]["qubits"])
    return out, prov


# ---------------------------------------------------------------------------
# manifest and files
# ---------------------------------------------------------------------------
def job_manifest(job: ZneJob, setting: Setting, cfg: HwConfig, backend_name: str, run_dir: Path, step: str, command: str,
                 selection: Optional[dict] = None) -> dict:
    pubs = []
    for p, info in zip(job.pubs, job.infos):
        pubs.append(dict(p.spec.to_dict(), **{k: v for k, v in info.items() if k not in ("gammas", "betas")},
                         n_2q=p.stats.get("n_2q"), n_2q_base=p.stats.get("n_2q_base"), depth_2q=p.stats.get("depth_2q"),
                         fidelity_est=p.stats.get("fidelity_est"), duration_s=p.stats.get("duration_s"),
                         transpile_seed=p.stats.get("transpile_seed"), measured_qubits=p.layout["measured"]["qubits"],
                         initial_layout=p.layout["initial"], **({"ops": p.stats.get("ops")} if getattr(p.spec, "arm", "") else {})))
    return {"schema": R.SCHEMA, "step": step, "kind": KIND, "state": "planned", "mode": cfg.mode, "created": utc_now(),
            "command": command, "tag": job.tag, "backend": backend_name, "calibration_id": None, "versions": R.versions(),
            "code": code_version(), "creds": creds_label(cfg.creds_file), "config": cfg.to_dict(),
            "primitive": cfg.primitive, "setting": setting.to_dict(), "sampler_options": None, "job_id": None,
            "billed": None, "run_dir": str(run_dir), "job_index": job.k, "n_jobs": job.n_jobs, "shots": job.shots,
            "estimate_s": job.estimate_s, "rep_delay_s": job.rep_delay_s, "circuit_duration_s": job.duration_s,
            "selection": selection, "pubs": pubs}


def write_job_files(job_dir: Path, job: ZneJob, backend) -> dict:
    """QPY, layouts and the pre-run calibration snapshot (as run.write_plan_files); returns the snapshot."""
    job_dir = Path(job_dir)
    layout_files(job_dir, job.pubs)
    qs, es = used_qubits_edges([p.layout for p in job.pubs])
    snap = calibration_snapshot(backend, qs, es, source="target")
    save_json(job_dir / "snapshot_pre.json", snap)
    return snap


def write_counts(job_dir: Path, manifest: dict, counts: Sequence[Dict[int, int]]) -> None:
    """pub_<k>_counts.json of every PUB, saved first (a submitted job is billed: its counts are kept even if a check fails);
    then the shots and the index range are checked."""
    pubs = manifest["pubs"]
    if len(counts) != len(pubs):
        raise ValueError(f"{len(counts)} count sets for {len(pubs)} PUBs")
    problems = []
    for p, c in zip(pubs, counts):
        total = int(sum(c.values()))
        save_json(Path(job_dir) / f"pub_{p['k']}_counts.json",
                  {"pub": p["k"], "name": p["name"], "target": p["target"], "lam": p["lam"], "n": p["n_vars"],
                   "shots": p["shots"], "total": total, "counts": {str(k): int(v) for k, v in sorted(c.items())}})
        if total != p["shots"]:
            problems.append(f"PUB {p['k']} returned {total} shots, expected {p['shots']}")
        if c and (min(c) < 0 or max(c) >= 1 << p["n_vars"]):
            problems.append(f"PUB {p['k']}: sample index outside 0..2^{p['n_vars']} - 1")
    if problems:
        raise ValueError("; ".join(problems))


def export_qx2_samples(run_dir: Path, manifest: dict, counts: Sequence[Dict[int, int]]) -> List[Path]:
    """Phase-1 sample sets of the QX2 PUBs, tagged ibm-<backend>_<setting>_p1_top<K>_<jobid>_zne<lambda> (never replaced)."""
    from .build import load_problem
    pubs = [(p, c) for p, c in zip(manifest["pubs"], counts) if p.get("K") and not p.get("arm")]      # fold mode: no Phase-1 samples
    if not pubs:
        return []
    problem = load_problem(run_dir)
    billed = (manifest.get("billed") or {}).get("used")
    out: List[Path] = []
    for p, c in pubs:
        if p["fingerprint"] != problem.fingerprint:
            raise ValueError(f"PUB {p['k']}: problem fingerprint {p['fingerprint']} != {problem.fingerprint}")
        pm = {"job_id": manifest["job_id"], "mode": manifest["mode"], "backend": manifest["backend"],
              "setting": {"name": manifest["setting"]["name"], "gauged": False}, "circ": f"top{p['K']}", "top_k": p["K"],
              "gammas": p["gammas"], "betas": p["betas"], "ideal_mean_energy": p["ideal_mean_energy"], "gauges": [],
              "tag": f"{manifest['tag']}_{p['name']}", "primitive": manifest["primitive"],
              "pubs": [{"fidelity_est": p.get("fidelity_est")}]}
        out += R.export_samples(problem, pm, {"main": {int(k): int(v) for k, v in c.items()}}, billed, ("main",),
                                {"main": f"zne{p['lam']}"}, {"zne_lambda": p["lam"], "zne_job": manifest["tag"],
                                                             "n_2q": p.get("n_2q"), "n_2q_base": p.get("n_2q_base")})
    return out


def finish_zne_job(job_dir: Path, manifest: dict, counts: Sequence[Dict[int, int]], job=None, backend=None) -> None:
    """Counts files, QX2 sample export, post-run snapshot, final manifest (also used by retrieve_zne_job)."""
    job_dir = Path(job_dir)
    write_counts(job_dir, manifest, counts)
    samples = export_qx2_samples(Path(manifest["run_dir"]), manifest, counts)
    if samples:
        manifest["samples"] = [s.name for s in samples]
    if backend is not None:
        lays = [load_json(p) for p in sorted(job_dir.glob("pub_*.layout.json"))]
        qs, es = used_qubits_edges(lays)
        props = None
        if job is not None:
            try:
                props = job.properties()
            except Exception as exc:
                log.warning("[hw-zne] job.properties() failed (%s): post snapshot uses the backend's current properties",
                            type(exc).__name__)
        snap = calibration_snapshot(backend, qs, es, source="job.properties" if props is not None else "backend",
                                    properties=props, refresh=props is None)
        save_json(job_dir / "snapshot_post.json", snap)
        manifest["calibration_id"] = snap.get("calibration_id") or manifest.get("calibration_id")
    manifest["state"] = "done"
    R.save_manifest(job_dir, manifest)


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def _local_job(backend, job: ZneJob, setting: Setting, job_dir: Path, manifest: dict, seed: int, method: str):
    """The local branch of run.execute_job with the choice of simulator per PUB (local_result.json is the same record)."""
    from . import fake as F
    if method not in LOCAL_METHODS:
        raise ValueError(f"sim_method must be one of {LOCAL_METHODS}, got {method!r}")
    check_equal_shots(job.pubs)
    methods = [("statevector" if int(p.stats.get("n_active_qubits") or 99) <= AUTO_MAX_QUBITS else "matrix_product_state")
               if method == "auto" else method for p in job.pubs]
    counts: List[Optional[Dict[int, int]]] = [None] * len(job.pubs)
    for m in ("statevector", "matrix_product_state"):
        ks = [k for k, x in enumerate(methods) if x == m]
        if ks:
            for k, c in zip(ks, F.run_local(backend, [job.pubs[k] for k in ks], seed, method=m)):
                counts[k] = c
    job_id = F.local_job_id(manifest.get("tag", ""))
    save_json(Path(job_dir) / "local_result.json", {"job_id": job_id, "seed": seed, "backend": manifest["backend"],
                                                    "sim_methods": methods,
                                                    "counts": [{str(k): v for k, v in c.items()} for c in counts]})
    manifest.update({"job_id": job_id, "state": "done", "sim_methods": methods, "options_requested": setting.options(),
                     "sampler_options": {"note": "recorded only: the local simulators ignore twirling and DD",
                                         "sim_methods": methods},
                     "billed": {"job_s": 0.0, "used": 0.0, "source": "local"}})
    R.save_manifest(job_dir, manifest)
    return counts


def _confirm_with_pubs(confirm: Callable[[str], None], job: ZneJob) -> Callable[[str], None]:
    """The typed confirmation of run.execute_job, preceded by the PUBs of the job."""
    pubs = ", ".join(f"{p.spec.target} lambda={p.spec.lam:g}{' ' + p.spec.arm + ' s' + str(p.spec.subset) if getattr(p.spec, 'arm', '') else ''} "
                     f"({p.stats['n_2q']} 2q gates)" for p in job.pubs)
    header = f"ZNE job {job.k} of {job.n_jobs}, one PUB per (circuit, lambda): {pubs}"
    return lambda msg: confirm(header + chr(10) + msg)


def run_zne(run_dir: Path, ks: Sequence[str], components: Sequence[str], setting_name: str, cfg: HwConfig,
            lambdas: Sequence[int] = DEFAULT_LAMBDAS, per_job: int = MAX_PUBS_PER_JOB, top_k: Optional[int] = None,
            step: str = STEP, command: str = "", out_dir: Optional[Path] = None,
            confirm: Callable[[str], None] = R.confirm_submit, seed: int = 0, sim_method: str = "auto", service=None,
            fold_gates: Optional[Sequence[str]] = None, subsets: int = 1, from_job: Optional[Path] = None, fold_seed: int = 0,
            randomizations: Optional[int] = None, wait: bool = True) -> dict:
    """hw-zne: plan every job (transpile, fold), then dry (plan files only), local (simulate) or submit them one after the
    other, each with its budget check and typed confirmation.  `wait` False (submit): every job is submitted without
    waiting for its result (state 'submitted', no counts); hw-retrieve <job id> rebuilds the outputs later.  `ks`: QX2 targets 'K[:l,l,...]' of a QX2 run folder;
    `components`: component names 'name[:l,l,...]' of a triplet run folder (exactly one of the two).  Returns {'jobs', 'plan',
    'service', 'estimate_total_s'}; the plan table is also written to out_dir/zne_plan.json.

    Fold mode (folding diagnosis, hsf2.hw.fold) when `fold_gates` (a list of 'cz' / 'all': one arm each), `subsets` > 1 or
    `from_job` is given: fractional lambdas, the gate sets folded, `subsets` random fold subsets per circuit (the shots split
    equally), up to FOLD_MAX_PUBS_PER_JOB PUBs per job.  `from_job`: a finished or dry zne job folder of THIS run folder whose
    lambda = 1 PUBs (QPY, layout, angles) are the base circuits, so that the same transpiled circuits are folded again.
    `randomizations`: twirling randomisations of the setting (32 for the twirled arm).  Without them nothing changes."""
    cfg.validate()
    run_dir = Path(run_dir)
    setting = get_setting(setting_name)
    if randomizations is not None:
        setting = with_randomizations(setting, randomizations)
    fold_mode = fold_gates is not None or from_job is not None or int(subsets) != 1
    arms = fold_arms(fold_gates) if fold_mode else []
    if setting.gauged or setting.fractional:
        raise ValueError("hw-zne uses settings a-d (no gauges, no fractional gates)")
    if bool(ks) == bool(components):
        raise ValueError("give either --K (QX2 run folder) or --components (triplet run folder), not both and not neither")
    if sim_method not in LOCAL_METHODS:
        raise ValueError(f"sim_method must be one of {LOCAL_METHODS}, got {sim_method!r}")
    if fold_mode:
        lambdas = [FD.lam_key(FD.check_fold_lambda(x)) for x in lambdas]
        if from_job is not None and (Path(from_job).resolve().parent.name != "hw" or Path(from_job).resolve().parents[1] != run_dir.resolve()):
            raise ValueError(f"--from-job {from_job} must be a job folder <run>/hw/<job> of the run folder {run_dir}")
    else:
        for lam in lambdas:
            check_lambda(lam)
    limit = FOLD_MAX_PUBS_PER_JOB if fold_mode else MAX_PUBS_PER_JOB
    if not 1 <= per_job <= limit:
        raise ValueError(f"--per-job must be between 1 and {limit}, got {per_job}")
    backend, service = R.get_job_backend(cfg, setting, service)
    name = _backend_name(backend)
    prov = None
    if fold_mode:
        items = list(ks) if ks else list(components)
        if from_job is not None:
            targets, prov = plan_job_targets(Path(from_job), items, lambdas, backend)
        elif ks:
            targets = plan_qx2_targets(run_dir, ks, lambdas, setting, cfg, backend, parse_target_fold)
        else:
            targets = plan_component_targets(run_dir, components, lambdas, setting, cfg, backend, top_k, parse_target_fold)
        jobs = plan_fold_jobs(targets, setting, cfg, backend, arms, subsets, fold_seed, per_job)
    else:
        targets = (plan_qx2_targets(run_dir, ks, lambdas, setting, cfg, backend) if ks else
                   plan_component_targets(run_dir, components, lambdas, setting, cfg, backend, top_k))
        jobs = plan_zne_jobs(targets, setting, cfg, backend, per_job)
    total = sum(j.estimate_s for j in jobs)
    table = plan_table(jobs)
    for r in table:
        log.info("[hw-zne] %s pub %d: %-26s lambda %g%s: %4d 2q gates (base %d), F_est %.3g", r["job"], r["pub"], r["target"],
                 r["lam"], f" -> {r['lam_eff']:.3f} {r['arm']} s{r['subset']}" if r.get("arm") else "", r["n_2q"], r["n_2q_base"],
                 r["fidelity_est"] or float("nan"))
    log.info("[hw-zne] %d PUBs x %d shots in %d jobs, setting %s; estimated %.1f s of QPU time in total (%s)", len(table),
             cfg.shots, len(jobs), setting.name, total, ", ".join(f"{j.estimate_s:.1f}" for j in jobs))
    if out_dir is not None:
        save_json(Path(out_dir) / "zne_plan.json", {"setting": setting.name, "shots": cfg.shots, "backend": name,
                                                    "estimate_total_s": total, "pubs": table})
    ledger = None
    if cfg.mode == "submit":
        from . import session
        ledger = HwLedger(cfg.ledger_path(), cfg.budget_s, creds_file=cfg.creds_file)
        ledger.check(total, session.service_remaining(service), cfg.reserve_s)       # all jobs, before the first one
    elif cfg.mode == "dry":
        led = HwLedger(cfg.ledger_path(), cfg.budget_s, creds_file=cfg.creds_file)
        log.info("[hw-zne] ledger %s: used %.1f s, remaining %.1f s of %.0f s; a submit of these jobs would book an estimated "
                 "%.1f s", led.path, led.used_seconds(), led.remaining_seconds(), led.budget_s, total)
    selection = {"targets": [{"label": t.label, "kind": t.kind, "lambdas": t.lams} for t in targets], "per_job": per_job,
                 "default_lambdas": list(lambdas)}
    if fold_mode:
        selection["fold"] = {"arms": arms, "subsets": int(subsets), "fold_seed": int(fold_seed), "twirl_randomizations": randomizations,
                             "shots_per_pub": cfg.shots // int(subsets), **(prov or {})}
    results: List[dict] = []
    for job in jobs:
        job_dir = R.new_job_dir(run_dir, job.tag)
        manifest = job_manifest(job, setting, cfg, name, run_dir, step, command, selection)
        write_job_files(job_dir, job, backend)
        R.save_manifest(job_dir, manifest)             # manifest (state planned) and QPY are on disk before anything is submitted
        log.info("[hw-zne] job %d/%d: %s | %d PUBs x %d shots | estimated %.1f s of QPU time", job.k, job.n_jobs, job_dir,
                 len(job.pubs), job.shots, job.estimate_s)
        res = {"job_dir": job_dir, "manifest": manifest, "job": job, "job_id": None, "counts": None}
        results.append(res)
        if cfg.mode == "dry":
            manifest["state"] = "dry"
            R.save_manifest(job_dir, manifest)
            continue
        try:
            if cfg.mode == "local":
                counts = _local_job(backend, job, setting, job_dir, manifest, seed, sim_method)
                ex = {"job_id": manifest["job_id"], "counts": counts, "billed": manifest["billed"], "job": None}
            else:
                ex = R.execute_job(backend, job.pubs, cfg, setting, f"hw-zne-{setting.name}", job.estimate_s, job_dir,
                                   manifest, str(run_dir), service, ledger, _confirm_with_pubs(confirm, job), wait, False, seed)
        except Exception as exc:
            if manifest.get("job_id") is None:             # refused or failed before anything was submitted or simulated
                manifest.update(state="aborted", aborted=f"{type(exc).__name__}: {exc}")
                R.save_manifest(job_dir, manifest)
            raise
        res.update(job_id=ex["job_id"], counts=ex["counts"], billed=ex["billed"])
        if ex["counts"] is None:                           # submitted without waiting: hw-retrieve <job id> finishes it
            log.info("[hw-zne] job %s submitted without waiting; later: python -m hsf2 hw-retrieve %s", job_dir.name,
                     ex["job_id"])
            continue
        finish_zne_job(job_dir, manifest, ex["counts"], ex.get("job"), backend)
    return {"jobs": results, "plan": table, "service": service, "estimate_total_s": total}


def retrieve_zne_job(job_id: str, cfg: HwConfig, run_dir: Optional[Path] = None, search_root: Optional[Path] = None,
                     service=None) -> dict:
    """Rebuild the outputs of a zne job whose shell was lost (as components.retrieve_components_job): counts from
    local_result.json or from IBM, billed time resolved and booked, counts files, samples and post snapshot."""
    from . import fake as F
    job_dir = R.find_job_dir(job_id, run_dir, search_root)
    manifest = load_json(job_dir / "manifest.json")
    if manifest.get("kind") != KIND:
        raise ValueError(f"{job_dir} is not a zne job (kind {manifest.get('kind')!r})")
    manifest["run_dir"] = str(job_dir.parents[1])
    job = None
    if job_id.startswith(F.LOCAL_PREFIX):
        stored = load_json(job_dir / "local_result.json")
        counts = [{int(k): int(v) for k, v in c.items()} for c in stored["counts"]]
        backend = F.fake_backend(cfg.fake_backend)
    else:
        from . import session
        service = service or session.get_service(cfg.creds_file)
        job = service.job(job_id)
        ledger = R.retrieve_ledger(cfg, manifest)
        entry = ledger.entry(job_id) or {}
        est = entry.get("estimate_s", manifest.get("estimate_s", 0.0))
        got = R.collect_job(job, job_dir, manifest, service, ledger, est, None, len(manifest["pubs"]), False, booked=entry)
        counts = got["counts"]
        backend = job.backend() if hasattr(job, "backend") else None
    if counts is not None:
        finish_zne_job(job_dir, manifest, counts, job, backend)
    return {"job_dir": job_dir, "manifest": manifest}


# ---------------------------------------------------------------------------
# extrapolation
# ---------------------------------------------------------------------------
def richardson_weights(lams: Sequence[float]) -> np.ndarray:
    """Lagrange weights w_i of the polynomial through (lambda_i, y_i) evaluated at lambda = 0."""
    lam = np.asarray(lams, float)
    w = np.ones(lam.size)
    for i in range(lam.size):
        for j in range(lam.size):
            if j != i:
                w[i] *= (0.0 - lam[j]) / (lam[i] - lam[j])
    return w


def extrapolate(lams: Sequence[float], Y: np.ndarray, floor: np.ndarray) -> Dict[str, np.ndarray]:
    """Extrapolate Y (L, R, J) = L lambdas x R replicates x J observables to lambda = 0 with the three models; `floor` (J,) is
    the fixed B of the exponential.  Returns {model: (R, J)} plus exp_A and exp_k; the exponential is NaN where Y - B does not
    keep one sign over all lambdas (the logarithm is undefined)."""
    lam = np.asarray(lams, float)
    Y = np.asarray(Y, float)
    if Y.ndim != 3 or Y.shape[0] != lam.size or lam.size < 2:
        raise ValueError(f"need Y of shape (L, R, J) with L = {lam.size} >= 2 noise factors, got {Y.shape}")
    lc = lam - lam.mean()
    slope = np.tensordot(lc, Y, axes=(0, 0)) / float(lc @ lc)
    linear = Y.mean(axis=0) - slope * lam.mean()
    rich = np.tensordot(richardson_weights(lam), Y, axes=(0, 0))
    Z = Y - np.asarray(floor, float)[None, None, :]
    sgn = np.sign(Z[0])
    ok = (sgn != 0) & np.all(Z * sgn[None] > 0, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lz = np.log(np.where(ok[None], np.abs(Z), 1.0))
        es = np.tensordot(lc, lz, axes=(0, 0)) / float(lc @ lc)
        ei = lz.mean(axis=0) - es * lam.mean()
    A = np.where(ok, sgn * np.exp(ei), np.nan)
    k = np.where(ok, -es, np.nan)
    expo = np.where(ok, np.asarray(floor, float)[None, :] + A, np.nan)
    return {"linear": linear, "richardson": rich, "exponential": expo, "exp_A": A, "exp_k": k}


# ---------------------------------------------------------------------------
# observables
# ---------------------------------------------------------------------------
@dataclass
class Observables:
    """Per-state functions whose shot mean is the observable: values(idx) -> (len(idx), J)."""
    names: List[str]
    ideal: np.ndarray
    floor: np.ndarray
    _funcs: List[Callable[[np.ndarray], np.ndarray]] = field(default_factory=list)

    def values(self, idx: np.ndarray) -> np.ndarray:
        return np.column_stack([f(idx) for f in self._funcs])


def correlation_state_values(p: np.ndarray, idx_all: np.ndarray, kmax: int,
                             transform: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """(len(idx_all), kmax): g_k(x) / sum_{|S|=k} c_S^2 of hsf2.hw.analysis.correlation_fidelity, whose shot mean is s_k.  One
    2^n vector at a time.  `transform`: applied to each full g_k before indexing (the read-out correction)."""
    c = walsh(np.array(p, np.float64))
    w = hamming_weights(c.size.bit_length() - 1)
    out = np.empty((idx_all.size, kmax))
    for k in range(1, kmax + 1):
        g = np.where(w == k, c, 0.0)
        denom = float(g @ g)
        if not denom > 0:
            raise ValueError(f"the ideal distribution has no correlation of order {k}: s_{k} is undefined")
        walsh(g)
        if transform is not None:
            g = transform(g)
        out[:, k - 1] = g[idx_all] / denom
        del g
    return out


def build_observables(E: np.ndarray, p: np.ndarray, idx_all: np.ndarray, masks: Dict[str, np.ndarray], kmax: int = KMAX,
                      readout_mats: Optional[Sequence[np.ndarray]] = None) -> Observables:
    """mean_energy, F_E, F_XEB, one P(mask) per entry of `masks` and s_1..s_kmax of the ideal distribution `p` over the states
    with energies `E`; `idx_all` is every state that any sample set contains (the s_k values are computed on those).
    `readout_mats` (P(read | true) of classical bit k, k = 0..n-1): every per-state function O is replaced by
    (A_{n-1}^-1 x ... x A_0^-1)^T O (readout.tensor_apply), whose shot mean is the read-out-corrected expectation of O.
    Ideal values and floors are not affected."""
    N = p.size
    idx_all = np.asarray(idx_all, np.int64)
    E_unif = float(E.mean())
    e_ideal = float(p @ E)
    if abs(E_unif - e_ideal) < 1e-12:
        raise ValueError("ideal mean energy equals the uniform one: F_E is undefined")
    sum_p2 = float(p @ p)
    tr = None
    if readout_mats is not None:
        from .readout import tensor_apply
        ops = [np.linalg.inv(np.asarray(m, float)).T for m in readout_mats]

        def tr(v):                                      # noqa: E306 -- the transposed inverse of the read-out
            return tensor_apply(np.asarray(v, np.float64), ops)
    Ev = E if tr is None else tr(E)                     # corrections are linear: constants pass unchanged
    pv = p if tr is None else tr(p)
    names, ideal, floor, funcs = ["mean_energy", "F_E", "F_XEB"], [e_ideal, 1.0, 1.0], [E_unif, 0.0, 0.0], [
        lambda i: Ev[i], lambda i: (E_unif - Ev[i]) / (E_unif - e_ideal), lambda i: (pv[i] - 1.0 / N) / (sum_p2 - 1.0 / N)]
    for nm, mask in masks.items():
        names.append(nm)
        ideal.append(float(p[mask].sum()))
        floor.append(float(mask.sum()) / N)
        mv = mask.astype(np.float64) if tr is None else tr(mask.astype(np.float64))
        funcs.append(lambda i, m=mv: m[i])
    sv = correlation_state_values(p, idx_all, kmax, tr)
    for k in range(kmax):
        names.append(f"s{k + 1}")
        ideal.append(1.0)
        floor.append(0.0)
        funcs.append(lambda i, col=sv[:, k]: col[np.searchsorted(idx_all, i)])
    return Observables(names, np.array(ideal), np.array(floor), funcs)


def analyze_counts(obs: Observables, counts_by_lam: Dict[int, Dict[int, int]], boot_n: int = BOOT_N, seed: int = BOOT_SEED,
                   tag: str = "") -> dict:
    """Points, bootstrap errors and the three extrapolations of every observable (see the module docstring).  The bootstrap
    resamples the shots of every lambda independently (multinomial) and repeats the extrapolation on every replicate."""
    lams = sorted(counts_by_lam)
    if len(lams) < 2:
        raise ValueError(f"extrapolation needs at least two noise factors, got {lams}")
    J = len(obs.names)
    Y = np.empty((len(lams), boot_n + 1, J))
    shots = {}
    for a, lam in enumerate(lams):
        idx, c = counts_arrays({str(k): v for k, v in counts_by_lam[lam].items()})
        o = np.argsort(idx)
        idx, c = idx[o], c[o]
        S = int(c.sum())
        shots[lam] = S
        V = obs.values(idx)
        lam_id = int(lam) if float(lam).is_integer() else int(round(float(lam) * 1e6))       # fractional lambdas: own streams
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(tag.encode()), lam_id]))
        Y[a, 0] = c @ V / S
        Y[a, 1:] = rng.multinomial(S, c / S, size=boot_n) @ V / S
    ex = extrapolate(lams, Y, obs.floor)
    res: Dict[str, dict] = {}
    for j, name in enumerate(obs.names):
        pts = {}
        for a, lam in enumerate(lams):
            lo, hi = np.percentile(Y[a, 1:, j], [16, 84])
            pts[str(lam)] = {"value": float(Y[a, 0, j]), "err": float(Y[a, 1:, j].std(ddof=1)), "lo68": float(lo),
                             "hi68": float(hi)}
        mods = {}
        for m in MODELS:
            v, rep = ex[m][0, j], ex[m][1:, j]
            good = rep[np.isfinite(rep)]
            rec: dict = {"value": None if not np.isfinite(v) else float(v), "n_valid": int(good.size)}
            if good.size >= 2 and np.isfinite(v):
                lo, hi = np.percentile(good, [16, 84])
                sd = float(good.std(ddof=1))
                rec.update(err=sd, lo68=float(lo), hi68=float(hi), brackets_ideal=bool(lo <= obs.ideal[j] <= hi),
                           z=None if sd == 0 else float((v - obs.ideal[j]) / sd))
            else:
                rec.update(err=None, lo68=None, hi68=None, brackets_ideal=None, z=None)
            if m == "exponential":
                rec.update(A=None if not np.isfinite(ex["exp_A"][0, j]) else float(ex["exp_A"][0, j]),
                           k=None if not np.isfinite(ex["exp_k"][0, j]) else float(ex["exp_k"][0, j]),
                           B_floor=float(obs.floor[j]))
            mods[m] = rec
        res[name] = {"ideal": float(obs.ideal[j]), "floor": float(obs.floor[j]), "points": pts, "models": mods}
    return {"lambdas": lams, "shots": shots, "boot_n": boot_n, "observables": res}


# ---------------------------------------------------------------------------
# analysis of job folders
# ---------------------------------------------------------------------------
def _load_zne_jobs(job_dirs: Sequence[Path]) -> Tuple[Path, List[Tuple[Path, dict]]]:
    job_dirs = [Path(d) for d in job_dirs]
    if not job_dirs:
        raise ValueError("no job folder to analyse")
    runs = set()
    out = []
    for jd in job_dirs:
        if jd.parent.name != "hw":
            raise ValueError(f"{jd} is not a job folder <run>/hw/<job>")
        runs.add(jd.parents[1])
        man = load_json(jd / "manifest.json")
        if man.get("kind") != KIND:
            raise ValueError(f"{jd} is not a zne job (kind {man.get('kind')!r}); hw-zne writes them")
        if man.get("state") != "done":
            raise ValueError(f"{jd} is in state {man.get('state')!r}, not done: no counts to analyse")
        out.append((jd, man))
    if len(runs) != 1:
        raise ValueError(f"the job folders belong to {len(runs)} different run folders: {sorted(map(str, runs))}")
    return runs.pop(), out


def _qx2_context(run_dir: Path, pubs: List[dict], idx_all: np.ndarray, masks_q: Sequence[float]):
    """(E, p, masks, extra) of a QX2 target: energies of the real objective's QUBO and the ideal distribution of the circuit."""
    from .analysis import ideal_probs
    from .build import load_problem
    pb = pubs[0]
    problem = load_problem(run_dir)
    if pb["fingerprint"] != problem.fingerprint:
        raise ValueError(f"problem fingerprint {pb['fingerprint']} of the job != {problem.fingerprint} of {run_dir}")
    E = problem.run.landscape(problem.qubo).E.astype(np.float64)
    p = ideal_probs(problem.run, problem.ising, int(pb["K"]), pb["gammas"], pb["betas"], problem.scale)
    e_ideal = float(p @ E)
    if abs(e_ideal - float(pb["ideal_mean_energy"])) > IDEAL_MEAN_TOL * max(1.0, abs(e_ideal)):
        raise ValueError(f"ideal <E> {e_ideal} of the statevector != {pb['ideal_mean_energy']} of the manifest: not the circuit "
                         "that was run")
    order = np.argsort(E, kind="stable")
    rank = np.empty(E.size, np.int32)
    rank[order] = np.arange(E.size, dtype=np.int32)
    masks = {f"P_best_{q * 100:g}pct": rank < int(round(q * E.size)) for q in masks_q}
    return E, p, masks, {"n_qubits": problem.n}


def _component_context(run_dir: Path, pubs: List[dict]):
    from . import components as C
    pb = pubs[0]
    comp = C.load_components(run_dir, [pb["component"]])[0]
    if comp.sha1 != pb["component_sha1"]:
        raise ValueError(f"{comp.path} changed since the job was planned")
    E = C.component_energies(comp.doc)
    ising = C.component_ising(comp.doc, check=False)
    scale = float(ising.scale())
    if abs(scale - float(pb["scale"])) > 1e-12 * max(1.0, abs(scale)):
        raise ValueError(f"{comp.name}: Ising scale {scale} != {pb['scale']} of the manifest")
    from .build import sparse_ising
    top_k = pb.get("top_k")
    ism = sparse_ising(ising, top_k)
    diag = (E - ising.offset) / scale if not top_k else None
    p = C.ideal_distribution(ism, scale, float(pb["gammas"][0]), float(pb["betas"][0]), diag)
    e_ideal = float(p @ E)
    if abs(e_ideal - float(pb["ideal_mean_energy"])) > 1e-6 * max(1.0, abs(e_ideal)):
        raise ValueError(f"{comp.name}: ideal <E> {e_ideal} != {pb['ideal_mean_energy']} of the manifest")
    E0 = float(E.min())
    return E, p, {"P_ground": E <= E0 + GROUND_TOL}, {"n_qubits": comp.n, "E0": E0, "group": pb.get("group", "")}


def _readout_mats(cal: dict, man: dict, pubs: List[dict]) -> Tuple[List[np.ndarray], List[int]]:
    """Per-bit read-out matrices of a target (classical-bit order), from a calibration that covers its measured qubits;
    every PUB of the target (all lambdas) must have measured the same qubits."""
    from . import readout as R
    qs = {tuple(int(q) for q in p["measured_qubits"]) for p in pubs}
    if len(qs) != 1:
        raise ValueError(f"target {pubs[0]['target']}: its PUBs measured different qubits {sorted(qs)}")
    qubits = list(qs.pop())
    cal_used = R.calibration_for(cal, qubits, bool(man["setting"]["measure_twirl"]), man.get("backend"), subset=True)
    return R.matrices_of(cal_used)[1], qubits


def analyze_zne(job_dirs: Sequence[Path], out_dir: Path, boot_n: int = BOOT_N, seed: int = BOOT_SEED,
                readout: Optional[Path] = None) -> dict:
    """hw-zne-analyze: extrapolate the observables of every target of the finished zne job folders (one run folder) to
    lambda = 0; writes zne_analysis.json, zne_extrapolation.csv and zne_points.csv into out_dir.  `readout`: a
    readout_cal.json covering the measured qubits; every observable is then read-out-corrected exactly (tensor inverse of
    the per-qubit model) before the extrapolation; read-out damping does not depend on lambda, so without it the
    extrapolated values are biased towards the read-out-damped ones."""
    import pandas as pd
    from .analysis import load_calibration
    run_dir, jobs = _load_zne_jobs(job_dirs)
    cal = None if readout is None else load_calibration(readout)
    man_of = {jd.name: man for jd, man in jobs}
    by_target: Dict[str, List[Tuple[Path, dict]]] = {}
    for jd, man in jobs:
        for p in man["pubs"]:
            if not p.get("arm"):
                by_target.setdefault(p["target"], []).append((jd, p))
    base_of: Dict[str, List[Tuple[Path, dict]]] = {}                  # fold mode: the unfolded PUBs, shared by every arm
    for jd, man in jobs:
        for p in man["pubs"]:
            if p.get("arm") == "base":
                base_of.setdefault(p["target"], []).append((jd, p))
            elif p.get("arm"):
                by_target.setdefault(f"{p['target']}[{p['arm']}]", []).append((jd, p))
    for label in list(by_target):
        if "[" in label:
            by_target[label] = base_of.get(label.split("[")[0], []) + by_target[label]
    targets = {}
    ext_rows, pt_rows = [], []
    for label, items in by_target.items():
        pubs = [p for _, p in items]
        counts: Dict[int, Dict[int, int]] = {}
        n2q: Dict[int, int] = {}
        for jd, p in items:
            c = {int(k): int(v) for k, v in load_json(jd / f"pub_{p['k']}_counts.json")["counts"].items()}
            lk = FD.lam_key(p["lam_eff"] if p.get("arm") else p["lam"])      # fold mode: fit over the achieved noise factor
            tot = counts.setdefault(lk, {})
            for k, v in c.items():
                tot[k] = tot.get(k, 0) + v
            n2q[lk] = int(p["n_2q"])
        idx_all = np.unique(np.concatenate([np.fromiter(c, np.int64) for c in counts.values()]))
        if pubs[0].get("K"):
            E, p_id, masks, extra = _qx2_context(run_dir, pubs, idx_all, TAIL_QS)
            kind = "qx2"
        else:
            E, p_id, masks, extra = _component_context(run_dir, pubs)
            kind = "component"
        mats, ro = None, None
        if cal is not None:
            mats, qubits = _readout_mats(cal, man_of[items[0][0].name], pubs)
            ro = {"calibration": str(readout), "method": "tensor-inverse", "qubits": qubits,
                  "p1_given_0": [float(m[1, 0]) for m in mats], "p0_given_1": [float(m[0, 1]) for m in mats]}
        obs = build_observables(E, p_id, idx_all, masks, readout_mats=mats)
        rec = analyze_counts(obs, counts, boot_n, seed, label)
        rec.update(kind=kind, target=label, n_2q=n2q, **extra, E_unif=float(E.mean()), E_ideal=float(p_id @ E), readout=ro)
        targets[label] = rec
        for name, o in rec["observables"].items():
            for lam, pt in o["points"].items():
                lk = FD.lam_key(float(lam))
                pt_rows.append({"target": label, "observable": name, "lam": lk, "n_2q": n2q[lk], "shots": rec["shots"][lk],
                                "value": pt["value"], "err": pt["err"], "ideal": o["ideal"], "floor": o["floor"]})
            for m, r in o["models"].items():
                ext_rows.append({"target": label, "observable": name, "model": m, "lambdas": ",".join(map(str, rec["lambdas"])),
                                 "value": r["value"], "err": r["err"], "lo68": r["lo68"], "hi68": r["hi68"], "ideal": o["ideal"],
                                 "floor": o["floor"], "brackets_ideal": r["brackets_ideal"], "z": r["z"],
                                 "n_valid": r["n_valid"], "value_lam1": o["points"][str(rec["lambdas"][0])]["value"]})
        log.info("[hw-zne-analyze] %s: lambdas %s", label, rec["lambdas"])
        for name in ("F_E", "F_XEB", "s1"):
            o = rec["observables"][name]
            log.info("[hw-zne-analyze]   %-8s ideal %.3f | points %s | extrapolated %s", name, o["ideal"],
                     " ".join(f"{o['points'][str(l)]['value']:.3f}" for l in rec["lambdas"]),
                     " ".join(f"{m} {r['value']:.3f}" if r["value"] is not None else f"{m} -" for m, r in o["models"].items()))
    res = {"schema": 1, "created": utc_now(), "run_dir": str(run_dir), "jobs": [jd.name for jd, _ in jobs],
           "config": {"models": list(MODELS), "boot_n": boot_n, "boot_seed": seed, "interval": "16/84 %", "kmax": KMAX,
                      "tail_q": list(TAIL_QS), "exponential_B": "uniform value of the observable (fixed)",
                      "errors": "bootstrap over the shots at each lambda; shot noise only",
                      "readout": None if readout is None else {"calibration": str(readout), "method": "tensor-inverse"}},
           "targets": targets}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res["files"] = {"json": str(save_json(out_dir / "zne_analysis.json", res)),
                    "csv": str(save_csv(pd.DataFrame(ext_rows), out_dir / "zne_extrapolation.csv")),
                    "points": str(save_csv(pd.DataFrame(pt_rows), out_dir / "zne_points.csv"))}
    return res

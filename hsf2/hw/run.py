"""One hardware job: plan (transpile, estimate), submit or simulate locally, retrieve, and write the record folder.

A job folder  <run folder>/hw/<DD-MM-YYYY-HH-MM>_<tag>/  holds everything needed to redo the analysis without IBM:
  manifest.json                  settings, options, versions, job id, billed time, the PUB list (angles, gauge, shots)
  pub_<k>_<name>.isa.qpy         the transpiled circuit of each PUB (QPY version 15)
  pub_<k>.layout.json            logical -> physical mapping, measured qubits (classical-bit order), active qubits, edges
  snapshot_pre.json / _post.json calibration of the used qubits and edges before / after the job
  pub_<k>_counts.json            raw counts of each PUB (in the gauge frame) ; gauge_<g>_counts.json for gauged settings
  local_result.json              (local mode) the simulated counts, so that retrieve_job can rebuild the outputs
and the sample sets  <run folder>/samples/ibm-<backend>_<setting>_p1_<topK|full>_<jobid8>_<main|control>.json  in the
Phase-1 format (hw-analyze, hw-validate read them).  Nothing is ever written into a Phase-1 code folder.

Submission is serialised and guarded: budget check against the ledger and IBM's remaining time, the typed confirmation
'submit' for every single job, the job id is written to the ledger and the manifest the moment IBM returns it, and the
billed time is resolved afterwards (job figure -> account difference -> estimate, see resolve_billed).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .._phase1 import ensure_phase1
from ..utils import code_version, load_json, save_json, stamp, unique_dir
from . import fake as F
from .build import (PubSpec, Problem, check_equal_shots, circuit_angles, ideal_mean_energy, load_problem,
                    make_pub_specs, sparse_ising)
from .config import HwConfig, creds_label
from .gauge import gauge_record, gauge_vectors, merge_counts, xor_counts
from .ledger import HwLedger
from .settings import Setting, apply_options, get_setting, options_dump
from .snapshot import calibration_snapshot, layout_files, used_qubits_edges, utc_now
from .transpile import IsaPub, transpile_job

log = logging.getLogger(__name__)

SCHEMA = 1
USAGE_TIMEOUT_S = 120.0
USAGE_INTERVAL_S = 3.0
ACCOUNT_TIMEOUT_S = 90.0
ACCOUNT_INTERVAL_S = 5.0


class SubmitAborted(RuntimeError):
    """The typed confirmation was not 'submit'."""


@dataclass
class JobPlan:
    """Everything decided before a job runs (no QPU time used by building it)."""
    run_dir: Path
    problem: Problem
    setting: Setting
    top_k: Optional[int]
    tag: str
    backend: object
    backend_name: str
    specs: List[PubSpec]
    pubs: List[IsaPub]
    gauges: np.ndarray
    gammas: List[float]
    betas: List[float]
    angle_info: dict
    ideal_energy: float
    estimate_s: float
    rep_delay_s: float
    duration_s: Optional[float]
    mode: str

    @property
    def shots(self) -> int:
        return check_equal_shots(self.specs)

    @property
    def circ(self) -> str:
        return f"top{self.top_k}" if self.top_k else "full"


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def versions() -> dict:
    import sys
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for dist, key in (("qiskit", "qiskit"), ("qiskit-ibm-runtime", "qiskit_ibm_runtime"), ("qiskit-aer", "qiskit_aer"),
                      ("mthree", "mthree")):
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = None
    return out


def get_job_backend(cfg: HwConfig, setting: Setting, service=None):
    """(backend, service): the fake backend for dry / local mode (the real one with cfg.real_target), the real backend
    for submit.  A real backend needs the IBM login; nothing is submitted by getting it."""
    from . import session
    fractional = setting.fractional
    if cfg.mode == "submit" or (cfg.mode == "dry" and cfg.real_target):
        service = service or session.get_service(cfg.creds_file)
        return session.get_backend(service, cfg.backend, fractional), service
    base = F.fake_backend(cfg.fake_backend)
    return (F.fractional_fake(base) if fractional else base), service


def plan_job(run_dir: Path, setting: Setting, top_k: Optional[int], cfg: HwConfig, backend, n_gauges: int = 8,
             gauge_seed: int = 0, tag: Optional[str] = None, problem: Optional[Problem] = None,
             cache: Optional[dict] = None) -> JobPlan:
    """Build, transpile and cost the job of `setting` with the `top_k` strongest couplings (None: all).

    `cache` (a dict kept by hw-ladder) makes the settings of one ladder run the SAME transpiled circuits on the same
    physical qubits, so they are compared on identical circuits even if the device is recalibrated between the jobs."""
    ensure_phase1()
    from acts_qaoa import ibm
    problem = problem or load_problem(run_dir)
    ism = sparse_ising(problem.ising, top_k)
    gammas, betas, info = circuit_angles(problem, ism, top_k)
    ng = int(n_gauges) if setting.gauged else 1
    specs = make_pub_specs(setting.fractional, setting.gauged, ng, cfg.shots, gammas, betas)
    gauges = gauge_vectors(problem.n, ng, gauge_seed)
    name = getattr(backend, "name", None)
    name = name() if callable(name) else str(name)
    key = (top_k, setting.fractional, ng, gauge_seed, name, cfg.transpile_seeds)
    if cache is not None and key in cache:
        pubs = cache[key]
        log.info("[hw] reusing the %d transpiled PUBs of an earlier job of this ladder (same circuits, same qubits)", len(pubs))
    else:
        t0 = time.time()
        pubs = transpile_job(ism, problem.scale, setting.fractional, specs, gauges, backend, cfg.transpile_seeds)
        log.info("[hw] transpiled %d PUBs in %.0f s", len(pubs), time.time() - t0)
        if cache is not None:
            cache[key] = pubs
    rep = ibm.rep_delay_seconds(backend)
    dur = max((p.stats.get("duration_s") or 0.0) for p in pubs) or None
    rnd = 16 if (setting.gate_twirl or setting.measure_twirl) else 1
    est = ibm.estimate_qpu_seconds([s.shots for s in specs], dur, rep, rnd)
    ie = ideal_mean_energy(problem, ism, gammas, betas)
    tag = tag or f"{setting.name}_K{top_k or 'full'}"
    return JobPlan(Path(run_dir), problem, setting, top_k, tag, backend, name, specs, pubs, gauges, gammas, betas, info,
                   ie, est, rep, dur, cfg.mode)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def build_manifest(plan: JobPlan, cfg: HwConfig, step: str, command: str, extra: Optional[dict] = None) -> dict:
    lay = plan.pubs[0].layout
    m = {"schema": SCHEMA, "step": step, "state": "planned", "mode": plan.mode, "created": utc_now(),
         "command": command, "tag": plan.tag, "backend": plan.backend_name, "calibration_id": None,
         "versions": versions(), "code": code_version(), "creds": creds_label(cfg.creds_file),
         "config": cfg.to_dict(), "primitive": cfg.primitive, "setting": plan.setting.to_dict(),
         "sampler_options": None, "job_id": None, "billed": None, "run_dir": str(plan.run_dir),
         "top_k": plan.top_k, "circ": plan.circ, "n_qubits": plan.problem.n,
         "fingerprint": plan.problem.fingerprint, "scale": plan.problem.scale,
         "gammas": plan.gammas, "betas": plan.betas, "angle_info": plan.angle_info,
         "ideal_mean_energy": plan.ideal_energy, "estimate_s": plan.estimate_s, "rep_delay_s": plan.rep_delay_s,
         "circuit_duration_s": plan.duration_s,
         "gauges": [gauge_record(g) for g in plan.gauges],
         "layout": {"measured_qubits": lay["measured"]["qubits"], "initial": lay["initial"]},
         "pubs": [dict(p.spec.to_dict(), n_2q=p.stats.get("n_2q"), fidelity_est=p.stats.get("fidelity_est"),
                       transpile_seed=p.stats.get("transpile_seed")) for p in plan.pubs]}
    if extra:
        m.update(extra)
    return m


def save_manifest(job_dir: Path, manifest: dict) -> Path:
    return save_json(Path(job_dir) / "manifest.json", manifest)


def write_plan_files(job_dir: Path, plan: JobPlan, snap_source: str = "target") -> dict:
    """QPY, layouts and the pre-run calibration snapshot; returns the snapshot."""
    layout_files(job_dir, plan.pubs)
    qs, es = used_qubits_edges([p.layout for p in plan.pubs])
    snap = calibration_snapshot(plan.backend, qs, es, source=snap_source)
    save_json(Path(job_dir) / "snapshot_pre.json", snap)
    return snap


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def make_sampler(backend, primitive: str, setting: Setting, shots: int, max_exec: Optional[int], strict: bool = True):
    """(sampler, applied options): the legacy SamplerV2 (default) or the executor Sampler, with the setting's options."""
    import warnings
    if primitive == "sampler_v2":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from qiskit_ibm_runtime import SamplerV2 as Prim
            sampler = Prim(mode=backend)
    elif primitive == "executor":
        try:
            from qiskit_ibm_runtime.executor_sampler import Sampler as Prim
        except ImportError as exc:
            raise RuntimeError("the executor Sampler (qiskit_ibm_runtime.executor_sampler) is not available in this "
                               f"qiskit-ibm-runtime ({exc}); use --primitive sampler_v2") from exc
        sampler = Prim(mode=backend)
    else:
        raise ValueError(f"unknown primitive {primitive!r}")
    opts = dict(setting.options())
    opts["default_shots"] = int(shots)
    if max_exec:
        opts["max_execution_time"] = int(max_exec)
    applied = apply_options(sampler.options, opts, strict)
    return sampler, applied


def pub_tuples(pubs: Sequence[IsaPub]) -> list:
    return [(p.isa, p.values, int(p.spec.shots)) for p in pubs]


def result_counts(result, n_pubs: int) -> List[Dict[int, int]]:
    """{index: count} of every PUB of a Sampler result (register 'meas', bit k of the index = classical bit k)."""
    from acts_qaoa.circuits import counts_to_index
    if len(result) != n_pubs:
        raise RuntimeError(f"result has {len(result)} PUB results, expected {n_pubs}")
    return [counts_to_index(result[i].data.meas.get_counts()) for i in range(n_pubs)]


def poll_usage(job, timeout_s: Optional[float] = None, interval_s: Optional[float] = None,
               sleep: Optional[Callable[[float], None]] = None, clock: Callable[[], float] = time.monotonic) -> Optional[float]:
    """IBM's per-job QPU seconds.  job.usage() is 0 while the usage status is 'pending' (runtime_job_v2.usage), which is
    indistinguishable from a real 0, so poll job.metrics()['usage'] until its status is not pending, then read
    qpu_charge_time_seconds.  None if it stays pending for `timeout_s` (default 120 s) or the metrics are unavailable."""
    timeout_s = USAGE_TIMEOUT_S if timeout_s is None else timeout_s
    interval_s = USAGE_INTERVAL_S if interval_s is None else interval_s
    sleep = sleep or time.sleep
    t0 = clock()
    while True:
        try:
            u = job.metrics().get("usage", {}) or {}
        except Exception as exc:
            log.warning("[hw] job.metrics() failed (%s): per-job usage unknown", type(exc).__name__)
            return None
        if u.get("status", "pending") != "pending":
            v = u.get("qpu_charge_time_seconds")
            return None if v is None else float(v)
        if clock() - t0 >= timeout_s:
            log.warning("[hw] job usage still pending after %.0f s", timeout_s)
            return None
        sleep(interval_s)


def account_after(service, before: Optional[float], timeout_s: Optional[float] = None, interval_s: Optional[float] = None,
                  sleep: Optional[Callable[[float], None]] = None, clock: Callable[[], float] = time.monotonic) -> Optional[float]:
    """Account usage after a job: IBM's counter lags the job by seconds, so wait until it exceeds `before`."""
    from . import session
    timeout_s = ACCOUNT_TIMEOUT_S if timeout_s is None else timeout_s
    interval_s = ACCOUNT_INTERVAL_S if interval_s is None else interval_s
    sleep = sleep or time.sleep
    t0 = clock()
    while True:
        v = session.account_consumed(service)
        if v is None or before is None or v > before or clock() - t0 >= timeout_s:
            return v
        sleep(interval_s)


def resolve_billed(job_s: Optional[float], before: Optional[float], after: Optional[float], estimate_s: float) -> dict:
    """What to book: IBM's per-job figure if it is positive, else the account-usage difference if positive, else the
    estimate (IBM's per-job figure was 0 s on 4 Nov 2026 for a job that certainly used QPU time)."""
    diff = None if (before is None or after is None) else float(after) - float(before)
    if job_s is not None and job_s > 0:
        used, source = float(job_s), "job"
    elif diff is not None and diff > 0:
        used, source = diff, "account_diff"
    else:
        used, source = float(estimate_s), "estimate"
    return {"job_s": job_s, "account_before_s": before, "account_after_s": after, "account_diff_s": diff,
            "estimate_s": float(estimate_s), "used": used, "source": source}


def confirm_submit(summary: str, input_fn: Callable[[str], str] = input) -> None:
    """Typed confirmation for ONE job (as in qx_hardware.sh); anything but 'submit' aborts."""
    try:
        ans = input_fn(f"{summary}\nType 'submit' to continue: ")
    except EOFError:
        raise SubmitAborted("no confirmation (non-interactive input): aborted") from None
    if ans.strip() != "submit":
        raise SubmitAborted("aborted: confirmation was not 'submit'")


def execute_job(backend, pubs: Sequence[IsaPub], cfg: HwConfig, setting: Setting, plan_name: str, estimate_s: float,
                job_dir: Path, manifest: dict, run_dir: str, service=None, ledger: Optional[HwLedger] = None,
                confirm: Callable[[str], None] = confirm_submit, wait: bool = True, server_dry_run: bool = False,
                seed: int = 0) -> dict:
    """Run the PUBs.  Returns {'job_id', 'counts' (per PUB or None), 'billed', 'job'}; updates and saves `manifest`.

    local: Aer simulation of the fake backend's noise, usage 0, result stored in local_result.json.
    submit: ledger and IBM budget check, confirmation, submission, job id to ledger + manifest at once, then wait for the
    result and book the billed time.  The 'estimate' booked at submission is replaced by the resolved figure."""
    check_equal_shots(pubs)
    job_dir = Path(job_dir)
    shots = int(pubs[0].spec.shots)
    if cfg.mode == "local":
        counts = F.run_local(backend, pubs, seed)
        job_id = F.local_job_id(manifest.get("tag", ""))
        save_json(job_dir / "local_result.json", {"job_id": job_id, "seed": seed, "backend": manifest["backend"],
                                                  "counts": [{str(k): v for k, v in c.items()} for c in counts]})
        manifest.update({"job_id": job_id, "state": "done", "sampler_options": {"note": "recorded only: the local "
                         "simulator ignores twirling and DD"}, "options_requested": setting.options(),
                         "billed": {"job_s": 0.0, "used": 0.0, "source": "local"}})
        save_manifest(job_dir, manifest)
        return {"job_id": job_id, "counts": counts, "billed": manifest["billed"], "job": None}
    if cfg.mode != "submit":
        raise ValueError(f"execute_job runs modes local and submit, not {cfg.mode!r}")
    if service is None:
        raise ValueError("submit needs the IBM service")
    from . import session
    if server_dry_run and cfg.primitive != "executor":
        raise ValueError("--server-dry-run needs --primitive executor")
    ledger = ledger or HwLedger(cfg.ledger_path(), cfg.budget_s, creds_file=cfg.creds_file)
    ledger.check(estimate_s, session.service_remaining(service), cfg.reserve_s)
    max_exec = int(max(30, min(np.ceil(1.5 * estimate_s), max(ledger.remaining_seconds(), 30))))
    sampler, applied = make_sampler(backend, cfg.primitive, setting, shots, max_exec)
    manifest.update({"sampler_options": options_dump(sampler.options), "options_applied": applied,
                     "max_execution_time": max_exec, "server_dry_run": bool(server_dry_run)})
    confirm(f"About to submit ONE {'SERVER-SIDE DRY-RUN ' if server_dry_run else ''}job to {manifest['backend']}: "
            f"{manifest['tag']}, {len(pubs)} PUBs x {shots} shots, setting {setting.name} ({setting.label}),\n"
            f"estimated {estimate_s:.1f} s of QPU time (ledger {ledger.path}: used {ledger.used_seconds():.1f} s, "
            f"remaining {ledger.remaining_seconds():.1f} s of {ledger.budget_s:.0f} s).")
    before = session.account_consumed(service)
    kw = {"dry_run": True} if server_dry_run else {}
    job = sampler.run(pub_tuples(pubs), **kw)
    job_id = job.job_id()
    manifest.update({"job_id": job_id, "state": "submitted", "submitted": utc_now(), "account_before_s": before})
    save_manifest(job_dir, manifest)                       # the id is on disk before anything can go wrong
    if not server_dry_run:
        ledger.add(job_id, manifest["backend"], estimate_s, plan_name, run_dir)
    log.info("[hw] SUBMITTED job %s (max_execution_time %d s); manifest saved", job_id, max_exec)
    if not wait:
        return {"job_id": job_id, "counts": None, "billed": None, "job": job}
    return {"job_id": job_id, "job": job, **collect_job(job, job_dir, manifest, service, ledger, estimate_s, before,
                                                       len(pubs), server_dry_run)}


def collect_job(job, job_dir: Path, manifest: dict, service, ledger: HwLedger, estimate_s: float,
                before: Optional[float], n_pubs: int, server_dry_run: bool = False,
                booked: Optional[dict] = None) -> dict:
    """Wait for `job`, resolve and book the billed time (also for a failed job: it can still consume QPU time), take
    the post-run snapshot.  Returns {'counts', 'billed'}.  `booked`: the job's existing ledger entry (hw-retrieve); a
    figure already booked is only replaced by IBM's own per-job figure, never by an estimate."""
    job_id = job.job_id()
    err = None
    result = None
    try:
        result = job.result()
    except Exception as exc:
        err = exc
        log.error("[hw] job %s did not return a result (%s: %s)", job_id, type(exc).__name__, exc)
    job_s = poll_usage(job)
    after = account_after(service, before) if before is not None else None
    billed = resolve_billed(job_s, before, after, estimate_s)
    booked = booked or {}
    keep = booked.get("quantum_seconds") is not None and billed["source"] != "job"
    if keep:
        billed = dict(billed, used=float(booked["quantum_seconds"]), source=booked.get("usage_source") or "ledger")
        log.info("[hw] job %s: keeping the booked %.2f s (%s)", job_id, billed["used"], billed["source"])
    elif not server_dry_run:
        ledger.book(job_id, billed["used"], billed["source"], manifest["backend"], manifest.get("run_dir", ""))
    manifest.update({"billed": billed, "retrieved": utc_now(), "state": "failed" if err else "done"})
    if err is not None:
        manifest["failed"] = f"{type(err).__name__}: {err}"
        save_manifest(job_dir, manifest)
        raise RuntimeError(f"job {job_id} did not return results ({type(err).__name__}: {err}); booked "
                           f"{billed['used']:.1f} s ({billed['source']})") from err
    save_manifest(job_dir, manifest)
    log.info("[hw] job %s done: billed %.2f s (%s)", job_id, billed["used"], billed["source"])
    return {"counts": None if server_dry_run else result_counts(result, n_pubs), "billed": billed}


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def sample_tag(manifest: dict, kind: str, full_id: bool = False) -> str:
    """12 characters of the job id (IBM ids are time-prefixed: 8 can collide for jobs submitted close together);
    the full id when even that collides with another job's file (export_samples)."""
    jid = str(manifest["job_id"])
    short = jid[len("local_"):] if jid.startswith("local_") else (jid if full_id else jid[:12])
    return f"ibm-{manifest['backend']}_{manifest['setting']['name']}_p1_{manifest['circ']}_{short}_{kind}"


def _foreign(path: Path, job_id: str) -> bool:
    """True if `path` exists and holds the samples of another job."""
    return path.exists() and (load_json(path).get("meta") or {}).get("job_id") != job_id


def write_counts(job_dir: Path, manifest: dict, counts: Sequence[Dict[int, int]]) -> Dict[str, Dict[int, int]]:
    """pub_<k>_counts.json, gauge_<g>_counts.json; returns the merged original-frame counts {'main': ..., 'control': ...}."""
    job_dir = Path(job_dir)
    pubs = manifest["pubs"]
    if len(counts) != len(pubs):
        raise ValueError(f"{len(counts)} count sets for {len(pubs)} PUBs")
    gauges = manifest["gauges"]
    merged: Dict[str, List[Dict[int, int]]] = {"main": [], "control": []}
    per_gauge: Dict[int, dict] = {}
    for p, c in zip(pubs, counts):
        save_json(job_dir / f"pub_{p['k']}_counts.json", {"pub": p["k"], "name": p["name"], "gauge": p["gauge"],
                                                          "shots": p["shots"], "total": int(sum(c.values())),
                                                          "counts": {str(k): int(v) for k, v in c.items()}})
        if sum(c.values()) != p["shots"]:
            raise ValueError(f"PUB {p['k']} returned {sum(c.values())} shots, expected {p['shots']}")
        x = xor_counts(c, gauges[p["gauge"]]["mask"])
        merged[p["kind"]].append(x)
        if p["gauged"]:
            per_gauge.setdefault(p["gauge"], {"gauge": gauges[p["gauge"]]})[p["kind"]] = {
                "gauge_frame": {str(k): int(v) for k, v in c.items()}, "original_frame": {str(k): int(v) for k, v in x.items()}}
    for g, d in per_gauge.items():
        save_json(job_dir / f"gauge_{g}_counts.json", d)
    return {k: merge_counts(v) for k, v in merged.items() if v}


def export_samples(problem: Problem, manifest: dict, merged: Dict[str, Dict[int, int]], billed_s: Optional[float],
                   kinds: Sequence[str] = ("main", "control"), suffix: Optional[Dict[str, str]] = None,
                   meta_extra: Optional[dict] = None) -> List[Path]:
    """The Phase-1 sample sets (`kinds`: main and control) of a job in <run folder>/samples; an existing file is never
    replaced.  `suffix` maps a kind to the end of its file tag (default: the kind; hw-zne uses zne<lambda>); `meta_extra`
    is added to the meta of every set."""
    ensure_phase1()
    from acts_qaoa.landscape import bootstrap_fidelity
    run = problem.run
    land = run.landscape(problem.qubo)
    out = []
    fid = max((p.get("fidelity_est") or 0.0) for p in manifest["pubs"]) or None
    jid = manifest["job_id"]
    suffix = suffix or {}
    full_id = any(_foreign(run.p("samples") / f"{sample_tag(manifest, suffix.get(k, k))}.json", jid) for k in kinds)
    if full_id:
        log.warning("[hw] the short sample tag of job %s is taken by another job: using the full job id", jid)
    for kind in kinds:
        counts = merged[kind]
        tag = sample_tag(manifest, suffix.get(kind, kind), full_id)
        path = run.p("samples") / f"{tag}.json"
        if _foreign(path, jid):
            raise FileExistsError(f"{path} holds the samples of another job; not exporting job {jid}")
        if path.exists():
            log.info("[hw] %s exists, not replaced", path.name)
            out.append(path)
            continue
        m = land.sample_metrics(counts)
        meta = {"kind": "ibm" if manifest["mode"] == "submit" else "local", "job_id": manifest["job_id"],
                "backend": manifest["backend"], "name": kind, "setting": manifest["setting"]["name"],
                "gammas": manifest["gammas"], "betas": manifest["betas"], "shots": int(sum(counts.values())),
                "ansatz": manifest["setting"]["name"], "sparsify_top_k": manifest["top_k"],
                "ideal_mean_energy": manifest["ideal_mean_energy"], "fidelity_est": fid,
                "quantum_seconds_job": billed_s, "n_gauges": len(manifest["gauges"]) if manifest["setting"]["gauged"] else 1,
                "hw_job": manifest["tag"], "primitive": manifest["primitive"], **(meta_extra or {})}
        if kind == "main":
            fb = bootstrap_fidelity(counts, land, manifest["ideal_mean_energy"])
            m.update({"F_measured": fb["F"], "F_lo68": fb["F_lo68"], "F_hi68": fb["F_hi68"], "F_est": fid})
        out.append(run.save_samples(tag, counts, meta, m))
        log.info("[hw] sample set %s: <E>=%.5f  r=%.3f%s", tag, m["mean_energy"], m["approx_ratio"],
                 f"  F_E {m['F_measured']:.3f}" if kind == "main" else "")
    return out


def finish_job(job_dir: Path, manifest: dict, counts: Sequence[Dict[int, int]], problem: Optional[Problem] = None,
               job=None, backend=None) -> dict:
    """Counts files, sample export, post-run snapshot and the final manifest (also used by retrieve_job)."""
    job_dir = Path(job_dir)
    merged = write_counts(job_dir, manifest, counts)
    problem = problem or load_problem(Path(manifest["run_dir"]))
    billed = (manifest.get("billed") or {}).get("used")
    samples = export_samples(problem, manifest, merged, billed)
    manifest["samples"] = [p.name for p in samples]
    if backend is not None:
        lays = [load_json(p) for p in sorted(job_dir.glob("pub_*.layout.json"))]
        qs, es = used_qubits_edges(lays)
        props = None
        if job is not None:
            try:
                props = job.properties()
            except Exception as exc:
                log.warning("[hw] job.properties() failed (%s): post snapshot uses the backend's current properties",
                            type(exc).__name__)
        snap = calibration_snapshot(backend, qs, es, source="job.properties" if props is not None else "backend",
                                    properties=props, refresh=props is None)
        save_json(job_dir / "snapshot_post.json", snap)
        manifest["calibration_id"] = snap.get("calibration_id") or manifest.get("calibration_id")
    manifest["state"] = "done"
    save_manifest(job_dir, manifest)
    return merged


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def new_job_dir(run_dir: Path, tag: str) -> Path:
    return unique_dir(Path(run_dir) / "hw", f"{stamp()}_{tag}")


def run_job(run_dir: Path, setting_name: str, top_k: Optional[int], cfg: HwConfig, step: str = "step2",
            n_gauges: int = 8, command: str = "", tag: Optional[str] = None, problem: Optional[Problem] = None,
            service=None, confirm: Callable[[str], None] = confirm_submit, wait: bool = True,
            server_dry_run: bool = False, seed: int = 0, gauge_seed: int = 0, cache: Optional[dict] = None) -> dict:
    """hw-run: plan, then dry (plan files only), local (simulate, export) or submit."""
    cfg.validate()
    setting = get_setting(setting_name)
    if cfg.primitive == "executor" and setting.fractional:
        log.warning("[hw] fractional gates with the executor Sampler have not been checked (risk in the design note)")
    problem = problem or load_problem(run_dir)
    backend, service = get_job_backend(cfg, setting, service)
    plan = plan_job(run_dir, setting, top_k, cfg, backend, n_gauges, gauge_seed, tag, problem, cache)
    job_dir = new_job_dir(run_dir, plan.tag)
    manifest = build_manifest(plan, cfg, step, command)
    write_plan_files(job_dir, plan)
    log.info("[hw] job folder %s | %s K=%s | %d PUBs x %d shots | estimated %.1f s of QPU time", job_dir, setting.name,
             top_k or "full", len(plan.pubs), plan.shots, plan.estimate_s)
    if cfg.mode == "dry":
        manifest["state"] = "dry"
        save_manifest(job_dir, manifest)
        return {"job_dir": job_dir, "manifest": manifest, "plan": plan, "service": service}
    ex =execute_job(backend, plan.pubs, cfg, setting, f"hw-{setting.name}", plan.estimate_s, job_dir, manifest,
                     str(run_dir), service, None, confirm, wait, server_dry_run, seed)
    if ex["counts"] is not None:
        finish_job(job_dir, manifest, ex["counts"], problem, ex.get("job"), backend)
    return {"job_dir": job_dir, "manifest": manifest, "plan": plan, "service": service, **ex}


def run_ladder(run_dir: Path, ks: Sequence[Optional[int]], settings: Sequence[str], cfg: HwConfig, n_gauges: int = 8,
               step: str = "step2", command: str = "", out_dir: Optional[Path] = None, readout: bool = False,
               confirm: Callable[[str], None] = confirm_submit, **kw) -> List[dict]:
    """hw-ladder: for each K the settings in the given order (a b c d e f d : d is repeated at the end to expose drift),
    one job each, strictly one after the other (one confirmation per job; a refused or failed job stops the ladder
    with the finished ones on disk)."""
    cfg.validate()
    for s in settings:
        get_setting(s)
    problem = load_problem(run_dir)
    service = None
    rows = []
    cache: dict = {}
    for k in ks:
        for i, s in enumerate(settings):
            tag = f"{s}_K{k or 'full'}" + ("" if settings.count(s) == 1 else f"_{settings[:i + 1].count(s)}")
            res = run_job(run_dir, s, k, cfg, step, n_gauges, command, tag, problem, service, confirm, True, cache=cache, **kw)
            service = res.get("service")
            row = {"K": k, "setting": s, "tag": tag, "job_dir": str(res["job_dir"]),
                   "job_id": res["manifest"].get("job_id"), "state": res["manifest"]["state"],
                   "billed_s": (res["manifest"].get("billed") or {}).get("used"),
                   "estimate_s": res["plan"].estimate_s}
            if readout and cfg.mode != "dry":
                row["readout_dir"] = str(run_readout(run_dir, res["job_dir"], cfg, step, command, confirm=confirm,
                                                     service=service)["job_dir"])
            rows.append(row)
            if out_dir is not None:
                save_json(Path(out_dir) / "ladder_summary.json", {"rows": rows})
    return rows


def find_job_dir(job_id: str, run_dir: Optional[Path] = None, search_root: Optional[Path] = None) -> Path:
    """The job folder whose manifest has `job_id` (under <run_dir>/hw, or anywhere under search_root)."""
    pats = [Path(run_dir) / "hw"] if run_dir else [Path(search_root)] if search_root else []
    if not pats:
        raise ValueError("give --run-dir (or a search root) to locate the job folder")
    hits = []
    for root in pats:
        glob = root.glob("*/manifest.json") if run_dir else root.glob("**/hw/*/manifest.json")
        for mf in glob:
            try:
                if load_json(mf).get("job_id") == job_id:
                    hits.append(mf.parent)
            except Exception:
                continue
    if not hits:
        raise FileNotFoundError(f"no job folder with job id {job_id} under {pats[0]}")
    return sorted(hits)[-1]


def retrieve_ledger(cfg: HwConfig, manifest: dict) -> HwLedger:
    """The ledger a job was booked in when it was submitted (recorded in its manifest), unless --ledger names another
    one; a recorded path whose folder does not exist here (job folder copied to another machine) falls back to the
    configured ledger."""
    path = Path(cfg.ledger).expanduser() if cfg.ledger else None
    if path is None:
        rec = (manifest.get("config") or {}).get("ledger")
        path = Path(rec) if rec and Path(rec).parent.is_dir() else cfg.ledger_path()
    return HwLedger(path, cfg.budget_s, creds_file=cfg.creds_file)


def retrieve_job(job_id: str, cfg: HwConfig, run_dir: Optional[Path] = None, search_root: Optional[Path] = None,
                 service=None) -> dict:
    """Rebuild the outputs of a job whose shell was lost: the manifest names the run folder and PUBs; the counts come from
    local_result.json (local jobs) or from IBM (service.job(job_id).result()).  Billed time is resolved and booked for
    real jobs.  Sample sets that already exist are kept."""
    job_dir = find_job_dir(job_id, run_dir, search_root)
    manifest = load_json(job_dir / "manifest.json")
    run_root = job_dir.parents[1]                  # <run folder>/hw/<job folder>: the folder may have been moved or copied
    manifest["run_dir"] = str(run_root)
    problem = load_problem(run_root)
    backend = None
    job = None
    if job_id.startswith(F.LOCAL_PREFIX):
        stored = load_json(job_dir / "local_result.json")
        counts = [{int(k): int(v) for k, v in c.items()} for c in stored["counts"]]
        backend = F.fake_backend(cfg.fake_backend)
        if manifest["setting"]["fractional"]:
            backend = F.fractional_fake(backend)
    else:
        from . import session
        service = service or session.get_service(cfg.creds_file)
        job = service.job(job_id)
        ledger = retrieve_ledger(cfg, manifest)
        entry = ledger.entry(job_id) or {}
        est = entry.get("estimate_s", manifest.get("estimate_s", 0.0))
        got = collect_job(job, job_dir, manifest, service, ledger, est, None, len(manifest["pubs"]),
                          bool(manifest.get("server_dry_run")), booked=entry)
        counts = got["counts"]
        backend = job.backend() if hasattr(job, "backend") else None
    merged = finish_job(job_dir, manifest, counts, problem, job, backend) if counts is not None else None
    return {"job_dir": job_dir, "manifest": manifest, "merged": merged}


# ---------------------------------------------------------------------------
# readout calibration and mitigation
# ---------------------------------------------------------------------------
def run_readout(run_dir: Path, ref_job_dir: Path, cfg: HwConfig, step: str = "step2", command: str = "",
                shots: Optional[int] = None, confirm: Callable[[str], None] = confirm_submit, service=None,
                seed: int = 0, wait: bool = True) -> dict:
    """hw-readout: calibrate the readout of the qubits that the job in `ref_job_dir` measured, with the measurement-
    twirling switch (and the other options) of that job's setting.  |0...0> and |1...1> with `shots` shots each (default:
    the job's shots per PUB); writes readout_cal.json (per-qubit 2x2 matrices, session id) into a new job folder.
    `wait` False (submit): return after the submission; hw-retrieve <job id> writes readout_cal.json later.  The reference
    job only has to be planned (its manifest names the measured qubits), not finished."""
    from . import readout as R
    from .snapshot import write_qpy
    ensure_phase1()
    from acts_qaoa import ibm
    cfg.validate()
    ref = load_json(Path(ref_job_dir) / "manifest.json")
    setting = get_setting(ref["setting"]["name"])
    if "layout" in ref:
        qubits = [int(q) for q in ref["layout"]["measured_qubits"]]
    else:                      # components / zne jobs: one circuit, hence one qubit list, per PUB; calibrate their union
        qubits = sorted({int(q) for p in ref["pubs"] for q in p["measured_qubits"]})
    backend, service = get_job_backend(cfg, setting, service)
    name = getattr(backend, "name", None)
    name = name() if callable(name) else str(name)
    shots = int(shots or ref["pubs"][0]["shots"])
    circs = R.calibration_circuits(qubits, int(backend.num_qubits))
    lay = {"n_logical": len(qubits), "initial": None, "final": None, "active_qubits": qubits, "edges": [],
           "measured": {"clbits": list(range(len(qubits))), "qubits": qubits}}
    pubs = [IsaPub(PubSpec(k, f"cal{k}", 0, [], [], shots), c, None, {"active_qubits": qubits, "duration_s": None}, lay)
            for k, c in enumerate(circs)]
    est = ibm.estimate_qpu_seconds([shots, shots], None, ibm.rep_delay_seconds(backend),
                                   16 if setting.measure_twirl else 1)
    tag = f"readout_{'tw' if setting.measure_twirl else 'bare'}_{ref['tag']}"
    job_dir = new_job_dir(run_dir, tag)
    session_id = f"{stamp()}_{ref['tag']}"
    manifest = {"schema": SCHEMA, "step": step, "kind": READOUT_KIND, "state": "planned", "mode": cfg.mode,
                "created": utc_now(), "command": command, "tag": tag, "backend": name, "versions": versions(),
                "code": code_version(), "creds": creds_label(cfg.creds_file), "config": cfg.to_dict(),
                "primitive": cfg.primitive, "setting": setting.to_dict(), "reference_job": str(ref_job_dir),
                "qubits": qubits, "session": session_id, "estimate_s": est, "run_dir": str(run_dir),
                "pubs": [p.spec.to_dict() for p in pubs]}
    for p in pubs:
        write_qpy(job_dir / f"pub_{p.spec.k}_{p.spec.name}.isa.qpy", p.isa)
    log.info("[hw-readout] %d qubits, 2 x %d shots, measurement twirl %s, estimated %.1f s", len(qubits), shots,
             setting.measure_twirl, est)
    if cfg.mode == "dry":
        manifest["state"] = "dry"
        save_manifest(job_dir, manifest)
        return {"job_dir": job_dir, "manifest": manifest, "service": service}
    ex = execute_job(backend, pubs, cfg, setting, f"hw-readout-{setting.name}", est, job_dir, manifest, str(run_dir),
                     service, None, confirm, wait, False, seed)
    if ex["counts"] is None:                       # submitted without waiting: state 'submitted', no readout_cal.json yet
        return {"job_dir": job_dir, "manifest": manifest, "cal": None, "service": service}
    cal = finish_readout(job_dir, manifest, ex["counts"])
    return {"job_dir": job_dir, "manifest": manifest, "cal": cal, "service": service}


READOUT_KIND = "readout_calibration"


def finish_readout(job_dir: Path, manifest: dict, counts) -> dict:
    """readout_cal.json and the two counts files of a finished read-out calibration job (hw-readout, hw-retrieve)."""
    from . import readout as R
    c0, c1 = counts
    setting = get_setting(manifest["setting"]["name"])
    shots = int(manifest["pubs"][0]["shots"])
    cal = R.calibration_record([int(q) for q in manifest["qubits"]], c0, c1, shots, setting.measure_twirl,
                               manifest["session"], manifest["backend"], manifest["job_id"])
    save_json(job_dir / "readout_cal.json", cal)
    for k, c in enumerate((c0, c1)):
        save_json(job_dir / f"pub_{k}_counts.json", {"pub": k, "name": f"cal{k}", "shots": shots,
                                                      "counts": {str(i): v for i, v in c.items()}})
    manifest["state"] = "done"
    save_manifest(job_dir, manifest)
    return cal


def retrieve_readout_job(job_id: str, cfg: HwConfig, run_dir: Optional[Path] = None, search_root: Optional[Path] = None,
                         service=None) -> dict:
    """Rebuild a read-out calibration job whose shell was lost or that was submitted without waiting: counts from
    local_result.json or from IBM (waits while the job is queued), billed time resolved and booked, readout_cal.json."""
    job_dir = find_job_dir(job_id, run_dir, search_root)
    manifest = load_json(job_dir / "manifest.json")
    if manifest.get("kind") != READOUT_KIND:
        raise ValueError(f"{job_dir} is not a read-out calibration job (kind {manifest.get('kind')!r})")
    manifest["run_dir"] = str(job_dir.parents[1])
    if job_id.startswith(F.LOCAL_PREFIX):
        stored = load_json(job_dir / "local_result.json")
        counts = [{int(k): int(v) for k, v in c.items()} for c in stored["counts"]]
    else:
        from . import session
        service = service or session.get_service(cfg.creds_file)
        job = service.job(job_id)
        ledger = retrieve_ledger(cfg, manifest)
        entry = ledger.entry(job_id) or {}
        est = entry.get("estimate_s", manifest.get("estimate_s", 0.0))
        counts = collect_job(job, job_dir, manifest, service, ledger, est, None, len(manifest["pubs"]), False,
                             booked=entry)["counts"]
    cal = finish_readout(job_dir, manifest, counts)
    return {"job_dir": job_dir, "manifest": manifest, "cal": cal}


def raw_counts(job_dir: Path, man: dict) -> Dict[str, Dict[int, int]]:
    """Merged original-frame raw counts of a finished job from its pub_<k>_counts.json."""
    parts: Dict[str, List[Dict[int, int]]] = {"main": [], "control": []}
    for p in man["pubs"]:
        c = {int(k): int(v) for k, v in load_json(Path(job_dir) / f"pub_{p['k']}_counts.json")["counts"].items()}
        parts[p["kind"]].append(xor_counts(c, man["gauges"][p["gauge"]]["mask"]))
    return {k: merge_counts(v) for k, v in parts.items()}


def mitigate_job(job_dir: Path, man: dict, cal: dict, prefer: str = "auto", kinds: Sequence[str] = ("main", "control")
                 ) -> tuple:
    """Readout-mitigated probabilities of the PUBs of a finished job (merged, original frame) with the calibration record
    `cal` (mthree, else the numpy solve).  The calibration must have the job's measurement-twirling switch and measured
    qubits (readout.calibration_for); the correction is applied in each PUB's gauge frame, then XOR'ed back.  Returns
    ({kind: {index: probability}}, [overhead per PUB], backend used)."""
    from . import readout as R
    job_dir = Path(job_dir)
    mq = [int(q) for q in man["layout"]["measured_qubits"]]
    cal = R.calibration_for(cal, mq, man["setting"]["measure_twirl"])
    n_device = max(mq) + 1
    parts: Dict[str, List[Dict[int, float]]] = {k: [] for k in kinds}
    overhead, used = [], None
    for p in man["pubs"]:
        if p["kind"] not in parts:
            continue
        counts = {int(k): int(v) for k, v in load_json(job_dir / f"pub_{p['k']}_counts.json")["counts"].items()}
        pr, ov, used = R.correct_counts(counts, cal, n_device, prefer=prefer)
        mask = man["gauges"][p["gauge"]]["mask"]
        parts[p["kind"]].append({k ^ mask: v for k, v in pr.items()})
        overhead.append(ov)
    merged: Dict[str, Dict[int, float]] = {}
    for kind, ds in parts.items():
        tot: Dict[int, float] = {}
        for d in ds:
            for k, v in d.items():
                tot[k] = tot.get(k, 0.0) + v / len(ds)
        merged[kind] = tot
    return merged, overhead, used


def apply_readout(job_dir: Path, cal_path: Path, prefer: str = "auto", out_name: str = "readout_mitigated.json",
                  with_energy: bool = True) -> dict:
    """Readout-mitigate the counts of a finished job with a calibration (mthree, else the numpy solve).  The calibration
    must have the job's measurement-twirling switch and measured qubits; the correction is applied in each PUB's gauge
    frame, then XOR'ed back.  Writes <job_dir>/<out_name>: probabilities of main and control and the mean energies."""
    from . import readout as R
    job_dir = Path(job_dir)
    man = load_json(job_dir / "manifest.json")
    cal = load_json(cal_path)
    merged, overhead, used = mitigate_job(job_dir, man, cal, prefer)
    res = {"mitigation": used, "mthree": R.mthree_status()[0], "overhead": overhead, "calibration": str(cal_path),
           "calibration_session": cal.get("session"),
           "probs": {k: {str(i): v for i, v in d.items()} for k, d in merged.items()}}
    if with_energy:
        problem = load_problem(job_dir.parents[1])
        E = problem.run.landscape(problem.qubo).E
        raw = raw_counts(job_dir, man)
        rawp = {k: {i: c / sum(d.values()) for i, c in d.items()} for k, d in raw.items()}
        res["mean_energy"] = {f"{k}_{s}": R.mean_energy(d, E) for s, src in (("mitigated", merged), ("raw", rawp))
                              for k, d in src.items()}
    save_json(job_dir / out_name, res)
    return res

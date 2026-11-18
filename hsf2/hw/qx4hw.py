"""Step 5: QX4 on hardware -- the 1-bit filter of TrackHHL on the smallest toy events (hw-qx4, hw-qx4-analyze).

Circuits (hsf2.qubo.linear).  The filter is H on the system qubits (|b> = uniform) and on the ancilla, e^{iAt} controlled by the
ancilla (one Lie-Trotter step of the Pauli decomposition of A, padded with alpha + beta to 2^q states), and H on the ancilla;
t = pi / (alpha + beta), so that a segment without an aligned neighbour (A-eigenvalue alpha + beta) has cos(lambda t / 2) = 0.
Post-selecting the ancilla on 0 therefore removes the isolated segments: the ideal conditional distribution over the segments
is exactly 0 on them.  Event: `n_particles x n_layers` straight tracks of hsf2.qubo.linear.toy_velo, seed 0 by default; 2 x 3 is
N = 8 segments on 4 qubits (3 Pauli terms), 2 x 5 is N = 16 on 5 qubits.  An optional single PUB runs the textbook HHL of
hsf2.qubo.linear.hhl_circuit with n_clock = 2 on the 1 x 3 event (N = 2): a fidelity check of the circuit's post-selection
probability P(ancilla = 1, clock = 0) against the QPE emulation (hsf2.qubo.linear.eigen_amplitudes); it has no tracking content.

Classical bits.  One register 'meas'; classical bit k holds qubit k: bits 0..q-1 are the system index s (segment s), then, for
the HHL, the clock bits, and the ancilla is the LAST bit.  A sample index is s + 2^q * ancilla (filter).

Observable.  Leakage L = P(s in rejected | post-selected) with `rejected` = segments without an aligned neighbour (and the
padded basis states); ideal 0.  Uniform control 1 - K / 2^q (K kept segments), white noise of fidelity F closed form
(white_noise_leakage).  L is a ratio, so zero-noise extrapolation (local folding, hsf2.hw.zne, lambda = 1, 3, 5) is applied to
the numerator P(post, s in kept) and to the denominator P(post) separately, and L = 1 - num / den is formed per bootstrap
replicate; L itself is never extrapolated.  Secondary: the segments kept by the filter (conditional frequency >= 1 / N) against
the classical solve (A S = b, S >= T) and the truth, as efficiency and purity, with the uniform control; total variation and
Hellinger distance to the ideal conditional distribution.  Read-out: the exact tensor inverse of the per-qubit model
(hsf2.hw.readout.tensor_apply) of the measured qubits.  Errors: bootstrap over the shots (16 / 84 %, shot noise only).

Job folder  <run folder>/hw/<DD-MM-YYYY-HH-MM>_qx4_<setting>_1/  (the files of an hw-zne job: the circuits are folded by hsf2.hw.zne)
  manifest.json  kind "qx4", step "step5"; per PUB: circuit, event, lambda, bit layout, ideal values, 2q counts, layout
  pub_<k>_<circuit>_lam<l>.isa.qpy, pub_<k>.layout.json, snapshot_pre / _post.json, pub_<k>_counts.json, local_result.json, and,
  after it, the read-out calibration job <run folder>/hw/<stamp>_readout_*/ (hw-readout pattern, own confirmation).
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._phase1 import ensure_phase1
from ..utils import load_json, save_csv, save_json
from . import run as R
from . import zne as Z
from .config import HwConfig
from .ledger import HwLedger
from .settings import Setting, get_setting
from .snapshot import utc_now
from .transpile import IsaPub

log = logging.getLogger(__name__)

KIND = "qx4"
STEP = "step5"
DEFAULT_EVENTS = "2x3,2x5"
DEFAULT_LAMS = (1, 3, 5)
OTHER_LAMS = (1,)
HHL_EVENT = (1, 3)                     # N = 2: the smallest system the HHL circuit is run on
HHL_CLOCK = 2
TRANSPILE_SEEDS = 6
MAX_PUBS = Z.MAX_PUBS_PER_JOB
BOOT_N = 1000
BOOT_SEED = 0
IDEAL_TOL = 1e-9                       # ideal post-selection probability of a rebuilt circuit vs the manifest
KEPT_FLOOR = 1e-12                     # a conditional probability below this is zero in the ideal distribution


# ---------------------------------------------------------------------------
# circuits
# ---------------------------------------------------------------------------
@dataclass
class Qx4Circuit:
    """One QX4 circuit with everything needed to read its counts (classical bit k = qubit k, ancilla last)."""
    label: str                      # filter<N> or hhl<N>
    circuit: str                    # "filter" | "hhl"
    n_particles: int
    n_layers: int
    seed: int
    alpha: float
    beta: float
    eps: float
    T: float
    n_clock: int
    N: int                          # segments
    q: int                          # system qubits (bits 0..q-1)
    n_bits: int
    t: float
    ab: float
    n_terms: int
    A: object
    b: np.ndarray
    C: object
    truth: np.ndarray               # (N,) same particle
    classical: np.ndarray           # (N,) classical solution S >= T
    kept: np.ndarray                # (2^q,) segment with an aligned neighbour (not rejected by the ideal filter)
    qc: object                      # no measurement
    qc_meas: object                 # measurements into the register 'meas', classical bit k <- qubit k
    anc_bit: int
    anc_value: int
    clock_bits: List[int] = field(default_factory=list)

    @property
    def post_offset(self) -> int:
        """Index of the post-selected state with system index 0 (clock 0, ancilla at its post-selected value)."""
        return self.anc_value << self.anc_bit

    @property
    def post_mask(self) -> np.ndarray:
        """(2^n_bits,) bool: ancilla (and clock) at the post-selected values."""
        idx = np.arange(1 << self.n_bits)
        m = ((idx >> self.anc_bit) & 1) == self.anc_value
        for c in self.clock_bits:
            m &= ((idx >> c) & 1) == 0
        return m

    @property
    def sys_of_index(self) -> np.ndarray:
        return np.arange(1 << self.n_bits) & ((1 << self.q) - 1)

    @property
    def kept_post_mask(self) -> np.ndarray:
        return self.post_mask & self.kept[self.sys_of_index]

    def meta(self) -> dict:
        """The fields of a manifest PUB record that describe the circuit (JSON types)."""
        return {"circuit": self.circuit, "label": self.label, "n_particles": self.n_particles, "n_layers": self.n_layers,
                "event_seed": self.seed, "alpha": self.alpha, "beta": self.beta, "eps": self.eps, "T": self.T,
                "n_clock": self.n_clock, "n_segments": self.N, "n_system_qubits": self.q, "n_vars": self.n_bits,
                "n_pauli_terms": self.n_terms, "t": self.t, "ab": self.ab, "anc_bit": self.anc_bit,
                "anc_value": self.anc_value, "clock_bits": list(self.clock_bits), "n_kept": int(self.kept.sum()),
                "n_rejected": int((~self.kept).sum()), "A_sha1": hashlib.sha1(
                    np.ascontiguousarray(self.A.toarray(), np.float64).tobytes()).hexdigest()}


def parse_events(text: str) -> List[Tuple[int, int]]:
    """'2x3,2x5' -> [(2, 3), (2, 5)]; the first one is the headline event."""
    out = []
    for item in str(text).split(","):
        m = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", item)
        if not m or int(m.group(1)) < 1 or int(m.group(2)) < 3:
            raise ValueError(f"event {item!r}: expected <particles>x<layers> with >= 1 particle and >= 3 layers (2x3,2x5)")
        out.append((int(m.group(1)), int(m.group(2))))
    if not out or len(set(out)) != len(out):
        raise ValueError(f"need distinct events, got {text!r}")
    return out


def as_lam(x):
    """A noise factor as int if it is integral (1, 3, 5), else float (1.5); must be >= 1 (a factor below 1 is not a folding)."""
    v = float(x)
    if not math.isfinite(v) or v < 1.0:
        raise ValueError(f"the noise factor must be a number >= 1, got {x!r}")
    return int(v) if v == int(v) else v


def parse_lams(text) -> list:
    """'1,3,5' -> [1, 3, 5]; '1.5,2,3' -> [1.5, 2, 3] (or a sequence of numbers).  Which factors can be folded is decided by the
    folding function (the default, zne.fold_two_qubit_gates, takes odd integers only and refuses the others at planning)."""
    items = [x for x in str(text).split(",") if x.strip()] if isinstance(text, str) else list(text)
    lams = sorted(as_lam(x) for x in items)
    if not lams or len(set(lams)) != len(lams):
        raise ValueError(f"need distinct noise factors, got {text!r}")
    return lams


def build_filter(n_particles: int, n_layers: int, seed: int = 0, alpha: float = 0.25, beta: float = 0.5, eps: float = 1e-4,
                 T: float = 0.8, circuit: str = "filter", n_clock: int = HHL_CLOCK) -> Qx4Circuit:
    """The QX4 circuit of the toy event (hsf2.qubo.linear.toy_velo, no noise, no smearing): the 1-bit filter or the HHL with
    `n_clock` clock qubits, in its measured form (classical bit k <- qubit k, ancilla last)."""
    from qiskit import ClassicalRegister
    from ..qubo import linear as L
    if circuit not in ("filter", "hhl"):
        raise ValueError(f"circuit must be 'filter' or 'hhl', got {circuit!r}")
    ev = L.toy_velo(int(n_particles), int(n_layers), seed=int(seed))
    seg = L.segments(ev)
    A, b, C, truth = L.build_system(ev, seg, alpha, beta, eps)
    N = len(seg)
    ab = alpha + beta
    t = math.pi / ab
    S = L.classical_solution(A, b)
    if circuit == "filter":
        qc, info = L.onebit_circuit(A, t, ab)
        anc_bit, anc_value, clocks, nc = info["system_qubits"], 0, [], 0
    else:
        if n_clock < 1:
            raise ValueError("the HHL needs at least one clock qubit")
        qc, info = L.hhl_circuit(A, t, ab, int(n_clock))
        q = info["system_qubits"]
        clocks, anc_bit, anc_value, nc = list(range(q, q + int(n_clock))), q + int(n_clock), 1, int(n_clock)
    q = info["system_qubits"]
    n_bits = info["qubits"]
    if qc.num_qubits != n_bits or anc_bit != n_bits - 1:
        raise RuntimeError("the ancilla must be the last qubit and the last classical bit")
    deg = np.asarray(C.sum(axis=1)).ravel()
    kept = np.zeros(1 << q, bool)
    kept[:N] = deg > 0
    meas = qc.copy()
    meas.add_register(ClassicalRegister(n_bits, "meas"))
    meas.measure(list(range(n_bits)), list(range(n_bits)))
    return Qx4Circuit(f"{circuit}{N}", circuit, int(n_particles), int(n_layers), int(seed), float(alpha), float(beta),
                      float(eps), float(T), nc, N, q, n_bits, float(t), float(ab), int(info["n_pauli_terms"]), A, b, C, truth,
                      S >= T, kept, qc, meas, anc_bit, anc_value, clocks)


def ideal_cond(circ: Qx4Circuit) -> dict:
    """Statevector of the (Trotterised) circuit: the probabilities over all 2^n_bits sample indices, P(post), the conditional
    distribution over the 2^q system states, P(post, kept) and the ideal leakage."""
    from qiskit.quantum_info import Statevector
    probs = np.asarray(Statevector(circ.qc).probabilities(), np.float64)
    if probs.size != 1 << circ.n_bits:
        raise RuntimeError(f"statevector of {probs.size} entries for {circ.n_bits} bits")
    post = probs[circ.post_offset + np.arange(1 << circ.q)]
    p_post = float(post.sum())
    if not p_post > 0:
        raise ValueError(f"{circ.label}: the ideal post-selection probability is 0")
    cond = post / p_post
    p_kept = float(post[circ.kept].sum())
    return {"probs": probs, "p_post": p_post, "cond": cond, "p_kept_post": p_kept, "leakage": 1.0 - p_kept / p_post}


def eigen_reference(circ: Qx4Circuit) -> dict:
    """The exact-evolution (no Trotter error) emulation of hsf2.qubo.linear.eigen_amplitudes for this circuit: for the filter
    the conditional distribution over the segments and P(post); for the HHL P(post) = Cmin^2 x the unnormalised success weight
    (Cmin = 2 pi / (2^n_clock t), the smallest clock value; the QPE kernel and 1 / lambda_k of the circuit's rotation)."""
    from ..qubo import linear as L
    if circ.circuit == "filter":
        amp, info = L.eigen_amplitudes(circ.A, circ.b, "onebit", t=circ.t, alpha_beta=circ.ab)
        return {"cond": np.abs(amp) ** 2, "p_post": float(info["postselection_probability"])}
    amp, info = L.eigen_amplitudes(circ.A, circ.b, "hhl-qpe", t=circ.t, n_clock=circ.n_clock)
    cmin = 2 * math.pi / ((1 << circ.n_clock) * circ.t)
    return {"cond": np.abs(amp) ** 2, "p_post": float(cmin ** 2 * info["success_weight"])}


def white_noise_leakage(circ: Qx4Circuit, F: float, ideal: Optional[dict] = None) -> float:
    """Closed form of L for counts  F p_ideal + (1 - F) / 2^n:  with w = (1 - F) / 2^n, n_post the post-selected sample indices
    and n_rej those of them in rejected segments,  L = [F (P0 - P0_kept) + w n_rej] / [F P0 + w n_post]  (0.29 at F = 0.74 for
    N = 8, where P0 = 1/8, n_post = 8, n_rej = 4)."""
    if not 0.0 <= F <= 1.0:
        raise ValueError(f"fidelity must lie in [0, 1], got {F}")
    ideal = ideal or ideal_cond(circ)
    w = (1.0 - F) / (1 << circ.n_bits)
    n_post = int(circ.post_mask.sum())
    n_rej = n_post - int(circ.kept_post_mask.sum())
    den = F * ideal["p_post"] + w * n_post
    return float((F * (ideal["p_post"] - ideal["p_kept_post"]) + w * n_rej) / den)


def uniform_leakage(circ: Qx4Circuit) -> float:
    """L of the uniform distribution over all sample indices: the rejected fraction of the 2^q system states."""
    return 1.0 - float(circ.kept.sum()) / (1 << circ.q)


# ---------------------------------------------------------------------------
# transpilation and planning
# ---------------------------------------------------------------------------
def transpile_filter(circ: Qx4Circuit, backend, n_seeds: int = TRANSPILE_SEEDS, k: int = 0, shots: int = 4000) -> IsaPub:
    """The circuit on `backend`: Phase-1 transpile_best (optimisation level 3, best estimated fidelity of `n_seeds` seeds).  The
    PUB carries no parameters; its spec is a lambda = 1 ZnePubSpec of target `circ.label`."""
    ensure_phase1()
    from acts_qaoa.circuits import transpile_best
    from .components import component_layout_record
    isa, st = transpile_best(circ.qc_meas, backend, list(range(int(n_seeds))))
    spec = Z.ZnePubSpec(k, "main", 0, [], [], int(shots), False, circ.label, 1)
    return IsaPub(spec, isa, None, st, component_layout_record(isa, circ.n_bits, st))


@dataclass
class Qx4Plan:
    job: Z.ZneJob
    circuits: Dict[str, Qx4Circuit]
    ideals: Dict[str, dict]
    selection: dict


def default_folder(isa, lam, seed):
    """The folding of hw-zne: every two-qubit gate G -> G (G^dagger G)^m, lambda = 2m + 1 (odd integers only; `seed` unused)."""
    return Z.fold_two_qubit_gates(isa, lam)


def fold_pub(base: IsaPub, spec: Z.ZnePubSpec, backend, folder: Callable) -> IsaPub:
    """The PUB of `base` at spec.lam: circuit = folder(base.isa, lam, transpile seed), stats recomputed.  With the default
    folder the 2q count must be exactly lam x the base count (zne.fold_pub); another folder only has to keep the parameters."""
    if folder is default_folder:
        return Z.fold_pub(base, spec, backend)
    ensure_phase1()
    from acts_qaoa.circuits import circuit_stats
    isa = folder(base.isa, spec.lam, base.stats.get("transpile_seed"))
    if [p.name for p in isa.parameters] != [p.name for p in base.isa.parameters]:
        raise RuntimeError("folding changed the circuit's parameters")
    st = circuit_stats(isa, backend)
    st["transpile_seed"] = base.stats.get("transpile_seed")
    st["n_2q_base"] = int(base.stats["n_2q"])
    return IsaPub(spec, isa, base.values, st, base.layout)


def plan_qx4(events: Sequence[Tuple[int, int]], lams: Sequence[float], lams_other: Sequence[float], hhl: bool, setting: Setting,
             cfg: HwConfig, backend, event_seed: int = 0, alpha: float = 0.25, beta: float = 0.5, eps: float = 1e-4,
             T: float = 0.8, n_clock: int = HHL_CLOCK, folder: Callable = default_folder) -> Qx4Plan:
    """One job: the first event at the noise factors `lams`, every other event at `lams_other`, optionally the HHL PUB; PUBs in
    this order, equal shots, folded by `folder(isa, lam, seed) -> circuit` (default: hsf2.hw.zne.fold_two_qubit_gates).  The job's QPU estimate is the Phase-1 conservative model."""
    ensure_phase1()
    from acts_qaoa import ibm
    if setting.gauged or setting.fractional:
        raise ValueError("hw-qx4 uses settings a-d (no gauges, no fractional gates)")
    if setting.gate_twirl:
        log.warning("[hw-qx4] setting %s twirls the gates: the design runs setting b (DD only)", setting.name)
    entries: List[Tuple[Qx4Circuit, List[int]]] = []
    for e, (npart, nlay) in enumerate(events):
        entries.append((build_filter(npart, nlay, event_seed, alpha, beta, eps, T), parse_lams(lams if e == 0 else lams_other)))
    if hhl:
        entries.append((build_filter(*HHL_EVENT, event_seed, alpha, beta, eps, T, "hhl", n_clock), [1]))
    labels = [c.label for c, _ in entries]
    if len(set(labels)) != len(labels):
        raise ValueError(f"two circuits have the same label: {labels}")
    n_pubs = sum(len(ls) for _, ls in entries)
    if n_pubs > MAX_PUBS:
        raise ValueError(f"{n_pubs} PUBs exceed the {MAX_PUBS} of one job")
    pubs, infos, ideals = [], [], {}
    k = 0
    for circ, ls in entries:
        ideal = ideal_cond(circ)
        ideals[circ.label] = ideal
        base = transpile_filter(circ, backend, cfg.transpile_seeds, 0, cfg.shots)
        log.info("[hw-qx4] %s: N=%d, %d qubits, %d Pauli terms, %d 2q gates, depth %s, F_est %.3g, ideal P(post) %.4f, ideal L %.3g, "
                 "physical qubits %s", circ.label, circ.N, circ.n_bits, circ.n_terms, base.stats["n_2q"], base.stats.get("depth"),
                 base.stats.get("fidelity_est", float("nan")), ideal["p_post"], ideal["leakage"], base.layout["measured"]["qubits"])
        info = dict(circ.meta(), ideal_p_post=ideal["p_post"], ideal_p_kept_post=ideal["p_kept_post"],
                    ideal_leakage=ideal["leakage"], gammas=[], betas=[])
        for lam in ls:
            pubs.append(fold_pub(base, Z.ZnePubSpec(k, "main", 0, [], [], int(cfg.shots), False, circ.label, lam), backend, folder))
            infos.append(info)
            k += 1
    rep = ibm.rep_delay_seconds(backend)
    rnd = 16 if (setting.gate_twirl or setting.measure_twirl) else 1
    dur = max((p.stats.get("duration_s") or 0.0) for p in pubs) or None
    est = ibm.estimate_qpu_seconds([cfg.shots] * len(pubs), dur, rep, rnd)
    job = Z.ZneJob(1, 1, f"qx4_{setting.name}_1", pubs, infos, est, rep, dur)
    selection = {"events": [list(e) for e in events], "event_seed": int(event_seed), "lams": parse_lams(lams),
                 "lams_other": parse_lams(lams_other), "hhl": bool(hhl), "n_clock": int(n_clock), "alpha": alpha, "beta": beta,
                 "eps": eps, "T": T, "transpile_seeds": int(cfg.transpile_seeds)}
    return Qx4Plan(job, {c.label: c for c, _ in entries}, ideals, selection)


def plan_table(plan: Qx4Plan) -> List[dict]:
    rows = Z.plan_table([plan.job])
    for r, p in zip(rows, plan.job.pubs):
        r["qubits"] = plan.circuits[p.spec.target].n_bits
    return rows


def readout_estimate(backend, shots: int, setting: Setting) -> float:
    """The estimate of run.run_readout (two calibration PUBs of `shots` shots)."""
    ensure_phase1()
    from acts_qaoa import ibm
    return float(ibm.estimate_qpu_seconds([shots, shots], None, ibm.rep_delay_seconds(backend), 16 if setting.measure_twirl else 1))


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def _confirm_header(confirm: Callable[[str], None], plan: Qx4Plan) -> Callable[[str], None]:
    pubs = ", ".join(f"{p.spec.target} lambda={p.spec.lam} ({p.stats['n_2q']} 2q gates)" for p in plan.job.pubs)
    header = f"QX4 job (1-bit filter / HHL on the smallest toy events), one PUB per (circuit, lambda): {pubs}"
    return lambda msg: confirm(header + chr(10) + msg)


def run_qx4(run_dir: Path, events: Sequence[Tuple[int, int]], setting_name: str, cfg: HwConfig,
            lams: Sequence[float] = DEFAULT_LAMS, lams_other: Sequence[float] = OTHER_LAMS, hhl: bool = True,
            readout: bool = True, cal_shots: Optional[int] = None, event_seed: int = 0, alpha: float = 0.25, beta: float = 0.5,
            eps: float = 1e-4, T: float = 0.8, n_clock: int = HHL_CLOCK, step: str = STEP, command: str = "",
            out_dir: Optional[Path] = None, confirm: Callable[[str], None] = R.confirm_submit, seed: int = 0,
            boot_n: int = BOOT_N, analysis_seed: int = BOOT_SEED, service=None,
            folder: Callable = default_folder) -> dict:
    """hw-qx4: plan the job (transpile, fold), then dry (plan files only), local (simulate) or submit.  The budget of the job and
    of its read-out calibration is checked before the first prompt; the job has its typed confirmation, the calibration
    (hw-readout pattern, run.run_readout) its own.  The analysis of a finished job (local or submitted) runs at the end, with
    the calibration if there is one.  Returns {'job_dir', 'manifest', 'plan', 'readout', 'analysis', 'service',
    'estimate_total_s'}; the plan table is also written to out_dir/qx4_plan.json."""
    cfg.validate()
    run_dir = Path(run_dir)
    setting = get_setting(setting_name)
    if setting.gauged or setting.fractional:
        raise ValueError("hw-qx4 uses settings a-d (no gauges, no fractional gates)")
    backend, service = R.get_job_backend(cfg, setting, service)
    name = Z._backend_name(backend)
    plan = plan_qx4(events, lams, lams_other, hhl, setting, cfg, backend, event_seed, alpha, beta, eps, T, n_clock, folder)
    job = plan.job
    ro_shots = int(cal_shots or cfg.shots)
    ro_est = readout_estimate(backend, ro_shots, setting) if readout else 0.0
    total = job.estimate_s + ro_est
    table = plan_table(plan)
    for r in table:
        log.info("[hw-qx4] pub %d: %-9s lambda %d: %2d qubits, %4d 2q gates (base %d), depth2q %s, F_est %.3g", r["pub"], r["target"],
                 r["lam"], r["qubits"], r["n_2q"], r["n_2q_base"], r["depth_2q"], r["fidelity_est"] or float("nan"))
    log.info("[hw-qx4] %d PUBs x %d shots, setting %s; estimated %.1f s of QPU time (job %.1f s%s)", len(table), cfg.shots,
             setting.name, total, job.estimate_s, f" + read-out calibration {ro_est:.1f} s" if readout else "")
    if out_dir is not None:
        save_json(Path(out_dir) / "qx4_plan.json", {"setting": setting.name, "shots": cfg.shots, "backend": name,
                                                    "estimate_job_s": job.estimate_s, "estimate_readout_s": ro_est,
                                                    "estimate_total_s": total, "selection": plan.selection, "pubs": table})
    ledger = None
    if cfg.mode == "submit":
        from . import session
        ledger = HwLedger(cfg.ledger_path(), cfg.budget_s, creds_file=cfg.creds_file)
        ledger.check(total, session.service_remaining(service), cfg.reserve_s)       # job and calibration, before the first prompt
    elif cfg.mode == "dry":
        led = HwLedger(cfg.ledger_path(), cfg.budget_s, creds_file=cfg.creds_file)
        log.info("[hw-qx4] ledger %s: used %.1f s, remaining %.1f s of %.0f s; a submit would book an estimated %.1f s", led.path,
                 led.used_seconds(), led.remaining_seconds(), led.budget_s, total)
    job_dir = R.new_job_dir(run_dir, job.tag)
    manifest = Z.job_manifest(job, setting, cfg, name, run_dir, step, command, plan.selection)
    manifest["kind"] = KIND
    Z.write_job_files(job_dir, job, backend)
    log.info("[hw-qx4] job folder %s | %d PUBs x %d shots | estimated %.1f s of QPU time", job_dir, len(job.pubs), job.shots,
             job.estimate_s)
    res = {"job_dir": job_dir, "manifest": manifest, "plan": plan, "job_id": None, "counts": None, "readout": None,
           "analysis": None, "service": service, "estimate_total_s": total}
    R.save_manifest(job_dir, manifest)                                   # the manifest is on disk before anything is submitted
    if cfg.mode == "dry":
        manifest["state"] = "dry"
        R.save_manifest(job_dir, manifest)
        if readout:
            res["readout"] = R.run_readout(run_dir, job_dir, cfg, step, command, ro_shots, confirm, service, seed)
        return res
    try:
        if cfg.mode == "local":
            counts = Z._local_job(backend, job, setting, job_dir, manifest, seed, "auto")
            ex = {"job_id": manifest["job_id"], "counts": counts, "billed": manifest["billed"], "job": None}
        else:
            ex = R.execute_job(backend, job.pubs, cfg, setting, f"hw-qx4-{setting.name}", job.estimate_s, job_dir, manifest,
                               str(run_dir), service, ledger, _confirm_header(confirm, plan), True, False, seed)
    except Exception as exc:
        if manifest.get("job_id") is None:                  # refused or failed before anything was submitted or simulated
            manifest.update(state="aborted", aborted=f"{type(exc).__name__}: {exc}")
            R.save_manifest(job_dir, manifest)
        raise
    res.update(job_id=ex["job_id"], counts=ex["counts"], billed=ex["billed"])
    Z.finish_zne_job(job_dir, manifest, ex["counts"], ex.get("job"), backend)
    cal_path = None
    try:
        if readout:
            res["readout"] = R.run_readout(run_dir, job_dir, cfg, step, command, ro_shots, confirm, service, seed)
            cal_path = Path(res["readout"]["job_dir"]) / "readout_cal.json"
    finally:                                                 # the analysis also runs if the calibration was refused
        if out_dir is not None:
            res["analysis"] = analyse_qx4(job_dir, out_dir, cal_path if cal_path is not None and cal_path.is_file() else None,
                                          boot_n, analysis_seed)
    return res


def retrieve_qx4_job(job_id: str, cfg: HwConfig, run_dir: Optional[Path] = None, search_root: Optional[Path] = None,
                     service=None) -> dict:
    """Rebuild the outputs of a qx4 job whose shell was lost (as zne.retrieve_zne_job): counts from local_result.json or IBM,
    billed time resolved and booked, counts files and post snapshot.  The analysis is hw-qx4-analyze."""
    from . import fake as F
    job_dir = R.find_job_dir(job_id, run_dir, search_root)
    manifest = load_json(job_dir / "manifest.json")
    if manifest.get("kind") != KIND:
        raise ValueError(f"{job_dir} is not a qx4 job (kind {manifest.get('kind')!r})")
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
        Z.finish_zne_job(job_dir, manifest, counts, job, backend)
    return {"job_dir": job_dir, "manifest": manifest}


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _num(x) -> Optional[float]:
    x = float(x)
    return x if math.isfinite(x) else None


def summarize(rows: np.ndarray, ideal: Optional[float] = None) -> dict:
    """Point (row 0), bootstrap spread (rows 1..) as std and 16 / 84 % of the finite replicates, and whether the 68 % interval
    brackets `ideal`."""
    v, rep = float(rows[0]), np.asarray(rows[1:], float)
    good = rep[np.isfinite(rep)]
    rec = {"value": _num(v), "err": None, "lo68": None, "hi68": None, "n_valid": int(good.size)}
    if good.size >= 2 and math.isfinite(v):
        lo, hi = np.percentile(good, [16, 84])
        rec.update(err=_num(good.std(ddof=1)), lo68=_num(lo), hi68=_num(hi))
        if ideal is not None:
            rec["ideal"] = float(ideal)
            rec["brackets_ideal"] = bool(lo <= ideal <= hi)
            if rec["err"]:
                rec["consistent_with_ideal_2sigma"] = bool(abs(v - ideal) <= 2.0 * rec["err"])
    elif ideal is not None:
        rec["ideal"] = float(ideal)
    return rec


def replicate_probs(counts: Dict[int, int], n_bits: int, boot_n: int, rng: np.random.Generator) -> np.ndarray:
    """(boot_n + 1, 2^n_bits) probabilities: row 0 the observed frequencies, the others multinomial redraws of the shots."""
    v = np.zeros(1 << n_bits)
    for k, c in counts.items():
        if not 0 <= int(k) < v.size:
            raise ValueError(f"sample index {k} outside 0..2^{n_bits} - 1")
        v[int(k)] += c
    S = int(v.sum())
    if S < 1:
        raise ValueError("empty counts")
    out = np.empty((boot_n + 1, v.size))
    out[0] = v / S
    if boot_n:
        out[1:] = rng.multinomial(S, v / S, size=boot_n) / S
    return out


def _correct_rows(P: np.ndarray, mats: Sequence[np.ndarray]) -> np.ndarray:
    """Exact tensor inverse of the per-qubit read-out model on every row (quasi-probabilities, rows sum to 1)."""
    from .readout import tensor_apply
    inv = [np.linalg.inv(np.asarray(m, float)) for m in mats]
    return np.stack([tensor_apply(r, inv) for r in P])


def _nearest(rows: np.ndarray) -> np.ndarray:
    """Rows -> the L2-nearest probability vectors (a row with a non-positive sum is NaN)."""
    from .readout import nearest_probability_distribution
    out = np.full(rows.shape, np.nan)
    for i, r in enumerate(rows):
        if np.all(np.isfinite(r)) and r.sum() > 0:
            out[i] = nearest_probability_distribution(r / r.sum())
    return out


def distribution_stats(circ: Qx4Circuit, P: np.ndarray, ideal_cond_vec: np.ndarray, clip: bool) -> Dict[str, np.ndarray]:
    """Per row of P (R, 2^n_bits): P(post), P(post, kept), L, and, from the conditional distribution over the system states,
    the total variation and Hellinger distance to the ideal one and, for the filter, the segments kept by the filter (conditional
    frequency >= 1 / N) as efficiency and purity against the classical solution and against the truth.  `clip`: the conditional
    distribution of read-out-corrected rows (quasi-probabilities) is first made a probability vector."""
    post_idx = circ.post_offset + np.arange(1 << circ.q)
    den = P[:, circ.post_mask].sum(axis=1)
    num = P[:, circ.kept_post_mask].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = den > 0
        cond = np.where(ok[:, None], P[:, post_idx] / np.where(ok, den, 1.0)[:, None], np.nan)
        leak = np.where(ok, 1.0 - num / np.where(ok, den, 1.0), np.nan)
    if clip:
        cond = _nearest(cond)
    tv = 0.5 * np.abs(cond - ideal_cond_vec[None, :]).sum(axis=1)
    hel = np.sqrt(np.clip(0.5 * ((np.sqrt(np.clip(cond, 0, None)) - np.sqrt(ideal_cond_vec)[None, :]) ** 2).sum(axis=1), 0, None))
    out = {"P_post": den, "P_kept_post": num, "L": leak, "TV": tv, "Hellinger": hel}
    if circ.circuit == "filter":
        keep = cond[:, :circ.N] >= 1.0 / circ.N - 1e-12
        keep &= np.isfinite(cond[:, :circ.N])
        for tag, ref in (("classical", circ.classical), ("truth", circ.truth)):
            tp = (keep & ref[None, :]).sum(axis=1)
            out[f"eff_{tag}"] = tp / max(int(ref.sum()), 1)
            out[f"pur_{tag}"] = tp / np.maximum(keep.sum(axis=1), 1)
        out["n_kept_hw"] = keep.sum(axis=1).astype(float)
        out["nan_row"] = (~np.isfinite(cond).all(axis=1)).astype(float)
    return out


def uniform_control(circ: Qx4Circuit, n_post: int, draws: int = 1000, seed: int = 0) -> dict:
    """Filter decisions and distances of `draws` samples of `n_post` post-selected shots from the uniform distribution over the
    2^q system states (the control that a device that returns noise would give): mean and standard deviation per statistic."""
    if circ.circuit != "filter" or n_post < 1:
        return {}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7]))
    cond = rng.multinomial(n_post, np.full(1 << circ.q, 1.0 / (1 << circ.q)), size=draws) / n_post
    P = np.zeros((draws, 1 << circ.n_bits))
    P[:, circ.post_offset + np.arange(1 << circ.q)] = cond
    st = distribution_stats(circ, P, ideal_cond(circ)["cond"], False)
    rec = {"n_post": int(n_post), "draws": int(draws)}
    for k, v in st.items():
        if k in ("P_post", "nan_row"):
            continue
        rec[k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1))}
    return rec


def ratio_extrapolation(lams: Sequence[int], num: np.ndarray, den: np.ndarray, floor_num: float, floor_den: float,
                        ideal_leak: Optional[float]) -> Dict[str, dict]:
    """Zero-noise extrapolation of the ratio: the numerator P(post, kept) and the denominator P(post) (rows: observed, then the
    bootstrap replicates; shape (L, R + 1)) are extrapolated separately with the three models of hsf2.hw.zne, and L = 1 - num /
    den is formed per replicate.  Returns {model: {num, den, L}} summaries (L also 'ideal' and 'brackets_ideal')."""
    Y = np.stack([num, den], axis=-1)                                      # (L, R + 1, 2)
    ex = Z.extrapolate(lams, Y, np.array([floor_num, floor_den]))
    out = {}
    for m in Z.MODELS:
        n_ex, d_ex = ex[m][:, 0], ex[m][:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            leak = np.where((d_ex > 0) & np.isfinite(n_ex) & np.isfinite(d_ex), 1.0 - n_ex / np.where(d_ex > 0, d_ex, 1.0), np.nan)
        out[m] = {"num": summarize(n_ex), "den": summarize(d_ex), "L": summarize(leak, ideal_leak)}
    return out


def analyze_target(circ: Qx4Circuit, ideal: dict, counts_by_lam: Dict[int, Dict[int, int]], mats: Optional[Sequence[np.ndarray]] = None,
                   boot_n: int = BOOT_N, seed: int = BOOT_SEED, tag: str = "") -> dict:
    """Points and zero-noise extrapolations of one circuit, raw and (with `mats`, the per-bit read-out matrices) read-out-
    corrected, from its counts at each noise factor; see the module docstring.  The replicates of the two variants are the
    same draws."""
    lams = sorted(counts_by_lam)
    probs = {}
    for lam in lams:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(tag.encode()), zlib.crc32(repr(lam).encode())]))
        probs[lam] = replicate_probs(counts_by_lam[lam], circ.n_bits, boot_n, rng)
    variants = {"raw": (False, probs)}
    if mats is not None:
        variants["readout"] = (True, {lam: _correct_rows(P, mats) for lam, P in probs.items()})
    n_post_states, n_kept_post = int(circ.post_mask.sum()), int(circ.kept_post_mask.sum())
    floor_den, floor_num = n_post_states / (1 << circ.n_bits), n_kept_post / (1 << circ.n_bits)
    res: dict = {"lambdas": lams, "shots": {str(l): int(sum(c.values())) for l, c in counts_by_lam.items()}, "variants": {}}
    for vname, (corrected, PP) in variants.items():
        stats = {lam: distribution_stats(circ, PP[lam], ideal["cond"], corrected) for lam in lams}
        points = {}
        for lam in lams:
            points[str(lam)] = {k: summarize(v, {"L": ideal["leakage"], "P_post": ideal["p_post"], "P_kept_post": ideal["p_kept_post"],
                                                 "TV": 0.0, "Hellinger": 0.0}.get(k)) for k, v in stats[lam].items()}
        rec: dict = {"points": points}
        if len(lams) >= 2 and circ.circuit == "filter":
            rec["extrapolation"] = ratio_extrapolation(lams, np.stack([stats[l]["P_kept_post"] for l in lams]),
                                                       np.stack([stats[l]["P_post"] for l in lams]), floor_num, floor_den,
                                                       ideal["leakage"])
        elif len(lams) >= 2:
            ex = Z.extrapolate(lams, np.stack([stats[l]["P_post"] for l in lams])[..., None], np.array([floor_den]))
            rec["extrapolation"] = {m: {"den": summarize(ex[m][:, 0], ideal["p_post"])} for m in Z.MODELS}
        res["variants"][vname] = rec
    return res


def _manifest_circuit(p: dict) -> Qx4Circuit:
    circ = build_filter(p["n_particles"], p["n_layers"], p["event_seed"], p["alpha"], p["beta"], p["eps"], p["T"], p["circuit"],
                        p["n_clock"] or HHL_CLOCK)
    meta = circ.meta()
    for k in ("A_sha1", "n_segments", "n_vars", "n_pauli_terms", "anc_bit", "anc_value", "clock_bits", "n_kept", "n_rejected"):
        if meta[k] != p[k]:
            raise ValueError(f"{p['target']}: {k} = {meta[k]} of the rebuilt circuit != {p[k]} of the manifest (not the circuit that was run)")
    return circ


def analyse_qx4(job_dir: Path, out_dir: Path, readout: Optional[Path] = None, boot_n: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """hw-qx4-analyze: per circuit of a finished qx4 job the leakage L, P(post), the filter decisions and the distances at every
    noise factor with bootstrap errors, and the ratio zero-noise extrapolation (see the module docstring); `readout`: a
    readout_cal.json covering the measured qubits adds the read-out-corrected variant.  Writes qx4_analysis.json and the csv
    files qx4_points.csv, qx4_extrapolation.csv, qx4_control.csv into out_dir."""
    import pandas as pd
    from .analysis import load_calibration
    job_dir = Path(job_dir)
    man = load_json(job_dir / "manifest.json")
    if man.get("kind") != KIND:
        raise ValueError(f"{job_dir} is not a qx4 job (kind {man.get('kind')!r}); hw-qx4 writes them")
    if man.get("state") != "done":
        raise ValueError(f"{job_dir} is in state {man.get('state')!r}, not done: no counts to analyse")
    cal = None if readout is None else load_calibration(readout)
    by_target: Dict[str, List[dict]] = {}
    for p in man["pubs"]:
        by_target.setdefault(p["target"], []).append(p)
    targets, pt_rows, ex_rows, ctl_rows = {}, [], [], []
    for label, pubs in by_target.items():
        circ = _manifest_circuit(pubs[0])
        ideal = ideal_cond(circ)
        if abs(ideal["p_post"] - float(pubs[0]["ideal_p_post"])) > IDEAL_TOL:
            raise ValueError(f"{label}: ideal P(post) {ideal['p_post']} != {pubs[0]['ideal_p_post']} of the manifest")
        counts: Dict[int, Dict[int, int]] = {}
        n2q: Dict[int, int] = {}
        fid: Dict[int, Optional[float]] = {}
        for p in pubs:
            c = {int(k): int(v) for k, v in load_json(job_dir / f"pub_{p['k']}_counts.json")["counts"].items()}
            tot = counts.setdefault(as_lam(p["lam"]), {})
            for k, v in c.items():
                tot[k] = tot.get(k, 0) + v
            n2q[as_lam(p["lam"])] = int(p["n_2q"])
            fid[as_lam(p["lam"])] = p.get("fidelity_est")
        mats, ro = None, None
        if cal is not None:
            mats, qubits = Z._readout_mats(cal, man, pubs)
            ro = {"calibration": str(readout), "method": "tensor-inverse", "qubits": qubits,
                  "p1_given_0": [float(m[1, 0]) for m in mats], "p0_given_1": [float(m[0, 1]) for m in mats]}
        rec = analyze_target(circ, ideal, counts, mats, boot_n, seed, label)
        eig = eigen_reference(circ)
        lam1 = min(counts)
        raw_post = sum(v for k, v in counts[lam1].items() if circ.post_mask[k])
        rec.update(circuit=circ.circuit, target=label, n_2q=n2q, n_segments=circ.N, n_qubits=circ.n_bits, n_kept=int(circ.kept.sum()),
                   n_rejected=int((~circ.kept).sum()), readout=ro,
                   ideal={"p_post": ideal["p_post"], "p_kept_post": ideal["p_kept_post"], "leakage": ideal["leakage"],
                          "cond": [float(x) for x in ideal["cond"]]},
                   eigen_emulation={"p_post": eig["p_post"], "cond": [float(x) for x in eig["cond"]]},
                   controls={"uniform_leakage": uniform_leakage(circ) if circ.circuit == "filter" else None,
                             "white_noise_leakage_at_F_est": (
                                 {str(l): white_noise_leakage(circ, min(max(float(f), 0.0), 1.0), ideal)
                                  for l, f in fid.items() if f is not None} if circ.circuit == "filter" else None),
                             "uniform_filter": uniform_control(circ, raw_post, seed=seed)},
                   classical={"active": [bool(x) for x in circ.classical], "truth": [bool(x) for x in circ.truth]})
        targets[label] = rec
        for vname, v in rec["variants"].items():
            for lam, pts in v["points"].items():
                for k, s in pts.items():
                    pt_rows.append({"target": label, "variant": vname, "lam": as_lam(lam), "n_2q": n2q[as_lam(lam)], "quantity": k,
                                    "value": s["value"], "err": s["err"], "lo68": s["lo68"], "hi68": s["hi68"], "ideal": s.get("ideal")})
            for m, qs in (v.get("extrapolation") or {}).items():
                for k, s in qs.items():
                    ex_rows.append({"target": label, "variant": vname, "model": m, "quantity": k,
                                    "lambdas": ",".join(map(str, rec["lambdas"])), "value": s["value"], "err": s["err"],
                                    "lo68": s["lo68"], "hi68": s["hi68"], "ideal": s.get("ideal"),
                                    "brackets_ideal": s.get("brackets_ideal"), "n_valid": s["n_valid"]})
        for k, s in (rec["controls"]["uniform_filter"] or {}).items():
            if isinstance(s, dict):
                ctl_rows.append({"target": label, "statistic": k, "mean": s["mean"], "sd": s["sd"]})
        for vname, v in rec["variants"].items():
            first = v["points"][str(rec["lambdas"][0])]
            if circ.circuit == "filter":
                log.info("[hw-qx4-analyze] %s %-7s lambda %d: P(post) %.4f (ideal %.4f), L %.3f +- %s (ideal %.3g, uniform %.3f)",
                         label, vname, rec["lambdas"][0], first["P_post"]["value"], ideal["p_post"], first["L"]["value"],
                         "n/a" if first["L"]["err"] is None else f"{first['L']['err']:.3f}", ideal["leakage"], uniform_leakage(circ))
                if "extrapolation" in v:
                    log.info("[hw-qx4-analyze] %s %-7s L extrapolated: %s", label, vname, " ".join(
                        (f"{m} {r['L']['value']:.3f} +- {r['L']['err']:.3f} ({'consistent' if r['L'].get('consistent_with_ideal_2sigma') else 'not consistent'} "
                        f"with {ideal['leakage']:.3g} within 2 sigma)" if r["L"]["err"] else f"{m} {r['L']['value']:.3f}") if r["L"]["value"] is not None else f"{m} -" for m, r in v["extrapolation"].items()))
            else:
                log.info("[hw-qx4-analyze] %s %-7s P(post) %.4f (ideal %.4f), TV to ideal conditional %.3f", label, vname,
                         first["P_post"]["value"], ideal["p_post"], first["TV"]["value"])
    res = {"schema": 1, "created": utc_now(), "job": job_dir.name, "run_dir": str(job_dir.parents[1]),
           "config": {"boot_n": boot_n, "boot_seed": seed, "interval": "16/84 %", "models": list(Z.MODELS),
                      "ratio": "numerator P(post, kept) and denominator P(post) extrapolated separately; L = 1 - num / den per "
                               "replicate; L itself is never extrapolated; an extrapolated L below 0 is kept (not clipped); success test: "
                               "consistent with 0 within 2 sigma, next to the lambda = 1 value",
                      "exponential_B": "uniform value of the numerator / denominator (fixed)",
                      "errors": "bootstrap over the shots at each lambda; shot noise only",
                      "readout": None if readout is None else {"calibration": str(readout), "method": "tensor-inverse"}},
           "targets": targets}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res["files"] = {"json": str(save_json(out_dir / "qx4_analysis.json", res)),
                    "points": str(save_csv(pd.DataFrame(pt_rows), out_dir / "qx4_points.csv")),
                    "extrapolation": str(save_csv(pd.DataFrame(ex_rows), out_dir / "qx4_extrapolation.csv")),
                    "control": str(save_csv(pd.DataFrame(ctl_rows), out_dir / "qx4_control.csv"))}
    return res

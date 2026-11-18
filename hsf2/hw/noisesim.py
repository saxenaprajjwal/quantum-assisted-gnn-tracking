"""Noise-simulation prediction of the folding diagnosis (0 QPU s): what fractional / all-gate / twirled folding would give.

Question: ZNE with CZ folding at lambda = 1, 3, 5 stayed 3.6-11 sigma short at 93-232 CZ.  Would folding the single-qubit gates
as well (arm ii), or twirling (arm iii), close the gap?  This module simulates the folded ISA circuits of a stored zne job with
a noise model built from the job's own calibration snapshot (snapshot_pre.json), and analyses them with the estimators of
hsf2.hw.zne (mean energy, F_E, F_XEB, 1 % and 10 % tails; the same extrapolation models over the ACHIEVED lambda).

Noise model (everything is a Pauli fault location; nothing else is random):
  * relaxation: T1 / T2 of every qubit over the time since its last gate (idle time included; ASAP schedule from the snapshot's
    gate durations, barriers synchronise their qubits, rz is virtual), in its Pauli-twirled form: p_X = p_Y = (1 - e^{-t/T1}) / 4,
    p_Z = (1 - e^{-t/T2}) / 2 - p_X.  T2 is the Hahn-echo value of the snapshot, so dynamical decoupling (setting b) is already in it;
  * gate errors: the snapshot's error of sx / x (single-qubit depolarising) and of cz (two-qubit depolarising), converted to a
    Pauli probability p = e (d + 1) / d, minus the relaxation of the gate's duration (counted above);
  * read-out: the snapshot's p(1|0) and p(0|1) per qubit.  The prediction is the read-out-CORRECTED one (the analysis of the
    hardware data applies the exact tensor inverse), so it is the distribution before the read-out; the raw values after the
    read-out are reported next to it;
  * coherent CZ error (optional, `phi`): an RZZ(phi) after every CZ (a ZZ phase), the same unitary every time it occurs, so that
    it adds up coherently (CZ G^dagger G folds triple it).  With twirling (arm iii) it becomes a Pauli ZZ error of probability
    sin^2(phi / 2) per CZ.
Not modelled: crosstalk and leakage, 1q coherent errors, the pulses of dynamical decoupling and of twirling (the twirled arm is
therefore the best case of twirling: the measured penalty of twirling, 0.08 in F_XEB at 232 CZ, comes on top).

Simulation.  Trajectories: every one is a statevector (Aer, one circuit per trajectory, the Paulis inserted as gates, so Aer
fuses them) and gives its exact outcome distribution |psi_t(x)|^2; the average over trajectories is the noisy distribution (no
shot noise, 13-15 qubits and K = 48 at 24 qubits alike; matrix-product states are slower at 232 CZ, tested).  Variance reduction:
the fault-free trajectory has the known probability P0 = prod(1 - p_i); the other trajectories are sampled conditioned on at least
one fault (first-fault method), and an observable is P0 O_0 + (1 - P0) mean_t O_t.  The error of a number is the bootstrap over
the trajectories (also through the extrapolation); the shot noise of a 4 000-shot hardware run (0.02-0.04) comes on top, and
`hw_like` also analyses a synthetic 4 000-shot sample with hsf2.hw.zne.analyze_counts.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import load_json, save_json
from . import fold as FD
from . import zne as Z

log = logging.getLogger(__name__)

PAULI_GATE = {1: "x", 2: "z", 3: "y"}          # codes of the xor group I = 0, X = 1, Z = 2, Y = 3
BOOT_TRAJ = 400
MODELS = Z.MODELS


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
    name: str
    gates: str                  # fold gate set: "cz" or "all"
    twirled: bool = False       # coherent CZ error becomes a Pauli ZZ error
    lams: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)


ARMS = {"0": Arm("0", "cz", False, (1.0, 3.0, 5.0)),         # what the 12 November job measured (validation of the model)
        "0b": Arm("0b", "cz", False, (1.0, 3.0)),                 # the same for the K = 48 job (lambda = 1, 3 only)
        "i": Arm("i", "cz", False, (1.0, 1.5, 2.0, 3.0)), "ii": Arm("ii", "all", False, (1.0, 1.5, 2.0, 3.0)),
        "iii": Arm("iii", "cz", True, (1.0, 1.5, 2.0, 3.0))}


def get_arm(arm, lambdas: Optional[Sequence[float]] = None) -> Arm:
    a = ARMS[arm] if isinstance(arm, str) else arm
    if a.gates not in FD.GATE_SETS:
        raise ValueError(f"arm {a.name}: unknown gate set {a.gates!r}")
    if lambdas is not None:
        a = Arm(a.name, a.gates, a.twirled, tuple(float(x) for x in lambdas))
    if 1.0 not in a.lams or len(set(a.lams)) != len(a.lams):
        raise ValueError(f"arm {a.name}: needs distinct noise factors including 1, got {a.lams}")
    return a


# ---------------------------------------------------------------------------
# calibration -> device noise
# ---------------------------------------------------------------------------
@dataclass
class DeviceNoise:
    t1: Dict[int, float]
    t2: Dict[int, float]
    p10: Dict[int, float]                   # P(read 1 | prepared 0)
    p01: Dict[int, float]                   # P(read 0 | prepared 1)
    gate1: Dict[Tuple[str, int], Tuple[float, float]]       # (name, qubit) -> (error, duration)
    cz: Dict[Tuple[int, int], Tuple[float, float]]          # (a, b) with a < b -> (error, duration)


def device_noise(snapshot: dict) -> DeviceNoise:
    """T1, T2, read-out probabilities, sx / x and cz errors and durations from a calibration snapshot; a missing value raises."""
    t1, t2, p10, p01, g1, cz = {}, {}, {}, {}, {}, {}
    for q, r in snapshot["qubits"].items():
        q = int(q)
        for key, dst in (("t1", t1), ("t2", t2), ("prob_meas1_prep0", p10), ("prob_meas0_prep1", p01)):
            if r.get(key) is None:
                raise ValueError(f"snapshot has no {key} for qubit {q}")
            dst[q] = float(r[key])
        for g in ("sx", "x"):
            if r.get(g) is None or r[g].get("error") is None or r[g].get("duration") is None:
                raise ValueError(f"snapshot has no {g} error / duration for qubit {q}")
            g1[(g, q)] = (float(r[g]["error"]), float(r[g]["duration"]))
    for e, r in snapshot["edges"].items():
        a, b = (int(x) for x in e.split("-"))
        if "cz" in r and r["cz"].get("error") is not None:
            cz[(min(a, b), max(a, b))] = (float(r["cz"]["error"]), float(r["cz"]["duration"]))
    return DeviceNoise(t1, t2, p10, p01, g1, cz)


def readout_matrices(dev: DeviceNoise, measured: Sequence[int]) -> List[np.ndarray]:
    """P(read i | true j) of classical bit k (physical qubit measured[k])."""
    return [np.array([[1 - dev.p10[q], dev.p01[q]], [dev.p10[q], 1 - dev.p01[q]]]) for q in measured]


# ---------------------------------------------------------------------------
# Pauli channels on the xor group
# ---------------------------------------------------------------------------
def relaxation_pauli(t: float, t1: float, t2: float) -> np.ndarray:
    """Pauli-twirled thermal relaxation over time t: probabilities of (I, X, Z, Y)."""
    if t <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    px = (1.0 - math.exp(-t / t1)) / 4.0
    pz = max(0.0, (1.0 - math.exp(-t / t2)) / 2.0 - px)
    return np.array([1.0 - 2 * px - pz, px, pz, px])


def depolarizing_pauli(p: float, n: int) -> np.ndarray:
    """n-qubit Pauli depolarising channel with total error probability p, on the 4^n xor-group elements."""
    out = np.full(4 ** n, p / (4 ** n - 1))
    out[0] = 1.0 - p
    return out


def convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Composition of two independent Pauli channels (xor convolution over the 4^n elements)."""
    idx = np.arange(a.size)
    return np.array([float(a @ b[idx ^ k]) for k in range(a.size)])


def tensor2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.outer(a, b).reshape(-1)           # element a_i * b_j at index 4 i + j


# ---------------------------------------------------------------------------
# circuit -> ops and fault locations
# ---------------------------------------------------------------------------
@dataclass
class Locations:
    """Fault locations: after op `after[i]` (-1: before the first op), a Pauli error on the sim qubits `qubits[i]` with total
    probability p[i] and the conditional distribution `cond[i]` over the non-identity elements (1..3 or 1..15)."""
    after: np.ndarray
    qubits: List[Tuple[int, ...]]
    p: np.ndarray
    cond: List[np.ndarray]

    @property
    def p0(self) -> float:
        return float(np.prod(1.0 - self.p))


def isa_ops(isa, values, measured: Sequence[int]) -> Tuple[List[tuple], List[int]]:
    """([(name, sim qubits, angle)], physical qubits in sim order): the bound ISA circuit without its measurements; sim qubits
    0..n-1 are the measured qubits in classical-bit order, then the other qubits the circuit touches."""
    if len(isa.parameters):
        if values is None or len(values) != len(isa.parameters):
            raise ValueError("the circuit has free parameters: pass their values")
        isa = isa.assign_parameters(dict(zip(isa.parameters, [float(v) for v in values])))
    raw = []
    for ins in isa.data:
        name = ins.operation.name
        if name == "measure":
            continue
        if name not in FD.ISA_NAMES or name == "delay":
            raise ValueError(f"cannot simulate the instruction {name}: not an ISA gate of the folded circuits")
        raw.append((name, tuple(isa.find_bit(q).index for q in ins.qubits),
                    float(ins.operation.params[0]) if name == "rz" else 0.0))
    used = {q for _, qs, _ in raw for q in qs}
    order = [int(q) for q in measured] + sorted(used - set(int(q) for q in measured))
    pos = {q: i for i, q in enumerate(order)}
    return [(n, tuple(pos[q] for q in qs), a) for n, qs, a in raw], order


def fault_locations(ops: Sequence[tuple], order: Sequence[int], dev: DeviceNoise, phi: float = 0.0, twirled: bool = False,
                    relaxation: bool = True, gate_errors: bool = True, error_scale: float = 1.0) -> Locations:
    """ASAP schedule of `ops` with the snapshot's durations and the Pauli fault locations of the noise model (module docstring).
    `error_scale` multiplies every incoherent fault probability (1: the snapshot as it is)."""
    n = len(order)
    ready = [0.0] * n                       # scheduling cursor (barriers move it)
    last = [0.0] * n                        # time of the last fault location on the qubit
    after, qubits, probs, conds = [], [], [], []

    def relax(q: int, t_end: float) -> np.ndarray:
        phys = order[q]
        return relaxation_pauli(t_end - last[q], dev.t1[phys], dev.t2[phys]) if relaxation else np.array([1.0, 0, 0, 0])

    def add(i: int, qs: Tuple[int, ...], dist: np.ndarray) -> None:
        if error_scale != 1.0:
            dist = np.concatenate([[1.0 - error_scale * (1.0 - dist[0])], error_scale * dist[1:]])
        p = 1.0 - float(dist[0])
        if p > 1e-12:
            after.append(i)
            qubits.append(qs)
            probs.append(p)
            conds.append(dist[1:] / p)

    for i, (name, qs, _) in enumerate(ops):
        if name == "barrier":
            t = max(ready[q] for q in qs)
            for q in qs:
                ready[q] = t
            continue
        if name == "rz":
            continue
        if name in ("sx", "x"):
            err, dur = dev.gate1[(name, order[qs[0]])]
        elif name == "cz":
            a, b = sorted(order[q] for q in qs)
            if (a, b) not in dev.cz:
                raise ValueError(f"the snapshot has no cz error for the edge {a}-{b}")
            err, dur = dev.cz[(a, b)]
        else:
            raise ValueError(f"unexpected gate {name}")
        t_end = max(ready[q] for q in qs) + dur
        chans = [relax(q, t_end) for q in qs]
        if len(qs) == 1:
            d = chans[0]
            if gate_errors:
                phys = order[qs[0]]
                p_tot = err * 1.5
                p_rel = 1.0 - relaxation_pauli(dur, dev.t1[phys], dev.t2[phys])[0] if relaxation else 0.0
                d = convolve(d, depolarizing_pauli(max(0.0, p_tot - p_rel), 1))
        else:
            d = tensor2(chans[0], chans[1])
            if gate_errors:
                p_rel = sum(1.0 - relaxation_pauli(dur, dev.t1[order[q]], dev.t2[order[q]])[0] for q in qs) if relaxation else 0.0
                d = convolve(d, depolarizing_pauli(max(0.0, err * 1.25 - p_rel), 2))
            if twirled and phi:
                zz = np.zeros(16)
                zz[0], zz[2 * 4 + 2] = 1.0 - math.sin(phi / 2) ** 2, math.sin(phi / 2) ** 2
                d = convolve(d, zz)
        add(i, tuple(qs), d)
        for q in qs:
            ready[q] = t_end
            last[q] = t_end
    t_final = max(ready) if ready else 0.0       # every qubit waits for the slowest one before the read-out
    for q in range(n):
        if relaxation and t_final > last[q]:
            add(len(ops) - 1, (q,), relax(q, t_final))
    return Locations(np.array(after, dtype=np.int64), qubits, np.array(probs), conds)


def fault_count_law(p: np.ndarray, kmax: int = 120) -> np.ndarray:
    """P(K = k), k = 0..kmax, of the number of faults K = sum of independent Bernoulli(p_i) (Poisson binomial, exact)."""
    P = np.zeros(kmax + 1)
    P[0] = 1.0
    for pi in p:
        P[1:] = P[1:] * (1.0 - pi) + P[:-1] * pi
        P[0] *= 1.0 - pi
    return P


def control_expectations(p: np.ndarray) -> np.ndarray:
    """E[(K, 1[K = 1], 1[K = 2]) | K >= 1] of the fault count K: the controls of the control-variate estimator."""
    if p.size == 0:
        return np.zeros(3)
    P = fault_count_law(p)
    cond = P[1:] / (1.0 - P[0])
    k = np.arange(1, P.size)
    return np.array([float(k @ cond), cond[0], cond[1]])


def controls(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, float)
    return np.column_stack([k, k == 1, k == 2]).astype(float)


def cv_mean(obs: np.ndarray, ctrl: np.ndarray, ctrl_exp: np.ndarray) -> np.ndarray:
    """Control-variate mean over trajectories: mean(obs) - (mean(ctrl) - E[ctrl]) beta, beta by least squares (the fault count
    explains most of the variance of an observable).  Fewer than 12 trajectories: the plain mean."""
    m = obs.mean(axis=0)
    if obs.shape[0] < 12:
        return m
    cm = ctrl.mean(axis=0)
    beta = np.linalg.lstsq(ctrl - cm, obs - m, rcond=1e-8)[0]
    return m - (cm - ctrl_exp) @ beta


def sample_faults(loc: Locations, T: int, rng: np.random.Generator) -> List[Dict[int, List[Tuple[int, int]]]]:
    """T fault patterns conditioned on at least one fault (first-fault method): {op index: [(sim qubit, pauli code), ...]}; the
    pattern's fault count is under the key -2 (a list of one int)."""
    L = loc.p.size
    if L == 0:
        raise ValueError("no fault location: the noise model is empty")
    surv = np.concatenate([[1.0], np.cumprod(1.0 - loc.p)])             # surv[i] = prod_{j < i} (1 - p_j)
    w = loc.p * surv[:-1]
    w = w / w.sum()
    out = []
    first = rng.choice(L, size=T, p=w)
    for t in range(T):
        f = int(first[t])
        hit = [f] + [int(j) for j in (f + 1 + np.nonzero(rng.random(L - f - 1) < loc.p[f + 1:])[0])]
        pat: Dict[int, List[Tuple[int, int]]] = {}
        for j in hit:
            k = int(rng.choice(loc.cond[j].size, p=loc.cond[j])) + 1
            qs = loc.qubits[j]
            codes = [k] if len(qs) == 1 else [k >> 2, k & 3]
            pat.setdefault(int(loc.after[j]), []).extend((q, c) for q, c in zip(qs, codes) if c)
        pat[-2] = [len(hit)]
        out.append(pat)
    return out


def build_circuit(ops: Sequence[tuple], n: int, n_meas: int, phi: float = 0.0, faults: Optional[dict] = None):
    """The simulation circuit: the ops, an RZZ(phi) after every CZ (if phi), the Pauli faults as gates, and a save of the outcome
    probabilities of the measured qubits."""
    import qiskit_aer  # noqa: F401  (adds QuantumCircuit.save_probabilities)
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    faults = faults or {}
    for q, c in faults.get(-1, ()):
        getattr(qc, PAULI_GATE[c])(q)
    for i, (name, qs, ang) in enumerate(ops):
        if name == "cz":
            qc.cz(*qs)
            if phi:
                qc.rzz(phi, *qs)
        elif name == "sx":
            qc.sx(qs[0])
        elif name == "x":
            qc.x(qs[0])
        elif name == "rz":
            qc.rz(ang, qs[0])
        for q, c in faults.get(i, ()):
            getattr(qc, PAULI_GATE[c])(q)
    qc.save_probabilities(qubits=list(range(n_meas)))
    return qc


def simulate_probs(circuits: Sequence, n: int, chunk: Optional[int] = None):
    """Outcome distributions of the circuits (Aer statevector, one chunk of circuits at a time); yields one vector each."""
    from qiskit_aer import AerSimulator
    big = n > 18
    sim = AerSimulator(method="statevector", precision="single" if big else "double", fusion_enable=True,
                       max_parallel_experiments=2 if big else 8, max_parallel_threads=8)
    chunk = chunk or (4 if big else 64)
    for i in range(0, len(circuits), chunk):
        res = sim.run(list(circuits[i:i + chunk]), shots=1).result()
        if not res.success:
            raise RuntimeError(f"Aer simulation failed: {res.status}")
        for k in range(len(circuits[i:i + chunk])):
            yield np.asarray(res.data(k)["probabilities"], np.float64)


# ---------------------------------------------------------------------------
# observables of a distribution
# ---------------------------------------------------------------------------
@dataclass
class Context:
    """The ideal distribution and the energies of a target; observables of a distribution q over the 2^n states."""
    E: np.ndarray
    p: np.ndarray
    masks: Dict[str, np.ndarray]
    names: List[str] = field(default_factory=list)
    ideal: np.ndarray = None
    floor: np.ndarray = None

    def __post_init__(self):
        N = self.p.size
        self.E_unif = float(self.E.mean())
        self.e_ideal = float(self.p @ self.E)
        self.sum_p2 = float(self.p @ self.p)
        if abs(self.E_unif - self.e_ideal) < 1e-12:
            raise ValueError("ideal mean energy equals the uniform one: F_E is undefined")
        self.names = ["mean_energy", "F_E", "F_XEB"] + list(self.masks)
        self.ideal = np.array([self.e_ideal, 1.0, 1.0] + [float(self.p[m].sum()) for m in self.masks.values()])
        self.floor = np.array([self.E_unif, 0.0, 0.0] + [float(m.sum()) / N for m in self.masks.values()])
        self.N = N
        self._idx = [np.nonzero(m)[0] for m in self.masks.values()]      # index lists: q[idx].sum() is faster than q[mask].sum()

    def observables(self, q: np.ndarray) -> np.ndarray:
        """mean_energy, F_E, F_XEB and P(mask) of the distribution q: the zne.build_observables functions, in expectation."""
        e = float(q @ self.E)
        return np.array([e, (self.E_unif - e) / (self.E_unif - self.e_ideal),
                         (float(q @ self.p) - 1.0 / self.N) / (self.sum_p2 - 1.0 / self.N)] + [float(q[i].sum()) for i in self._idx])


def make_context(E: np.ndarray, p: np.ndarray, extra_masks: Optional[Dict[str, np.ndarray]] = None) -> Context:
    """E, p and the masks of the 1 % and 10 % best states (as zne._qx2_context) plus `extra_masks` (e.g. P_ground)."""
    E = np.asarray(E, np.float64)
    order = np.argsort(E, kind="stable")
    rank = np.empty(E.size, np.int32)
    rank[order] = np.arange(E.size, dtype=np.int32)
    masks = {f"P_best_{q * 100:g}pct": rank < int(round(q * E.size)) for q in Z.TAIL_QS}
    masks.update(extra_masks or {})
    return Context(E, np.asarray(p, np.float64), masks)


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
@dataclass
class SimTarget:
    label: str
    kind: str
    isa: object
    values: np.ndarray
    measured: List[int]
    dev: DeviceNoise
    ctx: Context
    pub: dict                           # the lambda = 1 PUB record of the stored manifest
    job_dir: Path
    readout: List[np.ndarray]

    @property
    def n(self) -> int:
        return len(self.measured)


def load_target(job_dir: Path, label: str) -> SimTarget:
    """The lambda = 1 circuit of `label` in a stored zne job folder (QPY, layout, angles), its calibration snapshot and the
    ideal distribution / energies from the run folder (exactly as zne.analyze_zne builds them)."""
    from types import SimpleNamespace
    from .snapshot import read_qpy
    job_dir = Path(job_dir)
    man = load_json(job_dir / "manifest.json")
    if man.get("kind") != Z.KIND:
        raise ValueError(f"{job_dir} is not a zne job")
    base = [p for p in man["pubs"] if p["target"] == label and ((not p.get("arm") and float(p["lam"]) == 1.0) or p.get("arm") == "base")]
    if not base:
        raise ValueError(f"{job_dir.name} has no lambda = 1 PUB of target {label}: it has {sorted({p['target'] for p in man['pubs']})}")
    pub = base[0]
    lay = load_json(job_dir / f"pub_{pub['k']}.layout.json")
    isa = read_qpy(job_dir / f"pub_{pub['k']}_{pub['name']}.isa.qpy")
    gam, bet = list(pub["gammas"]), list(pub["betas"])
    values = np.array([float((gam if x.name.startswith("gamma") else bet)[int(x.name.split("[")[1].rstrip("]"))]) for x in isa.parameters])
    run_dir = job_dir.parents[1]
    if pub.get("K"):
        E, p, _, _ = Z._qx2_context(run_dir, [pub], None, Z.TAIL_QS)
        extra, kind = {}, "qx2"
    else:
        E, p, masks, _ = Z._component_context(run_dir, [pub])
        extra, kind = {k: v for k, v in masks.items() if k == "P_ground"}, "component"
    dev = device_noise(load_json(job_dir / "snapshot_pre.json"))
    measured = [int(q) for q in lay["measured"]["qubits"]]
    return SimTarget(label, kind, isa, values, measured, dev, make_context(E, p, extra), pub, job_dir, readout_matrices(dev, measured))


# ---------------------------------------------------------------------------
# one configuration: (target, folded circuit, phi, twirled)
# ---------------------------------------------------------------------------
@dataclass
class ConfigResult:
    lam_target: float
    lam_eff: float
    gates: str
    twirled: bool
    phi: float
    n_cz: int
    n_locations: int
    expected_faults: float
    p0: float
    trajectories: int
    obs0: np.ndarray                    # fault-free (coherent error only) observables
    obs_t: np.ndarray                   # (T, J) per-trajectory observables (conditioned on >= 1 fault)
    ctrl_t: np.ndarray                  # (T, 3) controls of the trajectories (fault count, K = 1, K = 2)
    ctrl_exp: np.ndarray                # (3,) their exact expectations given K >= 1
    value: np.ndarray                   # P0 obs0 + (1 - P0) cv_mean(obs_t)   (read-out corrected = before the read-out)
    raw: np.ndarray                     # the same after the read-out error
    counts: Optional[Dict[int, int]] = None     # synthetic hardware-like sample (read-out noisy), if requested
    seconds: float = 0.0


def simulate_config(tgt: SimTarget, lam: float, gates: str, phi: float = 0.0, twirled: bool = False, trajectories: int = 200,
                    subsets: int = 1, seed: int = 0, relaxation: bool = True, gate_errors: bool = True, shots: int = 0,
                    error_scale: float = 1.0) -> ConfigResult:
    """Trajectory simulation of `tgt` folded to `lam` (module docstring).  `subsets` random fold subsets share the trajectories."""
    t0 = time.time()
    ctx, n = tgt.ctx, tgt.n
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(round(lam * 1000)), 7]))      # same faults for every phi
    q0_sum = np.zeros(ctx.N)
    q_sum = np.zeros(ctx.N)
    p0s, obs_t, locs_n, exp_f, lam_eff, n_cz, ks, cexp = [], [], [], [], [], 0, [], []
    per = [trajectories // subsets + (1 if s < trajectories % subsets else 0) for s in range(subsets)]
    for s in range(subsets):
        circ, rec = FD.fold_gates(tgt.isa, lam, gates, seed=1000 * s)
        lam_eff.append(rec["lam_eff"])
        n_cz = circ.count_ops().get("cz", 0)
        ops, order = isa_ops(circ, tgt.values, tgt.measured)
        loc = fault_locations(ops, order, tgt.dev, phi, twirled, relaxation, gate_errors, error_scale)
        n_sim = len(order)
        p0s.append(loc.p0)
        locs_n.append(loc.p.size)
        exp_f.append(float(loc.p.sum()))
        cexp.append(control_expectations(loc.p))
        coh = 0.0 if twirled else phi
        q0 = next(simulate_probs([build_circuit(ops, n_sim, n, coh)], n_sim))
        q0_sum += q0
        pats = sample_faults(loc, per[s], rng) if loc.p.size else []          # no fault location: the noiseless circuit
        ks += [pat[-2][0] for pat in pats]
        circs = (build_circuit(ops, n_sim, n, coh, pat) for pat in pats)
        batch = []
        for pat_circ in circs:
            batch.append(pat_circ)
            if len(batch) == (4 if n_sim > 18 else 64):
                for q in simulate_probs(batch, n_sim):
                    q_sum += q
                    obs_t.append(ctx.observables(q))
                batch = []
        if batch:
            for q in simulate_probs(batch, n_sim):
                q_sum += q
                obs_t.append(ctx.observables(q))
    obs_t = np.array(obs_t).reshape(-1, len(ctx.names))
    p0 = float(np.mean(p0s))
    ctrl_t, ctrl_exp = controls(ks), np.average(cexp, axis=0, weights=np.maximum(per, 1))
    q0m = q0_sum / subsets
    obs0 = ctx.observables(q0m)
    if len(obs_t) == 0:
        if p0 < 1.0:
            raise ValueError("no trajectory was simulated although the noise model has fault locations: trajectories must be > 0")
        value, q = obs0, q0m
    else:
        value = p0 * obs0 + (1 - p0) * cv_mean(obs_t, ctrl_t, ctrl_exp)
        q = p0 * q0m + (1 - p0) * q_sum / len(obs_t)
    from .readout import tensor_apply
    q_raw = tensor_apply(q, tgt.readout)
    raw = ctx.observables(q_raw)
    counts = None
    if shots:
        c = rng.multinomial(int(shots), q_raw / q_raw.sum())
        idx = np.nonzero(c)[0]
        counts = {int(i): int(c[i]) for i in idx}
    return ConfigResult(float(lam), float(np.mean(lam_eff)), gates, bool(twirled), float(phi), int(n_cz), int(np.mean(locs_n)),
                        float(np.mean(exp_f)), p0, len(obs_t), obs0, obs_t, ctrl_t, ctrl_exp, value, raw, counts, time.time() - t0)


# ---------------------------------------------------------------------------
# arms: points, extrapolation, bootstrap over the trajectories
# ---------------------------------------------------------------------------
def extrapolate_arm(results: Sequence[ConfigResult], ctx: Context, boot: int = BOOT_TRAJ, seed: int = 0) -> dict:
    """Points and the three extrapolations (zne.extrapolate, over the ACHIEVED lambda) of the results of one arm; the error of a
    number is the bootstrap over the trajectories of every lambda.  Also the extrapolation of the raw (read-out damped) values."""
    res = sorted(results, key=lambda r: r.lam_eff)
    lams = [FD.lam_key(r.lam_eff) for r in res]
    if len(lams) < 2:
        raise ValueError("extrapolation needs at least two noise factors")
    J = len(ctx.names)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 99]))
    Y = np.empty((len(res), boot + 1, J))
    for a, r in enumerate(res):
        Y[a, 0] = r.value
        T = r.obs_t.shape[0]
        if T == 0:                                      # noiseless: nothing to resample
            Y[a, 1:] = r.value
            continue
        ids = rng.integers(0, T, size=(boot, T))
        Y[a, 1:] = [r.p0 * r.obs0 + (1 - r.p0) * cv_mean(r.obs_t[i], r.ctrl_t[i], r.ctrl_exp) for i in ids]
    ex = Z.extrapolate(lams, Y, ctx.floor)
    raw_ex = Z.extrapolate(lams, np.stack([r.raw for r in res])[:, None, :], ctx.floor)
    out = {"lambdas": lams, "lam_target": [r.lam_target for r in res], "n_cz": [r.n_cz for r in res], "p0": [r.p0 for r in res],
           "expected_faults": [r.expected_faults for r in res], "trajectories": [r.trajectories for r in res], "observables": {}}
    for j, name in enumerate(ctx.names):
        pts = {str(lam): {"value": float(Y[a, 0, j]), "err": float(Y[a, 1:, j].std(ddof=1)), "raw": float(res[a].raw[j])}
               for a, lam in enumerate(lams)}
        mods = {}
        for m in MODELS:
            v, rep = ex[m][0, j], ex[m][1:, j]
            good = rep[np.isfinite(rep)]
            mods[m] = {"value": None if not np.isfinite(v) else float(v),
                       "err": float(good.std(ddof=1)) if good.size > 1 else None,
                       "raw_value": None if not np.isfinite(raw_ex[m][0, j]) else float(raw_ex[m][0, j])}
        out["observables"][name] = {"ideal": float(ctx.ideal[j]), "floor": float(ctx.floor[j]), "points": pts, "models": mods}
    return out


def hw_like_analysis(tgt: SimTarget, results: Sequence[ConfigResult], boot_n: int = 200, seed: int = 0) -> Optional[dict]:
    """zne.analyze_counts on the synthetic hardware-like samples (read-out noisy counts, read-out corrected exactly, as the
    analysis of the hardware data): the numbers one 4 000-shot job of this configuration would give, shot noise included."""
    if any(r.counts is None for r in results):
        return None
    counts = {FD.lam_key(r.lam_eff): r.counts for r in results}
    idx_all = np.unique(np.concatenate([np.fromiter(c, np.int64) for c in counts.values()]))
    obs = Z.build_observables(tgt.ctx.E, tgt.ctx.p, idx_all, tgt.ctx.masks, kmax=0, readout_mats=tgt.readout)
    return Z.analyze_counts(obs, counts, boot_n, seed, tgt.label)


def measured_points(job_dir: Path, label: str, readout: Optional[Path] = None, boot_n: int = 200, seed: int = 0) -> dict:
    """The hardware numbers of a stored zne job for the same observables (mean energy, F_E, F_XEB, 1 % / 10 % tails and, for
    components, P_ground), read-out corrected with `readout` (a readout_cal.json) if given: zne.analyze_counts on its counts, at
    the lambdas of the job (CZ folding, achieved lambda = lambda)."""
    from .analysis import load_calibration
    job_dir = Path(job_dir)
    tgt = load_target(job_dir, label)
    man = load_json(job_dir / "manifest.json")
    counts: Dict[float, Dict[int, int]] = {}
    for p in man["pubs"]:
        if p["target"] != label or p.get("arm"):
            continue
        c = {int(k): int(v) for k, v in load_json(job_dir / f"pub_{p['k']}_counts.json")["counts"].items()}
        tot = counts.setdefault(FD.lam_key(p["lam"]), {})
        for k, v in c.items():
            tot[k] = tot.get(k, 0) + v
    mats = None
    if readout is not None:
        mats, _ = Z._readout_mats(load_calibration(readout), man, [p for p in man["pubs"] if p["target"] == label])
    idx_all = np.unique(np.concatenate([np.fromiter(c, np.int64) for c in counts.values()]))
    obs = Z.build_observables(tgt.ctx.E, tgt.ctx.p, idx_all, tgt.ctx.masks, kmax=0, readout_mats=mats)
    return Z.analyze_counts(obs, counts, boot_n, seed, label)


def calibrate_phi(tgt: SimTarget, measured_fxeb: float, grid: Sequence[float], trajectories: int = 200, seed: int = 0,
                  error_scale: float = 1.0) -> dict:
    """The smallest coherent ZZ phase phi (rad per CZ) in `grid` at which the simulated lambda = 1 F_XEB (read-out corrected)
    reaches `measured_fxeb`, by linear interpolation between the grid points that bracket it (the same fault patterns at every
    phi, so that the curve is smooth).  F_XEB(phi) is not monotone: a small coherent error can raise it (it sharpens the
    distribution), so the first crossing from the side of phi = 0 is taken.  Returns the curve and phi (None: never reached)."""
    grid = sorted(float(x) for x in grid)
    curve = []
    for phi in grid:
        r = simulate_config(tgt, 1.0, "cz", phi, False, trajectories, 1, seed, error_scale=error_scale)
        curve.append({"phi": phi, "F_XEB": float(r.value[2]), "F_E": float(r.value[1]), "seconds": r.seconds})
        log.info("[noisesim] %s calibration phi %.4f: F_XEB %.4f F_E %.4f", tgt.label, phi, r.value[2], r.value[1])
    f = np.array([c["F_XEB"] for c in curve])
    side = np.sign(f[0] - measured_fxeb)
    phi_fit = None
    for i in range(1, len(grid)):
        if np.sign(f[i] - measured_fxeb) != side and side != 0:
            t = (measured_fxeb - f[i - 1]) / (f[i] - f[i - 1])
            phi_fit = grid[i - 1] + t * (grid[i] - grid[i - 1])
            break
    return {"target": tgt.label, "measured_F_XEB": float(measured_fxeb), "curve": curve, "phi": phi_fit, "error_scale": error_scale}


# ---------------------------------------------------------------------------
# the prediction
# ---------------------------------------------------------------------------
def predict_arms(job_dir: Path, arms: Sequence = ("i", "ii", "iii"), lambdas: Optional[Sequence[float]] = None, shots: int = 4000,
                 targets: Optional[Sequence[str]] = None, phis=(0.0,), trajectories: int = 200,
                 subsets: int = 1, seed: int = 0, hw_like: Optional[bool] = None, out_dir: Optional[Path] = None,
                 boot: int = BOOT_TRAJ, error_scale: float = 1.0) -> dict:
    """Predicted F_E, F_XEB and tails per lambda and the extrapolations of every arm for the targets of a stored zne job folder.

    arms: names of ARMS ('i' CZ folding, 'ii' all-gate folding, 'iii' twirled CZ folding, '0' CZ folding at 1, 3, 5 as measured)
    or Arm objects; `lambdas` overrides the arms' lambda lists (lambda = 1 must be in it); phis: coherent ZZ phases (rad) for every
    target, or {target: [phi, ...]}; results {target: {phi: {arm: extrapolate_arm record}}}.  Identical configurations are
    simulated once (the unfolded lambda = 1 circuit, lambda = 3 of arms 0 and i, and arm iii at phi = 0, where twirling has no
    effect).  `shots`: size of the synthetic hardware-like samples (hw_like; default on up to 16 qubits)."""
    job_dir = Path(job_dir)
    man = load_json(job_dir / "manifest.json")
    labels = list(targets) if targets else sorted({p["target"] for p in man["pubs"]})
    arm_list = [get_arm(a, lambdas) for a in arms]
    out = {"job_dir": str(job_dir), "backend": man.get("backend"), "calibration_id": man.get("calibration_id"), "shots": shots,
           "arms": {a.name: {"gates": a.gates, "twirled": a.twirled, "lambdas": list(a.lams)} for a in arm_list},
           "phis": phis if isinstance(phis, dict) else [float(x) for x in phis], "trajectories": trajectories, "subsets": subsets, "seed": seed,
           "error_scale": error_scale, "method": "Aer statevector trajectories (exact distribution of each trajectory), Pauli faults + optional coherent RZZ",
           "targets": {}}
    for label in labels:
        tgt = load_target(job_dir, label)
        use_hw = (tgt.n <= 16) if hw_like is None else hw_like
        out["targets"][label] = {"n_qubits": tgt.n, "kind": tgt.kind, "measured_qubits": tgt.measured, "phis": {}}
        cache: Dict[tuple, ConfigResult] = {}
        for phi in (phis[label] if isinstance(phis, dict) else phis):
            rec_phi: Dict[str, dict] = {}
            for arm in arm_list:
                rs = []
                for lam in arm.lams:
                    key = (lam, arm.gates if lam != 1.0 else "cz", arm.twirled and float(phi) != 0.0, float(phi))
                    if key not in cache:
                        cache[key] = simulate_config(tgt, lam, key[1], phi, key[2], trajectories, subsets, seed,
                                                     shots=shots if use_hw else 0, error_scale=error_scale)
                        r = cache[key]
                        log.info("[noisesim] %s arm %s lam %.3f (%d CZ) phi %.4f: F_XEB %.4f F_E %.4f raw F_XEB %.4f | P0 %.3f, "
                                 "%.1f faults, %d traj, %.0f s", label, arm.name, r.lam_eff, r.n_cz, phi, r.value[2], r.value[1],
                                 r.raw[2], r.p0, r.expected_faults, r.trajectories, r.seconds)
                    rs.append(cache[key])
                rec = extrapolate_arm(rs, tgt.ctx, boot, seed)
                if use_hw:
                    rec["hw_like"] = hw_like_analysis(tgt, rs, seed=seed)
                rec_phi[arm.name] = rec
            out["targets"][label]["phis"][f"{phi:g}"] = rec_phi
            if out_dir is not None:
                save_json(Path(out_dir) / "prediction.json", out)
    return out


# ---------------------------------------------------------------------------
# command line:  python -m hsf2.hw.noisesim <zne job folder> --targets K48 --out results/fold_prediction_12nov
# ---------------------------------------------------------------------------
def summary_rows(label: str, pred_t: dict, measured: Optional[dict]) -> List[dict]:
    """One row per (phi, arm, observable, model): the predicted extrapolation next to the predicted lambda = 1 point and the
    measured lambda = 1 point and extrapolation of the stored job (same read-out correction)."""
    rows = []
    for phi, arms in pred_t["phis"].items():
        for arm, rec in arms.items():
            for name in ("F_E", "F_XEB", "P_best_10pct"):
                o = rec["observables"].get(name)
                if o is None:
                    continue
                m = measured["observables"].get(name) if measured else None
                lam1 = o["points"][str(rec["lambdas"][0])]
                for model, r in o["models"].items():
                    rows.append({"target": label, "phi": float(phi), "arm": arm, "observable": name, "model": model,
                                 "lambdas": ",".join(str(x) for x in rec["lambdas"]), "pred_lam1": lam1["value"],
                                 "pred_extrapolated": r["value"], "pred_err": r["err"], "ideal": o["ideal"],
                                 "meas_lam1": None if m is None else m["points"]["1"]["value"],
                                 "meas_extrapolated_" + model: None if m is None else m["models"][model]["value"]})
    return rows


def compact_table(out_dirs: Sequence[Path]) -> List[dict]:
    """One row per (target, phi, arm) from finished prediction folders (prediction.json + measured.json): predicted lambda = 1
    point, predicted extrapolations (exponential and linear) of F_XEB, F_E and the 10 % tail with their trajectory errors, and the
    measured lambda = 1 value and 12 November extrapolations (exponential, linear, Richardson) of the same quantities."""
    rows = []
    for d in map(Path, out_dirs):
        pred, meas = load_json(d / "prediction.json"), load_json(d / "measured.json")
        for label, t in pred["targets"].items():
            for phi, arms in t["phis"].items():
                for arm, rec in arms.items():
                    row = {"target": label, "phi": float(phi), "arm": arm, "gates": pred["arms"][arm]["gates"],
                           "twirled": pred["arms"][arm]["twirled"], "lambdas": ",".join(f"{x:g}" for x in rec["lambdas"]),
                           "n_cz_max": max(rec["n_cz"])}
                    for name, short in (("F_XEB", "xeb"), ("F_E", "fe"), ("P_best_10pct", "t10")):
                        o = rec["observables"].get(name)
                        if o is None:
                            continue
                        row[f"{short}_pred_lam1"] = o["points"][str(rec["lambdas"][0])]["value"]
                        for m in ("exponential", "linear", "richardson"):
                            row[f"{short}_pred_{m[:3]}"] = o["models"][m]["value"]
                            row[f"{short}_pred_{m[:3]}_err"] = o["models"][m]["err"]
                        mo = meas["observables"].get(name)
                        if mo is not None:
                            row[f"{short}_meas_lam1"] = mo["points"]["1"]["value"]
                            for m in ("exponential", "linear", "richardson"):
                                row[f"{short}_meas_{m[:3]}"] = mo["models"][m]["value"]
                        row[f"{short}_ideal"] = o["ideal"]
                    rows.append(row)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import pandas as pd
    from ..utils import code_version, save_csv, setup_logging, stamp, unique_dir
    ap = argparse.ArgumentParser(description="0-QPU noise-simulation prediction of the folding diagnosis (arms i, ii, iii)")
    ap.add_argument("job_dir", help="a finished zne job folder <run>/hw/<stamp>_zne_b_1 (QPY, layouts, snapshot_pre.json, counts)")
    ap.add_argument("--targets", nargs="+", required=True, help="target labels of the job, e.g. K48 event000001332_s2_f1_c31")
    ap.add_argument("--arms", nargs="+", default=["0", "i", "ii", "iii"], choices=sorted(ARMS))
    ap.add_argument("--out", required=True, help="parent folder of the output folder (a new <stamp>_<target> folder is made in it)")
    ap.add_argument("--readout", default=None, help="readout_cal.json of the job's qubits (measured numbers are read-out corrected)")
    ap.add_argument("--trajectories", type=int, default=200)
    ap.add_argument("--cal-trajectories", type=int, default=0, help="trajectories per phi of the calibration scan (default: --trajectories)")
    ap.add_argument("--grid", type=float, nargs="+", default=[0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25])
    ap.add_argument("--phis", type=float, nargs="+", default=None, help="skip the calibration and use these phases (always add 0)")
    ap.add_argument("--error-scale", type=float, default=1.0, help="multiplies every incoherent fault probability (default 1)")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--hw-like", action="store_true", help="also analyse a synthetic hardware-like sample (default: up to 16 qubits)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=BOOT_TRAJ)
    args = ap.parse_args(argv)
    job = Path(args.job_dir)
    for label in args.targets:
        out = unique_dir(Path(args.out), f"{stamp()}_{label}")
        setup_logging("INFO", out / "log.txt")
        save_json(out / "config.json", {"args": vars(args), "code": code_version(), "label": label,
                                        "note": "0-QPU noise simulation, see the docstring of hsf2/hw/noisesim.py"})
        tgt = load_target(job, label)
        ro = Path(args.readout) if args.readout else None
        meas = measured_points(job, label, ro, seed=args.seed)
        save_json(out / "measured.json", meas)
        phis = [0.0] + [float(x) for x in (args.phis or []) if float(x) != 0.0]
        if args.phis is None:
            cal = calibrate_phi(tgt, meas["observables"]["F_XEB"]["points"]["1"]["value"], args.grid,
                                args.cal_trajectories or args.trajectories, args.seed, args.error_scale)
            save_json(out / "calibration.json", cal)
            phis = [0.0] + ([cal["phi"]] if cal["phi"] else [])
            log.info("[noisesim] %s: coherent phase reproducing the measured lambda = 1 F_XEB %.3f: %s", label,
                     cal["measured_F_XEB"], cal["phi"])
        pred = predict_arms(job, args.arms, None, args.shots, [label], {label: phis}, args.trajectories, 1, args.seed,
                            True if args.hw_like else None, out, args.boot, args.error_scale)
        rows = summary_rows(label, pred["targets"][label], meas)
        save_csv(pd.DataFrame(rows), out / "summary.csv")
        log.info("[noisesim] %s done: %s", label, out)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

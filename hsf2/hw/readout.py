"""Readout calibration and readout-error mitigation (mthree) for the hardware jobs.

Calibration is run by our own runner (so it goes through the ledger): |0...0> and |1...1> on the same physical qubits as
the experiment, `shots` shots each, once per session or day, with THE SAME measurement-twirling switch as the experiment
(a twirled calibration must never be applied to bare counts, or the reverse).  The per-qubit 2x2 matrices
  M_q[i, j] = P(read i | prepared j)
are estimated from the two circuits and handed to mthree with M3Mitigation(system).cals_from_matrices (a list with one
entry per qubit of the device, None for qubits that were not calibrated).  mthree's cals_from_system is never used: it
submits its own jobs and would bypass the ledger.

mthree 2.8.1 imports qiskit.providers.backend.BackendV1, which qiskit >= 2.0 no longer has; mthree 3.0.0 works with
qiskit 2.x but ships no manylinux wheel for Python 3.11.  When mthree cannot be imported, `correct_counts` falls back
to the same reduced-matrix solve in numpy (A restricted to the observed bit strings, exact solve, nearest probability
distribution), which is as accurate but slower; `mitigation_backend()` says which one ran.

Bit order: a sample index has classical bit k in bit k; layout['measured']['qubits'][k] is the physical qubit of
classical bit k.  mthree takes counts as bit strings (bit 0 rightmost) and `qubits` in classical-bit order.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


def mthree_status() -> Tuple[Optional[str], str]:
    """(version or None, reason): is mthree importable (not merely installed)?"""
    try:
        import mthree
        from mthree import M3Mitigation  # noqa: F401
        return str(getattr(mthree, "__version__", "?")), "ok"
    except Exception as exc:      # ImportError, or the BackendV1 ImportError of mthree 2.8.1 on qiskit 2
        return None, f"{type(exc).__name__}: {exc}"


def mitigation_backend() -> str:
    v, _ = mthree_status()
    return f"mthree {v}" if v else "numpy reduced-matrix solve"


# ---------------------------------------------------------------------------
# calibration circuits and matrices
# ---------------------------------------------------------------------------
def calibration_circuits(qubits: Sequence[int], n_device: int):
    """[all-zeros, all-ones] circuits on the device's physical qubits `qubits`, measuring into one register 'meas' of
    len(qubits) bits in the order of `qubits` (classical bit k <- qubits[k]).  They only contain x and measure, so they
    are already in the ISA of every IBM backend."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    qubits = [int(q) for q in qubits]
    if len(set(qubits)) != len(qubits):
        raise ValueError("duplicate qubits in the readout calibration")
    if not qubits or min(qubits) < 0 or max(qubits) >= n_device:
        raise ValueError(f"qubits {qubits} do not fit a {n_device}-qubit device")
    out = []
    for ones in (False, True):
        qr, cr = QuantumRegister(n_device, "q"), ClassicalRegister(len(qubits), "meas")
        qc = QuantumCircuit(qr, cr, name="readout_ones" if ones else "readout_zeros")
        if ones:
            for q in qubits:
                qc.x(q)
        qc.barrier(qubits)
        for k, q in enumerate(qubits):
            qc.measure(q, k)
        out.append(qc)
    return out


def matrices_from_counts(counts0: Dict[int, int], counts1: Dict[int, int], n_bits: int) -> List[np.ndarray]:
    """Per-bit 2x2 readout matrices M[i, j] = P(read i | prepared j) from the all-zeros and all-ones counts
    ({sample index: shots}, bit k of the index = classical bit k)."""
    mats = []
    for k in range(n_bits):
        m = np.zeros((2, 2))
        for j, counts in ((0, counts0), (1, counts1)):
            tot = float(sum(counts.values()))
            if tot <= 0:
                raise ValueError("empty calibration counts")
            ones = float(sum(v for idx, v in counts.items() if (int(idx) >> k) & 1))
            m[1, j], m[0, j] = ones / tot, 1.0 - ones / tot
        mats.append(m)
    return mats


def calibration_record(qubits: Sequence[int], counts0: Dict[int, int], counts1: Dict[int, int], shots: int,
                       measure_twirl: bool, session: str, backend: str, job_id: Optional[str]) -> dict:
    """The readout_cal.json content: per physical qubit its 2x2 matrix and the probabilities p(1|0), p(0|1)."""
    mats = matrices_from_counts(counts0, counts1, len(qubits))
    return {"schema": 1, "session": session, "backend": backend, "job_id": job_id, "shots": int(shots),
            "measure_twirl": bool(measure_twirl), "qubits": [int(q) for q in qubits],
            "matrices": {str(q): m.tolist() for q, m in zip(qubits, mats)},
            "p1_given_0": {str(q): float(m[1, 0]) for q, m in zip(qubits, mats)},
            "p0_given_1": {str(q): float(m[0, 1]) for q, m in zip(qubits, mats)}}


def matrices_of(cal: dict) -> Tuple[List[int], List[np.ndarray]]:
    qs = [int(q) for q in cal["qubits"]]
    return qs, [np.array(cal["matrices"][str(q)], float) for q in qs]


def calibration_for(cal: dict, qubits: Sequence[int], measure_twirl: bool, backend: Optional[str] = None,
                    subset: bool = False) -> dict:
    """The calibration record to apply to counts that were measured on the physical `qubits` (classical-bit order) with the
    measurement-twirling switch `measure_twirl`.  A twirled calibration must never be applied to bare counts, or the reverse;
    the calibration's qubit list must be the measured qubits (subset=True: it may cover more qubits, and the matrices of
    `qubits` are taken in the given order; the matrices are per qubit, so a larger calibration is as good); `backend`, if
    given and recorded in the calibration, must agree.  Raises ValueError, never silently skips."""
    if bool(cal["measure_twirl"]) != bool(measure_twirl):
        raise ValueError(f"calibration measurement twirl {cal['measure_twirl']} != job setting {measure_twirl}: "
                         "a twirled calibration must never be mixed with bare counts")
    if backend and cal.get("backend") and str(cal["backend"]) != str(backend):
        raise ValueError(f"calibration of backend {cal['backend']} applied to counts of backend {backend}")
    qs = [int(q) for q in qubits]
    cq = [int(q) for q in cal["qubits"]]
    if cq == qs:
        return cal
    if not subset:
        raise ValueError(f"calibration qubits {cal['qubits']} differ from the job's measured qubits {qs}")
    missing = [q for q in qs if str(q) not in cal["matrices"]]
    if missing:
        raise ValueError(f"calibration (qubits {cq}) has no matrix for the measured qubits {missing}")
    return dict(cal, qubits=qs, matrices={str(q): cal["matrices"][str(q)] for q in qs})


# ---------------------------------------------------------------------------
# correction
# ---------------------------------------------------------------------------
def index_counts_to_bitstrings(counts: Dict[int, int], n_bits: int) -> Dict[str, int]:
    return {format(int(k), f"0{n_bits}b"): int(v) for k, v in counts.items()}


def nearest_probability_distribution(q: np.ndarray) -> np.ndarray:
    """L2-nearest probability vector to a quasi-distribution (Smolin, Gambetta, Smith 2012)."""
    q = np.asarray(q, float)
    order = np.argsort(q)
    sorted_q = q[order]
    acc, out = 0.0, np.zeros_like(q)
    n = q.size
    for i, v in enumerate(sorted_q):
        if v + acc / (n - i) < 0:
            acc += v
            continue
        out[order[i:]] = sorted_q[i:] + acc / (n - i)
        break
    return out


def correct_numpy(counts: Dict[int, int], qubit_mats: Sequence[np.ndarray]) -> Tuple[Dict[int, float], float]:
    """Reduced-matrix M3 correction in numpy.  `qubit_mats[k]` is the 2x2 matrix of classical bit k.
    Returns ({index: probability}, mitigation overhead = 1-norm of the quasi-distribution inverse, i.e. sum|q|)."""
    idx = np.array(sorted(counts), np.int64)
    p_obs = np.array([counts[int(i)] for i in idx], float)
    p_obs /= p_obs.sum()
    S = idx.size
    A = np.ones((S, S))
    for k, m in enumerate(qubit_mats):
        bits = ((idx >> k) & 1).astype(np.intp)
        A *= m[bits[:, None], bits[None, :]]            # A[x, y] = prod_k M_k[x_k, y_k] = P(read x | true y)
    A /= A.sum(axis=0, keepdims=True)                   # columns renormalised on the observed subspace (as M3)
    q = np.linalg.solve(A, p_obs)
    return ({int(i): float(v) for i, v in zip(idx, nearest_probability_distribution(q))}, float(np.abs(q).sum()))


def tensor_inverse(counts: Dict[int, int], qubit_mats: Sequence[np.ndarray]) -> np.ndarray:
    """Quasi-probabilities over ALL 2^n states: (A_{n-1}^-1 x ... x A_0^-1) p_obs, with `qubit_mats[k]` the 2x2 matrix
    P(read | true) of classical bit k (bit k of the index).  Exact for the uncorrelated read-out model and, unlike the
    reduced-subspace M3 solve, not restricted to the observed bit strings: a broad sample (e.g. 4 000 shots over 2^24
    states, almost no two of them one flip apart) is corrected too.  Entries can be negative; they sum to 1.  Memory:
    a few 2^n float64 vectors (n = 24: about 0.5 GB)."""
    n = len(qubit_mats)
    v = np.zeros(1 << n)
    for k, c in counts.items():
        v[int(k)] += c
    v /= v.sum()
    return tensor_apply(v, [np.linalg.inv(np.asarray(m, float)) for m in qubit_mats])


def tensor_apply(vec: np.ndarray, ops: Sequence[np.ndarray]) -> np.ndarray:
    """(ops[n-1] x ... x ops[0]) vec for a 2^n vector, ops[k] the 2x2 operator on bit k of the index.  With ops = A_k^-1
    it inverts the read-out of a distribution (tensor_inverse); with ops = (A_k^-1)^T it turns an observable O into the
    per-state values O~ whose plain shot mean is the read-out-corrected expectation of O (exact, so a shot bootstrap of
    O~ still works)."""
    n = len(ops)
    if vec.size != 1 << n:
        raise ValueError(f"vector of length {vec.size} for {n} bits")
    t = np.asarray(vec, np.float64).reshape([2] * n)      # C order: axis a <-> bit n - 1 - a of the index
    for k, m in enumerate(ops):
        a = n - 1 - k
        t = np.moveaxis(np.tensordot(np.asarray(m, float), t, axes=([1], [a])), 0, a)
    return np.ascontiguousarray(t).reshape(-1)


def correct_mthree(counts: Dict[int, int], qubits: Sequence[int], mats: Sequence[np.ndarray], n_device: int,
                   system=None) -> Tuple[Dict[int, float], float]:
    """M3Mitigation(system).cals_from_matrices(list of length n_device).apply_correction(counts, qubits=...)."""
    from mthree import M3Mitigation
    cals: List[Optional[np.ndarray]] = [None] * int(n_device)
    for q, m in zip(qubits, mats):
        cals[int(q)] = np.asarray(m, float)
    mit = M3Mitigation(system)
    mit.cals_from_matrices(cals)
    # full Hamming distance, the same reduced system as correct_numpy (mthree 3.0 truncates at min(n, 3) when no
    # distance is given, although its docstring says n), so the result does not depend on which backend is installed
    out = mit.apply_correction(index_counts_to_bitstrings(counts, len(qubits)), qubits=[int(q) for q in qubits],
                               distance=len(qubits), return_mitigation_overhead=True)
    # mthree 3.0: the quasi-distribution carries .mitigation_overhead; older versions may return (quasi, overhead)
    quasi, overhead = (out if isinstance(out, tuple) else (out, getattr(out, "mitigation_overhead", None)))
    probs = dict(quasi.nearest_probability_distribution())
    return ({int(k, 2) if isinstance(k, str) else int(k): float(v) for k, v in probs.items()},
            float("nan") if overhead is None else float(overhead))


def correct_counts(counts: Dict[int, int], cal: dict, n_device: int, system=None, prefer: str = "auto"
                   ) -> Tuple[Dict[int, float], float, str]:
    """Mitigated probability distribution of `counts` (sample-index keys) with the calibration record `cal` (its qubit
    list must be the experiment's measured qubits in classical-bit order).  Returns (probabilities, overhead, backend)."""
    qubits, mats = matrices_of(cal)
    if prefer not in ("auto", "mthree", "numpy"):
        raise ValueError("prefer must be auto, mthree or numpy")
    version, why = mthree_status()
    if prefer == "mthree" and version is None:
        raise RuntimeError(f"mthree requested but not usable: {why}")
    if prefer != "numpy" and version is not None:
        probs, ov = correct_mthree(counts, qubits, mats, n_device, system)
        return probs, ov, f"mthree {version}"
    if prefer == "auto" and version is None:
        log.warning("[hw] mthree not usable (%s): using the numpy reduced-matrix solve", why)
    probs, ov = correct_numpy(counts, mats)
    return probs, ov, "numpy"


def mean_energy(probs: Dict[int, float], energies: np.ndarray) -> float:
    idx = np.array(list(probs), np.int64)
    p = np.array(list(probs.values()), float)
    return float(p @ energies[idx] / p.sum())

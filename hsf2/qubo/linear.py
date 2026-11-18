"""QX4: linear-system track finding (Nicotra et al., JINST 18 (2023) P11028; TrackHHL, Chiotopoulos et al.,
EPJ Web Conf. 337 (2025) 01181, arXiv:2511.11458) -- the smallest meaningful emulation, with gate counts.

Problem (TrackHHL Eqs. 1-4).  Segments S_ab = hit pairs on consecutive layers.  Modified Denby-Peterson Hamiltonian
    H(S) = -1/2 sum_{abc} f(theta_abc) S_ab S_bc + alpha sum S_ab^2 + beta sum (1 - S_ab)^2,
    f = 1 if cos(theta_abc) >= 1 - eps else 0   (theta_abc: angle between segments ab and bc).
Relaxing S to real values, grad H = 0 is the linear system A S = b with
    A = (alpha + beta) I - C / 4,   b = beta (1, ..., 1),
C the symmetric 0/1 matrix of aligned consecutive segments (the overall factor 2 of the gradient is dropped, as in
the paper, whose b = beta 1).  A is real symmetric and sparse; it is positive definite (checked) when
alpha + beta > lambda_max(C) / 4.  Segments with S_i >= T are "active".

Algorithms compared on the same events
  classical   scipy sparse solve of A S = b, threshold T                         (the classical baseline)
  hhl-ideal   the normalised HHL output |x> = A^-1 |b> / ||A^-1 |b>|| read out with a finite number of shots
  hhl-qpe     the same with an n_c-bit phase estimation, emulated exactly in the eigenbasis of A: eigencomponent i
              is weighted by sum_k |a_k(lambda_i)|^2 C / lambda_k (QPE kernel a_k, clock values lambda_k)
  onebit      the 1-bit filter of TrackHHL: one ancilla, H - controlled e^{iAt} - H, post-select the ancilla on 0;
              eigencomponent i is weighted by (1 + e^{i lambda_i t}) / 2, i.e. |.|^2 = cos^2(lambda_i t / 2); t chosen
              so that isolated segments (C-eigenvalue 0, A-eigenvalue alpha + beta) land on cos = 0
Readout: shots from |amplitude|^2 over segments; a segment is active if its sampled frequency >= T_q / N_segments.

Resources.  The circuits are built with Qiskit (state preparation by Hadamards on ceil(log2 N) qubits, A padded with
(alpha + beta) on the extra basis states, e^{iAt} by one first-order Lie-Trotter step of the Pauli decomposition --
the Suzuki-Trotter simplification of TrackHHL) and transpiled to the basis {cx, rz, sx, x}; full HHL uses n_c clock
qubits, controlled powers e^{iA t 2^k}, an inverse QFT and a uniformly controlled RY.  The Pauli decomposition has up
to 4^n terms, so circuits are built only for small n; this is an emulation, not evidence for an HHL advantage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve


@dataclass
class VeloEvent:
    hits: np.ndarray          # (n, 3) x, y, z
    layer: np.ndarray
    pid: np.ndarray           # 0 = noise


def toy_velo(n_particles: int, n_layers: int = 5, spacing: float = 30.0, noise: int = 0, sigma: float = 0.0,
             seed: int = 0, cone: float = 0.3) -> VeloEvent:
    """Straight tracks from a primary vertex near the origin through planes z = spacing * (k + 1) (TrackHHL toy model:
    100 % efficiency, no field, optional resolution smearing and noise hits)."""
    rng = np.random.default_rng(seed)
    pv = np.array([rng.normal(0, 0.05), rng.normal(0, 0.05), rng.normal(0, 5.0)])
    th = rng.uniform(0.02, cone, n_particles)
    ph = rng.uniform(-np.pi, np.pi, n_particles)
    d = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    H, L, P = [], [], []
    for k in range(n_layers):
        z = spacing * (k + 1)
        t = (z - pv[2]) / d[:, 2]
        xy = pv[:2] + t[:, None] * d[:, :2] + rng.normal(0, sigma, (n_particles, 2))
        H.append(np.c_[xy, np.full(n_particles, z)])
        L.append(np.full(n_particles, k))
        P.append(np.arange(1, n_particles + 1))
    for _ in range(noise):
        k = int(rng.integers(n_layers))
        r = rng.uniform(0, spacing * (k + 1) * math.tan(cone))
        a = rng.uniform(-np.pi, np.pi)
        H.append(np.array([[r * math.cos(a), r * math.sin(a), spacing * (k + 1)]]))
        L.append(np.array([k]))
        P.append(np.array([0]))
    return VeloEvent(np.concatenate(H), np.concatenate(L), np.concatenate(P))


def segments(ev: VeloEvent) -> np.ndarray:
    """All hit pairs on consecutive layers (inner, outer)."""
    out = []
    for k in range(int(ev.layer.max())):
        a, b = np.nonzero(ev.layer == k)[0], np.nonzero(ev.layer == k + 1)[0]
        out.append(np.stack(np.meshgrid(a, b, indexing="ij"), -1).reshape(-1, 2))
    return np.concatenate(out) if out else np.zeros((0, 2), np.int64)


def build_system(ev: VeloEvent, seg: np.ndarray, alpha: float = 0.25, beta: float = 0.5, eps: float = 1e-4):
    """A (sparse), b, C and the truth label of every segment."""
    n = len(seg)
    by_inner: Dict[int, List[int]] = {}
    for i, (a, _) in enumerate(seg):
        by_inner.setdefault(int(a), []).append(i)
    rows, cols = [], []
    for i, (a, b) in enumerate(seg):
        v1 = ev.hits[b] - ev.hits[a]
        for j in by_inner.get(int(b), []):
            v2 = ev.hits[seg[j, 1]] - ev.hits[b]
            c = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            if c >= 1.0 - eps:
                rows += [i, j]
                cols += [j, i]
    C = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = (alpha + beta) * identity(n, format="csr") - 0.25 * C
    b = np.full(n, beta)
    truth = (ev.pid[seg[:, 0]] == ev.pid[seg[:, 1]]) & (ev.pid[seg[:, 0]] > 0)
    return A, b, C, truth


def spectrum_check(A) -> Dict[str, float]:
    w = np.linalg.eigvalsh(A.toarray())
    return {"lambda_min": float(w.min()), "lambda_max": float(w.max()), "condition_number": float(w.max() / w.min()),
            "positive_definite": bool(w.min() > 0), "dimension": int(A.shape[0])}


def classical_solution(A, b) -> np.ndarray:
    return spsolve(A.tocsc(), b)


def eigen_amplitudes(A, b, method: str, t: Optional[float] = None, n_clock: int = 4,
                     alpha_beta: Optional[float] = None) -> Tuple[np.ndarray, dict]:
    """Normalised output amplitudes over segments of the ideal HHL, the n_c-bit-QPE HHL or the 1-bit filter."""
    w, U = np.linalg.eigh(A.toarray())
    c = U.T @ (b / np.linalg.norm(b))
    info: dict = {}
    if method == "hhl-ideal":
        g = 1.0 / w
    elif method == "hhl-qpe":
        t = t if t is not None else 2 * np.pi * (1 - 2.0 ** -n_clock) / w.max()     # largest lambda -> top clock value
        M = 2 ** n_clock
        k = np.arange(M)
        phi = (w * t / (2 * np.pi)) % 1.0
        # QPE kernel |a_k(phi)|^2 = |sum_j e^{2 pi i j (phi - k/M)}|^2 / M^2
        delta = phi[:, None] - k[None, :] / M
        num = np.sin(np.pi * M * delta)
        den = M * np.sin(np.pi * delta)
        p = np.where(np.abs(den) < 1e-12, 1.0, (num / np.where(np.abs(den) < 1e-12, 1.0, den)) ** 2)
        lam_k = np.where(k == 0, np.nan, 2 * np.pi * k / (M * t))               # clock value 0 is skipped
        inv = np.nan_to_num(1.0 / lam_k)
        g = p @ inv
        info = {"t": t, "n_clock": n_clock, "clock_zero_weight": float((p[:, 0] * c ** 2).sum())}
    elif method == "onebit":
        ab = alpha_beta if alpha_beta is not None else float(np.median(np.diag(A.toarray())))
        t = t if t is not None else np.pi / ab        # isolated segments (eigenvalue alpha+beta): phase pi -> cos = 0
        g = (1.0 + np.exp(1j * w * t)) / 2.0
        info = {"t": t, "postselection_probability": float(np.sum(np.abs(c * g) ** 2))}
    else:
        raise ValueError(method)
    amp = U @ (c * g)
    norm = np.linalg.norm(amp)
    info["success_weight"] = float(norm ** 2)
    return amp / norm, info


def readout(amp: np.ndarray, shots: int, seed: int = 0, rel_threshold: float = 1.0) -> np.ndarray:
    """Active segments from `shots` samples: frequency >= rel_threshold / N (N = number of segments)."""
    p = np.abs(amp) ** 2
    p = p / p.sum()
    counts = np.random.default_rng(seed).multinomial(shots, p)
    return counts / shots >= rel_threshold / len(p)


def segment_metrics(active: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    tp = int((active & truth).sum())
    return {"efficiency": tp / max(int(truth.sum()), 1), "purity": tp / max(int(active.sum()), 1),
            "n_active": int(active.sum()), "n_true": int(truth.sum()), "n_segments": int(len(truth))}


# ---------------------------------------------------------------------------- circuits and gate counts
def _padded_operator(A, ab: float) -> Tuple[np.ndarray, int]:
    n = A.shape[0]
    q = max(1, int(math.ceil(math.log2(n))))
    M = np.eye(2 ** q) * ab
    M[:n, :n] = A.toarray()
    return M, q


def onebit_circuit(A, t: float, ab: float):
    """1-bit filter: Hadamards (|b> = uniform), ancilla H, controlled e^{iAt} (one Lie-Trotter step), H."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter
    M, q = _padded_operator(A, ab)
    op = SparsePauliOp.from_operator(M).simplify()
    qc = QuantumCircuit(q + 1)
    qc.h(range(q))
    qc.h(q)
    evo = PauliEvolutionGate(op, time=-t, synthesis=LieTrotter(reps=1))          # e^{-i(-t)A} = e^{iAt}
    qc.append(evo.control(1, annotated=False), [q] + list(range(q)))
    qc.h(q)
    return qc, {"n_pauli_terms": len(op), "system_qubits": q, "qubits": q + 1}


def hhl_circuit(A, t: float, ab: float, n_clock: int):
    """Textbook HHL: |b>, QPE with n_clock qubits (controlled e^{iA t 2^k}, one Lie-Trotter step each), inverse QFT,
    uniformly controlled RY on an ancilla (C / lambda_k), uncomputation."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate, QFTGate, RYGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter
    M, q = _padded_operator(A, ab)
    op = SparsePauliOp.from_operator(M).simplify()
    sysq = list(range(q))
    clk = list(range(q, q + n_clock))
    anc = q + n_clock
    qc = QuantumCircuit(q + n_clock + 1)
    qc.h(sysq)
    qpe = QuantumCircuit(q + n_clock)
    qpe.h(list(range(q, q + n_clock)))
    for k in range(n_clock):
        evo = PauliEvolutionGate(op, time=-t * 2 ** k, synthesis=LieTrotter(reps=1))
        qpe.append(evo.control(1, annotated=False), [q + k] + sysq)
    qpe.append(QFTGate(n_clock).inverse(), list(range(q, q + n_clock)))
    qc.append(qpe, sysq + clk)
    Mk = 2 ** n_clock
    lam = 2 * np.pi * np.arange(1, Mk) / (Mk * t)
    Cmin = lam.min()
    for k in range(1, Mk):                      # uniformly controlled rotation, one multi-controlled RY per clock value
        theta = 2 * math.asin(min(1.0, Cmin / lam[k - 1]))
        bits = [(k >> j) & 1 for j in range(n_clock)]
        for j, bt in enumerate(bits):
            if not bt:
                qc.x(clk[j])
        qc.append(RYGate(theta).control(n_clock, annotated=False), clk + [anc])
        for j, bt in enumerate(bits):
            if not bt:
                qc.x(clk[j])
    qc.append(qpe.inverse(), sysq + clk)
    return qc, {"n_pauli_terms": len(op), "system_qubits": q, "qubits": q + n_clock + 1}


def gate_counts(qc, basis: Sequence[str] = ("cx", "rz", "sx", "x"), opt: int = 1) -> Dict[str, int]:
    from qiskit import transpile
    tq = transpile(qc, basis_gates=list(basis), optimization_level=opt, seed_transpiler=0)
    ops = tq.count_ops()
    return {"total": int(sum(ops.values())), "cx": int(ops.get("cx", 0)), "depth": int(tq.depth())}


def run_study(sizes: Sequence[Tuple[int, int]], seeds: Sequence[int], alpha: float = 0.25, beta: float = 0.5,
              eps: float = 1e-4, T: float = 0.8, shots_factors: Sequence[int] = (1, 10, 100), noise_frac: float = 0.0,
              sigma: float = 0.0, circuit_max_qubits: int = 6, n_clock: int = 3) -> Dict[str, list]:
    """Efficiency / purity of the classical solve and of the quantum read-outs, and circuit resources, vs size."""
    import time
    rows, circuits = [], []
    ab = alpha + beta
    for n_p, n_l in sizes:
        for s in seeds:
            ev = toy_velo(n_p, n_l, noise=int(round(noise_frac * n_p * n_l)), sigma=sigma, seed=s)
            seg = segments(ev)
            A, b, C, truth = build_system(ev, seg, alpha, beta, eps)
            sc = spectrum_check(A) if A.shape[0] <= 3000 else {}
            t0 = time.perf_counter()
            S = classical_solution(A, b)
            t_cl = time.perf_counter() - t0
            row = {"n_particles": n_p, "n_layers": n_l, "seed": s, "n_segments": len(seg), "n_true": int(truth.sum()),
                   "nnz_A": int(A.nnz), "classical_s": t_cl, **sc,
                   **{f"classical_{k}": v for k, v in segment_metrics(S >= T, truth).items() if k in ("efficiency", "purity")}}
            for method, kw in (("hhl-ideal", {}), ("hhl-qpe", {"n_clock": n_clock}), ("onebit", {"alpha_beta": ab})):
                if A.shape[0] > 3000:
                    continue
                amp, info = eigen_amplitudes(A, b, method, **kw)
                row[f"{method}_success_weight"] = info.get("postselection_probability", info["success_weight"])
                for f in shots_factors:
                    m = segment_metrics(readout(amp, f * len(seg), seed=s), truth)
                    row[f"{method}_eff_shots{f}N"] = m["efficiency"]
                    row[f"{method}_pur_shots{f}N"] = m["purity"]
            rows.append(row)
        q = max(1, int(math.ceil(math.log2(max(len(seg), 2)))))
        if q <= circuit_max_qubits:
            qc, info = onebit_circuit(A, math.pi / ab, ab)
            g1 = gate_counts(qc)
            qh, infoh = hhl_circuit(A, math.pi / ab, ab, n_clock)
            gh = gate_counts(qh)
            circuits.append({"n_particles": n_p, "n_layers": n_l, "n_segments": len(seg), "system_qubits": q,
                             "pauli_terms": info["n_pauli_terms"], "onebit_qubits": info["qubits"],
                             "onebit_cx": g1["cx"], "onebit_total": g1["total"], "onebit_depth": g1["depth"],
                             "hhl_qubits": infoh["qubits"], "hhl_cx": gh["cx"], "hhl_total": gh["total"],
                             "hhl_depth": gh["depth"], "n_clock": n_clock})
    return {"rows": rows, "circuits": circuits}

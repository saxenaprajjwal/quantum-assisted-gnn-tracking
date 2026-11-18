"""QX6: QAOA p=1 on the same tuning problem at 16 to 32 qubits (simulation on lxplus batch nodes).

The problem.  One quadratic surrogate f(u) of the real objective -- Phase 1's fit_surrogate on every distinct real
evaluation of a QX2 closed-loop run, or on the 65,536 configurations of a QX1 landscape -- is discretised with b_p
bits for knob p: u_p = level_p / (2^b_p - 1), n = sum_p b_p qubits, and C(x) = -f(u(x)) is an exact QUBO
(qubo_mixed; for equal b_p it is Phase 1's qubo_from_surrogate).  The continuous problem is the same at every n and
only its resolution changes, so differences between sizes come from n alone.  Bits are added one knob at a time,
real knobs before the integer ones (bits_layout): 16 = 2 bits each, 24 = 3 bits each, 30 = 4 bits for the six real
knobs and 3 for max_degree/min_hits (exactly their 8 values), 32 = 4 bits each (the integer knobs then take
interpolated levels, rounded when a configuration is decoded).

Per size:
  exact        global optimum, E_max and <E>_uniform from a chunked scan of all 2^n states (ChunkedLandscape: 2^24
               states at a time, never a 2^n array), and the K lowest states
  qaoa         p = 1 with Phase-1 closed-form angles (optimize_p1, which also gives the exact <C>); the state is
               simulated with Qiskit Aer (statevector, OpenMP; single precision where single_precision_works(), else
               double) on the Phase-1 circuit; P(ground) and
               P(K lowest) are read inside Aer (save_amplitudes_squared), so the 2^n state never reaches Python
  qaoa-top24   the circuit with the 24 largest couplings (the Phase-1 hardware circuit), angles optimised for the
               full cost; its samples are scored with the full QUBO
  *-noise      the Phase-1 white-noise model p = F p_ideal + (1 - F) 2^-n, F from the two-qubit gate count of the
               circuit transpiled on a fake backend (hsf2.tuning.phase1.transpiled_2q; no hardware access)
  sa, random   Phase-1 simulated annealing (one final state per restart) and uniform sampling, as many samples as
               QAOA shots
The best sample of every method is ranked against all 2^n states (a second scan).  Optionally the exact optimum
and the QAOA pick of every size are evaluated on the real GNN objective.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..config import TRACKING_DEFAULTS, TRACKING_INTEGER, TRACKING_PARAM_NAMES
from ..utils import load_json, save_json
from .phase1 import transpiled_2q
from .space import full_space

from acts_qaoa.landscape import simulated_annealing  # noqa: E402  (Phase 1 is on the path after .phase1)
from acts_qaoa.qaoa_math import optimize_p1  # noqa: E402
from acts_qaoa.surrogate import QUBO, SurrogateFit, fit_surrogate  # noqa: E402

log = logging.getLogger(__name__)
CHUNK_BITS = 24
DEFAULT_SIZES = (16, 20, 24, 26, 28, 30, 32)
SPEC_VERSION = 1


# ---------------------------------------------------------------------------
# The problem at n qubits
# ---------------------------------------------------------------------------
def bits_layout(n: int, names: Sequence[str] = TRACKING_PARAM_NAMES, integer: Sequence[str] = TRACKING_INTEGER,
                min_bits: int = 2, max_bits: int = 4) -> List[int]:
    """Bits per knob for n qubits: min_bits each, then one bit at a time, real knobs before integer ones."""
    P = len(names)
    if not P * min_bits <= n <= P * max_bits:
        raise ValueError(f"{n} qubits: {P} knobs need {P * min_bits}..{P * max_bits}")
    order = [k for k, nm in enumerate(names) if nm not in integer] + [k for k, nm in enumerate(names) if nm in integer]
    bits, extra = [min_bits] * P, n - P * min_bits
    for _ in range(min_bits, max_bits):
        for k in order:
            if extra:
                bits[k] += 1
                extra -= 1
    return bits


def unit_from_index(idx, bits: Sequence[int]) -> np.ndarray:
    """u[..., p] = level_p / (2^b_p - 1); knob p sits on qubits sum(bits[:p]).. (least significant first)."""
    idx = np.asarray(idx, dtype=np.int64)
    out, off = [], 0
    for b in bits:
        out.append(((idx >> off) & ((1 << b) - 1)) / float((1 << b) - 1))
        off += b
    return np.stack(out, axis=-1)


def bits_from_index(idx, n: int) -> np.ndarray:
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    return ((idx[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.float64)


def params_from_index(idx: int, bits: Sequence[int], space=None) -> Dict[str, float]:
    """The configuration of state idx (integer knobs rounded by the space)."""
    return (space or full_space()).from_unit(unit_from_index(int(idx), bits).tolist())


def qubo_mixed(fit: SurrogateFit, bits: Sequence[int]) -> QUBO:
    """Exact QUBO of C(x) = -f(u(x)) with b_p bits for knob p (Phase-1 qubo_from_surrogate when all b_p are equal)."""
    c0, a, d, q = fit.components()
    bits = [int(b) for b in bits]
    if len(bits) != fit.n_params:
        raise ValueError(f"{len(bits)} bit counts for {fit.n_params} knobs")
    off = np.concatenate([[0], np.cumsum(bits)]).astype(int)
    n = int(off[-1])
    w = [(2.0 ** np.arange(b)) / (2 ** b - 1) for b in bits]
    L, W = np.zeros(n), np.zeros((n, n))
    for p, b in enumerate(bits):
        for i in range(b):
            k = off[p] + i
            L[k] += -a[p] * w[p][i] - d[p] * w[p][i] ** 2          # x^2 = x
            for j in range(i + 1, b):
                W[k, off[p] + j] += -2.0 * d[p] * w[p][i] * w[p][j]
    for p in range(len(bits)):
        for r in range(p + 1, len(bits)):
            if q[p, r] != 0.0:
                W[off[p]:off[p + 1], off[r]:off[r + 1]] += -q[p, r] * np.outer(w[p], w[r])
    return QUBO(n=n, const=-c0, L=L, W=W)


def check_qubo(qubo: QUBO, fit: SurrogateFit, bits: Sequence[int], n_check: int = 4000, seed: int = 0) -> float:
    """Max |QUBO - (-f(u))| and |Ising - QUBO| on random states; raises if the mapping is not exact."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 1 << qubo.n, size=n_check, dtype=np.int64)
    X = bits_from_index(idx, qubo.n)
    e_model = -fit.predict(unit_from_index(idx, bits))
    err = max(np.max(np.abs(qubo.energy_bits(X) - e_model)), np.max(np.abs(qubo.to_ising().energy_bits(X) - e_model)))
    if err > 1e-8 * max(1.0, float(np.max(np.abs(e_model)))):
        raise RuntimeError(f"QUBO/Ising do not reproduce the surrogate (max err {err:.2e})")
    return float(err)


# ---------------------------------------------------------------------------
# Exact statistics of all 2^n states, a chunk at a time
# ---------------------------------------------------------------------------
def _linear_all(c: np.ndarray, m: int) -> np.ndarray:
    """v[i] = sum_{k<m} c_k bit_k(i) for every i in [0, 2^m), by doubling."""
    v = np.zeros(1 << m)
    for k in range(m):
        h = 1 << k
        v[h:2 * h] = v[:h] + c[k]
    return v


class ChunkedLandscape:
    """All 2^n energies of a QUBO, 2^m at a time (m = min(n, chunk_bits)): exact statistics without a 2^n array.

    State i = (high << m) | low.  E(i) = const + E_high(high) + E_low(low) + sum_{k<m} x_k D_k(high) with
    D_k = sum_{j>=m} W_kj x_j; E_low and the linear term over the 2^m low states are built by doubling, so a chunk
    costs O(2^m) and the whole scan O(2^n).
    """

    def __init__(self, qubo: QUBO, chunk_bits: int = CHUNK_BITS) -> None:
        self.qubo, self.n = qubo, int(qubo.n)
        self.m = min(self.n, int(chunk_bits))
        self.n_states = 1 << self.n
        self._W = np.triu(qubo.W, 1)
        base = np.zeros(1 << self.m)
        for k in range(self.m):
            h = 1 << k
            base[h:2 * h] = base[:h] + qubo.L[k] + _linear_all(self._W[:k, k], k)
        self._base = base

    def chunks(self) -> Iterator[Tuple[int, np.ndarray]]:
        """(index of the first state, energies) of every chunk."""
        n, m, q, W = self.n, self.m, self.qubo, self._W
        for hi in range(1 << (n - m)):
            xh = ((hi >> np.arange(n - m)) & 1).astype(np.float64)
            e_high = q.const + float(q.L[m:] @ xh) + float(xh @ W[m:, m:] @ xh)
            yield hi << m, self._base + (_linear_all(W[:m, m:] @ xh, m) + e_high)

    def energies(self, idx) -> np.ndarray:
        return self.qubo.energy_bits(bits_from_index(idx, self.n))

    def scan(self, k_low: int = 1024) -> dict:
        """Minimum (first index on ties), maximum, uniform mean and the k_low lowest states."""
        e_min, i_min, e_max, tot = np.inf, -1, -np.inf, 0.0
        top_i, top_e = np.zeros(0, np.int64), np.zeros(0)
        for first, E in self.chunks():
            j = int(np.argmin(E))
            if E[j] < e_min:
                e_min, i_min = float(E[j]), first + j
            e_max = max(e_max, float(E.max()))
            tot += float(E.sum())
            k = min(k_low, E.size)
            part = np.argpartition(E, k - 1)[:k] if k < E.size else np.arange(E.size)
            top_i = np.concatenate([top_i, first + part.astype(np.int64)])
            top_e = np.concatenate([top_e, E[part]])
            if top_e.size > k_low:
                keep = np.argpartition(top_e, k_low - 1)[:k_low]
                top_i, top_e = top_i[keep], top_e[keep]
        o = np.lexsort((top_i, top_e))
        return {"e_min": e_min, "argmin": int(i_min), "e_max": e_max, "e_mean": tot / self.n_states,
                "top_index": top_i[o], "top_energy": top_e[o]}

    def count_below(self, values: Sequence[float], tol: float = 1e-10) -> List[int]:
        """Number of states with energy below each value - tol: the 0-based rank of a state of that energy."""
        v = np.asarray(values, float) - tol
        out = np.zeros(v.size, np.int64)
        for _, E in self.chunks():
            for k, x in enumerate(v):
                out[k] += int(np.count_nonzero(E < x))
        return out.tolist()


# ---------------------------------------------------------------------------
# QAOA p=1 with Qiskit Aer
# ---------------------------------------------------------------------------
def aer_memory_gb(n: int, precision: str = "single") -> float:
    """Statevector of n qubits: 8 bytes per amplitude in single precision, 16 in double."""
    return (8.0 if precision == "single" else 16.0) * (1 << n) / 1e9


_PROBE_SCRIPT = r"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
n, rng = 16, np.random.default_rng(0)
qc = QuantumCircuit(n)
qc.h(range(n))
for k in range(n):
    qc.rz(float(rng.normal()), k)
for k in range(n):
    for l in range(k + 1, n):
        qc.rzz(float(rng.normal()), k, l)
qc.rx(0.7, range(n))
qc.save_probabilities(label="p")
p = {}
for prec in ("double", "single"):
    sim = AerSimulator(method="statevector", precision=prec)
    p[prec] = np.asarray(sim.run(transpile(qc, sim, optimization_level=0), shots=1).result().data(0)["p"])
raise SystemExit(0 if np.allclose(p["single"], p["double"], atol=1e-6) else 3)
"""
_PROBE: Dict[str, bool] = {}


def single_precision_works(timeout_s: float = 300.0) -> bool:
    """Aer's single-precision diagonal kernels crash some builds (seen: qiskit-aer 0.17.2 on Windows, RZ/RZZ from 14
    qubits, a segmentation fault that cannot be caught in-process).  A 16-qubit dense QAOA-like circuit is run in a
    subprocess in both precisions; single is used only if it neither crashes nor disagrees with double."""
    if "single" not in _PROBE:
        import subprocess
        import sys
        try:
            r = subprocess.run([sys.executable, "-c", _PROBE_SCRIPT], capture_output=True, timeout=timeout_s)
            _PROBE["single"] = r.returncode == 0
        except subprocess.TimeoutExpired:
            _PROBE["single"] = False
        if not _PROBE["single"]:
            log.warning("[qx6] Aer single precision failed its check on this node: using double precision")
    return _PROBE["single"]


def aer_qaoa(ising, scale: float, gamma: float, beta: float, watch: Sequence[int], shots: int, seed: int,
             threads: int = 0, max_memory_mb: int = 0, precision: str = "double") -> dict:
    """Shots of the Phase-1 p=1 circuit (build_qaoa_circuit, H/scale) on Aer, and P(state) for the states in watch."""
    from acts_qaoa.circuits import bind, build_qaoa_circuit
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    h_t, j_t = ising.terms(scale)
    qc, gam, bet = build_qaoa_circuit(ising.n, h_t, j_t, p=1, measure=False)
    qc = bind(qc, gam, bet, [gamma], [beta])
    qc.save_amplitudes_squared([int(i) for i in watch], label="p_watch")
    qc.measure_all()
    opts = {"method": "statevector", "precision": precision, "seed_simulator": int(seed)}
    if threads:
        opts["max_parallel_threads"] = int(threads)
    if max_memory_mb:
        opts["max_memory_mb"] = int(max_memory_mb)
    sim = AerSimulator(**opts)
    t0 = time.time()
    res = sim.run(transpile(qc, sim, optimization_level=0), shots=int(shots), memory=True).result()
    if not res.success:
        raise RuntimeError(f"Aer failed for {ising.n} qubits: {res.status}")
    samples = np.array([int(s.replace(" ", ""), 2) for s in res.get_memory(0)], dtype=np.int64)
    return {"p_watch": np.asarray(res.data(0)["p_watch"], dtype=float), "samples": samples,
            "seconds": round(time.time() - t0, 2), "n_rzz": len(j_t)}


# ---------------------------------------------------------------------------
# One size
# ---------------------------------------------------------------------------
def _method_row(land: ChunkedLandscape, st: dict, samples: np.ndarray, p_ground: float, p_top: float,
                mean_energy: float, k_low: int, **extra) -> dict:
    E = land.energies(samples)
    j = int(np.argmin(E))
    span = st["e_max"] - st["e_min"]
    return {"best_index": int(samples[j]), "best_energy": float(E[j]),
            "best_gap": max(0.0, float((E[j] - st["e_min"]) / span)) if span > 0 else 0.0,   # rounding: >= 0
            "found_ground": bool(E[j] <= st["e_min"] + 1e-10), "n_distinct": int(np.unique(samples).size),
            "p_ground": float(p_ground), "amplification": float(p_ground * land.n_states),
            "p_top": float(p_top), "top_amplification": float(p_top * land.n_states / k_low),
            "mean_energy": float(mean_energy),
            "approx_ratio": float((st["e_max"] - mean_energy) / span) if span > 0 else 1.0, **extra}


def _noisy(rng, ideal: dict, samples: np.ndarray, F: float, st: dict, n_states: int, k_low: int) -> Tuple[np.ndarray, dict]:
    """White-noise model: a shot is the ideal circuit's with probability F, uniform otherwise."""
    shots = samples.size
    k = int(rng.binomial(shots, F))
    s = np.concatenate([samples[:k], rng.integers(0, n_states, size=shots - k, dtype=np.int64)])
    return s, {"p_ground": F * ideal["p_ground"] + (1.0 - F) / n_states,
               "p_top": F * ideal["p_top"] + (1.0 - F) * k_low / n_states,
               "mean_energy": F * ideal["mean_energy"] + (1.0 - F) * st["e_mean"]}


def run_size(fit: SurrogateFit, n: int, shots: int = 1024, seed: int = 0, k_low: int = 1024, sparse_top_k: int = 24,
             threads: int = 0, max_memory_gb: float = 0.0, noise: bool = True, noise_backend: str = "ibm_fez",
             chunk_bits: int = CHUNK_BITS, noise_fidelity: Optional[Dict[str, float]] = None,
             precision: str = "auto") -> dict:
    """Every method at n qubits (see the module docstring).  noise_fidelity={'full': F, 'top': F} skips transpiling;
    precision 'auto' = single if single_precision_works() else double."""
    if precision == "auto":
        precision = "single" if single_precision_works() else "double"
    t_all = time.time()
    rng = np.random.default_rng(seed)
    bits = bits_layout(n)
    qubo = qubo_mixed(fit, bits)
    err = check_qubo(qubo, fit, bits)
    ising = qubo.to_ising()
    scale = ising.scale()
    land = ChunkedLandscape(qubo, chunk_bits)
    t0 = time.time()
    st = land.scan(k_low)
    t_scan = time.time() - t0
    log.info("[qx6] n=%d bits %s: exact scan %.1f s, E_min %.6f (state %d), <E> %.6f, E_max %.6f", n, bits, t_scan,
             st["e_min"], st["argmin"], st["e_mean"], st["e_max"])
    watch = st["top_index"]
    res: Dict[str, dict] = {}
    res["random"] = _method_row(land, st, rng.integers(0, land.n_states, size=shots, dtype=np.int64),
                                1.0 / land.n_states, k_low / land.n_states, st["e_mean"], k_low)
    t0 = time.time()
    counts = simulated_annealing(qubo, restarts=shots, seed=seed)
    s_sa = np.repeat(np.array(list(counts), np.int64), list(counts.values()))
    E_sa = land.energies(s_sa)
    res["sa"] = _method_row(land, st, s_sa, float(np.mean(s_sa == st["argmin"])), float(np.mean(np.isin(s_sa, watch))),
                            float(E_sa.mean()), k_low, seconds=round(time.time() - t0, 2))

    need = aer_memory_gb(n, precision) * 1.15 + 1.0
    circuits = {"qaoa": ising}
    if sparse_top_k:
        sp = ising.sparsified(sparse_top_k)
        if sp.n_couplings(1e-12) < ising.n_couplings(1e-12):
            circuits[f"qaoa-top{sparse_top_k}"] = sp
    fid: Dict[str, dict] = {}
    for name, ism in circuits.items():
        if max_memory_gb and need > max_memory_gb:
            log.warning("[qx6] n=%d: %s skipped, the statevector needs ~%.1f GB > %.1f GB", n, name, need, max_memory_gb)
            res[name] = {"skipped": f"needs ~{need:.1f} GB > {max_memory_gb:.1f} GB"}
            continue
        if ism is ising:
            ang = optimize_p1(ising.h / scale, ising.full_J() / scale)
        else:     # the sparse circuit evolves under its own H; angles chosen for the FULL cost (Phase-1 practice)
            ang = optimize_p1(ism.h / scale, ism.full_J() / scale, h_obs=ising.h / scale, J_obs=ising.full_J() / scale)
        a = aer_qaoa(ism, scale, ang["gamma"], ang["beta"], watch, shots, seed, threads,
                     int(max_memory_gb * 1000) if max_memory_gb else 0, precision)
        mean = ising.offset + scale * ang["energy"]           # exact <C> of the full cost (closed form)
        res[name] = _method_row(land, st, a["samples"], a["p_watch"][0], float(a["p_watch"].sum()), mean, k_low,
                                gamma=ang["gamma"], beta=ang["beta"], seconds=a["seconds"], n_rzz=a["n_rzz"])
        log.info("[qx6] n=%d %-10s Aer %.1f s: P(ground) %.3g (x%.3g uniform), approx ratio %.4f", n, name,
                 a["seconds"], res[name]["p_ground"], res[name]["amplification"], res[name]["approx_ratio"])
        if noise or noise_fidelity:
            key = "full" if ism is ising else "top"
            if noise_fidelity:
                tr = {"fidelity_model": float(noise_fidelity[key]), "n_2q": None, "backend": "given"}
            else:
                tr = transpiled_2q(ising, scale, top_k=None if ism is ising else sparse_top_k,
                                   backend_name=noise_backend, seeds=(0,))
            F = float(tr["fidelity_model"])
            fid[name] = {k: tr.get(k) for k in ("backend", "n_2q", "depth_2q", "fidelity_model", "transpile_s")}
            s_n, nz = _noisy(rng, res[name], a["samples"], F, st, land.n_states, k_low)
            res[f"{name}-noise"] = _method_row(land, st, s_n, nz["p_ground"], nz["p_top"], nz["mean_energy"], k_low,
                                               fidelity=F, n_2q=tr.get("n_2q"))
    # rank of every method's best sample among all 2^n states
    names = [m for m in res if "best_energy" in res[m]]
    ranks = land.count_below([res[m]["best_energy"] for m in names])
    for m, r in zip(names, ranks):
        res[m]["best_rank"] = int(r)
        res[m]["best_fraction_better"] = r / land.n_states
    top = [{"index": int(i), "energy": float(e)} for i, e in zip(st["top_index"][:10], st["top_energy"][:10])]
    return {"n": n, "bits": bits, "n_states": land.n_states, "n_couplings": ising.n_couplings(1e-12),
            "scale": scale, "qubo_check_max_err": err, "aer_precision": precision,
            "e_min": st["e_min"], "argmin": st["argmin"],
            "e_max": st["e_max"], "e_mean": st["e_mean"], "k_low": k_low, "shots": shots, "seed": seed,
            "optimum_params": params_from_index(st["argmin"], bits),
            "predicted_optimum": float(fit.predict(unit_from_index(st["argmin"], bits)[None, :])[0]),
            "lowest_states": top, "methods": res, "noise_circuits": fid,
            "seconds": {"scan": round(t_scan, 1), "total": round(time.time() - t_all, 1)}}


# ---------------------------------------------------------------------------
# The surrogate and the study
# ---------------------------------------------------------------------------
STANDARD_SCORE = "eff - fake - dup/5 - t/5"


def _is_standard_loop(d: Path) -> bool:
    """A QX2 loop scored with the specified objective S (older loops record no score_definition: S)."""
    cfg = Path(d) / "config.json"
    if not cfg.exists():
        return True
    sd = load_json(cfg).get("score_definition")
    return sd is None or " ".join(str(sd).split()) == STANDARD_SCORE


def resolve_source(source: str, runs_root: Path) -> Path:
    """'qx2' -> newest finished QX2 loop (comparison.json) scored with the standard S, 'qx1' -> newest QX1 landscape,
    else a folder.  Objective-variant loops (t/20, DM, floor) are never picked by 'qx2': a queued job once resolved
    'qx2' at start time to the t/20 loop (QX6 n=32, 9 Nov 2026); name such a loop's folder explicitly instead."""
    marks = {"qx2": ("*qx2_loop*", "comparison.json"), "qx1": ("*qx1_landscape*", "landscape.npz")}
    if source in marks:
        pat, mark = marks[source]
        cands = [p.parent for p in Path(runs_root).glob(f"{pat}/{mark}")]
        if source == "qx2":
            cands = [d for d in cands if _is_standard_loop(d)]
        if not cands:
            raise FileNotFoundError(f"no {pat} folder with {mark} in {runs_root}")
        return max(cands, key=lambda d: (d / mark).stat().st_mtime)
    p = Path(source)
    if not ((p / "landscape.npz").exists() or (p / "results.json").exists()):
        raise FileNotFoundError(f"{p} is neither a QX1 landscape nor a QX2 loop folder")
    return p


def load_points(source: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Unit coordinates (full knob box) and objective of every distinct real evaluation in a QX1 landscape folder
    (all 65,536 sub-grid configurations) or a QX2 loop folder (results.json, all methods and seeds)."""
    sp = full_space()
    source = Path(source)
    if (source / "landscape.npz").exists():
        from .landscape import LandscapeTable
        tab = LandscapeTable(source)
        U = np.array([sp.to_unit(tab.enc.params_from_index(i)) for i in range(len(tab.score))])
        return U, np.asarray(tab.score, float), {"kind": "qx1-landscape", "n_events": tab.n_events}
    recs: Dict[tuple, Tuple[dict, float]] = {}
    for r in load_json(source / "results.json")["results"]:
        for x in r["records"]:
            if x["objective"] is not None:
                recs.setdefault(tuple(sorted(x["params"].items())), (x["params"], float(x["objective"])))
    if len(recs) < 50:
        raise ValueError(f"only {len(recs)} distinct evaluations in {source}/results.json")
    U = np.array([sp.to_unit(p) for p, _ in recs.values()])
    return U, np.array([y for _, y in recs.values()]), {"kind": "qx2-loop"}


def _flat(n: int, r: dict) -> List[dict]:
    rows = []
    for m, d in r["methods"].items():
        if "best_energy" in d:
            rows.append({"n": n, "method": m, **{k: d.get(k) for k in (
                "p_ground", "amplification", "top_amplification", "approx_ratio", "best_gap", "best_rank",
                "best_fraction_better", "found_ground", "fidelity", "n_2q", "seconds")}})
    return rows


def run_study(source: Path, out: Path, sizes: Sequence[int] = DEFAULT_SIZES, shots: int = 1024, seed: int = 0,
              k_low: int = 1024, sparse_top_k: int = 24, threads: int = 0, max_memory_gb: float = 0.0,
              noise: bool = True, noise_backend: str = "ibm_fez", precision: str = "auto",
              validate: Optional[Callable[[List[Tuple[str, dict]]], List[dict]]] = None, extra: Optional[dict] = None) -> dict:
    """Fit (or reuse) the surrogate, run every size not done yet (n<n>.json, resumable), validate, summarise."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = {"version": SPEC_VERSION, "source": str(source), "shots": shots, "seed": seed, "k_low": k_low,
            "sparse_top_k": sparse_top_k, "noise": noise, "noise_backend": noise_backend}
    spec_path = out / "spec.json"
    if spec_path.exists():
        old = load_json(spec_path)
        if {k: old.get(k) for k in spec} != spec:
            raise RuntimeError(f"{out} was made with other settings ({old}); use a new folder")
    else:
        save_json(spec_path, {**spec, **(extra or {})})
    sur_path = out / "surrogate.json"
    if sur_path.exists():
        fit = SurrogateFit.from_dict(load_json(sur_path)["fit"])
    else:
        U, y, info = load_points(source)
        fit = fit_surrogate(U, y, kind="quadratic", seed=0)
        save_json(sur_path, {"fit": fit.to_dict(), "n_points": int(len(y)), "y_max": float(y.max()),
                             "y_mean": float(y.mean()), **info})
    log.info("[qx6] surrogate on %d points: CV R^2 %.3f, CV RMSE %.4g", fit.n_samples, fit.cv_r2, fit.cv_rmse)
    if precision == "auto":
        precision = "single" if single_precision_works() else "double"
    log.info("[qx6] Aer precision %s", precision)
    results: Dict[int, dict] = {}
    for n in sizes:
        path = out / f"n{n}.json"
        if path.exists():
            results[n] = load_json(path)
            continue
        results[n] = run_size(fit, n, shots, seed, k_low, sparse_top_k, threads, max_memory_gb, noise, noise_backend,
                              precision=precision)
        save_json(path, results[n])
    if validate is not None and not (out / "validation.json").exists():
        todo = [("default", dict(TRACKING_DEFAULTS))]
        for n, r in results.items():
            todo.append((f"n{n}/exact", r["optimum_params"]))
            q = r["methods"].get("qaoa", {})
            if "best_index" in q:
                todo.append((f"n{n}/qaoa", params_from_index(q["best_index"], r["bits"])))
        save_json(out / "validation.json", {"rows": validate(todo)})
    rows = [row for n in sizes for row in _flat(n, results[n])]
    with open(out / "sizes.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    summ = {"surrogate": {"cv_r2": fit.cv_r2, "cv_rmse": fit.cv_rmse, "n_points": fit.n_samples},
            "sizes": {n: {k: results[n][k] for k in ("bits", "n_couplings", "e_min", "e_mean", "e_max",
                                                     "predicted_optimum", "optimum_params", "seconds")}
                      for n in sizes}, "rows": rows}
    save_json(out / "summary.json", summ)
    return summ

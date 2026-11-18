"""Spin-reversal gauges for the Ising circuits.

A gauge g in {0,1}^n (s_k = 1 - 2 g_k) maps the Hamiltonian to h_k -> s_k h_k, J_kl -> s_k s_l J_kl.  The circuit
starts in |+>^n and ends with RX mixers, both invariant under the bit flips X_k, so the gauged circuit prepares the
original state with the bits in the flipped qubits exchanged: a measured string x' belongs to the original problem as
x = x' XOR g.  Running several gauges averages the qubit-dependent readout asymmetry (T1 decay of |1>) and single
gate errors over the flips.  Gauge 0 is the identity.

Bit order: the integer index of a sample (Phase-1 counts_to_index) has qubit k in bit k.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def gauge_vectors(n: int, n_gauges: int, seed: int = 0) -> np.ndarray:
    """(n_gauges, n) uint8 array of distinct gauges; row 0 is the identity, the others are random and different from
    each other and from the identity (so n_gauges <= 2^n)."""
    if n_gauges < 1:
        raise ValueError(f"n_gauges must be >= 1, got {n_gauges}")
    if n_gauges > 2 ** min(n, 62):
        raise ValueError(f"cannot draw {n_gauges} distinct gauges on {n} qubits")
    rng = np.random.default_rng(seed)
    out = [np.zeros(n, np.uint8)]
    seen = {out[0].tobytes()}
    while len(out) < n_gauges:
        g = rng.integers(0, 2, size=n).astype(np.uint8)
        if g.tobytes() in seen:
            continue
        seen.add(g.tobytes())
        out.append(g)
    return np.array(out, np.uint8)


def gauge_mask(g: np.ndarray) -> int:
    """Gauge bits -> integer with qubit k in bit k (XOR this into a sample index)."""
    return int(sum(int(b) << k for k, b in enumerate(np.asarray(g, np.uint8))))


def gauge_signs(g: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * np.asarray(g, dtype=float)


def gauge_ising(ising, g: np.ndarray):
    """The gauged copy of a Phase-1 Ising: h -> s h, J -> s_k s_l J (offset unchanged)."""
    from .._phase1 import ensure_phase1
    ensure_phase1()
    from acts_qaoa.surrogate import Ising
    s = gauge_signs(g)
    if s.size != ising.n:
        raise ValueError(f"gauge has {s.size} entries, the Ising model has {ising.n} spins")
    return Ising(n=ising.n, offset=ising.offset, h=ising.h * s, J=ising.J * np.outer(s, s))


def xor_counts(counts: Dict[int, int], mask: int) -> Dict[int, int]:
    """Counts in the gauge frame -> counts of the original problem (index XOR mask)."""
    return {int(k) ^ int(mask): int(v) for k, v in counts.items()}


def merge_counts(parts) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for c in parts:
        for k, v in c.items():
            out[int(k)] = out.get(int(k), 0) + int(v)
    return out


def split_shots(total: int, n_gauges: int) -> int:
    """Shots per gauge; the total must divide evenly (equal shots in every PUB)."""
    if total % n_gauges:
        raise ValueError(f"{total} shots do not divide into {n_gauges} gauges with equal shots; "
                         f"use a multiple of {n_gauges}")
    return total // n_gauges


def gauge_record(g: np.ndarray) -> dict:
    return {"bits": [int(b) for b in g], "mask": gauge_mask(g)}


"""Search spaces of the tuning problem T1, expressed with the Phase-1 ParamSpace / Encoding.

24-bit space (QX2): the eight knobs of hsf2.config.TRACKING_RANGES, 3 bits each.  Phase-1 convention
(acts_qaoa.encoding): qubit k = bit k of the configuration index, knob p uses qubits 3p..3p+2 (least significant
first), u_p = level_p / 7 and value = lo + u_p (hi - lo), or lo (hi/lo)^u_p for the knobs of TRACKING_LOG
(KnobSpace); the integer knobs max_degree (1..8) and min_hits (3..10) span exactly 7, so their 8 levels are exactly
the integers.

16-bit sub-grid (QX1): every second level of every knob, 2 bits each, aligned so that the default configuration
(TRACKING_DEFAULTS, the ACTS GNN example settings) lies on it.  Knob p takes the 3-bit levels
o_p, o_p + 2, o_p + 4, o_p + 6 with o_p = (default level) mod 2, i.e. the 2-bit level l gives the 3-bit level
o_p + 2 l.  As a Phase-1 ParamSpace this is the box [v(o_p), v(o_p + 6)] (v = value of a 3-bit level, same spacing)
with a 2-bit Encoding, so every 16-bit configuration is also a 24-bit configuration and the QUBO stays exact (the
integer knobs move in steps of exactly 2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .._phase1 import ensure_phase1
from ..config import TRACKING_DEFAULTS, TRACKING_INTEGER, TRACKING_LOG, TRACKING_PARAM_NAMES, TRACKING_RANGES

ensure_phase1()
from acts_qaoa.config import ParamSpace  # noqa: E402
from acts_qaoa.encoding import Encoding  # noqa: E402

BITS_FULL = 3
BITS_SUB = 2


@dataclass
class KnobSpace(ParamSpace):
    """Phase-1 ParamSpace in which the knobs flagged in `log` are spaced geometrically: u = log(v/lo) / log(hi/lo).

    Every Phase-1 stage maps between u in [0, 1] and knob values only through to_unit / from_unit (grid, Latin-
    hypercube warm start, surrogate features, decoding of QUBO states), so they all follow this spacing with acts_qaoa
    unchanged.  The flags are stored in run_config.json (asdict).  Integer knobs stay linear.
    """
    log: List[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.log and len(self.log) != len(self.names):
            raise ValueError(f"log flags {self.log} do not match the knobs {self.names}")
        for k in range(len(self.log)):
            if self.log[k] and (self.integer[k] or not self.lo[k] > 0):
                raise ValueError(f"{self.names[k]}: geometric spacing needs a real knob with lo > 0")

    def is_log(self, k: int) -> bool:
        return bool(self.log) and bool(self.log[k])

    def to_unit(self, params: Mapping[str, float]) -> np.ndarray:
        u = super().to_unit(params)
        for k, n in enumerate(self.names):
            if self.is_log(k):
                u[k] = math.log(float(params[n]) / self.lo[k]) / math.log(self.hi[k] / self.lo[k])
        return u

    def from_unit(self, u: Sequence[float]) -> Dict[str, float]:
        out = super().from_unit(u)
        for k, n in enumerate(self.names):
            if self.is_log(k):
                uk = float(u[k])              # the end points exactly, so the default 1.0 stays on the grid
                out[n] = self.lo[k] if uk == 0.0 else self.hi[k] if uk == 1.0 else \
                    float(self.lo[k] * (self.hi[k] / self.lo[k]) ** uk)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "KnobSpace":
        return cls(list(d["names"]), [float(x) for x in d["lo"]], [float(x) for x in d["hi"]],
                   [bool(x) for x in d["integer"]], [bool(x) for x in d.get("log", [])])


def log_flags(space: ParamSpace) -> List[bool]:
    """Which knobs of a space are spaced geometrically (none for a plain Phase-1 ParamSpace)."""
    return [isinstance(space, KnobSpace) and space.is_log(k) for k in range(space.dim)]


def full_space(ranges: Mapping[str, Tuple[float, float]] = TRACKING_RANGES,
               names: Sequence[str] = TRACKING_PARAM_NAMES, log: Sequence[str] = TRACKING_LOG) -> KnobSpace:
    """The 8-knob box as a Phase-1 ParamSpace (integer flags for max_degree and min_hits, geometric deta/z0)."""
    return KnobSpace(list(names), [float(ranges[n][0]) for n in names], [float(ranges[n][1]) for n in names],
                     [n in TRACKING_INTEGER for n in names], [n in log for n in names])


def full_encoding(space: ParamSpace = None) -> Encoding:
    enc = Encoding(space or full_space(), BITS_FULL)
    bad = [n for n, ex in zip(enc.space.names, enc.exact) if not ex]
    if bad:
        raise ValueError(f"integer knobs {bad} do not span 2^{BITS_FULL}-1: the 3-bit grid would round them")
    return enc


def default_levels(enc: Encoding, defaults: Mapping[str, float] = TRACKING_DEFAULTS) -> np.ndarray:
    """3-bit levels of the default configuration; it must lie exactly on the grid."""
    lv = enc.levels_from_params(defaults)
    back = enc.params_from_levels(lv)
    for n in enc.space.names:
        if abs(float(back[n]) - float(defaults[n])) > 1e-9:
            raise ValueError(f"default {n}={defaults[n]} is not on the {enc.bits}-bit grid")
    return lv


def sub_offsets(enc_full: Encoding, defaults: Mapping[str, float] = TRACKING_DEFAULTS) -> np.ndarray:
    return default_levels(enc_full, defaults) % 2


def sub_space(enc_full: Encoding = None, defaults: Mapping[str, float] = TRACKING_DEFAULTS) -> KnobSpace:
    """The 16-bit QX1 sub-grid as a Phase-1 ParamSpace (use it with a 2-bit Encoding)."""
    enc_full = enc_full or full_encoding()
    sp = enc_full.space
    log = log_flags(sp)
    off = sub_offsets(enc_full, defaults)
    top = 2 * (2 ** BITS_SUB - 1)
    lo, hi = [], []
    for k in range(sp.dim):
        if log[k]:
            u0, u1 = off[k] / enc_full.mask, (off[k] + top) / enc_full.mask
            lo.append(sp.lo[k] if u0 == 0 else float(sp.lo[k] * (sp.hi[k] / sp.lo[k]) ** u0))
            hi.append(sp.hi[k] if u1 == 1 else float(sp.lo[k] * (sp.hi[k] / sp.lo[k]) ** u1))
            continue
        d = (sp.hi[k] - sp.lo[k]) / enc_full.mask
        lo.append(sp.lo[k] + off[k] * d)
        hi.append(sp.lo[k] + (off[k] + top) * d)
    return KnobSpace(list(sp.names), lo, hi, list(sp.integer), log)


def sub_encoding(enc_full: Encoding = None) -> Encoding:
    return Encoding(sub_space(enc_full), BITS_SUB)


def sub_to_full_levels(levels_sub: np.ndarray, enc_full: Encoding = None) -> np.ndarray:
    """3-bit levels of the 24-bit configuration equal to a 16-bit configuration."""
    enc_full = enc_full or full_encoding()
    return sub_offsets(enc_full) + 2 * np.asarray(levels_sub, dtype=np.int64)


def level_values(enc: Encoding) -> Dict[str, List[float]]:
    """The physical value of every level of every knob (the grid each knob is searched on)."""
    out = {}
    for k, n in enumerate(enc.space.names):
        vals = []
        for lv in range(enc.n_levels):
            levels = np.zeros(enc.n_params, np.int64)
            levels[k] = lv
            vals.append(enc.params_from_levels(levels)[n])
        out[n] = vals
    return out


def describe_space(enc: Encoding) -> dict:
    """JSON-ready description of a search space: box, bits, bit convention and the value of every level."""
    sp = enc.space
    return {"names": list(sp.names), "lo": list(sp.lo), "hi": list(sp.hi), "integer": list(sp.integer),
            "log": log_flags(sp), "bits_per_knob": enc.bits, "n_qubits": enc.n_qubits, "n_states": 1 << enc.n_qubits,
            "levels": level_values(enc),
            "convention": "Phase-1 acts_qaoa.encoding: qubit k = bit k of the index; knob p = qubits "
                          f"{enc.bits}p..{enc.bits}p+{enc.bits - 1}, least significant first; "
                          f"u_p = level_p/{enc.mask}; value = lo + u_p (hi - lo), or lo (hi/lo)^u_p where log is "
                          "true; Qiskit key = bin(index)"}

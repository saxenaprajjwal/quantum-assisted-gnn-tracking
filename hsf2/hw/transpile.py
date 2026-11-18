"""Transpilation of a job's circuits to the ISA of the backend (Phase-1 transpile_best, seeds, optional layout).

Normal settings: one parametrised circuit per gauge is transpiled once (gamma, beta stay parameters), and the main and
control PUBs only differ in the parameter values -- the gate sequence, hence the noise, is the same.  Gauges other than
0 are transpiled with the physical qubits of gauge 0 and its best transpile seed, so every gauge runs on the same
qubits.

Fractional setting (e): the backend has a native RZZ(theta) with theta in [0, pi/2].  The IBM transpiler cannot fold
unbound angles, so every PUB is bound first and transpiled separately (on the layout of the first one), and the runtime's
FoldRzzAngle pass is applied explicitly afterwards: qiskit-ibm-runtime 0.51's IBMBackend names the translation plugin
'ibm_dynamic_circuits' even for a fractional backend, which does not fold.  The ISA is then checked: it must contain
rzz, with every angle in (0, pi/2].
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .._phase1 import ensure_phase1
from .build import PubSpec, build_circuit
from .gauge import gauge_ising

log = logging.getLogger(__name__)

RZZ_TOL = 1e-9


@dataclass
class IsaPub:
    spec: PubSpec
    isa: object                    # transpiled circuit
    values: Optional[np.ndarray]   # parameter values (None: all angles are bound in the ISA)
    stats: dict                    # Phase-1 circuit_stats (+ transpile_seed)
    layout: dict                   # layout_record


def require_fractional_target(backend) -> None:
    ops = set(backend.target.operation_names)
    if "rzz" not in ops:
        raise RuntimeError(f"backend {getattr(backend, 'name', backend)} has no native rzz gate: setting (e) needs a "
                           "backend obtained with use_fractional_gates=True (a fractional-gate target)")


def fold_rzz(isa):
    """Fold every rzz angle of a bound ISA into [0, pi/2] (qiskit_ibm_runtime FoldRzzAngle; layout is kept)."""
    try:
        from qiskit.transpiler import PassManager
        from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle
    except ImportError as exc:
        raise RuntimeError("setting (e) needs qiskit_ibm_runtime.transpiler.passes.basis.FoldRzzAngle "
                           f"(not available in this qiskit-ibm-runtime: {exc})") from exc
    layout = getattr(isa, "_layout", None)
    out = PassManager([FoldRzzAngle()]).run(isa)
    if getattr(out, "_layout", None) is None and layout is not None:
        out._layout = layout
    return out


def rzz_angles(isa) -> List[float]:
    return [float(i.operation.params[0]) for i in isa.data if i.operation.name == "rzz"]


def check_fractional_isa(isa) -> List[float]:
    """The rzz angles of a fractional ISA; raises if there is none or one lies outside (0, pi/2]."""
    ang = rzz_angles(isa)
    if not ang:
        raise ValueError("fractional ISA contains no rzz gate")
    bad = [a for a in ang if not (0.0 < a <= math.pi / 2 + RZZ_TOL)]
    if bad:
        raise ValueError(f"{len(bad)} of {len(ang)} rzz angles outside (0, pi/2]: {bad[:5]} -- IBM rejects such a PUB")
    return ang


def layout_record(isa, n_logical: int, stats: dict) -> dict:
    """Logical -> physical mapping and the measured qubits of an ISA.

    measured.qubits[k] is the physical qubit whose result is classical bit k (bit k of a sample index), read from the
    measure instructions themselves, so it is right whatever routing did; this is the list to pass to mthree."""
    lay = getattr(isa, "layout", None)
    rec = {"n_logical": int(n_logical),
           "initial": [int(q) for q in lay.initial_index_layout(filter_ancillas=True)] if lay is not None else None,
           "final": [int(q) for q in lay.final_index_layout()] if lay is not None else None}
    pairs = sorted((isa.find_bit(c).index, isa.find_bit(q).index) for ins in isa.data
                   if ins.operation.name == "measure" for q, c in zip(ins.qubits, ins.clbits))
    rec["measured"] = {"clbits": [c for c, _ in pairs], "qubits": [q for _, q in pairs]}
    rec["active_qubits"] = [int(q) for q in stats["active_qubits"]]
    rec["edges"] = sorted({tuple(sorted(isa.find_bit(q).index for q in ins.qubits)) for ins in isa.data
                           if len(ins.qubits) == 2 and ins.operation.name != "barrier"})
    rec["edges"] = [list(e) for e in rec["edges"]]
    return rec


def _best(circuit, backend, seeds: Sequence[int], layout):
    from acts_qaoa.circuits import transpile_best
    return transpile_best(circuit, backend, seeds=list(seeds), initial_layout=layout)


def transpile_job(ism, scale: float, fractional: bool, specs: Sequence[PubSpec], gauges: np.ndarray, backend,
                  n_seeds: int = 4) -> List[IsaPub]:
    """Circuits of every PUB of a job on `backend`: see the module docstring."""
    ensure_phase1()
    from acts_qaoa.circuits import bind, circuit_stats, parameter_values
    if fractional:
        require_fractional_target(backend)
    seeds = list(range(int(n_seeds)))
    layout: Optional[List[int]] = None
    best_seed: Optional[int] = None
    cache: dict = {}
    out: List[Optional[IsaPub]] = [None] * len(specs)
    n = int(ism.n)
    for spec in specs:
        g = spec.gauge
        if g >= len(gauges):
            raise ValueError(f"PUB {spec.k} uses gauge {g}, but only {len(gauges)} gauges are defined")
        if g not in cache:
            qc, gam, bet, _, j_t = build_circuit(gauge_ising(ism, gauges[g]), scale)
            cache[g] = (qc, gam, bet)
            log.info("[hw] gauge %d circuit: %d RZZ per layer", g, len(j_t))
        qc, gam, bet = cache[g]
        use_seeds = seeds if layout is None else [best_seed]
        if fractional:
            isa, st = _best(bind(qc, gam, bet, spec.gammas, spec.betas), backend, use_seeds, layout)
            isa = fold_rzz(isa)
            check_fractional_isa(isa)
            seed = st["transpile_seed"]
            st = circuit_stats(isa, backend)
            st["transpile_seed"] = seed
            values = None
        else:
            if ("isa", g) not in cache:
                cache[("isa", g)] = _best(qc, backend, use_seeds, layout)
            isa, st = cache[("isa", g)]
            values = parameter_values(isa, gam, bet, spec.gammas, spec.betas)
        lay = layout_record(isa, n, st)
        if layout is None:
            layout, best_seed = lay["initial"], int(st["transpile_seed"])
            log.info("[hw] layout (best seed %d): %d 2q gates, depth2q %s, F_est %.3g, physical qubits %s", best_seed,
                     st["n_2q"], st.get("depth_2q"), st.get("fidelity_est", float("nan")), lay["measured"]["qubits"])
        out[spec.k] = IsaPub(spec, isa, values, st, lay)
    return [o for o in out if o is not None]

"""Local unitary folding of the gates of an ISA circuit to a fractional noise factor (folding diagnosis, 12 Nov 2026).

zne.fold_two_qubit_gates folds every CZ to lambda = 1, 3, 5: single-qubit gates stay as they are, so their error does not
scale with lambda.  fold_gates folds a chosen set of native Heron gates, G -> G (G^dagger G)^m, on a random subset of the gates
of each type, so that the effective noise factor of a type, lambda = 1 + 2 P / N (N gates of the type, P inserted G^dagger G
pairs), is as close as possible to the target (fractional lambda allowed: lambda = 1.5 folds a quarter of the gates once).

The circuit stays ISA-native on Heron (cz, sx, x, rz, measure, barrier, delay):
  CZ    is its own inverse;
  X     is its own inverse;
  SX    SX^dagger = RZ(pi) SX RZ(pi) up to a global phase (RZ is virtual: the folded SX costs one physical pulse, as an SX);
  RZ    is virtual, never folded.
A barrier on the gate's qubits separates the copies, so that neither the transpiler nor the runtime cancels them.

The record (per gate type: the number of gates, the inserted pairs, the indices of the gates that got an extra pair, the
achieved lambda; the seed) is returned next to the circuit and goes into the manifest.
"""
from __future__ import annotations

import math
import zlib
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

ISA_NAMES = ("cz", "sx", "x", "rz", "measure", "barrier", "delay")
FOLDABLE = ("cz", "sx", "x")
GATE_SETS = {"cz": ("cz",), "all": FOLDABLE}
MAX_LAMBDA = 9.0


def gate_set(gates: Union[str, Iterable[str]]) -> Tuple[str, ...]:
    """'cz' / 'all' or an iterable of gate names -> the foldable gate names, in canonical order; anything else raises."""
    if isinstance(gates, str):
        if gates not in GATE_SETS:
            raise ValueError(f"unknown gate set {gates!r}: use 'cz' or 'all' (or a set of {FOLDABLE})")
        return GATE_SETS[gates]
    names = {str(g) for g in gates}
    bad = sorted(names - set(FOLDABLE))
    if bad or not names:
        raise ValueError(f"cannot fold {bad or 'no gate'}: only {FOLDABLE} are foldable on the native gate set (rz is virtual)")
    return tuple(g for g in FOLDABLE if g in names)


def check_fold_lambda(lam) -> float:
    """lambda must be a real number in [1, MAX_LAMBDA] (fractional allowed)."""
    if isinstance(lam, bool) or not isinstance(lam, (int, float, np.integer, np.floating)) or not math.isfinite(float(lam)):
        raise ValueError(f"the noise factor must be a finite number, got {lam!r}")
    lam = float(lam)
    if not 1.0 <= lam <= MAX_LAMBDA:
        raise ValueError(f"the noise factor must lie in [1, {MAX_LAMBDA:g}], got {lam}")
    return lam


def check_isa(circ) -> None:
    """Only cz, sx, x, rz, measure, barrier, delay: anything else is not an ISA circuit of Heron and raises."""
    bad = sorted(set(circ.count_ops()) - set(ISA_NAMES))
    if bad:
        raise ValueError(f"the circuit is not in the Heron ISA {ISA_NAMES}: found {bad}")


def lam_key(lam: float) -> Union[int, float]:
    """A noise factor as a dictionary key: an int for whole values (1, 3, 5), a rounded float otherwise."""
    lam = round(float(lam), 6)
    return int(lam) if float(lam).is_integer() else lam


def folded_pairs(n: int, lam: float) -> int:
    """Number of G^dagger G pairs to insert in n gates for the noise factor lam: the integer P that makes 1 + 2 P / n closest."""
    return int(math.floor((lam - 1.0) * n / 2.0 + 0.5))


def _append_inverse(out, op, qs) -> None:
    name = op.name
    if name in ("cz", "x"):
        out.append(op.inverse(), qs)
    elif name == "sx":
        out.rz(math.pi, qs[0])                  # SX^dagger = RZ(pi) SX RZ(pi) (up to a global phase)
        out.append(op, qs)
        out.rz(math.pi, qs[0])
    else:
        raise ValueError(f"no inverse defined for {name}")


def fold_gates(isa_circuit, lam: float, gates: Union[str, Iterable[str]] = "cz", seed: int = 0):
    """(circuit, record): `isa_circuit` with the gates of the types `gates` folded to the noise factor `lam` (see the module
    docstring).  For every type with N gates, P = round((lam - 1) N / 2) pairs are inserted: every gate gets P // N pairs and
    a seeded random subset of P % N gates one more, so lam = 3 (and every odd integer) is deterministic.  The circuit's
    registers, qubit order, parameters, measurements, other gates, barriers and layout are unchanged; lam = 1 returns a copy."""
    lam = check_fold_lambda(lam)
    names = gate_set(gates)
    check_isa(isa_circuit)
    ops = isa_circuit.count_ops()
    plan: Dict[str, dict] = {}
    for g in names:
        n = int(ops.get(g, 0))
        if n == 0:
            plan[g] = {"n": 0, "pairs": 0, "base_pairs": 0, "folded_indices": [], "achieved_lam": 1.0}
            continue
        pairs = folded_pairs(n, lam)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(g.encode())]))
        extra = sorted(int(i) for i in rng.choice(n, size=pairs % n, replace=False)) if pairs % n else []
        plan[g] = {"n": n, "pairs": pairs, "base_pairs": pairs // n, "folded_indices": extra,
                   "achieved_lam": 1.0 + 2.0 * pairs / n}
    if not any(p["n"] for p in plan.values()):
        raise ValueError(f"the circuit has none of the gates {list(names)} to fold")
    extra_sets = {g: set(p["folded_indices"]) for g, p in plan.items()}
    seen = {g: 0 for g in names}
    out = isa_circuit.copy_empty_like()
    for ins in isa_circuit.data:
        op, qs, cs = ins.operation, ins.qubits, ins.clbits
        out.append(op, qs, cs)
        g = op.name
        if g in plan:
            k = seen[g]
            seen[g] += 1
            for _ in range(plan[g]["base_pairs"] + (1 if k in extra_sets[g] else 0)):
                out.barrier(*qs)
                _append_inverse(out, op, qs)
                out.barrier(*qs)
                out.append(op, qs, cs)
    if getattr(isa_circuit, "_layout", None) is not None:
        out._layout = isa_circuit._layout
    total_n = sum(p["n"] for p in plan.values())
    total_p = sum(p["pairs"] for p in plan.values())
    record = {"gates": list(names), "lam_target": lam, "seed": int(seed), "per_gate": plan,
              "lam_eff": 1.0 + 2.0 * total_p / total_n if total_n else 1.0,
              "lam_cz": plan["cz"]["achieved_lam"] if "cz" in plan else 1.0,
              "ops": {k: int(v) for k, v in out.count_ops().items()}}
    return out, record


def lambda_ladder(n: int, lams: Sequence[float]) -> list:
    """Achieved noise factors of n gates for the targets `lams` (for the plan tables)."""
    return [1.0 + 2.0 * folded_pairs(n, check_fold_lambda(x)) / n for x in lams]

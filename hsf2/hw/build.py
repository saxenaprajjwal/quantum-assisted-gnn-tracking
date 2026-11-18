"""Ising problem of a Phase-1 run folder -> p=1 QAOA circuits and the PUB list of one hardware job.

Everything numerical comes from Phase 1 (acts_qaoa), imported and never modified: Ising.sparsified (top-K couplings
by |J|, ties broken deterministically), the angle re-optimisation of a sparsified circuit (pipeline._circuit_angles),
build_qaoa_circuit / bind, and the closed-form p=1 energy.  The problem files are only read.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .._phase1 import ensure_phase1
from .gauge import split_shots

log = logging.getLogger(__name__)


@dataclass
class Problem:
    run: object                 # acts_qaoa.pipeline.Run (only read, and samples/ written by the sample export)
    qubo: object
    ising: object
    scale: float
    prob: dict
    entry: dict                 # angles.json["standard_p1"]
    run_dir: Path

    @property
    def n(self) -> int:
        return int(self.ising.n)

    @property
    def fingerprint(self) -> Optional[str]:
        return self.prob.get("fingerprint")


@dataclass
class PubSpec:
    """One PUB of a job: `kind` main or control, `gauge` index, angles, shots (equal in every PUB of a job)."""
    k: int
    kind: str
    gauge: int
    gammas: List[float]
    betas: List[float]
    shots: int
    gauged: bool = False

    @property
    def name(self) -> str:
        return f"{self.kind}_g{self.gauge}" if self.gauged else self.kind

    def to_dict(self) -> dict:
        return {"k": self.k, "name": self.name, "kind": self.kind, "gauge": self.gauge, "gammas": list(self.gammas),
                "betas": list(self.betas), "shots": self.shots, "gauged": self.gauged}


def load_problem(run_dir: Path) -> Problem:
    """run_config.json + problem.json + angles.json of `run_dir` (Phase-1 Run; nothing is written here)."""
    ensure_phase1()
    from acts_qaoa import pipeline as P1
    from acts_qaoa.config import RunConfig
    from ..utils import load_json
    run_dir = Path(run_dir)
    for need in ("run_config.json", "problem.json", "angles.json"):
        if not (run_dir / need).is_file():
            raise FileNotFoundError(f"{run_dir / need} not found (run folder of tune-loop / tune-compare expected)")
    cfg = RunConfig.from_dict(load_json(run_dir / "run_config.json"))
    run = P1.Run(run_dir, cfg, None)
    fit, qubo, ising, prob = run.load_problem()
    entry = P1._angles_entry(run, "standard", 1)
    return Problem(run, qubo, ising, float(prob["scale"]), prob, entry, run_dir)


def sparse_ising(ising, top_k: Optional[int]):
    """The Ising model the circuit evolves under: the `top_k` largest couplings (Phase-1 Ising.sparsified, tie-break
    included), all of them for top_k None / 0.  The h terms are always kept."""
    if top_k is None or top_k == 0:
        return ising
    if top_k < 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    return ising.sparsified(int(top_k))


def circuit_angles(problem: Problem, ism, top_k: Optional[int]) -> Tuple[List[float], List[float], dict]:
    """(gammas, betas, info) of the p=1 circuit of `ism`: the angles of angles.json, re-optimised for the sparsified
    circuit exactly as Phase 1 does (pipeline._circuit_angles), so a job with `top_k` couplings uses the angles of the
    earlier hardware jobs of the same K."""
    from acts_qaoa import pipeline as P1
    e = P1._circuit_angles(problem.entry, problem.ising, ism, problem.scale, bool(top_k))
    info = {"reoptimized_for_sparse": bool(e.get("reoptimized_for_sparse", False)),
            "original_gammas": e.get("original_gammas", e["gammas"]), "original_betas": e.get("original_betas", e["betas"])}
    return [float(x) for x in e["gammas"]], [float(x) for x in e["betas"]], info


def ideal_mean_energy(problem: Problem, ism, gammas: Sequence[float], betas: Sequence[float]) -> float:
    """<E_full> of the noiseless p=1 circuit state of `ism` (closed form, Phase-1 stage_ibm.ideal_energy)."""
    from acts_qaoa.qaoa_math import p1_energy
    if len(gammas) != 1:
        raise ValueError("closed form is for p = 1")
    ising, s = problem.ising, problem.scale
    e_n = p1_energy(ism.h / s, ism.full_J() / s, gammas[0], betas[0], h_obs=ising.h / s, J_obs=ising.full_J() / s)
    return float(ising.offset + s * e_n)


def build_circuit(ism, scale: float, p: int = 1):
    """(circuit with measure_all, gamma vector, beta vector, h terms, J terms): Phase-1 build_qaoa_circuit."""
    from acts_qaoa.circuits import build_qaoa_circuit
    h_t, j_t = ism.terms(scale)
    qc, gam, bet = build_qaoa_circuit(ism.n, h_t, j_t, p=p)
    return qc, gam, bet, h_t, j_t


def control_angles(fractional: bool, gammas: Sequence[float], betas: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Angles of the control PUB, whose ideal output is uniform.

    Normal settings: gamma = beta = 0 on the same transpiled circuit (as in Phase 1; the angles are circuit
    parameters, so the transpiler cannot remove any gate).  Fractional setting: its circuits are transpiled with
    bound angles, where beta = 0 would let the transpiler delete every diagonal gate in front of the measurement;
    beta = pi/2 (RX(pi) = X on every qubit) keeps the whole gate sequence and still gives the uniform distribution,
    so the control keeps the main circuit's gamma."""
    if fractional:
        return [float(g) for g in gammas], [math.pi / 2] * len(betas)
    return [0.0] * len(gammas), [0.0] * len(betas)


def make_pub_specs(fractional: bool, gauged: bool, n_gauges: int, total_shots: int, gammas: Sequence[float],
                   betas: Sequence[float]) -> List[PubSpec]:
    """PUBs of one job: main and control, each with `total_shots` shots in total.  A gauged job has one main and one
    control PUB per gauge, all with the same number of shots (total_shots / n_gauges): equal shots in every PUB."""
    cg, cb = control_angles(fractional, gammas, betas)
    ng = int(n_gauges) if gauged else 1
    shots = split_shots(int(total_shots), ng)
    specs: List[PubSpec] = []
    for g in range(ng):
        specs.append(PubSpec(len(specs), "main", g, list(gammas), list(betas), shots, gauged))
        specs.append(PubSpec(len(specs), "control", g, cg, cb, shots, gauged))
    check_equal_shots(specs)
    return specs


def check_equal_shots(pubs) -> int:
    """Every PUB of a job must have the same number of shots (unequal shots are deprecated since runtime 0.48 and
    the executor Sampler requires equal shots).  Returns the common value."""
    shots = [int(p.shots if hasattr(p, "shots") else p.spec.shots if hasattr(p, "spec") else p[-1]) for p in pubs]
    if not shots:
        raise ValueError("a job needs at least one PUB")
    if len(set(shots)) != 1:
        raise ValueError(f"unequal shots in the PUBs of one job: {shots}")
    if shots[0] < 1:
        raise ValueError(f"shots must be positive, got {shots[0]}")
    return shots[0]


def pack_components(components, backend, *args, **kwargs):
    """Not implemented and not planned: on 11 Nov 2026 the owner chose one component per PUB (no packing).  Kept as the
    place where packing would go: pack several QX3b components (component JSON written by
    hsf2/qubo/triplet.py) into ONE circuit on disjoint qubit regions of `backend`, with the isolated-vs-packed check.
    The hooks it will use are already here: PubSpec / make_pub_specs (equal shots), transpile_job (initial_layout of a
    region), layout_record (measured qubits per component) and run.finish_job (one sample set per component)."""
    raise NotImplementedError("component packing is Phase B of the hardware runner and is not implemented yet")

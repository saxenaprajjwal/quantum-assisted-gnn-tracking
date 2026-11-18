"""Step 3 of the hardware plan: QX3b components on hardware, one component per PUB.

Instances come from a `triplet` run folder: components.csv and components/<event>_s<sector>_f<frac>_c<id>.json (format in
the docstring of hsf2/qubo/triplet.py; local index i of a component is its i-th triplet, and bit i of a sample index).
  hard        both classical descents miss the exact optimum: min(descent_energy, greedy_descent_energy) > exact_energy + 1e-9
  benchmark   a seeded random sample of the other components of the same size range
Per component: QUBO -> Ising (z = 1 - 2x; h_i = -a_i/2 - sum_j b_ij/4, J_ij = b_ij/4, offset = sum a/2 + sum b/4, i.e. Phase-1
QUBO.to_ising, checked against the energies of the export), p = 1 angles from the Phase-1 closed form on the component's own
Ising (no sparsification unless top_k is given: the conflict couplings are essential), Phase-1 build_qaoa_circuit with
qubit i = variable i, and Phase-1 transpile_best (4 seeds, best estimated fidelity), each component on its own.  One PUB
per component with `shots` shots in every PUB, `per_job` PUBs per job, one typed confirmation per job; planning, budget check,
submission, billed-time resolution and booking are the ones of hsf2.hw.run (hw-run).

Job folder  <triplet run folder>/hw/<DD-MM-YYYY-HH-MM>_comp_<setting>_<k>/   (k = 1, 2, ... ; never inside a Phase-1 folder)
  manifest.json                  kind "components"; settings, options, versions, job id, billed time, the PUB list
  components.json                PUB k = which component (file, sha1, group, hard flag, its components.csv row)
  pub_<k>_main.isa.qpy           transpiled circuit of PUB k (QPY version 15)
  pub_<k>.layout.json            logical -> physical mapping, measured qubits, active qubits, edges
  snapshot_pre.json / _post.json calibration of the used qubits and edges before / after the job
  pub_<k>_counts.json            counts of PUB k: index = sum_i x_i 2^i, i.e. bit i = local triplet i, as exact_selection
  local_result.json              (local mode) the simulated counts, so that retrieve can rebuild the outputs
The analysis (hsf2.hw.comp_analysis, hw-components-analyze) reads these folders and runs after every local or submitted job.

Local mode (0 QPU s, no IBM access) has three simulators and an automatic choice, set by sim_method and recorded per PUB:
  statevector            Aer noise trajectories of the fake backend, i.e. the device noise; measured on the PC for a 14-variable
                         component: 0.07 s per shot (plus 8 s start-up), doubling with every further qubit
  matrix_product_state   the simulator of hw-run (Phase A); the dense conflict couplings of the components make its bond
                         dimension grow like 2^(n/2): about 4-8 s per shot at 14 qubits (two measurements), so it is not a default
  white-noise            counts drawn from F p_ideal + (1 - F) / 2^n with F the estimated circuit fidelity of the transpiled
                         circuit: a model, with no device noise simulated, for any n <= 22 (pipeline tests)
  auto                   statevector up to AUTO_MAX_QUBITS qubits, white-noise above (logged, and recorded per PUB)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._phase1 import ensure_phase1
from ..utils import code_version, load_json, save_json
from . import run as R
from .build import PubSpec, build_circuit, check_equal_shots, sparse_ising
from .config import HwConfig, creds_label
from .ledger import HwLedger
from .settings import Setting, get_setting
from .snapshot import calibration_snapshot, layout_files, used_qubits_edges, utc_now
from .transpile import IsaPub, check_fractional_isa, fold_rzz, layout_record, require_fractional_target

log = logging.getLogger(__name__)

KIND = "components"
STEP = "step3"
MAX_VARS = 24                      # components are exported, and solved exactly, up to 24 variables (triplet.py)
IDEAL_MAX_N = 22                   # exact statevector of the ideal circuit
HARD_TOL = 1e-9                    # a descent misses the optimum if it ends more than this above exact_energy
ENERGY_TOL = 1e-9
DEFAULT_NMIN, DEFAULT_NMAX = 14, 20
DEFAULT_PER_JOB = 10
SIM_METHODS = ("auto", "statevector", "matrix_product_state", "white-noise")
AUTO_MAX_QUBITS = 14               # auto: Aer noise trajectories up to this many active qubits, the white-noise model above
CSV_COLUMNS = ("event", "sector", "frac", "component", "n_variables", "n_quadruplet_couplings", "n_conflicts",
               "n_true_triplets", "n_fake_triplets", "exact_energy", "descent_energy", "greedy_descent_energy")


# ---------------------------------------------------------------------------
# instances: components.csv + components/*.json of a triplet run folder
# ---------------------------------------------------------------------------
@dataclass
class Component:
    """One exported QX3b component: its JSON document and its components.csv row."""
    name: str                       # file stem <event>_s<sector>_f<frac>_c<id>
    path: Path
    sha1: str                       # of the JSON file: a job records it, the analysis checks it
    doc: dict
    row: dict
    group: str = ""                 # hard / benchmark, set by select_components

    @property
    def n(self) -> int:
        return len(self.doc["linear"])

    @property
    def linear(self) -> np.ndarray:
        return np.asarray(self.doc["linear"], dtype=float)

    @property
    def quad(self) -> List[Tuple[int, int, float]]:
        return [(int(i), int(j), float(b)) for i, j, b in self.doc["quadratic"]]

    @property
    def true_triplet(self) -> np.ndarray:
        return np.asarray(self.doc["true_triplet"], dtype=bool)

    @property
    def hard(self) -> bool:
        return is_hard(self.row)

    def to_record(self, run_dir: Path) -> dict:
        """The entry of components.json (and of the manifest) for this component."""
        try:
            rel = self.path.relative_to(run_dir).as_posix()
        except ValueError:
            rel = self.path.name
        return {"component": self.name, "file": rel, "sha1": self.sha1, "n": self.n, "group": self.group,
                "hard": self.hard, "row": self.row}


def _plain(v):
    """pandas / numpy scalars -> JSON types; NaN -> None."""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) else float(v)
    return v


def is_hard(row: dict) -> bool:
    """Both classical descents miss the exact optimum (n <= 24, where the optimum is known)."""
    vals = [row.get(k) for k in ("exact_energy", "descent_energy", "greedy_descent_energy")]
    if any(v is None for v in vals):
        raise ValueError(f"component {row.get('component')} has no exact / descent energy in components.csv: "
                         "the hard flag is defined for exported components (n <= 24) only")
    e0, d, gd = (float(v) for v in vals)
    return min(d, gd) > e0 + HARD_TOL


def component_file_name(row: dict) -> str:
    """The JSON file of a components.csv row (names as written by hsf2.qubo.triplet.analyse_components)."""
    return (f"{row['event']}_s{int(row['sector'])}_f{float(row['frac']):g}_c{int(row['component'])}.json"
            .replace("/", "_").replace("\\", "_"))


def _qubo_energy(linear: np.ndarray, quad: Sequence[Tuple[int, int, float]], x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.dot(linear, x) + sum(b * x[i] * x[j] for i, j, b in quad))


def check_component_doc(doc: dict, row: dict, what: str) -> None:
    """The export and its components.csv row must describe the same component; raises ValueError otherwise."""
    for key in ("meta", "linear", "quadratic", "exact_energy", "exact_selection", "true_triplet"):
        if key not in doc:
            raise ValueError(f"{what}: no '{key}' in the component JSON")
    if row.get("exact_energy") is None or doc["exact_energy"] is None:
        raise ValueError(f"{what}: no exact energy (components.csv: {row.get('exact_energy')}, JSON: {doc['exact_energy']})")
    n = len(doc["linear"])
    if not (len(doc["exact_selection"]) == len(doc["true_triplet"]) == n == int(row["n_variables"])):
        raise ValueError(f"{what}: {n} variables, components.csv says {row['n_variables']}")
    zeta = float(doc["meta"].get("zeta", float("nan")))
    if not zeta > 0:
        raise ValueError(f"{what}: conflict weight zeta = {zeta} must be positive (conflicts are the positive couplings)")
    seen = set()
    for i, j, b in doc["quadratic"]:
        if not (0 <= i < j < n) or (i, j) in seen:
            raise ValueError(f"{what}: bad coupling ({i}, {j}) of a {n}-variable component")
        seen.add((i, j))
    n_quad = sum(1 for _, _, b in doc["quadratic"] if b < 0)
    n_conf = sum(1 for _, _, b in doc["quadratic"] if b > 0)
    if n_quad != int(row["n_quadruplet_couplings"]) or n_conf != int(row["n_conflicts"]):
        raise ValueError(f"{what}: {n_quad} quadruplet couplings and {n_conf} conflicts in the JSON, components.csv says "
                         f"{row['n_quadruplet_couplings']} and {row['n_conflicts']}")
    n_true = int(np.sum(doc["true_triplet"]))
    if n_true != int(row["n_true_triplets"]) or n - n_true != int(row["n_fake_triplets"]):
        raise ValueError(f"{what}: {n_true} true triplets in the JSON, components.csv says {row['n_true_triplets']}")
    lin, quad = np.asarray(doc["linear"], float), [(int(i), int(j), float(b)) for i, j, b in doc["quadratic"]]
    e_sel = _qubo_energy(lin, quad, doc["exact_selection"])
    if abs(e_sel - float(doc["exact_energy"])) > ENERGY_TOL or abs(float(doc["exact_energy"]) - float(row["exact_energy"])) > ENERGY_TOL:
        raise ValueError(f"{what}: energy of exact_selection {e_sel} vs exact_energy {doc['exact_energy']} "
                         f"(components.csv {row['exact_energy']})")


def load_components(run_dir: Path, only: Optional[Sequence[str]] = None) -> List[Component]:
    """Every component with n <= 24 variables of a triplet run folder (components.csv + components/), in file order.
    `only`: component names to load (the others are not read).  Raises if the CSV lacks a column, a JSON file is missing,
    or a file disagrees with its CSV row."""
    import pandas as pd
    run_dir = Path(run_dir)
    csv, cdir = run_dir / "components.csv", run_dir / "components"
    if not csv.is_file() or not cdir.is_dir():
        raise FileNotFoundError(f"{run_dir} is not a triplet run folder: components.csv and components/ are needed "
                                "(written by `python -m hsf2 triplet`)")
    df = pd.read_csv(csv)
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv} has no column {missing}; re-run `python -m hsf2 triplet` with the current code")
    wanted = None if only is None else set(only)
    out: List[Component] = []
    for rec in df.to_dict("records"):
        if int(rec["n_variables"]) > MAX_VARS:
            continue                                         # not exported (no exact optimum either)
        row = {k: _plain(v) for k, v in rec.items()}
        fname = component_file_name(row)
        if wanted is not None and Path(fname).stem not in wanted:
            continue
        path = cdir / fname
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found: components.csv lists it (n_variables {row['n_variables']})")
        raw = path.read_bytes()
        doc = json.loads(raw)
        check_component_doc(doc, row, path.name)
        out.append(Component(path.stem, path, hashlib.sha1(raw).hexdigest(), doc, row))
    if wanted is not None and {c.name for c in out} != wanted:
        raise FileNotFoundError(f"components not found in {run_dir}: {sorted(wanted - {c.name for c in out})}")
    return out


def _stream(seed: int, k: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(k)]))


def select_components(comps: Sequence[Component], nmin: int = DEFAULT_NMIN, nmax: int = DEFAULT_NMAX, hard_only: bool = False,
                      n_benchmark: int = 0, seed: int = 0, max_total: Optional[int] = None) -> List[Component]:
    """The components of a job set: every hard piece with nmin <= n <= nmax (group 'hard') and `n_benchmark` pieces drawn at
    random, with `seed`, from the other components of the range (group 'benchmark').  max_total caps the total: a seeded
    random subset of the hard pieces is kept, the benchmark sample is never reduced.  hard_only refuses a benchmark."""
    if not (2 <= nmin <= nmax <= MAX_VARS):
        raise ValueError(f"need 2 <= nmin <= nmax <= {MAX_VARS}, got nmin {nmin}, nmax {nmax}")
    if n_benchmark < 0:
        raise ValueError(f"--benchmark must be >= 0, got {n_benchmark}")
    if hard_only and n_benchmark:
        raise ValueError("--hard selects only the hard pieces: it cannot be combined with --benchmark")
    if max_total is not None and max_total < 1:
        raise ValueError(f"--max must be >= 1, got {max_total}")
    pool = [c for c in comps if nmin <= c.n <= nmax]
    hard = [c for c in pool if c.hard]
    ordinary = [c for c in pool if not c.hard]
    if n_benchmark > len(ordinary):
        raise ValueError(f"--benchmark {n_benchmark}: only {len(ordinary)} ordinary components with {nmin} <= n <= {nmax}")
    if max_total is not None and n_benchmark > max_total:
        raise ValueError(f"--max {max_total} is smaller than --benchmark {n_benchmark}")
    room = len(hard) if max_total is None else max_total - n_benchmark
    if hard and room < 1:
        raise ValueError(f"--max {max_total} leaves no room for the {len(hard)} hard pieces next to --benchmark {n_benchmark}")
    if room < len(hard):
        pick = sorted(_stream(seed, 1).choice(len(hard), size=room, replace=False).tolist()) if room > 0 else []
        hard = [hard[i] for i in pick]
    bench: List[Component] = []
    if n_benchmark:
        pick = sorted(_stream(seed, 0).choice(len(ordinary), size=n_benchmark, replace=False).tolist())
        bench = [ordinary[i] for i in pick]
    sel = [replace(c, group="hard") for c in hard] + [replace(c, group="benchmark") for c in bench]
    if not sel:
        raise ValueError(f"no component selected: {len(pool)} components with {nmin} <= n <= {nmax}, {len(hard)} of them "
                         "hard; widen the size range or ask for a benchmark sample (--benchmark N)")
    return sel


def selection_table(sel: Sequence[Component], pool_size: Optional[int] = None, pool_hard: Optional[int] = None) -> str:
    """The selection as text (hw-components --list)."""
    lines = [f"{'component':<38} {'n':>3} {'quad':>5} {'conf':>5} {'true':>4} {'fake':>4} {'E0':>10} {'descent':>10} "
             f"{'greedy+d':>10} {'group':<9}"]
    for c in sel:
        r = c.row
        lines.append(f"{c.name:<38} {c.n:>3} {r['n_quadruplet_couplings']:>5} {r['n_conflicts']:>5} {r['n_true_triplets']:>4} "
                     f"{r['n_fake_triplets']:>4} {r['exact_energy']:>10.5f} {r['descent_energy']:>10.5f} "
                     f"{r['greedy_descent_energy']:>10.5f} {c.group:<9}")
    n_hard = sum(c.group == "hard" for c in sel)
    lines.append(f"{len(sel)} components: {n_hard} hard, {len(sel) - n_hard} benchmark"
                 + ("" if pool_size is None else f" (of {pool_size} in the size range, {pool_hard} of them hard)"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# component -> QUBO -> Ising -> angles
# ---------------------------------------------------------------------------
def component_qubo(doc: dict):
    """Phase-1 QUBO of a component document: E(x) = sum_i linear[i] x_i + sum quadratic b_ij x_i x_j (const 0)."""
    ensure_phase1()
    from acts_qaoa.surrogate import QUBO
    n = len(doc["linear"])
    W = np.zeros((n, n))
    for i, j, b in doc["quadratic"]:
        W[int(i), int(j)] += float(b)
    return QUBO(n=n, const=0.0, L=np.asarray(doc["linear"], dtype=float), W=W)


def component_energies(doc: dict) -> np.ndarray:
    """E[index] of all 2^n selections (bit i of the index = triplet i), Phase-1 enumerate_energies."""
    ensure_phase1()
    from acts_qaoa.landscape import enumerate_energies
    return enumerate_energies(component_qubo(doc))


def ising_to_qubo(ism):
    """Exact inverse of QUBO.to_ising (z = 1 - 2x): the QUBO that has the energies of the Ising model `ism`."""
    ensure_phase1()
    from acts_qaoa.surrogate import QUBO
    Ju = np.triu(ism.J, 1)
    return QUBO(n=ism.n, const=float(ism.offset + ism.h.sum() + Ju.sum()), L=-2.0 * ism.h - 2.0 * (Ju + Ju.T).sum(axis=1),
                W=4.0 * Ju)


def component_ising(doc: dict, check: bool = True, seed: int = 0, n_random: int = 64):
    """Ising model of a component (h_i = -a_i/2 - sum_j b_ij/4, J_ij = b_ij/4, offset = sum a/2 + sum b/4).  The Ising and
    QUBO energies must agree on random selections and on the exported optimum (ValueError otherwise)."""
    q = component_qubo(doc)
    ising = q.to_ising()
    if check:
        rng = np.random.default_rng(seed)
        X = np.vstack([rng.integers(0, 2, size=(n_random, q.n)), np.asarray(doc["exact_selection"], dtype=int)[None, :]])
        e_q, e_i = q.energy_bits(X), ising.energy_bits(X)
        if not np.allclose(e_q, e_i, rtol=0, atol=1e-9):
            raise ValueError(f"Ising and QUBO energies differ by up to {np.abs(e_q - e_i).max():.3g} for {q.n} variables")
        if abs(float(e_i[-1]) - float(doc["exact_energy"])) > ENERGY_TOL:
            raise ValueError(f"Ising energy {e_i[-1]} of the exported optimum != exact_energy {doc['exact_energy']}")
    return ising


def component_angles(ising, top_k: Optional[int] = None):
    """(circuit Ising, scale, gammas, betas, info): the p = 1 angles that minimise the closed-form <E> of the component's
    full Ising for a circuit that evolves under its `top_k` strongest couplings (all of them for None / 0).  The circuit
    uses H / scale with the full model's scale, as Phase 1 does."""
    ensure_phase1()
    from acts_qaoa.qaoa_math import optimize_p1
    scale = float(ising.scale())
    if not scale > 0:
        raise ValueError("all Ising coefficients of this component are zero: no circuit to run")
    ism = sparse_ising(ising, top_k)
    r = optimize_p1(ism.h / scale, ism.full_J() / scale, h_obs=ising.h / scale, J_obs=ising.full_J() / scale)
    info = {"gamma": float(r["gamma"]), "beta": float(r["beta"]), "scale": scale, "top_k": top_k or None,
            "n_couplings": int(ism.n_couplings(1e-12)),
            "ideal_mean_energy": float(ising.offset + scale * r["energy"])}
    return ism, scale, [float(r["gamma"])], [float(r["beta"])], info


def ideal_distribution(ism, scale: float, gamma: float, beta: float, diag: Optional[np.ndarray] = None) -> np.ndarray:
    """Noiseless p = 1 distribution (2^n, float64, sums to 1) of the circuit that evolves under `ism` / scale; bit i of the
    index = qubit i = variable i.  `diag`: the circuit Hamiltonian (h.z + z.J.z)/scale of every basis state, if known."""
    ensure_phase1()
    from acts_qaoa.landscape import enumerate_energies
    from acts_qaoa.qaoa_math import StatevectorQAOA
    if ism.n > IDEAL_MAX_N:
        raise ValueError(f"the exact ideal distribution is computed for n <= {IDEAL_MAX_N} variables, got {ism.n}")
    if diag is None:
        diag = (enumerate_energies(ising_to_qubo(ism)) - ism.offset) / scale
    return np.asarray(StatevectorQAOA(diag, mixer="x", dtype=np.complex128).probs([gamma], [beta]), dtype=np.float64)


# ---------------------------------------------------------------------------
# circuits and plans
# ---------------------------------------------------------------------------
def component_layout_record(isa, n_logical: int, stats: dict) -> dict:
    """transpile.layout_record, with a fallback for qiskit's TranspileLayout.initial_index_layout(filter_ancillas=True), which
    raises IndexError when the layout lists an ancilla among its first n entries (qiskit 2.1.0, FakeFez, n = 13, transpiler
    seed 0, found with a 13-variable component): the initial layout is then read through input_qubit_mapping, and the final
    one is the measured qubits (classical bit k holds logical qubit k at the end of a measure_all circuit)."""
    try:
        return layout_record(isa, n_logical, stats)
    except IndexError:
        pass
    lay = isa.layout
    bare = isa.copy()
    bare._layout = None                                  # layout_record then reads the circuit only
    rec = layout_record(bare, n_logical, stats)
    initial: List[Optional[int]] = [None] * n_logical
    for virtual, phys in lay.initial_layout.get_virtual_bits().items():
        pos = lay.input_qubit_mapping.get(virtual)
        if pos is not None and pos < n_logical:
            initial[pos] = int(phys)
    if None in initial:
        raise RuntimeError(f"cannot read the initial layout of a {n_logical}-qubit circuit from its TranspileLayout")
    rec["initial"], rec["final"] = initial, list(rec["measured"]["qubits"])
    log.warning("[hw-comp] qiskit's initial_index_layout failed for this circuit; layout read from input_qubit_mapping")
    return rec


def transpile_component(ism, scale: float, spec: PubSpec, fractional: bool, backend, n_seeds: int = 4) -> IsaPub:
    """One component on `backend`: Phase-1 transpile_best over `n_seeds` seeds, the best estimated fidelity wins.  Normal
    settings keep gamma and beta as parameters (the PUB carries their values); the fractional setting (e) binds them first,
    folds the rzz angles into (0, pi/2] and checks the ISA, as hsf2.hw.transpile does."""
    ensure_phase1()
    from acts_qaoa.circuits import bind, circuit_stats, parameter_values, transpile_best
    qc, gam, bet, _, _ = build_circuit(ism, scale)
    seeds = list(range(int(n_seeds)))
    if fractional:
        require_fractional_target(backend)
        isa, st = transpile_best(bind(qc, gam, bet, spec.gammas, spec.betas), backend, seeds)
        isa = fold_rzz(isa)
        check_fractional_isa(isa)
        seed = st["transpile_seed"]
        st = circuit_stats(isa, backend)
        st["transpile_seed"] = seed
        values = None
    else:
        isa, st = transpile_best(qc, backend, seeds)
        values = parameter_values(isa, gam, bet, spec.gammas, spec.betas)
    return IsaPub(spec, isa, values, st, component_layout_record(isa, ism.n, st))


@dataclass
class ComponentPlan:
    """A component ready to run: its circuit Ising, angles and transpiled PUB."""
    comp: Component
    ising: object                   # the component's full Ising
    ism: object                     # the Ising the circuit evolves under (sparsified for top_k)
    scale: float
    info: dict                      # component_angles info
    pub: IsaPub


@dataclass
class ComponentJob:
    """The PUBs of one hardware job: one component each, equal shots."""
    k: int                          # job number, 1-based
    n_jobs: int
    tag: str
    plans: List[ComponentPlan]
    estimate_s: float
    rep_delay_s: float
    duration_s: Optional[float]

    @property
    def pubs(self) -> List[IsaPub]:
        return [p.pub for p in self.plans]

    @property
    def shots(self) -> int:
        return check_equal_shots(self.pubs)


def plan_component(comp: Component, k: int, shots: int, setting: Setting, backend, n_seeds: int,
                   top_k: Optional[int] = None) -> ComponentPlan:
    t0 = time.time()
    ising = component_ising(comp.doc)
    ism, scale, gammas, betas, info = component_angles(ising, top_k)
    spec = PubSpec(k, "main", 0, gammas, betas, int(shots), False)
    pub = transpile_component(ism, scale, spec, setting.fractional, backend, n_seeds)
    st = pub.stats
    log.info("[hw-comp] %s: n=%d, %d couplings, gamma %.3f beta %.3f, layout seed %s: %d 2q gates, depth2q %s, F_est %.3g "
             "(%.0f s)", comp.name, comp.n, info["n_couplings"], gammas[0], betas[0], st.get("transpile_seed"), st["n_2q"],
             st.get("depth_2q"), st.get("fidelity_est", float("nan")), time.time() - t0)
    return ComponentPlan(comp, ising, ism, scale, info, pub)


def plan_components(comps: Sequence[Component], setting: Setting, cfg: HwConfig, backend, per_job: int = DEFAULT_PER_JOB,
                    top_k: Optional[int] = None) -> List[ComponentJob]:
    """Group the components `per_job` to a job (in the given order) and transpile each one; the estimated QPU seconds of a
    job come from Phase-1 estimate_qpu_seconds (conservative, 1.5 x), with the longest circuit of the job."""
    ensure_phase1()
    from acts_qaoa import ibm
    if setting.gauged:
        raise ValueError("setting f (gauges) is not supported for components; use a-e")
    if per_job < 1:
        raise ValueError(f"--per-job must be >= 1, got {per_job}")
    if not comps:
        raise ValueError("no component to plan")
    rep = ibm.rep_delay_seconds(backend)
    rnd = 16 if (setting.gate_twirl or setting.measure_twirl) else 1
    chunks = [list(comps[i:i + per_job]) for i in range(0, len(comps), per_job)]
    jobs: List[ComponentJob] = []
    for j, chunk in enumerate(chunks, 1):
        plans = [plan_component(c, k, cfg.shots, setting, backend, cfg.transpile_seeds, top_k) for k, c in enumerate(chunk)]
        dur = max((p.pub.stats.get("duration_s") or 0.0) for p in plans) or None
        est = ibm.estimate_qpu_seconds([cfg.shots] * len(plans), dur, rep, rnd)
        jobs.append(ComponentJob(j, len(chunks), f"comp_{setting.name}_{j}", plans, est, rep, dur))
    return jobs


# ---------------------------------------------------------------------------
# manifest and plan files
# ---------------------------------------------------------------------------
def job_manifest(job: ComponentJob, setting: Setting, cfg: HwConfig, backend_name: str, run_dir: Path, step: str,
                 command: str, selection: Optional[dict] = None, top_k: Optional[int] = None) -> dict:
    pubs = []
    for p in job.plans:
        rec = p.comp.to_record(run_dir)
        pubs.append(dict(p.pub.spec.to_dict(), component=p.comp.name, group=p.comp.group, hard=rec["hard"], n=p.comp.n,
                         component_sha1=p.comp.sha1, n_couplings=p.info["n_couplings"], scale=p.scale,
                         ideal_mean_energy=p.info["ideal_mean_energy"], n_2q=p.pub.stats.get("n_2q"),
                         depth_2q=p.pub.stats.get("depth_2q"), fidelity_est=p.pub.stats.get("fidelity_est"),
                         duration_s=p.pub.stats.get("duration_s"), transpile_seed=p.pub.stats.get("transpile_seed"),
                         measured_qubits=p.pub.layout["measured"]["qubits"], initial_layout=p.pub.layout["initial"]))
    return {"schema": R.SCHEMA, "step": step, "kind": KIND, "state": "planned", "mode": cfg.mode, "created": utc_now(),
            "command": command, "tag": job.tag, "backend": backend_name, "calibration_id": None, "versions": R.versions(),
            "code": code_version(), "creds": creds_label(cfg.creds_file), "config": cfg.to_dict(),
            "primitive": cfg.primitive, "setting": setting.to_dict(), "sampler_options": None, "job_id": None,
            "billed": None, "run_dir": str(run_dir), "job_index": job.k, "n_jobs": job.n_jobs, "top_k": top_k or None,
            "shots": job.shots, "estimate_s": job.estimate_s, "rep_delay_s": job.rep_delay_s,
            "circuit_duration_s": job.duration_s, "selection": selection, "pubs": pubs}


def write_job_files(job_dir: Path, job: ComponentJob, backend, run_dir: Path) -> dict:
    """components.json, QPY, layouts and the pre-run calibration snapshot (as run.write_plan_files); returns the snapshot."""
    job_dir = Path(job_dir)
    save_json(job_dir / "components.json", {"schema": R.SCHEMA, "run_dir": str(run_dir), "job": job.tag,
                                            "pubs": [dict(pub=p.pub.spec.k, **p.comp.to_record(run_dir)) for p in job.plans]})
    layout_files(job_dir, job.pubs)
    qs, es = used_qubits_edges([p.pub.layout for p in job.plans])
    snap = calibration_snapshot(backend, qs, es, source="target")
    save_json(job_dir / "snapshot_pre.json", snap)
    return snap


# ---------------------------------------------------------------------------
# local simulation
# ---------------------------------------------------------------------------
def sim_method_for(method: str, plan: ComponentPlan) -> str:
    if method not in SIM_METHODS:
        raise ValueError(f"sim_method must be one of {SIM_METHODS}, got {method!r}")
    if method == "auto":
        n_act = int(plan.pub.stats.get("n_active_qubits") or plan.comp.n)
        return "statevector" if n_act <= AUTO_MAX_QUBITS else "white-noise"
    return method


def white_noise_counts(plan: ComponentPlan, seed: int) -> Dict[int, int]:
    """Counts of the white-noise model: F p_ideal + (1 - F) / 2^n with F = estimated fidelity of the transpiled circuit."""
    if plan.comp.n > IDEAL_MAX_N:
        raise ValueError(f"the white-noise local model needs the ideal distribution (n <= {IDEAL_MAX_N}), got n = "
                         f"{plan.comp.n}; use --sim-method statevector")
    spec = plan.pub.spec
    p = ideal_distribution(plan.ism, plan.scale, spec.gammas[0], spec.betas[0])
    F = min(max(float(plan.pub.stats.get("fidelity_est") or 0.0), 0.0), 1.0)
    mix = F * p + (1.0 - F) / p.size
    mix /= mix.sum()
    c = np.random.default_rng(int(seed)).multinomial(int(spec.shots), mix)
    return {int(i): int(v) for i, v in zip(np.nonzero(c)[0], c[c > 0])}


def run_local_counts(backend, job: ComponentJob, seed: int, method: str) -> Tuple[List[Dict[int, int]], List[str]]:
    """Counts of every PUB of a job on the local simulators (see the module docstring) and the method used for each."""
    from . import fake as F
    methods = [sim_method_for(method, p) for p in job.plans]
    counts: List[Optional[Dict[int, int]]] = [None] * len(job.plans)
    for m in ("statevector", "matrix_product_state"):
        ks = [k for k, x in enumerate(methods) if x == m]
        if ks:
            for k, c in zip(ks, F.run_local(backend, [job.plans[k].pub for k in ks], seed, method=m)):
                counts[k] = c
    for k, m in enumerate(methods):
        if m == "white-noise":
            counts[k] = white_noise_counts(job.plans[k], seed + k)
    if "white-noise" in methods:
        log.warning("[hw-comp] local mode: %d of %d PUBs come from the white-noise model (F = estimated circuit fidelity, no "
                    "device noise simulated); use --sim-method statevector for Aer noise", methods.count("white-noise"),
                    len(methods))
    return [c for c in counts if c is not None], methods


# ---------------------------------------------------------------------------
# running a job set
# ---------------------------------------------------------------------------
def write_counts(job_dir: Path, manifest: dict, counts: Sequence[Dict[int, int]]) -> None:
    """pub_<k>_counts.json for every PUB, saved first (the job is billed: its raw counts are kept even when a check
    fails, as in run.write_counts); then the shots and the index range are checked."""
    pubs = manifest["pubs"]
    if len(counts) != len(pubs):
        raise ValueError(f"{len(counts)} count sets for {len(pubs)} PUBs")
    problems = []
    for p, c in zip(pubs, counts):
        total = int(sum(c.values()))
        save_json(Path(job_dir) / f"pub_{p['k']}_counts.json",
                  {"pub": p["k"], "name": p["name"], "component": p["component"], "n": p["n"], "shots": p["shots"],
                   "total": total, "counts": {str(k): int(v) for k, v in sorted(c.items())}})
        if total != p["shots"]:
            problems.append(f"PUB {p['k']} returned {total} shots, expected {p['shots']}")
        if c and (min(c) < 0 or max(c) >= 1 << p["n"]):
            problems.append(f"PUB {p['k']}: sample index outside 0..2^{p['n']} - 1 (wrong register size?)")
    if problems:
        raise ValueError("; ".join(problems))


def finish_component_job(job_dir: Path, manifest: dict, counts: Sequence[Dict[int, int]], job=None, backend=None) -> None:
    """Counts files and the post-run snapshot (the part of run.finish_job that does not need a Phase-1 problem)."""
    job_dir = Path(job_dir)
    write_counts(job_dir, manifest, counts)
    if backend is not None:
        lays = [load_json(p) for p in sorted(job_dir.glob("pub_*.layout.json"))]
        qs, es = used_qubits_edges(lays)
        props = None
        if job is not None:
            try:
                props = job.properties()
            except Exception as exc:
                log.warning("[hw-comp] job.properties() failed (%s): post snapshot uses the backend's current properties",
                            type(exc).__name__)
        snap = calibration_snapshot(backend, qs, es, source="job.properties" if props is not None else "backend",
                                    properties=props, refresh=props is None)
        save_json(job_dir / "snapshot_post.json", snap)
        manifest["calibration_id"] = snap.get("calibration_id") or manifest.get("calibration_id")
    manifest["state"] = "done"
    R.save_manifest(job_dir, manifest)


def _local_job(backend, job: ComponentJob, setting: Setting, job_dir: Path, manifest: dict, seed: int,
               method: str) -> List[Dict[int, int]]:
    """The local branch of run.execute_job, with the choice of simulator (local_result.json is the same record)."""
    from . import fake as F
    check_equal_shots(job.pubs)
    counts, methods = run_local_counts(backend, job, seed, method)
    job_id = F.local_job_id(manifest.get("tag", ""))
    save_json(Path(job_dir) / "local_result.json", {"job_id": job_id, "seed": seed, "backend": manifest["backend"],
                                                    "sim_methods": methods,
                                                    "counts": [{str(k): v for k, v in c.items()} for c in counts]})
    note = {"note": "recorded only: the local simulators ignore twirling and DD", "sim_methods": methods}
    manifest.update({"job_id": job_id, "state": "done", "sampler_options": note, "options_requested": setting.options(),
                     "sim_methods": methods, "billed": {"job_s": 0.0, "used": 0.0, "source": "local"}})
    for p, m in zip(manifest["pubs"], methods):
        p["sim_method"] = m
    R.save_manifest(job_dir, manifest)
    return counts


def _confirm_with_components(confirm: Callable[[str], None], job: ComponentJob) -> Callable[[str], None]:
    """The typed confirmation of run.execute_job, preceded by the components of the job."""
    names = ", ".join(f"{p.comp.name} (n={p.comp.n}, {p.comp.group or 'ungrouped'})" for p in job.plans)
    header = f"Job {job.k} of {job.n_jobs}, one PUB per component: {names}"
    return lambda msg: confirm(header + chr(10) + msg)


def run_components(run_dir: Path, comps: Sequence[Component], setting_name: str, cfg: HwConfig, per_job: int = DEFAULT_PER_JOB,
                   top_k: Optional[int] = None, step: str = STEP, command: str = "", out_dir: Optional[Path] = None,
                   confirm: Callable[[str], None] = R.confirm_submit, wait: bool = True, seed: int = 0,
                   sim_method: str = "auto", selection: Optional[dict] = None, service=None,
                   analysis_seed: Optional[int] = None) -> dict:
    """hw-components: plan every job (transpile), then dry (plan files only), local (simulate) or submit them one after
    the other, each with its budget check and typed confirmation.  The analysis of the finished jobs (out_dir: where it
    writes; `analysis_seed`: seed of its controls, default `seed`) runs at the end, also if a later job was refused.  `seed`
    is the seed of the local simulation.  Returns {'jobs': [...], 'analysis', 'service', 'estimate_total_s'}."""
    cfg.validate()
    run_dir = Path(run_dir)
    setting = get_setting(setting_name)
    if setting.gauged:
        raise ValueError("setting f (gauges) is not supported for components; use a-e")
    if per_job < 1:
        raise ValueError(f"--per-job must be >= 1, got {per_job}")
    if not comps:
        raise ValueError("no component to run")
    if sim_method not in SIM_METHODS:
        raise ValueError(f"sim_method must be one of {SIM_METHODS}, got {sim_method!r}")
    if not wait and -(-len(comps) // per_job) > 1:
        raise ValueError("--no-wait needs a single job (raise --per-job): the account-usage fallback of the billed time "
                         "needs the jobs to run strictly one after the other")
    if cfg.primitive == "executor" and setting.fractional:
        log.warning("[hw-comp] fractional gates with the executor Sampler have not been checked (risk in the design note)")
    backend, service = R.get_job_backend(cfg, setting, service)
    name = getattr(backend, "name", None)
    name = name() if callable(name) else str(name)
    jobs = plan_components(comps, setting, cfg, backend, per_job, top_k)
    total = sum(j.estimate_s for j in jobs)
    log.info("[hw-comp] %d components -> %d jobs of up to %d PUBs x %d shots, setting %s; estimated %.1f s of QPU time in "
             "total (%s)", len(comps), len(jobs), per_job, cfg.shots, setting.name, total,
             ", ".join(f"{j.estimate_s:.1f}" for j in jobs))
    ledger = None
    if cfg.mode == "submit":
        from . import session
        ledger = HwLedger(cfg.ledger_path(), cfg.budget_s, creds_file=cfg.creds_file)
        ledger.check(total, session.service_remaining(service), cfg.reserve_s)      # all jobs, before the first one
    elif cfg.mode == "dry":                                                           # read only: what a submit would meet
        led = HwLedger(cfg.ledger_path(), cfg.budget_s, creds_file=cfg.creds_file)
        log.info("[hw-comp] ledger %s: used %.1f s, remaining %.1f s of %.0f s; a submit of these jobs would book an estimated "
                 "%.1f s", led.path, led.used_seconds(), led.remaining_seconds(), led.budget_s, total)
    results: List[dict] = []
    done: List[Path] = []
    try:
        for job in jobs:
            job_dir = R.new_job_dir(run_dir, job.tag)
            manifest = job_manifest(job, setting, cfg, name, run_dir, step, command, selection, top_k)
            write_job_files(job_dir, job, backend, run_dir)
            log.info("[hw-comp] job %d/%d: %s | %d PUBs x %d shots | estimated %.1f s of QPU time", job.k, job.n_jobs,
                     job_dir, len(job.plans), job.shots, job.estimate_s)
            res = {"job_dir": job_dir, "manifest": manifest, "job": job, "job_id": None, "counts": None}
            results.append(res)
            if cfg.mode == "dry":
                manifest["state"] = "dry"
                R.save_manifest(job_dir, manifest)
                continue
            try:
                if cfg.mode == "local":
                    counts = _local_job(backend, job, setting, job_dir, manifest, seed, sim_method)
                    ex = {"job_id": manifest["job_id"], "counts": counts, "billed": manifest["billed"], "job": None}
                else:
                    ex = R.execute_job(backend, job.pubs, cfg, setting, f"hw-comp-{setting.name}", job.estimate_s, job_dir,
                                       manifest, str(run_dir), service, ledger, _confirm_with_components(confirm, job), wait,
                                       False, seed)
            except Exception as exc:
                if manifest.get("job_id") is None:             # refused or failed before anything was submitted or simulated
                    manifest.update(state="aborted", aborted=f"{type(exc).__name__}: {exc}")
                    R.save_manifest(job_dir, manifest)         # the folder then says why it holds no counts
                raise
            res.update(job_id=ex["job_id"], counts=ex["counts"], billed=ex["billed"])
            if ex["counts"] is None:
                log.info("[hw-comp] job %s submitted, not waiting: rebuild its outputs with `python -m hsf2 hw-retrieve %s "
                         "--run-dir %s`, then run hw-components-analyze on %s", ex["job_id"], ex["job_id"], run_dir, job_dir)
            else:
                finish_component_job(job_dir, manifest, ex["counts"], ex.get("job"), backend)
                done.append(job_dir)
    finally:
        analysis = None
        if done and out_dir is not None:
            from .comp_analysis import analyze_jobs
            analysis = analyze_jobs(done, out_dir, seed if analysis_seed is None else analysis_seed)
    return {"jobs": results, "analysis": analysis, "service": service, "estimate_total_s": total}


# ---------------------------------------------------------------------------
# retrieval of a submitted component job
# ---------------------------------------------------------------------------
def retrieve_components_job(job_id: str, cfg: HwConfig, run_dir: Optional[Path] = None, search_root: Optional[Path] = None,
                            service=None) -> dict:
    """Rebuild the outputs of a component job whose shell was lost (as run.retrieve_job): counts from local_result.json
    (local jobs) or from IBM, billed time resolved and booked for real jobs, counts files and post snapshot.  The analysis
    is run with hw-components-analyze."""
    from . import fake as F
    job_dir = R.find_job_dir(job_id, run_dir, search_root)
    manifest = load_json(job_dir / "manifest.json")
    if manifest.get("kind") != KIND:
        raise ValueError(f"{job_dir} is not a components job (kind {manifest.get('kind')!r})")
    manifest["run_dir"] = str(job_dir.parents[1])          # the folder may have been moved or copied
    job = None
    if job_id.startswith(F.LOCAL_PREFIX):
        stored = load_json(job_dir / "local_result.json")
        counts = [{int(k): int(v) for k, v in c.items()} for c in stored["counts"]]
        backend = F.fake_backend(cfg.fake_backend)
        if manifest["setting"]["fractional"]:
            backend = F.fractional_fake(backend)
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
        finish_component_job(job_dir, manifest, counts, job, backend)
    return {"job_dir": job_dir, "manifest": manifest}


def retrieve_any(job_id: str, cfg: HwConfig, run_dir: Optional[Path] = None, search_root: Optional[Path] = None,
                 service=None) -> dict:
    """hw-retrieve for every kind of job: component jobs here, the others by run.retrieve_job (unchanged)."""
    job_dir = R.find_job_dir(job_id, run_dir, search_root)
    kind = load_json(job_dir / "manifest.json").get("kind")
    if kind == KIND:
        return retrieve_components_job(job_id, cfg, run_dir, search_root, service)
    if kind == "zne":                                       # hw-zne jobs (hsf2.hw.zne)
        from .zne import retrieve_zne_job
        return retrieve_zne_job(job_id, cfg, run_dir, search_root, service)
    if kind == "qx4":                                       # hw-qx4 jobs (hsf2.hw.qx4hw)
        from .qx4hw import retrieve_qx4_job
        return retrieve_qx4_job(job_id, cfg, run_dir, search_root, service)
    if kind == R.READOUT_KIND:                              # hw-readout calibration jobs: readout_cal.json
        return R.retrieve_readout_job(job_id, cfg, run_dir, search_root, service)
    return R.retrieve_job(job_id, cfg, run_dir, search_root, service)

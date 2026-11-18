"""Locate the Phase-1 package ``acts_qaoa`` (surrogate -> QUBO -> QAOA pipeline) and make it importable.

Search order
    1. $HSF_PHASE1_DIR                                  (set by hsf2_scripts/env_lxplus.sh on lxplus)
    2. the folder that contains hsf2/ itself            (lxplus: hsf2/ copied into the Optimization folder)
    3. <Part II root>/../HSF                            (the PC layout: Desktop/HSF next to Desktop/HSF part II)
    4. <Part II root>/../acts/Examples/Scripts/Optimization   (an older lxplus layout under my_acts/)

Only the tuning / quantum parts need Phase 1; data, graph, model and tracking work without it.  Part II uses some
Phase-1 internals (pipeline.Run, stage_build, _angles_entry, ...); ensure_phase1() checks that they exist, so an
incompatible acts_qaoa fails at start-up with a clear message instead of in the middle of a study.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]          # the folder that contains hsf2/

TESTED_VERSIONS = ("2.0.23",)
REQUIRED_API = {
    "acts_qaoa.pipeline": ("Run", "stage_warmstart", "stage_build", "diverse_select", "_angles_entry"),
    "acts_qaoa.config": ("ParamSpace", "RunConfig", "ObjectiveConfig", "EVENT_SEED_STRIDE"),
    "acts_qaoa.encoding": ("Encoding",),
    "acts_qaoa.surrogate": ("QUBO", "Ising", "fit_surrogate", "qubo_from_surrogate"),
    "acts_qaoa.landscape": ("Landscape", "enumerate_energies", "simulated_annealing", "uniform_counts",
                            "counts_from_probs"),
    "acts_qaoa.qaoa_math": ("StatevectorQAOA", "optimize_p1", "p1_energy"),
    "acts_qaoa.circuits": ("bind", "build_qaoa_circuit", "transpile_best", "aer_counts"),
    "acts_qaoa.ibm": ("fake_backend",),
    "acts_qaoa": ("__version__",),
}


def candidates() -> List[Path]:
    out = []
    env = os.environ.get("HSF_PHASE1_DIR")
    if env:
        out.append(Path(env).expanduser())
    out.append(ROOT)
    out.append(ROOT.parent / "HSF")
    out.append(ROOT.parent / "acts" / "Examples" / "Scripts" / "Optimization")
    return out


def find_phase1() -> Optional[Path]:
    for d in candidates():
        if (d / "acts_qaoa" / "__init__.py").is_file():
            return d.resolve()
    return None


def check_api() -> None:
    """Every Phase-1 name Part II uses must exist (raises ImportError listing what is missing)."""
    missing = []
    for mod, names in REQUIRED_API.items():
        m = importlib.import_module(mod)
        missing += [f"{mod}.{n}" for n in names if not hasattr(m, n)]
    if missing:
        import acts_qaoa
        raise ImportError(f"acts_qaoa {getattr(acts_qaoa, '__version__', '?')} at {Path(acts_qaoa.__file__).parent} "
                          f"lacks {missing}; Part II was tested with {TESTED_VERSIONS}")


def ensure_phase1() -> Path:
    """Make acts_qaoa importable (once), check its API and return its folder; clear error if it cannot be found."""
    d = find_phase1()
    if d is None:
        tried = "\n  ".join(str(c) for c in candidates())
        raise ImportError("Phase-1 package acts_qaoa not found. Set HSF_PHASE1_DIR to the folder that "
                          f"contains acts_qaoa/ (tried:\n  {tried})")
    if str(d) not in sys.path:
        sys.path.append(str(d))       # appended, so Phase-1 top-level folders never shadow anything else
    check_api()
    return d


def phase1_version() -> Optional[str]:
    try:
        ensure_phase1()
        import acts_qaoa
        return getattr(acts_qaoa, "__version__", None)
    except Exception:
        return None

"""Per-job record files: ISA circuits (QPY), layout JSON and calibration snapshots.

QPY is written with version=15, which the PC's qiskit 2.1 can read (qiskit 2.6 on lxplus would default to 17).
A calibration snapshot lists, for every qubit and every edge the job used: the target's error / duration of each gate
(after backend.refresh() on a real backend), and from backend.properties() T1, T2, readout error and the two
preparation-dependent readout errors prob_meas0_prep1 / prob_meas1_prep0.  A field the backend does not provide is
recorded as null, so a missing value is visible and the key set is always the same.
"""
from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

QPY_VERSION = 15

log = logging.getLogger(__name__)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def write_qpy(path: Path, circuit) -> Path:
    from qiskit import qpy
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        qpy.dump(circuit, f, version=QPY_VERSION)
    return path


def read_qpy(path: Path):
    from qiskit import qpy
    with open(path, "rb") as f:
        return qpy.load(f)[0]


def _f(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _gate_props(target, name: str, qargs: tuple) -> Optional[dict]:
    try:
        p = target[name][qargs]
    except Exception:
        return None
    if p is None:
        return None
    return {"error": _f(p.error), "duration": _f(p.duration)}


def _qubit_properties(props, q: int) -> dict:
    def get(fn):
        try:
            return _f(fn())
        except Exception:
            return None

    def qprop(name):
        return get(lambda: props.qubit_property(q, name)[0])

    out = {"t1": get(lambda: props.t1(q)), "t2": get(lambda: props.t2(q)),
           "readout_error": get(lambda: props.readout_error(q)),
           "prob_meas0_prep1": qprop("prob_meas0_prep1"), "prob_meas1_prep0": qprop("prob_meas1_prep0")}
    return out


def calibration_snapshot(backend, qubits: Iterable[int], edges: Iterable[Sequence[int]], source: str = "backend",
                         properties=None, refresh: bool = True) -> dict:
    """Errors and durations on the used qubits and edges.  `properties` overrides backend.properties() (post-run:
    job.properties(), the calibration at the job's start time); refresh=True first calls backend.refresh() on backends
    that have it (IBMBackend), so the target is today's calibration."""
    if refresh and hasattr(backend, "refresh") and "fake" not in type(backend).__module__.lower():
        try:
            backend.refresh()
        except Exception as exc:
            log.warning("[hw] backend.refresh() failed (%s): the snapshot may be stale", type(exc).__name__)
    qubits = sorted({int(q) for q in qubits})
    edges = sorted({tuple(sorted(int(x) for x in e)) for e in edges})
    target = backend.target
    props = properties
    if props is None:
        try:
            props = backend.properties()
        except Exception as exc:
            log.warning("[hw] backend.properties() failed (%s)", type(exc).__name__)
    cal_id = None
    if props is not None:
        lu = getattr(props, "last_update_date", None)
        cal_id = None if lu is None else str(lu)
    q_rec: Dict[str, dict] = {}
    for q in qubits:
        rec = _qubit_properties(props, q) if props is not None else {
            k: None for k in ("t1", "t2", "readout_error", "prob_meas0_prep1", "prob_meas1_prep0")}
        rec["measure"] = _gate_props(target, "measure", (q,))
        for g in ("sx", "x"):
            rec[g] = _gate_props(target, g, (q,))
        q_rec[str(q)] = rec
    e_rec: Dict[str, dict] = {}
    two_q = [n for n in target.operation_names if n in ("cz", "ecr", "cx", "rzz")]
    for a, b in edges:
        rec = {}
        for g in two_q:
            p = _gate_props(target, g, (a, b)) or _gate_props(target, g, (b, a))
            if p is not None:
                rec[g] = p
        e_rec[f"{a}-{b}"] = rec
    return {"date": utc_now(), "source": source, "backend": str(getattr(backend, "name", backend)),
            "calibration_id": cal_id, "qubits": q_rec, "edges": e_rec}


def used_qubits_edges(layouts: Sequence[dict]) -> tuple:
    """Union of the active qubits and 2q edges of several layout records."""
    qs: set = set()
    es: set = set()
    for lay in layouts:
        qs.update(lay["active_qubits"])
        qs.update(lay["measured"]["qubits"])
        es.update(tuple(e) for e in lay["edges"])
    return sorted(qs), sorted(es)


def layout_files(job_dir: Path, pubs) -> List[Path]:
    """Writes pub_<k>.layout.json and pub_<k>_<name>.isa.qpy for every IsaPub; returns the paths."""
    from ..utils import save_json
    out = []
    for p in pubs:
        k, name = p.spec.k, p.spec.name
        out.append(write_qpy(Path(job_dir) / f"pub_{k}_{name}.isa.qpy", p.isa))
        out.append(save_json(Path(job_dir) / f"pub_{k}.layout.json",
                             {"pub": k, "name": name, "gauge": p.spec.gauge, "transpile_seed": p.stats.get("transpile_seed"),
                              "stats": {x: p.stats.get(x) for x in ("n_2q", "depth", "depth_2q", "ops", "fidelity_est",
                                                                    "fidelity_gates", "fidelity_readout", "duration_s")},
                              **p.layout}))
    return out

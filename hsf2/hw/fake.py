"""Local mode: a fake IBM backend plus an Aer simulation of its noise (0 QPU s, no IBM access, no credentials).

  * backend: Phase-1 `ibm.fake_backend` (FakeFez by default) -- a calibration snapshot shipped with qiskit-ibm-runtime;
  * simulation: AerSimulator.from_backend(fake, method=matrix_product_state).  The QAOA circuits only entangle along the
    kept couplings, so the matrix-product state stays small, and Aer drops the idle qubits of the 156-qubit device;
  * the runtime options of the setting (twirling, DD, ...) are recorded but have no effect here -- the simulator applies
    the device's gate and readout noise without randomized compiling or dynamical decoupling;
  * job ids are `local_<8 hex>`, usage is 0, and the result is stored next to the manifest (local_result.json) so that
    `hw-retrieve` can rebuild the outputs.

`fractional_fake` adds a native RZZ(theta) and RX(theta) to a fake backend's target (same error and duration as its CZ
and SX), which is what a use_fractional_gates=True backend looks like to the transpiler; it can transpile but not run.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .._phase1 import ensure_phase1

log = logging.getLogger(__name__)

LOCAL_PREFIX = "local_"


def fake_backend(name: str = "FakeFez"):
    ensure_phase1()
    from acts_qaoa import ibm
    return ibm.fake_backend(name)


def fractional_fake(base):
    """`base` (a fake backend) with rzz(theta) on every cz edge and rx(theta) on every sx qubit."""
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate, RZZGate
    from qiskit.providers import BackendV2, Options
    from qiskit.transpiler import InstructionProperties

    tgt = copy.deepcopy(base.target)
    th = Parameter("theta")
    for gate, ref, name in ((RZZGate(th), "cz", "rzz"), (RXGate(th), "sx", "rx")):
        if name in tgt.operation_names:
            continue
        tgt.add_instruction(gate, {q: InstructionProperties(error=p.error, duration=p.duration)
                                   for q, p in tgt[ref].items() if p is not None})

    class FractionalFake(BackendV2):
        def __init__(self):
            super().__init__(name=f"{getattr(base, 'name', 'fake')}_fractional")
            self._target = tgt
            self._hsf2_base = base

        @property
        def target(self):
            return self._target

        @property
        def max_circuits(self):
            return None

        @classmethod
        def _default_options(cls):
            return Options(shots=1024)

        def properties(self):
            return base.properties()

        def run(self, *a, **k):
            raise NotImplementedError("fractional_fake only transpiles; local runs use the base backend's noise")

    return FractionalFake()


def local_simulator(backend, method: str = "matrix_product_state"):
    """AerSimulator with the noise of `backend` (of its base for a fractional_fake).  The fractional gates rzz / rx have
    no noise in the simulation (the base device has none for them); Aer runs them natively."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    base = getattr(backend, "_hsf2_base", None)
    if base is None:
        return AerSimulator.from_backend(backend, method=method)
    return AerSimulator(method=method, noise_model=NoiseModel.from_backend(base))


def local_job_id(tag: str) -> str:
    return LOCAL_PREFIX + hashlib.sha1(f"{time.time_ns()}:{tag}".encode()).hexdigest()[:8]


def bound_circuit(isa, values: Optional[np.ndarray]):
    return isa if values is None else isa.assign_parameters(dict(zip(isa.parameters, [float(v) for v in values])))


def run_local(backend, pubs: Sequence, seed: int = 0, method: str = "matrix_product_state") -> List[Dict[int, int]]:
    """Sample every IsaPub (`.isa`, `.values`, `.spec.shots`) on the local simulator; one {index: count} per PUB."""
    ensure_phase1()
    from acts_qaoa.circuits import counts_to_index
    sim = local_simulator(backend, method)
    out = []
    for k, p in enumerate(pubs):
        t0 = time.time()
        res = sim.run(bound_circuit(p.isa, p.values), shots=int(p.spec.shots), seed_simulator=int(seed) + k).result()
        if not res.success:
            raise RuntimeError(f"local simulation of PUB {k} failed: {res.status}")
        out.append(counts_to_index(res.get_counts()))
        log.info("[hw-local] PUB %d (%s): %d shots in %.1f s", k, p.spec.name, p.spec.shots, time.time() - t0)
    return out

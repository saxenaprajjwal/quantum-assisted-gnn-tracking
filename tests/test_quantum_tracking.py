"""QX4 linear-system tracking emulation and QX5 quantum edge classifier: conventions and known answers."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_linear_system_minimal_event():
    """TrackHHL Fig. 2: 2 particles, 3 layers -> 8 segments, 4 true; the system is positive definite."""
    from hsf2.qubo.linear import (build_system, classical_solution, eigen_amplitudes, segment_metrics, segments,
                                  spectrum_check, toy_velo)
    ev = toy_velo(2, n_layers=3, seed=1)
    seg = segments(ev)
    A, b, C, truth = build_system(ev, seg)
    assert len(seg) == 8 and truth.sum() == 4
    sc = spectrum_check(A)
    assert sc["positive_definite"] and sc["condition_number"] < 5.0
    S = classical_solution(A, b)
    assert S[truth].min() > S[~truth].max()                       # a threshold separates true from fake
    assert segment_metrics(S >= 0.5 * (S[truth].min() + S[~truth].max()), truth)["purity"] == 1.0
    ideal, _ = eigen_amplitudes(A, b, "hhl-ideal")
    assert np.allclose(np.abs(ideal), S / np.linalg.norm(S), atol=1e-12)        # HHL output = normalised A^-1 b
    one, info = eigen_amplitudes(A, b, "onebit", alpha_beta=0.75)
    assert np.allclose(one[~truth], 0.0, atol=1e-12)              # isolated (fake) segments are filtered out exactly
    assert np.all(np.abs(one[truth]) > 0.1) and 0 < info["postselection_probability"] < 1
    q8, _ = eigen_amplitudes(A, b, "hhl-qpe", n_clock=10)
    assert np.abs(np.abs(q8) @ np.abs(ideal)) > 0.99              # many clock bits -> ideal HHL


def test_linear_circuits_build_and_count():
    pytest.importorskip("qiskit")
    from hsf2.qubo.linear import build_system, gate_counts, hhl_circuit, onebit_circuit, segments, toy_velo
    ev = toy_velo(2, n_layers=3, seed=1)
    A, b, C, truth = build_system(ev, segments(ev))
    qc, info = onebit_circuit(A, np.pi / 0.75, 0.75)
    assert info["qubits"] == 4
    g1 = gate_counts(qc)
    qh, infoh = hhl_circuit(A, np.pi / 0.75, 0.75, n_clock=3)
    gh = gate_counts(qh)
    assert infoh["qubits"] == 7 and gh["cx"] > g1["cx"] > 0


def test_one_bit_circuit_matches_eigenbasis_emulation():
    """The Qiskit 1-bit circuit, post-selected on ancilla 0, reproduces the eigenbasis filter up to Trotter error."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    from hsf2.qubo.linear import build_system, eigen_amplitudes, onebit_circuit, segments, toy_velo
    ev = toy_velo(2, n_layers=3, seed=1)
    A, b, C, truth = build_system(ev, segments(ev))
    t = np.pi / 0.75
    qc, info = onebit_circuit(A, t, 0.75)
    sv = Statevector(qc).data.reshape(2, -1)[0]                   # ancilla (highest qubit) = 0
    em, _ = eigen_amplitudes(A, b, "onebit", t=t, alpha_beta=0.75)
    p_circ = np.abs(sv[: len(em)]) ** 2
    p_circ /= p_circ.sum()
    assert np.abs(p_circ - np.abs(em) ** 2).max() < 0.1           # one Lie-Trotter step: approximate, not exact
    assert p_circ[~truth].max() < 0.05


def test_quantum_head_matches_qiskit_statevector():
    pytest.importorskip("qiskit")
    import torch
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from hsf2.model.quantum_head import QuantumEdgeClassifier
    m = QuantumEdgeClassifier(n_qubits=3, n_layers=2, seed=4)
    with torch.no_grad():
        m.theta.uniform_(-2, 2)
        m.phi.uniform_(-2, 2)
    x = torch.tensor([[0.3, -1.2, 0.8], [2.0, 0.1, -0.4]])
    ours = m.expval(x).detach().numpy()
    for b in range(2):
        qc = QuantumCircuit(3)
        for l in range(2):
            for q in range(3):
                qc.ry(float(np.pi * np.tanh(float(x[b, q]))), q)
                qc.ry(m.theta[l, q].item(), q)
                qc.rz(m.phi[l, q].item(), q)
            for q in range(3):
                qc.cz(q, (q + 1) % 3)
        ref = Statevector(qc).expectation_value(SparsePauliOp("IIZ")).real
        assert abs(ref - ours[b]) < 1e-5

"""QX5: a variational quantum edge classifier, compared with classical classifiers on the same inputs.

Question it answers: given the same k input features per edge, does a k-qubit variational circuit separate true from
false edges as well as classical models with a comparable number of parameters?

Inputs (two settings, both reduced to k features by a PCA fitted on the training edges only):
  geometric  the 6 edge features of the Interaction Network (dr, dphi, dz, deta, z0, phi_slope), standardised;
  embedding  the trained IN's edge representation [h_src, h_dst, m] (3 x hidden), i.e. the quantum circuit replaces
             the IN's classifier head ("quantum head").
Circuit (k qubits, L layers, data re-uploading):
  for each layer l:  RY(pi * tanh(x_q)) on every qubit q (encoding),  RY(theta_lq) RZ(phi_lq),  CZ ring;
  output <Z_0>;  logit = w <Z_0> + c.     Parameters: 2 k L + 2.
It is simulated exactly (batched statevector in PyTorch, complex64), so training uses exact gradients (Adam on the
balanced binary cross-entropy); no shot noise, no hardware noise.  Classical references with the same k inputs:
logistic regression (k + 1 parameters) and an MLP with about as many parameters as the circuit.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch


class QuantumEdgeClassifier(torch.nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 3, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.k, self.L = int(n_qubits), int(n_layers)
        self.theta = torch.nn.Parameter(0.1 * torch.randn(self.L, self.k, generator=g))
        self.phi = torch.nn.Parameter(0.1 * torch.randn(self.L, self.k, generator=g))
        self.w = torch.nn.Parameter(torch.tensor(2.0))
        self.c = torch.nn.Parameter(torch.tensor(0.0))
        idx = torch.arange(2 ** self.k)
        self.register_buffer("bits", ((idx[:, None] >> torch.arange(self.k)) & 1).to(torch.float32))
        # CZ ring as a diagonal sign on the basis states
        sign = torch.ones(2 ** self.k)
        for q in range(self.k):
            r = (q + 1) % self.k
            if r == q:
                continue
            sign = sign * (1 - 2 * self.bits[:, q] * self.bits[:, r])
        self.register_buffer("cz", sign.to(torch.complex64))
        self.register_buffer("z0", (1 - 2 * self.bits[:, 0]).to(torch.float32))

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    @staticmethod
    def _apply_1q(psi: torch.Tensor, q: int, U: torch.Tensor, k: int) -> torch.Tensor:
        """psi [B, 2^k]; U [B, 2, 2] or [2, 2] acting on qubit q (bit q of the basis index)."""
        B = psi.shape[0]
        v = psi.reshape(B, 2 ** (k - q - 1), 2, 2 ** q)
        if U.dim() == 2:
            out = torch.einsum("ab,xybz->xyaz", U, v)
        else:
            out = torch.einsum("xab,xybz->xyaz", U, v)
        return out.reshape(B, 2 ** k)

    @staticmethod
    def _ry(a: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(a / 2), torch.sin(a / 2)
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)

    @staticmethod
    def _rz(a: torch.Tensor) -> torch.Tensor:
        e = torch.exp(-0.5j * a.to(torch.complex64))
        z = torch.zeros_like(e)
        return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)

    def expval(self, x: torch.Tensor) -> torch.Tensor:
        B, k = x.shape[0], self.k
        psi = torch.zeros(B, 2 ** k, dtype=torch.complex64)
        psi[:, 0] = 1.0
        enc = math.pi * torch.tanh(x)
        for l in range(self.L):
            for q in range(k):
                psi = self._apply_1q(psi, q, self._ry(enc[:, q]), k)
                psi = self._apply_1q(psi, q, self._ry(self.theta[l, q]), k)
                psi = self._apply_1q(psi, q, self._rz(self.phi[l, q]), k)
            psi = psi * self.cz
        return (psi.real ** 2 + psi.imag ** 2) @ self.z0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * self.expval(x) + self.c

    def circuit_resources(self) -> Dict[str, int]:
        ring = self.k if self.k > 2 else (1 if self.k == 2 else 0)
        return {"qubits": self.k, "layers": self.L, "parameters": self.n_parameters(),
                "single_qubit_rotations": 3 * self.k * self.L, "cz": ring * self.L}


def fit_pca(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    Z = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Z[: min(len(Z), 50000)], full_matrices=False)
    P = Vt[:k].T
    proj = Z @ P
    scale = proj.std(0) + 1e-9
    return mu, sd, P / scale


def apply_pca(X: np.ndarray, pca) -> np.ndarray:
    mu, sd, P = pca
    return ((X - mu) / sd) @ P


def train_binary(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 60, lr: float = 0.05,
                 batch: int = 1024, seed: int = 0) -> list:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    hist = []
    for _ in range(epochs):
        perm = rng.permutation(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), batch):
            j = torch.from_numpy(perm[i:i + batch])
            opt.zero_grad()
            loss = lossf(model(Xt[j]), yt[j])
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        hist.append(tot / len(Xt))
    return hist


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def mlp_like(n_in: int, n_params_target: int) -> torch.nn.Module:
    """One-hidden-layer MLP with about n_params_target parameters."""
    h = max(1, int(round((n_params_target - 1) / (n_in + 2))))
    return torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.Tanh(), torch.nn.Linear(h, 1), torch.nn.Flatten(0))


class Logistic(torch.nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(n_in, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


def edge_dataset(model, events, n_per_class: int, seed: int, source: str) -> Tuple[np.ndarray, np.ndarray]:
    """Balanced sample of edges (consecutive-hit labels) from the loose graphs of `events` (already hit-selected
    with the model's selection); features: 'geometric' (standardised IN edge features) or 'embedding' ([h_src, h_dst, m]
    of the trained IN)."""
    from ..graph.construct import build_graph
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for ev in events:
        g = build_graph(ev, model.pmap)
        x, e, s, d = model.tensors(ev, g)
        if source == "geometric":
            X = e.numpy()
        elif source == "embedding":
            with torch.no_grad():
                model.net.eval()
                X = model.net.embed(x, e, s, d).numpy()
        else:
            raise ValueError(source)
        Xs.append(X)
        ys.append(g.y.astype(np.int8))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    take = np.concatenate([rng.choice(pos, min(n_per_class, len(pos)), replace=False),
                           rng.choice(neg, min(n_per_class, len(neg)), replace=False)])
    return X[take], y[take].astype(np.float32)


def run_study(model, train_events, test_events, qubits: Tuple[int, ...] = (4, 6), layers: int = 3,
              n_per_class: int = 5000, epochs: int = 40, seed: int = 0) -> Dict[str, dict]:
    from .train import edge_metrics, roc_auc
    out: Dict[str, dict] = {}
    for source in ("geometric", "embedding"):
        Xtr, ytr = edge_dataset(model, train_events, n_per_class, seed, source)
        Xte, yte = edge_dataset(model, test_events, n_per_class, seed + 1, source)
        for k in qubits:
            if k > Xtr.shape[1]:
                continue
            pca = fit_pca(Xtr, k)                         # fitted on training edges only
            A, B = apply_pca(Xtr, pca), apply_pca(Xte, pca)
            res = {}
            q = QuantumEdgeClassifier(k, layers, seed)
            import time
            t0 = time.time()
            hq = train_binary(q, A, ytr, epochs=epochs, lr=0.05, seed=seed)
            sq = predict(q, B)
            res["quantum"] = {**edge_metrics(yte.astype(bool), sq), "parameters": q.n_parameters(),
                              "train_loss": hq[-1], "seconds": round(time.time() - t0, 1), **q.circuit_resources()}
            for name, net in (("logistic", Logistic(k)), ("mlp_same_params", mlp_like(k, q.n_parameters()))):
                h = train_binary(net, A, ytr, epochs=epochs, lr=0.02, seed=seed)
                s = predict(net, B)
                res[name] = {**edge_metrics(yte.astype(bool), s), "parameters": int(sum(p.numel() for p in net.parameters())),
                             "train_loss": h[-1]}
            out[f"{source}_k{k}"] = res
    # the full Interaction Network on the same test edges (reference upper bound)
    Xte_g, yte_g = edge_dataset(model, test_events, n_per_class, seed + 1, "embedding")
    with torch.no_grad():
        s_in = torch.sigmoid(model.net.classifier(torch.tensor(Xte_g, dtype=torch.float32)).squeeze(-1)).numpy()
    out["interaction_network"] = {**edge_metrics(yte_g.astype(bool), s_in), "parameters": model.net.n_parameters()}
    return out

"""Edge-classifying Interaction Network in plain PyTorch (no torch_geometric needed).

Follows DeZoort et al., "Charged particle tracking via edge-classifying interaction networks"
(Comput. Softw. Big Sci. 5 (2021) 26, arXiv:2103.16701) and the recurrent IN of Exa.TrkX (Ju et al.,
arXiv:2103.06995):

    h_i = phi_N(x_i),  m_k = phi_E(e_k)                                  (encoders)
    repeat T times (shared weights if `shared`):
        m_k <- m_k + phi_R([h_src(k), h_dst(k), m_k])                     (relational / edge update)
        h_i <- h_i + phi_O([h_i, mean_{k in in(i)} m_k, mean_{k in out(i)} m_k, log(1+deg_in), log(1+deg_out)])
    logit_k = phi_C([h_src(k), h_dst(k), m_k])                            (edge classifier)

Incoming and outgoing messages are aggregated separately (edges point away from the beam), and the mean
(not the sum) keeps the node update stable when the inference-time filters change the graph density.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn


def mlp(n_in: int, n_hidden: int, n_out: int, n_layers: int, layer_norm: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = n_in
    for _ in range(max(n_layers - 1, 0)):
        layers += [nn.Linear(d, n_hidden), nn.LayerNorm(n_hidden) if layer_norm else nn.Identity(), nn.SiLU()]
        d = n_hidden
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class InteractionNetwork(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 64, n_iter: int = 3, mlp_layers: int = 2,
                 shared: bool = True) -> None:
        super().__init__()
        self.n_iter = int(n_iter)
        self.shared = bool(shared)
        self.hidden = int(hidden)
        self.node_enc = mlp(node_in, hidden, hidden, mlp_layers)
        self.edge_enc = mlp(edge_in, hidden, hidden, mlp_layers)
        n_blocks = 1 if shared else max(self.n_iter, 1)
        self.edge_nets = nn.ModuleList([mlp(3 * hidden, hidden, hidden, mlp_layers) for _ in range(n_blocks)])
        self.node_nets = nn.ModuleList([mlp(3 * hidden + 2, hidden, hidden, mlp_layers) for _ in range(n_blocks)])
        self.classifier = mlp(3 * hidden, hidden, 1, mlp_layers)

    def forward(self, x: torch.Tensor, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x, e, src, dst)).squeeze(-1)

    def embed(self, x: torch.Tensor, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Input of the edge classifier, [h_src, h_dst, m] (3 x hidden per edge) -- used by the QX5 quantum head."""
        n = x.shape[0]
        h = self.node_enc(x)
        m = self.edge_enc(e)
        ones = torch.ones(len(src), dtype=x.dtype, device=x.device)
        deg_in = torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(0, dst, ones)
        deg_out = torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(0, src, ones)
        inv_in = (1.0 / deg_in.clamp(min=1.0)).unsqueeze(1)
        inv_out = (1.0 / deg_out.clamp(min=1.0)).unsqueeze(1)
        dfeat = torch.stack([torch.log1p(deg_in), torch.log1p(deg_out)], 1)
        for t in range(self.n_iter):
            k = 0 if self.shared else t
            m = m + self.edge_nets[k](torch.cat([h[src], h[dst], m], 1))
            agg_in = torch.zeros(n, self.hidden, dtype=x.dtype, device=x.device).index_add_(0, dst, m) * inv_in
            agg_out = torch.zeros(n, self.hidden, dtype=x.dtype, device=x.device).index_add_(0, src, m) * inv_out
            h = h + self.node_nets[k](torch.cat([h, agg_in, agg_out, dfeat], 1))
        return torch.cat([h[src], h[dst], m], 1)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


# ---------------------------------------------------------------------------
# Input features
# ---------------------------------------------------------------------------
def node_features(ev, names: Sequence[str]) -> np.ndarray:
    cols = []
    for n in names:
        if n == "r":
            cols.append(ev.r / 1000.0)
        elif n == "z":
            cols.append(ev.z / 1000.0)
        elif n == "phi":
            cols.append(ev.phi / np.pi)
        elif n == "eta":
            cols.append(ev.eta)
        elif n == "cosphi":
            cols.append(np.cos(ev.phi))
        elif n == "sinphi":
            cols.append(np.sin(ev.phi))
        else:
            raise KeyError(f"unknown node feature {n}")
    return np.stack(cols, 1).astype(np.float32)


def edge_matrix(g, names: Sequence[str]) -> np.ndarray:
    return np.stack([g.feats[n] for n in names], 1).astype(np.float32)


class Scaler:
    """Standardisation of node/edge features (mean/std from the training graphs), JSON-serialisable."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    @classmethod
    def fit(cls, arrays: Sequence[np.ndarray]) -> "Scaler":
        x = np.concatenate(arrays, 0).astype(np.float64)
        sd = x.std(0)
        return cls(x.mean(0), np.where(sd > 1e-12, sd, 1.0))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "Scaler":
        return cls(d["mean"], d["std"])

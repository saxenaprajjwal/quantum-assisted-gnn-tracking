"""Training and inference of the edge classifier (one graph = one event, CPU friendly)."""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import rankdata

from ..config import HitSelection, ModelConfig, TrainConfig
from ..graph.construct import Graph, LayerPairMap
from .interaction_network import InteractionNetwork, Scaler, edge_matrix, node_features

log = logging.getLogger(__name__)


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties)."""
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def edge_metrics(y: np.ndarray, s: np.ndarray, cut: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y, bool)
    p = s >= cut
    tp = float((p & y).sum())
    return {"auc": roc_auc(y, s), "edge_eff": tp / max(float(y.sum()), 1.0), "edge_purity": tp / max(float(p.sum()), 1.0),
            "n_edges": int(len(y)), "n_true": int(y.sum())}


class EdgeModel:
    """Network + feature scalers + graph map: everything needed to score the edges of a new event."""

    def __init__(self, net: InteractionNetwork, node_scaler: Scaler, edge_scaler: Scaler, mcfg: ModelConfig,
                 pmap: LayerPairMap, hit_selection: HitSelection, meta: Optional[dict] = None) -> None:
        self.net = net
        self.node_scaler = node_scaler
        self.edge_scaler = edge_scaler
        self.mcfg = mcfg
        self.pmap = pmap
        self.hit_selection = hit_selection
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ inference
    def tensors(self, ev, g: Graph, keep: Optional[np.ndarray] = None):
        src, dst = g.src, g.dst
        e = edge_matrix(g, self.mcfg.edge_features)
        if keep is not None:
            src, dst, e = src[keep], dst[keep], e[keep]
        x = self.node_scaler(node_features(ev, self.mcfg.node_features))
        return (torch.from_numpy(x), torch.from_numpy(self.edge_scaler(e)), torch.from_numpy(src.astype(np.int64)),
                torch.from_numpy(dst.astype(np.int64)))

    @torch.no_grad()
    def predict(self, ev, g: Graph, keep: Optional[np.ndarray] = None) -> np.ndarray:
        """Edge scores in [0, 1] for the edges of g (only the kept ones if a mask is given)."""
        n_e = int(keep.sum()) if keep is not None else g.n_edges
        if n_e == 0:
            return np.zeros(0, np.float32)
        self.net.eval()
        x, e, s, d = self.tensors(ev, g, keep)
        return torch.sigmoid(self.net(x, e, s, d)).numpy().astype(np.float32)

    # ------------------------------------------------------------------ persistence
    def save(self, out_dir: Path, extra: Optional[dict] = None) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), out_dir / "model.pt")
        info = {"model": self.mcfg.to_dict(), "node_scaler": self.node_scaler.to_dict(),
                "edge_scaler": self.edge_scaler.to_dict(), "pair_map": self.pmap.to_dict(),
                "hit_selection": self.hit_selection.to_dict(), "meta": self.meta}
        if extra:
            info.update(extra)
        (out_dir / "model.json").write_text(json.dumps(info, indent=1, default=float))

    @classmethod
    def load(cls, model_dir: Path) -> "EdgeModel":
        model_dir = Path(model_dir)
        info = json.loads((model_dir / "model.json").read_text())
        mcfg = ModelConfig.from_dict(info["model"])
        net = build_network(mcfg)
        net.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        net.eval()
        return cls(net, Scaler.from_dict(info["node_scaler"]), Scaler.from_dict(info["edge_scaler"]), mcfg,
                   LayerPairMap.from_dict(info["pair_map"]), HitSelection.from_dict(info["hit_selection"]),
                   info.get("meta", {}))


def build_network(mcfg: ModelConfig) -> InteractionNetwork:
    return InteractionNetwork(len(mcfg.node_features), len(mcfg.edge_features), mcfg.hidden, mcfg.n_iter,
                              mcfg.mlp_layers, mcfg.shared)


def train_model(train_set: Sequence[Tuple[object, Graph]], val_set: Sequence[Tuple[object, Graph]],
                mcfg: ModelConfig, tcfg: TrainConfig, pmap: LayerPairMap, hit_selection: HitSelection,
                progress: Optional[Callable[[dict], None]] = None) -> Tuple[EdgeModel, List[dict]]:
    """Train on (event, loose graph) pairs; the state with the lowest validation loss is returned."""
    if tcfg.threads and tcfg.threads > 0:
        torch.set_num_threads(int(tcfg.threads))
    torch.manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    if tcfg.max_train_events and len(train_set) > tcfg.max_train_events:
        train_set = list(train_set)[: tcfg.max_train_events]
    node_scaler = Scaler.fit([node_features(ev, mcfg.node_features) for ev, _ in train_set])
    edge_scaler = Scaler.fit([edge_matrix(g, mcfg.edge_features) for _, g in train_set if g.n_edges])
    net = build_network(mcfg)
    model = EdgeModel(net, node_scaler, edge_scaler, mcfg, pmap, hit_selection)
    lab = mcfg.label
    train_t = [model.tensors(ev, g) + (torch.from_numpy(g.labels(lab).astype(np.float32)),) for ev, g in train_set if g.n_edges]
    val_t = [model.tensors(ev, g) + (torch.from_numpy(g.labels(lab).astype(np.float32)),) for ev, g in val_set if g.n_edges]
    n_pos = sum(float(t[4].sum()) for t in train_t)
    n_all = sum(len(t[4]) for t in train_t)
    pos_weight = float(min((n_all - n_pos) / max(n_pos, 1.0), tcfg.max_pos_weight))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    opt = torch.optim.Adam(net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    total = max(tcfg.epochs * len(train_t), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: (tcfg.lr_min + 0.5 * (tcfg.lr - tcfg.lr_min) * (1 + math.cos(math.pi * min(step, total) / total))) / tcfg.lr)
    history: List[dict] = []
    best_state, best_loss = copy.deepcopy(net.state_dict()), float("inf")
    for epoch in range(1, tcfg.epochs + 1):
        t0 = time.time()
        net.train()
        losses = []
        for i in rng.permutation(len(train_t)):
            x, e, s, d, y = train_t[i]
            opt.zero_grad()
            loss = lossf(net(x, e, s, d), y)
            loss.backward()
            if tcfg.grad_clip and tcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), tcfg.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"),
               "lr": float(opt.param_groups[0]["lr"]), "time_s": 0.0}
        rec.update(evaluate_tensors(net, val_t, lossf))
        rec["time_s"] = round(time.time() - t0, 2)
        history.append(rec)
        if rec["val_loss"] < best_loss:
            best_loss = rec["val_loss"]
            best_state = copy.deepcopy(net.state_dict())
            rec["best"] = True
        if progress:
            progress(rec)
    net.load_state_dict(best_state)
    net.eval()
    best = min(history, key=lambda r: r["val_loss"]) if history else {}
    model.meta.update({"pos_weight": pos_weight, "n_parameters": net.n_parameters(), "best_epoch": best.get("epoch"),
                       "best_val": {k: best.get(k) for k in ("val_loss", "auc", "edge_eff", "edge_purity")},
                       "n_train_graphs": len(train_t), "n_val_graphs": len(val_t), "train": tcfg.to_dict()})
    return model, history


@torch.no_grad()
def evaluate_tensors(net: InteractionNetwork, tensors, lossf=None) -> Dict[str, float]:
    net.eval()
    ys, ss, losses = [], [], []
    for x, e, s, d, y in tensors:
        logit = net(x, e, s, d)
        if lossf is not None:
            losses.append(float(lossf(logit, y)) * len(y))
        ys.append(y.numpy())
        ss.append(torch.sigmoid(logit).numpy())
    if not ys:
        return {"val_loss": float("nan"), "auc": float("nan"), "edge_eff": float("nan"), "edge_purity": float("nan")}
    y, s = np.concatenate(ys), np.concatenate(ss)
    out = edge_metrics(y, s)
    out["val_loss"] = float(sum(losses) / max(len(y), 1)) if losses else float("nan")
    return out

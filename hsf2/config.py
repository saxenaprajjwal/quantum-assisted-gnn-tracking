"""Configuration dataclasses shared by all stages (saved next to every result as JSON)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Tuple


class _Cfg:
    """to_dict / from_dict for flat dataclasses (unknown keys in old files are ignored)."""

    def to_dict(self) -> dict:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, d: Optional[dict]):
        d = d or {}
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return cls(**kw)


@dataclass
class HitSelection(_Cfg):
    """Which hits enter the graph.

    S0  "ideal hit filtering" (as DeZoort et al. 2021): only hits of particles with pT >= pt_keep, no noise.
    S1  all hits of the event, including noise and low-pT particles (pt_keep = 0, keep_noise = True).
    """
    level: str = "S0"
    pt_keep: float = 1.0          # GeV
    keep_noise: bool = False

    @classmethod
    def preset(cls, level: str, pt: float = 1.0) -> "HitSelection":
        level = level.upper()
        if level == "S0":
            return cls("S0", float(pt), False)
        if level == "S1":
            return cls("S1", 0.0, True)
        raise ValueError(f"unknown hit level {level!r} (S0 or S1)")


@dataclass
class TargetConfig(_Cfg):
    """Reconstructable ("target") particles: the denominator of the efficiency."""
    pt_min: float = 1.0           # GeV
    eta_max: float = 4.0
    min_hits: int = 5             # hits of the particle among the selected hits
    match_min: float = 0.5        # ACTS truthMatchProbMin: majority-particle fraction of the track's hits


@dataclass
class GraphConfig(_Cfg):
    """Loose 'module-map-lite' graph: layer pairs and per-pair windows learned from truth edges."""
    pt_min_map: float = 1.0       # truth edges of particles above this pT define pairs and windows
    pair_min_count: int = 5       # a layer pair must carry at least this many truth edges ...
    pair_min_frac: float = 1e-4   # ... and this fraction of all truth edges
    quantile: float = 0.999       # window = quantile of |feature| over the truth edges of the pair ...
    margin: float = 1.2           # ... times this margin
    floor_dphi: float = 2e-3      # minimum windows (rad, unit eta, mm)
    floor_deta: float = 2e-3
    floor_z0: float = 20.0
    z0_min_dr: float = 20.0       # mm: no z0 window on pairs whose median |dr| is below this (disks, overlaps)
    cap_quantile: float = 0.999   # every window is capped at cap_margin * this quantile of ALL truth edges:
    cap_margin: float = 1.5       # a few secondaries / loopers (19 of 64,595 TrackML edges have |dphi| > 0.3)
                                  # must not open the windows of rarely used pairs


@dataclass
class ModelConfig(_Cfg):
    """Edge-classifying Interaction Network."""
    hidden: int = 64
    n_iter: int = 3               # message-passing steps
    mlp_layers: int = 2
    shared: bool = True           # one set of weights for all steps (recurrent IN)
    label: str = "consecutive"    # training target: consecutive hits of a particle, or "pid" (same particle)
    node_features: Tuple[str, ...] = ("r", "z", "eta")     # no absolute phi: the detector is phi-symmetric
    edge_features: Tuple[str, ...] = ("dr", "dphi", "dz", "deta", "z0", "phi_slope")


@dataclass
class TrainConfig(_Cfg):
    epochs: int = 20
    lr: float = 2e-3
    lr_min: float = 1e-4          # cosine decay to this
    weight_decay: float = 0.0
    max_pos_weight: float = 20.0  # BCE weight of true edges = min(n_false/n_true, this)
    seed: int = 0
    threads: int = 0              # torch threads (0 = leave as is / OMP_NUM_THREADS)
    max_train_events: int = 0     # 0 = all events of the split
    grad_clip: float = 1.0


# ---------------------------------------------------------------------------
# The tuning problem T1: inference-time knobs (no retraining)
# ---------------------------------------------------------------------------
TRACKING_PARAM_NAMES: List[str] = [
    "dphi_scale",       # graph filter: |dphi| <= dphi_scale * window(pair)     (needs GNN re-inference)
    "deta_scale",       # graph filter: |deta| <= deta_scale * window(pair)     (needs GNN re-inference)
    "z0_scale",         # graph filter: |z0|   <= z0_scale   * window(pair)     (needs GNN re-inference)
    "cut_hi",           # edges >= cut_hi may branch the walk-through (GNN4ITk thAdd)
    "cut_lo_frac",      # edges >= cut_lo_frac * cut_hi are usable (GNN4ITk thMin); 1 = no hysteresis
    "max_degree",       # junction removal: keep the best k incoming and k outgoing edges per hit
    "min_hits",         # minimum hits per track
    "merge_overlap",    # ambiguity resolution: max fraction of a track's hits already used by better tracks
]
TRACKING_INTEGER = ("max_degree", "min_hits")
FILTER_PARAMS = ("dphi_scale", "deta_scale", "z0_scale")

# Ranges re-fixed from the stage-A scans (lxplus, 30-31 Oct 2026, 20 tune events): the 1-D optima of S lie at
# dphi 0.4, deta 0.1, z0 0.075 (S_DM: 0.4, 0.2, 0.1) and S falls off below 0.05 in deta and z0.
TRACKING_RANGES: Dict[str, Tuple[float, float]] = {
    "dphi_scale": (0.3, 1.0),
    "deta_scale": (0.05, 1.0),
    "z0_scale": (0.05, 1.0),
    "cut_hi": (0.2, 0.9),
    "cut_lo_frac": (0.3, 1.0),
    "max_degree": (1, 8),
    "min_hits": (3, 10),
    "merge_overlap": (0.0, 0.7),
}
# Knobs whose 8 levels are spaced geometrically, value = lo (hi/lo)^(level/7): 0.05, 0.077, 0.118, 0.181, 0.277,
# 0.425, 0.652, 1.0.  A linear grid on [0.05, 1] would have one level below 0.18, where S peaks, and the QX1
# sub-grid none; the default 1.0 stays level 7.
TRACKING_LOG = ("deta_scale", "z0_scale")

# "Default" reference configuration, the analogue of the ACTS default in Phase 1: the settings of the ACTS GNN
# examples (edge cut 0.5, connected components with junction removal, >= 3 hits, no sharing, untightened graph).
TRACKING_DEFAULTS: Dict[str, float] = {
    "dphi_scale": 1.0,
    "deta_scale": 1.0,
    "z0_scale": 1.0,
    "cut_hi": 0.5,
    "cut_lo_frac": 1.0,
    "max_degree": 1,
    "min_hits": 3,
    "merge_overlap": 0.0,
}


@dataclass
class TrackingParams(_Cfg):
    dphi_scale: float = 1.0
    deta_scale: float = 1.0
    z0_scale: float = 1.0
    cut_hi: float = 0.5
    cut_lo_frac: float = 1.0
    max_degree: int = 1
    min_hits: int = 3
    merge_overlap: float = 0.0
    max_paths: int = 64           # walk-through: at most this many paths per connected component

    def __post_init__(self) -> None:
        self.max_degree = int(round(self.max_degree))
        self.min_hits = int(round(self.min_hits))

    @property
    def cut_lo(self) -> float:
        return float(self.cut_lo_frac) * float(self.cut_hi)

    def filter_key(self) -> Tuple[float, float, float]:
        return (round(float(self.dphi_scale), 6), round(float(self.deta_scale), 6), round(float(self.z0_scale), 6))


SCORE_EFFICIENCY_KEYS = ("efficiency", "efficiency_dm")


@dataclass
class ScoreConfig(_Cfg):
    """score = eff - fake - dup/k_dup - t/k_time  (Phase-1 objective; t = runtime relative to the default).

    time_mode "proxy" (the tuning objective): t = edges the GNN processes / edges of the unfiltered loose graph
    (= the default configuration, all *_scale = 1), summed over the events -- deterministic, so landscapes
    carry no timing noise.  "measured" uses m["time_rel"] (CPU time relative to the default, stored but not
    used for tuning); "none" drops the time term.  A missing time metric is an error, never silently 0.

    Sensitivity variants (v0.2.4; the defaults give the specified S):
      efficiency_key "efficiency_dm"   eff = double-majority efficiency (S_DM), so fragmented tracks count as lost;
      k_time 20                        a smaller weight of the time term;
      dm_floor f > 0                   - k_floor * max(0, f - eff_DM): a steep linear penalty below a DM-efficiency
                                       floor (a soft constraint that a quadratic surrogate can still follow).
    """
    k_dup: float = 5.0
    k_time: float = 5.0
    time_mode: str = "proxy"
    efficiency_key: str = "efficiency"
    dm_floor: float = 0.0
    k_floor: float = 10.0

    def __post_init__(self) -> None:
        if self.time_mode not in ("proxy", "measured", "none"):
            raise ValueError(f"time_mode must be proxy, measured or none, got {self.time_mode!r}")
        if self.efficiency_key not in SCORE_EFFICIENCY_KEYS:
            raise ValueError(f"efficiency_key must be one of {SCORE_EFFICIENCY_KEYS}, got {self.efficiency_key!r}")
        if not 0.0 <= self.dm_floor < 1.0 or self.k_floor <= 0 or self.k_time <= 0 or self.k_dup <= 0:
            raise ValueError("need 0 <= dm_floor < 1 and positive k_dup, k_time, k_floor")

    def to_dict(self) -> dict:
        """The variant fields only when they differ from the specified S, so S keeps its pre-v0.2.4 dictionary
        (GNNRunner cache keys contain it: caches written by earlier versions stay valid)."""
        d = asdict(self)
        for k, v in (("efficiency_key", "efficiency"), ("dm_floor", 0.0), ("k_floor", 10.0)):
            if d[k] == v:
                del d[k]
        return d

    @property
    def is_default(self) -> bool:
        return self == ScoreConfig()

    def tag(self) -> str:
        """Short name for folder labels: '' for S, else e.g. 'dm', 't20', 'floor0.99'."""
        parts = (["dm"] if self.efficiency_key == "efficiency_dm" else []) + \
            ([f"t{self.k_time:g}"] if self.k_time != 5.0 else []) + \
            ([f"dup{self.k_dup:g}"] if self.k_dup != 5.0 else []) + \
            ([f"floor{self.dm_floor:g}"] if self.dm_floor > 0 else []) + \
            (["notime"] if self.time_mode == "none" else [])
        return "_".join(parts)

    def describe(self) -> str:
        eff = "eff_DM" if self.efficiency_key == "efficiency_dm" else "eff"
        t = "" if self.time_mode == "none" else f" - t/{self.k_time:g}"
        floor = f" - {self.k_floor:g}*max(0, {self.dm_floor:g} - eff_DM)" if self.dm_floor > 0 else ""
        return f"{eff} - fake - dup/{self.k_dup:g}{t}{floor}"

    def time_term(self, m: Dict[str, float]) -> float:
        if self.time_mode == "none":
            return 0.0
        return float(m["time_proxy" if self.time_mode == "proxy" else "time_rel"])

    def score(self, m: Dict[str, float]) -> float:
        return self.physics(m) - self.time_term(m) / self.k_time

    def physics(self, m: Dict[str, float]) -> float:
        """The score without its time term."""
        s = float(m[self.efficiency_key]) - float(m["fakerate"]) - float(m["duplicaterate"]) / self.k_dup
        if self.dm_floor > 0:
            s -= self.k_floor * max(0.0, self.dm_floor - float(m["efficiency_dm"]))
        return s

    def score_arrays(self, m: Dict[str, "np.ndarray"]) -> Tuple["np.ndarray", "np.ndarray"]:
        """(physics, score) for arrays of metrics (exhaustive landscapes); the same formula as score()."""
        import numpy as np
        phys = np.asarray(m[self.efficiency_key], float) - np.asarray(m["fakerate"], float) \
            - np.asarray(m["duplicaterate"], float) / self.k_dup
        if self.dm_floor > 0:
            phys = phys - self.k_floor * np.maximum(0.0, self.dm_floor - np.asarray(m["efficiency_dm"], float))
        t = 0.0 if self.time_mode == "none" else np.asarray(m["time_proxy" if self.time_mode == "proxy" else "time_rel"], float)
        return phys, phys - t / self.k_time


@dataclass
class SplitConfig(_Cfg):
    """Disjoint event blocks (replace the Phase-1 seed strides k*1e6)."""
    n_train: int = 300
    n_val: int = 30
    n_tune: int = 50
    n_confirm: int = 100
    n_confirm_sets: int = 3

    def make(self, event_ids: List[str]) -> Dict[str, List[str]]:
        ids = sorted(event_ids)
        out: Dict[str, List[str]] = {}
        k = 0
        for name, n in [("train", self.n_train), ("val", self.n_val), ("tune", self.n_tune)] + \
                [(f"confirm{i + 1}", self.n_confirm) for i in range(self.n_confirm_sets)]:
            out[name] = ids[k:k + n]
            k += n
        out["unused"] = ids[k:]
        return out


@dataclass
class ExperimentRecord(_Cfg):
    """Everything needed to reproduce one result (written as config.json in each output folder)."""
    command: str = ""
    hits: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)
    graph: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    tracking: dict = field(default_factory=dict)
    data_dir: str = ""
    model_dir: str = ""
    split: str = ""
    events: List[str] = field(default_factory=list)
    code: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

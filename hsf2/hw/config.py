"""Configuration of one hardware command: backend, credentials path, ledger, budget, mode, primitive, shots.

Environment variables (all optional; the command line wins):
  HSF2_IBM_CREDS    path of a credentials file.  It is only passed on to Phase 1 (acts_qaoa.ibm.get_service), which
                    reads it; no hsf2 code opens it, and only its basename is ever written to a result.
  HSF2_QPU_LEDGER   ledger file; else ~/.hsf2_part2_qpu_ledger_<creds basename without extension>.json when
                    HSF2_IBM_CREDS is set, else ~/.hsf2_part2_qpu_ledger.json (one ledger per IBM account)
  HSF2_QPU_BUDGET   QPU seconds per 28-day ledger window (default 400, the owner-approved total)
  HSF2_IBM_BACKEND  backend name (default ibm_kingston, as in scripts/qx_hardware.sh)
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

MODES = ("dry", "local", "submit")
PRIMITIVES = ("sampler_v2", "executor")
DEFAULT_BUDGET_S = 400.0
DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_FAKE = "FakeFez"
DEFAULT_SHOTS = 4000
LEDGER_NAME = ".hsf2_part2_qpu_ledger"


def creds_label(creds_file: Optional[str]) -> Optional[str]:
    """Basename of the credentials file (the only thing of it that may be recorded)."""
    return Path(creds_file).name if creds_file else None


def default_ledger_path(creds_file: Optional[str] = None) -> Path:
    """HSF2_QPU_LEDGER, else one ledger per credentials file, else the shared Part-II ledger."""
    env = os.environ.get("HSF2_QPU_LEDGER")
    if env:
        return Path(env).expanduser()
    creds_file = creds_file or os.environ.get("HSF2_IBM_CREDS")
    if creds_file:
        return Path.home() / f"{LEDGER_NAME}_{Path(creds_file).stem}.json"
    return Path.home() / f"{LEDGER_NAME}.json"


def default_budget() -> float:
    txt = os.environ.get("HSF2_QPU_BUDGET")
    if txt is None or txt == "":
        return DEFAULT_BUDGET_S
    try:
        v = float(txt)
    except ValueError:
        raise ValueError(f"HSF2_QPU_BUDGET must be a number of seconds, got {txt!r}") from None
    if not v > 0:
        raise ValueError(f"HSF2_QPU_BUDGET must be positive, got {v}")
    return v


@dataclass
class HwConfig:
    backend: str = DEFAULT_BACKEND
    creds_file: Optional[str] = None
    ledger: Optional[str] = None
    budget_s: float = DEFAULT_BUDGET_S
    mode: str = "dry"
    primitive: str = "sampler_v2"
    shots: int = DEFAULT_SHOTS
    fake_backend: str = DEFAULT_FAKE
    real_target: bool = False           # dry mode: transpile on the real backend's current target (login, 0 QPU s)
    reserve_s: float = 0.0              # seconds kept free below the budget (a job is refused if it would eat them)
    transpile_seeds: int = 4

    @classmethod
    def from_env(cls, **over) -> "HwConfig":
        """Defaults from the environment; `over` entries that are None keep the default."""
        base = dict(backend=os.environ.get("HSF2_IBM_BACKEND") or DEFAULT_BACKEND,
                    creds_file=os.environ.get("HSF2_IBM_CREDS") or None, budget_s=default_budget())
        base.update({k: v for k, v in over.items() if v is not None})
        cfg = cls(**base)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"primitive must be one of {PRIMITIVES}, got {self.primitive!r}")
        if int(self.shots) < 1:
            raise ValueError(f"shots must be positive, got {self.shots}")
        if not self.budget_s > 0:
            raise ValueError(f"budget must be positive, got {self.budget_s}")
        if int(self.transpile_seeds) < 1:
            raise ValueError("transpile_seeds must be >= 1")

    def ledger_path(self) -> Path:
        return Path(self.ledger).expanduser() if self.ledger else default_ledger_path(self.creds_file)

    def to_dict(self) -> dict:
        """JSON-able record; the credentials path is reduced to its basename."""
        d = asdict(self)
        d["creds_file"] = creds_label(self.creds_file)
        d["ledger"] = str(self.ledger_path())
        return d

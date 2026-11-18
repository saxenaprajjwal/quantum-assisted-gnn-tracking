"""QPU-time ledger of Part II: Phase-1's QPULedger (same JSON entries, so QPULedger and scripts/qx_hardware.sh read the
file unchanged) plus the booking source and the budget guard.

One ledger per IBM account (hsf2.hw.config.default_ledger_path); the budget comes from HSF2_QPU_BUDGET (default 400 s per
28-day window).  IBM's per-job usage is 0 while its usage status is 'pending' (runtime_job_v2.usage), so a booked figure
carries its `usage_source`: job (IBM's per-job figure), account_diff (difference of service.usage() before / after) or
estimate (nothing better was available).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .._phase1 import ensure_phase1
from .config import default_budget, default_ledger_path

ensure_phase1()
from acts_qaoa.ibm import QPULedger  # noqa: E402

log = logging.getLogger(__name__)

WARN_FRACTION = 0.8
USAGE_SOURCES = ("job", "account_diff", "estimate")


class BudgetError(RuntimeError):
    """A job would exceed the QPU time that is left; nothing was submitted."""


class HwLedger(QPULedger):
    def __init__(self, path: Optional[Path] = None, budget_s: Optional[float] = None, window_days: int = 28,
                 creds_file: Optional[str] = None) -> None:
        super().__init__(Path(path) if path else default_ledger_path(creds_file),
                         default_budget() if budget_s is None else float(budget_s), window_days)

    def entry(self, job_id: str) -> Optional[dict]:
        hits = [e for e in self.entries if e["job_id"] == job_id]
        return hits[-1] if hits else None

    def book(self, job_id: str, seconds: Optional[float], source: str, backend: str = "?", run_dir: str = "") -> None:
        """Record the QPU seconds of `job_id` and where the figure comes from (job / account_diff / estimate).
        A job that is not in this ledger yet (submitted elsewhere) is added with a zero estimate."""
        if source not in USAGE_SOURCES:
            raise ValueError(f"usage source must be one of {USAGE_SOURCES}, got {source!r}")
        if seconds is not None and not seconds >= 0:
            raise ValueError(f"QPU seconds must be >= 0, got {seconds}")
        if self.entry(job_id) is None:
            self.add(job_id, backend, 0.0, "hw-booked", run_dir)
        for e in self.entries:
            if e["job_id"] == job_id:
                e["quantum_seconds"] = None if seconds is None else float(seconds)
                e["usage_source"] = source
        self.save()

    def check(self, estimate_s: float, service_remaining_s: Optional[float] = None, reserve_s: float = 0.0,
              force: bool = False) -> float:
        """Raise BudgetError if `estimate_s` exceeds the budget left (the ledger's, and IBM's reported remaining time
        if known), after keeping `reserve_s` free; warn when the window would pass 80 % of the budget.  Returns the
        remaining seconds (before this job)."""
        remaining = self.remaining_seconds() - reserve_s
        if service_remaining_s is not None:
            remaining = min(remaining, float(service_remaining_s) - reserve_s)
        if estimate_s > remaining and not force:
            raise BudgetError(f"estimated {estimate_s:.1f} s exceeds the QPU time left ({remaining:.1f} s after a "
                              f"{reserve_s:.0f} s reserve; ledger {self.path}, budget {self.budget_s:.0f} s"
                              + ("" if service_remaining_s is None else f", IBM reports {service_remaining_s:.0f} s left")
                              + ").  Nothing was submitted.")
        after = self.used_seconds() + estimate_s
        if after > WARN_FRACTION * self.budget_s:
            log.warning("[hw] this job brings the ledger to %.0f s of %.0f s (%.0f %% of the budget)", after,
                        self.budget_s, 100 * after / self.budget_s)
        return remaining

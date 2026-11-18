"""IBM service and backend access (the only place of hsf2.hw that talks to IBM) and the local fake backends.

The credentials file is never opened here: only its path is handed to Phase 1 (acts_qaoa.ibm.get_service, which reads
it).  The path comes from the argument or $HSF2_IBM_CREDS.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .._phase1 import ensure_phase1

log = logging.getLogger(__name__)


def creds_path(creds_file: Optional[str] = None) -> Optional[str]:
    return creds_file or os.environ.get("HSF2_IBM_CREDS") or None


def get_service(creds_file: Optional[str] = None):
    """QiskitRuntimeService from Phase 1's get_service(creds_file=...)  (None: $HSF2_IBM_CREDS, else Phase 1's default)."""
    ensure_phase1()
    from acts_qaoa import ibm
    return ibm.get_service(creds_file=creds_path(creds_file))


def get_backend(service, name: str, fractional: bool = False):
    """service.backend(name, use_fractional_gates=...).  A runtime that does not know the keyword is an error for the
    fractional setting (Phase 1 would silently fall back to the normal backend, and the setting would be wrong)."""
    try:
        return service.backend(name, use_fractional_gates=bool(fractional))
    except TypeError:
        if fractional:
            raise RuntimeError("this qiskit-ibm-runtime does not support use_fractional_gates: setting (e) "
                               "is not available") from None
        return service.backend(name)


def service_remaining(service) -> Optional[float]:
    ensure_phase1()
    from acts_qaoa import ibm
    return ibm.service_remaining_seconds(service)


def account_consumed(service) -> Optional[float]:
    """service.usage()['usage_consumed_seconds'] (None if this plan / runtime does not report it)."""
    try:
        v = service.usage().get("usage_consumed_seconds")
        return None if v is None else float(v)
    except Exception as exc:
        log.warning("[hw] account usage not available (%s)", type(exc).__name__)
        return None

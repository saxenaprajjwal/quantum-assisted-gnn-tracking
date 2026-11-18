"""Step-4 error-mitigation hooks (stubs): the Mitigator protocol and its not yet implemented members.

A mitigator rewrites the PUB list before submission (`prepare`) and turns the raw result into a corrected one
afterwards (`postprocess`).  Phase A implements none of them; each stub raises NotImplementedError with the reason, so
a run can never silently continue without the mitigation it asked for.
"""
from __future__ import annotations

from typing import Any, List, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Mitigator(Protocol):
    name: str

    def prepare(self, pubs: Sequence[Any]) -> List[Any]:
        """PUBs -> PUBs to submit (extra circuits, modified options, ...)."""

    def postprocess(self, result: Any) -> Any:
        """Raw job result -> mitigated result."""


class _Stub:
    name = "stub"
    why = ""

    def prepare(self, pubs):
        raise NotImplementedError(f"{self.name}: {self.why} (step 4, not implemented in Phase A)")

    def postprocess(self, result):
        raise NotImplementedError(f"{self.name}: {self.why} (step 4, not implemented in Phase A)")


class TREX(_Stub):
    name = "trex"
    why = "twirled readout error extinction needs the experimental measurement-twirl calibration of the executor primitive"


class ZNE(_Stub):
    name = "zne"
    why = "zero-noise extrapolation needs noise-amplified circuit folding and an extrapolation model"


class PEC(_Stub):
    name = "pec"
    why = "probabilistic error cancellation needs a learned noise model of the twirled layers"


class NoiseLearner(_Stub):
    name = "noise_learner"
    why = "noise learning of the used layers is a separate job type that is not wired into the runner"


MITIGATORS = {c.name: c for c in (TREX, ZNE, PEC, NoiseLearner)}


def get_mitigator(name: str) -> Mitigator:
    if name not in MITIGATORS:
        raise ValueError(f"unknown mitigator {name!r}; known: {sorted(MITIGATORS)}")
    return MITIGATORS[name]()

"""The error-suppression ladder of step 2: named settings a-f -> Sampler option dictionaries.

  a bare       gate twirling off, measurement twirling off, DD off
  b DD         off, off, XY4
  c twirl      on, on, off
  d both       on, on, XY4          (the current setting; repeated at the end of the ladder to expose drift)
  e fractional off, on, XY4         on a use_fractional_gates=True backend (native RZZ; angles folded into (0, pi/2])
  f gauges     on, on, XY4          8 spin-reversal gauges x 500 shots

Gate and measurement twirling are separate switches (Phase 1's --no-twirl switches both together).  The option names
are those of qiskit-ibm-runtime's SamplerOptions (verified in 0.51.0): twirling.enable_gates, twirling.enable_measure,
twirling.num_randomizations, twirling.shots_per_randomization, dynamical_decoupling.enable / .sequence_type.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

log = logging.getLogger(__name__)

NUM_RANDOMIZATIONS = 16
DD_SEQUENCE = "XY4"
DEFAULT_GAUGES = 8


@dataclass(frozen=True)
class Setting:
    name: str
    label: str
    gate_twirl: bool
    measure_twirl: bool
    dd: bool
    fractional: bool = False
    gauged: bool = False
    randomizations: int = NUM_RANDOMIZATIONS        # twirling.num_randomizations (folding diagnosis: 32, see with_randomizations)

    def options(self) -> Dict[str, object]:
        """Dotted SamplerOptions paths -> values (every switch explicit, nothing left at a default)."""
        return {"twirling.enable_gates": self.gate_twirl,
                "twirling.enable_measure": self.measure_twirl,
                "twirling.num_randomizations": self.randomizations,
                "twirling.shots_per_randomization": "auto",
                "dynamical_decoupling.enable": self.dd,
                "dynamical_decoupling.sequence_type": DD_SEQUENCE}

    def to_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "gate_twirl": self.gate_twirl,
                "measure_twirl": self.measure_twirl, "dd": self.dd, "fractional": self.fractional,
                "gauged": self.gauged, "options": self.options()}


SETTINGS: Dict[str, Setting] = {s.name: s for s in (
    Setting("a", "bare", False, False, False),
    Setting("b", "DD", False, False, True),
    Setting("c", "twirl", True, True, False),
    Setting("d", "both", True, True, True),
    # (e): gate twirling stays off, because the fractional RZZ cannot be twirled (A, from the IBM documentation)
    Setting("e", "fractional", False, True, True, fractional=True),
    Setting("f", "gauges", True, True, True, gauged=True),
)}


def get_setting(name: str) -> Setting:
    if name not in SETTINGS and len(name) > 1 and name[0] in SETTINGS and name[1:].isdigit():
        return with_randomizations(SETTINGS[name[0]], int(name[1:]))        # a name written by with_randomizations: d32
    if name not in SETTINGS:
        raise ValueError(f"unknown setting {name!r}; known: {sorted(SETTINGS)}")
    return SETTINGS[name]


def with_randomizations(setting: Setting, n: int) -> Setting:
    """`setting` with n twirling randomisations (a copy named <name><n>, e.g. d32; n = 16 returns it unchanged).  Only settings
    with gate or measurement twirling have randomisations."""
    import dataclasses
    n = int(n)
    if n < 1:
        raise ValueError(f"the number of twirling randomisations must be positive, got {n}")
    if n == setting.randomizations:
        return setting
    if not (setting.gate_twirl or setting.measure_twirl):
        raise ValueError(f"setting {setting.name} has no twirling: a number of randomisations makes no sense")
    return dataclasses.replace(setting, name=f"{setting.name}{n}", randomizations=n)


def apply_options(options, opts: Dict[str, object], strict: bool = True) -> Dict[str, object]:
    """Set dotted option paths on a SamplerOptions object; returns what was applied.

    strict: an option that this runtime version rejects raises (submit mode must never silently run a different
    setting).  strict=False only warns (local mode, where the simulator ignores most of them anyway)."""
    applied: Dict[str, object] = {}
    for path, value in opts.items():
        parts: Tuple[str, ...] = tuple(path.split("."))
        try:
            obj = options
            for a in parts[:-1]:
                obj = getattr(obj, a)
            if not hasattr(obj, parts[-1]):
                raise AttributeError(f"no option {parts[-1]!r}")
            setattr(obj, parts[-1], value)
            if getattr(obj, parts[-1]) != value:
                raise ValueError(f"reads back {getattr(obj, parts[-1])!r}")
            applied[path] = value
        except Exception as exc:
            msg = f"Sampler option {path}={value!r} cannot be set in this qiskit-ibm-runtime ({type(exc).__name__}: {exc})"
            if strict:
                raise RuntimeError(msg) from exc
            log.warning(msg)
    return applied


def options_dump(options) -> dict:
    """The complete option tree of a SamplerOptions object as plain JSON types (for the manifest)."""
    def conv(x):
        if hasattr(x, "__dataclass_fields__"):
            return {k: conv(getattr(x, k)) for k in x.__dataclass_fields__ if not k.startswith("_")}
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return f"<{type(x).__name__}>" if type(x).__name__ == "UnsetType" else str(x)
    return conv(options)

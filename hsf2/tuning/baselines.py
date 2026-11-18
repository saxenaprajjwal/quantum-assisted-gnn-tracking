"""Black-box baselines on the configuration grid, with a budget of B DISTINCT real evaluations each.

All three search exactly the grid the QUBO pipeline searches (the configurations of a Phase-1 Encoding), so
differences between methods come from the search, not from the search space.  The runner is a TaggedView, whose
record list (in order of first evaluation) is the method's trajectory.

random-grid  B distinct configurations drawn uniformly without replacement
tpe-grid     Optuna TPE over the integer levels of every knob -- Phase-1 baselines.run_optuna(on_grid=True) -- but
             run until B distinct configurations are evaluated (repeated suggestions are cached and free)
sa-grid      simulated annealing on the real objective: move = one knob to a neighbouring level, Metropolis
             acceptance, geometric temperature T0 -> T1 over the budget (temperatures in score units)
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def _distinct(runner) -> int:
    return len(runner.records())


def run_random_grid(runner, enc, budget: int, seed: int, tag: str) -> None:
    rng = np.random.default_rng(seed)
    n = 1 << enc.n_qubits
    for idx in rng.choice(n, size=min(budget, n), replace=False):
        runner.evaluate(enc.params_from_index(int(idx)), tag=tag)


def run_tpe_grid(runner, enc, budget: int, seed: int, tag: str, max_trials_factor: int = 50) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    names = enc.space.names

    def objective(trial):
        levels = [trial.suggest_int(n, 0, enc.mask) for n in names]
        obj = runner.evaluate(enc.params_from_levels(levels), tag=tag).get("objective")
        return float("nan") if obj is None else float(obj)

    def stop(study, trial):
        if _distinct(runner) >= budget:
            study.stop()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=max_trials_factor * budget, callbacks=[stop])


def run_sa_grid(runner, enc, budget: int, seed: int, tag: str, t0: float = 0.02, t1: float = 5e-4,
                max_steps_factor: int = 50) -> None:
    rng = np.random.default_rng(seed)
    levels = rng.integers(0, enc.n_levels, enc.n_params)
    cur = runner.evaluate(enc.params_from_levels(levels), tag=tag).get("objective")
    cur = -math.inf if cur is None else cur
    steps = 0
    while _distinct(runner) < budget and steps < max_steps_factor * budget:
        steps += 1
        frac = min(_distinct(runner) / max(budget - 1, 1), 1.0)
        T = t0 * (t1 / t0) ** frac
        cand = levels.copy()
        p = int(rng.integers(enc.n_params))
        step = int(rng.choice([-1, 1]))
        cand[p] = min(max(cand[p] + step, 0), enc.mask)
        if cand[p] == levels[p]:
            cand[p] = levels[p] - step                     # at the edge: move inwards
        val = runner.evaluate(enc.params_from_levels(cand), tag=tag).get("objective")
        if val is None:
            continue
        if val >= cur or rng.random() < math.exp((val - cur) / T):
            levels, cur = cand, val


BASELINES = {"random-grid": run_random_grid, "tpe-grid": run_tpe_grid, "sa-grid": run_sa_grid}


def trajectory(records: List[dict], truth=None) -> Dict[str, list]:
    """Best-so-far (by the runner's objective, and by `truth(record)` if given) over distinct evaluations."""
    best, best_true, out_b, out_t, chosen = -math.inf, None, [], [], None
    for r in records:
        o = r.get("objective")
        if o is not None and o > best:
            best, chosen = o, r
            best_true = truth(r) if truth else o
        out_b.append(best)
        out_t.append(best_true)
    return {"best_so_far": out_b, "best_true_so_far": out_t}

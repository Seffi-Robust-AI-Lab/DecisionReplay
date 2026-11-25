"""Simplex stacking of cross-fitted, recalibrated members."""

import numpy as np
from scipy.optimize import minimize

from data.schema import N_FOLDS, roles
from ..types import LOG_FLOOR
from .uniform_mean import member_choices, panel

NAME = "stack"


def simplex_fit(mass):
    """Return simplex weights maximising mean log score on informative fields."""
    masses = np.asarray(mass, dtype=np.float64)
    n_models = masses.shape[1]
    informative = np.any(masses > 0, axis=1) & np.isfinite(masses).all(axis=1)
    initial = np.full(n_models, 1.0 / n_models)
    if not informative.any() or n_models == 1:
        return initial
    x = np.maximum(masses[informative], LOG_FLOOR)
    field_weights = np.full(len(x), 1.0 / len(x))

    def objective(w):
        totals = np.maximum(x @ w, LOG_FLOOR)
        return -(field_weights * np.log(totals)).sum(), -(field_weights / totals) @ x

    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}
    fit = minimize(objective, initial, method="SLSQP", jac=True, bounds=[(0.0, 1.0)] * n_models,
                   constraints=[constraint], options={"ftol": 1e-12, "maxiter": 600})
    w = np.maximum(fit.x, 0)
    w /= w.sum()
    sparse = np.where(w < 1e-10, 0.0, w)
    sparse /= sparse.sum()
    return sparse if objective(sparse)[0] <= objective(w)[0] + 1e-10 else w


def weights(task, fold):
    """Return the simplex weights fitted on calibrated masses in the three fit folds."""
    cache = vars(task).setdefault("_stack_weights", {})
    if fold not in cache:
        fit_rows = task.rows(roles(fold)[0])
        cache[fold] = simplex_fit(task.view(fold, "calibrated")[fit_rows])
    return cache[fold]


def run(task, settings):
    """Return the calibrated simplex pool and its fitted weights."""
    fitted = np.stack([weights(task, f) for f in range(N_FOLDS)])
    return panel(task, fitted, member_choices(task),
                 ledger={"labeled_folds": "fit", "models_profiled": task.b.M})

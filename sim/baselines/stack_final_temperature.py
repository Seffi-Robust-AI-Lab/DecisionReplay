"""Final temperature chosen on out-of-fold predictions of the complete panel."""

import numpy as np

from data.schema import N_FOLDS, roles
from ..core import loss_of_mass
from . import stack
from .uniform_mean import equal_weights, member_choices, panel
from .uniform_temperature import best_temperature, fill_tempered_mass, menu

NAME = "stack_final_temperature"


def _inner_weights(task, folds):
    """Return simplex weights on two folds, with each calibrated using only the other."""
    cache = vars(task).setdefault("_inner_stack", {})
    if folds not in cache:
        rows = task.rows(folds)
        first, second = folds
        choice = np.where(task.fold[rows, None] == first,
                          task.recal_choice([second]), task.recal_choice([first]))
        mass = task.mass_menu[choice, rows[:, None], np.arange(task.b.M)]
        cache[folds] = stack.simplex_fit(mass)
    return cache[folds]


def temperatures(task, settings, equal=False):
    """Return five final temperatures fitted on cross-fitted complete pools."""
    candidates = menu(settings)
    cache = vars(task).setdefault("_pool_temperatures", {})
    key = candidates, equal
    if key not in cache:
        chosen = []
        for f in range(N_FOLDS):
            fit = roles(f)[0]
            mass = np.empty((len(candidates), task.nJ))
            for g in fit:
                two = tuple(h for h in fit if h != g)
                weights = equal_weights(task)[0] if equal else _inner_weights(task, two)
                fill_tempered_mass(task, mass, task.fold_blocks([g], judged_only=True),
                                   weights, task.recal_choice(two), candidates)
            losses = loss_of_mass(mass[:, task.rows(fit)]).mean(axis=1)
            chosen.append(candidates[best_temperature(losses, candidates)])
        cache[key] = chosen
    return cache[key]


def run(task, settings):
    """Return the calibrated simplex pool after its fitted final temperature."""
    weights = np.stack([stack.weights(task, f) for f in range(N_FOLDS)])
    return panel(task, weights, member_choices(task), temperatures(task, settings),
                 ledger={"labeled_folds": "fit", "models_profiled": task.b.M})

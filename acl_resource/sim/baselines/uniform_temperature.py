"""Final temperature fitted on the original equal-weight pool."""

import numpy as np

from ...data.schema import N_FOLDS, roles
from ..core import loss_of_mass, temper
from .uniform_mean import equal_weights, panel

NAME = "uniform_temperature"


def menu(settings):
    """Return the ordered temperature candidates specified by the experiment."""
    grid = settings["temperature_menu"]
    return tuple(float(grid["base"] ** (i / grid["steps_per_octave"]))
                 for i in range(grid["lowest_step"], grid["highest_step"] + 1))


def best_temperature(losses, candidates):
    """Return the minimum-loss candidate index, favouring temperatures close to one on ties."""
    losses = np.asarray(losses, dtype=np.float64)
    minimum = losses.min()
    tolerance = 32 * np.finfo(np.float64).eps * max(1.0, abs(minimum))
    tied = np.flatnonzero(losses <= minimum + tolerance)
    return int(min(tied, key=lambda i: (abs(np.log(candidates[i])), candidates[i])))


def fill_tempered_mass(task, table, blocks, weights, choices, candidates):
    """Fill and return menu masses for judged fields of the given width blocks."""
    for w, sel in blocks.items():
        rows = task.jpos[task.b.idx[w][sel]]
        vectors = task.pool_fields(w, weights, choices, sel)
        for j, temp in enumerate(candidates):
            table[j, rows] = task.block_mass(w, temper(vectors, temp), sel)
    return table


def flat_tables(task, settings):
    """Return temperature-by-field masses, loss sums by fold, and judged counts."""
    candidates = menu(settings)
    cache = vars(task).setdefault("_flat_tables", {})
    if candidates not in cache:
        mass = np.empty((len(candidates), task.nJ))
        blocks = task.fold_blocks(range(N_FOLDS), judged_only=True)
        fill_tempered_mass(task, mass, blocks, equal_weights(task)[0], None, candidates)
        loss = loss_of_mass(mass)
        sums = np.stack([loss[:, task.fold == g].sum(axis=1) for g in range(N_FOLDS)], axis=1)
        cache[candidates] = mass, sums, np.bincount(task.fold, minlength=N_FOLDS)
    return cache[candidates]


def run(task, settings):
    """Return the equal-weight pool with a fitted temperature for each rotation."""
    _, sums, counts = flat_tables(task, settings)
    candidates = menu(settings)
    temperatures = []
    for f in range(N_FOLDS):
        fit = list(roles(f)[0])
        losses = sums[:, fit].sum(axis=1) / counts[fit].sum()
        temperatures.append(candidates[best_temperature(losses, candidates)])
    return panel(task, equal_weights(task), temperatures=temperatures,
                 ledger={"labeled_folds": "fit", "models_profiled": task.b.M})

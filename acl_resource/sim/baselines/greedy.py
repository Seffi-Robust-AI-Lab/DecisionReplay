"""Forward panel selection with replacement and paired size selection."""

import numpy as np

from ...data.schema import N_FOLDS, roles
from ..core import loss_of_mass
from ..stats import select_size
from .uniform_mean import member_choices, panel

NAME = "greedy"


def path(task, fold):
    """Return prefix weights, from one to as many additions as there are models."""
    cache = vars(task).setdefault("_greedy_paths", {})
    if fold not in cache:
        mass = task.view(fold, "calibrated")[task.rows(roles(fold)[0])]
        n_models = task.b.M
        unit = np.eye(n_models)
        counts = np.zeros(n_models)
        prefixes = np.empty_like(unit)
        for size in range(1, n_models + 1):
            candidates = (counts[None, :] + unit) / size
            losses = loss_of_mass(mass @ candidates.T).mean(axis=0)
            counts[int(np.argmin(losses))] += 1
            prefixes[size - 1] = counts / size
        cache[fold] = prefixes
    return cache[fold]


def _size(task, fold, settings):
    """Return the smallest prefix within one paired standard error on validation."""
    validation = roles(fold)[1]
    rows = task.rows([validation])
    groups = np.flatnonzero(task.group_fold == validation)
    positions = np.searchsorted(groups, task.group[rows])
    utilities = -loss_of_mass(task.view(fold, "calibrated")[rows] @ path(task, fold).T)
    sums = np.zeros((len(groups), task.b.M))
    np.add.at(sums, positions, utilities)
    counts = np.bincount(positions, minlength=len(groups))
    seed = settings["seed_base"] + settings["seed_per_catalogue"] * task.cid + fold
    rng = np.random.default_rng([seed])
    probs = np.full(len(groups), 1.0 / len(groups))
    weights = rng.multinomial(len(groups), probs, size=settings["draws"])
    den = weights @ counts
    with np.errstate(invalid="ignore", divide="ignore"):
        replicates = (weights @ sums) / den[:, None]
    return select_size(utilities.mean(axis=0), replicates, n_groups=len(groups)).k


def run(task, settings):
    """Return the validation-selected prefix of the calibrated greedy panel."""
    sizes = [_size(task, f, settings["size_selection"]) for f in range(N_FOLDS)]
    weights = [path(task, f)[size - 1] for f, size in enumerate(sizes)]
    return panel(task, weights, member_choices(task), info={"size": sizes},
                 ledger={"labeled_folds": "fit and validation", "models_profiled": task.b.M})

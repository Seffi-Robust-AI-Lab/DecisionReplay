"""Selection of a single model by calibrated log loss."""

import numpy as np

from data.schema import N_FOLDS, roles
from .uniform_mean import member_choices, panel

NAME = "best_single"


def ranking(task, fold, n=3):
    """Return model indices by cross-fitted loss on the three fit or four training folds."""
    cache = vars(task).setdefault("_model_ranking", {})
    if (fold, n) not in cache:
        fit, _, train = roles(fold)
        loss = task.fit_loss(fold, np.eye(task.b.M), "calibrated", fit if n == 3 else train, n)
        cache[fold, n] = np.argsort(loss, kind="stable")
    return cache[fold, n]


def run(task, settings):
    """Return the best model selected and calibrated on the four non-test folds."""
    models = [int(ranking(task, f, n=4)[0]) for f in range(N_FOLDS)]
    weights = np.eye(task.b.M)[models]
    return panel(task, weights, member_choices(task, 4), info={"model": models},
                 ledger={"labeled_folds": "train4", "models_profiled": task.b.M})

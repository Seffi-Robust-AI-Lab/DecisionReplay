"""Equal weighting and shared evaluation of linear panels."""

import numpy as np

from ...data.schema import N_FOLDS
from ..core import temper
from ..types import NOT_CALLED, Outcome, empty_ledger

NAME = "uniform_mean"


def equal_weights(task):
    """Return equal weights for every model and rotation."""
    return np.full((N_FOLDS, task.b.M), 1.0 / task.b.M)


def member_choices(task, n=3):
    """Return the deployment transform per rotation and model."""
    return np.stack([task.recal(f, n)[0] for f in range(N_FOLDS)])


def panel(task, weights, choices=None, temperatures=None, ledger=None, info=None):
    """Return a stitched linear pool and its simultaneous positive-weight calls."""
    weights = np.asarray(weights, dtype=np.float64)
    pooled = task.stitch(weights, choices)
    stage = np.where(weights[task.b.fold] > 0, 0, NOT_CALLED).astype(np.int16)
    decisions = {"weights": weights.tolist()}
    if temperatures is not None:
        for w, vectors in pooled.items():
            folds = task.field_fold[task.b.idx[w]]
            for f, temperature in enumerate(temperatures):
                if temperature != 1.0:
                    sel = folds == f
                    vectors[sel] = temper(vectors[sel], temperature)
        decisions["temperature"] = list(map(float, temperatures))
    if choices is not None:
        decisions["member_transform"] = np.asarray(choices).tolist()
    if info:
        decisions.update(info)
    knowledge = empty_ledger()
    if ledger:
        knowledge.update(ledger)
    return Outcome(pooled, stage, decisions, knowledge)


def run(task, settings):
    """Return the equal-weight pool of original vectors."""
    return panel(task, equal_weights(task))

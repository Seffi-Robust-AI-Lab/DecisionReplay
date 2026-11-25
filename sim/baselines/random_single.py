"""Choose one original model uniformly for each request."""

import numpy as np

from ..core import transform
from ..types import MENU, NOT_CALLED, Outcome, empty_ledger
from data.schema import N_FOLDS

NAME = "random_single"


def request_vectors(task, models, choices=None):
    """Return each field's vector from its request's model, optionally calibrated by fold."""
    b = task.b
    pooled = {}
    for w, ids in b.idx.items():
        requests = b.field_row[ids]
        members = models[requests]
        vectors = b.P[w][np.arange(len(ids)), members]
        if choices is not None:
            menu = choices[b.fold[requests], members]
            for c in np.unique(menu):
                selected = menu == c
                vectors[selected] = transform(vectors[selected], *MENU[c])
        pooled[w] = vectors
    return pooled


def single_stage(task, models):
    """Return a plan calling the selected model once per request."""
    stage = np.full((task.R, task.b.M), NOT_CALLED, np.int16)
    stage[np.arange(task.R), models] = 0
    return stage


def run(task, settings):
    """Return the seeded single-model replay and call plan."""
    models = np.empty(task.R, np.int32)
    seed = settings[NAME]["seed"]
    for f in range(N_FOLDS):
        rows = np.flatnonzero(task.b.fold == f)
        rng = np.random.default_rng([seed, task.cid, f])
        models[rows] = rng.integers(task.b.M, size=len(rows))
    return Outcome(request_vectors(task, models), single_stage(task, models), {"seed": seed}, empty_ledger())

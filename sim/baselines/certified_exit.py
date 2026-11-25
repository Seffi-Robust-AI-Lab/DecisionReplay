"""Stop a stack panel once its remaining weight cannot change any request answer."""

import math

import numpy as np

from data.schema import N_FOLDS
from ..core import transform
from ..types import MENU, NOT_CALLED, Outcome, empty_ledger
from . import stack

NAME = "certified_exit"


def run(task, settings):
    """Return the first certified request prefixes and the partial weighted pools."""
    b = task.b
    pooled = {w: np.zeros((len(ids), w)) for w, ids in b.idx.items()}
    stage = np.full((b.R, b.M), NOT_CALLED, np.int16)
    info = {"weights": [], "order": []}
    for f in range(N_FOLDS):
        weights = stack.weights(task, f)
        weights = weights / weights.sum()
        order = np.argsort(-weights, kind="stable")
        order = order[weights[order] > 0]
        info["weights"].append(weights.tolist())
        info["order"].append(order.tolist())
        choices = task.recal(f)[0]
        guard = 32 * np.finfo(np.float64).eps * (len(order) + 1)
        active = b.fold == f
        for depth, m in enumerate(order):
            stage[active, m] = depth
            remaining = math.fsum(weights[order[depth + 1:]])
            unresolved = np.zeros(b.R, bool)
            for w, ids in b.idx.items():
                rows = b.field_row[ids]
                pos = np.flatnonzero(active[rows])
                if not len(pos):
                    continue
                if depth == len(order) - 1:
                    # All members were called. Match the panel's addition order even for numerical ties.
                    pooled[w][pos] = task.pool_fields(w, stack.weights(task, f), choices, pos)
                    continue
                vectors = transform(b.P[w][pos, m, :], *MENU[choices[m]])
                pooled[w][pos] += weights[m] * vectors
                two = np.partition(pooled[w][pos], w - 2, axis=1)[:, -2:]
                margin = two[:, 1] - two[:, 0]
                unresolved[rows[pos[~(margin > remaining + guard)]]] = True
            active = unresolved
            if not active.any():
                break
    ledger = empty_ledger()
    ledger.update(labeled_folds="fit", models_profiled=b.M)
    return Outcome(pooled, stage, info, ledger)

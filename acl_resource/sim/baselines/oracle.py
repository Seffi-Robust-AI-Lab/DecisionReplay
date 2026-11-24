"""Request-wise oracle over the original single-model predictions."""

import numpy as np

from ..types import Outcome, empty_ledger
from .majority_vote import votes
from .random_single import request_vectors, single_stage

NAME = "oracle"


def run(task, settings):
    """Return the model with the most correct judged fields on each request."""
    b = task.b
    counts = np.zeros((b.R, b.M), np.int32)
    for w, ids in b.idx.items():
        judged = b.judged[ids]
        rows = b.field_row[ids[judged]]
        positions = np.flatnonzero(judged)
        for m in range(b.M):
            right = b.acc[w][positions, votes(b, w, m)[judged]]
            np.add.at(counts[:, m], rows, right)
    models = counts.argmax(axis=1)
    ledger = empty_ledger()
    ledger.update(labeled_folds="test", models_profiled=b.M, note="Per-request selection using test labels.")
    return Outcome(request_vectors(task, models), single_stage(task, models), {}, ledger)

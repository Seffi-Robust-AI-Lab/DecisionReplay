"""Vote shares from the decoded options of all models."""

import numpy as np

from ..core import renormalise
from ..types import Outcome, empty_ledger

NAME = "majority_vote"


def votes(bench, width, model):
    """Return one model's decoded options in a width block without using labels."""
    ids = bench.idx[width]
    vectors = renormalise(bench.P[width][:, model, :])
    options = vectors.argmax(axis=1)
    yes_no = bench.is_noul[ids]
    if yes_no.any():
        options[yes_no] = vectors[yes_no, 1] >= 0.5
    return options


def run(task, settings):
    """Return fieldwise vote shares and a one-stage all-model plan."""
    b = task.b
    pooled = {}
    for w, ids in b.idx.items():
        counts = np.zeros((len(ids), w))
        positions = np.arange(len(ids))
        for m in range(b.M):
            counts[positions, votes(b, w, m)] += 1
        pooled[w] = counts / b.M
    return Outcome(pooled, np.zeros((b.R, b.M), np.int16), {}, empty_ledger())

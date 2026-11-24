"""Unsupervised reliability weighting from agreement on the four non-test folds."""

import numpy as np

from ...data.schema import N_FOLDS
from .uniform_mean import panel

NAME = "skill_pool"


def votes(bench):
    """Return each model's decoded option on every field."""
    out = np.empty((bench.F, bench.M), dtype=np.int64)
    for w, vectors in bench.P.items():
        ids = bench.idx[w]
        win = vectors.argmax(axis=2)
        yes_no = bench.is_noul[ids]
        if yes_no.any():
            win[yes_no] = vectors[yes_no, :, 1] >= 0.5
        out[ids] = win
    return out


def _patterns(vote, width):
    """Return distinct widths and agreement patterns with their field counts."""
    n_fields, n_models = vote.shape
    codes = np.empty_like(vote)
    next_code = np.ones(n_fields, dtype=np.int64)
    codes[:, 0] = 0
    rows = np.arange(n_fields)
    for m in range(1, n_models):
        match = vote[:, :m] == vote[:, m, None]
        new = ~match.any(axis=1)
        codes[:, m] = np.where(new, next_code, codes[rows, match.argmax(axis=1)])
        next_code += new
    patterns, count = np.unique(np.column_stack([width, codes]), axis=0, return_counts=True)
    return patterns[:, 0], patterns[:, 1:], count


def _skills(accuracy, widths, counts):
    """Return skills giving the fitted mean accuracies over the observed field widths."""
    lo, hi = np.full(len(accuracy), -60.0), np.full(len(accuracy), 60.0)
    widths = widths.astype(np.float64)
    total = counts.sum()
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        correct = 1.0 / (1.0 + (widths[None, :] - 1.0) * np.exp(-mid[:, None]))
        mean = (counts[None, :] * correct).sum(axis=1) / total
        up = mean < accuracy
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def em_skills(vote, width, settings):
    """Return model skills fitted by EM on unlabeled vote patterns."""
    n_fields, n_models = vote.shape
    if n_fields == 0:
        return np.zeros(n_models)
    widths, counts = np.unique(width, return_counts=True)
    bounds = settings["accuracy_bounds"]
    accuracy = np.clip(np.full(n_models, settings["start_accuracy"]), *bounds)
    pattern_width, code, count = _patterns(vote, width)
    offset = np.concatenate([[0], np.cumsum(pattern_width)[:-1]])
    cell = (offset[:, None] + code).ravel()
    patterns = np.repeat(np.arange(len(count)), pattern_width)
    for _ in range(settings["max_iter"]):
        skill = _skills(accuracy, widths, counts)
        support = np.bincount(cell, weights=np.tile(skill, len(count)), minlength=len(patterns))
        evidence = np.exp(support)
        total = np.bincount(patterns, weights=evidence, minlength=len(count))
        posterior = (evidence[cell] / np.repeat(total, n_models)).reshape(len(count), n_models)
        updated = np.clip((count @ posterior) / n_fields, *bounds)
        change = np.abs(updated - accuracy).max()
        accuracy = updated
        if change < settings["tolerance"]:
            break
    return _skills(accuracy, widths, counts)


def run(task, settings):
    """Return a pool weighted by nonnegative skills fitted without labels."""
    vote = votes(task.b)
    weights = []
    for f in range(N_FOLDS):
        train = task.field_fold != f
        skill = em_skills(vote[train], task.b.width[train].astype(np.int64), settings[NAME])
        positive = np.maximum(skill, 0.0)
        weights.append(positive / positive.sum() if positive.sum() > 0
                       else np.full(task.b.M, 1.0 / task.b.M))
    return panel(task, weights, ledger={"unlabeled_outputs": "train4", "models_profiled": task.b.M})

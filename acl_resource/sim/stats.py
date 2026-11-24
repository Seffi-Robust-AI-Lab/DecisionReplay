"""Paired split-group resampling, suite contrasts and validation size selection."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Selection:
    """Selected prefix, validation winner, paired errors and selection status."""

    k: int
    best_k: int
    paired_se: np.ndarray
    fallback: bool
    reason: str


def select_size(utilities, bootstrap_utilities, *, n_groups=None):
    """Return the smallest prefix within one paired standard error of the validation winner."""
    scores = np.asarray(utilities, dtype=np.float64)
    draws = np.asarray(bootstrap_utilities, dtype=np.float64)
    if scores.ndim != 1 or not scores.size:
        raise ValueError("Expected a nonempty prefix utility vector")
    if draws.ndim != 2 or draws.shape[1] != scores.size:
        raise ValueError("Expected one bootstrap column per prefix")
    errors = np.full(scores.shape, np.nan)
    if not np.isfinite(scores).any():
        return Selection(1, 1, errors, True, "no_finite_audit_utility")
    best = int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))
    if (n_groups is not None and n_groups < 2) or len(draws) < 2:
        return Selection(1, best + 1, errors, True, "insufficient_source_groups_or_draws")
    required = max(2, int(np.ceil(0.95 * len(draws))))
    winner_finite = np.isfinite(draws[:, best])
    if winner_finite.sum() < required:
        return Selection(1, best + 1, errors, True, "audit_winner_undefined_in_bootstrap")
    for j in np.flatnonzero(np.isfinite(scores)):
        valid = np.isfinite(draws[:, j]) & winner_finite
        if valid.sum() >= required:
            errors[j] = (draws[valid, best] - draws[valid, j]).std(ddof=1)
    within = np.flatnonzero(np.isfinite(errors) & (scores[best] - scores <= errors + 1e-12))
    if not within.size:
        return Selection(1, best + 1, errors, True, "unscorable_paired_audit")
    return Selection(int(within[0]) + 1, best + 1, errors, False, "paired_one_standard_error")


def group_multiplicities(group_folds, n_draws, *, seed, chunk_size=48):
    """Yield split-group multiplicities by draw, stratified by fold and independent of chunk size."""
    folds = np.asarray(group_folds)
    if folds.ndim != 1 or not len(folds) or n_draws < 1 or chunk_size < 1:
        raise ValueError("Invalid bootstrap dimensions")
    strata = [np.flatnonzero(folds == f) for f in np.unique(folds)]
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(strata))]
    for start in range(0, n_draws, chunk_size):
        count = min(n_draws - start, chunk_size)
        weights = np.zeros((count, len(folds)), dtype=np.float64)
        for groups, rng in zip(strata, streams):
            n = len(groups)
            sampled = rng.integers(n, size=(count, n), dtype=np.int64)
            shifted = sampled + (np.arange(count, dtype=np.int64) * n)[:, None]
            weights[:, groups] = np.bincount(shifted.ravel(), minlength=count * n).reshape(count, n)
        yield weights


def interval_and_test(delta, replicates):
    """Return the percentile interval, centred two-sided p-value and finite draw count."""
    values = np.asarray(replicates, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Expected one paired difference per draw")
    valid = np.isfinite(values)
    count = int(valid.sum())
    if not np.isfinite(delta) or count < max(2, np.ceil(0.95 * len(values))):
        return np.nan, np.nan, np.nan, count
    pair = values[valid]
    lo, hi = np.quantile(pair, [0.025, 0.975])
    extremal = np.count_nonzero(np.abs(pair - delta) >= abs(delta) - 1e-15)
    return float(lo), float(hi), (extremal + 1) / (count + 1), count


def holm_adjust(pvalues):
    """Return step-down multiplicity-adjusted p-values in input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected finite p-values in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def bench_replicates(T, per_field, draws=1999, seed=20261119):
    """Return bootstrap means by draw and column of the judged-field input table."""
    fields = np.asarray(per_field, dtype=np.float64)
    if fields.ndim == 1:
        fields = fields[:, None]
    counts = np.bincount(T.group, minlength=T.G).astype(np.float64)
    sums = np.stack([np.bincount(T.group, weights=col, minlength=T.G) for col in fields.T], axis=1)
    out = np.empty((draws, fields.shape[1]))
    pos = 0
    for weights in group_multiplicities(T.group_fold, draws, seed=[seed, T.cid]):
        end = pos + len(weights)
        with np.errstate(invalid="ignore"):
            out[pos:end] = (weights @ sums) / (weights @ counts)[:, None]
        pos = end
    return out


def suite_contrast(delta_by_cid, reps_by_cid, cids=None):
    """Return the reference equal-benchmark paired contrast, counting lower values as better."""
    cids = sorted(delta_by_cid) if cids is None else list(cids)
    differences = np.asarray([delta_by_cid[c] for c in cids])
    delta = float(np.mean(differences))
    reps = np.mean(np.stack([reps_by_cid[c] for c in cids], axis=1), axis=1)
    lo, hi, p, n = interval_and_test(delta, reps)
    return dict(delta=delta, ci_low=lo, ci_high=hi, p=p, draws=n, n_benchmarks=len(cids),
                better=int((differences < -1e-12).sum()), same=int((np.abs(differences) <= 1e-12).sum()),
                worse=int((differences > 1e-12).sum()))


def paired(tasks, loss_new, loss_ref, cids=None, draws=1999, seed=20261119):
    """Return the suite contrast, benchmark differences and paired replicate differences."""
    cids = sorted(loss_new) if cids is None else list(cids)
    delta, reps = {}, {}
    for cid in cids:
        task = tasks(cid) if callable(tasks) else tasks[cid]
        sampled = bench_replicates(task, np.column_stack([loss_new[cid], loss_ref[cid]]), draws, seed)
        delta[cid] = float(np.mean(loss_new[cid]) - np.mean(loss_ref[cid]))
        reps[cid] = sampled[:, 0] - sampled[:, 1]
    return suite_contrast(delta, reps, cids), delta, reps

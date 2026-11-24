"""Exact paired resampling, interval, selection and contrast parity tests."""

import numpy as np
import pytest

from acl_resource.sim import stats
from acl_resource.sim.engine import load_tasks


def test_bootstrap_stream_and_families(reference):
    from jdi import paper_stats

    folds = np.asarray([0, 2, 4, 1, 2, 3, 0, 4, 1, 3, 1, 0])
    seed = [20261119, 9]
    wanted = np.concatenate(list(paper_stats.group_multiplicities(folds, 1999, seed=seed)))
    for chunk in (9, 48, 53):
        computed = np.concatenate(list(stats.group_multiplicities(folds, 1999, seed=seed, chunk_size=chunk)))
        assert np.array_equal(computed, wanted)
    for p in ([0.05, 0.005, 0.05, 1, 0.25], [0], []):
        assert np.array_equal(stats.holm_adjust(p), paper_stats.holm_adjust(p))
    for draws in (wanted[:, 0], np.array([np.nan, 1, 0]), np.full(2, np.nan)):
        assert np.array_equal(stats.interval_and_test(0.5, draws),
                              paper_stats.interval_and_test(0.5, draws), equal_nan=True)


@pytest.mark.parametrize("n_groups", [1, 4, None])
def test_select_size(reference, n_groups):
    from jdi import paper_stats

    rng = np.random.default_rng(31)
    scores = np.array([-0.20, -0.15, -0.11, -0.11])
    draws = scores + rng.normal(0, 0.10, (499, len(scores)))
    for bad in (0, 1, 100):
        draws[:bad, 2] = np.nan
        actual = stats.select_size(scores, draws, n_groups=n_groups)
        wanted = paper_stats.select_size(scores, draws, n_groups=n_groups)
        assert (actual.k, actual.best_k, actual.fallback, actual.reason) == (
            wanted.k, wanted.best_k, wanted.fallback, wanted.reason)
        assert np.array_equal(actual.paired_se, wanted.paired_se, equal_nan=True)


def test_paired_contrast(corpus, reference):
    from ensemble_round2 import contrasts

    cids = [2, 9, 38, 59]
    tasks = load_tasks(corpus, cids)
    targets = {cid: reference.Task(cid) for cid in cids}
    a, b = {}, {}
    for cid in cids:
        losses = reference.loss_of_mass(tasks[cid].mass[:, :2])
        actual = stats.bench_replicates(tasks[cid], losses)
        wanted = contrasts.bench_replicates(targets[cid], losses)
        assert np.array_equal(actual, wanted, equal_nan=True)
        a[cid], b[cid] = losses.T
    actual, delta, reps = stats.paired(tasks, a, b, cids)
    wanted, reference_delta, reference_reps = contrasts.paired(targets, a, b, cids)
    assert actual == wanted
    assert delta == reference_delta
    for cid in cids:
        assert np.array_equal(reps[cid], reference_reps[cid], equal_nan=True)

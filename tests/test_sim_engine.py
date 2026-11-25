"""Exact parity of the task tables and rotations on the edge-case benchmarks."""

import numpy as np
import pytest

from data.schema import N_FOLDS, roles
from sim.engine import load_task


@pytest.mark.parametrize("cid", [2, 9, 31, 36, 38, 50, 56, 59])
def test_task_parity(corpus, reference, cid):
    task = load_task(corpus, cid)
    expected = reference.Task(cid)
    for name in ("cid", "name", "area", "F", "R", "field_fold", "J", "nJ", "jpos", "fold",
                 "width", "n_acc", "row_group", "G", "group_fold", "field_group", "group",
                 "mass", "mass_menu"):
        assert np.array_equal(getattr(task, name), getattr(expected, name)), name
    for computed, stored in zip(task.cal_stats, expected.cal_stats):
        assert np.array_equal(computed, stored)

    rng = np.random.default_rng(cid)
    weights = rng.random((N_FOLDS, task.b.M))
    weights[weights < 0.5] = 0.0
    for n in (3, 4):
        choices = []
        for f in range(N_FOLDS):
            deployment, cross = task.recal(f, n)
            deployment_ref, cross_ref = expected.recal(f, n)
            assert np.array_equal(deployment, deployment_ref)
            assert cross.keys() == cross_ref.keys()
            for g in cross:
                assert np.array_equal(cross[g], cross_ref[g])
            assert np.array_equal(task.choice_by_fold(f, n), expected.choice_by_fold(f, n))
            choices.append(deployment)
            for regime in ("raw", "calibrated"):
                assert np.array_equal(task.view(f, regime, n), expected.view(f, regime, n))
                for folds in (None, roles(f)[2]):
                    args = (f, weights, regime, folds, n)
                    assert np.array_equal(task.fit_loss(*args), expected.fit_loss(*args))
            blocks = task.fold_blocks([f], judged_only=True)
            reference_blocks = expected.fold_blocks([f], judged_only=True)
            assert blocks.keys() == reference_blocks.keys()
            vectors = {}
            for w, sel in blocks.items():
                assert np.array_equal(sel, reference_blocks[w])
                for choice in (None, deployment):
                    args = (w, weights[f], choice, sel)
                    vectors[w] = task.pool_fields(*args)
                    assert np.array_equal(vectors[w], expected.pool_fields(*args))
            assert np.array_equal(task.mass_of_blocks(vectors, blocks),
                                  expected.mass_of_blocks(vectors, reference_blocks), equal_nan=True)
        for choice in (None, np.asarray(choices)):
            pooled = task.stitch(weights, choice)
            stored = expected.stitch(weights, choice)
            for w in pooled:
                assert np.array_equal(pooled[w], stored[w])
            for actual, wanted in zip(task.decode(pooled), expected.decode(stored)):
                assert np.array_equal(actual, wanted)
        for regime in ("raw", "calibrated"):
            assert np.array_equal(task.stitched_loss(weights, regime, n),
                                  expected.stitched_loss(weights, regime, n))

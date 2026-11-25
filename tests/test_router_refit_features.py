"""Text features fitted on the fit requests only: the transforms must not see the other requests."""

import numpy as np
import pytest

from analysis import router_refit_features as RF
from data.schema import N_FOLDS
from sim import metrics
from sim.baselines import knn_router
from sim.engine import load_task

from .test_restyle_analyses import CIDS
from .test_sim_runner import _settings

pytest.importorskip("sklearn")


def test_transforms_ignore_requests_outside_the_fit_rows():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 20))
    fit_rows = np.arange(0, 60, 2)
    other = np.setdiff1d(np.arange(60), fit_rows)
    changed = x.copy()
    changed[other] = rng.normal(size=(len(other), 20)) * 5
    a, b = RF.pca_refit(x, fit_rows, 8), RF.pca_refit(changed, fit_rows, 8)
    assert a.shape == (60, 8) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)
    assert np.array_equal(a[fit_rows], b[fit_rows]) and not np.array_equal(a[other], b[other])

    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    texts = [" ".join(rng.choice(words, size=6)) for _ in range(40)]
    fit_rows = np.arange(0, 40, 2)
    changed = list(texts)
    for i in np.setdiff1d(np.arange(40), fit_rows):
        changed[i] = "omega " + changed[i]
    a, b = RF.tfidf_refit(texts, fit_rows, 4), RF.tfidf_refit(changed, fit_rows, 4)
    assert a.shape[0] == 40 and np.array_equal(a[fit_rows], b[fit_rows])


def test_refit_router_with_fixed_features_is_the_router(corpus):
    """With features that do not depend on the rotation, the refit path gives the stored router's answers."""
    settings, _ = _settings_dict()
    task = load_task(corpus, CIDS[0])
    features = knn_router.request_features(task.b, "numeric")
    scale = lambda f, fit_rows: _standardised(features, fit_rows)                                # noqa: E731
    refitted = RF.refit_router(task, settings, scale)
    ours = metrics.score(task, refitted)
    theirs = metrics.score(task, knn_router.run(task, settings))
    assert len(refitted.info["k"]) == N_FOLDS
    assert np.array_equal(ours.hit, theirs.hit) and np.array_equal(ours.mass, theirs.mass)


def _standardised(features, fit_rows):
    scale = features[fit_rows].std(axis=0)
    return (features - features[fit_rows].mean(axis=0)) / np.where(scale == 0, 1.0, scale)


def _settings_dict():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(dir=".") as directory:
        return _settings(Path(directory))

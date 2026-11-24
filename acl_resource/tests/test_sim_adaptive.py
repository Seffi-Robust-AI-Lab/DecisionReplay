"""Rotation isolation, request-level call plans and the adaptive policy limits."""

from dataclasses import dataclass, replace
from importlib import util
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from acl_resource.data.load import list_benchmarks
from acl_resource.data.schema import N_FOLDS, roles
from acl_resource.sim import core
from acl_resource.sim.baselines import BASELINES, certified_exit, confidence_cascade, knn_router, stack
from acl_resource.sim.engine import Task, load_task
from acl_resource.sim.types import MENU, NOT_CALLED


@pytest.fixture(scope="module")
def settings():
    """Return the shared experiment settings."""
    return json.loads((Path(__file__).parents[1] / "SETTINGS.json").read_text())


def shuffle_labels(task, fold):
    """Return a benchmark whose labels have been permuted within width on a held-out fold."""
    b = task.b
    source = np.arange(b.F)
    acc = {}
    rng = np.random.default_rng([task.cid, fold])
    for w, ids in b.idx.items():
        selected = np.flatnonzero(task.field_fold[ids] == fold)
        permuted = rng.permutation(selected)
        source[ids[selected]] = ids[permuted]
        acc[w] = b.acc[w].copy()
        acc[w][selected] = b.acc[w][permuted]
    assert any(not np.array_equal(acc[w], b.acc[w]) for w in acc)
    return replace(b, acc=acc, judged=b.judged[source], gold=b.gold[source], n_accepted=b.n_accepted[source])


def mask_uncalled(task, outcome, fold):
    """Return a benchmark with every uncalled vector and top probability on the fold masked."""
    b = task.b
    top = b.top.copy()
    vectors = {}
    for w, ids in b.idx.items():
        rows = b.field_row[ids]
        mask = (outcome.stage[rows] == NOT_CALLED) & (b.fold[rows, None] == fold)
        vectors[w] = b.P[w].copy()
        vectors[w][mask] = np.nan
        top[ids] = np.where(mask, np.nan, top[ids])
    assert any(np.isnan(block).any() for block in vectors.values())
    return replace(b, P=vectors, top=top)


def check_fold(task, fold, expected, actual):
    """Check all stage indices and pooled probabilities on the held-out requests."""
    requests = task.b.fold == fold
    assert np.array_equal(expected.stage[requests], actual.stage[requests])
    for w, ids in task.b.idx.items():
        keep = requests[task.b.field_row[ids]]
        assert np.isfinite(actual.pooled[w][keep]).all()
        assert np.array_equal(expected.pooled[w][keep], actual.pooled[w][keep])


@pytest.mark.parametrize("cid", [31, 56])
def test_label_permutation(corpus, settings, cid):
    task = load_task(corpus, cid)
    original = {name: module.run(task, settings) for name, module in BASELINES.items() if name != "oracle"}
    for fold in range(N_FOLDS):
        changed = Task(shuffle_labels(task, fold))
        for name, expected in original.items():
            check_fold(task, fold, expected, BASELINES[name].run(changed, settings))


@pytest.mark.parametrize("name", ["best_single", "knn_router", "confidence_cascade", "certified_exit"])
def test_uncalled_outputs(corpus, settings, name):
    task = load_task(corpus, 31)
    original = BASELINES[name].run(task, settings)
    for fold in range(N_FOLDS):
        masked = Task(mask_uncalled(task, original, fold))
        check_fold(task, fold, original, BASELINES[name].run(masked, settings))


def test_oracle_and_certified_winners(corpus, settings):
    for benchmark in list_benchmarks(corpus):
        task = load_task(corpus, benchmark["catalog_id"])
        best = BASELINES["oracle"].run(task, settings)
        oracle_hits = core.decode(task.b, best.pooled)[0].sum()
        assert ((best.stage == 0).sum(axis=1) == 1).all()
        for name, module in BASELINES.items():
            if name == "oracle":
                continue
            outcome = module.run(task, settings)
            hit, _, winner, _ = core.decode(task.b, outcome.pooled)
            assert hit.sum() <= oracle_hits, (task.cid, name)
            if name == "certified_exit":
                panel = stack.run(task, settings)
                assert np.array_equal(winner, core.decode(task.b, panel.pooled)[2])
                assert np.all((outcome.stage == NOT_CALLED) | (panel.stage == 0))
                assert np.all((outcome.stage >= 0).sum(axis=1) <= (panel.stage == 0).sum(axis=1))


def test_certified_reference_depth(corpus, settings, reference, test_root):
    # The standalone excerpt omits two imports that are present in its original module.
    path = test_root / "reference_excerpts" / "certified_depths.py"
    spec = util.spec_from_file_location("certification_reference", path)
    excerpt = util.module_from_spec(spec)
    excerpt.math, excerpt.dataclass = math, dataclass
    spec.loader.exec_module(excerpt)
    for cid in (1, 2, 9, 50, 56):
        task = load_task(corpus, cid)
        out = certified_exit.run(task, settings)
        b = task.b
        depths = np.zeros(b.R, int)
        for f in range(N_FOLDS):
            order = np.array(out.info["order"][f])
            weights = np.array(out.info["weights"][f])[order]
            choice = task.recal(f)[0]
            for w, ids in b.idx.items():
                for binary in (False, True):
                    pos = np.flatnonzero((task.field_fold[ids] == f) & (b.is_noul[ids] == binary))
                    if not len(pos):
                        continue
                    vectors = np.stack([core.transform(b.P[w][pos, m], *MENU[choice[m]]) for m in order], axis=1)
                    depth, _ = excerpt.certified_depths(vectors, weights, binary)
                    np.maximum.at(depths, b.field_row[ids[pos]], depth)
        assert np.array_equal(depths, (out.stage >= 0).sum(axis=1))


def test_router_limits(corpus, settings):
    task = load_task(corpus, 9)
    b = task.b
    config = {**settings, "knn_router": {**settings["knn_router"], "k_grid": [b.R]}}
    free = knn_router.run(task, config, penalty=0)
    costly = knn_router.run(task, config, penalty=1e12)
    for f in range(N_FOLDS):
        fit = roles(f)[0]
        fit_mask = np.isin(b.fold, fit)
        mass = task.view(f, "calibrated")
        judged_requests = b.field_row[task.J]
        utilities = []
        for r in np.flatnonzero(fit_mask):
            judged = judged_requests == r
            if judged.any():
                utilities.append(-core.loss_of_mass(mass[judged]).mean(axis=0))
        model = np.asarray(utilities).mean(axis=0).argmax()
        cheapest = b.cost[fit_mask].mean(axis=0).argmin()
        for output, m in ((free, model), (costly, cheapest)):
            assert np.all(output.stage[b.fold == f, m] == 0)
            assert np.all((output.stage[b.fold == f] >= 0).sum(axis=1) == 1)


def test_router_distance_ties():
    pool = np.array([[0.], [1.], [-1.], [1.], [-1.]])
    utility = np.eye(len(pool))
    scores = knn_router.neighbour_scores(pool, utility, np.array([[0.]]), [2, 3, 4, 5])
    for k, row in scores.items():
        assert np.array_equal(row[0], utility[:k].mean(axis=0))


def test_cascade_limits_and_cost(corpus, settings):
    task = load_task(corpus, 9)
    b = task.b
    assert any(len(chain) > 1 for chain in confidence_cascade.run(task, settings).info["chain"])
    for tau in (0, max(settings["confidence_cascade"]["tau_grid"])):
        out = confidence_cascade.run(task, settings, threshold=tau)
        for f, chain in enumerate(out.info["chain"]):
            called = chain[:1] if tau == 0 else chain
            expected = np.full(b.M, NOT_CALLED)
            expected[called] = np.arange(len(called))
            assert np.all(out.stage[b.fold == f] == expected)
            m = called[-1]
            for w, ids in b.idx.items():
                pos = np.flatnonzero(task.field_fold[ids] == f)
                vectors = core.transform(b.P[w][pos, m], *MENU[task.recal(f)[0][m]])
                assert np.array_equal(out.pooled[w][pos], vectors)
        cost = core.plan_costs(b, out.stage)
        assert cost["cost_usd"] == math.fsum(b.cost[out.stage >= 0])
        latency = np.zeros(b.R)
        for r in range(b.R):
            for depth in range(out.stage[r].max() + 1):
                latency[r] += b.lat[r, out.stage[r] == depth].max()
        assert cost["latency_mean_s"] == latency.mean()
        assert cost["latency_p50_s"] == np.percentile(latency, 50)
        assert cost["latency_p95_s"] == np.percentile(latency, 95)


def test_router_local_feature_folder(corpus, settings):
    task = load_task(corpus, 31)
    b = task.b
    with TemporaryDirectory(dir=".") as folder:
        # With identical coordinates all requests tie at distance zero.
        features = np.zeros((b.R, 2), np.float32)
        np.save(Path(folder) / f"catalog-{task.cid:03d}.npy", features)
        config = {**settings, "knn_router": {**settings["knn_router"], "features": folder, "k_grid": [1]}}
        out = knn_router.run(task, config, penalty=0)
    judged_requests = b.field_row[task.J]
    for f in range(N_FOLDS):
        rows = task.rows(roles(f)[0])
        first = judged_requests[rows].min()
        mass = task.view(f, "calibrated")[judged_requests == first]
        model = core.loss_of_mass(mass).mean(axis=0).argmin()
        assert np.all(out.stage[b.fold == f, model] == 0)


def test_cascade_validation_selection(corpus, settings):
    task = load_task(corpus, 31)
    b = task.b
    out = confidence_cascade.run(task, settings)
    single = BASELINES["best_single"].run(task, settings)
    grid = settings["confidence_cascade"]["tau_grid"]
    for f in range(N_FOLDS):
        _, val, _ = roles(f)
        valid = np.flatnonzero(b.fold == val)
        judged = task.rows([val])
        rows = b.field_row[task.J[judged]]
        budget = math.fsum(b.cost[valid, single.info["model"][f]])
        chain = out.info["chain"][f]
        confidence = {}
        for m in chain:
            top = np.empty(task.F)
            for w, ids in b.idx.items():
                vectors = core.transform(b.P[w][:, m], *MENU[task.recal(f)[0][m]])
                top[ids] = vectors.max(axis=1)
            confidence[m] = [min(top[b.field_row == r]) for r in valid]
        points = []
        for i, tau in enumerate(grid):
            models = np.empty(task.R, int)
            cost = []
            for j, r in enumerate(valid):
                for m in chain:
                    cost.append(b.cost[r, m])
                    models[r] = m
                    if confidence[m][j] >= tau:
                        break
            loss = core.loss_of_mass(task.view(f, "calibrated")[judged, models[rows]])
            if math.fsum(cost) <= budget:
                points.append((math.fsum(loss) / len(loss), i))
        expected = grid[min(points)[1]] if points else max(grid)
        assert out.info["tau"][f] == expected

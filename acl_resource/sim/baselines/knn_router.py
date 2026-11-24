"""Route requests by the calibrated utility of models on nearby fit requests."""

import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ...data.schema import N_FOLDS, roles
from ..core import loss_of_mass
from ..types import Outcome, empty_ledger
from . import best_single
from .random_single import request_vectors, single_stage

NAME = "knn_router"


def request_features(b, source):
    """Return numeric request features or the supplied local feature matrix."""
    if source != "numeric":
        x = np.load(Path(source) / f"catalog-{b.cid:03d}.npy", allow_pickle=False)
        if x.ndim != 2 or len(x) != b.R or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("request features must be a finite float32 matrix with one row per request")
        return x
    n = np.bincount(b.field_row, minlength=b.R)
    largest = np.zeros(b.R)
    np.maximum.at(largest, b.field_row, b.width)
    return np.column_stack((
        np.log1p(np.median(b.tin, axis=1)), n,
        np.bincount(b.field_row, b.width, minlength=b.R) / n, largest,
        np.bincount(b.field_row, b.is_noul, minlength=b.R) / n,
    ))


def column_means(values):
    """Return exactly summed means over the rows of a nonempty matrix."""
    return np.fromiter((math.fsum(col) / len(values) for col in values.T), np.float64)


def request_utilities(task, fold, fit):
    """Return fit request indices with judged fields and their mean calibrated log scores."""
    judged = task.rows(fit)
    rows = task.b.field_row[task.J[judged]]
    pool, starts, counts = np.unique(rows, return_index=True, return_counts=True)
    if not len(pool):
        raise ValueError("routing requires fit requests with judged fields")
    utility = -loss_of_mass(task.view(fold, "calibrated")[judged])
    means = utility[starts].copy()
    for i in np.flatnonzero(counts > 1):
        means[i] = column_means(utility[starts[i]:starts[i] + counts[i]])
    return pool, means


def neighbour_scores(pool_x, utility, query_x, k_grid):
    """Return model scores at each k, breaking distance ties by ascending fit request index."""
    n = len(pool_x)
    if not k_grid or min(k_grid) < 1:
        raise ValueError("k must be positive")
    sizes = sorted({min(k, n) for k in k_grid})
    scores = {}
    if sizes[-1] == n:
        sizes.pop()
        scores[n] = np.broadcast_to(column_means(utility), (len(query_x), utility.shape[1]))
    if sizes:
        tree = cKDTree(pool_x)
        # One extra neighbour detects an equal-distance boundary cut by the tree query.
        distances, nearest = tree.query(query_x, k=sizes[-1] + 1, workers=1)
        for k in sizes:
            scores[k] = np.empty((len(query_x), utility.shape[1]))
        for i, (dist, members) in enumerate(zip(distances, nearest)):
            if dist[-1] == dist[-2]:
                radius = np.nextafter(dist[-1], np.inf)
                members = np.asarray(tree.query_ball_point(query_x[i], radius))
                delta = pool_x[members] - query_x[i]
                dist = np.einsum("ij,ij->i", delta, delta)
            members = members[np.lexsort((members, dist))]
            for k in sizes:
                scores[k][i] = column_means(utility[members[:k]])
    return {k: scores[min(k, n)] for k in k_grid}


def validation_loss(task, fold, models):
    """Return mean calibrated judged log loss for validation request assignments."""
    _, val, _ = roles(fold)
    rows = task.rows([val])
    if not len(rows):
        raise ValueError("parameter selection requires judged validation fields")
    members = models[task.b.field_row[task.J[rows]]]
    loss = loss_of_mass(task.view(fold, "calibrated")[rows, members])
    return math.fsum(loss) / len(loss)


def validation_budget(task, fold, requests):
    """Return the validation cost of the rotation's four-fold best single model."""
    best = best_single.ranking(task, fold, n=4)[0]
    return math.fsum(task.b.cost[requests, best])


def choose_parameter(grid, losses, costs, budget):
    """Return the lowest-loss grid entry within the budget, or the cheapest entry if none is within it."""
    feasible = [i for i, cost in enumerate(costs) if cost <= budget]
    if not feasible:
        return grid[min(range(len(grid)), key=lambda i: (costs[i], i))]
    return grid[min(feasible, key=lambda i: (losses[i], i))]


def selection_record(grid, losses, costs, budget):
    """Return what the validation fold showed for every grid entry, for reports on the selection rule."""
    return dict(grid=[float(g) for g in grid], losses=[float(x) for x in losses], costs=[float(c) for c in costs],
                budget=float(budget), within_budget=int(sum(c <= budget for c in costs)))


def per_fold(value, selected, fold):
    """Return the parameter of a rotation: the selected one, a fixed one, or one of a list with one per fold."""
    if value is None:
        return selected
    if isinstance(value, (list, tuple)):
        if len(value) != N_FOLDS:
            raise ValueError("a parameter list needs one value per fold")
        return value[fold]
    return value


def fit_rotation(task, fold, features, source, k_grid, penalties):
    """Return selected k, selected penalty, test neighbour scores, relative fit costs and the selection record."""
    b = task.b
    fit, val, _ = roles(fold)
    fit_rows = np.flatnonzero(np.isin(b.fold, fit))
    valid = np.flatnonzero(b.fold == val)
    test = np.flatnonzero(b.fold == fold)
    if source == "numeric":
        scale = features[fit_rows].std(axis=0)
        features = (features - features[fit_rows].mean(axis=0)) / np.where(scale == 0, 1., scale)
    pool, utility = request_utilities(task, fold, fit)
    queries = np.concatenate((valid, test))
    scores = neighbour_scores(features[pool].astype(float), utility, features[queries].astype(float), k_grid)
    costs = column_means(b.cost[fit_rows])
    relative = costs / costs.max() if costs.max() else costs
    models = np.empty(task.R, np.int32)
    losses = []
    for k in k_grid:
        models[valid] = scores[k][:len(valid)].argmax(axis=1)
        losses.append(validation_loss(task, fold, models))
    k = k_grid[int(np.argmin(losses))]
    losses, costs = [], []
    for penalty in penalties:
        models[valid] = (scores[k][:len(valid)] - penalty * relative).argmax(axis=1)
        losses.append(validation_loss(task, fold, models))
        costs.append(math.fsum(b.cost[valid, models[valid]]))
    budget = validation_budget(task, fold, valid)
    selected = choose_parameter(penalties, losses, costs, budget)
    return k, selected, scores[k][len(valid):].copy(), relative, selection_record(penalties, losses, costs, budget)


def run(task, settings, *, penalty=None):
    """Return calibrated answers routed at the validation-selected penalty, a fixed one, or one per fold."""
    config = settings[NAME]
    source = config["features"]
    k_grid, penalties = tuple(config["k_grid"]), tuple(config["lambda_grid"])
    cache = vars(task).setdefault("_router_fits", {})
    key = (source, k_grid, penalties)
    if key not in cache:
        features = request_features(task.b, source)
        cache[key] = [fit_rotation(task, f, features, source, k_grid, penalties) for f in range(N_FOLDS)]
    models = np.empty(task.R, np.int32)
    info = {"k": [], "lambda": [], "selection": []}
    for f, (k, selected, scores, relative, record) in enumerate(cache[key]):
        value = per_fold(penalty, selected, f)
        models[task.b.fold == f] = (scores - value * relative).argmax(axis=1)
        info["k"].append(k)
        info["lambda"].append(value)
        info["selection"].append(record)
    choices = np.array([task.recal(f)[0] for f in range(N_FOLDS)])
    ledger = empty_ledger()
    ledger.update(labeled_folds="fit and validation", models_profiled=task.b.M)
    return Outcome(request_vectors(task, models, choices), single_stage(task, models), info, ledger)

"""Routing when the benchmark of a request is not known, and a second learned router.

    python3 -B -m analysis.router_unknown --corpus DIR --settings FILE --results RESULTS \
        --features DIR --out OUT [--project-root .] [--processes N]

``RESULTS`` is read (``rows.json`` and ``local/fields-<method>.npz``); ``--features`` is one folder of request
features (``features.json`` and one float32 matrix per benchmark, not reduced per benchmark, so that distances
across benchmarks mean something; its name, ``minilm`` on the corpus, is written ``<features>`` below).  ``OUT``
receives ``router_unknown.json`` (aggregates only) and ``local/fields-<policy>[.<rule>].npz`` (per-field arrays that
stay on the machine).  No model is called and the request text never enters this module.

Why.  Every router of the paper is fitted inside one benchmark: its recalibration, its neighbours, its k, its
penalty and its budget are the benchmark's own, so it knows which benchmark a request belongs to, and the only
router is a k-nearest-neighbour rule.  A review asked for the setting in which the benchmark is unknown and for a
second router.  This module runs both.  Added after the first run, on 23 November 2026.  The design below was
written into this docstring before the module was first run on the corpus; what changed afterwards is listed at
the end.

Design
------

Pool and folds.  The pool is the headline suite of the settings; a benchmark outside it gets no result.  Rotation
f takes fold f of every benchmark as test fold and ``schema.roles(f)`` gives the three fit folds, the validation
fold and the four training folds, as everywhere.  "Pooled" always means the equal-weight mean over the benchmarks
of the pool of a per-benchmark mean, summed exactly: the pooled log loss is the mean of the benchmarks' mean judged
log losses, the pooled cost the mean of the benchmarks' mean costs per request.

The paper's router uses the benchmark in five places.  Each gets a counterpart that is the same for all
benchmarks:

1. Recalibration.  One menu transform per model and rotation: the entry with the lowest pooled log loss on the
   folds at hand (from every task's ``cal_stats``), the first minimum within the tolerance of
   ``engine.select_recalibration``.  As in ``Task.recal``: a deployment transform from the three fit folds (from
   the four training folds for the single-model reference) and, for the folds it was fitted on, leave-one-fold-out
   transforms.  ``GlobalTask`` is a task whose ``recal`` returns this table, so ``Task.view`` and ``Task.fit_loss``
   and the router's ``request_utilities`` are used unchanged.  A benchmark without a judged field in the folds at
   hand is left out of that mean.
2. Returned vectors.  ``request_vectors(task, models, choices)`` with ``choices[f]`` the deployment transform of
   rotation f from (1).
3. Budget.  One per rotation, on the pooled validation cost per request, under three rules: ``uncapped`` (the
   penalty with the lowest pooled validation loss), ``capped_single`` (cost at most that of the single-model
   reference ``single_global`` below) and ``capped_bd`` (cost at most the pooled validation cost per request of the
   best single model of each benchmark, ``knn_router.validation_budget``), the last so that the result can be read
   beside the paper's router.  The choice follows ``knn_router.choose_parameter``.
4. Selection.  One k (or one alpha) and one penalty per rotation, on the pooled validation loss: first the value
   with the lowest loss at zero penalty (first minimum), then the penalty by the rule, as
   ``knn_router.fit_rotation`` does.  Grids: ``knn_router.k_grid`` and ``lambda_grid`` of the settings.  A
   benchmark's validation loss is the exactly rounded mean over its judged validation fields, as
   ``knn_router.validation_loss``.
5. Features and cost penalty.  The numeric request features are standardised with the pooled mean and standard
   deviation of the fit requests; the relative cost of a model is its pooled mean cost on the fit requests divided
   by the largest such mean.

Policies.  Each answers a request with the vectors of one model and is scored through the standard path,
``Outcome(request_vectors(...), single_stage(...))`` and ``metrics.score``.

``single_global``            per rotation the one model with the lowest pooled calibrated log loss on the four
                             training folds (recalibration (1) on four folds, cross-fitted), with its deployment
                             transform.  The reference when the benchmark is unknown.
``knn_across_<features>``,   the neighbours of a request are fit requests of all benchmarks of the pool (those with
``knn_across_numeric``       a judged field); the utility of a model on a fit request is its mean log score over
                             the request's judged fields under (1); the score of a model is the exactly summed
                             mean utility over the k nearest fit requests minus the penalty times its relative cost;
                             the request goes to the model with the highest score (ties: lower index).
``ridge_across_<features>``  ridge regression from the feature vector to the utilities of all models, fitted on
                             the same fit requests, each request one row: the least-squares loss summed over
                             requests and models plus alpha times the squared coefficients, with an intercept that
                             is not penalised (centred normal equations, solved in double precision by the linear
                             algebra library).  Alpha from (0.1, 1, 10, 100, 1000), chosen as k is, then the
                             penalty.  Its prediction takes the place of the neighbours' mean utility.
``ridge_within_<features>``  the same regression inside each benchmark, with the benchmark's own recalibration,
                             selection and budget, under the rules ``capped`` and ``uncapped`` of
                             ``operating_points``: a second, learned router that knows the benchmark, to stand
                             beside the paper's.
``knn_within_<features>``    (an addition to the brief) the paper's router with these features, replayed by this
                             module under ``capped`` and ``uncapped``; its capped rule is what the first check
                             below compares with the stored rows, its uncapped rule gives ``ridge_within`` a
                             like-for-like comparison.

One code path.  A pool of one benchmark makes every pooled quantity the benchmark's own: (1) is ``Task.recal``, the
pooled validation loss and cost are the benchmark's, ``capped_bd`` is the cap of the main table and ``single_global``
is the best single model.  The policies "within" are therefore ``fit_pool`` on one benchmark, and the checks below
test the code that produces the results across benchmarks.

The search.  Nearest fit requests are found exactly and without a tree, in double precision: the pool is reduced
to its distinct feature rows; a matrix product, in blocks of queries, screens the rows that can be among the k
nearest with a bound on its rounding error (the run stops if the bound fails on a screened row); their squared
distances are then computed from the differences, summed in four interleaved running sums as the tree search of
the stored router sums them; fit requests are ordered by distance and then by their position in the pool, which
is ascending by catalogue id and then by request index.  Many requests of some benchmarks have the same feature
vector, so the position decides often; the output says for how many requests the selected neighbourhood was cut
inside such a tie.

Checks that stop the run.  (a) ``knn_within_<features>.capped`` must reproduce the stored ``knn_router_text`` rows
of ``rows.json`` (log loss and accuracy, 1e-12) on every benchmark of the pool; a deviation is tolerated, counted
and reported only on a benchmark where the search met, at the edge of a neighbourhood, two different fit requests
whose distances agree to 1e-12 (a tie in floating point that the tree may break the other way); the per-field
arrays are also compared with the archive.  (b) ``single_global`` on a pool of one benchmark must reproduce the
stored ``best_single`` rows.  (c) The recalibration (1) on a pool of one benchmark must be ``Task.recal``.  (d) The
archives of the two stored methods must reproduce their rows.  (e) When ``RESULTS/router_study.json`` is there with
the same settings, ``knn_within_<features>.uncapped`` is compared with its uncapped rows in the same way as (a).
A test on the made-up corpus permutes the labels of a test fold and requires the same recalibration, parameters
and routed models for that rotation.

Output, ``OUT/router_unknown.json``.  ``rows``: benchmark and suite rows in the shared row schema (method =
policy, variant = rule; the own metric of the benchmarks in ``native`` when ``--project-root`` is given), for the
suites of the settings whose benchmarks all lie in the pool (the fits always use the whole pool).  ``suites``: per
suite the points by policy and rule next to the stored ``best_single``, ``knn_router``, ``knn_router_text`` and, when
the files are in ``RESULTS``, the control and the uncapped text router; the contrasts; the neighbours; the shares of
requests sent to each model; the choices of the policies within benchmarks.  ``selection``, ``reference``,
``recalibration``: what each rotation chose (losses are pooled validation log losses, costs and budgets pooled
validation costs in USD per request).  ``reproduction``: the outcome of the checks.

Contrasts, on log loss and on accuracy, each with two 95% intervals: ``within_low``/``within_high`` from
``stats.paired`` (source groups resampled within benchmark and fold, the draws and seed of the settings) and
``over_low``/``over_high`` from ``paper_facts._over_benchmarks`` (benchmarks resampled, the same draws and seed).
Every policy across benchmarks against ``single_global``, against the stored ``best_single`` and against the stored
``knn_router_text``; ``single_global`` against the two stored methods; ``ridge_within`` under both rules against the
two stored methods.  They are descriptive and not adjusted.

Neighbours.  For the two neighbour policies: the share of the selected-k neighbours of a test request that come
from the request's own benchmark (mean over the requests of a benchmark, then the equal-benchmark mean), the same
share if neighbours were drawn at random from the pool (an addition, to read the first against), and the requests
whose selected neighbourhood is cut inside a tie.

Where the brief left a choice.  The pooled standard deviation of (5) is that of the equal-weight mixture of the
benchmarks: the square root of the pooled mean squared deviation from the pooled mean.  The ridge regression
weighs every fit request equally (the neighbour rule also treats every fit request as one point), not every
benchmark.  The global position of a fit request is by ascending catalogue id, then request index.  Suites that
hold a benchmark outside the pool are not reported.

Additions to the brief.  ``knn_within_<features>``; the contrast of ``ridge_within.uncapped`` with
``knn_within.uncapped``; checks (b) to (e); the chance level of the neighbour share and the tie counts; the share
of benchmark-model cells whose transform under (1) differs from the benchmark's own.

Changes after the first run on the corpus: none.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.linalg import solve

from data.schema import N_FOLDS, roles, sha256_file
from sim import core, metrics, native, stats
from sim.baselines import knn_router
from sim.baselines.random_single import request_vectors, single_stage
from sim.engine import Task, load_task
from sim.types import MENU, Outcome, empty_ledger
from . import local_methods as LM
from . import run_experiments as RX
from .operating_points import POINT_KEYS, lowest_loss
from .paper_facts import _over_benchmarks

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
ACROSS_RULES = ("uncapped", "capped_single", "capped_bd")
WITHIN_RULES = {"capped": "capped_bd", "uncapped": "uncapped"}     # rule of operating_points -> rule on one benchmark
REFERENCE = "single_global"
BEST, TEXT = "best_single", LM.TEXT_ROUTER
TOLERANCE = 1e-12
NEAR = 1e-12                    # two distances closer than this, relatively, are a tie in floating point
EPS = float(np.finfo(np.float64).eps) / 2
SCREEN = 1 << 27                # bytes of one block of the screening product
PAIRS = 1 << 14                 # screened rows whose differences are held at once
MEANS = 1 << 10                 # queries whose neighbour utilities are summed at once


def label(policy, rule):
    """Name of a policy under a rule, as in the contrasts and the field archives."""
    return policy if rule is None else f"{policy}.{rule}"


def pooled(values):
    """Equal-weight mean over benchmarks, summed exactly: ``values`` holds one number or array per benchmark."""
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in values])
    flat = stacked.reshape(len(stacked), -1)
    total = np.fromiter((math.fsum(column) for column in flat.T), np.float64, count=flat.shape[1])
    return (total / len(stacked)).reshape(stacked.shape[1:])


# --------------------------------------------------------------------------------------------------
# the exact search
# --------------------------------------------------------------------------------------------------

def squared_distance(delta):
    """Row sums of squares of ``delta`` [n, d] in a fixed order: four interleaved running sums over the leading
    multiple of four coordinates, added left to right, then the remaining coordinates one by one."""
    square = delta * delta
    width = square.shape[1]
    body = width - width % 4
    if body:
        lanes = square[:, :4].copy()
        for start in range(4, body, 4):
            lanes += square[:, start:start + 4]
        total = ((lanes[:, 0] + lanes[:, 1]) + lanes[:, 2]) + lanes[:, 3]
    else:
        total = np.zeros(len(square))
    for column in range(body, width):
        total = total + square[:, column]
    return total


def nearest(pool_x, query_x, k):
    """The k nearest pool rows of every query row, nearest first; equal distances go to the lower pool position.

    Returns ``members`` [queries, k] (pool positions), ``distance`` [queries, k] (Euclidean, double precision) and
    ``group`` [queries, k] (the distinct pool row a member is a copy of).  Exact: see "The search" above.
    """
    pool_x, query_x = np.ascontiguousarray(pool_x), np.asarray(query_x)
    n, width = pool_x.shape
    if query_x.ndim != 2 or query_x.shape[1] != width or not 1 <= k <= n:
        raise ValueError("the search needs queries of the pool's width and 1 <= k <= pool size")
    rows, inverse, counts = np.unique(pool_x, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse.reshape(-1), kind="stable")          # pool positions by distinct row, ascending inside
    starts = np.cumsum(counts) - counts
    taken = np.minimum(counts, k)                                   # no more than k copies of a row can be needed
    rows = rows.astype(np.float64)
    distinct = len(rows)
    reach = min(k, distinct)
    gamma = width * EPS / (1 - width * EPS)                         # rounding of a sum of ``width`` products
    norm = np.einsum("ij,ij->i", rows, rows)
    largest = math.sqrt(float(norm.max()))
    step = max(1, SCREEN // (distinct * rows.itemsize))
    members = np.empty((len(query_x), k), np.int64)
    distance = np.empty((len(query_x), k))
    group = np.empty((len(query_x), k), np.int64)
    for first in range(0, len(query_x), step):
        wide = np.asarray(query_x[first:first + step], dtype=np.float64)
        length = np.sqrt(np.einsum("ij,ij->i", wide, wide))
        screen = wide @ rows.T                                      # |x|^2 - 2 q.x: the squared distance less |q|^2
        screen *= -2
        screen += norm
        bound = 16 * gamma * (length + largest) ** 2                # rounding of the product and of the distances
        edge = np.partition(screen, reach - 1, axis=1)[:, reach - 1] + 4 * bound
        q, r = np.nonzero(screen <= edge[:, None])
        exact = np.empty(len(q))
        for start in range(0, len(q), PAIRS):
            part = slice(start, start + PAIRS)
            exact[part] = squared_distance(rows[r[part]] - wide[q[part]])
        if np.any(np.abs(screen[q, r] + length[q] ** 2 - exact) > bound[q]):
            raise ArithmeticError("the rounding bound of the screening product does not hold")
        size = taken[r]
        pair = np.repeat(np.arange(len(q)), size)
        inside = np.arange(len(pair)) - np.repeat(np.cumsum(size) - size, size)
        member = order[starts[r][pair] + inside]
        far = np.sqrt(exact)[pair]
        owner = q[pair]
        rank = np.lexsort((member, far, owner))                     # by query, then distance, then pool position
        owner = owner[rank]
        place = np.arange(len(rank)) - np.searchsorted(owner, np.arange(len(wide)))[owner]
        keep = rank[place < k]
        if len(keep) != len(wide) * k:
            raise ArithmeticError("the search lost a neighbour")
        block = slice(first, first + len(wide))
        members[block] = member[keep].reshape(len(wide), k)
        distance[block] = far[keep].reshape(len(wide), k)
        group[block] = r[pair][keep].reshape(len(wide), k)
    return SimpleNamespace(members=members, distance=distance, group=group)


def neighbour_means(utility, members):
    """Exactly rounded mean utility of the listed fit requests, one row per query (as ``knn_router.column_means``)."""
    count, k = members.shape
    width = utility.shape[1]
    out = np.empty((count, width))
    for start in range(0, count, MEANS):
        block = utility[members[start:start + MEANS]].transpose(0, 2, 1).reshape(-1, k)
        sums = np.fromiter(map(math.fsum, block.tolist()), np.float64, count=len(block))
        out[start:start + MEANS] = sums.reshape(-1, width) / k
    return out


def knn_scores(pool_x, utility, query_x, k_grid):
    """Model scores at each k as ``knn_router.neighbour_scores`` defines them, with the exact search, and the search."""
    n = len(pool_x)
    if not k_grid or min(k_grid) < 1:
        raise ValueError("k must be positive")
    sizes = sorted({min(k, n) for k in k_grid})
    scores, found = {}, None
    if sizes[-1] == n:
        sizes.pop()
        scores[n] = np.broadcast_to(knn_router.column_means(utility), (len(query_x), utility.shape[1]))
    if sizes:
        found = nearest(pool_x, query_x, sizes[-1] + 1)             # one more: the pair at the edge of every k
        for k in sizes:
            scores[k] = neighbour_means(utility, found.members[:, :k])
    return {k: scores[min(k, n)] for k in k_grid}, found


def ridge_scores(pool_x, utility, query_x, alphas):
    """Predicted utilities at each alpha: ridge regression with an intercept that is not penalised."""
    x = np.array(pool_x, dtype=np.float64)
    centre = x.mean(axis=0)
    x -= centre
    level = utility.mean(axis=0)
    gram = x.T @ x
    cross = x.T @ (utility - level)
    queries = np.asarray(query_x, dtype=np.float64) - centre
    scores = {}
    for alpha in alphas:
        if not alpha > 0:
            raise ValueError("alpha must be positive")
        coefficients = solve(gram + alpha * np.eye(len(gram)), cross, assume_a="pos")
        scores[alpha] = queries @ coefficients + level
    return scores, None


def search_statistics(found, grid, value, pool_bench, query_bench, validations, count):
    """Per benchmark, over its test requests: summed share of own-benchmark neighbours at the selected k, the same
    for random neighbours, requests whose neighbourhood is cut inside a tie; and, over all queries and every k of
    the grid, pairs of different fit requests at the edge of a neighbourhood whose distances tie in floating point."""
    n = len(pool_bench)
    chance = np.bincount(pool_bench, minlength=count) / n
    test = query_bench[validations:]
    size = min(value, n)
    if size == n:
        same, cut = chance[test], np.zeros(len(test))
    else:
        same = (pool_bench[found.members[validations:, :size]] == test[:, None]).mean(axis=1)
        cut = (found.distance[validations:, size - 1] == found.distance[validations:, size]).astype(np.float64)
    ties = np.zeros(count, np.int64)
    for s in sorted({min(k, n) for k in grid} - {n}):
        a, b = found.distance[:, s - 1], found.distance[:, s]
        event = (found.group[:, s - 1] != found.group[:, s]) & (b - a <= NEAR * b)
        ties += np.bincount(query_bench[event], minlength=count)
    return (np.bincount(test, weights=same, minlength=count), np.bincount(test, weights=chance[test], minlength=count),
            np.bincount(test, weights=cut, minlength=count).astype(np.int64), ties)


# --------------------------------------------------------------------------------------------------
# recalibration for all benchmarks
# --------------------------------------------------------------------------------------------------

def global_choices(statistics):
    """{(fold, n): (deployment transform [M], {fold: leave-one-fold-out transform [M]})} for n = 3 fit folds and
    n = 4 training folds, from the ``cal_stats`` of every benchmark of the pool."""
    def choice(folds):
        folds = sorted(set(map(int, folds)))
        tables = []
        for sums, counts in statistics:
            n = int(sum(counts[g] for g in folds))
            if n:
                tables.append(np.sum([sums[g] for g in folds], axis=0) / n)
        if not tables:
            return np.zeros(statistics[0][0].shape[-1], dtype=np.int64)
        losses = pooled(tables)
        minima = losses.min(axis=0)
        tol = 32 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(minima))
        return (losses <= minima + tol).argmax(axis=0)

    table = {}
    for f in range(N_FOLDS):
        fit, _, train = roles(f)
        for n, used in ((3, fit), (4, train)):
            table[f, n] = choice(used), {g: choice(x for x in used if x != g) for g in used}
    return table


def deployment(table, n):
    """[folds, models] deployment transform of every rotation."""
    return np.stack([table[f, n][0] for f in range(N_FOLDS)])


class GlobalTask(Task):
    """A task whose recalibration is the table chosen for all benchmarks; views and losses of the engine follow it."""

    def __init__(self, task, table):
        vars(self).update({key: value for key, value in vars(task).items() if not key.startswith("_")})
        self._table, self._views = table, {}

    def recal(self, f, n=3):
        return self._table[f, n]


# --------------------------------------------------------------------------------------------------
# what a pool needs of one benchmark
# --------------------------------------------------------------------------------------------------

def profile_of(task):
    """What does not depend on the recalibration: folds, costs, the budget of the best single model, features."""
    b = task.b
    fit_cost, budget = [], []
    for f in range(N_FOLDS):
        fit, val, _ = roles(f)
        valid = np.flatnonzero(b.fold == val)
        if not len(valid):
            raise ValueError(f"benchmark {task.cid}: a validation fold is empty")
        fit_cost.append(knn_router.column_means(b.cost[np.flatnonzero(np.isin(b.fold, fit))]))
        budget.append(knn_router.validation_budget(task, f, valid) / len(valid))
    return dict(fold=b.fold, cost=b.cost, fit_cost=np.stack(fit_cost), budget=budget, stats=task.cal_stats,
                numeric=knn_router.request_features(b, "numeric"))


def rotations_of(task, table):
    """Per rotation, under the recalibration ``table``: fit requests and their utilities, the losses of the judged
    validation fields by model, and the mean loss of every model on the four training folds."""
    twin = GlobalTask(task, table)
    b, out = task.b, []
    for f in range(N_FOLDS):
        fit, val, train = roles(f)
        pool, utility = knn_router.request_utilities(twin, f, fit)
        rows = twin.rows([val])
        if not len(rows):
            raise ValueError("parameter selection requires judged validation fields")
        valid = np.flatnonzero(b.fold == val)
        out.append(dict(
            pool=pool, utility=utility, valid_loss=core.loss_of_mass(twin.view(f, "calibrated")[rows]),
            valid_position=np.searchsorted(valid, b.field_row[task.J[rows]]),
            train_loss=twin.fit_loss(f, np.eye(b.M), "calibrated", train, 4)))
        twin._views.clear()
    return out


# --------------------------------------------------------------------------------------------------
# the fit on a pool of benchmarks
# --------------------------------------------------------------------------------------------------

def fit_pool(cids, profiles, rotations, routers, penalties):
    """Fit the single-model reference and every router on the pool of the benchmarks ``cids``.

    ``routers``: dicts with ``policy``, ``scorer`` (``knn_scores`` or ``ridge_scores``), ``grid``, ``features``
    ({catalogue id: [requests, d]}) and ``standardise``.  Returns ``plans`` ({catalogue id: {(policy, rule): model
    of every request}}), ``selection`` (per policy and rotation), ``reference`` (per rotation) and ``search`` (per
    neighbour policy and catalogue id).
    """
    count = len(cids)
    requests = [len(profiles[c]["fold"]) for c in cids]
    plans = {c: {} for c in cids}
    selection = {router["policy"]: [] for router in routers}
    search = {router["policy"]: dict(same=np.zeros(count), chance=np.zeros(count), cut=np.zeros(count, np.int64),
                                     float_ties=np.zeros(count, np.int64))
              for router in routers if router["scorer"] is knn_scores}
    reference = []
    for f in range(N_FOLDS):
        fit, val, _ = roles(f)
        rot = [rotations[c][f] for c in cids]
        folds = [profiles[c]["fold"] for c in cids]
        valid = [np.flatnonzero(x == val) for x in folds]
        test = [np.flatnonzero(x == f) for x in folds]
        fitting = [np.flatnonzero(np.isin(x, fit)) for x in folds]
        validations = sum(len(v) for v in valid)
        edges_valid = np.cumsum([len(v) for v in valid])[:-1]
        edges_test = np.cumsum([len(t) for t in test])[:-1]

        def loss_of(models):
            """Pooled validation loss of the models given to the validation requests of all benchmarks."""
            return float(pooled([
                math.fsum(r["valid_loss"][np.arange(len(r["valid_position"])), part[r["valid_position"]]])
                / len(r["valid_position"]) for r, part in zip(rot, np.split(models, edges_valid))]))

        def cost_of(models):
            """Pooled validation cost per request."""
            return float(pooled([math.fsum(profiles[c]["cost"][v, part]) / len(v)
                                 for c, v, part in zip(cids, valid, np.split(models, edges_valid))]))

        train = pooled([r["train_loss"] for r in rot])
        single = int(np.argsort(train, kind="stable")[0])
        cost = pooled([profiles[c]["fit_cost"][f] for c in cids])
        relative = cost / cost.max() if cost.max() else cost
        budgets = dict(capped_single=cost_of(np.full(validations, single)),
                       capped_bd=float(pooled([profiles[c]["budget"][f] for c in cids])))
        reference.append(dict(model=single, training_loss=float(train[single]),
                              validation_cost=budgets["capped_single"]))
        pool_bench = np.repeat(np.arange(count), [len(r["pool"]) for r in rot])
        query_bench = np.concatenate([np.repeat(np.arange(count), [len(v) for v in valid]),
                                      np.repeat(np.arange(count), [len(t) for t in test])])
        utility = np.concatenate([r["utility"] for r in rot])
        for router in routers:
            x, grid, policy = router["features"], tuple(router["grid"]), router["policy"]
            if router["standardise"]:
                centre = pooled([x[c][rows].mean(axis=0) for c, rows in zip(cids, fitting)])
                spread = np.sqrt(pooled([((x[c][rows] - centre) ** 2).mean(axis=0) for c, rows in zip(cids, fitting)]))
                x = {c: (x[c] - centre) / np.where(spread == 0, 1.0, spread) for c in cids}
            pool_x = np.concatenate([x[c][r["pool"]] for c, r in zip(cids, rot)])
            query_x = np.concatenate([x[c][v] for c, v in zip(cids, valid)] + [x[c][t] for c, t in zip(cids, test)])
            scores, found = router["scorer"](pool_x, utility, query_x, grid)
            grid_losses = [loss_of(scores[g][:validations].argmax(axis=1)) for g in grid]
            value = grid[int(np.argmin(grid_losses))]
            losses, costs = [], []
            for penalty in penalties:
                models = (scores[value][:validations] - penalty * relative).argmax(axis=1)
                losses.append(loss_of(models))
                costs.append(cost_of(models))
            chosen = {"uncapped": lowest_loss(dict(grid=penalties, losses=losses))}
            for rule in ("capped_single", "capped_bd"):
                chosen[rule] = knn_router.choose_parameter(penalties, losses, costs, budgets[rule])
            for rule, penalty in chosen.items():
                models = (scores[value][validations:] - penalty * relative).argmax(axis=1)
                for c, size, rows, part in zip(cids, requests, test, np.split(models, edges_test)):
                    plans[c].setdefault((policy, rule), np.empty(size, np.int32))[rows] = part
            selection[policy].append(dict(
                value=value, grid_losses=grid_losses, penalty={rule: float(v) for rule, v in chosen.items()},
                losses=losses, costs=costs, budgets=budgets, fit_requests=len(pool_x),
                within_budget={rule: int(sum(c <= budgets[rule] for c in costs)) for rule in budgets}))
            if policy in search:
                parts = search_statistics(found, grid, value, pool_bench, query_bench, validations, count)
                for key, part in zip(("same", "chance", "cut", "float_ties"), parts):
                    search[policy][key] += part
    search = {policy: {c: {key: block[key][i].item() for key in block} for i, c in enumerate(cids)}
              for policy, block in search.items()}
    return dict(plans=plans, selection=selection, reference=reference, search=search)


# --------------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------------

def routed(task, models, choices, folds, across):
    """Outcome of answering every request with the vectors of one model under the transforms ``choices``."""
    ledger = empty_ledger()
    ledger.update(labeled_folds=folds, models_profiled=task.b.M)
    if across:
        ledger.update(other_benchmark_labels=True, note="the benchmark of a request is not given to the policy")
    return Outcome(request_vectors(task, models, choices), single_stage(task, models), {}, ledger)


def evaluate(task, entries, digest, root, across):
    """Rows, per-field arrays and routing shares of ``entries``: (policy, rule, models, transforms, labelled folds)."""
    rows, fields, routing = [], {}, {}
    for policy, rule, models, choices, folds in entries:
        outcome = routed(task, models, choices, folds, across)
        scored = metrics.score(task, outcome)
        row = RX._result_row(task, scored, policy, rule, digest)
        if root is not None:
            row["native"] = native.native_score(root, Path(root) / LM.ROWS, task.b, outcome.pooled)
        rows.append(row)
        fields[label(policy, rule)] = scored.mass, scored.hit.astype(bool)
        routing[label(policy, rule)] = [float(v) for v in np.bincount(models, minlength=task.b.M) / task.R]
    return rows, fields, routing


def _release_scorer():
    """Drop the kit's scorer of the benchmark just handled; a worker handles many benchmarks."""
    native._scorer.cache_clear()
    native._BENCHES.clear()


def deviation(row, reference):
    """Absolute differences of a row from a stored row."""
    out = {key: abs(row[key] - reference[key]) for key in ("log_loss", "accuracy")}
    if row["native"] is not None and reference["native"] is not None:
        out["native"] = abs(row["native"] - reference["native"])
    return out


def known(cid, stored, corpus, settings, results, features, root, digest):
    """Pass 1, the benchmark is known: the policies on a pool of one benchmark, the checks, the benchmark's profile.

    ``stored`` holds the benchmark's stored rows: ``best_single``, ``knn_router_text`` and, when there is one, the
    uncapped text router of the router study under the key ``uncapped``.
    """
    task = load_task(corpus, cid)
    b, name = task.b, Path(features).name
    text = knn_router.request_features(b, str(features))
    penalties = tuple(settings["knn_router"]["lambda_grid"])
    profile = profile_of(task)
    table = global_choices([profile["stats"]])
    for f in range(N_FOLDS):
        for n in (3, 4):
            own, cross = task.recal(f, n)
            if not (np.array_equal(table[f, n][0], own) and set(table[f, n][1]) == set(cross)
                    and all(np.array_equal(table[f, n][1][g], cross[g]) for g in cross)):
                raise ValueError(f"benchmark {cid}: the pooled recalibration of one benchmark is not its own")
    routers = [dict(policy=f"knn_within_{name}", scorer=knn_scores, grid=tuple(settings["knn_router"]["k_grid"]),
                    features={cid: text}, standardise=False),
               dict(policy=f"ridge_within_{name}", scorer=ridge_scores, grid=ALPHAS, features={cid: text},
                    standardise=False)]
    fit = fit_pool([cid], {cid: profile}, {cid: rotations_of(task, table)}, routers, penalties)
    entries = [(router["policy"], rule, fit["plans"][cid][router["policy"], source], deployment(table, 3),
                "fit and validation") for router in routers for rule, source in WITHIN_RULES.items()]
    single = np.asarray([r["model"] for r in fit["reference"]], np.int32)[b.fold]
    entries.append((REFERENCE, None, single, deployment(table, 4), "train4"))
    rows, fields, routing = evaluate(task, entries, digest, root, across=False)
    if root is not None:
        _release_scorer()
    by_label = {label(r["method"], r["variant"]): r for r in rows}
    archived = {}
    for method in (BEST, TEXT):
        mass, hit = RX._local_fields(results, method, task)
        if (abs(stored[method]["accuracy"] - hit.mean()) > TOLERANCE
                or abs(stored[method]["log_loss"] - core.loss_of_mass(mass).mean()) > TOLERANCE):
            raise ValueError(f"{method}: the archived fields do not reproduce rows.json on benchmark {cid}")
        archived[method] = mass, hit.astype(bool)
    capped = label(f"knn_within_{name}", "capped")
    check = dict(
        text=deviation(by_label[capped], stored[TEXT]), single=deviation(by_label[REFERENCE], stored[BEST]),
        fields_identical=bool(np.array_equal(fields[capped][0], archived[TEXT][0])
                              and np.array_equal(fields[capped][1], archived[TEXT][1])),
        float_ties=fit["search"][f"knn_within_{name}"][cid]["float_ties"])
    if "uncapped" in stored:
        check["uncapped"] = deviation(by_label[label(f"knn_within_{name}", "uncapped")], stored["uncapped"])
    chosen = {}
    for router in routers:
        records = fit["selection"][router["policy"]]
        chosen[router["policy"]] = dict(
            value=[r["value"] for r in records], capped=[r["penalty"]["capped_bd"] for r in records],
            uncapped=[r["penalty"]["uncapped"] for r in records],
            none_within_budget=[r["within_budget"]["capped_bd"] == 0 for r in records])
    own = {n: np.stack([task.recal(f, n)[0] for f in range(N_FOLDS)]) for n in (3, 4)}
    return dict(
        cid=cid, rows=[r for r in rows if r["method"] != REFERENCE],
        fields={key: value for key, value in fields.items() if key != REFERENCE}, archived=archived,
        routing={key: value for key, value in routing.items() if key != REFERENCE}, check=check, chosen=chosen,
        search=fit["search"][f"knn_within_{name}"][cid], profile=profile, own=own, requests=task.R, models=b.M,
        group=task.group, groups=task.G, group_fold=task.group_fold, distinct=int(len(np.unique(text, axis=0))),
        width=int(text.shape[1]))


def unknown_tables(cid, corpus, table):
    """Pass 2: what the pooled fits need of one benchmark under the recalibration for all benchmarks."""
    return rotations_of(load_task(corpus, cid), table)


def unknown(cid, plan, corpus, table, singles, digest, root):
    """Pass 3: rows and per-field arrays of the policies that do not know the benchmark."""
    task = load_task(corpus, cid)
    entries = [(REFERENCE, None, np.asarray(singles, np.int32)[task.b.fold], deployment(table, 4), "train4")]
    entries += [(policy, rule, models, deployment(table, 3), "fit and validation")
                for (policy, rule), models in plan.items()]
    out = evaluate(task, entries, digest, root, across=True)
    if root is not None:
        _release_scorer()
    return out


# --------------------------------------------------------------------------------------------------
# contrasts and the report
# --------------------------------------------------------------------------------------------------

def _number(value):
    """A float for JSON, or None when it is not finite."""
    return float(value) if np.isfinite(value) else None


def contrast(job, groups, cids, suites, draws, seed):
    """A minus B on every suite, for both metrics, with the interval within and the interval over benchmarks."""
    a, b, tables = job
    block = {}
    for metric, (left, right) in tables.items():
        effect, delta, reps = stats.paired(groups, left, right, cids, draws, seed)
        block[metric] = {}
        for suite, members in suites.items():
            within = effect if members == cids else stats.suite_contrast(delta, reps, members)
            over = _over_benchmarks([delta[c] for c in members], metric, draws, seed)
            block[metric][suite] = dict(
                a=a, b=b, difference=within["delta"], within_low=_number(within["ci_low"]),
                within_high=_number(within["ci_high"]), within_p=_number(within["p"]), over_low=over["low"],
                over_high=over["high"], benchmarks_better=over["better"], benchmarks_worse=over["worse"],
                n_benchmarks=len(members), draws=within["draws"])
    return f"{a} - {b}", block


def within_summary(parts, members, policy, grid, penalties):
    """Counts over the benchmark-rotation cells of a suite for a policy fitted inside each benchmark."""
    cells = [(parts[c]["chosen"][policy], f) for c in members for f in range(N_FOLDS)]
    return dict(
        cells=len(cells), grid=list(grid), value_chosen=[sum(c["value"][f] == g for c, f in cells) for g in grid],
        penalties=list(penalties), chosen=[sum(c["capped"][f] == p for c, f in cells) for p in penalties],
        chosen_uncapped=[sum(c["uncapped"][f] == p for c, f in cells) for p in penalties],
        none_within_budget=sum(bool(c["none_within_budget"][f]) for c, f in cells),
        cap_changed_choice=sum(c["capped"][f] != c["uncapped"][f] for c, f in cells))


def neighbour_summary(search, sizes, members):
    """Equal-benchmark means of the neighbour shares and the counts of neighbourhoods cut inside a tie."""
    share = {c: search[c]["same"] / sizes[c] for c in members}
    chance = {c: search[c]["chance"] / sizes[c] for c in members}
    cut = {c: search[c]["cut"] for c in members}
    return dict(
        same_benchmark_share=math.fsum(share.values()) / len(members),
        chance_share=math.fsum(chance.values()) / len(members),
        benchmarks_above_half=sum(v > 0.5 for v in share.values()),
        requests_cut_inside_tie=int(sum(cut.values())), requests=int(sum(sizes[c] for c in members)),
        share_cut_inside_tie=math.fsum(cut[c] / sizes[c] for c in members) / len(members),
        benchmarks_with_cut=sum(v > 0 for v in cut.values()),
        benchmarks_all_cut=sum(cut[c] == sizes[c] for c in members),
        float_ties=int(sum(search[c]["float_ties"] for c in members)))


def stored_points(results, digest, name, suite, stored_suite):
    """Suite points of the stored policies that the new ones are read against."""
    out = {}
    for method in (BEST, "knn_router", TEXT):
        if (method, suite) in stored_suite:
            out[method] = {None: {k: stored_suite[method, suite][k] for k in POINT_KEYS}}
    for file, methods in (("router_study.json", ("router_control", f"router_{name}")),
                          ("operating_points.json", ("single_by_cost", TEXT))):
        path = results / file
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["settings_sha256"] != digest:
            continue
        for r in document["rows"]:
            if r["catalog_id"] is None and r["suite"] == suite and r["method"] in methods:
                out.setdefault(f"{file[:-5]}:{r['method']}", {})[r["variant"]] = {k: r[k] for k in POINT_KEYS}
    return {method: {str(rule): point for rule, point in rules.items()} for method, rules in out.items()}


def run(corpus, settings_path, results, features, out, root=None, processes=1, progress=None):
    corpus, results, features, out = Path(corpus), Path(results), Path(features), Path(out)
    say = progress or (lambda message: None)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    document = json.loads((results / "rows.json").read_text(encoding="utf-8"))
    if document["settings_sha256"] != digest:
        raise SystemExit("rows.json was made with other settings; run the experiments first")
    stored = {(r["method"], r["catalog_id"]): r for r in document["rows"]
              if r["catalog_id"] is not None and r["variant"] is None}
    stored_suite = {(r["method"], r["suite"]): r for r in document["rows"]
                    if r["catalog_id"] is None and r["variant"] is None}
    available = sorted({c for _, c in stored})
    headline = settings["headline_suite"]
    cids = [c for c in available if c not in set(settings["suites"][headline]["exclude"])]
    if not cids or any((method, c) not in stored for c in cids for method in (BEST, TEXT)):
        raise SystemExit(f"rows.json needs {BEST} and {TEXT} on every benchmark of the pool; run local_methods first")
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in available if c not in set(definition["exclude"])]
        if members and set(members) <= set(cids):
            suites[suite] = members
    name = features.name
    meta = json.loads((features / "features.json").read_text(encoding="utf-8"))
    study = {}
    if (results / "router_study.json").is_file():
        other = json.loads((results / "router_study.json").read_text(encoding="utf-8"))
        if other["settings_sha256"] == digest:
            study = {r["catalog_id"]: r for r in other["rows"] if r["catalog_id"] is not None
                     and r["method"] == f"router_{name}" and r["variant"] == "uncapped"}
    k_grid, penalties = tuple(settings["knn_router"]["k_grid"]), tuple(settings["knn_router"]["lambda_grid"])
    draws, seed = settings["bootstrap"]["draws"], settings["bootstrap"]["seed"]

    with (ProcessPoolExecutor(processes) if processes > 1 else nullcontext()) as pool:
        each = pool.map if pool else map
        references = [dict({BEST: stored[BEST, c], TEXT: stored[TEXT, c]},
                           **({"uncapped": study[c]} if c in study else {})) for c in cids]
        work = partial(known, corpus=corpus, settings=settings, results=results, features=features, root=root,
                       digest=digest)
        parts = {part["cid"]: part for part in each(work, cids, references)}
        say(f"{len(cids)} benchmarks fitted one by one")

        # the checks on pools of one benchmark
        deviating = {c: parts[c]["check"] for c in cids
                     if max(parts[c]["check"]["text"]["log_loss"], parts[c]["check"]["text"]["accuracy"]) > TOLERANCE}
        unexplained = sorted(c for c, check in deviating.items() if not check["float_ties"])
        if unexplained:
            raise ValueError(f"knn_within_{name} does not reproduce {TEXT} on benchmarks {unexplained}, "
                             "and no tie in floating point explains it")
        worst = sorted(c for c in cids if max(parts[c]["check"]["single"]["log_loss"],
                                              parts[c]["check"]["single"]["accuracy"]) > TOLERANCE)
        if worst:
            raise ValueError(f"{REFERENCE} on a pool of one benchmark does not reproduce {BEST} on benchmarks {worst}")
        late = sorted(c for c in cids if "uncapped" in parts[c]["check"] and not parts[c]["check"]["float_ties"]
                      and max(parts[c]["check"]["uncapped"]["log_loss"],
                              parts[c]["check"]["uncapped"]["accuracy"]) > TOLERANCE)
        if late:
            raise ValueError(f"knn_within_{name}.uncapped does not reproduce router_study.json on benchmarks {late}")

        # the pool of all benchmarks
        profiles = {c: parts[c]["profile"] for c in cids}
        table = global_choices([profiles[c]["stats"] for c in cids])
        rotations = dict(zip(cids, each(partial(unknown_tables, corpus=corpus, table=table), cids)))
        text = {c: knn_router.request_features(SimpleNamespace(cid=c, R=parts[c]["requests"]), str(features))
                for c in cids}
        routers = [
            dict(policy=f"knn_across_{name}", scorer=knn_scores, grid=k_grid, features=text, standardise=False),
            dict(policy="knn_across_numeric", scorer=knn_scores, grid=k_grid,
                 features={c: profiles[c]["numeric"] for c in cids}, standardise=True),
            dict(policy=f"ridge_across_{name}", scorer=ridge_scores, grid=ALPHAS, features=text, standardise=False)]
        fit = fit_pool(cids, profiles, rotations, routers, penalties)
        del rotations, text
        say("the pool of all benchmarks fitted")
        singles = [r["model"] for r in fit["reference"]]
        scored = list(each(partial(unknown, corpus=corpus, table=table, singles=singles, digest=digest, root=root),
                           cids, [fit["plans"][c] for c in cids]))

        rows, fields, routing = [], {}, {}
        for c, (new_rows, new_fields, new_routing) in zip(cids, scored):
            rows.extend(new_rows + parts[c]["rows"])
            fields[c] = dict(new_fields, **parts[c]["fields"], **parts[c]["archived"])
            routing[c] = dict(new_routing, **parts[c]["routing"])
        across = [label(router["policy"], rule) for router in routers for rule in ACROSS_RULES]
        ridge_within = [label(f"ridge_within_{name}", rule) for rule in WITHIN_RULES]
        pairs = [(a, b) for a in across for b in (REFERENCE, BEST, TEXT)]
        pairs += [(REFERENCE, BEST), (REFERENCE, TEXT)]
        pairs += [(a, b) for a in ridge_within for b in (TEXT, BEST)]
        pairs += [(label(f"ridge_within_{name}", "uncapped"), label(f"knn_within_{name}", "uncapped"))]
        groups = {c: SimpleNamespace(cid=c, group=parts[c]["group"], G=parts[c]["groups"],
                                     group_fold=parts[c]["group_fold"]) for c in cids}
        losses = {key: {c: core.loss_of_mass(fields[c][key][0]) for c in cids} for key in fields[cids[0]]}
        hits = {key: {c: fields[c][key][1].astype(np.float64) for c in cids} for key in fields[cids[0]]}
        jobs = [(a, b, {"log_loss": (losses[a], losses[b]), "accuracy": (hits[a], hits[b])}) for a, b in pairs]
        say("policies scored")
        contrasts = dict(each(partial(contrast, groups=groups, cids=cids, suites=suites, draws=draws, seed=seed), jobs))

    for key in fields[cids[0]]:
        if key not in (BEST, TEXT):
            arrays = {f"{kind}_{c}": fields[c][key][i] for c in cids for i, kind in enumerate(("mass", "hit"))}
            RX._write(RX._field_file(out, key), lambda stream, a=arrays: np.savez_compressed(stream, **a))
    all_rows = RX._with_suites(rows, {suite: settings["suites"][suite] for suite in suites})
    sizes = {c: parts[c]["requests"] for c in cids}
    models = parts[cids[0]]["models"]
    searches = dict(fit["search"], **{f"knn_within_{name}": {c: parts[c]["search"] for c in cids}})
    report = {}
    for suite, members in suites.items():
        points = {}
        for r in all_rows:
            if r["catalog_id"] is None and r["suite"] == suite:
                points.setdefault(r["method"], {})[str(r["variant"])] = {k: r[k] for k in POINT_KEYS}
        report[suite] = dict(
            benchmarks=len(members), points=points, stored=stored_points(results, digest, name, suite, stored_suite),
            contrasts={pair: {metric: block[metric][suite] for metric in block} for pair, block in contrasts.items()},
            neighbours={policy: neighbour_summary(block, sizes, members) for policy, block in searches.items()},
            routing={key: [math.fsum(routing[c][key][m] for c in members) / len(members) for m in range(models)]
                     for key in routing[cids[0]]},
            within={f"knn_within_{name}": within_summary(parts, members, f"knn_within_{name}", k_grid, penalties),
                    f"ridge_within_{name}": within_summary(parts, members, f"ridge_within_{name}", ALPHAS, penalties)})
    checks = [parts[c]["check"] for c in cids]
    reproduction = dict(
        benchmarks=len(cids), tolerance=TOLERANCE, reference=TEXT,
        largest_log_loss_deviation=max(c["text"]["log_loss"] for c in checks),
        largest_accuracy_deviation=max(c["text"]["accuracy"] for c in checks),
        largest_native_deviation=max((c["text"]["native"] for c in checks if "native" in c["text"]), default=None),
        benchmarks_deviating=len(deviating),
        benchmarks_with_identical_fields=sum(c["fields_identical"] for c in checks),
        float_ties=int(sum(c["float_ties"] for c in checks)),
        benchmarks_with_float_ties=sum(c["float_ties"] > 0 for c in checks),
        single_largest_log_loss_deviation=max(c["single"]["log_loss"] for c in checks),
        single_largest_accuracy_deviation=max(c["single"]["accuracy"] for c in checks),
        uncapped_benchmarks_compared=sum("uncapped" in c for c in checks),
        uncapped_largest_log_loss_deviation=max((c["uncapped"]["log_loss"] for c in checks if "uncapped" in c),
                                                default=None),
        uncapped_largest_accuracy_deviation=max((c["uncapped"]["accuracy"] for c in checks if "uncapped" in c),
                                                default=None))
    differing = {str(n): float(np.mean([deployment(table, n) != parts[c]["own"][n] for c in cids])) for n in (3, 4)}
    RX._json(out / "router_unknown.json", dict(
        synthetic=document["synthetic"], settings_sha256=digest, pool=headline, benchmarks=cids, models=models,
        representation=name, draws=draws, seed=seed,
        features=dict({k: meta[k] for k in ("model", "max_tokens", "dtype", "device", "reduction", "revision",
                                            "normalised") if k in meta},
                      dimensions=parts[cids[0]]["width"],
                      distinct_vectors={str(c): parts[c]["distinct"] for c in cids},
                      benchmarks_with_one_vector=sum(parts[c]["distinct"] == 1 for c in cids),
                      benchmarks_under_half_distinct=sum(2 * parts[c]["distinct"] < sizes[c] for c in cids)),
        grids=dict(k=list(k_grid), alpha=list(ALPHAS), penalty=list(penalties)),
        rules=dict(across=list(ACROSS_RULES), within=list(WITHIN_RULES)),
        policies=dict(reference=REFERENCE, across=[router["policy"] for router in routers],
                      within=[f"knn_within_{name}", f"ridge_within_{name}"], stored=[BEST, TEXT]),
        rows=all_rows, selection=fit["selection"], reference=fit["reference"],
        recalibration=dict(menu=[list(entry) for entry in MENU], fit_folds=deployment(table, 3).tolist(),
                           training_folds=deployment(table, 4).tolist(), share_differing_from_own=differing),
        reproduction=reproduction, suites=report,
        neighbours_by_benchmark={policy: {str(c): dict(
            same_benchmark_share=block[c]["same"] / sizes[c], chance_share=block[c]["chance"] / sizes[c],
            requests_cut_inside_tie=block[c]["cut"], requests=sizes[c], float_ties=block[c]["float_ties"])
            for c in cids} for policy, block in searches.items()},
        routing_by_benchmark={str(c): routing[c] for c in cids}))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Routing when the benchmark of a request is unknown")
    for name in ("corpus", "settings", "results", "features", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--project-root")
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    report = run(a.corpus, a.settings, a.results, a.features, a.out, a.project_root, a.processes,
                 progress=lambda message: print(message, file=sys.stderr, flush=True))
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    block = report[headline]
    for source in ("stored", "points"):
        for policy, rules in sorted(block[source].items()):
            for rule, p in rules.items():
                own = "" if p["native"] is None else f"  own metric {p['native']:.4f}"
                print(f"{policy:34s} {rule:14s} log loss {p['log_loss']:.4f}  accuracy {100 * p['accuracy']:.2f}"
                      f"  USD per 1,000 {p['usd_per_1k_requests']:.4f}{own}")
    for policy, n in block["neighbours"].items():
        print(f"{policy}: own-benchmark neighbours {n['same_benchmark_share']:.3f} (chance {n['chance_share']:.3f}); "
              f"cut inside a tie on {n['benchmarks_with_cut']} benchmarks")
    return 0


if __name__ == "__main__":
    sys.exit(main())

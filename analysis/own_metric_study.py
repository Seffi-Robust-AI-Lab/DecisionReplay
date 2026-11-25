"""Intervals for the benchmarks' own metrics, and a single-model reference chosen by the own metric.

    python3 -B -m analysis.own_metric_study --corpus DIR --settings FILE --results RESULTS \
        --rows DIR --project-root . --out OUT [--features DIR] [--draws N] [--processes N]

Why.  The paper's reference is ``best_single``: for each test fold, the model with the lowest calibrated log loss
on the other four folds.  On accuracy the nearest-neighbour router is below that reference.  On the benchmarks' own
metrics (macro-F1, nDCG@10, exact match per case, ...) the router's equal-benchmark mean is above it, and the
stored results give that lead no interval and compare it with a reference that was not chosen by the own metric.
A review asked for both.  Everything here is computed from stored vectors; no model is called.  It is a local
step like ``sim.native``: it needs the benchmark rows and the pinned kit.

Design (written before the first run on the real corpus)
---------------------------------------------------------

Suite.  The headline suite of the settings, benchmarks weighted equally (``math.fsum(values) / n``).

Policies.  Rerun from the simulator with the settings: ``best_single``, ``knn_router``, ``knn_router_text`` (the
router with the feature folder of ``--features``; default: the folder named in the settings),
``stack_final_temperature``, ``greedy``, ``certified_exit``, ``confidence_cascade``, ``uniform_mean``.

Own metric.  The pooled vectors of a policy are decoded with ``core.decode`` and handed to the earlier project's
wrapper of the kit exactly as ``sim.native.native_score`` does (decoded option per field; p(yes) per field on the
two retrieval benchmarks).  The wrapper's ``reduce(decoded, "native", group_weights)`` takes a matrix
[draws, G] of multiplicities of the source split groups (G = the benchmark's split groups, in the order of their
codes in the corpus) and returns [draws, candidates]: every candidate of a benchmark is scored in one call, so all
of them see the same draws.

Checks that stop the run.
(1) ``rows.json`` was made with these settings.
(2) Every suite member of the corpus has a stored row of every policy with an own metric and an accuracy, and the
    rerun reproduces both within 1e-9.  The largest deviations are written to the output.
(3) The scorer's group codes are 0..G-1, the split groups of the corpus, so that weights and groups are aligned.
(4) With a weight of one on every group the weighted call returns the unweighted own metric of every candidate
    within 1e-9.
(5) The own metric of every candidate is defined on every suite member.

Part 1, intervals.  For each policy P and the reference B = ``best_single``:
* per benchmark, d = own(P) - own(B) with the 95% percentile interval of the paired replicate differences;
* on the suite, the mean of d over the benchmarks with two 95% percentile intervals:
  (a) ``groups``: source groups resampled inside each benchmark with the project's draws,
      ``stats.group_multiplicities(task.group_fold, draws, seed=[settings bootstrap seed, cid])`` (stratified by
      fold), fed to the scorer as group weights; the suite replicate is the mean over the benchmarks of the
      benchmark replicates (``stats.suite_contrast``).  These are the draws of the stored accuracy and log-loss
      contrasts.  Draws: the settings' 1,999 when the whole run then takes about 20 minutes or less, otherwise
      999, otherwise 499.  The number is decided before the run by a timing probe that prints seconds and
      yes/no checks and no metric value; it is passed as ``--draws`` and recorded in the output.
  (b) ``benchmarks``: the benchmarks resampled, by the rule of ``paper_facts._over_benchmarks`` that the paper
      uses for accuracy and log loss (``numpy.random.default_rng([seed, n])``, the settings' draws and seed).
* An interval needs a defined replicate in at least 95% of the draws (``stats.interval_and_test``); otherwise it
  is written as null.  ``p`` is the package's centred two-sided test; it is descriptive and not adjusted.
* Benchmarks ``above`` / ``level`` / ``below``: d > 1e-12, abs(d) <= 1e-12, d < -1e-12.

Part 2, a reference chosen by the own metric.  ``best_single_own``: for each test fold f,
* every model's vectors are taken in the calibrated view of rotation f on four folds (the vectors that
  ``best_single.ranking(task, f, n=4)`` scores: on a training fold the transform fitted on the other three
  training folds) and decoded;
* the own metric of each model is computed with weight 1 on the split groups of the four training folds and
  weight 0 on the groups of fold f;
* the model with the highest value is chosen.  Values within 1e-12 of the highest are tied, and a tie goes to the
  model that comes first in ``best_single.ranking(task, f, n=4)``.  A model whose value is undefined is never
  preferred; if every value is undefined the choice is that of ``best_single``.
The chosen models are recalibrated like ``best_single`` (the transform fitted on the four training folds); each
request is answered by the model chosen for its fold; own metric, accuracy, log loss and cost are computed on
the stitched answers, as for ``best_single``.  The answers of ``best_single_own`` are scored in the same call as
the policies, so part 2 uses the draws of part 1.

Fallback of part 2.  Whether weights of 0 and 1 give the kit's metric of the training requests is checked, not
assumed: for every rotation the weighted values of all models are compared with a scorer of the wrapper built
from the training requests only (``PaperScores(view, row_indices=...)``).  A benchmark where the two differ by
more than 1e-9 for any model (or are defined for different models), or where that scorer raises ``ValueError``,
falls back to the models of ``best_single`` and is named in the output with the reason.

Reported for part 2: own metric, accuracy, log loss and cost of ``best_single_own`` beside ``best_single``; the
number of benchmark-fold cells where the two choose different models; every policy against ``best_single_own``
on the own metric with both intervals (the brief asks for the two routers and ``stack_final_temperature``; the
others cost nothing more); the same per benchmark, so that the benchmarks where the two references differ by more
than one point (0.01) on the own metric can be listed.

Hindsight, for orientation only.  ``single_own_hindsight``: per benchmark the one model, the same for all five
folds, whose stitched answers (recalibrated like ``best_single``) have the highest own metric; ties within 1e-12
go to the lower model index.  It is chosen on the test folds, so it is an upper reference and not a policy; it is
labelled so in the output and gets no interval.

Output.  ``OUT/own_metric.json``: ``benchmarks`` (one block per suite member: the policies' own metric, accuracy,
log loss and cost; the models of both references per fold and their own metric on the training folds; the
hindsight model; the differences to both references with the within-benchmark interval; the checks) and
``summary`` (suite means, both kinds of interval for both references, the counts of part 2).  Aggregates only: no
per-item value, no path, no timing.  Two local files are written beside it and never leave the machine:
``OUT/local/fields-best_single_own.npz`` (accepted mass and hit per judged field, in the layout of the other
``fields-<method>.npz``) and ``OUT/local/own_metric_replicates.npz`` (the bootstrap replicates per benchmark).

Tests use the made-up corpus: with the kit where it can score the made-up rows, and with a stand-in scorer
(accuracy over judged fields, the same interface) everywhere else.  Added after the first run of the paper's
experiments, on 23 November 2026, in answer to a review.

Timing probe (before the run on the real corpus; it printed seconds and yes/no checks, no metric value).  On the
nine benchmarks whose weighted scoring the made-up corpus cannot exercise or that are the largest of their kind
(2, 36, 20, 22, 23, 40, 38, 5, 57), one weighted call with 48 draws and nine candidates took at most 0.007 s, so
the settings' 1,999 draws are used.  On all nine the group codes were aligned, all-ones weights gave the
unweighted metric, the output had the shape [draws, candidates] with every value defined, and weights of 0 and 1
gave the metric of a scorer built from the training requests (rotation 0).

Run on the real corpus (23 November 2026): once, on the 35 benchmarks of the headline suite, with 1,999 draws.
Every check passed and no benchmark fell back.  Nothing of the design above was changed after the run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from data.load import list_benchmarks
from data.schema import N_FOLDS, sha256_file
from sim import core, metrics, native, stats
from sim.baselines import BASELINES, best_single, knn_router
from sim.baselines.random_single import request_vectors
from sim.baselines.uniform_mean import member_choices, panel
from sim.engine import load_task
from sim.types import LOG_FLOOR
from . import paper_facts as PF
from . import run_experiments as RX

TEXT_ROUTER = "knn_router_text"
POLICIES = ("best_single", "knn_router", TEXT_ROUTER, "stack_final_temperature", "greedy", "certified_exit",
            "confidence_cascade", "uniform_mean")
REFERENCE = "best_single"
OWN_REFERENCE = "best_single_own"
HINDSIGHT = "single_own_hindsight"
CANDIDATES = POLICIES + (OWN_REFERENCE,)        # scored together, with the same draws
ASKED_AGAINST_OWN = ("knn_router", TEXT_ROUTER, "stack_final_temperature")
POINT_KEYS = ("accuracy", "log_loss", "usd_per_1k_requests", "calls_per_request")
TOLERANCE = 1e-9                                # stored values; weighted against unweighted or subset scoring
TIE = 1e-12                                     # two own-metric values this close are the same
ONE_POINT = 0.01


# --------------------------------------------------------------------------------------------------
# decoded answers and the two scorers
# --------------------------------------------------------------------------------------------------

@dataclass
class Answers:
    """Decoded answers of several candidates on every field, in the layout that the wrapper's scorer reads."""

    winner: np.ndarray                   # [candidates, F] int32, decoded option
    loss: np.ndarray                     # [candidates, F] log loss, NaN on fields that are not judged
    hit: np.ndarray                      # [candidates, F] bool
    yes: np.ndarray                      # [candidates, F] p(yes) on the retrieval benchmarks, NaN elsewhere

    def take(self, fields):
        return Answers(self.winner[:, fields], self.loss[:, fields], self.hit[:, fields], self.yes[:, fields])


def decode_all(bench, pooled_sets):
    """Answers of several sets of pooled vectors {width: [n_w, w]}, decoded as ``sim.native.native_score`` does."""
    winner, loss, hit, yes = [], [], [], []
    for pooled in pooled_sets:
        right, mass, choice, _ = core.decode(bench, pooled)
        field_loss = np.full(bench.F, np.nan)
        field_loss[bench.judged] = core.loss_of_mass(mass[bench.judged], LOG_FLOOR)
        p_yes = np.full(bench.F, np.nan)
        if bench.cid in native.RECALL_ONLY:
            for w in bench.widths:
                if w == 2:
                    v = pooled[w]
                    p_yes[bench.idx[w]] = (v / v.sum(axis=1, keepdims=True))[:, 1]
        winner.append(choice.astype(np.int32))
        loss.append(field_loss)
        hit.append(right)
        yes.append(p_yes)
    return Answers(np.stack(winner), np.stack(loss), np.stack(hit), np.stack(yes))


class KitScorer:
    """The benchmark's own metric of the pinned kit, through the earlier project's wrapper (as ``sim.native``)."""

    NAME = "kit"
    STORED = "native"                    # the column of rows.json that holds this metric

    def __init__(self, project_root, rows_dir, bench):
        key = id(bench)
        native._BENCHES[key] = bench
        self.scores, self._decoded = native._scorer(str(project_root), str(rows_dir), bench.cid, key)
        self.group_ids = np.asarray(self.scores.group_ids)

    def _hand(self, answers):
        return self._decoded(answers.winner, answers.loss, answers.hit, answers.yes)

    def own(self, answers, weights=None):
        """[candidates] own metric; with ``weights`` [draws, G] on the split groups, [draws, candidates]."""
        return np.asarray(self.scores.reduce(self._hand(answers), "native", weights), dtype=np.float64)

    def own_subset(self, answers, requests):
        """[candidates] own metric from a scorer of the wrapper that holds the given requests only."""
        part = type(self.scores)(self.scores.data, row_indices=np.asarray(requests))
        return np.asarray(part.reduce(self._hand(answers.take(part.field_indices)), "native"), dtype=np.float64)

    def close(self):
        """Release the cached rows of the benchmark."""
        native._scorer.cache_clear()
        native._BENCHES.clear()


class FieldAccuracy:
    """Stand-in for corpora that the kit cannot score: accuracy over judged fields, with the same interface."""

    NAME = "judged-field accuracy (stand-in)"
    STORED = "accuracy"

    def __init__(self, bench):
        self.bench = bench
        self.group_ids = np.unique(bench.split_group)
        self.judged = np.flatnonzero(bench.judged)
        self.groups = bench.split_group[bench.field_row[self.judged]]

    def own(self, answers, weights=None):
        hit = answers.hit[:, self.judged].astype(np.float64)
        if weights is None:
            return hit.mean(axis=1)
        counts = np.bincount(self.groups, minlength=weights.shape[1])
        sums = np.stack([np.bincount(self.groups, weights=h, minlength=weights.shape[1]) for h in hit])
        with np.errstate(invalid="ignore", divide="ignore"):
            return (weights @ sums.T) / (weights @ counts)[:, None]

    def own_subset(self, answers, requests):
        keep = np.isin(self.bench.field_row[self.judged], requests)
        return answers.hit[:, self.judged[keep]].astype(np.float64).mean(axis=1)

    def close(self):
        pass


# --------------------------------------------------------------------------------------------------
# the reference chosen by the own metric
# --------------------------------------------------------------------------------------------------

def training_weights(task, fold):
    """[1, G] weight 1 on the split groups of the four training folds of test fold ``fold``, 0 on its own."""
    return (task.group_fold != fold).astype(np.float64)[None, :]


def rotation_vectors(task, model, fold):
    """One model's vectors on every field in the calibrated view of rotation ``fold`` on four folds.

    On a training fold the transform is the one fitted on the other three training folds, on the test fold the
    one fitted on all four: the view that ``best_single.ranking(task, fold, n=4)`` scores.
    """
    return request_vectors(task, np.full(task.R, model, np.int32), task.choice_by_fold(fold, 4))


def choose(values, order):
    """The model with the highest value; ties within ``TIE`` go to the model that comes first in ``order``.

    An undefined value is never preferred; if every value is undefined the first model of ``order`` is returned.
    """
    v = np.asarray(values, dtype=np.float64)
    v = np.where(np.isfinite(v), v, -np.inf)
    tied = v >= v.max() - TIE
    return int(next(m for m in order if tied[m]))


def largest_gap(a, b):
    """Largest absolute difference of two vectors; infinite when one is defined where the other is not."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    defined = np.isfinite(a)
    if not np.array_equal(defined, np.isfinite(b)):
        return float("inf")
    return float(np.abs(a[defined] - b[defined]).max()) if defined.any() else 0.0


def own_reference(task, scorer):
    """What part 2 chooses on one benchmark, with the check that weights of 0 and 1 give the training metric."""
    M = task.b.M
    chosen, single, values_chosen, values_single, undefined = [], [], [], [], 0
    gap, reason = 0.0, None
    for f in range(N_FOLDS):
        ranking = best_single.ranking(task, f, n=4)
        answers = decode_all(task.b, [rotation_vectors(task, m, f) for m in range(M)])
        values = scorer.own(answers, training_weights(task, f))[0]
        try:
            gap = max(gap, largest_gap(values, scorer.own_subset(answers, np.flatnonzero(task.b.fold != f))))
        except ValueError as error:
            reason = f"the scorer of the training requests raised {type(error).__name__}"
        model = choose(values, ranking)
        chosen.append(model)
        single.append(int(ranking[0]))
        values_chosen.append(_num(values[model]))
        values_single.append(_num(values[ranking[0]]))
        undefined += int(not np.isfinite(values).any())
    if reason is None and not gap <= TOLERANCE:
        reason = "weights of 0 and 1 do not give the own metric of the training requests"
    return dict(models=single if reason else chosen, by_own_metric=chosen, best_single=single,
                training_own=dict(best_single_own=values_chosen, best_single=values_single),
                undefined_cells=undefined, subset_gap=gap, fallback=reason is not None, fallback_reason=reason)


def single_panel(task, models):
    """One model per test fold, recalibrated on the four training folds: the outcome that ``best_single`` builds."""
    models = [int(m) for m in models]
    return panel(task, np.eye(task.b.M)[models], member_choices(task, 4), info={"model": models},
                 ledger={"labeled_folds": "train4", "models_profiled": task.b.M})


def hindsight(task, scorer):
    """(model, own metric per model): the one model per benchmark with the best own metric on the test folds."""
    M = task.b.M
    values = scorer.own(decode_all(task.b, [single_panel(task, [m] * N_FOLDS).pooled for m in range(M)]))
    return choose(values, range(M)), values


# --------------------------------------------------------------------------------------------------
# one benchmark
# --------------------------------------------------------------------------------------------------

def run_policies(task, settings, features):
    """{policy: Outcome} of the rerun policies; the text router is the router with the given feature folder."""
    text = dict(settings, knn_router=dict(settings["knn_router"], features=str(features)))
    return {name: knn_router.run(task, text) if name == TEXT_ROUTER else BASELINES[name].run(task, settings)
            for name in POLICIES}


def replicates(task, scorer, answers, draws, seed):
    """[draws, candidates] own metric under the project's draws of split groups (stratified by fold)."""
    parts = [scorer.own(answers, weights)
             for weights in stats.group_multiplicities(task.group_fold, draws, seed=[seed, task.cid])]
    out = np.concatenate(parts)
    if out.shape != (draws, len(answers.winner)):
        raise ValueError(f"benchmark {task.cid}: the weighted scorer returned {out.shape}")
    return out


def differences(point, reps, names, reference):
    """Per candidate: own metric minus the reference's, with the interval of the paired replicate differences."""
    r = names.index(reference)
    out = {}
    for j, name in enumerate(names):
        if j == r:
            continue
        delta = float(point[j] - point[r])
        low, high, p, finite = stats.interval_and_test(delta, reps[:, j] - reps[:, r])
        out[name] = dict(difference=delta, ci_low=_num(low), ci_high=_num(high), p=_num(p), draws=finite)
    return out


def study(task, scorer, settings, stored, features, draws):
    """(aggregates, replicates [draws, candidates], (mass, hit) of the new reference) of one benchmark."""
    cid = task.cid
    if not np.array_equal(scorer.group_ids, np.arange(task.G)):     # check (3)
        raise ValueError(f"benchmark {cid}: the scorer's groups are not the split groups of the corpus")
    names = list(CANDIDATES)
    outcomes = run_policies(task, settings, features)
    points = {name: metrics.summarize(metrics.score(task, outcome)) for name, outcome in outcomes.items()}
    choice = own_reference(task, scorer)
    outcomes[OWN_REFERENCE] = single_panel(task, choice["models"])
    scored = metrics.score(task, outcomes[OWN_REFERENCE])
    points[OWN_REFERENCE] = metrics.summarize(scored)
    answers = decode_all(task.b, [outcomes[name].pooled for name in names])
    point = scorer.own(answers)
    if not np.isfinite(point).all():                            # check (5)
        raise ValueError(f"benchmark {cid}: the own metric of a candidate is undefined")
    gaps = {"own": 0.0, "accuracy": 0.0}
    for name in POLICIES:                                       # check (2): the stored rows are reproduced
        for key, value in (("own", point[names.index(name)]), ("accuracy", points[name]["accuracy"])):
            gap = abs(float(value) - stored[name][key])
            if not gap <= TOLERANCE:
                raise ValueError(f"benchmark {cid}, {name}: the rerun differs from the stored {key} by {gap}")
            gaps[key] = max(gaps[key], gap)
    ones = largest_gap(point, scorer.own(answers, np.ones((1, task.G)))[0])
    if not ones <= TOLERANCE:                                   # check (4)
        raise ValueError(f"benchmark {cid}: all-ones weights do not give the own metric (gap {ones})")
    reps = replicates(task, scorer, answers, draws, settings["bootstrap"]["seed"])

    best_model, by_model = hindsight(task, scorer)
    for name in (REFERENCE, OWN_REFERENCE):                     # one model in all folds is a hindsight candidate
        models = set(outcomes[name].info["model"])
        if len(models) == 1 and abs(by_model[models.pop()] - point[names.index(name)]) > TIE:
            raise ValueError(f"benchmark {cid}: the hindsight candidates do not reproduce {name}")
    block = dict(
        requests=task.R, judged_fields=task.nJ, split_groups=task.G,
        policies={name: dict({key: points[name][key] for key in POINT_KEYS}, own=float(point[j]))
                  for j, name in enumerate(names)},
        models=dict(best_single=choice["best_single"], best_single_own=choice["models"],
                    by_own_metric=choice["by_own_metric"],
                    cells_with_another_model=sum(a != b for a, b in zip(choice["best_single"], choice["models"]))),
        training_own=choice["training_own"], undefined_training_cells=choice["undefined_cells"],
        fallback=choice["fallback"], fallback_reason=choice["fallback_reason"],
        hindsight=dict(model=best_model, own=float(by_model[best_model]), chosen_on="test folds"),
        against_best_single=differences(point, reps, names, REFERENCE),
        against_best_single_own=differences(point, reps, names, OWN_REFERENCE),
        draws_defined=int(np.isfinite(reps).all(axis=1).sum()),
        checks=dict(stored_own_gap=gaps["own"], stored_accuracy_gap=gaps["accuracy"], all_ones_weights_gap=ones,
                    training_subset_gap=_num(choice["subset_gap"])))
    return block, reps, (scored.mass, scored.hit)


def one_benchmark(cid, corpus, settings, stored, features, factory, draws):
    """Load one benchmark, build its scorer, study it and release the scorer."""
    task = load_task(corpus, cid)
    scorer = factory(task.b)
    try:
        return study(task, scorer, settings, stored[cid], features, draws)
    finally:
        scorer.close()


# --------------------------------------------------------------------------------------------------
# the suite
# --------------------------------------------------------------------------------------------------

def _num(value):
    """A float for the JSON file, or None when the value is undefined."""
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def suite_differences(blocks, reps, cids, reference, bootstrap):
    """Every candidate minus ``reference`` on the own metric: suite mean, both intervals, benchmark counts."""
    names = list(CANDIDATES)
    r = names.index(reference)
    out = {}
    for j, name in enumerate(names):
        if j == r:
            continue
        delta = {c: blocks[c]["policies"][name]["own"] - blocks[c]["policies"][reference]["own"] for c in cids}
        effect = stats.suite_contrast(delta, {c: reps[c][:, j] - reps[c][:, r] for c in cids}, cids)
        # "accuracy" names the direction only: a higher own metric is better
        over = PF._over_benchmarks([delta[c] for c in cids], "accuracy", bootstrap["draws"], bootstrap["seed"])
        out[name] = dict(
            difference=over["mean"],
            groups=dict(ci_low=_num(effect["ci_low"]), ci_high=_num(effect["ci_high"]), p=_num(effect["p"]),
                        draws=effect["draws"]),
            benchmarks=dict(ci_low=over["low"], ci_high=over["high"], sign_test_p=over["sign_test_p"],
                            draws=bootstrap["draws"]),
            benchmarks_above=over["better"], benchmarks_below=over["worse"],
            benchmarks_level=over["benchmarks"] - over["better"] - over["worse"])
    return out


def summarise(blocks, reps, cids, bootstrap):
    """The suite block: means, both references with both intervals, the counts of part 2, the checks."""
    policies = {name: {key: _mean(blocks[c]["policies"][name][key] for c in cids) for key in POINT_KEYS + ("own",)}
                for name in CANDIDATES}
    policies[HINDSIGHT] = dict(own=_mean(blocks[c]["hindsight"]["own"] for c in cids), chosen_on="test folds")
    gap = {c: blocks[c]["policies"][OWN_REFERENCE]["own"] - blocks[c]["policies"][REFERENCE]["own"] for c in cids}
    cells = [blocks[c]["models"]["cells_with_another_model"] for c in cids]
    subset = [blocks[c]["checks"]["training_subset_gap"] for c in cids]
    return dict(
        benchmarks=len(cids), policies=policies,
        against_best_single=suite_differences(blocks, reps, cids, REFERENCE, bootstrap),
        against_best_single_own=suite_differences(blocks, reps, cids, OWN_REFERENCE, bootstrap),
        best_single_own=dict(
            cells=N_FOLDS * len(cids), cells_with_another_model=sum(cells),
            benchmarks_with_another_model=sum(n > 0 for n in cells),
            fallback_benchmarks=[c for c in cids if blocks[c]["fallback"]],
            undefined_training_cells=sum(blocks[c]["undefined_training_cells"] for c in cids),
            benchmarks_differing_by_more_than_one_point=[c for c in cids if abs(gap[c]) > ONE_POINT],
            benchmarks_above_best_single=sum(g > TIE for g in gap.values()),
            benchmarks_below_best_single=sum(g < -TIE for g in gap.values())),
        checks=dict(
            policy_benchmark_cells=len(POLICIES) * len(cids),
            largest_stored_own_gap=max(blocks[c]["checks"]["stored_own_gap"] for c in cids),
            largest_stored_accuracy_gap=max(blocks[c]["checks"]["stored_accuracy_gap"] for c in cids),
            largest_all_ones_weights_gap=max(blocks[c]["checks"]["all_ones_weights_gap"] for c in cids),
            largest_training_subset_gap=None if None in subset else max(subset),
            fewest_defined_draws=min(blocks[c]["draws_defined"] for c in cids)))


def run(corpus, settings_path, results, rows, project_root, out, features=None, draws=None, processes=1,
        scorer=None, cids=None):
    """Run the study and write ``own_metric.json``; ``scorer`` and ``cids`` are for tests on the made-up corpus."""
    corpus, results, out = Path(corpus), Path(results), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    document = json.loads((results / "rows.json").read_text(encoding="utf-8"))
    if document["settings_sha256"] != digest:                   # check (1)
        raise SystemExit("rows.json was made with other settings; run the experiments first")
    suite = settings["headline_suite"]
    draws = settings["bootstrap"]["draws"] if draws is None else int(draws)
    features = Path(settings[TEXT_ROUTER]["features"]) if features is None else Path(features)
    kind = KitScorer if scorer is None else scorer
    factory = partial(KitScorer, str(project_root), str(rows)) if scorer is None else scorer
    catalogue = {b["catalog_id"]: b for b in list_benchmarks(corpus)}
    members = [c for c in sorted(catalogue) if c not in set(settings["suites"][suite]["exclude"])]
    if cids is not None:
        members = [c for c in members if c in set(cids)]
    if not members or draws < 1 or processes < 1:
        raise SystemExit("need a suite member, a positive number of draws and a positive number of processes")
    cell = {(r["method"], r["catalog_id"]): r for r in document["rows"]
            if r["catalog_id"] is not None and r["variant"] is None}
    stored = {}
    for c in members:
        stored[c] = {}
        for name in POLICIES:
            row = cell.get((name, c))
            if row is None or row[kind.STORED] is None or row["accuracy"] is None:
                raise SystemExit(f"rows.json has no own metric and accuracy of {name} on benchmark {c}")
            stored[c][name] = dict(own=row[kind.STORED], accuracy=row["accuracy"])
    work = partial(one_benchmark, corpus=corpus, settings=settings, stored=stored, features=features,
                   factory=factory, draws=draws)
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            results_by_cid = dict(zip(members, pool.map(work, members)))
    else:
        results_by_cid = {c: work(c) for c in members}
    blocks, reps, fields = {}, {}, {}
    for c, (block, replicate, (mass, hit)) in results_by_cid.items():
        info = catalogue[c]
        blocks[c] = dict(block, name=info["name"], area=info["area"], metric=info.get("native_metric"))
        reps[c] = replicate
        fields[f"mass_{c}"], fields[f"hit_{c}"] = mass, hit
    content = dict(
        synthetic=document["synthetic"], settings_sha256=digest, suite=suite, scorer=kind.NAME,
        policies=list(POLICIES), reference=REFERENCE, own_reference=OWN_REFERENCE, hindsight=HINDSIGHT,
        text_router_features=features.name,
        bootstrap=dict(draws=draws, seed=settings["bootstrap"]["seed"],
                       draws_over_benchmarks=settings["bootstrap"]["draws"]),
        tolerances=dict(stored=TOLERANCE, tie=TIE, one_point=ONE_POINT),
        benchmarks={str(c): blocks[c] for c in members},
        summary=summarise(blocks, reps, members, settings["bootstrap"]))
    RX._write(RX._field_file(out, OWN_REFERENCE), lambda stream: np.savez_compressed(stream, **fields))
    arrays = {f"own_{c}": reps[c] for c in members}
    RX._write(out / "local" / "own_metric_replicates.npz",
              lambda stream: np.savez_compressed(stream, candidates=np.asarray(CANDIDATES), **arrays))
    RX._json(out / "own_metric.json", content)
    return content


def _points(value):
    return "    n/a" if value is None else f"{100 * value:+7.2f}"


def _table(title, block, means, reference):
    print(title)
    print(f"  {'candidate':26s} {'own':>7s} {'diff':>7s}  {'groups 95%':>17s}  {'benchmarks 95%':>17s}  above/level/below")
    for name, e in block.items():
        g, b = e["groups"], e["benchmarks"]
        print(f"  {name:26s} {100 * means[name]['own']:7.2f} {_points(e['difference'])}  "
              f"[{_points(g['ci_low'])},{_points(g['ci_high'])}]  [{_points(b['ci_low'])},{_points(b['ci_high'])}]"
              f"  {e['benchmarks_above']}/{e['benchmarks_level']}/{e['benchmarks_below']}")
    print(f"  reference {reference}: own {100 * means[reference]['own']:.2f}")


def _cell(block, key, name):
    e = block[key][name]
    return f"{name} {_points(e['difference'])} [{_points(e['ci_low'])},{_points(e['ci_high'])}]"


def _benchmark_lines(content):
    """The benchmarks that carry the routers' differences, and those where the two references differ."""
    blocks = content["benchmarks"]
    for name in ("knn_router", TEXT_ROUTER):
        order = sorted(blocks, key=lambda c: blocks[c]["against_best_single"][name]["difference"])
        print(f"{name} against best_single, the three lowest and the three highest benchmarks (points):")
        for c in order[:3] + order[-3:]:
            print(f"  {int(c):3d} {blocks[c]['name'][:30]:30s} {_cell(blocks[c], 'against_best_single', name)}")
    print("benchmarks where the two references differ by more than one point (against best_single_own, points):")
    for c in content["summary"]["best_single_own"]["benchmarks_differing_by_more_than_one_point"]:
        block = blocks[str(c)]
        own, single = (block["policies"][n]["own"] for n in (OWN_REFERENCE, REFERENCE))
        print(f"  {c:3d} {block['name'][:30]:30s} best_single {100 * single:.2f}, best_single_own {100 * own:.2f}; "
              + "; ".join(_cell(block, "against_best_single_own", n) for n in ASKED_AGAINST_OWN))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Intervals for the benchmarks' own metrics and a reference chosen by them")
    for name in ("corpus", "settings", "results", "rows", "project-root", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--features")
    ap.add_argument("--draws", type=int)
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    started = time.perf_counter()
    content = run(a.corpus, a.settings, a.results, a.rows, a.project_root, a.out, a.features, a.draws, a.processes)
    summary, checks = content["summary"], content["summary"]["checks"]
    print(f"suite {content['suite']}: {summary['benchmarks']} benchmarks, {content['bootstrap']['draws']} draws, "
          f"scorer {content['scorer']}, {time.perf_counter() - started:.0f} s")
    print("largest gap to rows.json: own metric %.1e, accuracy %.1e; all-ones weights %.1e; training subset %s" % (
        checks["largest_stored_own_gap"], checks["largest_stored_accuracy_gap"],
        checks["largest_all_ones_weights_gap"], checks["largest_training_subset_gap"]))
    _table("own metric against best_single (points)", summary["against_best_single"], summary["policies"], REFERENCE)
    _table("own metric against best_single_own (points)", summary["against_best_single_own"], summary["policies"],
           OWN_REFERENCE)
    for name in (REFERENCE, OWN_REFERENCE):
        p = summary["policies"][name]
        print(f"{name}: own {100 * p['own']:.2f}, accuracy {100 * p['accuracy']:.2f}, log loss {p['log_loss']:.4f}, "
              f"usd per 1k requests {p['usd_per_1k_requests']:.4f}")
    print("hindsight (chosen on the test folds): own %.2f" % (100 * summary["policies"][HINDSIGHT]["own"]))
    print("best_single_own:", json.dumps(summary["best_single_own"], sort_keys=True))
    _benchmark_lines(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())

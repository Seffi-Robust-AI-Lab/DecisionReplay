"""Routing when the benchmark of a request is unknown, on the made-up corpus."""

import dataclasses
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from analysis import router_unknown as RU
from analysis import run_experiments as RX
from data.load import list_benchmarks
from data.schema import N_FOLDS, sha256_file
from sim import metrics
from sim.baselines import best_single, knn_router
from sim.engine import Task, load_task
from sim.types import RESULT_ROW_KEYS

from .test_sim_runner import _settings

CIDS = [2, 11, 25, 59]          # 59 lies outside the headline suite
POOL = [2, 11, 25]
NAME = "madeup"


def _features(corpus, folder):
    """A made-up feature folder: unit float32 rows, many of them copies of one another, and a features.json."""
    folder.mkdir(parents=True)
    for b in list_benchmarks(corpus):
        x = np.random.default_rng(b["catalog_id"]).normal(size=(int(b["requests"]), 8)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        x[::4] = x[0]
        x[1::7] = x[1]
        np.save(folder / f"catalog-{b['catalog_id']:03d}.npy", x)
    (folder / "features.json").write_text(json.dumps(dict(model="made-up", normalised=True)))
    return folder


def _routers(features, profiles, k_grid):
    cids = sorted(profiles)
    return [dict(policy="knn_text", scorer=RU.knn_scores, grid=k_grid, features=features, standardise=False),
            dict(policy="knn_numeric", scorer=RU.knn_scores, grid=k_grid,
                 features={c: profiles[c]["numeric"] for c in cids}, standardise=True),
            dict(policy="ridge_text", scorer=RU.ridge_scores, grid=RU.ALPHAS, features=features, standardise=False)]


def _pool(tasks, features, settings):
    """Everything ``run`` builds for a pool of tasks: profiles, recalibration table, rotations and the fit."""
    cids = sorted(tasks)
    profiles = {c: RU.profile_of(tasks[c]) for c in cids}
    table = RU.global_choices([profiles[c]["stats"] for c in cids])
    rotations = {c: RU.rotations_of(tasks[c], table) for c in cids}
    routers = _routers({c: features[c] for c in cids}, profiles, tuple(settings["knn_router"]["k_grid"]))
    fit = RU.fit_pool(cids, profiles, rotations, routers, tuple(settings["knn_router"]["lambda_grid"]))
    return dict(cids=cids, profiles=profiles, table=table, rotations=rotations, fit=fit)


def _fit(tasks, features, settings):
    """The recalibration table and the fit of a pool of tasks."""
    pool = _pool(tasks, features, settings)
    return pool["table"], pool["fit"]


def _naive(pool, queries, k):
    """The k nearest rows by distance and then position, one query at a time."""
    pool = pool.astype(np.float64)
    out = []
    for q in queries.astype(np.float64):
        distance = np.sqrt(RU.squared_distance(pool - q))
        out.append(np.lexsort((np.arange(len(pool)), distance))[:k])
    return np.array(out)


def test_exact_search():
    rng = np.random.default_rng(1)
    for kind, width, n in ((np.float32, 8, 300), (np.float32, 384, 200), (np.float64, 5, 250), (np.float32, 3, 40)):
        x = rng.normal(size=(n, width)).astype(kind)
        x[::3] = x[0]                       # copies of one row: a tie that the position breaks
        x[5:40:2] = x[5]
        queries = np.concatenate([rng.normal(size=(60, width)).astype(kind), x[:20]])
        if kind is np.float64:              # a grid: different rows at the same distance
            x, queries = np.round(x), np.round(queries)
        for k in (1, 7, 31, n):
            found = RU.nearest(x, queries, k)
            assert np.array_equal(found.members, _naive(x, queries, k)), (kind, width, k)
            assert np.all(np.diff(found.distance, axis=1) >= 0)
        same = np.flatnonzero((x == x[0]).all(axis=1))                  # the copies of the first row, by position
        copies = RU.nearest(x, x[:1], len(same))
        assert np.array_equal(copies.members[0], same) and not copies.distance.any()
        assert len(set(copies.group[0])) == 1
    with pytest.raises(ValueError):
        RU.nearest(x, queries, n + 1)
    utility = rng.normal(size=(40, 3))
    scores, found = RU.knn_scores(x, utility, queries[:5], (5, 10 ** 9))
    assert found.members.shape == (5, 6)
    assert np.array_equal(scores[10 ** 9], np.broadcast_to(knn_router.column_means(utility), (5, 3)))
    assert np.array_equal(scores[5][2], knn_router.column_means(utility[found.members[2, :5]]))


def test_ridge_scores():
    """With little penalty the regression recovers a linear rule; with much it returns the mean utility."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 6)).astype(np.float32)
    weights, shift = rng.normal(size=(6, 3)), rng.normal(size=3)
    utility = x.astype(np.float64) @ weights + shift
    queries = rng.normal(size=(20, 6)).astype(np.float32)
    scores, found = RU.ridge_scores(x, utility, queries, (1e-9, 1e12))
    assert found is None
    assert np.allclose(scores[1e-9], queries.astype(np.float64) @ weights + shift, atol=1e-6)
    assert np.allclose(scores[1e12], utility.mean(axis=0), atol=1e-6)
    with pytest.raises(ValueError):
        RU.ridge_scores(x, utility, queries, (0.0,))
    # the same objective as the library's ridge regression with an intercept, on noisy utilities
    linear_model = pytest.importorskip("sklearn.linear_model")
    noisy = utility + rng.normal(size=utility.shape)
    scores, _ = RU.ridge_scores(x, noisy, queries, RU.ALPHAS)
    for alpha in RU.ALPHAS:
        model = linear_model.Ridge(alpha=alpha, fit_intercept=True, solver="cholesky").fit(x.astype(np.float64), noisy)
        assert np.allclose(scores[alpha], model.predict(queries.astype(np.float64)), rtol=0, atol=1e-9)


def test_pool_plan_is_a_plain_recomputation(corpus):
    """The models a pool of benchmarks routes to are those of a request-by-request recomputation."""
    settings = json.loads((Path(__file__).parents[1] / "SETTINGS.json").read_text())
    with TemporaryDirectory(dir=".") as directory:
        folder = _features(corpus, Path(directory) / NAME)
        tasks = {cid: load_task(corpus, cid) for cid in POOL}
        features = {cid: knn_router.request_features(tasks[cid].b, str(folder)) for cid in POOL}
    pool = _pool(tasks, features, settings)
    cids, profiles, rotations, fit = (pool[key] for key in ("cids", "profiles", "rotations", "fit"))
    same = {c: 0.0 for c in cids}
    for f in range(N_FOLDS):
        record = fit["selection"]["knn_text"][f]
        pool_x = np.concatenate([features[c][rotations[c][f]["pool"]] for c in cids]).astype(np.float64)
        utility = np.concatenate([rotations[c][f]["utility"] for c in cids])
        owner = np.concatenate([np.full(len(rotations[c][f]["pool"]), c) for c in cids])
        cost = RU.pooled([profiles[c]["fit_cost"][f] for c in cids])
        relative = cost / cost.max()
        k = record["value"]
        assert record["fit_requests"] == len(pool_x) > k
        for c in cids:
            test = np.flatnonzero(tasks[c].b.fold == f)
            expected = {rule: [] for rule in RU.ACROSS_RULES}
            for i in test:
                distance = np.sqrt(RU.squared_distance(pool_x - features[c][i].astype(np.float64)))
                members = np.lexsort((np.arange(len(pool_x)), distance))[:k]
                same[c] += float((owner[members] == c).mean())
                for rule in RU.ACROSS_RULES:
                    score = knn_router.column_means(utility[members]) - record["penalty"][rule] * relative
                    expected[rule].append(int(score.argmax()))
            for rule in RU.ACROSS_RULES:
                assert np.array_equal(fit["plans"][c]["knn_text", rule][test], expected[rule]), (f, rule, c)
        # the reference is one model, the best by the pooled loss on the four training folds
        train = RU.pooled([rotations[c][f]["train_loss"] for c in cids])
        assert fit["reference"][f]["model"] == int(np.argmin(train))
    # neighbours from the request's own benchmark: the search's sum is the recomputation's
    for c in cids:
        assert abs(fit["search"]["knn_text"][c]["same"] - same[c]) < 1e-9
        assert 0 < same[c] < tasks[c].R


def test_single_benchmark_restriction(corpus):
    """On a pool of one benchmark the pooled fit is the paper's router and its reference the best single model."""
    settings = json.loads((Path(__file__).parents[1] / "SETTINGS.json").read_text())
    with TemporaryDirectory(dir=".") as directory:
        folder = _features(corpus, Path(directory) / NAME)
        text = dict(settings, knn_router=dict(settings["knn_router"], features=str(folder)))
        for cid in POOL + [9]:
            task = load_task(corpus, cid)
            stored = knn_router.run(task, text)
            features = {cid: knn_router.request_features(task.b, str(folder))}
            table, fit = _fit({cid: task}, features, settings)
            for f in range(N_FOLDS):
                for n in (3, 4):
                    own, cross = task.recal(f, n)
                    assert np.array_equal(table[f, n][0], own)
                    assert all(np.array_equal(table[f, n][1][g], cross[g]) for g in cross)
            outcome = RU.routed(task, fit["plans"][cid]["knn_text", "capped_bd"], RU.deployment(table, 3),
                                "fit and validation", False)
            assert np.array_equal(outcome.stage, stored.stage)
            assert all(np.array_equal(outcome.pooled[w], stored.pooled[w]) for w in task.b.widths)
            records = fit["selection"]["knn_text"]
            assert [r["value"] for r in records] == stored.info["k"]
            assert [r["penalty"]["capped_bd"] for r in records] == [float(v) for v in stored.info["lambda"]]
            for mine, theirs in zip(records, stored.info["selection"]):
                assert mine["losses"] == theirs["losses"]
                assert mine["within_budget"]["capped_bd"] == theirs["within_budget"]
            assert [r["model"] for r in fit["reference"]] == [int(best_single.ranking(task, f, n=4)[0])
                                                             for f in range(N_FOLDS)]
            search = fit["search"]["knn_text"][cid]
            assert search["same"] == search["chance"] == task.R and 0 < search["cut"] <= task.R


def _permute_labels(bench, fold, seed):
    """The benchmark with the labels of one fold dealt out again among the fields of that fold of equal width."""
    rng = np.random.default_rng([seed, bench.cid])
    judged, gold, n_accepted = bench.judged.copy(), bench.gold.copy(), bench.n_accepted.copy()
    accepted = {}
    for w in bench.widths:
        ids = bench.idx[w]
        inside = np.flatnonzero(bench.fold[bench.field_row[ids]] == fold)
        moved = inside[rng.permutation(len(inside))]
        table = bench.acc[w].copy()
        table[inside] = bench.acc[w][moved]
        accepted[w] = table
        for column, source in ((judged, bench.judged), (gold, bench.gold), (n_accepted, bench.n_accepted)):
            column[ids[inside]] = source[ids[moved]]
    return dataclasses.replace(bench, judged=judged, gold=gold, n_accepted=n_accepted, acc=accepted)


def test_test_labels_decide_nothing(corpus):
    """Permuting the labels of a test fold changes neither the transforms, nor the parameters, nor the models."""
    settings = json.loads((Path(__file__).parents[1] / "SETTINGS.json").read_text())
    fold = 3
    with TemporaryDirectory(dir=".") as directory:
        folder = _features(corpus, Path(directory) / NAME)
        tasks = {cid: load_task(corpus, cid) for cid in POOL}
        features = {cid: knn_router.request_features(tasks[cid].b, str(folder)) for cid in POOL}
    permuted = {cid: Task(_permute_labels(tasks[cid].b, fold, 5)) for cid in POOL}
    assert any(not np.array_equal(tasks[c].b.acc[w], permuted[c].b.acc[w]) for c in POOL for w in tasks[c].b.widths)
    assert any(not np.array_equal(tasks[c].cal_stats[0][fold], permuted[c].cal_stats[0][fold]) for c in POOL)
    for pool in (POOL, POOL[:1]):                   # benchmark unknown, and one benchmark (the policies "within")
        table, fit = _fit({c: tasks[c] for c in pool}, features, settings)
        other_table, other = _fit({c: permuted[c] for c in pool}, features, settings)
        for n in (3, 4):
            assert np.array_equal(table[fold, n][0], other_table[fold, n][0])
            assert all(np.array_equal(table[fold, n][1][g], other_table[fold, n][1][g]) for g in table[fold, n][1])
        assert fit["reference"][fold] == other["reference"][fold]
        for policy, records in fit["selection"].items():
            assert records[fold] == other["selection"][policy][fold], policy
        for cid in pool:
            test = tasks[cid].b.fold == fold
            assert len(fit["plans"][cid]) == 9
            for key, models in fit["plans"][cid].items():
                assert np.array_equal(models[test], other["plans"][cid][key][test]), key
    # a fold that is not the test fold does decide: its rotation's utilities come from the permuted labels
    changed = (fold + 2) % N_FOLDS
    assert any(not np.array_equal(RU.rotations_of(tasks[c], table)[changed]["utility"],
                                  RU.rotations_of(permuted[c], table)[changed]["utility"]) for c in POOL[:1])


def _results(corpus, directory):
    """A results folder as the earlier steps leave it, with the text router that ``local_methods`` adds."""
    settings, path = _settings(directory)
    results = directory / "results"
    RX.main(["--corpus", str(corpus), "--settings", str(path), "--cids", *map(str, CIDS), "--out", str(results),
             "--only", "rows"])
    folder = _features(corpus, directory / "features" / NAME)
    text = dict(settings, knn_router=dict(settings["knn_router"], features=str(folder)))
    digest = sha256_file(path)
    document = json.loads((results / "rows.json").read_text())
    rows = [r for r in document["rows"] if r["catalog_id"] is not None]
    arrays = {}
    for cid in CIDS:
        task = load_task(corpus, cid)
        scored = metrics.score(task, knn_router.run(task, text))
        rows.append(RX._result_row(task, scored, RU.TEXT, None, digest))
        arrays[f"mass_{cid}"], arrays[f"hit_{cid}"] = scored.mass, scored.hit
    np.savez_compressed(results / "local" / f"fields-{RU.TEXT}.npz", **arrays)
    (results / "rows.json").write_text(json.dumps(dict(document, rows=RX._with_suites(rows, settings["suites"])),
                                                  sort_keys=True))
    return settings, path, results, folder


def test_report(corpus):
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path, results, folder = _results(corpus, directory)
        report = RU.run(corpus, path, results, folder, directory / "one", root=None, processes=1)
        RU.run(corpus, path, results, folder, directory / "two", root=None, processes=2)
        first = (directory / "one" / "router_unknown.json").read_bytes()
        assert first == (directory / "two" / "router_unknown.json").read_bytes()
        document = json.loads(first)
        stored = {(r["method"], r["catalog_id"]): r for r in json.loads((results / "rows.json").read_text())["rows"]}
        archives = {}
        for path_ in sorted((directory / "one" / "local").iterdir()):
            with np.load(path_, allow_pickle=False) as archive:
                archives[path_.name[len("fields-"):-len(".npz")]] = {key: archive[key] for key in archive.files}
        # a stored text router that is another policy stops the run
        with np.load(results / "local" / "fields-best_single.npz", allow_pickle=False) as archive:
            np.savez_compressed(results / "local" / f"fields-{RU.TEXT}.npz", **{k: archive[k] for k in archive.files})
        rows = json.loads((results / "rows.json").read_text())
        rows["rows"] = [r for r in rows["rows"] if r["method"] != RU.TEXT]
        rows["rows"] += [dict(r, method=RU.TEXT) for r in rows["rows"] if r["method"] == "best_single"]
        (results / "rows.json").write_text(json.dumps(rows, sort_keys=True))
        with pytest.raises(ValueError, match="does not reproduce"):
            RU.run(corpus, path, results, folder, directory / "three", root=None, processes=1)
        assert not (directory / "three").exists()

    assert document["synthetic"] is True and document["suites"] == json.loads(json.dumps(report))
    assert document["benchmarks"] == POOL and document["pool"] == settings["headline_suite"]
    assert document["representation"] == NAME
    across = [f"knn_across_{NAME}", "knn_across_numeric", f"ridge_across_{NAME}"]
    within = [f"knn_within_{NAME}", f"ridge_within_{NAME}"]
    expected = {(RU.REFERENCE, None)} | {(p, r) for p in across for r in RU.ACROSS_RULES} \
        | {(p, r) for p in within for r in RU.WITHIN_RULES}
    suites = {"main35", "main35_no_recall_only", "main35_no_largest", "main35_no_saturated"}
    assert set(document["suites"]) == suites
    cells = {(r["method"], r["variant"], r["catalog_id"], r["suite"]) for r in document["rows"]}
    assert cells == {(m, v, c, None) for m, v in expected for c in POOL} | {(m, v, None, s) for m, v in expected
                                                                           for s in suites}
    assert len(document["rows"]) == len(cells)
    assert all(set(r) == set(RESULT_ROW_KEYS) and r["calls_per_request"] == 1 for r in document["rows"])
    by_cell = {(r["method"], r["variant"], r["catalog_id"], r["suite"]): r for r in document["rows"]}

    # the checks: the pool of one benchmark is the stored text router and the stored best single model
    check = document["reproduction"]
    assert check["benchmarks"] == check["benchmarks_with_identical_fields"] == len(POOL)
    assert check["benchmarks_deviating"] == 0 and check["largest_log_loss_deviation"] == 0
    assert check["single_largest_log_loss_deviation"] <= 1e-12 and check["single_largest_accuracy_deviation"] == 0
    for cid in POOL:
        mine, theirs = by_cell[within[0], "capped", cid, None], stored[RU.TEXT, cid]
        for key in ("log_loss", "accuracy", "usd_per_1k_requests", "latency_mean_s"):
            assert mine[key] == theirs[key], (cid, key)

    # per-field archives: one per policy and rule, in the field order of the benchmark
    assert set(archives) == {RU.label(m, v) for m, v in expected}
    for (method, variant), cid in ((cell, c) for cell in expected for c in POOL):
        arrays, row = archives[RU.label(method, variant)], by_cell[method, variant, cid, None]
        assert arrays[f"mass_{cid}"].shape == arrays[f"hit_{cid}"].shape == (row["n_judged"],)
        assert abs(arrays[f"hit_{cid}"].mean() - row["accuracy"]) < 1e-12

    headline = document["suites"]["main35"]
    assert headline["benchmarks"] == len(POOL)
    assert set(headline["points"]) == {m for m, _ in expected}
    assert {"best_single", "knn_router", RU.TEXT} <= set(headline["stored"])
    pairs = {f"{RU.label(p, r)} - {b}" for p in across for r in RU.ACROSS_RULES
             for b in (RU.REFERENCE, RU.BEST, RU.TEXT)}
    pairs |= {f"{RU.REFERENCE} - {b}" for b in (RU.BEST, RU.TEXT)}
    pairs |= {f"{RU.label(within[1], r)} - {b}" for r in RU.WITHIN_RULES for b in (RU.BEST, RU.TEXT)}
    pairs |= {f"{RU.label(within[1], 'uncapped')} - {RU.label(within[0], 'uncapped')}"}
    for suite in suites:
        block = document["suites"][suite]
        assert set(block["contrasts"]) == pairs
        for name, effects in block["contrasts"].items():
            for metric in ("log_loss", "accuracy"):
                e = effects[metric]
                a, b = name.split(" - ")
                assert (e["a"], e["b"], e["n_benchmarks"]) == (a, b, block["benchmarks"])
                assert e["within_low"] <= e["within_high"] and e["over_low"] <= e["over_high"]
                assert e["benchmarks_better"] + e["benchmarks_worse"] <= block["benchmarks"]
                if b == RU.REFERENCE:           # the difference is the difference of the suite rows
                    policy, rule = a.split(".")
                    gap = block["points"][policy][rule][metric] - block["points"][RU.REFERENCE]["None"][metric]
                    assert abs(gap - e["difference"]) < 1e-12
        for key, shares in block["routing"].items():
            assert abs(sum(shares) - 1) < 1e-9 and len(shares) == document["models"], key
        for policy in across[:2] + within[:1]:
            n = block["neighbours"][policy]
            assert 0 <= n["chance_share"] <= 1 and 0 <= n["same_benchmark_share"] <= 1
            assert 0 <= n["benchmarks_all_cut"] <= n["benchmarks_with_cut"] <= block["benchmarks"]
            assert 0 <= n["requests_cut_inside_tie"] <= n["requests"]
        assert block["neighbours"][within[0]]["same_benchmark_share"] == 1
        for policy, grid in ((within[0], settings["knn_router"]["k_grid"]), (within[1], list(RU.ALPHAS))):
            summary = block["within"][policy]
            assert summary["grid"] == grid and summary["cells"] == N_FOLDS * block["benchmarks"]
            assert sum(summary["value_chosen"]) == sum(summary["chosen"]) == sum(summary["chosen_uncapped"]) \
                == summary["cells"]

    # one choice per rotation for all benchmarks
    assert set(document["selection"]) == set(across)
    for policy, records in document["selection"].items():
        grid = document["grids"]["alpha" if policy.startswith("ridge") else "k"]
        assert len(records) == N_FOLDS
        for record in records:
            assert record["value"] in grid and len(record["grid_losses"]) == len(grid)
            assert record["grid_losses"][grid.index(record["value"])] == min(record["grid_losses"])
            assert set(record["penalty"]) == set(RU.ACROSS_RULES)
            assert all(v in document["grids"]["penalty"] for v in record["penalty"].values())
            free = document["grids"]["penalty"].index(record["penalty"]["uncapped"])
            assert record["losses"][free] == min(record["losses"])
            for rule in ("capped_single", "capped_bd"):
                i = document["grids"]["penalty"].index(record["penalty"][rule])
                if record["within_budget"][rule]:
                    assert record["costs"][i] <= record["budgets"][rule]
                else:
                    assert record["costs"][i] == min(record["costs"])
    assert len(document["reference"]) == N_FOLDS
    assert all(0 <= r["model"] < document["models"] for r in document["reference"])
    for key in ("fit_folds", "training_folds"):
        table = np.array(document["recalibration"][key])
        assert table.shape == (N_FOLDS, document["models"]) and table.min() >= 0
        assert table.max() < len(document["recalibration"]["menu"])
    assert set(document["routing_by_benchmark"]) == {str(c) for c in POOL}
    assert document["features"]["dimensions"] == 8
    assert all(0 < document["features"]["distinct_vectors"][str(c)] for c in POOL)

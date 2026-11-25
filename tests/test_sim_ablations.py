"""Checks of training labels, transfer sources, and calibration order in the ablations."""

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from analysis import run_experiments as runner
from data.load import list_benchmarks
from data.schema import N_FOLDS, roles
from sim.baselines import transfer_single, transfer_temperature
from sim.engine import Task, load_task


def _settings():
    """Return experiment settings from the package."""
    return json.loads((Path(__file__).parents[1] / "SETTINGS.json").read_text())


def _change_other_labels(task, groups):
    """Return a new task with the accepted options outside the sample permuted by width."""
    b = task.b
    acc, judged, gold, counts = {}, b.judged.copy(), b.gold.copy(), b.n_accepted.copy()
    changed = False
    for w, ids in b.idx.items():
        sel = np.flatnonzero(~np.isin(task.field_group[ids], groups))
        source = np.roll(sel, 1)
        acc[w] = b.acc[w].copy()
        acc[w][sel] = b.acc[w][source]
        changed |= not np.array_equal(acc[w], b.acc[w])
        for target, original in ((judged, b.judged), (gold, b.gold), (counts, b.n_accepted)):
            target[ids[sel]] = original[ids[source]]
    assert changed
    return Task(replace(b, acc=acc, judged=judged, gold=gold, n_accepted=counts))


def test_budget_consumes_only_sampled_labels(corpus):
    settings = _settings()["label_budget"]
    task = load_task(corpus, 31)
    for fold in range(N_FOLDS):
        available = np.flatnonzero(task.group_fold != fold)
        rng = np.random.default_rng([settings["seed"], task.cid, fold, 0])
        sample = rng.permutation(available)[:min(settings["groups"])]
        before = runner._budget_fit(task, fold, sample)
        after = runner._budget_fit(_change_other_labels(task, sample), fold, sample)
        for left, right in zip(before, after):
            assert np.array_equal(left, right)


def test_budget_all_train4(corpus, reference):
    from ensemble_round2 import anchors, fits

    cid = 50
    task = load_task(corpus, cid)
    source = reference.task(cid)
    for fold in range(N_FOLDS):
        available = np.flatnonzero(task.group_fold != fold)
        model, weights, choices = runner._budget_fit(task, fold, available)
        assert model == anchors.ranking(source, fold, "calibrated", n=4)[0][0]
        assert np.array_equal(choices, source.recal(fold, n=4)[0])
        rows = source.rows(roles(fold)[2])
        assert np.array_equal(weights, fits.simplex_fit(source.view(fold, "calibrated", 4)[rows]))


def test_transfer_sources_and_equal_benchmark_means(corpus, reference):
    from ensemble_round2 import calibration, fits

    settings = _settings()
    target = 20
    ids = [2, 9, 20, 21, 31, 59]
    available = [b for b in list_benchmarks(corpus) if b["catalog_id"] in ids]
    area = reference.AREA_OF[target]
    sources = [c for c in ids if reference.AREA_OF[c] != area]

    def check_reads(function, requested):
        assert list(requested) == sources
        return map(function, requested)

    observed = runner._transfer_choices(corpus, available, [target], settings, check_reads)
    models, temperatures = [], []
    for cid in sources:
        source = reference.task(cid)
        models.append(reference.loss_of_mass(source.mass).mean(axis=0))
        _, sums, counts = calibration.flat_tables(cid)
        temperatures.append(sums.sum(axis=1) / counts.sum())
    model_mean = [math.fsum(column) / len(sources) for column in zip(*models)]
    temp_mean = [math.fsum(column) / len(sources) for column in zip(*temperatures)]
    expected = {"transfer_single": int(np.argmin(model_mean)),
                "transfer_temperature": fits.TEMPERATURES[fits.best_temperature(temp_mean)]}
    assert observed == {area: expected}


def test_calibration_order_against_equal_weight_control(corpus, reference):
    from ensemble_round2 import calibration, fits

    settings = _settings()
    for cid in (2, 31, 59):
        task = load_task(corpus, cid)
        blocks = task.fold_blocks(range(N_FOLDS), judged_only=True)
        outcomes = dict(runner._calibration_order(task, settings))

        def mass(name):
            pooled = outcomes[name].pooled
            return task.mass_of_blocks({w: pooled[w][sel] for w, sel in blocks.items()}, blocks)

        raw, sums, counts = calibration.flat_tables(cid)
        assert np.array_equal(mass("none"), raw[fits.TEMPERATURES.index(1.0)])
        temperatures = []
        for f in range(N_FOLDS):
            fit = list(roles(f)[0])
            temperatures.append(fits.best_temperature(sums[:, fit].sum(axis=1) / counts[fit].sum()))
        expected = raw[np.asarray(temperatures)[task.fold], np.arange(task.nJ)]
        assert np.array_equal(mass("final_temperature"), expected)
        plain, final, chosen, _ = calibration.stack_control(cid, mode="equal")
        assert np.array_equal(mass("calibrated_members"), plain)
        assert np.array_equal(mass("calibrated_members_and_final_temperature"), final)
        assert outcomes["calibrated_members_and_final_temperature"].info["temperature"] == chosen


def test_transfer_suite_matches_reference_and_isolates_target_area(corpus, reference):
    from ensemble_round2 import calibration, fits

    settings = _settings()
    tasks = {c: load_task(corpus, c) for c in (2, 20, 21, 31, 59)}
    sources = [c for c in tasks if tasks[c].area != tasks[20].area]
    models, temperatures = [], []
    for c in sources:
        source = reference.task(c)
        models.append(reference.loss_of_mass(source.mass).mean(axis=0))
        _, sums, counts = calibration.flat_tables(c)
        temperatures.append(sums.sum(axis=1) / counts.sum())
    model = int(np.argmin([math.fsum(column) / len(sources) for column in zip(*models)]))
    temp = fits.TEMPERATURES[fits.best_temperature(
        [math.fsum(column) / len(sources) for column in zip(*temperatures)])]
    changed = {c: t if c in sources else _change_other_labels(t, []) for c, t in tasks.items()}
    for module, choice_name, expected in ((transfer_single, "model", model),
                                          (transfer_temperature, "temperature", temp)):
        before = module.run_suite(tasks, settings)
        after = module.run_suite(changed, settings)
        assert set(before) == set(after) == set(tasks)
        for cid in (20, 21):
            assert before[cid].ledger["other_benchmark_labels"]
            assert before[cid].ledger["labeled_folds"] == "none"
            assert before[cid].info[choice_name] == after[cid].info[choice_name] == [expected] * N_FOLDS
            source = reference.task(cid)
            if module is transfer_single:
                weights = np.tile(np.eye(reference.M)[model], (N_FOLDS, 1))
            else:
                weights = np.full((N_FOLDS, reference.M), 1.0 / reference.M)
            reference_pool = source.stitch(weights)
            assert np.array_equal(before[cid].stage, np.where(weights[source.b.fold] > 0, 0, -1))
            for w, vectors in reference_pool.items():
                if module is transfer_temperature:
                    vectors = reference.temper(vectors, temp)
                assert np.array_equal(vectors, before[cid].pooled[w])
                assert np.array_equal(vectors, after[cid].pooled[w])

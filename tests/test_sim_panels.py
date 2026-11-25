"""Reference parity for every benchmark and checks of the fitted panel plans."""

import json
from pathlib import Path

import numpy as np

from data.schema import N_FOLDS, roles
from sim import core
from sim.baselines import (
    best_single, greedy, skill_pool, stack, stack_final_temperature, uniform_mean, uniform_temperature,
)
from sim.engine import load_task
from sim.metrics import score


def _settings():
    """Return the package experiment settings."""
    return json.loads((Path(__file__).parents[1] / "SETTINGS.json").read_text())


def _check_pool(task, source, method, settings, weights, choices=None, temperatures=None):
    """Check exact weights, plans, vectors, and decoded masses against the reference."""
    outcome = method.run(task, settings)
    context = (task.cid, method.NAME)
    assert np.array_equal(weights, outcome.info["weights"]), context
    assert np.array_equal(outcome.stage, np.where(weights[task.b.fold] > 0, 0, -1)), context
    pooled = source.stitch(weights, choices)
    if temperatures is not None:
        assert np.array_equal(temperatures, outcome.info["temperature"]), context
        from ensemble_round2.protocol import temper
        for w, block in pooled.items():
            for f in range(N_FOLDS):
                keep = source.field_fold[source.b.idx[w]] == f
                block[keep] = temper(block[keep], temperatures[f])
    for w in pooled:
        assert np.array_equal(outcome.pooled[w], pooled[w]), (context, w)
    hit, mass, _ = source.decode(pooled)
    scored = score(task, outcome)
    assert np.array_equal(scored.mass, mass[source.J]), context
    assert np.array_equal(scored.hit, hit[source.J]), context
    return outcome


def _unrounded_mass(task, outcome):
    """Return mass directly from the pooled vectors before the scoring renormalisation."""
    blocks = task.fold_blocks(range(N_FOLDS), judged_only=True)
    vectors = {w: outcome.pooled[w][sel] for w, sel in blocks.items()}
    return task.mass_of_blocks(vectors, blocks)


def test_reference_panels(corpus, reference):
    from ensemble_round2 import anchors, calibration, fits, reliability

    settings = _settings()
    for cid in reference.CIDS:
        task = load_task(corpus, cid)
        source = reference.task(cid)
        flat = np.full((N_FOLDS, reference.M), 1.0 / reference.M)
        choices = np.stack([source.recal(f)[0] for f in range(N_FOLDS)])
        choices4 = np.stack([source.recal(f, n=4)[0] for f in range(N_FOLDS)])
        single = [anchors.ranking(source, f, "calibrated", n=4)[0][0] for f in range(N_FOLDS)]
        _check_pool(task, source, best_single, settings, np.eye(reference.M)[single], choices4)
        _check_pool(task, source, uniform_mean, settings, flat)

        vote = reliability.all_votes(source)
        fit_weights = []
        for f in range(N_FOLDS):
            train = source.field_fold != f
            skills = reliability.em_skills(vote[train], source.b.width[train], **reliability.EM)[0]
            fit_weights.append(reliability.pool_weights(skills))
        _check_pool(task, source, skill_pool, settings, np.stack(fit_weights))

        masses, sums, counts = calibration.flat_tables(cid)
        actual_table = uniform_temperature.flat_tables(task, settings)
        for expected, actual in zip((masses, sums, counts), actual_table):
            assert np.array_equal(expected, actual), cid
        chosen = []
        for f in range(N_FOLDS):
            fit = list(roles(f)[0])
            index = fits.best_temperature(sums[:, fit].sum(axis=1) / counts[fit].sum())
            chosen.append(fits.TEMPERATURES[index])
        outcome = _check_pool(task, source, uniform_temperature, settings, flat, temperatures=chosen)
        indices = np.array([fits.TEMPERATURES.index(t) for t in chosen])
        assert np.array_equal(_unrounded_mass(task, outcome),
                              masses[indices[task.fold], np.arange(task.nJ)]), cid

        mass, final_mass, temperatures, weights = calibration.stack_control(cid)
        outcome = _check_pool(task, source, stack, settings, weights, choices)
        assert np.array_equal(_unrounded_mass(task, outcome), mass), cid
        final = _check_pool(task, source, stack_final_temperature, settings, weights, choices, temperatures)
        assert np.array_equal(_unrounded_mass(task, final), final_mass), cid


def test_greedy_prefix_and_plan(corpus):
    settings = _settings()
    task = load_task(corpus, 50)
    outcome = greedy.run(task, settings)
    for f in range(N_FOLDS):
        prefixes = greedy.path(task, f)
        mass = task.view(f, "calibrated")[task.rows(roles(f)[0])]
        winner = core.loss_of_mass(mass).mean(axis=0).argmin()
        assert np.array_equal(prefixes[0], np.eye(task.b.M)[winner])
        selected = prefixes[outcome.info["size"][f] - 1]
        assert np.array_equal(selected, outcome.info["weights"][f])
        assert np.all(outcome.stage[task.b.fold == f] == np.where(selected > 0, 0, -1))


def test_planted_single(corpus, planted):
    settings = _settings()
    specialist = planted["specialist"]
    chosen = []
    for cid in planted["areas"][specialist["area"]]:
        task = load_task(corpus, cid)
        chosen.extend(best_single.run(task, settings).info["model"])
    assert np.mean(np.asarray(chosen) == specialist["model"]) > 0.5

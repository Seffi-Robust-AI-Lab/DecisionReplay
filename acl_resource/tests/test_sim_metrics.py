"""Parity of scoring, call telemetry, and equal-benchmark means."""

import math

import numpy as np
import pytest

from acl_resource.sim import core
from acl_resource.sim.engine import load_task
from acl_resource.sim.metrics import score, suite_rows, summarize
from acl_resource.sim.types import NOT_CALLED, Outcome


@pytest.mark.parametrize("cid", [2, 9, 31, 36, 38, 50, 56, 59])
def test_scores_and_costs(corpus, reference, cid):
    from cluster_confidence_check.simulate import costs
    from ensemble_round2 import rows

    task = load_task(corpus, cid)
    expected = reference.Task(cid)
    rng = np.random.default_rng(cid)
    weights = rng.random((5, task.b.M))
    weights[weights < 0.5] = 0
    called = weights[task.b.fold] > 0
    pooled = task.stitch(weights)
    scored = score(task, Outcome(pooled, np.where(called, 0, NOT_CALLED)))
    ref = rows.score(expected, expected.stitch(weights), called, with_native=False)
    assert np.array_equal(scored.mass, ref.mass)
    assert np.array_equal(scored.hit, ref.hit)
    summary = summarize(scored)
    assert summary["accuracy"] == ref.accuracy
    assert summary["brier"] == core.brier(task.b, pooled)[task.J].mean()
    assert summary["ece"] == core.ece(scored.confidence, scored.hit)
    for key, floor in (("log_loss", 1e-15), ("log_loss_floor_1e-6", 1e-6), ("log_loss_floor_1e-3", 1e-3)):
        assert summary[key] == reference.loss_of_mass(ref.mass, floor).mean()
    reference_cost = costs(expected.b, called)
    for key in ("calls", "calls_per_request", "input_tokens_per_request", "output_tokens_per_request",
                "usd_per_1k_requests"):
        assert scored.cost[key] == reference_cost[key]
    assert scored.cost["cost_usd"] == reference_cost["cost_usd_sum"]
    assert scored.cost["cost_usd"] == math.fsum(task.b.cost[called])
    for measure in ("mean", "p50", "p95"):
        assert summary[f"latency_{measure}_s"] == reference_cost[f"latency_parallel_{measure}_s"]

    # Each called model has its own stage, so the request latencies add.
    stages = np.where(called, np.cumsum(called, axis=1) - 1, NOT_CALLED)
    cascade = score(task, Outcome(pooled, stages))
    elapsed = np.array([sum(task.b.lat[r, called[r]].tolist()) for r in range(task.R)])
    assert np.isclose(cascade.cost["latency_mean_s"], elapsed.mean(), rtol=1e-15, atol=0)
    assert cascade.cost["cost_usd"] == scored.cost["cost_usd"]


def test_suite_membership_and_weight():
    class Rows(dict):
        def __getitem__(self, cid):
            assert cid != 59
            return super().__getitem__(cid)

    inputs = Rows({2: dict(accuracy=0.25, n_judged=1, n_clipped=0),
                   9: dict(accuracy=0.75, n_judged=100, n_clipped=2), 59: None})
    row, = suite_rows(inputs, {"main35": {"exclude": [59]}})
    assert row["accuracy"] == 0.5
    assert row["n_judged"] == 101
    assert row["n_clipped"] == 2
    assert row["n_benchmarks"] == 2

"""A pool balanced by organisation and the pair groups, on the made-up corpus."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from analysis import pool_composition as PC
from sim import metrics
from sim.baselines import uniform_mean
from sim.engine import load_task

from .test_restyle_analyses import CIDS, _results


def test_weights_balance_organisations():
    models = [dict(organisation=o) for o in ("a", "a", "a", "b", "c", "c")]
    weights = PC.organisation_weights(models)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[:3].sum() == pytest.approx(1 / 3) and weights[3] == pytest.approx(1 / 3)
    # one model per organisation is the equal-weight pool
    assert np.allclose(PC.organisation_weights([dict(organisation=str(i)) for i in range(5)]), 0.2)


def test_pool_composition_report(corpus):
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path, out = _results(corpus, directory, ("rows",))
        target = directory / "followup"
        suites = PC.run(corpus, path, out, target, processes=1)
        document = json.loads((target / "pool_composition.json").read_text())
        assert document["suites"].keys() == suites.keys() and document["note"]
        task = load_task(corpus, CIDS[0])
        weights = PC.organisation_weights(task.b.models)
        scored = metrics.score(task, uniform_mean.panel(task, np.tile(weights, (5, 1))))
        one = metrics.score(task, PC.top_k_pool(task, [1] * 5))
        assert one.cost["calls_per_request"] == pytest.approx(1.0)
        validated = PC.top_k_pool(task)
        assert all(1 <= k <= task.b.M for k in validated.info["size"])
        for block in suites.values():
            assert {PC.RAW, PC.TEMPERED, PC.TOP_K, "uniform_mean", "best_single"} <= block["points"].keys()
            assert 1 <= block["top_k_sizes"]["mean"] <= task.b.M
            assert sum(block["top_k_sizes"]["cells_by_size"].values()) == block["top_k_sizes"]["cells"]
            assert block["points"][PC.RAW]["calls_per_request"] == pytest.approx(task.b.M)
            # the temperature keeps every decoded answer, so the two balanced pools are equally accurate
            assert block["points"][PC.RAW]["accuracy"] == pytest.approx(block["points"][PC.TEMPERED]["accuracy"])
            for contrast in block["contrasts"].values():
                for metric in PC.METRICS:
                    cell = contrast[metric]
                    assert cell["within_low"] <= cell["difference"] + 1e-12 <= cell["within_high"] + 2e-12 \
                        or cell["within_low"] == cell["within_high"]
        assert 0 <= scored.hit.mean() <= 1


def test_pair_groups(tmp_path):
    models = [dict(model_index=str(i), model_code=f"M{i:02d}", organisation=o, base_model=b)
              for i, (o, b) in enumerate((("x", "Q-4B"), ("x", ""), ("y", "Q-4B-Base"), ("z", "R-9B"), ("w", "")))]
    path = tmp_path / "pairs.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["catalog_id", "model_a", "model_b", "error_correlation", "error_correlation_given_others"])
        for cid in (1, 2):
            for a in range(5):
                for b in range(a + 1, 5):
                    writer.writerow([cid, a, b, 0.1 * (a + b), "" if (a, b) == (3, 4) else 0.01 * (a + b)])
    out = PC.error_correlation(path, models, [1, 2])
    assert out["same_organisation"]["pairs"] == 1
    assert out["other_organisation_same_base"]["pairs"] == 1 and list(out["shared_base_pairs"]) == ["M00+M02"]
    assert out["other_organisation_other_base"]["pairs"] == 8
    assert out["shared_base_pairs"]["M00+M02"]["error_correlation"] == pytest.approx(0.2)
    assert out["same_organisation"]["error_correlation_given_others"] == pytest.approx(0.01)
    assert PC.base_family("Q-4B-Base") == "Q-4B" and PC.base_family("") == ""

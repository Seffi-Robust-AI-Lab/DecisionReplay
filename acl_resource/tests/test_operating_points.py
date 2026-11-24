"""The report on operating points: the capped rule is the main-table policy, the uncapped rule ignores cost."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from acl_resource.analysis import operating_points
from acl_resource.analysis.run_experiments import main
from acl_resource.data.schema import N_FOLDS
from acl_resource.sim.baselines import confidence_cascade, knn_router
from acl_resource.sim.engine import load_task

from .test_sim_runner import _settings


def test_operating_points(corpus):
    cids = [2, 59]
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path = _settings(directory)
        out = directory / "out"
        main(["--corpus", str(corpus), "--settings", str(path), "--cids", *map(str, cids), "--out", str(out),
              "--only", "rows"])
        suites = operating_points.run(corpus, path, out, root=None, processes=1)
        document = json.loads((out / "operating_points.json").read_text())
        main_rows = {(r["method"], r["catalog_id"], r["suite"]): r
                     for r in json.loads((out / "rows.json").read_text())["rows"]}
    assert document["synthetic"] is True and document["suites"] == json.loads(json.dumps(suites))
    points = {(r["method"], r["variant"], r["catalog_id"], r["suite"]): r for r in document["rows"]}
    # the capped rule is the policy of the main table
    for name in ("knn_router", "confidence_cascade"):
        for cid in cids:
            mine, theirs = points[name, "capped", cid, None], main_rows[name, cid, None]
            for key in ("log_loss", "accuracy", "usd_per_1k_requests", "calls_per_request", "latency_mean_s"):
                assert mine[key] == theirs[key], (name, cid, key)
    assert "knn_router_text" not in {r["method"] for r in document["rows"]}     # no local features without a root
    headline = suites[settings["headline_suite"]]
    members = [c for c in cids if c not in settings["suites"][settings["headline_suite"]]["exclude"]]
    for name, block in headline["selection"].items():
        assert block["cells"] == N_FOLDS * len(members)
        assert sum(block["chosen"]) == sum(block["chosen_uncapped"]) == block["cells"]
        assert 0 <= block["none_within_budget"] <= block["cap_changed_choice"] + block["cells"]
    assert sum(headline["features"]["benchmarks_by_varying_features"]) == len(members)
    # a contrast of a policy with itself under the same rule does not exist; router against the reference does
    for name in ("router_capped_vs_single", "cascade_uncapped_vs_single", "mean_cascade_vs_cascade"):
        effect = headline["contrasts"][name]["log_loss"]
        assert effect["ci_low"] <= effect["difference"] <= effect["ci_high"]
    assert not any(name.startswith("text_router") for name in headline["contrasts"])


def test_rules_per_rotation(corpus):
    """The uncapped rule takes the lowest validation loss; a parameter list is applied fold by fold."""
    settings = json.loads((Path(__file__).parents[1] / "SETTINGS.json").read_text())
    task = load_task(corpus, 2)
    for run, parameter, key in ((knn_router.run, "penalty", "lambda"), (confidence_cascade.run, "threshold", "tau")):
        capped = run(task, settings)
        records = capped.info["selection"]
        assert len(records) == N_FOLDS
        free = [operating_points.lowest_loss(r) for r in records]
        for f, record in enumerate(records):
            i, j = record["grid"].index(capped.info[key][f]), record["grid"].index(free[f])
            assert record["losses"][j] == min(record["losses"])
            assert record["losses"][j] <= record["losses"][i]
            if record["within_budget"]:
                assert record["costs"][i] <= record["budget"]
            else:
                assert record["costs"][i] == min(record["costs"])
        outcome = run(task, settings, **{parameter: free})
        assert [float(v) for v in outcome.info[key]] == free
        fixed = [run(task, settings, **{parameter: v}) for v in sorted(set(free))]
        for f in range(N_FOLDS):
            same = fixed[sorted(set(free)).index(free[f])]
            rows = task.b.fold == f
            assert np.array_equal(outcome.stage[rows], same.stage[rows])


def test_mean_confidence(corpus):
    """The mean of the top probabilities is never below their minimum, so the mean cascade stops no later."""
    settings = json.loads((Path(__file__).parents[1] / "SETTINGS.json").read_text())
    task = load_task(corpus, 2)
    active = np.ones(task.R, bool)
    choice = task.recal(0)[0]
    low = confidence_cascade.request_confidence(task, active, 0, choice[0])
    mean = confidence_cascade.request_confidence(task, active, 0, choice[0], "mean")
    assert (mean >= low - 1e-12).all() and (mean <= 1 + 1e-12).all()
    fields = np.bincount(task.b.field_row, minlength=task.R)
    assert np.allclose(mean[fields == 1], low[fields == 1], rtol=0, atol=1e-15)
    mean_settings = dict(settings, confidence_cascade=dict(settings["confidence_cascade"], confidence="mean"))
    a = confidence_cascade.run(task, settings, threshold=0.9)
    b = confidence_cascade.run(task, mean_settings, threshold=0.9)
    assert ((b.stage >= 0).sum(axis=1) <= (a.stage >= 0).sum(axis=1)).all()

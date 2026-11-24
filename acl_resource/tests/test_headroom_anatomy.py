"""What the oracle's lead consists of, on the made-up corpus."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from acl_resource.analysis import headroom_anatomy
from acl_resource.sim.engine import load_task

from .test_restyle_analyses import CIDS, _results


def test_headroom_anatomy_report(corpus):
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path, out = _results(corpus, directory, ("rows",))
        target = directory / "followup"
        suites = headroom_anatomy.run(corpus, path, out, target, processes=1)
        document = json.loads((target / "headroom_anatomy.json").read_text())
        assert document["suites"].keys() == suites.keys() and document["taus"] == list(headroom_anatomy.TAUS)
        rows = json.loads((out / "rows.json").read_text())["rows"]
        stored = {(r["method"], r["suite"]): r["accuracy"] for r in rows
                  if r["catalog_id"] is None and r["variant"] is None}
        for suite, block in suites.items():
            lead = block["lead"]
            # the lead is the one of the result rows
            assert lead["oracle"] == pytest.approx(stored["oracle", suite], abs=1e-12)
            assert lead["reference"] == pytest.approx(stored["best_single", suite], abs=1e-12)
            assert lead["points"] == pytest.approx(100 * (lead["oracle"] - lead["reference"]), abs=1e-9)
            # a request-level oracle cannot beat "some model is right on the field"; more witnesses, fewer fields
            shares = [block["ceilings"][f"at_least_{k}"]["share"] for k in headroom_anatomy.WITNESSES]
            assert shares[0] >= lead["oracle"] - 1e-12 and shares == sorted(shares, reverse=True)
            groups = block["by_options"].values()
            assert sum(g["share_of_fields"] for g in groups) == pytest.approx(1.0, abs=1e-9)
            assert sum(g["lead_points"] for g in groups) == pytest.approx(lead["points"], abs=1e-9)
            # a stricter confidence threshold cannot add right answers; the reference's answers always stand
            points = [block["confident"][str(t)]["points_over_reference"] for t in headroom_anatomy.TAUS]
            assert points == sorted(points, reverse=True) and min(points) >= -1e-12
            assert all(block["confident"][str(t)]["accuracy"] <= shares[0] + 1e-12 for t in headroom_anatomy.TAUS)
            guessing = block["guessing"]
            assert 0 <= guessing["another_model_right"] <= 1 and 0 < guessing["uniform_guessing"] <= 1
            assert guessing["reference_wrong_share"] == pytest.approx(1 - lead["reference"], abs=1e-9)
            few = block["few_right"]
            if few["share_of_fields"]:
                low, high = 1 / block["models"], headroom_anatomy.FEW_RIGHT / block["models"]
                assert low - 1e-12 <= few["random_pick_right"] <= high + 1e-12
            assert 0 <= block["consensus"]["share_of_fields"] <= 1
            split = block["consensus"]["by_options"]
            assert sum(g["lead_points"] for g in split.values()) == pytest.approx(
                block["consensus"]["lead_points"], abs=1e-9)
            # with two options, "one to three right" and "all others agree on the other option" are one event
            if split["2"]["share_of_few_right_fields"] is not None:
                assert split["2"]["share_of_few_right_fields"] == pytest.approx(1.0)


def test_consensus_counts_one_wrong_option(corpus):
    task = load_task(corpus, CIDS[0])
    confidence, option, hits = headroom_anatomy.calibrated_decisions(task)
    assert confidence.shape == option.shape == hits.shape == (task.b.M, task.nJ)
    assert np.all((confidence > 0) & (confidence <= 1 + 1e-12))
    agreed = headroom_anatomy.wrong_consensus(task, option, hits)
    wrong = (~hits).sum(axis=0)
    # no more models can share a wrong option than are wrong; with two options every wrong model shares it
    assert np.all(agreed <= wrong)
    two = task.width == 2
    assert np.array_equal(agreed[two], wrong[two])

"""The learned router against the control, on the made-up corpus with made-up request features."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from acl_resource.analysis import router_control_contrast as RC
from acl_resource.analysis import router_study
from acl_resource.data.schema import N_FOLDS

from .test_restyle_analyses import CIDS, _results
from .test_router_unknown import _features


def test_router_control_contrast_report(corpus):
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path, out = _results(corpus, directory, ("rows",))
        features = _features(corpus, directory / "features" / "minilm")
        router_study.run(corpus, path, out, str(features.parent), processes=1)
        target = directory / "followup"
        suites = RC.run(corpus, path, out, features, target, processes=1)
        document = json.loads((target / "router_control_contrast.json").read_text())
        assert document["representation"] == "minilm" and document["suites"].keys() == suites.keys()
        for block in suites.values():
            assert block["benchmarks"] <= len(CIDS)
            for rule in RC.RULES:
                cell = block["ridge_minus_control"][rule]
                assert cell["cells"] == N_FOLDS * block["benchmarks"]
                assert 0 <= cell["cells_with_one_model"] <= cell["cells"]
                assert 0 <= cell["requests_sent_to_another_model"] <= 1
                for metric in RC.METRICS:
                    c = cell[metric]
                    assert c["within_low"] <= c["within_high"] and c["over_low"] <= c["over_high"]
                    assert c["benchmarks_better"] + c["benchmarks_worse"] <= block["benchmarks"]


def test_a_changed_control_stops_the_run(corpus):
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path, out = _results(corpus, directory, ("rows",))
        features = _features(corpus, directory / "features" / "minilm")
        router_study.run(corpus, path, out, str(features.parent), processes=1)
        study = json.loads((out / "router_study.json").read_text())
        for row in study["rows"]:
            if row["method"] == "router_control" and row["catalog_id"] is not None:
                row["log_loss"] += 0.5
        (out / "router_study.json").write_text(json.dumps(study))
        with pytest.raises(ValueError):
            RC.run(corpus, path, out, features, directory / "followup", processes=1)

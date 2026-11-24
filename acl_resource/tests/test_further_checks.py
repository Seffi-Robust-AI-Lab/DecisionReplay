"""Redrawn folds and the open track on the made-up corpus."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from acl_resource.analysis import fold_redraws, open_track
from acl_resource.analysis.run_experiments import main
from acl_resource.data import build_corpus
from acl_resource.data.load import load_benchmark
from acl_resource.data.rehydrate import rehydrate
from acl_resource.data.schema import N_FOLDS

from .test_sim_runner import _settings

ROWS = "data/decision_index/common-paid-65/inputs"
SOURCE = "results/decision_index/paper/dataset/candidate"


def test_redraw_keeps_groups_together(corpus):
    bench = load_benchmark(corpus, 2, labels=True)
    first, again, other = (fold_redraws.redraw(bench, seed) for seed in (1, 1, 2))
    assert np.array_equal(first.fold, again.fold) and not np.array_equal(first.fold, other.fold)
    assert not np.array_equal(first.fold, bench.fold)
    for group in np.unique(bench.split_group):
        assert len(np.unique(first.fold[bench.split_group == group])) == 1
    sizes = np.bincount([first.fold[bench.split_group == g][0] for g in np.unique(bench.split_group)],
                        minlength=N_FOLDS)
    assert sizes.max() - sizes.min() <= 1


def test_fold_redraws_report(corpus):
    cids = [2, 59]
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path = _settings(directory)
        out = directory / "out"
        main(["--corpus", str(corpus), "--settings", str(path), "--cids", *map(str, cids), "--out", str(out),
              "--only", "rows"])
        suites = fold_redraws.run(corpus, path, out, seeds=(1, 2), processes=1)
        document = json.loads((out / "fold_redraws.json").read_text())
    assert document["draws"] == 2 and document["synthetic"] is True
    block = suites[settings["headline_suite"]]
    for name, by_metric in block["methods"].items():
        loss = by_metric["log_loss"]
        assert len(loss["draws"]) == 2 and loss["low"] <= loss["mean"] <= loss["high"]
    unfitted = block["methods"]["uniform_mean"]["log_loss"]           # no fold enters an equal-weight pool
    assert unfitted["low"] == unfitted["high"] == unfitted["frozen"]
    assert block["contrasts_keeping_sign"] <= block["contrasts_compared"] == len(block["contrasts"])


def test_open_track(test_root):
    if not (test_root / SOURCE).is_dir():
        import pytest
        pytest.skip("no source export beside the test root")
    settings = Path(__file__).parents[1] / "SETTINGS.json"
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        build_corpus.build(test_root / SOURCE, test_root / ROWS, test_root / "meta", directory / "open", tier="open",
                           clearance=test_root / "meta/clearance.csv", names="coded")
        rehydrate(directory / "open", test_root / ROWS, directory / "labelled")
        document = open_track.run(directory / "labelled", settings, directory / "open_track.json")
        assert json.loads((directory / "open_track.json").read_text()) == json.loads(json.dumps(document))
    assert document["tier"] == "open" and document["synthetic"] is True and document["models"] >= 1
    for block in document["suites"].values():
        for name in open_track.METHODS:
            assert 0 <= block["methods"][name]["accuracy"] <= 1 and block["methods"][name]["log_loss"] >= 0
        assert block["methods"]["oracle"]["accuracy"] >= block["methods"]["best_single"]["accuracy"] - 1e-12

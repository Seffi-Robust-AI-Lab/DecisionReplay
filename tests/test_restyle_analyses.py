"""Field hardness, the router study and the benchmark metrics on the made-up corpus."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from analysis import benchmark_metrics, diagnostics, hardness, router_study
from analysis.run_experiments import main
from data.load import list_benchmarks

from .test_sim_runner import _settings

CIDS = [2, 59]


def _results(corpus, directory, sections=("rows", "frontier")):
    settings, path = _settings(directory)
    out = directory / "out"
    for section in sections:
        main(["--corpus", str(corpus), "--settings", str(path), "--cids", *map(str, CIDS), "--out", str(out),
              "--only", section])
    return settings, path, out


def test_hardness_report(corpus):
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path, out = _results(corpus, directory, ("rows",))
        suites = hardness.run(corpus, path, out, processes=1, draws=7)
        rows = json.loads((out / "rows.json").read_text())["rows"]
        # a misaligned archive stops the run
        archive = out / "local" / "fields-best_single.npz"
        with np.load(archive) as data:
            arrays = {k: data[k] for k in data.files}
        arrays[f"hit_{CIDS[0]}"] = arrays[f"hit_{CIDS[0]}"][::-1].copy()
        np.savez_compressed(archive, **arrays)
        with pytest.raises(ValueError):
            hardness.run(corpus, path, out, processes=1, draws=7)
    for suite, block in suites.items():
        assert abs(sum(b["share"] for b in block["bins"]) - 1) < 1e-12
        for b in block["bins"]:
            for value in b["accuracy"].values():
                assert value is None or 0 <= value <= 1
        if block["none_right"]["fields"]:
            for name in ("best_single", "random_single", "oracle", "knn_router", "confidence_cascade"):
                assert block["none_right"]["accuracy"][name] == 0
        subsets = block["subsets"]
        oracle = next(r for r in rows if r["method"] == "oracle" and r["suite"] == suite and r["catalog_id"] is None)
        assert abs(subsets["top_k"][-1] - oracle["accuracy"]) < 1e-12
        assert abs(subsets["random_mean"][-1] - oracle["accuracy"]) < 1e-12
        assert all(a <= b + 1e-15 for a, b in zip(subsets["top_k"], subsets["top_k"][1:]))
        assert all(a <= b + 1e-15 for a, b in zip(subsets["random_mean"], subsets["random_mean"][1:]))
        assert all(lo - 1e-12 <= m <= hi + 1e-12 for lo, m, hi in zip(subsets["random_low"], subsets["random_mean"],
                                                                         subsets["random_high"]))


def _fake_features(corpus, folder):
    """A made-up feature folder: one finite float32 matrix per benchmark and a features.json."""
    folder.mkdir(parents=True)
    for b in list_benchmarks(corpus):
        x = np.random.default_rng(b["catalog_id"]).normal(size=(int(b["requests"]), 4)).astype(np.float32)
        np.save(folder / f"catalog-{b['catalog_id']:03d}.npy", x)
    (folder / "features.json").write_text(json.dumps(dict(model="made-up", reduction="none")))


def test_router_study(corpus):
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path, out = _results(corpus, directory, ("rows",))
        _fake_features(corpus, directory / "features" / "madeup")
        suites = router_study.run(corpus, path, out, directory / "features", processes=1)
        hard = hardness.run(corpus, path, out, processes=1, draws=3)
        document = json.loads((out / "router_study.json").read_text())
    assert document["representations"] == ["numeric", "numeric_all", "madeup"]
    headline = settings["headline_suite"]
    recall = suites[headline]["recall"]
    for name in ("numeric", "numeric_all", "madeup"):
        for rule in ("capped", "uncapped"):
            for subset in ("few_right", "any_right"):
                block = recall[name][rule][subset]
                if block["fields"]:
                    assert block["at_1"] <= block["at_3"] <= block["at_5"] <= 1
    few = recall["numeric"]["capped"]["few_right"]
    if few["fields"]:              # recall at 1 is the router's accuracy on those fields
        assert abs(few["at_1"] - hard[headline]["few_right"]["accuracy"]["knn_router"]) < 1e-12
    for cid, shares in document["routing"].items():
        for key, values in shares.items():
            if key != "nested_k_all":
                assert abs(sum(values) - 1) < 1e-9, (cid, key)
    points = suites[headline]["points"]
    assert set(points) == {"router_numeric", "router_numeric_all", "router_madeup", "router_control"}
    # with every fitting request on the grid, the router can nest its control
    assert 0 <= suites[headline]["nested_cells_with_all"] <= suites[headline]["nested_cells"]


def test_pareto_helpers():
    points = [(1.0, 0.5), (2.0, 0.7), (2.0, 0.6), (3.0, 0.7), (0.5, 0.4)]
    front = benchmark_metrics.pareto(points)
    assert front == [0, 1, 4]
    assert benchmark_metrics.pareto_distances([points[1]], front, points, log_cost=False) == 0
    assert benchmark_metrics.pareto_distances([points[3]], front, points, log_cost=False) > 0


def test_benchmark_metrics(corpus):
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path, out = _results(corpus, directory, ("rows", "frontier"))
        diagnostics.run(corpus, out / "diagnostics")
        suites = benchmark_metrics.run(corpus, out, path, processes=1)
    for suite, block in suites.items():
        methods, refs = block["methods"], block["references"]
        assert methods["best_single"]["gain_b"] == 0 and methods["oracle"]["gap_o"] == 0
        assert refs["T"]["accuracy"] == max(s["accuracy"] for s in block["single_models"])
        assert refs["D"]["accuracy"] >= refs["T"]["accuracy"]
        for name, entry in methods.items():
            if "gain_b_ci" in entry and name != "best_single":
                low, high = entry["gain_b_ci"]
                assert low <= high
        policies = block["policies"]
        assert policies["best_single"]["perf_gain_B"] == 0 and policies["best_single"]["cost_save_B"] == 0
        front = [(c, a) for c, a, _, _ in block["frontier"]]
        for c, a in front:
            assert not any(c2 <= c and a2 >= a and (c2 < c or a2 > a) for c2, a2 in front)

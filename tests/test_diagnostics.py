"""End-to-end tests of the aggregate diagnostics on the made-up corpus."""

import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon

from data import load, schema
from sim import core
from sim.types import ECE_BINS, REPORT_FLOORS


def run_cli(corpus, out):
    """Run the documented command and return after it has written the reports."""
    subprocess.run(["python3", "-B", "-m", "analysis.diagnostics",
                    "--corpus", str(corpus), "--out", str(out)], check=True)


@pytest.fixture(scope="module")
def reports(corpus):
    """Yield the reports from one complete run, within the workspace."""
    with TemporaryDirectory(dir=".") as folder:
        out = Path(folder)
        run_cli(corpus, out)
        yield out


def rows(reports, table):
    """Return the CSV records with empty cells preserved."""
    with (reports / f"{table}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def summary(reports):
    """Return the suite summaries."""
    return json.loads((reports / "diagnostics.json").read_text(encoding="utf-8"))


def test_repeatable_files(reports, corpus):
    expected = {"models.csv", "reliability.csv", "pairs.csv", "headroom.csv", "failures.csv",
                "diagnostics.json"}
    assert {p.name for p in reports.iterdir()} == expected
    with TemporaryDirectory(dir=".") as folder:
        other = Path(folder)
        run_cli(corpus, other)
        assert {p.name for p in other.iterdir()} == expected
        for name in sorted(expected):
            assert (reports / name).read_bytes() == (other / name).read_bytes(), name


def test_aggregate_dimensions(reports, corpus, summary):
    cids = [r["catalog_id"] for r in load.list_benchmarks(corpus)]
    models = [r["model_index"] for r in load.list_models(corpus)]
    per_model = [(cid, m) for cid in cids for m in models]
    for table in ("models", "failures"):
        assert [(int(r["catalog_id"]), int(r["model_index"])) for r in rows(reports, table)] == per_model
    assert [int(r["catalog_id"]) for r in rows(reports, "headroom")] == cids
    assert [(int(r["catalog_id"]), int(r["model_i"]), int(r["model_j"]))
            for r in rows(reports, "pairs")] == [
        (cid, i, j) for cid in cids for i, j in combinations(models, 2)]
    assert [(int(r["catalog_id"]), int(r["model_index"]), int(r["bin"]))
            for r in rows(reports, "reliability")] == [
        (cid, m, k) for cid, m in per_model for k in range(ECE_BINS)]

    forbidden = re.compile(r"^(?:item|field|request)$|(?:^|_)(?:item|field|request)_(?:id|index|position|pos|key|row)(?:$|_)")
    for path in reports.glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as stream:
            assert not any(forbidden.search(col) for col in next(csv.reader(stream)))
    assert not any(forbidden.search(key) for key in json_keys(summary))
    assert summary["synthetic"] is True


def json_keys(value):
    """Yield the column names at every level of the JSON report."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from json_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from json_keys(child)


def test_copied_model(reports, corpus, planted):
    original, copy = sorted(planted["duplicate"].values())
    copied = [r for r in rows(reports, "pairs")
              if (int(r["model_i"]), int(r["model_j"])) == (original, copy)]
    assert len(copied) == len(load.list_benchmarks(corpus))
    for row in copied:
        if row["error_correlation"]:
            assert float(row["error_correlation"]) == 1.0
        assert float(row["disagreement_rate"]) == 0.0
        assert float(row["choice_disagreement_rate"]) == 0.0
        assert float(row["mean_js_distance"]) == 0.0
    assert any(row["error_correlation"] for row in copied)


def test_pair_groups(reports, corpus, summary, planted):
    model_info = load.list_models(corpus)
    groups = defaultdict(lambda: defaultdict(list))
    contributing = defaultdict(set)
    for row in rows(reports, "pairs"):
        i, j, cid = (int(row[k]) for k in ("model_i", "model_j", "catalog_id"))
        same_org = planted["organisations"][i] == planted["organisations"][j]
        base = model_info[i]["base_model"]
        same_base = bool(base) and base == model_info[j]["base_model"]
        assert row["same_organisation"].lower() == str(same_org).lower()
        assert row["same_base_model"].lower() == str(same_base).lower()
        if row["error_correlation"]:
            for group, member in (("same_organisation", same_org), ("same_base_model", same_base),
                                  ("other", not same_org and not same_base), ("different", not same_org)):
                if member:
                    groups[group][cid].append(float(row["error_correlation"]))
                    contributing[group].add((i, j))

    def suite_mean(group):
        values = [math.fsum(vals) / len(vals) for vals in groups[group].values()]
        return math.fsum(values) / len(values)

    assert suite_mean("same_organisation") > suite_mean("different")
    actual = summary["suites"]["all36"]["overall"]["pair_groups"]
    for group in ("same_organisation", "same_base_model", "other"):
        assert actual[group]["error_correlation"] == pytest.approx(suite_mean(group))
        assert actual[group]["n_benchmarks"] == len(groups[group])
        assert actual[group]["n_pairs"] == len(contributing[group])


def test_suite_means_and_specialist(reports, summary, planted):
    settings = json.loads(Path("SETTINGS.json").read_text(encoding="utf-8"))
    data = rows(reports, "models")
    assert summary["suites"].keys() == settings["suites"].keys()
    for name, config in settings["suites"].items():
        suite = summary["suites"][name]
        suite_rows = [r for r in data if int(r["catalog_id"]) not in config["exclude"]]
        assert set(suite["by_area"]) == {r["area"] for r in suite_rows}
        for area, scope in [(None, suite["overall"]), *suite["by_area"].items()]:
            expected = defaultdict(list)
            for row in suite_rows:
                if area is None or row["area"] == area:
                    expected[int(row["model_index"])].append(row)
            assert len(scope["models"]) == len(expected)
            for row in scope["models"]:
                for key in ("accuracy_first_max", "usd_per_1k_requests"):
                    vals = [float(r[key]) for r in expected[row["model_index"]] if r[key]]
                    assert row[key] == math.fsum(vals) / len(vals)
                    assert row["n_benchmarks"][key] == len(vals)
    specialist = planted["specialist"]
    suite = summary["suites"]["all36"]
    area_models = suite["by_area"][specialist["area"]]["models"]
    assert max(area_models, key=lambda r: r["accuracy_first_max"])["model_index"] == specialist["model"]
    for model_rows in (area_models, suite["overall"]["models"]):
        assert min(model_rows, key=lambda r: r["usd_per_1k_requests"])["model_index"] == specialist["model"]


def test_raw_formatting(reports, corpus, planted):
    records = {(int(r["catalog_id"]), int(r["model_index"])): r for r in rows(reports, "models")}
    rounded = set(planted["rounded_to_4_decimals"])
    sparse = set(planted["exact_zeros"])
    ties = set(planted["tied_maxima"])
    unrounded_zeros, untied_off_sum = set(), set()
    for info in load.list_benchmarks(corpus):
        bench = load.load_benchmark(corpus, info["catalog_id"], keep_raw=True)
        for m in range(bench.M):
            zero_count = off_sum_count = 0
            for w, block in bench.raw.items():
                raw = block[:, m]
                zeros = (raw == 0).any(axis=1)
                zero_count += np.count_nonzero(raw == 0)
                if zeros.any():
                    if m in rounded:
                        assert np.array_equal(raw[zeros], np.round(raw[zeros], 4))
                    else:
                        unrounded_zeros.add(m)
                off_sum = np.abs(load.fsum_rows(raw) - 1.0) > w * np.finfo(np.float64).eps
                off_sum_count += off_sum.sum()
                maxima = (raw == raw.max(axis=1, keepdims=True)).sum(axis=1)
                if (off_sum & (maxima == 1)).any():
                    untied_off_sum.add(m)
                if m not in rounded and m not in planted["single_precision"] and off_sum.any():
                    assert m in ties and (maxima[off_sum] > 1).all()
            row = records[bench.cid, m]
            assert float(row["zero_entry_share"]) == zero_count / bench.width.sum()
            assert float(row["off_sum_share"]) == off_sum_count / bench.F
    assert {m for (_, m), row in records.items() if int(row["zero_mass_count"]) > 0} == sparse
    assert unrounded_zeros == sparse
    assert untied_off_sum == rounded | set(planted["single_precision"])


def test_failure_partition_and_empty_bins(reports, summary):
    for row in rows(reports, "failures"):
        assert math.fsum(float(row[k]) for k in
                         ("right_share", "wrong_high_confidence_share", "wrong_low_confidence_share")
                         ) == pytest.approx(1.0)
        for key in ("zero_mass_share", "tied_maxima_share"):
            assert 0 <= float(row[key]) <= 1
    bins = rows(reports, "reliability")
    empty = [r for r in bins if int(r["count"]) == 0]
    assert empty and all(r["accuracy"] == "" and r["mean_confidence"] == "" for r in empty)
    benchmark_values = defaultdict(list)
    for row in bins:
        if row["accuracy"]:
            benchmark_values[int(row["model_index"]), int(row["bin"])].append(float(row["accuracy"]))
    for row in summary["suites"]["all36"]["overall"]["reliability"]:
        vals = benchmark_values[row["model_index"], row["bin"]]
        assert row["n_benchmarks"]["accuracy"] == len(vals)
        assert row["accuracy"] == (math.fsum(vals) / len(vals) if vals else None)


def test_constant_and_empty_indicators():
    from analysis.diagnostics import correctness_statistics
    for counts in ((4, 0, 0, 0), (0, 0, 0, 4), (1, 3, 0, 0)):
        stats = correctness_statistics(*counts)
        assert stats["error_correlation"] is None
        assert stats["q_statistic"] is None
        assert stats["disagreement_rate"] == (counts[1] + counts[2]) / sum(counts)
    assert all(x is None for x in correctness_statistics(0, 0, 0, 0).values())


def test_pair_means_with_partial_coverage():
    from analysis.diagnostics import DIMENSIONS, scope_summary
    benchmarks = []
    for cid, correlations in enumerate(((1.0, 0.0, None), (0.0, None, None))):
        tables = {name: [] for name in DIMENSIONS}
        for (i, j), value in zip(combinations(range(3), 2), correlations):
            tables["pairs"].append(dict(catalog_id=cid, model_i=i, model_j=j, same_organisation=True,
                                        same_base_model=False, error_correlation=value))
        benchmarks.append(tables)
    result = scope_summary(benchmarks)
    assert result["pair_groups"]["same_organisation"] == dict(
        error_correlation=0.25, n_benchmarks=2, n_pairs=2, n_pairs_total=3)
    assert result["pair_groups"]["other"]["error_correlation"] is None
    by_model = {r["model_index"]: r for r in result["pairs_by_model"]}
    assert by_model[0]["error_correlation"] == 0.25
    assert by_model[2]["n_benchmarks"]["error_correlation"] == 1


def test_distances_at_probability_extremes():
    from analysis.diagnostics import js_distances
    # Equal and disjoint support, a tiny perturbation, and a positive subnormal.
    d = 2.0 ** -32
    tiny = np.nextafter(0.0, 1.0)
    p = np.array([[1, 0], [1, 0], [0.5 - d, 0.5 + d], [1, 0]])
    q = np.array([[1, 0], [0, 1], [0.5 + d, 0.5 - d], [tiny, 1]])
    # For this symmetric perturbation, distance is d*sqrt(2/log(2)) to relative order d**2.
    expected = [0.0, 1.0, d * math.sqrt(2 / math.log(2)), 1.0]
    np.testing.assert_allclose(js_distances(p, q), expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("cid", [2, 9, 31, 59])
def test_model_scoring_and_usage(reports, corpus, planted, cid):
    bench = load.load_benchmark(corpus, cid, keep_raw=True)
    judged = bench.judged
    data = {int(r["model_index"]): r for r in rows(reports, "models") if int(r["catalog_id"]) == cid}
    failures = {int(r["model_index"]): r for r in rows(reports, "failures") if int(r["catalog_id"]) == cid}
    bins = defaultdict(list)
    for r in rows(reports, "reliability"):
        if int(r["catalog_id"]) == cid:
            bins[int(r["model_index"])].append(r)
    for m in sorted(set(planted["tied_maxima"] + planted["adapted"] + planted["rounded_to_4_decimals"])):
        row = data[m]
        vectors = core.single_model_vectors(bench, m)
        hit, mass, _, conf = core.decode(bench, vectors)
        assert float(row["accuracy_first_max"]) == hit[judged].mean()
        assert float(row["accuracy_reported"]) == core.reported_hits(bench, m)[judged].mean()
        assert float(row["brier"]) == pytest.approx(core.brier(bench, vectors)[judged].mean())
        assert float(row["ece"]) == pytest.approx(core.ece(conf[judged], hit[judged]))
        assert float(row["mean_confidence"]) == pytest.approx(conf[judged].mean())
        counts, means, accuracies = core.reliability_table(conf[judged], hit[judged])
        for r in bins[m]:
            k = int(r["bin"])
            assert int(r["count"]) == counts[k]
            if counts[k]:
                assert float(r["accuracy"]) == accuracies[k]
                assert float(r["mean_confidence"]) == means[k]
        failure = failures[m]
        assert float(failure["right_share"]) == hit[judged].mean()
        for key, mask in (("wrong_high_confidence_share", conf >= 0.9), ("wrong_low_confidence_share", conf < 0.9)):
            assert float(failure[key]) == (~hit & mask)[judged].mean()
        raw_mass = np.empty(bench.F)
        tied = np.empty(bench.F, dtype=bool)
        for w in bench.widths:
            raw = bench.raw[w][:, m]
            raw_mass[bench.idx[w]] = core.block_mass(bench, w, raw)
            tied[bench.idx[w]] = (raw == raw.max(axis=1, keepdims=True)).sum(axis=1) > 1
        assert float(failure["tied_maxima_share"]) == tied[judged].mean()
        assert float(failure["zero_mass_share"]) == (raw_mass[judged] == 0).mean()
        for kind, accepted in (("raw", raw_mass), ("normalised", mass)):
            for floor in REPORT_FLOORS:
                label = format(floor, ".0e").replace("-0", "-")
                expected = core.loss_of_mass(accepted[judged], floor).mean()
                assert float(row[f"log_loss_{kind}_floor_{label}"]) == pytest.approx(expected, abs=1e-14, rel=1e-14)
        assert int(row["zero_mass_count"]) == np.count_nonzero(raw_mass[judged] == 0)
        assert float(row["usd_per_1k_requests"]) == pytest.approx(1000 * math.fsum(bench.cost[:, m]) / bench.R)
        assert float(row["calls_per_request"]) == bench.calls[:, m].mean()
        usage = schema.read_table(schema.table_path(corpus, cid, "usage", m), schema.USAGE, ["attempts"])
        assert float(row["attempts_per_request"]) == usage["attempts"].to_numpy().mean()
        for p in (50, 95):
            assert float(row[f"latency_p{p}_s"]) == pytest.approx(np.percentile(bench.lat[:, m], p))


def test_pair_denominators_and_headroom(reports, corpus):
    bench = load.load_benchmark(corpus, 2)
    assert 0 < bench.judged.sum() < bench.F
    decoded = [core.decode(bench, core.single_model_vectors(bench, m)) for m in range(bench.M)]
    hits = np.column_stack([entry[0][bench.judged] for entry in decoded])
    pairs = [r for r in rows(reports, "pairs") if int(r["catalog_id"]) == bench.cid]
    for row in pairs:
        i, j = int(row["model_i"]), int(row["model_j"])
        a, b = hits[:, i], hits[:, j]
        assert float(row["disagreement_rate"]) == np.mean(a != b)
        assert float(row["double_fault_rate"]) == np.mean(~a & ~b)
        assert float(row["choice_disagreement_rate"]) == np.mean(decoded[i][2] != decoded[j][2])
        n11, n00 = int((a & b).sum()), int((~a & ~b).sum())
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        if n11 * n00 + n10 * n01:
            assert float(row["q_statistic"]) == (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)
        if a.min() != a.max() and b.min() != b.max():
            assert float(row["error_correlation"]) == pytest.approx(np.corrcoef(~a, ~b)[0, 1])
    first = pairs[0]
    i, j = int(first["model_i"]), int(first["model_j"])
    distances = np.concatenate([jensenshannon(bench.P[w][:, i], bench.P[w][:, j], base=2, axis=1)
                                for w in bench.widths])
    assert float(first["mean_js_distance"]) == pytest.approx(distances.mean(), rel=1e-12)
    row = next(r for r in rows(reports, "headroom") if int(r["catalog_id"]) == bench.cid)
    count = hits.sum(axis=1)
    assert float(row["best_single_share"]) == hits.mean(axis=0).max()
    assert float(row["at_least_one_share"]) == (count >= 1).mean()
    assert float(row["at_least_two_share"]) == (count >= 2).mean()
    assert float(row["majority_share"]) == (count > bench.M / 2).mean()
    assert float(row["all_share"]) == (count == bench.M).mean()
    assert float(row["none_share"]) == (count == 0).mean()
    assert float(row["mean_models_right"]) == count.mean()
    corr = [float(r["error_correlation"]) for r in pairs if r["error_correlation"]]
    assert float(row["effective_voters"]) == pytest.approx(bench.M / (1 + (bench.M - 1) * math.fsum(corr) / len(corr)))

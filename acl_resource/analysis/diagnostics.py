"""Benchmark and suite diagnostics for stored model predictions on NLP tasks."""

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from ..data import load, schema
from ..sim import core, types


# These columns identify a series across benchmarks. The other columns are measures.
DIMENSIONS = {
    "models": ("model_index",),
    "reliability": ("model_index", "bin"),
    "pairs": ("model_i", "model_j", "same_organisation", "same_base_model"),
    "headroom": (),
    "failures": ("model_index",),
}


def mean(values):
    """Return the accurately summed mean, or None for an empty sample."""
    return math.fsum(values) / len(values) if len(values) else None


def share(count, total):
    """Return a fraction, or None for an empty denominator."""
    return count / total if total else None


def usage_statistics(corpus, bench, m):
    """Return recorded cost, physical calls, attempts and latency per model."""
    calls = attempts = usd = p50 = p95 = None
    if bench.calls is not None:
        # The loader checks request alignment but does not expose attempts.  The open tier of schema version 2
        # has no attempt count.
        usage = schema.usage_schema(load.read_manifest(corpus)["schema_version"])
        if "attempts" in usage.names:
            path = schema.table_path(corpus, bench.cid, "usage", m)
            column = schema.read_table(path, usage, columns=["attempts"])["attempts"]
            attempts = share(int(column.to_numpy().sum()), bench.R)
        calls = share(int(bench.calls[:, m].sum()), bench.R)
        usd = share(1000 * math.fsum(bench.cost[:, m]), bench.R)
    if bench.lat is not None and bench.R:
        p50, p95 = map(float, np.percentile(bench.lat[:, m], [50, 95]))
    return dict(usd_per_1k_requests=usd, latency_p50_s=p50, latency_p95_s=p95,
                calls_per_request=calls, attempts_per_request=attempts)


def raw_statistics(bench, m, judged):
    """Return entry and sum diagnostics, judged tied-maxima share and raw judged masses."""
    entries = zeros = off_sum = ties = 0
    masses = []
    for w in bench.widths:
        raw = bench.raw[w][:, m, :]
        entries += raw.size
        zeros += int(np.count_nonzero(raw == 0))
        tolerance = w * np.finfo(raw.dtype).eps
        off_sum += int(np.count_nonzero(np.abs(raw.sum(axis=1) - 1) > tolerance))
        selected = judged[bench.idx[w]]
        if selected.any():
            values = raw[selected]
            ties += int(np.count_nonzero((values == values.max(axis=1)[:, None]).sum(axis=1) > 1))
            masses.append(core.block_mass(bench, w, values, sel=selected))
    formatting = dict(zero_entry_share=share(zeros, entries), off_sum_share=share(off_sum, bench.F))
    raw_mass = np.concatenate(masses) if masses else np.empty(0)
    return formatting, share(ties, int(judged.sum())), raw_mass


def model_rows(corpus, bench, m, judged):
    """Return model, reliability and failure rows, judged hits and all decoded options."""
    vectors = core.single_model_vectors(bench, m)
    hit, mass, choice, confidence = core.decode(bench, vectors)
    hit, mass, confidence = hit[judged], mass[judged], confidence[judged]
    formatting, tie_share, raw_mass = raw_statistics(bench, m, judged)
    n = len(hit)
    key = dict(catalog_id=bench.cid, area=bench.area, model_index=m)
    accuracy = share(int(hit.sum()), n)
    reported = mean(core.reported_hits(bench, m)[judged]) if n else None
    model = dict(key, accuracy_reported=reported, accuracy_first_max=accuracy)
    for kind, values in (("raw", raw_mass), ("normalised", mass)):
        for floor in types.REPORT_FLOORS:
            label = format(floor, ".0e").replace("e-0", "e-")
            model[f"log_loss_{kind}_floor_{label}"] = mean(core.loss_of_mass(values, floor))
    zero_mass = int(np.count_nonzero(raw_mass == 0))
    model.update(brier=mean(core.brier(bench, vectors)[judged]) if n else None,
                 ece=core.ece(confidence, hit) if n else None, mean_confidence=mean(confidence),
                 **formatting, zero_mass_count=zero_mass if bench.has_labels else None)
    model.update(usage_statistics(corpus, bench, m))

    counts, bin_confidence, bin_accuracy = core.reliability_table(confidence, hit)
    reliability = [
        dict(key, bin=b, count=int(count) if bench.has_labels else None,
             mean_confidence=float(bin_confidence[b]) if count else None,
             accuracy=float(bin_accuracy[b]) if count else None)
        for b, count in enumerate(counts)
    ]
    failures = dict(key, right_share=accuracy,
                    wrong_high_confidence_share=share(int(np.count_nonzero(~hit & (confidence >= 0.9))), n),
                    wrong_low_confidence_share=share(int(np.count_nonzero(~hit & (confidence < 0.9))), n),
                    zero_mass_share=share(zero_mass, n), tied_maxima_share=tie_share)
    return model, reliability, failures, hit, choice


def correctness_statistics(n11, n10, n01, n00):
    """Return correctness association and disagreement measures from joint counts."""
    # Python integer products avoid overflow in the correlation denominator.
    n11, n10, n01, n00 = map(int, (n11, n10, n01, n00))
    total = n11 + n10 + n01 + n00
    numerator = n11 * n00 - n10 * n01
    marginals = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    return dict(error_correlation=share(numerator, math.sqrt(marginals)),
                q_statistic=share(numerator, n11 * n00 + n10 * n01),
                disagreement_rate=share(n10 + n01, total), double_fault_rate=share(n00, total))


def js_distances(p, q):
    """Return one base-two distance per row, preserving small differences between models."""
    total = p + q
    delta = np.abs(p - q)
    t = np.divide(delta, total, out=np.zeros_like(total), where=total > 0)
    low = np.minimum(p, q)
    # Twice the divergence uses high*log(1+t) + low*log(1-t) per option.
    # Combining the low terms before taking logs avoids cancellation near equality.
    log_product = np.zeros_like(t)
    np.log1p(-t * t, out=log_product, where=t < 1)
    edge = (t == 1) & (low > 0)
    if edge.any():
        # The smaller probability is positive but 1-t rounded to zero.
        log_product[edge] = math.log(2) + np.log(2 * low[edge] / total[edge])
    terms = delta * np.log1p(t) + low * log_product
    return np.sqrt(terms.sum(axis=1) / (2 * math.log(2)))


def mean_js_distance(bench, i, j):
    """Return the field mean of the distances, weighting each width by its field count."""
    totals = [math.fsum(js_distances(bench.P[w][:, i], bench.P[w][:, j])) for w in bench.widths]
    return share(math.fsum(totals), bench.F)


def pair_rows(bench, hits, choices):
    """Return one row per unordered pair, with judged correctness and all-field similarity."""
    rows = []
    n = hits.shape[1]
    right_counts = hits.sum(axis=1)
    for i, j in combinations(range(bench.M), 2):
        n11 = int(np.count_nonzero(hits[i] & hits[j]))
        n10, n01 = int(right_counts[i]) - n11, int(right_counts[j]) - n11
        stats = correctness_statistics(n11, n10, n01, n - n11 - n10 - n01)
        left, right = bench.models[i], bench.models[j]
        different = int(np.count_nonzero(choices[i] != choices[j]))
        rows.append(dict(catalog_id=bench.cid, area=bench.area, model_i=i, model_j=j,
                         **stats, choice_disagreement_rate=share(different, bench.F),
                         mean_js_distance=mean_js_distance(bench, i, j),
                         same_organisation=left["organisation"] == right["organisation"],
                         same_base_model=bool(left["base_model"]) and left["base_model"] == right["base_model"]))
    return rows


def headroom_row(bench, hits, pairs):
    """Return shares of judged fields covered by the panel and its effective voter count."""
    n = hits.shape[1]
    votes = hits.sum(axis=0)
    rho = mean([p["error_correlation"] for p in pairs if p["error_correlation"] is not None])
    denominator = 1 + (bench.M - 1) * rho if rho is not None else None
    return dict(catalog_id=bench.cid, area=bench.area,
                best_single_share=share(int(hits.sum(axis=1).max()), n),
                at_least_one_share=share(int(np.count_nonzero(votes >= 1)), n),
                at_least_two_share=share(int(np.count_nonzero(votes >= 2)), n),
                majority_share=share(int(np.count_nonzero(2 * votes > bench.M)), n),
                all_share=share(int(np.count_nonzero(votes == bench.M)), n),
                none_share=share(int(np.count_nonzero(votes == 0)), n),
                mean_models_right=mean(votes), effective_voters=share(bench.M, denominator))


def benchmark_tables(corpus, bench):
    """Return all five aggregate tables for one benchmark loaded with raw vectors."""
    judged = bench.judged if bench.has_labels else np.zeros(bench.F, dtype=bool)
    hits = np.empty((bench.M, int(judged.sum())), dtype=bool)
    choices = np.empty((bench.M, bench.F), dtype=np.int32)
    tables = {name: [] for name in DIMENSIONS}
    for m in range(bench.M):
        model, reliability, failures, hits[m], choices[m] = model_rows(corpus, bench, m, judged)
        tables["models"].append(model)
        tables["reliability"].extend(reliability)
        tables["failures"].append(failures)
    tables["pairs"] = pair_rows(bench, hits, choices)
    tables["headroom"].append(headroom_row(bench, hits, tables["pairs"]))
    return tables


def grouped(rows, columns):
    """Return rows grouped by a tuple of dimension values, preserving input order."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[column] for column in columns)].append(row)
    return groups


def averages(rows, measures):
    """Return column means and the number of defined values used for each."""
    means, counts = {}, {}
    for key in measures:
        values = [row[key] for row in rows if row[key] is not None]
        means[key], counts[key] = mean(values), len(values)
    return means, counts


def summary_rows(rows, dimensions, measures):
    """Return equal-benchmark means for each series, with per-measure coverage."""
    result = []
    for key, benchmarks in grouped(rows, dimensions).items():
        means, counts = averages(benchmarks, measures)
        result.append(dict(zip(dimensions, key)) | means | {"n_benchmarks": counts})
    return result


def across_models(rows, dimensions, measures):
    """Return suite means after averaging available models or pairs within each benchmark."""
    within = []
    keys = ("catalog_id", *dimensions)
    for key, group in grouped(rows, keys).items():
        means, _ = averages(group, measures)
        within.append(dict(zip(keys, key)) | means)
    return summary_rows(within, dimensions, measures)


def pair_groups(rows):
    """Return group correlations and the counts of distinct member and contributing pairs."""
    groups = {"same_organisation": [], "same_base_model": [], "other": []}
    for row in rows:
        for name in ("same_organisation", "same_base_model"):
            if row[name]:
                groups[name].append(row)
        if not row["same_organisation"] and not row["same_base_model"]:
            groups["other"].append(row)
    result = {}
    for name, pairs in groups.items():
        defined = [p for p in pairs if p["error_correlation"] is not None]
        by_benchmark = grouped(defined, ("catalog_id",))
        correlations = [mean([p["error_correlation"] for p in group]) for group in by_benchmark.values()]
        result[name] = dict(error_correlation=mean(correlations), n_benchmarks=len(correlations),
                            n_pairs=len({(p["model_i"], p["model_j"]) for p in defined}),
                            n_pairs_total=len({(p["model_i"], p["model_j"]) for p in pairs}))
    return result


def scope_summary(benchmarks):
    """Return table means for a suite or its subset in one task area."""
    result = {"across_models": {}}
    for name, dimensions in DIMENSIONS.items():
        rows = [row for tables in benchmarks for row in tables[name]]
        measures = [k for k in rows[0] if k not in ("catalog_id", "area", *dimensions)] if rows else []
        result[name] = summary_rows(rows, dimensions, measures)
        if name == "pairs":
            result["pair_groups"] = pair_groups(rows)
            partners = [row | {"model_index": row[side]} for row in rows for side in ("model_i", "model_j")]
            result["pairs_by_model"] = across_models(partners, ("model_index",), measures)
        if name != "headroom":
            series = ("bin",) if "bin" in dimensions else ()
            result["across_models"][name] = across_models(rows, series, measures)
    return result


def suite_summaries(tables, benchmarks, suites):
    """Return overall and per-task-area summaries for the suites in the settings."""
    summaries = {}
    for name, config in suites.items():
        selected = [b for b in benchmarks if b["catalog_id"] not in config["exclude"]]
        by_area = {}
        for area in sorted({b["area"] for b in selected}):
            by_area[area] = scope_summary([tables[b["catalog_id"]] for b in selected if b["area"] == area])
        overall = scope_summary([tables[b["catalog_id"]] for b in selected])
        summaries[name] = dict(overall=overall, by_area=by_area)
    return summaries


def write_csv(path, rows):
    """Write ordered aggregate rows with round-trip float precision and empty undefined cells."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(corpus, out):
    """Write the five aggregate tables and their suite summaries."""
    benchmarks = sorted(load.list_benchmarks(corpus), key=lambda row: row["catalog_id"])
    manifest = load.read_manifest(corpus)
    synthetic = (manifest.get("source", {}).get("scope") == "synthetic" or manifest.get("scope") == "synthetic"
                 or any(b["name"].startswith("synthetic-") for b in benchmarks))
    settings = json.loads((Path(__file__).parents[1] / "SETTINGS.json").read_text(encoding="utf-8"))
    tables = {}
    for b in benchmarks:
        cid = b["catalog_id"]
        tables[cid] = benchmark_tables(corpus, load.load_benchmark(corpus, cid, keep_raw=True))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in DIMENSIONS:
        write_csv(out / f"{name}.csv", [row for bench in tables.values() for row in bench[name]])
    result = dict(synthetic=synthetic, suites=suite_summaries(tables, benchmarks, settings["suites"]))
    with (out / "diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def main(argv=None):
    """Run the diagnostics command for a processed corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.corpus, args.out)


if __name__ == "__main__":
    main()

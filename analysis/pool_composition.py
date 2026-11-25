"""How the make-up of the model pool bears on "pooling everything" and on how alike the models err.

    python3 -B -m analysis.pool_composition --corpus DIR --settings FILE --results RESULTS --out OUT \
        [--processes N]

Several endpoints of the pool are versions or tiers of one organisation's model, and some organisations document
the same base model.  A reviewer noted that an equal-weight pool then counts one family several times, and that
the error correlation is reported by organisation only.  This module adds two things; the design was fixed before
the first run on the corpus.

``balanced``           a pool with equal weight per organisation (an organisation's weight is shared equally by its
                       models), on the original vectors (``pool_by_organisation``) and with one temperature fitted
                       on the fit folds as for the equal-weight pool (``pool_by_organisation_temperature``).  It
                       calls every model, like the equal-weight pool, and reads no label without the temperature.
                       Contrasts: against the equal-weight pool with the same treatment (``uniform_mean``,
                       ``uniform_temperature``) and against the best model per benchmark (``best_single``), with
                       the within-benchmark interval of ``sim.stats`` and the interval over resampled benchmarks
                       of ``paper_facts``.
``top_k_validated``    the equal-weight pool of the k best-ranked calibrated models (the ranking of the stored
                       ``ranked_top_k`` panels, on the fit folds), with k chosen per benchmark and rotation by the
                       loss on the validation fold (first minimum over k = 1..M).  The paper lists top-k pools
                       among its baselines but reports them at fixed k only.  Contrasts: against ``best_single``,
                       ``stack_final_temperature`` and ``greedy``.
``error_correlation``  from ``RESULTS/diagnostics/homogenization_pairs.csv``: the mean error correlation of two
                       models, and the same given the number of other models that err, for pairs of one
                       organisation, pairs of different organisations that document the same base model (names
                       equal after a trailing "-Base" is dropped), and all other pairs of different organisations.
                       Benchmarks weigh equally; a pair without a defined correlation on a benchmark is left out
                       there.  Skipped with a note when the file is absent.

Checks that stop the run: the equal-weight pools recomputed here reproduce ``rows.json``; the top-k pool at a
fixed k of 3, built by the same code, reproduces ``ranked_top_k`` of ``RESULTS/frontier.json`` when that file is
there.  Writes
``OUT/pool_composition.json`` with aggregates only.  Added after the first run, on 23 November 2026, in answer to
a review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from data.schema import N_FOLDS, roles, sha256_file
from sim import core, metrics, stats
from sim.baselines import best_single, uniform_mean, uniform_temperature
from sim.engine import load_task
from . import run_experiments as RX
from .paper_facts import _over_benchmarks

RAW, TEMPERED, TOP_K = "pool_by_organisation", "pool_by_organisation_temperature", "top_k_validated"
PAIRS = ((RAW, "uniform_mean"), (TEMPERED, "uniform_temperature"), (RAW, "best_single"), (TEMPERED, "best_single"),
         (TOP_K, "best_single"), (TOP_K, "stack_final_temperature"), (TOP_K, "greedy"))
ARCHIVED = ("best_single", "stack_final_temperature", "greedy")
CHECK_SIZE = 3
METRICS = ("log_loss", "accuracy")
TOLERANCE = 1e-12


def prefix_weights(task, fold):
    """[M, M] equal weights over the k best-ranked models of the rotation, row k - 1 for size k."""
    order = best_single.ranking(task, fold)
    weights = np.zeros((task.b.M, task.b.M))
    for k in range(1, task.b.M + 1):
        weights[k - 1, order[:k]] = 1.0 / k
    return weights


def top_k_pool(task, sizes=None):
    """The equal-weight pool of the best-ranked calibrated models, at given sizes or at validated ones."""
    weights = np.zeros((N_FOLDS, task.b.M))
    chosen = []
    for f in range(N_FOLDS):
        prefixes = prefix_weights(task, f)
        if sizes is None:
            losses = task.fit_loss(f, prefixes, "calibrated", folds=[roles(f)[1]])
            k = int(np.argmin(losses)) + 1
        else:
            k = sizes[f]
        weights[f] = prefixes[k - 1]
        chosen.append(k)
    labelled = "fit" if sizes is not None else "fit and validation"
    return uniform_mean.panel(task, weights, uniform_mean.member_choices(task), info={"size": chosen},
                              ledger={"labeled_folds": labelled, "models_profiled": task.b.M})


def organisation_weights(models):
    """[M] weights: every organisation weighs the same, and its models share its weight equally."""
    organisations = [m["organisation"] for m in models]
    size = {o: organisations.count(o) for o in set(organisations)}
    return np.array([1.0 / (len(size) * size[o]) for o in organisations])


def fitted_temperatures(task, settings, weights):
    """One temperature per rotation for a fixed-weight pool, chosen on the fit folds like the equal-weight pool's."""
    candidates = uniform_temperature.menu(settings)
    mass = np.empty((len(candidates), task.nJ))
    uniform_temperature.fill_tempered_mass(task, mass, task.fold_blocks(range(N_FOLDS), judged_only=True), weights,
                                           None, candidates)
    loss = core.loss_of_mass(mass)
    sums = np.stack([loss[:, task.fold == g].sum(axis=1) for g in range(N_FOLDS)], axis=1)
    counts = np.bincount(task.fold, minlength=N_FOLDS)
    temperatures = []
    for f in range(N_FOLDS):
        fit = list(roles(f)[0])
        losses = sums[:, fit].sum(axis=1) / counts[fit].sum()
        temperatures.append(candidates[uniform_temperature.best_temperature(losses, candidates)])
    return temperatures


def one_benchmark(cid, corpus, settings, digest, results, stored, fixed, draws, seed):
    task = load_task(corpus, cid)
    weights = organisation_weights(task.b.models)
    stack = np.tile(weights, (N_FOLDS, 1))
    outcomes = {
        RAW: uniform_mean.panel(task, stack),
        TEMPERED: uniform_mean.panel(task, stack, temperatures=fitted_temperatures(task, settings, weights),
                                     ledger={"labeled_folds": "fit", "models_profiled": task.b.M}),
        TOP_K: top_k_pool(task),
        "uniform_mean": uniform_mean.run(task, settings),
        "uniform_temperature": uniform_temperature.run(task, settings),
    }
    scored = {name: metrics.score(task, outcome) for name, outcome in outcomes.items()}
    fields = {name: (core.loss_of_mass(s.mass), s.hit.astype(np.float64)) for name, s in scored.items()}
    for name in ("uniform_mean", "uniform_temperature"):
        reference = stored[name, cid]
        if (abs(reference["log_loss"] - fields[name][0].mean()) > TOLERANCE
                or abs(reference["accuracy"] - fields[name][1].mean()) > TOLERANCE):
            raise ValueError(f"{name} recomputed here does not reproduce rows.json on benchmark {cid}")
    if cid in fixed and task.b.M >= CHECK_SIZE:
        check = metrics.score(task, top_k_pool(task, [CHECK_SIZE] * N_FOLDS))
        if (abs(fixed[cid]["log_loss"] - core.loss_of_mass(check.mass).mean()) > TOLERANCE
                or abs(fixed[cid]["accuracy"] - check.hit.mean()) > TOLERANCE):
            raise ValueError(f"the top-{CHECK_SIZE} pool does not reproduce frontier.json on benchmark {cid}")
    for name in ARCHIVED:
        mass, hit = RX._local_fields(results, name, task)
        fields[name] = (core.loss_of_mass(mass), hit.astype(np.float64))
    rows = [RX._result_row(task, scored[name], name, None, digest) for name in (RAW, TEMPERED, TOP_K)]
    pairs = {}
    for a, b in PAIRS:
        for i, metric in enumerate(METRICS):
            sampled = stats.bench_replicates(task, np.column_stack([fields[a][i], fields[b][i]]), draws, seed)
            pairs[a, b, metric] = (float(fields[a][i].mean() - fields[b][i].mean()), sampled[:, 0] - sampled[:, 1])
    return cid, rows, pairs, dict(organisations=len(set(m["organisation"] for m in task.b.models)),
                                  largest_weight=float(weights.max()), smallest_weight=float(weights.min()),
                                  sizes=list(map(int, outcomes[TOP_K].info["size"])))


def base_family(name):
    """A documented base model without a trailing "-Base"; empty when none is documented."""
    return name[:-5] if name.endswith("-Base") else name


def error_correlation(path, models, cids):
    """Equal-benchmark mean correlations of pairs by organisation and by documented base model."""
    by_index = {int(m["model_index"]): m for m in models}

    def group(a, b):
        if by_index[a]["organisation"] == by_index[b]["organisation"]:
            return "same_organisation"
        base_a, base_b = base_family(by_index[a]["base_model"]), base_family(by_index[b]["base_model"])
        return "other_organisation_same_base" if base_a and base_a == base_b else "other_organisation_other_base"

    cells = {}
    wanted = set(cids)
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            cid = int(row["catalog_id"])
            if cid not in wanted:
                continue
            a, b = int(row["model_a"]), int(row["model_b"])
            for column in ("error_correlation", "error_correlation_given_others"):
                if row[column] not in ("", "nan"):
                    cells.setdefault((group(a, b), column), {}).setdefault(cid, []).append(float(row[column]))
                    cells.setdefault((f"pair:{by_index[a]['model_code']}+{by_index[b]['model_code']}",
                                      column), {}).setdefault(cid, []).append(float(row[column]))
    out = {}
    pairs = {}
    for a in by_index:
        for b in by_index:
            if a < b:
                pairs.setdefault(group(a, b), []).append((a, b))
    for name, members in pairs.items():
        block = dict(pairs=len(members))
        for column in ("error_correlation", "error_correlation_given_others"):
            per_benchmark = cells.get((name, column), {})
            block[column] = (math.fsum(math.fsum(v) / len(v) for v in per_benchmark.values()) / len(per_benchmark)
                             if per_benchmark else None)
            block[f"{column}_benchmarks"] = len(per_benchmark)
        out[name] = block
    out["shared_base_pairs"] = {}
    for a, b in pairs.get("other_organisation_same_base", []):
        key = f"pair:{by_index[a]['model_code']}+{by_index[b]['model_code']}"
        out["shared_base_pairs"][key[5:]] = {
            column: (math.fsum(math.fsum(v) / len(v) for v in cells[key, column].values()) / len(cells[key, column])
                     if (key, column) in cells else None)
            for column in ("error_correlation", "error_correlation_given_others")}
    return out


def run(corpus, settings_path, results, out, processes=1):
    corpus, results, out = Path(corpus), Path(results), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    document = json.loads((results / "rows.json").read_text(encoding="utf-8"))
    if document["settings_sha256"] != digest:
        raise SystemExit("rows.json was made with other settings; run the experiments first")
    stored = {(r["method"], r["catalog_id"]): r for r in document["rows"]
              if r["catalog_id"] is not None and r["variant"] is None}
    stored_suite = {(r["method"], r["suite"]): r for r in document["rows"]
                    if r["catalog_id"] is None and r["variant"] is None}
    cids = sorted({c for _, c in stored})
    draws, seed = settings["bootstrap"]["draws"], settings["bootstrap"]["seed"]
    fixed = {}
    if (results / "frontier.json").is_file():
        frontier = json.loads((results / "frontier.json").read_text(encoding="utf-8"))
        if frontier["settings_sha256"] == digest:
            fixed = {r["catalog_id"]: r for r in frontier["rows"] if r["catalog_id"] is not None
                     and r["method"] == "ranked_top_k" and r["variant"] == f"k={CHECK_SIZE}"}
    work = partial(one_benchmark, corpus=corpus, settings=settings, digest=digest, results=results, stored=stored,
                   fixed=fixed, draws=draws, seed=seed)
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            parts = list(pool.map(work, cids))
    else:
        parts = [work(c) for c in cids]
    rows = RX._with_suites([r for _, part, _, _ in parts for r in part], settings["suites"])
    pairs = {cid: p for cid, _, p, _ in parts}
    sizes = {cid: extra.pop("sizes") for cid, _, _, extra in parts}
    weights = parts[0][3]
    models = load_task(corpus, cids[0]).b.models
    pair_file = results / "diagnostics" / "homogenization_pairs.csv"
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in cids if c not in set(definition["exclude"])]
        if not members:
            continue
        keys = ("log_loss", "accuracy", "usd_per_1k_requests", "calls_per_request", "n_benchmarks")
        points = {r["method"]: {k: r[k] for k in keys} for r in rows if r["catalog_id"] is None and r["suite"] == suite}
        for name in ("uniform_mean", "uniform_temperature") + ARCHIVED:
            if (name, suite) in stored_suite:
                points[name] = {k: stored_suite[name, suite][k] for k in keys}
        chosen = [k for c in members for k in sizes[c]]
        histogram = np.bincount(chosen, minlength=max(chosen) + 1)
        contrasts = {}
        for a, b in PAIRS:
            block = {}
            for metric in METRICS:
                delta = {c: pairs[c][a, b, metric][0] for c in members}
                within = stats.suite_contrast(delta, {c: pairs[c][a, b, metric][1] for c in members}, members)
                over = _over_benchmarks([delta[c] for c in members], metric, draws, seed)
                block[metric] = dict(difference=within["delta"], within_low=within["ci_low"],
                                     within_high=within["ci_high"], over_low=over["low"], over_high=over["high"],
                                     benchmarks_better=over["better"], benchmarks_worse=over["worse"])
            contrasts[f"{a} - {b}"] = block
        suites[suite] = dict(benchmarks=len(members), points=points, contrasts=contrasts,
                             top_k_sizes=dict(mean=math.fsum(chosen) / len(chosen), cells=len(chosen),
                                              cells_by_size={str(k): int(n) for k, n in enumerate(histogram) if n}),
                             error_correlation=(error_correlation(pair_file, models, members)
                                                if pair_file.is_file() else None))
    RX._json(out / "pool_composition.json", dict(
        synthetic=document["synthetic"], settings_sha256=digest, weights=weights, draws=draws, seed=seed,
        note=None if pair_file.is_file() else "no homogenization_pairs.csv: error correlation skipped",
        suites=suites))
    return suites


def main(argv=None):
    ap = argparse.ArgumentParser(description="A pool balanced by organisation, and error correlation by base model")
    for name in ("corpus", "settings", "results", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    suites = run(a.corpus, a.settings, a.results, a.out, a.processes)
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    block = suites[headline]
    for name, p in block["points"].items():
        print(f"{name:34s} log loss {p['log_loss']:.4f}  accuracy {100 * p['accuracy']:.2f}")
    for name, c in block["contrasts"].items():
        a = c["accuracy"]
        print(f"{name}: accuracy {100 * a['difference']:+.2f} points [{100 * a['within_low']:+.2f}, "
              f"{100 * a['within_high']:+.2f}] within, [{100 * a['over_low']:+.2f}, {100 * a['over_high']:+.2f}] over")
    return 0


if __name__ == "__main__":
    sys.exit(main())

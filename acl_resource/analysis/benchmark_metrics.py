"""Benchmark metrics in the form of LLMRouterBench (Li et al., 2026), on our rows, against two kinds of reference.

    python3 -B -m acl_resource.analysis.benchmark_metrics --corpus DIR --results RESULTS --settings FILE [--processes N]

References, per suite:

``R``  the random model (``random_single``);
``T``  the top model: the one model with the highest mean accuracy over the suite's benchmarks, chosen on the test
       folds (the "Best Single" of LLMRouterBench, which is one model for all datasets chosen in hindsight);
``B``  the best single model of each benchmark, chosen and recalibrated on the four non-test folds (our reference);
``D``  the benchmark oracle: on each benchmark the model with the highest test accuracy (their "Dataset Oracle");
``O``  the per-request oracle (``oracle``).

Per method: AvgAcc (equal-benchmark mean accuracy), log loss, Gain@R, Gain@T, Gain@B = mean over benchmarks of
Acc(a, d) / Acc(ref, d) - 1, Gap@O = mean over benchmarks of 1 - Acc(a, d) / Acc(O, d), with paired group-bootstrap
intervals (items resampled within benchmarks, the draws and seed of the settings) for Gain@B and for the log-loss
difference from B.  Per policy with several settings (``frontier.json``) and per single-configuration method:
PerfGain = AvgAcc of its best setting / AvgAcc(ref) - 1 and CostSave = 1 - the cost of its cheapest setting that is
at least as accurate as ref / Cost(ref) (none when no setting is), for ref = T and B, the same two in log loss
against B, and ParetoDist, the mean L1 distance of its settings to the empirical Pareto frontier of accuracy
against cost (all settings of the built-in policies, the single-configuration built-in methods and the single
models; the oracle and the replayed methods are left out), on min-max normalised axes with cost on a log scale
(and on a linear scale).  Best settings are chosen on the test folds, as in LLMRouterBench, and are labelled so.
T's log loss uses its uncalibrated vectors and is reported only for information.

Writes ``RESULTS/benchmark_metrics.json``.  Added after the first run.
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

from ..data.schema import sha256_file
from ..sim import core, stats
from ..sim.engine import load_task
from . import run_experiments as RX

GRID_POLICIES = ("ranked_top_k", "greedy", "random_panel", "knn_router", "knn_router_text", "single_by_cost",
                 "confidence_cascade")
REPLAYED = ("tree_replay", "greedy_replay", "combiner_replay")
EXCLUDED_FROM_FRONTIER = ("oracle",) + REPLAYED
REFERENCE = "best_single"


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def _csv(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def replicates(cid, corpus, out, methods, bootstrap):
    """Per method: bootstrap draws of Acc(a)/Acc(B) - 1 and of mean log loss(a) - mean log loss(B) on one benchmark."""
    task = load_task(corpus, cid)
    fields = {}
    for name in methods:
        path = RX._field_file(out, name)
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            if f"hit_{cid}" in archive.files:
                fields[name] = archive[f"mass_{cid}"], archive[f"hit_{cid}"].astype(np.float64)
    if REFERENCE not in fields:
        return cid, {}
    ref_mass, ref_hit = fields[REFERENCE]
    ref_loss = core.loss_of_mass(ref_mass)
    out_reps = {}
    for name, (mass, hit) in fields.items():
        reps = stats.bench_replicates(task, np.column_stack((hit, ref_hit, core.loss_of_mass(mass), ref_loss)),
                                      **bootstrap)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = reps[:, 0] / reps[:, 1] - 1.0
        out_reps[name] = (ratio, reps[:, 2] - reps[:, 3])
    return cid, out_reps


def interval(draws):
    draws = np.asarray(draws, dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return None
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def pareto(points):
    """Indices of the points that no other point beats in both accuracy (higher) and cost (lower)."""
    keep = []
    for i, (cost, acc) in enumerate(points):
        if not any((c <= cost and a >= acc) and (c < cost or a > acc) for j, (c, a) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def pareto_distances(settings, front, all_points, log_cost):
    """Mean L1 distance of a policy's settings to the frontier on min-max normalised axes."""
    def x(cost):
        return math.log10(cost) if log_cost else cost
    xs = [x(c) for c, _ in all_points]
    ys = [a for _, a in all_points]
    span_x, span_y = (max(xs) - min(xs)) or 1.0, (max(ys) - min(ys)) or 1.0

    def norm(p):
        return (x(p[0]) - min(xs)) / span_x, (p[1] - min(ys)) / span_y
    targets = [norm(all_points[i]) for i in front]
    out = []
    for p in settings:
        px, py = norm(p)
        out.append(min(abs(px - tx) + abs(py - ty) for tx, ty in targets))
    return _mean(out)


def suite_block(suite, members, rows, frontier, models, reps):
    """Every metric of one suite."""
    bench = {(r["method"], r["catalog_id"]): r for r in rows if r["catalog_id"] in members and r["variant"] is None}
    suite_rows = {r["method"]: r for r in rows if r["suite"] == suite and r["catalog_id"] is None
                  and r["variant"] is None}
    methods = sorted(m for m in suite_rows if all((m, c) in bench for c in members))
    acc = {(int(r["catalog_id"]), int(r["model_index"])): float(r["accuracy_first_max"]) for r in models}
    loss = {(int(r["catalog_id"]), int(r["model_index"])): float(r["log_loss_normalised_floor_1e-15"]) for r in models}
    cost = {(int(r["catalog_id"]), int(r["model_index"])): float(r["usd_per_1k_requests"]) for r in models}
    p50 = {(int(r["catalog_id"]), int(r["model_index"])): float(r["latency_p50_s"]) for r in models}
    model_ids = sorted({m for _, m in acc})
    single = [dict(model_index=m, accuracy=_mean(acc[c, m] for c in members),
                   log_loss_uncalibrated=_mean(loss[c, m] for c in members),
                   usd_per_1k_requests=_mean(cost[c, m] for c in members),
                   latency_p50_s=_mean(p50[c, m] for c in members)) for m in model_ids]
    top = max(single, key=lambda s: (s["accuracy"], -s["model_index"]))
    ref_acc = {"R": {c: bench["random_single", c]["accuracy"] for c in members},
               "T": {c: acc[c, top["model_index"]] for c in members},
               "B": {c: bench[REFERENCE, c]["accuracy"] for c in members},
               "D": {c: max(acc[c, m] for m in model_ids) for c in members},
               "O": {c: bench["oracle", c]["accuracy"] for c in members}}
    references = dict(
        T=dict(model_index=top["model_index"], accuracy=top["accuracy"], usd_per_1k_requests=top["usd_per_1k_requests"],
               log_loss_uncalibrated=top["log_loss_uncalibrated"], latency_p50_s=top["latency_p50_s"]),
        D=dict(accuracy=_mean(ref_acc["D"].values()),
               log_loss_uncalibrated=_mean(min(loss[c, m] for m in model_ids) for c in members),
               models=sorted({max(model_ids, key=lambda m: (acc[c, m], -m)) for c in members})),
        B=dict(accuracy=suite_rows[REFERENCE]["accuracy"], log_loss=suite_rows[REFERENCE]["log_loss"],
               usd_per_1k_requests=suite_rows[REFERENCE]["usd_per_1k_requests"]),
        R=dict(accuracy=suite_rows["random_single"]["accuracy"], log_loss=suite_rows["random_single"]["log_loss"]),
        O=dict(accuracy=suite_rows["oracle"]["accuracy"], log_loss=suite_rows["oracle"]["log_loss"]))
    for ref in ("T", "D"):
        a = ref_acc[ref]
        references[ref].update(gain_b=_mean(a[c] / ref_acc["B"][c] - 1 for c in members),
                               gain_t=_mean(a[c] / ref_acc["T"][c] - 1 for c in members),
                               gap_o=_mean(1 - a[c] / ref_acc["O"][c] for c in members))
    block = {}
    for m in methods:
        a = {c: bench[m, c]["accuracy"] for c in members}
        r = suite_rows[m]
        entry = dict(avg_acc=r["accuracy"], log_loss=r["log_loss"], usd_per_1k_requests=r["usd_per_1k_requests"],
                     latency_mean_s=r["latency_mean_s"], calls_per_request=r["calls_per_request"],
                     gain_r=_mean(a[c] / ref_acc["R"][c] - 1 for c in members),
                     gain_t=_mean(a[c] / ref_acc["T"][c] - 1 for c in members),
                     gain_b=_mean(a[c] / ref_acc["B"][c] - 1 for c in members),
                     gain_d=_mean(a[c] / ref_acc["D"][c] - 1 for c in members),
                     gap_o=_mean(1 - a[c] / ref_acc["O"][c] for c in members),
                     log_loss_minus_b=r["log_loss"] - suite_rows[REFERENCE]["log_loss"])
        if all(m in reps.get(c, {}) for c in members):
            ratio = np.mean(np.stack([reps[c][m][0] for c in members]), axis=0)
            delta = np.mean(np.stack([reps[c][m][1] for c in members]), axis=0)
            entry["gain_b_ci"] = interval(ratio) if m != REFERENCE else [0.0, 0.0]
            entry["log_loss_minus_b_ci"] = interval(delta) if m != REFERENCE else [0.0, 0.0]
        block[m] = entry
    # settings of every policy: (cost, accuracy, log loss, variant)
    grids = {}
    for r in frontier:
        if r["suite"] == suite and r["catalog_id"] is None and r["method"] in GRID_POLICIES:
            grids.setdefault(r["method"], []).append(
                (r["usd_per_1k_requests"], r["accuracy"], r["log_loss"], r["variant"]))
    for m in methods:
        if m not in grids:
            r = suite_rows[m]
            grids[m] = [(r["usd_per_1k_requests"], r["accuracy"], r["log_loss"], None)]
    in_front = [(p[0], p[1], name, p[3]) for name, pts in grids.items() if name not in EXCLUDED_FROM_FRONTIER for p in pts]
    in_front += [(s["usd_per_1k_requests"], s["accuracy"], f"model_{s['model_index']}", None) for s in single]
    all_points = [(c, a) for c, a, _, _ in in_front]
    front = pareto(all_points)
    tb = {"T": (references["T"]["accuracy"], references["T"]["usd_per_1k_requests"]),
          "B": (references["B"]["accuracy"], references["B"]["usd_per_1k_requests"])}
    settings_block = {}
    for name, pts in sorted(grids.items()):
        best_acc = max(pts, key=lambda p: (p[1], -p[0]))
        best_ll = min(pts, key=lambda p: (p[2], p[0]))
        entry = dict(settings=len(pts), best_accuracy=best_acc[1], best_accuracy_variant=best_acc[3],
                     best_accuracy_cost=best_acc[0], best_log_loss=best_ll[2], best_log_loss_variant=best_ll[3],
                     cheapest_cost=min(p[0] for p in pts))
        for ref, (ref_a, ref_c) in tb.items():
            entry[f"perf_gain_{ref}"] = best_acc[1] / ref_a - 1
            ok = [p for p in pts if p[1] >= ref_a]
            entry[f"cost_save_{ref}"] = (1 - min(p[0] for p in ok) / ref_c) if ok else None
        ref_loss, ref_cost = references["B"]["log_loss"], references["B"]["usd_per_1k_requests"]
        entry["loss_gain_B"] = 1 - best_ll[2] / ref_loss
        ok = [p for p in pts if p[2] <= ref_loss]
        entry["cost_save_loss_B"] = (1 - min(p[0] for p in ok) / ref_cost) if ok else None
        if name not in EXCLUDED_FROM_FRONTIER:
            entry["pareto_dist"] = pareto_distances([(p[0], p[1]) for p in pts], front, all_points, log_cost=True)
            entry["pareto_dist_linear"] = pareto_distances([(p[0], p[1]) for p in pts], front, all_points,
                                                           log_cost=False)
        settings_block[name] = entry
    return dict(benchmarks=len(members), references=references, methods=block, policies=settings_block,
                frontier=sorted(([in_front[i][0], in_front[i][1], in_front[i][2], in_front[i][3]] for i in front),
                                key=lambda p: (p[0], -p[1])),
                single_models=single)


def run(corpus, results, settings_path, processes=1):
    corpus, results = Path(corpus), Path(results)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    document = json.loads((results / "rows.json").read_text(encoding="utf-8"))
    if document["settings_sha256"] != digest:
        raise SystemExit("rows.json was made with other settings; run the experiments first")
    rows = document["rows"]
    frontier = json.loads((results / "frontier.json").read_text(encoding="utf-8"))["rows"]
    models = _csv(results / "diagnostics" / "models.csv")
    cids = sorted({r["catalog_id"] for r in rows if r["catalog_id"] is not None})
    methods = sorted({r["method"] for r in rows if r["catalog_id"] is not None and r["variant"] is None})
    work = partial(replicates, corpus=corpus, out=results, methods=methods, bootstrap=settings["bootstrap"])
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            reps = dict(pool.map(work, cids))
    else:
        reps = dict(map(work, cids))
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in cids if c not in set(definition["exclude"])]
        if members:
            suites[suite] = suite_block(suite, members, rows, frontier, models, reps)
    RX._json(results / "benchmark_metrics.json", dict(
        synthetic=document["synthetic"], settings_sha256=digest, draws=settings["bootstrap"]["draws"],
        seed=settings["bootstrap"]["seed"], grid_policies=list(GRID_POLICIES),
        excluded_from_frontier=list(EXCLUDED_FROM_FRONTIER), suites=suites))
    return suites


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark metrics against random, top, best-single and oracle references")
    for name in ("corpus", "results", "settings"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    suites = run(a.corpus, a.results, a.settings, a.processes)
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    block = suites[headline]
    print("references", json.dumps(block["references"], sort_keys=True)[:400])
    for m, e in sorted(block["methods"].items()):
        print(f"{m:26s} acc {100 * e['avg_acc']:6.2f} gainT {100 * e['gain_t']:+6.2f} gainB {100 * e['gain_b']:+6.2f}"
              f" {e.get('gain_b_ci')} gapO {100 * e['gap_o']:6.2f}")
    for m, e in sorted(block["policies"].items()):
        print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

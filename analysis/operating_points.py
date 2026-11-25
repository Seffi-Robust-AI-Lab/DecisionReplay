"""Operating points of the router, the cascade and the per-benchmark control.

    python3 -B -m analysis.operating_points --corpus DIR --settings FILE --out RESULTS \
        [--project-root .] [--processes N]

The main table reports the router and the cascade at one operating point per rotation: the grid value with the
lowest validation loss among those whose validation cost does not exceed that of the best single model
("capped").  This module reports what that rule chose, and replays the same policies under a second rule that
ignores cost: the grid value with the lowest validation loss ("uncapped").  It does so for

* ``knn_router``               the router with numeric request features;
* ``knn_router_text``          the router with text embeddings (only where the local features exist);
* ``single_by_cost``           the control: one cost-penalised model per benchmark;
* ``confidence_cascade``       the cascade, confidence = lowest top probability among the request's fields;
* ``confidence_cascade_mean``  the cascade, confidence = mean top probability.

It writes ``RESULTS/operating_points.json``:

``rows``     benchmark and suite rows (method = policy, variant = rule), in the row schema of the other files;
``suites``   per suite: ``points`` (the suite rows by policy and rule), ``selection`` (how often each grid value
             was chosen over the benchmark-rotation cells, how often no value was within the budget, how often
             the cap changed the choice), ``features`` (how many numeric request features vary inside a
             benchmark), ``router_by_features`` (router minus best single model on the benchmarks with at
             least two varying features and on the others) and ``contrasts`` (paired group bootstrap, log loss
             and accuracy; descriptive, not adjusted).

With ``--project-root`` the benchmarks' own metric is added to the rows; it needs the benchmark rows and the
pinned kit.  Every analysis here was added after the first run on the corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from data.schema import N_FOLDS, sha256_file
from sim import metrics
from sim.baselines import best_single, confidence_cascade, knn_router
from sim.engine import load_task
from . import local_methods as LM
from . import run_experiments as RX

RULES = ("capped", "uncapped")
REFERENCE = "best_single"
CONTRASTS = (                       # name, A, B; A and B are "policy.rule" or the best single model
    ("router_capped_vs_single", "knn_router.capped", REFERENCE),
    ("router_uncapped_vs_single", "knn_router.uncapped", REFERENCE),
    ("text_router_capped_vs_single", "knn_router_text.capped", REFERENCE),
    ("text_router_uncapped_vs_single", "knn_router_text.uncapped", REFERENCE),
    ("control_capped_vs_single", "single_by_cost.capped", REFERENCE),
    ("control_uncapped_vs_single", "single_by_cost.uncapped", REFERENCE),
    ("router_capped_vs_control", "knn_router.capped", "single_by_cost.capped"),
    ("router_uncapped_vs_control", "knn_router.uncapped", "single_by_cost.uncapped"),
    ("text_router_capped_vs_control", "knn_router_text.capped", "single_by_cost.capped"),
    ("text_router_uncapped_vs_control", "knn_router_text.uncapped", "single_by_cost.uncapped"),
    ("cascade_capped_vs_single", "confidence_cascade.capped", REFERENCE),
    ("cascade_uncapped_vs_single", "confidence_cascade.uncapped", REFERENCE),
    ("mean_cascade_capped_vs_single", "confidence_cascade_mean.capped", REFERENCE),
    ("mean_cascade_uncapped_vs_single", "confidence_cascade_mean.uncapped", REFERENCE),
    ("mean_cascade_vs_cascade", "confidence_cascade_mean.capped", "confidence_cascade.capped"),
)
POINT_KEYS = ("log_loss", "log_loss_floor_1e-3", "accuracy", "native", "calls_per_request", "usd_per_1k_requests",
              "latency_mean_s", "latency_p50_s", "latency_p95_s", "n_benchmarks")


def policies(settings, with_text):
    """{policy: (run function, settings, name of the parameter, key of the chosen values in ``info``)}."""
    control = dict(settings, knn_router=dict(settings["knn_router"], k_grid=[RX.ALL_FIT_REQUESTS]))
    mean = dict(settings, confidence_cascade=dict(settings["confidence_cascade"], confidence="mean"))
    out = {"knn_router": (knn_router.run, settings, "penalty", "lambda"),
           "single_by_cost": (knn_router.run, control, "penalty", "lambda"),
           "confidence_cascade": (confidence_cascade.run, settings, "threshold", "tau"),
           "confidence_cascade_mean": (confidence_cascade.run, mean, "threshold", "tau")}
    if with_text:
        out["knn_router_text"] = (knn_router.run, LM.text_router_settings(settings), "penalty", "lambda")
    return out


def lowest_loss(record):
    """The grid value with the lowest validation loss, whatever it costs (ties: the earlier grid entry)."""
    return record["grid"][min(range(len(record["grid"])), key=lambda i: (record["losses"][i], i))]


def varying_features(bench):
    """How many of the numeric request features take more than one value in the benchmark."""
    x = knn_router.request_features(bench, "numeric")
    return int((x.max(axis=0) > x.min(axis=0)).sum())


def one_benchmark(cid, corpus, settings, root, digest, with_text):
    """Rows, per-rotation choices and paired replicates of one benchmark."""
    task = load_task(corpus, cid)
    reference = metrics.score(task, best_single.run(task, settings))
    fields = {REFERENCE: (reference.mass, reference.hit)}
    rows, chosen = [], {}
    for name, (run, config, parameter, key) in policies(settings, with_text).items():
        capped = run(task, config)
        records = capped.info["selection"]
        free = [lowest_loss(r) for r in records]
        outcomes = {"capped": capped, "uncapped": run(task, config, **{parameter: free})}
        for rule, outcome in outcomes.items():
            scored = metrics.score(task, outcome)
            row = RX._result_row(task, scored, name, rule, digest)
            if root is not None:
                row["native"] = LM.native.native_score(root, Path(root) / LM.ROWS, task.b, outcome.pooled)
            rows.append(row)
            fields[f"{name}.{rule}"] = scored.mass, scored.hit
        chosen[name] = dict(
            grid=records[0]["grid"], capped=[float(v) for v in capped.info[key]], uncapped=[float(v) for v in free],
            none_within_budget=[r["within_budget"] == 0 for r in records],
            k=[int(k) for k in capped.info.get("k", [])],
            stages=[len(chain) for chain in capped.info.get("chain", [])])
    specs = [dict(name=n, a=a, b=b) for n, a, b in CONTRASTS if a in fields and b in fields]
    paired = RX._paired_benchmark(task, fields, specs, settings)
    return cid, rows, chosen, paired, varying_features(task.b)


def selection_summary(chosen, cids):
    """Counts over the benchmark-rotation cells of a suite, per policy."""
    out = {}
    for name in sorted({n for c in cids for n in chosen[c]}):
        cells = [(chosen[c][name], f) for c in cids for f in range(N_FOLDS)]
        grid = chosen[cids[0]][name]["grid"]
        block = dict(
            cells=len(cells), grid=grid,
            chosen=[sum(c["capped"][f] == g for c, f in cells) for g in grid],
            chosen_uncapped=[sum(c["uncapped"][f] == g for c, f in cells) for g in grid],
            none_within_budget=sum(bool(c["none_within_budget"][f]) for c, f in cells),
            cap_changed_choice=sum(c["capped"][f] != c["uncapped"][f] for c, f in cells))
        if chosen[cids[0]][name]["k"]:
            values = sorted({k for c in cids for k in chosen[c][name]["k"]})
            block["k_grid"] = values
            block["k_chosen"] = [sum(c["k"][f] == k for c, f in cells) for k in values]
        if chosen[cids[0]][name]["stages"]:
            block["cells_by_stages"] = [sum(c["stages"][f] == n for c, f in cells) for n in (1, 2, 3)]
        out[name] = block
    return out


def by_features(rows, features, cids):
    """Router minus best single model on the benchmarks with at least two varying numeric features, and the rest."""
    cell = {(r["method"], r["variant"], r["catalog_id"]): r for r in rows if r["catalog_id"] is not None}
    groups = {"two_or_more": [c for c in cids if features[c] >= 2], "at_most_one": [c for c in cids if features[c] < 2]}
    out = {}
    for label, members in groups.items():
        block = dict(benchmarks=len(members))
        for rule in RULES:
            for metric in ("log_loss", "accuracy"):
                gaps = [cell["knn_router", rule, c][metric] - cell[REFERENCE, None, c][metric] for c in members]
                block[f"{rule}_{metric}_difference"] = math.fsum(gaps) / len(gaps) if gaps else None
        out[label] = block
    return out


def run(corpus, settings_path, out, root=None, processes=1):
    corpus, out = Path(corpus), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    document = json.loads((out / "rows.json").read_text(encoding="utf-8"))
    if document["settings_sha256"] != digest:
        raise SystemExit("rows.json was made with other settings; run the experiments first")
    features = Path(settings.get(LM.TEXT_ROUTER, {}).get("features", ""))
    with_text = (LM.TEXT_ROUTER in settings and root is not None
                 and (Path(root) / features / "features.json").is_file())
    reference = [r for r in document["rows"] if r["catalog_id"] is not None and r["method"] == REFERENCE
                 and r["variant"] is None]
    cids = sorted(r["catalog_id"] for r in reference)
    work = partial(one_benchmark, corpus=corpus, settings=settings, root=root, digest=digest, with_text=with_text)
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            results = list(pool.map(work, cids))
    else:
        results = [work(c) for c in cids]
    rows, chosen, paired, varying = [], {}, {}, {}
    for cid, new_rows, choice, pairs, count in results:
        rows.extend(new_rows)
        chosen[cid], paired[cid], varying[cid] = choice, pairs, count
    names = {r["method"] for r in rows}
    specs = [dict(name=n, a=a, b=b, in_family=False) for n, a, b in CONTRASTS
             if a.split(".")[0] in names and (b == REFERENCE or b.split(".")[0] in names)]
    contrast_rows = RX._contrast_rows(specs, settings, digest, paired)
    all_rows = RX._with_suites(rows, settings["suites"])
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in cids if c not in set(definition["exclude"])]
        if not members:
            continue
        points = {}
        for r in all_rows:
            if r["catalog_id"] is None and r["suite"] == suite:
                points.setdefault(r["method"], {})[r["variant"]] = {k: r[k] for k in POINT_KEYS}
        contrasts = {}
        for r in contrast_rows:
            if r["suite"] == suite:
                contrasts.setdefault(r["contrast"], {})[r["metric"]] = {
                    k: r[k] for k in ("a", "b", "difference", "ci_low", "ci_high", "p", "benchmarks_better",
                                      "benchmarks_same", "benchmarks_worse", "n_benchmarks", "draws")}
        counts = [sum(varying[c] == n for c in members) for n in range(6)]
        suites[suite] = dict(
            points=points, contrasts=contrasts, selection=selection_summary(chosen, members),
            features=dict(benchmarks=len(members), benchmarks_by_varying_features=counts,
                          two_or_more=sum(counts[2:]), at_most_one=sum(counts[:2])),
            router_by_features=by_features(rows + reference, varying, members))
    RX._json(out / "operating_points.json", dict(
        synthetic=document["synthetic"], settings_sha256=digest, rules=list(RULES), rows=all_rows, suites=suites))
    return suites


def main(argv=None):
    ap = argparse.ArgumentParser(description="Operating points of the router, the cascade and the control")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--project-root")
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    suites = run(a.corpus, a.settings, a.out, a.project_root, a.processes)
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    for name, rules in sorted(suites[headline]["points"].items()):
        print(name, {rule: (round(p["log_loss"], 4), round(p["usd_per_1k_requests"], 4)) for rule, p in rules.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

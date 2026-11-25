"""The methods that the simulator fits, on redrawn folds.

    python3 -B -m analysis.fold_redraws --corpus DIR --settings FILE --out RESULTS \
        [--seeds 1 2 3] [--processes N]

The corpus has one frozen assignment of source groups to five folds, and earlier studies used it.  This module
draws new assignments and reruns every built-in method of the settings on each: the source groups of a benchmark
are shuffled (seeded by the draw and the catalogue id) and dealt into five folds of nearly equal numbers of
groups, so a group still lies in one fold.  It then reports, per suite, each method's row on every draw beside
its row on the frozen folds, and the contrasts between built-in methods on every draw.

Methods that are replayed from stored weights or vectors of earlier studies cannot be refitted and are left out.
Writes ``RESULTS/fold_redraws.json``.  Added after the first run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from data.load import load_benchmark
from data.schema import N_FOLDS, sha256_file
from sim import metrics
from sim.baselines import BASELINES
from sim.engine import Task
from . import run_experiments as RX

SEEDS = (1, 2, 3)
METRICS = ("log_loss", "accuracy", "usd_per_1k_requests")


def redraw(bench, seed):
    """The benchmark with its source groups dealt into new folds."""
    groups, index = np.unique(bench.split_group, return_inverse=True)
    order = np.random.default_rng([seed, bench.cid]).permutation(len(groups))
    fold = np.empty(len(groups), dtype=bench.fold.dtype)
    fold[order] = np.arange(len(groups)) % N_FOLDS
    return dataclasses.replace(bench, fold=fold[index])


def one_benchmark(cid, corpus, settings, digest, seeds, methods):
    """Benchmark rows of every method on every draw."""
    bench = load_benchmark(corpus, cid, labels=True)
    rows = []
    for seed in seeds:
        task = Task(redraw(bench, seed))
        for name in methods:
            scored = metrics.score(task, BASELINES[name].run(task, settings))
            rows.append(RX._result_row(task, scored, name, f"draw={seed}", digest))
    return rows


def run(corpus, settings_path, out, seeds=SEEDS, processes=1):
    corpus, out = Path(corpus), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    document = json.loads((out / "rows.json").read_text(encoding="utf-8"))
    if document["settings_sha256"] != digest:
        raise SystemExit("rows.json was made with other settings; run the experiments first")
    methods = [m for m in settings["methods"] if m in BASELINES]
    frozen = {(r["method"], r["suite"]): r for r in document["rows"]
              if r["catalog_id"] is None and r["variant"] is None and r["method"] in methods}
    cids = sorted({r["catalog_id"] for r in document["rows"] if r["catalog_id"] is not None})
    work = partial(one_benchmark, corpus=corpus, settings=settings, digest=digest, seeds=tuple(seeds),
                   methods=methods)
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            results = list(pool.map(work, cids))
    else:
        results = [work(c) for c in cids]
    rows = RX._with_suites([r for part in results for r in part], settings["suites"])
    drawn = {(r["method"], r["variant"], r["suite"]): r for r in rows if r["catalog_id"] is None}
    variants = [f"draw={s}" for s in seeds]
    suites = {}
    for suite in settings["suites"]:
        if (methods[0], variants[0], suite) not in drawn:
            continue
        block = {}
        for name in methods:
            block[name] = {}
            for metric in METRICS:
                values = [drawn[name, v, suite][metric] for v in variants]
                block[name][metric] = dict(frozen=frozen[name, suite][metric], draws=values, low=min(values),
                                           high=max(values), mean=math.fsum(values) / len(values))
        contrasts = {}
        for spec in settings["contrasts"]:
            if spec["a"] not in methods or spec["b"] not in methods:
                continue
            contrasts[spec["name"]] = {}
            for metric in settings["contrast_metrics"]:
                values = [drawn[spec["a"], v, suite][metric] - drawn[spec["b"], v, suite][metric] for v in variants]
                stored = frozen[spec["a"], suite][metric] - frozen[spec["b"], suite][metric]
                contrasts[spec["name"]][metric] = dict(
                    frozen=stored, draws=values, low=min(values), high=max(values),
                    same_sign_as_frozen=all((v > 0) == (stored > 0) and (v < 0) == (stored < 0) for v in values))
        largest = max(max(abs(v - b["log_loss"]["frozen"]) for v in b["log_loss"]["draws"]) for b in block.values())
        suites[suite] = dict(
            methods=block, contrasts=contrasts, largest_log_loss_shift=largest,
            contrasts_compared=len(contrasts),
            contrasts_keeping_sign=sum(c["log_loss"]["same_sign_as_frozen"] for c in contrasts.values()))
    RX._json(out / "fold_redraws.json", dict(
        synthetic=document["synthetic"], settings_sha256=digest, seeds=list(seeds), draws=len(seeds),
        methods=methods, rows=[r for r in rows if r["catalog_id"] is None], suites=suites))
    return suites


def main(argv=None):
    ap = argparse.ArgumentParser(description="Built-in methods on redrawn folds")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    suites = run(a.corpus, a.settings, a.out, a.seeds, a.processes)
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    block = suites[headline]
    print(f"{block['contrasts_keeping_sign']} of {block['contrasts_compared']} contrasts keep their sign; "
          f"largest shift of a row {block['largest_log_loss_shift']:.4f} nats")
    return 0


if __name__ == "__main__":
    sys.exit(main())

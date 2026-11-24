"""Reference values for the router's recall at k: a fixed ranking per benchmark, and chance.

    python3 -B -m acl_resource.analysis.recall_references --corpus DIR --settings FILE --results RESULTS --out OUT \
        [--processes N]

``router_study`` reports, on the judged fields that one to three models get right, how often a right model is
among the k models that the router scores highest for the field's request.  That share has no reference in the
paper.  This module adds three, with the same weighting (every field weighs 1 / its benchmark's number of judged
fields) and the same k (1, 3, 5):

``fixed_ranking``  the ranking that ignores the request: the models of the benchmark ordered by their calibrated
                   log loss on the three fit folds (``best_single.ranking``), one ranking per test fold;
``control``        the router's own control, which scores every model on all fit requests and subtracts the cost
                   penalty that validation selects under the cost cap (``router_control`` of ``router_study``);
``chance``         k models drawn uniformly without replacement: 1 - C(M - r, k) / C(M, k) on a field with r right
                   models.

It reads ``RESULTS/router_study.json`` to print the routers' recall beside the references and writes
``OUT/recall_references.json``.  The design was fixed before the first run on the corpus.  Only aggregates are
written.  Added after the first run, on 23 November 2026, in answer to a review.
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

from ..data.schema import N_FOLDS, sha256_file
from ..sim.baselines import best_single, knn_router
from ..sim.engine import load_task
from . import run_experiments as RX
from .hardness import FEW_RIGHT, model_hits
from .router_study import RECALL_K, rankings, recall, settings_for

SUBSETS = ("few_right", "any_right")


def fixed_positions(task):
    """[R, M] rank of every model under the fit-fold log-loss ranking of the request's test fold."""
    position = np.empty((task.R, task.b.M), np.int64)
    for f in range(N_FOLDS):
        rank = np.empty(task.b.M, np.int64)
        rank[best_single.ranking(task, f, n=3)] = np.arange(task.b.M)
        position[task.b.fold == f] = rank
    return position


def chance(task, hits):
    """Expected hits of k uniformly drawn models on the two field subsets."""
    M = task.b.M
    right = hits.sum(axis=0)
    out = {}
    for label, mask in (("few_right", (right >= 1) & (right <= FEW_RIGHT)), ("any_right", right >= 1)):
        counts = np.bincount(right[mask], minlength=M + 1)
        block = dict(fields=int(mask.sum()), judged=int(task.nJ))
        for k in RECALL_K:
            block[f"at_{k}"] = math.fsum(int(counts[r]) * (1.0 - math.comb(M - r, k) / math.comb(M, k))
                                         for r in range(1, M + 1) if counts[r])
        out[label] = block
    return out


def one_benchmark(cid, corpus, settings):
    task = load_task(corpus, cid)
    hits = model_hits(task)
    config = settings_for(settings, "numeric", control=True)
    capped = knn_router.run(task, config)
    return cid, dict(fixed_ranking=recall(task, hits, fixed_positions(task)),
                     control=recall(task, hits, rankings(task, config, capped.info["lambda"])),
                     chance=chance(task, hits))


def summarise(parts, cids):
    out = {}
    for name in ("fixed_ranking", "control", "chance"):
        out[name] = {}
        for subset in SUBSETS:
            usable = [parts[c][name][subset] for c in cids if parts[c][name][subset]["fields"]]
            mass = math.fsum(u["fields"] / u["judged"] for u in usable)
            out[name][subset] = dict(
                benchmarks=len(usable), fields=sum(u["fields"] for u in usable),
                **{f"at_{k}": (math.fsum(u[f"at_{k}"] / u["judged"] for u in usable) / mass if mass else None)
                   for k in RECALL_K})
    return out


def run(corpus, settings_path, results, out, processes=1):
    corpus, results, out = Path(corpus), Path(results), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    document = json.loads((results / "rows.json").read_text(encoding="utf-8"))
    if document["settings_sha256"] != digest:
        raise SystemExit("rows.json was made with other settings; run the experiments first")
    cids = sorted({r["catalog_id"] for r in document["rows"] if r["catalog_id"] is not None})
    work = partial(one_benchmark, corpus=corpus, settings=settings)
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            parts = dict(pool.map(work, cids))
    else:
        parts = dict(map(work, cids))
    routers = {}
    study = results / "router_study.json"
    if study.is_file():
        stored = json.loads(study.read_text(encoding="utf-8"))
        if stored["settings_sha256"] == digest:
            routers = {suite: {name: {subset: {k: v for k, v in rules["capped"][subset].items() if k.startswith("at_")}
                                      for subset in SUBSETS}
                               for name, rules in block["recall"].items() if "capped" in rules}
                       for suite, block in stored["suites"].items()}
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in cids if c not in set(definition["exclude"])]
        if members:
            suites[suite] = dict(benchmarks=len(members), references=summarise(parts, members),
                                 routers_capped=routers.get(suite, {}))
    RX._json(out / "recall_references.json", dict(
        synthetic=document["synthetic"], settings_sha256=digest, few_right=FEW_RIGHT, recall_k=list(RECALL_K),
        suites=suites))
    return suites


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reference values for the router's recall at k")
    for name in ("corpus", "settings", "results", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    suites = run(a.corpus, a.settings, a.results, a.out, a.processes)
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    block = suites[headline]
    for name, subsets in {**block["routers_capped"], **block["references"]}.items():
        print(name, {k: round(v, 3) for k, v in subsets["few_right"].items() if k.startswith("at_")})
    return 0


if __name__ == "__main__":
    sys.exit(main())

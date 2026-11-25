"""How hard each judged field is for the panel, what each method gets on hard fields, and what more models buy.

    python3 -B -m analysis.hardness --corpus DIR --settings FILE --out RESULTS [--processes N]

It needs the local corpus and the per-field archives ``RESULTS/local/fields-<method>.npz`` that
``run_experiments`` and ``local_methods`` write.  For every judged field it counts how many of the models are
right, decoding each model's own vector as the simulator decodes any vector (the first maximum; on a yes/no field
"yes" when p(yes) >= 0.5).  It writes ``RESULTS/hardness.json`` with, per suite:

``bins``       for every count 0..M: the share of judged fields and each method's accuracy, benchmarks weighted
               equally (every field of a benchmark weighs 1 / its number of judged fields), and the pooled counts;
``few_right``  the same for the fields on which one to three models are right;
``subsets``    the accuracy of a per-request oracle restricted to k models, k = 1..M, for the k models ranked best
               on the fit folds (the ranking of the ``ranked_top_k`` panels) and for random subsets (mean and the
               5th and 95th percentiles over seeded draws of the equal-benchmark mean).

Checks that stop the run: each archive's hits reproduce the benchmark's accuracy in ``rows.json``; no method that
answers every request with one model's vector (best single, random model, oracle, routers, cascade) is right on a
field that no model gets right; the oracle restricted to all M models reproduces the stored oracle.  Only
aggregates are written.  Added after the first run.
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
from sim import core
from sim.baselines import best_single
from sim.engine import load_task
from . import run_experiments as RX

PICK_ONE = ("best_single", "random_single", "oracle", "knn_router", "knn_router_text", "confidence_cascade")
FEW_RIGHT = 3
DRAWS = 200
SEED = 20261121
TOLERANCE = 1e-12


def model_hits(task):
    """[M, judged fields] bool: is each model's own decoded option accepted, in the task's judged-field order."""
    b = task.b
    out = np.empty((b.M, task.nJ), bool)
    for m in range(b.M):
        hit, _, _, _ = core.decode(b, core.single_model_vectors(b, m))
        out[m] = hit[task.J]
    return out


def request_counts(task, hits):
    """Right fields per model and request (requests with judged fields only) and the fold of each request."""
    requests, inverse = np.unique(task.b.field_row[task.J], return_inverse=True)
    counts = np.stack([np.bincount(inverse, weights=h, minlength=len(requests)) for h in hits.astype(np.float64)])
    return counts, task.b.fold[requests]


def subset_curves(task, counts, request_fold, draws=DRAWS, seed=SEED):
    """Oracle accuracy of the top-k and of random k-model subsets, k = 1..M (rows of the random draws)."""
    M = counts.shape[0]
    top = np.zeros(M)
    random = np.zeros((draws, M))
    for f in range(N_FOLDS):
        sel = np.flatnonzero(request_fold == f)
        if not len(sel):
            continue
        block = counts[:, sel]
        order = best_single.ranking(task, f)
        top += np.maximum.accumulate(block[order], axis=0).sum(axis=1)
        for d in range(draws):
            perm = np.random.default_rng([seed, task.cid, f, d]).permutation(M)
            random[d] += np.maximum.accumulate(block[perm], axis=0).sum(axis=1)
    return top / task.nJ, random / task.nJ


def load_fields(out, methods, cid, n):
    """{method: hits} from the local archives that hold this benchmark."""
    fields = {}
    for name in methods:
        path = RX._field_file(out, name)
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            key = f"hit_{cid}"
            if key not in archive.files:
                continue
            hit = archive[key]
        if hit.shape != (n,):
            raise ValueError(f"{name}: field count differs on benchmark {cid}")
        fields[name] = hit.astype(bool)
    return fields


def one_benchmark(cid, corpus, out, methods, stored, draws, seed):
    """Histogram of models right, method hits by count, subset curves and the checks of one benchmark."""
    task = load_task(corpus, cid)
    hits = model_hits(task)
    right = hits.sum(axis=0)
    M = task.b.M
    fields = load_fields(out, methods, cid, task.nJ)
    for name, hit in fields.items():
        accuracy = stored.get((name, cid))
        if accuracy is None or abs(accuracy - hit.mean()) > TOLERANCE:
            raise ValueError(f"{name}: archived hits do not reproduce rows.json on benchmark {cid}")
        if name in PICK_ONE and np.any(hit & (right == 0)):
            raise ValueError(f"{name}: right on a field no model gets right (benchmark {cid}); fields misaligned")
    counts, request_fold = request_counts(task, hits)
    oracle = counts.max(axis=0).sum() / task.nJ
    if ("oracle", cid) in stored and abs(oracle - stored["oracle", cid]) > TOLERANCE:
        raise ValueError(f"the oracle over all models does not reproduce rows.json on benchmark {cid}")
    top, random = subset_curves(task, counts, request_fold, draws, seed)
    histogram = np.bincount(right, minlength=M + 1)
    by_count = {name: np.bincount(right, weights=hit.astype(np.float64), minlength=M + 1)
                for name, hit in fields.items()}
    return cid, dict(n=task.nJ, M=M, histogram=histogram, by_count=by_count, top=top, random=random,
                     oracle=float(oracle))


def aggregate(parts, cids, methods):
    """Equal-benchmark and pooled summaries over the given benchmarks."""
    M = parts[cids[0]]["M"]
    weight = {c: 1.0 / parts[c]["n"] for c in cids}
    present = [m for m in methods if all(m in parts[c]["by_count"] for c in cids)]
    bins = []
    for k in range(M + 1):
        mass = math.fsum(parts[c]["histogram"][k] * weight[c] for c in cids)
        pooled = int(sum(int(parts[c]["histogram"][k]) for c in cids))
        row = dict(models_right=k, share=mass / len(cids), fields=pooled,
                   benchmarks=sum(int(parts[c]["histogram"][k] > 0) for c in cids), accuracy={}, pooled_accuracy={})
        for m in present:
            right = math.fsum(parts[c]["by_count"][m][k] * weight[c] for c in cids)
            row["accuracy"][m] = right / mass if mass else None
            hits = math.fsum(parts[c]["by_count"][m][k] for c in cids)
            row["pooled_accuracy"][m] = hits / pooled if pooled else None
        bins.append(row)

    def band(low, high):
        mass = math.fsum(parts[c]["histogram"][low:high + 1].sum() * weight[c] for c in cids)
        pooled = int(sum(int(parts[c]["histogram"][low:high + 1].sum()) for c in cids))
        block = dict(models_right_min=low, models_right_max=high, share=mass / len(cids), fields=pooled,
                     accuracy={}, pooled_accuracy={})
        for m in present:
            right = math.fsum(parts[c]["by_count"][m][low:high + 1].sum() * weight[c] for c in cids)
            block["accuracy"][m] = right / mass if mass else None
            hits = math.fsum(parts[c]["by_count"][m][low:high + 1].sum() for c in cids)
            block["pooled_accuracy"][m] = hits / pooled if pooled else None
        return block

    top = np.mean([parts[c]["top"] for c in cids], axis=0)
    random = np.mean([parts[c]["random"] for c in cids], axis=0)          # [draws, M]: equal-benchmark mean per draw
    subsets = dict(sizes=list(range(1, M + 1)), top_k=[float(v) for v in top],
                   random_mean=[float(v) for v in random.mean(axis=0)],
                   random_low=[float(v) for v in np.percentile(random, 5, axis=0)],
                   random_high=[float(v) for v in np.percentile(random, 95, axis=0)],
                   oracle_all=float(np.mean([parts[c]["oracle"] for c in cids])))
    return dict(benchmarks=len(cids), methods=present, bins=bins, few_right=band(1, FEW_RIGHT),
                none_right=band(0, 0), all_right=band(M, M), subsets=subsets)


def run(corpus, settings_path, out, processes=1, draws=DRAWS, seed=SEED):
    corpus, out = Path(corpus), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    document = json.loads((out / "rows.json").read_text(encoding="utf-8"))
    if document["settings_sha256"] != digest:
        raise SystemExit("rows.json was made with other settings; run the experiments first")
    rows = [r for r in document["rows"] if r["catalog_id"] is not None and r["variant"] is None]
    stored = {(r["method"], r["catalog_id"]): r["accuracy"] for r in rows}
    methods = sorted({r["method"] for r in rows})
    cids = sorted({r["catalog_id"] for r in rows})
    work = partial(one_benchmark, corpus=corpus, out=out, methods=methods, stored=stored, draws=draws, seed=seed)
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            parts = dict(pool.map(work, cids))
    else:
        parts = dict(map(work, cids))
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in cids if c not in set(definition["exclude"])]
        if members:
            suites[suite] = aggregate(parts, members, methods)
    RX._json(out / "hardness.json", dict(
        synthetic=document["synthetic"], settings_sha256=digest, few_right=FEW_RIGHT, draws=draws, seed=seed,
        pick_one=[m for m in PICK_ONE if m in methods], suites=suites))
    return suites


def main(argv=None):
    ap = argparse.ArgumentParser(description="Field hardness and oracle subsets")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    suites = run(a.corpus, a.settings, a.out, a.processes)
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    block = suites[headline]
    few = block["few_right"]
    print(f"{block['benchmarks']} benchmarks; fields with 1-{FEW_RIGHT} models right: {few['share']:.3f}; "
          + ", ".join(f"{m} {v:.3f}" for m, v in sorted(few["accuracy"].items()) if v is not None))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The router with text representations whose transforms are fitted on the fit folds only.

    python3 -B -m acl_resource.analysis.router_refit_features --corpus DIR --settings FILE --results RESULTS \
        --features DIR --rows DIR --out OUT [--processes N]

``featurize_requests`` fits the TF-IDF vocabulary and its truncated SVD, and the PCA of the language-model states,
on all requests of a benchmark.  They read no label, but they see the requests of the test fold.  A reviewer asked
whether that matters.  For every benchmark and rotation this module refits both transforms on the requests of the
three fit folds only, applies them to all requests of the benchmark, and reruns the router under the capped rule
of the main table (same grids, same selection).  The sentence-encoder features have no fitted transform and are
not concerned.  The design was fixed before the first run on the corpus.

It writes ``OUT/router_refit_features.json`` with, per suite and representation: the router's row with the stored
(all-request) features and with the refitted ones, the paired difference in log loss and in accuracy with the
within-benchmark interval of ``sim.stats``, and the share of requests that are sent to another model.

Check that stops the run: the router with the stored features reproduces ``router_<representation>`` (capped) of
``RESULTS/router_study.json``.  The request text is read to build the TF-IDF features and is never written.
Added after the first run, on 23 November 2026, in answer to a review.
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

from ..data import tables as T
from ..data.schema import N_FOLDS, roles, sha256_file
from ..sim import core, metrics, stats
from ..sim.baselines import knn_router
from ..sim.baselines.random_single import request_vectors, single_stage
from ..sim.engine import load_task
from ..sim.types import Outcome, empty_ledger
from . import run_experiments as RX
from .featurize_requests import DIMENSIONS, unit_rows
from .router_study import called_model, settings_for

REPRESENTATIONS = ("tfidf", "smollm2")
TOLERANCE = 1e-12


def tfidf_refit(texts, fit_rows, dimensions=DIMENSIONS):
    """TF-IDF and truncated SVD of ``featurize_requests``, fitted on the fit requests, applied to all."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectoriser = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=50000,
                                 dtype=np.float64)
    fitted = vectoriser.fit_transform([texts[i] for i in fit_rows])
    k = min(dimensions, fitted.shape[0] - 1, fitted.shape[1] - 1)
    if k < 1:
        return unit_rows(np.zeros((len(texts), 1)))
    svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=7, random_state=0).fit(fitted)
    return unit_rows(svd.transform(vectoriser.transform(texts)))


def pca_refit(x, fit_rows, dimensions=DIMENSIONS):
    """PCA of ``featurize_requests``, fitted on the fit requests, applied to all."""
    from sklearn.decomposition import PCA

    k = min(dimensions, len(fit_rows) - 1, x.shape[1])
    if k < 1:
        return unit_rows(np.zeros((x.shape[0], 1)))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(x[fit_rows].astype(np.float64))
    return unit_rows(pca.transform(x.astype(np.float64)))


def refit_router(task, settings, features_of):
    """The capped router whose features for rotation f are ``features_of(f, fit request rows)``."""
    config = settings["knn_router"]
    k_grid, penalties = tuple(config["k_grid"]), tuple(config["lambda_grid"])
    models = np.empty(task.R, np.int32)
    info = {"k": [], "lambda": []}
    for f in range(N_FOLDS):
        fit_rows = np.flatnonzero(np.isin(task.b.fold, roles(f)[0]))
        k, selected, scores, relative, _ = knn_router.fit_rotation(task, f, features_of(f, fit_rows), "refit",
                                                                   k_grid, penalties)
        models[task.b.fold == f] = (scores - selected * relative).argmax(axis=1)
        info["k"].append(k)
        info["lambda"].append(selected)
    choices = np.array([task.recal(f)[0] for f in range(N_FOLDS)])
    ledger = empty_ledger()
    ledger.update(labeled_folds="fit and validation", models_profiled=task.b.M)
    return Outcome(request_vectors(task, models, choices), single_stage(task, models), info, ledger)


def one_benchmark(cid, corpus, settings, digest, features, rows_dir, stored):
    task = load_task(corpus, cid)
    rows, pairs, moved = [], {}, {}
    for name in REPRESENTATIONS:
        source = Path(features) / name
        if not (source / f"catalog-{cid:03d}.npy").is_file():
            continue
        before = knn_router.run(task, settings_for(settings, str(source)))
        scored_before = metrics.score(task, before)
        row = RX._result_row(task, scored_before, f"router_{name}", "all_requests", digest)
        reference = stored.get((f"router_{name}", cid))
        if reference is not None and (abs(reference["log_loss"] - row["log_loss"]) > TOLERANCE
                                      or abs(reference["accuracy"] - row["accuracy"]) > TOLERANCE):
            raise ValueError(f"router_{name} does not reproduce router_study.json on benchmark {cid}")
        if name == "tfidf":
            from .embed_requests import request_text
            from ..data.load import corpus_rows
            texts = [request_text(r) for r in corpus_rows(corpus, rows_dir, cid)]
            if len(texts) != task.R:
                raise ValueError(f"benchmark {cid}: rows do not match the corpus")
            features_of = lambda f, fit_rows, texts=texts: tfidf_refit(texts, fit_rows)          # noqa: E731
        else:
            raw = np.load(source / "unreduced" / f"catalog-{cid:03d}.npy", allow_pickle=False)
            if len(raw) != task.R or not np.isfinite(raw).all():
                raise ValueError(f"benchmark {cid}: stored states do not match the corpus")
            features_of = lambda f, fit_rows, raw=raw: pca_refit(raw, fit_rows)                  # noqa: E731
        after = refit_router(task, settings, features_of)
        scored_after = metrics.score(task, after)
        rows.append(row)
        rows.append(RX._result_row(task, scored_after, f"router_{name}", "fit_folds_only", digest))
        moved[name] = float((called_model(after.stage) != called_model(before.stage)).mean())
        pairs[name] = {}
        for metric, new, old in (("log_loss", core.loss_of_mass(scored_after.mass), core.loss_of_mass(scored_before.mass)),
                                 ("accuracy", scored_after.hit.astype(float), scored_before.hit.astype(float))):
            sampled = stats.bench_replicates(task, np.column_stack([new, old]))
            pairs[name][metric] = (float(new.mean() - old.mean()), sampled[:, 0] - sampled[:, 1])
    return cid, rows, pairs, moved


def run(corpus, settings_path, results, features, rows_dir, out, processes=1):
    corpus, results, out = Path(corpus), Path(results), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    study = json.loads((results / "router_study.json").read_text(encoding="utf-8"))
    if study["settings_sha256"] != digest:
        raise SystemExit("router_study.json was made with other settings")
    stored = {(r["method"], r["catalog_id"]): r for r in study["rows"]
              if r["catalog_id"] is not None and r["variant"] == "capped"}
    cids = sorted({c for _, c in stored})
    work = partial(one_benchmark, corpus=corpus, settings=settings, digest=digest, features=features,
                   rows_dir=rows_dir, stored=stored)
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            parts = list(pool.map(work, cids))
    else:
        parts = [work(c) for c in cids]
    rows = RX._with_suites([r for _, part, _, _ in parts for r in part], settings["suites"])
    pairs = {cid: p for cid, _, p, _ in parts}
    moved = {cid: m for cid, _, _, m in parts}
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in cids if c not in set(definition["exclude"])]
        block = {}
        for name in REPRESENTATIONS:
            if not members or any(name not in pairs[c] for c in members):
                continue
            points = {r["variant"]: {k: r[k] for k in ("log_loss", "accuracy", "usd_per_1k_requests", "n_benchmarks")}
                      for r in rows if r["catalog_id"] is None and r["suite"] == suite
                      and r["method"] == f"router_{name}"}
            block[name] = dict(points=points,
                               requests_sent_to_another_model=math.fsum(moved[c][name] for c in members) / len(members))
            for metric in ("log_loss", "accuracy"):
                block[name][f"{metric}_fit_only_minus_all"] = stats.suite_contrast(
                    {c: pairs[c][name][metric][0] for c in members},
                    {c: pairs[c][name][metric][1] for c in members}, members)
        if block:
            suites[suite] = block
    RX._json(out / "router_refit_features.json", dict(
        synthetic=study["synthetic"], settings_sha256=digest, representations=list(REPRESENTATIONS),
        dimensions=DIMENSIONS, suites=suites))
    return suites


def main(argv=None):
    ap = argparse.ArgumentParser(description="The router with text features fitted on the fit folds only")
    for name in ("corpus", "settings", "results", "features", "rows", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    suites = run(a.corpus, a.settings, a.results, a.features, a.rows, a.out, a.processes)
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    for name, block in suites[headline].items():
        d = block["log_loss_fit_only_minus_all"]
        print(name, {v: (round(p["log_loss"], 4), round(100 * p["accuracy"], 2)) for v, p in block["points"].items()},
              f"log loss {d['delta']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]",
              f"rerouted {100 * block['requests_sent_to_another_model']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

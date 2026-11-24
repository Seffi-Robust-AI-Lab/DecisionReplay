"""The router with several request representations, how often it ranks a right model high, and where it sends requests.

    python3 -B -m acl_resource.analysis.router_study --corpus DIR --settings FILE --out RESULTS \
        --features DIR [--processes N]

Representations: ``numeric`` (the five request features stored in the corpus) and every folder of ``--features``
that holds ``features.json`` and one float32 matrix per benchmark (``minilm``, ``tfidf``, ``smollm2``, built
locally by ``embed_requests`` and ``featurize_requests``), plus the control that chooses one cost-penalised model
per benchmark.  Each runs under the two rules of ``operating_points``: ``capped`` (the main-table rule) and
``uncapped`` (lowest validation loss at any cost).

It writes ``RESULTS/router_study.json``:

``rows``       benchmark and suite rows, method ``router_<representation>`` or ``router_control``, variant = rule;
``recall``     per suite, representation and rule: the share of judged fields on which a right model is among the
               k models that the router scores highest for the field's request (k = 1, 3, 5), on the fields where
               one to three models are right and on all fields where at least one is; benchmarks are weighted
               equally as in ``hardness`` (every field weighs 1 / its benchmark's number of judged fields), so
               recall at 1 equals the router's accuracy on those fields;
``routing``    per benchmark: the share of requests sent to each model by the router (numeric and sentence-encoder
               features, both rules) and the control, and, for the cascade (both rules), the share of requests
               answered by each model and the share that stop after one, two or three calls.

Checks that stop the run: the numeric and sentence-encoder routers under the capped rule reproduce ``knn_router``
and ``knn_router_text`` of ``rows.json``; the model ranked first is the model the router called.  The request text
never enters this module; it reads feature matrices only.  Added after the first run.
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
from ..sim import metrics
from ..sim.baselines import confidence_cascade, knn_router
from ..sim.types import NOT_CALLED
from ..sim.engine import load_task
from . import run_experiments as RX
from .hardness import FEW_RIGHT, model_hits
from .operating_points import RULES, lowest_loss

ORDER = ("numeric", "numeric_all", "minilm", "tfidf", "smollm2")
NESTED = "numeric_all"                      # numeric features with "every fitting request" added to the k grid
RECALL_K = (1, 3, 5)
ROUTED = ("numeric", "minilm")              # representations whose routing shares are reported
STORED = {"numeric": "knn_router", "minilm": "knn_router_text"}
TOLERANCE = 1e-12


def representations(features):
    """[(name, source)] in a fixed order: numeric first, then the local feature folders."""
    folders = sorted(p.name for p in Path(features).iterdir() if (p / "features.json").is_file()) if features else []
    names = [n for n in ORDER if n in ("numeric", NESTED) or n in folders] + [n for n in folders if n not in ORDER]
    return [(n, "numeric" if n in ("numeric", NESTED) else str(Path(features) / n)) for n in names]


def settings_for(settings, source, control=False, nested=False):
    router = dict(settings["knn_router"], features=source)
    if control:
        router["k_grid"] = [RX.ALL_FIT_REQUESTS]
    if nested:
        router["k_grid"] = list(router["k_grid"]) + [RX.ALL_FIT_REQUESTS]
    return dict(settings, knn_router=router)


def rankings(task, config, values):
    """[R, M] rank of every model for every request (0 = scored highest), from the router's cached rotations."""
    router = config["knn_router"]
    key = (router["features"], tuple(router["k_grid"]), tuple(router["lambda_grid"]))
    fits = vars(task)["_router_fits"][key]
    position = np.empty((task.R, task.b.M), np.int64)
    for f, (_, _, scores, relative, _) in enumerate(fits):
        rows = np.flatnonzero(task.b.fold == f)
        order = np.argsort(-(scores - values[f] * relative), axis=1, kind="stable")
        rank = np.empty_like(order)
        rank[np.arange(len(rows))[:, None], order] = np.arange(task.b.M)[None, :]
        position[rows] = rank
    return position


def called_model(stage):
    """The model a one-stage plan called for each request."""
    if not np.all((stage != NOT_CALLED).sum(axis=1) == 1):
        raise ValueError("expected one call per request")
    return stage.argmax(axis=1)


def recall(task, hits, position):
    """Right-model ranks of the judged fields: the best rank among the models that are right on the field."""
    rank = position[task.b.field_row[task.J]]
    best = np.where(hits.T, rank, task.b.M).min(axis=1)
    right = hits.sum(axis=0)
    out = {}
    for label, mask in (("few_right", (right >= 1) & (right <= FEW_RIGHT)), ("any_right", right >= 1)):
        out[label] = dict(fields=int(mask.sum()), judged=int(task.nJ),
                          **{f"at_{k}": int((best[mask] < k).sum()) for k in RECALL_K})
    return out


def one_benchmark(cid, corpus, settings, digest, reps, stored):
    task = load_task(corpus, cid)
    hits = model_hits(task)
    rows, recalls, routing = [], {}, {}
    for name, source in reps + [("control", "numeric")]:
        if name not in ("numeric", NESTED, "control") and not (Path(source) / f"catalog-{cid:03d}.npy").is_file():
            continue
        config = settings_for(settings, source, control=name == "control", nested=name == NESTED)
        capped = knn_router.run(task, config)
        free = [lowest_loss(r) for r in capped.info["selection"]]
        outcomes = {"capped": capped, "uncapped": knn_router.run(task, config, penalty=free)}
        values = {"capped": capped.info["lambda"], "uncapped": free}
        method = f"router_{name}"
        if name == NESTED:
            routing["nested_k_all"] = [int(k == RX.ALL_FIT_REQUESTS) for k in capped.info["k"]]
        for rule, outcome in outcomes.items():
            scored = metrics.score(task, outcome)
            rows.append(RX._result_row(task, scored, method, rule, digest))
            model = called_model(outcome.stage)
            if name in STORED and rule == "capped":
                reference = stored.get((STORED[name], cid))
                if reference is not None and (abs(reference["log_loss"] - rows[-1]["log_loss"]) > TOLERANCE
                                              or abs(reference["accuracy"] - rows[-1]["accuracy"]) > TOLERANCE):
                    raise ValueError(f"{method} does not reproduce {STORED[name]} on benchmark {cid}")
            if name != "control":
                position = rankings(task, config, values[rule])
                if not np.array_equal(position.argmin(axis=1), model):
                    raise ValueError(f"{method}: the first-ranked model is not the called model on benchmark {cid}")
                recalls[name, rule] = recall(task, hits, position)
            if name in ROUTED or name == "control":
                routing[f"{method}.{rule}"] = [float(v) for v in np.bincount(model, minlength=task.b.M) / task.R]
    capped = confidence_cascade.run(task, settings)
    free = [lowest_loss(r) for r in capped.info["selection"]]
    for rule, outcome in (("capped", capped), ("uncapped", confidence_cascade.run(task, settings, threshold=free))):
        depth = outcome.stage.max(axis=1)
        answer = outcome.stage.argmax(axis=1)
        routing[f"cascade.{rule}"] = [float(v) for v in np.bincount(answer, minlength=task.b.M) / task.R]
        routing[f"cascade_calls.{rule}"] = [float(v) for v in np.bincount(depth, minlength=3)[:3] / task.R]
    return cid, rows, recalls, routing


def summarise_recall(recalls, cids, names):
    out = {}
    for name in names:
        for rule in RULES:
            cells = [recalls[c][name, rule] for c in cids if (name, rule) in recalls[c]]
            if len(cells) != len(cids):
                continue
            block = {}
            for subset in ("few_right", "any_right"):
                usable = [cell[subset] for cell in cells if cell[subset]["fields"]]
                mass = math.fsum(u["fields"] / u["judged"] for u in usable)      # each field weighs 1 / judged fields
                block[subset] = dict(
                    benchmarks=len(usable), fields=sum(u["fields"] for u in usable),
                    **{f"at_{k}": (math.fsum(u[f"at_{k}"] / u["judged"] for u in usable) / mass if mass else None)
                       for k in RECALL_K},
                    **{f"pooled_at_{k}": sum(u[f"at_{k}"] for u in usable) / max(1, sum(u["fields"] for u in usable))
                       for k in RECALL_K})
            out.setdefault(name, {})[rule] = block
    return out


def run(corpus, settings_path, out, features, processes=1):
    corpus, out = Path(corpus), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    document = json.loads((out / "rows.json").read_text(encoding="utf-8"))
    if document["settings_sha256"] != digest:
        raise SystemExit("rows.json was made with other settings; run the experiments first")
    stored = {(r["method"], r["catalog_id"]): r for r in document["rows"]
              if r["catalog_id"] is not None and r["variant"] is None}
    cids = sorted({c for _, c in stored})
    reps = representations(features)
    work = partial(one_benchmark, corpus=corpus, settings=settings, digest=digest, reps=reps, stored=stored)
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            results = list(pool.map(work, cids))
    else:
        results = [work(c) for c in cids]
    rows, recalls, routing = [], {}, {}
    for cid, new_rows, rec, route in results:
        rows.extend(new_rows)
        recalls[cid], routing[str(cid)] = rec, route
    all_rows = RX._with_suites(rows, settings["suites"])
    names = [n for n, _ in reps]
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in cids if c not in set(definition["exclude"])]
        if not members:
            continue
        points = {}
        for r in all_rows:
            if r["catalog_id"] is None and r["suite"] == suite:
                points.setdefault(r["method"], {})[r["variant"]] = {
                    k: r[k] for k in ("log_loss", "accuracy", "usd_per_1k_requests", "latency_mean_s",
                                      "calls_per_request", "n_benchmarks")}
        chose_all = [v for c in members for v in routing[str(c)].get("nested_k_all", [])]
        suites[suite] = dict(points=points, recall=summarise_recall(recalls, members, names),
                             nested_cells=len(chose_all), nested_cells_with_all=sum(chose_all))
    info = {}
    for name, source in reps:
        if name not in ("numeric", NESTED):
            meta = json.loads((Path(source) / "features.json").read_text(encoding="utf-8"))
            info[name] = {k: meta[k] for k in ("model", "max_tokens", "dtype", "device", "reduction", "revision")
                          if k in meta}
    RX._json(out / "router_study.json", dict(
        synthetic=document["synthetic"], settings_sha256=digest, representations=names, features=info,
        few_right=FEW_RIGHT, recall_k=list(RECALL_K), rows=all_rows, suites=suites, routing=routing))
    return suites


def main(argv=None):
    ap = argparse.ArgumentParser(description="Router representations, recall and routing shares")
    for name in ("corpus", "settings", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--features")
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    suites = run(a.corpus, a.settings, a.out, a.features, a.processes)
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    for method, rules in sorted(suites[headline]["points"].items()):
        print(method, {rule: (round(p["log_loss"], 4), round(100 * p["accuracy"], 2)) for rule, p in rules.items()})
    for name, rules in suites[headline]["recall"].items():
        print("recall", name, {rule: {k: round(v, 3) for k, v in b["few_right"].items() if k.startswith("at_")}
                               for rule, b in rules.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

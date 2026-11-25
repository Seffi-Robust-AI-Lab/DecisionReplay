"""The routers with a representation of the whole request, answer options included, beside the stored one.

    python3 -B -m analysis.router_full_text --corpus DIR --settings FILE --results RESULTS \
        --old DIR --new DIR --out OUT [--processes N]

``--old`` is the stored sentence-encoder folder (state and instructions, cut at the encoder's length), ``--new``
the folder that ``embed_requests_full`` writes (the whole request with its answer options, no cut).  For both,
inside every benchmark and under the capped and the uncapped rule of ``operating_points``, the module runs

* the nearest-neighbour router of the paper (``knn_router`` with the grids of the settings), and
* the ridge-regression router of ``router_unknown`` (``ridge_within``),

and reports per suite: their rows; the paired differences new minus old, new minus the router's control and new
minus the best model per benchmark, in log loss and in accuracy, with the within-benchmark interval of
``sim.stats`` and the interval over resampled benchmarks of ``paper_facts``; the recall at k of the
nearest-neighbour router on the fields that one to three models get right (as ``router_study``); the share of
requests that the new representation sends to another model; and how many benchmarks have one vector for every
request, or distinct vectors for fewer than half of their requests, under each representation.

Checks that stop the run: with the old features the nearest-neighbour router reproduces ``router_minilm`` of
``RESULTS/router_study.json`` and the control reproduces ``router_control``; the ridge router with the old
features reproduces the per-field arrays archived in ``OUT/local`` when they are there.  The design was fixed
before the first run on the corpus.  It reads feature matrices only, never the request text.  Writes
``OUT/router_full_text.json``.  Added after the first run, on 23 November 2026, in answer to a review.
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
from sim import core, metrics, stats
from sim.baselines import knn_router
from sim.engine import load_task
from . import router_unknown as RU
from . import run_experiments as RX
from .hardness import model_hits
from .operating_points import lowest_loss
from .paper_facts import _over_benchmarks
from .router_study import RECALL_K, called_model, rankings, recall, settings_for

RULES = {"capped": "capped_bd", "uncapped": "uncapped"}
METRICS = ("log_loss", "accuracy")
PAIRS = (("knn_new", "knn_old"), ("knn_new", "control"), ("knn_new", "best_single"),
         ("ridge_new", "ridge_old"), ("ridge_new", "control"), ("ridge_new", "best_single"),
         ("ridge_new", "knn_new"))
TOLERANCE = 1e-12


def knn_outcomes(task, settings, source):
    """{rule: (outcome, penalties)} of the paper's router with the feature folder ``source``, and its settings."""
    config = settings_for(settings, str(source))
    capped = knn_router.run(task, config)
    free = [lowest_loss(r) for r in capped.info["selection"]]
    return {"capped": (capped, capped.info["lambda"]),
            "uncapped": (knn_router.run(task, config, penalty=free), free)}, config


def ridge_outcomes(task, settings, source, name):
    """{rule: (outcome, models)} of the ridge-regression router fitted inside the benchmark."""
    cid = task.cid
    policy = f"ridge_within_{name}"
    profile = RU.profile_of(task)
    table = RU.global_choices([profile["stats"]])
    router = dict(policy=policy, scorer=RU.ridge_scores, grid=RU.ALPHAS,
                  features={cid: knn_router.request_features(task.b, str(source))}, standardise=False)
    fit = RU.fit_pool([cid], {cid: profile}, {cid: RU.rotations_of(task, table)}, [router],
                      tuple(settings["knn_router"]["lambda_grid"]))
    out = {}
    for rule, key in RULES.items():
        models = fit["plans"][cid][policy, key]
        out[rule] = (RU.routed(task, models, RU.deployment(table, 3), "fit and validation", False), models)
    return out


def one_benchmark(cid, corpus, settings, digest, results, old, new, stored, out, draws, seed):
    task = load_task(corpus, cid)
    b = task.b
    hits = model_hits(task)
    x_old, x_new = (knn_router.request_features(b, str(folder)) for folder in (old, new))
    distinct = dict(old=int(len(np.unique(x_old, axis=0))), new=int(len(np.unique(x_new, axis=0))), requests=task.R)
    knn = {}
    for key, folder in (("old", old), ("new", new)):
        knn[key], config = knn_outcomes(task, settings, folder)
        knn[key] = dict(outcomes=knn[key], config=config)
    control_config = settings_for(settings, "numeric", control=True)
    control_capped = knn_router.run(task, control_config)
    control = {"capped": control_capped,
               "uncapped": knn_router.run(task, control_config,
                                          penalty=[lowest_loss(r) for r in control_capped.info["selection"]])}
    ridge = {"old": ridge_outcomes(task, settings, old, "old"), "new": ridge_outcomes(task, settings, new, "new")}
    mass, hit = RX._local_fields(results, "best_single", task)
    best = (core.loss_of_mass(mass), hit.astype(np.float64))
    rows, pairs, extra, recalls = [], {}, {}, {}
    for rule in RULES:
        scored = {
            "knn_old": metrics.score(task, knn["old"]["outcomes"][rule][0]),
            "knn_new": metrics.score(task, knn["new"]["outcomes"][rule][0]),
            "ridge_old": metrics.score(task, ridge["old"][rule][0]),
            "ridge_new": metrics.score(task, ridge["new"][rule][0]),
            "control": metrics.score(task, control[rule]),
        }
        for name, reference in (("knn_old", "router_minilm"), ("control", "router_control")):
            row = stored.get((reference, rule, cid))
            if row is not None and (abs(row["log_loss"] - core.loss_of_mass(scored[name].mass).mean()) > TOLERANCE
                                    or abs(row["accuracy"] - scored[name].hit.mean()) > TOLERANCE):
                raise ValueError(f"{reference} ({rule}) is not reproduced on benchmark {cid}")
        archive = Path(out) / "local" / f"fields-{RU.label('ridge_within_minilm', rule)}.npz"
        if archive.is_file():
            with np.load(archive, allow_pickle=False) as data:
                if f"mass_{cid}" in data.files and not np.array_equal(data[f"mass_{cid}"], scored["ridge_old"].mass):
                    raise ValueError(f"the ridge router with the old features is not reproduced on benchmark {cid}")
        fields = {name: (core.loss_of_mass(s.mass), s.hit.astype(np.float64)) for name, s in scored.items()}
        fields["best_single"] = best
        for name in ("knn_old", "knn_new", "ridge_old", "ridge_new"):
            rows.append(RX._result_row(task, scored[name], name, rule, digest))
        for a, c in PAIRS:
            for i, metric in enumerate(METRICS):
                sampled = stats.bench_replicates(task, np.column_stack([fields[a][i], fields[c][i]]), draws, seed)
                pairs[a, c, rule, metric] = (float(fields[a][i].mean() - fields[c][i].mean()),
                                             sampled[:, 0] - sampled[:, 1])
        sent = {key: called_model(knn[key]["outcomes"][rule][0].stage) for key in ("old", "new")}
        ridge_new = ridge["new"][rule][1]
        one_model = [len(np.unique(ridge_new[b.fold == f])) == 1 for f in range(N_FOLDS) if (b.fold == f).any()]
        extra[rule] = dict(knn_moved=float((sent["new"] != sent["old"]).mean()),
                           ridge_moved_from_control=float((ridge_new != called_model(control[rule].stage)).mean()),
                           ridge_cells=len(one_model), ridge_cells_one_model=int(sum(one_model)))
        for key in ("old", "new"):
            position = rankings(task, knn[key]["config"], knn[key]["outcomes"][rule][1])
            recalls[key, rule] = recall(task, hits, position)
    return cid, rows, pairs, extra, recalls, distinct


def summarise_recall(recalls, cids, key, rule):
    """Recall at k on the two field subsets, every field weighing 1 / its benchmark's number of judged fields."""
    out = {}
    for subset in ("few_right", "any_right"):
        usable = [recalls[c][key, rule][subset] for c in cids if recalls[c][key, rule][subset]["fields"]]
        mass = math.fsum(u["fields"] / u["judged"] for u in usable)
        out[subset] = {f"at_{k}": (math.fsum(u[f"at_{k}"] / u["judged"] for u in usable) / mass if mass else None)
                       for k in RECALL_K}
    return out


def run(corpus, settings_path, results, old, new, out, processes=1):
    corpus, results, out = Path(corpus), Path(results), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    study = json.loads((results / "router_study.json").read_text(encoding="utf-8"))
    if study["settings_sha256"] != digest:
        raise SystemExit("router_study.json was made with other settings")
    stored = {(r["method"], r["variant"], r["catalog_id"]): r for r in study["rows"] if r["catalog_id"] is not None}
    cids = sorted(c for c in {c for _, _, c in stored}
                  if all((Path(folder) / f"catalog-{c:03d}.npy").is_file() for folder in (old, new)))
    if not cids:
        raise SystemExit("no benchmark has both representations")
    draws, seed = settings["bootstrap"]["draws"], settings["bootstrap"]["seed"]
    work = partial(one_benchmark, corpus=corpus, settings=settings, digest=digest, results=results, old=old, new=new,
                   stored=stored, out=out, draws=draws, seed=seed)
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            parts = list(pool.map(work, cids))
    else:
        parts = [work(c) for c in cids]
    rows = RX._with_suites([r for _, part, _, _, _, _ in parts for r in part], settings["suites"])
    pairs = {cid: p for cid, _, p, _, _, _ in parts}
    extra = {cid: e for cid, _, _, e, _, _ in parts}
    recalls = {cid: r for cid, _, _, _, r, _ in parts}
    distinct = {cid: d for cid, _, _, _, _, d in parts}
    keys = ("log_loss", "accuracy", "usd_per_1k_requests", "n_benchmarks")
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in cids if c not in set(definition["exclude"])]
        if not members:
            continue
        points = {}
        for r in rows:
            if r["catalog_id"] is None and r["suite"] == suite and r["n_benchmarks"] == len(members):
                points.setdefault(r["method"], {})[r["variant"]] = {k: r[k] for k in keys}
        block = dict(benchmarks=len(members), points=points, contrasts={}, recall={}, routing={}, distinct={})
        for rule in RULES:
            for a, c in PAIRS:
                cell = {}
                for metric in METRICS:
                    delta = {m: pairs[m][a, c, rule, metric][0] for m in members}
                    within = stats.suite_contrast(delta, {m: pairs[m][a, c, rule, metric][1] for m in members},
                                                  members)
                    over = _over_benchmarks([delta[m] for m in members], metric, draws, seed)
                    cell[metric] = dict(difference=within["delta"], within_low=within["ci_low"],
                                        within_high=within["ci_high"], over_low=over["low"], over_high=over["high"],
                                        benchmarks_better=over["better"], benchmarks_worse=over["worse"])
                block["contrasts"][f"{a} - {c} ({rule})"] = cell
            block["recall"][rule] = {key: summarise_recall(recalls, members, key, rule) for key in ("old", "new")}
            block["routing"][rule] = dict(
                knn_requests_sent_to_another_model=math.fsum(extra[m][rule]["knn_moved"] for m in members) / len(members),
                ridge_requests_not_sent_to_the_control_model=math.fsum(
                    extra[m][rule]["ridge_moved_from_control"] for m in members) / len(members),
                ridge_cells=sum(extra[m][rule]["ridge_cells"] for m in members),
                ridge_cells_with_one_model=sum(extra[m][rule]["ridge_cells_one_model"] for m in members))
        for key in ("old", "new"):
            shares = [distinct[m][key] / distinct[m]["requests"] for m in members]
            block["distinct"][key] = dict(
                benchmarks_with_one_vector=sum(distinct[m][key] == 1 for m in members),
                benchmarks_below_half_distinct=sum(s < 0.5 for s in shares),
                benchmarks_below_a_tenth_distinct=sum(s < 0.1 for s in shares))
        suites[suite] = block
    RX._json(out / "router_full_text.json", dict(
        synthetic=study["synthetic"], settings_sha256=digest, old=Path(old).name, new=Path(new).name,
        alphas=list(RU.ALPHAS), draws=draws, seed=seed, recall_k=list(RECALL_K), suites=suites))
    return suites


def main(argv=None):
    ap = argparse.ArgumentParser(description="Routers with the whole request embedded, beside the stored features")
    for name in ("corpus", "settings", "results", "old", "new", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    suites = run(a.corpus, a.settings, a.results, a.old, a.new, a.out, a.processes)
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    block = suites[headline]
    for method, rules in sorted(block["points"].items()):
        print(method, {rule: (round(p["log_loss"], 4), round(100 * p["accuracy"], 2)) for rule, p in rules.items()})
    for name, cell in block["contrasts"].items():
        l, c = cell["log_loss"], cell["accuracy"]
        print(f"{name}: log loss {l['difference']:+.4f} [{l['within_low']:+.4f}, {l['within_high']:+.4f}]; "
              f"accuracy {100 * c['difference']:+.2f} [{100 * c['within_low']:+.2f}, {100 * c['within_high']:+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

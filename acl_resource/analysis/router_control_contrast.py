"""The learned router inside the benchmark against the router's control, paired.

    python3 -B -m acl_resource.analysis.router_control_contrast --corpus DIR --settings FILE --results RESULTS \
        --features DIR --out OUT [--processes N]

``router_unknown`` found that a ridge regression on the sentence embeddings, fitted inside each benchmark, is level
with the best model per benchmark.  Its suite row is slightly ahead of the control, which picks one cost-penalised
model per benchmark and never looks at the request, but the two were not paired there.  This module refits both
with the code of ``router_unknown`` and of ``router_study`` and reports, under the capped and the uncapped rule:

* the paired difference in log loss and in accuracy, ridge router minus control, with the within-benchmark
  interval of ``sim.stats`` and the interval over resampled benchmarks of ``paper_facts``;
* the share of requests that the ridge router sends to another model than the control;
* the share of the cells of benchmark and test fold in which the ridge router sends every request to one model,
  that is, acts as a per-benchmark choice.

Checks that stop the run: the control reproduces ``router_control`` of ``RESULTS/router_study.json``; the ridge
router reproduces the per-field arrays that ``router_unknown`` archived in ``OUT/local`` when they are there.
Writes ``OUT/router_control_contrast.json``.  The design was fixed before the run.  Added after the first run, on
23 November 2026, in answer to a review.
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
from ..sim import core, metrics, stats
from ..sim.baselines import knn_router
from ..sim.engine import load_task
from . import router_unknown as RU
from . import run_experiments as RX
from .operating_points import lowest_loss
from .paper_facts import _over_benchmarks
from .router_study import called_model, settings_for

RULES = {"capped": "capped_bd", "uncapped": "uncapped"}
METRICS = ("log_loss", "accuracy")
TOLERANCE = 1e-12


def one_benchmark(cid, corpus, settings, features, stored, out, draws, seed):
    task = load_task(corpus, cid)
    b, name = task.b, Path(features).name
    policy = f"ridge_within_{name}"
    profile = RU.profile_of(task)
    table = RU.global_choices([profile["stats"]])
    router = dict(policy=policy, scorer=RU.ridge_scores, grid=RU.ALPHAS,
                  features={cid: knn_router.request_features(b, str(features))}, standardise=False)
    fit = RU.fit_pool([cid], {cid: profile}, {cid: RU.rotations_of(task, table)}, [router],
                      tuple(settings["knn_router"]["lambda_grid"]))
    config = settings_for(settings, "numeric", control=True)
    capped = knn_router.run(task, config)
    free = [lowest_loss(r) for r in capped.info["selection"]]
    control = {"capped": capped, "uncapped": knn_router.run(task, config, penalty=free)}
    result = {}
    for rule, source in RULES.items():
        models = fit["plans"][cid][policy, source]
        ridge = metrics.score(task, RU.routed(task, models, RU.deployment(table, 3), "fit and validation", False))
        archive = Path(out) / "local" / f"fields-{RU.label(policy, rule)}.npz"
        if archive.is_file():
            with np.load(archive, allow_pickle=False) as data:
                if f"mass_{cid}" in data.files and not np.array_equal(data[f"mass_{cid}"], ridge.mass):
                    raise ValueError(f"the ridge router does not reproduce its archived fields on benchmark {cid}")
        other = metrics.score(task, control[rule])
        row = stored.get((rule, cid))
        if row is not None and (abs(row["log_loss"] - core.loss_of_mass(other.mass).mean()) > TOLERANCE
                                or abs(row["accuracy"] - other.hit.mean()) > TOLERANCE):
            raise ValueError(f"the control does not reproduce router_study.json on benchmark {cid}")
        chosen = called_model(control[rule].stage)
        one_model = [len(np.unique(models[b.fold == f])) == 1 for f in range(N_FOLDS) if (b.fold == f).any()]
        block = dict(moved=float((models != chosen).mean()), cells=len(one_model), cells_one_model=int(sum(one_model)))
        for metric, new, old in (("log_loss", core.loss_of_mass(ridge.mass), core.loss_of_mass(other.mass)),
                                 ("accuracy", ridge.hit.astype(float), other.hit.astype(float))):
            sampled = stats.bench_replicates(task, np.column_stack([new, old]), draws, seed)
            block[metric] = (float(new.mean() - old.mean()), sampled[:, 0] - sampled[:, 1])
        result[rule] = block
    return cid, result


def run(corpus, settings_path, results, features, out, processes=1):
    corpus, results, out = Path(corpus), Path(results), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    study = json.loads((results / "router_study.json").read_text(encoding="utf-8"))
    if study["settings_sha256"] != digest:
        raise SystemExit("router_study.json was made with other settings")
    stored = {(r["variant"], r["catalog_id"]): r for r in study["rows"]
              if r["catalog_id"] is not None and r["method"] == "router_control"}
    cids = sorted(c for c in {c for _, c in stored}
                  if (Path(features) / f"catalog-{c:03d}.npy").is_file())
    draws, seed = settings["bootstrap"]["draws"], settings["bootstrap"]["seed"]
    work = partial(one_benchmark, corpus=corpus, settings=settings, features=features, stored=stored, out=out,
                   draws=draws, seed=seed)
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            parts = dict(pool.map(work, cids))
    else:
        parts = dict(map(work, cids))
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in cids if c not in set(definition["exclude"])]
        if not members:
            continue
        block = {}
        for rule in RULES:
            cell = dict(
                requests_sent_to_another_model=math.fsum(parts[c][rule]["moved"] for c in members) / len(members),
                cells=sum(parts[c][rule]["cells"] for c in members),
                cells_with_one_model=sum(parts[c][rule]["cells_one_model"] for c in members),
                benchmarks_never_departing=sum(parts[c][rule]["moved"] == 0 for c in members))
            for metric in METRICS:
                delta = {c: parts[c][rule][metric][0] for c in members}
                within = stats.suite_contrast(delta, {c: parts[c][rule][metric][1] for c in members}, members)
                over = _over_benchmarks([delta[c] for c in members], metric, draws, seed)
                cell[metric] = dict(difference=within["delta"], within_low=within["ci_low"],
                                    within_high=within["ci_high"], over_low=over["low"], over_high=over["high"],
                                    benchmarks_better=over["better"], benchmarks_worse=over["worse"])
            block[rule] = cell
        suites[suite] = dict(benchmarks=len(members), ridge_minus_control=block)
    RX._json(out / "router_control_contrast.json", dict(
        synthetic=study["synthetic"], settings_sha256=digest, representation=Path(features).name,
        alphas=list(RU.ALPHAS), draws=draws, seed=seed, suites=suites))
    return suites


def main(argv=None):
    ap = argparse.ArgumentParser(description="The learned router inside the benchmark against the control")
    for name in ("corpus", "settings", "results", "features", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    suites = run(a.corpus, a.settings, a.results, a.features, a.out, a.processes)
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    for rule, cell in suites[headline]["ridge_minus_control"].items():
        l, c = cell["log_loss"], cell["accuracy"]
        print(f"{rule}: log loss {l['difference']:+.4f} [{l['within_low']:+.4f}, {l['within_high']:+.4f}] "
              f"over [{l['over_low']:+.4f}, {l['over_high']:+.4f}]; accuracy {100 * c['difference']:+.2f} "
              f"[{100 * c['within_low']:+.2f}, {100 * c['within_high']:+.2f}] over [{100 * c['over_low']:+.2f}, "
              f"{100 * c['over_high']:+.2f}]; rerouted {100 * cell['requests_sent_to_another_model']:.1f}%; "
              f"one model in {cell['cells_with_one_model']} of {cell['cells']} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())

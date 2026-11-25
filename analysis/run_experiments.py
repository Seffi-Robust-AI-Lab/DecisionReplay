"""Replay the configured policies and write aggregate results and local paired fields."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from data.load import list_benchmarks, read_manifest
from data.schema import N_FOLDS, roles, sha256_file
from sim import core, metrics, stats
from sim.baselines import (
    BASELINES, best_single, confidence_cascade, greedy, knn_router, stack,
    stack_final_temperature, transfer_single, transfer_temperature, uniform_mean, uniform_temperature,
)
from sim.engine import load_task, select_recalibration
from sim.types import RESULT_ROW_KEYS, empty_ledger


TRANSFER = {m.NAME: m for m in (transfer_single, transfer_temperature)}
BUILT_IN = BASELINES.keys() | TRANSFER.keys()
ALL_FIT_REQUESTS = 10 ** 9      # a neighbour count larger than any benchmark: the router then ignores the request


def _ledger(folds, models):
    """Return the target data used to fit a panel."""
    return dict(empty_ledger(), labeled_folds=folds, models_profiled=models)


def _result_row(task, scored, method, variant, digest):
    """Return a benchmark result in the shared row schema."""
    row = dict(method=method, variant=variant, suite=None, catalog_id=task.cid,
               n_benchmarks=1, n_requests=task.R, n_judged=task.nJ, ledger=scored.ledger,
               settings_sha256=digest, **metrics.summarize(scored))
    return {key: row[key] for key in RESULT_ROW_KEYS}


def _draw_mean(rows):
    """Return the mean of the individual replay metrics across draws on one benchmark."""
    row = rows[0].copy()
    if len(rows) == 1:
        return row
    for key in RESULT_ROW_KEYS:
        if key in ("method", "variant", "suite", "catalog_id", "n_benchmarks", "n_requests",
                   "n_judged", "ledger", "settings_sha256") or row[key] is None:
            continue
        row[key] = math.fsum(r[key] for r in rows) / len(rows)
    return row


def _frontier(task, settings, digest):
    """Yield benchmark rows for panel prefixes, random panels and adaptive policies."""
    config = settings["frontier"]
    folds = range(N_FOLDS)
    choices = uniform_mean.member_choices(task)
    ranks = np.stack([best_single.ranking(task, f) for f in folds])
    paths = np.stack([greedy.path(task, f) for f in folds])
    fit_ledger = _ledger("fit", task.b.M)

    def row(name, variant, outcome):
        return _result_row(task, metrics.score(task, outcome), name, variant, digest)

    def equal_prefix(order, size):
        weights = np.zeros_like(order, dtype=float)
        np.put_along_axis(weights, order[:, :size], 1.0 / size, axis=1)
        return uniform_mean.panel(task, weights, choices, ledger=fit_ledger)

    random_orders = [
        np.stack([np.random.default_rng([seed, task.cid, f]).permutation(task.b.M) for f in folds])
        for seed in config["random_panel_seeds"]
    ]
    for size in config["sizes"]:
        if not 1 <= size <= task.b.M:
            raise ValueError("panel size outside the available model range")
        variant = f"k={size}"
        yield row("ranked_top_k", variant, equal_prefix(ranks, size))
        yield row("greedy", variant, uniform_mean.panel(task, paths[:, size - 1], choices,
                                                        ledger=fit_ledger))
        yield _draw_mean([row("random_panel", variant, equal_prefix(order, size))
                          for order in random_orders])
    for penalty in settings["knn_router"]["lambda_grid"]:
        outcome = knn_router.run(task, settings, penalty=penalty)
        yield row("knn_router", f"lambda={penalty}", outcome)
    # Control for the router: with every fit request as a neighbour, the choice is one model per benchmark.
    constant = dict(settings, knn_router=dict(settings["knn_router"], k_grid=[ALL_FIT_REQUESTS]))
    for penalty in settings["knn_router"]["lambda_grid"]:
        outcome = knn_router.run(task, constant, penalty=penalty)
        yield row("single_by_cost", f"lambda={penalty}", outcome)
    for threshold in settings["confidence_cascade"]["tau_grid"]:
        outcome = confidence_cascade.run(task, settings, threshold=threshold)
        yield row("confidence_cascade", f"tau={threshold}", outcome)


def _calibration_order(task, settings):
    """Yield the four equal-weight pools differing in calibration order."""
    yield "none", uniform_mean.run(task, settings)
    yield "final_temperature", uniform_temperature.run(task, settings)
    weights = uniform_mean.equal_weights(task)
    choices = uniform_mean.member_choices(task)
    ledger = _ledger("fit", task.b.M)
    yield "calibrated_members", uniform_mean.panel(task, weights, choices, ledger=ledger)
    temps = stack_final_temperature.temperatures(task, settings, equal=True)
    yield "calibrated_members_and_final_temperature", uniform_mean.panel(
        task, weights, choices, temps, ledger=ledger)


def _budget_fit(task, fold, groups):
    """Return a calibrated single, simplex weights and deployment choices for sampled train4 groups."""
    train = roles(fold)[2]
    groups = np.asarray(groups)
    if len(groups) == np.count_nonzero(task.group_fold != fold):
        rows = task.rows(train)
        choice = task.recal(fold, n=4)[0]
        mass = task.view(fold, "calibrated", n=4)[rows]
        model = best_single.ranking(task, fold, n=4)[0]
    else:
        rows = np.flatnonzero(np.isin(task.group, groups))
        folds = task.fold[rows]
        table = task.mass_menu[:, rows]
        losses = core.loss_of_mass(table)
        sums = np.stack([losses[:, folds == g].sum(axis=1) for g in range(N_FOLDS)])
        counts = np.bincount(folds, minlength=N_FOLDS)
        choice = select_recalibration(sums, counts, train)
        mass = np.empty((len(rows), task.b.M))
        for g in train:
            members = select_recalibration(sums, counts, [h for h in train if h != g])
            sel = np.flatnonzero(folds == g)
            mass[sel] = table[members, sel[:, None], np.arange(task.b.M)]
        model = core.loss_of_mass(mass).mean(axis=0).argmin() if len(mass) else 0
    return int(model), stack.simplex_fit(mass), choice


def _budget_rows(task, settings, digest):
    """Yield paired label-budget comparisons, averaging each metric over the configured draws."""
    budget = settings["label_budget"]
    available = [np.flatnonzero(task.group_fold != f) for f in range(N_FOLDS)]
    full = [_budget_fit(task, f, groups) for f, groups in enumerate(available)]
    for size in [*budget["groups"], "all"]:
        use_all = [size == "all" or size >= len(groups) for groups in available]
        n_draws = 1 if all(use_all) else budget["draws"]
        rows = {"best_single": [], "stack": []}
        variant = f"groups={size}"
        for draw in range(n_draws):
            fits = []
            for f, groups in enumerate(available):
                if use_all[f]:
                    fits.append(full[f])
                else:
                    rng = np.random.default_rng([budget["seed"], task.cid, f, draw])
                    sampled = rng.permutation(groups)[:size]
                    fits.append(_budget_fit(task, f, sampled))
            models, weights, choices = zip(*fits)
            ledger = _ledger("train4" if size == "all" else f"up to {size} train4 groups", task.b.M)
            for name, panel in (("best_single", np.eye(task.b.M)[list(models)]), ("stack", weights)):
                outcome = uniform_mean.panel(task, np.asarray(panel), np.stack(choices), ledger=ledger)
                rows[name].append(_result_row(task, metrics.score(task, outcome), name, variant, digest))
        for draws in rows.values():
            yield _draw_mean(draws)


def _source_profile(cid, corpus, settings):
    """Return whole-benchmark loss by model and by flat-pool temperature."""
    task = load_task(corpus, cid)
    return {name: module.source_loss(task, settings) for name, module in TRANSFER.items()}


def _transfer_choices(corpus, benchmarks, cids, settings, map_benchmarks):
    """Return the model and temperature trained outside each selected target area."""
    areas = {r["catalog_id"]: r["area"] for r in benchmarks}
    target_areas = set(areas[c] for c in cids)
    sources = sorted(c for c, area in areas.items() if target_areas - {area})
    profile = partial(_source_profile, corpus=corpus, settings=settings)
    profiles = dict(zip(sources, map_benchmarks(profile, sources)))
    choices = {}
    for area in sorted(target_areas):
        other = transfer_single.source_cids(areas, area)
        if not other:
            raise ValueError("transfer requires source benchmarks in other task areas")
        choices[area] = {name: module.choose([profiles[c][name] for c in other], settings)
                         for name, module in TRANSFER.items()}
    return choices


def _ablations(task, settings, digest, transfer):
    """Yield calibration, group-budget and held-out-area benchmark rows."""
    for variant, outcome in _calibration_order(task, settings):
        yield _result_row(task, metrics.score(task, outcome), "uniform_mean", variant, digest)
    yield from _budget_rows(task, settings, digest)
    for name, module in TRANSFER.items():
        outcome = module.deploy(task, transfer[task.area][name])
        yield _result_row(task, metrics.score(task, outcome), name, None, digest)


def _field_file(out, method):
    """Return the local archive path of a method."""
    return out / "local" / f"fields-{method}.npz"


def _contrast_plan(settings, out):
    """Return runnable contrasts and a note for each missing locally supplied comparator."""
    active, notes = [], []
    for spec in settings["contrasts"]:
        names = (spec["a"], spec["b"])
        missing = [name for name in names if name not in BUILT_IN and not _field_file(out, name).is_file()]
        if missing and spec.get("needs_local_method"):
            notes.append(f"{spec['name']} skipped, missing local fields for {', '.join(missing)}")
        elif missing:
            raise ValueError(f"unknown contrast method {missing[0]}")
        else:
            active.append(spec)
    return active, notes


def _local_fields(out, name, task):
    """Return locally supplied masses and hits, checking both judged-field arrays."""
    with np.load(_field_file(out, name), allow_pickle=False) as archive:
        mass, hit = (archive[f"{key}_{task.cid}"] for key in ("mass", "hit"))
    if mass.shape != (task.nJ,) or hit.shape != (task.nJ,):
        raise ValueError(f"field count differs for {name} on benchmark {task.cid}")
    if not np.isfinite(mass).all() or not np.isin(hit, (0, 1)).all():
        raise ValueError(f"invalid fields for {name} on benchmark {task.cid}")
    return mass, hit


def _paired_benchmark(task, fields, contrasts, settings):
    """Return paired effects and their group bootstrap replicates for one benchmark."""
    bootstrap = settings["bootstrap"]
    result = {}
    for spec in contrasts:
        left, right = (fields[name] for name in (spec["a"], spec["b"]))
        for metric in settings["contrast_metrics"]:
            if metric == "log_loss":
                a, b = core.loss_of_mass(left[0]), core.loss_of_mass(right[0])
            elif metric == "accuracy":
                a, b = left[1], right[1]
            else:
                raise ValueError(f"unknown contrast metric {metric}")
            reps = stats.bench_replicates(task, np.column_stack((a, b)), **bootstrap)
            result[spec["name"], metric] = float(a.mean() - b.mean()), reps[:, 0] - reps[:, 1]
    return result


def _replay(cid, corpus, settings, out, digest, sections, contrasts, transfer):
    """Return one benchmark's released rows, local fields and paired replicates."""
    task = load_task(corpus, cid)
    rows = {section: [] for section in sections if section != "contrasts"}
    fields = {}
    methods = [name for name in settings["methods"] if name in BUILT_IN] if "rows" in rows else []
    for spec in contrasts:
        for name in (spec["a"], spec["b"]):
            if name not in methods:
                methods.append(name)
    for name in methods:
        if name in BUILT_IN:
            if name in BASELINES:
                outcome = BASELINES[name].run(task, settings)
            else:
                outcome = TRANSFER[name].deploy(task, transfer[task.area][name])
            scored = metrics.score(task, outcome)
            fields[name] = scored.mass, scored.hit
            if "rows" in rows and name in settings["methods"]:
                rows["rows"].append(_result_row(task, scored, name, None, digest))
        else:
            fields[name] = _local_fields(out, name, task)
    if "frontier" in rows:
        rows["frontier"].extend(_frontier(task, settings, digest))
    if "ablations" in rows:
        rows["ablations"].extend(_ablations(task, settings, digest, transfer))
    paired = _paired_benchmark(task, fields, contrasts, settings)
    local = {name: fields[name] for name in settings["methods"]
             if "rows" in rows and name in BUILT_IN}
    return cid, rows, local, paired


def _contrast_rows(contrasts, settings, digest, paired):
    """Return suite contrasts with multiplicity correction on the configured family."""
    rows = []
    for spec in contrasts:
        for suite, definition in settings["suites"].items():
            cids = sorted(set(paired) - set(definition["exclude"]))
            if not cids:
                continue
            for metric in settings["contrast_metrics"]:
                key = spec["name"], metric
                delta = {cid: paired[cid][key][0] for cid in cids}
                reps = {cid: paired[cid][key][1] for cid in cids}
                effect = stats.suite_contrast(delta, reps, cids)
                better, worse = effect["better"], effect["worse"]
                if metric == "accuracy":
                    better, worse = worse, better
                row = dict(contrast=spec["name"], a=spec["a"], b=spec["b"], suite=suite, metric=metric,
                           difference=effect["delta"], ci_low=effect["ci_low"], ci_high=effect["ci_high"],
                           p=effect["p"], p_holm=None, benchmarks_better=better,
                           benchmarks_same=effect["same"], benchmarks_worse=worse,
                           n_benchmarks=len(cids), draws=effect["draws"], settings_sha256=digest)
                for column in ("difference", "ci_low", "ci_high", "p"):
                    if not np.isfinite(row[column]):
                        row[column] = None
                rows.append(row)
    family = settings["holm_family"]
    suite = settings["headline_suite"] if family["suite"] == "headline" else family["suite"]
    outside = {spec["name"] for spec in contrasts if spec.get("in_family") is False}
    targets = [r for r in rows if r["suite"] == suite and r["metric"] == family["metric"]
               and r["p"] is not None and r["contrast"] not in outside]
    for row, adjusted in zip(targets, stats.holm_adjust([r["p"] for r in targets])):
        row["p_holm"] = float(adjusted)
    return rows


def _with_suites(rows, suites):
    """Return benchmark rows followed by suite rows for each method and variant."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["variant"]), {})[row["catalog_id"]] = row
    return rows + [row for by_cid in grouped.values() for row in metrics.suite_rows(by_cid, suites)]


def _supplied_rows(out, settings, cids, digest):
    """Return previously supplied benchmark rows for the configured local methods."""
    required = set(settings["methods"]) - BUILT_IN
    external = set(required)
    for spec in settings["contrasts"]:
        if spec.get("needs_local_method"):
            external.update({spec["a"], spec["b"]} - BUILT_IN)
    path = out / "rows.json"
    rows = []
    if external and path.is_file():
        with path.open(encoding="utf-8") as stream:
            for row in json.load(stream)["rows"]:
                if row["method"] in external and row["catalog_id"] in cids:
                    rows.append({k: row[k] for k in RESULT_ROW_KEYS})
    supplied = {(r["method"], r["catalog_id"]) for r in rows}
    for name in sorted(required):
        if any((name, cid) not in supplied for cid in cids):
            raise ValueError(f"expected a supplied benchmark row for {name} on every selected benchmark")
    return [dict(row, settings_sha256=digest) for row in rows]


def _write(path, writer):
    """Replace a result file after its complete contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _json(path, content):
    """Write a deterministic, strictly finite JSON document."""
    data = (json.dumps(content, sort_keys=True, allow_nan=False, indent=1) + "\n").encode("utf-8")
    _write(path, lambda stream: stream.write(data))


def run(corpus, settings, out, cids=None, only=None, processes=1):
    """Run the requested sections and return provenance and contrast availability."""
    corpus, settings_path, out = Path(corpus), Path(settings), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    sections = ("rows", "contrasts", "frontier", "ablations") if only is None else (only,)
    benchmarks = list_benchmarks(corpus)
    available = {r["catalog_id"] for r in benchmarks}
    cids = sorted(available if cids is None else set(cids))
    if not cids or set(cids) - available:
        raise ValueError("select at least one benchmark and use registered catalogue ids")
    if processes < 1:
        raise ValueError("processes must be positive")
    manifest = read_manifest(corpus)
    planted = corpus.parent.name == "processed" and (corpus.parent.parent / "PLANTED.json").is_file()
    synthetic = manifest.get("synthetic", planted or all(
        r["name"] == f"synthetic-{r['catalog_id']:03d}" for r in benchmarks))
    envelope = dict(synthetic=synthetic, settings_sha256=digest)
    contrasts, notes = _contrast_plan(settings, out) if "contrasts" in sections else ([], [])
    if only == "contrasts":
        cids = [cid for cid in cids if contrasts and
                any(cid not in suite["exclude"] for suite in settings["suites"].values())]
    all_rows = {s: [] for s in sections if s != "contrasts"}
    if "rows" in all_rows:
        all_rows["rows"] = _supplied_rows(out, settings, cids, digest)
    local, paired = {}, {}
    with (ProcessPoolExecutor(processes) if processes > 1 else nullcontext()) as pool:
        map_benchmarks = pool.map if pool else map
        names = set(settings["methods"]) if "rows" in sections else set()
        for contrast in contrasts:
            names.update((contrast["a"], contrast["b"]))
        needs_transfer = "ablations" in sections or bool(names & TRANSFER.keys())
        transfer = _transfer_choices(corpus, benchmarks, cids, settings, map_benchmarks) if needs_transfer else {}
        replay = partial(_replay, corpus=corpus, settings=settings, out=out, digest=digest,
                         sections=sections, contrasts=contrasts, transfer=transfer)
        for cid, rows, fields, result in map_benchmarks(replay, cids):
            for section, values in rows.items():
                all_rows[section].extend(values)
            for name, (mass, hit) in fields.items():
                archive = local.setdefault(name, {})
                archive[f"mass_{cid}"], archive[f"hit_{cid}"] = mass, hit
            paired[cid] = result
    for name, arrays in local.items():
        _write(_field_file(out, name), lambda stream: np.savez_compressed(stream, **arrays))
    for section, rows in all_rows.items():
        _json(out / f"{section}.json", dict(envelope, rows=_with_suites(rows, settings["suites"])))
    if "contrasts" in sections:
        rows = _contrast_rows(contrasts, settings, digest, paired)
        _json(out / "contrasts.json", dict(envelope, rows=rows))
    provenance = dict(envelope, rows=[], corpus_manifest_sha256=sha256_file(corpus / "manifest.json"),
                      notes=notes)
    _json(out / "run.json", provenance)
    return provenance


def main(argv=None):
    """Read command-line settings and run the offline experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cids", nargs="+", type=int)
    parser.add_argument("--only", choices=("rows", "contrasts", "frontier", "ablations"))
    parser.add_argument("--processes", type=int, default=1)
    run(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()

"""Local completion of a results folder: replayed methods of earlier studies and each benchmark's own metric.

    python3 -B -m analysis.local_methods --corpus DIR --settings FILE --out RESULTS \
        --project-root . [--processes N]

Run after ``run_experiments`` and before ``run_experiments --only contrasts``.  It needs files that exist only
where the earlier studies were run (fitted weights and pooled vectors) and the benchmark rows with the pinned kit
(for the benchmarks' own metrics).  It

* adds the methods ``tree_replay`` and ``greedy_replay`` (weights fitted by the earlier selection study) and
  ``combiner_replay`` (pooled vectors of the earlier conditional-logit combiner over all models) to ``rows.json``
  and writes their per-field arrays to ``local/fields-<method>.npz``, so that the contrasts can use them;
* fills the ``native`` column of every benchmark row of ``rows.json`` and recomputes the suite rows.

Nothing here fits a model: stored weights and vectors are replayed as they are.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from data.schema import N_FOLDS, sha256_file
from sim import metrics, native
from sim.baselines import BASELINES, knn_router
from sim.baselines.uniform_mean import member_choices, panel
from sim.engine import load_task
from sim.types import NOT_CALLED, Outcome, empty_ledger
from . import run_experiments as RX

STUDY = "results/decision_index/paper/study"
COMBINER = "ensemble_round2/results/combiner/vectors"
ROWS = "data/decision_index/common-paid-65/inputs"
PATH_METHODS = {"tree_replay": "tree", "greedy_replay": "greedy"}
COMBINER_METHOD = "combiner_replay"
TEXT_ROUTER = "knn_router_text"


def text_router_settings(settings):
    """Settings of the router with the local text embeddings as request features."""
    return dict(settings, knn_router=dict(settings["knn_router"], features=settings[TEXT_ROUTER]["features"]))


def stored_weights(root, cid, method):
    """[folds, models] weights of the earlier study's selected panel (calibrated members, log-loss objective)."""
    with gzip.open(Path(root) / STUDY / f"catalog-{cid:03d}" / "paths.jsonl.gz", "rt", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    pick = {r["fold"]: r for r in records
            if (r["regime"], r["objective"], r["method"]) == ("calibrated", "log_loss", method)}
    return np.stack([np.asarray(pick[f]["weights"], dtype=np.float64)[pick[f]["selected_k"] - 1]
                     for f in range(N_FOLDS)])


def replay_path(task, root, method):
    """Outcome of a stored panel: its weights on calibrated members, every member called at once."""
    weights = stored_weights(root, task.cid, method)
    return panel(task, weights, member_choices(task), info={"source": "stored weights of an earlier study"},
                 ledger={"labeled_folds": "fit and validation", "models_profiled": task.b.M,
                         "note": "weights fitted by an earlier study, replayed"})


def replay_combiner(task, root):
    """Outcome of the stored pooled vectors of the earlier combiner over all models."""
    path = Path(root) / COMBINER / f"vectors-all22_with_providers-{task.cid:03d}.npz"
    with np.load(path, allow_pickle=False) as z:
        pooled = {int(k): z[k] for k in z.files}
    b = task.b
    if sorted(pooled) != b.widths or any(pooled[w].shape != (len(b.idx[w]), w) for w in b.widths):
        raise ValueError(f"benchmark {task.cid}: stored vectors do not match the field layout")
    ledger = dict(empty_ledger(), labeled_folds="fit and validation", models_profiled=b.M,
                  other_benchmark_labels=True, note="pooled vectors of an earlier study, replayed")
    return Outcome(pooled, np.zeros((b.R, b.M), np.int16), {"source": "stored pooled vectors"}, ledger)


def one_benchmark(cid, corpus, settings, root, digest, methods, with_text, replays=True):
    """Rows and per-field arrays of the replayed methods, and the native score of every listed method.

    ``replays``: False leaves out the methods with stored fits.  Their weights and vectors were fitted on the pool
    of all models, so they cannot be replayed on another pool.
    """
    task = load_task(corpus, cid)
    rows, fields, natives = [], {}, {}

    def native_of(outcome):
        return native.native_score(root, Path(root) / ROWS, task.b, outcome.pooled)

    for name in methods:
        natives[name] = native_of(BASELINES[name].run(task, settings))
    replayed = {}
    if replays:
        replayed = {name: replay_path(task, root, source) for name, source in PATH_METHODS.items()}
        replayed[COMBINER_METHOD] = replay_combiner(task, root)
    frontier = []
    if with_text:
        text = text_router_settings(settings)
        replayed[TEXT_ROUTER] = knn_router.run(task, text)
        for penalty in settings["knn_router"]["lambda_grid"]:
            scored = metrics.score(task, knn_router.run(task, text, penalty=penalty))
            frontier.append(RX._result_row(task, scored, TEXT_ROUTER, f"lambda={penalty}", digest))
    for name, outcome in replayed.items():
        scored = metrics.score(task, outcome)
        row = RX._result_row(task, scored, name, None, digest)
        row["native"] = native_of(outcome)
        rows.append(row)
        fields[name] = scored.mass, scored.hit.astype(bool)
    return cid, rows, fields, natives, frontier


def run(corpus, settings_path, out, root, processes=1):
    corpus, out, root = Path(corpus), Path(out), Path(root)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    document = json.loads((out / "rows.json").read_text(encoding="utf-8"))
    if document["settings_sha256"] != digest:
        raise SystemExit("rows.json was made with other settings; run the experiments first")
    local_names = set(PATH_METHODS) | {COMBINER_METHOD, TEXT_ROUTER}
    features = Path(settings.get(TEXT_ROUTER, {}).get("features", ""))
    with_text = TEXT_ROUTER in settings and (root / features / "features.json").is_file()
    rows = [r for r in document["rows"] if r["catalog_id"] is not None and r["method"] not in local_names]
    cids = sorted({r["catalog_id"] for r in rows})
    methods = [m for m in settings["methods"] if m in BASELINES]
    work = partial(one_benchmark, corpus=corpus, settings=settings, root=root, digest=digest, methods=methods,
                   with_text=with_text, replays=settings.get("replay_stored_fits", True))
    archives = {}
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            results = list(pool.map(work, cids))
    else:
        results = [work(c) for c in cids]
    native_of, frontier_rows = {}, []
    for cid, new_rows, fields, natives, frontier in results:
        rows.extend(new_rows)
        frontier_rows.extend(frontier)
        for name, value in natives.items():
            native_of[name, cid] = value
        for name, (mass, hit) in fields.items():
            archives.setdefault(name, {}).update({f"mass_{cid}": mass, f"hit_{cid}": hit})
    for row in rows:
        if (row["method"], row["catalog_id"]) in native_of and row["variant"] is None:
            row["native"] = native_of[row["method"], row["catalog_id"]]
    order = {m: i for i, m in enumerate(settings["methods"] + sorted(local_names))}
    rows.sort(key=lambda r: (order.get(r["method"], len(order)), str(r["variant"]), r["catalog_id"]))
    for name, arrays in archives.items():
        RX._write(RX._field_file(out, name), lambda stream, a=arrays: np.savez_compressed(stream, **a))
    RX._json(out / "rows.json", dict(synthetic=document["synthetic"], settings_sha256=digest,
                                     rows=RX._with_suites(rows, settings["suites"])))
    if frontier_rows:
        document = json.loads((out / "frontier.json").read_text(encoding="utf-8"))
        kept = [r for r in document["rows"] if r["catalog_id"] is not None and r["method"] != TEXT_ROUTER]
        RX._json(out / "frontier.json", dict(synthetic=document["synthetic"], settings_sha256=digest,
                                             rows=RX._with_suites(kept + frontier_rows, settings["suites"])))
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Add replayed methods and native scores to a results folder")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    n = run(a.corpus, a.settings, a.out, a.project_root, a.processes)
    print(f"{n} benchmark rows with native scores")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""What the open subset supports on its own: methods that need probability vectors and labels, nothing else.

    python3 -B -m analysis.open_track --corpus DIR --settings FILE --out FILE

``DIR`` is an open-tier corpus whose labels table was rebuilt with ``data.rehydrate`` from benchmark
rows.  The open tier holds the vectors of the cleared models only, and no usage or latency, so this track runs
the policies that need neither cost nor latency and reports log loss and accuracy.  Routing, cascades and early
exit need cost and are not part of it.

The output has benchmark rows and, per suite of the settings, equal-benchmark means.  Added after the first run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from data.load import list_benchmarks, read_manifest
from data.schema import sha256_file
from sim import core
from sim.baselines import BASELINES
from sim.engine import load_task

METHODS = ("best_single", "oracle", "uniform_mean", "majority_vote", "skill_pool", "uniform_temperature", "stack",
           "stack_final_temperature", "greedy")


def score(task, pooled):
    """Mean log loss and accuracy over the judged fields of one benchmark."""
    hit, mass, _, _ = core.decode(task.b, pooled)
    judged = task.b.judged
    loss = core.loss_of_mass(mass[judged])
    return dict(log_loss=math.fsum(loss.tolist()) / len(loss), accuracy=float(np.count_nonzero(hit[judged])) / len(loss))


def run(corpus, settings_path, out):
    corpus = Path(corpus)
    manifest = read_manifest(corpus)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    benchmarks = list_benchmarks(corpus)
    rows = []
    for info in benchmarks:
        task = load_task(corpus, info["catalog_id"])
        if task.b.cost is not None:
            raise ValueError("the open track expects a corpus without usage tables")
        for name in METHODS:
            rows.append(dict(method=name, catalog_id=task.cid, n_judged=task.nJ,
                             **score(task, BASELINES[name].run(task, settings).pooled)))
    cids = sorted({r["catalog_id"] for r in rows})
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in cids if c not in set(definition["exclude"])]
        if not members:
            continue
        suites[suite] = dict(benchmarks=len(members), methods={
            name: {metric: math.fsum(r[metric] for r in rows if r["method"] == name and r["catalog_id"] in members)
                   / len(members) for metric in ("log_loss", "accuracy")}
            for name in METHODS})
    document = dict(
        synthetic=all(b["name"].startswith("synthetic-") for b in benchmarks),
        settings_sha256=sha256_file(settings_path), corpus_checksums_sha256=manifest["checksums_sha256"],
        tier=manifest["tier"], models=manifest["model_count"], benchmarks=len(cids), methods=list(METHODS),
        rows=rows, suites=suites)
    Path(out).write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return document


def main(argv=None):
    ap = argparse.ArgumentParser(description="Methods that the open subset supports on its own")
    for name in ("corpus", "settings", "out"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args(argv)
    document = run(a.corpus, a.settings, a.out)
    for suite, block in document["suites"].items():
        print(suite, block["benchmarks"], {m: round(v["log_loss"], 4) for m, v in block["methods"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Reproduction check: results of this package against the stored results of the earlier studies.

    python3 -B -m acl_resource.analysis.anchors --corpus DIR --results DIR --diagnostics DIR --project-root . --out FILE

A local step: the stored result files exist only where the earlier studies were run.  Every anchor compares
numbers that this package computed from the processed corpus with numbers stored by earlier code, and records the
number of values compared and the largest absolute gap.  Replayed quantities must agree to 1e-12.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from ..data.load import list_benchmarks, list_models, load_benchmark

TOLERANCE = 1e-12
STORED_ROWS = "ensemble_round2/results/rows_by_benchmark.csv"
STORED_SINGLE = "results/decision_index/expanded-paid-166/baselines.csv"
STORED_CONTRASTS = "ensemble_round2/results/comparisons.csv"
STORED_SELECTION = "results/decision_index/paper/statistics/primary_contrasts.csv"
STORED_LATENCY = "results/decision_index/paper/costs/suite_latency.csv"
# method of this package -> row name of the stored table
METHOD_ROWS = {
    "uniform_mean": "context_flat_mean_22",
    "best_single": "context_single_calibrated_four_folds",
    "skill_pool": "e6_skill_pool",
    "uniform_temperature": "e3b_flat_mean_temperature_fit_folds",
    "stack": "e3_stack_recalibrated",
    "stack_final_temperature": "e3_stack_recalibrated_final_temperature",
    "tree_replay": "context_tree_calibrated",
    "combiner_replay": "e5_all22_combiner_with_providers",
}
METHOD_COLUMNS = {"log_loss": "log_loss", "accuracy": "accuracy", "native": "native",
                  "calls_per_request": "calls_per_request", "usd_per_1k_requests": "usd_per_1k_requests"}


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _anchor(name, what, gaps, tolerance=TOLERANCE):
    gaps = [abs(float(g)) for g in gaps]
    return dict(anchor=name, what=what, values_compared=len(gaps), largest_gap=max(gaps) if gaps else None,
                tolerance=tolerance, reproduced=bool(gaps and max(gaps) <= tolerance))


def single_models(root, diagnostics, models):
    """The stored single-model rows: accuracy by reported choice, log loss on the vectors as collected."""
    stored = {(int(r["catalog_id"]), r["model"]): r for r in _csv(root / STORED_SINGLE) if r["baseline"] == "route"}
    acc, loss = [], []
    for r in _csv(diagnostics / "models.csv"):
        ref = stored[int(r["catalog_id"]), models[int(r["model_index"])]["model_name"]]
        acc.append(float(r["accuracy_reported"]) - float(ref["field_accuracy"]))
        loss.append(float(r["log_loss_raw_floor_1e-15"]) - float(ref["field_log_loss"]))
    return [_anchor("single_model_accuracy", "every model on every benchmark, provider's reported choice", acc),
            _anchor("single_model_log_loss", "every model on every benchmark, vectors as collected", loss)]


def methods(root, rows):
    stored = {(r["row"], int(r["catalog_id"])): r for r in _csv(root / STORED_ROWS)}
    out = []
    for method, name in METHOD_ROWS.items():
        gaps = []
        for r in rows:
            if r["method"] == method and r["variant"] is None and r["catalog_id"] is not None:
                ref = stored[name, r["catalog_id"]]
                gaps += [r[mine] - float(ref[theirs]) for mine, theirs in METHOD_COLUMNS.items()]
        out.append(_anchor(f"method_{method}", f"log loss, accuracy, own metric, calls and cost per benchmark ({name})", gaps))
    return out


def contrasts(root, rows):
    mine = {(r["contrast"], r["metric"]): r for r in rows if r["suite"] == "all36"}
    out = []
    stored = {r["comparison"]: r for r in _csv(root / STORED_CONTRASTS)}
    for name, key in (("C2", "H3b"), ("C3", "H6")):
        a, b = mine[name, "log_loss"], stored[key]
        out.append(_anchor(f"contrast_{name}", f"difference, interval and p-value (stored as {key})",
                           [a["difference"] - float(b["difference"]), a["ci_low"] - float(b["ci_low"]),
                            a["ci_high"] - float(b["ci_high"]), a["p"] - float(b["p"])]))
    stored = {r["comparison"]: r for r in _csv(root / STORED_SELECTION) if r["metric"] == "field_log_loss"}
    for name, key in (("C6", "tree_vs_singleton"), ("C7", "tree_vs_greedy")):
        a, b = mine[name, "log_loss"], stored[key]          # the stored difference is baseline minus tree
        out.append(_anchor(f"contrast_{name}", f"difference, interval and p-value (stored as {key}, sign reversed)",
                           [a["difference"] + float(b["delta"]), a["ci_low"] + float(b["ci_high"]),
                            a["ci_high"] + float(b["ci_low"]), a["p"] - float(b["p_unadjusted"])]))
    return out


def panel_latency(root, corpus):
    """All models called for every request: latency in parallel and in series, over all requests."""
    parallel, serial = [], []
    for info in list_benchmarks(corpus):
        b = load_benchmark(corpus, info["catalog_id"], labels=False)
        parallel.append(b.lat.max(axis=1))
        serial.append(b.lat.sum(axis=1))
    parallel, serial = np.concatenate(parallel), np.concatenate(serial)
    ref = next(r for r in _csv(root / STORED_LATENCY) if r["method"] == "all22_mean")
    return [
        _anchor("panel_latency_parallel", "median and 95th percentile, all models called in parallel",
                [np.percentile(parallel, 50) - float(ref["parallel_est_p50_s"]),
                 np.percentile(parallel, 95) - float(ref["parallel_est_p95_s"])]),
        _anchor("panel_latency_serial_median", "median, all models called in series",
                [np.percentile(serial, 50) - float(ref["serial_service_est_p50_s"])]),
        # Known gap of about 0.002 s: the stored figure treats the requests answered in field batches in a way
        # that the frozen corpus does not record.  Kept as an anchor so that the gap stays visible.
        _anchor("panel_latency_serial_p95", "95th percentile, all models called in series",
                [np.percentile(serial, 95) - float(ref["serial_service_est_p95_s"])]),
    ]


def run(corpus, results, diagnostics, root, out):
    corpus, results, diagnostics, root = Path(corpus), Path(results), Path(diagnostics), Path(root)
    rows = json.loads((results / "rows.json").read_text(encoding="utf-8"))["rows"]
    contrast_rows = json.loads((results / "contrasts.json").read_text(encoding="utf-8"))["rows"]
    anchors = (single_models(root, diagnostics, list_models(corpus)) + methods(root, rows)
               + contrasts(root, contrast_rows) + panel_latency(root, corpus))
    summary = dict(
        anchors=anchors, n_anchors=len(anchors), n_reproduced=sum(a["reproduced"] for a in anchors),
        values_compared=sum(a["values_compared"] for a in anchors),
        largest_gap=max(a["largest_gap"] for a in anchors if a["largest_gap"] is not None),
        all_reproduced=all(a["reproduced"] for a in anchors), tolerance=TOLERANCE,
        not_reproduced=[a["anchor"] for a in anchors if not a["reproduced"]],
        largest_gap_among_reproduced=max(a["largest_gap"] for a in anchors if a["reproduced"]))
    Path(out).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare results with the stored results of earlier studies")
    for name in ("corpus", "results", "diagnostics", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--project-root", default=".")
    a = ap.parse_args(argv)
    s = run(a.corpus, a.results, a.diagnostics, a.project_root, a.out)
    for x in s["anchors"]:
        print(f"{'ok ' if x['reproduced'] else 'NO '} {x['anchor']:36s} n={x['values_compared']:5d} gap={x['largest_gap']:.1e}")
    print(f"{s['n_reproduced']} of {s['n_anchors']} anchors reproduced; {s['values_compared']} values; largest gap {s['largest_gap']:.1e}")
    return 0 if s["all_reproduced"] else 1


if __name__ == "__main__":
    sys.exit(main())

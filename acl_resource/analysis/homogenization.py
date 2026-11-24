"""Two published measures of homogenization, computed for every pair of models on every benchmark.

    python3 -B -m acl_resource.analysis.homogenization --corpus DIR --out DIR [--settings FILE]

* Agreement when both are wrong (Kim et al., 2025): over the judged fields that two models both answer wrongly,
  the share on which they pick the same wrong option.  Chance level under a uniform choice among the wrong
  options: the mean of 1 / (number of wrong options).  Only fields with at least two wrong options count, since
  with one wrong option two wrong models must agree.
* Joint-failure ratio (Bommasani et al., 2022): the share of judged fields that all chosen models fail, divided
  by the product of their error rates.  1 means the failures are as frequent as under independence.

Two further measures were added after the first run, to separate what the models share from what the items
share:

* Error correlation given the other models: the correlation of two models' errors inside strata of judged
  fields on which the same number of the *other* models are wrong, pooled over the strata.  A field's stratum is
  a proxy for its difficulty.  The proxy is imperfect, so models that are independent given the true difficulty
  can still keep a small positive value; a difference between groups of pairs is the interpretable part.
* Agreement where no model is right: over the judged fields with at least two wrong options that every model
  answers wrongly, the share on which all models, and the share on which at least three quarters of them, pick
  the same wrong option.  Such fields are candidates for an error in the gold label.

Decisions use the first maximum of the normalised vector.  The output holds aggregates only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from ..data.load import list_benchmarks, list_models, load_benchmark
from ..sim import core

PAIR_COLUMNS = ("catalog_id", "model_a", "model_b", "same_organisation", "same_base_model", "judged_fields",
                "both_wrong", "joint_failure_ratio", "both_wrong_with_choice", "both_wrong_agreement",
                "chance_agreement", "error_correlation", "error_correlation_given_others")
PAIR_MEASURES = ("joint_failure_ratio", "both_wrong_agreement", "chance_agreement", "error_correlation",
                 "error_correlation_given_others")
PANEL_MEASURES = ("none_right", "joint_failure_ratio", "root_ratio", "none_right_unanimous",
                  "none_right_three_quarters")
GROUPS = ("same_organisation", "same_base_model", "different_organisation")


def decisions(bench):
    """(hit [J, M] bool, choice [J, M], wrong options [J]) on the judged fields, first-maximum rule."""
    hit = np.empty((bench.F, bench.M), bool)
    choice = np.empty((bench.F, bench.M), np.int64)
    for m in range(bench.M):
        h, _, c, _ = core.decode(bench, core.single_model_vectors(bench, m))
        hit[:, m], choice[:, m] = h, c
    j = bench.judged
    wrong_options = (bench.width.astype(np.int64) - bench.n_accepted.astype(np.int64))[j]
    return hit[j], choice[j], wrong_options


def stratified_correlation(a, b, stratum, n_strata):
    """(plain, within-stratum) correlation of two 0/1 arrays; None where a variance is zero."""
    def pooled(strata):
        n = np.bincount(strata, minlength=n_strata).astype(np.float64)
        sa, sb = (np.bincount(strata, weights=x, minlength=n_strata) for x in (a, b))
        sab = np.bincount(strata, weights=a * b, minlength=n_strata)
        keep = n > 0
        n, sa, sb, sab = n[keep], sa[keep], sb[keep], sab[keep]
        cov = float((sab - sa * sb / n).sum())
        var = float((sa - sa * sa / n).sum()) * float((sb - sb * sb / n).sum())
        return cov / math.sqrt(var) if var > 0 else None
    return pooled(np.zeros(len(a), np.int64)), pooled(stratum)


def modal_count(choice):
    """Per row, how many entries share the most frequent value."""
    ordered = np.sort(choice, axis=1)
    run = np.ones(len(ordered), np.int64)
    best = run.copy()
    for j in range(1, ordered.shape[1]):
        run = np.where(ordered[:, j] == ordered[:, j - 1], run + 1, 1)
        best = np.maximum(best, run)
    return best


def pair_rows(bench):
    """One dict per model pair of one benchmark; an undefined value is None."""
    hit, choice, wrong_options = decisions(bench)
    wrong = ~hit
    n = len(wrong)
    error = wrong.mean(axis=0)
    as_number = wrong.astype(np.float64)
    wrong_models = wrong.sum(axis=1)
    eligible = wrong_options >= 2
    chance = np.where(eligible, 1.0 / np.maximum(wrong_options, 1), 0.0)
    org = [m["organisation"] for m in bench.models]
    base = [m["base_model"] for m in bench.models]
    rows = []
    for a in range(bench.M):
        for b in range(a + 1, bench.M):
            both = wrong[:, a] & wrong[:, b]
            with_choice = both & eligible
            k = int(with_choice.sum())
            others = wrong_models - wrong[:, a] - wrong[:, b]
            plain, given = stratified_correlation(as_number[:, a], as_number[:, b], others, bench.M - 1)
            rows.append(dict(
                error_correlation=plain, error_correlation_given_others=given,
                catalog_id=bench.cid, model_a=a, model_b=b,
                same_organisation=org[a] == org[b], same_base_model=bool(base[a]) and base[a] == base[b],
                judged_fields=n, both_wrong=int(both.sum()),
                joint_failure_ratio=float(both.mean() / (error[a] * error[b])) if error[a] * error[b] > 0 else None,
                both_wrong_with_choice=k,
                both_wrong_agreement=float((choice[with_choice, a] == choice[with_choice, b]).mean()) if k else None,
                chance_agreement=float(chance[with_choice].mean()) if k else None))
    nobody = wrong.all(axis=1)
    none_right = float(nobody.mean())
    product = float(np.prod(error))
    with_choice = nobody & eligible
    most = modal_count(choice[with_choice]) if with_choice.any() else np.zeros(0, np.int64)
    panel = dict(
        none_right_with_choice=int(with_choice.sum()),
        none_right_unanimous=float((most == bench.M).mean()) if len(most) else None,
        none_right_three_quarters=float((most >= math.ceil(0.75 * bench.M)).mean()) if len(most) else None,
        catalog_id=bench.cid, judged_fields=n, none_right=none_right,
        joint_failure_ratio=none_right / product if product > 0 else None,
        root_ratio=(none_right ** (1.0 / bench.M)) / float(np.exp(np.log(error).mean())) if product > 0 else None)
    return rows, panel


def _mean(values):
    values = [v for v in values if v is not None]
    return (math.fsum(values) / len(values) if values else None), len(values)


def group_of(row):
    if row["same_base_model"]:
        return "same_base_model"
    return "same_organisation" if row["same_organisation"] else "different_organisation"


def summarize(rows, panels, suites, catalog_ids):
    """Equal-benchmark means per suite: first the mean over the pairs of a group within a benchmark."""
    out = {}
    for suite, spec in suites.items():
        cids = [c for c in catalog_ids if c not in set(spec.get("exclude", []))]
        block = {}
        for group in GROUPS + ("all_pairs",):
            per_bench = {k: [] for k in PAIR_MEASURES}
            pairs = set()
            for c in cids:
                mine = [r for r in rows if r["catalog_id"] == c and (group == "all_pairs" or group_of(r) == group)]
                pairs.update((r["model_a"], r["model_b"]) for r in mine)
                for k in per_bench:
                    per_bench[k].append(_mean([r[k] for r in mine])[0])
            block[group] = dict(pairs=len(pairs))
            for k, values in per_bench.items():
                block[group][k], block[group][k + "_benchmarks"] = _mean(values)
        block["all_models"] = {}
        for k in PANEL_MEASURES:
            block["all_models"][k], block["all_models"][k + "_benchmarks"] = _mean(
                [p[k] for p in panels if p["catalog_id"] in cids])
        mine = [p for p in panels if p["catalog_id"] in cids and p["none_right_with_choice"]]
        fields = sum(p["none_right_with_choice"] for p in mine)
        block["all_models"]["none_right_with_choice"] = fields
        for k in ("none_right_unanimous", "none_right_three_quarters"):
            block["all_models"][k + "_pooled"] = (
                math.fsum(p[k] * p["none_right_with_choice"] for p in mine) / fields if fields else None)
        out[suite] = block
    return out


def run(corpus, out, settings=None):
    corpus, out = Path(corpus), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    benchmarks = list_benchmarks(corpus)
    suites = (json.loads(Path(settings).read_text())["suites"] if settings else {"all": {"exclude": []}})
    rows, panels = [], []
    for info in benchmarks:
        r, p = pair_rows(load_benchmark(corpus, info["catalog_id"], labels=True, usage=False, latency=False))
        rows += r
        panels.append(p)
    with open(out / "homogenization_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(PAIR_COLUMNS)
        for r in rows:
            w.writerow(["" if r[c] is None else (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in PAIR_COLUMNS])
    summary = dict(
        synthetic=all(b["name"].startswith("synthetic-") for b in benchmarks),
        models=len(list_models(corpus)), benchmarks=len(benchmarks),
        suites=summarize(rows, panels, suites, [b["catalog_id"] for b in benchmarks]), panels=panels)
    (out / "homogenization.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Homogenization measures per model pair and benchmark")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings")
    a = ap.parse_args(argv)
    summary = run(a.corpus, a.out, a.settings)
    print(json.dumps({s: v["all_pairs"] for s, v in summary["suites"].items()}, sort_keys=True)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())

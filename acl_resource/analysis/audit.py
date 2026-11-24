#!/usr/bin/env python3
"""Read-only audit of the frozen corpus: 22 decision models on 36 text benchmarks.

    python3 -B acl_resource/analysis/audit.py            # recompute; write the summary and the audit document
    python3 -B acl_resource/analysis/audit.py --check    # recompute and compare with the stored summary

Run from the project root.  Every input is opened for reading only.  The two outputs go to acl_resource/docs/:
AUDIT_SUMMARY.json (every number) and DATA_AUDIT.md (docs/templates/DATA_AUDIT.template.md with numbers and tables
filled in).  Label files and gold masks are read to count structure; no item text, option text, gold value or
per-item value is printed or written.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
DOCS = PKG / "docs"

# Inputs, relative to the project root.  All read only.
CORPUS = "results/decision_index/paper/dataset/candidate"
PUBLIC = "results/decision_index/paper/dataset/public-stage"
LABELS = "data/decision_index/common-paid-65/inputs"
ARRAYS = "cluster_confidence_check/cache"
RIGHTS = "results/decision_index/paper/rights"
COLLECT = "results/decision_index/expanded-paid-166"
LATENCY = "results/decision_index/paper/costs/suite_latency.csv"
CARDS = "ensemble_round2/model_cards/cards.csv"
ROWS = "ensemble_round2/results/rows.csv"
INDEX = "data/decision_index/upstream/kit/decision_index/data/index-0.3.json"
RELATED = "ensemble_round2/related_work"
METHODS = "acl_resource/docs/method_compatibility.csv"
METHODS_TOPUP = "acl_resource/docs/method_compatibility_topup.csv"
TOPUP_NOTES = "acl_resource/related_work/evidence"

SATURATION = 0.95
SUM_TOLERANCE = 1e-9
STRAY = ("ruvector.db", "agentdb.rvf", "agentdb.rvf.lock", ".DS_Store")
# Screened papers that the comparison table and the routing and homogenization paragraphs need (corpus ids).
TOP_UP = (
    "doi:10.48550/arxiv.2403.12031", "doi:10.48550/arxiv.2601.07206", "doi:10.48550/arxiv.2309.15789",
    "doi:10.48550/arxiv.2410.02223", "doi:10.48550/arxiv.2310.12963", "doi:10.48550/arxiv.2307.02764",
    "doi:10.48550/arxiv.2410.10347", "openreview:Chn50flK4X", "openreview:EEPostHMtF",
    "doi:10.48550/arxiv.2506.07962", "doi:10.52202/068431-0265", "doi:10.1073/pnas.2018340118",
    "doi:10.48550/arxiv.2502.00674",
)
# Rows of the round-2 results table that correspond to the recommended baselines.
REFERENCE_ROWS = (
    ("context_flat_mean_22", "uniform pool of all 22 (mean of probabilities)"),
    ("e6_skill_pool", "label-free skill-weighted pool of all 22"),
    ("e3b_flat_mean_temperature_fit_folds", "uniform pool with one temperature"),
    ("context_single_calibrated_four_folds", "best single model, recalibrated"),
    ("context_tree_calibrated", "tree-guided selection (earlier manuscript), recalibrated"),
    ("e3_stack_recalibrated", "simplex stacking on recalibrated members"),
    ("e3_stack_recalibrated_final_temperature", "the same with a final temperature"),
    ("e5_all22_combiner_with_providers", "conditional-logit combiner over all 22"),
)
KIT = "data/decision_index/upstream/kit"
LEDGERS = "data/decision_index"


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------

def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def neumaier(x):
    """Row sums exactly as CPython >= 3.12 computes ``sum(list_of_floats)``; ``x`` is [n, w].

    Ported from cluster_confidence_check/data.py::builtin_sum (the stored pipeline divides by this total).
    """
    s = x[:, 0].copy()
    c = np.zeros_like(s)
    for j in range(1, x.shape[1]):
        xj = x[:, j]
        t = s + xj
        c += np.where(np.abs(s) >= np.abs(xj), (s - t) + xj, (xj - t) + s)
        s = t
    return np.where((c != 0) & np.isfinite(c), s + c, s)


def running_sum(x):
    """Row sums by plain left-to-right float addition; ``x`` is [n, w]."""
    s = x[:, 0].copy()
    for j in range(1, x.shape[1]):
        s = s + x[:, j]
    return s


def plain(o):
    """JSON-ready copy: numpy scalars and arrays become Python values, NaN becomes None."""
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if o != o else float(o)
    return o


# --------------------------------------------------------------------------------------------------
# corpus index, checksums
# --------------------------------------------------------------------------------------------------

def corpus_index():
    c = ROOT / CORPUS
    models = pd.read_csv(c / "models.csv")
    bench = pd.read_csv(c / "benchmarks.csv")
    manifest = json.loads((c / "manifest.json").read_text())
    if models.model_index.tolist() != list(range(len(models))):
        raise ValueError("model indexes are not 0..M-1")
    if bench.catalog_id.tolist() != manifest["catalog_ids"]:
        raise ValueError("benchmarks.csv and manifest.json list different catalogue ids")
    return models, bench, manifest


def checksums(manifest):
    c = ROOT / CORPUS
    listed = {}
    for line in (c / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        listed[name.strip()] = digest
    mismatched = sorted(n for n, d in listed.items() if not (c / n).is_file() or sha256_file(c / n) != d)
    present = sorted(str(p.relative_to(c)) for p in c.rglob("*") if p.is_file())
    unlisted = [n for n in present if n not in listed and n != "SHA256SUMS"]
    return dict(
        listed_files=len(listed), matching=len(listed) - len(mismatched), mismatched=mismatched,
        list_bound_to_manifest=sha256_file(c / "SHA256SUMS") == manifest["checksums_sha256"],
        unlisted_files=unlisted,
        unlisted_stray_hook_files=[n for n in unlisted if Path(n).name in STRAY],
        parquet_files=sum(1 for n in listed if n.endswith(".parquet")),
        bytes_listed=sum((c / n).stat().st_size for n in listed if (c / n).is_file()),
    )


def storage():
    """Sizes from file metadata only.  The ledgers are not opened."""
    def folder(rel):
        return int(sum(f.stat().st_size for f in (ROOT / rel).rglob("*") if f.is_file()))
    ledgers = []
    for f in sorted((ROOT / LEDGERS).rglob("*.sqlite3")):
        wal = f.with_name(f.name + "-wal")
        ledgers.append(dict(path=str(f.relative_to(ROOT)), bytes=int(f.stat().st_size),
                            wal_bytes=int(wal.stat().st_size) if wal.exists() else 0))
    return dict(ledgers=ledgers, ledger_files=len(ledgers),
                ledger_bytes=int(sum(x["bytes"] + x["wal_bytes"] for x in ledgers)),
                largest_wal_bytes=int(max(x["wal_bytes"] for x in ledgers)),
                corpus_bytes=folder(CORPUS), public_stage_bytes=folder(PUBLIC), labels_bytes=folder(LABELS),
                kit_bytes=folder(KIT), arrays_bytes=folder(ARRAYS))


# --------------------------------------------------------------------------------------------------
# labels: structure of the options (counts only)
# --------------------------------------------------------------------------------------------------

def option_structure(cid):
    """Counts from the local label file.  Option texts are hashed in memory and never leave this function."""
    by_text, by_key, widths, per_request, types = set(), set(), [], [], Counter()
    with gzip.open(ROOT / LABELS / f"{cid:03d}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            questions = json.loads(line)["questions"]
            per_request.append(len(questions))
            for q in questions.values():
                types[q["type"]] += 1
                if q["type"] == "choice":
                    texts = sorted(json.dumps(v, sort_keys=True, ensure_ascii=False) for v in q["criteria"].values())
                    keys = sorted(str(k) for k in q["criteria"])
                elif q["type"] == "noul":
                    texts = keys = ["no", "yes"]
                else:
                    raise ValueError(f"unsupported field type {q['type']!r}")
                widths.append(len(keys))
                by_text.add(hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).digest())
                by_key.add(hashlib.sha256(json.dumps(keys, ensure_ascii=False).encode("utf-8")).digest())
    w, r = np.asarray(widths), np.asarray(per_request)
    return dict(
        choice_fields=types["choice"], yes_no_fields=types["noul"],
        options_min=int(w.min()), options_median=float(np.median(w)), options_max=int(w.max()),
        distinct_option_counts=int(len(np.unique(w))),
        fields_per_request_min=int(r.min()), fields_per_request_median=float(np.median(r)),
        fields_per_request_max=int(r.max()),
        option_inventories_by_text=len(by_text), option_inventories_by_key=len(by_key),
        fixed_inventory=len(by_text) == 1, all_binary=bool(w.max() == 2),
    )


# --------------------------------------------------------------------------------------------------
# the 792 cells
# --------------------------------------------------------------------------------------------------

def scan(models, bench, cards, stored):
    """One pass over every benchmark and model.  Returns (per-benchmark, per-model, per-cell) records."""
    M = len(models)
    missing = [int(c) for c in bench.catalog_id if not (ROOT / ARRAYS / f"bench-{int(c):03d}.npz").is_file()]
    if missing:
        raise SystemExit(f"round-1 arrays are missing for catalogues {missing}; this script does not rebuild them")
    price_in = cards.set_index("model_index").price_prompt_usd_per_token.to_dict()
    price_out = cards.set_index("model_index").price_completion_usd_per_token.to_dict()
    stored_acc = stored.set_index(["catalog_id", "model"]).field_accuracy.to_dict()

    per_bench, cells = {}, []
    usage = {m: dict(tin=[], tout=[], cost=[], lat=[], calls=[], fields=[]) for m in range(M)}
    for info in bench.itertuples():
        cid = int(info.catalog_id)
        d = ROOT / CORPUS / f"catalog-{cid:03d}"
        items = pq.read_table(d / "items.parquet").to_pandas()
        R = len(items)
        if items.item_index.tolist() != list(range(R)):
            raise ValueError(f"catalogue {cid}: item indexes are not 0..R-1")
        fold = items.fold.to_numpy()
        with np.load(ROOT / ARRAYS / f"bench-{cid:03d}.npz", allow_pickle=False) as z:
            width = z["width"].astype(np.int64)
            is_noul, judged = z["is_noul"], z["judged"]
            field_row, field_pos = z["field_row"], z["field_pos"]
            F = len(width)
            off0 = np.concatenate([[0], np.cumsum(width)])
            accepted = np.zeros(off0[-1], bool)          # flat mask of accepted options, aligned with the vectors
            gold_class = np.full(F, -1, np.int64)
            groups = {int(w): np.flatnonzero(width == w) for w in np.unique(width)}
            for w, idx in groups.items():
                if not np.array_equal(z[f"idx_{w}"], idx):
                    raise ValueError(f"catalogue {cid}: cached field order differs")
                acc = z[f"acc_{w}"]
                accepted[(off0[idx][:, None] + np.arange(w)).ravel()] = acc.ravel()
                one = acc.sum(1) == 1
                gold_class[idx[one]] = z[f"lab_{w}"][one][np.arange(int(one.sum())), acc[one].argmax(1)]
        if not np.array_equal(np.bincount(field_row, minlength=R), items.field_count.to_numpy()):
            raise ValueError(f"catalogue {cid}: field counts differ between items and arrays")
        nj = int(judged.sum())
        starts = off0[:-1]
        hits = np.zeros((F, M), bool)                    # first-maximum rule
        zero_truth_bench = 0
        for m in range(M):
            t = pq.read_table(d / f"model-{m:02d}" / "predictions.parquet")
            col = t["probabilities"].combine_chunks()
            off = col.offsets.to_numpy().astype(np.int64)
            val = col.values.to_numpy(zero_copy_only=False)
            noul_col = pc.equal(t["question_type"], "noul").combine_chunks().to_numpy(zero_copy_only=False)
            layout_ok = bool(
                t.num_rows == F and off[0] == 0 and np.array_equal(off, off0)
                and np.array_equal(t["item_index"].to_numpy(), field_row)
                and np.array_equal(t["field_index"].to_numpy(), field_pos)
                and np.array_equal(noul_col, is_noul))
            if not layout_ok:
                raise ValueError(f"catalogue {cid}, model {m}: prediction layout differs from the field layout")
            rc_null = int(t["reported_choice"].null_count)
            rc = pc.fill_null(t["reported_choice"].combine_chunks(), -1).to_numpy(zero_copy_only=False).astype(np.int64)

            sums, exact, first = np.empty(F), np.empty(F), np.empty(F, np.int64)
            for w, idx in groups.items():
                x = val[starts[idx][:, None] + np.arange(w)]
                sums[idx] = running_sum(x)               # the total the stored counters were made with
                exact[idx] = neumaier(x)                 # the total the stored pipeline divides by
                first[idx] = x.argmax(1)
            # yes/no fields: "yes" (position 1) when its normalised probability is at least one half
            if is_noul.any():
                p0, p1 = val[starts[is_noul]], val[starts[is_noul] + 1]
                first[is_noul] = (p1 / (p0 + p1) >= 0.5).astype(np.int64)
            reported = np.where(rc >= 0, rc, first)
            hit_first = accepted[starts + first]
            hit_reported = accepted[starts + reported]
            hits[:, m] = hit_first
            mass = np.add.reduceat(val * accepted, starts)
            zero_truth = int(((mass == 0) & judged).sum())
            zero_truth_bench += zero_truth
            acc_first = float(hit_first[judged].mean())
            acc_reported = float(hit_reported[judged].mean())
            model_id = models.canonical_model_id[m]
            route_id = models.route_id[m]
            ref = stored_acc.get((cid, route_id), stored_acc.get((cid, model_id)))

            o = pq.read_table(d / f"model-{m:02d}" / "observations.parquet")
            adapted = int(o.num_rows - o["adapted_field_partitions"].null_count)
            od = o.drop_columns(["adapted_field_partitions"]).to_pandas()
            if od.item_index.tolist() != list(range(R)):
                raise ValueError(f"catalogue {cid}, model {m}: usage rows are not one per request")
            tin = od.leaf_input_tokens.to_numpy(float)
            tout = od.leaf_output_tokens.to_numpy(float)
            cost = od.leaf_cost_usd.to_numpy(float)
            calls = od.leaf_physical_calls.to_numpy(float)
            signed = cost - (tin * price_in[m] + tout * price_out[m])
            resid = np.abs(signed)
            beyond = resid > 5e-10 * np.maximum(calls, 1) + 1e-12
            for k, v in (("tin", tin), ("tout", tout), ("cost", cost), ("calls", calls),
                         ("lat", od.leaf_deployment_latency_s.to_numpy(float)),
                         ("fields", items.field_count.to_numpy(float))):
                usage[m][k].append(v)
            cells.append(dict(
                catalog_id=cid, model_index=m, fields=F, values=int(len(val)),
                nan=int(np.isnan(val).sum()), negative=int((val < 0).sum()), above_one=int((val > 1).sum()),
                empty_vectors=int((np.diff(off) == 0).sum()), zero_entries=int((val == 0).sum()),
                off_sum=int((np.abs(sums - 1) > SUM_TOLERANCE).sum()),
                off_sum_compensated=int((np.abs(exact - 1) > SUM_TOLERANCE).sum()),
                off_sum_1e6=int((np.abs(sums - 1) > 1e-6).sum()),
                choice_sum_not_one=int(((sums != 1.0) & ~is_noul).sum()),
                sum_min=float(sums.min()), sum_max=float(sums.max()),
                reported_null=rc_null, yes_no_fields=int(is_noul.sum()),
                reported_differs_first=int(((rc >= 0) & (rc != first)).sum()),
                reported_differs_at_exact_tie=int(((rc >= 0) & (rc != first)
                                                   & (val[starts + np.maximum(rc, 0)] == val[starts + first])).sum()),
                zero_mass_on_accepted=zero_truth,
                accuracy_first_max=acc_first, accuracy_reported=acc_reported, accuracy_stored=ref,
                requests=R, adapted_observations=adapted,
                transport=dict(Counter(od.transport.fillna("(null)"))),
                providers=dict(Counter(od.provider.fillna("(null)"))),
                nulls={c: int(n) for c, n in od.isna().sum().items() if n},
                input_tokens=int(tin.sum()), output_tokens=int(tout.sum()), cost_usd=float(cost.sum()),
                physical_calls=int(calls.sum()), realized_physical_calls=int(od.realized_physical_calls.sum()),
                price_card_max_abs_error_usd=float(resid.max()),
                price_card_rows_beyond_rounding=int(beyond.sum()),
                price_card_difference_usd=float(signed[beyond].sum()),
                price_card_ratio_min=float((cost[beyond] / (tin[beyond] * price_in[m])).min()) if beyond.any() else None,
                price_card_ratio_max=float((cost[beyond] / (tin[beyond] * price_in[m])).max()) if beyond.any() else None,
                price_card_beyond_by_provider=dict(Counter(od.provider.fillna("(null)")[beyond])),
            ))

        h = hits[judged]
        k = h.sum(1)
        per_model_acc = h.mean(0)
        gc = gold_class[judged & (gold_class >= 0)]
        per_bench[cid] = dict(
            requests=R, fields=F, judged_fields=nj, single_gold_fields=int((gold_class >= 0).sum()),
            split_groups=int(items.split_group_id.nunique()), native_groups=int(items.native_group_id.nunique()),
            folds=sorted(int(f) for f in np.unique(fold)),
            fold_requests=[int((fold == f).sum()) for f in range(5)],
            fold_split_groups=[int(items.split_group_id[fold == f].nunique()) for f in range(5)],
            fold_judged_fields=[int((judged & (fold[field_row] == f)).sum()) for f in range(5)],
            split_groups_in_two_folds=int((items.groupby("split_group_id").fold.nunique() > 1).sum()),
            native_groups_in_two_folds=int((items.groupby("native_group_id").fold.nunique() > 1).sum()),
            judged_gold_classes=int(len(np.unique(gc))),
            headroom=dict(
                best_single_hindsight=float(per_model_acc.max()), best_single_model=int(per_model_acc.argmax()),
                worst_single=float(per_model_acc.min()), mean_single=float(per_model_acc.mean()),
                any_correct=float((k >= 1).mean()), two_or_more=float((k >= 2).mean()),
                majority_12_or_more=float((k >= 12).mean()), all_correct=float((k == M).mean()),
                none_correct=float((k == 0).mean()), mean_models_correct=float(k.mean())),
            all_models_cost_per_1k_requests=float(1000 * sum(usage[m]["cost"][-1].sum() for m in range(M)) / R),
            **option_structure(cid),
        )
        print(f"  catalogue {cid:3d} {info.dataset[:28]:28s} requests {R:6d} fields {F:6d}", flush=True)

    per_model = {}
    for m in range(M):
        u = {k: np.concatenate(v) for k, v in usage[m].items()}
        mine = [c for c in cells if c["model_index"] == m]
        design = np.stack([u["tin"], u["tout"]], axis=1)
        if not u["tout"].any():
            design = design[:, :1]
        coef, *_ = np.linalg.lstsq(design, u["cost"], rcond=None)
        fit_resid = np.abs(u["cost"] - design @ coef)
        nulls = Counter()
        for c in mine:
            nulls.update(c["nulls"])
        providers, transport, beyond_by = Counter(), Counter(), Counter()
        for c in mine:
            providers.update(c["providers"])
            transport.update(c["transport"])
            beyond_by.update(c["price_card_beyond_by_provider"])
        ratios = [c[k] for c in mine for k in ("price_card_ratio_min", "price_card_ratio_max") if c[k] is not None]
        per_model[m] = dict(
            requests=int(len(u["cost"])), fields=int(sum(c["fields"] for c in mine)),
            input_tokens=int(u["tin"].sum()), output_tokens=int(u["tout"].sum()),
            requests_with_zero_output_tokens=int((u["tout"] == 0).sum()),
            requests_with_one_output_token_per_field=int((u["tout"] == u["fields"]).sum()),
            cost_usd=float(u["cost"].sum()), cost_per_1k_requests=float(1000 * u["cost"].mean()),
            physical_calls=int(u["calls"].sum()),
            latency_p50_s=float(np.percentile(u["lat"], 50)), latency_p95_s=float(np.percentile(u["lat"], 95)),
            latency_p99_s=float(np.percentile(u["lat"], 99)), latency_max_s=float(u["lat"].max()),
            adapted_observations=int(sum(c["adapted_observations"] for c in mine)),
            nulls=dict(sorted(nulls.items())), providers=dict(providers.most_common()), transport=dict(transport),
            price_card_input_usd_per_token=float(price_in[m]), price_card_output_usd_per_token=float(price_out[m]),
            price_card_max_abs_error_usd=float(max(c["price_card_max_abs_error_usd"] for c in mine)),
            price_card_rows_beyond_rounding=int(sum(c["price_card_rows_beyond_rounding"] for c in mine)),
            price_card_difference_usd=float(sum(c["price_card_difference_usd"] for c in mine)),
            price_card_ratio_range=[min(ratios), max(ratios)] if ratios else None,
            price_card_beyond_by_provider=dict(beyond_by),
            price_card_beyond_benchmarks=[c["catalog_id"] for c in mine if c["price_card_rows_beyond_rounding"]],
            fitted_input_usd_per_token=float(coef[0]),
            fitted_output_usd_per_token=float(coef[1]) if len(coef) > 1 else None,
            fitted_max_abs_error_usd=float(fit_resid.max()),
            values=int(sum(c["values"] for c in mine)), zero_entries=int(sum(c["zero_entries"] for c in mine)),
            off_sum=int(sum(c["off_sum"] for c in mine)),
            off_sum_share=float(sum(c["off_sum"] for c in mine) / sum(c["fields"] for c in mine)),
            off_sum_1e6=int(sum(c["off_sum_1e6"] for c in mine)),
            sum_min=float(min(c["sum_min"] for c in mine)), sum_max=float(max(c["sum_max"] for c in mine)),
            reported_differs_first=int(sum(c["reported_differs_first"] for c in mine)),
            zero_mass_on_accepted=int(sum(c["zero_mass_on_accepted"] for c in mine)),
        )
    return per_bench, per_model, cells


# --------------------------------------------------------------------------------------------------
# stored collection reports, rights, related work
# --------------------------------------------------------------------------------------------------

def attempt_outcomes(models):
    t = pd.read_csv(ROOT / COLLECT / "transport_by_route.csv").set_index("route_id")
    out, total = {}, Counter()
    for m, route in enumerate(models.route_id):
        c = Counter(json.loads(t.loc[route, "realized_outcomes_json"]))
        out[m] = dict(c)
        total.update(c)
    return dict(per_model=out, total=dict(total), attempts=int(sum(total.values())),
                refusal_or_abstention_outcomes=sorted(k for k in total if re.search("refus|abstain|filter", k, re.I)))


def stored_accuracy(models, bench, stored):
    acc = stored.pivot(index="catalog_id", columns="model", values="field_accuracy")
    nat = stored.pivot(index="catalog_id", columns="model", values="native_index_raw")
    ll = stored.pivot(index="catalog_id", columns="model", values="field_log_loss")
    route = list(models.route_id)
    per_bench = {}
    for cid in bench.catalog_id:
        a, n = acc.loc[cid, route], nat.loc[cid, route]
        per_bench[int(cid)] = dict(
            native_metric=stored.loc[stored.catalog_id == cid, "native_metric"].iloc[0],
            mean_field_accuracy=float(a.mean()), min_field_accuracy=float(a.min()), max_field_accuracy=float(a.max()),
            best_model=int(np.argmax(a.to_numpy())), mean_native_score=float(n.mean()), max_native_score=float(n.max()),
            mean_log_loss=float(ll.loc[cid, route].mean()),
            saturated_by_field_accuracy=bool(a.mean() > SATURATION),
            saturated_by_native_score=bool(n.mean() > SATURATION))
    best = acc[route].to_numpy().argmax(1)
    per_model = {m: dict(mean_field_accuracy=float(acc[r].mean()), mean_log_loss=float(ll[r].mean()),
                         mean_native_score=float(nat[r].mean()), benchmarks_where_best=int((best == m).sum()))
                 for m, r in enumerate(route)}
    return per_bench, per_model


def rights(models, bench):
    r = ROOT / RIGHTS
    clear = pd.read_csv(r / "clearance.csv")
    mo = pd.read_csv(r / "model_output_rights.csv").set_index("route_id")
    bs = pd.read_csv(r / "benchmark_sources.csv").set_index("catalog_id")
    p = ROOT / PUBLIC
    shards = sorted(p.glob("catalog-*/model-*/predictions.parquet"))
    cleared = clear[clear.numeric_public_status == "cleared"]
    return dict(
        cells=int(len(clear)), cell_status=dict(Counter(clear.numeric_public_status)),
        cleared_models=sorted(cleared.route_id.unique()), cleared_benchmarks=int(cleared.catalog_id.nunique()),
        cleared_vectors=int(cleared.model_field_vectors.sum()),
        benchmarks_without_cleared_cell=sorted(int(c) for c in set(bench.catalog_id) - set(cleared.catalog_id)),
        per_cell_notices=sorted(clear.per_cell_notices.unique()), review_dates=sorted(clear.review_date.unique()),
        model_status={m: mo.loc[route, "model_vector_status"] for m, route in enumerate(models.route_id)},
        model_serving_providers={m: mo.loc[route, "serving_providers"] for m, route in enumerate(models.route_id)},
        model_status_counts=dict(Counter(mo.model_vector_status)),
        benchmark_source_category={int(c): bs.loc[c, "source_category"] for c in bench.catalog_id},
        benchmark_vector_status={int(c): bs.loc[c, "numeric_vector_status"] for c in bench.catalog_id},
        benchmark_source_category_counts=dict(Counter(bs.source_category)),
        public_stage=dict(
            prediction_shards=len(shards), vectors=int(sum(pq.read_metadata(s).num_rows for s in shards)),
            usage_files=len(list(p.glob("catalog-*/model-*/observations.parquet"))),
            model_folders=sorted({s.parent.name for s in shards}),
            benchmarks=len({s.parent.parent.name for s in shards})),
    )


def related_work():
    rw = ROOT / RELATED
    bib = (rw / "references.bib").read_text()
    entries = re.findall(r"@(\w+)\{([^,\s]+),(.*?)\n\}", bib, flags=re.S)
    key_of = {}
    for _, key, body in entries:
        doi = re.search(r"doi\s*=\s*\{([^}]+)\}", body, flags=re.I)
        if doi:
            key_of["doi:" + doi.group(1).lower()] = key
    marker = r"\[@([^\]]+)\]"
    cited = sorted(set(re.findall(marker, (rw / "report.md").read_text())))
    section = (rw / "related_work_section.md").read_text()
    scope = {}
    for line in (rw / "evidence" / "INDEX.md").read_text().splitlines():
        if line.startswith("| R"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            scope[c[1].strip("`")] = c[2]
    corpus = json.loads((rw / "corpus.json").read_text())["papers"]
    by_id = {p["id"]: p for p in corpus}
    core = {p["id"] for p in json.loads((rw / "corpus-core.json").read_text())["papers"]}

    with open(ROOT / METHODS, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ids = [r["corpus_id"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(cited):
        raise ValueError("method_compatibility.csv does not list exactly the papers cited in the review: "
                         f"missing {sorted(set(cited) - set(ids))}, extra {sorted(set(ids) - set(cited))}")
    for r in rows:
        r["read_scope"] = scope.get(r["corpus_id"], "no note")
        r["bib_key"] = key_of.get(r["corpus_id"], "")
    methods = [r for r in rows if r["class"] != "none"]
    native = [r for r in methods if r["class"] == "N"]
    topup_rows = []
    if (ROOT / METHODS_TOPUP).is_file():
        read = {}
        for note in sorted((ROOT / TOPUP_NOTES).glob("R*.md")):
            text = note.read_text(encoding="utf-8")
            cid = re.search(r"^- corpus_id: (\S+)", text, flags=re.M).group(1)
            read[cid] = re.search(r"^- read_scope: (\w+)", text, flags=re.M).group(1)
        with open(ROOT / METHODS_TOPUP, newline="", encoding="utf-8") as fh:
            topup_rows = list(csv.DictReader(fh))
        if {r["corpus_id"] for r in topup_rows} != set(read):
            raise ValueError("method_compatibility_topup.csv does not list exactly the papers of the top-up notes")
        for r in topup_rows:
            r["read_scope"], r["bib_key"] = read[r["corpus_id"]], ""
    topup_methods = [r for r in topup_rows if r["class"] != "none"]
    return dict(
        topup_rows=topup_rows, topup_papers=len(topup_rows), topup_method_papers=len(topup_methods),
        topup_class_counts=dict(Counter(r["class"] for r in topup_methods)),
        topup_read_scope=dict(Counter(r["read_scope"] for r in topup_rows)),
        bib_entries=len(entries), bib_types=dict(Counter(t.lower() for t, _, _ in entries)),
        bib_keys=[k for _, k, _ in entries],
        section_paragraphs=re.findall(r"^\*\*(.+?)\.\*\*", section, flags=re.M),
        section_words=len(re.sub(r"\[@[^\]]+\](\{[^}]*\})?", "", section).split()),
        section_citation_markers=len(re.findall(marker, section)),
        section_papers=len(set(re.findall(marker, section))),
        report_cited_papers=len(cited),
        corpus_records=len(corpus), corpus_decisions=dict(Counter(p["screening"]["decision"] for p in corpus)),
        papers_read=len(core), evidence_notes=len(list((rw / "evidence").glob("R*.md"))),
        method_map_rows=sum(1 for _ in open(rw / "method_map.csv", newline="", encoding="utf-8")) - 1,
        read_scope_of_cited=dict(Counter(r["read_scope"] for r in rows)),
        method_rows=rows,
        method_papers=len(methods), not_methods=len(rows) - len(methods),
        class_counts=dict(Counter(r["class"] for r in methods)),
        native_binary_only=sum(1 for r in native if r["restriction"] == "binary"),
        native_fixed_inventory_only=sum(1 for r in native if r["restriction"] == "fixed_inventory"),
        native_fixed_inventory=sum(1 for r in native if "fixed_inventory" in r["restriction"]),
        native_in_part=sum(1 for r in native if r["restriction"].startswith("partly")),
        native_everywhere=sum(1 for r in native if r["restriction"] == "all"),
        top_up=[dict(corpus_id=i, in_corpus=i in by_id, read=i in core or any(r["corpus_id"] == i for r in topup_rows),
                     title=by_id[i]["title"] if i in by_id else None,
                     year=by_id[i]["year"] if i in by_id else None,
                     decision=by_id[i]["screening"]["decision"] if i in by_id else None) for i in TOP_UP],
    )


def reference_rows():
    r = pd.read_csv(ROOT / ROWS).set_index("row")
    lat = pd.read_csv(ROOT / LATENCY).set_index("method")
    keep = ["log_loss", "accuracy", "native", "calls_per_request", "usd_per_1k_requests", "n_benchmarks"]
    return dict(
        rows={k: dict(label=label, **{c: float(r.loc[k, c]) for c in keep}) for k, label in REFERENCE_ROWS},
        all_rows={k: {c: float(v[c]) for c in keep} for k, v in r.iterrows()},
        all22_parallel_p50_s=float(lat.loc["all22_mean", "parallel_est_p50_s"]),
        all22_parallel_p95_s=float(lat.loc["all22_mean", "parallel_est_p95_s"]),
        all22_serial_p50_s=float(lat.loc["all22_mean", "serial_service_est_p50_s"]),
        all22_serial_p95_s=float(lat.loc["all22_mean", "serial_service_est_p95_s"]),
    )


# --------------------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------------------

def build():
    models, bench, manifest = corpus_index()
    cards = pd.read_csv(ROOT / CARDS)
    if cards.route_id.tolist() != models.route_id.tolist():
        raise ValueError("model cards and models.csv are in different orders")
    base = pd.read_csv(ROOT / COLLECT / "baselines.csv")
    stored = base[base.baseline == "route"]
    pooled = base[(base.baseline == "all_models") & (base.pooling == "mean")]
    index = json.loads((ROOT / INDEX).read_text())
    area = {int(c): a["id"] for a in index["areas"] for c in a["benchmarks"]}

    print("checksums ...", flush=True)
    sums = checksums(manifest)
    print("cells ...", flush=True)
    per_bench, per_model, cells = scan(models, bench, cards, stored)
    acc_bench, acc_model = stored_accuracy(models, bench, stored)
    out = attempt_outcomes(models)
    cell = pd.DataFrame(cells)
    count_cols = ["requests", "fields", "judged_fields", "single_gold_fields", "split_groups", "native_groups", "vectors",
                  "observations", "adapted_observations", "successful_physical_calls", "attempts",
                  "raw_choice_sums_adjusted_for_ensemble", "reported_choice_differs_first_argmax"]
    by_cid = cell.groupby("catalog_id")
    diff = (cell.accuracy_reported - cell.accuracy_stored).abs()
    diff_first = (cell.accuracy_first_max - cell.accuracy_stored).abs()

    benchmarks = {}
    for b in bench.itertuples():
        cid = int(b.catalog_id)
        benchmarks[cid] = dict(name=b.dataset, area=area[cid], **per_bench[cid], **acc_bench[cid],
                               vectors=int(b.vectors), observations=int(b.observations), attempts=int(b.attempts))
    head_keys = list(next(iter(per_bench.values()))["headroom"])
    head = {k: float(np.mean([per_bench[c]["headroom"][k] for c in per_bench]))
            for k in head_keys if k != "best_single_model"}
    names = {int(c): n for c, n in zip(bench.catalog_id, bench.dataset)}

    def ids(test):
        return [c for c in benchmarks if test(benchmarks[c])]

    summary = dict(
        inputs=dict(corpus=CORPUS, public_stage=PUBLIC, labels=LABELS, arrays=ARRAYS, rights=RIGHTS,
                    collection_reports=COLLECT, model_cards=CARDS, index=INDEX, related_work=RELATED,
                    kit_commit=manifest["kit_commit"], protocol=manifest["protocol"],
                    schema_version=manifest["schema_version"]),
        checksums=sums,
        storage=storage(),
        corpus=dict(
            models=len(models), benchmarks=len(bench), cells=len(cell),
            organisations=int(cards.provider.nunique()),
            models_with_confirmed_base_model=int(cards.base_model_confirmed.notna().sum()),
            first_listing=str(cards.listed_date.min()), last_listing=str(cards.listed_date.max()),
            index_benchmarks_not_in_corpus=sorted(int(c) for a in index["areas"] for c in a["benchmarks"]
                                                  if int(c) not in set(bench.catalog_id)),
            **{c: int(bench[c].sum()) for c in count_cols},
            manifest_counts_equal=all(int(bench[c].sum()) == manifest["counts"][c] for c in count_cols),
            recount=dict(
                requests=int(sum(v["requests"] for v in per_bench.values())),
                fields=int(sum(v["fields"] for v in per_bench.values())),
                judged_fields=int(sum(v["judged_fields"] for v in per_bench.values())),
                single_gold_fields=int(sum(v["single_gold_fields"] for v in per_bench.values())),
                split_groups=int(sum(v["split_groups"] for v in per_bench.values())),
                native_groups=int(sum(v["native_groups"] for v in per_bench.values())),
                vectors=int(cell.fields.sum()), observations=int(cell.requests.sum()),
                adapted_observations=int(cell.adapted_observations.sum()),
                successful_physical_calls=int(cell.physical_calls.sum()),
                attempts=int(cell.realized_physical_calls.sum()),
                raw_choice_sums_adjusted_for_ensemble=int(cell.choice_sum_not_one.sum()),
                reported_choice_differs_first_argmax=int(cell.reported_differs_first.sum())),
            recount_equals_benchmarks_csv=bool(
                all(int(by_cid.fields.sum()[c]) == int(v) for c, v in zip(bench.catalog_id, bench.vectors))
                and all(int(by_cid.requests.sum()[c]) == int(v) for c, v in zip(bench.catalog_id, bench.observations))
                and all(int(by_cid.choice_sum_not_one.sum()[c]) == int(v)
                        for c, v in zip(bench.catalog_id, bench.raw_choice_sums_adjusted_for_ensemble))
                and all(int(by_cid.reported_differs_first.sum()[c]) == int(v)
                        for c, v in zip(bench.catalog_id, bench.reported_choice_differs_first_argmax))
                and all(int(by_cid.physical_calls.sum()[c]) == int(v)
                        for c, v in zip(bench.catalog_id, bench.successful_physical_calls))
                and all(int(by_cid.realized_physical_calls.sum()[c]) == int(v) for c, v in zip(bench.catalog_id, bench.attempts))
                and all(int(by_cid.adapted_observations.sum()[c]) == int(v)
                        for c, v in zip(bench.catalog_id, bench.adapted_observations))
                and all(per_bench[int(b.catalog_id)][k] == int(getattr(b, k)) for b in bench.itertuples()
                        for k in ("requests", "fields", "judged_fields", "single_gold_fields", "split_groups",
                                  "native_groups"))),
            folds=sorted({f for v in per_bench.values() for f in v["folds"]}),
            split_groups_in_two_folds=int(sum(v["split_groups_in_two_folds"] for v in per_bench.values())),
            native_groups_in_two_folds=int(sum(v["native_groups_in_two_folds"] for v in per_bench.values())),
            smallest_fold_requests=int(min(min(v["fold_requests"]) for v in per_bench.values())),
            smallest_fold_split_groups=int(min(min(v["fold_split_groups"]) for v in per_bench.values())),
            options_min=int(min(v["options_min"] for v in per_bench.values())),
            options_max=int(max(v["options_max"] for v in per_bench.values())),
            yes_no_fields=int(sum(v["yes_no_fields"] for v in per_bench.values())),
            choice_fields=int(sum(v["choice_fields"] for v in per_bench.values())),
            benchmarks_with_yes_no_fields=ids(lambda v: v["yes_no_fields"] > 0),
            unjudged_fields=int(sum(v["fields"] - v["judged_fields"] for v in per_bench.values())),
            fixed_inventory_benchmarks=ids(lambda v: v["fixed_inventory"]),
            per_item_inventory_benchmarks=ids(lambda v: not v["fixed_inventory"]),
            binary_benchmarks=ids(lambda v: v["all_binary"]),
            single_class_judged_benchmarks=ids(lambda v: v["fixed_inventory"] and v["judged_gold_classes"] == 1),
            structure=dict(
                fixed_inventory=len(ids(lambda v: v["fixed_inventory"])),
                per_item_inventory=len(ids(lambda v: not v["fixed_inventory"])),
                binary=len(ids(lambda v: v["all_binary"])),
                with_yes_no_fields=len(ids(lambda v: v["yes_no_fields"] > 0)),
                single_class_judged=len(ids(lambda v: v["fixed_inventory"] and v["judged_gold_classes"] == 1)),
                areas=len(index["areas"])),
            areas={a["id"]: sorted(int(c) for c in a["benchmarks"] if int(c) in benchmarks) for a in index["areas"]},
            largest_share_of_judged_fields=dict(
                catalog_id=int(bench.catalog_id[bench.judged_fields.idxmax()]),
                share=float(bench.judged_fields.max() / bench.judged_fields.sum())),
        ),
        vectors=dict(
            values=int(cell["values"].sum()), nan=int(cell.nan.sum()), negative=int(cell.negative.sum()),
            above_one=int(cell.above_one.sum()), empty_vectors=int(cell.empty_vectors.sum()),
            zero_entries=int(cell.zero_entries.sum()), off_sum=int(cell.off_sum.sum()),
            off_sum_compensated=int(cell.off_sum_compensated.sum()), off_sum_1e6=int(cell.off_sum_1e6.sum()),
            off_sum_share=float(cell.off_sum.sum() / cell.fields.sum()), sum_tolerance=SUM_TOLERANCE,
            sum_min=float(cell.sum_min.min()), sum_max=float(cell.sum_max.max()),
            choice_sum_not_exactly_one=int(cell.choice_sum_not_one.sum()),
            reported_null=int(cell.reported_null.sum()),
            reported_null_equals_yes_no_rows=bool((cell.reported_null == cell.yes_no_fields).all()),
            reported_differs_first=int(cell.reported_differs_first.sum()),
            reported_differs_at_exact_tie=int(cell.reported_differs_at_exact_tie.sum()),
            reported_differs_below_maximum=int((cell.reported_differs_first - cell.reported_differs_at_exact_tie).sum()),
            models_with_zero_entries=int(sum(1 for v in per_model.values() if v["zero_entries"])),
            models_with_off_sum=int(sum(1 for v in per_model.values() if v["off_sum"])),
            zero_mass_on_accepted=int(cell.zero_mass_on_accepted.sum()),
        ),
        decision_rule=dict(
            reported_choice_max_abs_difference_from_stored=float(diff.max()),
            reported_choice_cells_within_1e12=int((diff <= 1e-12).sum()),
            first_maximum_max_abs_difference_from_stored=float(diff_first.max()),
            first_maximum_cells_differing=int((diff_first > 1e-12).sum()),
            first_maximum_largest_difference_cell=dict(
                catalog_id=int(cell.catalog_id[diff_first.idxmax()]), model_index=int(cell.model_index[diff_first.idxmax()])),
        ),
        telemetry=dict(
            cost_usd=float(cell.cost_usd.sum()),
            cost_min_model_usd=float(min(v["cost_usd"] for v in per_model.values())),
            cost_max_model_usd=float(max(v["cost_usd"] for v in per_model.values())),
            input_tokens=int(cell.input_tokens.sum()), output_tokens=int(cell.output_tokens.sum()),
            latency_p50_min_s=float(min(v["latency_p50_s"] for v in per_model.values())),
            latency_p50_max_s=float(max(v["latency_p50_s"] for v in per_model.values())),
            latency_p95_min_s=float(min(v["latency_p95_s"] for v in per_model.values())),
            latency_p95_max_s=float(max(v["latency_p95_s"] for v in per_model.values())),
            latency_max_min_s=float(min(v["latency_max_s"] for v in per_model.values())),
            latency_max_max_s=float(max(v["latency_max_s"] for v in per_model.values())),
            leaf_columns_with_nulls=sorted({c for v in per_model.values() for c in v["nulls"] if c.startswith("leaf_")}),
            columns_with_nulls=sorted({c for v in per_model.values() for c in v["nulls"]}),
            models_with_zero_output_tokens=int(sum(1 for v in per_model.values() if v["output_tokens"] == 0)),
            price_card_models_exact=int(sum(1 for v in per_model.values() if v["price_card_rows_beyond_rounding"] == 0)),
            price_card_rows_beyond_rounding=int(sum(v["price_card_rows_beyond_rounding"] for v in per_model.values())),
            price_card_max_abs_error_usd=float(max(v["price_card_max_abs_error_usd"] for v in per_model.values())),
            price_card_difference_usd=float(sum(v["price_card_difference_usd"] for v in per_model.values())),
            price_card_models_with_output_price=int(sum(1 for v in per_model.values() if v["price_card_output_usd_per_token"])),
            price_card_models_off=[m for m, v in per_model.items() if v["price_card_rows_beyond_rounding"]],
            fitted_price_max_abs_error_usd=float(max(v["fitted_max_abs_error_usd"] for v in per_model.values())),
            all_models_cost_per_1k_requests_benchmark_mean=float(
                np.mean([v["all_models_cost_per_1k_requests"] for v in per_bench.values()])),
        ),
        attempts=out,
        accuracy=dict(
            saturated_by_field_accuracy=ids(lambda v: v["saturated_by_field_accuracy"]),
            saturated_by_native_score=ids(lambda v: v["saturated_by_native_score"]),
            hardest_by_field_accuracy=sorted(benchmarks, key=lambda c: benchmarks[c]["mean_field_accuracy"])[:4],
            uniform_mean_pool_log_loss_benchmark_mean=float(pooled.field_log_loss.mean()),
            uniform_mean_pool_accuracy_benchmark_mean=float(pooled.field_accuracy.mean()),
            mean_single_model_accuracy=float(np.mean([v["mean_field_accuracy"] for v in acc_model.values()])),
            threshold=SATURATION,
        ),
        headroom=head,
        rights=rights(models, bench),
        related_work=related_work(),
        reference=reference_rows(),
        models={m: dict(
            canonical_model_id=models.canonical_model_id[m], route_id=models.route_id[m],
            organisation=cards.provider[m], listed_date=cards.listed_date[m],
            base_model=cards.base_model_confirmed[m], adaptation=cards.adaptation[m],
            served_parameters=cards.served_params[m], input_modalities=cards.input_modalities[m],
            context_length=int(cards.context_length[m]), **per_model[m], **acc_model[m],
            attempt_outcomes=out["per_model"][m]) for m in range(len(models))},
        benchmarks=benchmarks,
        benchmark_names=names,
    )
    return plain(summary)


# --------------------------------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------------------------------

def table(head, rows, align=None):
    align = align or ["l"] + ["r"] * (len(head) - 1)
    line = "| " + " | ".join(head) + " |\n|" + "|".join("---" if a == "l" else "---:" for a in align) + "|\n"
    return line + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def n(v):
    return f"{int(v):,}"


def span(lo, hi):
    return n(lo) if lo == hi else f"{n(lo)}–{n(hi)}"


def pct(v, d=1):
    return f"{100 * float(v):.{d}f}"


def tables(s):
    mod = {int(k): v for k, v in s["models"].items()}
    ben = {int(k): v for k, v in s["benchmarks"].items()}
    r = s["rights"]
    short = lambda m: mod[m]["route_id"]
    t = {}
    t["models"] = table(
        ["#", "Model (route)", "Served by", "Listed", "Base model (adaptation)", "Input price, $/M tokens", "Output rights"],
        [(m, f"`{v['route_id']}`", r["model_serving_providers"][str(m)].replace(" | ", ", "), v["listed_date"],
          f"{v['base_model']} ({v['adaptation']})" if v["base_model"] else "not disclosed",
          f"{1e6 * v['price_card_input_usd_per_token']:.3f}", r["model_status"][str(m)]) for m, v in mod.items()],
        ["r", "l", "l", "l", "l", "r", "l"])
    t["benchmarks"] = table(
        ["Id", "Benchmark", "Area", "Native metric", "Requests", "Fields", "Judged", "Options", "Inventory", "Split groups",
         "Requests per fold", "Source terms"],
        [(c, v["name"], v["area"], v["native_metric"], n(v["requests"]), n(v["fields"]), n(v["judged_fields"]),
          span(v["options_min"], v["options_max"]),
          "fixed" if v["fixed_inventory"] else "per item", n(v["split_groups"]),
          span(min(v["fold_requests"]), max(v["fold_requests"])), r["benchmark_source_category"][str(c)])
         for c, v in ben.items()],
        ["r", "l", "l", "l", "r", "r", "r", "r", "l", "r", "r", "l"])
    t["telemetry"] = table(
        ["Model", "Input tokens", "Output tokens", "Requests with no output token", "Cost, $", "$ per 1,000 requests",
         "Latency p50, s", "p95, s", "max, s", "Adapted requests", "Retried attempts", "Failed attempts"],
        [(f"`{short(m)}`", n(v["input_tokens"]), n(v["output_tokens"]), n(v["requests_with_zero_output_tokens"]),
          f"{v['cost_usd']:.2f}", f"{v['cost_per_1k_requests']:.3f}", f"{v['latency_p50_s']:.3f}",
          f"{v['latency_p95_s']:.3f}", f"{v['latency_max_s']:.1f}", n(v["adapted_observations"]),
          n(sum(x for k, x in v["attempt_outcomes"].items() if "retry" in k)),
          n(v["attempt_outcomes"].get("error", 0))) for m, v in mod.items()])
    t["prices"] = table(
        ["Model", "Listed input price, $ per million tokens", "Requests where cost ≠ tokens × price",
         "Largest gap on one request, $", "Sum of gaps, $", "Cost ÷ (tokens × price) on those requests"],
        [(f"`{short(m)}`", f"{1e6 * v['price_card_input_usd_per_token']:.3f}", n(v["price_card_rows_beyond_rounding"]),
          f"{v['price_card_max_abs_error_usd']:.6f}" if v["price_card_rows_beyond_rounding"] else "0",
          f"{v['price_card_difference_usd']:+.3f}" if v["price_card_rows_beyond_rounding"] else "0",
          "–" if not v["price_card_ratio_range"] else
          f"{v['price_card_ratio_range'][0]:.2f}–{v['price_card_ratio_range'][1]:.2f}") for m, v in mod.items()])
    realized = ("realized_cost_usd", "realized_input_tokens", "realized_output_tokens")
    for v in mod.values():
        if len({v["nulls"].get(c, 0) for c in realized}) != 1 or set(v["nulls"]) - set(realized) - {"provider"}:
            raise ValueError("missing values no longer follow the pattern that the table assumes")
    t["nulls"] = table(
        ["Model", "Requests without `realized_*` tokens and cost", "Requests without `provider`"],
        [(f"`{short(m)}`", n(v["nulls"].get("realized_cost_usd", 0)), n(v["nulls"].get("provider", 0)))
         for m, v in mod.items()])
    t["hygiene"] = table(
        ["Model", "Vectors off 1 by more than 1e-9", "Share, %", "Sum range", "Entries equal to 0",
         "Judged fields with 0 on every accepted option", "Reported choice ≠ first maximum"],
        [(f"`{short(m)}`", n(v["off_sum"]), pct(v["off_sum"] / v["fields"], 2),
          f"{v['sum_min']:.4f}–{v['sum_max']:.4f}", n(v["zero_entries"]), n(v["zero_mass_on_accepted"]),
          n(v["reported_differs_first"])) for m, v in mod.items()])
    t["accuracy_benchmarks"] = table(
        ["Id", "Benchmark", "Mean accuracy, %", "Lowest model", "Highest model", "Mean native score, %", "Saturated",
         "Best single in hindsight, %", "Any model right, %", "All right, %", "None right, %"],
        [(c, v["name"], pct(v["mean_field_accuracy"]), pct(v["min_field_accuracy"]), pct(v["max_field_accuracy"]),
          pct(v["mean_native_score"]),
          "accuracy and native" if v["saturated_by_native_score"] and v["saturated_by_field_accuracy"]
          else "accuracy" if v["saturated_by_field_accuracy"] else "native" if v["saturated_by_native_score"] else "",
          pct(v["headroom"]["best_single_hindsight"]), pct(v["headroom"]["any_correct"]),
          pct(v["headroom"]["all_correct"]), pct(v["headroom"]["none_correct"])) for c, v in ben.items()],
        ["r", "l", "r", "r", "r", "r", "l", "r", "r", "r", "r"])
    t["accuracy_models"] = table(
        ["Model", "Mean accuracy, %", "Mean log loss", "Mean native score, %", "Benchmarks where it is the most accurate"],
        [(f"`{short(m)}`", pct(v["mean_field_accuracy"]), f"{v['mean_log_loss']:.3f}", pct(v["mean_native_score"]),
          v["benchmarks_where_best"]) for m, v in sorted(mod.items(), key=lambda kv: -kv[1]["mean_field_accuracy"])])
    outcomes = sorted(s["attempts"]["total"], key=lambda k: -s["attempts"]["total"][k])
    t["attempts"] = table(
        ["Outcome", "Attempts", "Models with at least one"],
        [(f"`{k}`", n(s["attempts"]["total"][k]), sum(1 for v in mod.values() if v["attempt_outcomes"].get(k))) for k in outcomes])
    t["completeness"] = table(
        ["Id", "Benchmark", "Requests", "Fields", "Judged fields", "Vectors (22 models)", "Model–request rows", "Attempts"],
        [(c, v["name"], n(v["requests"]), n(v["fields"]), n(v["judged_fields"]), n(v["vectors"]), n(v["observations"]),
          n(v["attempts"])) for c, v in ben.items()], ["r", "l", "r", "r", "r", "r", "r", "r"])
    gb = lambda b: f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB"
    st = s["storage"]
    t["ledgers"] = table(
        ["Ledger file", "Size", "Write-ahead log beside it"],
        [(f"`{x['path']}`", gb(x["bytes"]), gb(x["wal_bytes"]) if x["wal_bytes"] else "–") for x in st["ledgers"]
         if x["bytes"] >= 1e8], ["l", "r", "r"])
    t["storage"] = table(
        ["Layer", "Path", "Size", "Content", "Read by this audit"],
        [("Request ledgers", f"`{LEDGERS}/**/*.sqlite3`", gb(st["ledger_bytes"]),
          "one row per attempt: status, HTTP status, error, response, latency, provider, tokens, billed amount", "no"),
         ("Frozen numeric corpus", f"`{CORPUS}/`", gb(st["corpus_bytes"]),
          "items, probability vectors, usage per model and request", "yes, every file"),
         ("Staged open subset", f"`{PUBLIC}/`", gb(st["public_stage_bytes"]), "items and vectors of the cleared cells", "file counts"),
         ("Benchmark rows", f"`{LABELS}/`", gb(st["labels_bytes"]), "item text, options, gold answers", "structure counts only"),
         ("Pinned benchmark kit", f"`{KIT}/`", gb(st["kit_bytes"]), "builders, scorers, index definition", "area list"),
         ("Derived arrays (round 1)", f"`{ARRAYS}/`", gb(st["arrays_bytes"]), "vectors and accepted-option masks per benchmark", "yes"),
         ("Collection reports", f"`{COLLECT}/`", "", "stored single-model and pool results, attempt outcomes", "yes"),
         ("Rights review", f"`{RIGHTS}/`", "", "status per model, benchmark and cell", "yes"),
         ("Related work", f"`{RELATED}/`", "", "section, bibliography, report, corpus, evidence notes", "yes")],
        ["l", "l", "r", "l", "l"])
    t["areas"] = table(
        ["Area", "Benchmarks", "Catalogue ids"],
        [(a, len(c), ", ".join(map(str, c))) for a, c in s["corpus"]["areas"].items()], ["l", "r", "l"])

    rw = s["related_work"]
    shown = {"N": "N", "T": "T", "A": "A", "X": "X"}
    suffix = {"binary": "N-bin", "fixed_inventory": "N-fix", "partly_fixed_inventory": "N / N-fix",
              "partly_binary": "N / N-bin", "three_or_more_options": "N (3+ options)"}
    titles = (("label_free", "Label-free aggregation of votes or outputs"),
              ("calibration", "Calibration of pools and opinion pools"),
              ("combiners", "Learned combiners and selection"),
              ("routing", "Routing, cascades and stopping"),
              ("model_similarity", "Model similarity from weights or samples"),
              ("not_a_method", "Not combination methods (theory, reviews, benchmarks, evaluation)"))
    parts = []
    for g, title in titles:
        rows = [x for x in rw["method_rows"] if x["group"] == g]
        body = []
        for x in rows:
            cls = "–" if x["class"] == "none" else suffix.get(x["restriction"], shown[x["class"]]) if x["class"] == "N" \
                else shown[x["class"]] + (" + " + x["also"] if x["also"] else "")
            body.append((x["paper"], f"`{x['bib_key']}`" if x["bib_key"] else "", x["method"], cls,
                         x["read_scope"].replace("_", " "), x["note"]))
        parts.append(f"**{title}** ({len(rows)})\n\n" + table(
            ["Paper", "Key in `references.bib`", "Method", "Class", "Read", "Note"], body, ["l"] * 6))
    t["methods"] = "\n\n".join(parts)
    body = []
    for x in rw["topup_rows"]:
        cls = "–" if x["class"] == "none" else suffix.get(x["restriction"], "N") if x["class"] == "N" else x["class"]
        body.append((x["paper"], x["method"], cls, x["read_scope"].replace("_", " "), x["note"]))
    t["methods_topup"] = table(["Paper", "Method", "Class", "Read", "Note"], body, ["l"] * 5)
    t["top_up"] = table(
        ["Corpus id", "Year", "Title", "Screening", "Read"],
        [(f"`{x['corpus_id']}`", x["year"] or "", x["title"] or "not in the corpus", x["decision"] or "",
          "yes" if x["read"] else "no") for x in rw["top_up"]], ["l"] * 5)
    t["bib_keys"] = ", ".join(f"`{k}`" for k in rw["bib_keys"])
    ref = s["reference"]["rows"]
    t["reference_rows"] = table(
        ["Stored row", "What it is", "Log loss", "Accuracy, %", "Calls per request", "$ per 1,000 requests"],
        [(f"`{k}`", v["label"], f"{v['log_loss']:.4f}", pct(v["accuracy"], 2), f"{v['calls_per_request']:.2f}",
          f"{v['usd_per_1k_requests']:.4f}") for k, v in ref.items()], ["l", "l", "r", "r", "r", "r"])
    return t


def lookup(s, path):
    v = s
    for part in path.split("."):
        v = v[int(part)] if isinstance(v, list) else v[part]
    return v


def render(s):
    template = (DOCS / "templates" / "DATA_AUDIT.template.md").read_text(encoding="utf-8")
    tabs = tables(s)
    names = {int(k): v for k, v in s["benchmark_names"].items()}

    def fill(m):
        path, _, spec = m.group(1).partition("|")
        if path.startswith("table:"):
            return tabs[path[6:]]
        v = lookup(s, path)
        if spec == "n":
            return n(v)
        if spec.startswith("pct"):
            return pct(v, int(spec[3:] or 1))
        if spec.startswith("f"):
            return f"{float(v):.{int(spec[1:])}f}"
        if spec.startswith("usd"):
            return f"{float(v):,.{int(spec[3:])}f}"
        if spec.startswith("signed"):
            return f"{float(v):+.{int(spec[6:])}f}".replace("-", "−")
        if spec == "ids":
            return ", ".join(str(x) for x in v)
        if spec == "names":
            return ", ".join(f"{names[int(x)]} ({x})" for x in v)
        if spec == "list":
            return ", ".join(f"`{x}`" for x in v)
        if spec == "count":
            return str(len(v))
        if spec in ("", "raw"):
            return str(v)
        raise ValueError(f"unknown format {spec!r}")

    text = re.sub(r"\{\{([^{}]+)\}\}", fill, template)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="recompute and compare with the stored summary; write nothing")
    ap.add_argument("--render-only", action="store_true", help="rebuild DATA_AUDIT.md from the stored summary")
    args = ap.parse_args()
    target = DOCS / "AUDIT_SUMMARY.json"
    if args.render_only:
        summary = json.loads(target.read_text(encoding="utf-8"))
    else:
        summary = build()
        summary["environment"] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                      pyarrow=pa.__version__)
    if args.check:
        old = json.loads(target.read_text(encoding="utf-8"))
        keys = sorted(k for k in set(old) | set(summary) if k != "environment" and old.get(k) != summary.get(k))
        print("identical to the stored summary" if not keys else f"differs from the stored summary in: {keys}")
        return 1 if keys else 0
    text = json.dumps(summary, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    if not args.render_only:
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target.relative_to(ROOT)} ({len(text):,} bytes)")
    if (DOCS / "templates" / "DATA_AUDIT.template.md").is_file():
        doc = render(summary)
        (DOCS / "DATA_AUDIT.md").write_text(doc, encoding="utf-8")
        print(f"wrote {(DOCS / 'DATA_AUDIT.md').relative_to(ROOT)} ({len(doc.split()):,} words)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

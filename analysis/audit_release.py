"""Audit of the released corpus: the counts that the paper prints for the released models and benchmarks.

    python3 -B -m analysis.audit_release --corpus DIR --audit FILE --out FILE [--local DIR]
    python3 -B -m analysis.audit_release --corpus DIR --audit FILE --check-against-audit

``DIR`` is a corpus with labels, usage and latency: the open tier of schema version 2 with rebuilt labels, or the
local tier.  ``--audit`` is the stored audit of the whole collection (``docs/AUDIT_SUMMARY.json``).  It supplies
what does not depend on the pool: the model cards, the structure of every benchmark and the review of terms.
Everything that depends on the pool is recomputed here from the corpus tables, by the rules of ``analysis/audit.py``
(sums by plain left-to-right addition, the same tolerances).  ``--local`` names the local tier of the whole
collection; it adds what the open tier does not carry: the number of attempts, and how many of the requests that
a model's usual host served were billed off the listed price.

The output has the layout of the stored audit for the keys that the paper and its tables read.  Models are
numbered as in the corpus.  ``--check-against-audit`` runs the same code on the local tier of the whole collection
and compares its totals with the stored audit; it writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data import schema as S
from data.load import list_benchmarks, list_models, read_manifest

SUM_TOLERANCE = 1e-9            # as in analysis/audit.py


def running_sum(x):
    """Row sums by plain left-to-right float addition; ``x`` is [n, w] (as in analysis/audit.py)."""
    s = x[:, 0].copy()
    for j in range(1, x.shape[1]):
        s = s + x[:, j]
    return s


def _cell(corpus, cid, m, judged, accepted_lists, usage_schema, prices):
    """Counts of one benchmark and model: the vectors, the usage and the latency."""
    t = pq.read_table(S.table_path(corpus, cid, "predictions", m))
    col = t["probabilities"].combine_chunks()
    off = col.offsets.to_numpy().astype(np.int64)
    val = col.values.to_numpy(zero_copy_only=False)
    starts, width = off[:-1], np.diff(off)
    F = len(width)
    is_noul = pc.equal(t["question_type"], "noul").combine_chunks().to_numpy(zero_copy_only=False)
    rc = pc.fill_null(t["reported_choice"].combine_chunks(), -1).to_numpy(zero_copy_only=False).astype(np.int64)
    accepted = np.zeros(off[-1], bool)
    for f, positions in enumerate(accepted_lists):
        accepted[starts[f] + np.asarray(positions, np.int64)] = True
    sums, first = np.empty(F), np.empty(F, np.int64)
    for w in np.unique(width):
        idx = np.flatnonzero(width == w)
        x = val[starts[idx][:, None] + np.arange(w)]
        sums[idx] = running_sum(x)
        first[idx] = x.argmax(1)
    if is_noul.any():       # yes/no: "yes" (position 1) when its normalised probability is at least one half
        p0, p1 = val[starts[is_noul]], val[starts[is_noul] + 1]
        first[is_noul] = (p1 / (p0 + p1) >= 0.5).astype(np.int64)
    mass = np.add.reduceat(val * accepted, starts)
    differs = (rc >= 0) & (rc != first)

    u = S.read_table(S.table_path(corpus, cid, "usage", m), usage_schema)
    tin = u["input_tokens"].to_numpy().astype(float)
    tout = u["output_tokens"].to_numpy().astype(float)
    cost = u["cost_usd"].to_numpy().astype(float)
    calls = u["calls"].to_numpy().astype(float)
    beyond = np.abs(cost - (tin * prices[0] + tout * prices[1])) > 5e-10 * np.maximum(calls, 1) + 1e-12
    lat = S.read_table(S.table_path(corpus, cid, "latency", m), S.LATENCY)["latency_s"].to_numpy()
    cell = dict(
        fields=F, values=int(len(val)), zero_entries=int((val == 0).sum()),
        off_sum=int((np.abs(sums - 1) > SUM_TOLERANCE).sum()),
        reported_differs_first=int(differs.sum()),
        reported_differs_at_exact_tie=int((differs & (val[starts + np.maximum(rc, 0)] == val[starts + first])).sum()),
        zero_mass_on_accepted=int(((mass == 0) & judged).sum()),
        requests=int(len(cost)), input_tokens=int(tin.sum()), output_tokens=int(tout.sum()), cost_usd=float(cost.sum()),
        physical_calls=int(calls.sum()), adapted_observations=int((calls > 1).sum()),
        price_card_rows_beyond_rounding=int(beyond.sum()),
        attempts=int(u["attempts"].to_numpy().sum()) if "attempts" in usage_schema.names else None)
    return cell, lat, cost


def collect(corpus, audit):
    """The audit of ``corpus``; ``audit`` is the stored audit of the whole collection."""
    corpus = Path(corpus)
    manifest = read_manifest(corpus)
    usage_schema = S.usage_schema(manifest["schema_version"])
    models = list_models(corpus)
    originals = [str(int(m["model_code"][1:])) for m in models]          # index in the whole collection
    info = [audit["models"][o] for o in originals]
    for row, card in zip(models, info):
        if manifest["names"] == "real" and row["model_name"] != card["route_id"]:
            raise ValueError("the corpus and the stored audit name a model differently")
    benchmarks = list_benchmarks(corpus)
    removed = manifest.get(S.EXCLUDED_KEY, {})            # requests removed for every model, per benchmark
    M = len(models)
    per_model = [dict(lat=[], cost=[], cells=[]) for _ in range(M)]
    per_bench, cells = {}, []
    for b in benchmarks:
        cid = int(b["catalog_id"])
        labels = pq.read_table(S.table_path(corpus, cid, "labels"))
        judged = labels["judged"].to_numpy(zero_copy_only=False)
        accepted = labels["accepted"].to_pylist()
        stored = audit["benchmarks"][str(cid)]
        for m in range(M):
            prices = (info[m]["price_card_input_usd_per_token"], info[m]["price_card_output_usd_per_token"])
            cell, lat, cost = _cell(corpus, cid, m, judged, accepted, usage_schema, prices)
            cells.append(dict(cell, catalog_id=cid, model_index=m))
            per_model[m]["lat"].append(lat)
            per_model[m]["cost"].append(cost)
            per_model[m]["cells"].append(cell)
        options = pq.read_table(S.table_path(corpus, cid, "fields"), columns=["n_options"])["n_options"].to_numpy()
        per_bench[cid] = dict(name=stored["name"], area=stored["area"], native_metric=stored["native_metric"],
                              requests=int(b["requests"]), fields=int(b["fields"]), judged_fields=int(judged.sum()),
                              options_min=int(options.min()), options_max=int(options.max()),
                              fixed_inventory=stored["fixed_inventory"])
        gone = removed.get(str(cid), 0)
        if gone:
            per_bench[cid]["requests_removed"] = gone
        same = (per_bench[cid]["requests"] + gone == stored["requests"]
                and (per_bench[cid]["fields"] == stored["fields"] if not gone else
                     per_bench[cid]["fields"] < stored["fields"])
                and stored["options_min"] <= per_bench[cid]["options_min"]
                and per_bench[cid]["options_max"] <= stored["options_max"]
                and (gone or (per_bench[cid]["options_min"], per_bench[cid]["options_max"])
                     == (stored["options_min"], stored["options_max"])))
        if not same:
            raise ValueError(f"catalogue {cid}: the corpus and the stored audit differ in requests, fields or options")

    def total(key, rows=cells):
        return int(sum(c[key] for c in rows))

    model_rows = {}
    for m in range(M):
        mine, lat = per_model[m]["cells"], np.concatenate(per_model[m]["lat"])
        cost = np.concatenate(per_model[m]["cost"])
        model_rows[m] = dict(
            route_id=info[m]["route_id"], organisation=info[m]["organisation"], listed_date=info[m]["listed_date"],
            base_model=info[m]["base_model"], model_index_in_collection=int(originals[m]),
            price_card_input_usd_per_token=info[m]["price_card_input_usd_per_token"],
            price_card_output_usd_per_token=info[m]["price_card_output_usd_per_token"],
            requests=total("requests", mine), fields=total("fields", mine),
            input_tokens=total("input_tokens", mine), output_tokens=total("output_tokens", mine),
            cost_usd=float(cost.sum()), cost_per_1k_requests=float(1000 * cost.mean()),
            physical_calls=total("physical_calls", mine),
            latency_p50_s=float(np.percentile(lat, 50)), latency_p95_s=float(np.percentile(lat, 95)),
            zero_entries=total("zero_entries", mine), off_sum=total("off_sum", mine),
            price_card_rows_beyond_rounding=total("price_card_rows_beyond_rounding", mine))
    judged_fields = {cid: v["judged_fields"] for cid, v in per_bench.items()}
    largest = max(judged_fields, key=judged_fields.get)
    areas = sorted({v["area"] for v in per_bench.values()})
    status = {m: audit["rights"]["model_status"][originals[m]] for m in range(M)}
    category = {cid: audit["rights"]["benchmark_source_category"][str(cid)] for cid in per_bench}
    fields = sum(v["fields"] for v in per_bench.values())
    attempts = None if cells[0]["attempts"] is None else total("attempts")
    return dict(
        inputs=dict(corpus_checksums_sha256=manifest["checksums_sha256"], schema_version=manifest["schema_version"],
                    tier=manifest["tier"]),
        corpus=dict(
            models=M, benchmarks=len(per_bench), cells=len(cells),
            organisations=len({r["organisation"] for r in model_rows.values()}),
            models_with_confirmed_base_model=sum(1 for r in model_rows.values() if r["base_model"]),
            first_listing=min(r["listed_date"] for r in model_rows.values()),
            last_listing=max(r["listed_date"] for r in model_rows.values()),
            requests=sum(v["requests"] for v in per_bench.values()), fields=fields,
            judged_fields=sum(judged_fields.values()), vectors=total("fields"), observations=total("requests"),
            successful_physical_calls=total("physical_calls"), adapted_observations=total("adapted_observations"),
            attempts=attempts,
            options_min=min(v["options_min"] for v in per_bench.values()),
            options_max=max(v["options_max"] for v in per_bench.values()),
            structure=dict(fixed_inventory=sum(1 for v in per_bench.values() if v["fixed_inventory"]),
                           per_item_inventory=sum(1 for v in per_bench.values() if not v["fixed_inventory"]),
                           areas=len(areas)),
            areas={a: sorted(cid for cid, v in per_bench.items() if v["area"] == a) for a in areas},
            largest_share_of_judged_fields=dict(catalog_id=largest,
                                                share=judged_fields[largest] / sum(judged_fields.values()))),
        vectors=dict(
            values=total("values"), zero_entries=total("zero_entries"), off_sum=total("off_sum"),
            off_sum_share=total("off_sum") / total("fields"), sum_tolerance=SUM_TOLERANCE,
            reported_differs_first=total("reported_differs_first"),
            reported_differs_at_exact_tie=total("reported_differs_at_exact_tie"),
            models_with_zero_entries=sum(1 for r in model_rows.values() if r["zero_entries"]),
            models_with_off_sum=sum(1 for r in model_rows.values() if r["off_sum"]),
            zero_mass_on_accepted=total("zero_mass_on_accepted")),
        telemetry=dict(
            cost_usd=float(sum(c["cost_usd"] for c in cells)),
            cost_min_model_usd=min(r["cost_usd"] for r in model_rows.values()),
            cost_max_model_usd=max(r["cost_usd"] for r in model_rows.values()),
            input_tokens=total("input_tokens"), output_tokens=total("output_tokens"),
            latency_p50_min_s=min(r["latency_p50_s"] for r in model_rows.values()),
            latency_p50_max_s=max(r["latency_p50_s"] for r in model_rows.values()),
            models_with_zero_output_tokens=sum(1 for r in model_rows.values() if r["output_tokens"] == 0),
            price_card_models_exact=sum(1 for r in model_rows.values() if r["price_card_rows_beyond_rounding"] == 0),
            price_card_rows_beyond_rounding=total("price_card_rows_beyond_rounding"),
            price_card_models_with_output_price=sum(1 for r in model_rows.values()
                                                    if r["price_card_output_usd_per_token"]),
            price_card_models_off=[r["route_id"] for r in model_rows.values() if r["price_card_rows_beyond_rounding"]]),
        rights=dict(model_status=status, model_status_counts=dict(Counter(status.values())),
                    benchmark_source_category=category,
                    benchmark_source_category_counts=dict(Counter(category.values())),
                    review_dates=audit["rights"]["review_dates"]),
        models=model_rows, benchmarks=per_bench)


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def check_against_audit(summary, audit):
    """Differences between the audit computed here on the whole collection and the stored one (empty: none)."""
    pairs = [("corpus", k) for k in ("models", "benchmarks", "cells", "organisations", "requests", "fields",
                                     "judged_fields", "vectors", "observations", "successful_physical_calls",
                                     "adapted_observations", "attempts", "options_min", "options_max")]
    pairs += [("vectors", k) for k in ("values", "zero_entries", "off_sum", "reported_differs_first",
                                       "reported_differs_at_exact_tie", "models_with_zero_entries",
                                       "zero_mass_on_accepted")]
    pairs += [("telemetry", k) for k in ("input_tokens", "output_tokens", "price_card_models_exact",
                                         "price_card_rows_beyond_rounding")]
    differences = [f"{a}.{k}: {summary[a][k]} here, {audit[a][k]} stored" for a, k in pairs if summary[a][k] != audit[a][k]]
    if abs(summary["telemetry"]["cost_usd"] - audit["telemetry"]["cost_usd"]) > 1e-9:
        differences.append("telemetry.cost_usd")
    for m, row in summary["models"].items():
        stored = audit["models"][str(m)]
        if abs(row["latency_p50_s"] - stored["latency_p50_s"]) > 1e-12 or row["off_sum"] != stored["off_sum"]:
            differences.append(f"model {m}: median latency or vector sums")
    return differences


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit of the released corpus, in the layout of the stored audit")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--local", dest="local_tier",
                    help="local tier of the whole collection: adds the attempts and the price-card check by host")
    ap.add_argument("--out")
    ap.add_argument("--check-against-audit", action="store_true")
    a = ap.parse_args(argv)
    audit = json.loads(Path(a.audit).read_text(encoding="utf-8"))
    summary = _plain(collect(a.corpus, audit))
    if a.check_against_audit:
        differences = check_against_audit(summary, audit)
        print("\n".join(differences) if differences else "the audit computed here equals the stored audit")
        return 1 if differences else 0
    if a.local_tier:
        local = Path(a.local_tier)
        codes = {m["model_code"]: int(m["model_index"]) for m in list_models(local)}
        attempts = off_card_on_main_host = 0
        kept = {}
        for cid in read_manifest(a.corpus)["catalog_ids"]:          # requests are matched by their opaque id
            mine = S.read_table(S.table_path(a.corpus, cid, "items"), S.ITEMS)["item_id"].to_pylist()
            where = {request: i for i, request in enumerate(
                S.read_table(S.table_path(local, cid, "items"), S.ITEMS)["item_id"].to_pylist())}
            kept[cid] = np.asarray([where[request] for request in mine], dtype=np.int64)
        for position, m in enumerate(list_models(a.corpus)):
            card = summary["models"][str(position)]
            prices = (card["price_card_input_usd_per_token"], card["price_card_output_usd_per_token"])
            host, beyond = [], []
            for cid in read_manifest(a.corpus)["catalog_ids"]:
                usage = S.read_table(S.table_path(local, cid, "usage", codes[m["model_code"]]), S.USAGE)
                keep = kept[cid]                       # the requests of the audited corpus, in the local tier
                attempts += int(usage["attempts"].to_numpy()[keep].sum())
                tin, tout = (usage[k].to_numpy().astype(float)[keep] for k in ("input_tokens", "output_tokens"))
                cost, calls = (usage[k].to_numpy().astype(float)[keep] for k in ("cost_usd", "calls"))
                if not np.array_equal(cost, S.read_table(S.table_path(a.corpus, cid, "usage", position),
                                                         S.usage_schema(read_manifest(a.corpus)["schema_version"])
                                                         )["cost_usd"].to_numpy().astype(float)):
                    raise ValueError(f"catalogue {cid}: the local tier and the corpus differ in a request's cost")
                beyond.append(np.abs(cost - (tin * prices[0] + tout * prices[1])) > 5e-10 * np.maximum(calls, 1) + 1e-12)
                providers = usage["provider"].to_pylist()
                host.extend(str(providers[i]) for i in keep)                    # a missing host counts as one host
            host, beyond = np.asarray(host), np.concatenate(beyond)
            names, counts = np.unique(host, return_counts=True)
            main = host == names[counts.argmax()]
            off_card_on_main_host += int((beyond & main).sum())
        summary["corpus"]["attempts"] = attempts
        # Counts only: the serving host itself is not part of the release or of this file.
        summary["telemetry"]["price_card_rows_beyond_rounding_on_main_host"] = off_card_on_main_host
        summary["corpus"]["models_answering_in_smaller_batches"] = sum(
            1 for row in summary["models"].values() if row["physical_calls"] > row["requests"])
    if not a.out:
        ap.error("--out is required unless --check-against-audit is given")
    Path(a.out).write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    c, t = summary["corpus"], summary["telemetry"]
    print(f"{c['models']} models, {c['benchmarks']} benchmarks, {c['requests']:,} requests, {c['vectors']:,} vectors, "
          f"{t['cost_usd']:.2f} USD")
    return 0


if __name__ == "__main__":
    sys.exit(main())

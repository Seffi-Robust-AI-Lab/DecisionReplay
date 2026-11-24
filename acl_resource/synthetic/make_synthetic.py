"""Made-up corpus in the layout of the real inputs, for tests and for development without the real data.

    python3 -B -m acl_resource.synthetic.make_synthetic --out DIR [--seed 7]

Everything is generated from a seed: no real request, option, answer or model output is involved.  The result has
the folder layout and file schemas of the real inputs, the 36 catalogue ids and 22 model slots, and every edge
case found in the audit of the real corpus.  Written under DIR:

    results/decision_index/paper/dataset/candidate/     source export (items, predictions, observations)
    data/decision_index/common-paid-65/inputs/          benchmark rows NNN.jsonl.gz, with made-up text
    ensemble_round2/model_cards/cards.csv               made-up model cards
    meta/                                               benchmarks_meta.csv, models_meta.csv, clearance.csv
    processed/local/                                    the processed corpus (reference build)
    PLANTED.json                                        the structure planted on purpose, for tests

Planted structure (PLANTED.json): model 9 returns exactly the vectors of model 8; models of one organisation
share part of their noise; model 17 is the cheapest model and is strong only in the area "arts".
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..data import reference_build
from ..data import schema as S

M = 22
AREAS = {
    "knowledge": (25, 28, 30, 31, 32, 33, 43, 44, 57, 58),
    "language": (11, 12, 29, 38, 39, 40, 41, 42, 59),
    "retrieval": (4, 5, 36, 37, 56, 61),
    "tools": (1, 2, 3, 9, 62),
    "arts": (20, 21, 22, 23, 50, 64),
}
ORGS = ["orga"] * 3 + ["orgb"] * 2 + ["orgc"] + ["orgd"] * 5 + ["orge", "orgf"] + ["orgg"] * 4 + ["orgh"] * 2 \
    + ["orgi"] + ["orgj"] * 2
BASE = {0: "base/alpha-27b", 1: "base/alpha-9b", 2: "base/omni-30b", 5: "base/alpha-4b", 15: "base/beta-9b",
        17: "base/alpha-4b", 18: "base/alpha-27b"}
PRICE = [1.9e-7, 1.9e-8, 1.5e-7, 1.4e-7, 9e-8, 4.2e-8, 2e-8, 4.2e-8, 2e-8, 2e-8, 4.2e-8, 4.2e-8, 4.2e-8, 6e-8,
         6e-8, 4e-8, 6e-8, 5e-9, 2e-8, 1e-8, 4.2e-8, 4.2e-8]
DUPLICATE = (8, 9)                    # model 9 copies model 8
SPECIALIST, SPECIALIST_AREA = 17, "arts"
ROUNDED = (0, 1, 5, 15, 16)           # vectors rounded to 4 decimals: sums off by up to about 1e-3
SINGLE_PRECISION = (19,)              # vectors passed through float32: sums off by about 1e-8
SPARSE = (3, 4, 20, 21)               # small probabilities set to exactly 0
TIED = (3, 4)                         # sometimes two tied maxima, the later one reported
NO_OUTPUT_TOKENS = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 19)
ONE_OUTPUT_TOKEN_PER_FIELD = (11, 17, 18)
ADAPTED = (13, 16)                    # answered in field batches on requests with many fields
SECOND_PROVIDER = 1                   # a few requests at twice the price through another provider
DISCOUNTED = 12                       # a few requests below tokens x price
CLEARED, HELD_BENCHMARK = (2, 11), 59  # made-up release review: two models cleared, one benchmark held

# Structure per catalogue: fields per request (lo, hi), option counts (lo, hi), fixed option inventory, share of
# judged fields, and a special rule.
PROFILE = {
    1: dict(fields=(2, 4), options=(2, 2), fixed=True),
    2: dict(fields=(8, 12), options=(2, 2), fixed=True, judged=0.08, special="all_positive"),
    3: dict(options=(12, 12), fixed=True),
    4: dict(options=(20, 20), fixed=True),
    5: dict(options=(30, 30), fixed=True),
    9: dict(fields=(12, 18), options=(2, 9)),
    11: dict(fields=(5, 5), options=(3, 3), fixed=True),
    12: dict(options=(3, 3), fixed=True),
    20: dict(options=(2, 2)), 21: dict(options=(2, 2)),
    22: dict(options=(25, 25), fixed=True),
    23: dict(options=(2, 2)),
    25: dict(options=(4, 4)), 28: dict(options=(2, 2)), 29: dict(options=(4, 4)),
    30: dict(options=(4, 10)),
    31: dict(options=(2, 12), special="accepted_list"),
    32: dict(options=(2, 5)),
    33: dict(fields=(6, 12), options=(2, 2), fixed=True),
    36: dict(fields=(8, 12), options=(2, 2), fixed=True, judged=0.08, special="all_positive"),
    37: dict(options=(4, 4), fixed=True),
    38: dict(fields=(6, 14), options=(2, 2), fixed=True),
    39: dict(fields=(1, 3), options=(3, 3), fixed=True),
    40: dict(fields=(2, 3), options=(2, 2)),
    41: dict(options=(3, 3), fixed=True),
    42: dict(options=(2, 2), fixed=True, big_groups=True),
    43: dict(options=(2, 10)),
    44: dict(options=(2, 2), fixed=True),
    50: dict(options=(2, 5), special="accepted_list"),
    56: dict(fields=(9, 9), options=(2, 2), special="one_judged"),
    57: dict(options=(3, 10)), 58: dict(options=(2, 8)),
    59: dict(options=(2, 2), special="yes_no"),
    61: dict(options=(2, 2), fixed=True),
    62: dict(options=(4, 4)), 64: dict(options=(5, 5)),
}
OBSERVATIONS = pa.schema([
    ("item_index", pa.int32()), ("provider", pa.string()), ("transport", pa.string()),
    ("adapted_field_partitions", pa.list_(pa.list_(pa.int32()))),
    ("leaf_physical_calls", pa.int32()), ("realized_physical_calls", pa.int32()),
    ("leaf_input_tokens", pa.int64()), ("leaf_output_tokens", pa.int64()), ("leaf_cost_usd", pa.float64()),
    ("leaf_deployment_latency_s", pa.float64()), ("realized_input_tokens", pa.int64()),
    ("realized_output_tokens", pa.int64()), ("realized_known_input_tokens", pa.int64()),
    ("realized_known_output_tokens", pa.int64()), ("realized_cost_usd", pa.float64()),
    ("realized_known_cost_usd", pa.float64()), ("realized_missing_cost_usd", pa.int32()),
    ("realized_unresolved_reserved_nusd", pa.int64()), ("realized_service_sum_s", pa.float64()),
])


def _hash(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def _write(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _rows(cid, rng):
    """Benchmark rows with made-up text, plus per-request fold and group ids."""
    prof = PROFILE[cid]
    R = int(rng.integers(90, 150))
    if prof.get("big_groups"):
        n_split = 30
    else:
        n_split = max(40, int(R / rng.uniform(1.0, 1.8)))
    split = np.sort(rng.integers(0, n_split, R))
    split[:n_split] = np.arange(n_split)          # every group appears
    split = np.sort(split)
    group_fold = rng.permutation(np.arange(n_split) % S.N_FOLDS)
    native_sub = rng.integers(0, 2, R)             # up to two scoring groups inside a split group
    order = rng.permutation(R)
    lo, hi = prof.get("fields", (1, 1))
    olo, ohi = prof["options"]
    fixed = [f"label_{k:02d}" for k in range(ohi)] if prof.get("fixed") else None
    special = prof.get("special")
    rows, truth = [], []
    for i in range(R):
        n_fields = int(rng.integers(lo, hi + 1))
        questions, expected, row_truth = {}, {}, []
        scoring = None
        for j in range(n_fields):
            key = f"field_{j}"
            if special == "one_judged":
                key = "verdict" if j == 0 else f"aux_{j}"
            noul = special == "yes_no" or (special == "one_judged" and j % 3 != 2)
            if noul:
                answer = bool(rng.random() < 0.45)
                questions[key] = dict(type="noul", instructions=f"made-up yes/no question {j}",
                                      criteria=dict(yes="made-up yes text", no="made-up no text"))
                gold_pos, gold_value, n_opt = int(answer), answer, 2
            else:
                n_opt = int(rng.integers(olo, ohi + 1))
                keys = fixed if fixed else [f"option_{k}" for k in range(n_opt)]
                texts = ([f"made-up label text {k}" for k in range(n_opt)] if fixed else
                         [f"made-up option text {cid}-{i}-{j}-{k}" for k in range(n_opt)])
                questions[key] = dict(type="choice", instructions=f"made-up question {j}",
                                      criteria=dict(zip(keys, texts)))
                gold_pos = 1 if special == "all_positive" else int(rng.integers(0, n_opt))
                gold_value = keys[gold_pos]
            judged = rng.random() < prof.get("judged", 1.0)
            if special == "one_judged" and j > 0:
                gold_value = [None, gold_value][int(rng.integers(0, 2))]
            elif not judged:
                gold_value = None
            expected[key] = gold_value
            row_truth.append((gold_pos, n_opt, noul))
            if special == "accepted_list":
                extra = [k for k in range(n_opt) if k != gold_pos and rng.random() < 0.08]
                accepted = sorted([gold_pos] + extra[:2])
                scoring = dict(type="chess_reference_value" if cid == 31 else "human_group_rank",
                               accepted=[f"option_{k}" for k in accepted], scorer="made-up")
        if special == "one_judged":
            scoring = dict(type="made_up_primary", primary_field="verdict",
                           unscored_fields=[k for k in questions if k != "verdict"][:5])
        run_id = f"{_hash('run', cid, int(order[i]))[:12]}-{cid:03d}"
        row = dict(id=f"syn-{cid:03d}-{i:05d}", family="synthetic", split="test",
                   state=dict(text=f"made-up state {cid}-{i} " + "lorem " * int(rng.integers(5, 60))),
                   questions=questions, expected=expected, metadata=dict(synthetic=True))
        if scoring:
            row["scoring"] = scoring
        row["_evaluation"] = dict(run_id=run_id, catalog_id=cid, dataset=f"synthetic-{cid:03d}",
                                  group_id=f"ng-{int(split[i])}-{int(native_sub[i])}", track="main",
                                  payload_sha256=_hash("payload", cid, i), proxy_tokens=int(rng.integers(80, 4000)),
                                  benchmark_origin="synthetic")
        rows.append(row)
        truth.append(row_truth)
    return rows, truth, group_fold[split].astype(np.int8), split


def _vectors(cid, truth, rng, skill, org_code):
    """Probability vectors [field][model] -> list of float lists, and the reported choice."""
    out_p, out_rc = [[] for _ in range(M)], [[] for _ in range(M)]
    n_org = int(org_code.max()) + 1
    for row_truth in truth:
        hard = rng.normal(0.0, 0.8)                                 # difficulty of the request
        for gold, n_opt, noul in row_truth:
            shared = rng.normal(0.0, 1.0, n_opt)
            by_org = rng.normal(0.0, 1.0, (n_org, n_opt))
            mine = rng.normal(0.0, 1.0, (M, n_opt))
            logits = 0.7 * shared[None, :] + 0.8 * by_org[org_code] + 0.7 * mine
            logits[np.arange(M), gold] += skill - hard
            logits[DUPLICATE[1]] = logits[DUPLICATE[0]]
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            for m in range(M):
                v = p[m].copy()
                if m in SPARSE:
                    v[v < 0.02] = 0.0
                    if rng.random() < 0.01:
                        v[gold] = 0.0
                    if not v.any():
                        v[int(rng.integers(0, n_opt))] = 1.0
                    v /= v.sum()
                if noul:
                    yes = float(v[1])
                    v = np.array([1.0 - yes, yes])
                if m in ROUNDED:
                    v = np.round(v, 4)
                    if not v.any():
                        v[int(np.argmax(p[m]))] = 1.0
                elif m in SINGLE_PRECISION:
                    v = v.astype(np.float32).astype(np.float64)
                rc = None
                if not noul:
                    rc = int(np.argmax(v))
                    if m in TIED and n_opt > 2 and rng.random() < 0.02:
                        other = (rc + 1 + int(rng.integers(0, n_opt - 1))) % n_opt
                        if other > rc:
                            v[other] = v[rc]                         # exact tie; the later option is reported
                            rc = other
                out_p[m].append(v.tolist())
                out_rc[m].append(rc)
    out_p[DUPLICATE[1]] = out_p[DUPLICATE[0]]
    out_rc[DUPLICATE[1]] = out_rc[DUPLICATE[0]]
    return out_p, out_rc


def _observations(m, rows, rng):
    R = len(rows)
    counts = np.array([len(r["questions"]) for r in rows])
    proxy = np.array([r["_evaluation"]["proxy_tokens"] for r in rows])
    tin = (proxy * rng.uniform(0.9, 1.3) + 40 * counts).astype(np.int64)
    if m in NO_OUTPUT_TOKENS:
        tout = np.zeros(R, np.int64)
    elif m in ONE_OUTPUT_TOKEN_PER_FIELD:
        tout = counts.astype(np.int64)
    else:
        tout = (counts * rng.integers(20, 90, R)).astype(np.int64)
    cost = tin * PRICE[m]
    provider = np.array([f"host-{ORGS[m]}"] * R, dtype=object)
    transport = np.array(["native"] * R, dtype=object)
    partitions = [None] * R
    calls = np.ones(R, np.int32)
    if m == SECOND_PROVIDER:
        other = rng.random(R) < 0.03
        provider[other] = "host-other"
        cost[other] *= 2.0
    if m == DISCOUNTED:
        cheap = rng.random(R) < 0.05
        cost[cheap] *= rng.uniform(0.2, 0.9, int(cheap.sum()))
    if m in ADAPTED:
        for i in np.flatnonzero((counts >= 8) & (rng.random(R) < 0.5)):
            half = counts[i] // 2
            partitions[i] = [list(range(half)), list(range(half, counts[i]))]
            transport[i], provider[i], calls[i] = "capacity_adapted", None, 2
    retries = (rng.random(R) < 0.01).astype(np.int32)
    unknown = rng.random(R) < 0.005
    latency = rng.lognormal(np.log(0.3 + 0.02 * m), 0.5, R) + (rng.random(R) < 0.01) * rng.uniform(5, 40, R)

    def maybe(values, kind):
        return pa.array([None if u else v for v, u in zip(values.tolist(), unknown)], kind)

    return pa.table({
        "item_index": np.arange(R, dtype=np.int32), "provider": pa.array(provider.tolist(), pa.string()),
        "transport": transport.tolist(), "adapted_field_partitions": pa.array(partitions, pa.list_(pa.list_(pa.int32()))),
        "leaf_physical_calls": calls, "realized_physical_calls": calls + retries,
        "leaf_input_tokens": tin, "leaf_output_tokens": tout, "leaf_cost_usd": cost,
        "leaf_deployment_latency_s": latency, "realized_input_tokens": maybe(tin, pa.int64()),
        "realized_output_tokens": maybe(tout, pa.int64()), "realized_known_input_tokens": tin,
        "realized_known_output_tokens": tout, "realized_cost_usd": maybe(cost, pa.float64()),
        "realized_known_cost_usd": cost, "realized_missing_cost_usd": unknown.astype(np.int32),
        "realized_unresolved_reserved_nusd": np.zeros(R, np.int64),
        "realized_service_sum_s": latency * (1 + retries),
    }, schema=OBSERVATIONS)


def make(out, seed=7):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    source = out / "results/decision_index/paper/dataset/candidate"
    rows_dir = out / "data/decision_index/common-paid-65/inputs"
    rows_dir.mkdir(parents=True)
    source.mkdir(parents=True)
    area_of = {c: a for a, cs in AREAS.items() for c in cs}
    org_names = sorted(set(ORGS))
    org_code = np.array([org_names.index(o) for o in ORGS])
    base_rng = np.random.default_rng([seed, 0])
    org_skill = base_rng.normal(1.6, 0.5, len(org_names))
    model_skill = org_skill[org_code] + base_rng.normal(0.0, 0.3, M)
    bench_rows = []
    for cid in sorted(PROFILE):
        rng = np.random.default_rng([seed, cid])
        rows, truth, fold, split = _rows(cid, rng)
        skill = model_skill + rng.normal(0.0, 0.35, M) + rng.normal(0.0, 0.5)
        skill[SPECIALIST] = 3.2 if area_of[cid] == SPECIALIST_AREA else 0.2
        vectors, reported = _vectors(cid, truth, rng, skill, org_code)
        with gzip.GzipFile(rows_dir / f"{cid:03d}.jsonl.gz", "wb", mtime=0) as fh:
            for r in rows:
                fh.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
        R = len(rows)
        counts = [len(r["questions"]) for r in rows]
        d = source / f"catalog-{cid:03d}"
        _write(d / "items.parquet", pa.table({
            "item_index": np.arange(R, dtype=np.int32),
            "item_id": [S.public_id("item", cid, r["_evaluation"]["run_id"]) for r in rows],
            "fold": fold, "split_group_id": [_hash("split", cid, int(g)) for g in split],
            "native_group_id": [S.public_id("native_group", cid, r["_evaluation"]["group_id"]) for r in rows],
            "field_count": np.asarray(counts, np.int32),
            "row_sha256": [_hash("row", cid, i) for i in range(R)]}, schema=S.ITEMS))
        item = np.repeat(np.arange(R, dtype=np.int32), counts)
        pos = np.concatenate([np.arange(c, dtype=np.int32) for c in counts])
        qtype = [q["type"] for r in rows for q in r["questions"].values()]
        not_one = differs = adapted = calls = attempts = 0
        for m in range(M):
            _write(d / f"model-{m:02d}" / "predictions.parquet", pa.table({
                "item_index": item, "field_index": pos, "question_type": qtype,
                "probabilities": pa.array(vectors[m], pa.list_(pa.float64())),
                "reported_choice": pa.array(reported[m], pa.int32())}, schema=S.PREDICTIONS))
            obs = _observations(m, rows, np.random.default_rng([seed, cid, m]))
            _write(d / f"model-{m:02d}" / "observations.parquet", obs)
            for v, rc, t in zip(vectors[m], reported[m], qtype):
                if t == "choice":
                    total = v[0]
                    for x in v[1:]:
                        total = total + x
                    not_one += total != 1.0
                    differs += rc != int(np.argmax(v))
            adapted += obs.num_rows - obs["adapted_field_partitions"].null_count
            calls += int(np.sum(obs["leaf_physical_calls"].to_numpy()))
            attempts += int(np.sum(obs["realized_physical_calls"].to_numpy()))
        layouts = [reference_build.T.field_layout(cid, r) for r in rows]
        judged = sum(bool(a) for lay in layouts for _, _, a in lay)
        single = sum(len(a) == 1 for lay in layouts for _, _, a in lay)
        bench_rows.append(dict(
            catalog_id=cid, dataset=f"synthetic-{cid:03d}", requests=R, fields=len(item), judged_fields=judged,
            single_gold_fields=single, split_groups=len(set(split.tolist())),
            native_groups=len({r["_evaluation"]["group_id"] for r in rows}), vectors=len(item) * M,
            raw_choice_sums_adjusted_for_ensemble=int(not_one), reported_choice_differs_first_argmax=int(differs),
            observations=R * M, adapted_observations=int(adapted), successful_physical_calls=calls, attempts=attempts))
    with open(source / "benchmarks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(bench_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(bench_rows)
    names = [f"{ORGS[m]}/model-{m:02d}" for m in range(M)]
    with open(source / "models.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["model_index", "canonical_model_id", "route_id"])
        w.writerows([m, names[m], names[m]] for m in range(M))
    files = sorted(str(p.relative_to(source)) for p in source.rglob("*") if p.is_file())
    (source / "SHA256SUMS").write_text("".join(f"{S.sha256_file(source / f)}  {f}\n" for f in files))
    count_keys = [k for k in bench_rows[0] if k not in ("catalog_id", "dataset")]
    (source / "manifest.json").write_text(json.dumps(dict(
        catalog_ids=sorted(PROFILE), checksums_sha256=S.sha256_file(source / "SHA256SUMS"),
        counts={k: int(sum(b[k] for b in bench_rows)) for k in count_keys}, kit_commit="synthetic",
        model_count=M, protocol="synthetic", schema_version=S.SOURCE_ID_VERSION, scope="synthetic"), indent=1) + "\n")

    (out / "meta").mkdir()
    with open(out / "meta" / "benchmarks_meta.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["catalog_id", "area", "native_metric"])
        w.writerows([c, area_of[c], "accuracy"] for c in sorted(PROFILE))
    with open(out / "meta" / "models_meta.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["model_index", "organisation", "base_model", "input_price_usd_per_token"])
        w.writerows([m, ORGS[m], BASE.get(m, ""), repr(PRICE[m])] for m in range(M))
    with open(out / "meta" / "clearance.csv", "w", newline="") as fh:      # made-up release review, one row per cell
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["catalog_id", "route_id", "numeric_public_status"])
        for c in sorted(PROFILE):
            for m in range(M):
                status = "cleared" if m in CLEARED and c != HELD_BENCHMARK else ("unverified" if m < 2 else "conditional")
                w.writerow([c, names[m], status])
    cards = out / "ensemble_round2" / "model_cards"
    cards.mkdir(parents=True)
    with open(cards / "cards.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["model_index", "route_id", "canonical_model_id", "provider", "name", "listed_utc", "listed_date",
                    "listed_day", "context_length", "input_modalities", "price_prompt_usd_per_token",
                    "price_completion_usd_per_token", "hugging_face_id", "base_model_confirmed", "adaptation",
                    "served_params", "leaderboard_engine", "description"])
        for m in range(M):
            day = 20700 + 3 * m
            w.writerow([m, names[m], names[m], ORGS[m], f"Made-up model {m}", f"2026-10-{1 + m:02d}T00:00:00Z",
                        f"2026-10-{1 + m:02d}", day, 32768 * (1 + m % 3), "text", repr(PRICE[m]), "0.0",
                        BASE.get(m, ""), BASE.get(m, ""), "full fine-tune" if m in BASE else "", "", "",
                        "made-up description"])
    (out / "PLANTED.json").write_text(json.dumps(dict(
        seed=seed, duplicate=dict(copy=DUPLICATE[1], of=DUPLICATE[0]), organisations=ORGS,
        specialist=dict(model=SPECIALIST, area=SPECIALIST_AREA, cheapest=True),
        rounded_to_4_decimals=list(ROUNDED), single_precision=list(SINGLE_PRECISION), exact_zeros=list(SPARSE),
        tied_maxima=list(TIED), adapted=list(ADAPTED), second_provider_double_price=SECOND_PROVIDER,
        discounted=DISCOUNTED, cleared_models=list(CLEARED), held_benchmark=HELD_BENCHMARK, areas={a: list(c) for a, c in AREAS.items()}), indent=1) + "\n")
    reference_build.build(source, rows_dir, out / "meta", out / "processed" / "local")
    return bench_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write the made-up corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    rows = make(a.out, a.seed)
    print(json.dumps(dict(benchmarks=len(rows), requests=sum(r["requests"] for r in rows),
                          fields=sum(r["fields"] for r in rows), judged=sum(r["judged_fields"] for r in rows))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

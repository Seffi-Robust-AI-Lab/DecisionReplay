"""Minimal reference build of a tier "local" corpus.  The full builder must write the same bytes.

    python3 -B -m data.reference_build --source DIR --rows DIR --meta DIR --out DIR

``--source``: the source export (models.csv, benchmarks.csv, SHA256SUMS, catalog-NNN/...).
``--rows``:   folder with the benchmark rows, NNN.jsonl.gz (local only).
``--meta``:   folder with benchmarks_meta.csv (catalog_id, area, native_metric) and models_meta.csv
              (model_index, organisation, base_model, input_price_usd_per_token).

This file has no validation beyond the binding checks, no tiers and no release logic: those belong to
``build_corpus``.  It exists so that the table definitions can be tested and so that test corpora can be made.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

from . import schema as S
from . import tables as T


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([r[c] for c in columns])


def build(source, rows_dir, meta, out):
    source, rows_dir, meta, out = Path(source), Path(rows_dir), Path(meta), Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    models = _csv(source / "models.csv")
    bench = _csv(source / "benchmarks.csv")
    bmeta = {int(r["catalog_id"]): r for r in _csv(meta / "benchmarks_meta.csv")}
    mmeta = {int(r["model_index"]): r for r in _csv(meta / "models_meta.csv")}
    counts = dict(requests=0, fields=0, judged_fields=0, vectors=0, observations=0)
    rows_out = []
    for b in bench:
        cid = int(b["catalog_id"])
        src = source / f"catalog-{cid:03d}"
        S.catalog_dir(out, cid).mkdir(parents=True)
        shutil.copyfile(src / "items.parquet", S.table_path(out, cid, "items"))
        items = S.read_table(S.table_path(out, cid, "items"), S.ITEMS)
        rows = T.read_rows(rows_dir / f"{cid:03d}.jsonl.gz")
        T.check_binding(cid, items, rows)
        fields, labels = T.fields_and_labels(cid, rows)
        S.write_table(S.table_path(out, cid, "fields"), fields, S.FIELDS)
        S.write_table(S.table_path(out, cid, "labels"), labels, S.LABELS)
        S.write_table(S.table_path(out, cid, "splits"),
                      T.splits_table(items, [r["_evaluation"]["run_id"] for r in rows]), S.SPLITS)
        for m in range(len(models)):
            S.table_path(out, cid, "predictions", m).parent.mkdir()
            shutil.copyfile(src / f"model-{m:02d}" / "predictions.parquet", S.table_path(out, cid, "predictions", m))
            usage, latency = T.usage_and_latency(pq.read_table(src / f"model-{m:02d}" / "observations.parquet"))
            S.write_table(S.table_path(out, cid, "usage", m), usage, S.USAGE)
            S.write_table(S.table_path(out, cid, "latency", m), latency, S.LATENCY)
        judged = sum(labels["judged"].to_pylist())
        counts["requests"] += items.num_rows
        counts["fields"] += fields.num_rows
        counts["judged_fields"] += judged
        counts["vectors"] += fields.num_rows * len(models)
        counts["observations"] += items.num_rows * len(models)
        rows_out.append(dict(catalog_id=cid, name=b["dataset"], area=bmeta[cid]["area"],
                             native_metric=bmeta[cid]["native_metric"], requests=items.num_rows,
                             fields=fields.num_rows, judged_fields=judged))
    _write_csv(out / "benchmarks.csv", S.BENCHMARK_COLUMNS_LOCAL, rows_out)
    _write_csv(out / "models.csv", S.MODEL_COLUMNS, [
        dict(model_index=i, model_code=f"M{i:02d}", model_name=m["route_id"],
             organisation=mmeta[i]["organisation"], base_model=mmeta[i]["base_model"],
             input_price_usd_per_token=mmeta[i]["input_price_usd_per_token"]) for i, m in enumerate(models)])
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    sums = "".join(f"{S.sha256_file(out / f)}  {f}\n" for f in files)
    (out / "SHA256SUMS").write_text(sums, encoding="utf-8")
    manifest = dict(
        schema_version=S.SCHEMA_VERSION, tier="local", names="real",
        catalog_ids=[int(b["catalog_id"]) for b in bench], model_count=len(models), counts=counts,
        source=dict(checksums_sha256=S.sha256_file(source / "SHA256SUMS"), id_version=S.SOURCE_ID_VERSION),
        tables=list(S.TABLES_BY_TIER["local"]), checksums_sha256=S.sha256_file(out / "SHA256SUMS"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Minimal reference build of a local-tier corpus")
    for name in ("source", "rows", "meta", "out"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args(argv)
    manifest = build(a.source, a.rows, a.meta, a.out)
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

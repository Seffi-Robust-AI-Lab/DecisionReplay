"""Frozen data contract of the processed corpus (schema ``decision-replay-corpus-v1``).

A processed corpus is a folder.  It never holds request text, option text or gold text.

    manifest.json                               schema version, tier, names mode, counts, source binding
    SHA256SUMS                                  one line per file below
    models.csv                                  one row per model (MODEL_COLUMNS)
    benchmarks.csv                              one row per benchmark (BENCHMARK_COLUMNS)
    catalog-NNN/items.parquet                   ITEMS         one row per request           (copied from the source)
    catalog-NNN/fields.parquet                  FIELDS        one row per field
    catalog-NNN/splits.parquet                  SPLITS        one row per request
    catalog-NNN/labels.parquet                  LABELS        one row per field             (tier "local" only)
    catalog-NNN/model-MM/predictions.parquet    PREDICTIONS   one row per field             (copied from the source)
    catalog-NNN/model-MM/usage.parquet          USAGE         one row per request           (not in tier "open")
    catalog-NNN/model-MM/latency.parquet        LATENCY       one row per request           (tier "local" only)

Tiers.  "local": everything, for the machine that holds the benchmark rows.  "gated": no labels and no latency,
all cells.  "open": no labels, no latency, no usage, and only the cells cleared for open release.

Version 2 (``decision-replay-corpus-v2``, the authors' decision of 23 November 2026) exists for the open tier only.
Its open tier also carries, for every request and model, the request totals of PUBLIC_USAGE (calls, input and
output tokens, billed cost) and the measured latency.  It has no serving host, no transport and no attempt count.
Corpora of version 1 are read and verified as before.

Every module of the package reads and writes these tables through this file, so that two builds of the same
inputs are byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "decision-replay-corpus-v1"
PUBLIC_SCHEMA_VERSION = "decision-replay-corpus-v2"      # the open tier with usage and latency
SOURCE_ID_VERSION = "decision-fusion-numeric-v1"     # id scheme of the source export; do not change
N_FOLDS = 5
TIERS = ("local", "gated", "open")
QUESTION_TYPES = ("choice", "noul")                  # "noul" is a yes/no field: vector [1 - p, p]
TRANSPORTS = ("native", "capacity_adapted")

ITEMS = pa.schema([
    ("item_index", pa.int32()), ("item_id", pa.string()), ("fold", pa.int8()),
    ("split_group_id", pa.string()), ("native_group_id", pa.string()),
    ("field_count", pa.int32()), ("row_sha256", pa.string()),
])
PREDICTIONS = pa.schema([
    ("item_index", pa.int32()), ("field_index", pa.int32()), ("question_type", pa.string()),
    ("probabilities", pa.list_(pa.float64())), ("reported_choice", pa.int32()),
])
FIELDS = pa.schema([
    ("item_index", pa.int32()), ("field_index", pa.int32()), ("question_type", pa.string()),
    ("n_options", pa.int32()),
])
# split_group: integer code of the source split group, numbered by first appearance when requests are taken in
# run_rank order.  run_rank: rank of the request in the collection's run order (a permutation of 0..R-1).
SPLITS = pa.schema([
    ("item_index", pa.int32()), ("fold", pa.int8()), ("split_group", pa.int32()), ("run_rank", pa.int32()),
])
# accepted: positions of the accepted options, ascending; empty when the field is not judged.
LABELS = pa.schema([
    ("item_index", pa.int32()), ("field_index", pa.int32()), ("judged", pa.bool_()),
    ("accepted", pa.list_(pa.int32())),
])
# The accepted call or calls behind the stored answer.  provider is null for requests answered in field batches.
USAGE = pa.schema([
    ("item_index", pa.int32()), ("provider", pa.string()), ("transport", pa.string()),
    ("calls", pa.int32()), ("attempts", pa.int32()),
    ("input_tokens", pa.int64()), ("output_tokens", pa.int64()), ("cost_usd", pa.float64()),
])
LATENCY = pa.schema([("item_index", pa.int32()), ("latency_s", pa.float64())])
# Usage of the version-2 open tier: the request totals, without serving host, transport and attempts.
PUBLIC_USAGE = pa.schema([USAGE.field(name) for name in
                          ("item_index", "calls", "input_tokens", "output_tokens", "cost_usd")])

TABLES = {"items": ITEMS, "fields": FIELDS, "splits": SPLITS, "labels": LABELS,
          "predictions": PREDICTIONS, "usage": USAGE, "latency": LATENCY}
PER_BENCHMARK = ("items", "fields", "splits", "labels")
PER_MODEL = ("predictions", "usage", "latency")
TABLES_BY_TIER = {
    "local": ("items", "fields", "splits", "labels", "predictions", "usage", "latency"),
    "gated": ("items", "fields", "splits", "predictions", "usage"),
    "open": ("items", "fields", "splits", "predictions"),
}
PUBLIC_TABLES = ("items", "fields", "splits", "predictions", "usage", "latency")     # open tier of version 2
COPIED = ("items", "predictions")                    # byte copies of the source files

# models.csv.  With names mode "coded", model_name, organisation and base_model hold codes (M00, O03, B02 or "").
MODEL_COLUMNS = ("model_index", "model_code", "model_name", "organisation", "base_model",
                 "input_price_usd_per_token")
# benchmarks.csv.  judged_fields is written in tier "local" only (it is label information).
BENCHMARK_COLUMNS = ("catalog_id", "name", "area", "native_metric", "requests", "fields")
BENCHMARK_COLUMNS_LOCAL = BENCHMARK_COLUMNS + ("judged_fields",)
MANIFEST_KEYS = ("schema_version", "tier", "names", "catalog_ids", "model_count", "counts", "source",
                 "tables", "checksums_sha256")
# Optional key of an open tier of schema version 2 from which requests were removed for every model:
# {catalogue id as a string: number of removed requests}.  The benchmark rows of such a benchmark hold that many
# more requests than the corpus, and the rows are bound to the corpus by request id (tables.rows_of).
EXCLUDED_KEY = "excluded_requests"


def corpus_tables(version, tier):
    """Return the tables of a corpus of this schema version and tier."""
    if version == PUBLIC_SCHEMA_VERSION and tier == "open":
        return PUBLIC_TABLES
    if version == SCHEMA_VERSION and tier in TABLES_BY_TIER:
        return TABLES_BY_TIER[tier]
    raise ValueError("unsupported corpus schema version or tier")


def usage_schema(version):
    """Return the schema of the usage table of a corpus of this schema version."""
    if version == PUBLIC_SCHEMA_VERSION:
        return PUBLIC_USAGE
    if version == SCHEMA_VERSION:
        return USAGE
    raise ValueError("unsupported corpus schema version")


def catalog_dir(corpus, cid):
    return Path(corpus) / f"catalog-{int(cid):03d}"


def table_path(corpus, cid, table, model=None):
    """Path of one table file.  ``model`` is the model index for the per-model tables."""
    if table in PER_BENCHMARK:
        if model is not None:
            raise ValueError(f"{table} is not a per-model table")
        return catalog_dir(corpus, cid) / f"{table}.parquet"
    if table in PER_MODEL:
        if model is None:
            raise ValueError(f"{table} needs a model index")
        return catalog_dir(corpus, cid) / f"model-{int(model):02d}" / f"{table}.parquet"
    raise KeyError(table)


def write_table(path, table, schema):
    """The one way a derived table is written: fixed schema, one row group, zstd, no extra metadata."""
    table = table.cast(schema).replace_schema_metadata(None)
    if table.schema != schema:
        raise ValueError(f"{path}: schema differs from the contract")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", compression_level=9, version="2.6", data_page_version="2.0",
                   use_dictionary=False, write_statistics=False, row_group_size=max(len(table), 1),
                   store_schema=False)


def read_table(path, schema, columns=None):
    table = pq.read_table(path, columns=columns)
    expected = schema if columns is None else pa.schema([schema.field(c) for c in columns])
    if not table.schema.equals(expected, check_metadata=False):
        raise ValueError(f"{path}: schema differs from the contract")
    return table


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def public_id(kind, cid, value):
    """Opaque id of the source export: SHA-256 of a canonical JSON list.  ``kind`` is "item" or "native_group"."""
    blob = json.dumps([SOURCE_ID_VERSION, kind, int(cid), value], sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def roles(f):
    """(fit folds, validation fold, the four non-test folds) of test fold ``f``."""
    validation = (f + 1) % N_FOLDS
    fit = tuple(g for g in range(N_FOLDS) if g not in (f, validation))
    train = tuple(g for g in range(N_FOLDS) if g != f)
    return fit, validation, train

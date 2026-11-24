"""Tables for a dataset hub: the open tier of a corpus as one Parquet file per table and benchmark.

    python3 -B -m acl_resource.data.hub --corpus CORPUS --out OUT [--card CARD] [--notices NOTICES]

A corpus keeps one file per benchmark and model, and the benchmark and the model of a row are in the path of its
file.  A hub reads a table from a list of files, so here every row names its benchmark (``catalog_id``) and, in the
per-model tables, its model (``model_index``).  Both are integers; the tables ``benchmarks`` and ``models`` give
the names.  Nothing else is added and nothing is left out: every file is read back and compared with the corpus,
and the row totals are compared with the corpus manifest.

    OUT/README.md                         the dataset card (with --card): the template with the corpus's counts
    OUT/corpus_manifest.json              the manifest of the corpus, unchanged
    OUT/benchmarks.csv, OUT/models.csv    unchanged
    OUT/data/benchmarks/benchmarks.parquet, OUT/data/models/models.parquet
                                          the two lists again as tables, because a hub reads every table of a
                                          dataset in one file format
    OUT/release_notices.csv               the conditions that travel with the files (with --notices), unchanged
    OUT/data/TABLE/catalog-NNN.parquet    one file per table and benchmark; a per-model table has one row group
                                          per model, in model order
    OUT/SHA256SUMS                        one line per file above

Only an open tier without labels is exported, and the corpus is verified first.  The folder has no file named
``manifest.json``: its tables are not in the layout of a corpus, and the loader does not read them.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import build_corpus, load
from . import schema as S

ADDED = ("catalog_id", "model_index")                # the two columns a hub table has and a corpus table has not
COUNT_OF = {"items": "requests", "splits": "requests", "fields": "fields", "predictions": "vectors",
            "usage": "observations", "latency": "observations"}       # table -> key of the manifest's counts
ROW_OF = {"items": "request", "splits": "request", "fields": "field", "predictions": "field and model",
          "usage": "request and model", "latency": "request and model"}
NAMES = {"benchmarks": (S.BENCHMARK_COLUMNS, {"catalog_id": pa.int32(), "requests": pa.int32(), "fields": pa.int32()}),
         "models": (S.MODEL_COLUMNS, {"model_index": pa.int32(), "input_price_usd_per_token": pa.float64()})}
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def hub_schema(table, version):
    """(schema of the hub table, schema of the corpus table it is made from)."""
    base = S.usage_schema(version) if table == "usage" else S.TABLES[table]
    added = ADDED if table in S.PER_MODEL else ADDED[:1]
    return pa.schema([pa.field(name, pa.int32()) for name in added] + list(base)), base


def _same_array(x, y):
    if x.null_count != y.null_count or (x.null_count and not x.is_null().equals(y.is_null())):
        return False
    if pa.types.is_list(x.type):
        return pc.list_value_length(x).equals(pc.list_value_length(y)) and _same_array(x.flatten(), y.flatten())
    if pa.types.is_floating(x.type):
        return np.array_equal(x.to_numpy(zero_copy_only=False), y.to_numpy(zero_copy_only=False), equal_nan=True)
    return x.equals(y)


def same(a, b):
    """Whether two tables hold the same values in the same order (a missing value equals a missing value)."""
    if not a.schema.equals(b.schema, check_metadata=False) or a.num_rows != b.num_rows:
        return False
    return all(_same_array(pa.concat_arrays(x.chunks) if x.num_chunks else pa.array([], x.type),
                           pa.concat_arrays(y.chunks) if y.num_chunks else pa.array([], y.type))
               for x, y in zip(a.columns, b.columns))


def _parts(corpus, cid, table, base, models):
    """The corpus's rows of one table and benchmark, as (model index or None, table) in the order of the hub file."""
    if table in S.PER_MODEL:
        return [(m, S.read_table(S.table_path(corpus, cid, table, m), base).cast(base)) for m in range(models)]
    return [(None, S.read_table(S.table_path(corpus, cid, table), base).cast(base))]


def _write(path, schema, cid, parts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema, compression="zstd", compression_level=9) as writer:
        for m, part in parts:
            keys = [pa.array(np.full(part.num_rows, value, np.int32)) for value in (cid, m) if value is not None]
            writer.write_table(pa.Table.from_arrays(keys + part.columns, schema=schema))


def _check(path, schema, base, cid, parts):
    """Read a hub file back and compare it with the corpus; return its number of rows."""
    back = pq.read_table(path)
    if not back.schema.equals(schema, check_metadata=False):
        raise ValueError(f"{path.parent.name}/{path.name}: schema differs from the hub schema")
    start = 0
    for m, part in parts:
        piece = back.slice(start, part.num_rows)
        start += part.num_rows
        if not ((piece["catalog_id"].to_numpy() == int(cid)).all()
                and (m is None or (piece["model_index"].to_numpy() == m).all())
                and same(piece.select(base.names).cast(base), part)):
            raise ValueError(f"{path.parent.name}/{path.name}: rows differ from the corpus")
    if start != back.num_rows:
        raise ValueError(f"{path.parent.name}/{path.name}: more rows than the corpus")
    return back.num_rows


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def names_table(corpus, name):
    """``benchmarks.csv`` or ``models.csv`` as a table: the numeric columns as numbers, the others as text."""
    columns, numeric = NAMES[name]
    rows = _csv(Path(corpus) / f"{name}.csv")
    if any(tuple(row) != tuple(columns) for row in rows):
        raise ValueError(f"{name}.csv does not have the columns of an open tier")
    arrays = []
    for column in columns:
        kind = numeric.get(column, pa.string())
        convert = int if pa.types.is_integer(kind) else float if pa.types.is_floating(kind) else str
        arrays.append(pa.array([convert(row[column]) for row in rows], kind))
    return pa.table(arrays, names=list(columns))


def _type_name(kind):
    if pa.types.is_list(kind):
        return "list of " + _type_name(kind.value_type)
    return {"double": "float64"}.get(str(kind), str(kind))


def facts(corpus):
    """The counts and lists a guide or a card states about an open corpus, as text for ``fill``."""
    corpus = Path(corpus)
    manifest = load.read_manifest(corpus)
    models, benchmarks = load.list_models(corpus), load.list_benchmarks(corpus)
    counts, tables, version = manifest["counts"], list(manifest["tables"]), manifest["schema_version"]
    areas = {}
    for b in benchmarks:
        areas[b["area"]] = areas.get(b["area"], 0) + 1
    excluded = manifest.get(S.EXCLUDED_KEY, {})
    rows = ["| Configuration | One row per | Rows |", "|---|---|---|"]
    rows += [f"| `{t}` | {ROW_OF[t]} | {counts[COUNT_OF[t]]:,} |" for t in tables]
    rows += [f"| `benchmarks` | benchmark | {len(benchmarks):,} |", f"| `models` | model | {len(models):,} |"]
    columns = ["| Configuration | Columns |", "|---|---|"]
    for t in tables:
        schema = hub_schema(t, version)[0]
        columns.append(f"| `{t}` | " + ", ".join(f"`{f.name}` ({_type_name(f.type)})" for f in schema) + " |")
    for name in NAMES:
        schema = names_table(corpus, name).schema
        columns.append(f"| `{name}` | " + ", ".join(f"`{f.name}` ({_type_name(f.type)})" for f in schema) + " |")
    configs = []
    for t in tables + list(NAMES):
        configs += [f"- config_name: {t}", f"  data_files: data/{t}/*.parquet"]
        if t == "predictions":
            configs.append("  default: true")
    return {
        "schema_version": version,
        "model_count": f"{len(models):,}",
        "provider_count": f"{len({m['organisation'] for m in models}):,}",
        "benchmark_count": f"{len(benchmarks):,}",
        "requests": f"{counts['requests']:,}",
        "fields": f"{counts['fields']:,}",
        "areas": ", ".join(f"{area} ({n})" for area, n in areas.items()),
        "excluded_requests": f"{sum(excluded.values()):,}",
        "excluded_benchmarks": f"{len(excluded):,}",
        "table_rows": "\n".join(rows),
        "table_columns": "\n".join(columns),
        "configs": "\n".join(configs),
    }


def fill(text, values):
    """Replace every ``{{key}}`` of a template by its computed value; a key that is not computed is an error."""
    def one(match):
        if match.group(1) not in values:
            raise ValueError(f"the template asks for {match.group(1)}, which is not computed")
        return values[match.group(1)]
    return PLACEHOLDER.sub(one, text)


def export(corpus, out, card=None, notices=None):
    """Write the hub folder of an open corpus and return the number of rows of each table."""
    corpus, out = Path(corpus), Path(out)
    if out.exists():
        raise ValueError("the output folder exists already")
    manifest = build_corpus.verify(corpus)
    version, tables = manifest["schema_version"], list(manifest["tables"])
    if manifest["tier"] != "open" or "labels" in tables or list(corpus.rglob("labels.parquet")):
        raise ValueError("only an open tier without labels is exported")
    if tables != list(S.corpus_tables(version, "open")):
        raise ValueError("the tables of the manifest are not those of an open tier")
    if notices is not None:
        released = {m["model_name"] for m in load.list_models(corpus)}
        if {row["route_id"] for row in _csv(notices)} != released:
            raise ValueError("the notices are not those of the models of the corpus")
    text = None if card is None else fill(Path(card).read_text(encoding="utf-8"), facts(corpus))
    rows = dict.fromkeys(tables, 0)
    for table in tables:
        schema, base = hub_schema(table, version)
        for cid in manifest["catalog_ids"]:
            parts = _parts(corpus, cid, table, base, manifest["model_count"])
            path = out / "data" / table / f"catalog-{int(cid):03d}.parquet"
            _write(path, schema, cid, parts)
            rows[table] += _check(path, schema, base, cid, parts)
    for table, n in rows.items():
        if n != manifest["counts"][COUNT_OF[table]]:
            raise ValueError(f"{table}: {n} rows, and the manifest counts {manifest['counts'][COUNT_OF[table]]}")
    shutil.copyfile(corpus / "manifest.json", out / "corpus_manifest.json")
    for name in NAMES:
        shutil.copyfile(corpus / f"{name}.csv", out / f"{name}.csv")
        table, path = names_table(corpus, name), out / "data" / name / f"{name}.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(table, path, compression="zstd", compression_level=9)
        if not same(pq.read_table(path), table):
            raise ValueError(f"{name}: the table differs from {name}.csv")
    if notices is not None:
        shutil.copyfile(notices, out / "release_notices.csv")
    if text is not None:
        (out / "README.md").write_text(text, encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "SHA256SUMS").write_text(
        "".join(f"{S.sha256_file(p)}  {p.relative_to(out).as_posix()}\n" for p in files), encoding="utf-8")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the tables of an open corpus for a dataset hub")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--card", type=Path, help="template of the dataset card (docs/HUB_CARD.md)")
    parser.add_argument("--notices", type=Path, help="release notices of the models (data/meta/release_notices.csv)")
    args = parser.parse_args(argv)
    try:
        rows = export(args.corpus, args.out, args.card, args.notices)
    except (OSError, ValueError, KeyError) as error:
        print("hub: " + str(error), file=sys.stderr)
        return 1
    for table, n in rows.items():
        print(f"{table}: {n:,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

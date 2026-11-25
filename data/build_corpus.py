"""Build a processed corpus from a checksum-bound export, or verify an existing corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import schema as S
from . import tables as T
from . import tiers
from synthetic.make_synthetic import OBSERVATIONS

_HASH = re.compile(r"[0-9a-f]{64}")


def _check(condition, message):
    """Return normally when the condition holds, otherwise raise a validation error."""
    if not condition:
        raise ValueError(message)


def _csv(path, columns, exact=False):
    """Return CSV records after checking headers and record widths."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        valid = (header == list(columns) if exact else set(columns) <= set(header))
        _check(valid and len(header) == len(set(header)), f"{path.name} has invalid columns")
        rows = list(reader)
    _check(all(None not in r and None not in r.values() for r in rows),
           f"{path.name} contains an incomplete record")
    return rows


def _index(rows, key):
    """Return records indexed by a unique integer column."""
    result = {int(row[key]): row for row in rows}
    _check(len(result) == len(rows), f"duplicate {key} in metadata")
    return result


def _checksums(directory):
    """Return verified file digests and the digest of the exact list checked."""
    listing = directory / "SHA256SUMS"
    _check(listing.is_file() and not listing.is_symlink(), "missing or invalid SHA256SUMS")
    result = {}
    data = listing.read_bytes()
    for line in data.decode("utf-8").splitlines():
        parts = line.split(maxsplit=1)
        _check(len(parts) == 2 and _HASH.fullmatch(parts[0]), "invalid checksum entry")
        digest, name = parts
        path = Path(name)
        _check(not path.is_absolute() and ".." not in path.parts and name == path.as_posix()
               and path.parts and name not in result, "unsafe or duplicate checksum entry")
        target = directory / path
        _check(not any((directory / p).is_symlink() for p in (path, *path.parents)),
               "checksum entry uses a symbolic link")
        _check(target.is_file(), f"missing file in checksum list: {name}")
        _check(S.sha256_file(target) == digest, f"checksum mismatch: {name}")
        result[name] = digest
    return result, hashlib.sha256(data).hexdigest()


def _table(path, schema):
    """Return one Parquet file with the exact contracted schema."""
    try:
        return S.read_table(path, schema)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid {path.name} file or schema") from error


def _complete(table, nullable=()):
    """Return normally when all required column values are present."""
    for name in table.column_names:
        _check(name in nullable or not table[name].null_count, f"missing values in {name}")


def _items(items):
    """Check request indices, identifier strings, field counts and frozen folds."""
    _complete(items)
    _check(np.array_equal(items["item_index"].to_numpy(), np.arange(len(items))),
           "items are not in request order")
    _check((items["field_count"].to_numpy() > 0).all(), "request has no fields")
    fold = items["fold"].to_numpy()
    _check(((0 <= fold) & (fold < S.N_FOLDS)).all(), "fold outside the frozen range")
    for name in ("item_id", "native_group_id", "split_group_id", "row_sha256"):
        _check(all(_HASH.fullmatch(s) for s in items[name].unique().to_pylist()),
               f"invalid hash string in {name}")
    _check(len(items["item_id"].unique()) == len(items), "duplicate item_id")


def _fields(items, fields):
    """Check field positions, types and option counts against the requests."""
    _complete(fields)
    counts = items["field_count"].to_numpy()
    _check(counts.sum() == len(fields), "field counts differ from items")
    request = np.repeat(items["item_index"].to_numpy(), counts)
    starts = np.cumsum(counts) - counts
    _check(np.array_equal(request, fields["item_index"].to_numpy()) and
           np.array_equal(np.arange(len(fields)) - starts[request], fields["field_index"].to_numpy()),
           "fields are not in request and field order")
    kinds = fields["question_type"]
    widths = fields["n_options"].to_numpy()
    _check(set(kinds.unique().to_pylist()) <= set(S.QUESTION_TYPES), "unknown question type")
    _check((widths >= 2).all() and (widths[kinds.to_numpy() == "noul"] == 2).all(),
           "invalid field option count")


def _labels(fields, labels):
    """Return the judged field count after checking accepted option positions."""
    _complete(labels)
    for key in ("item_index", "field_index"):
        _check(fields[key].equals(labels[key]), "labels do not follow the fields table")
    for accepted, judged, width in zip(labels["accepted"].to_pylist(), labels["judged"].to_pylist(),
                                       fields["n_options"].to_numpy()):
        _check(judged == bool(accepted) and all(i is not None and 0 <= i < width for i in accepted)
               and accepted == sorted(set(accepted)), "invalid accepted option positions")
    return int(labels["judged"].to_numpy().sum())


def _predictions(predictions, fields):
    """Check each raw vector and reported choice against the corresponding field."""
    _complete(predictions, ("reported_choice",))
    for name in ("item_index", "field_index", "question_type"):
        _check(predictions[name].equals(fields[name]), "predictions differ from the field layout")
    vectors = predictions["probabilities"].combine_chunks()
    offsets = vectors.offsets.to_numpy()
    widths = fields["n_options"].to_numpy()
    _check(np.array_equal(np.diff(offsets), widths), "probability vector has incorrect option count")
    values = vectors.values.to_numpy(zero_copy_only=False)
    _check(np.isfinite(values).all() and (values >= 0).all(), "invalid probability")
    if len(widths):
        _check((np.maximum.reduceat(values, offsets[:-1]) > 0).all(), "probability vector has no mass")
    choice = predictions["reported_choice"]
    yes_no = fields["question_type"].to_numpy() == "noul"
    _check(np.array_equal(choice.is_null().to_numpy(), yes_no), "invalid reported_choice nulls")
    reported = choice.fill_null(0).to_numpy()
    _check(((0 <= reported) & (reported < widths)).all(), "reported_choice outside the options")


def _usage(usage, items):
    """Return observed provider names after checking one complete usage row per request.

    The usage table of the version-2 open tier has no provider, transport and attempts; it returns no name.
    """
    _complete(usage, ("provider",))
    _check(usage["item_index"].equals(items["item_index"]), "usage is not one row per request")
    calls = usage["calls"].to_numpy()
    _check((calls > 0).all(), "invalid calls or attempts")
    for name in ("input_tokens", "output_tokens", "cost_usd"):
        values = usage[name].to_numpy()
        _check(np.isfinite(values).all() and (values >= 0).all(), f"invalid {name}")
    if "provider" not in usage.column_names:
        return set()
    transport = usage["transport"]
    _check(set(transport.unique().to_pylist()) <= set(S.TRANSPORTS), "unknown transport")
    adapted = transport.to_numpy() == "capacity_adapted"
    provider = usage["provider"]
    _check(np.array_equal(provider.is_null().to_numpy(), adapted), "incomplete provider information")
    _check((calls[adapted] > 1).all() and (usage["attempts"].to_numpy() >= calls).all(),
           "invalid calls or attempts")
    names = set(provider.drop_null().unique().to_pylist())
    _check("" not in names, "empty provider name")
    return names


def _latency(latency, items):
    """Check that every request has one finite non-negative latency."""
    _complete(latency)
    values = latency["latency_s"].to_numpy()
    _check(latency["item_index"].equals(items["item_index"]), "latency is not one row per request")
    _check(np.isfinite(values).all() and (values >= 0).all(), "invalid latency")


def derive_benchmark(source, rows_dir, cid, excluded=0):
    """Return checked items and the three benchmark tables derived from bound rows.

    ``excluded``: the number of requests of the benchmark rows that the corpus does not hold (schema.EXCLUDED_KEY).
    """
    items = _table(S.table_path(source, cid, "items"), S.ITEMS)
    _items(items)
    rows = T.rows_of(cid, items["item_id"].to_pylist(), T.read_rows(rows_dir / f"{cid:03d}.jsonl.gz"), excluded)
    T.check_binding(cid, items, rows)
    fields, labels = T.fields_and_labels(cid, rows)
    _fields(items, fields)
    splits = T.splits_table(items, [row["_evaluation"]["run_id"] for row in rows])
    return items, fields, labels, splits


def _source_usage(source, cid, model):
    """Return usage and latency derived from a source observations file."""
    cell = S.table_path(source, cid, "predictions", model).parent
    return T.usage_and_latency(_table(cell / "observations.parquet", OBSERVATIONS))


def _preflight(source, rows, benchmarks, models, bmeta, tier):
    """Return benchmark summaries and provider names after checking every source cell."""
    summaries, providers = [], set()
    for b in benchmarks:
        cid = int(b["catalog_id"])
        items, fields, labels, splits = derive_benchmark(source, rows, cid)
        judged = int(labels["judged"].to_numpy().sum())
        counts = dict(requests=len(items), fields=len(fields), judged_fields=judged,
                      split_groups=len(splits["split_group"].unique()))
        for key, value in counts.items():
            _check(int(b[key]) == value, f"catalogue {cid}: {key} differs from benchmarks.csv")
        for m in range(len(models)):
            try:
                _predictions(_table(S.table_path(source, cid, "predictions", m), S.PREDICTIONS), fields)
                usage, latency = _source_usage(source, cid, m)
                observed = _usage(usage, items)
                _latency(latency, items)
            except ValueError as error:
                raise ValueError(f"catalogue {cid}, model {m}: {error}") from error
            if tier == "gated":
                providers.update(observed)
        summaries.append(dict(catalog_id=cid, name=b["dataset"], area=bmeta[cid]["area"],
                              native_metric=bmeta[cid]["native_metric"], requests=len(items),
                              fields=len(fields), judged_fields=judged))
    return summaries, sorted(providers)


def _payload(directory, cids, models, tables):
    """Yield each expected CSV and Parquet payload path in a processed corpus."""
    yield directory / "models.csv"
    yield directory / "benchmarks.csv"
    for cid in cids:
        for name in tables:
            if name in S.PER_BENCHMARK:
                yield S.table_path(directory, cid, name)
            else:
                for m in models:
                    yield S.table_path(directory, cid, name, m)


def _counts(benchmarks, model_count, tables):
    """Return the manifest counts for the stored tables in reference order."""
    counts = dict(requests=sum(int(b["requests"]) for b in benchmarks),
                  fields=sum(int(b["fields"]) for b in benchmarks))
    if "labels" in tables:
        counts["judged_fields"] = sum(int(b["judged_fields"]) for b in benchmarks)
    counts["vectors"] = counts["fields"] * model_count
    if "usage" in tables:
        counts["observations"] = counts["requests"] * model_count
    return counts


def _available(out):
    """Check that an output directory can be published without replacing data."""
    _check(not out.is_symlink() and (not out.exists() or (out.is_dir() and not any(out.iterdir()))),
           "output is not an empty directory")


@contextmanager
def staged_output(out):
    """Yield a temporary sibling and publish its contents on successful exit."""
    out = Path(out)
    _available(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as directory:
        stage = Path(directory)
        yield stage
        _available(out)
        stage.rename(out)


def write_manifest(directory, manifest):
    """Write the payload checksums and return the bound manifest."""
    files = (p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())
    names = sorted(f for f in files if f not in ("manifest.json", "SHA256SUMS"))
    sums = "".join(f"{S.sha256_file(directory / name)}  {name}\n" for name in names)
    checksum_path = directory / "SHA256SUMS"
    checksum_path.write_text(sums, encoding="utf-8")
    manifest = dict(manifest, checksums_sha256=S.sha256_file(checksum_path))
    # Local manifests follow the frozen reference writer's insertion order.
    data = json.dumps(manifest, indent=1, sort_keys=manifest["tier"] != "local", allow_nan=False)
    (directory / "manifest.json").write_text(data + "\n", encoding="utf-8")
    return manifest


def _write_csv(path, columns, rows):
    """Write selected CSV columns in contracted order and return nothing."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _excluded_requests(source, cids, models, rules):
    """{catalogue id: [requests] bool, True where a request is removed} for the rules of a release decision.

    A rule names a model of the source, a serving host and the number of requests of the corpus's benchmarks that the
    host served for the model; these requests are removed for every model.  The count of every rule must be found
    in the source, no more and no less.
    """
    route = {row["route_id"]: int(row["model_index"]) for row in models}
    found = {}
    remove = {}
    for rule in rules:
        key = (rule.get("model"), rule.get("serving_host"))
        _check(set(rule) == {"model", "serving_host", "expected_requests"} and key[0] in route and isinstance(key[1], str)
               and key[1] and type(rule["expected_requests"]) is int and rule["expected_requests"] > 0
               and key not in found, "invalid request exclusion")
        found[key] = 0
        for cid in cids:
            usage, _ = _source_usage(source, cid, route[key[0]])
            match = np.asarray([host == key[1] for host in usage["provider"].to_pylist()])
            if match.any():
                found[key] += int(match.sum())
                remove[cid] = remove.get(cid, np.zeros(len(match), bool)) | match
        _check(found[key] == rule["expected_requests"],
               f"the source holds {found[key]} requests of {key[0]} served by {key[1]}, not "
               f"{rule['expected_requests']}")
    _check(all(not mask.all() for mask in remove.values()), "a request exclusion removes a whole benchmark")
    return remove


def _write_benchmark(source, rows, stage, cid, models, tables, usage_schema=S.USAGE, remove=None):
    """Write the requested benchmark tables and selected model cells; return the benchmark's size.

    ``remove``: [requests] bool, requests that are left out for every model; the others are renumbered in order.
    """
    items, fields, labels, splits = derive_benchmark(source, rows, cid)
    S.catalog_dir(stage, cid).mkdir()
    new_index = None
    if remove is not None and remove.any():
        new_index = np.where(remove, -1, np.cumsum(~remove) - 1).astype(np.int64)
        kept = [row for row, gone in zip(T.read_rows(rows / f"{cid:03d}.jsonl.gz"), remove, strict=True) if not gone]
        items = T.project(items, new_index)
        _items(items)
        T.check_binding(cid, items, kept)
        fields, labels = T.fields_and_labels(cid, kept)
        _fields(items, fields)
        splits = T.splits_table(items, [row["_evaluation"]["run_id"] for row in kept])
        S.write_table(S.table_path(stage, cid, "items"), items, S.ITEMS)
    else:
        shutil.copyfile(S.table_path(source, cid, "items"), S.table_path(stage, cid, "items"))
    for name, table in (("fields", fields), ("labels", labels), ("splits", splits)):
        if name in tables:
            S.write_table(S.table_path(stage, cid, name), table, S.TABLES[name])
    for m, original in enumerate(models):
        path = S.table_path(stage, cid, "predictions", m)
        path.parent.mkdir()
        if new_index is None:
            shutil.copyfile(S.table_path(source, cid, "predictions", original), path)
        else:
            predictions = _table(S.table_path(source, cid, "predictions", original), S.PREDICTIONS)
            S.write_table(path, T.project(predictions, new_index), S.PREDICTIONS)
        if "usage" in tables:
            usage, latency = _source_usage(source, cid, original)
            if new_index is not None:
                usage, latency = T.project(usage, new_index), T.project(latency, new_index)
            S.write_table(S.table_path(stage, cid, "usage", m), usage.select(usage_schema.names), usage_schema)
            if "latency" in tables:
                S.write_table(S.table_path(stage, cid, "latency", m), latency, S.LATENCY)
    return dict(requests=len(items), fields=len(fields), judged_fields=int(labels["judged"].to_numpy().sum()))


def build(source, rows, meta, out, tier="local", clearance=None, names="real", version=S.SCHEMA_VERSION,
          exclusions=None):
    """Build a validated corpus atomically and return its manifest.

    ``version`` is the schema version: version 2 is the open tier with usage and latency, under listed names.
    ``exclusions`` (version 2 only): the request exclusions of the release decision (``exclude_host_requests``).
    """
    source, rows, meta, out = map(Path, (source, rows, meta, out))
    _check(version != S.PUBLIC_SCHEMA_VERSION or (tier == "open" and names == "real"),
           "schema version 2 is the open tier with listed model names")
    _check(not exclusions or version == S.PUBLIC_SCHEMA_VERSION, "requests are excluded from the released corpus only")
    _available(out)
    _check(not out.resolve().is_relative_to(source.resolve()), "source export must stay read-only")
    sums, source_hash = _checksums(source)
    _check({"benchmarks.csv", "models.csv"} <= sums.keys(), "missing CSV in source checksum list")
    benchmarks = _csv(source / "benchmarks.csv",
                      ("catalog_id", "dataset", "requests", "fields", "judged_fields", "split_groups"))
    models = _csv(source / "models.csv", ("model_index", "route_id"))
    _check(benchmarks and models, "source has no benchmark or model")
    selected, original, clearance_hash = tiers.select_cells(tier, benchmarks, models, clearance)
    cids = [int(b["catalog_id"]) for b in benchmarks]
    for cid in cids:
        paths = [S.table_path(source, cid, "items")]
        for m in range(len(models)):
            pred = S.table_path(source, cid, "predictions", m)
            paths.extend((pred, pred.with_name("observations.parquet")))
        _check(all(p.relative_to(source).as_posix() in sums for p in paths),
               f"catalogue {cid}: table missing from source checksum list")
    tables = S.corpus_tables(version, tier)
    bmeta = _index(_csv(meta / "benchmarks_meta.csv", ("catalog_id", "area", "native_metric")), "catalog_id")
    mmeta = _index(_csv(meta / "models_meta.csv", ("model_index", "organisation", "base_model",
                                                  "input_price_usd_per_token")), "model_index")
    _check(set(cids) <= bmeta.keys() and set(range(len(models))) <= mmeta.keys(), "incomplete metadata")
    model_rows = tiers.model_rows(models, mmeta, original, names)
    summaries, providers = _preflight(source, rows, benchmarks, models, bmeta, tier)
    summaries = [b for b in summaries if b["catalog_id"] in selected]
    cols = S.BENCHMARK_COLUMNS_LOCAL if tier == "local" else S.BENCHMARK_COLUMNS
    manifest = dict(schema_version=version, tier=tier, names=names,
                    catalog_ids=selected, model_count=len(original),
                    counts=_counts(summaries, len(original), tables),
                    source=dict(checksums_sha256=source_hash, id_version=S.SOURCE_ID_VERSION),
                    tables=list(tables))
    if tier != "local":
        manifest["providers"] = providers
    if tier == "open":
        manifest["clearance_sha256"] = clearance_hash
    remove = {}
    if exclusions:
        remove = _excluded_requests(source, selected, models, exclusions)
        manifest[S.EXCLUDED_KEY] = {str(cid): int(mask.sum()) for cid, mask in sorted(remove.items())}
    with staged_output(out) as stage:
        for summary in summaries:
            cid = summary["catalog_id"]
            size = _write_benchmark(source, rows, stage, cid, original, tables, S.usage_schema(version),
                                    remove.get(cid))
            summary.update(size)
        manifest["counts"] = _counts(summaries, len(original), tables)
        _write_csv(stage / "benchmarks.csv", cols, summaries)
        _write_csv(stage / "models.csv", S.MODEL_COLUMNS, model_rows)
        return write_manifest(stage, manifest)


def _manifest(directory):
    """Return the manifest after checking the schema, source binding and tier declarations."""
    path = directory / "manifest.json"
    _check(path.is_file() and not path.is_symlink(), "missing or invalid manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _check(isinstance(manifest, dict) and set(S.MANIFEST_KEYS) <= manifest.keys(), "missing manifest keys")
    tier = manifest["tier"]
    _check(tier in S.TIERS and manifest["names"] in ("real", "coded"), "invalid tier or names mode")
    version = manifest["schema_version"]
    _check(version in (S.SCHEMA_VERSION, S.PUBLIC_SCHEMA_VERSION), "unsupported schema version")
    _check(version != S.PUBLIC_SCHEMA_VERSION or (tier == "open" and manifest["names"] == "real"
                                                  and manifest.get("providers") == []),
           "schema version 2 is the open tier with listed model names and no provider list")
    source = manifest["source"]
    _check(isinstance(source, dict) and set(source) == {"id_version", "checksums_sha256"}
           and source["id_version"] == S.SOURCE_ID_VERSION, "invalid source binding")
    keys = set(S.MANIFEST_KEYS)
    digests = [manifest["checksums_sha256"], source["checksums_sha256"]]
    if tier == "open":
        keys.add("clearance_sha256")
        digests.append(manifest.get("clearance_sha256"))
    _check(all(isinstance(d, str) and _HASH.fullmatch(d) for d in digests), "invalid manifest digest")
    cids = manifest["catalog_ids"]
    _check(isinstance(cids, list) and cids and all(type(c) is int and c >= 0 for c in cids)
           and len(cids) == len(set(cids)), "invalid manifest catalogue ids")
    if S.EXCLUDED_KEY in manifest:
        excluded = manifest[S.EXCLUDED_KEY]
        keys.add(S.EXCLUDED_KEY)
        _check(version == S.PUBLIC_SCHEMA_VERSION and isinstance(excluded, dict) and excluded
               and all(isinstance(c, str) and c.isdigit() and int(c) in cids and str(int(c)) == c
                       and type(n) is int and n > 0 for c, n in excluded.items()),
               "invalid declaration of excluded requests")
    _check(type(manifest["model_count"]) is int and manifest["model_count"] > 0, "invalid model count")
    _check(isinstance(manifest["counts"], dict) and
           all(type(n) is int and n >= 0 for n in manifest["counts"].values()), "invalid manifest counts")
    present = set(S.corpus_tables(version, tier))
    if tier != "local":
        keys.add("providers")
        providers = manifest.get("providers")
        _check(isinstance(providers, list) and all(isinstance(s, str) and s for s in providers)
               and len(set(providers)) == len(providers), "invalid provider list")
        if "labels" in manifest:
            keys.add("labels")
            _check(manifest["labels"] == "rehydrated", "invalid label declaration")
            present.add("labels")
    expected = [name for name in S.TABLES_BY_TIER["local"] if name in present]
    _check(manifest["tables"] == expected, "table list differs from tier rules")
    _check(manifest.keys() == keys, "unexpected manifest keys")
    return manifest


def _model_csv(directory, manifest):
    """Check the number, order and naming of the model records."""
    models = _csv(directory / "models.csv", S.MODEL_COLUMNS, exact=True)
    _check(len(models) == manifest["model_count"], "models.csv differs from model count")
    originals = []
    for index, row in enumerate(models):
        code = row["model_code"]
        _check(int(row["model_index"]) == index and re.fullmatch(r"M[0-9]{2,}", code),
               "invalid model index or code")
        original = int(code[1:])
        _check(code == f"M{original:02d}", "invalid model code")
        originals.append(original)
        if manifest["names"] == "coded":
            _check(row["model_name"] == code, "uncoded model name")
    _check(originals == sorted(set(originals)), "model codes are not in source order")
    if manifest["schema_version"] == S.PUBLIC_SCHEMA_VERSION:
        listed = [row["model_name"] for row in models]
        _check(all(listed) and len(set(listed)) == len(listed), "listed model names must be unique and present")
    if manifest["tier"] != "open":
        _check(originals == list(range(len(models))), "incomplete model list")
    if manifest["names"] == "coded":
        for name, prefix in (("organisation", "O"), ("base_model", "B")):
            codes = list(dict.fromkeys(r[name] for r in models if r[name]))
            for order, code in enumerate(codes, 1):
                _check(re.fullmatch(prefix + r"[0-9]{2,}", code), f"invalid coded {name}")
                number = int(code[1:]) if manifest["tier"] == "open" else order
                _check(number > 0 and code == f"{prefix}{number:02d}", f"invalid coded {name}")


def _verify_benchmark(directory, b, manifest):
    """Return the actual benchmark counts and providers after validating its tables."""
    cid = int(b["catalog_id"])
    tables = manifest["tables"]
    items = _table(S.table_path(directory, cid, "items"), S.ITEMS)
    fields = _table(S.table_path(directory, cid, "fields"), S.FIELDS)
    splits = _table(S.table_path(directory, cid, "splits"), S.SPLITS)
    _items(items)
    _fields(items, fields)
    _complete(splits)
    ranks = splits["run_rank"].to_pylist()
    _check(sorted(ranks) == list(range(len(items))) and T.splits_table(items, ranks).equals(splits),
           "splits differ from frozen folds or split groups")
    counts = dict(requests=len(items), fields=len(fields))
    if "labels" in tables:
        labels = _table(S.table_path(directory, cid, "labels"), S.LABELS)
        counts["judged_fields"] = _labels(fields, labels)
    for key, n in counts.items():
        if key in b:
            _check(n == int(b[key]), f"{key} differs from benchmarks.csv")
    providers = set()
    for m in range(manifest["model_count"]):
        predictions = _table(S.table_path(directory, cid, "predictions", m), S.PREDICTIONS)
        _predictions(predictions, fields)
        if "usage" in tables:
            usage = _table(S.table_path(directory, cid, "usage", m), S.usage_schema(manifest["schema_version"]))
            providers.update(_usage(usage, items))
        if "latency" in tables:
            _latency(_table(S.table_path(directory, cid, "latency", m), S.LATENCY), items)
    return counts, providers


def verify(corpus):
    """Return a checked manifest after verifying the entire declared rectangle and file inventory."""
    directory = Path(corpus)
    manifest = _manifest(directory)
    sums, checksum_hash = _checksums(directory)
    _check(checksum_hash == manifest["checksums_sha256"], "checksum list differs from manifest")
    contents = set()
    for path in directory.rglob("*"):
        _check(not path.is_symlink(), "symbolic link in processed corpus")
        if not path.is_dir():
            contents.add(path.relative_to(directory).as_posix())
    _check(sums.keys() | {"manifest.json", "SHA256SUMS"} == contents, "unlisted file in corpus")
    _check({"benchmarks.csv", "models.csv"} <= sums.keys(), "missing CSV from checksum list")
    _model_csv(directory, manifest)
    cols = S.BENCHMARK_COLUMNS_LOCAL if manifest["tier"] == "local" else S.BENCHMARK_COLUMNS
    benchmarks = _csv(directory / "benchmarks.csv", cols, exact=True)
    cids = [int(b["catalog_id"]) for b in benchmarks]
    _check(cids == manifest["catalog_ids"], "benchmarks.csv differs from manifest")
    paths = _payload(directory, cids, range(manifest["model_count"]), manifest["tables"])
    _check(sums.keys() == {p.relative_to(directory).as_posix() for p in paths},
           "corpus files do not match the tier and model-by-benchmark rectangle")
    summaries, observed = [], set()
    for b in benchmarks:
        try:
            counts, providers = _verify_benchmark(directory, b, manifest)
        except ValueError as error:
            raise ValueError(f"catalogue {int(b['catalog_id'])}: {error}") from error
        summaries.append(counts)
        observed.update(providers)
    if manifest["tier"] != "local":
        _check(observed == set(manifest["providers"]), "usage strings differ from the manifest provider list")
    _check(manifest["counts"] == _counts(summaries, manifest["model_count"], manifest["tables"]),
           "manifest counts differ from corpus")
    return manifest


def main(argv=None):
    """Run the selected build or verification and return its exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path)
    mode.add_argument("--verify", type=Path)
    for name in ("rows", "meta", "out", "clearance"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--decision", type=Path,
                        help="release decision (JSON); with --schema-version 2 its request exclusions are applied")
    parser.add_argument("--tier", choices=S.TIERS)
    parser.add_argument("--names", choices=("real", "coded"))
    parser.add_argument("--schema-version", choices=("1", "2"), default="1",
                        help="2: the open tier with usage and latency, under listed model names")
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            if any((args.rows, args.meta, args.out, args.clearance, args.tier, args.names, args.decision)):
                parser.error("--verify takes only the processed corpus directory")
            manifest = verify(args.verify)
        else:
            if not all((args.rows, args.meta, args.out, args.tier)):
                parser.error("building requires --rows, --meta, --out and --tier")
            version = S.PUBLIC_SCHEMA_VERSION if args.schema_version == "2" else S.SCHEMA_VERSION
            exclusions = None
            if args.decision is not None:
                exclusions = json.loads(args.decision.read_text(encoding="utf-8")).get("exclude_host_requests")
            manifest = build(args.source, args.rows, args.meta, args.out, args.tier,
                             args.clearance, args.names or "real", version, exclusions)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f"corpus validation failed: {error}\n")
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

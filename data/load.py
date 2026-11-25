"""Loader: one benchmark of a processed corpus as plain arrays (``sim.types.BenchmarkData``).

Normalisation follows the stored results: a choice vector is divided by its ordinary Python sum and then by its
exactly rounded sum; a yes/no vector by the exactly rounded sum only.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from sim.types import BenchmarkData
from . import schema as S


def builtin_sum(x):
    """Totals over the last axis exactly as CPython >= 3.12 computes ``sum(list_of_floats)`` (Neumaier)."""
    s = x[..., 0].copy()
    c = np.zeros_like(s)
    for j in range(1, x.shape[-1]):
        xj = x[..., j]
        t = s + xj
        c += np.where(np.abs(s) >= np.abs(xj), (s - t) + xj, (xj - t) + s)
        s = t
    return np.where((c != 0) & np.isfinite(c), s + c, s)


def fsum_rows(a):
    """Exactly rounded sums over the last axis of a 2-D array (``math.fsum`` per row)."""
    if a.shape[1] == 1:
        return a[:, 0].copy()
    if a.shape[1] == 2:
        return a[:, 0] + a[:, 1]
    return np.fromiter(map(math.fsum, a.tolist()), dtype=np.float64, count=len(a))


def normalize(raw, is_noul):
    """raw [n, M, w], is_noul [n] -> normalised vectors of the same shape."""
    n, m, w = raw.shape
    v = raw.copy()
    choice = ~is_noul
    if choice.any():
        v[choice] = raw[choice] / builtin_sum(raw[choice])[..., None]
    flat = v.reshape(n * m, w)
    return (flat / fsum_rows(flat)[:, None]).reshape(n, m, w)


def read_manifest(corpus):
    return json.loads((Path(corpus) / "manifest.json").read_text(encoding="utf-8"))


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def list_models(corpus):
    """Rows of models.csv in model order, ``model_index`` as int."""
    rows = [dict(r, model_index=int(r["model_index"])) for r in _csv(Path(corpus) / "models.csv")]
    if [r["model_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("models.csv is not in model order")
    return rows


def list_benchmarks(corpus):
    """Rows of benchmarks.csv in catalogue order, ``catalog_id`` as int."""
    return [dict(r, catalog_id=int(r["catalog_id"])) for r in _csv(Path(corpus) / "benchmarks.csv")]


def corpus_rows(corpus, rows_dir, cid):
    """The benchmark rows of the corpus's requests, in the corpus's order (a local step: the rows hold text).

    A corpus from which requests were removed declares their number per benchmark; its rows are selected by request
    id, and the count of rows left over must be the declared one.
    """
    from . import tables as T
    corpus = Path(corpus)
    excluded = read_manifest(corpus).get(S.EXCLUDED_KEY, {}).get(str(int(cid)), 0)
    ids = S.read_table(S.table_path(corpus, cid, "items"), S.ITEMS)["item_id"].to_pylist()
    return T.rows_of(cid, ids, T.read_rows(Path(rows_dir) / f"{int(cid):03d}.jsonl.gz"), excluded)


def load_benchmark(corpus, cid, labels=None, usage=None, latency=None, keep_raw=False):
    """Load one benchmark.

    ``labels``, ``usage``, ``latency``: True = required, False = skip, None = load when the corpus has the table.
    """
    corpus = Path(corpus)
    info = {b["catalog_id"]: b for b in list_benchmarks(corpus)}[int(cid)]
    models = list_models(corpus)
    M = len(models)

    def want(flag, table, model=None):
        present = S.table_path(corpus, cid, table, model).is_file()
        if flag and not present:
            raise FileNotFoundError(f"catalogue {cid}: the corpus has no {table} table")
        return present if flag is None else bool(flag)

    items = S.read_table(S.table_path(corpus, cid, "items"), S.ITEMS)
    R = items.num_rows
    if items["item_index"].to_pylist() != list(range(R)):
        raise ValueError(f"catalogue {cid}: items are not in item order")
    fields = S.read_table(S.table_path(corpus, cid, "fields"), S.FIELDS)
    F = fields.num_rows
    field_row = fields["item_index"].to_numpy().astype(np.int32)
    field_pos = fields["field_index"].to_numpy().astype(np.int32)
    width = fields["n_options"].to_numpy().astype(np.int16)
    is_noul = np.asarray(fields["question_type"].to_pylist()) == "noul"
    if not np.array_equal(np.bincount(field_row, minlength=R), items["field_count"].to_numpy()):
        raise ValueError(f"catalogue {cid}: field counts differ between items and fields")
    if (np.diff(field_row) < 0).any() or (is_noul & (width != 2)).any():
        raise ValueError(f"catalogue {cid}: fields table is malformed")
    splits = S.read_table(S.table_path(corpus, cid, "splits"), S.SPLITS)
    fold = splits["fold"].to_numpy().astype(np.int8)
    if not np.array_equal(fold, items["fold"].to_numpy()):
        raise ValueError(f"catalogue {cid}: folds differ between items and splits")

    widths = sorted(int(w) for w in np.unique(width))
    offsets = np.concatenate([[0], np.cumsum(width.astype(np.int64))])
    idx = {w: np.flatnonzero(width == w).astype(np.int32) for w in widths}
    pos_in_w = np.empty(F, np.int32)
    for w in widths:
        pos_in_w[idx[w]] = np.arange(len(idx[w]), dtype=np.int32)

    values, reported = [], np.full((F, M), -1, np.int32)
    for m in range(M):
        t = S.read_table(S.table_path(corpus, cid, "predictions", m), S.PREDICTIONS)
        col = t["probabilities"].combine_chunks()
        offs = col.offsets.to_numpy()
        if (t.num_rows != F or not np.array_equal(offs, offsets)
                or not np.array_equal(t["item_index"].to_numpy(), field_row)
                or not np.array_equal(t["field_index"].to_numpy(), field_pos)
                or not np.array_equal(np.asarray(t["question_type"].to_pylist()) == "noul", is_noul)):
            raise ValueError(f"catalogue {cid}, model {m}: predictions do not follow the fields table")
        vals = col.values.to_numpy(zero_copy_only=False)
        if not np.isfinite(vals).all() or (vals < 0).any():
            raise ValueError(f"catalogue {cid}, model {m}: invalid probability")
        values.append(vals)
        rc = t["reported_choice"].to_pylist()
        reported[:, m] = [-1 if x is None else x for x in rc]

    P, raw_by_width, top = {}, {}, np.empty((F, M))
    for w in widths:
        take = offsets[idx[w]][:, None] + np.arange(w)[None, :]
        raw = np.stack([v[take] for v in values], axis=1)
        P[w] = normalize(raw, is_noul[idx[w]])
        top[idx[w]] = P[w].max(axis=2)
        if keep_raw:
            raw_by_width[w] = raw

    b = BenchmarkData(
        cid=int(cid), name=info["name"], area=info["area"], R=R, F=F, M=M, fold=fold,
        split_group=splits["split_group"].to_numpy().astype(np.int64),
        run_rank=splits["run_rank"].to_numpy().astype(np.int32),
        field_row=field_row, field_pos=field_pos, width=width, is_noul=is_noul, pos_in_w=pos_in_w,
        widths=widths, idx=idx, P=P, top=top, reported=reported, raw=raw_by_width if keep_raw else None,
        models=models, item_id=items["item_id"].to_pylist())

    if want(labels, "labels"):
        t = S.read_table(S.table_path(corpus, cid, "labels"), S.LABELS)
        if (not np.array_equal(t["item_index"].to_numpy(), field_row)
                or not np.array_equal(t["field_index"].to_numpy(), field_pos)):
            raise ValueError(f"catalogue {cid}: labels do not follow the fields table")
        accepted = t["accepted"].to_pylist()
        judged = np.asarray(t["judged"].to_pylist(), bool)
        if [bool(a) for a in accepted] != judged.tolist():
            raise ValueError(f"catalogue {cid}: judged flag and accepted list disagree")
        b.judged = judged
        b.n_accepted = np.asarray([len(a) for a in accepted], np.int16)
        b.gold = np.asarray([a[0] if len(a) == 1 else -1 for a in accepted], np.int16)
        b.acc = {}
        for w in widths:
            mask = np.zeros((len(idx[w]), w), bool)
            for k, f in enumerate(idx[w]):
                mask[k, accepted[f]] = True
            b.acc[w] = mask

    if want(usage, "usage", 0):
        b.calls = np.empty((R, M), np.int32)
        b.tin = np.empty((R, M), np.int64)
        b.tout = np.empty((R, M), np.int64)
        b.cost = np.empty((R, M), np.float64)
        usage_schema = S.usage_schema(read_manifest(corpus)["schema_version"])
        for m in range(M):
            t = S.read_table(S.table_path(corpus, cid, "usage", m), usage_schema)
            if t["item_index"].to_pylist() != list(range(R)):
                raise ValueError(f"catalogue {cid}, model {m}: usage is not one row per request")
            b.calls[:, m] = t["calls"].to_numpy()
            b.tin[:, m] = t["input_tokens"].to_numpy()
            b.tout[:, m] = t["output_tokens"].to_numpy()
            b.cost[:, m] = t["cost_usd"].to_numpy()
    if want(latency, "latency", 0):
        b.lat = np.empty((R, M), np.float64)
        for m in range(M):
            t = S.read_table(S.table_path(corpus, cid, "latency", m), S.LATENCY)
            if t["item_index"].to_pylist() != list(range(R)):
                raise ValueError(f"catalogue {cid}, model {m}: latency is not one row per request")
            b.lat[:, m] = t["latency_s"].to_numpy()
    return b

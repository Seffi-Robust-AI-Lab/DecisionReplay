"""Derivation of the processed tables from the source export and the local benchmark rows.

These functions are the definition of the tables in ``schema``.  They do no file management: the corpus builder
calls them and decides what is written where.  Benchmark rows (text, options, gold) are read here and nowhere
else; only positions, counts and codes leave these functions.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pyarrow as pa

from . import schema as S

# Source scoring kinds whose accepted answers are listed in the row instead of being the single expected answer.
ACCEPTED_LIST_SCORING = ("chess_reference_value", "human_group_rank")
# Catalogue 56 judges one field per request; its other fields carry predictions only.
ONLY_FIELD_JUDGED = {56: "verdict"}


def read_rows(path):
    """Benchmark rows of one catalogue, in file order (= ``item_index``)."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def field_layout(cid, row):
    """For every field of a row, in order: (question type, number of options, accepted positions).

    An empty accepted tuple means that the field is not judged.  The option order is the order of the row's
    ``criteria`` mapping; a yes/no field has the options (no, yes).
    """
    scoring = row.get("scoring") or {}
    kind = scoring.get("type")
    if kind == "routing_outcome":
        raise ValueError("routing_outcome scoring is not expected in this suite")
    unscored = set(scoring.get("unscored_fields", []))
    only = ONLY_FIELD_JUDGED.get(int(cid))
    out = []
    for key, q in row["questions"].items():
        if q["type"] == "choice":
            options = list(q["criteria"])
        elif q["type"] == "noul":
            options = ["no", "yes"]
        else:
            raise ValueError(f"unsupported field type {q['type']!r}")
        accepted = ()
        if key not in unscored and not (only is not None and key != only):
            gold = row["expected"][key]
            if kind in ACCEPTED_LIST_SCORING:
                names = list(scoring["accepted"])
            elif gold is None:
                names = None
            elif q["type"] == "noul":
                if not isinstance(gold, bool):
                    raise ValueError("non-boolean gold on a yes/no field")
                names = ["yes" if gold else "no"]
            else:
                names = [gold]
            if names is not None:
                accepted = tuple(i for i, o in enumerate(options) if o in names)
                if not accepted or not set(names) <= set(options):
                    raise ValueError("gold answer is not among the options")
        out.append((q["type"], len(options), accepted))
    return out


def row_positions(cid, item_ids, rows, excluded=None):
    """Positions, among the benchmark rows, of the requests ``item_ids``, in their order.

    The rows of a benchmark hold every request of the source; a corpus may hold fewer (requests removed from a
    release).  A row belongs to a request when the request id derived from the row equals the corpus's id.  Raises
    unless every request has exactly one row, the requests keep the order of the rows, and, when ``excluded`` is
    given, exactly that many rows are left over (0: the rows are the requests).
    """
    item_ids = list(item_ids)
    ids = [S.public_id("item", cid, row["_evaluation"]["run_id"]) for row in rows]
    if ids == item_ids:
        chosen = list(range(len(rows)))
    else:
        position = {request: i for i, request in enumerate(ids)}
        if len(position) != len(ids) or len(set(item_ids)) != len(item_ids):
            raise ValueError(f"catalogue {cid}: request ids are not unique")
        if not set(item_ids) <= position.keys():
            raise ValueError(f"catalogue {cid}: a request of the corpus has no benchmark row")
        chosen = [position[request] for request in item_ids]
        if chosen != sorted(chosen):
            raise ValueError(f"catalogue {cid}: the requests are not in the order of the benchmark rows")
    if excluded is not None and len(rows) - len(chosen) != excluded:
        raise ValueError(f"catalogue {cid}: the benchmark rows hold {len(rows) - len(chosen)} requests that the "
                         f"corpus does not have; {excluded} expected")
    return chosen


def rows_of(cid, item_ids, rows, excluded=None):
    """The benchmark rows of the requests ``item_ids``, in their order (see ``row_positions``)."""
    return [rows[i] for i in row_positions(cid, item_ids, rows, excluded)]


def project(table, new_index):
    """Keep the rows of the requests with ``new_index[item_index] >= 0`` and renumber their ``item_index``."""
    position = table.schema.get_field_index("item_index")
    index = np.asarray(new_index)[table["item_index"].to_numpy()]
    kept = index >= 0
    out = table.filter(pa.array(kept))
    return out.set_column(position, table.schema.field(position), pa.array(index[kept], type=table.schema.field(position).type))


def check_binding(cid, items, rows):
    """Raise unless the rows are the requests of the items table, in the same order."""
    ids = items["item_id"].to_pylist()
    groups = items["native_group_id"].to_pylist()
    counts = items["field_count"].to_pylist()
    if len(rows) != len(ids) or items["item_index"].to_pylist() != list(range(len(ids))):
        raise ValueError(f"catalogue {cid}: items table and benchmark rows disagree")
    for i, row in enumerate(rows):
        ev = row["_evaluation"]
        if int(ev["catalog_id"]) != int(cid):
            raise ValueError(f"catalogue {cid}: row {i} belongs to another benchmark")
        if ids[i] != S.public_id("item", cid, ev["run_id"]):
            raise ValueError(f"catalogue {cid}: row {i} is not item {i}")
        if groups[i] != S.public_id("native_group", cid, ev["group_id"]):
            raise ValueError(f"catalogue {cid}: row {i} has another scoring group than the export")
        if len(row["questions"]) != counts[i]:
            raise ValueError(f"catalogue {cid}: row {i} has another field count than the export")


def fields_and_labels(cid, rows):
    """(FIELDS table, LABELS table) of one catalogue."""
    item, field, qtype, width, judged, accepted = [], [], [], [], [], []
    for i, row in enumerate(rows):
        for j, (kind, n_options, acc) in enumerate(field_layout(cid, row)):
            item.append(i)
            field.append(j)
            qtype.append(kind)
            width.append(n_options)
            judged.append(bool(acc))
            accepted.append(list(acc))
    fields = pa.table({"item_index": item, "field_index": field, "question_type": qtype, "n_options": width},
                      schema=S.FIELDS)
    labels = pa.table({"item_index": item, "field_index": field, "judged": judged, "accepted": accepted},
                      schema=S.LABELS)
    return fields, labels


def splits_table(items, run_ids):
    """SPLITS table: fold, run rank and split-group code of every request.

    Requests are ranked by run id (plain string order).  Split groups are numbered by first appearance in that
    order, which is the numbering the stored bootstrap draws were made with.
    """
    n = len(run_ids)
    if len(set(run_ids)) != n or items.num_rows != n:
        raise ValueError("run ids must be unique and match the items table")
    order = sorted(range(n), key=run_ids.__getitem__)
    rank = np.empty(n, np.int32)
    rank[order] = np.arange(n, dtype=np.int32)
    group_ids = items["split_group_id"].to_pylist()
    code, group = {}, np.empty(n, np.int32)
    for i in order:
        group[i] = code.setdefault(group_ids[i], len(code))
    fold = items["fold"].to_numpy()
    low, high = np.full(len(code), S.N_FOLDS), np.full(len(code), -1)
    np.minimum.at(low, group, fold)
    np.maximum.at(high, group, fold)
    if (low != high).any():
        raise ValueError("a split group lies in two folds")
    return pa.table({"item_index": items["item_index"], "fold": items["fold"], "split_group": group,
                     "run_rank": rank}, schema=S.SPLITS)


def usage_and_latency(observations):
    """(USAGE table, LATENCY table) from one source ``observations`` table."""
    names = {"provider": "provider", "transport": "transport", "calls": "leaf_physical_calls",
             "attempts": "realized_physical_calls", "input_tokens": "leaf_input_tokens",
             "output_tokens": "leaf_output_tokens", "cost_usd": "leaf_cost_usd"}
    for column in ("leaf_physical_calls", "realized_physical_calls", "leaf_input_tokens", "leaf_output_tokens",
                   "leaf_cost_usd", "leaf_deployment_latency_s", "transport"):
        if observations[column].null_count:
            raise ValueError(f"source usage column {column} has missing values")
    usage = pa.table({"item_index": observations["item_index"],
                      **{new: observations[old] for new, old in names.items()}}, schema=S.USAGE)
    latency = pa.table({"item_index": observations["item_index"],
                        "latency_s": observations["leaf_deployment_latency_s"]}, schema=S.LATENCY)
    return usage, latency

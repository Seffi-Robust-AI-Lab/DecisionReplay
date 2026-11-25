"""Release matrix of the open tier: the review of provider and source terms plus the authors' release decision.

    python3 -B -m analysis.release --review DIR --decision FILE --out DIR [--check]

``--review`` is the folder of the review of provider and benchmark-source terms (read only).  It holds
``clearance.csv`` (one row per benchmark and model), ``model_output_rights.csv`` (one row per model) and
``serving_providers.csv`` (records per benchmark, model and serving host).  ``--decision`` is
``data/meta/release_decision.json``: the models that the authors hold back, the rule and the date, and
the requests that are removed for every model (``exclude_host_requests``: a released model, a serving host and the
number of requests that the host served for it).
Three files are written to ``--out`` (``data/meta``):

``release_clearance.csv``  the review's cell matrix with its rows, columns and values unchanged and two columns
                           added, ``release`` (released or held) and ``release_basis``; the corpus builder selects
                           the open tier from ``release``;
``release_notices.csv``    for each released model, the review's findings and conditions, to travel with the files;
``release_summary.json``   the counts that the paper quotes.

A cell is released when its model is not held and the review found no source clause that bars the anonymous
positional results.  The review's own status of a model (cleared, conditional, unverified) is kept as it was:
releasing a conditional or unverified model is the authors' decision, not a finding of the review.  The released
cells must form a rectangle (every released model on every released benchmark), as the open tier requires.
With ``--check`` nothing is written and the exit status says whether the files in ``--out`` are up to date.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_OK = "no_source_output_bar_identified"
NOTICE_COLUMNS = ("route_id", "canonical_model_id", "serving_providers", "review_date", "model_vector_status",
                  "output_use_finding", "public_benchmark_scope", "release_conditions_or_missing_authority",
                  "terms_urls")
EXCLUSION_COLUMN = "request_exclusion"        # added to the notices when the decision removes requests
FILES = ("release_clearance.csv", "release_notices.csv", "release_summary.json")


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _csv(columns, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def derive(review, decision):
    """Return {file name: text} for the three files, from the review folder and the decision document."""
    review = Path(review)
    columns, cells = _read(review / "clearance.csv")
    if {"release", "release_basis"} & set(columns):
        raise ValueError("the review's clearance already has a release column")
    term_columns, terms = _read(review / "model_output_rights.csv")
    _, serving = _read(review / "serving_providers.csv")
    if not set(NOTICE_COLUMNS) <= set(term_columns):
        raise ValueError("the model terms lack a column of the notices")
    held = dict(decision["held_models"])
    routes = sorted({c["route_id"] for c in cells})
    benchmarks = sorted({int(c["catalog_id"]) for c in cells})
    if not set(held) <= set(routes):
        raise ValueError("the decision holds a model that the review does not list")
    if len({(c["catalog_id"], c["route_id"]) for c in cells}) != len(cells) or len(cells) != len(routes) * len(benchmarks):
        raise ValueError("the review's clearance is not one row per benchmark and model")
    status = {t["route_id"]: t["model_vector_status"] for t in terms}
    if sorted(status) != routes or len(terms) != len(routes):
        raise ValueError("the model terms and the clearance list different models")
    if any(c["model_vector_status"] != status[c["route_id"]] for c in cells):
        raise ValueError("a cell's model status differs from the model terms")

    marked = []
    for c in cells:
        route = c["route_id"]
        if route in held:
            release, basis = "held", f"model held by the authors' decision of {decision['decided']}"
        elif c["benchmark_vector_status"] != SOURCE_OK:
            release, basis = "held", "benchmark source terms: " + c["benchmark_vector_status"]
        else:
            release = "released"
            basis = f"the authors' decision of {decision['decided']}; review status of the model: {status[route]}"
        marked.append(dict(c, release=release, release_basis=basis))
    released = [c for c in marked if c["release"] == "released"]
    models = sorted({c["route_id"] for c in released})
    cids = sorted({int(c["catalog_id"]) for c in released})
    if not released or len(released) != len(models) * len(cids):
        raise ValueError("the released cells do not form a rectangle of models and benchmarks")
    if sorted(set(routes) - set(models)) != sorted(held):
        raise ValueError("a model that the decision does not hold has no released cell")

    # serving hosts of the released cells; the records of each cell must add up to its requests
    requests = {(c["catalog_id"], c["route_id"]): int(c["original_requests"]) for c in cells}
    by_cell = defaultdict(Counter)
    for s in serving:
        by_cell[(s["catalog_id"], s["route_id"])][s["serving_provider"]] += int(s["original_requests"])
    for cell, count in requests.items():
        if sum(by_cell[cell].values()) != count:
            raise ValueError(f"serving records of cell {cell} do not add up to its requests")
    hosts = defaultdict(Counter)
    for c in released:
        hosts[c["route_id"]].update(by_cell[(c["catalog_id"], c["route_id"])])
    secondary, secondary_cells = {}, set()
    for route, count in sorted(hosts.items()):
        main = max(sorted(count), key=lambda h: count[h])
        others = {h: n for h, n in sorted(count.items()) if h != main}
        if others:
            secondary[route] = dict(main=main, records=dict(sorted(count.items())))
            secondary_cells |= {(c["catalog_id"], route) for c in released if c["route_id"] == route
                                and set(by_cell[(c["catalog_id"], route)]) - {main}}
    clause_hosts = set(decision.get("hosts_with_benchmark_clause", {}))
    clause_records = sum(n for c in released for h, n in by_cell[(c["catalog_id"], c["route_id"])].items()
                         if h in clause_hosts)
    clause_models = sorted({c["route_id"] for c in released
                            if set(by_cell[(c["catalog_id"], c["route_id"])]) & clause_hosts})

    # Requests removed for every model: those that a named host served for a named released model.  The number in
    # the decision must be the number of the review's serving records of the released cells.
    excluded, excluded_hosts, exclusion_notice = Counter(), {}, {}
    for rule in decision.get("exclude_host_requests", []):
        route, served_by = rule["model"], rule["serving_host"]
        if route not in models or (route, served_by) in excluded_hosts:
            raise ValueError("a request exclusion names a model that is not released, or names a pair twice")
        per_benchmark = {int(c["catalog_id"]): by_cell[(c["catalog_id"], route)].get(served_by, 0)
                         for c in released if c["route_id"] == route}
        if sum(per_benchmark.values()) != rule["expected_requests"] or not rule["expected_requests"]:
            raise ValueError(f"the review records {sum(per_benchmark.values())} requests of {route} served by {served_by}, "
                             f"not {rule['expected_requests']}")
        excluded.update({cid: n for cid, n in per_benchmark.items() if n})
        excluded_hosts[(route, served_by)] = rule["expected_requests"]
        exclusion_notice[route] = (exclusion_notice.get(route, "") + f"The {rule['expected_requests']} requests that "
                                   f"{served_by} served for this model were removed from the corpus for every model. ").strip()
    per_benchmark_requests = {int(c["catalog_id"]): int(c["original_requests"]) for c in released}
    if any(len({int(c["original_requests"]) for c in released if int(c["catalog_id"]) == cid}) != 1 for cid in cids):
        raise ValueError("the released models differ in the requests of a benchmark")
    requests_released = sum(per_benchmark_requests.values()) - sum(excluded.values())
    remaining, remaining_cells = {}, set()
    for route, entry in secondary.items():
        others = {h: n for h, n in entry["records"].items() if h != entry["main"] and (route, h) not in excluded_hosts}
        if others:
            remaining[route] = others
            remaining_cells |= {(c["catalog_id"], route) for c in released if c["route_id"] == route
                                and any(h in others for h in by_cell[(c["catalog_id"], route)])}
    clause_remaining = clause_records - sum(n for (route, host), n in excluded_hosts.items() if host in clause_hosts)

    review_counts = {kind: dict(sorted(Counter(status[r] for r in group).items()))
                     for kind, group in (("released", models), ("held", sorted(held)))}
    summary = dict(
        decided=decision["decided"], decided_by=decision["decided_by"], statement=decision["statement"],
        review_date=decision["review"]["date"],
        models_total=len(routes), models_released=len(models), models_held=len(held),
        benchmarks_total=len(benchmarks), benchmarks_released=len(cids),
        benchmarks_held=sorted(set(benchmarks) - set(cids)), released_benchmarks=cids,
        cells_total=len(cells), cells_released=len(released),
        vectors_total=sum(int(c["model_field_vectors"]) for c in cells),
        vectors_released=sum(int(c["model_field_vectors"]) for c in released),
        observations_released=sum(int(c["original_requests"]) for c in released),
        review_cleared_models=sum(1 for r in routes if status[r] == "cleared"),
        review_status_counts=review_counts,
        released_models=models, held_models=sorted(held),
        model_status={r: dict(review=status[r], release="held" if r in held else "released") for r in routes},
        secondary_hosts=secondary,
        secondary_host_models=len(secondary), secondary_host_cells=len(secondary_cells),
        secondary_host_records=sum(n for v in secondary.values() for h, n in v["records"].items() if h != v["main"]),
        benchmark_clause_hosts=sorted(clause_hosts), benchmark_clause_host_records=clause_records,
        benchmark_clause_host_models=len(clause_models),
    )
    # "observations_released" and "vectors_released" count the released cells as the review lists them, before any
    # request is removed; the keys below give what the corpus holds after the removal.
    summary.update(
        requests_released=requests_released, observations_in_corpus=requests_released * len(models),
        excluded_requests=sum(excluded.values()), excluded_request_benchmarks=len(excluded),
        excluded_requests_by_benchmark={str(cid): n for cid, n in sorted(excluded.items())},
        excluded_request_rules=len(excluded_hosts),
        secondary_hosts_in_corpus={route: dict(sorted(others.items())) for route, others in sorted(remaining.items())},
        secondary_host_models_in_corpus=len(remaining), secondary_host_cells_in_corpus=len(remaining_cells),
        secondary_host_records_in_corpus=sum(n for others in remaining.values() for n in others.values()),
        secondary_host_count_in_corpus=len({h for others in remaining.values() for h in others}),
        benchmark_clause_host_records_in_corpus=clause_remaining,
    )
    if "requests" in decision:
        summary["requests_decided"] = decision["requests"]["decided"]
    notices = [t for t in sorted(terms, key=lambda t: t["route_id"]) if t["route_id"] in set(models)]
    notice_columns = NOTICE_COLUMNS
    if excluded_hosts:
        notice_columns = NOTICE_COLUMNS + (EXCLUSION_COLUMN,)
        notices = [dict(t, **{EXCLUSION_COLUMN: exclusion_notice.get(t["route_id"], "")}) for t in notices]
    return {
        "release_clearance.csv": _csv(columns + ["release", "release_basis"], marked),
        "release_notices.csv": _csv(notice_columns, notices),
        "release_summary.json": json.dumps(summary, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Release matrix of the open tier from the review and the decision")
    ap.add_argument("--review", required=True, type=Path)
    ap.add_argument("--decision", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    documents = derive(a.review, json.loads(a.decision.read_text(encoding="utf-8")))
    if a.check:
        stale = [n for n in FILES if not (a.out / n).is_file() or (a.out / n).read_text(encoding="utf-8") != documents[n]]
        print("release files up to date" if not stale else "stale: " + ", ".join(stale))
        return 1 if stale else 0
    for name in FILES:
        (a.out / name).write_text(documents[name], encoding="utf-8")
    summary = json.loads(documents["release_summary.json"])
    print(f"released {summary['models_released']} of {summary['models_total']} models on "
          f"{summary['benchmarks_released']} benchmarks: {summary['cells_released']} of {summary['cells_total']} cells, "
          f"{summary['vectors_released']:,} vectors")
    return 0


if __name__ == "__main__":
    sys.exit(main())

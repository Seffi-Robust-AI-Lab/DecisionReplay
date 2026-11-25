"""Release matrix of the open tier on a made-up review: held models, held benchmarks, hosts, determinism."""

import csv
import json

import pytest

from analysis import release
from data import tiers

REVIEW_COLUMNS = ["catalog_id", "dataset", "route_id", "canonical_model_id", "numeric_public_status",
                  "public_vector_allowed", "model_vector_status", "benchmark_vector_status",
                  "observed_serving_providers", "original_requests", "model_field_vectors", "per_cell_notices"]
STATUS = {"org/a": "cleared", "org/b": "conditional", "org/c": "unverified"}
DECISION = dict(decided="2026-11-21", decided_by="the authors", statement="made up", held_models={"org/c": "C"},
                hosts_with_benchmark_clause={"HostB": "made up"}, review=dict(date="2026-11-19"))


def _write(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _review(folder):
    """Three models on three benchmarks; the third benchmark's source terms hold it; org/b used a second host once."""
    cells, serving = [], []
    for cid, source in ((1, release.SOURCE_OK), (2, release.SOURCE_OK), (3, "hold_specific_terms")):
        for route, status in STATUS.items():
            cells.append(dict(catalog_id=cid, dataset=f"bench-{cid}", route_id=route, canonical_model_id=route,
                              numeric_public_status=status if source == release.SOURCE_OK else "conditional",
                              public_vector_allowed="false", model_vector_status=status,
                              benchmark_vector_status=source, observed_serving_providers="HostA",
                              original_requests=10, model_field_vectors=20, per_cell_notices="notice"))
            split = (7, 3) if (route, cid) == ("org/b", 2) else (10, 0)
            for host, n in zip(("HostA", "HostB"), split):
                if n:
                    serving.append(dict(catalog_id=cid, route_id=route, serving_provider=host,
                                        original_requests=n, successful_physical_calls=n))
    _write(folder / "clearance.csv", REVIEW_COLUMNS, cells)
    terms = [dict(route_id=r, canonical_model_id=r, serving_providers="HostA", review_date="2026-11-19",
                  model_vector_status=s, output_use_finding="finding", public_benchmark_scope="scope",
                  release_conditions_or_missing_authority="conditions", terms_urls="https://example.org/terms")
             for r, s in STATUS.items()]
    _write(folder / "model_output_rights.csv", list(terms[0]), terms)
    _write(folder / "serving_providers.csv", list(serving[0]), serving)


def test_release_matrix(tmp_path):
    _review(tmp_path)
    documents = release.derive(tmp_path, DECISION)
    assert documents == release.derive(tmp_path, DECISION)
    rows = list(csv.DictReader(documents["release_clearance.csv"].splitlines()))
    with open(tmp_path / "clearance.csv", newline="", encoding="utf-8") as fh:
        review = list(csv.DictReader(fh))
    assert [{k: r[k] for k in review[0]} for r in rows] == review          # the review's columns and values kept
    released = {(int(r["catalog_id"]), r["route_id"]) for r in rows if r["release"] == "released"}
    assert released == {(c, r) for c in (1, 2) for r in ("org/a", "org/b")}
    summary = json.loads(documents["release_summary.json"])
    assert (summary["models_released"], summary["models_held"], summary["benchmarks_released"],
            summary["cells_released"], summary["vectors_released"]) == (2, 1, 2, 4, 80)
    assert summary["review_status_counts"] == {"held": {"unverified": 1}, "released": {"cleared": 1, "conditional": 1}}
    assert summary["secondary_host_records"] == 3 and summary["benchmark_clause_host_records"] == 3
    assert [n["route_id"] for n in csv.DictReader(documents["release_notices.csv"].splitlines())] == ["org/a", "org/b"]
    path = tmp_path / "release_clearance.csv"                             # the builder reads the release column
    path.write_text(documents["release_clearance.csv"], encoding="utf-8")
    assert tiers._cleared_cells(path, {1, 2, 3}, set(STATUS))[0] == released
    assert tiers._cleared_cells(tmp_path / "clearance.csv", {1, 2, 3}, set(STATUS))[0] == {(1, "org/a"), (2, "org/a")}


def test_release_rejects_unknown_model(tmp_path):
    _review(tmp_path)
    with pytest.raises(ValueError):
        release.derive(tmp_path, dict(DECISION, held_models={"org/x": "X"}))

"""The open tier of schema version 2: probability vectors with usage and latency, without the serving host."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq
import pytest

from analysis import lint_release
from data import build_corpus, load, schema
from data.rehydrate import rehydrate

SOURCE = Path("results/decision_index/paper/dataset/candidate")
ROWS = Path("data/decision_index/common-paid-65/inputs")


@pytest.fixture(scope="module")
def workspace():
    with TemporaryDirectory(dir=".") as folder:
        yield Path(folder)


@pytest.fixture(scope="module")
def public(test_root, workspace):
    """The version-2 open tier of the made-up corpus."""
    out = workspace / "open"
    build_corpus.build(test_root / SOURCE, test_root / ROWS, test_root / "meta", out, tier="open",
                       clearance=test_root / "meta/clearance.csv", names="real",
                       version=schema.PUBLIC_SCHEMA_VERSION)
    return out


def test_manifest_and_tables(public, planted):
    manifest = build_corpus.verify(public)
    assert manifest["schema_version"] == schema.PUBLIC_SCHEMA_VERSION
    assert manifest["tier"] == "open" and manifest["names"] == "real" and manifest["providers"] == []
    assert manifest["tables"] == list(schema.PUBLIC_TABLES)
    assert manifest["model_count"] == len(planted["cleared_models"])
    assert manifest["counts"]["observations"] == manifest["counts"]["requests"] * manifest["model_count"]
    assert "judged_fields" not in manifest["counts"]
    assert not list(public.rglob("labels.parquet"))


def test_usage_has_request_totals_and_no_host(public, corpus, planted):
    """Usage and latency equal the local tier's values; host, transport and attempts are not there."""
    manifest = load.read_manifest(public)
    for cid in manifest["catalog_ids"]:
        local = load.load_benchmark(corpus, cid, labels=False)
        mine = load.load_benchmark(public, cid)
        assert not mine.has_labels and mine.calls is not None and mine.lat is not None
        for new, old in enumerate(planted["cleared_models"]):
            names = pq.ParquetFile(schema.table_path(public, cid, "usage", new)).schema_arrow.names
            assert names == schema.PUBLIC_USAGE.names
            assert not {"provider", "transport", "attempts"} & set(names)
            for mine_array, local_array in ((mine.calls, local.calls), (mine.tin, local.tin), (mine.tout, local.tout),
                                            (mine.cost, local.cost), (mine.lat, local.lat)):
                assert (mine_array[:, new] == local_array[:, old]).all()
            assert (schema.table_path(public, cid, "predictions", new).read_bytes()
                    == schema.table_path(corpus, cid, "predictions", old).read_bytes())


def test_release_lint_accepts_version_two(public):
    errors, _ = lint_release.check_release(public)
    assert errors == []


def test_full_usage_table_is_refused(public, corpus, workspace):
    """A usage table with the serving host does not pass as version 2, in the verifier or in the release lint."""
    out = shutil.copytree(public, workspace / "tampered")
    cid = load.read_manifest(out)["catalog_ids"][0]
    shutil.copyfile(schema.table_path(corpus, cid, "usage", 0), schema.table_path(out, cid, "usage", 0))
    errors, _ = lint_release.check_release(out)
    assert any("usage columns differ" in e or "forbidden" in e for e in errors)
    with pytest.raises(ValueError):
        build_corpus.verify(out)


def test_version_two_is_the_open_tier_with_listed_names(test_root, workspace):
    for tier, names in (("gated", "real"), ("local", "real"), ("open", "coded")):
        with pytest.raises(ValueError):
            build_corpus.build(test_root / SOURCE, test_root / ROWS, test_root / "meta", workspace / f"no-{tier}-{names}",
                               tier=tier, clearance=test_root / "meta/clearance.csv", names=names,
                               version=schema.PUBLIC_SCHEMA_VERSION)


def test_rehydrated_copy_runs_like_the_local_tier(public, corpus, planted, test_root, workspace):
    """With rebuilt labels the version-2 open tier holds what a policy needs: labels, cost and latency."""
    out = workspace / "labelled"
    rehydrate(public, test_root / ROWS, out)
    manifest = build_corpus.verify(out)
    assert manifest["labels"] == "rehydrated" and "labels" in manifest["tables"]
    for cid in manifest["catalog_ids"]:
        local = load.load_benchmark(corpus, cid)
        mine = load.load_benchmark(out, cid)
        assert mine.has_labels and (mine.judged == local.judged).all() and (mine.gold == local.gold).all()
        for new, old in enumerate(planted["cleared_models"]):
            assert (mine.cost[:, new] == local.cost[:, old]).all() and (mine.lat[:, new] == local.lat[:, old]).all()


# ---- requests removed for every model (the release decision's ``exclude_host_requests``)

RULE_MODEL, RULE_HOST = 1, "host-other"          # the made-up corpus serves a few requests of model 1 through it


def served_by_host(corpus, cid):
    """[requests] bool: the local tier's requests of ``cid`` that the rule's host served for the rule's model."""
    usage = pq.read_table(schema.table_path(corpus, cid, "usage", RULE_MODEL), columns=["provider"])
    return np.asarray([host == RULE_HOST for host in usage["provider"].to_pylist()])


@pytest.fixture(scope="module")
def rule(public, corpus):
    name = load.list_models(corpus)[RULE_MODEL]["model_name"]
    total = sum(int(served_by_host(corpus, cid).sum()) for cid in load.read_manifest(public)["catalog_ids"])
    assert total > 0
    return dict(model=name, serving_host=RULE_HOST, expected_requests=total)


@pytest.fixture(scope="module")
def trimmed(test_root, workspace, rule):
    out = workspace / "trimmed"
    build_corpus.build(test_root / SOURCE, test_root / ROWS, test_root / "meta", out, tier="open",
                       clearance=test_root / "meta/clearance.csv", names="real",
                       version=schema.PUBLIC_SCHEMA_VERSION, exclusions=[rule])
    return out


def test_excluded_requests_are_gone_for_every_model(trimmed, public, corpus, planted, rule):
    manifest = build_corpus.verify(trimmed)
    full = load.read_manifest(public)
    declared = manifest[schema.EXCLUDED_KEY]
    assert sum(declared.values()) == rule["expected_requests"]
    assert manifest["counts"]["requests"] == full["counts"]["requests"] - rule["expected_requests"]
    assert manifest["counts"]["observations"] == manifest["counts"]["requests"] * manifest["model_count"]
    errors, _ = lint_release.check_release(trimmed)
    assert errors == []
    for cid in manifest["catalog_ids"]:
        gone = served_by_host(corpus, cid)
        assert declared.get(str(cid), 0) == int(gone.sum())
        local = load.load_benchmark(corpus, cid, labels=False)
        mine = load.load_benchmark(trimmed, cid)
        assert mine.R == local.R - int(gone.sum())
        assert mine.item_id == [request for request, out in zip(local.item_id, gone) if not out]
        assert (mine.fold == local.fold[~gone]).all()
        keep_field = ~gone[local.field_row]
        assert (mine.width == local.width[keep_field]).all()
        for new, old in enumerate(planted["cleared_models"]):
            assert (mine.cost[:, new] == local.cost[~gone, old]).all()
            assert (mine.lat[:, new] == local.lat[~gone, old]).all()
            assert (mine.calls[:, new] == local.calls[~gone, old]).all()
            assert (mine.top[:, new] == local.top[keep_field, old]).all()
            assert (mine.reported[:, new] == local.reported[keep_field, old]).all()
        # source groups keep their requests together: two kept requests share a group as they did before
        kept = np.flatnonzero(~gone)
        pairs = {}
        for position, request in enumerate(kept):
            pairs.setdefault(int(local.split_group[request]), set()).add(int(mine.split_group[position]))
        assert all(len(codes) == 1 for codes in pairs.values())
        assert len({next(iter(codes)) for codes in pairs.values()}) == len(pairs)


def test_untouched_benchmarks_are_byte_identical(trimmed, public):
    declared = load.read_manifest(trimmed)[schema.EXCLUDED_KEY]
    for cid in load.read_manifest(public)["catalog_ids"]:
        if str(cid) in declared:
            continue
        for path in sorted(schema.catalog_dir(public, cid).rglob("*.parquet")):
            relative = path.relative_to(public)
            assert (trimmed / relative).read_bytes() == path.read_bytes()


def test_wrong_exclusion_count_is_refused(test_root, workspace, rule):
    for wrong in (rule["expected_requests"] + 1, rule["expected_requests"] - 1):
        with pytest.raises(ValueError):
            build_corpus.build(test_root / SOURCE, test_root / ROWS, test_root / "meta", workspace / f"wrong-{wrong}",
                               tier="open", clearance=test_root / "meta/clearance.csv", names="real",
                               version=schema.PUBLIC_SCHEMA_VERSION, exclusions=[dict(rule, expected_requests=wrong)])
    with pytest.raises(ValueError):                 # version 1 has no request exclusions
        build_corpus.build(test_root / SOURCE, test_root / ROWS, test_root / "meta", workspace / "wrong-version",
                           tier="open", clearance=test_root / "meta/clearance.csv", names="coded", exclusions=[rule])


def test_labels_are_rebuilt_from_the_full_benchmark_rows(trimmed, corpus, test_root, workspace):
    """A user holds every row of a benchmark; the label rebuild binds them to the corpus by request id."""
    out = workspace / "trimmed-labelled"
    rehydrate(trimmed, test_root / ROWS, out)
    manifest = build_corpus.verify(out)
    assert manifest["labels"] == "rehydrated" and manifest[schema.EXCLUDED_KEY] == load.read_manifest(trimmed)[schema.EXCLUDED_KEY]
    for cid in manifest["catalog_ids"]:
        gone = served_by_host(corpus, cid)
        local = load.load_benchmark(corpus, cid)
        mine = load.load_benchmark(out, cid)
        keep_field = ~gone[local.field_row]
        assert (mine.judged == local.judged[keep_field]).all() and (mine.gold == local.gold[keep_field]).all()
        rows = load.corpus_rows(out, test_root / ROWS, cid)
        assert len(rows) == mine.R and sum(len(row["questions"]) for row in rows) == mine.F


def test_a_wrong_declaration_of_excluded_requests_stops_the_label_rebuild(trimmed, test_root, workspace):
    import json
    out = shutil.copytree(trimmed, workspace / "trimmed-tampered")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    cid = next(iter(manifest[schema.EXCLUDED_KEY]))
    manifest[schema.EXCLUDED_KEY][cid] += 1
    build_corpus.write_manifest(out, {k: v for k, v in manifest.items() if k != "checksums_sha256"})
    with pytest.raises(ValueError):
        rehydrate(out, test_root / ROWS, workspace / "trimmed-tampered-labelled")


def test_request_features_follow_the_corpus(trimmed, corpus, test_root, workspace):
    """Features built on every request are copied, and refitted on the kept requests where requests were removed."""
    from analysis import features_for_corpus, featurize_requests
    from analysis.embed_requests import request_text
    declared = load.read_manifest(trimmed)[schema.EXCLUDED_KEY]
    cids = sorted(int(c) for c in declared)[:2] + [c for c in load.read_manifest(trimmed)["catalog_ids"]
                                                   if str(c) not in declared][:1]
    full = workspace / "features" / "tfidf"
    featurize_requests.run("tfidf", test_root / ROWS, corpus, full, cids=cids)
    # the features exist for three benchmarks only; the tool walks the corpus's benchmark list, so give it those
    listed = load.list_benchmarks
    try:
        features_for_corpus.list_benchmarks = lambda c: [b for b in listed(c) if b["catalog_id"] in cids]
        report = features_for_corpus.run(workspace / "features", test_root / ROWS, trimmed, workspace / "features_trimmed")
    finally:
        features_for_corpus.list_benchmarks = listed
    done = report["tfidf"]
    for cid in cids:
        mine = np.load(workspace / "features_trimmed" / "tfidf" / f"catalog-{cid:03d}.npy")
        rows = load.corpus_rows(trimmed, test_root / ROWS, cid)
        assert mine.shape[0] == len(rows)
        if str(cid) in declared:
            assert done[cid] == "refitted"
            assert np.array_equal(mine, featurize_requests.tfidf_matrix([request_text(r) for r in rows]))
        else:
            assert done[cid] == "copied"
            assert (workspace / "features_trimmed" / "tfidf" / f"catalog-{cid:03d}.npy").read_bytes() == \
                (full / f"catalog-{cid:03d}.npy").read_bytes()


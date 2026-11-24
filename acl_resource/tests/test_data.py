"""Corpus builds, release boundaries, frozen folds and local label reconstruction."""

import csv
import gzip
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pytest

from acl_resource.data import build_corpus, load, schema, splits, tables
from acl_resource.data.rehydrate import rehydrate


SOURCE = Path("results/decision_index/paper/dataset/candidate")
ROWS = Path("data/decision_index/common-paid-65/inputs")
SETTINGS = Path("acl_resource/SETTINGS.json")


@pytest.fixture(scope="module")
def workspace():
    """Yield a workspace for builds and remove its contents after the tests."""
    with TemporaryDirectory(dir=".") as folder:
        yield Path(folder)


@pytest.fixture(scope="module")
def source(workspace, test_root):
    """Return a copy of the source with an extra file outside the checksum list."""
    path = shutil.copytree(test_root / SOURCE, workspace / "source")
    (path / "extra.txt").write_text("outside the checksum list\n")
    return path


@pytest.fixture(scope="module")
def builds(source, test_root, workspace):
    """Return the three processed corpora shared by read-only tests."""
    paths = {tier: workspace / tier for tier in schema.TIERS}
    for tier, out in paths.items():
        names = "real" if tier == "local" else "coded"
        clearance = test_root / "meta/clearance.csv" if tier == "open" else None
        build_corpus.build(source, test_root / ROWS, test_root / "meta", out,
                           tier=tier, clearance=clearance, names=names)
    return paths


def relative_files(folder):
    """Return the relative names of all files in the folder."""
    return {path.relative_to(folder) for path in folder.rglob("*") if path.is_file()}


def refresh_checksums(folder):
    """Recompute data checksums and their manifest binding after a deliberate test edit."""
    files = relative_files(folder) - {Path("manifest.json"), Path("SHA256SUMS")}
    (folder / "SHA256SUMS").write_text("".join(
        f"{schema.sha256_file(folder / path)}  {path}\n" for path in sorted(files, key=str)))
    manifest = load.read_manifest(folder)
    manifest["checksums_sha256"] = schema.sha256_file(folder / "SHA256SUMS")
    (folder / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")


def coded_values(rows, column, prefix):
    """Return expected codes in order of the original, unfiltered model metadata."""
    distinct = dict.fromkeys(row[column] for row in rows if row[column])
    return {name: f"{prefix}{i:02d}" for i, name in enumerate(distinct, start=1)} | {"": ""}


@pytest.mark.parametrize("tier", schema.TIERS)
def test_tier_contents(tier, builds, corpus, planted, test_root):
    out = builds[tier]
    manifest = load.read_manifest(out)
    source_models = load.list_models(corpus)
    original_indexes = planted["cleared_models"] if tier == "open" else range(len(source_models))
    benchmarks = load.list_benchmarks(corpus)
    cids = [b["catalog_id"] for b in benchmarks
            if tier != "open" or b["catalog_id"] != planted["held_benchmark"]]
    assert manifest["catalog_ids"] == cids
    assert manifest["model_count"] == len(original_indexes)
    assert manifest["tier"] == tier
    assert manifest["tables"] == list(schema.TABLES_BY_TIER[tier])
    assert manifest["source"] == load.read_manifest(corpus)["source"]
    if tier == "open":
        assert manifest["clearance_sha256"] == schema.sha256_file(test_root / "meta/clearance.csv")
    expected_columns = schema.BENCHMARK_COLUMNS_LOCAL if tier == "local" else schema.BENCHMARK_COLUMNS
    output_benchmarks = load.list_benchmarks(out)
    assert tuple(output_benchmarks[0]) == expected_columns
    assert output_benchmarks == [{key: b[key] for key in expected_columns} for b in benchmarks
                                 if b["catalog_id"] in cids]

    models = load.list_models(out)
    assert manifest["names"] == ("real" if tier == "local" else "coded")
    orgs = coded_values(source_models, "organisation", "O")
    bases = coded_values(source_models, "base_model", "B")
    for new, old in enumerate(original_indexes):
        expected = dict(source_models[old], model_index=new)
        if tier != "local":
            expected.update(model_name=expected["model_code"], organisation=orgs[expected["organisation"]],
                            base_model=bases[expected["base_model"]])
        assert models[new] == expected
        assert tuple(models[new]) == schema.MODEL_COLUMNS

    expected_files = {Path(name) for name in ("models.csv", "benchmarks.csv", "manifest.json", "SHA256SUMS")}
    for cid in cids:
        for table in schema.TABLES_BY_TIER[tier]:
            if table in schema.PER_BENCHMARK:
                expected_files.add(schema.table_path(out, cid, table).relative_to(out))
            else:
                for new, old in enumerate(original_indexes):
                    path = schema.table_path(out, cid, table, new)
                    expected_files.add(path.relative_to(out))
                    if table == "predictions" and tier == "open":
                        assert path.read_bytes() == schema.table_path(corpus, cid, table, old).read_bytes()
    assert relative_files(out) == expected_files
    build_corpus.verify(out)


@pytest.mark.parametrize("tier", schema.TIERS)
def test_loader_reads_each_tier(tier, builds):
    out = builds[tier]
    manifest = load.read_manifest(out)
    benchmark = load.load_benchmark(out, manifest["catalog_ids"][0])
    assert benchmark.M == manifest["model_count"]
    assert benchmark.has_labels == (tier == "local")
    assert (benchmark.calls is not None) == (tier != "open")
    assert (benchmark.lat is not None) == (tier == "local")


def test_local_reference_and_repeat(builds, corpus, source, test_root, workspace):
    with TemporaryDirectory(dir=workspace) as folder:
        first, second = builds["local"], Path(folder) / "second"
        build_corpus.build(source, test_root / ROWS, test_root / "meta", second, tier="local")
        assert relative_files(first) == relative_files(corpus) == relative_files(second)
        for file in sorted(relative_files(corpus)):
            assert (first / file).read_bytes() == (corpus / file).read_bytes() == (second / file).read_bytes()


def test_corrupted_source_checksum(test_root, workspace):
    with TemporaryDirectory(dir=workspace) as folder:
        source = shutil.copytree(test_root / SOURCE, Path(folder) / "source")
        path = source / "models.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        out = Path(folder) / "out"
        with pytest.raises(ValueError):
            build_corpus.build(source, test_root / ROWS, test_root / "meta", out)
        assert not out.exists()


def test_tampered_output_checksum(corpus, workspace):
    with TemporaryDirectory(dir=workspace) as folder:
        out = shutil.copytree(corpus, Path(folder) / "out")
        path = out / "models.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(ValueError):
            build_corpus.verify(out)


@pytest.mark.parametrize("fault", ["negative", "nonfinite", "empty", "reported_null"])
def test_invalid_prediction_before_writing(fault, test_root, workspace, monkeypatch):
    with TemporaryDirectory(dir=workspace) as folder:
        source = shutil.copytree(test_root / SOURCE, Path(folder) / "source")
        cid = load.read_manifest(source)["catalog_ids"][0]
        path = schema.table_path(source, cid, "predictions", 1)
        table = schema.read_table(path, schema.PREDICTIONS)
        column = "reported_choice" if fault == "reported_null" else "probabilities"
        col = schema.PREDICTIONS.field(column)
        values = table[column].to_pylist()
        if fault == "empty":
            values[0] = []
        elif fault == "reported_null":
            assert table["question_type"][0].as_py() == "choice"
            values[0] = None
        else:
            values[0][0] = float("nan") if fault == "nonfinite" else -1.0
        changed = table.set_column(table.schema.get_field_index(column), col, pa.array(values, col.type))
        schema.write_table(path, changed, schema.PREDICTIONS)
        refresh_checksums(source)

        def unexpected_write(*args, **kwargs):
            pytest.fail("a table was written before source validation finished")

        monkeypatch.setattr(schema, "write_table", unexpected_write)
        monkeypatch.setattr(shutil, "copyfile", unexpected_write)
        out = Path(folder) / "out"
        with pytest.raises(ValueError):
            build_corpus.build(source, test_root / ROWS, test_root / "meta", out)
        assert not out.exists()


def test_nonempty_destination(source, workspace, test_root):
    with TemporaryDirectory(dir=workspace) as folder:
        out = Path(folder)
        marker = out / "existing"
        marker.write_text("keep")
        with pytest.raises(ValueError):
            build_corpus.build(source, test_root / ROWS, test_root / "meta", out)
        assert list(out.iterdir()) == [marker]
        assert marker.read_text() == "keep"


@pytest.mark.parametrize("tier", ["gated", "open"])
def test_release_rejects_free_text_with_matching_checksums(tier, builds, workspace):
    with TemporaryDirectory(dir=workspace) as folder:
        out = shutil.copytree(builds[tier], Path(folder) / "out")
        cid = load.read_manifest(out)["catalog_ids"][-1]
        if tier == "gated":
            path = schema.table_path(out, cid, "usage", len(load.list_models(out)) - 1)
            expected, column = schema.USAGE, "provider"
        else:
            path = schema.table_path(out, cid, "items")
            expected, column = schema.ITEMS, "row_sha256"
        table = schema.read_table(path, expected)
        values = table[column].to_pylist()
        values[-1] = "unexpected free text"
        i = table.schema.get_field_index(column)
        schema.write_table(path, table.set_column(i, expected.field(i), pa.array(values)), expected)
        refresh_checksums(out)
        with pytest.raises(ValueError):
            build_corpus.verify(out)


@pytest.mark.parametrize("change", ["forbidden", "missing"])
def test_verify_enforces_open_table_inventory(change, builds, workspace):
    with TemporaryDirectory(dir=workspace) as folder:
        out = shutil.copytree(builds["open"], Path(folder) / "out")
        cid = load.read_manifest(out)["catalog_ids"][-1]
        if change == "forbidden":
            shutil.copyfile(schema.table_path(builds["local"], cid, "labels"),
                            schema.table_path(out, cid, "labels"))
        else:
            schema.table_path(out, cid, "predictions", 0).unlink()
        refresh_checksums(out)
        with pytest.raises(ValueError):
            build_corpus.verify(out)


def test_open_requires_common_cleared_benchmark(source, test_root, corpus, workspace):
    with TemporaryDirectory(dir=workspace) as folder:
        out = Path(folder) / "out"
        with pytest.raises(ValueError):
            build_corpus.build(source, test_root / ROWS, test_root / "meta", out, tier="open")
        clearance = Path(folder) / "clearance.csv"
        models = load.list_models(corpus)
        cids = load.read_manifest(corpus)["catalog_ids"]
        with clearance.open("w", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["catalog_id", "route_id", "numeric_public_status"])
            for i in range(2):
                writer.writerow([cids[i], models[i]["model_name"], "cleared"])
        with pytest.raises(ValueError):
            build_corpus.build(source, test_root / ROWS, test_root / "meta", out,
                               tier="open", clearance=clearance)
        assert not out.exists()


@pytest.mark.parametrize("tier", ["gated", "open"])
def test_rehydrated_labels_match_local(tier, builds, corpus, workspace, test_root):
    with TemporaryDirectory(dir=workspace) as folder:
        out = Path(folder) / "with-labels"
        rehydrate(builds[tier], test_root / ROWS, out)
        manifest = load.read_manifest(out)
        assert manifest["tier"] == tier
        assert manifest["labels"] == "rehydrated"
        for cid in manifest["catalog_ids"]:
            label_path = schema.table_path(out, cid, "labels")
            assert label_path.read_bytes() == schema.table_path(corpus, cid, "labels").read_bytes()
        assert load.load_benchmark(out, manifest["catalog_ids"][0]).has_labels
        build_corpus.verify(out)


def write_rows(path, rows):
    """Write temporary, local benchmark rows and return nothing."""
    with gzip.GzipFile(path, "wb", mtime=0) as fh:
        for row in rows:
            fh.write((json.dumps(row) + "\n").encode("utf-8"))


def test_rehydrate_checks_derived_fields(builds, test_root, workspace):
    with TemporaryDirectory(dir=workspace) as folder:
        rows_dir = shutil.copytree(test_root / ROWS, Path(folder) / "rows")
        cid = load.read_manifest(builds["gated"])["catalog_ids"][0]
        path = rows_dir / f"{cid:03d}.jsonl.gz"
        rows = tables.read_rows(path)
        question = next(q for q in rows[0]["questions"].values() if q["type"] == "choice")
        question["criteria"]["extra"] = "extra option"
        write_rows(path, rows)
        out = Path(folder) / "out"
        with pytest.raises(ValueError):
            rehydrate(builds["gated"], rows_dir, out)
        assert not out.exists()


def test_rehydrate_checks_run_ranks(builds, test_root, workspace):
    with TemporaryDirectory(dir=workspace) as folder:
        corpus = shutil.copytree(builds["open"], Path(folder) / "open")
        cid = load.read_manifest(corpus)["catalog_ids"][0]
        path = schema.table_path(corpus, cid, "splits")
        table = schema.read_table(path, schema.SPLITS)
        ranks = table["run_rank"].to_pylist()
        first = {}
        for i, group in enumerate(table["split_group"].to_pylist()):
            if group in first:
                j = first[group]
                ranks[i], ranks[j] = ranks[j], ranks[i]
                break
            first[group] = i
        else:
            pytest.fail("fixture needs two requests in the same split group")
        col = schema.SPLITS.field("run_rank")
        table = table.set_column(table.schema.get_field_index(col.name), col, pa.array(ranks, col.type))
        schema.write_table(path, table, schema.SPLITS)
        refresh_checksums(corpus)
        build_corpus.verify(corpus)
        out = Path(folder) / "with-labels"
        with pytest.raises(ValueError, match="splits"):
            rehydrate(corpus, test_root / ROWS, out)
        assert not out.exists()


def test_leakage_and_tracks(builds, test_root):
    settings = json.loads(SETTINGS.read_text())
    with (test_root / SOURCE / "benchmarks.csv").open(newline="") as fh:
        source = {int(row["catalog_id"]): row for row in csv.DictReader(fh)}
    for tier in ("local", "open"):
        path = builds[tier]
        report = splits.leakage_report(path)
        info = load.list_benchmarks(path)
        cids = [b["catalog_id"] for b in info]
        assert set(report) == set(cids)
        for cid in cids:
            stats = report[cid]
            assert stats["split_groups_in_multiple_folds"] == 0
            assert len(stats["requests_per_fold"]) == schema.N_FOLDS
            assert sum(stats["requests_per_fold"]) == int(source[cid]["requests"])
            assert sum(stats["split_groups_per_fold"]) == int(source[cid]["split_groups"])
            if tier == "local":
                assert sum(stats["judged_fields_per_fold"]) == int(source[cid]["judged_fields"])
            else:
                assert "judged_fields_per_fold" not in stats
        tracks = splits.tracks(path, settings)
        rotations = tracks["rotations"]
        assert [r["test"] for r in rotations] == list(range(schema.N_FOLDS))
        for rotation in rotations:
            fit, validation, _ = schema.roles(rotation["test"])
            assert rotation["fit"] == list(fit) and rotation["validation"] == validation
        assert tracks["suites"] == {name: [cid for cid in cids if cid not in suite["exclude"]]
                                    for name, suite in settings["suites"].items()}
        areas = {b["area"] for b in info}
        assert set(tracks["leave_one_area_out"]) == areas
        for area in areas:
            target = [b["catalog_id"] for b in info if b["area"] == area]
            assert tracks["leave_one_area_out"][area] == {
                "target": target, "source": [cid for cid in cids if cid not in target]}


@pytest.mark.parametrize("pair", [(1, 62), (2, 3)])
def test_shared_source_benchmarks_must_stay_together(pair, builds, workspace):
    with TemporaryDirectory(dir=workspace) as folder:
        out = shutil.copytree(builds["open"], Path(folder) / "out")
        rows = load.list_benchmarks(out)
        area = next(row["area"] for row in rows if row["catalog_id"] == pair[0])
        changed = next(row for row in rows if row["catalog_id"] == pair[1])
        changed["area"] = next(row["area"] for row in rows if row["area"] != area)
        with (out / "benchmarks.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=schema.BENCHMARK_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        refresh_checksums(out)
        with pytest.raises(ValueError):
            splits.tracks(out, SETTINGS)


def test_duplicate_report_counts_requests_in_multiple_folds(test_root, workspace):
    with TemporaryDirectory(dir=workspace) as folder:
        root = Path(folder)
        cid = 59
        rows_dir = root / ROWS
        rows_dir.mkdir(parents=True)
        dest = schema.table_path(root / SOURCE, cid, "items")
        dest.parent.mkdir(parents=True)
        shutil.copyfile(schema.table_path(test_root / SOURCE, cid, "items"), dest)
        folds = schema.read_table(dest, schema.ITEMS)["fold"].to_pylist()
        path = rows_dir / f"{cid:03d}.jsonl.gz"
        rows = tables.read_rows(test_root / ROWS / path.name)
        same = next(i for i in range(1, len(rows)) if folds[i] == folds[0])
        other = next(i for i in range(1, len(rows)) if folds[i] != folds[0])
        rows[same]["state"] = rows[0]["state"]
        rows[same]["questions"] = rows[0]["questions"]
        write_rows(path, rows)
        assert splits.duplicate_report(rows_dir) == {cid: 0}
        rows[other]["questions"] = rows[0]["questions"]
        rows[other]["state"]["text"] = "\t".join(rows[0]["state"]["text"].upper().split())
        write_rows(path, rows)
        assert splits.duplicate_report(rows_dir) == {cid: 3}

"""Tables for a dataset hub: the records of the open tier, with the benchmark and the model as columns."""

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from acl_resource.analysis import build_repo, lint_release
from acl_resource.data import build_corpus, hub, load, schema

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = Path("results/decision_index/paper/dataset/candidate")
ROWS = Path("data/decision_index/common-paid-65/inputs")


@pytest.fixture(scope="module")
def public(test_root, tmp_path_factory):
    """The version-2 open tier of the made-up corpus."""
    out = tmp_path_factory.mktemp("hub") / "open"
    build_corpus.build(test_root / SOURCE, test_root / ROWS, test_root / "meta", out, tier="open",
                       clearance=test_root / "meta/clearance.csv", names="real",
                       version=schema.PUBLIC_SCHEMA_VERSION)
    return out


@pytest.fixture(scope="module")
def notices(public, tmp_path_factory):
    path = tmp_path_factory.mktemp("notices") / "release_notices.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["route_id", "release_conditions_or_missing_authority"])
        writer.writerows([m["model_name"], "none"] for m in load.list_models(public))
    return path


@pytest.fixture(scope="module")
def folder(public, notices, tmp_path_factory):
    out = tmp_path_factory.mktemp("hub_out") / "tables"
    rows = hub.export(public, out, card=PACKAGE / "docs" / "HUB_CARD.md", notices=notices)
    return out, rows


def test_tables_hold_the_corpus(public, folder):
    out, rows = folder
    manifest = load.read_manifest(public)
    counts, models = manifest["counts"], manifest["model_count"]
    assert rows == {"items": counts["requests"], "fields": counts["fields"], "splits": counts["requests"],
                    "predictions": counts["vectors"], "usage": counts["observations"],
                    "latency": counts["observations"]}
    for table in manifest["tables"]:
        expected, base = hub.hub_schema(table, manifest["schema_version"])
        added = [name for name in expected.names if name not in base.names]
        assert added == list(hub.ADDED if table in schema.PER_MODEL else hub.ADDED[:1])
        assert all(pa.types.is_integer(expected.field(name).type) for name in added)
        for cid in manifest["catalog_ids"]:
            back = pq.read_table(out / "data" / table / f"catalog-{cid:03d}.parquet")
            assert back.schema.equals(expected, check_metadata=False)
            assert set(back["catalog_id"].to_pylist()) <= {cid}
            if table in schema.PER_MODEL:
                parts = [schema.read_table(schema.table_path(public, cid, table, m), base) for m in range(models)]
                assert back["model_index"].to_pylist() == [m for m, part in enumerate(parts) for _ in range(len(part))]
            else:
                parts = [schema.read_table(schema.table_path(public, cid, table), base)]
            assert back.select(base.names).to_pylist() == [row for part in parts for row in part.to_pylist()]


def test_names_tables_hold_the_lists(public, folder):
    """A hub reads every table of a dataset in one format, so the two lists of names are tables as well."""
    out, _ = folder
    for name, rows in (("benchmarks", load.list_benchmarks(public)), ("models", load.list_models(public))):
        back = pq.read_table(out / "data" / name / f"{name}.parquet").to_pylist()
        assert len(back) == len(rows) and list(back[0]) == list(rows[0])
        for mine, theirs in zip(back, rows):
            assert all(mine[key] == (type(mine[key])(value) if value is not None else None)
                       for key, value in theirs.items())
        assert (out / f"{name}.csv").read_bytes() == (public / f"{name}.csv").read_bytes()


def test_folder_inventory(public, folder, notices):
    out, _ = folder
    listed = [line.split("  ", 1)[1] for line in (out / "SHA256SUMS").read_text().splitlines()]
    assert listed == sorted(p.relative_to(out).as_posix() for p in out.rglob("*")
                            if p.is_file() and p.name != "SHA256SUMS")
    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert schema.sha256_file(out / name) == digest
    assert (out / "corpus_manifest.json").read_bytes() == (public / "manifest.json").read_bytes()
    assert (out / "models.csv").read_bytes() == (public / "models.csv").read_bytes()
    assert (out / "release_notices.csv").read_bytes() == notices.read_bytes()
    assert not (out / "manifest.json").exists() and not list(out.rglob("labels*"))
    assert lint_release.check_release(out)[0] == []


def test_card_is_filled(public, folder):
    out, _ = folder
    card = (out / "README.md").read_text(encoding="utf-8")
    manifest = load.read_manifest(public)
    assert "{{" not in card and card.startswith("---\n")
    for table in list(manifest["tables"]) + ["benchmarks", "models"]:
        assert f"- config_name: {table}\n  data_files: data/{table}/*.parquet\n" in card
        assert list((out / "data" / table).glob("*.parquet"))
    assert f"| `items` | request | {manifest['counts']['requests']:,} |" in card
    assert "license" not in card.split("---")[1]
    with pytest.raises(ValueError):
        hub.fill("{{not_computed}}", hub.facts(public))


def test_refusals(public, corpus, folder, notices, tmp_path):
    with pytest.raises(ValueError):                                # a tier with labels
        hub.export(corpus, tmp_path / "local")
    with pytest.raises(ValueError):                                # an existing folder is not written into
        hub.export(public, folder[0])
    other = tmp_path / "notices.csv"
    other.write_text(notices.read_text(encoding="utf-8") + "some/other-model,none\n", encoding="utf-8")
    with pytest.raises(ValueError):                                # notices of a model that is not in the corpus
        hub.export(public, tmp_path / "with_notices", notices=other)
    assert not (tmp_path / "local").exists() and not (tmp_path / "with_notices").exists()


def test_changed_value_is_noticed():
    a = pa.table({"x": pa.array([[0.25, 0.75], [1.0]], pa.list_(pa.float64())), "y": pa.array([1, None], pa.int32())})
    assert hub.same(a, pa.concat_tables([a.slice(0, 1), a.slice(1, 1)]))
    assert not hub.same(a, a.set_column(0, "x", pa.array([[0.25, 0.75], [0.5]], pa.list_(pa.float64()))))
    assert not hub.same(a, a.set_column(0, "x", pa.array([[0.25], [0.75, 1.0]], pa.list_(pa.float64()))))
    assert not hub.same(a, a.set_column(1, "y", pa.array([1, 2], pa.int32())))
    nan = pa.table({"x": pa.array([float("nan")])})
    assert hub.same(nan, nan)


def test_repository_from_a_package(public, tmp_path):
    """The repository is the package's code folder with the guide beside it."""
    code = tmp_path / "package" / "supplementary" / "acl_resource"
    for path in sorted(public.rglob("*")):
        if path.is_file():
            build_repo.copy(path, code / "data" / "processed" / "open" / path.relative_to(public))
    for path in sorted((PACKAGE / "docs" / "repo").iterdir()):
        build_repo.copy(path, code / "docs" / "repo" / path.name)
    count, counts = build_repo.build(tmp_path / "package", tmp_path / "repo")
    assert counts == load.read_manifest(public)["counts"]
    assert count == sum(p.is_file() for p in (tmp_path / "repo").rglob("*"))
    guide = (tmp_path / "repo" / "README.md").read_text(encoding="utf-8")
    assert "{{" not in guide and f"{counts['requests']:,} requests" in guide
    assert (tmp_path / "repo" / ".gitignore").is_file() and (tmp_path / "repo" / "requirements.txt").is_file()
    assert build_corpus.verify(tmp_path / "repo" / "acl_resource" / "data" / "processed" / "open")["tier"] == "open"
    with pytest.raises(ValueError):
        build_repo.build(tmp_path / "package", tmp_path / "repo")

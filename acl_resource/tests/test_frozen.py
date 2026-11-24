"""Tests of the frozen layer: tables, loader, scoring primitives, the made-up corpus."""

from __future__ import annotations

import numpy as np

from acl_resource.data import schema as S
from acl_resource.data.load import list_benchmarks, list_models, load_benchmark, read_manifest
from acl_resource.sim import core
from acl_resource.sim.types import NOT_CALLED


def test_manifest_and_checksums(corpus):
    manifest = read_manifest(corpus)
    assert manifest["schema_version"] == S.SCHEMA_VERSION and manifest["tier"] == "local"
    assert S.sha256_file(corpus / "SHA256SUMS") == manifest["checksums_sha256"]
    for line in (corpus / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        assert S.sha256_file(corpus / name.strip()) == digest
    assert len(list_models(corpus)) == manifest["model_count"] == 22
    assert [b["catalog_id"] for b in list_benchmarks(corpus)] == manifest["catalog_ids"]


def test_every_benchmark_loads_and_is_leak_free(corpus):
    for info in list_benchmarks(corpus):
        b = load_benchmark(corpus, info["catalog_id"])
        assert b.R == int(info["requests"]) and b.F == int(info["fields"])
        assert int(b.judged.sum()) == int(info["judged_fields"])
        for w in b.widths:
            assert np.allclose(b.P[w].sum(axis=2), 1.0, atol=1e-12)
        fold_of_group = {}
        for g, f in zip(b.split_group.tolist(), b.fold.tolist()):
            assert fold_of_group.setdefault(g, f) == f
        assert sorted(b.run_rank.tolist()) == list(range(b.R))


def test_planted_duplicate_and_decoding(corpus, planted):
    copy, original = planted["duplicate"]["copy"], planted["duplicate"]["of"]
    b = load_benchmark(corpus, 9)
    for w in b.widths:
        assert np.array_equal(b.P[w][:, copy], b.P[w][:, original])
    hit, mass, choice, confidence = core.decode(b, core.single_model_vectors(b, original))
    assert hit.shape == (b.F,) and (mass[b.judged] >= 0).all() and (confidence <= 1 + 1e-12).all()
    assert np.isfinite(core.brier(b, core.single_model_vectors(b, original))[b.judged]).all()


def test_plan_costs(corpus):
    b = load_benchmark(corpus, 20)
    one = np.full((b.R, b.M), NOT_CALLED, np.int16)
    one[:, 3] = 0
    serial = np.full((b.R, b.M), NOT_CALLED, np.int16)
    serial[:, 3], serial[:, 5] = 0, 1
    parallel = serial.copy()
    parallel[:, 5] = 0
    a, s, p = core.plan_costs(b, one), core.plan_costs(b, serial), core.plan_costs(b, parallel)
    assert a["calls_per_request"] == 1 and s["calls_per_request"] == 2
    assert abs(s["cost_usd"] - p["cost_usd"]) < 1e-15 and s["latency_mean_s"] > p["latency_mean_s"]
    assert abs(s["latency_mean_s"] - (b.lat[:, 3] + b.lat[:, 5]).mean()) < 1e-12

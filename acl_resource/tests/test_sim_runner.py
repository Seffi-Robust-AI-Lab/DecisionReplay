"""End-to-end experiment files, paired local comparators and process determinism."""

import hashlib
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from acl_resource.analysis.run_experiments import main
from acl_resource.data.load import list_benchmarks
from acl_resource.sim.types import CONTRAST_ROW_KEYS, RESULT_ROW_KEYS


def _settings(directory):
    """Return the settings and their file with shorter grids for the integration run."""
    settings = json.loads((Path("acl_resource") / "SETTINGS.json").read_text())
    settings["bootstrap"]["draws"] = 49
    settings["size_selection"]["draws"] = 49
    settings["frontier"]["sizes"] = [1, 2, 22]
    settings["frontier"]["random_panel_seeds"] = settings["frontier"]["random_panel_seeds"][:2]
    settings["label_budget"]["groups"] = settings["label_budget"]["groups"][:1]
    settings["label_budget"]["draws"] = 2
    for grid in ("tau_grid", "lambda_grid", "k_grid"):
        method = "confidence_cascade" if grid == "tau_grid" else "knn_router"
        values = settings[method][grid]
        settings[method][grid] = [values[0], values[-1]]
    settings["temperature_menu"].update(lowest_step=-1, highest_step=1)
    path = directory / "settings.json"
    path.write_text(json.dumps(settings, sort_keys=True))
    return settings, path


def _json(directory, section):
    """Return a result document."""
    return json.loads((directory / f"{section}.json").read_bytes())


def _expected_points(settings):
    """Return the method and variant pairs for each type of replay."""
    frontier = {(method, f"k={size}") for size in settings["frontier"]["sizes"]
                for method in ("ranked_top_k", "greedy", "random_panel")}
    frontier.update(("knn_router", f"lambda={p}") for p in settings["knn_router"]["lambda_grid"])
    frontier.update(("single_by_cost", f"lambda={p}") for p in settings["knn_router"]["lambda_grid"])
    frontier.update(("confidence_cascade", f"tau={t}") for t in settings["confidence_cascade"]["tau_grid"])
    ablations = {("uniform_mean", kind) for kind in (
        "none", "final_temperature", "calibrated_members", "calibrated_members_and_final_temperature")}
    ablations.update((method, f"groups={size}") for size in [*settings["label_budget"]["groups"], "all"]
                     for method in ("best_single", "stack"))
    ablations.update((method, None) for method in ("transfer_single", "transfer_temperature"))
    return {"rows": {(m, None) for m in settings["methods"]}, "frontier": frontier, "ablations": ablations}


def _check_suite_coverage(rows, points, cids, settings):
    """Check that every selected benchmark and every nonempty suite has exactly one row per point."""
    expected = {(m, v, cid, None) for m, v in points for cid in cids}
    for suite, spec in settings["suites"].items():
        if set(cids) - set(spec["exclude"]):
            expected.update((m, v, None, suite) for m, v in points)
    actual = {(r["method"], r["variant"], r["catalog_id"], r["suite"]) for r in rows}
    assert actual == expected and len(rows) == len(expected)


def _check_contrasts(out, names, settings, cids, digest):
    """Check suite membership, sign convention, paired draw counts and adjustment scope."""
    rows = _json(out, "contrasts")["rows"]
    expected = {(name, metric, suite) for name in names for metric in settings["contrast_metrics"]
                for suite, spec in settings["suites"].items() if set(cids) - set(spec["exclude"])}
    assert {(r["contrast"], r["metric"], r["suite"]) for r in rows} == expected
    assert len(rows) == len(expected)
    benchmark = {(r["method"], r["catalog_id"]): r for r in _json(out, "rows")["rows"]
                 if r["catalog_id"] is not None}
    for r in rows:
        assert set(r) == set(CONTRAST_ROW_KEYS)
        assert r["settings_sha256"] == digest
        selected = sorted(set(cids) - set(settings["suites"][r["suite"]]["exclude"]))
        n = len(selected)
        differences = np.array([benchmark[r["a"], cid][r["metric"]] - benchmark[r["b"], cid][r["metric"]]
                                for cid in selected])
        assert abs(r["difference"] - math.fsum(differences) / n) < 1e-15
        improvements = -differences if r["metric"] == "log_loss" else differences
        assert r["benchmarks_better"] == (improvements > 1e-12).sum()
        assert r["benchmarks_worse"] == (improvements < -1e-12).sum()
        assert n == r["n_benchmarks"] == sum(r[f"benchmarks_{key}"] for key in ("better", "same", "worse"))
        assert r["draws"] == settings["bootstrap"]["draws"]
        assert 0 < r["p"] <= 1
        outside = {c["name"] for c in settings["contrasts"] if c.get("in_family") is False}
        if (r["suite"] == settings["headline_suite"] and r["metric"] == settings["holm_family"]["metric"]
                and r["contrast"] not in outside):
            assert r["p"] <= r["p_holm"] <= 1
        else:
            assert r["p_holm"] is None
    return rows


def test_runner_outputs(corpus):
    cids = [2, 59]
    counts = {entry["catalog_id"]: int(entry["judged_fields"]) for entry in list_benchmarks(corpus)}
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path = _settings(directory)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        args = ["--corpus", str(corpus), "--settings", str(path), "--cids", *map(str, cids)]
        first, second = directory / "first", directory / "second"
        main([*args, "--out", str(first), "--processes", "1"])
        main([*args, "--out", str(second), "--processes", "2"])
        sections = ("rows", "contrasts", "frontier", "ablations", "run")
        for section in sections:
            filename = f"{section}.json"
            assert (first / filename).read_bytes() == (second / filename).read_bytes(), section
            doc = _json(first, section)
            assert doc["settings_sha256"] == digest and doc["synthetic"] is True
        manifest_hash = hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest()
        assert _json(first, "run")["corpus_manifest_sha256"] == manifest_hash
        for section, points in _expected_points(settings).items():
            rows = _json(first, section)["rows"]
            _check_suite_coverage(rows, points, cids, settings)
            assert all(set(row) == set(RESULT_ROW_KEYS) and row["settings_sha256"] == digest for row in rows)
        for method in settings["methods"]:
            with np.load(first / "local" / f"fields-{method}.npz", allow_pickle=False) as fields:
                assert set(fields.files) == {f"{key}_{c}" for c in cids for key in ("mass", "hit")}
                for c in cids:
                    assert fields[f"mass_{c}"].shape == fields[f"hit_{c}"].shape == (counts[c],)
        active = [spec["name"] for spec in settings["contrasts"] if not spec.get("needs_local_method")]
        _check_contrasts(first, active, settings, cids, digest)
        notes = " ".join(_json(first, "run")["notes"])
        assert all(spec["name"] in notes for spec in settings["contrasts"] if spec.get("needs_local_method"))

        # Supply identical predictions under a local method name. Its contrast must be identically zero.
        local = first / "local"
        shutil.copyfile(local / "fields-best_single.npz", local / "fields-tree_replay.npz")
        rows = _json(first, "rows")
        rows["rows"] += [dict(r, method="tree_replay") for r in rows["rows"] if r["method"] == "best_single"]
        (first / "rows.json").write_text(json.dumps(rows, sort_keys=True))
        existing = (first / "rows.json").read_bytes()
        main([*args, "--out", str(first), "--only", "contrasts"])
        assert (first / "rows.json").read_bytes() == existing
        contrasts = _check_contrasts(first, active + ["C6"], settings, cids, digest)
        for r in contrasts:
            if r["contrast"] == "C6":
                assert r["ci_low"] == r["difference"] == r["ci_high"] == 0
                assert r["p"] == 1 and r["n_benchmarks"] == r["benchmarks_same"]
        notes = " ".join(_json(first, "run")["notes"])
        assert "C6" not in notes and "C7" in notes

        # This suite excludes 59. Neither local archive contains fields for that benchmark.
        kept = cids[0]
        with np.load(local / "fields-best_single.npz", allow_pickle=False) as fields:
            supplied = {f"{key}_{kept}": fields[f"{key}_{kept}"] for key in ("mass", "hit")}
        for name in ("tree_replay", "greedy_replay"):
            np.savez_compressed(local / f"fields-{name}.npz", **supplied)
        suite = settings["headline_suite"]
        settings["suites"] = {suite: settings["suites"][suite]}
        settings["contrasts"] = [s for s in settings["contrasts"] if s["name"] in ("C6", "C7")]
        path.write_text(json.dumps(settings, sort_keys=True))
        main([*args, "--out", str(first), "--only", "contrasts"])
        contrasts = _json(first, "contrasts")["rows"]
        assert {r["contrast"] for r in contrasts} == {"C6", "C7"}
        assert all(r["n_benchmarks"] == 1 and r["difference"] == 0 for r in contrasts)
        assert not _json(first, "run")["notes"]

        main([*args, "--out", str(first), "--only", "rows"])
        external = [r for r in _json(first, "rows")["rows"] if r["method"] == "tree_replay"]
        assert {r["catalog_id"] for r in external} == {None, *cids}

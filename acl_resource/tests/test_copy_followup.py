"""The package builders take the follow-up aggregates and nothing else from results/followup."""

import json

import pytest

from acl_resource.analysis.build_package import copy_followup


def test_copy_followup_takes_top_level_json_only(tmp_path):
    results = tmp_path / "results" / "main"
    results.mkdir(parents=True)
    followup = tmp_path / "results" / "followup"
    (followup / "local").mkdir(parents=True)
    (followup / "followup_facts.json").write_text(json.dumps({"suite": "main35"}))
    (followup / "fold_redraws.json").write_text("{}")
    (followup / "rows.json").write_text("{}")                     # a copy of the main rows: left out
    (followup / "local" / "fields-x.npz").write_bytes(b"per-item arrays")
    (followup / "notes.txt").write_text("not json")
    target = tmp_path / "package" / "results" / "followup"
    assert copy_followup(results, target) == 2
    assert sorted(p.name for p in target.iterdir()) == ["fold_redraws.json", "followup_facts.json"]


def test_copy_followup_without_folder(tmp_path):
    results = tmp_path / "results" / "main"
    results.mkdir(parents=True)
    assert copy_followup(results, tmp_path / "out") == 0


def test_copy_followup_refuses_forbidden_names(tmp_path):
    from acl_resource.analysis.build_package import FORBIDDEN_NAMES
    names = [n for n in FORBIDDEN_NAMES if n.endswith(".json")]
    if not names:
        pytest.skip("no forbidden JSON name to try")
    results = tmp_path / "results" / "main"
    results.mkdir(parents=True)
    followup = tmp_path / "results" / "followup"
    followup.mkdir(parents=True)
    (followup / names[0]).write_text("{}")
    with pytest.raises(ValueError):
        copy_followup(results, tmp_path / "out")

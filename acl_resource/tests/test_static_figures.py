"""The hand-drawn figures of the paper agree with the versioned aggregate inputs, and a changed input is noticed."""

import copy
import csv
import json
import shutil
from pathlib import Path

import pytest

from acl_resource.analysis import static_figures as SF

PACKAGE = Path("acl_resource")
# The supplied drawings show the first study (22 models, 36 benchmarks) and are kept beside the paper under their own
# name; the checker is tested on them against that study's audit and diagnostics.
FOLDER = PACKAGE / "paper" / (SF.FOLDER + "_22_models")
INPUTS = {"audit": PACKAGE / "docs" / "AUDIT_SUMMARY.json", "release": PACKAGE / "data" / "meta" / "release_summary.json",
          "diagnostics": PACKAGE / "results" / "main" / "diagnostics" / "diagnostics.json",
          "models": PACKAGE / "data" / "meta" / "models_meta.csv",
          "benchmarks": PACKAGE / "data" / "meta" / "benchmarks_meta.csv"}
pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in [*INPUTS.values(), *(FOLDER / name for name in SF.LABELS)])
    or shutil.which("pdftotext") is None or shutil.which("pdftocairo") is None,
    reason="the drawings, their aggregate inputs or poppler are not present")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def inputs():
    """Return the arguments of ``static_figures.check`` from files that hold no per-item value."""
    audit = json.loads(INPUTS["audit"].read_text(encoding="utf-8"))
    release = json.loads(INPUTS["release"].read_text(encoding="utf-8"))
    settings = json.loads((PACKAGE / "SETTINGS.json").read_text(encoding="utf-8"))
    suite = json.loads(INPUTS["diagnostics"].read_text(encoding="utf-8"))["suites"][settings["headline_suite"]]
    models = [dict(row, model_name=audit["models"][row["model_index"]]["route_id"]) for row in read_csv(INPUTS["models"])]
    return dict(audit=audit, release=release, diagnostics=suite["overall"]["models"], corpus_models=models,
                catalog=read_csv(INPUTS["benchmarks"]))


def test_drawings_agree_with_the_inputs(inputs):
    compared = SF.check(FOLDER, **inputs)
    assert sorted(compared) == sorted(SF.LABELS) and all(compared.values())


@pytest.mark.parametrize("change", ["requests", "collection cost", "released model", "held benchmark", "accuracy",
                                    "cost", "latency", "organisation"])
def test_a_changed_input_is_noticed(inputs, change):
    changed = copy.deepcopy(inputs)
    if change == "requests":
        changed["audit"]["corpus"]["requests"] += 1
    elif change == "collection cost":
        changed["audit"]["telemetry"]["cost_usd"] += 0.01
    elif change == "released model":
        name = changed["corpus_models"][0]["model_name"]
        status = changed["release"]["model_status"][name]
        status["release"] = "held" if status["release"] == "released" else "released"
    elif change == "held benchmark":
        changed["release"]["benchmarks_held"] = [int(changed["catalog"][0]["catalog_id"])]
    elif change == "accuracy":
        changed["diagnostics"][0]["accuracy_first_max"] -= 0.004
    elif change == "cost":
        changed["diagnostics"][0]["usd_per_1k_requests"] *= 1.08
    elif change == "latency":
        changed["diagnostics"][0]["latency_p50_s"] += 0.03
    else:
        changed["corpus_models"][1]["organisation"] = "another organisation"
    with pytest.raises(ValueError):
        SF.check(FOLDER, **changed)

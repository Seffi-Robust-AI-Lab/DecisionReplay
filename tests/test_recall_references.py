"""Reference values for the router's recall at k, on the made-up corpus."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from analysis import recall_references
from analysis.hardness import model_hits
from analysis.router_study import RECALL_K, recall
from sim.engine import load_task

from .test_restyle_analyses import CIDS, _results


def test_recall_references_report(corpus):
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path, out = _results(corpus, directory, ("rows",))
        target = directory / "followup"
        suites = recall_references.run(corpus, path, out, target, processes=1)
        document = json.loads((target / "recall_references.json").read_text())
        assert document["recall_k"] == list(RECALL_K) and document["suites"].keys() == suites.keys()
        for block in suites.values():
            for name in ("fixed_ranking", "control", "chance"):
                for subset in recall_references.SUBSETS:
                    cell = block["references"][name][subset]
                    values = [cell[f"at_{k}"] for k in RECALL_K]
                    if cell["fields"]:
                        # a longer list of models cannot hold fewer right ones
                        assert values == sorted(values) and 0 <= values[0] and values[-1] <= 1 + 1e-12


def test_fixed_ranking_and_chance(corpus):
    task = load_task(corpus, CIDS[0])
    hits = model_hits(task)
    position = recall_references.fixed_positions(task)
    assert position.shape == (task.R, task.b.M)
    assert np.array_equal(np.sort(position, axis=1), np.tile(np.arange(task.b.M), (task.R, 1)))
    # with every model in the list a right model is always found, by ranking and by chance
    full = recall(task, hits, position)
    assert full["any_right"]["fields"] == int((hits.sum(axis=0) >= 1).sum())
    expected = recall_references.chance(task, hits)
    M = task.b.M
    right = hits.sum(axis=0)
    mask = right >= 1
    by_hand = sum(1.0 - math.comb(M - int(r), 1) / math.comb(M, 1) for r in right[mask])
    assert abs(expected["any_right"]["at_1"] - by_hand) < 1e-9
    # the first-ranked model's recall is its accuracy on the fields where a model is right
    first = position.argmin(axis=1)[task.b.field_row[task.J]]
    assert full["any_right"]["at_1"] == int(hits[first, np.arange(task.nJ)][mask].sum())

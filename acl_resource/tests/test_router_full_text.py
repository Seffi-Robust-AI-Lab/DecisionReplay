"""The whole-request representation: its text and pooling, and the comparison of routers on two feature folders."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from acl_resource.analysis import embed_requests_full as EF
from acl_resource.analysis import router_full_text as RT
from acl_resource.analysis import router_study
from acl_resource.analysis.embed_requests import request_text
from acl_resource.data.load import list_benchmarks
from acl_resource.data.schema import N_FOLDS

from .test_restyle_analyses import CIDS, _results
from .test_router_unknown import _features


def test_text_holds_the_options_and_keeps_their_order():
    row = dict(state={"b": 1, "a": "context"},
               questions={"q1": dict(type="choice", instructions="pick one", criteria={"second": "omega", "first": "alpha"}),
                          "q2": dict(type="yesno", instructions={"task": "decide", "candidate": "delta"},
                                     criteria={"no": "it is not", "yes": "it is"})})
    text = EF.full_text(row)
    assert text.startswith(json.dumps(row["state"], sort_keys=True))
    for piece in ("pick one", "omega", "alpha", "delta", "it is not"):
        assert piece in text
    assert text.index("omega") < text.index("alpha")                       # the options keep their order
    assert "omega" not in request_text(row)                                # the stored text leaves them out
    assert EF.question_text(dict(type="choice", instructions=None, criteria=None)) == "\n"


def test_windows_cover_every_token_once():
    assert EF.windows(0, 5) == [(0, 0)]
    assert EF.windows(5, 5) == [(0, 5)]
    spans = EF.windows(23, 5)
    assert spans[0] == (0, 5) and spans[-1] == (20, 23)
    assert [b for _, b in spans[:-1]] == [a for a, _ in spans[1:]]
    with pytest.raises(ValueError):
        EF.windows(3, 0)


def test_pooling_weighs_windows_by_their_tokens():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    pooled = EF.pool_windows(vectors, [3, 1, 2], np.array([0, 0, 1]), 3)
    assert pooled.dtype == np.float32 and pooled.shape == (3, 2)
    expected = np.array([3.0, 1.0]) / np.sqrt(10.0)
    assert np.allclose(pooled[0], expected, atol=1e-6) and np.allclose(pooled[1], [0.0, 1.0])
    assert np.array_equal(pooled[2], [0.0, 0.0])                           # a request without a window stays zero


def _other_features(corpus, folder):
    """A second made-up feature folder with other values and fewer copies."""
    folder.mkdir(parents=True)
    for b in list_benchmarks(corpus):
        x = np.random.default_rng(1000 + b["catalog_id"]).normal(size=(int(b["requests"]), 6)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        np.save(folder / f"catalog-{b['catalog_id']:03d}.npy", x)
    (folder / "features.json").write_text(json.dumps(dict(model="made-up", normalised=True)))
    return folder


def test_router_full_text_report(corpus):
    with TemporaryDirectory(dir=".") as directory:
        directory = Path(directory)
        settings, path, out = _results(corpus, directory, ("rows",))
        old = _features(corpus, directory / "features" / "minilm")
        router_study.run(corpus, path, out, str(old.parent), processes=1)
        new = _other_features(corpus, directory / "other" / "minilm_full")
        target = directory / "followup"
        suites = RT.run(corpus, path, out, old, new, target, processes=1)
        document = json.loads((target / "router_full_text.json").read_text())
        assert document["old"] == "minilm" and document["new"] == "minilm_full"
        for block in suites.values():
            assert {"knn_old", "knn_new", "ridge_old", "ridge_new"} <= block["points"].keys()
            assert len(block["contrasts"]) == len(RT.PAIRS) * len(RT.RULES)
            for rule in RT.RULES:
                routing = block["routing"][rule]
                assert routing["ridge_cells"] == N_FOLDS * block["benchmarks"]
                assert 0 <= routing["knn_requests_sent_to_another_model"] <= 1
                for key in ("old", "new"):
                    values = [block["recall"][rule][key]["any_right"][f"at_{k}"] for k in router_study.RECALL_K]
                    assert values == sorted(values)
            # the made-up old folder repeats rows, the new one does not
            assert block["distinct"]["new"]["benchmarks_below_half_distinct"] == 0
            assert block["distinct"]["old"]["benchmarks_with_one_vector"] == 0
            assert block["benchmarks"] <= len(CIDS)


def test_merge_needs_every_benchmark_once(corpus, tmp_path):
    cids = [b["catalog_id"] for b in list_benchmarks(corpus)]
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    np.save(first / f"catalog-{cids[0]:03d}.npy", np.zeros((1, 2), np.float32))
    np.save(second / f"catalog-{cids[0]:03d}.npy", np.zeros((1, 2), np.float32))
    with pytest.raises(ValueError, match="two runs"):
        EF.merge([(first, "cpu"), (second, "mps")], tmp_path, corpus, tmp_path / "out")
    (second / f"catalog-{cids[0]:03d}.npy").unlink()
    with pytest.raises(ValueError, match="no matrix"):
        EF.merge([(first, "cpu"), (second, "mps")], tmp_path, corpus, tmp_path / "out")

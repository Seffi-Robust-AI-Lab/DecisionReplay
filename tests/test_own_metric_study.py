"""The study of the benchmarks' own metrics, on the made-up corpus.

The kit scores some of the made-up benchmarks.  The tests that need it are skipped where the kit, its wrapper or
the made-up benchmark rows are absent; everything else runs with the stand-in scorer (accuracy over judged fields).
"""

import dataclasses
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from analysis import own_metric_study as OM
from analysis import run_experiments as RX
from analysis.run_experiments import main
from data.schema import N_FOLDS, sha256_file
from sim import core, metrics, native, stats
from sim.baselines import best_single
from sim.engine import Task, load_task

from .test_restyle_analyses import _fake_features
from .test_sim_runner import _settings

CIDS = [1, 4, 11, 30]     # suite members the kit scores on made-up rows: exact per case, macro-F1 (two kinds), tracks
ROWS = "data/decision_index/common-paid-65/inputs"
KIT = Path("jdi/paper_scores.py").is_file() and Path("data/decision_index/upstream/kit/decision_index").is_dir()
needs_kit = pytest.mark.skipif(not KIT, reason="the pinned kit and its wrapper are not beside the working folder")
DRAWS = 29


def _stored(corpus, directory, cids, rows_dir=None):
    """A made-up results folder with the stored rows that the study checks itself against.

    ``run_experiments`` writes the built-in methods; the text router's row is added here, and with ``rows_dir``
    the kit's own metric of every policy (what ``local_methods`` does on the real corpus).
    """
    settings, path = _settings(directory)
    out = directory / "out"
    main(["--corpus", str(corpus), "--settings", str(path), "--cids", *map(str, cids), "--out", str(out),
          "--only", "rows"])
    features = directory / "features" / "madeup"
    _fake_features(corpus, features)
    digest = sha256_file(path)
    document = json.loads((out / "rows.json").read_text())
    rows = [r for r in document["rows"] if r["catalog_id"] is not None]
    for cid in cids:
        task = load_task(corpus, cid)
        outcomes = OM.run_policies(task, settings, features)
        rows.append(RX._result_row(task, metrics.score(task, outcomes[OM.TEXT_ROUTER]), OM.TEXT_ROUTER, None, digest))
        if rows_dir is not None:
            for row in rows:
                if row["catalog_id"] == cid and row["method"] in outcomes:
                    row["native"] = native.native_score(".", rows_dir, task.b, outcomes[row["method"]].pooled)
    RX._json(out / "rows.json", dict(document, rows=RX._with_suites(rows, settings["suites"])))
    return settings, path, out, features


@pytest.fixture(scope="module")
def stored(corpus, tmp_path_factory):
    return _stored(corpus, tmp_path_factory.mktemp("own_metric"), CIDS)


def _permuted_labels(bench, fold, seed):
    """A copy of the benchmark with the labels permuted among the fields of one fold (inside each option count)."""
    rng = np.random.default_rng(seed)
    field_fold = bench.fold[bench.field_row]
    acc = {w: a.copy() for w, a in bench.acc.items()}
    judged, n_accepted, gold = bench.judged.copy(), bench.n_accepted.copy(), bench.gold.copy()
    for w in bench.widths:
        ids = bench.idx[w]
        rows = np.flatnonzero(field_fold[ids] == fold)
        acc[w][rows] = acc[w][rng.permutation(rows)]
        count = acc[w].sum(axis=1)
        judged[ids], n_accepted[ids] = count > 0, count
        gold[ids] = np.where(count == 1, acc[w].argmax(axis=1), -1)
    return dataclasses.replace(bench, acc=acc, judged=judged, n_accepted=n_accepted, gold=gold)


class _ShiftedSubset(OM.FieldAccuracy):
    """A scorer whose weights of 0 and 1 do not give the metric of the training requests."""

    def own_subset(self, answers, requests):
        return super().own_subset(answers, requests) + 1e-6


class _NoSubset(OM.FieldAccuracy):
    """A scorer that cannot be built for a subset of the requests."""

    def own_subset(self, answers, requests):
        raise ValueError("no scorer for a subset")


def test_choose_breaks_ties_by_the_given_order():
    assert OM.choose([0.5, 0.7, 0.7, np.nan], [2, 1, 0, 3]) == 2
    assert OM.choose([0.5, 0.7, 0.7 + 5e-13, 0.1], [1, 2, 0, 3]) == 1        # inside the tie tolerance
    assert OM.choose([0.5, 0.7, 0.7 + 5e-12, 0.1], [1, 2, 0, 3]) == 2        # outside it
    assert OM.choose([np.nan, 0.0], [0, 1]) == 1                             # an undefined value is never preferred
    assert OM.choose([np.nan, np.nan], [1, 0]) == 1                          # nothing defined: the first of the order
    assert OM.largest_gap([1.0, np.nan], [1.0, np.nan]) == 0.0 and OM.largest_gap([1.0, np.nan], [1.0, 2.0]) == np.inf


def test_training_weights_are_zero_one_and_leave_the_test_fold_out(corpus):
    for cid in CIDS:
        task = load_task(corpus, cid)
        scorer = OM.FieldAccuracy(task.b)
        answers = OM.decode_all(task.b, [core.single_model_vectors(task.b, 0)])
        for f in range(N_FOLDS):
            w = OM.training_weights(task, f)
            assert w.shape == (1, task.G) and set(np.unique(w)) <= {0.0, 1.0}
            test_groups = np.unique(task.row_group[task.b.fold == f])
            training_groups = np.unique(task.row_group[task.b.fold != f])
            assert not np.intersect1d(test_groups, training_groups).size
            assert not w[0, test_groups].any() and w[0, training_groups].all()
            assert w.sum() == len(training_groups) == task.G - len(test_groups)
            # with these weights the scorer returns the metric of the training folds
            by_hand = answers.hit[0, task.J][task.fold != f].mean()
            assert scorer.own(answers, w)[0, 0] == pytest.approx(by_hand, abs=1e-12)
            assert scorer.own_subset(answers, np.flatnonzero(task.b.fold != f))[0] == pytest.approx(by_hand, abs=1e-12)


def test_candidates_are_scored_on_the_vectors_of_the_log_loss_ranking(corpus):
    task = load_task(corpus, CIDS[1])
    blocks = task.fold_blocks(range(N_FOLDS), judged_only=True)
    for f in (0, 3):
        for m in (0, 8, 21):
            vectors = OM.rotation_vectors(task, m, f)
            view = task.view(f, "calibrated", 4)[:, m]
            assert np.allclose(task.mass_of_blocks(vectors, blocks), view, rtol=0, atol=1e-12)
            # on the test fold they are the vectors that the stitched single-model panel answers with
            stitched = OM.single_panel(task, [m] * N_FOLDS).pooled
            for w in task.b.widths:
                test = task.field_fold[task.b.idx[w]] == f
                assert np.allclose(vectors[w][test], stitched[w][test], rtol=0, atol=1e-15)
    reference = OM.single_panel(task, [int(best_single.ranking(task, f, n=4)[0]) for f in range(N_FOLDS)])
    stored = best_single.run(task, {})
    assert all(np.array_equal(reference.pooled[w], stored.pooled[w]) for w in task.b.widths)
    assert np.array_equal(reference.stage, stored.stage) and reference.ledger == stored.ledger


def test_duplicate_models_tie_and_the_earlier_one_is_chosen(corpus, planted):
    copy, original = planted["duplicate"]["copy"], planted["duplicate"]["of"]
    for cid in CIDS[:2]:
        task = load_task(corpus, cid)
        scorer = OM.FieldAccuracy(task.b)
        for f in range(N_FOLDS):
            answers = OM.decode_all(task.b, [OM.rotation_vectors(task, m, f) for m in (original, copy)])
            values = scorer.own(answers, OM.training_weights(task, f))[0]
            ranking = list(best_single.ranking(task, f, n=4))
            assert values[0] == values[1] and ranking.index(original) < ranking.index(copy)
        assert copy not in OM.own_reference(task, scorer)["by_own_metric"]


def test_labels_of_the_test_fold_do_not_enter_the_choice(corpus):
    for cid in (CIDS[1], CIDS[2]):          # one option count each, so the permutation moves most labels of the fold
        task = load_task(corpus, cid)
        base = OM.own_reference(task, OM.FieldAccuracy(task.b))
        assert not base["fallback"] and base["models"] == base["by_own_metric"]
        for f in range(N_FOLDS):
            changed = Task(_permuted_labels(task.b, f, seed=[cid, f]))
            assert changed.nJ == task.nJ
            assert not np.array_equal(changed.mass[changed.fold == f], task.mass[task.fold == f])    # the labels moved
            assert np.array_equal(changed.mass[changed.fold != f], task.mass[task.fold != f])
            again = OM.own_reference(changed, OM.FieldAccuracy(changed.b))
            assert again["by_own_metric"][f] == base["by_own_metric"][f]
            assert again["best_single"][f] == base["best_single"][f]
            for key in ("best_single_own", "best_single"):
                assert again["training_own"][key][f] == pytest.approx(base["training_own"][key][f], abs=1e-12)
            # control: the other rotations train on that fold, so what they see does change
            assert any(again["training_own"]["best_single_own"][g] != base["training_own"]["best_single_own"][g]
                       for g in range(N_FOLDS) if g != f)


def test_report_with_the_stand_in_scorer(corpus, stored, tmp_path):
    settings, path, results, features = stored
    target = tmp_path / "followup"
    content = OM.run(corpus, path, results, None, None, target, features=features, draws=DRAWS,
                     scorer=OM.FieldAccuracy, cids=CIDS)
    text = (target / "own_metric.json").read_bytes()
    document = json.loads(text)
    assert document == json.loads(json.dumps(content)) and document["synthetic"] is True
    assert document["suite"] == settings["headline_suite"] and document["policies"] == list(OM.POLICIES)
    assert document["scorer"] == OM.FieldAccuracy.NAME and document["text_router_features"] == "madeup"
    assert document["bootstrap"] == dict(draws=DRAWS, seed=settings["bootstrap"]["seed"],
                                         draws_over_benchmarks=settings["bootstrap"]["draws"])
    blocks, summary = document["benchmarks"], document["summary"]
    assert sorted(blocks, key=int) == [str(c) for c in CIDS] and summary["benchmarks"] == len(CIDS)
    assert set(summary["policies"]) == set(OM.CANDIDATES) | {OM.HINDSIGHT}
    assert summary["policies"][OM.HINDSIGHT]["chosen_on"] == "test folds"
    for name in OM.CANDIDATES:
        cells = [blocks[str(c)]["policies"][name] for c in CIDS]
        assert summary["policies"][name]["own"] == math.fsum(p["own"] for p in cells) / len(cells)
        assert all(p["own"] == p["accuracy"] for p in cells)           # the stand-in is the judged-field accuracy
    for reference, key in ((OM.REFERENCE, "against_best_single"), (OM.OWN_REFERENCE, "against_best_single_own")):
        assert set(summary[key]) == set(OM.CANDIDATES) - {reference}
        for name, effect in summary[key].items():
            gaps = [blocks[str(c)]["policies"][name]["own"] - blocks[str(c)]["policies"][reference]["own"] for c in CIDS]
            assert effect["difference"] == pytest.approx(math.fsum(gaps) / len(gaps), abs=1e-15)
            assert effect["benchmarks_above"] == sum(g > 1e-12 for g in gaps)
            assert effect["benchmarks_below"] == sum(g < -1e-12 for g in gaps)
            assert effect["benchmarks_above"] + effect["benchmarks_level"] + effect["benchmarks_below"] == len(CIDS)
            assert effect["groups"]["draws"] == DRAWS and effect["benchmarks"]["draws"] == settings["bootstrap"]["draws"]
            for kind in ("groups", "benchmarks"):
                assert effect[kind]["ci_low"] <= effect[kind]["ci_high"]
            for c, gap in zip(CIDS, gaps):
                cell = blocks[str(c)][key][name]
                assert cell["difference"] == pytest.approx(gap, abs=1e-15) and cell["ci_low"] <= cell["ci_high"]
                assert cell["draws"] == DRAWS
    # the two references mirror each other
    one, other = summary["against_best_single"][OM.OWN_REFERENCE], summary["against_best_single_own"][OM.REFERENCE]
    assert one["difference"] == -other["difference"] and one["benchmarks_above"] == other["benchmarks_below"]
    assert one["groups"]["ci_low"] == pytest.approx(-other["groups"]["ci_high"], abs=1e-12)
    # part 2
    count = summary["best_single_own"]
    assert count["cells"] == N_FOLDS * len(CIDS) and count["fallback_benchmarks"] == []
    assert count["cells_with_another_model"] == sum(
        a != b for c in CIDS for a, b in zip(blocks[str(c)]["models"]["best_single"],
                                             blocks[str(c)]["models"]["best_single_own"]))
    assert count["benchmarks_differing_by_more_than_one_point"] == [
        c for c in CIDS if abs(blocks[str(c)]["policies"][OM.OWN_REFERENCE]["own"]
                               - blocks[str(c)]["policies"][OM.REFERENCE]["own"]) > OM.ONE_POINT]
    with np.load(target / "local" / "fields-best_single_own.npz") as fields, \
            np.load(target / "local" / "own_metric_replicates.npz") as replicates:
        assert list(replicates["candidates"]) == list(OM.CANDIDATES)
        for c in CIDS:
            block, task = blocks[str(c)], load_task(corpus, c)
            assert block["models"]["best_single"] == [int(best_single.ranking(task, f, n=4)[0]) for f in range(N_FOLDS)]
            assert block["models"]["best_single_own"] == block["models"]["by_own_metric"] and not block["fallback"]
            assert block["hindsight"]["chosen_on"] == "test folds" and 0 <= block["hindsight"]["model"] < task.b.M
            # on the training folds the model chosen by the own metric is at least as good as the log-loss choice
            for own, single in zip(block["training_own"]["best_single_own"], block["training_own"]["best_single"]):
                assert own >= single
            checks = block["checks"]
            assert max(checks.values()) <= OM.TOLERANCE and block["draws_defined"] == DRAWS
            assert fields[f"mass_{c}"].shape == fields[f"hit_{c}"].shape == (task.nJ,)
            assert fields[f"hit_{c}"].mean() == block["policies"][OM.OWN_REFERENCE]["accuracy"]
            draws = replicates[f"own_{c}"]
            assert draws.shape == (DRAWS, len(OM.CANDIDATES))
            low, high = np.quantile(draws[:, 1] - draws[:, 0], [0.025, 0.975])
            cell = block["against_best_single"][OM.POLICIES[1]]
            assert (cell["ci_low"], cell["ci_high"]) == pytest.approx((low, high), abs=1e-12)
    # a second run writes the same bytes
    OM.run(corpus, path, results, None, None, tmp_path / "again", features=features, draws=DRAWS,
           scorer=OM.FieldAccuracy, cids=CIDS)
    assert (tmp_path / "again" / "own_metric.json").read_bytes() == text
    for banned in (str(tmp_path), str(Path.cwd()), "seconds"):
        assert banned.encode() not in text


def test_stored_rows_that_the_rerun_does_not_reproduce_stop_the_run(corpus, stored, tmp_path):
    settings, path, results, features = stored

    def run_with(change):
        folder = tmp_path / change.__name__
        shutil.copytree(results, folder)
        document = json.loads((folder / "rows.json").read_text())
        document = change(document)
        (folder / "rows.json").write_text(json.dumps(document))
        return OM.run(corpus, path, folder, None, None, folder / "followup", features=features, draws=3,
                      scorer=OM.FieldAccuracy, cids=CIDS[:2])

    def shifted(document):
        for row in document["rows"]:
            if (row["method"], row["catalog_id"], row["variant"]) == ("greedy", CIDS[1], None):
                row["accuracy"] += 1e-6
        return document

    def without_text_router(document):
        return dict(document, rows=[r for r in document["rows"] if r["method"] != OM.TEXT_ROUTER])

    def without_a_value(document):
        for row in document["rows"]:
            if (row["method"], row["catalog_id"], row["variant"]) == ("uniform_mean", CIDS[0], None):
                row["accuracy"] = None
        return document

    def other_settings(document):
        return dict(document, settings_sha256="0" * 64)

    with pytest.raises(ValueError, match="greedy"):
        run_with(shifted)
    for change in (without_text_router, without_a_value, other_settings):
        with pytest.raises(SystemExit):
            run_with(change)
    # the kit's own metric is not in rows.json of the made-up runs: a run with the kit stops before it scores
    with pytest.raises(SystemExit, match="own metric"):
        OM.run(corpus, path, results, tmp_path / "rows", ".", tmp_path / "followup", features=features, draws=3,
               cids=CIDS[:2])


@pytest.mark.parametrize("scorer", [_ShiftedSubset, _NoSubset])
def test_fallback_to_best_single_is_named(corpus, stored, tmp_path, scorer):
    settings, path, results, features = stored
    content = OM.run(corpus, path, results, None, None, tmp_path / "followup", features=features, draws=5,
                     scorer=scorer, cids=CIDS[:2])
    assert content["summary"]["best_single_own"]["fallback_benchmarks"] == CIDS[:2]
    assert content["summary"]["best_single_own"]["cells_with_another_model"] == 0
    for c in CIDS[:2]:
        block = content["benchmarks"][str(c)]
        assert block["fallback"] and block["fallback_reason"]
        assert block["models"]["best_single_own"] == block["models"]["best_single"]
        assert block["policies"][OM.OWN_REFERENCE] == block["policies"][OM.REFERENCE]
        cell = block["against_best_single"][OM.OWN_REFERENCE]
        assert cell["difference"] == cell["ci_low"] == cell["ci_high"] == 0


@needs_kit
def test_kit_scorer_takes_group_weights_on_made_up_rows(corpus, test_root):
    rows = test_root / ROWS
    if not rows.is_dir():
        pytest.skip("no made-up benchmark rows beside the corpus")
    for cid in CIDS:
        task = load_task(corpus, cid)
        scorer = OM.KitScorer(".", rows, task.b)
        try:
            assert np.array_equal(scorer.group_ids, np.arange(task.G))
            pooled = [core.single_model_vectors(task.b, m) for m in range(task.b.M)]
            answers = OM.decode_all(task.b, pooled)
            point = scorer.own(answers)
            assert point.shape == (task.b.M,) and np.isfinite(point).all()
            for m in (0, 7):
                assert point[m] == pytest.approx(native.native_score(".", rows, task.b, pooled[m]), abs=1e-12)
            ones = scorer.own(answers, np.ones((1, task.G)))
            assert ones.shape == (1, task.b.M) and np.allclose(ones[0], point, rtol=0, atol=1e-12)
            draws = np.concatenate(list(stats.group_multiplicities(task.group_fold, 7, seed=[3, cid])))
            assert scorer.own(answers, draws).shape == (7, task.b.M)
            rng = np.random.default_rng(cid)
            for f in range(N_FOLDS):
                weights = OM.training_weights(task, f)
                values = scorer.own(answers, weights)[0]
                # weights of 0 and 1 give the metric of a scorer that holds the training requests only
                assert OM.largest_gap(values, scorer.own_subset(answers, np.flatnonzero(task.b.fold != f))) <= 1e-12
                # and the answers on the test fold do not enter: scramble them
                test = np.flatnonzero(task.field_fold == f)
                scrambled = OM.Answers(answers.winner.copy(), answers.loss, answers.hit.copy(), answers.yes)
                scrambled.winner[:, test] = rng.integers(0, task.b.width[test], size=(task.b.M, len(test)))
                scrambled.hit[:, test] = ~scrambled.hit[:, test]
                assert not np.array_equal(scrambled.winner, answers.winner)
                assert np.allclose(scorer.own(scrambled, weights)[0], values, rtol=0, atol=1e-12)
                assert not np.allclose(scorer.own(scrambled), point, rtol=0, atol=1e-12)      # control
        finally:
            scorer.close()


@needs_kit
def test_report_with_the_kit(corpus, test_root, tmp_path):
    rows = test_root / ROWS
    if not rows.is_dir():
        pytest.skip("no made-up benchmark rows beside the corpus")
    settings, path, results, features = _stored(corpus, tmp_path, CIDS, rows)
    content = OM.run(corpus, path, results, rows, ".", tmp_path / "followup", features=features, draws=19, cids=CIDS)
    assert content["scorer"] == OM.KitScorer.NAME and content["summary"]["benchmarks"] == len(CIDS)
    checks = content["summary"]["checks"]
    assert checks["policy_benchmark_cells"] == len(OM.POLICIES) * len(CIDS) and checks["fewest_defined_draws"] == 19
    for key in ("largest_stored_own_gap", "largest_stored_accuracy_gap", "largest_all_ones_weights_gap",
                "largest_training_subset_gap"):
        assert checks[key] <= OM.TOLERANCE
    assert content["summary"]["best_single_own"]["fallback_benchmarks"] == []
    blocks = content["benchmarks"]
    # the kit's metric is not the judged-field accuracy where it scores per case or by macro-F1
    assert any(abs(b["policies"][OM.REFERENCE]["own"] - b["policies"][OM.REFERENCE]["accuracy"]) > 1e-6
               for b in blocks.values())
    for block in blocks.values():
        for own, single in zip(block["training_own"]["best_single_own"], block["training_own"]["best_single"]):
            assert own >= single
        # one model for all folds can do no better than the hindsight model
        for name in (OM.REFERENCE, OM.OWN_REFERENCE):
            if len(set(block["models"][name])) == 1:
                assert block["policies"][name]["own"] <= block["hindsight"]["own"] + OM.TIE

"""Tests of the homogenization measures on the made-up corpus."""

from __future__ import annotations

from analysis import homogenization as H
from data.load import load_benchmark


def test_copied_model_agrees_whenever_both_are_wrong(corpus, planted):
    copy, original = planted["duplicate"]["copy"], planted["duplicate"]["of"]
    for cid in (9, 30, 57):
        bench = load_benchmark(corpus, cid)
        rows, panel = H.pair_rows(bench)
        pair = next(r for r in rows if (r["model_a"], r["model_b"]) == (original, copy))
        assert pair["both_wrong_agreement"] == 1.0 and pair["chance_agreement"] < 1.0
        error = pair["both_wrong"] / pair["judged_fields"]
        assert abs(pair["joint_failure_ratio"] - 1.0 / error) < 1e-9      # identical failures: P(F) / P(F)^2
        assert 0.0 <= panel["none_right"] <= 1.0


def test_binary_benchmark_has_no_wrong_option_choice(corpus):
    rows, _ = H.pair_rows(load_benchmark(corpus, 20))
    assert all(r["both_wrong_with_choice"] == 0 and r["both_wrong_agreement"] is None for r in rows)


def test_run_is_deterministic_and_grouped(corpus, tmp_path):
    a = H.run(corpus, tmp_path / "a")
    b = H.run(corpus, tmp_path / "b")
    assert a == b
    assert (tmp_path / "a" / "homogenization_pairs.csv").read_bytes() == (tmp_path / "b" / "homogenization_pairs.csv").read_bytes()
    suite = a["suites"]["all"]
    assert suite["same_organisation"]["joint_failure_ratio"] > suite["different_organisation"]["joint_failure_ratio"]
    assert suite["all_pairs"]["pairs"] == 231 and a["synthetic"] is True


def test_error_correlation_given_the_other_models(corpus, planted):
    """A copied model errs with its original in every stratum; the plain value equals a direct correlation."""
    import numpy as np
    copy, original = planted["duplicate"]["copy"], planted["duplicate"]["of"]
    bench = load_benchmark(corpus, 9)
    rows, panel = H.pair_rows(bench)
    hit, choice, _ = H.decisions(bench)
    wrong = (~hit).astype(float)
    for r in rows[:25]:
        a, b = wrong[:, r["model_a"]], wrong[:, r["model_b"]]
        if a.std() > 0 and b.std() > 0:
            assert abs(r["error_correlation"] - np.corrcoef(a, b)[0, 1]) < 1e-9
        if r["error_correlation_given_others"] is not None:
            assert -1 - 1e-9 <= r["error_correlation_given_others"] <= 1 + 1e-9
    pair = next(r for r in rows if (r["model_a"], r["model_b"]) == (original, copy))
    assert abs(pair["error_correlation"] - 1.0) < 1e-9 and abs(pair["error_correlation_given_others"] - 1.0) < 1e-9
    counts = H.modal_count(np.array([[1, 1, 2, 1], [0, 1, 2, 3], [5, 5, 5, 5]]))
    assert counts.tolist() == [3, 1, 4]
    if panel["none_right_with_choice"]:
        assert 0.0 <= panel["none_right_unanimous"] <= panel["none_right_three_quarters"] <= 1.0

"""Derived numbers that the paper's prose quotes, computed from the result files.

    python3 -B -m acl_resource.analysis.paper_facts --results DIR --diagnostics DIR --settings FILE --out FILE

Every value here is a simple function of rows.json, frontier.json, ablations.json, contrasts.json,
operating_points.json and the diagnostics: a difference, a ratio, a rank correlation, the best point of a curve,
an interval over resampled benchmarks.  They live in one file so that the prose never does arithmetic.  The
headline suite of the settings is used throughout.  Where the prose makes a statement that the numbers must
support ("on every suite", "at no setting"), the statement is checked here and the run fails if it is false.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

LOWER_IS_BETTER = {"log_loss": True, "accuracy": False}


def _suite_rows(path, suite):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [r for r in doc["rows"] if r["catalog_id"] is None and r["suite"] == suite]


def _by(rows, method):
    return {r["variant"]: r for r in rows if r["method"] == method}


def _mean(values):
    return math.fsum(values) / len(values)


def _over_benchmarks(gaps, metric, draws, seed):
    """Mean of per-benchmark differences with a percentile interval from resampling the benchmarks."""
    values = np.asarray(gaps, dtype=np.float64)
    rng = np.random.default_rng([seed, len(values)])
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    low, high = (float(q) for q in np.quantile(means, [0.025, 0.975]))
    sign = -1.0 if LOWER_IS_BETTER[metric] else 1.0
    better, worse = int((sign * values > 1e-12).sum()), int((sign * values < -1e-12).sum())
    return dict(mean=_mean(gaps), low=low, high=high, better=better, worse=worse, benchmarks=len(values),
                sign_test_p=float(binomtest(better, better + worse, 0.5).pvalue) if better + worse else None,
                excludes_zero=bool(low > 0 or high < 0))


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _homogeneity(diagnostics, cids, overall, draws, seed):
    """What the pair tables say beyond the suite means: spread over benchmarks, pooled rates, medians."""
    wanted = {str(c) for c in cids}
    by_bench = {}
    for r in _csv(diagnostics / "pairs.csv"):
        if r["catalog_id"] in wanted and r["error_correlation"]:
            same, base = r["same_organisation"] == "True", r["same_base_model"] == "True"
            group = "same_organisation" if same else (None if base else "other")
            if group:
                by_bench.setdefault(r["catalog_id"], {}).setdefault(group, []).append(float(r["error_correlation"]))
    gaps = [_mean(g["same_organisation"]) - _mean(g["other"]) for _, g in sorted(by_bench.items())
            if "same_organisation" in g and "other" in g]
    groups = overall["pair_groups"]
    stored = groups["same_organisation"]["error_correlation"] - groups["other"]["error_correlation"]
    if len(gaps) != len(wanted) or abs(_mean(gaps) - stored) > 1e-9:
        raise ValueError("the organisation gap of the pair table differs from the diagnostics summary")
    gap = _over_benchmarks(gaps, "accuracy", draws, seed)           # "better" = same organisation more alike
    level = overall["across_models"]["pairs"][0]["error_correlation"]
    voters = [float(r["effective_voters"]) for r in _csv(diagnostics / "headroom.csv")
              if r["catalog_id"] in wanted and r["effective_voters"]]
    agree = cases = 0.0
    ratios = {}
    for r in _csv(diagnostics / "homogenization_pairs.csv"):
        if r["catalog_id"] not in wanted:
            continue
        if r["both_wrong_agreement"]:
            k = int(r["both_wrong_with_choice"])
            agree, cases = agree + k * float(r["both_wrong_agreement"]), cases + k
        if r["joint_failure_ratio"]:
            ratios.setdefault(r["catalog_id"], []).append(float(r["joint_failure_ratio"]))
    models = len(overall["models"])
    parent = list(range(models))                      # organisations as the components of the same-organisation pairs

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    first = min(wanted, key=int)
    for r in _csv(diagnostics / "pairs.csv"):
        if r["catalog_id"] == first and r["same_organisation"] == "True":
            parent[find(int(r["model_i"]))] = find(int(r["model_j"]))
    sizes = sorted((sum(find(i) == root for i in range(models)) for root in {find(i) for i in range(models)}),
                   reverse=True)
    if gap["better"] != gap["benchmarks"]:
        raise ValueError("the prose says that same-organisation pairs are more alike on every benchmark")
    return dict(
        organisations=len(sizes), models_in_two_largest_organisations=sum(sizes[:2]),
        organisation_gap=gap["mean"], organisation_gap_low=gap["low"], organisation_gap_high=gap["high"],
        benchmarks_same_organisation_more_alike=gap["better"], benchmarks=gap["benchmarks"],
        organisation_gap_share_of_level=gap["mean"] / level,
        effective_voters_min=min(voters), effective_voters_max=max(voters),
        effective_voters_from_mean_correlation=models / (1 + (models - 1) * level),
        both_wrong_agreement_pooled=agree / cases, both_wrong_cases=int(cases),
        joint_failure_ratio_median=float(np.median([_mean(v) for v in ratios.values()])),
        joint_failure_ratio_benchmarks=len(ratios))


def run(results, diagnostics, settings_path, out):
    results, diagnostics = Path(results), Path(diagnostics)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    suite = settings["headline_suite"]
    rows = {r["method"]: r for r in _suite_rows(results / "rows.json", suite) if r["variant"] is None}
    frontier = _suite_rows(results / "frontier.json", suite)
    ablations = _suite_rows(results / "ablations.json", suite)
    diag = json.loads((diagnostics / "diagnostics.json").read_text(encoding="utf-8"))["suites"][suite]["overall"]
    facts = {"suite": suite, "benchmarks": rows["best_single"]["n_benchmarks"]}

    models = diag["models"]
    accuracy = [m["accuracy_first_max"] for m in models]
    cost = [m["usd_per_1k_requests"] for m in models]
    rho = spearmanr(accuracy, cost)
    facts["models"] = dict(
        n=len(models), accuracy_min=min(accuracy), accuracy_max=max(accuracy),
        cost_min=min(cost), cost_max=max(cost), cost_ratio=max(cost) / min(cost),
        spearman_accuracy_cost=float(rho.statistic), spearman_p=float(rho.pvalue),
        ece_min=min(m["ece"] for m in models), ece_max=max(m["ece"] for m in models))

    best, stack, exit_ = rows["best_single"], rows["stack"], rows["certified_exit"]
    if abs(stack["accuracy"] - exit_["accuracy"]) > 1e-12:
        raise ValueError("early exit must decode exactly like the stacked panel")
    facts["early_exit"] = dict(
        calls_saved_share=1 - exit_["calls_per_request"] / stack["calls_per_request"],
        cost_saved_share=1 - exit_["usd_per_1k_requests"] / stack["usd_per_1k_requests"],
        latency_ratio=exit_["latency_mean_s"] / stack["latency_mean_s"])
    facts["relative_to_best_single"] = {
        name: dict(cost_ratio=r["usd_per_1k_requests"] / best["usd_per_1k_requests"],
                   accuracy_points=100 * (r["accuracy"] - best["accuracy"]),
                   log_loss_difference=r["log_loss"] - best["log_loss"])
        for name, r in rows.items() if name != "best_single"}
    # The conclusion says in words: pooling loses and fitted weights gain a point at more than three times the cost.
    # Its clause on routers is guarded where the routers' numbers are (followup_facts).
    relative = facts["relative_to_best_single"]
    stacked = relative["stack_final_temperature"]
    if not (0.5 <= stacked["accuracy_points"] < 1.5 and 3 < stacked["cost_ratio"] < 4
            and relative["uniform_mean"]["accuracy_points"] < 0):
        raise ValueError("the conclusion's summary in words no longer matches the results")
    facts["oracle_gap_points"] = 100 * (rows["oracle"]["accuracy"] - best["accuracy"])
    facts["best_method_gap_points"] = 100 * (max(r["accuracy"] for n, r in rows.items() if n != "oracle")
                                             - best["accuracy"])

    curves = {}
    for method in ("ranked_top_k", "greedy", "random_panel"):
        points = _by(frontier, method)
        sizes = sorted(int(v.split("=")[1]) for v in points)
        loss = {k: points[f"k={k}"]["log_loss"] for k in sizes}
        best_k = min(sizes, key=lambda k: (loss[k], k))
        curves[method] = dict(best_size=best_k, best_log_loss=loss[best_k], log_loss_size_1=loss[sizes[0]],
                              log_loss_all=loss[sizes[-1]], all=sizes[-1],
                              first_size_within_0_001=min(k for k in sizes if loss[k] <= loss[best_k] + 0.001))
    facts["panel_size"] = curves

    routers = [m for m in ("knn_router", "knn_router_text") if _by(frontier, m)]
    control = _by(frontier, "single_by_cost")
    facts["router"] = {}
    for name in routers:
        points = _by(frontier, name)
        better = [v for v in points if points[v]["log_loss"] < control[v]["log_loss"]]
        cheaper_and_better = [v for v in points if points[v]["log_loss"] < control[v]["log_loss"]
                              and points[v]["usd_per_1k_requests"] <= control[v]["usd_per_1k_requests"]]
        facts["router"][name] = dict(
            grid_points=len(points), points_with_lower_loss_than_control=len(better),
            points_dominating_control=len(cheaper_and_better),
            largest_loss_gap_to_control=max(points[v]["log_loss"] - control[v]["log_loss"] for v in points),
            smallest_loss_gap_to_control=min(points[v]["log_loss"] - control[v]["log_loss"] for v in points))
    cheapest = max(control, key=lambda v: float(v.split("=")[1]))
    facts["cheap_single"] = dict(
        setting=cheapest, accuracy=control[cheapest]["accuracy"], log_loss=control[cheapest]["log_loss"],
        usd_per_1k_requests=control[cheapest]["usd_per_1k_requests"],
        cost_share_of_best_single=control[cheapest]["usd_per_1k_requests"] / best["usd_per_1k_requests"],
        accuracy_points_below_best_single=100 * (best["accuracy"] - control[cheapest]["accuracy"]))
    cascade = _by(frontier, "confidence_cascade")
    facts["cascade"] = dict(
        points=len(cascade),
        points_cheaper_than_best_single_with_lower_loss=sum(
            1 for r in cascade.values() if r["usd_per_1k_requests"] <= best["usd_per_1k_requests"]
            and r["log_loss"] < best["log_loss"]),
        lowest_log_loss=min(r["log_loss"] for r in cascade.values()))

    order = _by(ablations, "uniform_mean")
    facts["calibration_order"] = {k: order[k]["log_loss"] for k in order}
    budget = {}
    for method in ("best_single", "stack"):
        points = _by(ablations, method)
        budget[method] = {v.split("=")[1]: points[v]["log_loss"] for v in points}
    facts["label_budget"] = budget
    facts["no_target_labels"] = dict(
        transfer_single=_by(ablations, "transfer_single")[None]["log_loss"],
        transfer_temperature=_by(ablations, "transfer_temperature")[None]["log_loss"],
        uniform_mean=rows["uniform_mean"]["log_loss"])
    facts["label_budget"]["groups_where_selection_beats_transfer"] = min(
        (int(k) for k, v in budget["best_single"].items()
         if k != "all" and v < facts["no_target_labels"]["transfer_single"]), default=None)

    # order of the methods on the other suites, against the headline suite
    document = json.loads((results / "rows.json").read_text(encoding="utf-8"))["rows"]
    by_suite = {}
    for r in document:
        if r["catalog_id"] is None and r["variant"] is None:
            by_suite.setdefault(r["suite"], {})[r["method"]] = r
    names = sorted(by_suite[suite])
    held = set(settings["suites"][suite]["exclude"])
    inside = sorted(s for s in by_suite if set(settings["suites"][s]["exclude"]) > held)     # parts of the headline suite
    whole = sorted(s for s in by_suite if not settings["suites"][s]["exclude"])              # every benchmark
    rank = {other: {metric: float(spearmanr([by_suite[suite][m][metric] for m in names],
                                            [by_suite[other][m][metric] for m in names]).statistic)
                    for metric in ("log_loss", "accuracy")}
            for other in inside + whole}
    one_swap = 1 - 12 / (len(names) * (len(names) ** 2 - 1))          # Spearman after one swap of neighbours
    if any(rank[o]["accuracy"] < 1 - 1e-12 for o in rank):
        raise ValueError("the prose says that the accuracy order is the same on every suite; it is not")
    if any(abs(rank[o]["log_loss"] - 1) > 1e-9 and abs(rank[o]["log_loss"] - one_swap) > 1e-9 for o in rank):
        raise ValueError("the prose says that the log-loss order differs by one swap of neighbours at most")
    facts["sensitivity"] = dict(
        other_suites=len(inside), methods=len(names), rank_correlation=rank,
        min_rank_correlation_log_loss=min(rank[o]["log_loss"] for o in inside),
        suites_with_identical_log_loss_order=sum(rank[o]["log_loss"] > 1 - 1e-12 for o in inside),
        every_benchmark_suites=len(whole))
    transfer = facts["no_target_labels"]["transfer_single"]
    if not (budget["best_single"]["4"] > transfer and budget["stack"]["4"] > transfer
            and budget["best_single"]["12"] < transfer and budget["stack"]["12"] < transfer):
        raise ValueError("the prose on label budgets no longer matches the results")
    facts["label_budget"]["stack_12_minus_transfer"] = budget["stack"]["12"] - transfer

    draws, seed = settings["bootstrap"]["draws"], settings["bootstrap"]["seed"]
    excluded = set(settings["suites"][suite]["exclude"])
    bench = {(r["method"], r["catalog_id"]): r for r in document
             if r["catalog_id"] is not None and r["variant"] is None and r["catalog_id"] not in excluded}
    cids = sorted({c for _, c in bench})

    # the planned and the descriptive contrasts when benchmarks, not items, are resampled
    stored = {(r["contrast"], r["metric"]): r
              for r in json.loads((results / "contrasts.json").read_text(encoding="utf-8"))["rows"] if r["suite"] == suite}
    facts["over_benchmarks"] = {}
    for spec in settings["contrasts"]:
        if (spec["name"], "log_loss") not in stored:
            continue
        block = {}
        for metric in settings["contrast_metrics"]:
            gaps = [bench[spec["a"], c][metric] - bench[spec["b"], c][metric] for c in cids]
            block[metric] = _over_benchmarks(gaps, metric, draws, seed)
            if abs(block[metric]["mean"] - stored[spec["name"], metric]["difference"]) > 1e-12:
                raise ValueError(f"{spec['name']}: the benchmark rows do not give the stored difference")
        facts["over_benchmarks"][spec["name"]] = block
    intervals = facts["over_benchmarks"]
    including_zero = sorted(name for name, b in intervals.items() if not b["log_loss"]["excludes_zero"])
    facts["over_benchmarks_including_zero"] = including_zero
    if any(name.startswith("C") for name in including_zero):
        raise ValueError("the prose says that every planned contrast excludes zero when benchmarks are resampled: "
                         + str(including_zero))
    # The planned contrasts that the settings fixed and the ones that this pool allows.  The paper names the ones
    # that remain and says that the two others concern one method whose stored fit cannot be replayed here.
    planned = [spec for spec in settings["contrasts"] if spec.get("in_family", True)]
    computed = [spec["name"] for spec in planned if (spec["name"], "log_loss") in stored]
    missing = [spec for spec in planned if spec["name"] not in computed]
    facts["planned_contrasts"] = dict(fixed=len(planned), computed=len(computed), names=computed,
                                      not_computed=[spec["name"] for spec in missing])
    if settings.get("replay_stored_fits", True) is False and not (
            computed == ["C1", "C2", "C3", "C4", "C5", "C8", "C9"] and len(planned) == 9
            and all(spec["a"] == "tree_replay" for spec in missing)):
        raise ValueError("the prose on the planned contrasts no longer matches the results")
    facts["bootstrap"] = dict(draws=draws)
    facts["over_benchmarks_summary"] = dict(
        planned=sum(1 for name in intervals if name.startswith("C")),
        planned_excluding_zero=sum(1 for name, b in intervals.items()
                                   if name.startswith("C") and b["log_loss"]["excludes_zero"]),
        descriptive=sum(1 for name in intervals if name.startswith("D")),
        descriptive_excluding_zero=sum(1 for name, b in intervals.items()
                                       if name.startswith("D") and b["log_loss"]["excludes_zero"]),
        widest_ratio=max((b["log_loss"]["high"] - b["log_loss"]["low"])
                         / (stored[name, "log_loss"]["ci_high"] - stored[name, "log_loss"]["ci_low"])
                         for name, b in intervals.items()),
        narrowest_ratio=min((b["log_loss"]["high"] - b["log_loss"]["low"])
                            / (stored[name, "log_loss"]["ci_high"] - stored[name, "log_loss"]["ci_low"])
                            for name, b in intervals.items()))

    # the label budget on both metrics, and against the model chosen without target labels
    ablation_bench = {(r["method"], r["variant"], r["catalog_id"]): r
                      for r in json.loads((results / "ablations.json").read_text(encoding="utf-8"))["rows"]
                      if r["catalog_id"] is not None and r["catalog_id"] not in excluded}
    accuracy = {m: {v.split("=")[1]: r["accuracy"] for v, r in _by(ablations, m).items()}
                for m in ("best_single", "stack")}
    facts["label_budget"]["accuracy"] = accuracy
    facts["no_target_labels"]["transfer_single_accuracy"] = _by(ablations, "transfer_single")[None]["accuracy"]
    facts["label_budget"]["against_transfer"] = {
        f"{method}_{groups}": {
            metric: _over_benchmarks([ablation_bench[method, f"groups={groups}", c][metric]
                                      - ablation_bench["transfer_single", None, c][metric] for c in cids],
                                     metric, draws, seed)
            for metric in ("log_loss", "accuracy")}
        for method in ("best_single", "stack") for groups in ("4", "12", "48")}
    against = facts["label_budget"]["against_transfer"]
    four, twelve = against["best_single_4"], against["best_single_12"]
    if not (four["log_loss"]["mean"] > 0 and against["stack_4"]["log_loss"]["mean"] > 0
            and twelve["log_loss"]["high"] < 0 and twelve["accuracy"]["low"] > 0
            and against["stack_12"]["log_loss"]["high"] < 0 and against["stack_12"]["accuracy"]["low"] > 0
            and all(budget["stack"][k] < budget["best_single"][k] for k in budget["best_single"])):
        raise ValueError("the prose on label budgets (4 and 12 groups against no target labels) no longer holds")
    def floor_gap(key):
        return (_by(ablations, "best_single")["groups=12"][key] - _by(ablations, "transfer_single")[None][key])
    facts["label_budget"]["single_12_minus_transfer_by_floor"] = {
        "1e-15": floor_gap("log_loss"), "1e-6": floor_gap("log_loss_floor_1e-6"), "1e-3": floor_gap("log_loss_floor_1e-3")}
    facts["label_budget"]["all_groups_stack_minus_main_row"] = budget["stack"]["all"] - rows["stack"]["log_loss"]

    # the log-loss floor
    shift = {m: r["log_loss"] - r["log_loss_floor_1e-3"] for m, r in rows.items()}
    moved = sorted(m for m, v in shift.items() if abs(v) > 0.003)
    if moved != ["majority_vote", "random_single"]:
        raise ValueError("the prose names the two rows that depend on the log-loss floor; the set changed: " + str(moved))
    order = {key: sorted(rows, key=lambda m: (rows[m][key], m)) for key in ("log_loss", "log_loss_floor_1e-3")}
    facts["floor"] = dict(
        rows_moving_more_than_0_003=len(moved), largest_shift_of_other_rows=max(abs(v) for m, v in shift.items()
                                                                                 if m not in moved),
        rows_changing_rank=sum(order["log_loss"].index(m) != order["log_loss_floor_1e-3"].index(m) for m in rows),
        majority_vote_rank=1 + order["log_loss"].index("majority_vote"),
        majority_vote_rank_at_1e_3=1 + order["log_loss_floor_1e-3"].index("majority_vote"), rows=len(rows))

    # what the oracle selector bounds, and what the recall-only benchmarks contribute
    lower = sorted(m for m, r in rows.items() if m != "oracle" and r["log_loss"] < rows["oracle"]["log_loss"])
    facts["oracle"] = dict(rows_with_lower_log_loss=len(lower), log_loss=rows["oracle"]["log_loss"])
    if "stack_final_temperature" not in lower:
        raise ValueError("the prose says that stacking with a temperature has a lower log loss than the oracle selector")
    recall_only = set(settings["suites"]["no_recall_only"]["exclude"])
    without = next(s for s in inside if set(settings["suites"][s]["exclude"]) == held | recall_only)
    facts["recall_only"] = {
        name: {label: 100 * (by_suite[s][name]["accuracy"] - by_suite[s]["best_single"]["accuracy"])
               for label, s in (("headline", suite), ("without", without))}
        for name in ("combiner_replay", "stack_final_temperature") if name in rows}
    facts["recall_only"]["suite_without"] = without
    facts["recall_only"]["benchmarks_without"] = by_suite[without]["best_single"]["n_benchmarks"]
    facts["pool_like_for_like"] = (facts["calibration_order"]["calibrated_members_and_final_temperature"]
                                   - best["log_loss"])
    any_right = {int(r["catalog_id"]): float(r["at_least_one_share"]) for r in _csv(diagnostics / "headroom.csv")}
    above = {m: sorted(c for c in cids if bench[m, c]["accuracy"] > any_right[c] + 1e-12)
             for m in rows if m != "oracle"}
    facts["above_any_model_right"] = dict(
        methods=sum(bool(v) for v in above.values()),
        benchmarks=len({c for v in above.values() for c in v}),
        all_recall_only=all(set(v) <= recall_only for v in above.values()),
        combiner_benchmarks=len(above.get("combiner_replay", [])))
    if "combiner_replay" in rows:
        lead = facts["recall_only"]["combiner_replay"]
        if not (len(above["combiner_replay"]) == 1 and facts["above_any_model_right"]["all_recall_only"]
                and lead["without"] < lead["headline"] / 1.5):
            raise ValueError("the prose on the combiner and the recall-only benchmarks no longer holds")

    # the benchmarks' own metric: point estimates only, no interval exists
    facts["own_metric"] = {}
    for name in ("knn_router", "knn_router_text", "confidence_cascade", "stack_final_temperature", "certified_exit"):
        if name not in rows or rows[name]["native"] is None:
            continue
        gaps = [bench[name, c]["native"] - bench["best_single", c]["native"] for c in cids]
        facts["own_metric"][name] = dict(
            points_above_best_single=100 * (rows[name]["native"] - best["native"]),
            benchmarks_above=sum(g > 1e-12 for g in gaps), benchmarks_below=sum(g < -1e-12 for g in gaps),
            benchmarks_equal=sum(abs(g) <= 1e-12 for g in gaps),
            points_without_two_largest_gains=100 * _mean(sorted(gaps)[:-2]),
            two_largest_gains=sorted(c for _, c in sorted(zip(gaps, cids))[-2:]),
            share_of_gains_from_two_benchmarks=math.fsum(sorted(gaps)[-2:]) / math.fsum(g for g in gaps if g > 0),
            cost_share_of_best_single=rows[name]["usd_per_1k_requests"] / best["usd_per_1k_requests"])
    for name in routers:
        own = facts["own_metric"].get(name)
        if own and not (own["points_above_best_single"] > 0 > own["points_without_two_largest_gains"]
                        and own["benchmarks_below"] > own["benchmarks_above"]):
            raise ValueError("the prose on the routers' own-metric lead (carried by two benchmarks) no longer holds")
    if "knn_router" in facts["own_metric"] and facts["own_metric"]["knn_router"]["two_largest_gains"] != [38, 40]:
        raise ValueError("the prose names ACOS and iSarcasmEval as the benchmarks with the router's largest own-metric gains")

    # fixed settings of the frontier: one threshold or cost weight for all benchmarks
    def cheaper_or_equal(points):
        inside = [v for v in points if points[v]["usd_per_1k_requests"] <= best["usd_per_1k_requests"]]
        v = min(inside, key=lambda v: (points[v]["log_loss"], v))
        return dict(setting=v, value=float(v.split("=")[1]), log_loss=points[v]["log_loss"],
                    accuracy=points[v]["accuracy"],
                    cost_share_of_best_single=points[v]["usd_per_1k_requests"] / best["usd_per_1k_requests"],
                    log_loss_above_best_single=points[v]["log_loss"] - best["log_loss"],
                    accuracy_points_below_best_single=100 * (best["accuracy"] - points[v]["accuracy"]))

    def dominated(point, others):
        return any(o["usd_per_1k_requests"] <= point["usd_per_1k_requests"] and o["log_loss"] <= point["log_loss"]
                   for o in others)

    facts["cascade"]["best_fixed_threshold_within_single_cost"] = cheaper_or_equal(cascade)
    last = max(cascade, key=lambda v: float(v.split("=")[1]))
    facts["cascade"]["every_stage"] = dict(
        log_loss=cascade[last]["log_loss"], calls_per_request=cascade[last]["calls_per_request"],
        cost_share_of_best_single=cascade[last]["usd_per_1k_requests"] / best["usd_per_1k_requests"])
    facts["cascade"]["points_dominated_by_a_control_setting"] = sum(
        dominated(r, control.values()) for r in cascade.values())
    facts["cascade"]["points_with_lower_loss_than_best_single"] = sum(
        r["log_loss"] < best["log_loss"] for r in cascade.values())
    for name in routers:
        points = _by(frontier, name)
        zero = min(points, key=lambda v: float(v.split("=")[1]))
        facts["router"][name].update(
            best_fixed_weight_within_single_cost=cheaper_or_equal(points),
            settings_dominated_by_a_control_setting=sum(dominated(r, control.values()) for r in points.values()),
            settings_dominating_a_control_setting=sum(dominated(c, [r]) for r in points.values()
                                                      for c in control.values()),
            settings_with_lower_loss_than_best_single=sum(r["log_loss"] < best["log_loss"] for r in points.values()),
            no_cost_weight=dict(log_loss=points[zero]["log_loss"], accuracy=points[zero]["accuracy"],
                                cost_share_of_best_single=points[zero]["usd_per_1k_requests"]
                                / best["usd_per_1k_requests"]))
    facts["control"] = dict(
        best_fixed_weight_within_single_cost=cheaper_or_equal(control),
        settings_with_lower_loss_than_best_single=sum(r["log_loss"] < best["log_loss"] for r in control.values()))
    if facts["cascade"]["points_with_lower_loss_than_best_single"] or any(
            facts["router"][n]["settings_with_lower_loss_than_best_single"] for n in routers):
        raise ValueError("the prose says that no fixed setting of the router or the cascade has a lower log loss "
                         "than the best model per benchmark")
    facts["relative_to_stacking"] = {
        name: dict(cost_share=rows[name]["usd_per_1k_requests"] / rows["stack_final_temperature"]["usd_per_1k_requests"],
                   calls_share=rows[name]["calls_per_request"] / rows["stack_final_temperature"]["calls_per_request"])
        for name in ("greedy", "certified_exit") if name in rows}
    if not (facts["early_exit"]["latency_ratio"] > 1 and 0 < facts["early_exit"]["cost_saved_share"] < 1
            and facts["relative_to_stacking"]["greedy"]["cost_share"] < 1):
        raise ValueError("the prose says that early exit and greedy selection cost less than the stacked panel, and "
                         "that early exit is slower")

    facts["homogeneity"] = _homogeneity(diagnostics, cids, diag, draws, seed)

    # statements of the appendix on further checks
    summary = json.loads((diagnostics / "homogenization.json").read_text(encoding="utf-8"))["suites"][suite]
    if "error_correlation_given_others" in summary["all_pairs"]:
        given = {g: summary[g]["error_correlation_given_others"] for g in ("all_pairs", "same_organisation",
                                                                         "different_organisation")}
        if not (given["all_pairs"] < summary["all_pairs"]["error_correlation"] / 3
                and given["same_organisation"] > 2 * given["different_organisation"] > 0):
            raise ValueError("the prose on error correlation given item difficulty no longer holds")
    path = results / "fold_redraws.json"
    if path.is_file():
        redrawn = json.loads(path.read_text(encoding="utf-8"))["suites"][suite]
        single = redrawn["methods"]["best_single"]["log_loss"]
        if not (redrawn["contrasts_keeping_sign"] == redrawn["contrasts_compared"] and single["low"] > single["frozen"]):
            raise ValueError("the prose on redrawn folds no longer holds")
    path = results / "open_track.json"
    if path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
        subset = document["suites"][suite]["methods"]
        best_open = subset["best_single"]
        if not (subset["uniform_mean"]["log_loss"] > best_open["log_loss"]
                > max(subset["stack_final_temperature"]["log_loss"], subset["greedy"]["log_loss"])
                and subset["uniform_mean"]["accuracy"] < best_open["accuracy"]
                < min(subset["stack_final_temperature"]["accuracy"], subset["greedy"]["accuracy"])):
            raise ValueError("the prose on the open subset no longer holds")
        if not all(subset[m]["log_loss"] > rows[m]["log_loss"] and subset[m]["accuracy"] < rows[m]["accuracy"]
                   for m in document["methods"]):
            raise ValueError("the prose says that every method is worse on the open subset than on all models")
        facts["open_subset"] = dict(
            models=document["models"], benchmarks=document["suites"][suite]["benchmarks"],
            points_over_best_single={m: 100 * (subset[m]["accuracy"] - best_open["accuracy"])
                                     for m in ("uniform_mean", "stack_final_temperature", "greedy")},
            points_over_best_single_all_models={m: 100 * (rows[m]["accuracy"] - best["accuracy"])
                                                for m in ("uniform_mean", "stack_final_temperature", "greedy")})

    # the selection rule of the router and the cascade, and the same policies tuned without a cost cap
    path = results / "operating_points.json"
    if path.is_file():
        ops = json.loads(path.read_text(encoding="utf-8"))["suites"][suite]
        block = {}
        for name, by_rule in ops["points"].items():
            block[name] = {rule: dict(cost_share_of_best_single=p["usd_per_1k_requests"] / best["usd_per_1k_requests"],
                                      own_metric_points_above_best_single=(
                                          None if p["native"] is None or best["native"] is None
                                          else 100 * (p["native"] - best["native"])))
                           for rule, p in by_rule.items()}
            chosen = ops["selection"][name]
            block[name]["cap_changed_choice_share"] = chosen["cap_changed_choice"] / chosen["cells"]
            block[name]["none_within_budget_share"] = chosen["none_within_budget"] / chosen["cells"]
        for name in ("knn_router", "confidence_cascade", "knn_router_text"):
            if name in block and ops["points"][name]["capped"]["log_loss"] != rows[name]["log_loss"]:
                raise ValueError(f"{name}: the capped operating point is not the row of the main table")
        effects = ops["contrasts"]
        for name in ("router_capped_vs_single", "text_router_capped_vs_single", "cascade_capped_vs_single",
                     "router_uncapped_vs_single", "router_capped_vs_control", "router_uncapped_vs_control",
                     "text_router_capped_vs_control"):
            if name in effects and not effects[name]["log_loss"]["ci_low"] > 0:
                raise ValueError(f"the prose says that the interval of {name} lies above zero; it does not")
        for name in ("text_router_uncapped_vs_single", "cascade_uncapped_vs_single", "text_router_uncapped_vs_control"):
            if name in effects and not effects[name]["log_loss"]["ci_low"] < 0 < effects[name]["log_loss"]["ci_high"]:
                raise ValueError(f"the prose says that the interval of {name} includes zero; it does not")
        control = effects["control_capped_vs_single"]["log_loss"]
        if not control["ci_low"] < 0 < control["ci_high"]:
            raise ValueError("the prose says that the control's log loss cannot be told from the best single model's")
        uncapped = {name: e for name, e in effects.items()            # the routers and the cascade, not the control
                    if name.endswith("_uncapped_vs_single") and not name.startswith("control")}
        accuracy = effects["cascade_uncapped_vs_single"]["accuracy"]
        by_features = ops["router_by_features"]
        own = [p["native"] for name in routers for p in ops["points"][name].values()]
        if not (all(e["log_loss"]["difference"] > 0 and e["accuracy"]["difference"] < 0 for e in uncapped.values())
                and accuracy["ci_low"] < 0 < accuracy["ci_high"]
                and all(block[name]["uncapped"]["cost_share_of_best_single"] < 1 for name in routers)
                and block["confidence_cascade"]["uncapped"]["cost_share_of_best_single"] > 1
                and ops["selection"]["confidence_cascade"]["cells_by_stages"][0] == 0
                and ops["selection"]["confidence_cascade"]["none_within_budget"] == 0
                and abs(by_features["two_or_more"]["capped_log_loss_difference"]) < 0.005
                and by_features["at_most_one"]["capped_log_loss_difference"] > 0.01
                # the cascade on the mean confidence "behaves alike": clearly worse under the cap, and below the
                # reference in accuracy without it (the prose gives both differences in points)
                and effects["mean_cascade_capped_vs_single"]["log_loss"]["ci_low"] > 0
                and effects["mean_cascade_capped_vs_single"]["accuracy"]["difference"] < 0
                and effects["mean_cascade_uncapped_vs_single"]["accuracy"]["difference"] < 0
                and all(v is not None and v > best["native"] for v in own)):
            raise ValueError("the appendix on operating points makes a statement that the results no longer support")
        block["router_cost_relative_to_control"] = {
            rule: ops["points"]["knn_router"][rule]["usd_per_1k_requests"]
            / ops["points"]["single_by_cost"][rule]["usd_per_1k_requests"] for rule in ("capped", "uncapped")}
        block["cascade_gap_closed_by_lifting_the_cap"] = 1 - (
            effects["cascade_uncapped_vs_single"]["log_loss"]["difference"]
            / effects["cascade_capped_vs_single"]["log_loss"]["difference"])
        facts["operating_points"] = block

    restyle = _restyle(results, diagnostics, settings, suite, rows, facts)
    if restyle:
        facts["restyle"] = restyle

    Path(out).write_text(json.dumps(facts, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return facts


COMBINATIONS = ("uniform_mean", "majority_vote", "skill_pool", "uniform_temperature", "stack",
                "stack_final_temperature", "greedy", "certified_exit")       # combination methods fitted here
ADAPTIVE = ("knn_router", "knn_router_text", "single_by_cost", "confidence_cascade")


def _restyle(results, diagnostics, settings, suite, rows, facts=None):
    """Facts behind the bold findings of the restyled paper, with a guard for every finding.

    Uses hardness.json, benchmark_metrics.json and router_study.json when they exist (all added after the first
    run); returns None when they do not.
    """
    paths = {name: results / f"{name}.json" for name in ("hardness", "benchmark_metrics", "router_study")}
    if not all(path.is_file() for path in paths.values()):
        return None
    hard = json.loads(paths["hardness"].read_text(encoding="utf-8"))["suites"]
    bm = json.loads(paths["benchmark_metrics"].read_text(encoding="utf-8"))["suites"][suite]
    study = json.loads(paths["router_study"].read_text(encoding="utf-8"))["suites"][suite]
    out = {}

    def check(condition, finding):
        if not condition:
            raise ValueError("the restyled paper states a finding that the results no longer support: " + finding)

    # Complementary per request, less per task area.
    models = _csv(diagnostics / "models.csv")
    exclude = set(settings["suites"][suite]["exclude"])
    accuracy = {(int(r["catalog_id"]), int(r["model_index"])): float(r["accuracy_first_max"]) for r in models
                if int(r["catalog_id"]) not in exclude}
    area_of = {int(r["catalog_id"]): r["area"] for r in models}
    cids = sorted({c for c, _ in accuracy})
    ids = sorted({m for _, m in accuracy})
    winners = sorted({max(ids, key=lambda m: (accuracy[c, m], -m)) for c in cids})
    groups = {}
    for r in _csv(diagnostics / "pairs.csv"):
        if r["same_organisation"].lower() == "true":
            i, j = int(r["model_i"]), int(r["model_j"])
            gi, gj = groups.setdefault(i, {i}), groups.setdefault(j, {j})
            if gi is not gj:
                gi |= gj
                for m in gi:
                    groups[m] = gi
    organisation = {m: min(groups.get(m, {m})) for m in ids}
    leaders = {}
    for area in sorted({area_of[c] for c in cids}):
        members = [c for c in cids if area_of[c] == area]
        means = {m: sum(accuracy[c, m] for c in members) / len(members) for m in ids}
        leaders[area] = max(ids, key=lambda m: (means[m], -m))
    by_org = {}
    for area, m in leaders.items():
        by_org.setdefault(organisation[m], []).append(m)
    top_org = max(by_org.values(), key=len)
    out["complementarity"] = dict(benchmark_winners=len(winners), areas=len(leaders),
                                  areas_led_by_one_organisation=len(top_org),
                                  models_leading_those_areas=len(set(top_org)))
    out["complementarity"]["models"] = len(ids)
    check(len(winners) >= len(ids) / 2, "more than half of the models are the most accurate on some benchmark")

    # Pooling everything loses; fitted weights win by little.
    methods = bm["methods"]
    check(methods["uniform_mean"]["gain_b"] < 0 < methods["stack_final_temperature"]["gain_b"] < 0.02,
          "pooling loses, fitted weights win by little")

    # Hard fields: every combination fitted here is below the best single model; the replayed combiner only with
    # the recall-only benchmarks.
    few = hard[suite]["few_right"]["accuracy"]
    below = [m for m in COMBINATIONS if m in few]
    check(all(few[m] < few["best_single"] for m in below), "combination is below the best single on hard fields")
    no_recall = hard.get(suite + "_no_recall_only", {}).get("few_right", {}).get("accuracy", {})
    if "combiner_replay" in few:
        check(few["combiner_replay"] > few["best_single"] and "combiner_replay" in no_recall
              and no_recall["combiner_replay"] <= no_recall["best_single"],
              "the replayed combiner exceeds the best single on hard fields only with the recall-only benchmarks")
    # Where the oracle's lead over B lies: its share by the number of models right on a field.
    bins = hard[suite]["bins"]
    M = len(bins) - 1
    lead = [b["share"] * ((b["accuracy"].get("oracle") or 0) - (b["accuracy"].get("best_single") or 0)) for b in bins]
    total = math.fsum(lead)
    minority = [k for k in range(M + 1) if 2 * k < M]
    out["hard_fields"] = dict(combinations_below_best_single=len(below),
                              gap_stacking=few["best_single"] - few["stack_final_temperature"],
                              lead_points=total, lead_share_one_to_three=math.fsum(lead[1:4]) / total,
                              lead_share_minority=math.fsum(lead[k] for k in minority) / total,
                              minority_max_right=max(minority), minority_field_share=math.fsum(bins[k]["share"]
                                                                                              for k in minority))

    # More models add less than better-chosen ones; from some size on random subsets cover more.
    sub = hard[suite]["subsets"]
    top, random, low = sub["top_k"], sub["random_mean"], sub["random_low"]
    M = len(top)
    crossing = next((k for k in range(1, M + 1) if all(random[j] > top[j] for j in range(k - 1, M - 1))), M)
    below_band = [k for k in range(1, M) if top[k - 1] < low[k - 1]]
    # The strongest models are alike: their errors correlate more than those of all pairs.
    ranked_models = sorted({int(r["model_index"]) for r in models},
                           key=lambda m: -sum(accuracy[c, m] for c in cids))[:5]
    corr = {}
    for r in _csv(diagnostics / "pairs.csv"):
        if int(r["catalog_id"]) in exclude or not r["error_correlation"]:
            continue
        corr.setdefault((int(r["model_i"]), int(r["model_j"])), []).append(float(r["error_correlation"]))
    mean_pair = {k: sum(v) / len(v) for k, v in corr.items()}
    top_pairs = [v for (i, j), v in mean_pair.items() if i in ranked_models and j in ranked_models]
    out["top_models_error_correlation"] = sum(top_pairs) / len(top_pairs)
    out["all_pairs_error_correlation"] = sum(mean_pair.values()) / len(mean_pair)
    check(out["top_models_error_correlation"] > out["all_pairs_error_correlation"],
          "the strongest models are more alike than the average pair")
    out["subsets"] = dict(random_above_top_from=crossing, gain_first_three=top[2] - top[0], gain_all=top[M - 1] - top[0],
                          largest_random_margin=max(random[j] - top[j] for j in range(crossing - 1, M - 1)),
                          gain_last_eleven=top[M - 1] - top[10],
                          top_below_band_from=below_band[0] if below_band else None,
                          top_below_band_to=below_band[-1] if below_band else None)

    # Routing beats the top model T at a fraction of its cost, but not B; no adaptive policy reaches B.
    policies, refs = bm["policies"], bm["references"]
    check(methods["knn_router"]["avg_acc"] > refs["T"]["accuracy"], "the capped router is more accurate than T")
    ranked = sorted(bm["single_models"], key=lambda s: (-s["accuracy"], s["model_index"]))
    second = ranked[1]
    router_cost = rows["knn_router"]["usd_per_1k_requests"]
    check(router_cost > refs["T"]["usd_per_1k_requests"] and rows["best_single"]["usd_per_1k_requests"] > router_cost,
          "the capped router costs more than Best Single and less than the best model per benchmark")
    out["routing"] = dict(router_cost_share_of_top=router_cost / refs["T"]["usd_per_1k_requests"],
                          router_cost_share_of_second=router_cost / second["usd_per_1k_requests"],
                          second_accuracy=second["accuracy"], top_minus_second_points=ranked[0]["accuracy"]
                          - second["accuracy"],
                          router_points_over_top=100 * (methods["knn_router"]["avg_acc"] - refs["T"]["accuracy"]),
                          reference_points_over_top=100 * (rows["best_single"]["accuracy"] - refs["T"]["accuracy"]),
                          reference_cost_ratio_to_top=rows["best_single"]["usd_per_1k_requests"]
                          / refs["T"]["usd_per_1k_requests"])

    # The representation of the request barely matters.
    points = study["points"]
    names = [p for p in points if p.startswith("router_") and p not in ("router_control", "router_numeric_all")]
    losses = [points[p]["capped"]["log_loss"] for p in names]
    accs = [points[p]["capped"]["accuracy"] for p in names]
    recall3 = [study["recall"][p.removeprefix("router_")]["capped"]["few_right"]["at_3"] for p in names
               if p.removeprefix("router_") in study["recall"]]
    check(max(losses) - min(losses) < 0.015 and min(losses) > rows["best_single"]["log_loss"]
          and max(accs) < rows["best_single"]["accuracy"], "the representation of the request barely matters")
    if "router_numeric_all" in points:
        nested = points["router_numeric_all"]["capped"]
        control = points["router_control"]["capped"]
        check(abs(nested["log_loss"] - control["log_loss"]) < 0.001 and nested["log_loss"] > rows["best_single"]["log_loss"],
              "with every fitting request on its grid the router matches its control and stays behind B_d")
        out["nested"] = dict(log_loss=nested["log_loss"], accuracy=nested["accuracy"],
                             log_loss_minus_control=nested["log_loss"] - control["log_loss"],
                             cells=study.get("nested_cells"), cells_with_all=study.get("nested_cells_with_all"))
    out["representations"] = dict(count=len(names), log_loss_spread=max(losses) - min(losses),
                                  log_loss_min=min(losses), log_loss_max=max(losses),
                                  accuracy_min=min(accs), accuracy_max=max(accs),
                                  recall3_min=min(recall3), recall3_max=max(recall3))

    # Latency separates models of similar cost and accuracy.
    single = bm["single_models"]
    pairs = []
    for i, a in enumerate(single):
        for c in single[i + 1:]:
            cost_gap = abs(a["usd_per_1k_requests"] - c["usd_per_1k_requests"]) / max(a["usd_per_1k_requests"],
                                                                                      c["usd_per_1k_requests"])
            if cost_gap <= 0.10 and abs(a["accuracy"] - c["accuracy"]) <= 0.01:
                pairs.append(max(a["latency_p50_s"], c["latency_p50_s"]) / min(a["latency_p50_s"], c["latency_p50_s"]))
    check(pairs and max(pairs) > 1.3, "models of similar cost and accuracy differ in latency")
    rho = spearmanr([x["accuracy"] for x in single], [x["latency_p50_s"] for x in single])
    out["latency_accuracy_spearman"] = float(rho.statistic)
    out["latency_accuracy_spearman_p"] = float(rho.pvalue)
    check(abs(rho.statistic) < 0.3, "latency is not clearly related to accuracy")
    out["latency"] = dict(similar_pairs=len(pairs), largest_ratio=max(pairs), cost_tolerance=0.10,
                          accuracy_tolerance=0.01, p50_min=min(s["latency_p50_s"] for s in single),
                          p50_max=max(s["latency_p50_s"] for s in single))

    # Who traces the Pareto frontier.
    members = {}
    for _, _, name, _ in bm["frontier"]:
        key = "single_models" if name.startswith("model_") else name
        members[key] = members.get(key, 0) + 1
    check({"single_by_cost", "knn_router_text", "greedy", "certified_exit"} <= set(members),
          "the control, the text router, greedy panels and early exit lie on the frontier")
    out["frontier"] = members

    # Early exit saves cost, not latency; the router's lead on the own metric comes from two benchmarks.
    exit_latency, panel_latency = rows["certified_exit"]["latency_mean_s"], rows["stack_final_temperature"]["latency_mean_s"]
    check(abs(exit_latency - panel_latency) / panel_latency < 0.1, "early exit has about the latency of the full panel")
    own = (facts or {}).get("own_metric", {}).get("knn_router", {})
    if own:
        check(own["points_above_best_single"] > 0 and own["points_without_two_largest_gains"] < 0,
              "the router's lead on the benchmarks' own metrics is carried by two benchmarks")
    out["early_exit_latency_ratio"] = exit_latency / panel_latency

    # Early exit lies on the frontier.
    check(policies["certified_exit"]["pareto_dist"] == 0, "early exit lies on the frontier")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Derived numbers for the paper's prose")
    for name in ("results", "diagnostics", "settings", "out"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args(argv)
    facts = run(a.results, a.diagnostics, a.settings, a.out)
    print(json.dumps({k: facts[k] for k in ("models", "early_exit", "router", "cheap_single", "cascade")},
                     indent=1, sort_keys=True)[:1800])
    return 0


if __name__ == "__main__":
    sys.exit(main())

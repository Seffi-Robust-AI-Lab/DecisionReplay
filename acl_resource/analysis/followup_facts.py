"""The numbers that the paper takes from the follow-up analyses, under plain keys, with guards.

    python3 -B -m acl_resource.analysis.followup_facts --followup DIR [--suite main35]

The follow-up analyses of 23 November 2026 (``headroom_anatomy``, ``pool_composition``,
``router_refit_features``, ``router_unknown``, ``router_control_contrast``, ``router_full_text``,
``own_metric_study``) write their results to one folder, under keys that LaTeX cannot address (they hold spaces
and parentheses).  This module copies the values that the paper quotes into ``DIR/followup_facts.json`` under
plain keys, so that the prose reads them with ``\\num[...]{fu....}`` like every other number.  It computes nothing
new.  It also checks every qualitative statement that the prose makes about these values (an interval that
includes or excludes zero, a sign, an order), and stops when one no longer holds, as ``paper_facts`` does for
the main results.  Added after the first run, on 23 November 2026.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import run_experiments as RX


def _load(folder, name):
    return json.loads((Path(folder) / name).read_text(encoding="utf-8"))


def _interval(cell):
    """Difference and both intervals of a contrast block with within_* and over_* bounds."""
    return dict(difference=cell["difference"], within_low=cell["within_low"], within_high=cell["within_high"],
                over_low=cell["over_low"], over_high=cell["over_high"])


def _includes_zero(low, high):
    return low <= 0 <= high


def _check(condition, statement):
    if not condition:
        raise ValueError("a statement of the paper no longer holds: " + statement)


def _excludes_zero(cell, bounds="within"):
    return not _includes_zero(cell[bounds + "_low"], cell[bounds + "_high"])


def collect(folder, suite="main35"):
    h = _load(folder, "headroom_anatomy.json")["suites"][suite]
    p = _load(folder, "pool_composition.json")["suites"][suite]
    f = _load(folder, "router_refit_features.json")["suites"][suite]
    u = _load(folder, "router_unknown.json")["suites"][suite]
    c = _load(folder, "router_control_contrast.json")["suites"][suite]["ridge_minus_control"]
    t = _load(folder, "router_full_text.json")["suites"][suite]
    o = _load(folder, "own_metric.json")
    if o["suite"] != suite:
        raise ValueError("own_metric.json was made for another suite")
    s = o["summary"]

    facts = {}
    # the questions that only a few models answer correctly
    facts["few_right"] = dict(confidence_right=h["few_right"]["mean_confidence_of_right_models"],
                              confidence_wrong=h["few_right"]["mean_confidence_of_wrong_models"])
    _check(facts["few_right"]["confidence_right"] < facts["few_right"]["confidence_wrong"],
           "right models are less confident on the questions that few models get right")

    pts = p["points"]
    top, stack = pts["top_k_validated"], pts["stack_final_temperature"]
    facts["pool"] = dict(
        top_k_accuracy=top["accuracy"], top_k_log_loss=top["log_loss"],
        top_k_cost_share_of_stacking=top["usd_per_1k_requests"] / stack["usd_per_1k_requests"],
        top_k_calls=top["calls_per_request"],
        top_k_minus_stacking=_interval(p["contrasts"]["top_k_validated - stack_final_temperature"]["accuracy"]),
        top_k_minus_stacking_log_loss=p["contrasts"]["top_k_validated - stack_final_temperature"]["log_loss"]["difference"],
        balanced_minus_uniform=_interval(p["contrasts"]["pool_by_organisation - uniform_mean"]["accuracy"]))
    tk = facts["pool"]["top_k_minus_stacking"]
    _check(_includes_zero(tk["within_low"], tk["within_high"]) and _includes_zero(tk["over_low"], tk["over_high"]),
           "the validated top-k pool is level with stacking in accuracy")
    _check(facts["pool"]["top_k_minus_stacking_log_loss"] > 0, "stacking keeps a lead in log loss")
    bu = facts["pool"]["balanced_minus_uniform"]
    _check(bu["within_high"] < 0 and bu["over_high"] < 0, "the pool balanced by organisation is worse")

    tf = f["tfidf"]
    facts["refit"] = dict(tfidf_log_loss=tf["points"]["fit_folds_only"]["log_loss"],
                          tfidf_accuracy=tf["points"]["fit_folds_only"]["accuracy"],
                          tfidf_usd=tf["points"]["fit_folds_only"]["usd_per_1k_requests"],
                          tfidf_difference=tf["log_loss_fit_only_minus_all"]["delta"],
                          tfidf_low=tf["log_loss_fit_only_minus_all"]["ci_low"],
                          tfidf_high=tf["log_loss_fit_only_minus_all"]["ci_high"])
    _check(facts["refit"]["tfidf_low"] > 0, "the TF-IDF router is worse when fitted on the fitting folds only")

    up, uc = u["points"], u["contrasts"]
    across = up["knn_across_minilm"]["uncapped"]
    within = up["ridge_within_minilm"]["capped"]
    best = u["stored"]["best_single"]["None"]
    facts["unknown"] = dict(
        across_accuracy=across["accuracy"], global_accuracy=up["single_global"]["None"]["accuracy"],
        across_minus_global=_interval(uc["knn_across_minilm.uncapped - single_global"]["accuracy"]),
        across_minus_text=_interval(uc["knn_across_minilm.uncapped - knn_router_text"]["accuracy"]),
        across_minus_best=_interval(uc["knn_across_minilm.uncapped - best_single"]["accuracy"]),
        same_benchmark_share=u["neighbours"]["knn_across_minilm"]["same_benchmark_share"],
        chance_share=u["neighbours"]["knn_across_minilm"]["chance_share"],
        ridge_accuracy=within["accuracy"], ridge_log_loss=within["log_loss"],
        ridge_cost_share=within["usd_per_1k_requests"] / best["usd_per_1k_requests"],
        ridge_uncapped_cost_share=up["ridge_within_minilm"]["uncapped"]["usd_per_1k_requests"] / best["usd_per_1k_requests"],
        ridge_minus_best=_interval(uc["ridge_within_minilm.capped - best_single"]["accuracy"]),
        ridge_minus_best_log_loss=_interval(uc["ridge_within_minilm.capped - best_single"]["log_loss"]),
        ridge_minus_text=_interval(uc["ridge_within_minilm.capped - knn_router_text"]["accuracy"]))
    un = facts["unknown"]
    _check(un["across_minus_global"]["within_low"] > 0 and un["across_minus_global"]["over_low"] > 0,
           "the router without the label beats one global model")
    _check(_includes_zero(un["across_minus_text"]["within_low"], un["across_minus_text"]["within_high"])
           and _includes_zero(un["across_minus_text"]["over_low"], un["across_minus_text"]["over_high"]),
           "without the label the router is level with the text router in accuracy")
    _check(un["across_minus_best"]["within_high"] < 0 and _includes_zero(un["across_minus_best"]["over_low"],
                                                                         un["across_minus_best"]["over_high"]),
           "without the label the router is below the best model per benchmark when source groups are resampled, "
           "not when benchmarks are")
    _check(un["same_benchmark_share"] > 0.75, "most neighbours come from the query's own benchmark")
    # the ridge router under the cap: level in accuracy, cheaper, and ahead in log loss within benchmarks only
    cell = un["ridge_minus_best"]
    _check(not _excludes_zero(cell, "within") and not _excludes_zero(cell, "over"),
           "under the cap the ridge router is level with the best model per benchmark in accuracy")
    cell = un["ridge_minus_best_log_loss"]
    _check(cell["within_high"] < 0 and not _excludes_zero(cell, "over"),
           "under the cap the ridge router has the lower log loss when source groups are resampled, not when "
           "benchmarks are")
    _check(un["ridge_cost_share"] < 1 and un["ridge_uncapped_cost_share"] < 1,
           "the ridge router is cheaper than the best model per benchmark, with and without the cap")
    capped_across = uc["knn_across_minilm.capped_bd - knn_router_text"]
    un["across_capped_minus_text"] = _interval(capped_across["accuracy"])
    un["across_capped_minus_text_log_loss"] = capped_across["log_loss"]["difference"]
    _check(_includes_zero(un["across_capped_minus_text"]["within_low"], un["across_capped_minus_text"]["within_high"])
           and _includes_zero(un["across_capped_minus_text"]["over_low"], un["across_capped_minus_text"]["over_high"]),
           "under the same cap the router without the label is level with the text router in accuracy")
    _check(un["across_capped_minus_text_log_loss"] > 0, "the router without the label is worse in log loss")
    un["ridge_uncapped_minus_best_log_loss"] = _interval(uc["ridge_within_minilm.uncapped - best_single"]["log_loss"])
    un["ridge_uncapped_minus_best"] = _interval(uc["ridge_within_minilm.uncapped - best_single"]["accuracy"])
    _check(un["ridge_minus_text"]["within_low"] > 0 and un["ridge_minus_text"]["over_low"] > 0,
           "the ridge router beats the paper's text router")

    capped = c["capped"]
    facts["control"] = dict(ridge_minus_control=_interval(capped["accuracy"]),
                            log_loss_better=capped["log_loss"]["benchmarks_better"],
                            log_loss_worse=capped["log_loss"]["benchmarks_worse"],
                            cells_one_model=capped["cells_with_one_model"], cells=capped["cells"])
    rc = facts["control"]["ridge_minus_control"]
    _check(_includes_zero(rc["over_low"], rc["over_high"]), "the ridge router is level with the control over benchmarks")
    _check(facts["control"]["log_loss_worse"] > facts["control"]["log_loss_better"],
           "the ridge router is worse than the control in log loss on more benchmarks")

    tc = t["contrasts"]
    facts["full_text"] = dict(
        knn_minus_stored=_interval(tc["knn_new - knn_old (capped)"]["accuracy"]),
        knn_minus_best=_interval(tc["knn_new - best_single (capped)"]["accuracy"]),
        knn_minus_control=_interval(tc["knn_new - control (capped)"]["accuracy"]),
        ridge_minus_stored=_interval(tc["ridge_new - ridge_old (capped)"]["accuracy"]),
        knn_accuracy=t["points"]["knn_new"]["capped"]["accuracy"],
        knn_log_loss=t["points"]["knn_new"]["capped"]["log_loss"],
        stored_one_vector=t["distinct"]["old"]["benchmarks_with_one_vector"],
        stored_below_half=t["distinct"]["old"]["benchmarks_below_half_distinct"],
        new_below_half=t["distinct"]["new"]["benchmarks_below_half_distinct"])
    ft = facts["full_text"]
    ft["ridge_minus_best"] = _interval(tc["ridge_new - best_single (capped)"]["accuracy"])
    ft["ridge_uncapped_minus_best"] = _interval(tc["ridge_new - best_single (uncapped)"]["accuracy"])
    ft["ridge_minus_best_log_loss"] = _interval(tc["ridge_new - best_single (capped)"]["log_loss"])
    ft["ridge_uncapped_minus_best_log_loss"] = _interval(tc["ridge_new - best_single (uncapped)"]["log_loss"])
    no_recall = _load(folder, "router_full_text.json")["suites"][suite + "_no_recall_only"]["contrasts"]
    ft["ridge_minus_best_log_loss_no_recall_only"] = _interval(no_recall["ridge_new - best_single (capped)"]["log_loss"])
    ft["ridge_minus_stored_log_loss"] = _interval(tc["ridge_new - ridge_old (capped)"]["log_loss"])
    ft["ridge_cost_share"] = t["points"]["ridge_new"]["capped"]["usd_per_1k_requests"] / best["usd_per_1k_requests"]
    ft["ridge_uncapped_cost_share"] = (t["points"]["ridge_new"]["uncapped"]["usd_per_1k_requests"]
                                       / best["usd_per_1k_requests"])
    # The four variants of the ridge router (with or without the cap, either representation) against the best model
    # per benchmark.  The paper says: every variant is cheaper and has the lower log loss when source groups are
    # resampled; the two on the whole request also when benchmarks are; the two without the cap are more accurate
    # when source groups are resampled; none is more accurate when benchmarks are; none is less accurate.
    accuracy = dict(stored_capped=un["ridge_minus_best"], stored_uncapped=un["ridge_uncapped_minus_best"],
                    whole_capped=ft["ridge_minus_best"], whole_uncapped=ft["ridge_uncapped_minus_best"])
    log_loss = dict(stored_capped=un["ridge_minus_best_log_loss"],
                    stored_uncapped=un["ridge_uncapped_minus_best_log_loss"],
                    whole_capped=ft["ridge_minus_best_log_loss"], whole_uncapped=ft["ridge_uncapped_minus_best_log_loss"])
    facts["ridge"] = dict(
        variants=len(accuracy),
        more_accurate_within=sorted(k for k, cell in accuracy.items() if cell["within_low"] > 0),
        more_accurate_over=sorted(k for k, cell in accuracy.items() if cell["over_low"] > 0),
        less_accurate_within=sorted(k for k, cell in accuracy.items() if cell["within_high"] < 0),
        lower_log_loss_within=sorted(k for k, cell in log_loss.items() if cell["within_high"] < 0),
        lower_log_loss_over=sorted(k for k, cell in log_loss.items() if cell["over_high"] < 0),
        largest_cost_share=max(un["ridge_cost_share"], un["ridge_uncapped_cost_share"], ft["ridge_cost_share"],
                               ft["ridge_uncapped_cost_share"]))
    rr = facts["ridge"]
    rr["whole_request_variants"] = len(rr["lower_log_loss_over"])
    _check(rr["largest_cost_share"] < 1, "every ridge router is cheaper than the best model per benchmark")
    _check(len(rr["lower_log_loss_within"]) == rr["variants"],
           "every ridge router has the lower log loss when source groups are resampled")
    _check(rr["lower_log_loss_over"] == ["whole_capped", "whole_uncapped"],
           "the two ridge routers on the whole request have the lower log loss when benchmarks are resampled, the others not")
    _check(rr["more_accurate_within"] == ["stored_uncapped", "whole_uncapped"] and not rr["more_accurate_over"]
           and not rr["less_accurate_within"],
           "the two ridge routers without the cap are more accurate when source groups are resampled; none is when "
           "benchmarks are, and none is less accurate")
    _check(max(accuracy[k]["within_low"] for k in rr["more_accurate_within"]) < 0.001,
           "the ridge routers without the cap are more accurate only narrowly (lower bounds under a tenth of a point)")
    _check(not _excludes_zero(ft["ridge_minus_best"], "within") and not _excludes_zero(ft["ridge_minus_best"], "over"),
           "under the cap the ridge router on the whole request is level in accuracy")
    cell = ft["ridge_minus_best_log_loss_no_recall_only"]
    _check(not _excludes_zero(cell, "within") and abs(cell["difference"]) < abs(ft["ridge_minus_best_log_loss"]["difference"]),
           "without the two recall-only benchmarks the ridge router's lead in log loss is smaller and includes zero")
    # the whole request and the nearest-neighbour router
    _check(ft["knn_minus_best"]["within_high"] < 0 and ft["knn_minus_best"]["over_high"] < 0,
           "with the whole request the kNN router stays below the best model per benchmark")
    _check(not _excludes_zero(ft["knn_minus_stored"], "within") and not _excludes_zero(ft["knn_minus_stored"], "over"),
           "the whole request does not move the kNN router's accuracy")
    _check(abs(ft["ridge_minus_stored"]["difference"]) < 0.001 and not _excludes_zero(ft["ridge_minus_stored"], "within"),
           "the whole request leaves the ridge router's accuracy")
    _check(ft["new_below_half"] == 0, "every benchmark has distinct vectors for most requests")

    own = s["against_best_single"]["knn_router"]
    against_own = s["against_best_single_own"]
    stack_own, stack_reference = s["against_best_single"]["stack_final_temperature"], against_own["stack_final_temperature"]
    facts["own"] = dict(
        router_minus_best=own["difference"], router_groups_low=own["groups"]["ci_low"],
        router_groups_high=own["groups"]["ci_high"], router_over_low=own["benchmarks"]["ci_low"],
        router_over_high=own["benchmarks"]["ci_high"],
        router_above=own["benchmarks_above"], router_below=own["benchmarks_below"],
        reference_own=s["policies"]["best_single"]["own"], own_reference_own=s["policies"]["best_single_own"]["own"],
        router_minus_own=against_own["knn_router"]["difference"],
        text_router_minus_own=against_own["knn_router_text"]["difference"],
        stack_minus_best=stack_own["difference"], stack_groups_low=stack_own["groups"]["ci_low"],
        stack_groups_high=stack_own["groups"]["ci_high"], stack_over_low=stack_own["benchmarks"]["ci_low"],
        stack_over_high=stack_own["benchmarks"]["ci_high"], stack_above=stack_own["benchmarks_above"],
        stack_below=stack_own["benchmarks_below"],
        stack_minus_own=stack_reference["difference"], stack_above_own=stack_reference["benchmarks_above"],
        stack_below_own=stack_reference["benchmarks_below"],
        isarcasm_stack=o["benchmarks"]["40"]["policies"]["stack_final_temperature"]["own"],
        isarcasm_own_reference=o["benchmarks"]["40"]["policies"]["best_single_own"]["own"])
    ow = facts["own"]
    _check(ow["router_over_low"] < 0 < ow["router_over_high"], "the router's own-metric lead includes zero over benchmarks")
    _check(ow["router_below"] > ow["router_above"], "the router is below the reference on most benchmarks")
    for name in ("knn_router", "knn_router_text"):
        cell = against_own[name]
        _check(cell["groups"]["ci_high"] < 0 and cell["benchmarks"]["ci_high"] < 0,
               "both routers are below the reference chosen by the own metric")
    _check(ow["stack_groups_low"] > 0 and ow["stack_over_low"] > 0 and ow["stack_above"] > 3 * ow["stack_below"],
           "stacking is above the best model per benchmark on the own metrics, with both intervals and on most benchmarks")
    _check(ow["stack_minus_own"] < 0 and stack_reference["groups"]["ci_high"] < 0
           and _includes_zero(stack_reference["benchmarks"]["ci_low"], stack_reference["benchmarks"]["ci_high"])
           and ow["stack_above_own"] > ow["stack_below_own"],
           "stacking is below the reference chosen by the own metric on average and above it on most benchmarks")
    _check(ow["isarcasm_stack"] < ow["isarcasm_own_reference"] / 2, "log loss is a poor guide on iSarcasmEval")

    # the twenty redrawn fold assignments
    folds = _load(folder, "fold_redraws.json")["suites"][suite]
    methods, contrasts = folds["methods"], folds["contrasts"]
    facts["folds"] = dict(
        draws=len(methods["best_single"]["log_loss"]["draws"]),
        **{f"{name}_{key}": value for name in ("C5", "C8", "D1")
           for key, value in (("frozen", contrasts[name]["accuracy"]["frozen"]),
                              ("mean", sum(contrasts[name]["accuracy"]["draws"])
                               / len(contrasts[name]["accuracy"]["draws"])))})
    fo = facts["folds"]
    _check(fo["C5_frozen"] <= fo["C5_mean"] and fo["D1_frozen"] <= fo["D1_mean"],
           "the frozen folds do not favour the fitted combinations over the best model per benchmark")
    _check(fo["C8_frozen"] < 0 and fo["C8_mean"] < 0, "the router trails the best model per benchmark on the redrawn folds too")
    _check(all(c["log_loss"]["same_sign_as_frozen"] and c["accuracy"]["same_sign_as_frozen"]
               for c in contrasts.values()), "every contrast keeps its sign on every draw")

    # rows of two tables, copied as they are
    def row(name, group, point, own=None):
        return dict(name=name, group=group, log_loss=point["log_loss"], accuracy=point["accuracy"],
                    own=own if own is not None else point.get("native"),
                    usd=point["usd_per_1k_requests"], calls=point.get("calls_per_request"))

    new_knn = t["points"]["knn_new"]["capped"]
    facts["table"] = dict(
        rows=[
            row("Best model per benchmark", "Told the benchmark", best),
            row("Best model per benchmark, by own metric", "Told the benchmark", s["policies"]["best_single_own"],
                own=s["policies"]["best_single_own"]["own"]),
            row("Ridge router, text embeddings", "Told the benchmark", within),
            row("Ridge router, text embeddings, no cap", "Told the benchmark", up["ridge_within_minilm"]["uncapped"]),
            row("kNN router, whole request", "Told the benchmark", new_knn),
            row("Top-$k$ pool, validated $k$", "Told the benchmark", top),
            row("Pool with equal weight per provider", "Told the benchmark", pts["pool_by_organisation"]),
            row("One model for all benchmarks", "Not told the benchmark", up["single_global"]["None"]),
            row("kNN router, text embeddings", "Not told the benchmark", across),
            row("kNN router, request features", "Not told the benchmark", up["knn_across_numeric"]["uncapped"]),
            row("Ridge router, text embeddings", "Not told the benchmark", up["ridge_across_minilm"]["uncapped"]),
        ],
        representations=dict(
            whole=dict(log_loss=new_knn["log_loss"], accuracy=new_knn["accuracy"], usd=new_knn["usd_per_1k_requests"]),
            ridge=dict(log_loss=within["log_loss"], accuracy=within["accuracy"], usd=within["usd_per_1k_requests"]),
            tfidf_refit=dict(log_loss=facts["refit"]["tfidf_log_loss"], accuracy=facts["refit"]["tfidf_accuracy"],
                             usd=facts["refit"]["tfidf_usd"])))
    return facts


def main(argv=None):
    ap = argparse.ArgumentParser(description="The follow-up numbers that the paper quotes, with guards")
    ap.add_argument("--followup", required=True)
    ap.add_argument("--suite", default="main35")
    a = ap.parse_args(argv)
    facts = collect(a.followup, a.suite)
    RX._json(Path(a.followup) / "followup_facts.json", dict(suite=a.suite, **facts))
    print(f"wrote {Path(a.followup) / 'followup_facts.json'}: {sum(len(v) for v in facts.values())} values")
    return 0


if __name__ == "__main__":
    sys.exit(main())

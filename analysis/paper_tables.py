"""Tables of the paper that need display names, grouping or a wide layout.

Each function takes rows that the exporter has already read and returns LaTeX (booktabs, no vertical rules).
Nothing here computes a result: it selects, orders and formats.
"""

from __future__ import annotations

NAMES = {
    "best_single": "Best model per benchmark", "oracle": "Per-request oracle", "random_single": "Random model",
    "uniform_mean": "Equal-weight pool", "majority_vote": "Majority vote", "skill_pool": "Skill-weighted pool",
    "uniform_temperature": "Equal-weight pool + temperature", "stack": "Stacking",
    "stack_final_temperature": "Stacking + temperature", "greedy": "Greedy selection",
    "knn_router": "kNN router, request features", "knn_router_text": "kNN router, text embeddings",
    "confidence_cascade": "Confidence cascade", "certified_exit": "Certified early exit",
    "tree_replay": "Tree-guided selection", "greedy_replay": "Greedy selection (stored fit)",
    "combiner_replay": "Conditional-logit combiner",
}
SHORT = {
    "best_single": "Best per benchmark", "uniform_mean": "Pool", "uniform_temperature": "Pool+T",
    "stack_final_temperature": "Stack+T", "greedy": "Greedy", "knn_router": "Router", "knn_router_text": "Router (text)",
    "confidence_cascade": "Cascade", "certified_exit": "Exit", "combiner_replay": "Combiner", "oracle": "Oracle",
    "tree_replay": "Tree", "greedy_replay": "Greedy (stored)", "skill_pool": "Skill pool", "stack": "Stack",
}
GROUPS = (
    ("Reference points", ("best_single", "random_single")),          # the oracle is computed, not printed
    ("No labels", ("uniform_mean", "majority_vote", "skill_pool")),
    ("Supervised combination", ("uniform_temperature", "stack", "stack_final_temperature", "greedy", "tree_replay")),
    ("Routing and cascades", ("knn_router", "knn_router_text", "confidence_cascade", "certified_exit")),
)
APPENDIX_GROUPS = GROUPS + (("Stored fits", ("greedy_replay", "combiner_replay")),)
TEXT = ("knn_router_text",)                                # needs the benchmark text
REPLAYED = ("tree_replay", "greedy_replay", "combiner_replay")   # fitted by earlier code, replayed here
KNOWN = ("C2", "C3", "C6", "C7")                 # estimated with an interval by earlier studies on the same folds
KNOWN_ARMS = ("C1", "C4", "C5")                  # the point estimates of both arms were stored by earlier studies
AREAS = ("knowledge", "language", "retrieval", "tools", "arts")
POLICIES = (("single_by_cost", "Per-benchmark control"), ("knn_router", "kNN router, request features"),
            ("knn_router_text", "kNN router, text embeddings"), ("confidence_cascade", "Cascade, lowest confidence"),
            ("confidence_cascade_mean", "Cascade, mean confidence"))
VERSUS_SINGLE = {"single_by_cost": "control", "knn_router": "router", "knn_router_text": "text_router",
                 "confidence_cascade": "cascade", "confidence_cascade_mean": "mean_cascade"}
WIDE_METHODS = ("best_single", "uniform_mean", "uniform_temperature", "stack_final_temperature", "greedy",
                "knn_router", "knn_router_text", "confidence_cascade", "combiner_replay")
SUITE_NAMES = {"all36": "All", "main35": "Headline", "main35_no_recall_only": "No recall-only",
               "main35_no_largest": "No largest", "main35_no_saturated": "No saturated"}


def esc(text):
    return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


SHORT_TEX = {"best_single": r"Best ($\mathcal{B}_d$)"}   # short names that carry the paper's symbol


def short(method, fallback=None):
    """Short name of a method in a LaTeX table: the paper's symbol where it has one, else the escaped short name."""
    if method in SHORT_TEX:
        return SHORT_TEX[method]
    return esc(SHORT.get(method, method if fallback is None else fallback))


def num(value, decimals, scale=1.0):
    return "--" if value is None else f"{scale * value:,.{decimals}f}"


def signed(value, decimals, scale=1.0):
    return "--" if value is None else f"${scale * value:+.{decimals}f}$".replace("-", "{-}")


def _wrap(body, caption, label, wide=True, size=r"\small"):
    env = "table*" if wide else "table"
    return (f"\\begin{{{env}}}[tp]\n\\centering{size}\n{body}\n\\caption{{{caption}}}\n\\label{{tab:{label}}}\n"
            f"\\end{{{env}}}\n")


def _mark(method):
    return r"$^{\dagger}$" if method in TEXT else (r"$^{\ddagger}$" if method in REPLAYED else "")


def main_results(rows, suite, uncapped=None):
    """Methods on the headline suite, grouped.

    ``uncapped``: {method: its row without the cost cap} (``operating_points.json``); each is printed below the
    capped row of its method.
    """
    by = {r["method"]: r for r in rows if r["suite"] == suite and r["catalog_id"] is None and r["variant"] is None}
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Method & Log loss & Acc.\ (\%) & Own (\%) & Calls & USD/1k & Latency (s) \\"]
    n_bench = None
    for title, methods in GROUPS:
        present = [m for m in methods if m in by]
        if not present:
            continue
        lines += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for m in present:
            r = by[m]
            n_bench = r["n_benchmarks"]
            lines.append(" & ".join([
                esc(NAMES[m]) + _mark(m), num(r["log_loss"], 3), num(r["accuracy"], 1, 100), num(r["native"], 1, 100),
                num(r["calls_per_request"], 2), num(r["usd_per_1k_requests"], 3), num(r["latency_mean_s"], 2)]) + r" \\")
            u = (uncapped or {}).get(m)
            if u:
                lines.append(" & ".join([
                    r"\quad without the cost cap", num(u["log_loss"], 3), num(u["accuracy"], 1, 100),
                    num(u.get("native"), 1, 100), num(u["calls_per_request"], 2), num(u["usd_per_1k_requests"], 3),
                    num(u["latency_mean_s"], 2)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    shown = [m for _, methods in GROUPS for m in methods if m in by]
    caption = (f"Methods on the headline suite ({n_bench} benchmarks, equal weight). Log loss is in nats per "
               "judged question; the own metric is each benchmark's native score. Latency is the mean per request, "
               "with the calls of one stage in parallel. The router and the cascade are shown at the setting chosen "
               r"under a cost cap (Appendix~\ref{sec:setup})"
               + ("; the cascade also without it" if uncapped else "")
               + r". $^{\dagger}$ needs the benchmark text."
               + (r" $^{\ddagger}$ fitted by earlier analysis code and replayed." if any(m in REPLAYED for m in shown)
                  else ""))
    return _wrap("\n".join(lines), caption, "main-results", size=r"\small\setlength{\tabcolsep}{5pt}")


def _p(value):
    return "--" if value is None else (f"{value:.4f}" if value < 0.01 else f"{value:.2f}")


def _names(contrasts):
    """Contrast names in words: "C1 to C5, C8 and C9" for C1..C5, C8, C9."""
    runs = []
    for c in contrasts:
        if runs and c[0] == runs[-1][-1][0] and int(c[1:]) == int(runs[-1][-1][1:]) + 1:
            runs[-1].append(c)
        else:
            runs.append([c])
    parts = [part for r in runs for part in ([f"{r[0]} to {r[-1]}"] if len(r) > 2 else r)]
    if len(parts) < 2:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def contrasts(rows, suite, facts=None, earlier=True):
    """All contrasts on the headline suite: difference, two intervals, adjusted p-value.

    ``earlier``: mark the contrasts that the first study's earlier analyses had estimated on its pool and folds.
    It is false for the released corpus, whose pool differs; what had been seen of that pool before its run (point
    estimates of the pools, stacking and greedy selection) is said in the paper's appendix on reproducibility.
    """
    loss = {r["contrast"]: r for r in rows if r["suite"] == suite and r["metric"] == "log_loss"}
    acc = {r["contrast"]: r for r in rows if r["suite"] == suite and r["metric"] == "accuracy"}
    over = (facts or {}).get("over_benchmarks", {})
    order = sorted(loss, key=lambda c: (c[0], int(c[1:])))
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r" & A $-$ B & $\Delta$ log loss & Within benchmarks & Benchmarks resampled & $p$ (Holm) & "
             r"$\Delta$ acc.\ (pts) & Better / worse \\", r"\midrule"]
    for c in order:
        r, a = loss[c], acc.get(c)
        if c == "D1":
            lines.append(r"\midrule")
        mark = "" if not earlier else (r"$^{\ast}$" if c in KNOWN else (r"$^{\circ}$" if c in KNOWN_ARMS else ""))
        wide = over.get(c, {}).get("log_loss")
        if a is None:
            points = "--"
        elif a["ci_low"] == a["ci_high"] == a["difference"] == 0:
            points = "0"
        else:
            points = signed(a["difference"], 2, 100)
        lines.append(" & ".join([
            c + mark, short(r["a"], NAMES[r["a"]]) + " $-$ " + short(r["b"], NAMES[r["b"]]),
            signed(r["difference"], 4), f"[{signed(r['ci_low'], 4)}, {signed(r['ci_high'], 4)}]",
            f"[{signed(wide['low'], 3)}, {signed(wide['high'], 3)}]" if wide else "--", _p(r["p_holm"]), points,
            f"{r['benchmarks_better']} / {r['benchmarks_worse']}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    draws = next(iter(loss.values()))["draws"] if loss else 0
    floor = min((r["p_holm"] for r in loss.values() if r["p_holm"] is not None), default=None)
    planned, later = [c for c in order if c.startswith("C")], [c for c in order if c.startswith("D")]
    caption = ("Paired contrasts on the headline suite: A minus B, so a negative log-loss difference favours A. "
               f"Both intervals are 95\\% percentile intervals from {draws:,} bootstrap draws: one resamples the "
               "source groups within each benchmark, the other resamples the benchmarks. "
               "Better / worse counts the benchmarks on which A has the lower / higher log loss (ties are left out). "
               "Pool: equal-weight pool; +T: with a temperature; Stack: stacking"
               + ("; Tree: tree-guided selection" if any("tree_replay" in (r["a"], r["b"]) for r in loss.values()) else "")
               + "; Exit: early exit."
               + (f" The contrasts {_names(planned)} were fixed before the first run; their $p$-values "
                  "(within-benchmark resampling, Holm-adjusted) are descriptive"
                  + (f", and {_p(floor)} is the smallest value that these draws allow" if floor is not None else "")
                  + "." if planned else "")
               + (f" {_names(later)} were added later and are not adjusted." if later else "")
               + " An accuracy difference of 0 is zero by construction."
               + (r" $^{\ast}$ estimated with an interval by our earlier analyses on the same "
                  r"folds; $^{\circ}$ the point estimates of both arms were known from them." if earlier else ""))
    return _wrap("\n".join(lines), caption, "contrasts", size=r"\scriptsize\setlength{\tabcolsep}{4pt}")


def contrasts_accuracy(rows, suite, facts=None):
    """The contrasts of ``contrasts`` in accuracy, with the within-benchmark and the benchmark-resampled interval."""
    acc = {r["contrast"]: r for r in rows if r["suite"] == suite and r["metric"] == "accuracy"}
    over = (facts or {}).get("over_benchmarks", {})
    order = sorted(acc, key=lambda c: (c[0], int(c[1:])))
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r" & A $-$ B & $\Delta$ acc.\ (pts) & Within benchmarks & Benchmarks resampled & Better / worse \\",
             r"\midrule"]
    for c in order:
        a = acc[c]
        if c == "D1":
            lines.append(r"\midrule")
        wide = over.get(c, {}).get("accuracy")
        zero = a["ci_low"] == a["ci_high"] == a["difference"] == 0
        lines.append(" & ".join([
            c, short(a["a"], NAMES[a["a"]]) + " $-$ " + short(a["b"], NAMES[a["b"]]),
            "0" if zero else signed(a["difference"], 2, 100),
            "--" if zero else f"[{signed(a['ci_low'], 2, 100)}, {signed(a['ci_high'], 2, 100)}]",
            "--" if zero or not wide else f"[{signed(wide['low'], 2, 100)}, {signed(wide['high'], 2, 100)}]",
            f"{a['benchmarks_better']} / {a['benchmarks_worse']}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("The paired contrasts in accuracy, in percentage points, with the same two "
               "kinds of 95\\% interval. Better / worse counts the benchmarks on which A is more / less accurate. "
               "A difference of 0 is zero by construction.")
    return _wrap("\n".join(lines), caption, "contrasts-accuracy", wide=True,
                 size=r"\scriptsize\setlength{\tabcolsep}{3pt}")


def followup_policies(table):
    """Policies of the follow-up analyses (``followup_facts.json``: facts.table.rows) beside the reference."""
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Policy & Log loss & Acc.\ (\%) & Own (\%) & Calls & USD/1k \\"]
    group = None
    for r in table["rows"]:
        if r["group"] != group:
            group = r["group"]
            lines += [r"\midrule", rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{esc(group)}}}}} \\"]
        lines.append(" & ".join([r["name"], num(r["log_loss"], 3), num(r["accuracy"], 1, 100), num(r["own"], 1, 100),
                                 num(r["calls"], 2), num(r["usd"], 3)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("Policies of the further analyses on the headline suite (benchmarks weighted equally). Routers are "
               "fitted inside each benchmark unless they are not told the benchmark; then one recalibration, one "
               "setting and one budget serve all benchmarks and the neighbours come from every benchmark. The "
               "ridge router regresses each model's log score on the request's sentence embedding. Routers are "
               "shown under the cost cap unless marked; the routers that are not told the benchmark are shown "
               "without it. The top-$k$ pool chooses $k$ on the validation fold. --: not computed.")
    return _wrap("\n".join(lines), caption, "followup-policies", wide=True,
                 size=r"\scriptsize\setlength{\tabcolsep}{3pt}")


def per_benchmark(rows, benchmarks, metric, decimals, scale, title, label):
    """One row per benchmark, one column per method."""
    methods = [m for m in WIDE_METHODS if any(r["method"] == m for r in rows)]
    cell = {(r["method"], r["catalog_id"]): r[metric] for r in rows
            if r["catalog_id"] is not None and r["variant"] is None}
    lines = [r"\begin{tabular}{@{}l" + "r" * len(methods) + r"@{}}", r"\toprule",
             "Benchmark & " + " & ".join(short(m) for m in methods) + r" \\", r"\midrule"]
    area = None
    place = {a: i for i, a in enumerate(AREAS)}
    for b in sorted(benchmarks, key=lambda b: (place.get(b["area"], len(place)), int(b["catalog_id"]))):
        if area is not None and b["area"] != area:
            lines.append(r"\addlinespace")
        area = b["area"]
        cid = int(b["catalog_id"])
        lines.append(esc(b["name"][:24]) + " & " + " & ".join(num(cell.get((m, cid)), decimals, scale) for m in methods) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _wrap("\n".join(lines), title, label, size=r"\scriptsize\setlength{\tabcolsep}{4pt}")


def sensitivity(rows, suites, metric, decimals, scale, title, label):
    """One row per method, one column per suite."""
    names = [s for s in SUITE_NAMES if s in suites]
    cell = {(r["method"], r["suite"]): r for r in rows if r["catalog_id"] is None and r["variant"] is None}
    count = {s: next((r["n_benchmarks"] for (m, x), r in cell.items() if x == s), None) for s in names}
    lines = [r"\begin{tabular}{@{}l" + "r" * len(names) + r"@{}}", r"\toprule",
             "Method & " + " & ".join(f"{SUITE_NAMES[s]} ({count[s]})" for s in names) + r" \\", r"\midrule"]
    for _, methods in APPENDIX_GROUPS:
        for m in methods:
            if any((m, s) in cell for s in names):
                lines.append(esc(NAMES[m]) + _mark(m) + " & " + " & ".join(
                    num(cell[m, s][metric], decimals, scale) if (m, s) in cell else "--" for s in names) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _wrap("\n".join(lines), title, label, size=r"\footnotesize")


def methods_detail(rows, suite):
    """Every method on the headline suite: log loss at three floors, Brier score, calibration error, latency."""
    by = {r["method"]: r for r in rows if r["suite"] == suite and r["catalog_id"] is None and r["variant"] is None}
    lines = [r"\begin{tabular}{@{}lrrrrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Log loss, floor} & & & & & & \multicolumn{2}{c}{Latency (s)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){10-11}",
             r"Method & $10^{-15}$ & $10^{-6}$ & $10^{-3}$ & Clipped & Brier & ECE & Acc.\ (\%) & Own (\%) & "
             r"median & 95th pct. \\"]
    for title, methods in APPENDIX_GROUPS:
        present = [m for m in methods if m in by]
        if not present:
            continue
        lines += [r"\midrule", rf"\multicolumn{{11}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for m in present:
            r = by[m]
            lines.append(" & ".join([
                esc(NAMES[m]) + _mark(m), num(r["log_loss"], 3), num(r["log_loss_floor_1e-6"], 3),
                num(r["log_loss_floor_1e-3"], 3), f"{r['n_clipped']:,}", num(r["brier"], 3), num(r["ece"], 3),
                num(r["accuracy"], 1, 100), num(r["native"], 1, 100), num(r["latency_p50_s"], 2),
                num(r["latency_p95_s"], 2)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("Every method on the headline suite, with the columns that the main table leaves out. Log loss "
               "clips the probability of the accepted options from below at the floor; ``clipped'' counts the "
               r"judged fields below $10^{-15}$. The Brier score is the multi-class form; ECE is the expected "
               "calibration error with ten bins. Latency percentiles are over requests, averaged over benchmarks. "
               r"$^{\dagger}$ needs the benchmark text. $^{\ddagger}$ fitted by an earlier study and replayed: "
               "the tree-guided and the earlier greedy selection from their stored weights, the combiner from its "
               "stored pooled vectors.")
    return _wrap("\n".join(lines), caption, "methods-detail", size=r"\scriptsize\setlength{\tabcolsep}{4pt}")


def operating_points(document, suite, reference):
    """The router, the control and the cascade under the capped and the uncapped selection rule."""
    block = document["suites"][suite]
    points, effects, chosen = block["points"], block["contrasts"], block["selection"]
    lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
             r"Policy & Rule & Log loss & Acc.\ (\%) & Own (\%) & Cost & Calls & $\Delta$ log loss to best per benchmark "
             r"& Cap binds \\", r"\midrule"]
    for name, label in POLICIES:
        if name not in points:
            continue
        for rule in document["rules"]:
            r = points[name][rule]
            e = effects[f"{VERSUS_SINGLE[name]}_{rule}_vs_single"]["log_loss"]
            binds = (f"{chosen[name]['cap_changed_choice']} / {chosen[name]['cells']}" if rule == "capped" else "")
            lines.append(" & ".join([
                esc(label) + _mark(name) if rule == "capped" else "", rule, num(r["log_loss"], 4),
                num(r["accuracy"], 2, 100), num(r["native"], 1, 100),
                num(r["usd_per_1k_requests"] / reference["usd_per_1k_requests"], 0, 100) + r"\%",
                num(r["calls_per_request"], 2),
                f"{signed(e['difference'], 4)} [{signed(e['ci_low'], 4)}, {signed(e['ci_high'], 4)}]", binds]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("Operating points of the router, its per-benchmark control and the cascade on the headline suite. "
               "``Capped'' is the rule of the main table: per benchmark and test fold, the setting with the lowest "
               "validation loss among those whose validation cost does not exceed that of the best model per benchmark. "
               "``Uncapped'' takes the lowest validation loss at any cost. Cost is relative to the best model per benchmark "
               f"({num(reference['log_loss'], 4)} nats, {num(reference['accuracy'], 2, 100)}\\% accuracy, "
               f"{num(reference['native'], 1, 100)}\\% own metric). Intervals: paired bootstrap over source groups, "
               "not adjusted. ``Cap binds'' counts the combinations of benchmark and test fold in which the cap changed the "
               r"choice. $^{\dagger}$ needs the benchmark text.")
    return _wrap("\n".join(lines), caption, "operating-points", size=r"\scriptsize\setlength{\tabcolsep}{4pt}")


def open_track(document, rows, suite):
    """Methods on the open subset beside the same methods on all models."""
    block = document["suites"][suite]
    full = {r["method"]: r for r in rows if r["suite"] == suite and r["catalog_id"] is None and r["variant"] is None}
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             rf" & \multicolumn{{2}}{{c}}{{Open subset ({document['models']} models)}} & "
             r"\multicolumn{2}{c}{All models} \\", r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Method & Log loss & Acc.\ (\%) & Log loss & Acc.\ (\%) \\", r"\midrule"]
    for m in document["methods"]:
        mine, theirs = block["methods"][m], full.get(m)
        lines.append(" & ".join([
            esc(NAMES[m]), num(mine["log_loss"], 3), num(mine["accuracy"], 1, 100),
            num(theirs["log_loss"], 3) if theirs else "--", num(theirs["accuracy"], 1, 100) if theirs else "--"])
            + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = (f"What the open subset supports on its own: the {document['models']} released models on the "
               f"{block['benchmarks']} benchmarks of the headline suite, with labels rebuilt from the benchmark "
               "rows. The open subset holds no cost or latency, so routing, cascades and early exit cannot be "
               "replayed from it. Right: the same methods on all models (Table~\\ref{tab:main-results}).")
    return _wrap("\n".join(lines), caption, "open-track", size=r"\footnotesize")


def fold_redraws(document, suite, stored_fits=True):
    """Fitted methods on the frozen folds and on redrawn folds.

    ``stored_fits``: the run also has methods with stored fits, which cannot be refitted; the caption says so.
    """
    block = document["suites"][suite]["methods"]
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Log loss} & \multicolumn{2}{c}{Accuracy (\%)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Method & Frozen folds & Redrawn folds & Frozen folds & Redrawn folds \\", r"\midrule"]
    for _, methods in GROUPS:
        for m in methods:
            if m not in block:
                continue
            loss, acc = block[m]["log_loss"], block[m]["accuracy"]
            lines.append(" & ".join([
                esc(NAMES[m]), num(loss["frozen"], 4), f"{num(loss['low'], 4)} to {num(loss['high'], 4)}",
                num(acc["frozen"], 2, 100), f"{num(acc['low'], 2, 100)} to {num(acc['high'], 2, 100)}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("The methods that the simulator fits, on the frozen folds and on "
               f"{document['draws']} redrawn assignments of the source groups to five folds (lowest and highest "
               "value over the draws)."
               + (" The methods with stored fits cannot be refitted and are left out." if stored_fits else ""))
    return _wrap("\n".join(lines), caption, "fold-redraws", size=r"\footnotesize")


def models(summary, corpus_models, coded=False):
    """Per-model diagnostics on the headline suite, most accurate first."""
    name = {int(m["model_index"]): m for m in corpus_models}
    label = "model_code" if coded else "model_name"
    if coded:          # price and latency per code would identify the model on the gateway
        lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
                 r"Model & Accuracy (\%) & Log loss & Calibration error \\", r"\midrule"]
    else:
        lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
                 r"Model & Accuracy & Log loss & Calibration & USD per & Median \\",
                 r" & (\%) & & error & 1,000 requests & latency (s) \\", r"\midrule"]
    for r in sorted(summary, key=lambda r: -r["accuracy_first_max"]):
        m = name[int(r["model_index"])]
        cells = [r"\texttt{" + esc(m[label]) + "}", num(r["accuracy_first_max"], 1, 100),
                 num(r["log_loss_normalised_floor_1e-15"], 3), num(r["ece"], 3)]
        if not coded:
            cells += [num(r["usd_per_1k_requests"], 3), num(r["latency_p50_s"], 2)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("The models on the headline suite (equal weight per benchmark), by accuracy. Each model is scored "
               "as a panel of one: normalised vectors, first maximum. Calibration error is the expected "
               "calibration error with ten bins."
               + ("" if coded else " Cost and median latency are means over the benchmarks of the suite."))
    return _wrap("\n".join(lines), caption, "models", wide=True, size=r"\footnotesize")


def _released(audit, release, i):
    """Whether model ``i`` (an index as a string) is in the open subset of the release decision."""
    return release["model_status"][audit["models"][i]["route_id"]]["release"] == "released"


def rights(audit, coded=False, release=None):
    """Status of every model in the review of output terms, and in the open subset when the release is given."""
    status = audit["rights"]["model_status"]
    ordered = sorted(audit["models"].items(), key=lambda kv: int(kv[0]))
    extra = (r" & Open subset" if release else "", "l" if release else "")
    mark = (lambda i: " & " + ("yes" if _released(audit, release, i) else "held")) if release else (lambda i: "")
    tail = (", and whether its vectors are in the open subset" if release else "")
    if coded:
        lines = [r"\begin{tabular}{@{}ll" + extra[1] + r"@{}}", r"\toprule", r"Model & Review of output terms" + extra[0] + r" \\",
                 r"\midrule"]
        lines += [r"\texttt{" + f"M{int(i):02d}" + "} & " + esc(status[i]) + mark(i) + r" \\" for i, _ in ordered]
        caption = ("The models, under the codes of the corpus, with the status of their outputs in our review of "
                   "provider terms" + tail + ".")
    else:
        lines = [r"\begin{tabular}{@{}llll" + extra[1] + r"@{}}", r"\toprule",
                 r"Model & Listed & Base model & Review of output terms" + extra[0] + r" \\", r"\midrule"]
        for i, m in ordered:
            lines.append(" & ".join([r"\texttt{" + esc(m["route_id"]) + "}", esc(m["listed_date"]),
                                     esc(m["base_model"] or "not documented"), esc(status[i])]) + mark(i) + r" \\")
        caption = ("The models, with the date on which each was listed, the base model where the provider "
                   "documents one, and the status of its outputs in our review of provider terms" + tail + ".")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _wrap("\n".join(lines), caption, "rights", size=r"\scriptsize\setlength{\tabcolsep}{4pt}")


ANCHOR_NAMES = {        # internal names of stored results -> the names used in the paper
    "context_flat_mean_22": "equal-weight pool", "context_single_calibrated_four_folds": "best model per benchmark",
    "e6_skill_pool": "skill-weighted pool", "e3b_flat_mean_temperature_fit_folds": "pool with a temperature",
    "e3_stack_recalibrated": "stacking", "e3_stack_recalibrated_final_temperature": "stacking with a final temperature",
    "context_tree_calibrated": "tree-guided selection", "e5_all22_combiner_with_providers": "conditional-logit combiner",
}


def anchors(document):
    """The reproduction check: one row per anchor."""
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.66\textwidth}rrl@{}}", r"\toprule",
             r"Stored quantity & Values & Largest gap & Reproduced \\",
             r"\midrule"]
    for a in document["anchors"]:
        what = a["what"]
        for code, name in ANCHOR_NAMES.items():
            what = what.replace(f"({code})", f"({name})")
        if a["anchor"].startswith("contrast_"):
            what = f"contrast {a['anchor'].split('_', 1)[1]}: difference, interval and p-value"
        gap = "0" if a["largest_gap"] == 0 else f"{a['largest_gap']:.1e}".replace("e-0", "e-").replace("e-", r"$\cdot 10^{-") + "}$"
        lines.append(" & ".join([esc(what), f"{a['values_compared']:,}", gap,
                                 "yes" if a["reproduced"] else "no"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("Reproduction of stored results of earlier studies. Each row compares numbers computed by the "
               "simulator from the corpus with numbers stored by earlier code; a quantity counts as reproduced "
               "when the largest absolute gap is at most $10^{-12}$.")
    return _wrap("\n".join(lines), caption, "anchors", size=r"\footnotesize")


# Short forms of audited cells of comparison_table.csv; the guard strings must occur in the audited cell.
COMPACT = (
    ("RouterBench", "11 to 14", "405,467", "no", "from price lists", "no", "linked; no licence stated",
     {"models": "11 in the stated total", "model_query_instances": "405,467", "probability_vectors": "no",
      "cost_source": "API pricing", "latency": "no (not stored)", "licence_stated": "not stated"}),
    ("LLMRouterBench", "33", "391,645", "no", "stored; origin not stated", "estimated", "linked; no licence stated",
     {"models": "33 in two pools", "model_query_instances": "391,645", "probability_vectors": "no",
      "cost_source": "is not stated", "latency": "approximated", "licence_stated": "not stated"}),
    ("EmbedLLM correctness dataset", "112", "not stated", "no", "not stated", "not stated", "not stated",
     {"models": "112", "probability_vectors": "no", "cost_source": "not stated", "latency": "not stated",
      "licence_stated": "not stated"}),
)


def compact_comparison(records, audit, release=None):
    """Main-text comparison with the stored collections that were read in full."""
    by = {r["name"]: r for r in records}
    lines = [r"\begin{tabular}{@{}lrrllll@{}}", r"\toprule",
             r"Collection & Models & Model--query & Probability & Cost & Latency & Open release \\",
             r" & & records & vectors & & & \\", r"\midrule"]
    for name, models_, records_, vectors, cost, latency, open_cell, guards in COMPACT:
        row = by[name]
        for column, needle in guards.items():
            if needle not in row[column]:
                raise ValueError(f"comparison table: the audited cell {name}/{column} no longer says {needle!r}")
        short = name.replace(" correctness dataset", "")
        lines.append(" & ".join([esc(short), models_, records_, vectors, cost, latency, open_cell]) + r" \\")
    c, r = audit["corpus"], audit["rights"]
    lines.append(r"\midrule")
    lines.append(" & ".join([
        "This corpus", f"{c['models']}", f"{c['observations']:,}", "yes", "billed", "measured",
        f"{release['models_released'] if release else len(r['cleared_models'])} of {c['models']} models, vectors only"])
                 + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("Stored collections of model outputs that we read in full, and this corpus. A record is one model "
               "on one query (here: one request, with a probability vector for each of its questions).")
    return _wrap("\n".join(lines), caption, "comparison-compact", size=r"\footnotesize\setlength{\tabcolsep}{3.5pt}")


# --------------------------------------------------------------------------------------------------------------
# Tables of the restyled paper (added after the first run).
# --------------------------------------------------------------------------------------------------------------

MODEL_ABBR = {            # short labels of the listed model names, for tables and figure labels (real-name build)
    "cloudflare/clef": "Clef", "cloudflare/clef-flash": "Clef-Flash", "cloudflare/clef-omni": "Clef-Omni",
    "heygen/heygen-decision-1": "HG-1", "heygen/heygen-decision-1-turbo": "HG-1-Turbo",
    "jaredpalmer/kev-4b": "Kev-4B", "liquid/d1": "D1", "liquid/d1-ultra": "D1-Ultra", "liquid/d1.1": "D1.1",
    "liquid/d1.2": "D1.2", "liquid/d1.2-ultra": "D1.2-Ultra", "microsoft/microsoft-decision-1.1": "MS-1.1",
    "morph/reflex": "Reflex", "nace-ai/drex-max": "Drex-Max", "nace-ai/drex-max-finance": "Drex-Max-Fin",
    "nace-ai/drex-v1.5": "Drex-1.5", "nace-ai/drex-v2.0": "Drex-2.0",
    "perplexity/pplx-decider-flash-v1.2": "Decider-Flash", "perplexity/pplx-decider-v1.2": "Decider",
    "respan/span-02": "Span-02", "typesafe/jev-1.13": "Jev-1.13", "typesafe/jev-1.14": "Jev-1.14",
}
BENCH_ABBR = {
    1: "BFCL", 2: "TRet", 3: "APIB", 4: "BK77", 5: "CLINC", 9: "Home", 11: "CNLI", 12: "ANLI", 20: "BPoMP",
    21: "Humi", 22: "POP", 23: "cfcol", 25: "GPQA", 28: "Wino", 29: "HSwag", 30: "GSM8K", 31: "Chess", 32: "MuSR",
    33: "SATA", 36: "BRIGHT", 37: "ESCI", 38: "ACOS", 39: "FinE", 40: "iSarc", 41: "VAST", 42: "NLI4CT", 43: "CRUX",
    44: "CLad", 50: "Haber", 56: "Phish", 57: "MMLUP", 58: "BBH", 59: "RAGT", 61: "HoVer", 62: "W2C", 64: "NYC",
}
AREA_NAMES = {"knowledge": "Knowledge", "language": "Language", "retrieval": "Retrieval", "tools": "Tools",
              "arts": "Arts"}


def model_labels(corpus_models, coded=False):
    """{model index: label}: the corpus code, or the short form of the listed name."""
    out = {}
    for m in corpus_models:
        i = int(m["model_index"])
        out[i] = m["model_code"] if coded else MODEL_ABBR.get(m["model_name"], m["model_name"].split("/")[-1])
    return out


CHECK = {"yes": r"\cmark", "no": r"\xmark", "partial": r"\pmark", "not stated": r"\nmark"}
COST = {"billed": "billed", "list": "price list", "billed or list": "billed or list", "tokens": "tokens only",
        "no": r"\xmark", "not stated": r"\nmark"}
LATENCY = {"measured": "measured", "estimated": "estimated", "no": r"\xmark", "not stated": r"\nmark"}


def comparison_checks(records, values):
    """The comparison with stored collections of model outputs, in the manner of LLMRouterBench's Table 1.

    records: rows of comparison_checks.csv; values: replacements for the placeholders of our own row.
    """
    lines = [r"\begin{tabular}{@{}lccccccrrc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Evaluated} & \multicolumn{3}{c}{Stored per model and item} & & & \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Collection & Comb. & Rout. & Casc. & Probs. & Cost & Latency & Datasets & Models & Release \\",
             r"\midrule"]
    notes = []
    symbols = (r"\dag", r"\ddag", r"\S", r"\P")
    for row in records:
        ours = row["name"] == "this_corpus"
        if ours:
            lines.append(r"\midrule")
        cells = {k: row[k] for k in row}
        for k, v in list(cells.items()):
            for placeholder, value in values.items():
                if isinstance(v, str):
                    cells[k] = cells[k].replace(placeholder, value)
        name = r"\textbf{DecisionReplay}" if ours else esc(row["name"]) + r"~\citep{" + row["key"] + "}"
        mark = ""
        if cells["note"]:
            notes.append(cells["note"])
            mark = r"\textsuperscript{" + symbols[len(notes) - 1] + "}"
        lines.append(" & ".join([
            name, CHECK[cells["combination"]] + (mark if cells["combination"] == "partial" else ""),
            CHECK[cells["routing"]], CHECK[cells["cascades"]],
            CHECK[cells["probability_vectors"]] + (mark if cells["probability_vectors"] == "yes" and not ours
                                                   and cells["release"] != "yes" else ""),
            COST[cells["cost"]], LATENCY[cells["latency"]], esc(cells["datasets"]), esc(cells["models"]),
            CHECK[cells["release"]] + (mark if cells["release"] in ("partial", "not stated") else "")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = (r"Comparison with stored collections of model outputs. Comb., Rout., Casc.: combination, routing and "
               r"cascade methods evaluated; Probs.: a probability for every answer option. \cmark{} yes, \xmark{} no, "
               r"\pmark{} in part, \nmark{} not stated. Release: per-item outputs of every model are public. "
               + " ".join(r"\textsuperscript{" + symbols[i] + "}" + esc(n[0].upper() + n[1:]) + "."
                          for i, n in enumerate(notes)))
    return _wrap("\n".join(lines), caption, "comparison-checks", size=r"\footnotesize\setlength{\tabcolsep}{3.6pt}")


def statistics(audit, settings_headline_count=None, release=None):
    """Overall statistics of the corpus.

    With ``release`` (the release summary of a partial release) a second column gives the open subset.  Without it
    the corpus is what is released, and one column says so.
    """
    c, t = audit["corpus"], audit["telemetry"]
    if release is None:
        rows = [
            ("Benchmarks", f"{c['benchmarks']:,}"), ("Task areas", f"{c['structure']['areas']}"),
            ("Models", f"{c['models']:,}"), ("Providers", f"{c['organisations']:,}"),
            ("Requests", f"{c['requests']:,}"), ("Questions", f"{c['fields']:,}"),
            ("\\quad with an accepted answer", f"{c['judged_fields']:,}"),
            ("Probability vectors", f"{c['vectors']:,}"),
            ("Model--request records", f"{c['observations']:,}"),
            ("Input tokens", f"{t['input_tokens'] / 1e9:.2f}B"),
            ("Collection cost", rf"\${t['cost_usd']:,.2f}"),
        ]
        lines = [r"\begin{tabular}{@{}lr@{}}", r"\toprule"]
        lines += [" & ".join(row) + r" \\" for row in rows]
        lines += [r"\bottomrule", r"\end{tabular}"]
        caption = ("Overall statistics of DecisionReplay. Every model--request record holds the probability vectors "
                   "of its questions, the token counts, the billed cost and the measured latency, and all of them "
                   "are released; request text and gold answers are rebuilt from each benchmark's own release.")
        return _wrap("\n".join(lines), caption, "statistics", wide=False, size=r"\footnotesize")
    r = audit["rights"]
    open_ = (dict(benchmarks=release["benchmarks_released"], models=release["models_released"],
                  vectors=release["vectors_released"], records=f"{release['observations_released']:,}") if release else
             dict(benchmarks=r["cleared_benchmarks"], models=len(r["cleared_models"]), vectors=r["cleared_vectors"],
                  records="--"))
    rows = [
        ("Benchmarks", f"{c['benchmarks']:,}", f"{open_['benchmarks']:,}"),
        ("Task areas", f"{c['structure']['areas']}", f"{c['structure']['areas']}"),
        ("Models", f"{c['models']:,}", f"{open_['models']}"),
        ("Requests", f"{c['requests']:,}", "--"),
        ("Questions", f"{c['fields']:,}", "--"),
        ("\\quad with an accepted answer", f"{c['judged_fields']:,}", "--"),
        ("Probability vectors", f"{c['vectors']:,}", f"{open_['vectors']:,}"),
        ("Model--request records", f"{c['observations']:,}", open_["records"]),
        ("Input tokens", f"{t['input_tokens'] / 1e9:.2f}B", "--"),
        ("Cost and latency per call", "yes", "no"),
        ("Collection cost", rf"\${t['cost_usd']:,.2f}", "--"),
    ]
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r" & Corpus & Open subset \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("Overall statistics of DecisionReplay. The open subset holds the probability vectors of the "
               + ("released models (Appendix~\\ref{sec:rights})" if release else "models whose terms allow redistribution")
               + "; request text and gold answers are rebuilt from each benchmark's own release.")
    return _wrap("\n".join(lines), caption, "statistics", wide=False, size=r"\footnotesize")


def top_by_area(model_rows, cids, labels):
    """The three most accurate models of each task area (equal weight per benchmark of the area), one row per area."""
    by = {}
    for r in model_rows:
        cid = int(r["catalog_id"])
        if cid in cids:
            by.setdefault(r["area"], {}).setdefault(int(r["model_index"]), []).append(float(r["accuracy_first_max"]))
    lines = [r"\begin{tabular}{@{}llll@{}}", r"\toprule", r"Task area & First & Second & Third \\", r"\midrule"]
    for area in AREAS:
        if area not in by:
            continue
        means = sorted(((sum(v) / len(v), m) for m, v in by[area].items()), key=lambda t: (-t[0], t[1]))[:3]
        n = len(next(iter(by[area].values())))
        cells = [rf"{esc(labels[m])} {100 * value:.1f}" for value, m in means]
        cells[0] = r"\textbf{" + cells[0] + "}"
        lines.append(f"{AREA_NAMES[area]} ({n}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("The three most accurate models of each task area, with their accuracy in percent (number of "
               "benchmarks in parentheses; benchmarks weighted equally).")
    return _wrap("\n".join(lines), caption, "top-models", wide=False, size=r"\scriptsize\setlength{\tabcolsep}{2.6pt}")


REPRESENTATIONS = (("numeric", "Five request features"), ("numeric_all", "\\quad with all neighbours"),
                   ("tfidf", "TF-IDF + SVD"), ("minilm", "MiniLM-L6 encoder"), ("smollm2", "SmolLM2-360M states"))


def representations(document, suite, reference, extra=None):
    """The router with each request representation, and the one-model-per-benchmark control (capped rule).

    ``extra`` (``followup_facts.json``: facts.table.representations) adds the router on the whole request and the
    TF-IDF router whose transform is fitted on the fitting folds only, which then stands in the TF-IDF row.
    """
    block = document["suites"][suite]
    points = block["points"]
    cost = reference["usd_per_1k_requests"]
    refit = (extra or {}).get("tfidf_refit")
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Representation & Log loss & Acc. (\%) & Cost \\", r"\midrule"]

    def row(label, p):
        share = f"{100 * p['usd_per_1k_requests'] / cost:.0f}\\%" if p.get("usd_per_1k_requests") is not None else "--"
        return " & ".join([label, f"{p['log_loss']:.3f}", f"{100 * p['accuracy']:.1f}", share]) + r" \\"

    for name, label in REPRESENTATIONS:
        key = f"router_{name}"
        if key not in points:
            continue
        p = points[key]["capped"]
        if name == "tfidf" and refit:
            p = dict(log_loss=refit["log_loss"], accuracy=refit["accuracy"], usd_per_1k_requests=refit.get("usd"))
        lines.append(row(esc(label), p))
    if extra:
        whole = extra["whole"]
        lines.append(row("Whole request, MiniLM", dict(whole, usd_per_1k_requests=whole["usd"])))
    ridge = (extra or {}).get("ridge")
    if ridge:
        lines.append(r"\midrule")
        lines.append(row(r"Ridge router, MiniLM-L6 encoder$^{\ast}$", dict(ridge, usd_per_1k_requests=ridge["usd"])))
    lines.append(r"\midrule")
    if "router_control" in points:
        lines.append(row("Per-benchmark control", points["router_control"]["capped"]))
    lines.append(" & ".join(["$\\mathcal{B}_d$", f"{reference['log_loss']:.3f}",
                             f"{100 * reference['accuracy']:.1f}", r"100\%"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    shown = sum(1 for name, _ in REPRESENTATIONS if f"router_{name}" in points and name != "numeric_all") + bool(extra)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    caption = (f"The kNN router with {words.get(shown, shown)} representations of the request, under the cost cap; "
               "``all neighbours'' adds every fitting request to the grid of neighbour counts. Cost: share of the "
               "best model per benchmark's cost.")
    if extra:
        caption += (" Text is cut at 256 tokens and leaves out the answer options, except for the whole request."
                    + (" The TF-IDF transform is fitted on the fitting folds only." if refit else "")
                    + (r" $^{\ast}$ A ridge regression in place of the nearest-neighbour rule, added after a first "
                       "draft." if ridge else ""))
    return _wrap("\n".join(lines), caption, "representations", wide=False,
                 size=r"\scriptsize\setlength{\tabcolsep}{2.8pt}")


def model_pool(corpus_models, audit, labels, coded=False, release=None):
    """The model pool (appendix): listed name, label, organisation, price, latency, base model, review and release."""
    status = audit["rights"]["model_status"]
    info = audit["models"]
    opened = (lambda i: " & " + ("yes" if _released(audit, release, i) else "held")) if release else (lambda i: "")
    if coded:
        lines = [r"\begin{tabular}{@{}lll" + ("l" if release else "") + r"@{}}", r"\toprule",
                 r"Model & Organisation & Review of output terms" + (" & Open subset" if release else "") + r" \\",
                 r"\midrule"]
        orgs = {}
        for m in sorted(corpus_models, key=lambda r: int(r["model_index"])):
            orgs.setdefault(m["organisation"], f"O{len(orgs) + 1:02d}")
            i = str(int(m["model_index"]))
            lines.append(" & ".join([r"\texttt{" + esc(m["model_code"]) + "}", orgs[m["organisation"]],
                                     esc(status[i])]) + opened(i) + r" \\")
        caption = ("The model pool under the codes of the corpus, with the organisation code, the status in our "
                   "review of provider terms" + (" and whether the model is in the open subset." if release else "."))
    else:
        base_column = r">{\raggedright\arraybackslash}p{3.1cm}" if release else "l"
        lines = [r"\begin{tabular}{@{}llrr" + base_column + "l" + ("l" if release else "") + r"@{}}", r"\toprule",
                 r"Model & Abbr. & Input price & Median & Documented & Review" + (" & Open" if release else "") + r" \\",
                 r" & & (\$ per 1M tokens) & latency (s) & base model & of terms" + (" & subset" if release else "")
                 + r" \\", r"\midrule"]
        for m in sorted(corpus_models, key=lambda r: int(r["model_index"])):
            i = str(int(m["model_index"]))
            price = 1e6 * float(info[i]["price_card_input_usd_per_token"])
            lines.append(" & ".join([r"\texttt{" + esc(m["model_name"]) + "}", esc(labels[int(i)]),
                                     f"{price:.3f}", f"{info[i]['latency_p50_s']:.2f}",
                                     esc(info[i]["base_model"] or "--").replace("/", r"/\allowbreak{}"),
                                     esc(status[i])]) + opened(i) + r" \\")
        caption = ("The model pool: listed name, the abbreviation used in tables and figures, the listed input "
                   "price (no model charged for output tokens), the median latency over all requests, the base "
                   "model where the provider documents one, the status of its outputs in our review of "
                   "terms" + (", and whether its vectors are in the open subset." if release else "."))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _wrap("\n".join(lines), caption, "model-pool", size=r"\scriptsize\setlength{\tabcolsep}{3.4pt}")


def benchmark_list(catalog, held=()):
    """The benchmarks (appendix): name, abbreviation, task area, own metric and size."""
    place = {a: i for i, a in enumerate(AREAS)}
    lines = [r"\begin{tabular}{@{}lllp{2.8cm}rrr@{}}", r"\toprule",
             r"Benchmark & Abbr. & Area & Own metric & Requests & Questions & Judged \\", r"\midrule"]
    area = None
    for b in sorted(catalog, key=lambda b: (place.get(b["area"], 9), int(b["catalog_id"]))):
        cid = int(b["catalog_id"])
        if cid in held:
            continue
        if area is not None and b["area"] != area:
            lines.append(r"\addlinespace")
        area = b["area"]
        lines.append(" & ".join([esc(b["name"]), BENCH_ABBR.get(cid, str(cid)), AREA_NAMES[b["area"]],
                                 esc(b["native_metric"].replace('"', "")), f"{int(b['requests']):,}",
                                 f"{int(b['fields']):,}", f"{int(b['judged_fields']):,}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = "The benchmarks with the abbreviations used in the tables, their task area, own metric and size."
    return _wrap("\n".join(lines), caption, "benchmarks", size=r"\scriptsize\setlength{\tabcolsep}{2.8pt}")


def _ranks(values, lower_better):
    """Indices of the best and the second-best distinct value (ties share a rank)."""
    finite = sorted({v for v in values if v is not None}, reverse=not lower_better)
    best = finite[0] if finite else None
    second = finite[1] if len(finite) > 1 else None
    return best, second


def model_benchmark(model_rows, method_rows, catalog, labels, metric, title, label, held=(), methods=()):
    """Per model and benchmark, with the best and second-best model of each benchmark shaded (appendix).

    model_rows: diagnostics/models.csv; method_rows: rows.json benchmark rows; metric: accuracy, log_loss or cost.
    Benchmarks are split into two tables by task area.
    """
    key = {"accuracy": ("accuracy_first_max", "accuracy", 100.0, 1, False),
           "log_loss": ("log_loss_normalised_floor_1e-15", "log_loss", 1.0, 3, True),
           "cost": ("usd_per_1k_requests", "usd_per_1k_requests", 1.0, 3, True)}[metric]
    model_key, method_key, scale, digits, lower = key
    place = {a: i for i, a in enumerate(AREAS)}
    shown = [b for b in sorted(catalog, key=lambda b: (place.get(b["area"], 9), int(b["catalog_id"])))
             if int(b["catalog_id"]) not in held]
    halves = (("knowledge", "language"), ("retrieval", "tools", "arts"))
    cell = {(int(r["model_index"]), int(r["catalog_id"])): float(r[model_key]) for r in model_rows}
    method_cell = {(r["method"], r["catalog_id"]): r[method_key] for r in method_rows
                   if r["catalog_id"] is not None and r["variant"] is None}
    models_ = sorted({m for m, _ in cell})
    out = []
    for part, areas in enumerate(halves):
        bench = [b for b in shown if b["area"] in areas]
        cids = [int(b["catalog_id"]) for b in bench]
        groups = [(a, sum(1 for b in bench if b["area"] == a)) for a in areas if any(b["area"] == a for b in bench)]
        header1 = " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{AREA_NAMES[a]}}}" for a, n in groups)
        rules, start = [], 2
        for _, n in groups:
            rules.append(rf"\cmidrule(lr){{{start}-{start + n - 1}}}")
            start += n
        lines = [r"\begin{tabular}{@{}l" + "r" * len(cids) + r"@{}}", r"\toprule", " & " + header1 + r" \\",
                 "".join(rules), "Model & " + " & ".join(BENCH_ABBR.get(c, str(c)) for c in cids) + r" \\",
                 r"\midrule"]
        marks = {c: _ranks([cell.get((m, c)) for m in models_], lower) for c in cids}
        for m in models_:
            entries = []
            for c in cids:
                v = cell.get((m, c))
                text = "--" if v is None else f"{scale * v:.{digits}f}"
                best, second = marks[c]
                if v is not None and v == best:
                    text = r"\cellcolor{bestcell}" + text
                elif v is not None and v == second:
                    text = r"\cellcolor{secondcell}" + text
                entries.append(text)
            lines.append(r"\textbf{" + esc(labels[m]) + "} & " + " & ".join(entries) + r" \\")
        if methods:
            lines.append(r"\midrule")
            for method in methods:
                if not all((method, c) in method_cell for c in cids):
                    continue
                lines.append(short(method) + " & " + " & ".join(
                    num(method_cell[method, c], digits, scale) for c in cids) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        out.append(_wrap("\n".join(lines), title + (" (continued)" if part else ""), f"{label}-{part + 1}",
                         size=r"\tiny\setlength{\tabcolsep}{2.4pt}"))
    return "\n".join(out)


def metric_table(block, labels_of_methods=None):
    """Every method on the benchmark metrics (appendix)."""
    methods = block["methods"]
    order = [m for _, group in APPENDIX_GROUPS for m in group if m in methods]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Method & AvgAcc & Log loss & Gain@$\mathcal{R}$ & Gain@$\mathcal{B}$ & Gain@$\mathcal{B}_d$ & 95\% interval \\", r"\midrule"]
    for m in order:
        e = methods[m]
        ci = e.get("gain_b_ci")
        lines.append(" & ".join([esc(NAMES.get(m, m)) + _mark(m), f"{100 * e['avg_acc']:.2f}", f"{e['log_loss']:.3f}",
                                 signed(e["gain_r"], 2, 100), signed(e["gain_t"], 2, 100), signed(e["gain_b"], 2, 100),
                                 ("[" + ", ".join(signed(v, 2, 100) for v in ci) + "]") if ci and m != "best_single"
                                 else "--"]) + r" \\")
    refs = block["references"]
    lines.append(r"\midrule")
    lines.append(" & ".join(["Best Single $\\mathcal{B}$ (test folds)", f"{100 * refs['T']['accuracy']:.2f}", "--", "--", "--",
                             signed(refs["T"]["gain_b"], 2, 100), "--"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("Benchmark metrics of the methods of Table~\\ref{tab:main-results}, in percent except log loss: gain over "
               "the random model $\\mathcal{R}$, Best Single $\\mathcal{B}$ and the best model per benchmark $\\mathcal{B}_d$, "
               "with the 95\\% interval of Gain@$\\mathcal{B}_d$ (resampling within benchmarks). Gain@$b$ is the mean "
               "over benchmarks of the accuracy ratio to $b$, minus one, as in LLMRouterBench.")
    return _wrap("\n".join(lines), caption, "benchmark-metrics", size=r"\footnotesize\setlength{\tabcolsep}{4pt}")


POLICY_NAMES = {"ranked_top_k": "Top-k pool", "random_panel": "Random panel", "single_by_cost": "Per-benchmark control"}


def policy_table(block):
    """PerfGain, CostSave and ParetoDist of every policy (appendix)."""
    policies = block["policies"]

    def pct(v):
        return "N/A" if v is None else signed(v, 1, 100)
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Against $\mathcal{B}$} & \multicolumn{2}{c}{Against $\mathcal{B}_d$} & \multicolumn{2}{c}{} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Policy & Settings & PerfGain & CostSave & PerfGain & CostSave & CostSave (loss) & ParetoDist \\",
             r"\midrule"]
    for name, e in sorted(policies.items(), key=lambda kv: (kv[1].get("pareto_dist") is None,
                                                            kv[1].get("pareto_dist") or 0)):
        if name == "oracle":
            continue
        label = POLICY_NAMES.get(name) or NAMES.get(name, name)
        lines.append(" & ".join([esc(label) + _mark(name), str(e["settings"]), pct(e["perf_gain_T"]),
                                 pct(e["cost_save_T"]), pct(e["perf_gain_B"]), pct(e["cost_save_B"]),
                                 pct(e["cost_save_loss_B"]),
                                 f"{e['pareto_dist']:.3f}" if e.get("pareto_dist") is not None else "--"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("Policies over their settings (best settings chosen on the test folds, as in LLMRouterBench), in "
               "percent: PerfGain, the accuracy of the best setting over the reference's; CostSave, the cost saved "
               "by the cheapest setting at least as accurate as the reference (N/A: none is), also with log loss in "
               "place of accuracy; ParetoDist, the mean distance of the settings to the frontier of "
               "Figure~\\ref{fig:pareto}. Replayed methods are not on the frontier.")
    return _wrap("\n".join(lines), caption, "policies", size=r"\footnotesize\setlength{\tabcolsep}{4pt}")


def settings_table(settings):
    """The grids and seeds of the experiments (appendix), from the settings file."""
    r, c, f, b, lb = (settings["knn_router"], settings["confidence_cascade"], settings["frontier"],
                      settings["bootstrap"], settings["label_budget"])

    def values(v):
        return ", ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in v)
    rows = [("kNN router", "neighbour count $k$ (tuned on the validation fold)", values(r["k_grid"])),
            ("kNN router", "cost weight $\\lambda$", values(r["lambda_grid"])),
            ("Confidence cascade", "threshold $\\tau$; at most " + str(c["max_stages"]) + " models", values(c["tau_grid"])),
            ("Panels by size", "number of members $k$", f"{min(f['sizes'])} to {max(f['sizes'])}"),
            ("Random panels", "seeds", values(f["random_panel_seeds"])),
            ("Label budgets", f"labelled source groups ({lb['draws']} draws each)", values(lb["groups"]) + ", all"),
            ("Bootstrap", "paired draws; seed", f"{b['draws']:,}; {b['seed']}")]
    lines = [r"\begin{tabular}{@{}llp{0.36\linewidth}@{}}", r"\toprule", r"Policy & Setting & Grid \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = ("Grids and seeds of the experiments, from the settings file whose hash every result file records. "
               "The router and the cascade take the grid value chosen on the validation fold under the cost cap "
               "(Appendix~\\ref{sec:setup}); the panels take a size chosen on the validation fold.")
    return _wrap("\n".join(lines), caption, "settings", size=r"\footnotesize")

"""Generate the paper's assets from the frozen corpus and aggregate results."""

import argparse
import csv
import json
import math
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from acl_resource.data.schema import sha256_file
from acl_resource.sim.types import RESULT_ROW_KEYS


# Figure output


TASK_AREAS = ("knowledge", "language", "retrieval", "tools", "arts")
INK = "#20485b"
PALETTE = ("#0072b2", "#bc5100", "#008563", "#814f82", "#333333")
MARKERS = ("o", "s", "^", "D", "x")


def _figure(width, height):
    """Return a vector-ready figure with dimensions in centimetres."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width / 2.54, height / 2.54))
    FigureCanvasAgg(figure)
    return figure


def _mean(values):
    """Return the equal-benchmark mean, or NaN for an empty cell."""
    return math.fsum(values) / len(values) if values else math.nan


def _model_blocks(pairs, models):
    """Return organisation blocks ordered by their smallest model index."""
    neighbours = {model: set() for model in models}
    for row in pairs:
        i, j = int(row["model_i"]), int(row["model_j"])
        neighbours.setdefault(i, set())
        neighbours.setdefault(j, set())
        if row["same_organisation"].lower() == "true":
            neighbours[i].add(j)
            neighbours[j].add(i)
    blocks = []
    remaining = set(neighbours)
    while remaining:
        block = set()
        queue = [min(remaining)]
        while queue:
            index = queue.pop()
            if index not in block:
                block.add(index)
                queue.extend(neighbours[index] - block)
        blocks.append(sorted(block))
        remaining -= block
    return blocks


def _similarity(pairs, blocks):
    """Return the suite correlation heat map and its coverage caption."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    models = [model for block in blocks for model in block]
    positions = {model: index for index, model in enumerate(models)}
    values = defaultdict(list)
    for row in pairs:
        if row["error_correlation"]:
            key = tuple(sorted((int(row["model_i"]), int(row["model_j"]))))
            values[key].append(float(row["error_correlation"]))
    size = len(models)
    matrix = np.full((size, size), np.nan)
    counts = np.zeros((size, size), dtype=int)
    for (i, j), scores in values.items():
        a, b = positions[i], positions[j]
        matrix[a, b] = matrix[b, a] = _mean(scores)
        counts[a, b] = counts[b, a] = len(scores)
    fig = _figure(16, 13)
    ax = fig.subplots()
    fig.subplots_adjust(left=.09, right=.79, top=.985, bottom=.11)
    cmap = LinearSegmentedColormap.from_list("correlation", ["#4c91b7", "#fafafa", "#873700"])
    mesh = ax.pcolormesh(np.arange(size + 1) - .5, np.arange(size + 1) - .5,
                         matrix, cmap=cmap, vmin=-1, vmax=1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set(xticks=range(size), xticklabels=models, yticks=range(size), yticklabels=models,
           xlabel="Model index", ylabel="Model index")
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelrotation=90)
    border = -.5
    for block in blocks[:-1]:
        border += len(block)
        ax.axvline(border, color="0.35", linewidth=.7)
        ax.axhline(border, color="0.35", linewidth=.7)
    observed = set(counts[counts > 0].tolist())
    for a in range(size):
        for b in range(size):
            if a == b:
                ax.plot([b - .3, b + .3], [a + .3, a - .3], color="0.35", linewidth=.8)
            elif counts[a, b] == 0:
                ax.plot(b, a, marker="x", markersize=4, color="0.25")
            elif len(observed) > 1:
                ax.text(b, a, str(counts[a, b]), ha="center", va="center",
                        color="white" if matrix[a, b] > .65 else "black")
            if matrix[a, b] < 0:
                ax.plot(b - .34, a + .34, marker=".", markersize=2, color="black")
    bar = fig.colorbar(mesh, cax=fig.add_axes([.82, .21, .026, .67]), ticks=[-1, -.5, 0, .5, 1])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    bar.set_label("Error correlation")
    coverage = (f"{next(iter(observed))} benchmarks per defined pair." if len(observed) == 1
                else "Cell numbers count contributing benchmarks." if observed
                else "No pair has a defined correlation.")
    caption = ("Mean error correlation by model pair. Lines delimit organisations. "
               "Slashes indicate self-comparisons, crosses indicate missing pairs and dots mark "
               "negative correlations. " + coverage)
    return fig, caption


def _reliability(rows, models):
    """Return reliability panels and per-bin benchmark counts."""
    bins = [int(row["bin"]) for row in rows]
    bins = list(range(min(bins), max(bins) + 1)) if bins else []
    points = defaultdict(list)
    for row in rows:
        if int(row["count"]) and row["mean_confidence"] and row["accuracy"]:
            points[int(row["model_index"]), int(row["bin"])].append(
                (float(row["mean_confidence"]), float(row["accuracy"])))
    columns = min(4, len(models))
    lines = math.ceil(len(models) / columns)
    fig = _figure(16, 1 + 2.75 * lines)
    axes = list(fig.subplots(lines, columns, squeeze=False).flat)
    fig.subplots_adjust(left=.09, right=.985, top=.975, bottom=.12, hspace=1.0, wspace=.25)
    for ax, model in zip(axes, models):
        cells = [points[model, index] for index in bins]
        ax.plot([0, 1], [0, 1], color=".6", ls="--", lw=.7)
        ax.plot([_mean([x for x, y in cell]) for cell in cells],
                [_mean([y for x, y in cell]) for cell in cells],
                color=INK, marker="o", ms=2.5, lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 1], yticks=[0, 1])
        ax.tick_params(length=2, pad=1)
        ax.text(.04, .92, str(model), transform=ax.transAxes, ha="left", va="top")
        counts = [str(len(cell)) for cell in cells]
        half = math.ceil(len(counts) / 2)
        ax.text(.5, -.4, " ".join(counts[:half]) + "\n" + " ".join(counts[half:]),
                transform=ax.transAxes, ha="center", va="top", linespacing=.96)
    for ax in axes[len(models):]:
        ax.set_visible(False)
    fig.supylabel("Accuracy (share of judged fields)", fontsize=10, x=.005)
    fig.supxlabel("Mean confidence (probability)", fontsize=10, y=.005)
    return fig, ("Reliability by model index. The dashed line indicates equality. Counts under "
                 "each panel give contributing benchmarks in increasing bin order, read across "
                 "each line. Empty bins interrupt the curve.")


def _label_points(ax, points):
    """Place model indices next to points, avoiding other indices and markers."""
    from matplotlib.transforms import Bbox

    scale = ax.figure.dpi / 72
    bounds = ax.get_window_extent()
    locations = [ax.transData.transform((x, y)) for x, y, label in points]
    obstacles = [Bbox.from_bounds(x - 2 * scale, y - 2 * scale, 4 * scale, 4 * scale)
                 for x, y in locations]
    for (x, y, label), (px, py) in zip(points, locations):
        width, height = (len(label) * 6.2 + 1) * scale, 11 * scale
        best = None
        for distance in (8, 13, 19, 27, 36, 48, 65):
            for step in range(16):
                angle = math.tau * step / 16
                dx, dy = distance * math.cos(angle), distance * math.sin(angle)
                box = Bbox.from_bounds(px + dx * scale - width / 2,
                                       py + dy * scale - height / 2, width, height)
                if not (bounds.contains(*box.min) and bounds.contains(*box.max)):
                    continue
                overlaps = 0.
                for other in obstacles:
                    if box.overlaps(other):
                        overlap = Bbox.intersection(box, other)
                        overlaps += overlap.width * overlap.height
                score = overlaps / scale**2 + distance / 100
                if best is None or score < best[0]:
                    best = score, box, dx, dy
            if best is not None and best[0] < 1:
                break
        if best is None:
            continue
        _, box, dx, dy = best
        obstacles.append(box)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center",
                    arrowprops={"arrowstyle": "-", "lw": .5, "color": ".45",
                                "shrinkA": 0, "shrinkB": 1.5})


def _accuracy_cost(rows, models):
    """Return the model cost panels and coverage for each axis."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    groups = defaultdict(list)
    for row in rows:
        for area in ("Suite", row["area"]):
            groups[area, int(row["model_index"])].append(row)
    fig = _figure(16, 15)
    axes = list(fig.subplots(3, 2, sharex=True, sharey=True).flat)
    fig.subplots_adjust(left=.11, right=.98, top=.97, bottom=.09, wspace=.09, hspace=.13)
    panels = []
    for ax, area in zip(axes, ("Suite", *TASK_AREAS)):
        summaries = []
        for model in models:
            records = groups[area, model]
            cost = [float(r["usd_per_1k_requests"]) for r in records if r["usd_per_1k_requests"]]
            accuracy = [float(r["accuracy_first_max"]) for r in records if r["accuracy_first_max"]]
            if cost and accuracy:
                n = str(len(cost)) if len(cost) == len(accuracy) else f"{len(cost)}/{len(accuracy)}"
                summaries.append((_mean(cost), _mean(accuracy), model, n))
        counts = {n for _, _, _, n in summaries}
        points = [(x, y, str(model) if len(counts) == 1 else f"{model}({n})")
                  for x, y, model, n in summaries]
        ax.scatter([x for x, y, label in points], [y for x, y, label in points],
                   s=9, color=INK)
        panels.append(points)
        n = f" (n={next(iter(counts))})" if len(counts) == 1 else ""
        ax.text(.98, .98, area.capitalize() + n, transform=ax.transAxes, va="top", ha="right",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": .5})
    costs = [x for panel in panels for x, _, _ in panel]
    log_cost = bool(costs) and min(costs) > 0
    if log_cost:
        axes[0].set_xscale("log")
        axes[0].xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        for ax in axes:
            ax.minorticks_off()
    axes[0].margins(x=.12, y=.19)
    fig.supxlabel("Cost (USD per 1,000 requests)", fontsize=10, y=.008)
    fig.supylabel("First-maximum accuracy (share of judged fields)", fontsize=10, x=.008)
    fig.canvas.draw()
    for ax, points in zip(axes, panels):
        _label_points(ax, points)
    caption = ("Mean accuracy and cost by model index for the suite and each task area. "
               "Panel counts give contributing benchmarks. Counts that vary by model appear "
               "in parentheses, with unequal cost and accuracy counts separated by a slash.")
    if log_cost:
        caption += " The cost axis is logarithmic."
    return fig, caption


def _headroom(rows):
    """Return benchmark headroom and four aligned field-share columns."""
    rows = sorted(rows, key=lambda row: (TASK_AREAS.index(row["area"]), int(row["catalog_id"])))
    small = 7.5                                   # the type size of the other figures
    fig = _figure(16, 1.7 + .30 * len(rows))
    axes = fig.subplots(1, 5, sharey=True, gridspec_kw={"width_ratios": [1.55, 1, 1, 1, 1]})
    fig.subplots_adjust(left=.17, right=.985, top=.955, bottom=.085, wspace=.18)
    labels = []
    for y, row in enumerate(rows):
        new_area = y == 0 or rows[y - 1]["area"] != row["area"]
        from .paper_tables import BENCH_ABBR
        name = BENCH_ABBR.get(int(row["catalog_id"]), row["catalog_id"])
        labels.append((row["area"].capitalize() + "   " if new_area else "") + name)
        if new_area and y:
            for ax in axes:
                ax.axhline(y - .5, color=".72", lw=.55)
        endpoints = [row["best_single_share"], row["at_least_one_share"]]
        if all(endpoints):
            axes[0].plot([float(v) for v in endpoints], [y, y], color=INK, lw=.9)
        for value, marker in zip(endpoints, ("o", "|")):
            if value:
                axes[0].plot(float(value), y, color=INK, marker=marker, ms=3.2, clip_on=False)
        for ax, key in zip(axes[1:], ("at_least_two_share", "majority_share", "all_share", "none_share")):
            if row[key]:
                ax.plot(float(row[key]), y, color=INK, marker="o", ms=2.4, clip_on=False)
    axes[0].set_yticks(range(len(rows)), labels, fontsize=small)
    axes[0].set_ylim(len(rows) - .5, -.5)
    for ax, label in zip(axes, ("Single to any", "At least two", "Majority", "All", "None")):
        ax.set(xlim=(0, 1), xticks=[0, 1])
        ax.grid(axis="x", linewidth=.5, color=".8")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=small)
        ax.text(.5, 1.013, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=small)
    fig.supxlabel("Share of judged questions", fontsize=small, y=.013)
    return fig, ("Headroom by benchmark. Left: the share of judged "
                 "questions that the most accurate single model gets right (dot) and that at least one model gets "
                 "right (tick). Other columns: the share that at least two models, a majority, all models and no "
                 "model get right.")


def _series_labels(ax, ends):
    """Return end labels spaced vertically with leaders to their series."""
    if not ends:
        return
    low, high = ax.get_ylim()
    height = high - low
    gap = height * .13
    ends = sorted(ends, key=lambda item: item[1])
    positions = []
    for _, y, _ in ends:
        positions.append(max(y, low + gap / 2, positions[-1] + gap if positions else low))
    for i in reversed(range(len(positions))):
        ceiling = positions[i + 1] - gap if i + 1 < len(positions) else high - gap / 2
        positions[i] = min(positions[i], ceiling)
    for (x, y, label), pos in zip(ends, positions):
        ax.annotate(label, xy=(x, y), xytext=(1.025, pos), textcoords=ax.get_yaxis_transform(),
                    va="center", ha="left", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": ".45", "lw": .65})


CURVE_LABELS = {"ranked_top_k": "Top-k pool", "greedy": "Greedy", "random_panel": "Random pool",
                "knn_router": "Router", "confidence_cascade": "Cascade",
                "best_single": "Best single", "stack": "Stack"}


def _coverage(rows):
    """Return benchmark coverage for supplied suite operating points."""
    counts = sorted({row["n_benchmarks"] for row in rows})
    if not counts:
        return " No operating points reported."
    if len(counts) == 1:
        return f" Each point includes {counts[0]} benchmarks."
    return " Contributing benchmarks per point range from " + f"{counts[0]} to {counts[-1]}."


def _frontier(rows, xkey, xlabel):
    """Return all supplied operating points and their nondominated frontier."""
    fig = _figure(7.7, 7.0)
    ax = fig.subplots()
    rows = [row for row in rows if row[xkey] is not None and row["accuracy"] is not None]
    methods = sorted({row["method"] for row in rows})
    legend = len(methods) + 1 > 4
    fig.subplots_adjust(left=.22, right=.975 if legend else .72, top=.96, bottom=.42 if legend else .2)
    ends = []
    points = []
    for i, method in enumerate(methods):
        pairs = sorted((row[xkey], row["accuracy"]) for row in rows if row["method"] == method)
        label = CURVE_LABELS.get(method, method.replace("_", " "))
        ax.scatter(*zip(*pairs), label=label, marker=MARKERS[i % len(MARKERS)],
                   color=PALETTE[i % len(PALETTE)], s=14)
        ends.append((*pairs[-1], label))
        points.extend(pairs)
    frontier = []
    for x, accuracy in sorted(set(points), key=lambda pair: (pair[0], -pair[1])):
        if not frontier or accuracy > frontier[-1][1]:
            frontier.append((x, accuracy))
    if frontier:
        ax.step(*zip(*frontier), where="post", color="0.25", lw=.9, zorder=0, label="Frontier")
        ends.append((*frontier[-1], "Frontier"))
    ax.set(xlabel=xlabel, ylabel="Accuracy (share)")
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(.45, -.32), ncol=2,
                  frameon=False, columnspacing=.7, handlelength=1.2, labelspacing=.2)
    else:
        _series_labels(ax, ends)
    if not rows:
        ax.set_axis_off()
    return fig, ("Accuracy of the evaluated operating points. The step line connects "
                 "nondominated points in increasing " + ("cost." if xkey.startswith("usd") else "latency.")
                 + _coverage(rows))


def _size_curve(rows, budget):
    """Return log loss against panel size or ordered label budgets."""
    prefix = "groups=" if budget else "k="
    rows = [row for row in rows if (row["variant"] or "").startswith(prefix)]
    series = defaultdict(dict)
    for row in rows:
        level = row["variant"].removeprefix(prefix)
        if level != "all":
            level = int(level)
        series[row["method"]][level] = row["log_loss"]
    levels = sorted({level for data in series.values() for level in data},
                    key=lambda level: math.inf if level == "all" else level)
    fig = _figure(7.7, 5.8)
    ax = fig.subplots()
    fig.subplots_adjust(left=.22, right=.71 if budget else .96, bottom=.24, top=.96)
    ends = []
    for i, (method, data) in enumerate(sorted(series.items())):
        x = list(range(len(levels))) if budget else levels
        y = [data.get(level) for level in levels]
        label = CURVE_LABELS.get(method, method.replace("_", " "))
        ax.plot(x, y, marker=MARKERS[i % len(MARKERS)], color=PALETTE[i % len(PALETTE)],
                ms=3, lw=.9, label=label)
        available = [(position, loss) for position, loss in zip(x, y) if loss is not None]
        if available:
            ends.append((*available[-1], label))
    if budget:
        ax.set_xticks(range(len(levels)), [str(level) for level in levels])
    ax.set(xlabel="Label budget (split groups)" if budget else "Panel size (members)",
           ylabel="Log loss (nats per question)")
    if len(ends) > 4 or not budget:           # the panel curves meet at the full panel, so end labels would collide
        ax.legend(frameon=False, fontsize=8, loc="upper right" if not budget else "best")
    else:
        _series_labels(ax, ends)
    available_rows = [row for row in rows if row["log_loss"] is not None]
    if not available_rows:
        ax.set_axis_off()
    if budget:
        return fig, ("Mean log loss per judged question by label budget. The full budget uses every available "
                     "training group." + _coverage(available_rows))
    return fig, ("Log loss on the headline suite by panel size: equal-weight pools of the k best-ranked models "
                 "(top-k pool) and of k random models (random pool, mean over seeds), and greedy selection with "
                 "k members.")


def _make_figures(out, paths, data, documents=None):
    """Return captions and source selectors for the vector PDFs written to disk."""
    documents = documents or {}
    settings = data["SETTINGS.json"]
    suite = settings["headline_suite"]
    excluded = set(settings["suites"][suite]["exclude"])
    suite_sources = [("SETTINGS.json", f"headline_suite and suites.{suite}.exclude")]
    suite_selector = f"catalog_id outside suites.{suite}.exclude"
    pairs_file = "diagnostics/pairs.csv"
    models_file = "diagnostics/models.csv"
    reliability_file = "diagnostics/reliability.csv"
    headroom_file = "diagnostics/headroom.csv"
    diag_file = "diagnostics/diagnostics.json"
    diag_made_up = data[diag_file]["synthetic"]
    pairs = read_csv(paths[pairs_file])
    models = read_csv(paths[models_file])
    blocks = _model_blocks(pairs, {int(row["model_index"]) for row in models})
    order = [model for block in blocks for model in block]
    order_sources = [(models_file, "all model_index values"),
                     (pairs_file, "all model_i, model_j, same_organisation")]
    folder = out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    assets = {}

    def selected(rows):
        """Return rows in the headline suite according to the settings."""
        return [row for row in rows if int(row["catalog_id"]) not in excluded]

    def save(stem, figure_caption, sources, synthetic, suite_caption=True):
        """Write one PDF and record its caption, size and source selectors."""
        fig, caption = figure_caption
        if suite_caption:
            count = len({int(r["catalog_id"]) for r in read_csv(paths["corpus/benchmarks.csv"])} - excluded)
            caption = f"Headline suite ({count} benchmarks). " + caption
        if synthetic:
            caption += " Synthetic corpus."
        fig.savefig(folder / f"{stem}.pdf",
                    metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None})
        assets[stem] = {"caption": caption, "sources": sources}

    diag_sources = [(diag_file, "synthetic")]
    base = suite_sources + order_sources + diag_sources
    from . import paper_figures

    organisation = {int(r["model_index"]): r["organisation"] for r in read_csv(paths["corpus/models.csv"])}
    if PRESENTATION["model_names"] == "coded":
        codes = organisation_codes(read_csv(paths["corpus/models.csv"]))
        organisation = {m: codes[name] for m, name in organisation.items()}
    save("similarity", paper_figures.similarity(selected(pairs), blocks, organisation),
         base + [(pairs_file, suite_selector + ": error_correlation by model_i and model_j"),
                 ("corpus/models.csv", "model_index, organisation")],
         diag_made_up, suite_caption=False)
    save("reliability", _reliability(selected(read_csv(paths[reliability_file])), order),
         base + [(reliability_file, suite_selector + ": count, mean_confidence, accuracy by model_index and bin")],
         diag_made_up)
    save("accuracy_cost", _accuracy_cost(selected(models), order),
         base + [(models_file, suite_selector + ": usd_per_1k_requests, accuracy_first_max by model_index and area")],
         diag_made_up)
    held = set(PRESENTATION["held_benchmarks"])
    save("headroom", _headroom([r for r in read_csv(paths[headroom_file]) if int(r["catalog_id"]) not in held]),
         diag_sources + [(headroom_file, "all catalog_id, area, best_single_share, at_least_one_share, "
                          "at_least_two_share, majority_share, all_share, none_share")],
         diag_made_up, suite_caption=False)
    frontier_file = "results/frontier.json"
    frontier = data[frontier_file]
    rows = [row for row in frontier["rows"] if row["suite"] == suite]
    source = f"rows with suite={suite}"
    for stem, xkey, xlabel in [("frontier_cost", "usd_per_1k_requests", "Cost (USD per 1,000 requests)"),
                               ("frontier_latency", "latency_mean_s", "Mean latency (s per request)")]:
        save(stem, _frontier(rows, xkey, xlabel),
             suite_sources + [(frontier_file, f"synthetic and {source}: method, {xkey}, accuracy, n_benchmarks")],
             frontier["synthetic"])
    suite_rows = [r for r in data["results/rows.json"]["rows"] if r["suite"] == suite and r["catalog_id"] is None]
    single_models = data[diag_file]["suites"][suite]["overall"]["models"]
    save("operating_points", paper_figures.operating_points(rows, suite_rows, single_models),
         suite_sources + [(frontier_file, f"synthetic and {source}: method, variant, accuracy, usd_per_1k_requests, latency_mean_s"),
                          ("results/rows.json", f"suite={suite}, catalog_id=null: accuracy, usd_per_1k_requests, latency_mean_s"),
                          (diag_file, f"suites.{suite}.overall.models: accuracy_first_max, usd_per_1k_requests")],
         frontier["synthetic"] or diag_made_up)
    save("panel_size", _size_curve(rows, False),
         suite_sources + [(frontier_file, f"synthetic and {source}, variant=k=*: method, log_loss, n_benchmarks")],
         frontier["synthetic"], suite_caption=False)
    budget_file = "results/ablations.json"
    budgets = data[budget_file]
    rows = [row for row in budgets["rows"] if row["suite"] == suite]
    save("label_budget", paper_figures.label_budget(rows, data["SETTINGS.json"]["label_budget"]["draws"]),
         suite_sources + [(budget_file, f"synthetic and {source}, variant=groups=* and method=transfer_single: "
                                        "method, log_loss, accuracy, n_benchmarks"),
                          ("SETTINGS.json", "label_budget.draws")],
         budgets["synthetic"], suite_caption=False)
    restyle_figures(save, documents, data, paths, suite, frontier, suite_sources)
    return assets


def restyle_figures(save, documents, data, paths, suite, frontier, suite_sources):
    """Figures of the restyled paper, from the documents bound with --json (each only when it is given)."""
    from . import paper_figures as PF
    from . import paper_tables as PT

    coded = PRESENTATION["model_names"] == "coded"
    labels = PT.model_labels(read_csv(paths["corpus/models.csv"]), coded)
    if "bm" in documents:
        role, bm = documents["bm"]
        block = bm["suites"][suite]
        selector = f"suites.{suite}"
        # The third panel is in accuracy points: the paired differences to the best model per benchmark.
        contrast = {r["contrast"]: r for r in data["results/contrasts.json"]["rows"]
                    if r["suite"] == suite and r["metric"] == "accuracy"}
        specs = {(r["a"], r["b"]): c for c, r in contrast.items()}
        points = {}
        for method in block["methods"]:
            name = specs.get((method, "best_single"))
            if name and contrast[name]["difference"] is not None:
                r = contrast[name]
                points[method] = (100 * r["difference"], 100 * r["ci_low"], 100 * r["ci_high"])
        stacked, exit_ = block["methods"].get("stack_final_temperature"), block["methods"].get("certified_exit")
        if stacked and exit_ and "stack_final_temperature" in points and stacked["avg_acc"] == exit_["avg_acc"]:
            points["certified_exit"] = points["stack_final_temperature"]      # the same answers, question by question
        extra, bar_sources = [], [("results/contrasts.json", f"suite={suite}, metric=accuracy: a - best_single")]
        if "ops" in documents:
            ops = documents["ops"][1]["suites"][suite]
            p, e = ops["points"]["confidence_cascade"]["uncapped"], ops["contrasts"]["cascade_uncapped_vs_single"]
            extra.append(dict(label="Cascade, no cap", colour="#f3a98a", log_loss=p["log_loss"], avg_acc=p["accuracy"],
                              points=tuple(100 * e["accuracy"][k] for k in ("difference", "ci_low", "ci_high"))))
            bar_sources.append((documents["ops"][0], f"suites.{suite}: points.confidence_cascade.uncapped, "
                                                     "contrasts.cascade_uncapped_vs_single"))
        save("metric_bars", PF.metric_bars(block, points, extra),
             suite_sources + [(role, selector + ": methods, references")] + bar_sources,
             bm["synthetic"], suite_caption=False)
        save("bubbles", PF.bubbles(block, None if coded else labels),
             suite_sources + [(role, selector + ".single_models"), ("corpus/models.csv", "model_index, model_name")],
             bm["synthetic"], suite_caption=False)
        save("gains", PF.gains(block), suite_sources + [(role, selector + ".policies")], bm["synthetic"],
             suite_caption=False)
        rows = [r for r in frontier["rows"] if r["suite"] == suite and r["catalog_id"] is None]
        save("pareto", PF.pareto(block, rows),
             suite_sources + [(role, selector + ": policies, methods, references, frontier, single_models"),
                              ("results/frontier.json", f"rows with suite={suite}: method, accuracy, usd_per_1k_requests")],
             bm["synthetic"] or frontier["synthetic"], suite_caption=False)
    if "hard" in documents:
        role, hard = documents["hard"]
        block = hard["suites"][suite]
        save("hardness", PF.hardness(block), suite_sources + [(role, f"suites.{suite}.bins")], hard["synthetic"],
             suite_caption=False)
        # The figure of the oracle over k models (PF.subsets) is not written: the paper prints no oracle.
    if "rstudy" in documents:
        role, study = documents["rstudy"]
        held = set(PRESENTATION["held_benchmarks"])
        catalog = {int(b["catalog_id"]): b for b in read_benchmarks(paths, documents)}
        chosen = donut_benchmarks(catalog, read_csv(paths["diagnostics/models.csv"]), held, study["routing"])
        if chosen:
            save("donuts", PF.donuts(study["routing"], [(c, PT.BENCH_ABBR.get(c, catalog[c]["name"])) for c in chosen],
                                     labels),
                 [(role, "routing: " + ", ".join(str(c) for c in chosen)), ("corpus/models.csv", "model_index")],
                 study["synthetic"], suite_caption=False)


def donut_benchmarks(catalog, model_rows, held, routing):
    """The benchmark with the most judged fields in each task area, and the one with the lowest mean model accuracy."""
    shown = {c: b for c, b in catalog.items() if c not in held and str(c) in routing}
    chosen = []
    for area in ("knowledge", "language", "retrieval", "tools", "arts"):
        members = [c for c, b in shown.items() if b["area"] == area]
        if members:
            chosen.append(max(members, key=lambda c: (int(shown[c]["judged_fields"]), -c)))
    accuracy = {}
    for r in model_rows:
        c = int(r["catalog_id"])
        if c in shown:
            accuracy.setdefault(c, []).append(float(r["accuracy_first_max"]))
    if accuracy:
        hardest = min(accuracy, key=lambda c: (sum(accuracy[c]) / len(accuracy[c]), c))
        if hardest not in chosen:
            chosen.append(hardest)
    return chosen


def write_figures(out, paths, data, documents=None):
    """Return source selectors and captions for the deterministic vector PDFs."""
    import os

    # Keep plotting-library caches out of both the release and the calling account.
    with tempfile.TemporaryDirectory(dir=out.parent) as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib

            with matplotlib.rc_context(rc={"pdf.fonttype": 42, "pdf.use14corefonts": False,
                                           "font.family": "DejaVu Sans", "font.size": 10,
                                           "text.usetex": False, "axes.spines.top": False,
                                           "axes.spines.right": False, "axes.linewidth": .65}):
                return _make_figures(out, paths, data, documents)
        finally:
            if previous is None:
                del os.environ["MPLCONFIGDIR"]
            else:
                os.environ["MPLCONFIGDIR"] = previous


# Number and table output

PACKAGE = Path(__file__).resolve().parents[1]
METRICS = ("log_loss", "accuracy", "calls_per_request", "usd_per_1k_requests", "latency_mean_s")
METRIC_HEADER = ("Method", "Log loss", "Accuracy", "Calls/req.", "USD/1,000 req.", "Latency (s)")
ANCHORS = {"best_single", "uniform_mean"}


def read_json(path):
    """Return the decoded JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    """Return the CSV records in file order."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def tex_text(value):
    """Return literal text escaped for a LaTeX table or caption."""
    special = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "%": r"\%", "$": r"\$",
               "#": r"\#", "&": r"\&", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(special.get(c, c) for c in str(value))


def format_number(fmt, value):
    """Return a finite numeric scalar formatted for LaTeX (format ``text``: a string, escaped)."""
    if fmt == "text":
        if not isinstance(value, str):
            raise ValueError("text format requires a string")
        return tex_text(value)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("expected a finite numeric scalar")
    if fmt == "n":
        if value != int(value):
            raise ValueError("integer format requires an integer")
        return f"{int(value):,}"
    if fmt == "p":
        if not 0 <= value <= 1:
            raise ValueError("p-value outside the unit interval")
        return r"\ensuremath{< 0.001}" if value < .001 else f"{value:.3f}"
    if not re.fullmatch(r"f[0-4]|a[0-4]|s?pts[0-2]|pct[0-2]|usd[2-4]|s[1-4]", fmt):
        raise ValueError("unknown number format " + fmt)
    digits = int(fmt[-1])
    if fmt.startswith("a"):                      # the size of a difference whose direction the sentence states
        return f"{abs(value):.{digits}f}"
    if fmt.startswith("pts"):                    # the size of a difference of two shares, in percentage points
        return f"{abs(value) * 100:.{digits}f}"
    if fmt.startswith("spts"):                   # the same with its sign
        return r"\ensuremath{" + f"{value * 100:+.{digits}f}" + "}"
    if fmt.startswith("pct"):
        return f"{value * 100:.{digits}f}" + r"\%"
    if fmt.startswith("usd"):
        return r"\$" + f"{value:.{digits}f}"
    if fmt.startswith("s"):
        return r"\ensuremath{" + f"{value:+.{digits}f}" + "}"
    return f"{value:.{digits}f}"


def row_index(rows, fields):
    """Return rows and their source offsets indexed by the given fields."""
    indexed = {}
    for offset, row in enumerate(rows):
        key = tuple(row[field] for field in fields)
        if key in indexed:
            raise ValueError("ambiguous input rows for " + str(key))
        indexed[key] = (offset, row)
    return indexed


def resolve_number(key, documents, suites, contrasts, headline):
    """Return the scalar, input role and exact JSON path for a prose key."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
        raise ValueError("invalid number key")
    prefix, *fields = key.split(".")
    if prefix == "rows":
        suite, method, metric = fields
        if metric not in RESULT_ROW_KEYS:
            raise KeyError(metric)
        offset, row = suites[(headline if suite == "headline" else suite, method)]
        return row[metric], "results/rows.json", f"rows.{offset}.{metric}"
    if prefix == "contrasts":
        name, suite, metric, field = fields
        if field not in ("difference", "ci_low", "ci_high", "p", "p_holm"):
            raise KeyError(field)
        offset, row = contrasts[(name, headline if suite == "headline" else suite, metric)]
        return row[field], "results/contrasts.json", f"rows.{offset}.{field}"
    role, value = documents[prefix]
    if not fields:
        raise KeyError(key)
    for part in fields:
        if isinstance(value, list):
            if not part.isascii() or not part.isdecimal():
                raise KeyError(part)
            value = value[int(part)]
        else:
            value = value[part]
    return value, role, ".".join(fields)


def write_numbers(out, paper, documents, rows, contrasts, headline, synthetic):
    """Write the macro and return provenance selectors for the requested pairs."""
    from acl_resource.analysis.lint_paper import iter_numbers

    suites = row_index(rows, ("suite", "catalog_id", "method"))
    suites = {(suite, method): item for (suite, cid, method), item in suites.items() if cid is None}
    contrasts = row_index(contrasts, ("contrast", "suite", "metric"))
    definitions, sources = {}, {}
    for path in [paper / "main.tex", *sorted((paper / "sections").glob("*.tex"))]:
        try:
            for fmt, key, line in iter_numbers(path.read_text(encoding="utf-8")):
                if (fmt, key) in definitions:
                    continue
                try:
                    value, role, selector = resolve_number(key, documents, suites, contrasts, headline)
                    definitions[fmt, key] = format_number(fmt, value)
                except (IndexError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{line}: invalid number [{fmt}]{{{key}}} ({error})") from None
                refs = [(role, selector)]
                if (key.startswith("rows.headline.") or
                        (key.startswith("contrasts.") and key.split(".")[2] == "headline")):
                    refs.append(("SETTINGS.json", "headline_suite"))
                sources[f"{fmt}:{key}"] = refs
        except ValueError as error:
            raise ValueError(f"{path.relative_to(paper)}:{str(error).removeprefix('line ')}") from None

    lines = [r"\newif\ifpapersynthetic", r"\papersynthetictrue" if synthetic else r"\papersyntheticfalse",
             r"\newif\ifpapercoded", r"\papercodedtrue" if PRESENTATION["model_names"] == "coded" else r"\papercodedfalse",
             r"\newif\ifpaperheld", r"\paperheldtrue" if PRESENTATION["held_benchmarks"] else r"\paperheldfalse",
             r"\newcommand{\num}[2][]{%",
             r"  \ifcsname paper@num@#1@#2\endcsname\csname paper@num@#1@#2\endcsname",
             r"  \else\PackageError{paper}{Undefined number [#1]{\detokenize{#2}}}{Regenerate the paper assets.}\fi}"]
    for (fmt, key), rendered in sorted(definitions.items()):
        lines.append(r"\expandafter\def\csname paper@num@" + fmt + "@" + key + r"\endcsname{" + rendered + "}")
    (out / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sources


def tabular(header, rows, columns, flexible=False):
    """Return a booktabs grid with pre-escaped cells."""
    env = "tabularx" if flexible else "tabular"
    width = r"{\linewidth}" if flexible else ""
    lines = [r"\begin{" + env + "}" + width + "{@{}" + columns + "@{}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    return "\n".join([*lines, r"\bottomrule", r"\end{" + env + "}"])


def table(body, caption, label, wide=True):
    """Return a labelled table float at its natural width."""
    env = "table*" if wide else "table"
    return "\n".join([r"\begin{" + env + "}[tp]", r"\centering\small", body,
                      r"\caption{" + caption + "}", r"\label{tab:" + label + "}", r"\end{" + env + "}", ""])


def read_benchmarks(paths, documents):
    """The corpus's list of benchmarks.

    A corpus whose labels were rebuilt lists no judged questions per benchmark (its manifest has the total); the
    audit, computed from the rebuilt labels, has them, and they are taken from there.
    """
    benchmarks = read_csv(paths["corpus/benchmarks.csv"])
    if benchmarks and "judged_fields" not in benchmarks[0] and "audit" in documents:
        audited = documents["audit"][1].get("benchmarks", {})
        if all(str(int(b["catalog_id"])) in audited for b in benchmarks):
            for b in benchmarks:
                entry = audited[str(int(b["catalog_id"]))]
                if (int(b["requests"]), int(b["fields"])) != (entry["requests"], entry["fields"]):
                    raise ValueError("the corpus and the audit differ in the size of a benchmark")
                b["judged_fields"] = str(entry["judged_fields"])
    return benchmarks


def corpus_table(benchmarks):
    """Return task-area counts computed from the benchmark catalogue."""
    fields = ["requests", "fields"]
    if benchmarks and "judged_fields" in benchmarks[0]:
        fields.append("judged_fields")
    rows = []
    for area in dict.fromkeys(row["area"] for row in benchmarks):
        members = [row for row in benchmarks if row["area"] == area]
        counts = [len(members), *(sum(int(row[k]) for row in members) for k in fields)]
        rows.append([tex_text(area), *(format_number("n", n) for n in counts)])
    header = ["Task area", "Benchmarks", *(k.replace("_", " ").capitalize() for k in fields)]
    return table(tabular(header, rows, "l" + "r" * (len(header) - 1)), "Corpus statistics by task area.", "corpus")


def corpus_display_name(row):
    """Return the resource label without a bibliographic author attribution."""
    name, reuse, _ = row["name"].partition(" as reused by ")
    if reuse:
        return name + " (as reused)"
    if " of " in name and " et al." in name:
        subtitle = re.search(r"\(([^()]+)\)\s*$", name)
        return subtitle[1] if subtitle else row["corpus_id"]
    return name


def comparison_tables(records, manifest, audit):
    """Return the size and stored-output comparisons, preserving source qualifications."""
    sizes, features = [], []
    for row in records:
        name = corpus_display_name(row)
        model, queries = row["models"], row["queries"]
        if name == "this_corpus":
            counts = manifest["counts"]
            name = "This corpus"
            model = format_number("n", manifest["model_count"]) + " models"
            model += " from " + format_number("n", audit["corpus"]["organisations"]) + " providers"
            queries = queries.replace("{{n_requests}}", format_number("n", counts["requests"]))
            queries = queries.replace("{{n_fields}}", format_number("n", counts["fields"]))
            queries += " across " + format_number("n", len(manifest["catalog_ids"])) + " benchmarks"
        sizes.append([tex_text(c) for c in (name, model, queries)])
        resource = tex_text(name)
        if row["name"] != "this_corpus" and row["corpus_id"]:
            identifier = re.sub(r"([/:.])", lambda m: m[0] + r"\allowbreak{}", tex_text(row["corpus_id"]))
            resource += r"\newline " + identifier
        features.append([resource, *(tex_text(row[k]) for k in ("probability_vectors", "token_counts", "cost"))])
    para = r">{\raggedright\arraybackslash}X"
    return (table(tabular(["Corpus", "Models", "Queries"], sizes, para * 3, True),
                  "Earlier evaluation corpora and the frozen corpus.", "comparison"),
            table(tabular(["Corpus", "Probability vectors", "Token counts", "Cost"], features, para * 4, True),
                  "Outputs and telemetry in the comparison corpora.", "comparison-features"))


def metric_grid(rows, methods):
    """Return a grid of available policies in settings order."""
    by_method = {row["method"]: row for row in rows}
    values = []
    for method in methods:
        if method in by_method:
            row = by_method[method]
            values.append([tex_text(method), *[
                format_number("f3", row[key]) if row.get(key) is not None else "" for key in METRICS]])
    return tabular(METRIC_HEADER, values, "lrrrrr")


def contrast_grid(rows, methods):
    """Return intervals in the recorded direction for comparisons involving an anchor."""
    intervals = {}
    for row in rows:
        if row["a"] not in methods or row["b"] not in methods:
            continue
        if not {row["a"], row["b"]} & ANCHORS or None in (row["ci_low"], row["ci_high"]):
            continue
        interval = "[" + ", ".join(format_number("s3", row[k]) for k in ("ci_low", "ci_high")) + "]"
        intervals.setdefault((row["contrast"], row["a"], row["b"]), {})[row["metric"]] = interval
    if not intervals:
        return ""
    grid = [[tex_text(a), tex_text(b),
             cells.get("log_loss", ""), cells.get("accuracy", "")] for (_, a, b), cells in intervals.items()]
    return "\n" + r"\par\smallskip" + "\n" + tabular(
        ("Method A", "Method B", "Log loss (A $-$ B)", "Accuracy (A $-$ B)"), grid, "llrr")


def diagnostic_table(summary, suite):
    """Return headline-suite diagnostics for each model."""
    keys = ("accuracy_reported", "accuracy_first_max", "log_loss_normalised_floor_1e-15",
            "brier", "ece", "mean_confidence")
    header = ("Model", "Reported acc.", "First-max acc.", "Log loss", "Brier", "Cal. error", "Confidence")
    rows = [[format_number("n", r["model_index"]), *[
        format_number("f3", r[k]) if r.get(k) is not None else "" for k in keys]]
        for r in sorted(summary, key=lambda r: r["model_index"])]
    return table(tabular(header, rows, "rrrrrrr"),
                 "Per-model diagnostics on " + tex_text(suite) + ". Log loss uses normalised vectors.", "diagnostics")


def write_tables(out, paths, data, documents):
    """Write the table fragments and return the source selectors for each fragment."""
    settings = data["SETTINGS.json"]
    rows, contrasts = data["results/rows.json"]["rows"], data["results/contrasts.json"]["rows"]
    suite, methods = settings["headline_suite"], settings["methods"]
    headline = [r for r in rows if r["suite"] == suite and r["catalog_id"] is None and r["method"] in methods]
    comparisons = [r for r in contrasts if r["suite"] == suite]
    main = metric_grid(headline, methods) + contrast_grid(comparisons, {r["method"] for r in headline})
    catalog = read_benchmarks(paths, documents)
    records = read_csv(paths["related_work/comparison_table.csv"])
    manifest = documents["corpus"][1]
    if any(r["name"] == "this_corpus" for r in records) and "audit" not in documents:
        raise ValueError("the comparison row requires --json audit=FILE for corpus.organisations")
    audit = documents["audit"][1] if "audit" in documents else {}
    sizes, features = comparison_tables(records, manifest, audit)
    diag_role = "diagnostics/diagnostics.json"
    diag_path = f"suites.{suite}.overall.models"
    diagnostics = data[diag_role]["suites"][suite]["overall"]["models"]
    from . import paper_tables as PT

    held = set(PRESENTATION["held_benchmarks"])
    coded = PRESENTATION["model_names"] == "coded"
    corpus_models = read_csv(paths["corpus/models.csv"])
    bench_rows = [r for r in rows if r["catalog_id"] is not None and r["catalog_id"] not in held]
    # A suite is shown when the held benchmarks are outside it and, unless it is the headline suite, when it differs
    # from the headline suite on this corpus (on the released corpus the suite of all benchmarks is the headline one).
    in_corpus = {int(b["catalog_id"]) for b in catalog}
    headline_members = in_corpus - set(settings["suites"][suite]["exclude"])
    shown_suites = {name: spec for name, spec in settings["suites"].items()
                    if held <= set(spec["exclude"]) and name in PT.SUITE_NAMES
                    and (name == suite or in_corpus - set(spec["exclude"]) != headline_members)}
    # False for the released corpus: it has no method with stored fits, and the marks of the contrasts table refer
    # to estimates on the first study's pool.  What had been seen of the released pool is said in the appendix.
    stored_fits = settings.get("replay_stored_fits", True)
    uncapped = None
    if "ops" in documents:          # the cascade without the cost cap, printed below its capped row
        uncapped = {"confidence_cascade": documents["ops"][1]["suites"][suite]["points"]["confidence_cascade"]["uncapped"]}
    shown_catalog = [b for b in catalog if int(b["catalog_id"]) not in held]
    files = {
        "corpus": corpus_table(catalog), "comparison": sizes, "comparison_features": features,
        "model_diagnostics": PT.models(diagnostics, corpus_models, coded),
        "main_results": PT.main_results(rows, suite, uncapped),
        "contrasts": PT.contrasts(contrasts, suite, documents["facts"][1] if "facts" in documents else None,
                                  earlier=stored_fits),
        "contrasts_accuracy": PT.contrasts_accuracy(contrasts, suite,
                                                    documents["facts"][1] if "facts" in documents else None),
        "methods_detail": PT.methods_detail(rows, suite),
        "per_benchmark": PT.per_benchmark(
            bench_rows, shown_catalog, "log_loss", 3, 1.0,
            "Log loss per benchmark (nats per judged field). Blank lines separate the five task areas.",
            "per-benchmark-loss") + PT.per_benchmark(
            bench_rows, shown_catalog, "accuracy", 1, 100.0,
            "Accuracy per benchmark (\\% of judged fields).", "per-benchmark-accuracy"),
        "sensitivity": PT.sensitivity(
            rows, shown_suites, "log_loss", 3, 1.0,
            "Log loss on every suite (number of benchmarks in parentheses).", "sensitivity-loss") + PT.sensitivity(
            rows, shown_suites, "accuracy", 1, 100.0,
            "Accuracy (\\%) on every suite.", "sensitivity-accuracy"),
    }
    refs = {
        "corpus": [("corpus/benchmarks.csv", "all rows grouped by area: catalog_id, requests, fields, judged_fields when present")],
        "comparison": [("related_work/comparison_table.csv", "all rows: name, models, queries"),
                       ("corpus/manifest.json", "model_count, catalog_ids, counts.requests, counts.fields")],
        "comparison_features": [("related_work/comparison_table.csv", "all rows: name, corpus_id, probability_vectors, token_counts, cost")],
        "model_diagnostics": [(diag_role, diag_path), ("SETTINGS.json", "headline_suite")],
        "main_results": [("results/rows.json", f"suite={suite}, catalog_id=null; settings methods"),
                         ("results/contrasts.json", f"suite={suite}; reported methods involving best_single or uniform_mean"),
                         ("SETTINGS.json", "headline_suite, methods")],
    }
    if any(r["name"] == "this_corpus" for r in records):
        refs["comparison"].append((documents["audit"][0], "corpus.organisations"))
    if "audit" in documents and "models" in audit and "cleared_models" in audit.get("rights", {}):
        release = documents["rel"][1] if "rel" in documents else None
        files["comparison_compact"] = PT.compact_comparison(records, audit, release)
        refs["comparison_compact"] = [("related_work/comparison_table.csv", "rows RouterBench, LLMRouterBench, EmbedLLM"),
                                      (documents["audit"][0], "corpus.models, corpus.observations, rights.cleared_models")]
        files["rights"] = PT.rights(audit, coded, release)
        refs["rights"] = [(documents["audit"][0], "models.*.route_id, listed_date, base_model; rights.model_status")]
        if release:
            hosts = release["secondary_hosts"]
            if not (release["secondary_host_models"] == 1
                    and all(len(v["records"]) - 1 == 2 for v in hosts.values())
                    and release["benchmark_clause_host_models"] == 1
                    and release["review_status_counts"]["released"].get("cleared", 0)
                    == audit["rights"]["model_status_counts"]["cleared"]):
                raise ValueError("the prose on the release (one model served by two other hosts) no longer holds")
            refs["comparison_compact"].append((documents["rel"][0], "models_released"))
            refs["rights"].append((documents["rel"][0], "model_status.*.release"))
    if "anchors" in documents:
        files["anchors"] = PT.anchors(documents["anchors"][1])
        refs["anchors"] = [(documents["anchors"][0], "anchors")]
    if "ops" in documents:
        reference = next(r for r in rows if r["suite"] == suite and r["catalog_id"] is None
                         and r["method"] == "best_single" and r["variant"] is None)
        files["operating_points"] = PT.operating_points(documents["ops"][1], suite, reference)
        refs["operating_points"] = [(documents["ops"][0], f"suites.{suite}: points, contrasts, selection"),
                                    ("results/rows.json", f"suite={suite}, method=best_single"),
                                    ("SETTINGS.json", "headline_suite")]
    if "open" in documents:
        files["open_track"] = PT.open_track(documents["open"][1], rows, suite)
        refs["open_track"] = [(documents["open"][0], f"suites.{suite}.methods, models, methods"),
                              ("results/rows.json", f"suite={suite}, catalog_id=null, variant=null: log_loss, accuracy"),
                              ("SETTINGS.json", "headline_suite")]
    if "folds" in documents:
        files["fold_redraws"] = PT.fold_redraws(documents["folds"][1], suite, stored_fits)
        refs["fold_redraws"] = [(documents["folds"][0], f"suites.{suite}.methods: log_loss, accuracy; draws"),
                                ("SETTINGS.json", "headline_suite")]
    restyle_tables(files, refs, paths, data, documents, settings, rows, catalog, corpus_models, audit, held, coded)
    refs["contrasts"] = [("results/contrasts.json", f"suite={suite}: every contrast"), ("SETTINGS.json", "headline_suite")]
    refs["contrasts_accuracy"] = [("results/contrasts.json", f"suite={suite}, metric=accuracy"),
                                  ("SETTINGS.json", "headline_suite")]
    if "facts" in documents:
        refs["contrasts"].append((documents["facts"][0], "over_benchmarks.*.log_loss: low, high"))
        refs["contrasts_accuracy"].append((documents["facts"][0], "over_benchmarks.*.accuracy: low, high"))
    refs["methods_detail"] = [("results/rows.json", f"suite={suite}, catalog_id=null, variant=null: log loss at "
                               "three floors, n_clipped, brier, ece, accuracy, native, latency percentiles"),
                              ("SETTINGS.json", "headline_suite")]
    refs["per_benchmark"] = [("results/rows.json", "catalog_id not null, variant null: log_loss, accuracy"),
                             ("corpus/benchmarks.csv", "catalog_id, name, area")]
    refs["sensitivity"] = [("results/rows.json", "catalog_id null, variant null: log_loss, accuracy by suite"),
                           ("SETTINGS.json", "suites")]
    refs["model_diagnostics"].append(("corpus/models.csv", "model_index, model_name"))
    refs["main_results"] = [("results/rows.json", f"suite={suite}, catalog_id=null, variant=null"),
                            ("SETTINGS.json", "headline_suite")]
    if uncapped:
        refs["main_results"].append((documents["ops"][0], f"suites.{suite}.points.confidence_cascade.uncapped"))
    release_prose(documents)
    folder = out / "tables"
    folder.mkdir()
    for name, content in files.items():
        (folder / (name + ".tex")).write_text(content, encoding="utf-8")
    return {f"tables/{name}.tex": sources for name, sources in refs.items()}


def release_prose(documents):
    """Stop when the appendix's sentences on the collection and the release no longer match the audit and the release.

    They say: the corpus is the release; the review's counts are those of the released models; one model was served
    partly by one other host; the requests that a host with a benchmark clause served for that model were removed
    for every model; the billed cost follows the listed price except for the requests of that model that the other
    host served; some models answered requests in smaller batches; every model has a vector for every question.
    """
    if "audit" not in documents or "rel" not in documents:
        return
    audit, release = documents["audit"][1], documents["rel"][1]
    telemetry, corpus = audit.get("telemetry", {}), audit.get("corpus", {})
    if "price_card_rows_beyond_rounding_on_main_host" not in telemetry:
        return                                                     # the audit of the whole collection: other prose
    removed = {cid: b["requests_removed"] for cid, b in audit["benchmarks"].items() if b.get("requests_removed")}
    statements = {
        "the corpus is the release": (corpus["models"], corpus["benchmarks"], corpus["requests"])
        == (release["models_released"], release["benchmarks_released"], release["requests_released"]),
        "the review's counts are those of the released models":
            audit["rights"]["model_status_counts"] == release["review_status_counts"]["released"],
        "one model had one other serving host, for some of its requests": release["secondary_host_models_in_corpus"] == 1
        and release["secondary_host_count_in_corpus"] == 1 and release["secondary_host_records_in_corpus"] > 0,
        "the requests that a host with a benchmark clause served for one model were removed for every model":
            release["excluded_request_rules"] == 1 and release["excluded_requests"] > 0
            and release["benchmark_clause_host_records_in_corpus"] == 0
            and release["benchmark_clause_host_records"] == release["excluded_requests"]
            and removed == release["excluded_requests_by_benchmark"],
        "one model is billed off the listed price, on the requests that the other host served":
            corpus["models"] - telemetry["price_card_models_exact"] == 1
            and telemetry["price_card_rows_beyond_rounding"] == release["secondary_host_records_in_corpus"]
            and telemetry["price_card_rows_beyond_rounding_on_main_host"] == 0,
        "some models answered requests in smaller batches": corpus["models_answering_in_smaller_batches"] >= 2
        and corpus["adapted_observations"] > 0,
        "every model has a vector for every question, after retries":
            corpus["vectors"] == corpus["fields"] * corpus["models"]
            and corpus["attempts"] > corpus["successful_physical_calls"],
    }
    failed = [name for name, holds in statements.items() if not holds]
    if failed:
        raise ValueError("the prose on the collection and the release no longer holds: " + "; ".join(failed))


def restyle_tables(files, refs, paths, data, documents, settings, rows, catalog, corpus_models, audit, held, coded):
    """Tables of the restyled paper, each only when its inputs are given."""
    from . import paper_tables as PT

    suite = settings["headline_suite"]
    labels = PT.model_labels(corpus_models, coded)
    shown = [b for b in catalog if int(b["catalog_id"]) not in held]
    files["benchmarks"] = PT.benchmark_list(catalog, held)
    files["settings"] = PT.settings_table(settings)
    refs["settings"] = [("SETTINGS.json", "knn_router, confidence_cascade, frontier, label_budget, bootstrap")]
    refs["benchmarks"] = [("corpus/benchmarks.csv", "catalog_id, name, area, native_metric, requests, fields, judged_fields")]
    rights = audit.get("rights", {})
    if "related_work/comparison_checks.csv" in paths and "audit" in documents and "corpus" in audit:
        cleared = len(rights.get("cleared_models", []))
        values = {"{{n_benchmarks}}": format_number("n", audit["corpus"]["benchmarks"]),
                  "{{n_models}}": format_number("n", audit["corpus"]["models"]),
                  "{{n_cleared}}": format_number("n", cleared),
                  "{{n_released}}": format_number("n", documents["rel"][1]["models_released"] if "rel" in documents
                                                  else cleared)}
        files["comparison_checks"] = PT.comparison_checks(read_csv(paths["related_work/comparison_checks.csv"]), values)
        refs["comparison_checks"] = [("related_work/comparison_checks.csv", "all rows"),
                                     (documents["audit"][0], "corpus.benchmarks, corpus.models")]
    if ("audit" in documents and "telemetry" in audit and "models" in audit and "structure" in audit.get("corpus", {})
            and "model_status" in rights):
        # The audit of the whole collection lists what the review cleared: the release is then a part of the corpus,
        # and the tables show both.  The audit of the released corpus does not: the corpus is the release.
        partial = all(k in rights for k in ("cleared_vectors", "cleared_models", "cleared_benchmarks"))
        release = documents["rel"][1] if partial and "rel" in documents else None
        files["statistics"] = PT.statistics(audit, None, release)
        refs["statistics"] = [(documents["audit"][0], "corpus, telemetry, rights")]
        files["model_pool"] = PT.model_pool(corpus_models, audit, labels, coded, release)
        refs["model_pool"] = [("corpus/models.csv", "model_index, model_code, model_name, organisation"),
                              (documents["audit"][0], "models.*: price_card_input_usd_per_token, latency_p50_s, "
                                                      "base_model; rights.model_status")]
        if release:
            refs["statistics"].append((documents["rel"][0], "benchmarks_released, models_released, vectors_released, "
                                                            "observations_released"))
            refs["model_pool"].append((documents["rel"][0], "model_status.*.release"))
    model_rows = read_csv(paths["diagnostics/models.csv"])
    exclude = set(settings["suites"][suite]["exclude"]) | held
    cids = {int(b["catalog_id"]) for b in catalog} - exclude
    files["top_models"] = PT.top_by_area(model_rows, cids, labels)
    refs["top_models"] = [("diagnostics/models.csv", f"catalog_id outside suites.{suite}.exclude: area, model_index, "
                                                     "accuracy_first_max"), ("SETTINGS.json", "suites")]
    bench_rows = [r for r in rows if r["catalog_id"] is not None and r["catalog_id"] not in held]
    tables = [("accuracy", "Accuracy (\\%) of every model on every benchmark, and of four methods below the "
               "rule. Shaded: the best and second-best model of each benchmark.", "model-accuracy")]
    files["model_benchmark"] = "\n".join(PT.model_benchmark(
        model_rows, bench_rows, shown, labels, metric, title, label, held,
        ("best_single", "stack_final_temperature", "knn_router", "confidence_cascade")) for metric, title, label in tables)
    refs["model_benchmark"] = [("diagnostics/models.csv", "accuracy_first_max by model_index and catalog_id"),
                               ("results/rows.json", "catalog_id not null, variant null: accuracy")]
    if "bm" in documents:
        role, bm = documents["bm"]
        files["benchmark_metrics"] = PT.metric_table(bm["suites"][suite])
        files["policies"] = PT.policy_table(bm["suites"][suite])
        refs["benchmark_metrics"] = [(role, f"suites.{suite}: methods, references")]
        refs["policies"] = [(role, f"suites.{suite}.policies")]
    if "rstudy" in documents:
        role, study = documents["rstudy"]
        reference = next(r for r in rows if r["suite"] == suite and r["catalog_id"] is None
                         and r["method"] == "best_single" and r["variant"] is None)
        extra = documents["fu"][1]["table"]["representations"] if "fu" in documents else None
        files["representations"] = PT.representations(study, suite, reference, extra)
        refs["representations"] = [(role, f"suites.{suite}: points, recall"),
                                   ("results/rows.json", f"suite={suite}, method=best_single")]
        if extra is not None:
            refs["representations"].append((documents["fu"][0], "table.representations"))
    if "fu" in documents:
        files["followup_policies"] = PT.followup_policies(documents["fu"][1]["table"])
        refs["followup_policies"] = [(documents["fu"][0], "table.rows")]


def write_tikz(out, paths, data, documents, catalog, suite, synthetic):
    """Figure 1 (the overview) and Figure 2 (the modules) as TikZ, drawn from the inputs by ``overview_figures``.

    The overview shows the corpus as a grid (a row per model, a column per benchmark) with three values per model,
    so it needs the audit, the corpus's lists of models and benchmarks and the per-model diagnostics; the modules
    strip prints three totals of the audit.  Neither is written when the audit does not hold the totals.
    """
    from . import overview_figures as OF

    if "audit" not in documents:
        return {}
    role, audit = documents["audit"]
    c, t = audit.get("corpus", {}), audit.get("telemetry", {})
    if "cost_usd" not in t or not all(k in c for k in ("models", "organisations", "benchmarks", "requests", "fields",
                                                       "vectors", "observations")):
        return {}
    folder = out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    (folder / "pipeline.tex").write_text(
        OF.modules(dict(models=c["models"], observations=c["observations"], cost_usd=t["cost_usd"])), encoding="utf-8")
    written["figures/pipeline.tex"] = [(role, "corpus.models, corpus.observations, telemetry.cost_usd")]

    models = sorted(read_csv(paths["corpus/models.csv"]), key=lambda r: int(r["model_index"]))
    order = [b for area, _ in OF.AREAS for b in sorted((b for b in catalog if b["area"] == area),
                                                       key=lambda b: int(b["catalog_id"]))]
    diag_file = "diagnostics/diagnostics.json"
    rows = data[diag_file]["suites"][suite]["overall"]["models"]
    values = {int(r["model_index"]): dict(accuracy=r["accuracy_first_max"], cost=r["usd_per_1k_requests"],
                                          latency=r["latency_p50_s"]) for r in rows}
    counts = (len(models), len(order), len({m["organisation"] for m in models}))
    agrees = (counts == (c["models"], c["benchmarks"], c["organisations"]) and len(order) == len(catalog)
              and sorted(values) == [int(m["model_index"]) for m in models])
    if not agrees:
        if synthetic:        # made-up inputs need not describe one corpus; the overview is then left out
            return written
        raise ValueError("the overview cannot be drawn: the audit, the corpus lists and the diagnostics disagree "
                         "on the models, the benchmarks or the organisations")
    # The audit of the whole collection lists what a review cleared: the release is then a part of the corpus.
    rights = audit.get("rights", {})
    partial = all(k in rights for k in ("cleared_vectors", "cleared_models", "cleared_benchmarks"))
    try:
        picture = OF.overview([(int(m["model_index"]), m["organisation"]) for m in models],
                              [(int(b["catalog_id"]), b["area"]) for b in order], values,
                              dict(requests=c["requests"], fields=c["fields"], vectors=c["vectors"],
                                   organisations=c["organisations"]), released=not partial)
    except ValueError:
        if synthetic:        # a made-up corpus may have areas too narrow for their names, or values off the strips
            return written
        raise
    (folder / "overview.tex").write_text(picture, encoding="utf-8")
    written["figures/overview.tex"] = [
        (role, "corpus.requests, corpus.fields, corpus.vectors, corpus.models, corpus.benchmarks, "
               "corpus.organisations"),
        ("corpus/models.csv", "model_index, organisation"), ("corpus/benchmarks.csv", "area by catalog_id"),
        (diag_file, f"suites.{suite}.overall.models: accuracy_first_max, usd_per_1k_requests, latency_p50_s "
                    "by model_index")]
    return written


def check_static_figures(paper, paths, documents, suite, synthetic):
    """Return {file name: (sha256, selectors compared)} for the hand-drawn figures of the paper folder.

    The drawings show the real corpus, so they are compared with the inputs unless those are made up.  The coded
    build does not show the drawn overview, so the source of that build may come without it.
    """
    from . import static_figures as SF

    folder = paper / SF.FOLDER
    names = [name for name in sorted(SF.LABELS) if (folder / name).is_file()]
    if not names:
        return {}
    compared = {name: [] for name in names}
    if not synthetic:
        coded = PRESENTATION["model_names"] == "coded"
        shown = [name for name in sorted(SF.LABELS) if not (coded and name in SF.CODED_BUILD_OMITS)]
        if any(key not in documents for key in ("audit", "rel", "diag")) or not set(shown) <= set(names):
            raise ValueError("the static figures need " + " and ".join(shown) + " and the audit, rel and diag documents")
        compared = SF.check(folder, documents["audit"][1], documents["rel"][1],
                            documents["diag"][1]["suites"][suite]["overall"]["models"],
                            read_csv(paths["corpus/models.csv"]), read_csv(paths["corpus/benchmarks.csv"]), names)
    return {name: (sha256_file(folder / name),
                   [(documents[key][0] if key in documents else key, selector) for key, selector in compared[name]])
            for name in names}


def input_paths(args):
    """Return logical input filenames and the bindings for named JSON documents."""
    paths = {"SETTINGS.json": getattr(args, "settings", None) or PACKAGE / "SETTINGS.json"}
    for role, folder, filenames in [
            ("corpus", args.corpus, ("manifest.json", "benchmarks.csv", "models.csv")),
            ("results", args.results, ("rows.json", "contrasts.json", "frontier.json", "ablations.json", "run.json")),
            ("diagnostics", args.diagnostics, ("models.csv", "reliability.csv", "pairs.csv", "headroom.csv", "diagnostics.json")),
            ("related_work", args.related_work, ("comparison_table.csv", "references.bib"))]:
        paths.update({role + "/" + name: folder / name for name in filenames})
    if (args.related_work / "comparison_checks.csv").is_file():
        paths["related_work/comparison_checks.csv"] = args.related_work / "comparison_checks.csv"
    for role, folder, filenames in []:
        paths.update({role + "/" + name: folder / name for name in filenames})
    for name in ("failures.csv", "homogenization.json", "homogenization_pairs.csv"):
        if (args.diagnostics / name).is_file():
            paths["diagnostics/" + name] = args.diagnostics / name
    bindings = {"corpus": "corpus/manifest.json"}
    for arg in args.json:
        name, sep, filename = arg.partition("=")
        if not sep or not filename or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("--json expects name=FILE")
        if name in bindings or name in ("rows", "contrasts"):
            raise ValueError("reserved or repeated JSON name: " + name)
        path = Path(filename)
        role = next((k for k, v in paths.items() if v.resolve() == path.resolve()), f"json/{name}/{path.name}")
        paths[role] = path
        bindings[name] = role
    return paths, bindings


def synthetic_input(value):
    """Return whether a JSON document declares synthetic data at any level."""
    if isinstance(value, dict):
        return value.get("synthetic") is True or any(synthetic_input(v) for v in value.values())
    return isinstance(value, list) and any(synthetic_input(v) for v in value)


def synthetic_sources(paths, data):
    """Return the JSON and CSV inputs that declare made-up data."""
    sources = [(role, "synthetic=true") for role in sorted(data) if synthetic_input(data[role])]
    for role, path in paths.items():
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8", newline="") as stream:
                rows = csv.DictReader(stream)
                if "synthetic" in (rows.fieldnames or []) and any((r["synthetic"] or "").strip().lower() == "true" for r in rows):
                    sources.append((role, "synthetic=true"))
    return sources


def generate(out, paper, paths, bindings):
    """Write the assets and their source hashes into an empty staging directory."""
    hashes = {role: sha256_file(path) for role, path in paths.items()}
    json_roles = set(bindings.values()) | {role for role in paths if role.endswith(".json")}
    data = {role: read_json(paths[role]) for role in json_roles}
    documents = {key: (role, data[role]) for key, role in bindings.items()}
    flags = synthetic_sources(paths, data)
    rows = data["results/rows.json"]["rows"]
    contrasts = data["results/contrasts.json"]["rows"]
    settings = data["SETTINGS.json"]
    numbers = write_numbers(out, paper, documents, rows, contrasts, settings["headline_suite"], bool(flags))
    tables = write_tables(out, paths, data, documents)
    figures = write_figures(out, paths, data, documents)
    captions = [r"\newcommand{\assetcaption}[1]{\csname paper@caption@#1\endcsname}"]
    figure_sources = {}
    tikz = write_tikz(out, paths, data, documents, read_benchmarks(paths, documents), settings["headline_suite"],
                      bool(flags))
    static = check_static_figures(paper, paths, documents, settings["headline_suite"], bool(flags))
    for stem, asset in sorted(figures.items()):
        text = tex_text(asset["caption"])
        for plain, symbol in CAPTION_SYMBOLS:
            text = text.replace(plain, symbol)
        captions.append(r"\expandafter\def\csname paper@caption@" + stem + r"\endcsname{" + text + "}")
        figure_sources[f"figures/{stem}.pdf"] = asset["sources"]
    (out / "figure_captions.tex").write_text("\n".join(captions) + "\n", encoding="utf-8")

    def provenance(selectors):
        """Return portable source selectors bound to the exact bytes used."""
        return [{"file": role, "selector": key, "sha256": hashes[role]} for role, key in selectors]

    for role, path in paths.items():
        if sha256_file(path) != hashes[role]:
            raise ValueError(role + " changed during export")
    manifest = {"inputs": hashes, "synthetic": bool(flags), "synthetic_sources": provenance(flags)}
    figure_sources.update(tikz)
    for group, assets in (("numbers", numbers), ("tables", tables), ("figures", figure_sources)):
        manifest[group] = {name: provenance(sources) for name, sources in assets.items()}
    manifest["static_figures"] = {name: {"sha256": digest, "checked_against": provenance(sources)}
                                  for name, (digest, sources) in static.items()}
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


CAPTION_SYMBOLS = (      # reference names in figure captions -> the symbols of the paper (after escaping)
    ("Gain@B\\_d", r"Gain@$\mathcal{B}_d$"), ("(B\\_d)", r"($\mathcal{B}_d$)"), (" B\\_d", r" $\mathcal{B}_d$"),
    ("Best Single B ", r"Best Single $\mathcal{B}$ "), ("Best Single B.", r"Best Single $\mathcal{B}$."),
    (" k ", " $k$ "),
)


def product_hashes(folder):
    """Return the relative filenames and digests in a generated tree."""
    return {path.relative_to(folder): sha256_file(path) for path in folder.rglob("*") if path.is_file()}


PRESENTATION = {"model_names": "real", "held_benchmarks": ()}     # how results are shown; set by ``export``


def organisation_codes(models):
    """{organisation: code}, numbered in the order of the model list (the rule of the open tier)."""
    codes = {}
    for row in sorted(models, key=lambda r: int(r["model_index"])):
        codes.setdefault(row["organisation"], f"O{len(codes) + 1:02d}")
    return codes


def export(args):
    """Generate a complete directory or return the names of stale products in check mode."""
    PRESENTATION["model_names"] = args.model_names
    PRESENTATION["held_benchmarks"] = tuple(sorted(args.hold))
    paths, bindings = input_paths(args)
    target = args.out
    paper = target.parent if (target.parent / "main.tex").is_file() else PACKAGE / "paper"
    if target.is_symlink() or any(p.resolve().is_relative_to(target.resolve()) for p in [paper / "main.tex", *paths.values()]):
        raise ValueError("output directory overlaps the inputs")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".paper-assets-") as scratch:
        stage = Path(scratch) / "generated"
        stage.mkdir()
        generate(stage, paper, paths, bindings)
        if args.check:
            before, after = product_hashes(target), product_hashes(stage)
            return sorted(str(k) for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        backup = Path(scratch) / "previous"
        if target.exists():
            if not target.is_dir():
                raise ValueError("output must be a directory")
            products = {p.relative_to(stage) for p in stage.rglob("*")}
            for existing in sorted(target.rglob("*")):
                relative = existing.relative_to(target)
                if existing.is_symlink() or relative not in products:
                    raise ValueError(f"output contains an unrelated entry: {relative}")
            target.rename(backup)
        try:
            stage.rename(target)
        except OSError:
            if backup.exists():
                backup.rename(target)
            raise
    return []


def main(argv=None):
    """Return the status of an export or a byte-for-byte freshness check."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("corpus", "results", "diagnostics", "related-work", "out"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--json", action="append", default=[], metavar="name=FILE")
    parser.add_argument("--settings", type=Path, help="settings file of the results (default: SETTINGS.json)")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model-names", choices=("real", "coded"), default="real",
                        help="show the models under their listed names or under the codes of the corpus")
    parser.add_argument("--hold", type=int, nargs="*", default=[], metavar="CATALOG_ID",
                        help="benchmarks whose results are left out of every table (and every suite that has them)")
    args = parser.parse_args(argv)
    try:
        stale = export(args)
    except (OSError, UnicodeError, KeyError, ValueError) as error:
        message = str(error)
        if isinstance(error, OSError) and error.filename:
            message = f"{Path(error.filename).name}: {error.strerror}"
        print("export: " + str(message), file=sys.stderr)
        return 1
    for name in stale:
        print("stale product: " + name, file=sys.stderr)
    return int(bool(stale))


if __name__ == "__main__":
    raise SystemExit(main())

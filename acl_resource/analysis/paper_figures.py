"""Main-text figures: similarity of the models' errors, operating points of the methods, label budgets.

Both take plain rows that the exporter has already read and return (figure, caption).  Colours: four categorical
hues that pass a colour-blind separation check, plus grey for context; every series also has its own marker.
"""

from __future__ import annotations

from collections import defaultdict

BLUE, ORANGE, PURPLE, GREEN, GREY, INK = "#0072B2", "#D55E00", "#8E5BB5", "#009E73", "#8a8a8a", "#222222"
FONT = 7.5

LINES = (            # frontier method, label, colour, marker, line style
    ("single_by_cost", "Per-benchmark control", BLUE, "o", "-"),
    ("knn_router_text", "kNN router (text)", ORANGE, "s", "-"),
    ("knn_router", "kNN router (features)", ORANGE, "s", "--"),
    ("confidence_cascade", "Confidence cascade", PURPLE, "^", "-"),
    ("greedy", "Greedy selection", GREEN, "D", "-"),
    ("ranked_top_k", "Top-k pool", GREY, "x", "-"),
)
POINTS = (           # method, label, marker, label offset in points for the cost and the latency panel
    ("best_single", "Best per benchmark", "o", (4, -11), (6, -10)),
    ("stack_final_temperature", "Stacking", "s", (5, 5), (5, 5)),
    ("certified_exit", "Early exit", "^", (-5, 7), (-5, 7)),
    ("uniform_mean", "Equal-weight pool", "X", (-7, -10), (-7, -10)),
)
Y_RANGE = (74.0, 90.5)


def _figure(width_cm, height_cm):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(width_cm / 2.54, height_cm / 2.54))
    FigureCanvasAgg(figure)
    return figure


def similarity(pairs, blocks, organisation):
    """Heat map of the mean error correlation of every model pair, models grouped by organisation.

    pairs: rows of pairs.csv for the suite; blocks: lists of model indexes, one per organisation;
    organisation: {model index: organisation label}.
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    order = [m for block in blocks for m in block]
    position = {m: i for i, m in enumerate(order)}
    values = defaultdict(list)
    for row in pairs:
        if row["error_correlation"]:
            values[int(row["model_i"]), int(row["model_j"])].append(float(row["error_correlation"]))
    n = len(order)
    matrix = np.full((n, n), np.nan)
    for (i, j), v in values.items():
        matrix[position[i], position[j]] = matrix[position[j], position[i]] = sum(v) / len(v)
    low = float(np.floor(np.nanmin(matrix) * 10) / 10)
    high = float(np.ceil(np.nanmax(matrix) * 10) / 10)
    width, height, side = 7.7, 4.9, 4.3                 # centimetres; the heat map is a square of ``side``
    fig = _figure(width, height)
    ax = fig.add_axes([0.235, 0.06, side / width, side / height])
    cmap = LinearSegmentedColormap.from_list("one_hue", ["#f1f6fb", "#9cc3e2", "#2f7ebc", "#0b3c6f"])
    cmap.set_bad("#ffffff")
    mesh = ax.pcolormesh(np.arange(n + 1), np.arange(n + 1), matrix, cmap=cmap, vmin=low, vmax=high,
                         edgecolors="none", rasterized=False)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    edge, centres, labels = 0, [], []
    for block in blocks:
        centres.append(edge + len(block) / 2)
        labels.append(organisation[block[0]])
        edge += len(block)
        if edge < n:
            ax.axhline(edge, color="white", linewidth=1.6)
            ax.axvline(edge, color="white", linewidth=1.6)
    ax.set_yticks(centres)
    ax.set_yticklabels(labels, fontsize=FONT)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar_axis = fig.add_axes([0.235 + side / width + 0.03, 0.16, 0.028, 0.68])
    ticks = [t / 10 for t in range(int(round(low * 10)), int(round(high * 10)) + 1, 2)]
    bar = fig.colorbar(mesh, cax=bar_axis, ticks=ticks)
    bar.solids.set_edgecolor("face")
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=FONT, length=2)
    bar.set_label("Error correlation", fontsize=FONT)
    inside = [matrix[position[a], position[b]] for block in blocks for a in block for b in block if a < b]
    caption = (f"Mean correlation between the errors of two models on the headline suite, for all pairs of the {n} "
               "models. Models "
               "are grouped by provider (white lines); the diagonal is left blank. "
               f"{len(inside)} pairs are of two models of one provider.")
    return fig, caption


def operating_points(frontier, rows, single_models):
    """Accuracy against cost and against latency for the settings of every method.

    frontier, rows: suite rows of frontier.json and rows.json; single_models: suite means per model with
    ``accuracy_first_max`` and ``usd_per_1k_requests``.
    """
    from matplotlib.ticker import FixedLocator, NullLocator

    fig = _figure(16.0, 7.0)
    left = fig.add_axes([0.075, 0.30, 0.43, 0.67])
    right = fig.add_axes([0.555, 0.30, 0.43, 0.67], sharey=left)
    by_method = defaultdict(list)
    for r in frontier:
        if r["accuracy"] is not None:
            by_method[r["method"]].append(r)
    suite = {r["method"]: r for r in rows if r["variant"] is None}
    handles = []
    if single_models:
        h = left.scatter([m["usd_per_1k_requests"] for m in single_models],
                         [100 * m["accuracy_first_max"] for m in single_models],
                         s=9, color="#c9c9c9", linewidths=0, zorder=1, label="Single models")
        handles.append(h)
    for method, label, colour, marker, style in LINES:
        points = by_method.get(method)
        if not points:
            continue
        for axis, key in ((left, "usd_per_1k_requests"), (right, "latency_mean_s")):
            xy = sorted((p[key], 100 * p["accuracy"]) for p in points if p[key] is not None)
            line, = axis.plot([x for x, _ in xy], [y for _, y in xy], color=colour, marker=marker, markersize=3.2,
                              linewidth=1.1, linestyle=style, markerfacecolor="white" if style == "--" else colour,
                              markeredgewidth=0.8, zorder=2, label=label)
        handles.append(line)
    for method, label, marker, offset_cost, offset_latency in POINTS:
        r = suite.get(method)
        if r is None:
            continue
        for axis, key, offset in ((left, "usd_per_1k_requests", offset_cost),
                                  (right, "latency_mean_s", offset_latency)):
            if r.get(key) is None:
                continue
            axis.scatter([r[key]], [100 * r["accuracy"]], s=34, marker=marker, color=INK, zorder=4,
                         edgecolors="white", linewidths=0.7)
            axis.annotate(label, (r[key], 100 * r["accuracy"]), xytext=offset, textcoords="offset points",
                          fontsize=FONT, color=INK, ha="left" if offset[0] > 0 else "right", zorder=5)
    left.set_xscale("log")
    ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4]
    left.xaxis.set_major_locator(FixedLocator(ticks))
    left.xaxis.set_minor_locator(NullLocator())
    left.set_xticklabels([f"{t:g}" for t in ticks])
    left.set_xlabel("Cost (USD per 1,000 requests, log scale)", fontsize=FONT)
    right.set_xlabel("Mean latency (seconds per request)", fontsize=FONT)
    left.set_ylabel("Accuracy (% of judged questions)", fontsize=FONT)
    for axis in (left, right):
        axis.tick_params(labelsize=FONT, length=2.5, width=0.6)
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    right.tick_params(labelleft=False)
    left.set_ylim(*Y_RANGE)
    below = sum(1 for m in single_models if 100 * m["accuracy_first_max"] < Y_RANGE[0])
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=FONT, handlelength=2.0,
               columnspacing=1.2, handletextpad=0.5, bbox_to_anchor=(0.5, 0.0))
    caption = ("Accuracy against cost (left) and against mean latency (right). Each line is one method "
               "over its settings; black symbols are single configurations; grey dots are the single models.")
    if below:
        caption += f" {below} single model{'s lie' if below > 1 else ' lies'} below the plotted range."
    return fig, caption


BUDGET_SERIES = (("best_single", "Best per benchmark", BLUE, "o"), ("stack", "Stacking", GREEN, "D"))


def label_budget(rows, draws):
    """Log loss and accuracy of two fitted policies against the number of labelled source groups.

    rows: suite rows of ablations.json (variants ``groups=N`` and the model chosen without target labels).
    """
    points = defaultdict(dict)
    for r in rows:
        if (r["variant"] or "").startswith("groups="):
            points[r["method"]][r["variant"].split("=")[1]] = r
    levels = sorted({k for by in points.values() for k in by}, key=lambda k: float("inf") if k == "all" else int(k))
    transfer = next((r for r in rows if r["method"] == "transfer_single" and r["variant"] is None), None)
    fig = _figure(7.7, 4.6)
    panels = ((fig.add_axes([0.135, 0.36, 0.345, 0.60]), "log_loss", 1.0, "Log loss (nats)"),
              (fig.add_axes([0.635, 0.36, 0.345, 0.60]), "accuracy", 100.0, "Accuracy (%)"))
    handles = []
    for axis, key, scale, label in panels:
        handles = []
        for method, name, colour, marker in BUDGET_SERIES:
            if method not in points:
                continue
            xy = [(i, scale * points[method][k][key]) for i, k in enumerate(levels) if k in points[method]]
            line, = axis.plot([x for x, _ in xy], [y for _, y in xy], color=colour, marker=marker, markersize=3.2,
                              linewidth=1.1, label=name)
            handles.append(line)
        if transfer is not None:
            handles.append(axis.axhline(scale * transfer[key], color=GREY, linewidth=1.0, linestyle="--",
                                        label="No target labels"))
        axis.set_xticks(range(len(levels)))
        axis.set_xticklabels(levels)
        axis.set_xlim(-0.3, len(levels) - 0.7)
        axis.set_xlabel("Labelled groups", fontsize=FONT)
        axis.set_ylabel(label, fontsize=FONT)
        axis.tick_params(labelsize=FONT, length=2.5, width=0.6)
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=FONT, handlelength=1.2,
               columnspacing=0.6, handletextpad=0.3, borderaxespad=0.2, bbox_to_anchor=(0.5, 0.0))
    caption = ("Log loss (left) and accuracy (right) on the headline suite of the best model per benchmark and the "
               f"stacked panel, fitted on a sample of labelled source groups of the target benchmark (mean of {draws} draws). Dashed: a "
               "model chosen on the other task areas with no target label. With all groups both use the four "
               "folds that are not the test fold.")
    return fig, caption


# --------------------------------------------------------------------------------------------------------------
# Figures of the restyled paper (added after the first run).  Colours follow the method in every figure; the order
# of the bars and lines below passes the colour-vision checks of the dataviz validator on adjacent pairs, and the
# three colours of the scatter plots (router, cascade, greedy) pass on all pairs.  Single configurations in scatter
# plots are black symbols with direct labels; references are grey or hatched.
# --------------------------------------------------------------------------------------------------------------

POOL, SKILL, STACK, EXIT, GREEDY, ROUTER, CASCADE = (
    "#e87ba4", "#6250d6", "#eda100", "#008300", "#1baf7a", "#2a78d6", "#eb6834")
SECONDARY, LIGHT = "#52514e", "#b7b6b0"
SMALL = 6.5

BAR_METHODS = (          # method, label, colour (the order passes the adjacent-pair checks)
    ("uniform_mean", "Equal-weight pool", POOL),
    ("stack_final_temperature", "Stacking", STACK),
    ("certified_exit", "Early exit", EXIT),
    ("greedy", "Greedy", GREEDY),
    ("knn_router", "kNN router", ROUTER),
    ("confidence_cascade", "Cascade", CASCADE),
)


def _clean(axis, grid="y"):
    axis.tick_params(labelsize=SMALL, length=2.0, width=0.6, pad=1.5)
    if grid:
        axis.grid(axis=grid, color="#e6e6e6", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def metric_bars(block, points=None, extra=None):
    """Three panels: log loss, average accuracy, and the accuracy difference to the best model per benchmark.

    block: one suite of benchmark_metrics.json.  points: {method: (difference, low, high)} in accuracy points, the
    paired difference to the best model per benchmark with its 95% interval (contrasts.json); without it the third
    panel is left out.  extra: further bars, each a dict(label, colour, log_loss, avg_acc, points) for a policy that
    is not a row of the block (the cascade without its cost cap).  The dashed line is the best model per benchmark.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    methods, refs = block["methods"], block["references"]
    bars = [dict(label=label, colour=colour, log_loss=methods[m]["log_loss"], avg_acc=methods[m]["avg_acc"],
                 points=(points or {}).get(m)) for m, label, colour in BAR_METHODS if m in methods]
    bars += list(extra or [])
    panels = [("Log loss", "↓", lambda e: e["log_loss"], "{:.3f}", refs["B"]["log_loss"]),
              ("Accuracy (%)", "↑", lambda e: 100 * e["avg_acc"], "{:.1f}", 100 * refs["B"]["accuracy"])]
    if points:
        panels.append(("Accuracy $-$ B$_d$ (points)", "↑", lambda e: e["points"][0] if e["points"] else 0.0,
                       "{:+.1f}", None))
    fig = _figure(16.0, 4.5)
    step = 0.96 / len(panels)
    for i, (title, arrow, value, fmt, line) in enumerate(panels):
        axis = fig.add_axes([0.05 + i * step, 0.20, step - 0.065, 0.67])
        xs = list(range(len(bars)))
        values = [value(e) for e in bars]
        axis.bar(xs, values, width=0.78, color=[e["colour"] for e in bars], edgecolor="white", linewidth=0.8, zorder=2)
        if line is None:
            for x, e in zip(xs, bars):
                if e["points"]:
                    axis.plot([x, x], [e["points"][1], e["points"][2]], color=INK, linewidth=0.7, zorder=3)
            axis.axhline(0, color=INK, linewidth=0.6, zorder=3)
        else:
            axis.axhline(line, color=INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=4)
        low, high = min(0.0, min(values)), max(0.0, max(values))
        pad = 0.34 * (high - low or 1.0)
        axis.set_ylim(low - (pad if low < 0 else 0), high + pad)
        for x, v in zip(xs, values):
            axis.annotate(fmt.format(v), (x, v), xytext=(0, 1.5 if v >= 0 else -1.5), textcoords="offset points",
                          ha="center", va="bottom" if v >= 0 else "top", fontsize=5.4, color=INK, zorder=5,
                          rotation=90, bbox={"boxstyle": "square,pad=0.05", "fc": "white", "ec": "none", "alpha": 0.85})
        axis.set_xticks([])
        axis.set_xlim(-0.6, len(bars) - 0.4)
        axis.set_title(f"{title} {arrow}", fontsize=FONT, fontweight="bold", pad=3)
        _clean(axis)
        axis.spines["bottom"].set_visible(False)
    handles = [Patch(facecolor=e["colour"], edgecolor="white", label=e["label"]) for e in bars]
    handles.append(Line2D([0], [0], color=INK, linewidth=0.8, linestyle=(0, (3, 2)), label="B$_d$"))
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=SMALL, handlelength=1.4,
               columnspacing=1.0, handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
    caption = ("Log loss, average accuracy" + (" and accuracy difference to the best model per benchmark (B_d)"
                                                if points else "")
               + " on the headline suite (benchmarks weighted equally). Dashed: B_d."
               + (" Whiskers: 95% intervals of the difference (resampling within benchmarks)." if points else "")
               + " Stacking: with a final temperature. The kNN router and the cascade are shown under the cost cap, "
               "at most the cost of B_d"
               + ("; the last bar is the cascade without the cap" if extra else "")
               + ". The pools and stacking are not capped. Early exit returns the stacked panel's answers, so its "
               "accuracy equals stacking's.")
    return fig, caption


HARD_LINES = (            # method, label, colour, marker (adjacent order passes the checks)
    ("uniform_mean", "Equal-weight pool", POOL, "x"),
    ("stack_final_temperature", "Stacking", STACK, "s"),
    ("knn_router", "kNN router", ROUTER, "^"),
    ("confidence_cascade", "Cascade", CASCADE, "D"),
)


def hardness(block):
    """Share of judged fields by the number of models right (top) and each method's accuracy there (bottom)."""
    bins = block["bins"]
    ks = [b["models_right"] for b in bins]
    fig = _figure(7.7, 5.8)
    top = fig.add_axes([0.13, 0.67, 0.85, 0.29])
    bottom = fig.add_axes([0.13, 0.27, 0.85, 0.35], sharex=top)
    shares = [100 * b["share"] for b in bins]
    top.bar(ks, shares, width=0.8, color=LIGHT, edgecolor="white", linewidth=0.6, zorder=2)
    for k, v in zip(ks, shares):
        if v >= 5:
            top.annotate(f"{v:.0f}", (k, v), xytext=(0, 1), textcoords="offset points", ha="center", va="bottom",
                         fontsize=5.2, color=INK)
    top.set_ylabel("Questions (%)", fontsize=SMALL)
    top.tick_params(labelbottom=False)
    _clean(top)
    handles = []
    line, = bottom.plot(ks, [100 * (b["accuracy"].get("best_single") or 0) for b in bins], color=INK, marker="o",
                        markersize=2.6, linewidth=1.0, label="Best per benchmark (B$_d$)", zorder=4)
    handles.append(line)
    for method, label, colour, marker in HARD_LINES:
        if method not in bins[0]["accuracy"]:
            continue
        ys = [100 * b["accuracy"][method] if b["accuracy"][method] is not None else float("nan") for b in bins]
        line, = bottom.plot(ks, ys, color=colour, marker=marker, markersize=2.6, linewidth=1.0, label=label,
                            markerfacecolor="white" if marker == "x" else colour, zorder=3)
        handles.append(line)
    bottom.set_ylabel("Accuracy (%)", fontsize=SMALL)
    bottom.set_xlabel("Number of models (of " + str(len(ks) - 1) + ") right on the question", fontsize=SMALL)
    bottom.set_xticks([k for k in ks if k % 2 == 0])
    bottom.set_xlim(-0.7, ks[-1] + 0.7)
    bottom.set_ylim(-3, 103)
    _clean(bottom)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=SMALL, handlelength=1.6,
               columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.54, -0.01))
    caption = ("Top: judged questions by how many of the models are right on them. Bottom: accuracy of the shown "
               "methods on those questions. Benchmarks weighted equally.")
    return fig, caption


def subsets(block):
    """Oracle accuracy of k-model subsets: the k models ranked best on the fit folds and random subsets."""
    s = block["subsets"]
    ks = s["sizes"]
    fig = _figure(7.7, 5.0)
    axis = fig.add_axes([0.13, 0.30, 0.84, 0.66])

    def draw(ax, size):
        ax.fill_between(ks, [100 * v for v in s["random_low"]], [100 * v for v in s["random_high"]],
                        color="#d8d7d2", linewidth=0, zorder=1)
        r, = ax.plot(ks, [100 * v for v in s["random_mean"]], color=SECONDARY, marker="s", markersize=size,
                     linewidth=1.0, markerfacecolor="white", label="Random $k$ (5–95%)", zorder=3)
        t, = ax.plot(ks, [100 * v for v in s["top_k"]], color=INK, marker="o", markersize=size, linewidth=1.0,
                     label="Top $k$ (ranked on fitting folds)", zorder=4)
        return t, r
    top, random = draw(axis, 2.6)
    axis.set_xlabel("Number of models in the subset", fontsize=SMALL)
    axis.set_ylabel("Oracle accuracy (%)", fontsize=SMALL)
    axis.set_xticks([k for k in ks if k in (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)])
    _clean(axis)
    inset = axis.inset_axes([0.42, 0.10, 0.55, 0.50])
    draw(inset, 1.8)
    zoom = [k for k in ks if k >= 8]
    values = ([100 * v for k, v in zip(ks, s["top_k"]) if k >= 8]
              + [100 * v for k, v in zip(ks, s["random_low"]) if k >= 8]
              + [100 * v for k, v in zip(ks, s["random_high"]) if k >= 8])
    inset.set_xlim(zoom[0] - 0.5, zoom[-1] + 0.5)
    inset.set_ylim(min(values) - 0.1, max(values) + 0.1)
    inset.set_xticks([k for k in zoom if k % 2 == 0])
    inset.tick_params(labelsize=5.2, length=1.5, pad=1)
    inset.grid(color="#e6e6e6", linewidth=0.5)
    inset.set_facecolor("white")
    axis.indicate_inset_zoom(inset, edgecolor="#9a9a9a", linewidth=0.6)
    fig.legend(handles=[top, random], loc="lower center", ncol=2, frameon=False, fontsize=SMALL, handlelength=1.6,
               columnspacing=1.0, handletextpad=0.4, bbox_to_anchor=(0.55, 0.0))
    caption = ("Accuracy of an oracle that picks, for each request, the best of k models: the k models ranked best "
               "on the fitting folds, and random subsets of k models (the inset enlarges k of 8 and more). Benchmarks "
               "weighted equally.")
    return fig, caption


def bubbles(block, labels=None):
    """Accuracy against cost of every model, with the bubble area proportional to its median latency.

    labels: {model index: short name}, or None to leave the bubbles unlabelled (the coded build).
    """
    from matplotlib.ticker import FixedLocator, NullLocator

    single = block["single_models"]
    fig = _figure(7.7, 5.8)
    axis = fig.add_axes([0.13, 0.16, 0.84, 0.81])
    xs = [s["usd_per_1k_requests"] for s in single]
    ys = [100 * s["accuracy"] for s in single]
    lat = [s["latency_p50_s"] for s in single]
    scale = 150.0
    axis.scatter(xs, ys, s=[scale * v for v in lat], color=ROUTER, alpha=0.28, edgecolors=ROUTER, linewidths=0.6,
                 zorder=3)
    axis.set_xscale("log")
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2) if min(xs) / 1.5 <= t <= max(xs) * 1.5]
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xticklabels([f"{t:g}" for t in ticks])
    axis.set_xlabel("Cost (USD per 1,000 requests, log scale)", fontsize=SMALL)
    axis.set_ylabel("Accuracy (%)", fontsize=SMALL)
    axis.margins(x=0.12, y=0.12)
    _clean(axis, grid="both")
    for size in (0.3, 0.5, 0.8):
        axis.scatter([], [], s=scale * size, color=ROUTER, alpha=0.28, edgecolors=ROUTER, linewidths=0.6,
                     label=f"{size:g} s")
    axis.legend(title="Median latency", title_fontsize=SMALL, fontsize=SMALL, frameon=False, loc="lower right",
                labelspacing=1.1, borderpad=0.6, handletextpad=0.9)
    if labels:
        fig.canvas.draw()
        _place_labels(axis, [(x, y, labels[s["model_index"]]) for x, y, s in zip(xs, ys, single)])
    caption = ("Accuracy against cost of each of the models (benchmarks weighted equally); the area of a bubble is "
               "proportional to the model's median latency per request, averaged over benchmarks.")
    return fig, caption


def _place_labels(axis, points, fontsize=4.7):
    """Short labels beside points, avoiding each other and the points."""
    import math as _m
    from matplotlib.transforms import Bbox

    scale = axis.figure.dpi / 72
    bounds = axis.get_window_extent()
    locations = [axis.transData.transform((x, y)) for x, y, _ in points]
    obstacles = [Bbox.from_bounds(px - 2 * scale, py - 2 * scale, 4 * scale, 4 * scale) for px, py in locations]
    for (x, y, label), (px, py) in zip(points, locations):
        width, height = (len(label) * fontsize * 0.55 + 1) * scale, (fontsize + 2) * scale
        best = None
        for distance in (5, 7, 9, 12, 15, 19, 24, 30, 37):
            for step in range(24):
                angle = _m.tau * step / 24
                dx, dy = distance * _m.cos(angle), distance * _m.sin(angle)
                box = Bbox.from_bounds(px + dx * scale - width / 2, py + dy * scale - height / 2, width, height)
                if not (bounds.contains(*box.min) and bounds.contains(*box.max)):
                    continue
                overlap = sum(Bbox.intersection(box, o).width * Bbox.intersection(box, o).height
                              for o in obstacles if box.overlaps(o))
                score = overlap / scale ** 2 + distance / 50
                if best is None or score < best[0]:
                    best = score, box, dx, dy
            if best is not None and best[0] < 1:
                break
        if best is None:
            continue
        _, box, dx, dy = best
        obstacles.append(box)
        axis.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha="center", va="center",
                      fontsize=fontsize, color=INK, zorder=6,
                      arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#9a9a9a", "shrinkA": 0, "shrinkB": 1.5}
                      if _m.hypot(dx, dy) > 10 else None)


GAIN_ROWS = (             # policy, label
    ("knn_router", "kNN router"),
    ("knn_router_text", "kNN router (text)"),
    ("single_by_cost", "Per-benchmark control"),
    ("confidence_cascade", "Confidence cascade"),
    ("greedy", "Greedy selection"),
    ("ranked_top_k", "Top-k pool"),
    ("stack_final_temperature", "Stacking"),
    ("certified_exit", "Early exit"),
)
PERF, SAVE = "#e34948", SECONDARY


def gains(block):
    """PerfGain and CostSave of each policy against Best Single B and against the best model per benchmark B_d.

    Four panels, each on its own scale; a CostSave below -100% is drawn as a marker at the end of the axis.
    """
    policies = block["policies"]
    rows = [(p, label) for p, label in GAIN_ROWS if p in policies]
    ys = list(range(len(rows)))[::-1]
    fig = _figure(16.0, 4.6)
    panels = (("T", "perf_gain", "PerfGain against B (%)", PERF), ("T", "cost_save", "CostSave against B (%)", SAVE),
              ("B", "perf_gain", "PerfGain against B$_d$ (%)", PERF), ("B", "cost_save", "CostSave against B$_d$ (%)", SAVE))
    perf_values = [100 * policies[p][f"perf_gain_{ref}"] for ref in ("T", "B") for p, _ in rows]
    perf_lim = max(abs(v) for v in perf_values) * 1.45 + 0.3
    for i, (ref, key, title, colour) in enumerate(panels):
        axis = fig.add_axes([0.165 + i * 0.209, 0.12, 0.168, 0.76])
        for y in ys[::2]:
            axis.axhspan(y - 0.5, y + 0.5, color="#f3f3f1", zorder=0, linewidth=0)
        for y, (p, _) in zip(ys, rows):
            value = policies[p][f"{key}_{ref}"]
            if value is None:
                axis.annotate("N/A", (0, y), xytext=(3, 0), textcoords="offset points", ha="left", va="center",
                              fontsize=5.4, color=SECONDARY)
                continue
            v = 100 * value
            if key == "cost_save" and v < -100:
                axis.plot([-100], [y], marker="<", markersize=4.5, color=colour, zorder=3)
                axis.annotate(f"{v:+.0f}", (-100, y), xytext=(5, 0), textcoords="offset points", ha="left",
                              va="center", fontsize=5.4, color=INK)
                continue
            axis.barh(y, v, height=0.62, color=colour, edgecolor="white", linewidth=0.5, zorder=2)
            axis.annotate(f"{v:+.1f}", (v, y), xytext=(2 if v >= 0 else -2, 0), textcoords="offset points",
                          ha="left" if v >= 0 else "right", va="center", fontsize=5.4, color=INK)
        axis.axvline(0, color=INK, linewidth=0.6, zorder=3)
        axis.set_xlim(*((-perf_lim, perf_lim) if key == "perf_gain" else (-110, 125)))
        axis.set_ylim(-0.6, len(rows) - 0.4)
        axis.set_yticks(ys)
        axis.set_yticklabels([label for _, label in rows] if i == 0 else [""] * len(rows), fontsize=SMALL)
        axis.set_title(title, fontsize=SMALL, fontweight="bold", pad=3)
        _clean(axis, grid="x")
        axis.tick_params(axis="y", length=0)
    caption = ("Best achievable accuracy gain (PerfGain) and largest cost saving at no loss of accuracy (CostSave) of "
               "each policy over its settings, against Best Single B (one model for all benchmarks, chosen on the test "
               "folds, as in LLMRouterBench) and against the best model per benchmark B_d. Settings are chosen on the "
               "test folds. N/A: no setting is as accurate as the reference; arrows: the cheapest such setting costs more than twice the reference.")
    return fig, caption


PARETO_LINES = (          # policy, label, colour, marker, line style
    ("knn_router", "Router", ROUTER, "o", "-"),
    ("knn_router_text", "Router, text", ROUTER, "s", "--"),
    ("single_by_cost", "Control", SECONDARY, "v", ":"),
    ("confidence_cascade", "Cascade", CASCADE, "^", "-"),
    ("greedy", "Greedy", GREEDY, "D", "-"),
)
PARETO_POINTS = (         # method or reference, label, marker, label offset in points, alignment
    ("best_single", "B$_d$", "o", (2, -9), "left"),
    ("stack_final_temperature", "Stacking", "s", (3, 4), "left"),
    ("certified_exit", "Early exit", "^", (-2, 6), "right"),
)


def pareto(block, frontier_rows):
    """Accuracy against cost of every setting, with the empirical Pareto frontier and ParetoDist in the legend."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator

    policies, methods, refs = block["policies"], block["methods"], block["references"]
    fig = _figure(7.7, 6.0)
    axis = fig.add_axes([0.16, 0.13, 0.81, 0.79])
    single = block["single_models"]
    axis.scatter([s["usd_per_1k_requests"] for s in single], [100 * s["accuracy"] for s in single], s=8,
                 color="#c9c9c9", linewidths=0, zorder=1)
    handles = [Line2D([0], [0], marker="o", color="#c9c9c9", linewidth=0, markersize=3.5, label="Single models")]
    front = block["frontier"]
    fx, fy = [p[0] for p in front], [100 * p[1] for p in front]
    axis.step(fx, fy, where="post", color="#8fb8e8", linewidth=2.2, alpha=0.8, zorder=0)
    handles.append(Line2D([0], [0], color="#8fb8e8", linewidth=2.2, label="Pareto frontier"))
    for policy, label, colour, marker, style in PARETO_LINES:
        points = sorted((r["usd_per_1k_requests"], 100 * r["accuracy"]) for r in frontier_rows
                        if r["method"] == policy)
        if not points or policy not in policies:
            continue
        line, = axis.plot([x for x, _ in points], [y for _, y in points], color=colour, marker=marker,
                          markersize=2.8, linewidth=0.9, linestyle=style, zorder=3,
                          markerfacecolor="white" if style != "-" else colour, markeredgewidth=0.7,
                          label=f"{label} ({policies[policy]['pareto_dist']:.3f})")
        handles.append(line)
    for method, label, marker, offset, align in PARETO_POINTS:
        if method not in methods:
            continue
        e = methods[method]
        axis.scatter([e["usd_per_1k_requests"]], [100 * e["avg_acc"]], s=26, marker=marker, color=INK, zorder=5,
                     edgecolors="white", linewidths=0.6)
        axis.annotate(label, (e["usd_per_1k_requests"], 100 * e["avg_acc"]), xytext=offset, textcoords="offset points",
                      ha=align, fontsize=5.4, color=INK, zorder=6)
    t = refs["T"]
    axis.scatter([t["usd_per_1k_requests"]], [100 * t["accuracy"]], s=26, marker="P", color=INK, zorder=5,
                 edgecolors="white", linewidths=0.6)
    axis.annotate("Best Single (B)", (t["usd_per_1k_requests"], 100 * t["accuracy"]), xytext=(-4, -9),
                  textcoords="offset points", ha="right", fontsize=5.4, color=INK, zorder=6)
    axis.set_xscale("log")
    xs = [s["usd_per_1k_requests"] for s in single] + fx
    ticks = [v for v in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4) if min(xs) / 1.5 <= v <= max(xs) * 1.5]
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xticklabels([f"{v:g}" for v in ticks])
    axis.set_xlabel("Cost (USD per 1,000 requests, log scale)", fontsize=SMALL)
    axis.set_ylabel("Average accuracy (%)", fontsize=SMALL)
    low = min(100 * s["accuracy"] for s in single)
    floor = 64.0
    axis.set_ylim(max(low - 1.5, floor), max(fy + [100 * methods[m]["avg_acc"] for m, *_ in PARETO_POINTS
                                                   if m in methods]) + 1.4)
    below = sum(1 for x in single if 100 * x["accuracy"] < max(low - 1.5, floor))
    axis.set_xlim(min(xs) / 1.25, max(xs) * 2.4)
    _clean(axis, grid="both")
    axis.annotate("", xy=(0.02, 1.035), xytext=(0.42, 1.035), xycoords="axes fraction",
                  arrowprops={"arrowstyle": "-|>", "color": SECONDARY, "lw": 1.0})
    axis.text(0.22, 1.05, "cheaper", transform=axis.transAxes, ha="center", va="bottom", fontsize=SMALL,
              color=SECONDARY)
    axis.annotate("", xy=(-0.165, 0.98), xytext=(-0.165, 0.62), xycoords="axes fraction",
                  arrowprops={"arrowstyle": "-|>", "color": SECONDARY, "lw": 1.0}, annotation_clip=False)
    axis.text(-0.19, 0.80, "better", transform=axis.transAxes, rotation=90, ha="center", va="center",
              fontsize=SMALL, color=SECONDARY)
    from matplotlib.ticker import MultipleLocator
    axis.yaxis.set_major_locator(MultipleLocator(5))
    axis.legend(handles=handles, loc="lower right", ncol=2, frameon=True, framealpha=0.92, edgecolor="none",
                fontsize=5.6, handlelength=1.5, columnspacing=0.8, handletextpad=0.35, labelspacing=0.25,
                borderpad=0.3)
    caption = ("Average accuracy against cost for the single models and every setting of the adaptive policies, "
               "with the empirical Pareto frontier (Control: the per-benchmark control). Black symbols: B_d, stacking, "
               "early exit and Best Single B. In parentheses: "
               "ParetoDist, the mean distance of a policy's settings to the frontier on normalised axes (cost on a "
               "log scale).")
    if below:
        caption += f" {below} single model{'s lie' if below > 1 else ' lies'} below the plotted range."
    return fig, caption


DONUT_ROWS = (("router_numeric.capped", "kNN router"), ("router_numeric.uncapped", "kNN router, no cap"),
              ("cascade.capped", "Cascade (answering model)"))


def donuts(routing, benchmarks, labels, accuracy=None):
    """Share of requests per model for a few benchmarks: router under both rules and the cascade's answering model.

    routing: router_study.json "routing"; benchmarks: [(catalog id, name)]; labels: {model index: label}.
    """
    from matplotlib.patches import Patch

    palette = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
    totals = defaultdict(float)
    for cid, _ in benchmarks:
        for key, _ in DONUT_ROWS:
            for m, v in enumerate(routing[str(cid)][key]):
                totals[m] += v
    named = [m for m, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:len(palette)]]
    colour = {m: palette[i] for i, m in enumerate(sorted(named))}
    cols = len(DONUT_ROWS)
    rows = len(benchmarks)
    fig = _figure(16.0, 1.0 + 3.0 * ((rows + 1) // 2))
    for b, (cid, name) in enumerate(benchmarks):
        for j, (key, title) in enumerate(DONUT_ROWS):
            col = (b % 2) * cols + j
            row = b // 2
            axis = fig.add_axes([0.02 + col * 0.162, 1 - (row + 1) * (3.0 / (1.0 + 3.0 * ((rows + 1) // 2))) + 0.02,
                                 0.14, 2.3 / (1.0 + 3.0 * ((rows + 1) // 2))])
            shares = routing[str(cid)][key]
            parts, colours, texts = [], [], []
            other = 0.0
            for m, v in enumerate(shares):
                if v <= 0:
                    continue
                if m in colour and v >= 0.05:
                    parts.append(v)
                    colours.append(colour[m])
                    texts.append(f"{100 * v:.0f}%" if v >= 0.12 else "")
                else:
                    other += v
            if other > 0:
                parts.append(other)
                colours.append("#cfcfcf")
                texts.append(f"{100 * other:.0f}%" if other >= 0.12 else "")
            wedges, inner = axis.pie(parts, colors=colours, labels=None, startangle=90, counterclock=False,
                                     wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 0.8},
                                     autopct=None)
            for wedge, text in zip(wedges, texts):
                if text:
                    import math as _m
                    angle = _m.radians((wedge.theta1 + wedge.theta2) / 2)
                    axis.text(0.78 * _m.cos(angle), 0.78 * _m.sin(angle), text, ha="center", va="center",
                              fontsize=4.8, color=INK)
            axis.set_title((name + "\n" if j == 1 else "\n") + title, fontsize=5.4, pad=1)
            axis.set_aspect("equal")
    handles = [Patch(facecolor=colour[m], label=labels[m]) for m in sorted(named)]
    handles.append(Patch(facecolor="#cfcfcf", label="Others (or < 5%)"))
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=SMALL, handlelength=1.0,
               columnspacing=0.9, bbox_to_anchor=(0.5, 0.0))
    caption = ("Share of requests sent to each model by the kNN router under the cost cap and without it, and the "
               "model whose answer the confidence cascade returns. Models below 5% are grouped as Others.")
    return fig, caption

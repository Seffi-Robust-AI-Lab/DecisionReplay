"""Check the hand-drawn figures of the paper against the data they show.

``paper/static_figures/overview.pdf`` (the overview, Figure 1) and ``pipeline.pdf`` (the modules, Figure 2) are drawn
outside this code.  Both print corpus totals; the overview also shows the open subset as a grid of cells (one row per
model, one column per benchmark) and three values per model as dots.  ``check`` compares all of that with the audit,
the release summary, the per-model diagnostics and the corpus tables, so a drawing that no longer agrees with the data
stops the export.  Text is read with ``pdftotext`` and shapes with ``pdftocairo`` (both from poppler).
"""

import math
import re
import subprocess
from collections import Counter
from pathlib import Path

FOLDER = "static_figures"
CODED_BUILD_OMITS = ("overview.pdf",)       # it shows cost and latency per model, which the coded build leaves out
AREAS = ("knowledge", "language", "retrieval", "tools", "arts")       # the column blocks of the grid, left to right
AXES = (("accuracy_first_max", (60.0, 70.0, 80.0, 90.0), lambda v: 100 * v, 0.1),      # per cent
        ("usd_per_1k_requests", (-2.0, -1.0, 0.0), math.log10, 0.015),                  # log10 of USD
        ("latency_p50_s", (0.0, 0.5, 1.0), float, 0.01))                                # seconds
LABELS = {      # digits printed as axis ticks, units and step numbers, which no input determines
    "overview.pdf": ("60", "90", "$0.01", "$1", "0", "1", "1"),          # 60, 90%; $0.01, $1; 0, 1 s; "per 1k req."
    "pipeline.pdf": ("1", "2", "3", "4", "5", "1", "2"),                 # five modules; stage 1, stage 2
}
NUMBER = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"


def poppler(tool, *args):
    """Return the standard output of a poppler tool."""
    proc = subprocess.run([tool, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                          errors="replace")
    if proc.returncode:
        raise ValueError(f"{tool} could not read a static figure")
    return proc.stdout


def printed_numbers(pdf):
    """Return every number printed in a PDF, as text and sorted."""
    return sorted(re.findall(r"\$?\d+(?:[.,]\d+)*", poppler("pdftotext", str(pdf), "-")))


def expected_numbers(name, audit, release):
    """Return the numbers a drawing has to print, and the selectors they come from."""
    c = audit["corpus"]
    if name == "overview.pdf":
        values = [c["requests"], c["fields"], c["benchmarks"], c["models"], c["organisations"], c["vectors"],
                  release["models_released"], release["benchmarks_released"], release["models_held"],
                  release["benchmarks_total"] - release["benchmarks_released"]]
        text = [f"{int(v):,}" for v in values]
        sources = [("audit", "corpus.requests, corpus.fields, corpus.benchmarks, corpus.models, corpus.organisations, "
                             "corpus.vectors"),
                   ("rel", "models_released, benchmarks_released, models_held, benchmarks_total")]
    else:
        text = [f"${audit['telemetry']['cost_usd']:.2f}", f"{int(c['attempts']):,}", f"{int(c['models']):,}"]
        sources = [("audit", "telemetry.cost_usd, corpus.attempts, corpus.models")]
    return sorted(text + list(LABELS[name])), sources


def shapes(pdf):
    """Return the drawn paths of a PDF (not its text): fill, stroke, bounding box and the boxes of its subpaths."""
    svg = poppler("pdftocairo", "-svg", str(pdf), "-")
    out = []
    for tag in re.finditer(r"<path\b([^>]*)/>", svg[svg.find("</defs>"):]):
        attributes = dict(re.findall(r'([\w:-]+)="([^"]*)"', tag[1]))
        d = attributes.get("d", "")
        t = [float(v) for v in re.findall(NUMBER, attributes.get("transform", ""))]
        t = t if len(t) == 6 else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        boxes = []
        for sub in re.split(r"(?=M)", d):
            values = [float(v) for v in re.findall(NUMBER, re.sub(r"[A-Za-z]", " ", sub))]
            xs = [t[0] * x + t[2] * y + t[4] for x, y in zip(values[0::2], values[1::2])]
            ys = [t[1] * x + t[3] * y + t[5] for x, y in zip(values[0::2], values[1::2])]
            if xs and (max(xs) > min(xs) or max(ys) > min(ys)):
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
        if boxes:
            box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                   max(b[3] for b in boxes))
            out.append({"fill": attributes.get("fill", "none"), "stroke": attributes.get("stroke", "none"),
                        "box": box, "boxes": boxes, "curved": "C" in d})
    return out


def cluster(values, tolerance=0.6):
    """Return the sorted centres of groups of nearly equal coordinates."""
    groups = []
    for value in sorted(values):
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(g) / len(g) for g in groups]


def nearest(centres, value):
    """Return the index of the centre nearest to a coordinate."""
    return min(range(len(centres)), key=lambda i: abs(centres[i] - value))


def blocks(centres):
    """Return the sizes of the blocks of evenly spaced coordinates that wider gaps separate."""
    gaps = [b - a for a, b in zip(centres, centres[1:])]
    sizes = [1]
    for gap in gaps:
        if gap > 1.25 * min(gaps):
            sizes.append(1)
        else:
            sizes[-1] += 1
    return sizes


def luminance(colour):
    """Return the mean of the channels of an ``rgb(r%, g%, b%)`` colour."""
    return sum(float(v) for v in re.findall(r"[\d.]+", colour)) / 3


def check_overview(pdf, release, diagnostics, corpus_models, catalog):
    """Raise ValueError unless the grid and the dots of the overview show the release and the per-model values."""
    drawn = shapes(pdf)
    grids = [s for s in drawn if len(s["boxes"]) >= 20 and s["fill"] != "none"]
    if not 1 <= len({s["fill"] for s in grids}) <= 2:
        raise ValueError("overview.pdf: expected the grid of cells in one or two colours")
    dark = min((s["fill"] for s in grids), key=luminance)
    cells = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, s["fill"] == dark) for s in grids for b in s["boxes"]]
    columns, rows = cluster(c[0] for c in cells), cluster(c[1] for c in cells)
    models = sorted(corpus_models, key=lambda m: int(m["model_index"]))
    order = [b for area in AREAS for b in sorted((b for b in catalog if b["area"] == area),
                                                 key=lambda b: int(b["catalog_id"]))]
    if len(order) != len(catalog):
        raise ValueError("overview.pdf: a benchmark lies outside the five task areas of the grid")
    if (len(rows), len(columns)) != (len(models), len(order)) or len(cells) != len(models) * len(order):
        raise ValueError(f"overview.pdf: the grid has {len(rows)} rows and {len(columns)} columns of {len(cells)} "
                         f"cells, the corpus {len(models)} models and {len(order)} benchmarks")
    organisations = [n for _, n in sorted(Counter(m["organisation"] for m in models).items(),
                                          key=lambda item: min(int(m["model_index"]) for m in models
                                                               if m["organisation"] == item[0]))]
    areas = [sum(b["area"] == area for b in order) for area in AREAS]
    if blocks(rows) != organisations or blocks(columns) != [n for n in areas if n]:
        raise ValueError("overview.pdf: the blocks of the grid differ from the organisations or the task areas")
    held = {int(c) for c in release["benchmarks_held"]}
    for x, y, is_dark in cells:
        model, benchmark = models[nearest(rows, y)], order[nearest(columns, x)]
        opened = (release["model_status"][model["model_name"]]["release"] == "released"
                  and int(benchmark["catalog_id"]) not in held)
        if is_dark != opened:
            raise ValueError(f"overview.pdf: the cell of model {model['model_index']} and benchmark "
                             f"{benchmark['catalog_id']} is drawn as {'open' if is_dark else 'held back'}")

    right, top, bottom = max(c[0] for c in cells), min(rows) - 6, max(rows) + 6
    beside = [s for s in drawn if s["box"][0] > right and len(s["boxes"]) == 1]
    stripes = [s for s in beside if s["fill"] != "none" and not s["curved"] and top < s["box"][1] < bottom
               and s["box"][2] - s["box"][0] > 40 and s["box"][3] - s["box"][1] < 12]
    panels = sorted({(round(s["box"][0]), round(s["box"][2])) for s in stripes})
    dots = [((s["box"][0] + s["box"][2]) / 2, (s["box"][1] + s["box"][3]) / 2) for s in beside
            if s["fill"] != "none" and s["curved"] and top < s["box"][1] < bottom
            and abs((s["box"][2] - s["box"][0]) - (s["box"][3] - s["box"][1])) < 0.2]
    lines = cluster(s["box"][0] for s in beside if s["stroke"] != "none" and s["box"][2] - s["box"][0] < 0.1
                    and s["box"][3] - s["box"][1] > (bottom - top) / 2)
    if len(panels) != len(AXES) or len(dots) != len(AXES) * len(models):
        raise ValueError(f"overview.pdf: expected {len(AXES)} strips with one dot per model, found {len(panels)} "
                         f"strips and {len(dots)} dots")
    values = {int(r["model_index"]): r for r in diagnostics}
    for (low, high), (key, ticks, scale, tolerance) in zip(panels, AXES):
        grid = [x for x in lines if low - 1 <= x <= high + 1]
        step = (grid[-1] - grid[0]) / (len(ticks) - 1) if len(grid) == len(ticks) else 0
        if not step or any(abs(x - grid[0] - i * step) > 0.2 for i, x in enumerate(grid)):
            raise ValueError(f"overview.pdf: the grid lines of the {key} strip are not the expected axis")
        inside = [d for d in dots if low - 4 <= d[0] <= high + 4]
        if sorted(nearest(rows, y) for _, y in inside) != list(range(len(models))):
            raise ValueError(f"overview.pdf: the {key} strip does not have one dot per model")
        for x, y in inside:
            index = int(models[nearest(rows, y)]["model_index"])
            shown = ticks[0] + (x - grid[0]) / (grid[-1] - grid[0]) * (ticks[-1] - ticks[0])
            if abs(shown - scale(float(values[index][key]))) > tolerance:
                raise ValueError(f"overview.pdf: the {key} dot of model {index} is not at its value")
    return [("diag", "suites.<headline suite>.overall.models.*: accuracy_first_max, usd_per_1k_requests, "
                     "latency_p50_s by model_index"),
            ("rel", "model_status.*.release, benchmarks_held"), ("corpus/models.csv", "model_index, organisation"),
            ("corpus/benchmarks.csv", "area by catalog_id")]


def check(folder, audit, release, diagnostics, corpus_models, catalog, names=None):
    """Raise ValueError unless the drawings (all, or those named) agree with the inputs; return what was compared."""
    compared = {}
    for name in sorted(LABELS if names is None else names):
        pdf = Path(folder) / name
        expected, sources = expected_numbers(name, audit, release)
        printed = printed_numbers(pdf)
        if printed != expected:
            missing = sorted((Counter(expected) - Counter(printed)).elements())
            extra = sorted((Counter(printed) - Counter(expected)).elements())
            raise ValueError(f"{name}: printed numbers differ from the inputs (not printed: {', '.join(missing) or '-'}; "
                             f"not expected: {', '.join(extra) or '-'})")
        if name == "overview.pdf":
            sources = sources + check_overview(pdf, release, diagnostics, corpus_models, catalog)
        compared[name] = sources
    return compared

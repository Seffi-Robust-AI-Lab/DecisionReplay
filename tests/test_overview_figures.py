"""Figures 1 and 2 are drawn from their inputs: the grid, the dots and the printed totals follow them."""

import math
import re
import shutil
import subprocess

import pytest

from analysis import overview_figures as OF

MODELS = [(0, "north"), (1, "north"), (2, "south"), (3, "east"), (4, "east"), (5, "east")]
AREAS = ("knowledge",) * 3 + ("language",) * 3 + ("retrieval",) * 3 + ("tools",) * 2 + ("arts",) * 2
BENCHMARKS = [(i + 1, area) for i, area in enumerate(AREAS)]
VALUES = {i: dict(accuracy=0.71 + 0.03 * i, cost=0.02 * 2 ** i, latency=0.3 + 0.1 * i) for i in range(6)}
TOTALS = dict(requests=12345, fields=23456, vectors=140736, organisations=3)
PAIR = r"\((-?[\d.]+),(-?[\d.]+)\)"


def after(text, note):
    """Return the lines that follow a comment of the picture, up to the next comment."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("% " + note))
    rest = lines[start + 1:]
    stop = next((i for i, line in enumerate(rest) if line.startswith("% ")), len(rest))
    return lines[start], rest[:stop]


def blocks(starts):
    """Return the sizes of the groups of evenly spaced coordinates that wider gaps separate."""
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    sizes = [1]
    for gap in gaps:
        if gap > min(gaps) + 0.5:
            sizes.append(1)
        else:
            sizes[-1] += 1
    return sizes


@pytest.fixture(scope="module")
def picture():
    return OF.overview(MODELS, BENCHMARKS, VALUES, TOTALS)


def test_the_grid_has_a_cell_per_model_and_benchmark(picture):
    _, lines = after(picture, "cells:")
    corners = [(float(x), float(y)) for x, y in re.findall(PAIR + " rectangle", lines[0])]
    assert len(corners) == len(MODELS) * len(BENCHMARKS)
    columns, rows = sorted({x for x, _ in corners}), sorted({y for _, y in corners})
    assert (len(columns), len(rows)) == (len(BENCHMARKS), len(MODELS))
    assert blocks(columns) == [3, 3, 3, 2, 2]              # the task areas, in the order of the paper
    assert blocks(rows) == [2, 1, 3]                       # the organisations, in the order of the models


def test_every_dot_sits_at_the_value_of_its_model(picture):
    transforms = {"accuracy": lambda v: 100 * v, "cost": math.log10, "latency": float}
    for key, transform in transforms.items():
        note, lines = after(picture, f"dots: {key}")
        low, high, first, last = (float(v) for v in re.search(
            r"axis (-?[\d.]+) to (-?[\d.]+) from x=([\d.]+) to x=([\d.]+)", note).groups())
        dots = [float(re.search(PAIR, line)[1]) for line in lines if "circle" in line][:len(MODELS)]
        assert len(dots) == len(MODELS)
        for (index, _), x in zip(MODELS, dots):
            shown = low + (x - first) / (last - first) * (high - low)
            assert shown == pytest.approx(transform(VALUES[index][key]), abs=0.005 * (high - low))


def test_the_totals_are_printed_and_follow_the_inputs(picture):
    for text in (r"\textbf{12,345} requests", r"\textbf{23,456} questions", r"\textbf{140,736} probability vectors",
                 r"\textbf{6} models, \textbf{3} providers", r"\textbf{13} fixed-choice benchmarks",
                 r"all released: \textbf{6} models \texttimes{} \textbf{13} benchmarks"):
        assert text in picture
    changed = OF.overview(MODELS, BENCHMARKS, VALUES, dict(TOTALS, requests=12346), released=False)
    assert r"\textbf{12,346} requests" in changed and "all released" not in changed


def test_the_axes_hold_the_values():
    assert OF.axis("accuracy", [0.724, 0.819])[0] == [70, 80, 90]
    assert OF.axis("accuracy", [0.447, 0.736])[0] == [40, 50, 60, 70, 80]
    assert OF.axis("accuracy", [0.12, 0.93])[0] == [0, 20, 40, 60, 80, 100]
    ticks, _, low, high = OF.axis("cost", [0.018, 0.405])
    assert (ticks, low, high) == ([-2, -1, 0], "$0.01", "$1")
    ticks, _, low, high = OF.axis("latency", [0.32, 0.95])
    assert (ticks, low, high) == ([0.0, 0.5, 1.0], "0", "1 s")
    assert OF.axis("latency", [0.4, 3.2])[0] == [0.0, 1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("change", ["split organisation", "unknown area", "missing value", "long total",
                                    "narrow areas"])
def test_inputs_that_cannot_be_drawn_stop_the_export(change):
    models, benchmarks, values, totals = list(MODELS), list(BENCHMARKS), dict(VALUES), dict(TOTALS)
    if change == "split organisation":
        models[5] = (5, "north")
    elif change == "unknown area":
        benchmarks[0] = (1, "another area")
    elif change == "missing value":
        del values[3]
    elif change == "narrow areas":                         # the names of the areas would run into each other
        benchmarks = [(i + 1, area) for i, area in enumerate(
            ("knowledge",) * 30 + ("language", "retrieval", "tools", "arts"))]
    else:
        totals["requests"] = 10 ** 40                      # the line would run into the labels of the axes
    with pytest.raises(ValueError):
        OF.overview(models, benchmarks, values, totals)


def test_the_modules_strip_prints_its_totals():
    strip = OF.modules(dict(models=6, observations=74070, cost_usd=12.3456))
    for text in (r"\$12.35 in total", r"\textbf{74,070} records", "6 models"):
        assert text in strip
    with pytest.raises(ValueError):
        OF.modules(dict(models=6, observations=74070.5, cost_usd=12.3456))


def test_text_widths_come_from_the_font():
    assert OF.text_width("requests", 10) < OF.text_width([("requests", True)], 10) < OF.text_width("requests", 12)
    assert OF.text_width("", 10) == 0


@pytest.mark.skipif(shutil.which("pdflatex") is None or shutil.which("pdftotext") is None,
                    reason="LaTeX or poppler is not present")
def test_the_pictures_compile_to_the_size_of_the_column_and_the_page(tmp_path, picture):
    strip = OF.modules(dict(models=6, observations=74070, cost_usd=12.3456))
    for name, text, width, wanted in (("overview", picture, 7.7, ("12,345", "23,456", "140,736")),
                                      ("pipeline", strip, 16.0, ("$12.35", "74,070", "6 models"))):
        (tmp_path / f"{name}.tex").write_text(text, encoding="utf-8")
        (tmp_path / f"page_{name}.tex").write_text(
            "\\documentclass[border=0pt]{standalone}\n\\usepackage[T1]{fontenc}\n\\usepackage{times}\n"
            "\\usepackage{xcolor}\n\\usepackage{tikz}\n\\begin{document}\n\\input{" + name + ".tex}\n"
            "\\end{document}\n", encoding="utf-8")
        done = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"page_{name}.tex"],
                              cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        if done.returncode and "standalone.cls" in done.stdout:
            pytest.skip("the standalone class is not installed")
        assert done.returncode == 0, done.stdout[-2000:]
        info = subprocess.run(["pdfinfo", f"page_{name}.pdf"], cwd=tmp_path, stdout=subprocess.PIPE, text=True).stdout
        points = float(re.search(r"Page size:\s+([\d.]+) x", info)[1])
        assert points == pytest.approx(width / 2.54 * 72, abs=0.2)
        printed = subprocess.run(["pdftotext", f"page_{name}.pdf", "-"], cwd=tmp_path, stdout=subprocess.PIPE,
                                 text=True).stdout
        assert all(text in printed for text in wanted)


def exporter_inputs(tmp_path):
    """Return the arguments of the exporter's ``write_tikz`` for the small corpus of this module."""
    models = tmp_path / "models.csv"
    models.write_text("model_index,organisation\n" + "".join(f"{i},{name}\n" for i, name in MODELS), encoding="utf-8")
    catalog = [dict(catalog_id=str(i), area=area) for i, area in BENCHMARKS]
    rows = [dict(model_index=i, accuracy_first_max=v["accuracy"], usd_per_1k_requests=v["cost"],
                 latency_p50_s=v["latency"]) for i, v in VALUES.items()]
    data = {"diagnostics/diagnostics.json": {"suites": {"suite": {"overall": {"models": rows}}}}}
    audit = dict(corpus=dict(TOTALS, models=len(MODELS), benchmarks=len(BENCHMARKS), observations=74070),
                 telemetry=dict(cost_usd=12.3456))
    return dict(paths={"corpus/models.csv": models}, data=data, documents={"audit": ("audit.json", audit)},
                catalog=catalog, suite="suite")


def test_the_exporter_writes_both_pictures_with_their_sources(tmp_path):
    from analysis import export_paper_assets as exporter

    inputs = exporter_inputs(tmp_path)
    written = exporter.write_tikz(tmp_path / "out", synthetic=False, **inputs)
    assert sorted(written) == ["figures/overview.tex", "figures/pipeline.tex"] and all(written.values())
    assert {role for role, _ in written["figures/overview.tex"]} == {
        "audit.json", "corpus/models.csv", "corpus/benchmarks.csv", "diagnostics/diagnostics.json"}
    overview = (tmp_path / "out" / "figures" / "overview.tex").read_text(encoding="utf-8")
    assert overview == OF.overview(MODELS, BENCHMARKS, VALUES, TOTALS)
    assert r"\$12.35 in total" in (tmp_path / "out" / "figures" / "pipeline.tex").read_text(encoding="utf-8")

    # An audit that lists what a review cleared describes a corpus of which the release is a part.
    partial = exporter_inputs(tmp_path)
    partial["documents"]["audit"][1]["rights"] = dict(cleared_vectors=1, cleared_models=[0], cleared_benchmarks=1)
    exporter.write_tikz(tmp_path / "partial", synthetic=False, **partial)
    assert "all released" not in (tmp_path / "partial" / "figures" / "overview.tex").read_text(encoding="utf-8")


def test_inputs_that_disagree_stop_the_export_of_real_data(tmp_path):
    from analysis import export_paper_assets as exporter

    inputs = exporter_inputs(tmp_path)
    inputs["documents"]["audit"][1]["corpus"]["organisations"] += 1
    with pytest.raises(ValueError):
        exporter.write_tikz(tmp_path / "real", synthetic=False, **inputs)
    # Made-up inputs need not describe one corpus: the overview is left out and the strip is still written.
    assert sorted(exporter.write_tikz(tmp_path / "made_up", synthetic=True, **inputs)) == ["figures/pipeline.tex"]
    assert not (tmp_path / "made_up" / "figures" / "overview.tex").exists()

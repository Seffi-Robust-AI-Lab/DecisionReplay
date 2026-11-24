"""Exercise the public export, review build and release checks on the made-up corpus."""

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pytest

from acl_resource.data import schema


PAPER = Path("acl_resource/paper")
SKELETON = Path("acl_resource/tests/paper_skeleton")  # placeholder text that needs only the made-up inputs
INPUTS = Path("acl_resource/tests/paper_inputs")      # made-up results; generated locally, not versioned
pytestmark = pytest.mark.skipif(not INPUTS.is_dir(), reason="made-up paper inputs are not present")
SETTINGS = Path("acl_resource/SETTINGS.json")


def command(*args, cwd=None):
    """Return status and combined diagnostics from one command."""
    return subprocess.run(args, cwd=cwd, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)


def module(name, *args):
    """Return the result of invoking a public analysis command from the project root."""
    return command("python3", "-B", "-m", "acl_resource.analysis." + name, *args)


def passed(result):
    """Assert a successful command and return its output."""
    assert result.returncode == 0, result.stdout
    return result.stdout


def require_programs(*names):
    """Skip a tool-specific check if an executable is unavailable."""
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        pytest.skip("missing " + ", ".join(missing))


@pytest.fixture(scope="module")
def workspace():
    """Yield an isolated directory under the workspace for copies, tool caches and builds."""
    with TemporaryDirectory(prefix=".paper-test-", dir=".") as directory, pytest.MonkeyPatch.context() as env:
        root = Path(directory).resolve()
        for key in ("TMPDIR", "MPLCONFIGDIR", "TEXMFVAR"):
            local = root / key.lower()
            local.mkdir()
            env.setenv(key, str(local))
        yield root


def copy_paper(root, source=PAPER, generated=False):
    """Return a paper copy with the same relative bibliography layout."""
    paper = root / PAPER
    ignore = ["build"] if generated else ["build", "generated"]
    if Path(source) == PAPER:
        # The real paper quotes results that exist for the real corpus only (stored results of earlier studies,
        # native scores).  The tests take its build files and the skeleton's text.
        shutil.copytree(source, paper, ignore=shutil.ignore_patterns(*ignore, "sections", "main.tex", "drafts"))
        shutil.copytree(SKELETON, paper, dirs_exist_ok=True)
    else:
        shutil.copytree(source, paper, ignore=shutil.ignore_patterns(*ignore))
    shutil.copytree(INPUTS / "related_work", paper.parent / "related_work")
    return paper


def export(paper, corpus, results, *options):
    """Return the result of exporting all inputs to a paper's generated directory."""
    diag = INPUTS / "diagnostics"
    return module("export_paper_assets", "--corpus", corpus, "--results", results,
                  "--diagnostics", diag, "--json", "audit=" + str(INPUTS / "audit.json"),
                  "--json", "diag=" + str(diag / "diagnostics.json"),
                  "--json", "homog=" + str(diag / "homogenization.json"),
                  "--related-work", INPUTS / "related_work", "--out", paper / "generated", *options)


@pytest.fixture(scope="module")
def example(workspace, corpus):
    """Return an exported copy of the paper and its private result input directory."""
    paper = copy_paper(workspace / "example")
    results = workspace / "results"
    shutil.copytree(INPUTS / "results", results)
    passed(export(paper, corpus, results))
    return paper, results


def digests(folder):
    """Return sorted relative product names mapped to their checksums."""
    return {p.relative_to(folder): schema.sha256_file(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def test_reproducible_exports_and_stale_inputs(workspace, corpus, example):
    paper, results = example
    other = copy_paper(workspace / "repeat")
    passed(export(other, corpus, results))
    original = digests(paper / "generated")
    assert original == digests(other / "generated")
    passed(export(paper, corpus, results, "--check"))

    rows_file = results / "rows.json"
    before = rows_file.read_bytes()
    try:
        data = json.loads(before)
        headline = json.loads(SETTINGS.read_text())["headline_suite"]
        row = next(r for r in data["rows"] if r["suite"] == headline and r["method"] == "best_single")
        row["log_loss"] += 1
        rows_file.write_text(json.dumps(data, sort_keys=True))
        stale = export(paper, corpus, results, "--check")
        assert stale.returncode != 0, stale.stdout
        assert digests(paper / "generated") == original
    finally:
        rows_file.write_bytes(before)


def read_json(path):
    """Return the JSON value stored at the given path."""
    return json.loads(path.read_text())


def test_number_formats_and_sources(workspace, corpus, example):
    paper = copy_paper(workspace / "formats")
    _, results = example
    headline = read_json(SETTINGS)["headline_suite"]
    row = next(r for r in read_json(results / "rows.json")["rows"]
               if r["suite"] == headline and r["method"] == "best_single")
    contrast = next(r for r in read_json(results / "contrasts.json")["rows"]
                    if r["suite"] == headline and r["contrast"] == "C1" and r["metric"] == "accuracy")
    positive = next(r for r in read_json(results / "contrasts.json")["rows"]
                    if r["suite"] == headline and r["contrast"] == "C1" and r["metric"] == "log_loss")
    count = read_json(corpus / "manifest.json")["counts"]["requests"]
    first = read_json(INPUTS / "diagnostics" / "homogenization.json")["panels"][0]
    probes = [("n", "corpus.counts.requests", f"{count:,}"),
              ("n", "homog.panels.0.catalog_id", str(first["catalog_id"])),
              ("p", "contrasts.C1.headline.accuracy.p", "<0.001"),
              ("p", "contrasts.C1.headline.log_loss.p_holm", f'{positive["p_holm"]:.3f}'),
              ("s2", "contrasts.C1.headline.log_loss.difference", f'{positive["difference"]:+.2f}')]
    for precision in range(5):
        probes.append((f"f{precision}", "rows.headline.best_single.log_loss", f'{row["log_loss"]:.{precision}f}'))
    for precision in range(3):
        probes.append((f"pct{precision}", "rows.headline.best_single.accuracy", f'{100*row["accuracy"]:.{precision}f}%'))
    for precision in range(2, 5):
        probes.append((f"usd{precision}", "rows.headline.best_single.usd_per_1k_requests",
                       f'${row["usd_per_1k_requests"]:.{precision}f}'))
    for precision in range(1, 5):
        probes.append((f"s{precision}", "contrasts.C1.headline.accuracy.difference",
                       f'{contrast["difference"]:+.{precision}f}'.replace("-", "−")))
    source = [r"\documentclass{article}", r"\usepackage[T1]{fontenc}", r"\usepackage{times}",
              r"\input{generated/numbers}", r"\begin{document}", r"\pagestyle{empty}"]
    source.extend(rf"\noindent Value \num[{fmt}]{{{key}}}\par" for fmt, key, _ in probes)
    (paper / "main.tex").write_text("\n".join(source + [r"\end{document}"]) + "\n")
    passed(export(paper, corpus, results))
    manifest = (paper / "generated" / "manifest.json").read_text()
    for _, key, _ in probes:
        assert key in manifest
    require_programs("pdflatex", "pdftotext")
    passed(command("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex", cwd=paper))
    rendered = passed(command("pdftotext", "-raw", paper / "main.pdf", "-"))
    values = [line[len("Value"):].strip().replace(" ", "") for line in rendered.splitlines() if line.startswith("Value")]
    assert values == [expected for _, _, expected in probes]


def test_generated_figures_and_tables(example):
    paper, _ = example
    generated = paper / "generated"
    provenance = read_json(generated / "manifest.json")
    assert provenance["synthetic"] is True
    assert provenance["inputs"]["results/rows.json"] == schema.sha256_file(INPUTS / "results" / "rows.json")
    for group in ("numbers", "tables", "figures"):
        for records in provenance[group].values():
            assert records
            for record in records:
                assert record["selector"] and record["sha256"] == provenance["inputs"][record["file"]]
    for table in sorted((generated / "tables").glob("*.tex")):
        assert table.relative_to(generated).as_posix() in provenance["tables"]
        assert r"\toprule" in table.read_text()
    figures = sorted((generated / "figures").glob("*.pdf"))
    assert figures
    for figure in figures:
        assert figure.relative_to(generated).as_posix() in provenance["figures"]
        assert b"CreationDate" not in figure.read_bytes()
    require_programs("pdfinfo", "pdffonts")
    for figure in figures:
        info = passed(command("pdfinfo", figure))
        size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info)
        assert size, figure.name
        width_cm = float(size[1]) * 2.54 / 72
        assert min(abs(width_cm - cm) for cm in (7.7, 16.0)) < 0.02, figure.name
        font_rows = passed(command("pdffonts", figure)).splitlines()[2:]
        assert font_rows, figure.name
        for font in font_rows:
            assert "Type 3" not in font and font.split()[-5] == "yes", (figure.name, font)


def append_line(path, source):
    """Append a source line and return its one-based line number."""
    original = path.read_text().rstrip("\n") + "\n"
    path.write_text(original + source + "\n")
    return original.count("\n") + 1


def test_unresolved_export_reports_call_site(workspace, corpus, example):
    original, results = example
    paper = copy_paper(workspace / "unresolved", original)
    probe = paper / "sections" / "probe.tex"
    for source in (r"\num[f2]{audit.unavailable}", r"\num{corpus.model_count}"):
        probe.write_text("% Number call.\n" + source + "\n")
        result = export(paper, corpus, results)
        assert result.returncode != 0 and "probe.tex:2" in result.stdout
        if "unavailable" in source:
            assert "audit.unavailable" in result.stdout


def test_lint_sources_and_literals(workspace, example):
    original, _ = example
    paper = copy_paper(workspace / "draft", original, generated=True)
    args = ("--paper", paper, "--sources-only")
    passed(module("lint_paper", *args))
    section = paper / "sections" / "introduction.tex"
    append_line(section, r"\literal{37} \label{case2} \ref{case2} $a_{2}$ % 200")
    notices = passed(module("lint_paper", *args))
    assert "literal" in notices.lower() and section.name in notices
    main = paper / "main.tex"
    generated = paper / "generated" / "numbers.tex"
    address = "contact" + chr(64) + ".".join(("example", "invalid"))
    planted_path = os.sep + os.path.join("unused", "file")
    for path, source in (
        (section, "There were 47."),
        (section, r"\section{Round 2}"),
        (section, address),
        (main, r"\author{Identifying credit}"),
        (generated, planted_path),
        (section, r"\ref{absent-label}"),
        (section, r"\citet{absent-citation}"),
        (section, r"\num[n]{corpus.unavailable}"),
    ):
        before = path.read_bytes()
        line = append_line(path, source)
        result = module("lint_paper", *args)
        assert result.returncode != 0, (path.name, result.stdout)
        assert f"{path.name}:{line}" in result.stdout, result.stdout
        path.write_bytes(before)
    passed(module("lint_paper", *args))

    term = "withheld-marker"
    terms = workspace / "terms.txt"
    terms.write_text(term + "\n")
    line = append_line(section, term)
    result = module("lint_paper", *args, "--terms", terms)
    assert result.returncode != 0 and f"{section.name}:{line}" in result.stdout
    assert term not in result.stdout


def build(paper):
    """Return the result of the review build using the paper's actual settings."""
    require_programs("latexmk", "pdflatex")
    return command("latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex", cwd=paper)


def test_pdf_build_and_submission_guard(example):
    paper, _ = example
    passed(build(paper))
    pdf = paper / "build" / "main.pdf"
    log = pdf.with_suffix(".log").read_text(errors="replace")
    assert not re.search(r"(?:Citation|Reference).*undefined|undefined (?:references|citations)", log, re.I)
    passed(module("lint_paper", "--paper", paper))
    result = module("lint_paper", "--paper", paper, "--submission")
    assert result.returncode != 0 and "placeholder" in result.stdout.lower()
    assert re.search(r"synthetic|made-up", result.stdout, re.I)
    require_programs("pdftotext", "pdfinfo")
    assert "(A4)" in passed(command("pdfinfo", pdf))
    pages = passed(command("pdftotext", "-raw", pdf, "-")).strip("\n\f").split("\f")
    assert pages and all(re.search(r"MADE[\s‐‑–-]*UP\s+DATA", page) for page in pages)


def test_submission_after_removing_draft_markers(workspace, example):
    require_programs("latexmk", "pdflatex")
    original, _ = example
    paper = copy_paper(workspace / "submission-fixture", original, generated=True)
    for path in [paper / "main.tex", *sorted((paper / "sections").glob("*.tex"))]:
        lines = path.read_text().splitlines()
        lines = [line for line in lines if not line.lstrip().startswith(r"\placeholder{")]
        path.write_text("\n".join(lines) + "\n")
    numbers = paper / "generated" / "numbers.tex"
    numbers.write_text(numbers.read_text().replace(r"\papersynthetictrue", r"\papersyntheticfalse"))
    passed(build(paper))
    passed(module("lint_paper", "--paper", paper, "--submission"))
    require_programs("pdftotext")
    text = passed(command("pdftotext", "-raw", paper / "build" / "main.pdf", "-"))
    assert "MADE-UP DATA" not in text


def test_undefined_number_stops_latex(workspace, example):
    require_programs("latexmk", "pdflatex")
    original, _ = example
    paper = copy_paper(workspace / "undefined", original, generated=True)
    section = paper / "sections" / "introduction.tex"
    append_line(section, r"\num[f3]{corpus.undefined}")
    result = build(paper)
    assert result.returncode != 0 and "corpus.undefined" in result.stdout


def test_submission_guards_on_clean_paper(workspace):
    require_programs("pdffonts", "pdfinfo", "pdftotext")
    from matplotlib import rc_context
    from matplotlib.figure import Figure

    paper = workspace / "font-paper"
    section = paper / "sections" / "limitations.tex"
    section.parent.mkdir(parents=True)
    section.write_text(r"\section*{Limitations}" + "\n")
    numbers = paper / "generated" / "numbers.tex"
    numbers.parent.mkdir()
    numbers.write_text(r"\newif\ifpapersynthetic" + "\n")
    (paper / "main.tex").write_text("\n".join([
        r"\documentclass[a4paper]{article}", r"\usepackage[review]{acl}",
        r"\newcommand{\placeholder}[1]{[Draft: #1]}", r"\input{generated/numbers}",
        r"\begin{document}", r"\input{sections/limitations}", r"\end{document}"]))
    pdf = workspace / "font.pdf"
    for font_type in (42, 3):
        with rc_context({"pdf.fonttype": font_type}):
            fig = Figure(figsize=(210 / 25.4, 297 / 25.4))
            fig.text(.1, .9, "Limitations")
            fig.savefig(pdf, metadata={"CreationDate": None, "Creator": None, "Producer": None})
        args = ("--paper", paper, "--pdf", pdf, "--submission")
        result = module("lint_paper", *args)
        if font_type == 42:
            passed(result)
            for path, source in [(section, r"\placeholder{Draft}"), (numbers, r"\papersynthetictrue")]:
                clean = path.read_bytes()
                line = append_line(path, source)
                flagged = module("lint_paper", *args)
                assert flagged.returncode != 0 and f"{path.name}:{line}" in flagged.stdout
                path.write_bytes(clean)
            passed(module("lint_paper", *args))
        else:
            assert "Type 3" in passed(command("pdffonts", pdf))
            assert result.returncode != 0 and re.search(r"Type\s*3", result.stdout, re.I)


def test_release_guards(workspace):
    folder = workspace / "release"
    corpus_dir = folder / "processed" / "gated"
    corpus_dir.mkdir(parents=True)
    manifest = {"tier": "gated", "providers": ["channel"], "synthetic": True}
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    fields = schema.table_path(corpus_dir, 1, "fields")
    valid = dict(item_index=0, field_index=0, question_type=schema.QUESTION_TYPES[0], n_options=2)
    table = pa.Table.from_pylist([valid], schema=schema.FIELDS)
    schema.write_table(fields, table, schema.FIELDS)
    output = passed(module("lint_release", folder))
    assert "processed" in output and "bytes" in output.lower()

    stray = corpus_dir / "ruvector.db"
    stray.touch()
    result = module("lint_release", folder)
    assert result.returncode != 0 and stray.name in result.stdout
    stray.unlink()
    free = pa.Table.from_pylist([dict(valid, question_type="unrestricted text")], schema=schema.FIELDS)
    schema.write_table(fields, free, schema.FIELDS)
    result = module("lint_release", folder)
    assert result.returncode != 0 and fields.name in result.stdout
    assert "unrestricted text" not in result.stdout
    schema.write_table(fields, table, schema.FIELDS)

    forbidden = schema.table_path(corpus_dir, 1, "labels")
    schema.write_table(forbidden, pa.Table.from_pylist([], schema=schema.LABELS), schema.LABELS)
    result = module("lint_release", folder)
    assert result.returncode != 0 and forbidden.name in result.stdout
    forbidden.unlink()
    passed(module("lint_release", folder))


def test_held_benchmark_absent_from_figures(workspace, corpus):
    """A held benchmark appears in no figure: the headroom figure once listed every benchmark."""
    require_programs("pdftotext")
    from acl_resource.analysis.paper_tables import BENCH_ABBR
    paper = copy_paper(workspace / "held")
    results = workspace / "held-results"
    shutil.copytree(INPUTS / "results", results)
    rows = {int(r["catalog_id"]) for r in csv.DictReader(open(INPUTS / "diagnostics" / "headroom.csv"))}
    held = sorted(rows)[0]
    passed(export(paper, corpus, results, "--hold", str(held)))
    text = command("pdftotext", paper / "generated" / "figures" / "headroom.pdf", "-").stdout
    assert BENCH_ABBR[held] not in text.split()

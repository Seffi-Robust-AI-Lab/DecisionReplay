"""Assemble the package that leaves this machine: paper, anonymised supplementary material, notes for the authors.

    python3 -B -m acl_resource.analysis.build_package --out DIR --results RESULTS --open OPEN_TIER \
        --pdf main=FILE [--zip FILE]

The package is built from named files only; nothing is taken from the git repository.  It never holds request
text, option text, gold answers, per-item results (the ``local`` folder of the results), the local or gated tier,
request embeddings, downloaded papers or agent logs.  Three top-level folders:

``paper``              the main manuscript and its LaTeX source with the generated assets;
``supplementary``      code, tests, the made-up corpus generator, settings, aggregate result files and the open
                       tier (the released corpus: probability vectors, usage and latency of the released models);
``notes_for_authors``  working documents that are not for submission.

After the build, ``acl_resource.analysis.lint_release`` is run on the folder by the caller.  With ``--zip`` the
folder is also written as an archive with sorted entries and fixed time stamps, so the same files give the same
bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {"local", "gated", "features", "fulltext", "cache", "logs", "paper_inputs", "build", ".git",
                   "__pycache__", "dist"}
FORBIDDEN_NAMES = {".DS_Store", "ruvector.db", "agentdb.rvf", "agentdb.rvf.lock"}

SUPPLEMENTARY = (           # patterns relative to the package folder acl_resource/
    "SETTINGS.json", "SETTINGS_RELEASE.json", "INPUTS.json", "reproduce.sh", "reproduce_release.sh",
    "reproduce_followup.sh", "__init__.py",
    "data/*.py", "data/meta/*", "sim/*.py", "sim/baselines/*.py", "analysis/*.py", "synthetic/*.py",
    "tests/*.py", "tests/paper_skeleton/main.tex", "tests/paper_skeleton/sections/*.tex",
    "docs/CONTRACT.md", "docs/DATASET_CARD.md", "docs/LICENSE_LAYERS.md", "docs/CHANGES_AFTER_FIRST_RUN.md",
    "docs/EXPERIMENT_PLAN.md", "docs/AUDIT_RELEASE.json", "docs/HUB_CARD.md", "docs/repo/*",
)
PAPER_SOURCE = (
    "build_paper.sh", "paper/main.tex", "paper/latexmkrc", "paper/sections/*.tex", "paper/acl-style/*",
    "paper/generated/*", "paper/generated/tables/*", "paper/generated/figures/*", "paper/static_figures/*",
    "related_work/build_tex.py", "related_work/references.bib", "related_work/benchmarks.bib",
    "related_work/related_work_main.md", "related_work/related_work_extended.md",
    "related_work/comparison_table.csv", "related_work/comparison_checks.csv",
)
# Hand-drawn figures: none until they are redrawn for the released corpus.  The drawings of the first study
# (paper/static_figures_22_models) show models that are not released and never go into a package.
OPTIONAL = {"paper/static_figures/*"}
NOTES = (
    "docs/ARR_CHECKLIST.md", "docs/DATA_AUDIT.md", "docs/AUDIT_SUMMARY.json", "docs/PLAN.md", "docs/FIGURES.md",
    "docs/method_compatibility.csv", "docs/method_compatibility_topup.csv", "docs/reviews/*.md",
    "docs/agents/*.md", "related_work/WRITER_NOTES.md", "related_work/BRIEF_reader.md",
    "related_work/comparison_table_secondhand.csv", "related_work/bib_verification.csv",
    "related_work/benchmarks_verification.csv", "related_work/audit/*", "related_work/evidence/*.md",
)


def allowed(relative):
    """Whether a path, relative to the folder it is taken from, may go into a package."""
    return relative.name not in FORBIDDEN_NAMES and not relative.name.endswith(".pyc") \
        and not (set(relative.parts) & FORBIDDEN_PARTS)


def copy_followup(results, target):
    """The aggregate files of the follow-up analyses beside the results folder (``results/followup``), if any.

    Only the top-level JSON files are taken: ``local/`` holds per-item arrays and request representations, and
    ``rows.json`` there is a copy of the main one.  The paper build reads ``followup_facts.json`` and
    ``fold_redraws.json`` from this folder.
    """
    folder = Path(results).parent / "followup"
    if not folder.is_dir():
        return 0
    count = 0
    for path in sorted(folder.glob("*.json")):
        if path.name == "rows.json":
            continue
        if not allowed(Path(path.name)) or path.is_symlink():
            raise ValueError(f"followup/{path.name} may not go into a package")
        copy(path, Path(target) / path.name)
        count += 1
    return count


def collect(root, patterns):
    """Files under ``root`` that match one of the patterns, sorted, with the forbidden ones refused."""
    found = []
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if not matches and pattern not in OPTIONAL:
            raise ValueError(f"nothing matches {pattern}")
        for path in matches:
            if path.is_dir():
                continue
            if not allowed(path.relative_to(root)) or path.is_symlink():
                raise ValueError(f"{path.relative_to(root)} may not go into a package")
            found.append(path)
    return sorted(set(found))


def tree(root):
    """Every file below ``root``; any forbidden entry is an error."""
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if not allowed(relative) or path.is_symlink():
            raise ValueError(f"{relative} may not go into a package")
        files.append(path)
    return files


def copy(path, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)


def build(out, results, open_tier, pdfs):
    out, results, open_tier = Path(out), Path(results), Path(open_tier)
    if out.exists():
        raise ValueError("the output folder exists already")
    if out.resolve().is_relative_to((PACKAGE / "data").resolve()) or out.resolve().is_relative_to(results.resolve()):
        raise ValueError("choose an output folder outside the data and results folders")
    if not pdfs:
        pdfs = [("main", PACKAGE / "paper" / "build" / "main.pdf")]
    if [name for name, path in pdfs] != ["main"]:
        raise ValueError("the submission package takes one manuscript: --pdf main=FILE")
    main_pdf = Path(pdfs[0][1])
    if not main_pdf.is_file():
        raise ValueError(f"main manuscript does not exist: {main_pdf}")
    numbers = (PACKAGE / "paper" / "generated" / "numbers.tex").read_text(encoding="utf-8").splitlines()
    if not {r"\papercodedfalse", r"\paperheldfalse"}.issubset(numbers):
        raise ValueError("regenerate the main paper assets with listed model names and all benchmarks")
    code = out / "supplementary" / "acl_resource"
    for path in collect(PACKAGE, SUPPLEMENTARY):
        copy(path, code / path.relative_to(PACKAGE))
    for path in sorted(results.rglob("*")):                       # aggregate result files only
        relative = path.relative_to(results)
        if path.is_file() and relative.parts[0] != "local" and not relative.parts[0].startswith("corpus"):
            if not allowed(relative) or path.is_symlink():
                raise ValueError(f"{relative} may not go into a package")
            copy(path, code / "results" / "main" / relative)
    copy_followup(results, code / "results" / "followup")
    for path in tree(open_tier):
        copy(path, code / "data" / "processed" / "open" / path.relative_to(open_tier))
    source = out / "paper" / "source"
    for path in collect(PACKAGE, PAPER_SOURCE):
        copy(path, source / path.relative_to(PACKAGE))
    copy(main_pdf, out / "paper" / "main.pdf")
    notes = out / "notes_for_authors"
    for path in collect(PACKAGE, NOTES):
        copy(path, notes / path.relative_to(PACKAGE))
    for name in ("README.md", "OPEN_DECISIONS.md"):
        copy(PACKAGE / "docs" / "package" / name, (out if name == "README.md" else notes) / name)
    files = [p for p in sorted(out.rglob("*")) if p.is_file()]
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}" for p in files]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files) + 1


def archive(folder, target):
    """A zip of the folder with sorted entries and one fixed time stamp."""
    folder, target = Path(folder), Path(target)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo((Path(folder.name) / path.relative_to(folder)).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if path.suffix in (".parquet", ".pdf", ".png") else zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, path.read_bytes())
    return target.stat().st_size


def main(argv=None):
    ap = argparse.ArgumentParser(description="Assemble the package of paper, supplementary material and notes")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--open", required=True, type=Path)
    ap.add_argument("--pdf", action="append", default=[], metavar="main=FILE")
    ap.add_argument("--zip", type=Path)
    a = ap.parse_args(argv)
    try:
        pdfs = [tuple(item.split("=", 1)) for item in a.pdf]
        count = build(a.out, a.results, a.open, pdfs)
    except (OSError, ValueError) as error:
        print("package: " + str(error), file=sys.stderr)
        return 1
    print(f"{count} files in {a.out.name}")
    if a.zip:
        print(f"{a.zip.name}: {archive(a.out, a.zip):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

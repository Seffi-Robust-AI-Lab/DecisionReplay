"""Assemble a working copy for continuing the paper with an assistant (for example GPT).

    python3 -B -m acl_resource.analysis.build_handoff --out DIR --results RESULTS --corpus CORPUS --open OPEN_TIER \
        --pdf real=FILE [--zip FILE]

The copy keeps the project's layout, so the commands of ``HANDOFF.md`` (at its root) work from there:
``acl_resource/`` with the code, the tests and the made-up corpus generator, the LaTeX source with the generated
assets of the last paper build, the related-work sources, the documentation, the aggregate result files, the
three corpus files that the paper build reads (``data/corpus_meta/``), the open tier and the made-up inputs of the
paper-asset tests (``tests/paper_inputs/``, refused unless every JSON file in it declares ``synthetic``); ``notes/``
with the open
decisions, reviews, reading notes and agent specifications; ``pdf/`` with the drafts.

Built from named files only, like ``build_package``; nothing is taken from the git repository.  It never holds
request text, option text, gold answers, per-item results, the local or gated tier, the outputs of the models held
back from the open subset, request embeddings, downloaded papers or agent logs.  After the build the caller runs
``lint_release`` and ``scan_item_text`` on the whole folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .build_package import (NOTES, PACKAGE, PAPER_SOURCE, SUPPLEMENTARY, allowed, archive, collect, copy,
                            copy_followup, tree)

CODE = SUPPLEMENTARY + PAPER_SOURCE + ("docs/ARR_CHECKLIST.md", "docs/DATA_AUDIT.md", "docs/AUDIT_SUMMARY.json",
                                      "docs/PLAN.md", "docs/FIGURES.md")
NOTE_ONLY = tuple(p for p in NOTES if p not in CODE)
CORPUS_FILES = ("manifest.json", "benchmarks.csv", "models.csv")


def build(out, results, corpus, open_tier, pdfs):
    out, results, corpus, open_tier = Path(out), Path(results), Path(corpus), Path(open_tier)
    if out.exists():
        raise ValueError("the output folder exists already")
    if out.resolve().is_relative_to((PACKAGE / "data").resolve()) or out.resolve().is_relative_to(results.resolve()):
        raise ValueError("choose an output folder outside the data and results folders")
    code = out / "acl_resource"
    for path in collect(PACKAGE, CODE):
        copy(path, code / path.relative_to(PACKAGE))
    for path in sorted(results.rglob("*")):                       # aggregate result files only
        relative = path.relative_to(results)
        if path.is_file() and relative.parts[0] != "local" and not relative.parts[0].startswith("corpus"):
            if not allowed(relative) or path.is_symlink():
                raise ValueError(f"{relative} may not go into a package")
            copy(path, code / "results" / "main" / relative)
    copy_followup(results, code / "results" / "followup")
    for name in CORPUS_FILES:                                     # metadata only: no table of the corpus
        copy(corpus / name, code / "data" / "corpus_meta" / name)
    for path in tree(open_tier):
        copy(path, code / "data" / "processed" / "open" / path.relative_to(open_tier))
    made_up = PACKAGE / "tests" / "paper_inputs"
    if made_up.is_dir():
        files = tree(made_up)
        if any(p.suffix == ".json" and json.loads(p.read_text(encoding="utf-8")).get("synthetic") is not True
               for p in files):
            raise ValueError("tests/paper_inputs holds a JSON file that does not declare synthetic data")
        for path in files:
            copy(path, code / "tests" / "paper_inputs" / path.relative_to(made_up))
    for path in collect(PACKAGE, NOTE_ONLY):
        copy(path, out / "notes" / path.relative_to(PACKAGE))
    copy(PACKAGE / "docs" / "package" / "OPEN_DECISIONS.md", out / "notes" / "OPEN_DECISIONS.md")
    copy(PACKAGE / "docs" / "package" / "HANDOFF.md", out / "HANDOFF.md")
    for name, pdf in pdfs:
        copy(Path(pdf), out / "pdf" / f"{name}.pdf")
    files = [p for p in sorted(out.rglob("*")) if p.is_file()]
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}" for p in files]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files) + 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Assemble a working copy for continuing the paper with an assistant")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--open", required=True, type=Path)
    ap.add_argument("--pdf", action="append", default=[], metavar="name=FILE")
    ap.add_argument("--zip", type=Path)
    a = ap.parse_args(argv)
    try:
        count = build(a.out, a.results, a.corpus, a.open, [tuple(item.split("=", 1)) for item in a.pdf])
    except (OSError, ValueError) as error:
        print("handoff: " + str(error), file=sys.stderr)
        return 1
    print(f"{count} files in {a.out.name}")
    if a.zip:
        print(f"{a.zip.name}: {archive(a.out, a.zip):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

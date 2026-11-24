"""Assemble a code repository from a built package: its supplementary folder with a guide beside it.

    python3 -B -m acl_resource.analysis.build_repo --package PACKAGE --out DIR

``PACKAGE`` is a folder written by ``build_package``.  Its ``supplementary/acl_resource`` is copied unchanged, so
the repository holds what the release lint has seen: the code, the tests, the settings, the aggregate result files
and the open tier of the corpus.  Beside it go ``README.md`` (``docs/repo/README.md`` with the counts of the
corpus), ``requirements.txt`` and ``.gitignore``, all three from the package.  The paper and the notes for the
authors stay in the package.

A package whose corpus is not an open tier without labels is refused.  After the build the caller runs
``lint_release`` and ``scan_item_text`` on the folder, and only then puts it under version control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acl_resource.analysis.build_package import copy, tree
from acl_resource.data import hub, load

GUIDE = {"README.md": "README.md", "requirements.txt": "requirements.txt", "gitignore": ".gitignore"}


def build(package, out):
    """Write the repository folder and return (number of files, counts of the corpus)."""
    package, out = Path(package), Path(out)
    source = package / "supplementary" / "acl_resource"
    if not source.is_dir():
        raise ValueError("the package has no supplementary code folder")
    if out.exists():
        raise ValueError("the output folder exists already")
    corpus = source / "data" / "processed" / "open"
    manifest = load.read_manifest(corpus)
    if manifest.get("tier") != "open" or "labels" in manifest["tables"] or list(source.rglob("labels.parquet")):
        raise ValueError("the corpus of the package is not an open tier without labels")
    files = tree(source)
    for path in files:
        copy(path, out / "acl_resource" / path.relative_to(source))
    guide = source / "docs" / "repo"
    values = hub.facts(corpus)
    for name, target in GUIDE.items():
        text = (guide / name).read_text(encoding="utf-8")
        (out / target).write_text(hub.fill(text, values), encoding="utf-8")
    return len(files) + len(GUIDE), manifest["counts"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Assemble a code repository from a built package")
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        count, counts = build(args.package, args.out)
    except (OSError, ValueError, KeyError) as error:
        print("repository: " + str(error), file=sys.stderr)
        return 1
    print(f"{count} files in {args.out.name}")
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

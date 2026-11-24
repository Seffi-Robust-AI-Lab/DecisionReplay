"""Compare the result files of two or more runs of ``reproduce.sh``.

    python3 -B -m acl_resource.analysis.compare_runs --out FILE RESULTS_DIR RESULTS_DIR [RESULTS_DIR ...]

Every JSON and CSV result file must be byte-identical across the runs, and every per-field archive must hold the
same arrays.  ``timing.json`` is left out: it records the run time.  The report holds file names relative to the
results folders, checksums and counts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SKIP = {"timing.json", "determinism.json"}


def files(folder):
    """Result files of a run, relative to its folder; the rebuilt corpus is checked through its manifest."""
    folder = Path(folder)
    names = [p.relative_to(folder).as_posix() for p in sorted(folder.rglob("*"))
             if p.is_file() and p.suffix in (".json", ".csv", ".npz")
             and not p.relative_to(folder).parts[0].startswith("corpus")]
    return [n for n in names if n not in SKIP]


def digest(path):
    """Checksum of a file; of the arrays, not of the container, for a compressed archive."""
    h = hashlib.sha256()
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            for key in sorted(archive.files):
                value = archive[key]
                h.update(key.encode() + str(value.dtype).encode() + str(value.shape).encode())
                h.update(np.ascontiguousarray(value).tobytes())
    else:
        h.update(path.read_bytes())
    return h.hexdigest()


def run(folders, out):
    folders = [Path(f) for f in folders]
    names = files(folders[0])
    report, differing = [], []
    for name in names:
        present = [(f / name).is_file() for f in folders]
        sums = [digest(f / name) if ok else None for f, ok in zip(folders, present)]
        same = all(present) and len(set(sums)) == 1
        report.append(dict(file=name, sha256=sums[0], identical=same))
        if not same:
            differing.append(name)
    extra = sorted({n for f in folders[1:] for n in files(f)} - set(names))
    summary = dict(runs=len(folders), reruns=len(folders) - 1, files_compared=len(names),
                   files_identical=len(names) - len(differing), files_only_in_later_runs=extra,
                   all_identical=not differing and not extra, files=report)
    Path(out).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare the result files of several runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("folders", nargs="+")
    a = ap.parse_args(argv)
    if len(a.folders) < 2:
        ap.error("give at least two results folders")
    summary = run(a.folders, a.out)
    print(f"{summary['files_identical']} of {summary['files_compared']} files identical over {summary['runs']} runs")
    return 0 if summary["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Count requests whose text also occurs in another fold of the same benchmark.  A local step (reads rows).

    python3 -B -m analysis.leakage --rows DIR --out FILE [--corpus DIR]

Without ``--corpus`` the count is over the source export and its rows.  With ``--corpus`` it is over the requests
of that corpus: its benchmarks, and in each of them the rows of the requests that the corpus holds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data import splits


def corpus_report(rows_dir, corpus):
    """{catalogue id: requests of the corpus whose text also occurs in another fold}."""
    from data import schema as S
    from data.load import corpus_rows, read_manifest
    report = {}
    for cid in read_manifest(corpus)["catalog_ids"]:
        items = S.read_table(S.table_path(corpus, cid, "items"), S.ITEMS)
        report[cid] = splits._duplicate_count(cid, items, corpus_rows(corpus, rows_dir, cid))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Duplicate request text across folds: counts only")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus")
    a = ap.parse_args(argv)
    report = corpus_report(a.rows, a.corpus) if a.corpus else splits.duplicate_report(a.rows)
    out = dict(requests_with_duplicate_text_across_folds=int(sum(report.values())),
               benchmarks_affected=int(sum(1 for v in report.values() if v)), benchmarks=len(report),
               per_benchmark={str(k): int(v) for k, v in sorted(report.items()) if v})
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

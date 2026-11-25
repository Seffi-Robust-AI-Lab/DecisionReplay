"""Request representations for a corpus that holds fewer requests than the benchmark rows.  A local step.

    python3 -B -m analysis.features_for_corpus --features DIR --rows DIR --corpus DIR --out DIR

``--features`` holds one folder per representation, each with ``features.json`` and one matrix per benchmark with
one row per request of the benchmark rows (built by ``embed_requests``, ``featurize_requests`` or
``embed_requests_full`` on a corpus with every request).  ``--corpus`` is a corpus from which requests were removed
(``manifest.json``: ``excluded_requests``).  For every representation a folder of the same name is written to
``--out`` with one row per request of that corpus:

* a benchmark that keeps all its requests: the matrix is copied as it is;
* a representation computed request by request (the sentence encoders): the rows of the kept requests are taken;
* TF-IDF with a truncated SVD and the decoder states with a PCA are fitted per benchmark, so they are fitted again
  on the kept requests (the unreduced decoder states of the kept requests are taken, no model is run).

Requests are matched by request id (``data.tables.row_positions``), never by position alone.  The matrices are
derived from benchmark text and stay local.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from data import schema as S
from data import tables as T
from data.load import list_benchmarks, read_manifest


def _positions(corpus, rows_dir, cid, excluded):
    rows = T.read_rows(Path(rows_dir) / f"{cid:03d}.jsonl.gz")
    ids = S.read_table(S.table_path(corpus, cid, "items"), S.ITEMS)["item_id"].to_pylist()
    return rows, np.asarray(T.row_positions(cid, ids, rows, excluded), dtype=np.int64)


def one_representation(source, rows_dir, corpus, out):
    """Write the folder ``out`` for the representation in ``source``; return {catalogue id: "copied" | "subset" | "refitted"}."""
    from . import featurize_requests as FR
    from .embed_requests import request_text
    source, corpus, out = Path(source), Path(corpus), Path(out)
    info = json.loads((source / "features.json").read_text(encoding="utf-8"))
    kind = info.get("kind")                       # "tfidf", "lm", or absent for an encoder applied per request
    whole_request = "window" in info              # embed_requests_full: counts of tokens and windows per benchmark
    excluded = read_manifest(corpus).get(S.EXCLUDED_KEY, {})
    out.mkdir(parents=True, exist_ok=True)
    if kind == "lm":
        (out / "unreduced").mkdir(exist_ok=True)
    entries, done, tokeniser = {}, {}, None
    for b in list_benchmarks(corpus):
        cid = b["catalog_id"]
        name = f"catalog-{cid:03d}.npy"
        gone = excluded.get(str(cid), 0)
        entry = dict(info["benchmarks"][str(cid)])
        x = np.load(source / name, allow_pickle=False)
        if not gone:
            if x.shape[0] != int(b["requests"]):
                raise ValueError(f"benchmark {cid}: the matrix does not match the corpus")
            shutil.copyfile(source / name, out / name)
            if kind == "lm":
                shutil.copyfile(source / "unreduced" / name, out / "unreduced" / name)
            entries[str(cid)], done[cid] = entry, "copied"
            continue
        rows, keep = _positions(corpus, rows_dir, cid, gone)
        if x.shape[0] != len(rows) or len(keep) != int(b["requests"]):
            raise ValueError(f"benchmark {cid}: the matrix does not have one row per benchmark row")
        if kind == "tfidf":
            new, done[cid] = FR.tfidf_matrix([request_text(rows[i]) for i in keep]), "refitted"
        elif kind == "lm":
            raw = np.load(source / "unreduced" / name, allow_pickle=False)
            if raw.shape[0] != len(rows):
                raise ValueError(f"benchmark {cid}: the unreduced states do not have one row per benchmark row")
            np.save(out / "unreduced" / name, raw[keep])
            new, done[cid] = FR.reduce_pca(raw[keep]), "refitted"
        else:
            new, done[cid] = np.ascontiguousarray(x[keep]), "subset"
        if new.dtype != np.float32 or not np.isfinite(new).all():
            raise ValueError(f"benchmark {cid}: the derived matrix is not a finite float32 matrix")
        np.save(out / name, new)
        entry.update(requests=int(new.shape[0]), dimensions=int(new.shape[1]))
        if "sha256" in entry:
            entry["sha256"] = FR.checksum(out / name)
        if whole_request:
            from . import embed_requests_full as EF
            if tokeniser is None:
                from transformers import AutoTokenizer
                tokeniser = AutoTokenizer.from_pretrained(info["model"])
            ids = tokeniser([EF.full_text(rows[i]) for i in keep], add_special_tokens=False, truncation=False,
                            verbose=False)["input_ids"]
            entry.update(windows=int(sum(len(EF.windows(len(s), int(info["window"]))) for s in ids)),
                         tokens=int(sum(len(s) for s in ids)), distinct=int(len(np.unique(new, axis=0))))
        entries[str(cid)] = entry
    info["benchmarks"] = entries
    info["requests"] = ("the requests of a corpus from which requests were removed: copied matrices, and for the "
                        "benchmarks with removed requests " + ("matrices fitted again on the kept requests"
                                                               if kind in ("tfidf", "lm") else
                                                               "the rows of the kept requests"))
    (out / "features.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return done


def run(features, rows_dir, corpus, out):
    features, out = Path(features), Path(out)
    report = {}
    for folder in sorted(p for p in features.iterdir() if (p / "features.json").is_file()):
        if folder.resolve() == out.resolve() or out.resolve().is_relative_to(folder.resolve()):
            continue
        report[folder.name] = one_representation(folder, rows_dir, corpus, out / folder.name)
    if not report:
        raise ValueError("no representation found")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Request representations for a corpus with removed requests")
    for name in ("features", "rows", "corpus", "out"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args(argv)
    for name, done in run(a.features, a.rows, a.corpus, a.out).items():
        counts = {how: sum(1 for v in done.values() if v == how) for how in ("copied", "subset", "refitted")}
        print(name, {k: v for k, v in counts.items() if v})
    return 0


if __name__ == "__main__":
    sys.exit(main())

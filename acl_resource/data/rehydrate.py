"""Rebuild label tables in a private copy of a gated or open corpus."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from . import schema as S
from .build_corpus import derive_benchmark, staged_output, verify, write_manifest


def _labels(corpus, rows_dir, cid, excluded=0):
    """Return labels after checking row binding, field layout and frozen splits."""
    _, fields, labels, splits = derive_benchmark(corpus, rows_dir, cid, excluded)
    if not fields.equals(S.read_table(S.table_path(corpus, cid, "fields"), S.FIELDS)):
        raise ValueError(f"catalogue {cid}: derived fields differ from the corpus")
    if not splits.equals(S.read_table(S.table_path(corpus, cid, "splits"), S.SPLITS)):
        raise ValueError(f"catalogue {cid}: derived splits differ from the corpus")
    return labels


def _payload_hashes(directory, manifest):
    """Return payload paths and hashes bound by the manifest's checksum digest."""
    data = (directory / "SHA256SUMS").read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["checksums_sha256"]:
        raise ValueError("checksum list changed during rehydration")
    entries = (line.split(maxsplit=1) for line in data.decode("utf-8").splitlines())
    return {path: digest for digest, path in entries}


def rehydrate(corpus, rows, out):
    """Return the manifest of the verified corpus copy with locally rebuilt labels."""
    corpus, rows, out = Path(corpus), Path(rows), Path(out)
    if out.resolve().is_relative_to(corpus.resolve()):
        raise ValueError("choose a separate directory for the private corpus")
    manifest = verify(corpus)
    if manifest["tier"] not in ("gated", "open") or "labels" in manifest["tables"]:
        raise ValueError("rehydration requires a gated or open corpus without labels")

    payload = _payload_hashes(corpus, manifest)
    with staged_output(out) as stage:
        for name in payload:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(corpus / name, target)
        judged_fields = 0
        for cid in manifest["catalog_ids"]:
            labels = _labels(stage, rows, cid, manifest.get(S.EXCLUDED_KEY, {}).get(str(cid), 0))
            S.write_table(S.table_path(stage, cid, "labels"), labels, S.LABELS)
            judged_fields += int(labels["judged"].to_numpy().sum())
        manifest["counts"]["judged_fields"] = judged_fields
        manifest["labels"] = "rehydrated"
        present = set(manifest["tables"]) | {"labels"}
        manifest["tables"] = [t for t in S.TABLES_BY_TIER["local"] if t in present]
        manifest = write_manifest(stage, manifest)
        copied = _payload_hashes(stage, manifest)
        if any(copied[name] != digest for name, digest in payload.items()):
            raise ValueError("corpus payload changed during rehydration")
    return manifest


def main(argv=None):
    """Return zero after rebuilding the labels from the supplied rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "rows", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        rehydrate(args.corpus, args.rows, args.out)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f"rehydration failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen-fold reports and evaluation tracks for a processed corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from . import load
from . import schema as S
from . import tables as T


def _fold_counts(corpus, cid, has_labels):
    """Return aggregate request, split-group and optional judged-field counts."""
    splits = S.read_table(S.table_path(corpus, cid, "splits"), S.SPLITS)
    fold_by_item = {}
    requests = [0] * S.N_FOLDS
    groups = [set() for _ in range(S.N_FOLDS)]
    for row in splits.to_pylist():
        fold = row["fold"]
        if fold not in range(S.N_FOLDS):
            raise ValueError(f"catalogue {cid} has an invalid fold")
        fold_by_item[row["item_index"]] = fold
        requests[fold] += 1
        groups[fold].add(row["split_group"])
    folds_per_group = Counter(group for fold_groups in groups for group in fold_groups)
    result = {
        "split_groups_in_multiple_folds": sum(n > 1 for n in folds_per_group.values()),
        "requests_per_fold": requests,
        "split_groups_per_fold": [len(fold_groups) for fold_groups in groups],
    }
    if has_labels:
        labels = S.read_table(S.table_path(corpus, cid, "labels"), S.LABELS,
                              columns=["item_index", "judged"])
        judged = [0] * S.N_FOLDS
        for row in labels.to_pylist():
            if row["judged"]:
                judged[fold_by_item[row["item_index"]]] += 1
        result["judged_fields_per_fold"] = judged
    return result


def leakage_report(corpus):
    """Return fold counts by catalogue id, including split groups crossing folds."""
    has_labels = "labels" in load.read_manifest(corpus)["tables"]
    return {b["catalog_id"]: _fold_counts(corpus, b["catalog_id"], has_labels)
            for b in load.list_benchmarks(corpus)}


def tracks(corpus, settings):
    """Return rotations, suite benchmark lists and leave-one-area-out benchmark lists."""
    if isinstance(settings, (str, Path)):
        settings = json.loads(Path(settings).read_text(encoding="utf-8"))
    area_of = {b["catalog_id"]: b["area"] for b in load.list_benchmarks(corpus)}
    for first, second in ((1, 62), (2, 3)):
        if first in area_of and second in area_of and area_of[first] != area_of[second]:
            raise ValueError(f"catalogues {first} and {second} must share a task area")

    rotations = []
    for test in range(S.N_FOLDS):
        fit, validation, _ = S.roles(test)
        rotations.append(dict(fit=list(fit), validation=validation, test=test))
    suites = {suite: [cid for cid in area_of if cid not in spec["exclude"]]
              for suite, spec in settings["suites"].items()}
    leave_one_area_out = {}
    for area in sorted(set(area_of.values())):
        leave_one_area_out[area] = {
            "target": [cid for cid, a in area_of.items() if a == area],
            "source": [cid for cid, a in area_of.items() if a != area],
        }
    return dict(rotations=rotations, suites=suites, leave_one_area_out=leave_one_area_out)


def _normalised_json(value):
    """Return canonical JSON with lower-case, whitespace-collapsed string keys and values."""
    if isinstance(value, dict):
        members = sorted((_normalised_json(k), _normalised_json(v)) for k, v in value.items())
        return "{" + ",".join(key + ":" + val for key, val in members) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_normalised_json(element) for element in value) + "]"
    if isinstance(value, str):
        value = " ".join(value.lower().split())
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _duplicate_count(cid, items, rows):
    """Return the number of requests with matching text in a different fold."""
    T.check_binding(cid, items, rows)
    counts = defaultdict(Counter)
    for row, fold in zip(rows, items["fold"].to_pylist(), strict=True):
        payload = _normalised_json([row["state"], row["questions"]])
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        counts[digest][fold] += 1
    return sum(by_fold.total() for by_fold in counts.values() if len(by_fold) > 1)


def duplicate_report(rows_dir):
    """Return cross-fold duplicate request counts by catalogue id in the source layout."""
    rows_dir = Path(rows_dir).resolve()
    layout = Path("data/decision_index/common-paid-65/inputs")
    if rows_dir.parts[-len(layout.parts):] != layout.parts or not rows_dir.is_dir():
        raise ValueError("benchmark rows must be in the data tree paired with the source export")
    root = rows_dir.parents[len(layout.parts) - 1]
    source = root / "results/decision_index/paper/dataset/candidate"
    report = {}
    for path in sorted(rows_dir.glob("[0-9][0-9][0-9].jsonl.gz")):
        cid = int(path.name.split(".")[0])
        items = S.read_table(S.table_path(source, cid, "items"), S.ITEMS)
        report[cid] = _duplicate_count(cid, items, T.read_rows(path))
    return report


def main(argv=None):
    """Write tracks.json to the requested directory and return the exit status."""
    parser = argparse.ArgumentParser(description="Write the evaluation tracks of a processed corpus")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = tracks(args.corpus, args.settings)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tracks.json").write_text(json.dumps(result, sort_keys=True, indent=1) + "\n",
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

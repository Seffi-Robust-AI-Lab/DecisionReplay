"""The comparison of result folders: identical files pass, a changed value or a missing file fails."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from analysis import compare_runs


def write_run(folder, value=1.0, extra=False):
    (folder / "local").mkdir(parents=True)
    (folder / "corpus").mkdir()
    (folder / "rows.json").write_text(json.dumps({"rows": [value]}))
    (folder / "timing.json").write_text(json.dumps({"seconds": hash(str(folder)) % 100}))
    (folder / "corpus" / "manifest.json").write_text(str(folder))            # not a result file
    with open(folder / "local" / "fields-a.npz", "wb") as stream:
        np.savez_compressed(stream, mass=np.array([value, 0.5]), hit=np.array([True, False]))
    if extra:
        (folder / "extra.csv").write_text("a\n1\n")


def test_compare_runs():
    with TemporaryDirectory(dir=".") as directory:
        root = Path(directory)
        for name, value, extra in (("a", 1.0, False), ("b", 1.0, False), ("c", 2.0, False), ("d", 1.0, True)):
            write_run(root / name, value, extra)
        same = compare_runs.run([root / "a", root / "b"], root / "same.json")
        assert same["all_identical"] and same["files_compared"] == same["files_identical"] == 2
        assert [f["file"] for f in same["files"]] == ["local/fields-a.npz", "rows.json"]
        changed = compare_runs.run([root / "a", root / "c"], root / "changed.json")
        assert not changed["all_identical"] and changed["files_identical"] == 0
        more = compare_runs.run([root / "a", root / "d"], root / "more.json")
        assert not more["all_identical"] and more["files_only_in_later_runs"] == ["extra.csv"]
        assert compare_runs.main(["--out", str(root / "x.json"), str(root / "a"), str(root / "c")]) == 1

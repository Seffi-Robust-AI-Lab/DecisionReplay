"""Each benchmark's own metric, for pooled vectors.  A local step: it needs the benchmark rows and the pinned kit.

The benchmark kit defines one metric per benchmark (accuracy, macro-F1, nDCG@10, exact match per case, ...).  The
earlier project wrapped the kit's scorers in ``jdi.paper_scores``; this module feeds that wrapper with the
decisions of a policy.  It imports code from the project folder that holds the kit, so it is not part of what
runs on a released corpus alone.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data import tables as T
from . import core
from .types import LOG_FLOOR

RECALL_ONLY = (2, 36)          # retrieval benchmarks whose native metric ranks candidates by p(yes)


class _ModelCount:
    """The wrapper reads only the number of models from its probability argument."""

    def __init__(self, m):
        self.m = m

    def __getitem__(self, i):
        return np.empty((self.m, 1))


@lru_cache(maxsize=None)
def _scorer(project_root, rows_dir, cid, bench_key):
    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from jdi.paper_scores import Decoded, PaperScores      # earlier project's wrapper around the pinned kit
    bench = _BENCHES[bench_key]
    rows = T.read_rows(Path(rows_dir) / f"{cid:03d}.jsonl.gz")
    if bench.item_id is not None:          # a corpus may hold fewer requests than the rows: bind them by request id
        rows = T.rows_of(cid, bench.item_id, rows)
    if len(rows) != bench.R or sum(len(row["questions"]) for row in rows) != bench.F:
        raise ValueError(f"benchmark {cid}: the benchmark rows do not match the corpus")
    names = tuple(key for row in rows for key in row["questions"])
    gold = [()] * bench.F
    for w in bench.widths:
        for k, f in enumerate(bench.idx[w]):
            gold[f] = tuple(int(i) for i in np.flatnonzero(bench.acc[w][k]))
    view = SimpleNamespace(cid=cid, rows=rows, field_rows=bench.field_row, field_names=names,
                           groups=bench.split_group, gold=tuple(gold), probs=_ModelCount(bench.M), folds=bench.fold)
    return PaperScores(view), Decoded


_BENCHES = {}


def native_score(project_root, rows_dir, bench, pooled):
    """The benchmark's own metric for pooled vectors {width: [n_w, w]} (full precision)."""
    key = id(bench)
    _BENCHES[key] = bench
    scores, Decoded = _scorer(str(project_root), str(rows_dir), bench.cid, key)
    hit, mass, choice, _ = core.decode(bench, pooled)
    loss = np.full(bench.F, np.nan)
    loss[bench.judged] = core.loss_of_mass(mass[bench.judged], LOG_FLOOR)
    yes = np.full(bench.F, np.nan)
    if bench.cid in RECALL_ONLY:
        for w in bench.widths:
            if w == 2:
                v = pooled[w]
                yes[bench.idx[w]] = (v / v.sum(axis=1, keepdims=True))[:, 1]
    d = Decoded(choice[None, :].astype(np.int32), loss[None, :], hit[None, :], yes[None, :])
    return float(scores.reduce(d, "native")[0])

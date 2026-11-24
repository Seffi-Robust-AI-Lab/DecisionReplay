"""Fold-specific scoring and calibration tables for one NLP benchmark."""

from functools import cached_property

import numpy as np

from ..data.load import list_benchmarks, load_benchmark
from ..data.schema import N_FOLDS, roles
from . import core
from .types import MENU


def select_recalibration(sums, counts, folds):
    """Return the loss-minimising transform per model from the specified fold totals."""
    folds = sorted(set(map(int, folds)))
    n = int(sum(counts[g] for g in folds))
    if n == 0:
        return np.zeros(sums.shape[-1], dtype=np.int64)
    losses = np.sum([sums[g] for g in folds], axis=0) / n
    minima = losses.min(axis=0)
    tol = 32 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(minima))
    return (losses <= minima + tol).argmax(axis=0)


class Task:
    """Judged-field and split-group tables in the order of the processed corpus."""

    def __init__(self, bench):
        if not bench.has_labels:
            raise ValueError("A task needs the benchmark labels table")
        self.b = bench
        self.cid, self.name, self.area = bench.cid, bench.name, bench.area
        self.F, self.R = bench.F, bench.R
        self.field_fold = bench.fold[bench.field_row].astype(np.int64)
        self.J = np.flatnonzero(bench.judged)
        self.nJ = len(self.J)
        self.jpos = np.full(self.F, -1, dtype=np.int64)
        self.jpos[self.J] = np.arange(self.nJ)
        self.fold = self.field_fold[self.J]
        self.width = bench.width[self.J].astype(np.int64)
        self.n_acc = bench.n_accepted[self.J].astype(np.int64)
        self.row_group = bench.split_group
        self.G = len(np.unique(self.row_group))
        self.group_fold = np.empty(self.G, dtype=np.int64)
        self.group_fold[self.row_group] = bench.fold
        self.field_group = self.row_group[bench.field_row]
        self.group = self.field_group[self.J]
        self._recal = {}
        self._views = {}

    def block_mass(self, w, V, sel=None):
        """Return accepted mass for fields selected within a width block."""
        return core.block_mass(self.b, w, V, sel)

    @cached_property
    def mass_menu(self):
        """Return accepted masses indexed by menu entry, judged field and model."""
        out = np.empty((len(MENU), self.nJ, self.b.M))
        for w in self.b.widths:
            ids = self.b.idx[w]
            sel = np.flatnonzero(self.b.judged[ids])
            if len(sel):
                positions = self.jpos[ids[sel]]
                vectors = self.b.P[w][sel]
                for c, (kind, par) in enumerate(MENU):
                    out[c, positions] = self.block_mass(w, core.transform(vectors, kind, par), sel)
        return out

    @property
    def mass(self):
        """Return accepted mass on the original vectors for each judged field and model."""
        return self.mass_menu[0]

    @cached_property
    def cal_stats(self):
        """Return summed log losses by fold, transform and model, and field counts by fold."""
        loss = core.loss_of_mass(self.mass_menu)
        sums = np.stack([loss[:, self.fold == g].sum(axis=1) for g in range(N_FOLDS)])
        return sums, np.bincount(self.fold, minlength=N_FOLDS)

    def recal_choice(self, folds):
        """Return the first minimum-loss transform within rounding error for each model."""
        return select_recalibration(*self.cal_stats, folds)

    def recal(self, f, n=3):
        """Return deployment and leave-one-fold-out transforms fitted on n folds."""
        if (f, n) not in self._recal:
            fit, _, train = roles(f)
            used = fit if n == 3 else train
            cross = {g: self.recal_choice(x for x in used if x != g) for g in used}
            self._recal[f, n] = self.recal_choice(used), cross
        return self._recal[f, n]

    def choice_by_fold(self, f, n=3):
        """Return the transform table indexed by physical fold and model for rotation f."""
        deployment, cross = self.recal(f, n)
        table = np.tile(deployment, (N_FOLDS, 1))
        for g, choice in cross.items():
            table[g] = choice
        return table

    def view(self, f, regime="raw", n=3):
        """Return original or cross-fitted accepted masses for rotation f."""
        if regime == "raw":
            return self.mass
        if regime != "calibrated":
            raise ValueError(regime)
        if (f, n) not in self._views:
            choice = self.choice_by_fold(f, n)[self.fold]
            self._views[f, n] = self.mass_menu[choice, np.arange(self.nJ)[:, None], np.arange(self.b.M)]
        return self._views[f, n]

    def rows(self, folds):
        """Return positions in the judged table for the given folds."""
        return np.flatnonzero(np.isin(self.fold, list(folds)))

    def fit_loss(self, f, W, regime="raw", folds=None, n=3):
        """Return mean log loss on the selected folds for each row of weights."""
        sel = self.rows(roles(f)[0] if folds is None else folds)
        return core.loss_of_mass(self.view(f, regime, n)[sel] @ np.atleast_2d(W).T).mean(axis=0)

    def stitched_loss(self, W, regime="raw", n=3):
        """Return judged losses from the five rotations in field order."""
        W = np.asarray(W, dtype=np.float64)
        out = np.empty(self.nJ)
        for f in range(N_FOLDS):
            sel = np.flatnonzero(self.fold == f)
            weights = W[f] / W[f].sum()
            out[sel] = core.loss_of_mass(self.view(f, regime, n)[sel] @ weights)
        return out

    def stitch(self, W, choice=None):
        """Return pooled vectors from the five held-out folds, grouped by width."""
        W = np.asarray(W, dtype=np.float64)
        out = {}
        for w in self.b.widths:
            folds = self.field_fold[self.b.idx[w]]
            pooled = np.empty((len(folds), w))
            for f in range(N_FOLDS):
                sel = np.flatnonzero(folds == f)
                if not len(sel):
                    continue
                weights = W[f] / W[f].sum()
                members = np.flatnonzero(weights > 0)
                if choice is None:
                    vectors = self.b.P[w][np.ix_(sel, members)]
                    pooled[sel] = np.einsum("m,nmo->no", weights[members], vectors)
                else:
                    partial = np.zeros((len(sel), w))
                    for c in np.unique(choice[f][members]):
                        models = members[choice[f][members] == c]
                        vectors = self.b.P[w][np.ix_(sel, models)]
                        partial += np.einsum("m,nmo->no", weights[models], core.transform(vectors, *MENU[c]))
                    pooled[sel] = partial
            out[w] = pooled
        return out

    def pool_fields(self, w, weights, choice=None, sel=None):
        """Return pooled vectors of selected fields of one width with a single weight vector."""
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        members = np.flatnonzero(weights > 0)
        rows = np.arange(len(self.b.idx[w]))
        if sel is not None:
            rows = rows[sel]

        def selected(models):
            """Return selected vectors with the model-major layout of the reference contraction."""
            return self.b.P[w][rows[None, :], models[:, None]].transpose(1, 0, 2)

        if choice is None:
            return np.einsum("m,nmo->no", weights[members], selected(members))
        choice = np.asarray(choice)
        out = np.zeros((len(rows), w))
        for c in np.unique(choice[members]):
            models = members[choice[members] == c]
            out += np.einsum("m,nmo->no", weights[models], core.transform(selected(models), *MENU[c]))
        return out

    def fold_blocks(self, folds, judged_only=False):
        """Return selected field positions within each nonempty width block."""
        folds = list(folds)
        blocks = {}
        for w in self.b.widths:
            ids = self.b.idx[w]
            keep = np.isin(self.field_fold[ids], folds)
            if judged_only:
                keep &= self.b.judged[ids]
            sel = np.flatnonzero(keep)
            if len(sel):
                blocks[w] = sel
        return blocks

    def mass_of_blocks(self, vectors, blocks):
        """Return judged masses of selected blocks, with NaN outside the blocks."""
        out = np.full(self.nJ, np.nan)
        for w, sel in blocks.items():
            out[self.jpos[self.b.idx[w][sel]]] = self.block_mass(w, vectors[w], sel)
        return out

    def decode(self, pooled):
        """Return hit, accepted mass and decoded option for every field."""
        return core.decode(self.b, pooled)[:3]


def load_task(corpus, cid):
    """Return the task for one catalogue id of a processed corpus."""
    return Task(load_benchmark(corpus, cid, labels=True))


def load_tasks(corpus, cids=None):
    """Return a dictionary of tasks for the given catalogue ids or the full corpus."""
    if cids is None:
        cids = [row["catalog_id"] for row in list_benchmarks(corpus)]
    return {cid: load_task(corpus, cid) for cid in cids}

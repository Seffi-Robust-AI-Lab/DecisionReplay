"""Single-model selection using only benchmarks in other task areas."""

import math

import numpy as np

from ...data.schema import N_FOLDS
from ..core import loss_of_mass
from .uniform_mean import panel

NAME = "transfer_single"


def source_cids(areas, held_area):
    """Return the ordered catalogue ids of benchmarks outside the held-out area."""
    return sorted(cid for cid, area in areas.items() if area != held_area)


def equal_mean(profiles):
    """Return the mean loss of each candidate, giving each source benchmark equal weight."""
    profiles = list(profiles)
    if not profiles:
        raise ValueError("transfer requires source benchmarks in other task areas")
    return np.array([math.fsum(column) / len(profiles) for column in zip(*profiles)])


def run_held_out_areas(tasks, settings, source_loss, choose, deploy):
    """Return transferred outcomes using one fitted choice per held-out task area."""
    areas = {cid: task.area for cid, task in tasks.items()}
    if tasks and len(set(areas.values())) < 2:
        raise ValueError("transfer requires source benchmarks in other task areas")
    profiles = {cid: source_loss(tasks[cid], settings) for cid in sorted(tasks)}
    choices = {area: choose((profiles[c] for c in source_cids(areas, area)), settings)
               for area in sorted(set(areas.values()))}
    return {cid: deploy(tasks[cid], choices[areas[cid]]) for cid in sorted(tasks)}


def source_loss(task, settings):
    """Return whole-benchmark original-vector mean log loss for each model."""
    mass = np.empty((task.nJ, task.b.M))
    for w, sel in task.fold_blocks(range(N_FOLDS), judged_only=True).items():
        rows = task.jpos[task.b.idx[w][sel]]
        mass[rows] = task.block_mass(w, task.b.P[w][sel], sel)
    return loss_of_mass(mass).mean(axis=0)


def choose(profiles, settings):
    """Return the lowest-loss model on the source benchmarks, breaking ties by model index."""
    return int(equal_mean(profiles).argmin())


def deploy(task, choice):
    """Return original vectors from the source-selected model on every target request."""
    models = [choice] * N_FOLDS
    weights = np.eye(task.b.M)[models]
    return panel(task, weights, ledger={"other_benchmark_labels": True}, info={"model": models})


def run_suite(tasks, settings):
    """Return transferred single-model outcomes by catalogue id."""
    return run_held_out_areas(tasks, settings, source_loss, choose, deploy)

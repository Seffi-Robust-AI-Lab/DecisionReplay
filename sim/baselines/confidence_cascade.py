"""Replay a cost-ordered chain, stopping at a request confidence threshold."""

import math

import numpy as np

from data.schema import N_FOLDS, roles
from ..core import loss_of_mass, transform
from ..types import MENU, NOT_CALLED, Outcome, empty_ledger
from .knn_router import (choose_parameter, column_means, per_fold, selection_record, validation_budget,
                         validation_loss)
from .random_single import request_vectors

NAME = "confidence_cascade"
CONFIDENCE = ("lowest", "mean")         # how the top probabilities of a request's fields become one confidence


def cost_chain(task, fold, max_stages):
    """Return the cheapest, middle and lowest-loss models on the fit cost/loss frontier."""
    fit, _, _ = roles(fold)
    cost = column_means(task.b.cost[np.isin(task.b.fold, fit)])
    losses = column_means(loss_of_mass(task.view(fold, "calibrated")[task.rows(fit)]))
    order = np.lexsort((np.arange(task.b.M), losses, cost))
    frontier = []
    for model in order:
        if not frontier or losses[model] < losses[frontier[-1]]:
            frontier.append(int(model))
    if max_stages < 1:
        raise ValueError("the chain needs at least one stage")
    if len(frontier) <= max_stages and len(frontier) <= 3:
        return frontier
    if max_stages == 1:
        return [frontier[0]]
    if max_stages == 2:
        return [frontier[0], frontier[-1]]
    return [frontier[0], frontier[len(frontier) // 2], frontier[-1]]


def request_confidence(task, active, model, choice, kind="lowest"):
    """Return the lowest (or the mean) calibrated top probability over each active request's fields."""
    if kind not in CONFIDENCE:
        raise ValueError("unknown confidence " + str(kind))
    b = task.b
    confidence = np.full(task.R, np.inf) if kind == "lowest" else np.zeros(task.R)
    for w, ids in b.idx.items():
        rows = b.field_row[ids]
        pos = np.flatnonzero(active[rows])
        top = transform(b.P[w][pos, model, :], *MENU[choice]).max(axis=1)
        if kind == "lowest":
            np.minimum.at(confidence, rows[pos], top)
        else:
            np.add.at(confidence, rows[pos], top)
    if kind == "mean":
        confidence /= np.maximum(np.bincount(b.field_row, minlength=task.R), 1)
    return confidence


def replay(task, fold, requests, chain, threshold, stage, models, confidence=None, kind="lowest"):
    """Fill in the called stages and stopping models of the given requests."""
    active = np.zeros(task.R, bool)
    active[requests] = True
    choices = task.recal(fold)[0]
    for depth, m in enumerate(chain):
        stage[active, m] = depth
        models[active] = m
        if depth == len(chain) - 1:
            break
        conf = request_confidence(task, active, m, choices[m], kind) if confidence is None else confidence[m]
        active &= conf < threshold
        if not active.any():
            break


def fit_rotation(task, fold, max_stages, grid, kind="lowest"):
    """Return the fit chain, the validation-selected threshold and the selection record."""
    b = task.b
    _, val, _ = roles(fold)
    valid = b.fold == val
    requests = np.flatnonzero(valid)
    chain = cost_chain(task, fold, max_stages)
    choices = task.recal(fold)[0]
    confidence = {m: request_confidence(task, valid, m, choices[m], kind) for m in chain[:-1]}
    costs, losses = [], []
    for threshold in grid:
        stage = np.full((task.R, b.M), NOT_CALLED, np.int16)
        models = np.empty(task.R, np.int32)
        replay(task, fold, requests, chain, threshold, stage, models, confidence)
        costs.append(math.fsum(b.cost[stage != NOT_CALLED]))
        losses.append(validation_loss(task, fold, models))
    budget = validation_budget(task, fold, requests)
    return chain, choose_parameter(grid, losses, costs, budget), selection_record(grid, losses, costs, budget)


def run(task, settings, *, threshold=None):
    """Return a cascade at the validation-selected threshold, a fixed one, or one per fold.

    ``settings[NAME]["confidence"]`` is "lowest" (the default: the lowest top probability among the request's
    fields) or "mean".
    """
    config = settings[NAME]
    grid = tuple(config["tau_grid"])
    kind = config.get("confidence", "lowest")
    cache = vars(task).setdefault("_cascade_fits", {})
    key = (config["max_stages"], grid, kind)
    if key not in cache:
        cache[key] = [fit_rotation(task, f, *key) for f in range(N_FOLDS)]
    stage = np.full((task.R, task.b.M), NOT_CALLED, np.int16)
    models = np.empty(task.R, np.int32)
    info = {"chain": [], "tau": [], "selection": []}
    for f, (chain, selected, record) in enumerate(cache[key]):
        value = per_fold(threshold, selected, f)
        replay(task, f, np.flatnonzero(task.b.fold == f), chain, value, stage, models, kind=kind)
        info["chain"].append(chain)
        info["tau"].append(value)
        info["selection"].append(record)
    choices = np.array([task.recal(f)[0] for f in range(N_FOLDS)])
    ledger = empty_ledger()
    ledger.update(labeled_folds="fit and validation", models_profiled=task.b.M)
    return Outcome(request_vectors(task, models, choices), stage, info, ledger)

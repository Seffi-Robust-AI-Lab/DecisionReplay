"""Shared scoring primitives: recalibration transforms, decoding, accepted mass, losses, cost of a call plan.

These are the rules of the stored results, in one place, so that the simulator, the baselines and the diagnostics
all score the same way.  Everything here is a pure function of arrays.
"""

from __future__ import annotations

import math

import numpy as np

from ..data.load import fsum_rows
from .types import ECE_BINS, LOG_FLOOR, NOT_CALLED


# --------------------------------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------------------------------

def transform(P, kind, par):
    """One recalibration transform on vectors with the options on the last axis."""
    if kind == "identity":
        return P
    if kind == "uniform":
        return (1.0 - par) * P + par / P.shape[-1]
    if kind != "temperature":
        raise ValueError(kind)
    powered = np.power(P, 1.0 / par)                 # zeros stay zero
    return powered / powered.sum(axis=-1, keepdims=True)


def temper(P, T):
    """p^(1/T), renormalised, for any positive temperature T."""
    return P if T == 1.0 else transform(P, "temperature", T)


def loss_of_mass(mass, floor=LOG_FLOOR):
    """Log loss of an accepted mass, with the mass clipped to [floor, 1]."""
    return -np.log(np.clip(mass, floor, 1.0))


# --------------------------------------------------------------------------------------------------
# decoding and scoring of pooled vectors
# --------------------------------------------------------------------------------------------------

def block_mass(bench, w, V, sel=None):
    """Accepted mass of vectors V [n, ..., w] for the fields ``idx[w][sel]`` (the whole block if ``sel`` is None).

    A yes/no field whose answer is "no" takes 1 - p(yes).
    """
    ids = bench.idx[w] if sel is None else bench.idx[w][sel]
    acc = bench.acc[w] if sel is None else bench.acc[w][sel]
    mass = np.einsum("n...o,no->n...", V, acc.astype(np.float64))
    no = bench.is_noul[ids] & (bench.gold[ids] == 0)
    if no.any():
        mass[no] = 1.0 - V[no][..., 1]
    return mass


def renormalise(v):
    """Divide every row of [n, w] by its exactly rounded sum."""
    return v / fsum_rows(v)[:, None]


def decode(bench, pooled):
    """Per-field outcome of pooled vectors {width: [n_w, w]}.

    Returns (hit [F] bool, mass [F], choice [F] int64, confidence [F]): renormalise, take the first maximum, on a
    yes/no field "yes" when p(yes) >= 0.5.  ``hit`` and ``mass`` are meaningful on judged fields only (hit is
    False and mass is 0 on the others).  ``confidence`` is the probability of the chosen option.
    """
    hit = np.zeros(bench.F, bool)
    mass = np.zeros(bench.F)
    choice = np.empty(bench.F, np.int64)
    confidence = np.empty(bench.F)
    for w in bench.widths:
        ids = bench.idx[w]
        p = renormalise(pooled[w])
        rows = np.arange(len(ids))
        win = p.argmax(axis=1)
        noul = bench.is_noul[ids]
        if noul.any():
            win[noul] = p[noul, 1] >= 0.5
        choice[ids] = win
        confidence[ids] = p[rows, win]
        if bench.has_labels:
            hit[ids] = bench.acc[w][rows, win]
            mass[ids] = block_mass(bench, w, p)
    return hit, mass, choice, confidence


def brier(bench, pooled):
    """Per-field Brier score [F] with the accepted options merged into one class (NaN on unjudged fields).

    For a field with one accepted option this is the usual multi-class Brier score, sum_o (p_o - y_o)^2.
    """
    out = np.full(bench.F, np.nan)
    for w in bench.widths:
        ids = bench.idx[w]
        p = renormalise(pooled[w])
        m = block_mass(bench, w, p)
        rest = np.where(bench.acc[w], 0.0, p * p).sum(axis=1)
        value = rest + (1.0 - m) ** 2
        j = bench.judged[ids]
        out[ids[j]] = value[j]
    return out


def ece(confidence, hit, bins=ECE_BINS):
    """Expected calibration error with equal-width bins: sum_b |hits_b - confidence_b| / n."""
    confidence = np.asarray(confidence, dtype=np.float64)
    hit = np.asarray(hit, dtype=np.float64)
    if len(confidence) == 0:
        return float("nan")
    b = np.minimum(bins - 1, (bins * confidence).astype(np.int64))
    gap = np.bincount(b, weights=hit, minlength=bins) - np.bincount(b, weights=confidence, minlength=bins)
    return float(np.abs(gap).sum() / len(confidence))


def reliability_table(confidence, hit, bins=ECE_BINS):
    """Per bin: count, mean confidence, accuracy (NaN for an empty bin)."""
    confidence = np.asarray(confidence, dtype=np.float64)
    hit = np.asarray(hit, dtype=np.float64)
    b = np.minimum(bins - 1, (bins * confidence).astype(np.int64))
    n = np.bincount(b, minlength=bins).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.bincount(b, weights=confidence, minlength=bins) / n
        acc = np.bincount(b, weights=hit, minlength=bins) / n
    return n.astype(np.int64), conf, acc


def single_model_vectors(bench, m):
    """Pooled-vector form {width: [n_w, w]} of one model's own normalised vectors."""
    return {w: bench.P[w][:, m, :] for w in bench.widths}


def reported_hits(bench, m):
    """[F] bool: is the option the provider reported accepted?  Yes/no fields use the 0.5 rule on p(yes)."""
    choice = bench.reported[:, m].astype(np.int64).copy()
    hit = np.zeros(bench.F, bool)
    for w in bench.widths:
        ids = bench.idx[w]
        c = choice[ids]
        noul = bench.is_noul[ids]
        if noul.any():
            c[noul] = bench.P[w][noul, m, 1] >= 0.5
        if (c < 0).any():
            raise ValueError("a choice field has no reported option")
        hit[ids] = bench.acc[w][np.arange(len(ids)), c]
    return hit


# --------------------------------------------------------------------------------------------------
# cost and latency of a call plan
# --------------------------------------------------------------------------------------------------

def plan_costs(bench, stage):
    """Cost and latency of a call plan ``stage`` [R, M] (NOT_CALLED or the stage index of each call).

    Cost and tokens are sums over the called model-request pairs.  The latency of a request is the sum over its
    stages of the slowest call of the stage; ``latency_one_stage`` puts every call in one stage and
    ``latency_serial`` one call per stage, for reference.
    """
    stage = np.asarray(stage)
    if stage.shape != (bench.R, bench.M):
        raise ValueError("stage must be [requests, models]")
    called = stage != NOT_CALLED
    R = bench.R
    n_calls = int(called.sum())
    tin, tout = int(bench.tin[called].sum()), int(bench.tout[called].sum())
    usd = math.fsum(bench.cost[called].tolist())
    out = dict(
        calls=n_calls, calls_per_request=n_calls / R,
        input_tokens_per_request=tin / R, output_tokens_per_request=tout / R,
        cost_usd=usd, usd_per_1k_requests=1000 * usd / R,
        stages_mean=float((stage.max(axis=1) + 1).mean()),
    )
    if bench.lat is not None:
        total = np.zeros(R)
        for s in range(int(stage.max()) + 1):
            total += np.where(stage == s, bench.lat, 0.0).max(axis=1)
        one = np.where(called, bench.lat, 0.0).max(axis=1)
        serial = np.where(called, bench.lat, 0.0).sum(axis=1)
        out.update(
            latency_mean_s=float(total.mean()), latency_p50_s=float(np.percentile(total, 50)),
            latency_p95_s=float(np.percentile(total, 95)),
            latency_one_stage_mean_s=float(one.mean()), latency_one_stage_p50_s=float(np.percentile(one, 50)),
            latency_one_stage_p95_s=float(np.percentile(one, 95)),
            latency_serial_mean_s=float(serial.mean()), latency_serial_p50_s=float(np.percentile(serial, 50)),
            latency_serial_p95_s=float(np.percentile(serial, 95)),
        )
    return out

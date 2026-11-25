"""Frozen interfaces shared by the loader, the simulator, the baselines and the analyses.

Vocabulary.  A *request* is one call to a model; it holds one or more *fields* (typed questions).  A model returns
one probability vector per field.  Routing, cascading and early exit decide per request; scoring is per judged
field; resampling is per split group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

LOG_FLOOR = 1e-15                       # stored rule: accepted mass is clipped to [LOG_FLOOR, 1] before the log
REPORT_FLOORS = (1e-15, 1e-6, 1e-3)     # log loss is reported at these floors
ECE_BINS = 10
NOT_CALLED = -1

# Recalibration menu of the stored results, in this order (the order decides ties).
MENU = (
    ("identity", 1.0),
    ("temperature", 0.5), ("temperature", 0.75), ("temperature", 1.25), ("temperature", 1.5), ("temperature", 2.0),
    ("uniform", 0.01), ("uniform", 0.05), ("uniform", 0.10), ("uniform", 0.25),
)


@dataclass
class BenchmarkData:
    """All arrays of one benchmark.  Requests are in ``item_index`` order, fields in (request, field) order.

    Vectors are grouped by option count ("width"), so no padding is needed: for a width ``w``, ``idx[w]`` holds the
    field indexes with that many options and ``P[w][k]`` is the [M, w] block of field ``idx[w][k]``.
    """

    cid: int
    name: str
    area: str
    R: int                               # requests
    F: int                               # fields
    M: int                               # models
    fold: np.ndarray                     # [R] int8, frozen fold of the request
    split_group: np.ndarray              # [R] int64, code of the source split group (unit of resampling)
    run_rank: np.ndarray                 # [R] int32
    field_row: np.ndarray                # [F] int32, request of the field
    field_pos: np.ndarray                # [F] int32, position of the field inside its request
    width: np.ndarray                    # [F] int16, number of options
    is_noul: np.ndarray                  # [F] bool, yes/no field (vector is [1 - p, p])
    pos_in_w: np.ndarray                 # [F] int32, position of the field inside idx[width]
    widths: list                         # sorted distinct widths
    idx: dict                            # width -> [n_w] int32 field indexes
    P: dict                              # width -> [n_w, M, w] float64, normalised vectors
    top: np.ndarray                      # [F, M] float64, largest normalised probability
    reported: np.ndarray                 # [F, M] int32, option the provider reported; -1 on yes/no fields
    # labels: None when the corpus has no labels table
    judged: Optional[np.ndarray] = None  # [F] bool
    gold: Optional[np.ndarray] = None    # [F] int16, the accepted option when there is exactly one, else -1
    n_accepted: Optional[np.ndarray] = None   # [F] int16
    acc: Optional[dict] = None           # width -> [n_w, w] bool, accepted options
    # usage and latency per request and model: None when the tier does not hold them
    calls: Optional[np.ndarray] = None   # [R, M] int32
    tin: Optional[np.ndarray] = None     # [R, M] int64 input tokens
    tout: Optional[np.ndarray] = None    # [R, M] int64 output tokens
    cost: Optional[np.ndarray] = None    # [R, M] float64 USD
    lat: Optional[np.ndarray] = None     # [R, M] float64 seconds
    raw: Optional[dict] = None           # width -> [n_w, M, w] vectors as stored, when asked for
    models: list = field(default_factory=list)    # rows of models.csv as dicts
    item_id: Optional[list] = None       # [R] opaque request ids, by which benchmark rows are bound to the requests

    @property
    def has_labels(self):
        return self.judged is not None


@dataclass
class Outcome:
    """What a policy did on one benchmark, every request handled by the rotation that holds its fold out.

    ``pooled[w][k]`` is the final probability vector of field ``idx[w][k]`` (it need not sum to one; the scorer
    renormalises).  ``stage[r, m]`` is NOT_CALLED, or the stage (0, 1, ...) in which model ``m`` was called for
    request ``r``.  Models of one stage are called in parallel, stages run one after another.
    """

    pooled: dict
    stage: np.ndarray                    # [R, M] int16
    info: dict = field(default_factory=dict)      # JSON-ready: chosen settings per fold, weights, panel sizes
    ledger: dict = field(default_factory=dict)    # what the policy saw of the target benchmark before testing


def empty_ledger():
    """Ledger of a policy that saw nothing of the target benchmark."""
    return dict(labeled_folds="none", unlabeled_outputs="none", models_profiled=0, other_benchmark_labels=False,
                note="")


# One row of results/*.json "rows": a method on one benchmark (or on a suite, with catalog_id = null).
RESULT_ROW_KEYS = (
    "method", "variant", "suite", "catalog_id", "n_benchmarks", "n_requests", "n_judged",
    "log_loss", "log_loss_floor_1e-6", "log_loss_floor_1e-3", "n_clipped",
    "accuracy", "brier", "ece", "native",
    "calls_per_request", "input_tokens_per_request", "output_tokens_per_request", "usd_per_1k_requests",
    "latency_mean_s", "latency_p50_s", "latency_p95_s", "stages_mean",
    "ledger", "settings_sha256",
)
# One row of results/*.json "contrasts": A - B on a suite.
CONTRAST_ROW_KEYS = (
    "contrast", "a", "b", "suite", "metric", "difference", "ci_low", "ci_high", "p", "p_holm",
    "benchmarks_better", "benchmarks_same", "benchmarks_worse", "n_benchmarks", "draws", "settings_sha256",
)

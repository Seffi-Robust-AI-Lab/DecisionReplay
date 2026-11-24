"""Judged-field scores and equal-benchmark summaries of offline policies."""

from dataclasses import dataclass
import math

import numpy as np

from . import core
from .types import REPORT_FLOORS, RESULT_ROW_KEYS


@dataclass
class Scored:
    """Field scores in judged-field order with the cost and information of the policy."""

    cid: int
    mass: np.ndarray
    hit: np.ndarray
    brier: np.ndarray
    confidence: np.ndarray
    cost: dict
    info: dict
    ledger: dict


def score(task, outcome):
    """Return judged-field scores and the cost of the policy's actual call plan."""
    hit, mass, _, confidence = core.decode(task.b, outcome.pooled)
    brier = core.brier(task.b, outcome.pooled)
    return Scored(task.cid, mass[task.J], hit[task.J], brier[task.J], confidence[task.J],
                  core.plan_costs(task.b, outcome.stage), outcome.info, outcome.ledger)


def summarize(scored):
    """Return the metric columns of one benchmark row."""
    mass = scored.mass
    row = {
        "n_clipped": int((mass < REPORT_FLOORS[0]).sum()),
        "accuracy": float(scored.hit.mean()) if len(mass) else None,
        "brier": float(scored.brier.mean()) if len(mass) else None,
        "ece": core.ece(scored.confidence, scored.hit) if len(mass) else None,
        "native": None,
    }
    for key, floor in zip(("log_loss", "log_loss_floor_1e-6", "log_loss_floor_1e-3"), REPORT_FLOORS):
        row[key] = float(core.loss_of_mass(mass, floor).mean()) if len(mass) else None
    for key in ("calls_per_request", "input_tokens_per_request", "output_tokens_per_request",
                "usd_per_1k_requests", "latency_mean_s", "latency_p50_s", "latency_p95_s", "stages_mean"):
        row[key] = scored.cost.get(key)
    return row


def suite_rows(rows_by_cid, suites):
    """Return one row per nonempty suite of a method and variant, averaging benchmarks equally."""
    result = []
    for suite, spec in suites.items():
        cids = sorted(set(rows_by_cid).difference(spec["exclude"]))
        if not cids:
            continue
        rows = [rows_by_cid[cid] for cid in cids]
        row = dict(rows[0], catalog_id=None, suite=suite, n_benchmarks=len(cids))
        for key in RESULT_ROW_KEYS:
            if key not in row:
                continue
            if key in ("n_judged", "n_requests"):
                row[key] = sum(r[key] for r in rows)
            elif key == "n_clipped":
                total = math.fsum(r[key] for r in rows)
                row[key] = int(total) if total.is_integer() else total
            elif key == "ledger" and any(r[key] != rows[0][key] for r in rows):
                row[key] = {str(cid): r[key] for cid, r in zip(cids, rows)}
            elif key not in ("method", "variant", "suite", "catalog_id", "n_benchmarks", "ledger", "settings_sha256"):
                values = [r[key] for r in rows]
                row[key] = None if None in values else math.fsum(values) / len(cids)
        result.append(row)
    return result

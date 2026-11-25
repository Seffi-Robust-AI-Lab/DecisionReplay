"""Replay policies indexed by their settings names."""

from . import (
    best_single, certified_exit, confidence_cascade, greedy, knn_router, majority_vote,
    oracle, random_single, skill_pool, stack, stack_final_temperature, uniform_mean,
    uniform_temperature,
)

BASELINES = {module.NAME: module for module in (
    best_single, oracle, random_single, uniform_mean, majority_vote, skill_pool,
    uniform_temperature, stack, stack_final_temperature, greedy, knn_router,
    confidence_cascade, certified_exit,
)}

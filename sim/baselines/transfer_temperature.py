"""Final flat-pool temperature selected using other task areas."""

from data.schema import N_FOLDS
from .transfer_single import equal_mean, run_held_out_areas
from .uniform_mean import equal_weights, panel
from .uniform_temperature import best_temperature, flat_tables, menu

NAME = "transfer_temperature"


def source_loss(task, settings):
    """Return whole-benchmark mean flat-pool log loss for each temperature."""
    _, sums, counts = flat_tables(task, settings)
    return sums.sum(axis=1) / counts.sum()


def choose(profiles, settings):
    """Return the temperature minimising equal-benchmark mean source loss."""
    candidates = menu(settings)
    return candidates[best_temperature(equal_mean(profiles), candidates)]


def deploy(task, choice):
    """Return the flat target pool tempered with the source-selected temperature."""
    return panel(task, equal_weights(task), temperatures=[choice] * N_FOLDS,
                 ledger={"other_benchmark_labels": True})


def run_suite(tasks, settings):
    """Return transferred flat-pool outcomes by catalogue id."""
    return run_held_out_areas(tasks, settings, source_loss, choose, deploy)

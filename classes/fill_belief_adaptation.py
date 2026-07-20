"""Production-named API for online fill-belief adaptation.

Thin re-export of :mod:`fill_rl` under stable names; aliases are
identity-equivalent to the validated implementation.
"""

from .fill_rl import (
    RLFillMM,
    bce_score,
    brier_score,
    calibration_table,
    delta_ticks_from_X,
    depth_calibration_table,
    resolve_vol_use_realized,
    run_fill_rl_multi_sim,
    run_fill_rl_sim,
    update_fill_nn,
)

FillBeliefLoggingMM = RLFillMM
run_single_agent_adaptation_sim = run_fill_rl_sim
run_population_adaptation_sim = run_fill_rl_multi_sim
update_fill_belief_nn = update_fill_nn

__all__ = [
    "FillBeliefLoggingMM",
    "run_single_agent_adaptation_sim",
    "run_population_adaptation_sim",
    "update_fill_belief_nn",
    "resolve_vol_use_realized",
    "calibration_table",
    "depth_calibration_table",
    "bce_score",
    "brier_score",
    "delta_ticks_from_X",
]

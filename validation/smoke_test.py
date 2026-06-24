"""End-to-end smoke test for the calibration stack.

Run it with::

    python -m research_core.validation.smoke_test

It builds a tiny synthetic two-dimension event set and then exercises the
production calibration path: Poisson, univariate single-exponential, and
multivariate single-exponential (in-process, ``n_workers=1``). The point is to
catch import breakage and obvious regressions in a few seconds, so it never
touches the WSE HDF5 files or the SQLite databases. It is not a statistical
test. The data is random and the trial budget is tiny.
"""

from __future__ import annotations

import matplotlib

# Headless backend: the smoke test should run on CI and over SSH without a
# display. Set before anything imports pyplot.
matplotlib.use("Agg")

import numpy as np
import optuna

from research_core.classes import HawkesCalibration

MARKS = ["MO_bid", "MO_ask"]


def _synthetic_events(n_days: int = 2, per_minute: float = 5.0,
                      horizon: float = 600.0, seed: int = 0):
    """Poisson-scattered event times for a handful of short synthetic days."""
    rng = np.random.default_rng(seed)
    expected = per_minute * horizon / 60.0
    days = []
    for _ in range(n_days):
        day = []
        for _ in MARKS:
            n = int(rng.poisson(expected))
            times = np.sort(rng.uniform(0.0, horizon, size=max(n, 1)))
            day.append(np.ascontiguousarray(times, dtype=np.float64))
        days.append(day)
    end_times = np.full(n_days, horizon, dtype=np.float64)
    return days, end_times


def main() -> int:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    events, end_times = _synthetic_events()
    cal = HawkesCalibration(events, MARKS, end_times)

    poisson = cal.fit_poisson()
    assert np.isfinite(poisson["total_ll"]), "Poisson total_ll is not finite"

    univariate = cal.fit_univariate_hawkes(n_trials=5, beta_max=10.0)
    assert np.isfinite(univariate["per_event_ll"]), \
        "univariate per-event score is not finite"

    multivariate = cal.fit_multivariate_hawkes(
        n_trials=5, n_workers=1, beta_max=10.0,
    )
    branching = multivariate["branching_ratio"]
    assert 0.0 <= branching < float("inf"), \
        f"branching ratio out of range: {branching}"

    print("\nsmoke_test OK: Poisson, univariate and multivariate "
          "single-exponential fits all ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

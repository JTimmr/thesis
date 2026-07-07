"""Hawkes process calibration for WSE order flow.

The production model is the single-exponential multivariate Hawkes process.
Each kernel is ``φ_ij(t) = α_ij β_ij exp(−β_ij t)``, fitted by maximum
likelihood: Optuna searches the decay rates ``β_ij`` and the ``tick`` learner
returns the baseline ``μ`` and adjacency ``α``. The same code path fits the
Poisson baseline and the per-dimension univariate Hawkes used as sanity checks.

Calibration runs in two time domains. Raw clock time is the default. The other
is seasonality-adjusted ``τ``-time, where a shared intraday profile is
integrated to ``τ(t) = ∫₀ᵗ s̄(u) du`` so deterministic time-of-day variation is
removed before fitting.

Goodness of fit uses the time-rescaling theorem. The fitted compensator turns
each dimension's events into interarrivals that should look Exponential(1) when
the model is right.

``tick`` is a hard dependency (see ``pyproject.toml``). It pins to older Python
on Windows, so calibration usually runs in its own environment.
"""

from __future__ import annotations

import os
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import optuna
from tick.hawkes import HawkesExpKern

from research_core.classes.parallelisation import run_parallel_optuna


# --- Seasonality helpers ---

def get_average_seasonality_shape(seasonality_profiles: dict, marks_order: List[str], normalize: bool = True) -> dict:
    """Average the per-dimension intraday intensity profiles into one shared shape.

    Each dimension has a (grid, profile) tuple from estimate_seasonality_profiles.
    This function normalizes each profile to mean=1, stacks them, and computes the
    pointwise average and std. The cumulative integral (trapezoidal) is used later
    to map clock time -> tau-time.

    Returns dict with keys: grid, profile, std, n_patterns, cumulative_integral.
    """
    grid = np.array(seasonality_profiles[marks_order[0]][0])
    profiles = []
    for dim_name in marks_order:
        raw_profile = np.array(seasonality_profiles[dim_name][1])
        if normalize:
            raw_profile = raw_profile / raw_profile.mean()
        profiles.append(raw_profile)

    profiles_matrix = np.array(profiles)
    avg_profile = profiles_matrix.mean(axis=0)
    std_profile = profiles_matrix.std(axis=0)

    # Trapezoidal cumulative integral: used to map clock-time -> tau-time
    midpoints = (avg_profile[:-1] + avg_profile[1:]) / 2.0
    cum_integral = np.concatenate([[0.0], np.cumsum(np.diff(grid) * midpoints)])

    return {
        "grid": grid,
        "profile": avg_profile,
        "std": std_profile,
        "n_patterns": len(profiles),
        "cumulative_integral": cum_integral,
    }


def clock_to_tau(t, *, grid: np.ndarray, cumulative_integral: np.ndarray) -> np.ndarray:
    """Convert clock-time (seconds since open) to tau-time via linear interpolation.

    Clips input to [0, end_of_day] then looks up the pre-computed cumulative
    seasonality integral at that clock-time. The result is a deseasonalised
    time axis where 1 unit of tau corresponds to 1 unit of average-intensity time.
    """
    t = np.atleast_1d(t)
    t = np.clip(t, 0.0, grid[-1])
    return np.interp(t, grid, cumulative_integral)


def format_mean_with_ci(values, decimal_format: str = ".6f") -> str:
    """Format a sample mean with its 95% t-distribution confidence interval.

    Example output: '0.123456  95% CI [0.120000, 0.126912] (+/-2.4%)'
    Returns just the mean if fewer than 2 finite values are available.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)

    if n == 0:
        return f"{np.nan:{decimal_format}}"
    mean_value = values.mean()
    if n < 2:
        return f"{mean_value:{decimal_format}}"

    standard_error = values.std(ddof=1) / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, df=n - 1)
    ci_half_width = t_critical * standard_error
    lower = mean_value - ci_half_width
    upper = mean_value + ci_half_width

    if mean_value != 0:
        relative_pct = ci_half_width / abs(mean_value) * 100
    else:
        relative_pct = float("inf")

    return f"{mean_value:{decimal_format}}  95% CI [{lower:{decimal_format}}, {upper:{decimal_format}}] (+/-{relative_pct:.1f}%)"


# --- Optuna objectives ---

class UnivariateHawkesObjective:
    """Optuna objective for univariate (self-exciting) Hawkes calibration."""

    def __init__(self, beta_min, beta_max, max_iter, tol, events, end_times):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.max_iter = max_iter
        self.tol = tol
        self.events = events
        self.end_times = end_times

    def __call__(self, trial):
        beta = trial.suggest_float("beta", self.beta_min, self.beta_max, log=True)
        decays = np.array([[beta]])
        model = HawkesExpKern(decays=decays, max_iter=self.max_iter, tol=self.tol)
        model.fit(self.events)
        alpha = float(model.adjacency[0, 0])
        if alpha >= 1.0:
            return -np.inf
        score = model.score(events=self.events, end_times=self.end_times)
        return float(score)


class MultivariateHawkesObjective:
    """Optuna objective for multivariate Hawkes calibration (single-exp)."""

    def __init__(self, beta_min, beta_max, n_nodes, marks_order, max_iter, tol, events, end_times):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_nodes = n_nodes
        self.marks_order = marks_order
        self.max_iter = max_iter
        self.tol = tol
        self.events = events
        self.end_times = end_times

    def build_decay_matrix(self, trial):
        decays = np.zeros((self.n_nodes, self.n_nodes))
        for i, di in enumerate(self.marks_order):
            for j, dj in enumerate(self.marks_order):
                decays[i, j] = trial.suggest_float(
                    f"beta_{di}__{dj}", self.beta_min, self.beta_max, log=True,
                )
        return decays

    def __call__(self, trial):
        decays_matrix = self.build_decay_matrix(trial)
        model = HawkesExpKern(decays=decays_matrix, max_iter=self.max_iter, tol=self.tol)
        model.fit(self.events)
        score = model.score(events=self.events, end_times=self.end_times)
        adjacency = model.adjacency
        branching_ratio = max(np.linalg.eigvals(adjacency).real)
        if branching_ratio >= 1.0:
            return -np.inf
        return float(score)


# --- Goodness-of-fit utilities ---

def plot_time_rescaling_cdf(interarrival_times, title: str):
    """Goodness-of-fit plot: compare model-implied interarrivals against Exp(1).

    If the Hawkes model is correct, the compensator transforms event times into
    interarrivals that should be independent Exponential(1) random variables.
    This plot shows their empirical CDF against the theoretical 45-degree line
    (uniform on [0,1] after probability transform) with a KS confidence band.

    A model that fits well produces a CDF that hugs the diagonal inside the
    gray band. Departures mean the intensity model is misspecified.
    """
    interarrival_times = np.asarray(interarrival_times)
    interarrival_times = interarrival_times[interarrival_times > 0]

    cumulative = np.cumsum(interarrival_times)
    n = len(cumulative)
    if n > 0:
        T = float(cumulative[-1])
    else:
        T = 0.0

    x = np.hstack([0.0, np.repeat(cumulative, 2), T])
    y = np.repeat(np.arange(n + 1), 2) / n

    ks_band = 1.36 / np.sqrt(n)

    plt.figure(figsize=(6, 5), dpi=100)
    plt.plot(x, y, "k-", label="Data")
    plt.fill_between(
        [0, T * ks_band, T * (1 - ks_band), T],
        [0, 0, 1 - 2 * ks_band, 1 - ks_band],
        [ks_band, 2 * ks_band, 1, 1],
        color="lightgray",
        label="95% confidence interval",
    )
    plt.xlim([0, T])
    plt.ylim([0, 1])
    plt.ylabel("Cumulative distribution function")
    plt.xlabel("Transformed time")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_all_seasonality_patterns(seasonality_profiles, marks_order, figsize=(12, 7), show_average=True, show_uncertainty=True, normalize=True, print_stats=True):
    """Plot all intraday seasonality patterns on the same figure.

    Parameters
    ----------
    seasonality_profiles : dict
        Mapping dim_name -> (grid, mean_profile, ...).
    marks_order : list
        Order of dimensions to plot.
    figsize, show_average, show_uncertainty, normalize, print_stats :
        Display options.

    Returns
    -------
    fig, ax, stats_results
    """
    colors = {
        "MO_bid": "firebrick",
        "MO_ask": "indianred",
        "LO_bid": "steelblue",
        "LO_ask": "royalblue",
        "CXL_bid": "seagreen",
        "CXL_ask": "forestgreen",
    }
    linestyles = {
        "MO_bid": "-",
        "MO_ask": "--",
        "LO_bid": "-",
        "LO_ask": "--",
        "CXL_bid": "-",
        "CXL_ask": "--",
    }

    fig, ax = plt.subplots(figsize=figsize)

    common_grid = np.array(seasonality_profiles[marks_order[0]][0])
    all_profiles_normalized = []
    profile_names = []

    for dim_name in marks_order:
        data = seasonality_profiles[dim_name]
        grid, profile = np.array(data[0]), np.array(data[1])

        grid_hours = grid / 3600.0
        profile_names.append(dim_name)

        profile_norm = profile / profile.mean()
        all_profiles_normalized.append(profile_norm)

        color = colors.get(dim_name, "gray")
        ls = linestyles.get(dim_name, "-")

        if normalize:
            plot_profile = profile_norm
        else:
            plot_profile = profile
        ax.plot(grid_hours, plot_profile, label=dim_name, color=color, linestyle=ls, linewidth=1.5, alpha=0.7)

    profiles_matrix = np.array(all_profiles_normalized)
    grid_hours = common_grid / 3600.0
    avg_profile = profiles_matrix.mean(axis=0)
    std_profile = profiles_matrix.std(axis=0)

    if show_average:
        ax.plot(grid_hours, avg_profile, label="Average", color="black", linestyle="-", linewidth=2.5, zorder=10)
    if show_uncertainty:
        ax.fill_between(grid_hours, avg_profile - std_profile, avg_profile + std_profile, color="gray", alpha=0.3, label="±1 std", zorder=5)

    if normalize:
        ylabel = "Normalized intensity (mean=1)"
    else:
        ylabel = "Event intensity (events/second)"
    ax.set_xlabel("Time since market open (hours)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Intraday Seasonality Patterns by Event Type", fontsize=13)
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.1)

    stats_results: dict = {
        "correlations": {},
        "rmse_to_average": {},
        "pairwise_correlations": {},
    }

    for i, name in enumerate(profile_names):
        corr, p_value = stats.pearsonr(profiles_matrix[i], avg_profile)
        stats_results["correlations"][name] = {
            "correlation": corr, "p_value": p_value,
        }
        rmse = float(np.sqrt(np.mean((profiles_matrix[i] - avg_profile) ** 2)))
        stats_results["rmse_to_average"][name] = rmse

    n_patterns = len(profile_names)
    for i in range(n_patterns):
        for j in range(i + 1, n_patterns):
            corr, p_value = stats.pearsonr(
                profiles_matrix[i], profiles_matrix[j],
            )
            pair_name = f"{profile_names[i]} vs {profile_names[j]}"
            stats_results["pairwise_correlations"][pair_name] = {
                "correlation": corr, "p_value": p_value,
            }

    all_corrs = [v["correlation"] for v in stats_results["correlations"].values()]
    all_pairwise_corrs = [v["correlation"] for v in stats_results["pairwise_correlations"].values()]
    stats_results["summary"] = {
        "mean_corr_with_average": float(np.mean(all_corrs)),
        "min_corr_with_average": float(np.min(all_corrs)),
        "mean_pairwise_corr": float(np.mean(all_pairwise_corrs)),
        "min_pairwise_corr": float(np.min(all_pairwise_corrs)),
        "mean_rmse_to_average": float(
            np.mean(list(stats_results["rmse_to_average"].values()))
        ),
    }

    friedman_stat, friedman_p = stats.friedmanchisquare(*profiles_matrix)
    if friedman_p > 0.05:
        friedman_interpretation = "Patterns are similar (can use average)"
    else:
        friedman_interpretation = "Patterns differ significantly"
    stats_results["friedman_test"] = {
        "statistic": friedman_stat,
        "p_value": friedman_p,
        "interpretation": friedman_interpretation,
    }

    if print_stats:
        print("\n1. Correlation with average pattern:")
        print(
            f"   {'Event Type':<12} {'Correlation':>12} "
            f"{'p-value':>12} {'RMSE':>10}"
        )
        print("   " + "-" * 48)
        for name in profile_names:
            corr = stats_results["correlations"][name]["correlation"]
            p = stats_results["correlations"][name]["p_value"]
            rmse = stats_results["rmse_to_average"][name]
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                sig = ""
            print(f"   {name:<12} {corr:>12.4f} {p:>11.2e}{sig} {rmse:>10.4f}")

        print("\n2. Summary statistics:")
        s = stats_results["summary"]
        print(f"   Mean correlation with average: {s['mean_corr_with_average']:.4f}")
        print(f"   Min correlation with average:  {s['min_corr_with_average']:.4f}")
        print(f"   Mean pairwise correlation:     {s['mean_pairwise_corr']:.4f}")
        print(f"   Min pairwise correlation:      {s['min_pairwise_corr']:.4f}")

    plt.tight_layout()

    stats_results["average_shape"] = {
        "grid_seconds": common_grid,
        "grid_hours": grid_hours,
        "profile": avg_profile,
        "std": std_profile,
        "normalized": normalize,
    }

    return fig, ax, stats_results


def compensator_interarrivals_single(day_sequences: list, decays: np.ndarray, adjacency: np.ndarray, baseline: np.ndarray) -> list:
    """Compute compensator increments between events for goodness-of-fit testing.

    The "time-rescaling theorem" says: if you have the true conditional intensity
    lambda(t), then the integral of lambda between consecutive events (called the
    compensator increment) should be Exp(1)-distributed. This function computes
    those increments for a fitted single-exponential Hawkes process.

    The algorithm walks through all events in chronological order, maintaining the
    kernel state (sum of decaying contributions from past events). Between events,
    it integrates the intensity analytically (closed-form for exponential kernels)
    to get the compensator increment for each dimension.

    Parameters
    ----------
    day_sequences : list of arrays, one per dimension (single day).
        day_sequences[k] is a sorted array of event times for dimension k.
    decays : (n_nodes, n_nodes) matrix of decay rates beta_ij.
    adjacency : (n_nodes, n_nodes) matrix of excitation weights alpha_ij.
    baseline : (n_nodes,) vector of baseline intensities mu_i.

    Returns
    -------
    list of arrays, one per dimension. Each array contains the compensator
    increments between consecutive events in that dimension. Under a correctly
    specified model, each array should look like i.i.d. Exp(1) samples.
    """
    n_nodes = len(day_sequences)

    if day_sequences:
        all_times = np.concatenate(day_sequences)
    else:
        all_times = np.array([])
    if len(all_times) == 0:
        return [np.array([]) for _ in range(n_nodes)]

    # Tag each event with its dimension index, then sort everything by time
    marks = np.concatenate([
        np.full(len(seq), idx, dtype=np.int64)
        for idx, seq in enumerate(day_sequences)
    ])
    order = np.argsort(all_times)
    sorted_t = all_times[order]
    sorted_m = marks[order]

    # kernel_state[i,j] tracks the decaying contribution from past events in
    # dimension j to dimension i's intensity
    kernel_state = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    # comp[i] is the running compensator (cumulative integrated intensity) for dim i
    compensator = np.zeros(n_nodes, dtype=np.float64)
    # last_comp[i] is the compensator value at the previous event in dim i
    compensator_at_last_event = np.zeros(n_nodes, dtype=np.float64)
    increments: List[list] = [[] for _ in range(n_nodes)]

    prev_t = 0.0
    idx = 0
    while idx < len(sorted_t):
        cur_t = sorted_t[idx]
        dt = cur_t - prev_t

        # Phase 1: Integrate intensity from prev_t to cur_t (closed-form for exp kernels)
        if dt > 0:
            compensator += baseline * dt
            decay_factor = np.exp(-decays * dt)
            compensator += (kernel_state * (1.0 - decay_factor) / decays).sum(axis=1)
            kernel_state *= decay_factor

        # Phase 2: Record compensator increments for all events at this timestamp
        start = idx
        while start < len(sorted_t) and sorted_t[start] == cur_t:
            dim_idx = sorted_m[start]
            increments[dim_idx].append(compensator[dim_idx] - compensator_at_last_event[dim_idx])
            compensator_at_last_event[dim_idx] = compensator[dim_idx]
            start += 1

        # Phase 3: Apply excitation jumps from these events to the kernel state
        while idx < start:
            src = sorted_m[idx]
            kernel_state[:, src] += adjacency[:, src] * decays[:, src]
            idx += 1

        prev_t = cur_t

    return [np.array(inc) for inc in increments]


# --- Main class ---

class HawkesCalibration:
    """Unified calibration for Poisson / Hawkes processes.

    Parameters
    ----------
    timestamps_by_day : list[list[np.ndarray]]
        Outer list = days (realizations).  Inner list = dimensions (one array
        per mark).  timestamps_by_day[d][k] is a 1-D float64 array of event
        times (seconds since market open) for dimension k on day d.
    marks_order : list[str]
        Ordered dimension names, e.g. ['MO_bid', 'MO_ask', ...].
    end_times : np.ndarray, shape (n_days,)
        Observation end time for each day (seconds since market open).
    seasonality_profiles : dict, optional
        Mapping dim_name → (grid, profile, ...) for intraday seasonality.
    max_iter : int
        Maximum EM iterations for tick learners (default 1_000_000).
    tol : float
        Convergence tolerance for tick learners (default 1e-9).
    """

    def __init__(self, timestamps_by_day: List[List[np.ndarray]], marks_order: List[str], end_times: np.ndarray, seasonality_profiles: Optional[dict] = None, max_iter: int = 1_000_000, tol: float = 1e-9):
        self.marks_order = list(marks_order)
        self.n_nodes = len(self.marks_order)
        self.max_iter = max_iter
        self.tol = tol

        self.timestamps_by_day = [
            [
                np.ascontiguousarray(day_seq[k], dtype=np.float64)
                for k in range(self.n_nodes)
            ]
            for day_seq in timestamps_by_day
        ]
        self.end_times = np.ascontiguousarray(end_times, dtype=np.float64)
        self.n_days = len(self.timestamps_by_day)
        assert self.n_days == len(self.end_times), \
            "timestamps_by_day and end_times must have the same length"

        self.seasonality_profiles = seasonality_profiles
        self._tau_events: Optional[List[List[np.ndarray]]] = None
        self._tau_end_times: Optional[np.ndarray] = None
        self._transform_time = None
        self._avg_seasonality: Optional[dict] = None

        if seasonality_profiles is not None:
            self.build_tau()

    # --- τ-time construction ---

    def build_tau(self):
        """Build tau-time events and end times from seasonality profiles.

        Tau-time removes deterministic intraday variation: tau(t) is the integral
        from 0 to t of the average seasonality profile. Event times are mapped
        through that transform before Hawkes fitting.
        """
        avg_data = get_average_seasonality_shape(self.seasonality_profiles, self.marks_order, normalize=True)
        transform_fn = partial(clock_to_tau, grid=avg_data["grid"], cumulative_integral=avg_data["cumulative_integral"])
        self._transform_time = transform_fn
        self._avg_seasonality = avg_data

        tau_events = []
        for day_seq in self.timestamps_by_day:
            day_tau = []
            for k in range(self.n_nodes):
                seq = day_seq[k]
                if len(seq) > 0:
                    tau_seq = transform_fn(seq)
                else:
                    tau_seq = np.array([], dtype=np.float64)
                day_tau.append(np.ascontiguousarray(tau_seq, dtype=np.float64))
            tau_events.append(day_tau)

        tau_end = np.array([
            float(transform_fn(np.array([T]))[0])
            for T in self.end_times
        ], dtype=np.float64)

        self._tau_events = tau_events
        self._tau_end_times = tau_end

    # --- Helper: pick the right events / end_times ---

    def resolve_events(self, use_tau: bool):
        """Return (events, end_times) for the requested time domain."""
        if use_tau:
            if self._tau_events is None:
                self.build_tau()
            return self._tau_events, self._tau_end_times
        return self.timestamps_by_day, self.end_times

    # --- GOF helper ---

    def _plot_gof_for_dims(self, gof_dims, events, adjacency, baseline, decays, label):
        """For each requested dimension, compute compensator increments and plot the GOF CDF."""
        for dim_name in gof_dims:
            dim_idx = self.marks_order.index(dim_name)
            compensator_segments = []
            for day_seq in events:
                if len(day_seq[dim_idx]) == 0:
                    continue
                day_compensator = compensator_interarrivals_single(
                    day_seq, decays, adjacency, baseline,
                )[dim_idx]
                if len(day_compensator) > 0:
                    compensator_segments.append(day_compensator)
            if compensator_segments:
                compensator_all = np.concatenate(compensator_segments)
            else:
                compensator_all = np.array([])
            if len(compensator_all) > 0:
                plot_time_rescaling_cdf(compensator_all, f"{label}: {dim_name}")

    # --- 1. Poisson calibration ---

    def fit_poisson(self, day_keys: List[str], use_tau: bool = False, gof_dims: List[str] = []) -> dict:
        """Fit independent homogeneous Poisson processes.

        Parameters
        ----------
        use_tau : bool
            If True, use seasonality-adjusted τ-time.
        day_keys : list[str]
            Labels for each day (for per-day reporting).
        gof_dims : list[str]
            Dimensions for which GOF plots are produced.

        Returns
        -------
        dict with keys:
            pooled_df    : pd.DataFrame with pooled MLE results
            daily_df     : pd.DataFrame with per-day MLE results
            total_ll     : float, total log-likelihood
            per_event_ll : float, per-event log-likelihood
        """
        events, end_times = self.resolve_events(use_tau)

        if len(end_times):
            T_total = float(np.sum(end_times))
        else:
            T_total = 0.0

        counts = np.array([
            sum(len(day_seq[k]) for day_seq in events)
            for k in range(self.n_nodes)
        ], dtype=float)

        if T_total > 0:
            mu = counts / T_total
        else:
            mu = np.zeros_like(counts)
        se_mu = np.sqrt(mu / T_total)
        mu_ci_lo = mu - 1.96 * se_mu
        mu_ci_hi = mu + 1.96 * se_mu

        ll = np.where(mu > 0, counts * np.log(mu) - mu * T_total, -np.inf)
        per_ev = np.where(counts > 0, ll / counts, np.nan)

        pooled_df = pd.DataFrame({
            "dim": self.marks_order,
            "mu": mu,
            "mu_ci_lower": mu_ci_lo,
            "mu_ci_upper": mu_ci_hi,
            "n_events": counts.astype(int),
            "log_likelihood": ll,
            "per_event": per_ev,
        })

        total_ll = float(np.nansum(ll))
        total_n = int(np.nansum(counts))
        if total_n:
            per_event_ll = total_ll / total_n
        else:
            per_event_ll = np.nan

        if use_tau:
            label = "τ-time"
        else:
            label = "raw"

        if gof_dims:
            dummy_adj = np.zeros((self.n_nodes, self.n_nodes))
            dummy_decays = np.ones((self.n_nodes, self.n_nodes))
            self._plot_gof_for_dims(
                gof_dims, events, dummy_adj, mu, dummy_decays,
                f"Poisson ({label})",
            )

        daily_rows = []
        for d, (dk, day_seq) in enumerate(zip(day_keys, events)):
            T_day = float(end_times[d])
            for k, dim_name in enumerate(self.marks_order):
                n_ev = len(day_seq[k])
                if T_day > 0:
                    mu_day = n_ev / T_day
                else:
                    mu_day = 0.0
                if mu_day > 0 and n_ev > 0:
                    ll_day = n_ev * np.log(mu_day) - mu_day * T_day
                else:
                    ll_day = np.nan
                if n_ev > 0:
                    per_event_day = ll_day / n_ev
                else:
                    per_event_day = np.nan
                daily_rows.append({
                    "day": dk, "dim": dim_name, "mu": mu_day,
                    "n_events": n_ev, "T_day": T_day,
                    "log_likelihood": ll_day,
                    "per_event": per_event_day,
                })
        daily_df = pd.DataFrame(daily_rows)

        per_day_summary = daily_df.groupby("day").agg(
            total_ll=("log_likelihood", "sum"),
            total_n=("n_events", "sum"),
        ).reset_index()
        per_day_summary["per_event"] = (
            per_day_summary["total_ll"] / per_day_summary["total_n"]
        )

        print(f"\n-- Poisson per-day calibration ({label}, mean +/- 95% CI) --")
        for dim_name in self.marks_order:
            mu_values = daily_df[daily_df["dim"] == dim_name]["mu"].values
            print(f"  {dim_name:8s}  mu = {format_mean_with_ci(mu_values)}")
        print(
            "\n  Per-event (per-day MLE): "
            f"{format_mean_with_ci(per_day_summary['per_event'].values)}"
        )

        return {
            "pooled_df": pooled_df,
            "daily_df": daily_df,
            "total_ll": total_ll,
            "per_event_ll": per_event_ll,
        }

    # --- 2. Univariate Hawkes: single exponential ---

    def fit_univariate_hawkes(self, use_tau: bool = False, n_trials: int = 100, beta_min: float = 0.01, beta_max: float = 20.0, gof_dims: List[str] = []) -> dict:
        """Fit independent univariate Hawkes per dimension (single-exp kernel).

        Returns dict with keys:
            df       : pd.DataFrame with per-dimension results
            models   : dict[dim_name -> fitted HawkesExpKern]
            total_ll, per_event_ll : floats
        """
        events, end_times = self.resolve_events(use_tau)

        if use_tau:
            label = "τ-time"
        else:
            label = "raw"
        results = []
        models: Dict[str, object] = {}

        for k, dim_name in enumerate(self.marks_order):
            realizations = [day_seq[k] for day_seq in events]
            total_ev = sum(len(s) for s in realizations)
            if total_ev == 0:
                continue

            if use_tau:
                ev_list = [[seq] for seq in realizations]
                dim_end = end_times
            else:
                filtered = [s for s in realizations if len(s)]
                ev_list = [[s] for s in filtered]
                dim_end = np.array([float(s[0].max()) for s in ev_list])

            study = optuna.create_study(direction="maximize")
            objective = UnivariateHawkesObjective(
                beta_min, beta_max, self.max_iter, self.tol, ev_list, dim_end,
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best_beta = study.best_params["beta"]

            best_model = HawkesExpKern(
                decays=np.array([[best_beta]]),
                max_iter=self.max_iter, tol=self.tol,
            )
            best_model.fit(ev_list)
            models[dim_name] = best_model

            alpha_hat = float(best_model.adjacency[0, 0])
            mu_hat = float(best_model.baseline[0])
            ll = float(best_model.score(events=ev_list, end_times=dim_end))

            results.append({
                "dim": dim_name, "beta": best_beta, "alpha": alpha_hat,
                "mu": mu_hat, "score": ll, "n_events": total_ev,
                "stable_alpha_lt_1": alpha_hat < 1.0,
            })

        df = pd.DataFrame(results)

        total_ll = float((df["score"] * df["n_events"]).sum())
        total_n = int(df["n_events"].sum())
        per_event_ll = total_ll / total_n

        if gof_dims:
            valid_gof_dims = [dim_name for dim_name in gof_dims if dim_name in models]
            if valid_gof_dims:
                full_adjacency = np.zeros((self.n_nodes, self.n_nodes))
                full_decays = np.ones((self.n_nodes, self.n_nodes))
                full_baseline = np.zeros(self.n_nodes)
                for fitted_name, model in models.items():
                    dim_idx = self.marks_order.index(fitted_name)
                    full_adjacency[dim_idx, dim_idx] = float(model.adjacency[0, 0])
                    full_decays[dim_idx, dim_idx] = float(model.decays[0, 0])
                    full_baseline[dim_idx] = float(model.baseline[0])
                self._plot_gof_for_dims(
                    valid_gof_dims, events, full_adjacency, full_baseline,
                    full_decays, f"Self-exciting ({label})",
                )

        return {
            "df": df,
            "models": models,
            "total_ll": total_ll,
            "per_event_ll": per_event_ll,
        }

    # --- 3. Multivariate Hawkes: single exponential ---

    def fit_multivariate_hawkes(self, use_tau: bool = False, n_trials: int = 1000, n_workers: int = 12, beta_min: float = 0.1, beta_max: float = 20.0, gof_dims: List[str] = []) -> dict:
        """Fit the multivariate single-exponential Hawkes process.

        This is the production calibration behind every figure. Optuna searches
        the per-kernel decay rates ``β_{ij}``; the model is then refit at the
        best matrix to read off the baseline ``μ`` and adjacency ``α``.

        With ``n_workers > 1`` the search is spread across subprocess workers
        (:func:`parallelisation.run_parallel_optuna`), which is the fast path on
        Windows where multiprocessing uses spawn. ``n_workers == 1`` runs the
        study in-process. This is slower, but useful for smoke tests.

        Parameters
        ----------
        use_tau : bool
            Fit in seasonality-adjusted τ-time instead of raw clock time.
        n_trials : int
            Total Optuna trials (split across workers).
        n_workers : int
            Number of parallel worker processes.
        beta_min, beta_max : float
            Search bounds for the decay rates.
        gof_dims : list[str], optional
            Dimensions to render time-rescaling GOF plots for.

        Returns
        -------
        dict
            Keys ``adjacency``, ``baseline``, ``decays``, ``branching_ratio``,
            ``score`` (per-event), and the fitted ``model``.
        """
        os.environ["OMP_NUM_THREADS"] = "1"

        events, end_times = self.resolve_events(use_tau)
        if use_tau:
            label = "τ-time"
        else:
            label = "raw"

        if n_workers > 1:
            data_dict = {
                "beta_min": beta_min,
                "beta_max": beta_max,
                "n_nodes": self.n_nodes,
                "marks_order": self.marks_order,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "events_dense": events,
                "end_times_array": end_times,
            }
            opt_results = run_parallel_optuna(
                data_dict,
                objective_type="single",
                n_workers=n_workers,
                n_trials=n_trials,
                study_name=f"hawkes_single_multi_{label}_{int(time.time())}",
            )
            best_params = opt_results["best_params"]
            if best_params is None:
                raise RuntimeError(
                    "Parallel optimisation failed: no trials completed"
                )
        else:
            study = optuna.create_study(direction="maximize")
            objective = MultivariateHawkesObjective(
                beta_min, beta_max, self.n_nodes, self.marks_order,
                self.max_iter, self.tol, events, end_times,
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params

        best_decays = np.zeros((self.n_nodes, self.n_nodes))
        for i, di in enumerate(self.marks_order):
            for j, dj in enumerate(self.marks_order):
                best_decays[i, j] = best_params[f"beta_{di}__{dj}"]

        model = HawkesExpKern(decays=best_decays, max_iter=self.max_iter, tol=self.tol)
        model.fit(events)

        adjacency = model.adjacency
        baseline = model.baseline
        branching_ratio = float(max(np.linalg.eigvals(adjacency).real))
        score = float(model.score(events=events, end_times=end_times))

        print(f"Branching ratio: {branching_ratio:.4f}")
        print(f"Per-event score ({label}): {score:.6f}")

        if gof_dims:
            self._plot_gof_for_dims(
                gof_dims, events, adjacency, baseline, best_decays,
                f"Multivariate Hawkes ({label})",
            )

        return {
            "adjacency": adjacency,
            "baseline": baseline,
            "decays": best_decays,
            "branching_ratio": branching_ratio,
            "score": score,
            "model": model,
        }

    # --- 4. Goodness-of-fit (single-exponential, any dimension) ---

    def goodness_of_fit(self, dim_name: str, adjacency: np.ndarray, baseline: np.ndarray, decays: np.ndarray, use_tau: bool = False, title: Optional[str] = None):
        """Plot the time-rescaling GOF for one dimension of a single-exp fit.

        The fitted compensator turns the dimension's events into interarrivals
        that are Exponential(1) under a correct model. ``plot_time_rescaling_cdf``
        then compares their empirical CDF against the 45-degree line with a KS
        band.

        Parameters
        ----------
        dim_name : str
            Dimension to evaluate.
        adjacency : np.ndarray, shape (n_nodes, n_nodes)
            Fitted adjacency ``α``.
        baseline : np.ndarray, shape (n_nodes,)
            Fitted baseline ``μ``.
        decays : np.ndarray, shape (n_nodes, n_nodes)
            Decay matrix ``β``.
        use_tau : bool
            Evaluate in τ-time instead of raw time.
        title : str, optional
            Plot title; a default is built from the dimension and time domain.

        Returns
        -------
        np.ndarray
            Concatenated compensator increments across days.
        """
        events, _ = self.resolve_events(use_tau)
        if use_tau:
            label = "τ-time"
        else:
            label = "raw"
        dim_idx = self.marks_order.index(dim_name)

        compensator_segments = []
        for day_seq in events:
            if len(day_seq[dim_idx]) == 0:
                continue
            day_compensator = compensator_interarrivals_single(
                day_seq, np.asarray(decays), adjacency, baseline,
            )[dim_idx]
            if len(day_compensator) > 0:
                compensator_segments.append(day_compensator)

        if compensator_segments:
            compensator_all = np.concatenate(compensator_segments)
        else:
            compensator_all = np.array([])
        if title is None:
            title = f"GOF ({label}): {dim_name}"
        if len(compensator_all) > 0:
            plot_time_rescaling_cdf(compensator_all, title)
        return compensator_all

    # --- 5. Save / load calibration results ---

    @staticmethod
    def save_params(path: Union[str, Path], **kwargs):
        """Pickle calibration results to *path*."""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(kwargs, f)
        print(f"Saved: {Path(path).name}")

    @staticmethod
    def load_params(path: Union[str, Path]) -> dict:
        """Load pickled calibration results."""
        with open(path, "rb") as f:
            return pickle.load(f)

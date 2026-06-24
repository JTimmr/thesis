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

A sum-of-exponentials ("triple kernel") path lives in the appendix at the
bottom of this module. It is kept separate on purpose so it can be deleted in
one block if the thesis stays with single-exponential kernels.

``tick`` is a hard dependency (see ``pyproject.toml``). It pins to older Python
on Windows, so calibration usually runs in its own environment.
"""

from __future__ import annotations

import os
import pickle
import time as _time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import optuna
from tick.hawkes import HawkesExpKern, HawkesSumExpKern

from research_core.classes.parallelisation import run_parallel_optuna


# ═══════════════════════════════════════════════════════════════════════════════
# Seasonality helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_average_seasonality_shape(
    seasonality_profiles: dict,
    marks_order: Optional[List[str]] = None,
    normalize: bool = True,
) -> dict:
    """Compute average intraday seasonality across event types.

    Returns dict with keys: grid, profile, std, n_patterns, cumulative_integral.
    """
    if marks_order is None:
        marks_order = list(seasonality_profiles.keys())

    all_profiles = []
    common_grid = None

    for dim_name in marks_order:
        if dim_name not in seasonality_profiles:
            continue
        data = seasonality_profiles[dim_name]
        if isinstance(data, tuple) and len(data) >= 2:
            grid = np.array(data[0])
            profile = np.array(data[1])
        else:
            continue
        if common_grid is None:
            common_grid = grid
        if normalize:
            profile = profile / profile.mean()
        all_profiles.append(profile)

    if len(all_profiles) == 0:
        raise ValueError("No valid seasonality profiles found")

    profiles_matrix = np.array(all_profiles)
    avg_profile = profiles_matrix.mean(axis=0)
    std_profile = profiles_matrix.std(axis=0)

    cum_integral = np.concatenate([
        [0.0],
        np.cumsum(
            np.diff(common_grid)
            * (avg_profile[:-1] + avg_profile[1:]) / 2.0
        ),
    ])

    return {
        "grid": common_grid,
        "profile": avg_profile,
        "std": std_profile,
        "n_patterns": len(all_profiles),
        "cumulative_integral": cum_integral,
    }


def create_average_time_transformer(
    seasonality_profiles: dict,
    marks_order: Optional[List[str]] = None,
    normalize: bool = True,
):
    """Return (transform_func, avg_data) for τ(t) = ∫₀ᵗ s_avg(u) du."""
    avg_data = get_average_seasonality_shape(
        seasonality_profiles, marks_order, normalize
    )
    transform_time = partial(
        _transform_time_from_average_profile,
        grid=avg_data["grid"],
        cumulative_integral=avg_data["cumulative_integral"],
    )

    return transform_time, avg_data


def _transform_time_from_average_profile(
    t,
    *,
    grid: np.ndarray,
    cumulative_integral: np.ndarray,
) -> np.ndarray:
    """Map clock time to tau-time using the average seasonality integral."""
    t = np.atleast_1d(t)
    t = np.clip(t, 0.0, grid[-1])
    return np.interp(t, grid, cumulative_integral)


def _format_mean_ci(vals, fmt: str = ".6f") -> str:
    """Format the sample mean and a 95% t confidence interval."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    mean_value = vals.mean() if n > 0 else np.nan
    if n < 2:
        return f"{mean_value:{fmt}}"
    sem = vals.std(ddof=1) / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, df=n - 1)
    lower, upper = mean_value - t_critical * sem, mean_value + t_critical * sem
    pct = (t_critical * sem / abs(mean_value) * 100) if mean_value != 0 else float("inf")
    return f"{mean_value:{fmt}}  95% CI [{lower:{fmt}}, {upper:{fmt}}] (+/-{pct:.1f}%)"


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna objectives
# ═══════════════════════════════════════════════════════════════════════════════

class _SelfObjective:
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
        try:
            model.fit(self.events)
            alpha = float(model.adjacency[0, 0])
            if alpha >= 1.0:
                return -np.inf
            score = model.score(events=self.events, end_times=self.end_times)
        except Exception as e:
            print("CRASH:", e, flush=True)
            raise
        return float(score)


class _MutualObjective:
    """Optuna objective for multivariate Hawkes calibration (single-exp)."""

    def __init__(
        self, beta_min, beta_max, n_nodes, marks_order, max_iter, tol,
        events, end_times,
    ):
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
        model = HawkesExpKern(
            decays=decays_matrix, max_iter=self.max_iter, tol=self.tol,
        )
        try:
            model.fit(self.events)
            score = model.score(events=self.events, end_times=self.end_times)
        except Exception as e:
            print("CRASH:", e, flush=True)
            raise

        A = model.adjacency
        br = max(np.linalg.eigvals(A).real)
        if br >= 1.0:
            return -np.inf
        print(f"Trial {trial.number} -> {score:.6f}")
        return float(score)


# ═══════════════════════════════════════════════════════════════════════════════
# Goodness-of-fit utilities
# ═══════════════════════════════════════════════════════════════════════════════

def plot_time_rescaling_cdf(interarrival_times, title: str):
    """KS-style CDF plot for transformed interarrival times."""
    interarrival_times = np.asarray(interarrival_times)
    interarrival_times = interarrival_times[interarrival_times > 0]

    cumulative = np.cumsum(interarrival_times)
    n = len(cumulative)
    T = float(cumulative[-1]) if n > 0 else 0.0

    x = np.hstack([0.0, np.repeat(cumulative, 2), T])
    y = np.repeat(np.arange(n + 1), 2) / n

    ks_bw = 1.36 / np.sqrt(n)

    plt.figure(figsize=(6, 5), dpi=100)
    plt.plot(x, y, "k-", label="Data")
    plt.fill_between(
        [0, T * ks_bw, T * (1 - ks_bw), T],
        [0, 0, 1 - 2 * ks_bw, 1 - ks_bw],
        [ks_bw, 2 * ks_bw, 1, 1],
        color="#dddddd",
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


def plot_all_seasonality_patterns(
    seasonality_profiles,
    marks_order=None,
    figsize=(12, 7),
    show_average=True,
    show_uncertainty=True,
    normalize=True,
    print_stats=True,
):
    """Plot all intraday seasonality patterns on the same figure.

    Parameters
    ----------
    seasonality_profiles : dict
        Mapping dim_name -> (grid, mean_profile, ...).
    marks_order : list, optional
        Order of dimensions to plot.
    figsize, show_average, show_uncertainty, normalize, print_stats :
        Display options.

    Returns
    -------
    fig, ax, stats_results
    """
    if marks_order is None:
        marks_order = list(seasonality_profiles.keys())

    colors = {
        "MO_bid": "#E74C3C",
        "MO_ask": "#C0392B",
        "LO_bid": "#3498DB",
        "LO_ask": "#2980B9",
        "CXL_bid": "#2ECC71",
        "CXL_ask": "#27AE60",
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

    all_profiles = []
    all_profiles_normalized = []
    profile_names = []
    common_grid = None

    for dim_name in marks_order:
        if dim_name not in seasonality_profiles:
            print(f"Warning: {dim_name} not found in seasonality_profiles")
            continue
        data = seasonality_profiles[dim_name]
        if isinstance(data, tuple) and len(data) >= 2:
            grid = np.array(data[0])
            profile = np.array(data[1])
        else:
            print(f"Warning: Unexpected format for {dim_name}")
            continue

        if common_grid is None:
            common_grid = grid

        grid_hours = grid / 3600.0
        all_profiles.append(profile)
        profile_names.append(dim_name)

        if normalize:
            profile_norm = profile / profile.mean()
            all_profiles_normalized.append(profile_norm)

        color = colors.get(dim_name, "gray")
        ls = linestyles.get(dim_name, "-")

        if normalize:
            ax.plot(
                grid_hours, profile_norm, label=dim_name,
                color=color, linestyle=ls, linewidth=1.5, alpha=0.7,
            )
        else:
            ax.plot(
                grid_hours, profile, label=dim_name,
                color=color, linestyle=ls, linewidth=1.5, alpha=0.7,
            )

    profiles_matrix = (
        np.array(all_profiles_normalized)
        if normalize
        else np.array(all_profiles)
    )
    grid_hours = common_grid / 3600.0
    avg_profile = profiles_matrix.mean(axis=0)
    std_profile = profiles_matrix.std(axis=0)

    if show_average:
        ax.plot(
            grid_hours, avg_profile, label="Average",
            color="black", linestyle="-", linewidth=2.5, zorder=10,
        )
    if show_uncertainty:
        ax.fill_between(
            grid_hours, avg_profile - std_profile, avg_profile + std_profile,
            color="gray", alpha=0.3, label="\u00b11 std", zorder=5,
        )

    ylabel = (
        "Normalized intensity (mean=1)"
        if normalize
        else "Event intensity (events/second)"
    )
    ax.set_xlabel("Time since market open (hours)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Intraday Seasonality Patterns by Event Type", fontsize=13)
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.1)

    # ── Statistical tests ─────────────────────────────────────────────
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

    all_corrs = [
        v["correlation"] for v in stats_results["correlations"].values()
    ]
    all_pairwise_corrs = [
        v["correlation"]
        for v in stats_results["pairwise_correlations"].values()
    ]
    stats_results["summary"] = {
        "mean_corr_with_average": float(np.mean(all_corrs)),
        "min_corr_with_average": float(np.min(all_corrs)),
        "mean_pairwise_corr": float(np.mean(all_pairwise_corrs)),
        "min_pairwise_corr": float(np.min(all_pairwise_corrs)),
        "mean_rmse_to_average": float(
            np.mean(list(stats_results["rmse_to_average"].values()))
        ),
    }

    try:
        friedman_stat, friedman_p = stats.friedmanchisquare(
            *profiles_matrix
        )
        stats_results["friedman_test"] = {
            "statistic": friedman_stat,
            "p_value": friedman_p,
            "interpretation": (
                "Patterns are similar (can use average)"
                if friedman_p > 0.05
                else "Patterns differ significantly"
            ),
        }
    except Exception as e:
        stats_results["friedman_test"] = {"error": str(e)}

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
            sig = (
                "***" if p < 0.001
                else "**" if p < 0.01
                else "*" if p < 0.05
                else ""
            )
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


def compensator_interarrivals_single(
    day_sequences: list,
    decays: np.ndarray,
    adjacency: np.ndarray,
    baseline: np.ndarray,
) -> list:
    """Compensator interarrivals for **single-exponential** kernels.

    Parameters
    ----------
    day_sequences : list of arrays, one per dimension (single day)
    decays        : (n_nodes, n_nodes)
    adjacency     : (n_nodes, n_nodes)
    baseline      : (n_nodes,)

    Returns
    -------
    list of arrays, one per dimension, with compensator increments.
    """
    n_nodes = len(day_sequences)

    all_times = np.concatenate(day_sequences) if day_sequences else np.array([])
    if len(all_times) == 0:
        return [np.array([]) for _ in range(n_nodes)]

    marks = np.concatenate([
        np.full(len(seq), idx, dtype=np.int64)
        for idx, seq in enumerate(day_sequences)
    ])

    order = np.argsort(all_times)
    sorted_t = all_times[order]
    sorted_m = marks[order]

    kernel_state = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    comp = np.zeros(n_nodes, dtype=np.float64)
    last_comp = np.zeros(n_nodes, dtype=np.float64)
    increments: List[list] = [[] for _ in range(n_nodes)]

    prev_t = 0.0
    idx = 0
    while idx < len(sorted_t):
        cur_t = sorted_t[idx]
        dt = cur_t - prev_t
        if dt > 0:
            comp += baseline * dt
            decay_f = np.exp(-decays * dt)
            comp += (kernel_state * (1.0 - decay_f) / decays).sum(axis=1)
            kernel_state *= decay_f

        # record increments for all events at this time
        start = idx
        while start < len(sorted_t) and sorted_t[start] == cur_t:
            d = sorted_m[start]
            increments[d].append(comp[d] - last_comp[d])
            last_comp[d] = comp[d]
            start += 1

        # apply jumps
        while idx < start:
            src = sorted_m[idx]
            kernel_state[:, src] += adjacency[:, src] * decays[:, src]
            idx += 1

        prev_t = cur_t

    return [np.array(inc) for inc in increments]


# ═══════════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════════

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

    def __init__(
        self,
        timestamps_by_day: List[List[np.ndarray]],
        marks_order: List[str],
        end_times: np.ndarray,
        seasonality_profiles: Optional[dict] = None,
        max_iter: int = 1_000_000,
        tol: float = 1e-9,
    ):
        # ── validate & store ──────────────────────────────────────
        self.marks_order = list(marks_order)
        self.n_nodes = len(self.marks_order)
        self.max_iter = max_iter
        self.tol = tol

        # Ensure contiguous float64 arrays everywhere
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

        # ── seasonality ───────────────────────────────────────────
        self.seasonality_profiles = seasonality_profiles
        self._tau_events: Optional[List[List[np.ndarray]]] = None
        self._tau_end_times: Optional[np.ndarray] = None
        self._transform_time = None
        self._avg_seasonality: Optional[dict] = None

        if seasonality_profiles is not None:
            self._build_tau()

    # ──────────────────────────────────────────────────────────────
    # τ-time construction
    # ──────────────────────────────────────────────────────────────

    def _build_tau(self):
        """Build τ-time events and end times from seasonality profiles."""
        transform_fn, avg_data = create_average_time_transformer(
            self.seasonality_profiles,
            marks_order=self.marks_order,
            normalize=True,
        )
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

    def build_tau_events(self):
        """Public accessor; builds τ-time if not already done.

        Returns (tau_events, tau_end_times).
        """
        if self._tau_events is None:
            if self.seasonality_profiles is None:
                raise ValueError(
                    "Cannot build τ-time events without seasonality_profiles."
                )
            self._build_tau()
        return self._tau_events, self._tau_end_times

    # ──────────────────────────────────────────────────────────────
    # Helper: pick the right events / end_times
    # ──────────────────────────────────────────────────────────────

    def _resolve_events(self, use_tau: bool):
        """Return (events, end_times) for the requested time domain."""
        if use_tau:
            ev, et = self.build_tau_events()
            return ev, et
        return self.timestamps_by_day, self.end_times

    # ══════════════════════════════════════════════════════════════
    # 1.  Poisson calibration
    # ══════════════════════════════════════════════════════════════

    def fit_poisson(
        self,
        use_tau: bool = False,
        day_keys: Optional[list] = None,
        gof_dims: Optional[List[str]] = None,
    ) -> dict:
        """Fit independent homogeneous Poisson processes.

        Parameters
        ----------
        use_tau : bool
            If True, use seasonality-adjusted τ-time.
        day_keys : list, optional
            Labels for each day (for per-day reporting).
        gof_dims : list[str], optional
            Dimensions for which GOF plots are produced.

        Returns
        -------
        dict with keys:
            pooled_df    : pd.DataFrame with pooled MLE results
            daily_df     : pd.DataFrame with per-day MLE results
            total_ll     : float, total log-likelihood
            per_event_ll : float, per-event log-likelihood
        """
        events, end_times = self._resolve_events(use_tau)
        if gof_dims is None:
            gof_dims = []
        if day_keys is None:
            day_keys = [f"day_{i}" for i in range(self.n_days)]

        T_total = float(np.sum(end_times)) if len(end_times) else 0.0

        counts = np.array([
            sum(len(day_seq[k]) for day_seq in events)
            for k in range(self.n_nodes)
        ], dtype=float)

        mu = counts / T_total if T_total > 0 else np.zeros_like(counts)
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
        per_event_ll = total_ll / total_n if total_n else np.nan

        print(pooled_df)
        label = "τ-time" if use_tau else "raw"
        print(f"\nPoisson total score (pooled, {label}):     {total_ll:.4f}")
        print(f"Poisson per-event score (pooled, {label}): {per_event_ll:.6f}")

        # ── GOF ──────────────────────────────────────────────────
        for dim_name in gof_dims:
            row = pooled_df[pooled_df["dim"] == dim_name].iloc[0]
            mu_dim = row["mu"]
            dim_idx = self.marks_order.index(dim_name)
            s_list = []
            for day_seq in events:
                seq = day_seq[dim_idx]
                if len(seq) > 0:
                    s_list.append(mu_dim * np.diff(np.concatenate([[0.0], seq])))
            s_all = np.concatenate(s_list) if s_list else np.array([])
            if len(s_all) > 0:
                plot_time_rescaling_cdf(s_all, f"Poisson ({label}): {dim_name}")
            else:
                print(f"No events for GOF: {dim_name}")

        # ── Per-day calibration ──────────────────────────────────
        daily_rows = []
        for d, (dk, day_seq) in enumerate(zip(day_keys, events)):
            T_day = float(end_times[d])
            for k, dim_name in enumerate(self.marks_order):
                n_ev = len(day_seq[k])
                mu_day = n_ev / T_day if T_day > 0 else 0.0
                ll_day = (
                    n_ev * np.log(mu_day) - mu_day * T_day
                    if mu_day > 0 and n_ev > 0 else np.nan
                )
                daily_rows.append({
                    "day": dk, "dim": dim_name, "mu": mu_day,
                    "n_events": n_ev, "T_day": T_day,
                    "log_likelihood": ll_day,
                    "per_event": ll_day / n_ev if n_ev > 0 else np.nan,
                })
        daily_df = pd.DataFrame(daily_rows)

        _pday = daily_df.groupby("day").agg(
            total_ll=("log_likelihood", "sum"),
            total_n=("n_events", "sum"),
        ).reset_index()
        _pday["per_event"] = _pday["total_ll"] / _pday["total_n"]

        print(f"\n-- Poisson per-day calibration ({label}, mean +/- 95% CI) --")
        for dim_name in self.marks_order:
            vals = daily_df[daily_df["dim"] == dim_name]["mu"].values
            print(f"  {dim_name:8s}  mu = {_format_mean_ci(vals)}")
        print(
            "\n  Per-event (per-day MLE): "
            f"{_format_mean_ci(_pday['per_event'].values)}"
        )

        return {
            "pooled_df": pooled_df,
            "daily_df": daily_df,
            "total_ll": total_ll,
            "per_event_ll": per_event_ll,
        }

    # ══════════════════════════════════════════════════════════════
    # 2.  Univariate Hawkes: single exponential
    # ══════════════════════════════════════════════════════════════

    def fit_univariate_hawkes(
        self,
        use_tau: bool = False,
        n_trials: int = 100,
        beta_min: float = 0.01,
        beta_max: float = 20.0,
        gof_dims: Optional[List[str]] = None,
    ) -> dict:
        """Fit independent univariate Hawkes per dimension (single-exp kernel).

        Returns dict with keys:
            df       : pd.DataFrame with per-dimension results
            models   : dict[dim_name -> fitted HawkesExpKern]
            total_ll, per_event_ll : floats
        """
        events, end_times = self._resolve_events(use_tau)
        if gof_dims is None:
            gof_dims = []

        label = "τ-time" if use_tau else "raw"
        results = []
        models: Dict[str, object] = {}

        # For τ-time: shared end times across dims
        if use_tau:
            shared_end = end_times
        else:
            shared_end = None  # per-dim end times computed below

        for k, dim_name in enumerate(self.marks_order):
            realizations = [day_seq[k] for day_seq in events]
            total_ev = sum(len(s) for s in realizations)
            if total_ev == 0:
                continue

            if use_tau:
                # wrap each day as [[tau_seq]]
                ev_list = [[seq] for seq in realizations]
                dim_end = shared_end
            else:
                filtered = [s for s in realizations if len(s)]
                ev_list = [[s] for s in filtered]
                dim_end = np.array([float(s[0].max()) for s in ev_list])

            # ── Optuna search over β ─────────────────────────────
            study = optuna.create_study(direction="maximize")
            objective = _SelfObjective(
                beta_min, beta_max, self.max_iter, self.tol, ev_list, dim_end,
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best_beta = study.best_params["beta"]

            # ── Refit at best β ──────────────────────────────────
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
        print(df)

        total_ll = float((df["score"] * df["n_events"]).sum())
        total_n = int(df["n_events"].sum())
        per_event_ll = total_ll / total_n

        print(f"\nSelf-exciting total score (pooled, {label}):     {total_ll:.4f}")
        print(f"Self-exciting per-event score (pooled, {label}): {per_event_ll:.6f}")

        unstable = df[~df["stable_alpha_lt_1"]]
        if len(unstable):
            print("\nUnstable (alpha >= 1):")
            print(unstable[["dim", "alpha"]])
        else:
            print("\nAll self-exciting processes satisfy alpha < 1")

        # ── GOF ──────────────────────────────────────────────────
        for gof_dim in gof_dims:
            if gof_dim not in models:
                print(f"Model not found for {gof_dim}")
                continue
            m = models[gof_dim]
            beta = float(m.decays[0, 0])
            adj = m.adjacency
            bl = m.baseline
            dim_idx = self.marks_order.index(gof_dim)
            s_list = []
            for day_seq in events:
                seq = day_seq[dim_idx]
                if len(seq) > 0:
                    if use_tau:
                        data_seq = seq
                    else:
                        data_seq = seq
                    s_day = compensator_interarrivals_single(
                        [data_seq],
                        np.array([[beta]], dtype=np.float64),
                        adj, bl,
                    )[0]
                    if len(s_day) > 0:
                        s_list.append(s_day)
            s_all = np.concatenate(s_list) if s_list else np.array([])
            if len(s_all) > 0:
                plot_time_rescaling_cdf(
                    s_all, f"Self-exciting ({label}): {gof_dim}",
                )
            else:
                print(f"No events for GOF: {gof_dim}")

        return {
            "df": df,
            "models": models,
            "total_ll": total_ll,
            "per_event_ll": per_event_ll,
        }

    # ══════════════════════════════════════════════════════════════
    # 3.  Multivariate Hawkes: single exponential
    # ══════════════════════════════════════════════════════════════

    def fit_multivariate_hawkes(
        self,
        use_tau: bool = False,
        n_trials: int = 1000,
        n_workers: int = 12,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        gof_dims: Optional[List[str]] = None,
    ) -> dict:
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

        events, end_times = self._resolve_events(use_tau)
        label = "τ-time" if use_tau else "raw"
        if gof_dims is None:
            gof_dims = []

        print(f"\n{'='*60}")
        print(f"Multivariate single-exp ({label}): "
              f"{n_trials} trials, {n_workers} worker(s)")
        print(f"{'='*60}\n")

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
                study_name=f"hawkes_single_multi_{label}_{int(_time.time())}",
            )
            best_params = opt_results["best_params"]
            if best_params is None:
                raise RuntimeError(
                    "Parallel optimisation failed: no trials completed"
                )
        else:
            study = optuna.create_study(direction="maximize")
            objective = _MutualObjective(
                beta_min, beta_max, self.n_nodes, self.marks_order,
                self.max_iter, self.tol, events, end_times,
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params

        best_decays = np.zeros((self.n_nodes, self.n_nodes))
        for i, di in enumerate(self.marks_order):
            for j, dj in enumerate(self.marks_order):
                best_decays[i, j] = best_params[f"beta_{di}__{dj}"]

        # ── refit at best decays ──────────────────────────────────
        model = HawkesExpKern(
            decays=best_decays, max_iter=self.max_iter, tol=self.tol,
        )
        model.fit(events)

        A = model.adjacency
        baseline = model.baseline
        br = float(max(np.linalg.eigvals(A).real))
        score = float(model.score(events=events, end_times=end_times))

        print(f"\nBranching ratio: {br:.4f}")
        print("\nAdjacency matrix:")
        print(pd.DataFrame(A, index=self.marks_order, columns=self.marks_order))
        print(f"\nDecay matrix ({label}):")
        print(pd.DataFrame(best_decays, index=self.marks_order, columns=self.marks_order))
        print("\nBaseline intensities:")
        print(pd.Series(baseline, index=self.marks_order, name="baseline"))

        # ── GOF ──────────────────────────────────────────────────
        for dim_name in gof_dims:
            dim_idx = self.marks_order.index(dim_name)
            s_list = []
            for day_seq in events:
                if len(day_seq[dim_idx]) == 0:
                    continue
                s_day = compensator_interarrivals_single(
                    day_seq, best_decays, A, baseline,
                )[dim_idx]
                if len(s_day) > 0:
                    s_list.append(s_day)
            s_all = np.concatenate(s_list) if s_list else np.array([])
            if len(s_all) > 0:
                plot_time_rescaling_cdf(
                    s_all, f"Multivariate Hawkes ({label}): {dim_name}",
                )
            else:
                print(f"No events for GOF: {dim_name}")

        return {
            "adjacency": A,
            "baseline": baseline,
            "decays": best_decays,
            "branching_ratio": br,
            "score": score,
            "model": model,
        }

    # ══════════════════════════════════════════════════════════════
    # 4.  Goodness-of-fit  (single-exponential, any dimension)
    # ══════════════════════════════════════════════════════════════

    def goodness_of_fit(
        self,
        dim_name: str,
        adjacency: np.ndarray,
        baseline: np.ndarray,
        decays: np.ndarray,
        use_tau: bool = False,
        title: Optional[str] = None,
    ):
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
        events, _ = self._resolve_events(use_tau)
        label = "τ-time" if use_tau else "raw"
        dim_idx = self.marks_order.index(dim_name)

        s_list = []
        for day_seq in events:
            if len(day_seq[dim_idx]) == 0:
                continue
            s_day = compensator_interarrivals_single(
                day_seq, np.asarray(decays), adjacency, baseline,
            )[dim_idx]
            if len(s_day) > 0:
                s_list.append(s_day)

        s_all = np.concatenate(s_list) if s_list else np.array([])
        if title is None:
            title = f"GOF ({label}): {dim_name}"
        if len(s_all) > 0:
            plot_time_rescaling_cdf(s_all, title)
        else:
            print(f"No events for GOF: {dim_name}")
        return s_all

    # ══════════════════════════════════════════════════════════════
    # 5.  Save / load calibration results
    # ══════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════
    # TRIPLE KERNEL / QUINTET APPENDIX (experimental)
    #
    # Everything below fits sum-of-exponential ("triple") kernels via Optuna.
    # It is the path used for the quintet calibration experiments. Nothing in
    # the single-exponential production calibration above depends on it, so the
        # whole block, these methods plus the module-level appendix at the end of
        # the file, can be deleted in one go if the thesis stays single-exp.
    # ══════════════════════════════════════════════════════════════

    # Univariate: sum-of-exponentials (Optuna β search)
    def fit_univariate_hawkes_sumexp_optuna(
        self,
        use_tau: bool = False,
        n_trials: int = 200,
        n_workers: int = 1,
        beta_ranges: Optional[List[Tuple[float, float]]] = None,
        gof_dims: Optional[List[str]] = None,
        penalty: str = "l2",
        C: float = 1e3,
    ) -> dict:
        """Fit independent univariate Hawkes per dimension (sum-of-exp).

        Shared decay rates are **optimized per dimension** via Optuna.
        ``beta_ranges`` is a list of ``(lo, hi)`` tuples, one per component,
        ordered fastest-to-slowest.  For each trial ``HawkesSumExpKern``
        fits baseline and the alpha coefficients.

        When ``n_workers > 1``, each dimension's study is distributed
        across parallel subprocess workers.

        Returns dict with keys:
            df       -- pd.DataFrame with per-dimension results
            models   -- dict[dim_name -> fitted HawkesSumExpKern]
            total_ll, per_event_ll -- floats
        """
        if beta_ranges is None:
            beta_ranges = [(10.0, 100.0), (1.0, 10.0), (0.005, 0.01)]
        n_components = len(beta_ranges)

        events, end_times = self._resolve_events(use_tau)
        label = "τ-time" if use_tau else "raw"
        if gof_dims is None:
            gof_dims = []

        results = []
        models: Dict[str, object] = {}

        for k, dim_name in enumerate(self.marks_order):
            realizations = [day_seq[k] for day_seq in events]
            total_ev = sum(len(s) for s in realizations)
            if total_ev == 0:
                continue

            ev_list = []
            et_list = []
            for s, et in zip(realizations, end_times):
                if len(s) > 0:
                    ev_list.append(
                        [np.ascontiguousarray(s, dtype=np.float64)]
                    )
                    et_list.append(et)

            print(f"\n-- {dim_name} ({total_ev:,} events), "
                  f"{n_workers} worker(s) --")

            if n_workers > 1:
                data_dict = {
                    "beta_ranges": beta_ranges,
                    "penalty": penalty,
                    "C": C,
                    "max_iter": self.max_iter,
                    "tol": self.tol,
                    "events": ev_list,
                    "end_times": et_list,
                }
                opt_results = run_parallel_optuna(
                    data_dict,
                    objective_type="sumexp_self",
                    n_workers=n_workers,
                    n_trials=n_trials,
                    study_name=(f"sumexp_self_{dim_name}_{label}_"
                                f"{int(_time.time())}"),
                )
                best_params = opt_results["best_params"]
                if best_params is None:
                    print(f"  Skipping {dim_name}: no trials completed")
                    continue
                best_betas = np.array([
                    best_params[f"beta_{i}"]
                    for i in range(n_components)
                ])
            else:
                study = optuna.create_study(direction="maximize")
                objective = _SumExpSelfObjective(
                    beta_ranges, penalty, C,
                    self.max_iter, self.tol,
                    ev_list, et_list,
                )
                study.optimize(
                    objective, n_trials=n_trials, show_progress_bar=False,
                )
                best_betas = np.array([
                    study.best_params[f"beta_{i}"]
                    for i in range(n_components)
                ])

            best_learner = HawkesSumExpKern(
                decays=best_betas,
                penalty=penalty, C=C,
                max_iter=self.max_iter, tol=self.tol, verbose=False,
            )
            best_learner.fit(ev_list)
            models[dim_name] = best_learner

            bl = best_learner.baseline
            adj = best_learner.adjacency  # (1, 1, n_components)
            mu_hat = float(bl[0])
            alphas = {f"alpha_{i}": float(adj[0, 0, i])
                      for i in range(n_components)}
            kernel_norm = sum(alphas.values())
            sc = float(best_learner.score(
                events=ev_list, end_times=et_list,
            ))

            row = {
                "dim": dim_name,
                "mu": mu_hat,
                **alphas,
                "kernel_norm": kernel_norm,
                **{f"beta_{i}": float(best_betas[i])
                   for i in range(n_components)},
                "score": sc,
                "n_events": total_ev,
                "stable": kernel_norm < 1.0,
            }
            results.append(row)

        df = pd.DataFrame(results)
        print(df)

        total_sc = float((df["score"] * df["n_events"]).sum())
        total_n = int(df["n_events"].sum())
        per_ev = total_sc / total_n if total_n else np.nan

        print(f"\n{n_components}-exp Optuna self-exciting total score "
              f"({label}):     {total_sc:.4f}")
        print(f"{n_components}-exp Optuna self-exciting per-event score "
              f"({label}): {per_ev:.6f}")

        # ── GOF ──────────────────────────────────────────────────
        for gof_dim in gof_dims:
            if gof_dim not in models:
                print(f"Model not found for {gof_dim}")
                continue
            lrn = models[gof_dim]
            row = df[df["dim"] == gof_dim].iloc[0]
            decays_vec = np.array([row[f"beta_{i}"]
                                   for i in range(n_components)])
            adj_3d = lrn.adjacency
            bl = lrn.baseline
            dim_idx = self.marks_order.index(gof_dim)
            s_list = []
            for day_seq in events:
                seq = day_seq[dim_idx]
                if len(seq) > 0:
                    s_day = compensator_interarrivals_sumexp(
                        [seq], decays_vec, adj_3d, bl,
                    )[0]
                    if len(s_day) > 0:
                        s_list.append(s_day)
            s_all = np.concatenate(s_list) if s_list else np.array([])
            if len(s_all) > 0:
                plot_time_rescaling_cdf(
                    s_all,
                    f"{n_components}-exp Optuna self-exciting "
                    f"({label}): {gof_dim}",
                )
            else:
                print(f"No events for GOF: {gof_dim}")

        return {
            "df": df,
            "models": models,
            "total_ll": total_sc,
            "per_event_ll": per_ev,
        }

    # Multivariate: sum-of-exponentials (Optuna β search)
    def fit_multivariate_hawkes_sumexp_optuna(
        self,
        use_tau: bool = False,
        n_trials: int = 300,
        n_workers: int = 1,
        beta_ranges: Optional[List[Tuple[float, float]]] = None,
        gof_dims: Optional[List[str]] = None,
        penalty: str = "l2",
        C: float = 1e3,
        slow_self_floor: Optional[dict] = None,
        rho_target: float = 0.95,
    ) -> dict:
        """Fit multivariate Hawkes with sum-of-exponential kernels.

        Shared decay rates are **optimized** via Optuna.
        ``beta_ranges`` is a list of ``(lo, hi)`` tuples, one per component,
        ordered fastest-to-slowest.  For each trial ``HawkesSumExpKern``
        fits all baselines and the alpha matrices.

        Supercritical kernels (ρ > ``rho_target``) are scaled down to
        ``rho_target`` before scoring, so Optuna optimises the fit quality
        of the model that will actually run in simulation.

        When ``n_workers > 1``, trials are distributed across parallel
        subprocess workers via :func:`parallelisation.run_parallel_optuna`.

        Returns
        -------
        dict with keys:
            adjacency      -- (n_nodes, n_nodes, n_components)
            baseline       -- (n_nodes,)
            decays         -- 1-D array of length n_components
            kernel_norms   -- (n_nodes, n_nodes)
            branching_ratio -- float
            score          -- float
            model          -- fitted HawkesSumExpKern
            best_betas     -- 1-D array of optimized decay rates
        """
        if beta_ranges is None:
            beta_ranges = [(10.0, 100.0), (1.0, 10.0), (0.005, 0.01)]
        n_components = len(beta_ranges)

        events, end_times = self._resolve_events(use_tau)
        label = "τ-time" if use_tau else "raw"
        if gof_dims is None:
            gof_dims = []

        print(f"\n{'='*60}")
        print(f"{n_components}-exp Optuna multivariate ({label}): "
              f"{n_trials} trials, {n_workers} worker(s)")
        print(f"{'='*60}\n")

        if n_workers > 1:
            data_dict = {
                "beta_ranges": beta_ranges,
                "penalty": penalty,
                "C": C,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "events": events,
                "end_times": end_times,
                "slow_self_floor": slow_self_floor,
                "rho_target": rho_target,
            }
            opt_results = run_parallel_optuna(
                data_dict,
                objective_type="sumexp",
                n_workers=n_workers,
                n_trials=n_trials,
                study_name=f"sumexp_multi_{label}_{int(_time.time())}",
            )
            best_params = opt_results["best_params"]
            if best_params is None:
                raise RuntimeError("Parallel optimisation failed: "
                                   "no trials completed")
            best_betas = np.array([
                best_params[f"beta_{i}"]
                for i in range(n_components)
            ])
        else:
            study = optuna.create_study(direction="maximize")
            objective = _SumExpObjective(
                beta_ranges, penalty, C,
                self.max_iter, self.tol,
                events, end_times,
                slow_self_floor=slow_self_floor,
                rho_target=rho_target,
            )
            study.optimize(
                objective, n_trials=n_trials, show_progress_bar=False,
            )
            best_betas = np.array([
                study.best_params[f"beta_{i}"]
                for i in range(n_components)
            ])

        beta_str = ", ".join(
            f"β_{i}={b:.4f}" for i, b in enumerate(best_betas)
        )
        print(f"\nBest betas: {beta_str}")

        # ── Refit at best betas ───────────────────────────────────
        learner = HawkesSumExpKern(
            decays=best_betas, penalty=penalty, C=C,
            max_iter=self.max_iter, tol=self.tol, verbose=False,
        )
        learner.fit(events)

        bl = learner.baseline
        adj_raw = learner.adjacency
        kernel_norms_raw = adj_raw.sum(axis=2)
        rho_raw = float(max(np.linalg.eigvals(kernel_norms_raw).real))

        if rho_raw > rho_target:
            scale = rho_target / rho_raw
            adj = adj_raw * scale
        else:
            scale = 1.0
            adj = adj_raw

        kernel_norms = adj.sum(axis=2)
        br = float(max(np.linalg.eigvals(kernel_norms).real))
        sc = float(learner.score(events=events, end_times=end_times,
                                 adjacency=adj))

        print(f"\nMultivariate {n_components}-exp Optuna Hawkes ({label})")
        if scale < 1.0:
            print(f"  Raw rho: {rho_raw:.4f} -> scaled by {scale:.4f} to rho={br:.4f}")
        print(f"  Branching ratio: {br:.4f}")
        print(f"  Score: {sc:.6f}")
        print("\nBaseline:")
        print(pd.Series(bl, index=self.marks_order, name="baseline"))
        for u in range(n_components):
            print(f"\nalpha_{u} (beta={best_betas[u]:.4f}):")
            print(pd.DataFrame(adj[:, :, u], index=self.marks_order,
                               columns=self.marks_order))
        print("\nKernel norms (sum of alpha_u):")
        print(pd.DataFrame(kernel_norms, index=self.marks_order,
                           columns=self.marks_order))

        # ── GOF ──────────────────────────────────────────────────
        for dim_name in gof_dims:
            dim_idx = self.marks_order.index(dim_name)
            s_list = []
            for day_seq in events:
                if len(day_seq[dim_idx]) == 0:
                    continue
                s_day = compensator_interarrivals_sumexp(
                    day_seq, best_betas, adj, bl,
                )[dim_idx]
                if len(s_day) > 0:
                    s_list.append(s_day)
            s_all = np.concatenate(s_list) if s_list else np.array([])
            if len(s_all) > 0:
                plot_time_rescaling_cdf(
                    s_all,
                    f"{n_components}-exp Optuna Multivariate "
                    f"({label}): {dim_name}",
                )
            else:
                print(f"No events for GOF: {dim_name}")

        # ── visualise ────────────────────────────────────────────
        n_plots = n_components + 1
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        for u in range(n_components):
            im = axes[u].imshow(adj[:, :, u], cmap="viridis")
            plt.colorbar(im, ax=axes[u], label=f"α_{u}")
            axes[u].set_xticks(range(self.n_nodes))
            axes[u].set_xticklabels(self.marks_order, rotation=45,
                                    ha="right")
            axes[u].set_yticks(range(self.n_nodes))
            axes[u].set_yticklabels(self.marks_order)
            axes[u].set_title(f"α_{u} (β={best_betas[u]:.4f})")

        im_n = axes[-1].imshow(kernel_norms, cmap="viridis")
        plt.colorbar(im_n, ax=axes[-1], label="kernel norm")
        axes[-1].set_xticks(range(self.n_nodes))
        axes[-1].set_xticklabels(self.marks_order, rotation=45, ha="right")
        axes[-1].set_yticks(range(self.n_nodes))
        axes[-1].set_yticklabels(self.marks_order)
        axes[-1].set_title("Kernel norms (Σ α_u)")

        fig.suptitle(
            f"{n_components}-exp Optuna adjacency ({label})", fontsize=14,
        )
        fig.tight_layout()
        plt.show()

        return {
            "adjacency": adj,
            "baseline": bl,
            "decays": best_betas,
            "kernel_norms": kernel_norms,
            "branching_ratio": br,
            "score": sc,
            "model": learner,
            "best_betas": best_betas,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRIPLE KERNEL / QUINTET APPENDIX (experimental)
#
# Sum-of-exponential objectives and the matching compensator, used only by the
# fit_*_sumexp_optuna methods above. Kept at module level and at the bottom of
# the file so the triple-kernel path can be deleted in one block. The
# single-exponential production calibration does not touch anything here.
# ═══════════════════════════════════════════════════════════════════════════════


def compensator_interarrivals_sumexp(
    day_sequences: list,
    decays: np.ndarray,
    adjacency: np.ndarray,
    baseline: np.ndarray,
) -> list:
    """Compensator interarrivals for sum-of-exponential kernels.

    Works for any number of components ``U`` (``U = 3`` for the triple kernel).
    The kernel from source ``j`` to target ``i`` is
    ``φ_ij(t) = Σ_u α^u_ij β^u exp(−β^u t)``.

    Parameters
    ----------
    day_sequences : list of np.ndarray
        One event-time array per dimension, for a single day.
    decays : np.ndarray, shape (U,)
        Decay rates ``β^u``.
    adjacency : np.ndarray, shape (n_nodes, n_nodes, U)
        Component adjacencies ``α^u_ij``.
    baseline : np.ndarray, shape (n_nodes,)
        Baseline intensities ``μ``.

    Returns
    -------
    list of np.ndarray
        Compensator increments per dimension; Exponential(1) under a good fit.
    """
    n_nodes = len(day_sequences)
    U = len(decays)

    all_times = np.concatenate(day_sequences) if day_sequences else np.array([])
    if len(all_times) == 0:
        return [np.array([]) for _ in range(n_nodes)]

    marks = np.concatenate([
        np.full(len(seq), idx, dtype=np.int64)
        for idx, seq in enumerate(day_sequences)
    ])

    order = np.argsort(all_times)
    sorted_t = all_times[order]
    sorted_m = marks[order]

    # kernel_state[u, i, j]: auxiliary state for decay u, target i, source j
    kernel_state = np.zeros((U, n_nodes, n_nodes), dtype=np.float64)
    comp = np.zeros(n_nodes, dtype=np.float64)
    last_comp = np.zeros(n_nodes, dtype=np.float64)
    increments: List[list] = [[] for _ in range(n_nodes)]

    prev_t = 0.0
    idx = 0
    while idx < len(sorted_t):
        cur_t = sorted_t[idx]
        dt = cur_t - prev_t
        if dt > 0:
            comp += baseline * dt
            for u in range(U):
                beta_u = decays[u]
                decay_f = np.exp(-beta_u * dt)
                # ∫₀^dt kernel_state · exp(−β t') dt' = kernel_state · (1 − e^{−β dt}) / β
                comp += (kernel_state[u] * (1.0 - decay_f) / beta_u).sum(axis=1)
                kernel_state[u] *= decay_f

        start = idx
        while start < len(sorted_t) and sorted_t[start] == cur_t:
            d = sorted_m[start]
            increments[d].append(comp[d] - last_comp[d])
            last_comp[d] = comp[d]
            start += 1

        while idx < start:
            src = sorted_m[idx]
            for u in range(U):
                kernel_state[u, :, src] += adjacency[:, src, u] * decays[u]
            idx += 1

        prev_t = cur_t

    return [np.array(inc) for inc in increments]


class _SumExpSelfObjective:
    """Optuna objective for univariate sum-of-exponentials Hawkes.

    Searches over N shared decay rates (one per component), ordered
    fastest-to-slowest.  Each beta_i is drawn from its own range.
    """

    def __init__(self, beta_ranges, penalty, C,
                 max_iter, tol, events, end_times):
        self.beta_ranges = beta_ranges
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.events = events
        self.end_times = end_times

    def __call__(self, trial):
        betas = []
        for idx, (lo, hi) in enumerate(self.beta_ranges):
            betas.append(trial.suggest_float(
                f"beta_{idx}", lo, hi, log=True,
            ))
        if sorted(betas, reverse=True) != betas:
            return -np.inf

        decays = np.array(betas)
        learner = HawkesSumExpKern(
            decays=decays, penalty=self.penalty, C=self.C,
            max_iter=self.max_iter, tol=self.tol, verbose=False,
        )
        try:
            learner.fit(self.events)
            kernel_norm = float(learner.adjacency.sum())
            if kernel_norm >= 1.0:
                return -np.inf
            score = float(learner.score(
                events=self.events, end_times=self.end_times,
            ))
        except Exception as e:
            print("CRASH:", e, flush=True)
            raise
        return score


class _SumExpObjective:
    """Optuna objective for multivariate sum-of-exponentials Hawkes.

    Searches over N shared decay rates (one per component), ordered
    fastest-to-slowest.  Each beta_i is drawn from its own range.

    Supercritical kernels (spectral radius > ``rho_target``) are scaled
    down to exactly ``rho_target`` before scoring, so the log-likelihood
    reflects the fit quality of the model that will actually be used in
    simulation.

    Parameters
    ----------
    slow_self_floor : dict or None
        Soft penalty ensuring the slowest kernel carries a minimum fraction
        of each specified dimension's self-excitation.
        Keys: ``"dims"`` (list of int) and ``"r_target"`` (float, e.g. 0.20).
        Penalty per dimension: ``max(0, r_target / r - 1)`` where
        ``r = α_slow[dim,dim] / Σ_k α_k[dim,dim]``.
    rho_target : float
        Target spectral radius.  Kernels with ρ > rho_target are scaled
        to this value before the score is computed.
    """

    def __init__(self, beta_ranges, penalty, C,
                 max_iter, tol, events, end_times,
                 slow_self_floor=None, rho_target=0.95):
        self.beta_ranges = beta_ranges
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.events = events
        self.end_times = end_times
        self.slow_self_floor = slow_self_floor
        self.rho_target = rho_target

    def __call__(self, trial):
        betas = []
        for idx, (lo, hi) in enumerate(self.beta_ranges):
            betas.append(trial.suggest_float(
                f"beta_{idx}", lo, hi, log=True,
            ))
        if sorted(betas, reverse=True) != betas:
            return -np.inf

        decays = np.array(betas)
        learner = HawkesSumExpKern(
            decays=decays, penalty=self.penalty, C=self.C,
            max_iter=self.max_iter, tol=self.tol, verbose=False,
        )
        try:
            learner.fit(self.events)
            adj = learner.adjacency
            kernel_norms = adj.sum(axis=2)
            rho = float(max(np.linalg.eigvals(kernel_norms).real))

            if rho > self.rho_target:
                scaled_adj = adj * (self.rho_target / rho)
            else:
                scaled_adj = adj

            score = float(learner.score(
                events=self.events, end_times=self.end_times,
                adjacency=scaled_adj,
            ))
        except Exception as e:
            print("CRASH:", e, flush=True)
            raise

        if self.slow_self_floor is not None:
            r_target = self.slow_self_floor["r_target"]
            for dim in self.slow_self_floor["dims"]:
                total = adj[dim, dim, :].sum()
                if total > 0:
                    r = adj[dim, dim, -1] / total
                    if r > 0:
                        score -= max(0.0, (r_target / r) - 1.0)
                    else:
                        score -= 100.0
        beta_str = ", ".join(f"beta_{i}={b:.4f}" for i, b in enumerate(betas))
        print(f"Trial {trial.number} -> {score:.6f}  (rho={rho:.4f}, {beta_str})")
        return score

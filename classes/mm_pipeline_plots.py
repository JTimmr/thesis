"""Publication figures for the supervised fill-belief and MM competition pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .mm_pipeline_analysis import dynamic_padded_bounds, run_cluster_bootstrap_interval

PathLike = Union[str, Path]

# Colorblind-friendly Tableau palette, with neutral gray for references.
color_navy = "tab:blue"
color_darkorange = "tab:orange"
color_seagreen = "tab:green"
color_firebrick = "tab:red"
color_dimgray = "dimgray"
color_goldenrod = "tab:purple"

population_colors = {
    1: "tab:purple",
    2: "tab:blue",
    5: "tab:green",
    10: "gold",
}
stage_colors = (
    color_navy,
    color_darkorange,
    color_seagreen,
)

tercile_colors = {
    "low": color_dimgray,
    "mid": color_goldenrod,
    "high": color_darkorange,
}

side_colors = {"bid": color_navy, "ask": color_firebrick}
side_markers = {"bid": "o", "ask": "s"}
side_linestyles = {"bid": "-", "ask": "--"}

comparison_styles = {
    "solo_adapted": (":", color_dimgray, "o"),
    "independent": ("-", color_navy, "o"),
    "population_adapted": ("--", color_seagreen, "s"),
    "coordinated": ("-.", color_darkorange, "D"),
}


def set_report_style() -> None:
    """Apply matplotlib defaults for thesis report figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.6,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_report_figure(
    fig: Figure,
    output_path: PathLike,
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> Path:
    """Save a figure as PDF, SVG, or 300 dpi PNG based on the file suffix."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        fig.savefig(path, format="pdf", bbox_inches=bbox_inches)
    elif suffix == ".svg":
        fig.savefig(path, format="svg", bbox_inches=bbox_inches)
    elif suffix == ".png":
        fig.savefig(path, format="png", dpi=dpi, bbox_inches=bbox_inches)
    else:
        raise ValueError(f"Unsupported figure format: {suffix!r} (use .pdf, .svg, or .png)")
    return path


def apply_y_limits(
    ax: plt.Axes,
    values: Iterable[float],
    *,
    ci_low: Optional[Iterable[float]] = None,
    ci_high: Optional[Iterable[float]] = None,
    ymin_floor: Optional[float] = None,
    ymax_cap: Optional[float] = None,
) -> None:
    """Set y-axis limits from data and optional confidence bands."""
    ymin, ymax = dynamic_padded_bounds(values, ci_low=ci_low, ci_high=ci_high)
    if ymin_floor is not None:
        ymin = min(ymin, ymin_floor)
    if ymax_cap is not None:
        ymax = max(ymax, ymax_cap)
    ax.set_ylim(ymin, ymax)


def apply_x_limits(
    ax: plt.Axes,
    values: Iterable[float],
    *,
    ci_low: Optional[Iterable[float]] = None,
    ci_high: Optional[Iterable[float]] = None,
) -> None:
    """Set x-axis limits from data and optional confidence bands."""
    xmin, xmax = dynamic_padded_bounds(values, ci_low=ci_low, ci_high=ci_high)
    ax.set_xlim(xmin, xmax)


def apply_probability_y_limits(
    ax: plt.Axes,
    values: Iterable[float],
    *,
    ci_low: Optional[Iterable[float]] = None,
    ci_high: Optional[Iterable[float]] = None,
) -> None:
    """Use a zero baseline without expanding small probabilities to [0, 1]."""
    _, ymax = dynamic_padded_bounds(
        values,
        ci_low=ci_low,
        ci_high=ci_high,
        padding_fraction=0.10,
        min_padding=0.001,
    )
    ax.set_ylim(0.0, min(1.0, ymax))


def positive_log_bounds(
    values: Iterable[float],
    *,
    ci_low: Optional[Iterable[float]] = None,
    ci_high: Optional[Iterable[float]] = None,
) -> Tuple[float, float]:
    """Multiplicatively padded bounds for positive-valued log axes."""
    positive_values = []
    for series in (values, ci_low, ci_high):
        if series is None:
            continue
        array = np.asarray(list(series), dtype=float).ravel()
        finite_positive = array[np.isfinite(array) & (array > 0)]
        if finite_positive.size:
            positive_values.append(finite_positive)
    stacked = np.concatenate(positive_values)
    log_min, log_max = dynamic_padded_bounds(
        np.log10(stacked),
        padding_fraction=0.08,
        min_padding=0.08,
    )
    return float(10**log_min), float(10**log_max)


def plot_ci_ribbon(
    ax: plt.Axes,
    x_values: np.ndarray,
    center: np.ndarray,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    *,
    color: str,
    alpha: float = 0.18,
) -> None:
    if ax.get_yscale() == "log":
        positive = np.concatenate([center, ci_lo, ci_hi])
        positive = positive[np.isfinite(positive) & (positive > 0)]
        lower_display_bound = positive.min() / 2.0
        ci_lo = np.maximum(ci_lo, lower_display_bound)
    ax.fill_between(x_values, ci_lo, ci_hi, color=color, alpha=alpha, linewidth=0.0)


def plot_mean_ci_line(
    ax: plt.Axes,
    x_values: np.ndarray,
    center: np.ndarray,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    marker: str = "o",
) -> None:
    plot_ci_ribbon(ax, x_values, center, ci_lo, ci_hi, color=color)
    ax.plot(
        x_values,
        center,
        linestyle=linestyle,
        marker=marker,
        color=color,
        label=label,
        markersize=5,
    )


def population_color_map(n_agents_values: Sequence[int]) -> Dict[int, str]:
    ordered = sorted({int(value) for value in n_agents_values})
    return {
        n_agents: population_colors.get(n_agents, color_dimgray)
        for n_agents in ordered
    }


def side_panel_label(side_value: Union[int, str]) -> str:
    if side_value in (1, "1", "bid", "Bid"):
        return "Bid"
    return "Ask"


def plot_fill_by_intensity(
    summary_df: pd.DataFrame,
    *,
    delta_band_label: str = r"$\delta \in [0.10, 0.20)$ PLN",
) -> Figure:
    """Fill rate vs opposite-side MO intensity for bid and ask panels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    global_y_limits = positive_log_bounds(
        summary_df[
            ["empirical_fill", "predicted_fill", "exponential_fill"]
        ].to_numpy().ravel(),
        ci_low=summary_df[
            ["empirical_fill_ci_lo", "predicted_fill_ci_lo"]
        ].to_numpy().ravel(),
        ci_high=summary_df[
            ["empirical_fill_ci_hi", "predicted_fill_ci_hi"]
        ].to_numpy().ravel(),
    )
    x_axis_max = {"bid": 3.5, "ask": 6.0}
    for axis, side_label in zip(axes, ("bid", "ask")):
        axis.set_yscale("log")
        side_frame = summary_df.loc[summary_df["side"] == side_label].sort_values("intensity_bin_center")
        if side_frame.empty:
            axis.set_title(f"{side_label.capitalize()}: no data")
            continue
        x_values = side_frame["intensity_bin_center"].to_numpy(dtype=float)
        plot_mean_ci_line(
            axis,
            x_values,
            side_frame["empirical_fill"].to_numpy(dtype=float),
            side_frame["empirical_fill_ci_lo"].to_numpy(dtype=float),
            side_frame["empirical_fill_ci_hi"].to_numpy(dtype=float),
            color=side_colors[side_label],
            label="Empirical fill rate",
            linestyle=side_linestyles[side_label],
            marker=side_markers[side_label],
        )
        plot_mean_ci_line(
            axis,
            x_values,
            side_frame["predicted_fill"].to_numpy(dtype=float),
            side_frame["predicted_fill_ci_lo"].to_numpy(dtype=float),
            side_frame["predicted_fill_ci_hi"].to_numpy(dtype=float),
            color=color_dimgray,
            label="MLP predicted fill",
            linestyle="--",
            marker="s",
        )
        exponential_level = float(side_frame["exponential_fill"].iloc[0])
        axis.axhline(
            exponential_level,
            color=color_darkorange,
            linestyle=":",
            linewidth=1.4,
            label="Exponential benchmark",
        )
        axis.set_xlabel("Opposite-side MO intensity")
        axis.set_ylabel("Fill probability")
        axis.set_title(f"{side_label.capitalize()} quotes ({delta_band_label})")
        axis.set_ylim(*global_y_limits)
        apply_x_limits(axis, x_values)
        axis.set_xlim(axis.get_xlim()[0], x_axis_max[side_label])
        axis.legend(loc="upper left")
    fig.suptitle("Fill probability vs opposite-side market-order intensity", fontsize=12)
    return fig


def plot_reliability_by_intensity(summary_df: pd.DataFrame) -> Figure:
    """Reliability diagram split by opposite-side MO-intensity tercile."""
    fig, axis = plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
    axis.set_xscale("log")
    axis.set_yscale("log")
    mlp_frame = summary_df.loc[summary_df["curve"] == "mlp"].copy()
    for tercile_name in ("low", "mid", "high"):
        tercile_frame = mlp_frame.loc[
            mlp_frame["intensity_tercile"] == tercile_name
        ].sort_values("predicted_bin_center")
        if tercile_frame.empty:
            continue
        x_values = tercile_frame["predicted_bin_center"].to_numpy(dtype=float)
        plot_mean_ci_line(
            axis,
            x_values,
            tercile_frame["observed_fill"].to_numpy(dtype=float),
            tercile_frame["observed_fill_ci_lo"].to_numpy(dtype=float),
            tercile_frame["observed_fill_ci_hi"].to_numpy(dtype=float),
            color=tercile_colors[tercile_name],
            label=f"{tercile_name} intensity tercile",
            linestyle="-",
            marker="o",
        )

    exponential_frame = summary_df.loc[
        summary_df["curve"] == "exponential"
    ].sort_values("predicted_bin_center")
    if not exponential_frame.empty:
        x_values = exponential_frame["predicted_bin_center"].to_numpy(dtype=float)
        plot_mean_ci_line(
            axis,
            x_values,
            exponential_frame["observed_fill"].to_numpy(dtype=float),
            exponential_frame["observed_fill_ci_lo"].to_numpy(dtype=float),
            exponential_frame["observed_fill_ci_hi"].to_numpy(dtype=float),
            color=color_darkorange,
            label="Exponential benchmark",
            linestyle="--",
            marker="s",
        )

    common_bounds = positive_log_bounds(
        np.concatenate(
            [
                summary_df["predicted_bin_center"].to_numpy(dtype=float),
                summary_df["observed_fill"].to_numpy(dtype=float),
            ]
        ),
        ci_low=summary_df["observed_fill_ci_lo"].to_numpy(dtype=float),
        ci_high=summary_df["observed_fill_ci_hi"].to_numpy(dtype=float),
    )
    lower_bound = common_bounds[0]
    upper_bound = max(1.0, common_bounds[1])
    axis.plot(
        [lower_bound, upper_bound],
        [lower_bound, upper_bound],
        color=color_dimgray,
        linestyle=":",
        linewidth=1.0,
        label="Perfect calibration",
    )
    axis.set_xlabel("Predicted fill probability")
    axis.set_ylabel("Observed fill rate")
    axis.set_title("Reliability by opposite-side MO-intensity tercile")
    axis.set_xlim(lower_bound, upper_bound)
    axis.set_ylim(lower_bound, upper_bound)
    axis.legend(loc="best")
    return fig


def plot_fill_by_depth_and_intensity(summary_df: pd.DataFrame) -> Figure:
    """Fill probability vs quote depth by opposite-side MO-intensity tercile."""
    fig, axis = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    axis.set_yscale("log")
    observed_by_depth = summary_df.groupby("delta_bin_center")["empirical_fill"].max()
    observed_centers = observed_by_depth.index[observed_by_depth > 0].to_numpy(dtype=float)
    all_centers = np.sort(summary_df["delta_bin_center"].unique().astype(float))
    if observed_centers.size:
        last_observed = observed_centers.max()
        following_centers = all_centers[all_centers > last_observed]
        if following_centers.size:
            display_max = following_centers.min()
        else:
            display_max = last_observed
        plot_frame = summary_df.loc[summary_df["delta_bin_center"] <= display_max]
    else:
        plot_frame = summary_df

    for tercile_name in ("low", "mid", "high"):
        tercile_frame = plot_frame.loc[
            plot_frame["intensity_tercile"] == tercile_name
        ].sort_values("delta_bin_center")
        if tercile_frame.empty:
            continue
        x_values = tercile_frame["delta_bin_center"].to_numpy(dtype=float)
        plot_mean_ci_line(
            axis,
            x_values,
            tercile_frame["empirical_fill"].to_numpy(dtype=float),
            tercile_frame["empirical_fill_ci_lo"].to_numpy(dtype=float),
            tercile_frame["empirical_fill_ci_hi"].to_numpy(dtype=float),
            color=tercile_colors[tercile_name],
            label=f"{tercile_name} tercile (empirical)",
            linestyle="-",
            marker="o",
        )
        axis.plot(
            x_values,
            tercile_frame["mlp_fill"].to_numpy(dtype=float),
            linestyle="--",
            marker="s",
            color=tercile_colors[tercile_name],
            label=f"{tercile_name} tercile (MLP)",
        )
        plot_ci_ribbon(
            axis,
            x_values,
            tercile_frame["mlp_fill"].to_numpy(dtype=float),
            tercile_frame["mlp_fill_ci_lo"].to_numpy(dtype=float),
            tercile_frame["mlp_fill_ci_hi"].to_numpy(dtype=float),
            color=tercile_colors[tercile_name],
            alpha=0.10,
        )
        axis.plot(
            x_values,
            tercile_frame["exponential_fill"].to_numpy(dtype=float),
            linestyle=":",
            marker=".",
            color=tercile_colors[tercile_name],
            alpha=0.85,
            label=f"{tercile_name} tercile (exponential)",
        )

    axis.set_xlabel("Quote depth δ (PLN)")
    axis.set_ylabel("Fill probability")
    axis.set_title("Fill probability vs depth by opposite-side MO-intensity tercile")
    apply_x_limits(axis, plot_frame["delta_bin_center"].to_numpy(dtype=float))
    axis.set_ylim(
        *positive_log_bounds(
            plot_frame[
                ["empirical_fill", "mlp_fill", "exponential_fill"]
            ].to_numpy().ravel(),
            ci_low=plot_frame[
                ["empirical_fill_ci_lo", "mlp_fill_ci_lo"]
            ].to_numpy().ravel(),
            ci_high=plot_frame[
                ["empirical_fill_ci_hi", "mlp_fill_ci_hi"]
            ].to_numpy().ravel(),
        )
    )
    axis.legend(loc="best", ncol=2, fontsize=8)
    return fig


def plot_depth_calibration(
    summary_df: pd.DataFrame,
    *,
    before_stage: Optional[str] = None,
    after_stage: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """Depth calibration: competition small multiples or corrected-FIFO before/after overlay."""
    if "stage" in summary_df.columns and "n_agents" not in summary_df.columns:
        return _plot_fifo_depth_calibration(
            summary_df,
            before_stage=before_stage,
            after_stage=after_stage,
            title=title,
        )
    return _plot_scenario_depth_calibration(summary_df, title=title)


def _plot_fifo_depth_calibration(
    summary_df: pd.DataFrame,
    *,
    before_stage: Optional[str],
    after_stage: Optional[str],
    title: Optional[str],
) -> Figure:
    stages = list(summary_df["stage"].dropna().unique())
    if before_stage is None:
        before_stage = stages[0]
    if after_stage is None:
        if len(stages) > 1:
            after_stage = stages[-1]
        else:
            after_stage = stages[0]

    fig, axis = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    stage_styles = {
        before_stage: ("-", color_dimgray, "o", "Placement-time phantom model"),
        after_stage: ("--", color_seagreen, "s", "Corrected exact-FIFO model"),
    }
    y_values = []
    y_ci_lo = []
    y_ci_hi = []
    x_values = []
    for stage_name, (linestyle, color, marker, label) in stage_styles.items():
        stage_frame = summary_df.loc[summary_df["stage"] == stage_name].sort_values("dbucket")
        if stage_frame.empty:
            continue
        x_axis = stage_frame["dbucket"].to_numpy(dtype=float)
        x_values.extend(x_axis.tolist())
        for prefix, line_label in (("pred", "predicted"), ("realized", "realized")):
            center = stage_frame[f"{prefix}_mean"].to_numpy(dtype=float)
            ci_lo = stage_frame[f"{prefix}_ci_lo"].to_numpy(dtype=float)
            ci_hi = stage_frame[f"{prefix}_ci_hi"].to_numpy(dtype=float)
            y_values.extend(center.tolist())
            y_ci_lo.extend(ci_lo.tolist())
            y_ci_hi.extend(ci_hi.tolist())
            plot_mean_ci_line(
                axis,
                x_axis,
                center,
                ci_lo,
                ci_hi,
                color=color,
                label=f"{label} ({line_label})",
                linestyle=linestyle,
                marker=marker,
            )

    axis.set_xlabel("Quoted depth δ (ticks behind mid)")
    axis.set_ylabel("Fill probability per 1 s window")
    axis.set_title(title or "Corrected exact-FIFO depth calibration")
    apply_x_limits(axis, x_values)
    apply_probability_y_limits(axis, y_values, ci_low=y_ci_lo, ci_high=y_ci_hi)
    axis.legend(loc="best")
    return fig


def _plot_scenario_depth_calibration(
    summary_df: pd.DataFrame,
    *,
    title: Optional[str],
) -> Figure:
    n_values = sorted(summary_df["n_agents"].dropna().unique())
    n_columns = max(1, len(n_values))
    n_rows = 1
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.4 * n_columns, 4.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    y_values = []
    y_ci_lo = []
    y_ci_hi = []
    x_values = []

    for panel_index, n_agents in enumerate(n_values):
        axis = axes_flat[panel_index]
        panel_color = population_color_map(n_values)[int(n_agents)]
        panel_frame = summary_df.loc[summary_df["n_agents"] == n_agents].sort_values("dbucket")
        x_axis = panel_frame["dbucket"].to_numpy(dtype=float)
        x_values.extend(x_axis.tolist())
        realized = panel_frame["realized_mean"].to_numpy(dtype=float)
        realized_lo = panel_frame["realized_ci_lo"].to_numpy(dtype=float)
        realized_hi = panel_frame["realized_ci_hi"].to_numpy(dtype=float)
        predicted = panel_frame["pred_mean"].to_numpy(dtype=float)
        predicted_lo = panel_frame["pred_ci_lo"].to_numpy(dtype=float)
        predicted_hi = panel_frame["pred_ci_hi"].to_numpy(dtype=float)
        y_values.extend(np.concatenate([realized, predicted]).tolist())
        y_ci_lo.extend(np.concatenate([realized_lo, predicted_lo]).tolist())
        y_ci_hi.extend(np.concatenate([realized_hi, predicted_hi]).tolist())
        plot_mean_ci_line(
            axis,
            x_axis,
            realized,
            realized_lo,
            realized_hi,
            color=panel_color,
            label="Realized",
            linestyle="-",
            marker="o",
        )
        plot_mean_ci_line(
            axis,
            x_axis,
            predicted,
            predicted_lo,
            predicted_hi,
            color=color_dimgray,
            label="Predicted (NN)",
            linestyle="--",
            marker="s",
        )
        axis.set_title(f"N = {int(n_agents)}")
        axis.set_xlabel("Quoted depth δ (ticks behind mid)")
        axis.set_ylabel("Fill probability per 1 s window")
        axis.legend(loc="best", fontsize=8)

    for panel_index in range(len(n_values), len(axes_flat)):
        axes_flat[panel_index].axis("off")

    if y_values:
        for axis in axes_flat[: len(n_values)]:
            apply_x_limits(axis, x_values)
            apply_probability_y_limits(
                axis,
                y_values,
                ci_low=y_ci_lo,
                ci_high=y_ci_hi,
            )

    fig.suptitle(title or "Predicted vs realized fill probability by depth", fontsize=12)
    return fig


def plot_adaptation_pnl(summary_df: pd.DataFrame) -> Figure:
    """Realized PnL trajectory with run observations and bootstrap intervals."""
    rows = []
    for adaptation_step, step_frame in summary_df.groupby("adaptation_step", sort=True):
        mean_pnl, ci_lo, ci_hi = run_cluster_bootstrap_interval(
            step_frame["realized_pnl"].to_numpy(dtype=float),
            n_bootstrap=2000,
            bootstrap_seed=17 + int(adaptation_step),
        )
        row = {
            "adaptation_step": int(adaptation_step),
            "stage": str(step_frame["stage"].iloc[0]),
            "mean_pnl": mean_pnl,
            "pnl_ci_lo": ci_lo,
            "pnl_ci_hi": ci_hi,
        }
        rows.append(row)

    ordered = pd.DataFrame(rows).sort_values("adaptation_step").reset_index(drop=True)
    x_values = np.arange(len(ordered), dtype=float)
    labels = ordered["stage"].tolist()
    fig, axis = plt.subplots(figsize=(10.0, 5.2), constrained_layout=True)

    for step_index, step_value in enumerate(ordered["adaptation_step"].tolist()):
        run_rows = summary_df.loc[
            summary_df["adaptation_step"] == step_value,
            "realized_pnl",
        ].to_numpy(dtype=float)
        if run_rows.size > 1:
            jitter = np.linspace(-0.10, 0.10, run_rows.size)
        else:
            jitter = np.zeros(1)
        if step_index == 0:
            simulation_label = "Individual simulation"
        else:
            simulation_label = None
        axis.scatter(
            np.full(run_rows.size, step_index) + jitter,
            run_rows,
            s=20,
            alpha=0.35,
            color=color_navy,
            label=simulation_label,
        )

    mean_pnl = ordered["mean_pnl"].to_numpy(dtype=float)
    pnl_ci_lo = ordered["pnl_ci_lo"].to_numpy(dtype=float)
    pnl_ci_hi = ordered["pnl_ci_hi"].to_numpy(dtype=float)
    axis.errorbar(
        x_values,
        mean_pnl,
        yerr=[mean_pnl - pnl_ci_lo, pnl_ci_hi - mean_pnl],
        marker="o",
        color=color_navy,
        capsize=4,
        linewidth=1.8,
        label="Mean ± 95% CI",
    )
    axis.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axis.set_title("Realized PnL during supervised fill-belief adaptation")
    axis.set_xlabel("Evaluated fill-belief checkpoint")
    axis.set_ylabel("Realized PnL per simulation (PLN)")
    axis.set_xticks(x_values, labels, rotation=25, ha="right")
    apply_y_limits(
        axis,
        np.concatenate(
            [mean_pnl, summary_df["realized_pnl"].to_numpy(dtype=float)]
        ),
        ci_low=pnl_ci_lo,
        ci_high=pnl_ci_hi,
    )
    axis.legend(loc="best")
    return fig


def plot_quoted_depth_by_population(summary_df: pd.DataFrame, *, title: Optional[str] = None) -> Figure:
    """Mean quoted depth vs number of market makers with 95% confidence intervals."""
    ordered = summary_df.sort_values("n_agents")
    x_values = ordered["n_agents"].to_numpy(dtype=float)
    mean_depth = ordered["mean_depth"].to_numpy(dtype=float)
    ci_lo = ordered["ci_lo"].to_numpy(dtype=float)
    ci_hi = ordered["ci_hi"].to_numpy(dtype=float)
    color_lookup = population_color_map(ordered["n_agents"].tolist())

    fig, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    axis.plot(
        x_values,
        mean_depth,
        color=color_dimgray,
        linewidth=1.4,
        label="Population trend",
    )
    for n_agents, x_value, center, lower, upper in zip(
        ordered["n_agents"],
        x_values,
        mean_depth,
        ci_lo,
        ci_hi,
    ):
        axis.errorbar(
            x_value,
            center,
            yerr=[[center - lower], [upper - center]],
            color=color_lookup[int(n_agents)],
            marker="o",
            linestyle="none",
            capsize=4,
            label=f"N = {int(n_agents)}",
        )
    axis.set_xticks(x_values)
    axis.set_xlabel("Number of market makers (N)")
    axis.set_ylabel("Mean quoted depth δ (ticks)")
    axis.set_title(title or "Quoted depth vs competition (95% CI)")
    apply_x_limits(axis, x_values)
    apply_y_limits(axis, mean_depth, ci_low=ci_lo, ci_high=ci_hi)
    axis.legend(loc="best")
    return fig


def plot_cross_stage_quoted_depth(summary_df: pd.DataFrame, *, title: Optional[str] = None) -> Figure:
    """Quoted depth vs N for multiple pipeline stages on one axis."""
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    y_values = []
    y_ci_lo = []
    y_ci_hi = []
    x_values = []
    for stage_index, (stage_name, stage_frame) in enumerate(summary_df.groupby("stage")):
        ordered = stage_frame.sort_values("n_agents")
        x_axis = ordered["n_agents"].to_numpy(dtype=float)
        mean_depth = ordered["mean_depth"].to_numpy(dtype=float)
        ci_lo = ordered["ci_lo"].to_numpy(dtype=float)
        ci_hi = ordered["ci_hi"].to_numpy(dtype=float)
        color = stage_colors[stage_index % len(stage_colors)]
        if stage_index == 0:
            linestyle = "-"
        else:
            linestyle = "--"
        if stage_index % 2 == 0:
            marker = "o"
        else:
            marker = "s"
        plot_mean_ci_line(
            axis,
            x_axis,
            mean_depth,
            ci_lo,
            ci_hi,
            color=color,
            label=str(stage_name),
            linestyle=linestyle,
            marker=marker,
        )
        x_values.extend(x_axis.tolist())
        y_values.extend(mean_depth.tolist())
        y_ci_lo.extend(ci_lo.tolist())
        y_ci_hi.extend(ci_hi.tolist())

    axis.set_xlabel("Number of market makers (N)")
    axis.set_ylabel("Mean quoted depth δ (ticks)")
    axis.set_title(title or "Cross-stage quoted depth vs competition (95% CI)")
    apply_x_limits(axis, x_values)
    apply_y_limits(axis, y_values, ci_low=y_ci_lo, ci_high=y_ci_hi)
    axis.legend(loc="best")
    return fig


def plot_cross_stage_pnl(summary_df: pd.DataFrame, *, title: Optional[str] = None) -> Figure:
    """Mean per-agent PnL vs population size across benchmark stages."""
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    y_values = []
    y_ci_lo = []
    y_ci_hi = []
    x_values = []
    for stage_index, (stage_name, stage_frame) in enumerate(summary_df.groupby("stage")):
        ordered = stage_frame.sort_values("n_agents")
        x_axis = ordered["n_agents"].to_numpy(dtype=float)
        mean_pnl = ordered["mean_pnl"].to_numpy(dtype=float)
        ci_lo = ordered["ci_lo"].to_numpy(dtype=float)
        ci_hi = ordered["ci_hi"].to_numpy(dtype=float)
        plot_mean_ci_line(
            axis,
            x_axis,
            mean_pnl,
            ci_lo,
            ci_hi,
            color=stage_colors[stage_index % len(stage_colors)],
            label=str(stage_name),
            linestyle=("-", "--", "-.")[stage_index % 3],
            marker=("o", "s", "D")[stage_index % 3],
        )
        x_values.extend(x_axis.tolist())
        y_values.extend(mean_pnl.tolist())
        y_ci_lo.extend(ci_lo.tolist())
        y_ci_hi.extend(ci_hi.tolist())

    axis.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    axis.set_xlabel("Number of market makers (N)")
    axis.set_ylabel("Mean realized PnL per agent (PLN)")
    axis.set_title(title or "Realized PnL across benchmark stages")
    apply_x_limits(axis, x_values)
    apply_y_limits(axis, y_values, ci_low=y_ci_lo, ci_high=y_ci_hi)
    axis.legend(loc="best")
    return fig


def plot_estimated_fill_by_depth(summary_df: pd.DataFrame, *, title: Optional[str] = None) -> Figure:
    """Estimated NN fill probability vs quoted depth, overlaid by population size."""
    fig, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    color_lookup = population_color_map(summary_df["n_agents"].tolist())
    y_values = []
    y_ci_lo = []
    y_ci_hi = []
    x_values = []
    for n_agents, group in summary_df.groupby("n_agents"):
        ordered = group.sort_values("dbucket")
        x_axis = ordered["dbucket"].to_numpy(dtype=float)
        predicted = ordered["pred_mean"].to_numpy(dtype=float)
        ci_lo = ordered["pred_ci_lo"].to_numpy(dtype=float)
        ci_hi = ordered["pred_ci_hi"].to_numpy(dtype=float)
        plot_mean_ci_line(
            axis,
            x_axis,
            predicted,
            ci_lo,
            ci_hi,
            color=color_lookup[int(n_agents)],
            label=f"N = {int(n_agents)}",
            linestyle="-",
            marker="o",
        )
        x_values.extend(x_axis.tolist())
        y_values.extend(predicted.tolist())
        y_ci_lo.extend(ci_lo.tolist())
        y_ci_hi.extend(ci_hi.tolist())

    axis.set_xlabel("Quoted depth δ (ticks behind mid)")
    axis.set_ylabel("Estimated fill probability per 1 s window")
    axis.set_title(title or "Estimated (NN) fill probability vs depth")
    apply_x_limits(axis, x_values)
    apply_probability_y_limits(axis, y_values, ci_low=y_ci_lo, ci_high=y_ci_hi)
    axis.legend(loc="best", title="Population size")
    return fig


def plot_pnl_distributions(summary_df: pd.DataFrame, *, title: Optional[str] = None) -> Figure:
    """Simulation-level per-agent PnL distributions by population size."""
    n_values = sorted(summary_df["n_agents"].dropna().unique())
    color_lookup = population_color_map(n_values)
    fig, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    box_rows = []
    y_values = []
    for n_agents in n_values:
        pnl_values = summary_df.loc[summary_df["n_agents"] == n_agents, "per_agent_pnl"].to_numpy(dtype=float)
        y_values.extend(pnl_values.tolist())
        box_rows.append(
            {
                "med": float(np.median(pnl_values)),
                "mean": float(np.mean(pnl_values)),
                "q1": float(np.percentile(pnl_values, 25)),
                "q3": float(np.percentile(pnl_values, 75)),
                "whislo": float(np.min(pnl_values)),
                "whishi": float(np.max(pnl_values)),
                "fliers": [],
            }
        )

    boxplot = axis.bxp(
        box_rows,
        positions=np.arange(1, len(n_values) + 1),
        showmeans=True,
        patch_artist=True,
        widths=0.6,
    )
    for patch, n_agents in zip(boxplot["boxes"], n_values):
        patch.set_facecolor(color_lookup[int(n_agents)])
        patch.set_alpha(0.25)
        patch.set_edgecolor(color_lookup[int(n_agents)])
    axis.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axis.set_xticks(np.arange(1, len(n_values) + 1), [str(int(n)) for n in n_values])
    axis.set_xlabel("Number of market makers (N)")
    axis.set_ylabel("Mean realized PnL per simulation (PLN)")
    axis.set_title(title or "PnL distribution vs competition")
    apply_y_limits(axis, y_values)
    return fig


def plot_inventory_dispersion(
    summary_df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    lot_size: float = 10_000.0,
) -> Figure:
    """Absolute inventory dispersion by population size."""
    shares_per_lot = float(lot_size)
    n_values = sorted(summary_df["n_agents"].dropna().unique())
    color_lookup = population_color_map(n_values)
    fig, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    box_rows = []
    y_values = []
    for n_agents in n_values:
        inventory_values = (
            summary_df.loc[summary_df["n_agents"] == n_agents, "abs_inventory"].to_numpy(
                dtype=float
            )
            / shares_per_lot
        )
        y_values.extend(inventory_values.tolist())
        box_rows.append(
            {
                "med": float(np.median(inventory_values)),
                "q1": float(np.percentile(inventory_values, 25)),
                "q3": float(np.percentile(inventory_values, 75)),
                "whislo": float(np.min(inventory_values)),
                "whishi": float(np.max(inventory_values)),
                "fliers": [],
            }
        )

    boxplot = axis.bxp(
        box_rows,
        positions=np.arange(1, len(n_values) + 1),
        showfliers=False,
        patch_artist=True,
        widths=0.6,
    )
    for patch, n_agents in zip(boxplot["boxes"], n_values):
        patch.set_facecolor(color_lookup[int(n_agents)])
        patch.set_alpha(0.25)
        patch.set_edgecolor(color_lookup[int(n_agents)])
    axis.set_xticks(np.arange(1, len(n_values) + 1), [str(int(n)) for n in n_values])
    axis.set_xlabel("Number of market makers (N)")
    axis.set_ylabel("|Inventory| (lots)")
    axis.set_title(title or "Inventory dispersion vs competition")
    apply_y_limits(axis, y_values, ymin_floor=0.0)
    return fig


def plot_inventory_depth(
    summary_df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    lot_size: float = 10_000.0,
) -> Figure:
    """Ask and bid quote-depth curves vs inventory with bootstrap confidence ribbons."""
    shares_per_lot = float(lot_size)
    n_values = sorted(summary_df["n_agents"].dropna().unique())
    color_lookup = population_color_map(n_values)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True, constrained_layout=True)
    y_values = []
    y_ci_lo = []
    y_ci_hi = []
    x_values = []

    for axis, side_value in zip(axes, (2, 1)):
        side_label = side_panel_label(side_value)
        side_frame = summary_df.loc[summary_df["side"] == side_value]
        for n_agents in n_values:
            curve = side_frame.loc[side_frame["n_agents"] == n_agents].sort_values("bin_center")
            if curve.empty:
                continue
            x_axis = curve["bin_center"].to_numpy(dtype=float) / shares_per_lot
            mean_depth = curve["mean_depth"].to_numpy(dtype=float)
            ci_lo = curve["ci_lo"].to_numpy(dtype=float)
            ci_hi = curve["ci_hi"].to_numpy(dtype=float)
            plot_mean_ci_line(
                axis,
                x_axis,
                mean_depth,
                ci_lo,
                ci_hi,
                color=color_lookup[int(n_agents)],
                label=f"N = {int(n_agents)}",
                linestyle="-",
                marker="o",
            )
            x_values.extend(x_axis.tolist())
            y_values.extend(mean_depth.tolist())
            y_ci_lo.extend(ci_lo.tolist())
            y_ci_hi.extend(ci_hi.tolist())
        axis.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
        axis.set_xlabel("Inventory level (lots)")
        axis.set_title(f"Average {side_label} quote depth vs inventory")
        axis.legend(loc="best", fontsize=8, title="Population size")

    axes[0].set_ylabel("Quoted depth δ (ticks)")
    if y_values:
        for axis in axes:
            apply_x_limits(axis, x_values)
            apply_y_limits(axis, y_values, ci_low=y_ci_lo, ci_high=y_ci_hi)
    fig.suptitle(title or "Quote depth vs inventory", fontsize=12)
    return fig


def plot_inventory_depth_comparison(
    summary_df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    lot_size: float = 10_000.0,
) -> Figure:
    """Part 5 six-panel figure with inventory-depth curves per panel.

    Ask/bid panels in the same row share a y-axis; each population size ``N``
    gets its own depth range so competitive compression remains readable.
    """
    shares_per_lot = float(lot_size)
    panel_ns = sorted({int(value) for value in summary_df["n_agents"].unique() if int(value) > 1})
    n_rows = max(1, len(panel_ns))
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(13.5, 4.3 * n_rows),
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )

    for row_index, n_agents in enumerate(panel_ns):
        ask_axis = axes[row_index, 0]
        bid_axis = axes[row_index, 1]
        row_y_values = []
        row_y_ci_lo = []
        row_y_ci_hi = []
        row_x_values = []
        for side_value, axis in ((2, ask_axis), (1, bid_axis)):
            side_label = side_panel_label(side_value)
            panel_frame = summary_df.loc[
                (summary_df["n_agents"] == n_agents) & (summary_df["side"] == side_value)
            ]
            for curve_label, curve_frame in panel_frame.groupby("curve_label"):
                ordered = curve_frame.sort_values("bin_center")
                if ordered.empty:
                    continue
                linestyle, color, marker = comparison_styles.get(
                    str(curve_label),
                    ("-", color_dimgray, "o"),
                )
                x_axis = ordered["bin_center"].to_numpy(dtype=float) / shares_per_lot
                mean_depth = ordered["mean_depth"].to_numpy(dtype=float)
                axis.plot(
                    x_axis,
                    mean_depth,
                    linestyle=linestyle,
                    marker=marker,
                    color=color,
                    linewidth=1.8,
                    markersize=4,
                    label=str(curve_label).replace("_", " "),
                )
                if {"ci_lo", "ci_hi"}.issubset(ordered.columns):
                    ci_lo = ordered["ci_lo"].to_numpy(dtype=float)
                    ci_hi = ordered["ci_hi"].to_numpy(dtype=float)
                    plot_ci_ribbon(
                        axis,
                        x_axis,
                        mean_depth,
                        ci_lo,
                        ci_hi,
                        color=color,
                    )
                    row_y_ci_lo.extend(ci_lo.tolist())
                    row_y_ci_hi.extend(ci_hi.tolist())
                row_x_values.extend(x_axis.tolist())
                row_y_values.extend(mean_depth.tolist())
            axis.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
            axis.set_title(f"N = {n_agents} — average {side_label} quote depth vs inventory")
            axis.set_xlabel("Inventory level (lots)")
            if row_index == 0:
                axis.legend(loc="upper right", fontsize=8)
        axes[row_index, 0].set_ylabel("Quoted depth δ (ticks)")
        if row_x_values:
            for axis in (ask_axis, bid_axis):
                apply_x_limits(axis, row_x_values)
                apply_y_limits(
                    axis,
                    row_y_values,
                    ci_low=row_y_ci_lo or None,
                    ci_high=row_y_ci_hi or None,
                )

    fig.suptitle(title or "Optimal quote depth vs inventory", fontsize=12)
    return fig


def plot_depth_differences(
    summary_df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    lot_size: float = 10_000.0,
) -> Figure:
    """Six-panel paired quote-depth differences with run-cluster intervals."""
    shares_per_lot = float(lot_size)
    panel_ns = sorted(summary_df["n_agents"].dropna().unique())
    n_rows = max(1, len(panel_ns))
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(13.0, 4.0 * n_rows),
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )

    for row_index, n_agents in enumerate(panel_ns):
        row_y_values = []
        row_y_ci_lo = []
        row_y_ci_hi = []
        row_x_values = []
        for side_value, axis in ((2, axes[row_index, 0]), (1, axes[row_index, 1])):
            side_label = side_panel_label(side_value)
            panel_frame = summary_df.loc[
                (summary_df["n_agents"] == n_agents) & (summary_df["side"] == side_value)
            ].sort_values("bin_center")
            if panel_frame.empty:
                axis.set_title(f"N = {int(n_agents)} — {side_label}: no paired data")
                continue
            comparison_column = (
                "comparison_label"
                if "comparison_label" in panel_frame.columns
                else None
            )
            comparison_groups = (
                panel_frame.groupby(comparison_column)
                if comparison_column is not None
                else [("Coordinated − independent", panel_frame)]
            )
            for comparison_index, (comparison_label, comparison_frame) in enumerate(
                comparison_groups
            ):
                ordered = comparison_frame.sort_values("bin_center")
                x_axis = ordered["bin_center"].to_numpy(dtype=float) / shares_per_lot
                mean_difference = ordered["mean_difference"].to_numpy(dtype=float)
                ci_lo = ordered["ci_lo"].to_numpy(dtype=float)
                ci_hi = ordered["ci_hi"].to_numpy(dtype=float)
                plot_mean_ci_line(
                    axis,
                    x_axis,
                    mean_difference,
                    ci_lo,
                    ci_hi,
                    color=(color_darkorange, color_seagreen)[comparison_index % 2],
                    label=str(comparison_label),
                    linestyle=("-", "--")[comparison_index % 2],
                    marker=("o", "s")[comparison_index % 2],
                )
                row_x_values.extend(x_axis.tolist())
                row_y_values.extend(mean_difference.tolist())
                row_y_ci_lo.extend(ci_lo.tolist())
                row_y_ci_hi.extend(ci_hi.tolist())
            axis.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
            axis.set_title(f"N = {int(n_agents)} — {side_label} depth differences")
            axis.set_xlabel("Inventory level (lots)")
            if row_index == 0:
                axis.legend(loc="best", fontsize=8)
        axes[row_index, 0].set_ylabel("Δ quoted depth δ (ticks)")
        if row_x_values:
            for axis in axes[row_index]:
                apply_x_limits(axis, row_x_values)
                apply_y_limits(
                    axis,
                    row_y_values,
                    ci_low=row_y_ci_lo,
                    ci_high=row_y_ci_hi,
                )

    fig.suptitle(title or "Paired quote-depth differences from independent competition", fontsize=12)
    return fig


__all__ = [
    "apply_x_limits",
    "apply_y_limits",
    "dynamic_padded_bounds",
    "plot_adaptation_pnl",
    "plot_cross_stage_quoted_depth",
    "plot_cross_stage_pnl",
    "plot_depth_calibration",
    "plot_depth_differences",
    "plot_estimated_fill_by_depth",
    "plot_fill_by_depth_and_intensity",
    "plot_fill_by_intensity",
    "plot_inventory_depth",
    "plot_inventory_depth_comparison",
    "plot_inventory_dispersion",
    "plot_pnl_distributions",
    "plot_quoted_depth_by_population",
    "plot_reliability_by_intensity",
    "save_report_figure",
    "set_report_style",
]

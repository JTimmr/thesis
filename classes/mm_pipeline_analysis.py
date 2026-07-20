"""Conservative run-cluster bootstrap summaries for MM competition shards."""

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def load_competition_shards(competition_root: PathLike, log_names: Sequence[str] = ("fill_probe", "state", "summary")) -> Dict[str, pd.DataFrame]:
    """Load parquet competition shards written by ``write_competition_log_shards``."""
    root = Path(competition_root)
    setup_dirs = sorted(root.glob("setup_*"))
    shards: Dict[str, pd.DataFrame] = {}

    for log_name in log_names:
        frames = []
        for setup_dir in setup_dirs:
            if log_name == "summary":
                summary_path = setup_dir / "summary.parquet"
                if summary_path.is_file():
                    frames.append(pd.read_parquet(summary_path))
            else:
                for run_path in sorted(setup_dir.glob(f"{log_name}_run*.parquet")):
                    frames.append(pd.read_parquet(run_path))
        if frames:
            shards[log_name] = pd.concat(frames, ignore_index=True)
        else:
            shards[log_name] = pd.DataFrame()

    return shards


def add_fill_probe_derived_columns(fill_probe_df: pd.DataFrame, max_depth_bucket: float = 20.0) -> pd.DataFrame:
    """Add realized fill fraction and rounded depth buckets used in calibration plots."""
    frame = fill_probe_df.copy()
    frame["realized"] = frame["filled_qty"] / frame["size"].clip(lower=1)
    frame["dbucket"] = frame["delta_ticks"].round().clip(lower=0, upper=max_depth_bucket)
    return frame


def build_run_level_metrics(
    fill_probe_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    tick_size: float = 0.05,
) -> pd.DataFrame:
    """Collapse probe rows and per-agent summaries to one row per (n_agents, run_id).

    Simulator mark-to-market is stored in native tick-index units. Multiply by
    ``tick_size`` (``0.05`` = ``1/20`` for KGHM) so reported PnL is in PLN.
    """
    fill_metrics = pd.DataFrame(columns=["n_agents", "run_id"])
    if not fill_probe_df.empty:
        fill_metrics = (
            fill_probe_df.groupby(["n_agents", "run_id"], as_index=False)
            .agg(
                mean_quoted_depth=("delta_ticks", "mean"),
                mean_predicted_fill=("pred_h", "mean"),
                mean_realized_fill=("realized", "mean"),
                n_fill_probe_rows=("realized", "size"),
            )
        )

    summary_metrics = pd.DataFrame(columns=["n_agents", "run_id"])
    if not summary_df.empty:
        summary_metrics = (
            summary_df.groupby(["n_agents", "run_id"], as_index=False)
            .agg(
                per_agent_pnl=("realized_pnl", "mean"),
                total_pnl=("realized_pnl", "sum"),
                n_agents_in_run=("agent_id", "size"),
            )
        )
        price_scale = float(tick_size)
        summary_metrics["per_agent_pnl"] *= price_scale
        summary_metrics["total_pnl"] *= price_scale

    if fill_metrics.empty and summary_metrics.empty:
        return pd.DataFrame(
            columns=[
                "n_agents", "run_id", "mean_quoted_depth", "mean_predicted_fill",
                "mean_realized_fill", "n_fill_probe_rows", "per_agent_pnl", "total_pnl",
                "n_agents_in_run",
            ]
        )
    if fill_metrics.empty:
        return summary_metrics
    if summary_metrics.empty:
        return fill_metrics
    return fill_metrics.merge(summary_metrics, on=["n_agents", "run_id"], how="outer")


def run_cluster_bootstrap_interval(
    run_values: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap mean and percentile CI treating each run value as one cluster."""
    values = np.asarray(run_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan

    point_estimate = float(values.mean())
    if values.size == 1:
        return point_estimate, point_estimate, point_estimate

    alpha = 1.0 - confidence_level
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap_means = np.empty(int(n_bootstrap), dtype=float)
    for draw_idx in range(int(n_bootstrap)):
        resampled = rng.choice(values, size=values.size, replace=True)
        bootstrap_means[draw_idx] = resampled.mean()

    ci_lo = float(np.percentile(bootstrap_means, lower_q))
    ci_hi = float(np.percentile(bootstrap_means, upper_q))
    return point_estimate, ci_lo, ci_hi


def aggregate_quoted_depth_vs_n(
    fill_probe_df: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Mean quoted depth by N with run-cluster bootstrap confidence intervals."""
    if fill_probe_df.empty:
        return pd.DataFrame(columns=["n_agents", "mean_depth", "ci_lo", "ci_hi", "n_runs"])

    run_depth = (
        fill_probe_df.groupby(["n_agents", "run_id"], as_index=False)["delta_ticks"]
        .mean()
        .rename(columns={"delta_ticks": "run_mean_depth"})
    )

    rows = []
    for n_agents, group in run_depth.groupby("n_agents"):
        mean_depth, ci_lo, ci_hi = run_cluster_bootstrap_interval(
            group["run_mean_depth"].to_numpy(),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed + int(n_agents),
        )
        rows.append({
            "n_agents": int(n_agents),
            "mean_depth": mean_depth,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_runs": int(group["run_id"].nunique()),
        })

    return pd.DataFrame(rows).sort_values("n_agents").reset_index(drop=True)


def aggregate_depth_calibration(
    fill_probe_df: pd.DataFrame,
    *,
    min_count_per_bucket: int = 10,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Predicted and realized fill rates by depth bucket with run-cluster intervals."""
    if fill_probe_df.empty:
        return pd.DataFrame(columns=[
            "n_agents", "dbucket", "pred_mean", "pred_ci_lo", "pred_ci_hi",
            "realized_mean", "realized_ci_lo", "realized_ci_hi", "n_runs", "total_count",
        ])

    run_bucket = (
        fill_probe_df.groupby(["n_agents", "run_id", "dbucket"], as_index=False)
        .agg(
            pred_mean=("pred_h", "mean"),
            realized_mean=("realized", "mean"),
            bucket_count=("realized", "size"),
        )
    )

    rows = []
    for (n_agents, dbucket), group in run_bucket.groupby(["n_agents", "dbucket"]):
        total_count = int(group["bucket_count"].sum())
        if total_count < int(min_count_per_bucket):
            continue

        seed_offset = int(n_agents) * 1000 + int(dbucket)
        pred_mean, pred_ci_lo, pred_ci_hi = run_cluster_bootstrap_interval(
            group["pred_mean"].to_numpy(),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed + seed_offset,
        )
        realized_mean, realized_ci_lo, realized_ci_hi = run_cluster_bootstrap_interval(
            group["realized_mean"].to_numpy(),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed + seed_offset + 1,
        )
        rows.append({
            "n_agents": int(n_agents),
            "dbucket": float(dbucket),
            "pred_mean": pred_mean,
            "pred_ci_lo": pred_ci_lo,
            "pred_ci_hi": pred_ci_hi,
            "realized_mean": realized_mean,
            "realized_ci_lo": realized_ci_lo,
            "realized_ci_hi": realized_ci_hi,
            "n_runs": int(group["run_id"].nunique()),
            "total_count": total_count,
        })

    return pd.DataFrame(rows).sort_values(["n_agents", "dbucket"]).reset_index(drop=True)


def aggregate_adaptation_depth_calibration(
    run_depth_df: pd.DataFrame,
    *,
    min_count_per_bucket: int = 10,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Aggregate per-run adaptation depth tables with run-cluster intervals."""
    rows = []
    grouping_columns = ["adaptation_step", "stage", "dbucket"]
    for group_keys, group in run_depth_df.groupby(grouping_columns):
        total_count = int(group["cnt"].sum())
        if total_count < int(min_count_per_bucket):
            continue
        adaptation_step, stage, dbucket = group_keys
        seed_offset = int(adaptation_step) * 1000 + int(dbucket)
        pred_mean, pred_ci_lo, pred_ci_hi = run_cluster_bootstrap_interval(
            group["pred"].to_numpy(),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed + seed_offset,
        )
        realized_mean, realized_ci_lo, realized_ci_hi = run_cluster_bootstrap_interval(
            group["real"].to_numpy(),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed + seed_offset + 1,
        )
        rows.append(
            {
                "adaptation_step": int(adaptation_step),
                "stage": str(stage),
                "dbucket": float(dbucket),
                "pred_mean": pred_mean,
                "pred_ci_lo": pred_ci_lo,
                "pred_ci_hi": pred_ci_hi,
                "realized_mean": realized_mean,
                "realized_ci_lo": realized_ci_lo,
                "realized_ci_hi": realized_ci_hi,
                "n_runs": int(group["run_id"].nunique()),
                "total_count": total_count,
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["adaptation_step", "dbucket"])
        .reset_index(drop=True)
    )


def assign_inventory_bins(inventory_values: np.ndarray, inventory_bin_edges: Sequence[float]) -> np.ndarray:
    """Map inventory levels to fixed bin indices shared across runs."""
    edges = np.asarray(inventory_bin_edges, dtype=float)
    return np.clip(np.digitize(inventory_values, edges[1:-1]), 0, edges.size - 2)


def aggregate_inventory_quote_depth(
    fill_probe_df: pd.DataFrame,
    inventory_bin_edges: Sequence[float],
    *,
    side: Optional[int] = None,
    min_count_per_bin: int = 50,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Quote depth vs fixed inventory bins with per-run means and bootstrap intervals."""
    if fill_probe_df.empty:
        return pd.DataFrame(columns=[
            "n_agents", "side", "inventory_bin", "bin_center", "mean_depth",
            "ci_lo", "ci_hi", "n_runs", "total_count",
        ])

    edges = np.asarray(inventory_bin_edges, dtype=float)
    frame = fill_probe_df.copy()
    if side is not None:
        frame = frame.loc[frame["side"] == int(side)]
    if frame.empty:
        return pd.DataFrame(columns=[
            "n_agents", "side", "inventory_bin", "bin_center", "mean_depth",
            "ci_lo", "ci_hi", "n_runs", "total_count",
        ])

    frame["inventory_bin"] = assign_inventory_bins(frame["inventory"].to_numpy(dtype=float), edges)
    frame["bin_center"] = 0.5 * (edges[frame["inventory_bin"]] + edges[frame["inventory_bin"] + 1])

    run_bin = (
        frame.groupby(["n_agents", "side", "inventory_bin", "bin_center", "run_id"], as_index=False)
        .agg(run_mean_depth=("delta_ticks", "mean"), bin_count=("delta_ticks", "size"))
    )

    rows = []
    for (n_agents, side_value, inventory_bin, bin_center), group in run_bin.groupby(
        ["n_agents", "side", "inventory_bin", "bin_center"]
    ):
        total_count = int(group["bin_count"].sum())
        if total_count < int(min_count_per_bin):
            continue

        seed_offset = int(n_agents) * 1000 + int(inventory_bin) * 10 + int(side_value)
        mean_depth, ci_lo, ci_hi = run_cluster_bootstrap_interval(
            group["run_mean_depth"].to_numpy(),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed + seed_offset,
        )
        rows.append({
            "n_agents": int(n_agents),
            "side": int(side_value),
            "inventory_bin": int(inventory_bin),
            "bin_center": float(bin_center),
            "mean_depth": mean_depth,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_runs": int(group["run_id"].nunique()),
            "total_count": total_count,
        })

    return pd.DataFrame(rows).sort_values(["n_agents", "side", "inventory_bin"]).reset_index(drop=True)


def paired_scenario_bootstrap_difference(
    left_run_metrics: pd.DataFrame,
    right_run_metrics: pd.DataFrame,
    value_column: str,
    *,
    match_columns: Sequence[str] = ("n_agents", "run_id"),
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Paired run-cluster bootstrap of right minus left on matched scenarios."""
    match_cols = list(match_columns)
    paired = left_run_metrics.merge(
        right_run_metrics,
        on=match_cols,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if paired.empty:
        return pd.DataFrame(columns=[
            "n_agents", "mean_difference", "ci_lo", "ci_hi", "n_pairs",
        ])

    left_col = f"{value_column}_left"
    right_col = f"{value_column}_right"
    paired["paired_difference"] = paired[right_col] - paired[left_col]

    rows = []
    if "n_agents" in paired.columns:
        group_col = "n_agents"
    else:
        group_col = match_cols[0]
    for group_key, group in paired.groupby(group_col):
        mean_difference, ci_lo, ci_hi = run_cluster_bootstrap_interval(
            group["paired_difference"].to_numpy(),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed + int(group_key),
        )
        rows.append({
            group_col: int(group_key),
            "mean_difference": mean_difference,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_pairs": int(group.shape[0]),
        })

    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def paired_inventory_depth_difference(
    left_fill_probe: pd.DataFrame,
    right_fill_probe: pd.DataFrame,
    inventory_bin_edges: Sequence[float],
    *,
    min_count_per_bin: int = 50,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Paired run-level difference in inventory-conditioned quote depth."""
    edges = np.asarray(inventory_bin_edges, dtype=float)

    def build_run_bins(fill_probe: pd.DataFrame) -> pd.DataFrame:
        frame = fill_probe.copy()
        frame["inventory_bin"] = assign_inventory_bins(
            frame["inventory"].to_numpy(dtype=float),
            edges,
        )
        frame["bin_center"] = 0.5 * (
            edges[frame["inventory_bin"]] + edges[frame["inventory_bin"] + 1]
        )
        return (
            frame.groupby(
                ["n_agents", "side", "inventory_bin", "bin_center", "run_id"],
                as_index=False,
            )
            .agg(run_mean_depth=("delta_ticks", "mean"), bin_count=("delta_ticks", "size"))
        )

    paired = build_run_bins(left_fill_probe).merge(
        build_run_bins(right_fill_probe),
        on=["n_agents", "side", "inventory_bin", "bin_center", "run_id"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    paired["depth_difference"] = (
        paired["run_mean_depth_right"] - paired["run_mean_depth_left"]
    )

    rows = []
    grouping_columns = ["n_agents", "side", "inventory_bin", "bin_center"]
    for group_keys, group in paired.groupby(grouping_columns):
        total_count = int(
            np.minimum(group["bin_count_left"], group["bin_count_right"]).sum()
        )
        if total_count < int(min_count_per_bin):
            continue
        n_agents, side, inventory_bin, bin_center = group_keys
        seed_offset = (
            int(n_agents) * 1000 + int(inventory_bin) * 10 + int(side)
        )
        mean_difference, ci_lo, ci_hi = run_cluster_bootstrap_interval(
            group["depth_difference"].to_numpy(),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed + seed_offset,
        )
        rows.append(
            {
                "n_agents": int(n_agents),
                "side": int(side),
                "inventory_bin": int(inventory_bin),
                "bin_center": float(bin_center),
                "mean_difference": mean_difference,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_pairs": int(group["run_id"].nunique()),
                "total_count": total_count,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["n_agents", "side", "inventory_bin"])
        .reset_index(drop=True)
    )


def dynamic_padded_bounds(
    values: Iterable[float],
    *,
    ci_low: Optional[Iterable[float]] = None,
    ci_high: Optional[Iterable[float]] = None,
    padding_fraction: float = 0.08,
    min_padding: float = 0.05,
) -> Tuple[float, float]:
    """Axis limits that include values and confidence intervals with dynamic padding."""
    all_values = []
    for series in (values, ci_low, ci_high):
        if series is None:
            continue
        array = np.asarray(list(series), dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size:
            all_values.append(finite)

    if not all_values:
        return 0.0, 1.0

    stacked = np.concatenate(all_values)
    ymin = float(stacked.min())
    ymax = float(stacked.max())
    span = ymax - ymin
    if span <= 0.0:
        span = max(abs(ymax), min_padding)
    padding = max(float(min_padding), float(padding_fraction) * span)
    return ymin - padding, ymax + padding


def save_analysis_summary(summary_tables: Dict[str, pd.DataFrame], output_dir: PathLike) -> None:
    """Write named analysis tables as compact parquet shards."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in summary_tables.items():
        frame.to_parquet(out_dir / f"{table_name}.parquet", index=False)


def load_analysis_summary(output_dir: PathLike) -> Dict[str, pd.DataFrame]:
    """Load parquet tables written by ``save_analysis_summary``."""
    out_dir = Path(output_dir)
    paths = sorted(out_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No analysis summary parquet files under {out_dir}")
    return {path.stem: pd.read_parquet(path) for path in paths}

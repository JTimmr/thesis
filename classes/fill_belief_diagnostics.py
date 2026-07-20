"""Static phantom-NN fill-belief diagnostics for the validated uniform workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import curve_fit

from .helpers import resolve_data_path
from .mm_backtest_parallel import _load_nn_bundle

PathLike = Union[str, Path]

# --- Validated static phantom-NN workflow ---

phantom_size = 10000
symbol = "SIM"
phantom_labels_dir = f"phantom_labels_sim_1M_size{phantom_size}_uniform"
static_checkpoint_rel = f"phantom_models_sim_1M_size{phantom_size}_uniform_logq/fillprob_mlp_best.pt"

delta_band_lo = 0.10
delta_band_hi = 0.20
delta_reference = 0.15

bootstrap_replicates = 1000
bootstrap_seed = 42

intensity_bin_min_rows = 50
delta_bin_min_rows = 300
delta_bin_min_fills = 10
reliability_bins = 25
reliability_min_rows = 20
fill_vs_delta_bins = 22

tercile_names = ("low", "mid", "high")

FILL_VS_INTENSITY_COLUMNS = (
    "side",
    "intensity_bin_center",
    "intensity_bin_lo",
    "intensity_bin_hi",
    "n_rows",
    "n_fills",
    "predicted_fill",
    "predicted_fill_ci_lo",
    "predicted_fill_ci_hi",
    "empirical_fill",
    "empirical_fill_ci_lo",
    "empirical_fill_ci_hi",
    "exponential_fill",
)

RELIABILITY_COLUMNS = (
    "curve",
    "intensity_tercile",
    "intensity_lo",
    "intensity_hi",
    "predicted_bin_center",
    "n_rows",
    "observed_fill",
    "observed_fill_ci_lo",
    "observed_fill_ci_hi",
)

FILL_VS_DELTA_TERCILE_COLUMNS = (
    "intensity_tercile",
    "intensity_lo",
    "intensity_hi",
    "delta_bin_center",
    "n_rows",
    "n_fills",
    "reliable",
    "mlp_fill",
    "mlp_fill_ci_lo",
    "mlp_fill_ci_hi",
    "empirical_fill",
    "empirical_fill_ci_lo",
    "empirical_fill_ci_hi",
    "exponential_fill",
)


def load_fill_model(ckpt_path: str) -> Dict[str, Any]:
    """Load the eval-mode fill MLP and normalisation bundle (same as ``_load_nn_bundle``)."""
    return _load_nn_bundle(ckpt_path)


def read_checkpoint_split(ckpt_path: PathLike) -> Dict[str, Any]:
    """Read train/val/test day lists and phantom paths saved beside the checkpoint."""
    config_path = Path(ckpt_path).with_name("train_config.json")
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    return {
        "symbol": config["symbol"],
        "phantom_dir": Path(config["phantom_dir"]),
        "train_days": list(config["train_days"]),
        "val_days": list(config["val_days"]),
        "test_days": list(config["test_days"]),
        "queue_transform": config["queue_transform"],
    }


def read_checkpoint_feat_cols(ckpt_path: PathLike) -> List[str]:
    # Deferred so summary-only diagnostics do not initialize the PyTorch DLLs.
    import torch

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return list(checkpoint["feat_cols"])


def phantom_day_path(phantom_dir: PathLike, day_key: str, asset_symbol: str = symbol) -> Path:
    return Path(phantom_dir) / f"{asset_symbol}_{day_key}.parquet"


def opposite_mo_intensity(
    side: np.ndarray,
    mo_bid: np.ndarray,
    mo_ask: np.ndarray,
) -> np.ndarray:
    """Opposite-side MO Hawkes intensity: ask intensity for bid phantoms and vice versa."""
    side_array = np.asarray(side)
    mo_bid_array = np.asarray(mo_bid, dtype=np.float64)
    mo_ask_array = np.asarray(mo_ask, dtype=np.float64)
    return np.where(side_array == 1, mo_ask_array, mo_bid_array)


def exponential_fill(delta: Union[np.ndarray, float], amplitude: float, decay: float) -> np.ndarray:
    delta_array = np.asarray(delta, dtype=np.float64)
    return amplitude * np.exp(-decay * delta_array)


def fit_train_exponential(train_parquet_paths: Sequence[PathLike]) -> Tuple[float, float]:
    """Fit h(delta)=A*exp(-k*delta) on positive-delta rows from train parquets only."""
    grouped_parts = []
    for parquet_path in train_parquet_paths:
        day_frame = pd.read_parquet(parquet_path, columns=["delta", "label"])
        day_frame = day_frame.loc[day_frame["delta"] > 0]
        grouped_parts.append(
            day_frame.groupby("delta", as_index=False).agg(
                fill_sum=("label", "sum"),
                count=("label", "size"),
            )
        )
    grouped = (
        pd.concat(grouped_parts, ignore_index=True)
        .groupby("delta", as_index=False)
        .agg(fill_sum=("fill_sum", "sum"), count=("count", "sum"))
        .sort_values("delta")
    )
    empirical_rate = grouped["fill_sum"] / grouped["count"]
    amplitude_decay, _ = curve_fit(
        exponential_fill,
        grouped["delta"].to_numpy(dtype=np.float64),
        empirical_rate.to_numpy(dtype=np.float64),
        sigma=1.0 / np.sqrt(grouped["count"].to_numpy(dtype=np.float64)),
        p0=[1.0, 1.0],
        maxfev=10000,
        bounds=([0.0, 0.0], [np.inf, np.inf]),
    )
    return float(amplitude_decay[0]), float(amplitude_decay[1])


def normalize_phantom_rows(
    day_frame: pd.DataFrame,
    feat_cols: Sequence[str],
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    queue_transform: Optional[str],
) -> np.ndarray:
    norm_cols = list(feat_cols) + ["delta", "queue_ahead"]
    norm_values = day_frame[norm_cols].to_numpy(dtype=np.float32)
    if queue_transform == "log1p":
        queue_ahead_index = norm_cols.index("queue_ahead")
        norm_values[:, queue_ahead_index] = np.log1p(norm_values[:, queue_ahead_index])
    norm_values = (norm_values - feat_mean) / feat_std
    side_encoding = (day_frame["side"].to_numpy(dtype=np.float32) - 1.0).reshape(-1, 1)
    return np.hstack([norm_values, side_encoding]).astype(np.float32)


def infer_day_predictions(
    bundle: Dict[str, Any],
    feat_cols: Sequence[str],
    day_parquet_path: PathLike,
    day_key: str,
) -> pd.DataFrame:
    """Run one test-day forward pass with checkpoint normalisation and queue transform."""
    torch = bundle["torch"]
    model = bundle["model"]
    day_frame = pd.read_parquet(day_parquet_path)
    features = normalize_phantom_rows(
        day_frame,
        feat_cols,
        np.asarray(bundle["feat_mean"], dtype=np.float32),
        np.asarray(bundle["feat_std"], dtype=np.float32),
        bundle.get("queue_transform"),
    )
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        predicted_fill = torch.sigmoid(logits / bundle["temperature"]).numpy().astype(np.float32)
    mo_bid = day_frame["hawkes_intensity_MO_bid"].to_numpy(dtype=np.float32)
    mo_ask = day_frame["hawkes_intensity_MO_ask"].to_numpy(dtype=np.float32)
    side = day_frame["side"].to_numpy(dtype=np.int8)
    return pd.DataFrame(
        {
            "day": day_key,
            "side": side,
            "delta": day_frame["delta"].to_numpy(dtype=np.float32),
            "label": day_frame["label"].to_numpy(dtype=np.float32),
            "predicted_fill": predicted_fill,
            "opposite_mo_intensity": opposite_mo_intensity(side, mo_bid, mo_ask).astype(np.float32),
        }
    )


def build_test_predictions_cache(
    ckpt_path: PathLike,
    cache_path: PathLike,
    phantom_dir: Optional[PathLike] = None,
    test_days: Optional[Sequence[str]] = None,
    asset_symbol: str = symbol,
) -> Path:
    """Stream test-day inference into one compact parquet (skips if cache already exists)."""
    cache_file = Path(cache_path)
    if cache_file.exists():
        return cache_file

    split = read_checkpoint_split(ckpt_path)
    if phantom_dir is None:
        label_dir = split["phantom_dir"]
    else:
        label_dir = Path(phantom_dir)
    if test_days is None:
        day_keys = split["test_days"]
    else:
        day_keys = list(test_days)
    bundle = load_fill_model(str(ckpt_path))
    feat_cols = read_checkpoint_feat_cols(ckpt_path)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_file.with_suffix(".tmp.parquet")
    temporary_cache.unlink(missing_ok=True)
    parquet_writer = None
    for day_key in day_keys:
        day_predictions = infer_day_predictions(
            bundle,
            feat_cols,
            phantom_day_path(label_dir, day_key, asset_symbol),
            day_key,
        )
        day_table = pa.Table.from_pandas(day_predictions, preserve_index=False)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(temporary_cache, day_table.schema)
        parquet_writer.write_table(day_table)
    if parquet_writer is not None:
        parquet_writer.close()
    temporary_cache.replace(cache_file)
    return cache_file


def load_test_predictions(cache_path: PathLike) -> pd.DataFrame:
    return pd.read_parquet(cache_path)


def day_cluster_bootstrap_mean(
    day_values: Dict[str, float],
    n_bootstrap: int = bootstrap_replicates,
    seed: int = bootstrap_seed,
) -> Tuple[float, float, float]:
    """Cluster bootstrap 95% CI for the mean of per-day values."""
    days = sorted(day_values)
    values = np.array([day_values[day_key] for day_key in days], dtype=np.float64)
    point_estimate = float(values.mean())
    random_generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
    for replicate_index in range(n_bootstrap):
        sampled_days = random_generator.choice(days, size=len(days), replace=True)
        bootstrap_means[replicate_index] = np.mean([day_values[day_key] for day_key in sampled_days])
    ci_lo, ci_hi = np.percentile(bootstrap_means, [2.5, 97.5])
    return point_estimate, float(ci_lo), float(ci_hi)


def day_cluster_bootstrap_rate(
    day_counts: Dict[str, float],
    day_sums: Dict[str, float],
    n_bootstrap: int = bootstrap_replicates,
    seed: int = bootstrap_seed,
) -> Tuple[float, float, float]:
    """Cluster bootstrap 95% CI for a pooled rate sum/count across days."""
    days = sorted(day_counts)
    point_numerator = sum(day_sums[day_key] for day_key in days)
    point_denominator = sum(day_counts[day_key] for day_key in days)
    if point_denominator > 0:
        point_estimate = point_numerator / point_denominator
    else:
        point_estimate = np.nan
    random_generator = np.random.default_rng(seed)
    bootstrap_rates = np.empty(n_bootstrap, dtype=np.float64)
    for replicate_index in range(n_bootstrap):
        sampled_days = random_generator.choice(days, size=len(days), replace=True)
        numerator = sum(day_sums[day_key] for day_key in sampled_days)
        denominator = sum(day_counts[day_key] for day_key in sampled_days)
        if denominator > 0:
            bootstrap_rates[replicate_index] = numerator / denominator
        else:
            bootstrap_rates[replicate_index] = np.nan
    ci_lo, ci_hi = np.percentile(bootstrap_rates, [2.5, 97.5])
    return float(point_estimate), float(ci_lo), float(ci_hi)


def intensity_tercile_edges(intensity_values: np.ndarray) -> np.ndarray:
    return np.quantile(intensity_values.astype(np.float64), [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])


def tercile_mask(
    intensity_values: np.ndarray,
    tercile_edges: np.ndarray,
    tercile_index: int,
) -> np.ndarray:
    lo_edge = tercile_edges[tercile_index]
    hi_edge = tercile_edges[tercile_index + 1]
    if tercile_index < 2:
        return (intensity_values >= lo_edge) & (intensity_values < hi_edge)
    return (intensity_values >= lo_edge) & (intensity_values <= hi_edge)


def aggregate_fill_vs_opposite_intensity(
    predictions: pd.DataFrame,
    exponential_amplitude: float,
    exponential_decay: float,
    delta_lo: float = delta_band_lo,
    delta_hi: float = delta_band_hi,
    n_intensity_bins: int = 15,
    min_rows_per_bin: int = intensity_bin_min_rows,
    n_bootstrap: int = bootstrap_replicates,
    seed: int = bootstrap_seed,
) -> pd.DataFrame:
    """Fill rate vs opposite-side MO intensity in a fixed delta band, bid/ask panels."""
    rows: List[Dict[str, Any]] = []
    exponential_at_reference = float(exponential_fill(delta_reference, exponential_amplitude, exponential_decay))
    for side_value, side_label in ((1, "bid"), (2, "ask")):
        side_band = predictions[
            (predictions["side"] == side_value)
            & (predictions["delta"] >= delta_lo)
            & (predictions["delta"] < delta_hi)
        ]
        if side_band.empty:
            continue
        intensity_values = side_band["opposite_mo_intensity"].to_numpy(dtype=np.float64)
        intensity_edges = np.linspace(intensity_values.min(), intensity_values.max(), n_intensity_bins + 1)
        for bin_index in range(n_intensity_bins):
            bin_lo = intensity_edges[bin_index]
            bin_hi = intensity_edges[bin_index + 1]
            bin_mask = (intensity_values >= bin_lo) & (intensity_values < bin_hi)
            if int(bin_mask.sum()) < min_rows_per_bin:
                continue
            bin_frame = side_band.loc[bin_mask]
            day_pred_sums: Dict[str, float] = {}
            day_label_sums: Dict[str, float] = {}
            day_counts: Dict[str, float] = {}
            for day_key, day_group in bin_frame.groupby("day"):
                day_counts[day_key] = float(len(day_group))
                day_label_sums[day_key] = float(day_group["label"].sum())
                day_pred_sums[day_key] = float(day_group["predicted_fill"].sum())
            predicted_fill, predicted_lo, predicted_hi = day_cluster_bootstrap_rate(
                day_counts,
                day_pred_sums,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            empirical_fill, empirical_lo, empirical_hi = day_cluster_bootstrap_rate(
                day_counts,
                day_label_sums,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            rows.append(
                {
                    "side": side_label,
                    "intensity_bin_center": 0.5 * (bin_lo + bin_hi),
                    "intensity_bin_lo": bin_lo,
                    "intensity_bin_hi": bin_hi,
                    "n_rows": int(bin_mask.sum()),
                    "n_fills": int(bin_frame["label"].sum()),
                    "predicted_fill": predicted_fill,
                    "predicted_fill_ci_lo": predicted_lo,
                    "predicted_fill_ci_hi": predicted_hi,
                    "empirical_fill": empirical_fill,
                    "empirical_fill_ci_lo": empirical_lo,
                    "empirical_fill_ci_hi": empirical_hi,
                    "exponential_fill": exponential_at_reference,
                }
            )
    summary = pd.DataFrame(rows)
    return summary[list(FILL_VS_INTENSITY_COLUMNS)]


def aggregate_reliability_by_intensity_tercile(
    predictions: pd.DataFrame,
    exponential_amplitude: float,
    exponential_decay: float,
    n_bins: int = reliability_bins,
    min_rows_per_bin: int = reliability_min_rows,
    n_bootstrap: int = bootstrap_replicates,
    seed: int = bootstrap_seed,
) -> pd.DataFrame:
    """Reliability diagram points split by opposite-side MO-intensity tercile."""
    intensity_values = predictions["opposite_mo_intensity"].to_numpy(dtype=np.float64)
    tercile_edges = intensity_tercile_edges(intensity_values)
    rows: List[Dict[str, Any]] = []

    for tercile_index, tercile_name in enumerate(tercile_names):
        tercile_frame = predictions.loc[tercile_mask(intensity_values, tercile_edges, tercile_index)]
        tercile_valid = tercile_frame.loc[tercile_frame["predicted_fill"] >= 1e-5]
        if tercile_valid.empty:
            continue
        predicted_values = tercile_valid["predicted_fill"].to_numpy(dtype=np.float64)
        probability_edges = np.geomspace(predicted_values[predicted_values > 0].min(), 1.0, n_bins + 1)
        for bin_lo, bin_hi in zip(probability_edges[:-1], probability_edges[1:]):
            bin_mask = (predicted_values >= bin_lo) & (predicted_values < bin_hi)
            if int(bin_mask.sum()) < min_rows_per_bin:
                continue
            bin_frame = tercile_valid.iloc[np.where(bin_mask)[0]]
            day_label_sums: Dict[str, float] = {}
            day_counts: Dict[str, float] = {}
            for day_key, day_group in bin_frame.groupby("day"):
                day_counts[day_key] = float(len(day_group))
                day_label_sums[day_key] = float(day_group["label"].sum())
            observed_fill, observed_lo, observed_hi = day_cluster_bootstrap_rate(
                day_counts,
                day_label_sums,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            rows.append(
                {
                    "curve": "mlp",
                    "intensity_tercile": tercile_name,
                    "intensity_lo": float(tercile_edges[tercile_index]),
                    "intensity_hi": float(tercile_edges[tercile_index + 1]),
                    "predicted_bin_center": float(predicted_values[bin_mask].mean()),
                    "n_rows": int(bin_mask.sum()),
                    "observed_fill": observed_fill,
                    "observed_fill_ci_lo": observed_lo,
                    "observed_fill_ci_hi": observed_hi,
                }
            )

    positive_delta = predictions.loc[predictions["delta"] > 0].copy()
    positive_delta["exponential_fill"] = exponential_fill(
        positive_delta["delta"].to_numpy(dtype=np.float64),
        exponential_amplitude,
        exponential_decay,
    )
    exponential_valid = positive_delta.loc[positive_delta["exponential_fill"] >= 1e-5]
    exponential_predictions = exponential_valid["exponential_fill"].to_numpy(dtype=np.float64)
    probability_edges = np.geomspace(exponential_predictions[exponential_predictions > 0].min(), 1.0, n_bins + 1)
    for bin_lo, bin_hi in zip(probability_edges[:-1], probability_edges[1:]):
        bin_mask = (exponential_predictions >= bin_lo) & (exponential_predictions < bin_hi)
        if int(bin_mask.sum()) < min_rows_per_bin:
            continue
        bin_frame = exponential_valid.iloc[np.where(bin_mask)[0]]
        day_label_sums = {}
        day_counts = {}
        for day_key, day_group in bin_frame.groupby("day"):
            day_counts[day_key] = float(len(day_group))
            day_label_sums[day_key] = float(day_group["label"].sum())
        observed_fill, observed_lo, observed_hi = day_cluster_bootstrap_rate(
            day_counts,
            day_label_sums,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append(
            {
                "curve": "exponential",
                "intensity_tercile": "all",
                "intensity_lo": np.nan,
                "intensity_hi": np.nan,
                "predicted_bin_center": float(exponential_predictions[bin_mask].mean()),
                "n_rows": int(bin_mask.sum()),
                "observed_fill": observed_fill,
                "observed_fill_ci_lo": observed_lo,
                "observed_fill_ci_hi": observed_hi,
            }
        )

    summary = pd.DataFrame(rows)
    return summary[list(RELIABILITY_COLUMNS)]


def aggregate_fill_vs_delta_by_intensity_tercile(
    predictions: pd.DataFrame,
    exponential_amplitude: float,
    exponential_decay: float,
    n_delta_bins: int = fill_vs_delta_bins,
    min_rows_per_bin: int = delta_bin_min_rows,
    min_fills_per_bin: int = delta_bin_min_fills,
    n_bootstrap: int = bootstrap_replicates,
    seed: int = bootstrap_seed,
) -> pd.DataFrame:
    """Fill probability vs delta by opposite-side MO-intensity tercile, with exponential benchmark."""
    intensity_values = predictions["opposite_mo_intensity"].to_numpy(dtype=np.float64)
    tercile_edges = intensity_tercile_edges(intensity_values)
    rows: List[Dict[str, Any]] = []

    for tercile_index, tercile_name in enumerate(tercile_names):
        tercile_frame = predictions.loc[tercile_mask(intensity_values, tercile_edges, tercile_index)]
        positive_delta = tercile_frame.loc[tercile_frame["delta"] > 0]
        if positive_delta.empty:
            continue
        delta_values = positive_delta["delta"].to_numpy(dtype=np.float64)
        delta_edges = np.geomspace(delta_values.min(), delta_values.max(), n_delta_bins + 1)
        for bin_lo, bin_hi in zip(delta_edges[:-1], delta_edges[1:]):
            bin_mask = (delta_values >= bin_lo) & (delta_values < bin_hi)
            if int(bin_mask.sum()) < min_rows_per_bin:
                continue
            bin_frame = positive_delta.iloc[np.where(bin_mask)[0]]
            day_label_sums: Dict[str, float] = {}
            day_pred_sums: Dict[str, float] = {}
            day_counts: Dict[str, float] = {}
            for day_key, day_group in bin_frame.groupby("day"):
                day_counts[day_key] = float(len(day_group))
                day_label_sums[day_key] = float(day_group["label"].sum())
                day_pred_sums[day_key] = float(day_group["predicted_fill"].sum())
            mlp_fill, mlp_lo, mlp_hi = day_cluster_bootstrap_rate(
                day_counts,
                day_pred_sums,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            empirical_fill, empirical_lo, empirical_hi = day_cluster_bootstrap_rate(
                day_counts,
                day_label_sums,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            delta_center = 0.5 * (bin_lo + bin_hi)
            rows.append(
                {
                    "intensity_tercile": tercile_name,
                    "intensity_lo": float(tercile_edges[tercile_index]),
                    "intensity_hi": float(tercile_edges[tercile_index + 1]),
                    "delta_bin_center": delta_center,
                    "n_rows": int(bin_mask.sum()),
                    "n_fills": int(bin_frame["label"].sum()),
                    "reliable": bool(bin_frame["label"].sum() >= min_fills_per_bin),
                    "mlp_fill": mlp_fill,
                    "mlp_fill_ci_lo": mlp_lo,
                    "mlp_fill_ci_hi": mlp_hi,
                    "empirical_fill": empirical_fill,
                    "empirical_fill_ci_lo": empirical_lo,
                    "empirical_fill_ci_hi": empirical_hi,
                    "exponential_fill": float(exponential_fill(delta_center, exponential_amplitude, exponential_decay)),
                }
            )

    summary = pd.DataFrame(rows)
    return summary[list(FILL_VS_DELTA_TERCILE_COLUMNS)]


def build_static_phantom_report_summaries(
    ckpt_path: Optional[PathLike] = None,
    cache_dir: Optional[PathLike] = None,
) -> Dict[str, Path]:
    """Build cached test predictions and the three plot-ready summary parquets."""
    if ckpt_path is None:
        checkpoint_path = resolve_data_path(static_checkpoint_rel)
    else:
        checkpoint_path = Path(ckpt_path)
    if cache_dir is None:
        summary_dir = checkpoint_path.parent / "diagnostics_cache"
    else:
        summary_dir = Path(cache_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    split = read_checkpoint_split(checkpoint_path)
    train_paths = [
        phantom_day_path(split["phantom_dir"], day_key, split["symbol"])
        for day_key in split["train_days"]
    ]
    exponential_amplitude, exponential_decay = fit_train_exponential(train_paths)

    predictions_path = summary_dir / "test_predictions.parquet"
    build_test_predictions_cache(checkpoint_path, predictions_path)
    predictions = load_test_predictions(predictions_path)

    fill_vs_intensity_path = summary_dir / "fill_vs_opposite_intensity.parquet"
    reliability_path = summary_dir / "reliability_by_intensity_tercile.parquet"
    fill_vs_delta_path = summary_dir / "fill_vs_delta_by_intensity_tercile.parquet"
    exponential_path = summary_dir / "exponential_fit.json"

    fill_vs_intensity = aggregate_fill_vs_opposite_intensity(
        predictions,
        exponential_amplitude,
        exponential_decay,
    )
    reliability = aggregate_reliability_by_intensity_tercile(
        predictions,
        exponential_amplitude,
        exponential_decay,
    )
    fill_vs_delta = aggregate_fill_vs_delta_by_intensity_tercile(
        predictions,
        exponential_amplitude,
        exponential_decay,
    )

    fill_vs_intensity.to_parquet(fill_vs_intensity_path, index=False)
    reliability.to_parquet(reliability_path, index=False)
    fill_vs_delta.to_parquet(fill_vs_delta_path, index=False)
    with open(exponential_path, "w", encoding="utf-8") as exponential_file:
        json.dump(
            {
                "amplitude": exponential_amplitude,
                "decay": exponential_decay,
                "delta_reference": delta_reference,
            },
            exponential_file,
            indent=2,
        )

    return {
        "predictions": predictions_path,
        "fill_vs_opposite_intensity": fill_vs_intensity_path,
        "reliability_by_intensity_tercile": reliability_path,
        "fill_vs_delta_by_intensity_tercile": fill_vs_delta_path,
        "exponential_fit": exponential_path,
    }

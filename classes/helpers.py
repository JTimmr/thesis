"""Shared paths and SQLite loaders for Hawkes calibration.

This module sits between the SQLite databases produced by ``extract`` and the
fitting code in ``calibrate``. It resolves data paths, lists trading days,
loads per-mark event times into the ``list[list[np.ndarray]]`` layout the
calibrator expects, and estimates the intraday seasonality profiles used to
build τ-time. ``plot_mm_result_compact`` is an unrelated plotting helper kept
here because the market-maker notebooks import it.

SQLite contract
---------------
The loaders assume the schema written by ``research_core.classes.extract``:

- ``mo_orders(day, first_time_ns, side)`` with ``side`` in {'buy', 'sell'};
  ``MO_bid`` maps to buy market orders, ``MO_ask`` to sell.
- ``orders(day, timestamp, event_type, side)`` with ``event_type`` in
  {'LO', 'CXL'} and integer ``side`` in {1 = bid, 2 = ask}.

Event times are returned as seconds since the day's market open.
"""

from __future__ import annotations

import glob as _glob
import pickle
import re as _re
import sqlite3
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def project_root() -> Path:
    """Return the repository root (this package lives directly under it)."""
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Return the canonical data directory for databases and artefacts.

    All SQLite databases created or read by ``research_core`` live here.
    The directory is created on first access if it does not exist.
    """
    d = project_root() / "data"
    d.mkdir(exist_ok=True)
    return d


def resolve_data_path(path: Union[str, Path]) -> Path:
    """Resolve *path* so it lives inside :func:`data_dir`.

    * Absolute paths are returned unchanged.
    * Relative paths (including bare filenames) are resolved relative to
      ``data_dir()``.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return data_dir() / p


def load_day_events_from_sqlite(
    conn: sqlite3.Connection,
    day_key: str,
    market_open_str: str,
    marks: Sequence[str],
) -> List[np.ndarray]:
    """Load per-mark event times (seconds since market open) for one day.

    Used by calibration workflows to extract per-mark timestamps for Hawkes fitting.
    """
    day_date_str = day_key.lstrip("d")
    ref_ns = pd.Timestamp(
        f"{day_date_str} {market_open_str}", tz="Europe/Warsaw"
    ).value

    day_seq: List[np.ndarray] = []
    for mark in marks:
        if mark.startswith("MO_"):
            side = "buy" if mark == "MO_bid" else "sell"
            rows = conn.execute(
                "SELECT first_time_ns FROM mo_orders "
                "WHERE day = ? AND side = ? ORDER BY first_time_ns",
                (day_key, side),
            ).fetchall()
            if rows:
                ns_arr = np.array([r[0] for r in rows], dtype=np.int64)
                seconds = (ns_arr - ref_ns).astype(np.float64) / 1e9
            else:
                seconds = np.array([], dtype=np.float64)

        elif mark.startswith("LO_") or mark.startswith("CXL_"):
            event_type = "LO" if mark.startswith("LO_") else "CXL"
            side = 1 if mark.endswith("_bid") else 2
            rows = conn.execute(
                "SELECT timestamp FROM orders "
                "WHERE day = ? AND event_type = ? AND side = ? "
                "ORDER BY timestamp",
                (day_key, event_type, side),
            ).fetchall()
            if rows:
                ts = pd.to_datetime([r[0] for r in rows], format="ISO8601")
                ns_arr = ts.values.astype(np.int64)
                seconds = (ns_arr - ref_ns).astype(np.float64) / 1e9
            else:
                seconds = np.array([], dtype=np.float64)
        else:
            seconds = np.array([], dtype=np.float64)

        day_seq.append(np.ascontiguousarray(np.sort(seconds), dtype=np.float64))
    return day_seq


def list_day_keys_from_sqlite(db_path: Union[str, Path]) -> List[str]:
    """Return sorted distinct ``day`` keys from an empirical order-flow SQLite file."""
    db_path = resolve_data_path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT DISTINCT day FROM orders ORDER BY day").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def compute_end_times(timestamps_by_day: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """Per-day horizon: max event time (seconds since open) across marks."""
    return np.array(
        [
            max((seq.max() if len(seq) else 0.0) for seq in day_seq)
            for day_seq in timestamps_by_day
        ],
        dtype=np.float64,
    )


def load_events_from_sqlite_bulk(
    db_path: Union[str, Path],
    day_keys: Sequence[str],
    market_open_str: str,
    marks_order: Sequence[str],
) -> List[List[np.ndarray]]:
    """Bulk-load event timestamps from SQLite for all requested days at once.

    Instead of issuing ``len(marks_order)`` queries per day, this reads the
    ``mo_orders`` and ``orders`` tables with just 2 queries total, then
    splits in Python.  Orders of magnitude faster on large databases.

    Returns ``list[list[np.ndarray]]`` -- outer = days, inner = dimensions.
    """
    day_keys = list(day_keys)
    if len(day_keys) == 0:
        return []

    placeholders = ",".join(["?"] * len(day_keys))
    marks_order = list(marks_order)

    ref_ns_by_day: Dict[str, int] = {
        dk: pd.Timestamp(
            f"{dk.lstrip('d')} {market_open_str}", tz="Europe/Warsaw"
        ).value
        for dk in day_keys
    }

    buckets: Dict[str, Dict[str, np.ndarray]] = {
        dk: {mark: np.array([], dtype=np.float64) for mark in marks_order}
        for dk in day_keys
    }

    conn = sqlite3.connect(str(db_path))

    # ── MO events ──────────────────────────────────────────────────────
    t0 = _time.time()
    mo_df = pd.read_sql_query(
        f"SELECT day, first_time_ns, side FROM mo_orders "
        f"WHERE day IN ({placeholders}) ORDER BY day, first_time_ns",
        conn,
        params=day_keys,
    )
    print(f"  MO query: {len(mo_df):,} rows in {_time.time() - t0:.2f}s")

    if not mo_df.empty:
        mo_df["mark"] = np.where(mo_df["side"] == "buy", "MO_bid", "MO_ask")
        mo_df["ref_ns"] = mo_df["day"].map(ref_ns_by_day).astype(np.int64)
        mo_df["seconds"] = (
            mo_df["first_time_ns"].astype(np.int64) - mo_df["ref_ns"]
        ) / 1e9
        for (dk, mark), grp in mo_df.groupby(["day", "mark"], sort=False):
            if dk in buckets:
                buckets[dk][mark] = grp["seconds"].to_numpy(dtype=np.float64)
    del mo_df

    # ── LO + CXL events ───────────────────────────────────────────────
    t0 = _time.time()
    orders_df = pd.read_sql_query(
        f"SELECT day, timestamp, event_type, side FROM orders "
        f"WHERE day IN ({placeholders}) AND event_type IN ('LO','CXL') "
        f"AND side IN (1,2) ORDER BY day, timestamp",
        conn,
        params=day_keys,
    )
    print(f"  LO/CXL query: {len(orders_df):,} rows in {_time.time() - t0:.2f}s")
    conn.close()

    if not orders_df.empty:
        side_label = np.where(
            orders_df["side"].astype(int) == 1, "bid", "ask"
        )
        orders_df["mark"] = (
            orders_df["event_type"].astype(str) + "_" + side_label
        )
        t0 = _time.time()
        ts_parsed = pd.to_datetime(orders_df["timestamp"], utc=True)
        ts_ns = ts_parsed.values.astype("datetime64[ns]").astype(np.int64)
        orders_df["ref_ns"] = (
            orders_df["day"].map(ref_ns_by_day).astype(np.int64)
        )
        orders_df["seconds"] = (
            ts_ns - orders_df["ref_ns"].to_numpy(dtype=np.int64)
        ) / 1e9
        print(f"  Timestamp conversion: {_time.time() - t0:.2f}s")

        for (dk, mark), grp in orders_df.groupby(["day", "mark"], sort=False):
            if dk in buckets and mark in buckets[dk]:
                buckets[dk][mark] = grp["seconds"].to_numpy(dtype=np.float64)
    del orders_df

    # ── Assemble in day_keys order ─────────────────────────────────────
    timestamps_by_day: List[List[np.ndarray]] = []
    for dk in day_keys:
        day_seq: List[np.ndarray] = []
        for mark in marks_order:
            arr = buckets[dk][mark]
            day_seq.append(
                np.ascontiguousarray(np.sort(arr), dtype=np.float64)
            )
        timestamps_by_day.append(day_seq)

    return timestamps_by_day


def list_day_keys(
    asset_name: str,
    *,
    source: str = "auto",
    sqlite_dir: Optional[Union[str, Path]] = None,
    event_output_dir: Optional[Union[str, Path]] = None,
    orders_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Return sorted day keys for *asset_name*.

    Searches (in order): SQLite DB, ``.npz`` event files, HDF5 source data.
    Pass ``source="hdf5"`` to skip SQLite / npz and go straight to HDF5.

    Parameters
    ----------
    sqlite_dir : path, optional
        Folder containing ``<asset>_order_flow.sqlite``.
    event_output_dir : path, optional
        Folder containing ``<asset>_<day>_events.npz`` files.
    orders_dir : path, optional
        Folder containing ``<asset>_lob_2017_zlib.h5``.
    """
    root = project_root()
    if sqlite_dir is None:
        sqlite_dir = root / "data"
    if event_output_dir is None:
        event_output_dir = root / "data"
    if orders_dir is None:
        orders_dir = root / "data" / "WSELOB-2017" / "orders"

    sqlite_dir = Path(sqlite_dir)
    event_output_dir = Path(event_output_dir)
    orders_dir = Path(orders_dir)

    if source == "auto":
        db = sqlite_dir / f"{asset_name}_order_flow.sqlite"
        if db.exists():
            conn = sqlite3.connect(str(db))
            days = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT day FROM orders ORDER BY day"
                )
            ]
            conn.close()
            if days:
                return days

        pattern = str(event_output_dir / f"{asset_name}_*_events.npz")
        npz_files = sorted(_glob.glob(pattern))
        if npz_files:
            days = []
            for p in npz_files:
                m = _re.search(r"(d\d{8})_events\.npz$", p)
                if m:
                    days.append(m.group(1))
            if days:
                return sorted(days)

    file_path = orders_dir / f"{asset_name}_lob_2017_zlib.h5"
    if not file_path.exists():
        raise FileNotFoundError(f"No data source found for {asset_name}")
    with pd.HDFStore(str(file_path), mode="r") as hdf:
        keys = [k.lstrip("/") for k in hdf.keys()]
    return sorted(keys)


# ═══════════════════════════════════════════════════════════════════════════════
# Seasonality estimation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def epanechnikov(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel: K(u) = 0.75 * (1 - u^2) for |u| <= 1."""
    out = 0.75 * (1 - u ** 2)
    out[np.abs(u) > 1] = 0.0
    return out


def estimate_seasonality_for_day(
    event_times: np.ndarray,
    T_day: float,
    grid: np.ndarray,
    h: float,
) -> np.ndarray:
    """Estimate intraday intensity profile for a single day using Epanechnikov kernel.

    Parameters
    ----------
    event_times : 1-D array of event times (seconds since open).
    T_day : observation horizon for this day (seconds).
    grid : evaluation grid (seconds).
    h : bandwidth (seconds).
    """
    if len(event_times) == 0:
        return np.zeros_like(grid)
    diffs = (grid[:, None] - event_times[None, :]) / h
    weights = epanechnikov(diffs)
    num = np.sum(weights, axis=1) / h
    denom = np.zeros_like(grid)
    for i, t in enumerate(grid):
        a = max(0.0, t - h)
        b = min(T_day, t + h)
        u = np.linspace(a, b, 50)
        denom[i] = np.trapz(epanechnikov((t - u) / h) / h, u)
    denom[denom == 0] = 1.0
    return num / denom


def estimate_seasonality_profiles(
    timestamps_by_day: List[List[np.ndarray]],
    marks_order: Sequence[str],
    end_times: Sequence[float],
    *,
    day_keys: Optional[Sequence[str]] = None,
    bandwidth: float = 300.0,
    grid_points: int = 400,
    cache_path: Optional[Union[str, Path]] = None,
    force_recompute: bool = False,
) -> dict:
    """Estimate intraday seasonality profiles for each event type.

    Returns a dict mapping each dimension name to a 4-tuple
    ``(grid, mean_profile, day_profiles_dict, day_keys)``, where
    ``day_profiles_dict`` is keyed by the trading-day labels.

    Parameters
    ----------
    timestamps_by_day, marks_order, end_times :
        Per-day, per-mark event times; dimension names; per-day horizons.
    day_keys : sequence of str, optional
        Trading-day labels (e.g. ``"d20170111"``), aligned with
        ``timestamps_by_day``. When omitted, positional ``"day_{i}"`` labels
        are used. Passing the real keys keeps the per-day profiles traceable
        back to specific sessions.
    bandwidth, grid_points :
        Epanechnikov bandwidth (seconds) and number of grid points.
    cache_path : path, optional
        Pickle results here, and reload them on the next call unless
        ``force_recompute`` is set.
    """
    marks_order = list(marks_order)

    if day_keys is None:
        day_keys_list = [f"day_{i}" for i in range(len(timestamps_by_day))]
    else:
        day_keys_list = list(day_keys)
        if len(day_keys_list) != len(timestamps_by_day):
            raise ValueError(
                "day_keys length does not match timestamps_by_day "
                f"({len(day_keys_list)} vs {len(timestamps_by_day)})"
            )

    # ── Try loading from cache ─────────────────────────────────────────
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            print(f"Loading seasonality from cache: {cache_path.name}")
            try:
                with open(cache_path, "rb") as f:
                    loaded = pickle.load(f)
                profiles = (
                    loaded.get("seasonality_profiles", loaded)
                    if isinstance(loaded, dict)
                    else loaded
                )
                first_key = next(iter(profiles), None)
                if (
                    first_key
                    and isinstance(profiles[first_key], tuple)
                    and len(profiles[first_key]) == 4
                ):
                    print(f"Loaded seasonality for: {list(profiles.keys())}")
                    return profiles
                print("Incompatible cache format -- will recompute")
            except Exception as e:
                print(f"Cache load error: {e}")

    # ── Compute from scratch ───────────────────────────────────────────
    print("Computing seasonality profiles...")
    T_max = float(np.max(end_times)) if len(end_times) else 28200.0
    grid = np.linspace(0, T_max, grid_points)

    seasonality_profiles: Dict[str, tuple] = {}

    for dim_idx, dim_name in enumerate(marks_order):
        print(f"  {dim_name}...")
        day_profiles = []
        computed_day_keys: List[str] = []
        for dk, day_seq, T_day in zip(
            day_keys_list, timestamps_by_day, end_times
        ):
            events_day = np.asarray(day_seq[dim_idx], dtype=float)
            profile = estimate_seasonality_for_day(
                events_day, T_day, grid, bandwidth
            )
            mask = grid <= T_day
            mean_val = np.trapz(profile[mask], grid[mask]) / T_day
            if mean_val > 0:
                profile = profile / mean_val
            day_profiles.append(profile)
            computed_day_keys.append(dk)

        day_profiles_arr = np.array(day_profiles)
        mean_profile = np.nanmean(day_profiles_arr, axis=0)
        day_profiles_dict = {
            computed_day_keys[i]: day_profiles_arr[i]
            for i in range(len(day_profiles_arr))
        }
        seasonality_profiles[dim_name] = (
            grid.copy(),
            mean_profile.copy(),
            day_profiles_dict,
            computed_day_keys.copy(),
        )

        peak_idx = int(np.argmax(mean_profile))
        trough_idx = int(np.argmin(mean_profile))
        print(
            f"    Peak: {mean_profile[peak_idx]:.3f}, "
            f"Trough: {mean_profile[trough_idx]:.3f}, "
            f"Ratio: {mean_profile[peak_idx] / mean_profile[trough_idx]:.2f}"
        )

    # ── Save to cache ──────────────────────────────────────────────────
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(seasonality_profiles, f)
        print(f"Saved seasonality cache: {cache_path.name}")

    print("Seasonality estimation complete.")
    return seasonality_profiles


def _scale_times_mm(raw):
    if len(raw) == 0:
        return np.array(raw, dtype=float), "time"
    arr = np.array(raw, dtype=float)
    span = arr[-1] - arr[0]
    if span > 7200:
        return arr / 3600.0, "time (hours)"
    if span > 120:
        return arr / 60.0, "time (minutes)"
    return arr, "time (seconds)"


def plot_mm_result_compact(run_payload):
    """Recreate the 4-panel market-maker result figure from compact payload."""
    t_pnl = np.asarray(run_payload["pnl_t"], dtype=float)
    mids = np.asarray(run_payload["pnl_mid"], dtype=float)
    pnls = np.asarray(run_payload["pnl_mtm"], dtype=float)

    t_tr = np.asarray(run_payload["trade_t"], dtype=float)
    side = np.asarray(run_payload["trade_side"], dtype=int)
    inv = np.asarray(run_payload["trade_inv"], dtype=float)
    cash = np.asarray(run_payload["trade_cash"], dtype=float)

    buy_mask = side == 1
    sell_mask = side == -1

    t_pnl_s, xlabel = _scale_times_mm(t_pnl)
    t_tr_s, _ = _scale_times_mm(t_tr)

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    # 1) MtM PnL
    axes[0].plot(t_pnl_s, pnls, lw=1, color="tab:green", label="Mark-to-market PnL")
    axes[0].axhline(0, ls="--", color="grey", alpha=0.5)
    axes[0].set_ylabel("PnL")
    axes[0].set_title(f"Market Maker Performance (run {run_payload.get('run_id', '?')})")
    axes[0].legend(loc="upper left")

    # 2) Inventory + buy/sell markers
    if len(t_tr_s):
        axes[1].step(t_tr_s, inv, where="post", lw=1, color="tab:blue", label="Inventory")
        axes[1].scatter(t_tr_s[buy_mask], inv[buy_mask], marker="^", color="green", s=15,
                        alpha=0.7, label="Buy fill", zorder=3)
        axes[1].scatter(t_tr_s[sell_mask], inv[sell_mask], marker="v", color="red", s=15,
                        alpha=0.7, label="Sell fill", zorder=3)
    axes[1].axhline(0, ls="--", color="grey", alpha=0.5)
    axes[1].set_ylabel("Inventory")
    axes[1].legend(loc="upper left")

    # 3) Cash
    if len(t_tr_s):
        axes[2].step(t_tr_s, cash, where="post", lw=1, color="tab:purple", label="Cash")
    axes[2].axhline(0, ls="--", color="grey", alpha=0.5)
    axes[2].set_ylabel("Cash")
    axes[2].legend(loc="upper left")

    # 4) Mid price
    axes[3].plot(t_pnl_s, mids, lw=0.5, color="tab:orange", label="Mid price")
    axes[3].set_ylabel("Mid price")
    axes[3].set_xlabel(xlabel)
    axes[3].legend(loc="upper left")

    plt.tight_layout()
    plt.show()


__all__ = [
    "project_root",
    "data_dir",
    "resolve_data_path",
    "load_day_events_from_sqlite",
    "load_events_from_sqlite_bulk",
    "list_day_keys_from_sqlite",
    "list_day_keys",
    "compute_end_times",
    "epanechnikov",
    "estimate_seasonality_for_day",
    "estimate_seasonality_profiles",
    "plot_mm_result_compact",
    "_scale_times_mm",
]

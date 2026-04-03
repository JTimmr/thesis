"""Extract Hawkes event timestamps and order-flow data from WSE HDF5 files.

Events:
- LO_bid / LO_ask: limit order additions (side from orders)
- CXL_bid / CXL_ask: order cancellations (side from orders)
- MO_bid / MO_ask: market orders inferred from trades + order book

With full_extraction=True, also captures:
- Per-LO/CXL book-state snapshots for SQLite database
- Per-MO-fill depth snapshots
- Aggregated aggressive-MO data
"""

import sqlite3
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from pandas import HDFStore

from .orderbook import HeapOrderBook, compute_bbo_series, find_continuous_trading_start
from ..data.schema import (
    CREATE_ORDERS_TABLE,
    CREATE_FILLS_TABLE,
    CREATE_MO_ORDERS_TABLE,
)

TZ = "Europe/Warsaw"

_N_ORDER_COLS = 41
_INSERT_ORDERS_SQL = (
    "INSERT INTO orders VALUES (" + ",".join(["?"] * _N_ORDER_COLS) + ")"
)

_N_FILL_COLS = 30
_INSERT_FILLS_SQL = (
    "INSERT INTO fills VALUES (" + ",".join(["?"] * _N_FILL_COLS) + ")"
)

_N_MO_COLS = 33
_INSERT_MO_SQL = (
    "INSERT INTO mo_orders VALUES (" + ",".join(["?"] * _N_MO_COLS) + ")"
)

_CLS_KEYS = [
    "quote_sell", "quote_buy", "mid_sell", "mid_buy",
    "tick_used", "last_side_fallback", "unclassified",
]
_CLS_MAP = {
    "quote_sell": "quote", "quote_buy": "quote",
    "mid_sell": "midpoint", "mid_buy": "midpoint",
    "tick_used": "tick",
    "last_side_fallback": "last_side",
    "unclassified": "unclassified",
}


# ─────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────

def list_day_keys_hdf(orders_file: Path) -> list:
    """Return sorted day keys stored in the HDF5 orders file."""
    with HDFStore(str(orders_file), mode="r") as hdf:
        keys = [k.lstrip("/") for k in hdf.keys()]
    return sorted(keys)


def load_orders_day(orders_file: Path, day_key: str) -> pd.DataFrame:
    """Load and normalize one day of orders data."""
    df = pd.read_hdf(str(orders_file), f"/{day_key}")
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True).dt.tz_convert(TZ)
    df["priority_date"] = pd.to_datetime(df["priority_date"], unit="ns", utc=True).dt.tz_convert(TZ)
    df["order_date"] = pd.to_datetime(df["order_date"], unit="ns", utc=True).dt.tz_convert(TZ)
    df["price"] = df["price"] / (10 ** df["price_level"])
    df["action_type"] = df["action_type"].astype(str)
    df["order_type"] = df["order_type"].astype(str)
    df["order_type_num"] = pd.to_numeric(df["order_type"], errors="coerce")
    return df


def load_trades_day(trades_file: Path, day_key: str, time_field: str) -> pd.DataFrame:
    """Load and normalize one day of trades data."""
    df = pd.read_hdf(str(trades_file), f"/{day_key}")
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True).dt.tz_convert(TZ)
    df["trading_datetime"] = pd.to_datetime(df["trading_datetime"], unit="ns", utc=True).dt.tz_convert(TZ)
    df["price_float"] = df["price"] / (10 ** df["price_level"])
    df["trade_time"] = df[time_field]

    if "opening_trade_indicator" in df.columns:
        df["opening_trade_indicator"] = df["opening_trade_indicator"].astype(str).str.strip()
    if "trade_origin" in df.columns:
        df["trade_origin"] = df["trade_origin"].astype(str).str.strip()

    return df


def filter_market_hours(df: pd.DataFrame, market_open: str, market_close: str, time_col: str):
    """Filter records to the market open/close interval (local time)."""
    if market_open is None or market_close is None:
        return df
    local_time = df[time_col].dt.tz_convert(TZ)
    start = local_time.dt.normalize() + pd.to_timedelta(market_open)
    end = local_time.dt.normalize() + pd.to_timedelta(market_close)
    return df[(local_time >= start) & (local_time <= end)]


def to_seconds_since(ts: pd.Series, ref_time: pd.Timestamp) -> np.ndarray:
    """Convert timestamps to float seconds since ref_time."""
    ts = pd.Series(ts)
    if ts.empty:
        return np.array([])
    values = ts.view("int64").to_numpy()
    ref = int(ref_time.value)
    return (values - ref).astype("float64") / 1e9


# ─────────────────────────────────────────────────────────────
# Lee–Ready trade classification
# ─────────────────────────────────────────────────────────────

def infer_trade_side(
    trade_price,
    best_bid,
    best_ask,
    trade_ts,
    quote_ts,
    last_price,
    last_tick_dir,
    last_side=None,
    side_stats=None,
    tick_size=0.01,
    last_good_bid=None,
    last_good_ask=None,
    last_good_quote_ts=None,
    max_quote_staleness=None,
    use_last_side_fallback=False,
):
    """Research-grade Lee-Ready trade classification.

    Hierarchy:
    1. Quote rule (strictly prior quotes only)
    2. Midpoint rule (spread > 1 tick only, discrete grid)
    3. Extended tick rule (zero-tick carry-forward)

    Enhancements:
    - Crossed-market fallback to last valid BBO (if recent)
    - Trade-through detection (flag only, no forced classification)
    - Locked-market exclusion from quote rule
    - Optional staleness control
    - Optional last-side fallback (off by default)
    """
    if side_stats is None:
        side_stats = {}

    side_stats["n_trades"] = side_stats.get("n_trades", 0) + 1

    use_bid, use_ask = None, None

    strictly_prior = (
        best_bid is not None
        and best_ask is not None
        and quote_ts is not None
        and quote_ts < trade_ts
    )

    if strictly_prior:
        use_bid, use_ask = best_bid, best_ask
    else:
        side_stats["quote_not_strictly_prior"] = side_stats.get("quote_not_strictly_prior", 0) + 1

    if (
        use_bid is not None
        and max_quote_staleness is not None
        and quote_ts is not None
        and (trade_ts - quote_ts) > max_quote_staleness
    ):
        side_stats["stale_quote_skipped"] = side_stats.get("stale_quote_skipped", 0) + 1
        use_bid, use_ask = None, None

    if use_bid is not None and use_ask is not None:
        if use_bid > use_ask:
            side_stats["crossed_market"] = side_stats.get("crossed_market", 0) + 1
            if (
                last_good_bid is not None
                and last_good_ask is not None
                and last_good_quote_ts is not None
                and last_good_quote_ts < trade_ts
                and (max_quote_staleness is None
                    or (trade_ts - last_good_quote_ts) <= max_quote_staleness)
                and last_good_bid < last_good_ask
            ):
                use_bid, use_ask = last_good_bid, last_good_ask
            else:
                use_bid, use_ask = None, None
        elif use_bid == use_ask:
            side_stats["locked_market"] = side_stats.get("locked_market", 0) + 1
            use_bid, use_ask = None, None

    if use_bid is not None and use_ask is not None:
        bid_ticks = int(round(use_bid / tick_size))
        ask_ticks = int(round(use_ask / tick_size))
        trade_ticks = int(round(trade_price / tick_size))
        spread_ticks = ask_ticks - bid_ticks

        if trade_ticks > ask_ticks + 1:
            side_stats["trade_through_buy"] = side_stats.get("trade_through_buy", 0) + 1
            use_bid, use_ask = None, None
        elif trade_ticks < bid_ticks - 1:
            side_stats["trade_through_sell"] = side_stats.get("trade_through_sell", 0) + 1
            use_bid, use_ask = None, None
        else:
            if trade_ticks <= bid_ticks:
                side_stats["quote_sell"] = side_stats.get("quote_sell", 0) + 1
                return "sell"
            if trade_ticks >= ask_ticks:
                side_stats["quote_buy"] = side_stats.get("quote_buy", 0) + 1
                return "buy"

            if spread_ticks > 1:
                mid_ticks = (bid_ticks + ask_ticks) // 2
                if trade_ticks < mid_ticks:
                    side_stats["mid_sell"] = side_stats.get("mid_sell", 0) + 1
                    return "sell"
                if trade_ticks > mid_ticks:
                    side_stats["mid_buy"] = side_stats.get("mid_buy", 0) + 1
                    return "buy"
                side_stats["mid_exact"] = side_stats.get("mid_exact", 0) + 1
            else:
                side_stats["one_tick_spread"] = side_stats.get("one_tick_spread", 0) + 1

    if last_price is not None:
        if trade_price > last_price:
            inferred_tick = "buy"
        elif trade_price < last_price:
            inferred_tick = "sell"
        else:
            inferred_tick = last_tick_dir
        if inferred_tick is not None:
            side_stats["tick_used"] = side_stats.get("tick_used", 0) + 1
            return inferred_tick

    if use_last_side_fallback and last_side is not None:
        side_stats["last_side_fallback"] = side_stats.get("last_side_fallback", 0) + 1
        return last_side

    side_stats["unclassified"] = side_stats.get("unclassified", 0) + 1
    return None


# ─────────────────────────────────────────────────────────────
# Order book helpers
# ─────────────────────────────────────────────────────────────

def apply_order_to_book(row, ob, miss_counts, miss_times):
    """Apply one order message to the heap book, track misses."""
    action = row.action_type
    ok = ob.apply_action(action, row.order_id, row.side, row.price, row.volume)
    if not ok and action in ("M", "D", "Y"):
        miss_counts[action] += 1
        miss_times[action].append(row.time)


def snapshot_depth_levels(book_qty, best_price, tick_size, n_levels, direction):
    """Snapshot volume at n_levels price levels starting from best_price.

    Parameters
    ----------
    book_qty : dict
        {price: volume} from HeapOrderBook (bid_qty or ask_qty)
    best_price : float
        Starting price level (best bid or best ask)
    tick_size : float
        Minimum price increment
    n_levels : int
        Number of levels to snapshot
    direction : int
        +1 for ascending prices (ask side), -1 for descending (bid side)

    Returns
    -------
    list[int]
        Volume at each level
    """
    depths = []
    for i in range(n_levels):
        p = round(best_price + direction * i * tick_size, 8)
        depths.append(book_qty.get(p, 0))
    return depths


# ─────────────────────────────────────────────────────────────
# MO aggregation
# ─────────────────────────────────────────────────────────────

def aggregate_fills_into_mos(fills, group_gap_us=100, tick_size=0.01):
    """Group consecutive same-side fills within group_gap_us into
    single aggressive market orders.

    Parameters
    ----------
    fills : list[dict]
        Each dict has at least: time_ns, volume, price, side,
        best_bid, best_ask, ticks_from_bbo, and depth columns.
    group_gap_us : int
        Maximum gap in microseconds between fills in the same MO.
    tick_size : float
        Minimum price increment (for ticks_walked computation).

    Returns
    -------
    list[dict]
        Aggregated MOs with: first_time_ns, side, mo_volume, n_fills,
        min_price, max_price, best_bid, best_ask, ticks_walked,
        ratio_L0, microprice, and depth columns from the first fill.
    """
    if not fills:
        return []

    fills_sorted = sorted(fills, key=lambda f: f["time_ns"])

    groups = []
    current_group = [fills_sorted[0]]

    for f in fills_sorted[1:]:
        prev = current_group[-1]
        dt_us = (f["time_ns"] - prev["time_ns"]) / 1000
        if f["side"] == prev["side"] and dt_us <= group_gap_us:
            current_group.append(f)
        else:
            groups.append(current_group)
            current_group = [f]
    groups.append(current_group)

    aggregated = []
    for group in groups:
        first = group[0]
        total_vol = sum(f["volume"] for f in group)
        prices = [f["price"] for f in group]

        agg = {
            "first_time_ns": first["time_ns"],
            "side": first["side"],
            "mo_volume": total_vol,
            "n_fills": len(group),
            "min_price": min(prices),
            "max_price": max(prices),
            "best_bid": first["best_bid"],
            "best_ask": first["best_ask"],
            "ticks_walked": int(round((max(prices) - min(prices)) / tick_size)),
        }

        for k, v in first.items():
            if k.startswith(("opp_depth_", "bid_depth_", "ask_depth_")):
                agg[k] = v

        if "microprice" in first:
            agg["microprice"] = first["microprice"]
        if "cls_method" in first:
            agg["cls_method"] = first["cls_method"]

        opp_L0 = first.get("opp_depth_L0", 0)
        agg["ratio_L0"] = total_vol / opp_L0 if opp_L0 > 0 else None

        aggregated.append(agg)

    return aggregated


def merge_bookwalking_mos(mos, tick_size=0.01, ratio_tolerance=0.01):
    """Stage 2 aggregation: merge consecutive MOs that represent book-walking.

    If an MO consumes ~100% of L0 (ratio ~ 1), and the next MO is:
    - Same side, AND
    - At a different price (walked to next level)
    Then merge them into a single aggressive order.
    """
    if not mos:
        return []

    n_input = len(mos)
    mos_sorted = sorted(mos, key=lambda m: m["first_time_ns"])

    n_ratio1 = sum(
        1 for m in mos_sorted
        if m.get("ratio_L0") is not None and abs(m["ratio_L0"] - 1.0) <= ratio_tolerance
    )

    merged = []
    current_group = [mos_sorted[0]]
    n_merge_events = 0

    for mo in mos_sorted[1:]:
        prev = current_group[-1]
        prev_ratio = prev.get("ratio_L0")
        prev_consumes_L0 = (
            prev_ratio is not None and
            abs(prev_ratio - 1.0) <= ratio_tolerance
        )
        same_side = mo["side"] == prev["side"]
        price_changed = mo["min_price"] != prev["max_price"]

        if prev_consumes_L0 and same_side and price_changed:
            current_group.append(mo)
            n_merge_events += 1
        else:
            merged.append(_merge_mo_group(current_group, tick_size))
            current_group = [mo]

    merged.append(_merge_mo_group(current_group, tick_size))

    n_output = len(merged)
    print(f"    Stage 2 merge: {n_input} -> {n_output} MOs "
          f"(ratio~1: {n_ratio1}, merge events: {n_merge_events})")

    return merged


def _merge_mo_group(group, tick_size):
    """Merge a group of MOs into a single aggregated MO."""
    if len(group) == 1:
        return group[0]

    first = group[0]
    total_vol = sum(m["mo_volume"] for m in group)
    total_fills = sum(m["n_fills"] for m in group)
    all_min_prices = [m["min_price"] for m in group]
    all_max_prices = [m["max_price"] for m in group]

    merged = {
        "first_time_ns": first["first_time_ns"],
        "side": first["side"],
        "mo_volume": total_vol,
        "n_fills": total_fills,
        "min_price": min(all_min_prices),
        "max_price": max(all_max_prices),
        "best_bid": first["best_bid"],
        "best_ask": first["best_ask"],
        "ticks_walked": int(round((max(all_max_prices) - min(all_min_prices)) / tick_size)),
    }

    for k, v in first.items():
        if k.startswith(("opp_depth_", "bid_depth_", "ask_depth_")):
            merged[k] = v

    if "microprice" in first:
        merged["microprice"] = first["microprice"]
    if "cls_method" in first:
        merged["cls_method"] = first["cls_method"]

    opp_L0 = first.get("opp_depth_L0", 0)
    merged["ratio_L0"] = total_vol / opp_L0 if opp_L0 > 0 else None

    return merged


# ─────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────

def extract_events_for_day(
    orders_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    market_open: str,
    market_close: str,
    record_bbo: bool = False,
    tick_size: float = 0.01,
    return_diagnostics: bool = False,
    full_extraction: bool = False,
    day_key: str = None,
    depth_levels: int = 5,
    mo_depth_levels: int = 10,
    group_gap_us: int = 100,
):
    """Extract per-event timestamps (seconds since market open).

    Uses HeapOrderBook (heap-based) for BBO computation and the
    research-grade Lee-Ready algorithm for MO side inference:
      quote rule -> midpoint rule -> extended tick rule.

    Trades are pre-filtered so that only *continuous-session book trades*
    are considered as market orders:
      - opening_trade_indicator == 'S'  (Core Continuous)
      - trade_origin == 'B'             (Orders from the book)

    Parameters
    ----------
    tick_size : float
        Minimum price increment for the instrument.
    full_extraction : bool
        If True, also capture per-LO/CXL book-state snapshots,
        per-MO-fill depth snapshots, and aggregated aggressive-MO data.
    day_key : str
        Trading day key (e.g. '20170102').
    depth_levels : int
        Number of price levels to snapshot on each side of the book
        for LO/CXL events (default 5).
    mo_depth_levels : int
        Number of opposite-side depth levels to snapshot for each
        MO fill (default 10).
    group_gap_us : int
        Maximum gap in microseconds between fills in the same
        aggressive MO (default 100).
    """

    orders_df_full = orders_df
    orders_df = filter_market_hours(orders_df, market_open, market_close, "time")
    trades_df = filter_market_hours(trades_df, market_open, market_close, "trade_time")

    n_trades_raw = len(trades_df)
    n_dropped_oti = 0
    n_dropped_origin = 0

    if "opening_trade_indicator" in trades_df.columns:
        mask_oti = trades_df["opening_trade_indicator"] == "S"
        n_dropped_oti = int((~mask_oti).sum())
        trades_df = trades_df[mask_oti]

    if "trade_origin" in trades_df.columns:
        mask_origin = trades_df["trade_origin"] == "B"
        n_dropped_origin = int((~mask_origin).sum())
        trades_df = trades_df[mask_origin]

    if trades_df.empty:
        day_start = orders_df["time"].min()
    else:
        day_start = min(orders_df["time"].min(), trades_df["trade_time"].min())
    cutoff = day_start

    if market_open is not None and market_close is not None:
        cutoff = day_start.normalize() + pd.to_timedelta(market_open)

    _market_open_ts = day_start.normalize() + pd.to_timedelta(market_open) if market_open else None
    _market_close_ts = day_start.normalize() + pd.to_timedelta(market_close) if market_close else None

    is_bid = orders_df["side"] == 1
    is_ask = orders_df["side"].isin([2, 5])
    is_add = orders_df["action_type"] == "A"
    is_cxl = orders_df["action_type"] == "D"
    is_lo = orders_df["order_type_num"] == 2

    lo_bid = orders_df[is_add & is_lo & is_bid]["time"]
    lo_ask = orders_df[is_add & is_lo & is_ask]["time"]
    cxl_bid = orders_df[is_cxl & is_bid]["time"]
    cxl_ask = orders_df[is_cxl & is_ask]["time"]

    _fill_times_set = set(trades_df["trade_time"])
    n_fill_d_removed = 0
    if _fill_times_set:
        n_cxl_before = len(cxl_bid) + len(cxl_ask)
        cxl_bid = cxl_bid[~cxl_bid.isin(_fill_times_set)]
        cxl_ask = cxl_ask[~cxl_ask.isin(_fill_times_set)]
        n_fill_d_removed = n_cxl_before - len(cxl_bid) - len(cxl_ask)

    ob = HeapOrderBook()
    miss_counts = {"M": 0, "D": 0, "Y": 0, "SIDE": 0}
    miss_times = {"M": [], "D": [], "Y": [], "SIDE": []}
    side_stats = {
        "n_trades_raw": n_trades_raw,
        "n_dropped_oti": n_dropped_oti,
        "n_dropped_origin": n_dropped_origin,
        "n_trades": 0,
        "quote_not_strictly_prior": 0,
        "crossed_market": 0,
        "locked_market": 0,
        "trade_through_buy": 0,
        "trade_through_sell": 0,
        "quote_sell": 0,
        "quote_buy": 0,
        "mid_sell": 0,
        "mid_buy": 0,
        "mid_exact": 0,
        "one_tick_spread": 0,
        "tick_used": 0,
        "unclassified": 0,
    }

    first_stable_time = None
    if full_extraction:
        _ts_bbo, _bb_bbo, _ba_bbo = compute_bbo_series(orders_df)
        _mask_valid = ~np.isnan(_bb_bbo) & ~np.isnan(_ba_bbo)
        if _mask_valid.any():
            _ts_bbo = _ts_bbo[_mask_valid]
            _bb_bbo = _bb_bbo[_mask_valid]
            _ba_bbo = _ba_bbo[_mask_valid]
            _stable_idx, _ = find_continuous_trading_start(_bb_bbo, _ba_bbo)
            if _stable_idx < len(_ts_bbo):
                first_stable_time = pd.Timestamp(_ts_bbo[_stable_idx])
                if first_stable_time.tzinfo is None:
                    first_stable_time = first_stable_time.tz_localize(TZ)

        if first_stable_time is None:
            first_stable_time = cutoff

        _order_delta0 = {}
        _n_lo_total = 0
        _n_bid_count = 0
        _n_ask_count = 0
        _last_event_time = None
        _last_mid = None
        lo_cxl_rows = []
        mo_fill_rows = []

    orders_sorted = orders_df_full.sort_values("time")
    trades_sorted = trades_df.sort_values("trade_time")

    bbo_times = []
    bbo_bids = []
    bbo_asks = []
    last_bbo = None

    mo_bid_times = []
    mo_ask_times = []
    _mo_fill_ts = []

    order_idx = 0
    last_trade_price = None
    last_trade_side = None
    last_tick_dir = None
    last_good_bid = None
    last_good_ask = None
    last_bbo_change_ts = None
    last_good_bbo_ts = None

    orders_tuples = list(orders_sorted.itertuples(index=False))
    trades_records = trades_sorted.to_dict("records")

    if return_diagnostics:
        diag_rows = []

    # ──────────────────────────────────────────────────────────
    # Main loop: iterate trades, advance book, classify MOs
    # ──────────────────────────────────────────────────────────
    for tr in trades_records:
        t = tr["trade_time"]
        _pre_fill_snap = None

        while order_idx < len(orders_tuples) and orders_tuples[order_idx].time <= t:
            orow = orders_tuples[order_idx]

            if full_extraction and orow.time >= first_stable_time and (
                _market_close_ts is None or orow.time <= _market_close_ts
            ):
                _bb, _ba = ob.get_bbo()
                if _bb is not None and _ba is not None and _ba > _bb:
                    _is_fill_d = (orow.action_type == "D" and orow.time == t)

                    # Snapshot the book BEFORE the first fill-induced
                    # deletion so that mo_orders records reflect the
                    # state the aggressor actually traded against.
                    if _is_fill_d and _pre_fill_snap is None:
                        _pre_fill_snap = {
                            'bb': _bb, 'ba': _ba,
                            'ask_opp': snapshot_depth_levels(
                                ob.ask_qty, _ba, tick_size,
                                mo_depth_levels, +1),
                            'bid_opp': snapshot_depth_levels(
                                ob.bid_qty, _bb, tick_size,
                                mo_depth_levels, -1),
                            'bid_depths': snapshot_depth_levels(
                                ob.bid_qty, _bb, tick_size,
                                depth_levels, -1),
                            'ask_depths': snapshot_depth_levels(
                                ob.ask_qty, _ba, tick_size,
                                depth_levels, +1),
                            'bb_sz': ob.bid_qty.get(_bb, 0),
                            'ba_sz': ob.ask_qty.get(_ba, 0),
                        }

                    if not _is_fill_d:
                        _capture_order_event(
                            orow, _bb, _ba, ob, tick_size, depth_levels,
                            day_key, lo_cxl_rows,
                            _order_delta0, _n_lo_total, _n_bid_count, _n_ask_count,
                            _last_event_time, _last_mid,
                        )

                    is_lo_event = (
                        orow.action_type == "A"
                        and hasattr(orow, "order_type_num")
                        and orow.order_type_num == 2
                        and orow.side in (1, 2)
                    )
                    is_cxl_event = orow.action_type == "D" and orow.side in (1, 2)

                    if is_lo_event:
                        _n_lo_total += 1
                        if orow.side == 1:
                            _n_bid_count += 1
                        else:
                            _n_ask_count += 1
                    elif is_cxl_event:
                        _n_lo_total = max(0, _n_lo_total - 1)
                        if orow.side == 1:
                            _n_bid_count = max(0, _n_bid_count - 1)
                        else:
                            _n_ask_count = max(0, _n_ask_count - 1)

                    if (is_lo_event or is_cxl_event) and not _is_fill_d:
                        _last_event_time = orow.time
                        _last_mid = 0.5 * (_bb + _ba)

            apply_order_to_book(orow, ob, miss_counts, miss_times)

            bb, ba = ob.get_bbo()
            if bb is not None:
                current_bbo = (bb, ba)
                if current_bbo != last_bbo:
                    last_bbo_change_ts = orow.time
                    if record_bbo:
                        bbo_times.append(last_bbo_change_ts)
                        bbo_bids.append(bb)
                        bbo_asks.append(ba)
                    last_bbo = current_bbo
            order_idx += 1

        best_bid, best_ask = ob.get_bbo()
        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            last_good_bid, last_good_ask = best_bid, best_ask
            last_good_bbo_ts = last_bbo_change_ts

        trade_price = tr["price_float"]

        if return_diagnostics or full_extraction:
            _snap = {k: side_stats.get(k, 0) for k in _CLS_KEYS}

        side = infer_trade_side(
            trade_price=trade_price,
            best_bid=best_bid,
            best_ask=best_ask,
            trade_ts=t,
            quote_ts=last_bbo_change_ts,
            last_price=last_trade_price,
            last_tick_dir=last_tick_dir,
            last_side=last_trade_side,
            side_stats=side_stats,
            tick_size=tick_size,
            last_good_bid=last_good_bid,
            last_good_ask=last_good_ask,
            last_good_quote_ts=last_good_bbo_ts,
        )

        _cls_src = None
        if return_diagnostics or full_extraction:
            _cls_src = "unclassified"
            for _k in _CLS_KEYS:
                if side_stats.get(_k, 0) > _snap[_k]:
                    _cls_src = _CLS_MAP[_k]
                    break

        if return_diagnostics:
            _mid, _sp = np.nan, np.nan
            if best_bid is not None and best_ask is not None and best_bid < best_ask:
                _mid = (best_bid + best_ask) / 2.0
                _sp = round((best_ask - best_bid) / tick_size)
            _sign = 1.0 if side == "buy" else (-1.0 if side == "sell" else np.nan)
            diag_rows.append((t, trade_price, best_bid, best_ask, _mid, _sign, _cls_src, _sp))

        if side is None:
            miss_counts["SIDE"] += 1
            miss_times["SIDE"].append(t)
        elif side == "buy":
            mo_bid_times.append(t)
            _mo_fill_ts.append((t.value, "buy"))
        elif side == "sell":
            mo_ask_times.append(t)
            _mo_fill_ts.append((t.value, "sell"))

        # Use pre-fill snapshot (book state before fill-induced deletions)
        # when available; fall back to post-fill state otherwise.
        if full_extraction and side is not None:
            _fill_bb = _pre_fill_snap['bb'] if _pre_fill_snap else best_bid
            _fill_ba = _pre_fill_snap['ba'] if _pre_fill_snap else best_ask

            if _fill_bb is not None and _fill_ba is not None:
                trade_vol = tr.get("volume", 0)

                if _pre_fill_snap is not None:
                    opp_depths = (_pre_fill_snap['ask_opp'] if side == "buy"
                                  else _pre_fill_snap['bid_opp'])
                    bid_depths = _pre_fill_snap['bid_depths']
                    ask_depths = _pre_fill_snap['ask_depths']
                    _bb_sz = _pre_fill_snap['bb_sz']
                    _ba_sz = _pre_fill_snap['ba_sz']
                else:
                    if side == "buy":
                        opp_depths = snapshot_depth_levels(
                            ob.ask_qty, _fill_ba, tick_size,
                            mo_depth_levels, +1)
                    else:
                        opp_depths = snapshot_depth_levels(
                            ob.bid_qty, _fill_bb, tick_size,
                            mo_depth_levels, -1)
                    bid_depths = snapshot_depth_levels(
                        ob.bid_qty, _fill_bb, tick_size, depth_levels, -1)
                    ask_depths = snapshot_depth_levels(
                        ob.ask_qty, _fill_ba, tick_size, depth_levels, +1)
                    _bb_sz = ob.bid_qty.get(_fill_bb, 0)
                    _ba_sz = ob.ask_qty.get(_fill_ba, 0)

                _tfb = round(abs(trade_price - (_fill_ba if side == "buy"
                                                else _fill_bb)) / tick_size)
                _denom = _bb_sz + _ba_sz
                _microprice = (
                    (_fill_bb * _ba_sz + _fill_ba * _bb_sz) / _denom
                    if _denom > 0
                    else 0.5 * (_fill_bb + _fill_ba)
                )

                fill_dict = {
                    "time_ns": int(t.value) if hasattr(t, "value") else int(pd.Timestamp(t).value),
                    "volume": trade_vol,
                    "price": trade_price,
                    "side": side,
                    "cls_method": _cls_src,
                    "best_bid": _fill_bb,
                    "best_ask": _fill_ba,
                    "ticks_from_bbo": _tfb,
                    "microprice": _microprice,
                }
                for i in range(mo_depth_levels):
                    fill_dict[f"opp_depth_L{i}"] = int(opp_depths[i])
                for i in range(depth_levels):
                    fill_dict[f"bid_depth_L{i}"] = int(bid_depths[i])
                    fill_dict[f"ask_depth_L{i}"] = int(ask_depths[i])

                mo_fill_rows.append(fill_dict)

        if last_trade_price is not None:
            if trade_price > last_trade_price:
                last_tick_dir = "buy"
            elif trade_price < last_trade_price:
                last_tick_dir = "sell"
        last_trade_price = trade_price
        if side is not None:
            last_trade_side = side

    # ── Process remaining orders ──
    while order_idx < len(orders_tuples):
        orow = orders_tuples[order_idx]

        if full_extraction and first_stable_time is not None and orow.time >= first_stable_time and (
            _market_close_ts is None or orow.time <= _market_close_ts
        ):
            _bb, _ba = ob.get_bbo()
            if _bb is not None and _ba is not None and _ba > _bb:
                _capture_order_event(
                    orow, _bb, _ba, ob, tick_size, depth_levels,
                    day_key, lo_cxl_rows,
                    _order_delta0, _n_lo_total, _n_bid_count, _n_ask_count,
                    _last_event_time, _last_mid,
                )
                is_lo_event = (
                    orow.action_type == "A"
                    and hasattr(orow, "order_type_num")
                    and orow.order_type_num == 2
                    and orow.side in (1, 2)
                )
                is_cxl_event = orow.action_type == "D" and orow.side in (1, 2)

                if is_lo_event:
                    _n_lo_total += 1
                    if orow.side == 1:
                        _n_bid_count += 1
                    else:
                        _n_ask_count += 1
                elif is_cxl_event:
                    _n_lo_total = max(0, _n_lo_total - 1)
                    if orow.side == 1:
                        _n_bid_count = max(0, _n_bid_count - 1)
                    else:
                        _n_ask_count = max(0, _n_ask_count - 1)

                if is_lo_event or is_cxl_event:
                    _last_event_time = orow.time
                    _last_mid = 0.5 * (_bb + _ba)

        apply_order_to_book(orow, ob, miss_counts, miss_times)

        if record_bbo:
            bb, ba = ob.get_bbo()
            if bb is not None:
                current_bbo = (bb, ba)
                if current_bbo != last_bbo:
                    bbo_times.append(orow.time)
                    bbo_bids.append(bb)
                    bbo_asks.append(ba)
                    last_bbo = current_bbo
        order_idx += 1

    # ── Aggregate per-fill timestamps into per-MO timestamps ──
    if _mo_fill_ts:
        _mo_fill_ts.sort(key=lambda x: x[0])
        agg_bid_times = []
        agg_ask_times = []
        prev_ns, prev_side = _mo_fill_ts[0]
        if prev_side == "buy":
            agg_bid_times.append(pd.Timestamp(prev_ns, tz=TZ))
        else:
            agg_ask_times.append(pd.Timestamp(prev_ns, tz=TZ))

        for ns, s in _mo_fill_ts[1:]:
            dt_us = (ns - prev_ns) / 1000
            if s == prev_side and dt_us <= group_gap_us:
                pass
            else:
                if s == "buy":
                    agg_bid_times.append(pd.Timestamp(ns, tz=TZ))
                else:
                    agg_ask_times.append(pd.Timestamp(ns, tz=TZ))
            prev_ns, prev_side = ns, s

        n_fills_total = len(mo_bid_times) + len(mo_ask_times)
        n_mos_total = len(agg_bid_times) + len(agg_ask_times)
        mo_bid_times = agg_bid_times
        mo_ask_times = agg_ask_times
    else:
        n_fills_total = 0
        n_mos_total = 0

    mo_orders = []
    if full_extraction:
        mo_orders_stage1 = aggregate_fills_into_mos(
            mo_fill_rows, group_gap_us=group_gap_us, tick_size=tick_size
        )
        mo_orders = merge_bookwalking_mos(
            mo_orders_stage1, tick_size=tick_size
        )

    result = {
        "MO_bid": to_seconds_since(pd.Series(mo_bid_times), cutoff),
        "MO_ask": to_seconds_since(pd.Series(mo_ask_times), cutoff),
        "LO_bid": to_seconds_since(lo_bid, cutoff),
        "LO_ask": to_seconds_since(lo_ask, cutoff),
        "CXL_bid": to_seconds_since(cxl_bid, cutoff),
        "CXL_ask": to_seconds_since(cxl_ask, cutoff),
        "_miss_counts": miss_counts,
        "_miss_times": miss_times,
        "_side_stats": side_stats,
        "_n_fills": n_fills_total,
        "_n_mos": n_mos_total,
        "_n_fill_d_removed": n_fill_d_removed,
        "_bbo": {
            "times": bbo_times,
            "best_bid": bbo_bids,
            "best_ask": bbo_asks,
        },
    }
    if return_diagnostics:
        result["_diagnostics"] = pd.DataFrame(
            diag_rows,
            columns=["time", "price", "best_bid", "best_ask",
                     "mid", "sign", "source", "spread_ticks"],
        )
    if full_extraction:
        result["_lo_cxl_rows"] = lo_cxl_rows
        result["_mo_fills"] = mo_fill_rows
        result["_mo_orders"] = mo_orders

    return result


def _capture_order_event(
    orow, best_bid, best_ask, ob, tick_size, depth_levels,
    day_key, lo_cxl_rows,
    order_delta0, n_lo_total, n_bid_count, n_ask_count,
    last_event_time, last_mid,
):
    """Capture a LO or CXL event into lo_cxl_rows (in-place append).

    Book state is read BEFORE the order is applied to the book.
    Only LO additions (A + order_type_num==2) and cancellations (D) are recorded.
    """
    spread = best_ask - best_bid
    spread_ticks = int(round(spread / tick_size))
    total_bid = sum(ob.bid_qty.values())
    total_ask = sum(ob.ask_qty.values())
    bb_size = ob.bid_qty.get(best_bid, 0)
    ba_size = ob.ask_qty.get(best_ask, 0)
    total = total_bid + total_ask
    imbalance = (total_bid - total_ask) / total if total > 0 else 0.0
    mid_price = 0.5 * (best_bid + best_ask)
    ticks_from_mid = int(round((orow.price - mid_price) / tick_size))

    denom = bb_size + ba_size
    microprice = (
        (best_bid * ba_size + best_ask * bb_size) / denom
        if denom > 0
        else mid_price
    )

    dp_mid = (mid_price - last_mid) if last_mid is not None else None

    bid_depths = snapshot_depth_levels(ob.bid_qty, best_bid, tick_size, depth_levels, -1)
    ask_depths = snapshot_depth_levels(ob.ask_qty, best_ask, tick_size, depth_levels, +1)

    if last_event_time is None:
        dt_prev = None
    else:
        dt_prev = (orow.time - last_event_time).total_seconds()

    is_lo = (
        orow.action_type == "A"
        and hasattr(orow, "order_type_num")
        and orow.order_type_num == 2
        and orow.side in (1, 2)
    )
    if is_lo:
        if orow.side == 1:
            ticks_from_best = int(round((best_bid - orow.price) / tick_size))
            best_same = best_bid
            delta0 = orow.price - best_ask
            queue_ahead = ob.bid_qty.get(orow.price, 0)
        else:
            ticks_from_best = int(round((orow.price - best_ask) / tick_size))
            best_same = best_ask
            delta0 = best_bid - orow.price
            queue_ahead = ob.ask_qty.get(orow.price, 0)

        order_delta0[orow.order_id] = delta0

        lo_cxl_rows.append((
            day_key, orow.time.isoformat(), "LO", orow.order_id, orow.side,
            orow.price, best_bid, best_ask, best_same,
            bb_size, ba_size, total_bid, total_ask,
            mid_price, ticks_from_mid, spread, spread_ticks,
            imbalance, ticks_from_best, queue_ahead, orow.volume,
            delta0, None, None, dt_prev,
            n_lo_total, 0,
            microprice, n_bid_count, n_ask_count, dp_mid,
            *bid_depths, *ask_depths,
        ))
        return

    is_cxl = orow.action_type == "D" and orow.side in (1, 2)
    if is_cxl:
        delta0 = order_delta0.get(orow.order_id, None)
        if orow.side == 1:
            delta_t = orow.price - best_ask
        else:
            delta_t = best_bid - orow.price

        y_ratio = None
        if delta0 is not None and delta0 != 0:
            y_ratio = delta_t / delta0

        lo_cxl_rows.append((
            day_key, orow.time.isoformat(), "CXL", orow.order_id, orow.side,
            orow.price, best_bid, best_ask,
            best_bid if orow.side == 1 else best_ask,
            bb_size, ba_size, total_bid, total_ask,
            mid_price, ticks_from_mid, spread, spread_ticks,
            imbalance, 0, None, orow.volume,
            delta0, delta_t, y_ratio, dt_prev,
            n_lo_total, 1,
            microprice, n_bid_count, n_ask_count, dp_mid,
            *bid_depths, *ask_depths,
        ))


# ─────────────────────────────────────────────────────────────
# Convenience: run full extraction to SQLite
# ─────────────────────────────────────────────────────────────

def run_full_extraction(
    asset: str,
    orders_h5: Path,
    trades_h5: Path,
    db_path: Path,
    *,
    market_open: str = "09:00:00",
    market_close: str = "16:50:00",
    tick_size: float = 0.01,
    time_field: str = "time",
    force: bool = False,
    day_keys: Optional[List[str]] = None,
):
    """Run the full extraction pipeline from raw HDF5 to SQLite.

    Parameters
    ----------
    asset : str
        Asset name (for logging).
    orders_h5 : Path
        Path to the HDF5 orders file.
    trades_h5 : Path
        Path to the HDF5 trades file.
    db_path : Path
        Output SQLite database path.
    force : bool
        If True, drop and recreate tables.
    day_keys : list or None
        Specific day keys to extract.  If ``None``, all days in the HDF5
        file are used.
    """
    if day_keys is None:
        day_keys = list_day_keys_hdf(orders_h5)

    print(f"Extracting {len(day_keys)} days for {asset} -> {Path(db_path).name}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    if force:
        cur.execute("DROP TABLE IF EXISTS orders")
        cur.execute("DROP TABLE IF EXISTS fills")
        cur.execute("DROP TABLE IF EXISTS mo_orders")
    cur.execute(CREATE_ORDERS_TABLE)
    cur.execute(CREATE_FILLS_TABLE)
    cur.execute(CREATE_MO_ORDERS_TABLE)
    conn.commit()

    for i, day_key in enumerate(day_keys):
        orders_df = load_orders_day(orders_h5, day_key)
        trades_df = load_trades_day(trades_h5, day_key, time_field)

        events = extract_events_for_day(
            orders_df, trades_df,
            market_open, market_close,
            tick_size=tick_size,
            full_extraction=True,
            day_key=day_key,
        )

        lo_cxl_rows = events.pop("_lo_cxl_rows", [])
        mo_fills = events.pop("_mo_fills", [])
        mo_orders = events.pop("_mo_orders", [])

        if lo_cxl_rows:
            cur.executemany(_INSERT_ORDERS_SQL, lo_cxl_rows)

        if mo_fills:
            fill_tuples = []
            for f in mo_fills:
                fill_tuples.append((
                    day_key,
                    f["time_ns"], f["volume"], f["price"],
                    f["side"], f.get("cls_method"),
                    f["best_bid"], f["best_ask"],
                    f["ticks_from_bbo"], f.get("microprice"),
                    *[f.get(f"opp_depth_L{j}", 0) for j in range(10)],
                    *[f.get(f"bid_depth_L{j}", 0) for j in range(5)],
                    *[f.get(f"ask_depth_L{j}", 0) for j in range(5)],
                ))
            cur.executemany(_INSERT_FILLS_SQL, fill_tuples)

        if mo_orders:
            mo_tuples = []
            for o in mo_orders:
                mo_tuples.append((
                    day_key,
                    o["first_time_ns"], o["side"], o.get("cls_method"),
                    o["mo_volume"], o["n_fills"],
                    o["min_price"], o["max_price"],
                    o["best_bid"], o["best_ask"],
                    o["ticks_walked"], o.get("ratio_L0"),
                    o.get("microprice"),
                    *[o.get(f"opp_depth_L{j}", 0) for j in range(10)],
                    *[o.get(f"bid_depth_L{j}", 0) for j in range(5)],
                    *[o.get(f"ask_depth_L{j}", 0) for j in range(5)],
                ))
            cur.executemany(_INSERT_MO_SQL, mo_tuples)

        conn.commit()

        n_lo = len(lo_cxl_rows)
        n_mo = len(mo_orders)
        print(f"  [{i+1}/{len(day_keys)}] {day_key}  LO/CXL={n_lo}  MOs={n_mo}")

    conn.close()
    print(f"\nExtraction complete: {Path(db_path).name}")

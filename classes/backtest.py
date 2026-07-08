"""Replay-based market-maker backtester for empirical and simulated SQLite databases.

Feeds recorded BBO updates and market-order events to any agent that
implements the standard ``on_event(sim, t, fills)`` / ``liquidate(sim, t)``
protocol, then collects per-window statistics (PnL, trades, inventory,
drawdown) for distributional analysis and parameter sweeps.
"""

from __future__ import annotations

import itertools
import math
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hawkes_filter import (
    HawkesFilter,
    HawkesFilterFactory,
    classify_event,
    classify_mo,
    resolve_filter_factory,
)
from .helpers import resolve_data_path
from .market_maker import _write_mm_sqlite, mm_spendable

# --- ReplaySimulator ---

class ReplaySimulator:
    """Feeds recorded BBO to an MM agent and detects paper fills using
    the depth snapshot from the empirical / simulated data.

    Exposes the same ``agent_place_order``, ``agent_cancel_order``,
    ``agent_market_order`` interface that ``Simulate`` provides so any
    agent written for the full simulator works here unchanged.

    **BBO units** — ``bbo_in_tick_index`` and ``price_native_to_pln`` tell
    agents how to convert the mid to economic (PLN) scale for models such
    as Avellaneda–Stoikov: ``mid_pln = mid * price_native_to_pln``.  For
    empirical WSE replay, BBO is usually already PLN (``bbo_in_tick_index``
    False, ``price_native_to_pln`` 1).  For tick-indexed books (full
    ``Simulate`` or simulated SQLite), set ``bbo_in_tick_index`` True and
    ``price_native_to_pln`` to PLN per tick (often equal to ``tick_size``).
    """

    class _OB:
        """Minimal order-book facade expected by MM agents."""

        def __init__(self):
            self.order_map: Dict[int, list] = {}
            self._bb: Optional[float] = None
            self._ba: Optional[float] = None

        def get_bbo(self) -> Tuple[Optional[float], Optional[float]]:
            return self._bb, self._ba

    def __init__(
        self,
        tick_size: float = 1.0,
        *,
        bbo_in_tick_index: bool = False,
        price_native_to_pln: Optional[float] = None,
    ):
        self.ob = self._OB()
        self.tick_size = tick_size
        self.bbo_in_tick_index = bool(bbo_in_tick_index)
        if price_native_to_pln is not None:
            self.price_native_to_pln = float(price_native_to_pln)
        elif self.bbo_in_tick_index:
            self.price_native_to_pln = float(tick_size)
        else:
            self.price_native_to_pln = 1.0
        self._next_oid = 1_000_000
        self.mm_bid: Optional[Tuple[int, float, int]] = None
        self.mm_ask: Optional[Tuple[int, float, int]] = None
        self.hawkes_filter: Optional[HawkesFilter] = None
        self.book_state: Optional[SimpleNamespace] = None

    def update_bbo(self, bb, ba):
        bb, ba = float(bb), float(ba)
        if np.isfinite(bb) and np.isfinite(ba):
            self.ob._bb = bb
            self.ob._ba = ba

    # --- agent interface (paper orders) ---

    def agent_place_order(self, side: int, price, volume, t) -> int:
        oid = self._next_oid
        self._next_oid += 1
        self.ob.order_map[oid] = [side, price, volume]
        if side == 1:
            self.mm_bid = (oid, price, volume)
        else:
            self.mm_ask = (oid, price, volume)
        return oid

    def agent_cancel_order(self, oid: int, t) -> bool:
        self.ob.order_map.pop(oid, None)
        if self.mm_bid is not None and self.mm_bid[0] == oid:
            self.mm_bid = None
        if self.mm_ask is not None and self.mm_ask[0] == oid:
            self.mm_ask = None
        return True

    def agent_market_order(self, side: int, volume, t) -> List[Tuple[float, int]]:
        """Liquidation: fill at current best price (paper trade)."""
        bb, ba = self.ob.get_bbo()
        if side == 1 and ba is not None:
            return [(ba, int(volume))]
        if side == 2 and bb is not None:
            return [(bb, int(volume))]
        return []

    # --- fill detection ---

    def _check_side(self, mm_order, level_sign, ref_price, mo_volume, opp_depth):
        if mm_order is None:
            return None
        oid, px, vol = mm_order
        ts = self.tick_size
        level = round(level_sign * (px - ref_price) / ts)

        if level < 0:
            fill_qty = int(min(float(mo_volume), vol))
        elif level < len(opp_depth):
            depth_arr = np.nan_to_num(opp_depth[: level + 1], nan=0.0)
            remaining = float(mo_volume) - float(np.sum(depth_arr))
            if remaining > 0:
                fill_qty = int(min(remaining, vol))
            else:
                fill_qty = 0
        else:
            return None

        if fill_qty <= 0:
            return None

        if level_sign == 1:
            resting_side = 2
        else:
            resting_side = 1
        new_vol = vol - fill_qty
        if new_vol <= 0:
            self.ob.order_map.pop(oid, None)
            if resting_side == 2:
                self.mm_ask = None
            else:
                self.mm_bid = None
        else:
            self.ob.order_map[oid][2] = new_vol
            if resting_side == 2:
                self.mm_ask = (oid, px, new_vol)
            else:
                self.mm_bid = (oid, px, new_vol)

        return (oid, px, fill_qty, resting_side)

    def process_mo(self, mo_side, mo_volume, best_bid, best_ask,
                   opp_depth) -> list:
        """Check whether an empirical MO would fill the MM's resting order.

        ``mo_side`` accepts both string (``"buy"``/``"sell"``) for empirical
        databases and integer (``1``/``2``) for simulated databases.
        """
        if mo_side in ("buy", 1):
            fill = self._check_side(
                self.mm_ask, +1, best_ask, mo_volume, opp_depth
            )
        elif mo_side in ("sell", 2):
            fill = self._check_side(
                self.mm_bid, -1, best_bid, mo_volume, opp_depth
            )
        else:
            fill = None
        if fill is not None:
            return [fill]
        return []


class CashCappedReplaySimulator(ReplaySimulator):
    """Replay simulator that clips MM *buy* fills to ``mm_spendable``.

    ``ReplaySimulator._check_side`` applies the fill to the book before
    ``on_event`` runs; constrained agents must therefore execute at most
    ``floor(spendable / price)`` on bid-side hits (BBO marks on
    inventory) so resting volume stays consistent with the capital
    rule.
    """

    def __init__(
        self,
        tick_size: float = 1.0,
        *,
        bbo_in_tick_index: bool = False,
        price_native_to_pln: Optional[float] = None,
    ):
        super().__init__(
            tick_size,
            bbo_in_tick_index=bbo_in_tick_index,
            price_native_to_pln=price_native_to_pln,
        )
        self._cash_agent = None

    def set_agent(self, agent) -> None:
        self._cash_agent = agent

    def _check_side(self, mm_order, level_sign, ref_price, mo_volume, opp_depth):
        if mm_order is None:
            return None
        oid, px, vol = mm_order
        ts = self.tick_size
        level = round(level_sign * (px - ref_price) / ts)

        if level < 0:
            fill_qty = int(min(float(mo_volume), vol))
        elif level < len(opp_depth):
            depth_arr = np.nan_to_num(opp_depth[: level + 1], nan=0.0)
            remaining = float(mo_volume) - float(np.sum(depth_arr))
            if remaining > 0:
                fill_qty = int(min(remaining, vol))
            else:
                fill_qty = 0
        else:
            return None

        if fill_qty <= 0:
            return None

        if level_sign == 1:
            resting_side = 2
        else:
            resting_side = 1
        ag = self._cash_agent
        if getattr(ag, "_constrained", False) and resting_side == 1:
            px_f = float(px)
            if px_f <= 0 or not math.isfinite(px_f):
                return None
            bb, ba = self.ob.get_bbo()
            sp = mm_spendable(float(ag.cash), int(ag.inventory), bb, ba)
            cap = int(sp // px_f)
            fill_qty = min(fill_qty, cap)
            if fill_qty <= 0:
                return None

        new_vol = vol - fill_qty
        if new_vol <= 0:
            self.ob.order_map.pop(oid, None)
            if resting_side == 2:
                self.mm_ask = None
            else:
                self.mm_bid = None
        else:
            self.ob.order_map[oid][2] = new_vol
            if resting_side == 2:
                self.mm_ask = (oid, px, new_vol)
            else:
                self.mm_bid = (oid, px, new_vol)

        return (oid, px, fill_qty, resting_side)


# --- SweepResult ---

@dataclass
class SweepResult:
    """Stores per-window results for every parameter combination in a sweep.

    Attributes
    ----------
    param_names : list[str]
        Names of the swept parameters (in order).
    grid : dict
        ``{(val_0, val_1, ...): pd.DataFrame}`` – one results DataFrame
        per parameter combination, same schema as ``MMBacktester.run_all``.
    """
    param_names: List[str] = field(default_factory=list)
    grid: Dict[tuple, pd.DataFrame] = field(default_factory=dict)

    def mean(self, metric: str = "pnl") -> Dict[tuple, float]:
        return {k: float(df[metric].mean()) for k, df in self.grid.items()}

    def std(self, metric: str = "pnl") -> Dict[tuple, float]:
        return {k: float(df[metric].std()) for k, df in self.grid.items()}

    def to_summary_df(self, metric: str = "pnl") -> pd.DataFrame:
        """One-row-per-combo summary with mean, std, median, count."""
        rows = []
        for combo, df in self.grid.items():
            row = dict(zip(self.param_names, combo))
            row[f"{metric}_mean"] = df[metric].mean()
            row[f"{metric}_std"] = df[metric].std()
            row[f"{metric}_median"] = df[metric].median()
            row["n_windows"] = len(df)
            rows.append(row)
        return pd.DataFrame(rows)


# --- MMBacktester ---

_OPP_DEPTH_COLS = [f"opp_depth_L{i}" for i in range(10)]
_BID_DEPTH_COLS = [f"bid_depth_L{i}" for i in range(40)]
_ASK_DEPTH_COLS = [f"ask_depth_L{i}" for i in range(40)]


class MMBacktester:
    """Replay-based market-maker backtester.

    Works with both empirical (WSE) and simulated SQLite databases.
    Empirical DBs use trading days as natural windows; simulated DBs
    use N-event windows (user-specified).

    Parameters
    ----------
    db_path : str or Path
        Path to a SQLite database.  Relative paths are resolved via
        ``resolve_data_path`` (i.e. relative to ``research_core/data/``).
    tick_size : float or None
        Native price step for replay fill geometry (difference ``px − ref``
        divided by this when mapping to depth levels).  Defaults to 0.05
        for empirical (PLN BBO) and **1.0** for simulated **tick-index**
        BBO (one integer tick per step).
    db_type : str
        ``"auto"`` (default), ``"empirical"``, or ``"simulated"``.
    bbo_in_tick_index : bool or None
        If True, ``best_bid`` / ``best_ask`` are tick indices (``Simulate``).
        If False, they are already PLN (typical empirical DB).  ``None`` =
        False for empirical, True for simulated.
    price_native_to_pln : float or None
        PLN per native price unit (``mid_pln = mid * price_native_to_pln``).
        For tick-index BBO, set to venue tick size in PLN (default **0.05**
        for simulated DBs to match ``Simulate``'s default).  For PLN BBO,
        use ``1.0`` (default when ``bbo_in_tick_index`` is False).
    """

    def __init__(self, db_path: Union[str, Path], *,
                 tick_size: Optional[float] = None,
                 db_type: str = "auto",
                 bbo_in_tick_index: Optional[bool] = None,
                 price_native_to_pln: Optional[float] = None,
                 load_book_state: bool = False,
                 hawkes: Union[
                     None, bool, HawkesFilter, HawkesFilterFactory
                 ] = True,
                 skip_opening: bool = False):
        """
        Parameters
        ----------
        load_book_state
            If True, load 10-level bid/ask depth snapshots and imbalance
            from the ``orders`` table and expose them on
            ``sim.book_state`` during replay.  Required for NN
            fill-probability callbacks that need live book features.
        hawkes
            Online Hawkes intensity filter to attach to the replay
            simulator.  ``True`` (default) installs the KGHM **single-kernel**
            multivariate calibration shared with
            ``Simulate(arrival_mode='hawkes_multivariate')``.
            Pass a :class:`HawkesFilter` instance to use its parameters
            as a template (a fresh clone is built per window), a
            zero-arg callable for full control, or ``False`` / ``None``
            to disable.  When enabled, each window's simulator exposes
            the live filter as ``sim.hawkes_filter``.
        skip_opening
            If True, detect a sustained flat BBO period at the start of
            empirical trading days (opening auction + initial settling)
            and skip agent interaction until the continuous session
            begins.  The Hawkes filter and BBO state are still updated
            during the skipped period so that the agent starts with
            correct state.  Disabled by default.
        """
        self.db_path = resolve_data_path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.load_book_state = bool(load_book_state)
        self.skip_opening = bool(skip_opening)
        if db_type == "auto":
            self.db_type = self._detect_db_type()
        else:
            self.db_type = db_type
        if tick_size is not None:
            self.tick_size = tick_size
        elif self.db_type == "empirical":
            self.tick_size = 0.05
        else:
            self.tick_size = 1.0
        if bbo_in_tick_index is None:
            self.bbo_in_tick_index = self.db_type != "empirical"
        else:
            self.bbo_in_tick_index = bool(bbo_in_tick_index)
        if price_native_to_pln is not None:
            self.price_native_to_pln = float(price_native_to_pln)
        elif not self.bbo_in_tick_index:
            self.price_native_to_pln = 1.0
        elif self.db_type == "simulated":
            # Match ``Simulate(..., tick_size=0.05)`` when recording tick BBO.
            self.price_native_to_pln = 0.05
        else:
            self.price_native_to_pln = float(self.tick_size)

        self._hawkes_factory: Optional[HawkesFilterFactory] = (
            resolve_filter_factory(hawkes)
        )

    # --- DB introspection ---

    def _detect_db_type(self) -> str:
        conn = sqlite3.connect(str(self.db_path))
        try:
            cols = [
                row[1]
                for row in conn.execute("PRAGMA table_info(orders)").fetchall()
            ]
            if "day" in cols:
                return "empirical"
            return "simulated"
        finally:
            conn.close()

    # --- window listing ---

    def list_windows(self, window_size: Optional[int] = None) -> list:
        """Return the list of window identifiers.

        For empirical DBs each window is a day key string (e.g. ``"d20170111"``).
        For simulated DBs windows are ``(start_row, end_row)`` tuples of
        *window_size* LO/CXL events each.  Market orders falling within
        the same timestamp range are loaded separately.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            if self.db_type == "empirical":
                rows = conn.execute(
                    "SELECT DISTINCT day FROM orders ORDER BY day"
                ).fetchall()
                return [r[0] for r in rows]
            else:
                n_orders = conn.execute(
                    "SELECT COUNT(*) FROM orders"
                ).fetchone()[0]
                ws = window_size or 10_000
                windows = []
                for start in range(0, n_orders, ws):
                    end = min(start + ws, n_orders)
                    if end - start > 0:
                        windows.append((start, end))
                return windows
        finally:
            conn.close()

    # --- data loading ---

    def _load_empirical_day(self, conn: sqlite3.Connection,
                            day_id: str):
        """Load orders + mo_orders DataFrames for one empirical day."""
        base_cols = (
            "SELECT timestamp, event_type, order_id, side, order_price, "
            "volume, best_bid, best_ask"
        )
        if self.load_book_state:
            base_cols += ", " + ", ".join(_BID_DEPTH_COLS)
            base_cols += ", " + ", ".join(_ASK_DEPTH_COLS)
            base_cols += ", imbalance"
        orders_df = pd.read_sql_query(
            base_cols + " FROM orders WHERE day = ? ORDER BY timestamp",
            conn,
            params=(day_id,),
        )
        if not orders_df.empty:
            ts_col = orders_df["timestamp"]
            if pd.api.types.is_numeric_dtype(ts_col):
                orders_df["time_ns"] = ts_col.astype("int64")
            else:
                # KGHM empirical DB mixes ISO8601 with/without fractional seconds.
                try:
                    orders_df["time_ns"] = (
                        pd.to_datetime(ts_col, utc=True).astype("int64")
                    )
                except ValueError:
                    orders_df["time_ns"] = (
                        pd.to_datetime(ts_col, format="mixed", utc=True)
                        .astype("int64")
                    )

        mos_df = pd.read_sql_query(
            "SELECT first_time_ns AS time_ns, side AS mo_side, mo_volume, "
            "ticks_walked, best_bid, best_ask, "
            + ", ".join(_OPP_DEPTH_COLS)
            + " FROM mo_orders WHERE day = ? ORDER BY first_time_ns",
            conn,
            params=(day_id,),
        )
        return orders_df, mos_df

    def _load_sim_window(self, conn: sqlite3.Connection,
                         start: int, end: int):
        """Load a chunk of simulated events as (orders_df, mos_df).

        ``start`` / ``end`` are row offsets into the ``orders`` table.
        Market orders are loaded by matching the timestamp range of the
        loaded orders so that both tables stay synchronised.
        """
        base_cols = (
            "SELECT timestamp AS time_ns, event_type, order_id, side, "
            "order_price, volume, best_bid, best_ask"
        )
        if self.load_book_state:
            base_cols += ", " + ", ".join(_BID_DEPTH_COLS)
            base_cols += ", " + ", ".join(_ASK_DEPTH_COLS)
            base_cols += ", imbalance"
        orders_df = pd.read_sql_query(
            base_cols + " FROM orders ORDER BY timestamp "
            f"LIMIT {end - start} OFFSET {start}",
            conn,
        )

        if orders_df.empty:
            mos_df = pd.DataFrame()
        else:
            t_min = float(orders_df["time_ns"].iloc[0])
            t_max = float(orders_df["time_ns"].iloc[-1])
            mos_df = pd.read_sql_query(
                "SELECT timestamp AS time_ns, side AS mo_side, mo_volume, "
                "ticks_walked, best_bid, best_ask, "
                + ", ".join(_OPP_DEPTH_COLS)
                + " FROM mo_orders WHERE timestamp >= ? AND timestamp <= ? "
                "ORDER BY timestamp",
                conn,
                params=(t_min, t_max),
            )

        return orders_df, mos_df

    # --- replay engine ---

    def _replay(self, orders_df: pd.DataFrame, mos_df: pd.DataFrame,
                agent, *, replay_start_s: Optional[float] = None,
                max_replay_s: Optional[float] = None) -> dict:
        """Run the event-by-event replay loop and return per-window stats.

        Stats are computed as per-window deltas so that the results are
        correct both when a fresh agent is created per window (factory
        mode) and when the same agent is reused across windows
        (persistent mode).

        Parameters
        ----------
        replay_start_s : float, optional
            Seconds from window start before the measurement window opens.
            Events before this time still update BBO / Hawkes state but do
            not invoke the agent (warm-up).  Stats are measured from this
            point.  Defaults to ``0``.
        max_replay_s : float, optional
            Length of the measurement window in seconds, starting at
            ``replay_start_s``.  When ``replay_start_s`` is ``0`` this
            matches the legacy behaviour (stop at ``max_replay_s`` from
            window open).
        """
        if orders_df.empty:
            return {}

        replay_start_seconds = float(replay_start_s or 0.0)
        if max_replay_s is not None:
            replay_end_seconds = replay_start_seconds + float(max_replay_s)
        else:
            replay_end_seconds = None

        inv_before = getattr(agent, "inventory", 0)
        cash_before = agent.cash
        trades_before = len(agent.trade_log)
        snaps_before = len(agent.pnl_snapshots)
        measuring = replay_start_seconds <= 0.0

        sim_kw = dict(
            tick_size=self.tick_size,
            bbo_in_tick_index=self.bbo_in_tick_index,
            price_native_to_pln=self.price_native_to_pln,
        )
        if getattr(agent, "_constrained", False):
            sim = CashCappedReplaySimulator(**sim_kw)
            sim.set_agent(agent)
        else:
            sim = ReplaySimulator(**sim_kw)

        # Online Hawkes filter (fresh per window so day boundaries reset).
        if self._hawkes_factory is not None:
            hf: Optional[HawkesFilter] = self._hawkes_factory()
            hf.reset(t0=0.0)
        else:
            hf = None
        sim.hawkes_filter = hf

        o_times = orders_df["time_ns"].values
        o_bb = orders_df["best_bid"].values
        o_ba = orders_df["best_ask"].values
        o_etype = orders_df["event_type"].values
        o_side = orders_df["side"].values
        n_o = len(o_times)

        has_book = self.load_book_state and _BID_DEPTH_COLS[0] in orders_df.columns
        if has_book:
            o_bid_depth = orders_df[_BID_DEPTH_COLS].values
            o_ask_depth = orders_df[_ASK_DEPTH_COLS].values
            o_imbalance = orders_df["imbalance"].values
            sim.book_state = SimpleNamespace(
                bid_depths=np.zeros(40, dtype=np.float64),
                ask_depths=np.zeros(40, dtype=np.float64),
                imbalance=0.0,
            )

        if len(mos_df) > 0:
            m_times = mos_df["time_ns"].values
            m_bb = mos_df["best_bid"].values
            m_ba = mos_df["best_ask"].values
            m_side = mos_df["mo_side"].values
            m_vol = mos_df["mo_volume"].values
            m_depth = mos_df[_OPP_DEPTH_COLS].values
            n_m = len(m_times)
        else:
            n_m = 0

        oi, mi = 0, 0

        if self.db_type == "empirical":
            t0 = o_times[0]
            if n_m > 0:
                t0 = min(t0, m_times[0])
            INT_MAX = np.iinfo(np.int64).max
            time_scale = 1e9
        else:
            t0 = 0.0
            INT_MAX = float("inf")
            time_scale = 1.0

        # Opening detection: skip the auction/settling period at start of day.
        # The opening auction produces a price jump then a sustained flat
        # period where mid barely moves.  We detect the transition to
        # continuous trading by counting mid-price *reversals* (direction
        # changes).  The flat period has 0-1 reversals; once we see several,
        # the continuous session has begun.  During the skip, Hawkes filter
        # and BBO state still update so the agent starts with correct context.
        opening_active = (
            self.skip_opening and self.db_type == "empirical"
        )
        if opening_active:
            _open_prev_mid: Optional[float] = None
            _open_prev_dir: int = 0  # +1 up, -1 down, 0 unknown
            _open_reversals: int = 0
            _OPEN_REVERSAL_THRESH = 4
            _OPEN_MIN_MOVE = self.tick_size * 0.5

        t = 0.0
        while oi < n_o or mi < n_m:
            if oi < n_o:
                ot = o_times[oi]
            else:
                ot = INT_MAX
            if mi < n_m:
                mt = m_times[mi]
            else:
                mt = INT_MAX

            next_t = float(min(ot, mt) - t0) / time_scale
            if replay_end_seconds is not None and next_t >= replay_end_seconds:
                break

            if ot <= mt:
                t = float(o_times[oi] - t0) / time_scale
                sim.update_bbo(o_bb[oi], o_ba[oi])
                if has_book:
                    sim.book_state.bid_depths = o_bid_depth[oi]
                    sim.book_state.ask_depths = o_ask_depth[oi]
                    sim.book_state.imbalance = float(o_imbalance[oi])
                if hf is not None:
                    try:
                        hf.update(t, classify_event(o_etype[oi], o_side[oi]))
                    except ValueError:
                        pass
                if not measuring and t >= replay_start_seconds:
                    measuring = True
                    inv_before = getattr(agent, "inventory", 0)
                    cash_before = agent.cash
                    trades_before = len(agent.trade_log)
                    snaps_before = len(agent.pnl_snapshots)
                if opening_active:
                    bb_now, ba_now = sim.ob.get_bbo()
                    if bb_now is not None and ba_now is not None:
                        mid_now = (bb_now + ba_now) / 2.0
                        if _open_prev_mid is not None:
                            diff = mid_now - _open_prev_mid
                            if abs(diff) >= _OPEN_MIN_MOVE:
                                if diff > 0:
                                    cur_dir = 1
                                else:
                                    cur_dir = -1
                                if (_open_prev_dir != 0
                                        and cur_dir != _open_prev_dir):
                                    _open_reversals += 1
                                    if _open_reversals >= _OPEN_REVERSAL_THRESH:
                                        opening_active = False
                                _open_prev_dir = cur_dir
                        _open_prev_mid = mid_now
                elif measuring:
                    agent.on_event(sim, t, [])
                oi += 1
            else:
                t = float(m_times[mi] - t0) / time_scale
                sim.update_bbo(m_bb[mi], m_ba[mi])
                if hf is not None:
                    try:
                        hf.update(t, classify_mo(m_side[mi]))
                    except ValueError:
                        pass
                if not measuring and t >= replay_start_seconds:
                    measuring = True
                    inv_before = getattr(agent, "inventory", 0)
                    cash_before = agent.cash
                    trades_before = len(agent.trade_log)
                    snaps_before = len(agent.pnl_snapshots)
                if not opening_active and measuring:
                    fills = sim.process_mo(
                        m_side[mi], m_vol[mi], m_bb[mi], m_ba[mi], m_depth[mi]
                    )
                    agent.on_event(sim, t, fills)
                mi += 1

        if not measuring:
            return {}

        agent.liquidate(sim, t)

        window_trades = agent.trade_log[trades_before:]
        if window_trades:
            inv_series = [r[-2] for r in window_trades]
        else:
            inv_series = [0]
        max_inv = max(abs(v) for v in inv_series)

        window_snaps = agent.pnl_snapshots[snaps_before:]
        if window_snaps:
            pnl_curve = [s[2] for s in window_snaps]
        else:
            pnl_curve = [0]
        peak = np.maximum.accumulate(pnl_curve)
        intraday_dd = float(np.min(np.array(pnl_curve) - peak))

        bb_end, ba_end = sim.ob.get_bbo()
        if bb_end and ba_end:
            mid_end = (bb_end + ba_end) / 2.0
        else:
            mid_end = 0.0
        mid_start = float((o_bb[0] + o_ba[0]) / 2.0)
        equity_before = cash_before + inv_before * mid_start
        equity_after = agent.cash + agent.inventory * mid_end

        return {
            "pnl": equity_after - equity_before,
            "n_trades": len(window_trades),
            "max_inventory": max_inv,
            "intraday_dd": intraday_dd,
        }

    # --- public run methods ---

    def run_single(self, window_id, agent, *, replay_start_s: Optional[float] = None,
                   max_replay_s: Optional[float] = None) -> dict:
        """Run backtest on a single window, return stats dict.

        The *agent* is mutated in place (fills its logs).  The returned
        dict contains ``pnl``, ``n_trades``, ``max_inventory``,
        ``intraday_dd``.

        Parameters
        ----------
        replay_start_s : float, optional
            Warm-up offset before the measurement window (see ``_replay``).
        max_replay_s : float, optional
            Measurement window length in seconds from ``replay_start_s``.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            if self.db_type == "empirical":
                orders_df, mos_df = self._load_empirical_day(conn, window_id)
            else:
                start, end = window_id
                orders_df, mos_df = self._load_sim_window(conn, start, end)
            return self._replay(
                orders_df, mos_df, agent,
                replay_start_s=replay_start_s,
                max_replay_s=max_replay_s,
            )
        finally:
            conn.close()

    def run_all(self, agent_factory: Callable[[], Any], *,
                windows: Optional[list] = None,
                window_size: Optional[int] = None,
                seed: Optional[int] = 42,
                carry_cash: bool = False,
                verbose: bool = True,
                mm_sqlite_path: Optional[Union[str, Path]] = None,
                mm_sqlite_side_is_string: bool = False) -> pd.DataFrame:
        """Run backtest across all windows, return a DataFrame of per-window results.

        Parameters
        ----------
        agent_factory
            Callable that returns a fresh agent instance.
        windows
            Subset of window IDs to run.  Defaults to all.
        window_size
            For simulated DBs: number of events per window.
        seed
            Random seed for reproducibility.  Set to ``None`` to skip.
        carry_cash
            When True, each window's agent starts with the ending cash
            of the previous window (and ``budget``, if present, is set
            to that same value initially).  Constrained MMs then set
            ``budget`` to **spendable** (gross inventory marks on BBO)
            whenever the agent syncs with a BBO.  The agent is still
            freshly created per window (clean logs, zero inventory),
            but its cash reflects cumulative performance.
        verbose
            Print progress every 50 windows.
        """
        if seed is not None:
            random.seed(seed)

        if windows is None:
            windows = self.list_windows(window_size=window_size)

        rows: list[dict] = []
        merged_trades: List[Any] = []
        merged_quotes: List[Any] = []
        merged_pnl: List[Any] = []
        prev_cash: Optional[float] = None
        conn = sqlite3.connect(str(self.db_path))
        try:
            for i, wid in enumerate(windows):
                if self.db_type == "empirical":
                    orders_df, mos_df = self._load_empirical_day(conn, wid)
                else:
                    start, end = wid
                    orders_df, mos_df = self._load_sim_window(conn, start, end)

                agent = agent_factory()
                if carry_cash and prev_cash is not None:
                    agent.cash = prev_cash
                    if hasattr(agent, "budget"):
                        agent.budget = prev_cash
                stats = self._replay(orders_df, mos_df, agent)
                if not stats:
                    continue
                prev_cash = agent.cash
                stats["window"] = wid
                rows.append(stats)

                if mm_sqlite_path is not None:
                    merged_trades.extend(getattr(agent, "trade_log", []) or [])
                    merged_quotes.extend(getattr(agent, "quote_log", []) or [])
                    merged_pnl.extend(getattr(agent, "pnl_snapshots", []) or [])

                if verbose and (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(windows)} windows done ...")
        finally:
            conn.close()

        if mm_sqlite_path is not None and (
                merged_trades or merged_quotes or merged_pnl):
            out_p = Path(mm_sqlite_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.exists():
                out_p.unlink()
            stub = SimpleNamespace(
                agent_id="mm_session",
                trade_log=merged_trades,
                quote_log=merged_quotes,
                pnl_snapshots=merged_pnl,
            )
            _write_mm_sqlite(
                stub, out_p, side_is_string=mm_sqlite_side_is_string)

        df = pd.DataFrame(rows)
        if not df.empty:
            cols = ["window", "pnl", "n_trades", "max_inventory", "intraday_dd"]
            df = df[[c for c in cols if c in df.columns]]
        return df

    def run_sweep(self, agent_class, param_grid: Dict[str, Sequence], *,
                  fixed_params: Optional[Dict[str, Any]] = None,
                  windows: Optional[list] = None,
                  window_size: Optional[int] = None,
                  seed: Optional[int] = 42,
                  verbose: bool = True) -> SweepResult:
        """Grid search over agent parameters, preserving full per-window data.

        Parameters
        ----------
        agent_class
            The MM agent class to instantiate.
        param_grid
            ``{"param_name": [val1, val2, ...], ...}`` – every combination
            is evaluated.
        fixed_params
            Extra keyword arguments passed to *agent_class* on every
            instantiation (e.g. ``{"tick_size": 0.05, "size": 100}``).
        windows
            Subset of window IDs.  Defaults to all.
        window_size
            For simulated DBs: events per window.
        seed
            Random seed, re-applied before each parameter combination.
        verbose
            Print a line per combination.
        """
        if windows is None:
            windows = self.list_windows(window_size=window_size)

        fixed = fixed_params or {}
        names = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        sweep_result = SweepResult(param_names=names)

        for combo in combos:
            kwargs = {**fixed, **dict(zip(names, combo))}

            def make_agent(_kw=kwargs):
                return agent_class(**_kw)

            df = self.run_all(
                make_agent, windows=windows, window_size=window_size,
                seed=seed, verbose=False,
            )
            sweep_result.grid[combo] = df

            if verbose:
                if not df.empty:
                    mean_pnl = df["pnl"].mean()
                else:
                    mean_pnl = 0.0
                label = ", ".join(f"{n}={v}" for n, v in zip(names, combo))
                print(f"  {label}: mean PnL = {mean_pnl:+.2f}")

        return sweep_result

    # --- plotting: single run ---

    @staticmethod
    def plot_single(agent, title: Optional[str] = None):
        """Plot PnL, inventory, cash, mid; for AS agents also spread and quotes.

        When ``quote_log`` rows have length ≥ 9 (Avellaneda–Stoikov logging:
        bid/ask, sizes, mid, sigma, reservation, reason), a fifth panel shows
        quoted spread and the mid panel overlays reservation and bid/ask.
        """
        snaps = agent.pnl_snapshots
        trades = agent.trade_log
        if not snaps:
            print("No PnL snapshots to plot.")
            return

        times = np.array([s[0] for s in snaps])
        mids = np.array([s[1] for s in snaps])
        pnls = np.array([s[2] for s in snaps])

        if trades:
            trade_t = np.array([r[0] for r in trades])
            trade_inv = np.array([r[-2] for r in trades])
            trade_cash = np.array([r[-1] for r in trades])
            trade_side_raw = [r[1] for r in trades]
        else:
            trade_t = np.array([])
            trade_inv = np.array([])
            trade_cash = np.array([])
            trade_side_raw = []
        if trade_side_raw and isinstance(trade_side_raw[0], str):
            buy_mask = np.array([s == "BUY" for s in trade_side_raw])
        elif trade_side_raw:
            buy_mask = np.array([s > 0 for s in trade_side_raw])
        else:
            buy_mask = np.array([], dtype=bool)
        if len(buy_mask):
            sell_mask = ~buy_mask
        else:
            sell_mask = np.array([], dtype=bool)

        quote_log = getattr(agent, "quote_log", None) or []
        has_as_quotes = bool(
            quote_log and len(quote_log[0]) >= 9
        )
        t_quote = np.array([], dtype=float)
        bid_q = np.array([], dtype=float)
        ask_q = np.array([], dtype=float)
        res_q = np.array([], dtype=float)
        if has_as_quotes:
            t_quote = np.array([float(r[0]) for r in quote_log], dtype=float)
            bid_q = np.array([float(r[1]) for r in quote_log], dtype=float)
            ask_q = np.array([float(r[2]) for r in quote_log], dtype=float)
            res_q = np.array([float(r[7]) for r in quote_log], dtype=float)

        ref_parts = [times]
        if has_as_quotes and len(t_quote):
            ref_parts.append(t_quote)
        if len(trade_t):
            ref_parts.append(trade_t)
        if ref_parts:
            ref = np.concatenate(ref_parts)
        else:
            ref = times
        if len(ref) < 2:
            ref = times

        times_s, xlabel = scale_plot_times(times, ref_for_span=ref)
        trade_t_s, _ = scale_plot_times(trade_t, ref_for_span=ref)
        t_quote_s, _ = scale_plot_times(t_quote, ref_for_span=ref)

        if has_as_quotes:
            nrows = 5
            fig_h = 16.0
        else:
            nrows = 4
            fig_h = 13.0
        fig, axes = plt.subplots(nrows, 1, figsize=(14, fig_h), sharex=True)
        axes = np.atleast_1d(axes)

        iax = 0
        axes[iax].plot(times_s, pnls, lw=1, color="tab:green",
                       label="Mark-to-market PnL")
        axes[iax].axhline(0, ls="--", color="grey", alpha=0.5)
        axes[iax].set_ylabel("PnL")
        axes[iax].set_title(title or "Market Maker Backtest")
        axes[iax].legend(loc="upper left")
        iax += 1

        if len(trade_t_s):
            axes[iax].step(trade_t_s, trade_inv, where="post", lw=1,
                           color="tab:blue", label="Inventory")
            if buy_mask.any():
                axes[iax].scatter(trade_t_s[buy_mask], trade_inv[buy_mask],
                                 marker="^", color="green", s=15, alpha=0.7,
                                 label="Buy fill", zorder=3)
            if sell_mask.any():
                axes[iax].scatter(trade_t_s[sell_mask], trade_inv[sell_mask],
                                 marker="v", color="red", s=15, alpha=0.7,
                                 label="Sell fill", zorder=3)
        axes[iax].axhline(0, ls="--", color="grey", alpha=0.5)
        axes[iax].set_ylabel("Inventory")
        axes[iax].legend(loc="upper left")
        iax += 1

        if len(trade_t_s):
            axes[iax].step(trade_t_s, trade_cash, where="post", lw=1,
                           color="tab:purple", label="Cash")
        axes[iax].axhline(0, ls="--", color="grey", alpha=0.5)
        axes[iax].set_ylabel("Cash")
        axes[iax].legend(loc="upper left")
        iax += 1

        if has_as_quotes:
            spread = ask_q - bid_q
            ok_sp = np.isfinite(bid_q) & np.isfinite(ask_q)
            if ok_sp.any():
                axes[iax].plot(
                    t_quote_s[ok_sp], spread[ok_sp], lw=0.8, color="tab:brown",
                    label="Quoted spread (ask − bid)",
                )
                axes[iax].legend(loc="upper left")
            axes[iax].axhline(0, ls="--", color="grey", alpha=0.4)
            axes[iax].set_ylabel("Spread")
            iax += 1

        ax_mid = axes[iax]
        color_res = "tab:cyan"
        color_bid = "tab:blue"
        color_ask = "tab:red"
        fill_alpha = 0.09

        if has_as_quotes and len(t_quote_s):
            m_fin = np.isfinite(res_q) & np.isfinite(bid_q)
            if m_fin.any():
                ax_mid.fill_between(
                    t_quote_s[m_fin], res_q[m_fin], bid_q[m_fin],
                    color=color_bid, alpha=fill_alpha, linewidth=0,
                    interpolate=True,
                )
            m_fin = np.isfinite(res_q) & np.isfinite(ask_q)
            if m_fin.any():
                ax_mid.fill_between(
                    t_quote_s[m_fin], res_q[m_fin], ask_q[m_fin],
                    color=color_ask, alpha=fill_alpha, linewidth=0,
                    interpolate=True,
                )
            mb = np.isfinite(bid_q)
            if mb.any():
                ax_mid.plot(
                    t_quote_s[mb], bid_q[mb], lw=0.9, color=color_bid,
                    alpha=0.55, label="Quoted bid",
                )
            ma = np.isfinite(ask_q)
            if ma.any():
                ax_mid.plot(
                    t_quote_s[ma], ask_q[ma], lw=0.9, color=color_ask,
                    alpha=0.55, label="Quoted ask",
                )
            mr = np.isfinite(res_q)
            if mr.any():
                ax_mid.plot(
                    t_quote_s[mr], res_q[mr], lw=1.0, color=color_res,
                    label="Reservation price",
                )

        ax_mid.plot(times_s, mids, lw=0.6, color="tab:orange",
                    label="Mid price", zorder=4)
        ax_mid.set_ylabel("Mid / quotes")
        ax_mid.set_xlabel(xlabel)
        ax_mid.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        plt.show()

    # --- plotting: multi-window summary ---

    @staticmethod
    def plot_summary(results_df: pd.DataFrame, title: Optional[str] = None):
        """4-panel overview: cumulative PnL, PnL distribution,
        trades per window, and maximum inventory.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        cum = results_df["pnl"].cumsum()
        axes[0, 0].plot(cum.values, lw=1, color="tab:green")
        axes[0, 0].axhline(0, ls="--", color="grey", alpha=0.4)
        axes[0, 0].set_title("Cumulative PnL")
        axes[0, 0].set_xlabel("Window index")

        axes[0, 1].hist(results_df["pnl"], bins=30,
                        color="tab:blue", edgecolor="white", alpha=0.8)
        axes[0, 1].axvline(0, ls="--", color="grey")
        axes[0, 1].set_title("PnL distribution")
        axes[0, 1].set_xlabel("PnL")

        axes[1, 0].bar(range(len(results_df)), results_df["n_trades"],
                       color="tab:orange", alpha=0.7)
        axes[1, 0].set_title("Trades per window")
        axes[1, 0].set_xlabel("Window index")

        axes[1, 1].plot(results_df["max_inventory"].values, lw=0.8,
                        color="tab:red")
        axes[1, 1].set_title("Max absolute inventory per window")
        axes[1, 1].set_xlabel("Window index")

        if title:
            fig.suptitle(title, fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

    # --- plotting: sweep heatmap ---

    @staticmethod
    def plot_sweep_heatmap(sweep: SweepResult,
                           param_x: str, param_y: str,
                           metric: str = "pnl",
                           agg: str = "mean"):
        """Heatmap of an aggregate metric across two swept parameters."""
        names = sweep.param_names
        ix = names.index(param_x)
        iy = names.index(param_y)

        x_vals = sorted({k[ix] for k in sweep.grid})
        y_vals = sorted({k[iy] for k in sweep.grid})

        grid = np.full((len(y_vals), len(x_vals)), np.nan)
        for combo, df in sweep.grid.items():
            if df.empty:
                continue
            xi = x_vals.index(combo[ix])
            yi = y_vals.index(combo[iy])
            if agg == "mean":
                grid[yi, xi] = df[metric].mean()
            elif agg == "median":
                grid[yi, xi] = df[metric].median()
            elif agg == "std":
                grid[yi, xi] = df[metric].std()

        fig, ax = plt.subplots(figsize=(9, 6))
        vmax = np.nanmax(np.abs(grid)) or 1
        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn",
                       origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals])
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_title(f"{agg.title()} {metric} by {param_y} and {param_x}")

        for yi_idx in range(len(y_vals)):
            for xi_idx in range(len(x_vals)):
                val = grid[yi_idx, xi_idx]
                if np.isfinite(val):
                    if abs(val) < vmax * 0.6:
                        text_color = "black"
                    else:
                        text_color = "white"
                    ax.text(xi_idx, yi_idx, f"{val:+.1f}",
                            ha="center", va="center", fontsize=9,
                            color=text_color)

        fig.colorbar(im, ax=ax, label=f"{agg.title()} {metric}")
        plt.tight_layout()
        plt.show()

    # --- plotting: sweep distributions ---

    @staticmethod
    def plot_sweep_distributions(sweep: SweepResult,
                                 param_x: str, param_y: str,
                                 metric: str = "pnl",
                                 bins: int = 20):
        """Grid of histograms: one per cell in the sweep, preserving
        the full distribution instead of collapsing to a single mean.
        """
        names = sweep.param_names
        ix = names.index(param_x)
        iy = names.index(param_y)

        x_vals = sorted({k[ix] for k in sweep.grid})
        y_vals = sorted({k[iy] for k in sweep.grid}, reverse=True)

        n_rows = len(y_vals)
        n_cols = len(x_vals)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.2 * n_cols, 2.5 * n_rows),
                                 sharex=True, sharey=True)
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for ri, yv in enumerate(y_vals):
            for ci, xv in enumerate(x_vals):
                ax = axes[ri, ci]
                matching = [
                    df for combo, df in sweep.grid.items()
                    if combo[ix] == xv and combo[iy] == yv and not df.empty
                ]
                if matching:
                    window_metric = matching[0][metric]
                    ax.hist(window_metric, bins=bins, color="tab:blue",
                            edgecolor="white", alpha=0.8)
                    ax.axvline(window_metric.mean(), color="tab:red", ls="--", lw=1)
                    ax.axvline(0, color="grey", ls="--", alpha=0.5)
                ax.set_title(f"{param_y}={yv}, {param_x}={xv}", fontsize=8)
                if ri == n_rows - 1:
                    ax.set_xlabel(metric, fontsize=8)
                if ci == 0:
                    ax.set_ylabel("count", fontsize=8)
                ax.tick_params(labelsize=7)

        fig.suptitle(f"Distribution of {metric} per window", fontsize=12)
        plt.tight_layout()
        plt.show()


# --- plot time scaling ---

def scale_plot_times(raw, ref_for_span: np.ndarray):
    """Scale timestamps to seconds / minutes / hours for plotting.

    *ref_for_span* chooses the unit so quote times and PnL times share
    the same axis scaling.
    """
    if len(raw) == 0:
        return np.array(raw, dtype=float), "time"
    arr = np.asarray(raw, dtype=float)
    ref = np.asarray(ref_for_span, dtype=float)
    if len(ref) < 2:
        span = 0.0
    else:
        span = float(ref[-1] - ref[0])
    if span > 7200:
        return arr / 3600.0, "time (hours)"
    if span > 120:
        return arr / 60.0, "time (minutes)"
    return arr, "time (seconds)"

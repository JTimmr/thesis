"""Replay-based market-maker backtester for empirical and simulated SQLite databases.

Feeds recorded BBO updates and market-order events to any agent that
implements the standard ``on_event(sim, t, fills)`` / ``liquidate(sim, t)``
protocol, then collects per-window statistics (PnL, trades, inventory,
drawdown) for distributional analysis and parameter sweeps.
"""

from __future__ import annotations

import itertools
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .helpers import resolve_data_path

# ═══════════════════════════════════════════════════════════════════════════════
# ReplaySimulator – thin adapter between recorded data and agent interface
# ═══════════════════════════════════════════════════════════════════════════════

class ReplaySimulator:
    """Feeds recorded BBO to an MM agent and detects paper fills using
    the depth snapshot from the empirical / simulated data.

    Exposes the same ``agent_place_order``, ``agent_cancel_order``,
    ``agent_market_order`` interface that ``Simulate`` provides so any
    agent written for the full simulator works here unchanged.
    """

    class _OB:
        """Minimal order-book facade expected by MM agents."""

        def __init__(self):
            self.order_map: Dict[int, list] = {}
            self._bb: Optional[float] = None
            self._ba: Optional[float] = None

        def get_bbo(self) -> Tuple[Optional[float], Optional[float]]:
            return self._bb, self._ba

    def __init__(self, tick_size: float = 1.0):
        self.ob = self._OB()
        self.tick_size = tick_size
        self._next_oid = 1_000_000
        self.mm_bid: Optional[Tuple[int, float, int]] = None
        self.mm_ask: Optional[Tuple[int, float, int]] = None

    def update_bbo(self, bb, ba):
        try:
            bb, ba = float(bb), float(ba)
            if np.isfinite(bb) and np.isfinite(ba):
                self.ob._bb = bb
                self.ob._ba = ba
        except (TypeError, ValueError):
            pass

    # ── agent interface (paper orders) ────────────────────────────────

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

    # ── fill detection ────────────────────────────────────────────────

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
            fill_qty = int(min(remaining, vol)) if remaining > 0 else 0
        else:
            return None

        if fill_qty <= 0:
            return None

        resting_side = 2 if level_sign == 1 else 1
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

    def process_mo(self, mo_side: str, mo_volume, best_bid, best_ask,
                   opp_depth) -> list:
        """Check whether an empirical MO would fill the MM's resting order."""
        if mo_side == "buy":
            result = self._check_side(
                self.mm_ask, +1, best_ask, mo_volume, opp_depth
            )
        elif mo_side == "sell":
            result = self._check_side(
                self.mm_bid, -1, best_bid, mo_volume, opp_depth
            )
        else:
            result = None
        return [result] if result is not None else []


# ═══════════════════════════════════════════════════════════════════════════════
# SweepResult – container for parameter-sweep output with full distributions
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# MMBacktester – main class
# ═══════════════════════════════════════════════════════════════════════════════

_OPP_DEPTH_COLS = [f"opp_depth_L{i}" for i in range(10)]


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
        Price tick size.  Defaults to 0.05 for empirical DBs and
        1.0 for simulated DBs.
    db_type : str
        ``"auto"`` (default), ``"empirical"``, or ``"simulated"``.
    """

    def __init__(self, db_path: Union[str, Path], *,
                 tick_size: Optional[float] = None,
                 db_type: str = "auto"):
        self.db_path = resolve_data_path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.db_type = self._detect_db_type() if db_type == "auto" else db_type
        self.tick_size = tick_size if tick_size is not None else (
            0.05 if self.db_type == "empirical" else 1.0
        )

    # ── DB introspection ──────────────────────────────────────────────

    def _detect_db_type(self) -> str:
        conn = sqlite3.connect(str(self.db_path))
        try:
            cols = [
                row[1]
                for row in conn.execute("PRAGMA table_info(orders)").fetchall()
            ]
            return "empirical" if "day" in cols else "simulated"
        finally:
            conn.close()

    # ── window listing ────────────────────────────────────────────────

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

    # ── data loading ──────────────────────────────────────────────────

    def _load_empirical_day(self, conn: sqlite3.Connection,
                            day_id: str):
        """Load orders + mo_orders DataFrames for one empirical day."""
        orders_df = pd.read_sql_query(
            "SELECT timestamp, event_type, order_id, side, order_price, "
            "volume, best_bid, best_ask "
            "FROM orders WHERE day = ? ORDER BY timestamp",
            conn,
            params=(day_id,),
        )
        if not orders_df.empty:
            orders_df["time_ns"] = (
                pd.to_datetime(orders_df["timestamp"]).astype("int64")
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
        orders_df = pd.read_sql_query(
            "SELECT timestamp AS time_ns, event_type, order_id, side, "
            "order_price, volume, best_bid, best_ask "
            "FROM orders ORDER BY timestamp "
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

    # ── replay engine ─────────────────────────────────────────────────

    def _replay(self, orders_df: pd.DataFrame, mos_df: pd.DataFrame,
                agent) -> dict:
        """Run the event-by-event replay loop and return per-window stats.

        Stats are computed as per-window deltas so that the results are
        correct both when a fresh agent is created per window (factory
        mode) and when the same agent is reused across windows
        (persistent mode).
        """
        if orders_df.empty:
            return {}

        inv_before = getattr(agent, "inventory", 0)
        cash_before = agent.cash
        trades_before = len(getattr(agent, "trade_log", []))
        snaps_before = len(getattr(agent, "pnl_snapshots", []))

        sim = ReplaySimulator(tick_size=self.tick_size)

        o_times = orders_df["time_ns"].values
        o_bb = orders_df["best_bid"].values
        o_ba = orders_df["best_ask"].values
        n_o = len(o_times)

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

        t = 0.0
        while oi < n_o or mi < n_m:
            ot = o_times[oi] if oi < n_o else INT_MAX
            mt = m_times[mi] if mi < n_m else INT_MAX

            if ot <= mt:
                t = float(o_times[oi] - t0) / time_scale
                sim.update_bbo(o_bb[oi], o_ba[oi])
                agent.on_event(sim, t, [])
                oi += 1
            else:
                t = float(m_times[mi] - t0) / time_scale
                sim.update_bbo(m_bb[mi], m_ba[mi])
                fills = sim.process_mo(
                    m_side[mi], m_vol[mi], m_bb[mi], m_ba[mi], m_depth[mi]
                )
                agent.on_event(sim, t, fills)
                mi += 1

        agent.liquidate(sim, t)

        window_trades = (agent.trade_log[trades_before:]
                         if hasattr(agent, "trade_log") else [])
        inv_series = [r[-2] for r in window_trades] if window_trades else [0]
        max_inv = max(abs(v) for v in inv_series)

        window_snaps = (agent.pnl_snapshots[snaps_before:]
                        if hasattr(agent, "pnl_snapshots") else [])
        pnl_curve = [s[2] for s in window_snaps] if window_snaps else [0]
        peak = np.maximum.accumulate(pnl_curve)
        intraday_dd = float(np.min(np.array(pnl_curve) - peak))

        bb_end, ba_end = sim.ob.get_bbo()
        mid_end = (bb_end + ba_end) / 2.0 if bb_end and ba_end else 0.0
        mid_start = float((o_bb[0] + o_ba[0]) / 2.0)
        equity_before = cash_before + inv_before * mid_start
        equity_after = agent.cash + agent.inventory * mid_end

        return {
            "pnl": equity_after - equity_before,
            "n_trades": len(window_trades),
            "max_inventory": max_inv,
            "intraday_dd": intraday_dd,
        }

    # ── public run methods ────────────────────────────────────────────

    def run_single(self, window_id, agent) -> dict:
        """Run backtest on a single window, return stats dict.

        The *agent* is mutated in place (fills its logs).  The returned
        dict contains ``pnl``, ``n_trades``, ``max_inventory``,
        ``intraday_dd``.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            if self.db_type == "empirical":
                orders_df, mos_df = self._load_empirical_day(conn, window_id)
            else:
                start, end = window_id
                orders_df, mos_df = self._load_sim_window(conn, start, end)
            return self._replay(orders_df, mos_df, agent)
        finally:
            conn.close()

    def run_all(self, agent_factory: Callable[[], Any], *,
                windows: Optional[list] = None,
                window_size: Optional[int] = None,
                seed: Optional[int] = 42,
                carry_cash: bool = False,
                verbose: bool = True) -> pd.DataFrame:
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
            of the previous window (and budget, if constrained).  The
            agent is still freshly created per window (clean logs,
            zero inventory), but its capital reflects cumulative
            performance.
        verbose
            Print progress every 50 windows.
        """
        if seed is not None:
            random.seed(seed)

        if windows is None:
            windows = self.list_windows(window_size=window_size)

        rows: list[dict] = []
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

                if verbose and (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(windows)} windows done ...")
        finally:
            conn.close()

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

        result = SweepResult(param_names=names)

        for combo in combos:
            kwargs = {**fixed, **dict(zip(names, combo))}

            def _factory(_kw=kwargs):
                return agent_class(**_kw)

            df = self.run_all(
                _factory, windows=windows, window_size=window_size,
                seed=seed, verbose=False,
            )
            result.grid[combo] = df

            if verbose:
                mean_pnl = df["pnl"].mean() if not df.empty else 0.0
                label = ", ".join(f"{n}={v}" for n, v in zip(names, combo))
                print(f"  {label}: mean PnL = {mean_pnl:+.2f}")

        return result

    # ── plotting: single run ──────────────────────────────────────────

    @staticmethod
    def plot_single(agent, title: Optional[str] = None):
        """4-panel plot for a single-run agent (PnL, inventory, cash, mid)."""
        snaps = getattr(agent, "pnl_snapshots", [])
        trades = getattr(agent, "trade_log", [])
        if not snaps:
            print("No PnL snapshots to plot.")
            return

        times = np.array([s[0] for s in snaps])
        mids = np.array([s[1] for s in snaps])
        pnls = np.array([s[2] for s in snaps])

        trade_t = np.array([r[0] for r in trades]) if trades else np.array([])
        trade_inv = np.array([r[-2] for r in trades]) if trades else np.array([])
        trade_cash = np.array([r[-1] for r in trades]) if trades else np.array([])
        trade_side_raw = [r[1] for r in trades] if trades else []
        if trade_side_raw and isinstance(trade_side_raw[0], str):
            buy_mask = np.array([s == "BUY" for s in trade_side_raw])
        else:
            buy_mask = np.array([s > 0 for s in trade_side_raw]) if trade_side_raw else np.array([], dtype=bool)
        sell_mask = ~buy_mask if len(buy_mask) else np.array([], dtype=bool)

        times_s, xlabel = _scale_times(times)
        trade_t_s, _ = _scale_times(trade_t)

        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

        axes[0].plot(times_s, pnls, lw=1, color="tab:green",
                     label="Mark-to-market PnL")
        axes[0].axhline(0, ls="--", color="grey", alpha=0.5)
        axes[0].set_ylabel("PnL")
        axes[0].set_title(title or "Market Maker Backtest")
        axes[0].legend(loc="upper left")

        if len(trade_t_s):
            axes[1].step(trade_t_s, trade_inv, where="post", lw=1,
                         color="tab:blue", label="Inventory")
            if buy_mask.any():
                axes[1].scatter(trade_t_s[buy_mask], trade_inv[buy_mask],
                                marker="^", color="green", s=15, alpha=0.7,
                                label="Buy fill", zorder=3)
            if sell_mask.any():
                axes[1].scatter(trade_t_s[sell_mask], trade_inv[sell_mask],
                                marker="v", color="red", s=15, alpha=0.7,
                                label="Sell fill", zorder=3)
        axes[1].axhline(0, ls="--", color="grey", alpha=0.5)
        axes[1].set_ylabel("Inventory")
        axes[1].legend(loc="upper left")

        if len(trade_t_s):
            axes[2].step(trade_t_s, trade_cash, where="post", lw=1,
                         color="tab:purple", label="Cash")
        axes[2].axhline(0, ls="--", color="grey", alpha=0.5)
        axes[2].set_ylabel("Cash")
        axes[2].legend(loc="upper left")

        axes[3].plot(times_s, mids, lw=0.5, color="tab:orange",
                     label="Mid price")
        axes[3].set_ylabel("Mid price")
        axes[3].set_xlabel(xlabel)
        axes[3].legend(loc="upper left")

        plt.tight_layout()
        plt.show()

    # ── plotting: multi-window summary ────────────────────────────────

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

    # ── plotting: sweep heatmap ───────────────────────────────────────

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
                    ax.text(xi_idx, yi_idx, f"{val:+.1f}",
                            ha="center", va="center", fontsize=9,
                            color="black" if abs(val) < vmax * 0.6 else "white")

        fig.colorbar(im, ax=ax, label=f"{agg.title()} {metric}")
        plt.tight_layout()
        plt.show()

    # ── plotting: sweep distributions ─────────────────────────────────

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
                    data = matching[0][metric]
                    ax.hist(data, bins=bins, color="tab:blue",
                            edgecolor="white", alpha=0.8)
                    ax.axvline(data.mean(), color="tab:red", ls="--", lw=1)
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


# ── module-level helper ───────────────────────────────────────────────────────

def _scale_times(raw):
    if len(raw) == 0:
        return np.array(raw, dtype=float), "time"
    arr = np.asarray(raw, dtype=float)
    span = arr[-1] - arr[0]
    if span > 7200:
        return arr / 3600.0, "time (hours)"
    if span > 120:
        return arr / 60.0, "time (minutes)"
    return arr, "time (seconds)"

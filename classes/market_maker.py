"""
Example market-making agents for use with ``Simulate(agents=[...])``.
"""
from __future__ import annotations

import math
import sqlite3
import time
from typing import Optional

import numpy as np

def _get_ergodic_solver():
    from . import _ergodic_solver
    return _ergodic_solver


def _float_or_nan(value):
    if value is None:
        return float('nan')
    return float(value)


# --- MM SQLite schema ---

mm_create_trades_sql = """
CREATE TABLE IF NOT EXISTS mm_trades (
    agent_id     TEXT,
    timestamp    REAL,
    side         INTEGER,
    price        REAL,
    qty          INTEGER,
    best_bid     REAL,
    best_ask     REAL,
    inventory    INTEGER,
    cash         REAL
)"""

mm_create_quotes_sql = """
CREATE TABLE IF NOT EXISTS mm_quotes (
    agent_id     TEXT,
    timestamp    REAL,
    bid_price    REAL,
    ask_price    REAL,
    bid_size     INTEGER,
    ask_size     INTEGER,
    reason       TEXT,
    mid          REAL,
    sigma        REAL,
    reservation  REAL
)"""

mm_create_pnl_sql = """
CREATE TABLE IF NOT EXISTS mm_pnl (
    agent_id     TEXT,
    timestamp    REAL,
    mid          REAL,
    mtm          REAL
)"""

mm_insert_trade_sql = "INSERT INTO mm_trades VALUES (?,?,?,?,?,?,?,?,?)"
mm_insert_quote_sql = "INSERT INTO mm_quotes VALUES (?,?,?,?,?,?,?,?,?,?)"
mm_insert_pnl_sql = "INSERT INTO mm_pnl VALUES (?,?,?,?)"


def mm_spendable(cash: float, inventory: int, bb, ba) -> float:
    """Capacity for new quotes after gross inventory marks on BBO.

    ``cash − max(0,inv)×bb − max(0,−inv)×ba``, clamped at 0.  Used for
    constrained sizing, replay buy-clip, and ``agent.budget`` sync.
    """
    if bb is None or ba is None:
        return 0.0
    bb_f, ba_f = float(bb), float(ba)
    if bb_f <= 0 or ba_f <= 0:
        return 0.0
    if not math.isfinite(bb_f) or not math.isfinite(ba_f):
        return 0.0
    if not math.isfinite(float(cash)):
        return 0.0
    inv = int(inventory)
    long_r = max(0, inv) * bb_f
    short_r = max(0, -inv) * ba_f
    return max(0.0, float(cash) - long_r - short_r)


def symmetric_margin_quote_sizes(
    cash: float,
    inventory: int,
    bb,
    ba,
    bid_price: float,
    ask_price: float,
    size: int,
) -> tuple[int, int]:
    """Constrained resting sizes: half spendable USD per side, int shares.

    Long inventory: flatten-first ask (``min(size, inv)``) plus
    ``short_add`` capped by the ask-half envelope.  Short inventory:
    cover-first bid plus ``long_add`` from the bid-half.  Flat: both
    halves apply to bid and incremental-short ask only.
    """
    inv = int(inventory)
    sz = int(size)
    if bid_price <= 0 or ask_price <= 0 or sz <= 0:
        return 0, 0
    sp = mm_spendable(cash, inv, bb, ba)
    half = sp / 2.0
    if half > 0:
        bid_from_half = int(half // float(bid_price))
        short_add_from_half = int(half // float(ask_price))
    else:
        bid_from_half = 0
        short_add_from_half = 0

    if inv > 0:
        flatten = min(sz, inv)
        short_add = min(short_add_from_half, max(0, sz - flatten))
        ask_size = flatten + short_add
        bid_size = min(sz, bid_from_half)
    elif inv < 0:
        cover_flat = min(sz, -inv)
        long_add = min(bid_from_half, max(0, sz - cover_flat))
        bid_size = cover_flat + long_add
        ask_size = min(sz, short_add_from_half)
    else:
        bid_size = min(sz, bid_from_half)
        ask_size = min(sz, short_add_from_half)

    return max(0, bid_size), max(0, ask_size)


def sync_cash_budget(agent, bb=None, ba=None) -> None:
    """Set ``budget`` to spendable (BBO) or ``cash`` when BBO missing."""
    if not getattr(agent, "_constrained", False):
        return
    if bb is not None and ba is not None:
        agent.budget = mm_spendable(
            float(agent.cash), int(agent.inventory), bb, ba)
    else:
        agent.budget = float(agent.cash)


def _write_mm_sqlite(agent, db_path, side_is_string=False):
    """Write MM trade / quote / PnL logs to SQLite."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute(mm_create_trades_sql)
    cur.execute(mm_create_quotes_sql)
    cur.execute(mm_create_pnl_sql)

    aid = agent.agent_id

    if agent.trade_log:
        rows = []
        for row in agent.trade_log:
            t, side, px, qty, bb, ba, inv, cash = row
            if side_is_string:
                if side == "BUY":
                    side = 1
                else:
                    side = -1
            rows.append((aid, float(t), int(side), float(px), int(qty),
                         float(bb), float(ba), int(inv), float(cash)))
        cur.executemany(mm_insert_trade_sql, rows)

    if agent.quote_log:
        rows = []
        for row in agent.quote_log:
            if len(row) == 6:
                t, bp, ap, bs, as_, reason = row
                mid = sigma = res = None
            else:
                t, bp, ap, bs, as_, mid, sigma, res, reason = row
            rows.append((aid, float(t), float(bp), float(ap),
                         int(bs), int(as_), reason,
                         float(mid) if mid is not None else None,
                         float(sigma) if sigma is not None else None,
                         float(res) if res is not None else None))
        cur.executemany(mm_insert_quote_sql, rows)

    if agent.pnl_snapshots:
        rows = [(aid, float(t), float(m), float(mtm))
                for t, m, mtm in agent.pnl_snapshots]
        cur.executemany(mm_insert_pnl_sql, rows)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_mm_trades_agent "
                "ON mm_trades(agent_id, timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mm_quotes_agent "
                "ON mm_quotes(agent_id, timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mm_pnl_agent "
                "ON mm_pnl(agent_id, timestamp)")

    conn.commit()
    conn.close()


# --- Ergodic Guéant market maker (infinite-horizon / long-run optimal) ---

class TimeGridRealizedVol:
    """Online trailing time-grid realised fractional volatility.

    Mirrors :func:`research_core.classes.phantom_labels.realized_vol_time_grid`:
    log-mid returns on a uniform grid, trailing RMS over ``rv_window_s``.
    """

    var_floor = 1e-8

    def __init__(self, rv_window_s=200.0, rv_sample_dt_s=1.0, vol_floor=1e-4):
        self._vol_floor = float(vol_floor)
        self._rv_dt = max(1e-9, float(rv_sample_dt_s))
        self._rv_nwin = max(1, int(round(float(rv_window_s) / self._rv_dt)))
        self._rv_buf = np.zeros(self._rv_nwin, dtype=np.float64)
        self._rv_pos = 0
        self._rv_count = 0
        self._rv_sum = 0.0
        self._rv_t0 = None
        self._rv_last_k = -1
        self._rv_prev_grid_mid = None
        self._rv_cur_mid = None

    def update(self, mid, t_sec):
        if not (mid > 0):
            return
        if self._rv_t0 is None:
            self._rv_t0 = t_sec
            self._rv_last_k = 0
            self._rv_prev_grid_mid = mid
            self._rv_cur_mid = mid
            return

        k_new = int(math.floor((t_sec - self._rv_t0) / self._rv_dt))
        if k_new <= self._rv_last_k:
            self._rv_cur_mid = mid
            return

        n_steps = k_new - self._rv_last_k
        if n_steps > self._rv_nwin:
            self._rv_buf[:] = 0.0
            self._rv_pos = 0
            self._rv_count = 0
            self._rv_sum = 0.0
            self._rv_last_k = k_new
            self._rv_prev_grid_mid = mid
            self._rv_cur_mid = mid
            return

        prev_grid_mid = self._rv_prev_grid_mid
        sampled = self._rv_cur_mid
        for _ in range(self._rv_last_k + 1, k_new + 1):
            if (prev_grid_mid is not None and prev_grid_mid > 0
                    and sampled is not None and sampled > 0):
                r = math.log(sampled / prev_grid_mid)
                r2 = r * r
            else:
                r2 = 0.0
            old = self._rv_buf[self._rv_pos]
            self._rv_buf[self._rv_pos] = r2
            self._rv_pos = (self._rv_pos + 1) % self._rv_nwin
            if self._rv_count < self._rv_nwin:
                self._rv_count += 1
                self._rv_sum += r2
            else:
                self._rv_sum += r2 - old
            prev_grid_mid = sampled
        self._rv_prev_grid_mid = prev_grid_mid
        self._rv_last_k = k_new
        self._rv_cur_mid = mid

    @property
    def sigma(self):
        if self._rv_count > 0:
            var = self._rv_sum / self._rv_count
        else:
            var = 0.0
        return max(math.sqrt(max(var, self.var_floor)), self._vol_floor)


class ErgodicMM:
    """Market maker using the long-run (ergodic) Guéant optimal quoting formulas.

    Unlike :class:`AvellanedaStoikovMM` which uses a finite-horizon rolling
    lookahead, this agent quotes at the *stationary* optimal deltas derived
    from the ergodic HJB equation.  There is no ``horizon`` parameter;
    instead, gamma controls inventory penalty strength directly through
    the inventory-penalty function  S(q) = -gamma * sigma^2 * q^2 / 2.

    Optimal bid/ask deltas (distance from mid) for inventory q:

        b0   = (1/gamma) * ln(1 + gamma/k)
        c    = (sigma^2 * gamma) / (2 * k * A) * (1 + gamma/k)^(1 + k/gamma)
        s    = sqrt(c)

        delta_b(q) = b0 + (2*q + 1) * s / 2
        delta_a(q) = b0 - (2*q - 1) * s / 2

    where A is the market-order arrival intensity (set to 1.0 by convention)
    and k is the exponential fill-decay parameter.

    Volatility can be estimated in two modes (``vol_mode``):

    * ``"ewma_event"`` (default) — EWMA of squared fractional mid returns
      on each BBO update (same as :class:`AvellanedaStoikovMM`).
    * ``"realized_time"`` — trailing realised vol on a uniform time grid
      (200 s window, 1 s steps), matching phantom-label ``recent_vol``.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient (> 0).  Enters both the inventory
        penalty  S(q) = -gamma * sigma^2 * q^2 / 2  and the base spread
        component  (1/gamma) * ln(1 + gamma/k).  Units: 1/PLN^2.
    k : float
        Exponential decay of fill probability with quoting depth.
        Calibrated from market-order penetration data.
    vol_halflife : int
        EWMA half-life (in number of mid-price updates) for the
        squared-return volatility estimator.
    vol_floor : float
        Minimum fractional volatility to prevent numerical instability.
    vol_mode : str
        ``"ewma_event"`` or ``"realized_time"``.
    rv_window_s, rv_sample_dt_s : float
        Trailing window and grid spacing for ``realized_time`` mode.
    size : int
        Order size placed on each side.
    tick_size : float
        Minimum price increment for rounding quotes.
    intensity_A : float
        Market-order arrival intensity A.  Set to 1.0 by convention
        (absorbed into the calibrated k).
    initial_cash : float or None
        Starting cash ledger.  ``None`` means unlimited capital.
    initial_inventory : int
        Starting inventory (number of shares already held).
    """

    def __init__(self, gamma=1.0, k=1.5,
                 vol_halflife=50, vol_floor=1e-4, size=1, tick_size=1,
                 intensity_A=1.0, verbose=True,
                 initial_cash=None, initial_inventory=0,
                 agent_id=None,
                 vol_mode="ewma_event",
                 rv_window_s=200.0, rv_sample_dt_s=1.0):
        self.agent_id = agent_id
        self.gamma = float(gamma)
        self.k = float(k)
        self.intensity_A = float(intensity_A)
        self.size = int(size)
        self.tick_size = float(tick_size)
        self.verbose = verbose
        self._constrained = initial_cash is not None

        if vol_mode not in ("ewma_event", "realized_time"):
            raise ValueError(
                f"vol_mode must be 'ewma_event' or 'realized_time', got {vol_mode!r}"
            )
        self.vol_mode = vol_mode

        # EWMA volatility estimator (fractional returns)
        self._alpha = 1.0 - 0.5 ** (1.0 / vol_halflife)
        self._vol_floor = float(vol_floor)
        self._ewma_var = self._vol_floor ** 2
        self._prev_mid = None
        self._prev_t = None
        if vol_mode == "realized_time":
            self._rv = TimeGridRealizedVol(rv_window_s, rv_sample_dt_s, vol_floor)
        else:
            self._rv = None

        # Order / position state
        self.inventory = int(initial_inventory)
        if initial_cash is not None:
            self.cash = float(initial_cash)
            self.budget = float(initial_cash)
        else:
            self.cash = 0.0
            self.budget = 0.0
        self.bid_oid = None
        self.ask_oid = None
        self._bid_price = None
        self._ask_price = None
        self._prev_bb = None
        self._prev_ba = None

        self.trade_log = []      # (t, side, price, qty, bb, ba, inventory, cash)
        self.pnl_snapshots = []  # (t, mid, mtm)
        self.quote_log = []      # (t, bid, ask, bsz, asz, mid, sigma, res, reason)
        self.n_quotes_bid = 0
        self.n_quotes_ask = 0

    @property
    def sigma(self):
        """Current fractional volatility estimate (floored)."""
        return max(math.sqrt(self._ewma_var), self._vol_floor)

    @property
    def sigma_realized(self):
        """Trailing time-grid realised vol (fractional), or EWMA if unavailable."""
        if self._rv is not None:
            return self._rv.sigma
        return self.sigma

    def _sigma_frac_for_quotes(self):
        if self.vol_mode == "realized_time" and self._rv is not None:
            return self._rv.sigma
        return self.sigma

    # --- internal helpers ---

    def _update_vol(self, mid, t=None):
        """Update volatility estimators with a new mid-price."""
        if self._prev_mid is not None and self._prev_mid > 0:
            ret = (mid - self._prev_mid) / self._prev_mid
            self._ewma_var = (1.0 - self._alpha) * self._ewma_var + self._alpha * ret * ret
        self._prev_mid = mid
        if t is not None and self._rv is not None:
            self._rv.update(mid, float(t))

    def _compute_quotes(self, mid, sim, return_deltas=False):
        """Return ``(bid, ask, sigma_abs_pln, reservation_mid_native)``.

        All quantities are in PLN, matching the numerical solver and the
        Gueant analytical formulas.  ``self.k`` is expected in PLN units
        (1/PLN), as returned by ``ErgodicMM.calibrate()``.

        When ``return_deltas`` is True the continuous (pre-rounding) bid/ask
        offsets ``(delta_b, delta_a)`` in PLN are appended to the tuple.
        """
        native_to_pln = float(getattr(sim, "price_native_to_pln", 1.0) or 1.0)
        tick_idx = bool(getattr(sim, "bbo_in_tick_index", False))
        mid_pln = float(mid) * native_to_pln

        sigma_frac = self._sigma_frac_for_quotes()
        sigma_abs_pln = sigma_frac * mid_pln
        sigma2 = sigma_abs_pln ** 2

        g = self.gamma
        k = self.k
        A = self.intensity_A
        if self.size > 0:
            q = int(round(self.inventory / self.size))
        else:
            q = self.inventory

        b0 = (1.0 / g) * math.log(1.0 + g / k)
        c = (sigma2 * g) / (2.0 * k * A) * (1.0 + g / k) ** (1.0 + k / g)
        s = math.sqrt(max(0.0, c))

        delta_b = b0 + 0.5 * (2 * q + 1) * s
        delta_a = b0 - 0.5 * (2 * q - 1) * s

        raw_bid_pln = mid_pln - delta_b
        raw_ask_pln = mid_pln + delta_a

        raw_bid_n = raw_bid_pln / native_to_pln
        raw_ask_n = raw_ask_pln / native_to_pln

        if tick_idx:
            bid = float(max(1.0, math.floor(raw_bid_n)))
            ask = float(max(bid + 1.0, math.ceil(raw_ask_n)))
        else:
            ts = float(self.tick_size)
            bid = round(max(ts, math.floor(raw_bid_n / ts) * ts), 8)
            ask = round(max(bid + ts, math.ceil(raw_ask_n / ts) * ts), 8)

        reservation_native = float(mid)
        if return_deltas:
            return bid, ask, sigma_abs_pln, reservation_native, float(delta_b), float(delta_a)
        return bid, ask, sigma_abs_pln, reservation_native

    def _snap_pnl(self, sim, t):
        bb, ba = sim.ob.get_bbo()
        if bb is None or ba is None:
            return
        mid = (bb + ba) / 2.0
        mtm = self.cash + self.inventory * mid
        self.pnl_snapshots.append((float(t), float(mid), float(mtm)))

    def _cancel_all(self, sim, t):
        if self.bid_oid is not None:
            sim.agent_cancel_order(self.bid_oid, t)
            self.bid_oid = None
            self._bid_price = None
        if self.ask_oid is not None:
            sim.agent_cancel_order(self.ask_oid, t)
            self.ask_oid = None
            self._ask_price = None

    def _place_quotes(
        self,
        sim,
        t,
        bid_price,
        ask_price,
        reason="initial",
        mid=None,
        *,
        sigma_abs_pln: Optional[float] = None,
        reservation_native: Optional[float] = None,
    ):
        gbb, gba = sim.ob.get_bbo()
        bid_size = self.size
        ask_size = self.size
        if self._constrained:
            bid_size, ask_size = symmetric_margin_quote_sizes(
                self.cash, self.inventory, gbb, gba,
                bid_price, ask_price, self.size)
            sync_cash_budget(self, gbb, gba)

        placed_bid = bid_price > 0 and bid_size > 0
        placed_ask = ask_price > 0 and ask_size > 0

        if placed_bid:
            self.bid_oid = sim.agent_place_order(1, bid_price, bid_size, t)
            self._bid_price = bid_price
            self.n_quotes_bid += 1
        if placed_ask:
            self.ask_oid = sim.agent_place_order(2, ask_price, ask_size, t)
            self._ask_price = ask_price
            self.n_quotes_ask += 1

        if placed_bid or placed_ask:
            mid_f = _float_or_nan(mid)
            sig_log = _float_or_nan(sigma_abs_pln)
            res_log = _float_or_nan(reservation_native)
            self.quote_log.append((
                float(t),
                float(bid_price) if placed_bid else float('nan'),
                float(ask_price) if placed_ask else float('nan'),
                int(bid_size) if placed_bid else 0,
                int(ask_size) if placed_ask else 0,
                mid_f, sig_log, res_log,
                reason))

    # --- agent callback ---

    def on_event(self, sim, t, fills):
        bb, ba = sim.ob.get_bbo()
        bb_f = _float_or_nan(bb)
        ba_f = _float_or_nan(ba)

        got_fill = False
        for oid, px, qty, side in fills:
            if oid != self.bid_oid and oid != self.ask_oid:
                continue
            got_fill = True
            if side == 1:
                self.inventory += int(qty)
                self.cash -= float(px) * int(qty)
                self.trade_log.append((
                    float(t), +1, float(px), int(qty),
                    bb_f, ba_f,
                    int(self.inventory), float(self.cash)))
                if oid == self.bid_oid and oid not in sim.ob.order_map:
                    self.bid_oid = None
                    self._bid_price = None
            else:
                self.inventory -= int(qty)
                self.cash += float(px) * int(qty)
                self.trade_log.append((
                    float(t), -1, float(px), int(qty),
                    bb_f, ba_f,
                    int(self.inventory), float(self.cash)))
                if oid == self.ask_oid and oid not in sim.ob.order_map:
                    self.ask_oid = None
                    self._ask_price = None

        if got_fill:
            if self._constrained:
                sync_cash_budget(self, bb, ba)
            self._snap_pnl(sim, t)

        if bb is None or ba is None:
            return

        if not got_fill and bb == self._prev_bb and ba == self._prev_ba:
            return
        self._prev_bb = bb
        self._prev_ba = ba

        mid = (bb + ba) / 2.0
        self._update_vol(mid, t)

        bid_price, ask_price, sig_pln, res_nat = self._compute_quotes(mid, sim)

        need_requote = (
            self.bid_oid is None
            or self.ask_oid is None
            or bid_price != self._bid_price
            or ask_price != self._ask_price
        )
        if need_requote:
            self._cancel_all(sim, t)
            if got_fill:
                reason = "fill"
            else:
                reason = "params_change"
            self._place_quotes(
                sim, t, bid_price, ask_price, reason, mid,
                sigma_abs_pln=sig_pln, reservation_native=res_nat,
            )
            if not got_fill:
                self._snap_pnl(sim, t)

    # --- liquidation ---

    def liquidate(self, sim, t):
        """Close out remaining inventory via market order."""
        self._cancel_all(sim, t)

        if self.inventory == 0:
            if self.verbose:
                print("Nothing to liquidate (inventory = 0)")
            if self._constrained:
                bb0, ba0 = sim.ob.get_bbo()
                sync_cash_budget(self, bb0, ba0)
            self._snap_pnl(sim, t)
            return

        bb, ba = sim.ob.get_bbo()
        bb_f = _float_or_nan(bb)
        ba_f = _float_or_nan(ba)

        if self.inventory > 0:
            fills = sim.agent_market_order(2, self.inventory, t)
            for px, qty in fills:
                self.inventory -= int(qty)
                self.cash += float(px) * int(qty)
                self.trade_log.append((
                    float(t), -1, float(px), int(qty),
                    bb_f, ba_f,
                    int(self.inventory), float(self.cash)))
        elif self.inventory < 0:
            need = -self.inventory
            cover_qty = need
            if self._constrained and ba is not None and float(ba) > 0:
                sp = mm_spendable(
                    float(self.cash), int(self.inventory), bb, ba)
                cover_qty = min(need, int(sp // float(ba)))
            if cover_qty > 0:
                fills = sim.agent_market_order(1, cover_qty, t)
                for px, qty in fills:
                    self.inventory += int(qty)
                    self.cash -= float(px) * int(qty)
                    self.trade_log.append((
                        float(t), +1, float(px), int(qty),
                        bb_f, ba_f,
                        int(self.inventory), float(self.cash)))

        if self._constrained:
            bb2, ba2 = sim.ob.get_bbo()
            sync_cash_budget(self, bb2, ba2)
        self._snap_pnl(sim, t)
        if self.verbose:
            print(f"Liquidated -> inventory: {self.inventory}, cash: {self.cash:+,.0f}")

    def to_sqlite(self, db_path):
        """Write trade, quote and PnL logs to a SQLite database."""
        _write_mm_sqlite(self, db_path, side_is_string=False)

    # --- calibration (same methodology as AvellanedaStoikovMM) ---

    @staticmethod
    def calibrate_k(mo_df, tick_size=0.01):
        """Estimate fill-decay parameter *k* from empirical MO penetration data.

        For each market order, computes cumulative opposite-side depth
        at levels 0..9 and checks whether the MO volume exceeded that
        depth.  The survival probabilities are fitted to
        ``P(penetrate beyond d) ~ exp(-k * d)`` via log-linear OLS.

        Returns
        -------
        float
            Fitted *k* value (positive).
        """
        depth_cols = [f"opp_depth_L{i}" for i in range(10)]
        present = [c for c in depth_cols if c in mo_df.columns]
        n_levels = len(present)
        if n_levels < 2:
            raise ValueError("mo_df must contain at least opp_depth_L0 and L1")

        depths = mo_df[present].values
        cum_depth = np.cumsum(depths, axis=1)
        vol = mo_df["mo_volume"].values[:, None]

        penetrates = vol > cum_depth
        p_survive = penetrates.mean(axis=0)

        mask = p_survive > 0
        levels = np.arange(n_levels)
        x = levels[mask]
        y = np.log(p_survive[mask])

        if len(x) < 2:
            print(f"  calibrate_k: only {len(x)} usable level(s), "
                  f"returning k=1.5 (default)")
            return 1.5

        coeffs = np.polyfit(x, y, 1)
        k = float(-coeffs[0])

        if k <= 0:
            print(f"  calibrate_k: fitted k={k:.4f} <= 0, "
                  f"clamping to 0.01")
            k = 0.01

        return k

    @staticmethod
    def calibrate_vol_halflife(orders_df, candidate_halflifes=(20, 50, 100, 200, 500)):
        """Pick the EWMA half-life that best tracks realised variance.

        Returns
        -------
        int
            Best half-life from the candidate set.
        """
        bb = orders_df["best_bid"].values.astype(float)
        ba = orders_df["best_ask"].values.astype(float)
        mid = (bb + ba) / 2.0

        valid = (mid > 0) & np.isfinite(mid)
        mid = mid[valid]
        if len(mid) < 600:
            default = candidate_halflifes[len(candidate_halflifes) // 2]
            print(f"  calibrate_vol_halflife: too few data points "
                  f"({len(mid)}), returning default {default}")
            return default

        ret = np.diff(mid) / mid[:-1]
        ret2 = ret ** 2

        window = 500
        realized = np.convolve(ret2, np.ones(window) / window, mode="valid")
        offset = window - 1

        best_hl = candidate_halflifes[0]
        best_mse = np.inf

        for hl in candidate_halflifes:
            alpha = 1.0 - 0.5 ** (1.0 / hl)
            ewma = np.empty(len(ret2))
            ewma[0] = ret2[0]
            for i in range(1, len(ret2)):
                ewma[i] = (1.0 - alpha) * ewma[i - 1] + alpha * ret2[i]

            ewma_aligned = ewma[offset:]
            mse = float(np.mean((ewma_aligned - realized) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_hl = hl

        return best_hl

    @staticmethod
    def calibrate_A(mo_df, orders_df=None, tick_size=0.01):
        """Estimate the MO arrival intensity *A* (per-side fills/second at delta=0).

        Computes the per-side market-order arrival rate from the
        ``first_time_ns`` column in ``mo_df``.  Session duration per day
        is estimated from the span between first and last MO.

        Parameters
        ----------
        mo_df : pd.DataFrame
            Must contain ``day`` and ``first_time_ns``.
        orders_df : pd.DataFrame, optional
            Unused; accepted for API consistency with ``calibrate()``.
        tick_size : float
            Unused; accepted for API consistency.

        Returns
        -------
        float
            Estimated A (per-side MO arrivals per second).
        """
        days = mo_df["day"].unique()
        total_mo = 0
        total_seconds = 0.0

        for day in days:
            day_mos = mo_df[mo_df["day"] == day]
            n_mo_day = len(day_mos)

            if n_mo_day > 1:
                duration_sec = (
                    day_mos["first_time_ns"].max() - day_mos["first_time_ns"].min()
                ) / 1e9
            else:
                duration_sec = 0.0

            if duration_sec < 60:
                continue

            total_mo += n_mo_day
            total_seconds += duration_sec

        if total_seconds < 1.0:
            print("  calibrate_A: insufficient data, returning A=1.0 (default)")
            return 1.0

        A = (total_mo / 2.0) / total_seconds
        return A

    @classmethod
    def calibrate(cls, mo_df, orders_df, tick_size=0.01):
        """Calibrate structural parameters from empirical data.

        ``calibrate_k`` returns k in per-tick units (fill decay per depth
        level).  This method converts it to per-PLN units so that all
        Gueant formulas work consistently in PLN — matching the numerical
        solver in ``numerical_as.ipynb``.

        Returns
        -------
        dict
            ``{"k": ..., "vol_halflife": ..., "tick_size": ..., "intensity_A": ...}``
            where ``k`` is in 1/PLN.
        """
        k_tick = cls.calibrate_k(mo_df, tick_size=tick_size)
        k_pln = k_tick / float(tick_size)
        vol_halflife = cls.calibrate_vol_halflife(orders_df)
        A = cls.calibrate_A(mo_df, orders_df, tick_size=tick_size)
        return {
            "k": k_pln, "vol_halflife": vol_halflife,
            "tick_size": tick_size, "intensity_A": A,
        }


# --- Numerical ergodic MM: solves the discrete HJB at every BBO change ---

class NumericalErgodicMM:
    """Ergodic market maker that solves the HJB equation numerically.

    Mirrors :class:`ErgodicMM` (same EWMA volatility, same ``k`` in
    1/PLN, same fill / liquidation logic) but replaces the closed-form
    Guéant deltas with a discrete-tick HJB solve at every re-quote
    event.

    The value function ``phi`` is **warm-started** from the previous
    solve, so consecutive solves typically converge in only a handful
    of relaxation steps.  ``Q`` (half-width of the inventory grid)
    grows on demand: when current ``|inventory|`` would touch the
    boundary the grid is expanded and ``phi`` is reset to zero — the
    next solve cold-starts at the new size, but every solve after that
    again warm-starts from the converged state.  To avoid many such
    expansions in a row as ``|q|`` increases, ``Q`` starts at
    ``max(Q_floor, |q_0| + Q_margin)`` instead of only
    ``|q_0| + Q_margin``.  ``Q`` is also *reclaimed*: once a smaller
    grid would have sufficed for ``Q_shrink_patience`` consecutive
    solves, it shrinks back (keeping the central, converged slice of
    ``phi`` as the warm start), so a transient early inventory spike no
    longer leaves every later solve paying for an oversized lattice.

    Parameters
    ----------
    gamma, k, vol_halflife, vol_floor, size, tick_size, intensity_A,
    verbose, initial_cash, initial_inventory, agent_id
        Identical to :class:`ErgodicMM`.
    max_iter : int
        Hard cap on relaxation iterations per solve.  50 000 is
        comfortable: cold starts take a few hundred to a few thousand;
        warm starts complete in 1-2 iterations.
    relax_step, tol
        Standard relaxation parameters.
    delta_lo : float
        Lower bound (PLN) of the discrete delta grid.  Negative values
        let the solver place quotes inside the spread for aggressive
        liquidation at high inventory.
    max_delta : float
        Upper bound (PLN) of the discrete delta grid.  Default ``5.0``;
        quotes at 100+ ticks from mid are never optimal in practice, so
        a wider grid only adds builder overhead when the fill law is
        state-dependent.
    Q_margin : int
        Headroom kept between the current inventory and the grid
        boundary.  ``Q`` grows whenever ``|inventory| + Q_margin``
        would exceed the current grid half-width.  The inventory
        half-width is initialised to at least :attr:`Q_floor`
        (default 30) so typical paths that eventually need
        ``|q| + Q_margin`` near 28--30 do not pay a chain of grid
        expansions (each resets ``phi``) while ``|q|`` ramps up.
    Q_shrink_patience : int
        Number of consecutive solves a smaller grid must suffice before
        ``Q`` is reclaimed (shrunk back toward ``|q| + Q_margin``,
        floored at :attr:`Q_floor`).  Larger values shrink more
        conservatively; the shrink keeps the warm start.
    Q_floor : int, instance attribute
        Floor on ``Q`` (default 30).  Independent of ``Q_margin``;
        growth still uses only ``|inventory| + Q_margin``.
    solver_tick : float, optional
        Resolution of the internal δ-grid the HJB solver searches over.
        Defaults to ``tick_size`` (the market tick).  Setting this
        smaller than ``tick_size`` (e.g. ``solver_tick=0.001`` while
        ``tick_size=0.05``) lets the solver pick δ values much closer
        to the analytical continuous optimum, which makes the final
        ``floor``/``ceil`` rounding agree with :class:`ErgodicMM`'s
        on a much higher fraction of events.  Cost: the per-side
        lambda grid grows roughly proportionally to
        ``tick_size / solver_tick``, and each Hamiltonian maximiser
        does ``log₂`` more comparisons per call.

    Notes
    -----
    The default fill law is the exponential
    ``lambda(delta) = intensity_A * exp(-k * delta)``, identical to
    :class:`ErgodicMM`.  To use a *state-conditional* fill law (e.g.
    one that reads ``sim.hawkes_filter.intensity(t)`` or the live order
    book depth), pass either a pair of arrival-rate callbacks
    ``lam_b(z, delta)`` / ``lam_a(z, delta)`` or a pair of
    fill-probability callbacks ``h_b(z, delta)`` / ``h_a(z, delta)``
    (Poisson-linked over ``poisson_tau``).  When supplied, the lambda
    grids are rebuilt before every solve from
    ``z = state_extractor(sim)`` (defaults to the simulator itself).
    The exponential default path remains state-independent and cached.
    """

    def __init__(self, gamma=1.0, k=1.5,
                 vol_halflife=50, vol_floor=1e-4, size=1, tick_size=1,
                 intensity_A=1.0, verbose=True,
                 initial_cash=None, initial_inventory=0,
                 agent_id=None,
                 max_iter=50_000, relax_step=0.2, tol=1e-4,
                 delta_lo=-0.50, max_delta=5.0, Q_margin=20,
                 Q_shrink_patience=256,
                 solver_tick=None, solver_engine="scan",
                 candidate_grid="uniform",
                 # custom fill-law hooks (default: built-in exponential)
                 lam_b=None, lam_a=None, h_b=None, h_a=None,
                 state_extractor=None, poisson_tau=1.0, h_clamp=1e-9,
                 rv_window_s=200.0, rv_sample_dt_s=1.0):
        """Construct the agent.

        Parameters
        ----------
        gamma, k, vol_halflife, vol_floor, size, tick_size, intensity_A,
        verbose, initial_cash, initial_inventory, agent_id, max_iter,
        relax_step, tol, delta_lo, max_delta, Q_margin
            Solver and book-side knobs (see class docstring).
        lam_b, lam_a : callable(z, delta) -> float, optional
            Per-side arrival-rate functions.  When supplied, the lambda
            grid is rebuilt before every solve using the current state
            ``z = state_extractor(sim)`` (defaults to ``sim`` itself).
        h_b, h_a : callable(z, delta) -> float, optional
            Per-side fill-probability functions over horizon
            ``poisson_tau``; converted to arrival rates via the Poisson
            link ``lambda = -ln(1 - h) / tau``.
        state_extractor : callable(sim) -> z, optional
            Extracts the state object handed to ``lam_*`` / ``h_*``.
            Defaults to passing ``sim`` directly, which is usually
            sufficient (read ``sim.hawkes_filter``, ``sim.ob.*_qty``,
            etc. inside the callback).
        poisson_tau, h_clamp : float
            Horizon and probability clamp for h-mode only.

        When *none* of ``lam_b/lam_a/h_b/h_a`` is supplied the agent
        falls back to the built-in exponential model
        ``lambda(delta) = intensity_A * exp(-k * delta)`` and the grid
        is cached once at construction (state-independent, identical to
        the previous behaviour).
        """
        self.agent_id = agent_id
        self.gamma = float(gamma)
        self.k = float(k)
        self.intensity_A = float(intensity_A)
        self.size = int(size)
        self.tick_size = float(tick_size)
        # Internal δ-grid resolution used by the HJB solver.  Decoupled
        # from ``tick_size`` (the market tick used for the final bid/ask
        # rounding) so the numerical agent can search δ on a much finer
        # grid while still quoting on the market grid.  Defaults to
        # ``tick_size`` for backward compatibility.
        if solver_tick is not None:
            self.solver_tick = float(solver_tick)
        else:
            self.solver_tick = float(tick_size)
        self.verbose = verbose
        self._constrained = initial_cash is not None

        self._alpha = 1.0 - 0.5 ** (1.0 / vol_halflife)
        self._vol_floor = float(vol_floor)
        self._ewma_var = self._vol_floor ** 2
        self._prev_mid = None
        self._prev_t = None
        self._ewma_dt = None

        # Time-grid realised-volatility tracker (matches the NN training
        # feature ``recent_vol`` in vol_mode='realized_time': RMS of log-mid
        # returns sampled on a uniform ``rv_sample_dt_s`` grid over the trailing
        # ``rv_window_s`` seconds; see classes/phantom_labels.realized_vol_time_grid).
        # Advanced online from _update_vol(mid, t); exposed via ``sigma_realized``.
        self._rv_dt = max(1e-9, float(rv_sample_dt_s))
        self._rv_nwin = max(1, int(round(float(rv_window_s) / self._rv_dt)))
        self._rv_buf = np.zeros(self._rv_nwin, dtype=np.float64)
        self._rv_pos = 0
        self._rv_count = 0
        self._rv_sum = 0.0
        self._rv_t0 = None
        self._rv_last_k = -1
        self._rv_prev_grid_mid = None
        self._rv_cur_mid = None

        self.inventory = int(initial_inventory)
        if initial_cash is not None:
            self.cash = float(initial_cash)
            self.budget = float(initial_cash)
        else:
            self.cash = 0.0
            self.budget = 0.0
        self.bid_oid = None
        self.ask_oid = None
        self._bid_price = None
        self._ask_price = None
        self._prev_bb = None
        self._prev_ba = None

        self.trade_log = []
        self.pnl_snapshots = []
        self.quote_log = []
        self.n_quotes_bid = 0
        self.n_quotes_ask = 0

        self.max_iter = int(max_iter)
        self.relax_step = float(relax_step)
        self.tol = float(tol)
        self.delta_lo = float(delta_lo)
        self.max_delta = float(max_delta)
        # "scan": per-query linear scan (original).  "hull": line-envelope
        # maximisation -- same discrete maximum computed in O(log n) per
        # query instead of O(n); identical relaxation scheme otherwise.
        if solver_engine not in ("scan", "hull"):
            raise ValueError(f"solver_engine must be 'scan' or 'hull', "
                             f"got {solver_engine!r}")
        self.solver_engine = str(solver_engine)
        if candidate_grid not in ("uniform", "legal"):
            raise ValueError(
                "candidate_grid must be 'uniform' or 'legal', "
                f"got {candidate_grid!r}"
            )
        self.candidate_grid = str(candidate_grid)
        self.Q_margin = int(Q_margin)
        # Floor on the inventory half-width.  ``Q`` grows on demand and is
        # reclaimed (shrunk) once the larger grid has gone unused for
        # ``Q_shrink_patience`` consecutive solves — early intensity spikes
        # no longer leave the lattice permanently oversized.
        self.Q_floor = 30
        self.Q_shrink_patience = int(Q_shrink_patience)
        self._Q_low_streak = 0

        # The HJB inventory lattice is indexed in LOTS (1 lot = ``size``
        # shares), not raw shares.  Whole-LO matching means a fill always
        # moves inventory by exactly ``size`` shares == one lot, so each
        # lattice step corresponds to one real fill.  This keeps the grid
        # the same size regardless of ``size`` (identical math to the
        # ``size=1`` problem) instead of solving thousands of per-share
        # inventory states.
        q_lots0 = abs(self._inventory_lots())
        self._Q = max(self.Q_floor, q_lots0 + self.Q_margin)
        self._phi = np.zeros(2 * self._Q + 1, dtype=np.float64)

        # Resolve fill-law mode from the user-supplied hooks.
        user_lam = (lam_b is not None) or (lam_a is not None)
        user_h = (h_b is not None) or (h_a is not None)
        if user_lam and user_h:
            raise ValueError(
                "Pass (lam_b, lam_a) OR (h_b, h_a), not both."
            )
        if user_lam and not (lam_b is not None and lam_a is not None):
            raise ValueError("lam_b and lam_a must be provided together.")
        if user_h and not (h_b is not None and h_a is not None):
            raise ValueError("h_b and h_a must be provided together.")

        if user_h:
            self._mode = "h"
            self._fn_b, self._fn_a = h_b, h_a
            self._state_dependent = True
        elif user_lam:
            self._mode = "lam"
            self._fn_b, self._fn_a = lam_b, lam_a
            self._state_dependent = True
        else:
            self._mode = "lam"
            self._fn_b = self._lam_fn
            self._fn_a = self._lam_fn
            self._state_dependent = False

        self.state_extractor = state_extractor
        self.poisson_tau = float(poisson_tau)
        self.h_clamp = float(h_clamp)

        # Per-side lambda grids.  Built once at __init__ for the
        # exponential default (state-independent); rebuilt per solve
        # when state-dependent hooks are supplied.
        self._dg_b = self._lg_b = None
        self._dg_a = self._lg_a = None
        self._pg_b = self._pg_a = None
        self._legal_dg_b = self._legal_dg_a = None
        if not self._state_dependent and self.candidate_grid == "uniform":
            self._build_grids(z=None)

        self.total_solve_time = 0.0
        self.n_solves = 0
        self.total_iters = 0

    @property
    def sigma(self):
        return max(math.sqrt(self._ewma_var), self._vol_floor)

    @property
    def sigma_realized(self):
        """Trailing time-grid realised vol (fractional), matching the NN
        ``recent_vol`` training feature.  Floored at sqrt(1e-8) as in
        :func:`research_core.classes.phantom_labels.realized_vol_time_grid`."""
        if self._rv_count > 0:
            var = self._rv_sum / self._rv_count
        else:
            var = 0.0
        return math.sqrt(max(var, 1e-8))

    def _lam_fn(self, _z, d):
        """Built-in exponential ``lambda(delta) = A · exp(-k δ)``.

        Vectorised: accepts either a scalar or a numpy array of deltas
        and returns matching shape, so the grid builder takes the fast
        single-call path.
        """
        return self.intensity_A * np.exp(-self.k * np.asarray(d, dtype=np.float64))

    def _legal_candidate_grids(self, sim):
        """Build side-specific executable prices and midpoint deltas.

        This uses the same legal-tick geometry as phantom-label creation:
        candidates are actual resting prices, while deltas may be half-ticks
        when the BBO midpoint lies between market ticks.
        """
        bb, ba = sim.ob.get_bbo()
        if bb is None or ba is None:
            raise ValueError("cannot build legal candidates without a valid BBO")

        native_to_pln = float(
            getattr(sim, "price_native_to_pln", 1.0) or 1.0
        )
        bb_pln = float(bb) * native_to_pln
        ba_pln = float(ba) * native_to_pln

        from .phantom_labels import build_legal_tick_state

        unused_depth = np.zeros(1, dtype=np.float64)
        pg_b_pln, dg_b, _ = build_legal_tick_state(
            bb_pln, ba_pln, unused_depth, unused_depth, 1,
            tick_size=self.tick_size, delta_lo=self.delta_lo,
            max_delta=self.max_delta,
        )
        pg_a_pln, dg_a, _ = build_legal_tick_state(
            bb_pln, ba_pln, unused_depth, unused_depth, 2,
            tick_size=self.tick_size, delta_lo=self.delta_lo,
            max_delta=self.max_delta,
        )
        if dg_b.size == 0 or dg_a.size == 0:
            raise ValueError(
                "legal candidate grid is empty for "
                f"BBO=({bb!r}, {ba!r}), delta bounds="
                f"({self.delta_lo}, {self.max_delta})"
            )

        pg_b = pg_b_pln / native_to_pln
        pg_a = pg_a_pln / native_to_pln
        if bool(getattr(sim, "bbo_in_tick_index", False)):
            pg_b = np.rint(pg_b)
            pg_a = np.rint(pg_a)
        return pg_b, dg_b, pg_a, dg_a

    def _build_grids(self, z, sim=None):
        """(Re)tabulate per-side lambda grids from the active fill functions.

        Called once at construction in the state-independent (exponential)
        case, and once per quote when state-dependent hooks are in use.
        """
        grid_b = grid_a = None
        if self.candidate_grid == "legal":
            if sim is None:
                raise ValueError("legal candidate grids require the live simulator")
            self._pg_b, grid_b, self._pg_a, grid_a = (
                self._legal_candidate_grids(sim)
            )
            # Publish these before invoking h(z, delta): the exact queue-ahead
            # path maps every assessed delta back to its executable price.
            self._legal_dg_b = grid_b
            self._legal_dg_a = grid_a

        ergodic_solver = _get_ergodic_solver()
        if self._mode == "lam":
            self._dg_b, self._lg_b = ergodic_solver.precompute_lam_grid_discrete(
                self._fn_b, z, self.solver_tick, self.max_delta,
                delta_lo=self.delta_lo, lam_floor=0.0, delta_grid=grid_b,
            )
            self._dg_a, self._lg_a = ergodic_solver.precompute_lam_grid_discrete(
                self._fn_a, z, self.solver_tick, self.max_delta,
                delta_lo=self.delta_lo, lam_floor=0.0, delta_grid=grid_a,
            )
        else:
            self._dg_b, self._lg_b = ergodic_solver.precompute_lam_grid_discrete_from_h(
                self._fn_b, z, self.poisson_tau, self.solver_tick, self.max_delta,
                delta_lo=self.delta_lo, h_clamp=self.h_clamp,
                delta_grid=grid_b,
            )
            self._dg_a, self._lg_a = ergodic_solver.precompute_lam_grid_discrete_from_h(
                self._fn_a, z, self.poisson_tau, self.solver_tick, self.max_delta,
                delta_lo=self.delta_lo, h_clamp=self.h_clamp,
                delta_grid=grid_a,
            )

    def _update_vol(self, mid, t=None):
        if self._prev_mid is not None and self._prev_mid > 0:
            ret = (mid - self._prev_mid) / self._prev_mid
            self._ewma_var = (1.0 - self._alpha) * self._ewma_var + self._alpha * ret * ret
        self._prev_mid = mid

        if t is not None:
            t_sec = float(t)
            if self._prev_t is not None:
                dt = t_sec - self._prev_t
                if dt > 0:
                    if self._ewma_dt is None:
                        self._ewma_dt = dt
                    else:
                        self._ewma_dt = (1.0 - self._alpha) * self._ewma_dt + self._alpha * dt
            self._prev_t = t_sec
            self._update_rv(mid, t_sec)

    def _update_rv(self, mid, t_sec):
        """Advance the trailing time-grid realised-vol estimator to ``t_sec``.

        Resamples the mid onto a uniform ``self._rv_dt`` grid (LOCF) and keeps a
        ring buffer of the last ``self._rv_nwin`` squared log-returns.  Mirrors
        ``realized_vol_time_grid``: each grid step contributes one squared
        return (the log-move that became effective at that grid time), so a
        constant mid pushes zeros and lets the estimate decay.
        """
        if not (mid > 0):
            return
        if self._rv_t0 is None:
            self._rv_t0 = t_sec
            self._rv_last_k = 0
            self._rv_prev_grid_mid = mid
            self._rv_cur_mid = mid
            return

        k_new = int(math.floor((t_sec - self._rv_t0) / self._rv_dt))
        if k_new <= self._rv_last_k:
            self._rv_cur_mid = mid  # latest in-effect mid for the next grid step
            return

        n_steps = k_new - self._rv_last_k
        if n_steps > self._rv_nwin:
            # Gap exceeds the window — the entire trailing buffer is stale.
            # Reset and fill with floor; the next non-gap step restarts cleanly.
            self._rv_buf[:] = 0.0
            self._rv_pos = 0
            self._rv_count = 0
            self._rv_sum = 0.0
            self._rv_last_k = k_new
            self._rv_prev_grid_mid = mid
            self._rv_cur_mid = mid
            return

        prev_grid_mid = self._rv_prev_grid_mid
        sampled = self._rv_cur_mid  # mid in effect across the gap (pre-update)
        for _ in range(self._rv_last_k + 1, k_new + 1):
            if (prev_grid_mid is not None and prev_grid_mid > 0
                    and sampled is not None and sampled > 0):
                r = math.log(sampled / prev_grid_mid)
                r2 = r * r
            else:
                r2 = 0.0
            old = self._rv_buf[self._rv_pos]
            self._rv_buf[self._rv_pos] = r2
            self._rv_pos = (self._rv_pos + 1) % self._rv_nwin
            if self._rv_count < self._rv_nwin:
                self._rv_count += 1
                self._rv_sum += r2
            else:
                self._rv_sum += r2 - old
            prev_grid_mid = sampled
        self._rv_prev_grid_mid = prev_grid_mid
        self._rv_last_k = k_new
        self._rv_cur_mid = mid

    def _inventory_lots(self):
        """Current inventory in lots (1 lot = ``size`` shares).

        Whole-LO matching makes inventory an exact multiple of ``size``, so
        this is an exact integer; ``round`` only guards against any stray
        non-multiple (e.g. a partial liquidation fill).
        """
        if self.size > 0:
            return int(round(self.inventory / self.size))
        return int(self.inventory)

    def _ensure_Q(self, required_Q):
        """Resize the inventory grid toward ``required_Q`` (in lots).

        Growth is immediate: zero-padding the existing ``phi`` creates a
        discontinuity at the old boundary that the relaxation solver
        struggles to fix, so the grid is reset to zero and the next solve
        cold-starts at the new size (every solve after that warm-starts
        again).

        Shrinking is *hysteretic*: ``Q`` is only reclaimed once the
        smaller size has been sufficient for ``Q_shrink_patience``
        consecutive solves, which avoids thrashing the grid (and the
        cold-start cost) when inventory hovers near a boundary.  Unlike
        growth, a shrink keeps the warm start — the converged inner
        states are exactly the central slice of the current ``phi`` — so
        the next solve still benefits from it.
        """
        target = max(self.Q_floor, int(required_Q))

        if target > self._Q:
            self._Q = target
            self._phi = np.zeros(2 * target + 1, dtype=np.float64)
            self._Q_low_streak = 0
            return

        if target == self._Q:
            self._Q_low_streak = 0
            return

        # target < self._Q: a smaller grid would suffice this solve.
        self._Q_low_streak += 1
        if self._Q_low_streak < self.Q_shrink_patience:
            return

        # Sustained low demand — reclaim the grid, preserving the central
        # (converged) slice of phi as the warm start for the smaller grid.
        off = self._Q - target
        self._phi = self._phi[off:off + (2 * target + 1)].copy()
        self._Q = target
        self._Q_low_streak = 0

    def _solve_phi(self, sigma2):
        """Run the discrete HJB relaxation, warm-started from ``self._phi``.

        ``qs`` are inventory states in LOTS, so the running-inventory penalty
        ``-0.5 * gamma * sigma^2 * q^2`` is per lot: holding one lot
        (``size`` shares) carries the same risk weight a single unit would in
        the ``size=1`` problem ("100 shares behaves like 1").
        """
        Q = self._Q
        qs = np.arange(-Q, Q + 1, dtype=np.float64)
        S = (-0.5 * self.gamma * float(sigma2) * qs * qs).astype(np.float64)

        if self.solver_engine == "hull":
            ergodic_solver = _get_ergodic_solver()
            solver = ergodic_solver.solve_ergodic_discrete_hull
        else:
            ergodic_solver = _get_ergodic_solver()
            solver = ergodic_solver.solve_ergodic_discrete
        t0 = time.perf_counter()
        phi, g, n_iter, converged, _last_dphi = solver(
            S, self._lg_b, self._lg_a, self._dg_b, self._dg_a,
            self.gamma, self.relax_step, self.tol, self.max_iter,
            Q, self._phi.astype(np.float64, copy=True),
        )
        self.total_solve_time += time.perf_counter() - t0
        self.n_solves += 1
        self.total_iters += int(n_iter)
        self._phi = phi
        return g, n_iter, converged

    @staticmethod
    def _legal_price_for_delta(delta, delta_grid, price_grid):
        """Return the executable price paired with a solver-selected delta."""
        idx = int(np.argmin(np.abs(delta_grid - float(delta))))
        if not math.isclose(
            float(delta_grid[idx]), float(delta), rel_tol=0.0, abs_tol=1e-10,
        ):
            raise RuntimeError(
                f"selected delta {delta!r} is absent from legal candidate grid"
            )
        return float(price_grid[idx]), idx

    def _compute_quotes(self, mid, sim, return_deltas=False):
        """Return ``(bid, ask, sigma_abs_pln, reservation_native)``.

        When ``return_deltas`` is True the solver's (pre-rounding) bid/ask
        offsets ``(delta_b, delta_a)`` in PLN are appended to the tuple.
        """
        native_to_pln = float(getattr(sim, "price_native_to_pln", 1.0) or 1.0)
        tick_idx = bool(getattr(sim, "bbo_in_tick_index", False))
        mid_pln = float(mid) * native_to_pln

        sigma_abs_pln = self.sigma * mid_pln
        sigma2 = sigma_abs_pln ** 2

        # When the fill law is state-dependent (h-mode), lambda is in
        # per-second units (from the Poisson link with tau in seconds).
        # sigma^2 is per-event; scale to per-second for unit consistency.
        if self._state_dependent and self._ewma_dt is not None and self._ewma_dt > 0:
            sigma2 = sigma2 / self._ewma_dt

        # State-dependent fill laws are rebuilt every quote.  Legal grids are
        # BBO-dependent, so they are also rebuilt even for a stationary law.
        if self._state_dependent or self.candidate_grid == "legal":
            if self.state_extractor is not None:
                z = self.state_extractor(sim)
            else:
                z = sim
            self._build_grids(z, sim=sim)

        # Solve / index the lattice in lots (1 lot = ``size`` shares).
        q_lots = self._inventory_lots()
        self._ensure_Q(abs(q_lots) + self.Q_margin)
        self._solve_phi(sigma2)

        ergodic_solver = _get_ergodic_solver()
        delta_b, delta_a = ergodic_solver.optimal_deltas_discrete(
            q_lots, self._phi, self.gamma,
            self._lg_b, self._lg_a, self._dg_b, self._dg_a, self._Q,
        )

        if self.candidate_grid == "legal":
            if q_lots >= self._Q:
                bid = 0.0
                bid_idx = None
            else:
                bid, bid_idx = self._legal_price_for_delta(
                    delta_b, self._dg_b, self._pg_b,
                )
            if q_lots <= -self._Q:
                ask = 0.0
                ask_idx = None
            else:
                ask, ask_idx = self._legal_price_for_delta(
                    delta_a, self._dg_a, self._pg_a,
                )

            # With delta_lo <= 0 both sides can independently choose the
            # midpoint tick. Keep the old ask-side tie-break, but move to the
            # next *assessed legal candidate* instead of post-hoc rounding.
            if bid > 0.0 and ask > 0.0 and ask <= bid:
                valid_asks = np.flatnonzero(self._pg_a > bid)
                if valid_asks.size:
                    ask_idx = int(valid_asks[0])
                    ask = float(self._pg_a[ask_idx])
                    delta_a = float(self._dg_a[ask_idx])
                else:
                    valid_bids = np.flatnonzero(self._pg_b < ask)
                    if not valid_bids.size:
                        raise RuntimeError("legal bid/ask candidate grids cross")
                    bid_idx = int(valid_bids[0])
                    bid = float(self._pg_b[bid_idx])
                    delta_b = float(self._dg_b[bid_idx])
        else:
            # Uniform deltas retain the historical midpoint inversion and
            # floor/ceil executable-price snapping.
            raw_bid_pln = mid_pln - float(delta_b)
            raw_ask_pln = mid_pln + float(delta_a)
            raw_bid_n = raw_bid_pln / native_to_pln
            raw_ask_n = raw_ask_pln / native_to_pln

            if tick_idx:
                bid = float(max(1.0, math.floor(raw_bid_n)))
                ask = float(max(bid + 1.0, math.ceil(raw_ask_n)))
            else:
                ts = float(self.tick_size)
                bid = round(max(ts, math.floor(raw_bid_n / ts) * ts), 8)
                ask = round(max(bid + ts, math.ceil(raw_ask_n / ts) * ts), 8)

        if return_deltas:
            return bid, ask, sigma_abs_pln, float(mid), float(delta_b), float(delta_a)
        return bid, ask, sigma_abs_pln, float(mid)

    def _snap_pnl(self, sim, t):
        bb, ba = sim.ob.get_bbo()
        if bb is None or ba is None:
            return
        mid = (bb + ba) / 2.0
        mtm = self.cash + self.inventory * mid
        self.pnl_snapshots.append((float(t), float(mid), float(mtm)))

    def _cancel_all(self, sim, t):
        if self.bid_oid is not None:
            sim.agent_cancel_order(self.bid_oid, t)
            self.bid_oid = None
            self._bid_price = None
        if self.ask_oid is not None:
            sim.agent_cancel_order(self.ask_oid, t)
            self.ask_oid = None
            self._ask_price = None

    def _place_quotes(self, sim, t, bid_price, ask_price, reason="initial",
                      mid=None, *,
                      sigma_abs_pln: Optional[float] = None,
                      reservation_native: Optional[float] = None):
        gbb, gba = sim.ob.get_bbo()
        bid_size = self.size
        ask_size = self.size
        if self._constrained:
            bid_size, ask_size = symmetric_margin_quote_sizes(
                self.cash, self.inventory, gbb, gba,
                bid_price, ask_price, self.size)
            sync_cash_budget(self, gbb, gba)

        placed_bid = bid_price > 0 and bid_size > 0
        placed_ask = ask_price > 0 and ask_size > 0

        if placed_bid:
            self.bid_oid = sim.agent_place_order(1, bid_price, bid_size, t)
            self._bid_price = bid_price
            self.n_quotes_bid += 1
        if placed_ask:
            self.ask_oid = sim.agent_place_order(2, ask_price, ask_size, t)
            self._ask_price = ask_price
            self.n_quotes_ask += 1

        if placed_bid or placed_ask:
            mid_f = _float_or_nan(mid)
            sig_log = _float_or_nan(sigma_abs_pln)
            res_log = _float_or_nan(reservation_native)
            self.quote_log.append((
                float(t),
                float(bid_price) if placed_bid else float('nan'),
                float(ask_price) if placed_ask else float('nan'),
                int(bid_size) if placed_bid else 0,
                int(ask_size) if placed_ask else 0,
                mid_f, sig_log, res_log, reason))

    def on_event(self, sim, t, fills):
        bb, ba = sim.ob.get_bbo()
        bb_f = _float_or_nan(bb)
        ba_f = _float_or_nan(ba)

        got_fill = False
        for oid, px, qty, side in fills:
            if oid != self.bid_oid and oid != self.ask_oid:
                continue
            got_fill = True
            if side == 1:
                self.inventory += int(qty)
                self.cash -= float(px) * int(qty)
                self.trade_log.append((
                    float(t), +1, float(px), int(qty),
                    bb_f, ba_f, int(self.inventory), float(self.cash)))
                if oid == self.bid_oid and oid not in sim.ob.order_map:
                    self.bid_oid = None
                    self._bid_price = None
            else:
                self.inventory -= int(qty)
                self.cash += float(px) * int(qty)
                self.trade_log.append((
                    float(t), -1, float(px), int(qty),
                    bb_f, ba_f, int(self.inventory), float(self.cash)))
                if oid == self.ask_oid and oid not in sim.ob.order_map:
                    self.ask_oid = None
                    self._ask_price = None

        if got_fill:
            if self._constrained:
                sync_cash_budget(self, bb, ba)
            self._snap_pnl(sim, t)

        if bb is None or ba is None:
            return

        if not got_fill and bb == self._prev_bb and ba == self._prev_ba:
            return
        self._prev_bb = bb
        self._prev_ba = ba

        mid = (bb + ba) / 2.0
        self._update_vol(mid, t)

        bid_price, ask_price, sig_pln, res_nat = self._compute_quotes(mid, sim)

        need_requote = (
            self.bid_oid is None
            or self.ask_oid is None
            or bid_price != self._bid_price
            or ask_price != self._ask_price
        )
        if need_requote:
            self._cancel_all(sim, t)
            if got_fill:
                reason = "fill"
            else:
                reason = "params_change"
            self._place_quotes(
                sim, t, bid_price, ask_price, reason, mid,
                sigma_abs_pln=sig_pln, reservation_native=res_nat,
            )
            if not got_fill:
                self._snap_pnl(sim, t)

    def liquidate(self, sim, t):
        """Close out remaining inventory via market order."""
        self._cancel_all(sim, t)

        if self.inventory == 0:
            if self.verbose:
                print("Nothing to liquidate (inventory = 0)")
            if self._constrained:
                bb0, ba0 = sim.ob.get_bbo()
                sync_cash_budget(self, bb0, ba0)
            self._snap_pnl(sim, t)
            return

        bb, ba = sim.ob.get_bbo()
        bb_f = _float_or_nan(bb)
        ba_f = _float_or_nan(ba)

        if self.inventory > 0:
            fills = sim.agent_market_order(2, self.inventory, t)
            for px, qty in fills:
                self.inventory -= int(qty)
                self.cash += float(px) * int(qty)
                self.trade_log.append((
                    float(t), -1, float(px), int(qty),
                    bb_f, ba_f, int(self.inventory), float(self.cash)))
        elif self.inventory < 0:
            need = -self.inventory
            cover_qty = need
            if self._constrained and ba is not None and float(ba) > 0:
                sp = mm_spendable(
                    float(self.cash), int(self.inventory), bb, ba)
                cover_qty = min(need, int(sp // float(ba)))
            if cover_qty > 0:
                fills = sim.agent_market_order(1, cover_qty, t)
                for px, qty in fills:
                    self.inventory += int(qty)
                    self.cash -= float(px) * int(qty)
                    self.trade_log.append((
                        float(t), +1, float(px), int(qty),
                        bb_f, ba_f, int(self.inventory), float(self.cash)))

        if self._constrained:
            bb2, ba2 = sim.ob.get_bbo()
            sync_cash_budget(self, bb2, ba2)
        self._snap_pnl(sim, t)
        if self.verbose:
            print(f"Liquidated -> inventory: {self.inventory}, cash: {self.cash:+,.0f}")

    def to_sqlite(self, db_path):
        _write_mm_sqlite(self, db_path, side_is_string=False)

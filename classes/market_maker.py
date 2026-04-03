"""
Example market-making agents for use with ``Simulate(agents=[...])``.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

class SimpleMarketMaker:
    """Market-making agent that quotes symmetrically around the BBO.

    Places a bid and an ask ``offset`` ticks from the current BBO.
    Quotes are automatically cancelled and re-placed whenever the BBO
    moves from where the quotes were originally placed, keeping the
    agent's orders aligned with the current market (same re-quoting
    regime as ``AvellanedaStoikovMM``).

    Works with both integer-tick simulators (``tick_size=1``, the default)
    and decimal-price empirical data (e.g. ``tick_size=0.05``).

    Parameters
    ----------
    offset : int
        Number of ticks away from the best price.
    size : int
        Order size placed on each side.
    tick_size : float
        Price increment per tick.  Defaults to 1 (integer ticks).
    verbose : bool
        When True (default) keeps a detailed ``order_log`` and prints
        status messages during liquidation.  Set to False for batch
        runs / parameter sweeps to save memory.
    initial_cash : float or None
        Starting cash balance.  When set, the MM operates under a hard
        budget constraint.  A separate *budget* tracks how much
        capital is still available: opening any position (buying or
        shorting) costs budget, closing a position (selling owned
        stock or covering a short) returns budget at the fill price.
        The budget is split 50/50 between the bid and ask sides.
        When ``None`` (default) the MM has unlimited capital (classic
        unconstrained mode).
    initial_inventory : int
        Starting inventory (number of shares already held).  Only
        meaningful when *initial_cash* is set.
    """

    def __init__(self, offset=1, size=100, tick_size=1,
                 verbose=True, initial_cash=None, initial_inventory=0):
        self.offset = int(offset)
        self.size = int(size)
        self.tick_size = float(tick_size)
        self.verbose = verbose
        self._constrained = initial_cash is not None

        self.bid_oid = None
        self.ask_oid = None
        self._quote_bb = None
        self._quote_ba = None

        self.inventory = int(initial_inventory)
        self.cash = float(initial_cash) if initial_cash is not None else 0.0
        self.budget = float(initial_cash) if initial_cash is not None else 0.0

        self.trade_log = []         # [(t, side, price, qty, inventory_after, cash_after)]
        self.order_log = [] if verbose else None
        self.pnl_snapshots = []     # [(t, mid, mark_to_market)]

    # -- helpers --

    def _place_quotes(self, sim, t):
        """Place bid and ask limit orders around the current BBO."""
        bb, ba = sim.ob.get_bbo()
        if bb is None or ba is None:
            return

        bid_price = round(bb - self.offset * self.tick_size, 8)
        ask_price = round(ba + self.offset * self.tick_size, 8)

        bid_size = self.size
        ask_size = self.size
        if self._constrained:
            half_budget = self.budget / 2.0
            bid_size = min(bid_size, int(half_budget // bid_price)) if bid_price > 0 and half_budget > 0 else 0
            ask_size = min(ask_size, max(0, self.inventory) + int(half_budget // ask_price)) if ask_price > 0 and half_budget > 0 else 0

        if bid_price > 0 and bid_size > 0:
            self.bid_oid = sim.agent_place_order(1, bid_price, bid_size, t)
            if self.verbose:
                self.order_log.append((t, "PLACE", "BID", bid_price, bid_size, bb, ba))
        if ask_price > 0 and ask_size > 0:
            self.ask_oid = sim.agent_place_order(2, ask_price, ask_size, t)
            if self.verbose:
                self.order_log.append((t, "PLACE", "ASK", ask_price, ask_size, bb, ba))

        self._quote_bb = bb
        self._quote_ba = ba

    def _cancel_all(self, sim, t):
        """Cancel any live quotes."""
        if self.bid_oid is not None:
            if self.verbose:
                bb, ba = sim.ob.get_bbo()
                entry = sim.ob.order_map.get(self.bid_oid)
                px = entry[1] if entry else None
                self.order_log.append((t, "CANCEL", "BID", px, None, bb, ba))
            sim.agent_cancel_order(self.bid_oid, t)
            self.bid_oid = None
        if self.ask_oid is not None:
            if self.verbose:
                bb, ba = sim.ob.get_bbo()
                entry = sim.ob.order_map.get(self.ask_oid)
                px = entry[1] if entry else None
                self.order_log.append((t, "CANCEL", "ASK", px, None, bb, ba))
            sim.agent_cancel_order(self.ask_oid, t)
            self.ask_oid = None

    def _snap_pnl(self, sim, t):
        """Record a mark-to-market PnL snapshot."""
        bb, ba = sim.ob.get_bbo()
        if bb is not None and ba is not None:
            mid = (bb + ba) / 2.0
            mtm = self.cash + self.inventory * mid
            self.pnl_snapshots.append((float(t), float(mid), float(mtm)))

    # -- main callback --

    def on_event(self, sim, t, fills):
        got_fill = False
        for oid, price, qty, resting_side in fills:
            if oid == self.bid_oid:
                old_abs = abs(self.inventory)
                self.inventory += int(qty)
                self.cash -= float(price) * int(qty)
                if self._constrained:
                    new_abs = abs(self.inventory)
                    self.budget += (old_abs - new_abs) * float(price)
                self.trade_log.append(
                    (float(t), "BUY", float(price), int(qty),
                     int(self.inventory), float(self.cash)))
                got_fill = True
                if oid not in sim.ob.order_map:
                    self.bid_oid = None

            elif oid == self.ask_oid:
                old_abs = abs(self.inventory)
                self.inventory -= int(qty)
                self.cash += float(price) * int(qty)
                if self._constrained:
                    new_abs = abs(self.inventory)
                    self.budget += (old_abs - new_abs) * float(price)
                self.trade_log.append(
                    (float(t), "SELL", float(price), int(qty),
                     int(self.inventory), float(self.cash)))
                got_fill = True
                if oid not in sim.ob.order_map:
                    self.ask_oid = None

        if got_fill:
            self._snap_pnl(sim, t)

        if self.bid_oid is None and self.ask_oid is None:
            self._place_quotes(sim, t)
            self._snap_pnl(sim, t)
            return

        bb, ba = sim.ob.get_bbo()
        if bb != self._quote_bb or ba != self._quote_ba:
            self._cancel_all(sim, t)
            self._place_quotes(sim, t)

    # -- reporting --

    def mark_to_market(self, mid_price):
        """Unrealised + realised PnL at a given mid price."""
        return self.cash + self.inventory * mid_price

    def summary(self, mid_price=None):
        n_trades = len(self.trade_log)
        buys  = sum(q for _, s, _, q, *_ in self.trade_log if s == "BUY")
        sells = sum(q for _, s, _, q, *_ in self.trade_log if s == "SELL")
        print(f"SimpleMarketMaker  |  trades: {n_trades}  "
              f"(bought {buys}, sold {sells})")
        print(f"  inventory: {self.inventory:+d}  |  cash: {self.cash:+,.0f}")
        if mid_price is not None:
            mtm = self.mark_to_market(mid_price)
            print(f"  mark-to-market PnL: {mtm:+,.0f}  (mid={mid_price})")
        if self.order_log is not None:
            print(f"  order actions logged: {len(self.order_log)}")
        print(f"  pnl snapshots: {len(self.pnl_snapshots)}")

    @staticmethod
    def _scale_times(raw):
        """Auto-scale raw times to seconds/minutes/hours."""
        if len(raw) == 0:
            return raw, "time"
        span = raw[-1] - raw[0]
        if span > 7200:
            return [t / 3600.0 for t in raw], "time (hours)"
        elif span > 120:
            return [t / 60.0 for t in raw], "time (minutes)"
        else:
            return raw, "time (seconds)"

    def plot_pnl(self):
        """Plot mark-to-market PnL, cash, inventory, and mid-price over time."""

        times = [s[0] for s in self.pnl_snapshots]
        mids  = [s[1] for s in self.pnl_snapshots]
        pnls  = [s[2] for s in self.pnl_snapshots]

        buy_t  = [r[0] for r in self.trade_log if r[1] == "BUY"]
        sell_t = [r[0] for r in self.trade_log if r[1] == "SELL"]
        buy_inv  = [r[-2] for r in self.trade_log if r[1] == "BUY"]
        sell_inv = [r[-2] for r in self.trade_log if r[1] == "SELL"]

        inv_t  = [r[0] for r in self.trade_log]
        inv_v  = [r[-2] for r in self.trade_log]
        cash_t = [r[0] for r in self.trade_log]
        cash_v = [r[-1] for r in self.trade_log]

        times, xlabel = self._scale_times(times)
        buy_t, _ = self._scale_times(buy_t)
        sell_t, _ = self._scale_times(sell_t)
        inv_t, _ = self._scale_times(inv_t)
        cash_t, _ = self._scale_times(cash_t)

        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

        ax = axes[0]
        ax.plot(times, pnls, lw=1, color="tab:green", label="Mark-to-market PnL")
        ax.axhline(0, ls="--", color="grey", alpha=0.5)
        ax.set_ylabel("PnL")
        ax.set_title("Market Maker Performance")
        ax.legend(loc="upper left")

        ax = axes[1]
        if inv_t:
            ax.step(inv_t, inv_v, where="post", lw=1, color="tab:blue", label="Inventory")
            ax.scatter(buy_t, buy_inv, marker="^", color="green", s=15, alpha=0.7, label="Buy fill", zorder=3)
            ax.scatter(sell_t, sell_inv, marker="v", color="red", s=15, alpha=0.7, label="Sell fill", zorder=3)
        ax.axhline(0, ls="--", color="grey", alpha=0.5)
        ax.set_ylabel("Inventory")
        ax.legend(loc="upper left")

        ax = axes[2]
        if cash_t:
            ax.step(cash_t, cash_v, where="post", lw=1, color="tab:purple", label="Cash")
        ax.axhline(0, ls="--", color="grey", alpha=0.5)
        ax.set_ylabel("Cash")
        ax.legend(loc="upper left")

        ax = axes[3]
        ax.plot(times, mids, lw=0.5, color="tab:orange", label="Mid price")
        ax.set_ylabel("Mid price")
        ax.set_xlabel(xlabel)
        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.show()

    def print_trade_log(self, last_n=20):
        """Print the last N trades."""
        trades = self.trade_log[-last_n:]
        print(f"Last {len(trades)} trades (of {len(self.trade_log)} total):")
        for t, side, px, qty, inv, cash in trades:
            print(f"  t={t:>12.2f}  {side:4s}  {qty}@{px}  "
                  f"inv={inv:+d}  cash={cash:+,.0f}")

    def liquidate(self, sim, t):
        """Close out remaining inventory via market order at end of sim."""
        self._cancel_all(sim, t)

        if self.inventory == 0:
            if self.verbose:
                print("Nothing to liquidate (inventory = 0)")
            return

        if self.inventory > 0:
            fills = sim.agent_market_order(2, self.inventory, t)
            for px, qty in fills:
                old_abs = abs(self.inventory)
                self.inventory -= int(qty)
                self.cash += float(px) * int(qty)
                if self._constrained:
                    self.budget += (old_abs - abs(self.inventory)) * float(px)
                self.trade_log.append(
                    (float(t), "SELL", float(px), int(qty),
                     int(self.inventory), float(self.cash)))
        else:
            fills = sim.agent_market_order(1, -self.inventory, t)
            for px, qty in fills:
                old_abs = abs(self.inventory)
                self.inventory += int(qty)
                self.cash -= float(px) * int(qty)
                if self._constrained:
                    self.budget += (old_abs - abs(self.inventory)) * float(px)
                self.trade_log.append(
                    (float(t), "BUY", float(px), int(qty),
                     int(self.inventory), float(self.cash)))

        self._snap_pnl(sim, t)
        if self.verbose:
            print(f"Liquidated -> inventory: {self.inventory}, cash: {self.cash:+,.0f}")

    def print_order_log(self, last_n=20):
        """Print the last N order actions (only available when verbose=True)."""
        if self.order_log is None:
            print("Order log not available (verbose=False).")
            return
        actions = self.order_log[-last_n:]
        print(f"Last {len(actions)} order actions (of {len(self.order_log)} total):")
        for t, action, side, px, sz, bb, ba in actions:
            sz_str = str(sz) if sz else "-"
            px_str = str(px) if px else "-"
            print(f"  t={t:>12.2f}  {action:6s} {side:3s}  "
                  f"px={px_str}  sz={sz_str}  BBO=[{bb}, {ba}]")


CompactMarketMaker = SimpleMarketMaker


# ═══════════════════════════════════════════════════════════════════════════════
# Avellaneda-Stoikov optimal-quoting market maker
# ═══════════════════════════════════════════════════════════════════════════════

class AvellanedaStoikovMM:
    """Market maker using the Avellaneda-Stoikov optimal quoting model.

    Quotes are placed symmetrically around a *reservation price* that is
    skewed away from the mid-price proportionally to current inventory,
    with a half-spread determined by volatility, risk-aversion, and the
    order-arrival decay parameter *k*.

    Reservation price:
        r = mid - q * gamma * sigma^2 * horizon

    Optimal half-spread:
        delta = gamma * sigma^2 * horizon / 2  +  (1/gamma) * ln(1 + gamma/k)

    Volatility is estimated online via an EWMA of squared mid-price
    returns.  The terminal-time horizon is kept as a fixed rolling
    lookahead (stationary behaviour, no end-of-session urgency).

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient (> 0).  Higher values penalise
        inventory more aggressively and widen the spread.
    k : float
        Exponential decay of fill probability with quoting depth.
        Depends on the market-order size distribution (stable for a
        given simulator configuration).
    horizon : float
        Rolling lookahead that replaces (T - t) in the AS formulas.
        Measured in the same time units as the simulator clock.
    vol_halflife : int
        EWMA half-life (in number of mid-price updates) for the
        squared-return volatility estimator.
    vol_floor : float
        Minimum volatility to prevent numerical instability.
    size : int
        Order size placed on each side.
    tick_size : int
        Minimum price increment for rounding quotes.
    initial_cash : float or None
        Starting cash balance.  When set, the MM operates under a hard
        budget constraint.  A separate *budget* tracks how much
        capital is still available: opening any position (buying or
        shorting) costs budget, closing a position (selling owned
        stock or covering a short) returns budget at the fill price.
        The budget is split 50/50 between the bid and ask sides.
        When ``None`` (default) the MM has unlimited capital (classic
        unconstrained mode).
    initial_inventory : int
        Starting inventory (number of shares already held).  Only
        meaningful when *initial_cash* is set.
    """

    def __init__(self, gamma=0.1, k=1.5, horizon=5000.0,
                 vol_halflife=50, vol_floor=1e-4, size=1, tick_size=1,
                 verbose=True, initial_cash=None, initial_inventory=0):
        self.gamma = float(gamma)
        self.k = float(k)
        self.horizon = float(horizon)
        self.size = int(size)
        self.tick_size = float(tick_size)
        self.verbose = verbose
        self._constrained = initial_cash is not None

        # EWMA volatility estimator
        self._alpha = 1.0 - 0.5 ** (1.0 / vol_halflife)
        self._vol_floor = float(vol_floor)
        self._ewma_var = self._vol_floor ** 2
        self._prev_mid = None

        # Order / position state
        self.inventory = int(initial_inventory)
        self.cash = float(initial_cash) if initial_cash is not None else 0.0
        self.budget = float(initial_cash) if initial_cash is not None else 0.0
        self.bid_oid = None
        self.ask_oid = None
        self._bid_price = None
        self._ask_price = None

        self.trade_log = []      # (t, side_sign, inventory_after, cash_after)
        self.pnl_snapshots = []  # (t, mid, mtm)

    @property
    def sigma(self):
        """Current volatility estimate (floored)."""
        return max(math.sqrt(self._ewma_var), self._vol_floor)

    # ── internal helpers ──────────────────────────────────────────────

    def _update_vol(self, mid):
        """Update the EWMA variance estimate with a new mid-price."""
        if self._prev_mid is not None and self._prev_mid > 0:
            ret = (mid - self._prev_mid) / self._prev_mid
            self._ewma_var = (1.0 - self._alpha) * self._ewma_var + self._alpha * ret * ret
        self._prev_mid = mid

    def _compute_quotes(self, mid):
        """Return (bid_price, ask_price) on the tick grid.

        k is used in per-level units as calibrated — this gives wider
        spreads that provide proper adverse-selection protection and
        makes gamma behave in the intuitive direction (higher = wider).
        """
        sigma2 = self.sigma ** 2
        reservation = mid - self.inventory * self.gamma * sigma2 * self.horizon
        half_spread = (
            self.gamma * sigma2 * self.horizon / 2.0
            + (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        )
        raw_bid = reservation - half_spread
        raw_ask = reservation + half_spread

        ts = self.tick_size
        bid = round(max(ts, math.floor(raw_bid / ts) * ts), 8)
        ask = round(max(bid + ts, math.ceil(raw_ask / ts) * ts), 8)
        return bid, ask

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

    def _place_quotes(self, sim, t, bid_price, ask_price):
        bid_size = self.size
        ask_size = self.size
        if self._constrained:
            half_budget = self.budget / 2.0
            bid_size = min(bid_size, int(half_budget // bid_price)) if bid_price > 0 and half_budget > 0 else 0
            ask_size = min(ask_size, max(0, self.inventory) + int(half_budget // ask_price)) if ask_price > 0 and half_budget > 0 else 0

        if bid_price > 0 and bid_size > 0:
            self.bid_oid = sim.agent_place_order(1, bid_price, bid_size, t)
            self._bid_price = bid_price
        if ask_price > 0 and ask_size > 0:
            self.ask_oid = sim.agent_place_order(2, ask_price, ask_size, t)
            self._ask_price = ask_price

    # ── agent callback ────────────────────────────────────────────────

    def on_event(self, sim, t, fills):
        # 1. Process fills
        got_fill = False
        for oid, px, qty, side in fills:
            got_fill = True
            old_abs = abs(self.inventory)
            if side == 1:
                self.inventory += int(qty)
                self.cash -= float(px) * int(qty)
                if self._constrained:
                    self.budget += (old_abs - abs(self.inventory)) * float(px)
                self.trade_log.append((float(t), +1, int(self.inventory), float(self.cash)))
                if oid == self.bid_oid and oid not in sim.ob.order_map:
                    self.bid_oid = None
                    self._bid_price = None
            else:
                self.inventory -= int(qty)
                self.cash += float(px) * int(qty)
                if self._constrained:
                    self.budget += (old_abs - abs(self.inventory)) * float(px)
                self.trade_log.append((float(t), -1, int(self.inventory), float(self.cash)))
                if oid == self.ask_oid and oid not in sim.ob.order_map:
                    self.ask_oid = None
                    self._ask_price = None

        if got_fill:
            self._snap_pnl(sim, t)

        # 2. Update volatility
        bb, ba = sim.ob.get_bbo()
        if bb is None or ba is None:
            return
        mid = (bb + ba) / 2.0
        self._update_vol(mid)

        # 3. Compute optimal quotes
        bid_price, ask_price = self._compute_quotes(mid)

        # 4. Re-quote if prices changed or orders are missing
        need_requote = (
            self.bid_oid is None
            or self.ask_oid is None
            or bid_price != self._bid_price
            or ask_price != self._ask_price
        )
        if need_requote:
            self._cancel_all(sim, t)
            self._place_quotes(sim, t, bid_price, ask_price)
            if not got_fill:
                self._snap_pnl(sim, t)

    # ── liquidation ───────────────────────────────────────────────────

    def liquidate(self, sim, t):
        """Close out remaining inventory via market order."""
        self._cancel_all(sim, t)

        if self.inventory == 0:
            if self.verbose:
                print("Nothing to liquidate (inventory = 0)")
            self._snap_pnl(sim, t)
            return

        if self.inventory > 0:
            fills = sim.agent_market_order(2, self.inventory, t)
            for px, qty in fills:
                old_abs = abs(self.inventory)
                self.inventory -= int(qty)
                self.cash += float(px) * int(qty)
                if self._constrained:
                    self.budget += (old_abs - abs(self.inventory)) * float(px)
                self.trade_log.append((float(t), -1, int(self.inventory), float(self.cash)))
        elif self.inventory < 0:
            fills = sim.agent_market_order(1, -self.inventory, t)
            for px, qty in fills:
                old_abs = abs(self.inventory)
                self.inventory += int(qty)
                self.cash -= float(px) * int(qty)
                if self._constrained:
                    self.budget += (old_abs - abs(self.inventory)) * float(px)
                self.trade_log.append((float(t), +1, int(self.inventory), float(self.cash)))

        self._snap_pnl(sim, t)
        if self.verbose:
            print(f"Liquidated -> inventory: {self.inventory}, cash: {self.cash:+,.0f}")

    # ── calibration ────────────────────────────────────────────────

    @staticmethod
    def calibrate_k(mo_df, tick_size=0.01):
        """Estimate the fill-decay parameter *k* from empirical MO data.

        For each market order, computes cumulative opposite-side depth
        at levels 0..9 and checks whether the MO volume exceeded that
        depth (i.e. penetrated beyond that level).  The survival
        probabilities are then fitted to ``P(penetrate beyond d) ~
        exp(-k * d)`` via log-linear least squares.

        Parameters
        ----------
        mo_df : pd.DataFrame
            Must contain ``mo_volume`` and ``opp_depth_L0`` through
            ``opp_depth_L9``.
        tick_size : float
            Not used in the calculation but accepted for API
            consistency with ``calibrate()``.

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

        print(f"  calibrate_k: fitted k = {k:.4f}  "
              f"(from {len(x)} levels, {len(mo_df)} MOs)")
        return k

    @staticmethod
    def calibrate_vol_halflife(orders_df, candidate_halflifes=None):
        """Pick the EWMA half-life that best tracks realised variance.

        Computes mid-price returns from BBO data, then for each
        candidate half-life evaluates the EWMA variance series and
        compares it against a rolling realised variance.  Returns the
        half-life with the lowest mean-squared error.

        Parameters
        ----------
        orders_df : pd.DataFrame
            Must contain ``best_bid`` and ``best_ask``, sorted by time.
        candidate_halflifes : list[int] or None
            Half-lives to evaluate.  Defaults to
            ``[20, 50, 100, 200, 500]``.

        Returns
        -------
        int
            Best half-life from the candidate set.
        """
        if candidate_halflifes is None:
            candidate_halflifes = [20, 50, 100, 200, 500]

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

        # Rolling realised variance (window = 500)
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

        print(f"  calibrate_vol_halflife: best half-life = {best_hl}  "
              f"(MSE = {best_mse:.2e}, from {len(candidate_halflifes)} "
              f"candidates)")
        return best_hl

    @classmethod
    def calibrate(cls, mo_df, orders_df, tick_size=0.01):
        """Calibrate structural parameters from empirical data.

        Estimates ``k`` (fill-decay) from ``mo_orders`` data and
        ``vol_halflife`` from BBO data.  Returns a dict that can be
        unpacked into the constructor alongside user-chosen preference
        parameters (``gamma``, ``horizon``, etc.).

        Parameters
        ----------
        mo_df : pd.DataFrame
            Market-order data (from ``mo_orders`` table).
        orders_df : pd.DataFrame
            Order-book snapshot data with ``best_bid``, ``best_ask``.
        tick_size : float
            Minimum price increment for the instrument.

        Returns
        -------
        dict
            ``{"k": ..., "vol_halflife": ..., "tick_size": ...}``
        """
        print("AvellanedaStoikovMM.calibrate:")
        k = cls.calibrate_k(mo_df, tick_size=tick_size)
        vol_halflife = cls.calibrate_vol_halflife(orders_df)
        params = {"k": k, "vol_halflife": vol_halflife, "tick_size": tick_size}
        print(f"  -> {params}")
        return params



"""
AnalyseMarket — unified analysis class for order-flow SQLite databases.

Works with both:
  - Empirical order-flow databases (partitioned by `day`, with `cls_method`)
  - Simulation output databases (continuous `timestamp`, with `bbo` and `intensities` tables)
"""

import json
import sqlite3
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.stats import linregress, norm, lognorm, beta as beta_dist, gaussian_kde
from scipy.optimize import minimize_scalar, curve_fit

from .extract import filter_market_hours
from .helpers import resolve_data_path


class AnalyseMarket:
    """Unified analysis class for order-flow SQLite databases."""

    # --- Construction ---
    def __init__(self, db_path, tick_size=0.05, day=None, verbose=True,
                 market_open="09:00:00", market_close="16:50:00"):
        """
        Parameters
        ----------
        db_path : str or Path
            Path to the SQLite database.
        tick_size : float
            Minimum price increment (used for display / conversions).
        day : str or None
            If set, restrict all queries to this day (real data only).
            Example: ``day='d20170110'``.
        verbose : bool
            If False, skip construction prints (useful when looping over many DBs).
        market_open, market_close : str or None
            Local session window in ``Europe/Warsaw``, same defaults as
            ``extract.run_full_extraction``.  When either is ``None``, session
            filtering is disabled.  For mid-price paths, filtering applies only
            when ``has_day`` (empirical WSE-style DB); simulation DBs are unchanged.
        """
        self.db_path = resolve_data_path(db_path)
        self.tick_size = tick_size
        self.day = day
        self.market_open = market_open
        self.market_close = market_close

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = self._conn()
        cur = conn.cursor()

        # Available tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        self.tables = {row[0] for row in cur.fetchall()}

        # Schema detection
        order_cols = self._table_cols(conn, "orders")
        self.has_day = "day" in order_cols
        mo_cols = (self._table_cols(conn, "mo_orders")
                   if "mo_orders" in self.tables else [])
        self.has_cls_method = "cls_method" in mo_cols
        self.has_bbo = "bbo" in self.tables
        self.has_intensities = "intensities" in self.tables

        conn.close()

        # Cancellation calibration cache
        self._cancel_pass_done = False

        if self.has_day:
            src = "real data"
        else:
            src = "simulation"
        if verbose:
            print(f"AnalyseMarket | {self.db_path.name} | {src}")
            print(f"  Tables: {sorted(self.tables)}")
            if self.day:
                print(f"  Day filter: {self.day}")

    # --- Internal helpers ---
    def _conn(self):
        """Return a fresh SQLite connection."""
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _table_cols(conn, table):
        cur = conn.cursor()
        try:
            cur.execute(f"PRAGMA table_info({table})")
            return [row[1] for row in cur.fetchall()]
        except sqlite3.Error:
            return []

    def _day_clause(self, table_alias=""):
        """SQL fragment: `` AND day = '...' `` when day is set."""
        if self.day and self.has_day:
            if table_alias:
                col = f"{table_alias}.day"
            else:
                col = "day"
            return f" AND {col} = '{self.day}'"
        return ""

    # --- Mid-price helpers ---
    def _mid_price_query(self):
        """Return the SQL query string for mid-price retrieval."""
        if self.has_bbo:
            return "SELECT timestamp, mid_price FROM bbo ORDER BY timestamp"
        return ("SELECT timestamp, mid_price FROM orders "
                "WHERE mid_price IS NOT NULL"
                + self._day_clause()
                + " ORDER BY timestamp")

    def _parse_ts(self, raw_ts):
        """Convert a raw timestamp (string or float) to float seconds.

        For string (ISO-8601) timestamps the conversion is deferred to
        ``_get_mid_prices`` where the full array can be batch-converted
        with ``_collapse_trading_time``.  Here we just pass floats
        through and flag strings.
        """
        if isinstance(raw_ts, str):
            return raw_ts  # handled later in batch
        return float(raw_ts)

    @staticmethod
    def _posix_unit_from_magnitude(ts_num):
        """Infer pandas ``unit`` for epoch-encoded numeric timestamps."""
        m = float(np.nanmedian(np.abs(np.asarray(ts_num, dtype=np.float64))))
        if m > 1e16:
            return "ns"
        if m > 1e13:
            return "us"
        if m > 1e11:
            return "ms"
        return "s"

    @staticmethod
    def _numeric_ts_is_absolute_wall_clock(ts_num):
        """True if *ts_num* looks like POSIX (or ms/ns) wall time, not sim elapsed.

        Simulation clocks in this project usually start near 0 and stay < 1e8 s.
        Empirical Unix seconds are ~1.7e9; ns timestamps ~1.7e18.
        """
        a = np.asarray(ts_num, dtype=np.float64)
        if a.size < 2 or not np.isfinite(a).all():
            return False
        return float(np.nanmin(a)) >= 1.0e8

    def _maybe_filter_empirical_session(self, ts_dt, mids):
        """Restrict (ts_dt, mids) to exchange session for empirical DBs only.

        Uses ``extract.filter_market_hours`` (``Europe/Warsaw``).  Skipped when
        ``not self.has_day`` (simulation) or when ``market_open`` /
        ``market_close`` are ``None``.
        """
        if not self.has_day:
            return ts_dt, mids
        if self.market_open is None or self.market_close is None:
            return ts_dt, mids
        df = pd.DataFrame({
            "_t": pd.Series(ts_dt),
            "_m": np.asarray(mids, dtype=np.float64),
        })
        df = filter_market_hours(df, self.market_open, self.market_close, "_t")
        if len(df) < 5:
            return None, None
        return df["_t"].reset_index(drop=True), df["_m"].to_numpy(dtype=np.float64)

    def _get_mid_prices(self):
        """Return (timestamps, mid_prices) NumPy arrays.

        Timestamps are always returned as float64 seconds.
        ISO strings and **numeric POSIX-style** clocks (Unix s/ms/us/ns) are
        converted to datetimes and passed through ``_collapse_trading_time`` so
        fixed-interval LOCF grids do not step through closed markets (which
        otherwise repeats the same mid and explodes kurtosis on ΔP).

        For empirical databases (``has_day``), rows are also restricted to the
        configured local session window (defaults match ``extract``).

        Pure simulation runs with a small continuous clock (values starting
        near 0) are left as-is.

        Uses a raw SQLite cursor instead of ``pd.read_sql`` to avoid
        the memory overhead of a pandas DataFrame.
        """
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(self._mid_price_query())
            rows = cur.fetchall()
            if len(rows) < 5:
                return None, None

            ts_raw = [r[0] for r in rows]
            mids = np.array([r[1] for r in rows], dtype=np.float64)

            if isinstance(ts_raw[0], str):
                ts_dt = pd.to_datetime(ts_raw, utc=True)
                ts_dt, mids = self._maybe_filter_empirical_session(ts_dt, mids)
                if mids is None:
                    return None, None
                ts = self._collapse_trading_time(pd.Series(ts_dt))
            else:
                ts_num = np.array(ts_raw, dtype=np.float64)
                if self._numeric_ts_is_absolute_wall_clock(ts_num):
                    unit = self._posix_unit_from_magnitude(ts_num)
                    ts_dt = pd.to_datetime(ts_num, unit=unit, utc=True)
                    ts_dt, mids = self._maybe_filter_empirical_session(ts_dt, mids)
                    if mids is None:
                        return None, None
                    ts = self._collapse_trading_time(pd.Series(ts_dt))
                else:
                    ts = ts_num

            return ts, mids
        finally:
            conn.close()

    def _get_sampled_mid_prices(self, interval_sec):
        """Last LOCF mid at each grid point spaced by *interval_sec* on the mid axis.

        Always uses ``_get_mid_prices`` so string ISO and numeric POSIX clocks
        share the same (collapsed) time axis.  Pure simulation floats keep a
        continuous axis.

        Returns a 1-D ``float64`` array of sampled mids, or ``None``.
        """
        if interval_sec <= 0:
            return None
        ts, mids = self._get_mid_prices()
        if mids is None or len(mids) < 5:
            return None
        grid = np.arange(ts[0], ts[-1], interval_sec)
        if len(grid) < 2:
            return None
        idx = np.clip(
            np.searchsorted(ts, grid, side="right") - 1,
            0, len(mids) - 1,
        )
        return mids[idx]

    def _get_mid_returns(self):
        """Return (timestamps, log_returns) or (None, None)."""
        ts, mids = self._get_mid_prices()
        if mids is None or len(mids) < 5:
            return None, None
        log_mids = np.log(mids)
        if not np.all(np.isfinite(log_mids)):
            return None, None
        return ts[1:], np.diff(log_mids)

    # --- Statistics helpers ---
    @staticmethod
    def _ppf_normal(p):
        """Approximate inverse-Normal CDF (Beasley-Springer-Moro)."""
        a = [-3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00,  2.938163982698783e+00]
        d = [ 7.784695709041462e-03,  3.224671290700398e-01,
              2.445134137142996e+00,  3.754408661907416e+00]

        p_low  = 0.02425
        p_high = 1.0 - p_low

        if p < p_low:
            q = np.sqrt(-2.0 * np.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
        else:
            q = np.sqrt(-2.0 * np.log(1.0 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)

    @staticmethod
    def _acf(x, max_lag):
        """Autocorrelation function up to *max_lag* (loop version)."""
        xd = x - x.mean()
        var = np.dot(xd, xd)
        if var < 1e-30:
            return np.zeros(max_lag)
        lags = np.arange(1, min(max_lag + 1, len(x)))
        return np.array([np.dot(xd[l:], xd[:-l]) / var for l in lags])

    @staticmethod
    def _acf_fft(x, max_lag):
        """FFT-based autocorrelation — fast for large *max_lag*."""
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n < 2:
            return np.zeros(max_lag + 1)
        xc = x - x.mean()
        xf = np.fft.rfft(xc, n=2 * n)
        ac = np.fft.irfft(xf * np.conj(xf))[:max_lag + 1]
        if ac[0] != 0:
            ac /= ac[0]
        else:
            ac[:] = 0.0
        return ac  # ac[0] = 1, ac[k] = rho(k)

    # --- Summary ---
    def summary(self):
        """Print database summary statistics."""
        conn = self._conn()
        cur = conn.cursor()

        n_orders = cur.execute(
            "SELECT COUNT(*) FROM orders WHERE 1=1"
            + self._day_clause()
        ).fetchone()[0]

        if self.has_day:
            days = [r[0] for r in cur.execute(
                "SELECT DISTINCT day FROM orders ORDER BY day"
            ).fetchall()]
            print(f"Orders: {n_orders:,} rows across {len(days)} days")
            print(f"Days: {days[0]} … {days[-1]}")
        else:
            print(f"Orders: {n_orders:,} rows")

        if "fills" in self.tables:
            n_fills = cur.execute(
                "SELECT COUNT(*) FROM fills"
            ).fetchone()[0]
            print(f"Fills: {n_fills:,}")

        if "mo_orders" in self.tables:
            n_mos = cur.execute(
                "SELECT COUNT(*) FROM mo_orders"
            ).fetchone()[0]
            print(f"Market orders: {n_mos:,}")

        if self.has_bbo:
            n_bbo = cur.execute(
                "SELECT COUNT(*) FROM bbo"
            ).fetchone()[0]
            print(f"BBO snapshots: {n_bbo:,}")

        conn.close()

    def trade_classification(self):
        """Print trade classification breakdown (real data with cls_method)."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table.")
            return

        conn = self._conn()
        cur = conn.cursor()
        n_fills = cur.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
        n_mos = cur.execute("SELECT COUNT(*) FROM mo_orders").fetchone()[0]
        mo_buy = cur.execute(
            "SELECT COUNT(*) FROM mo_orders WHERE side='buy'"
        ).fetchone()[0]
        mo_sell = cur.execute(
            "SELECT COUNT(*) FROM mo_orders WHERE side='sell'"
        ).fetchone()[0]

        print(f"Raw fills: {n_fills:,}")
        print(f"Aggregated MOs: {n_mos:,}")
        print(f"  Buy: {mo_buy:,}  |  Sell: {mo_sell:,}")

        if self.has_cls_method:
            for method in ["quote", "midpoint", "tick", "unclassified"]:
                cnt = cur.execute(
                    "SELECT COUNT(*) FROM mo_orders "
                    "WHERE cls_method=?", (method,)
                ).fetchone()[0]
                if n_mos > 0:
                    pct = 100 * cnt / n_mos
                else:
                    pct = 0
                print(f"  {method}: {cnt:,} ({pct:.2f}%)")

        conn.close()

    # --- Spread Analysis ---
    def spread_diagnostics(self, max_ticks=200):
        """Spread PMF (log-log) and linear frequency plot."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT CAST(spread_ticks AS INTEGER) AS st, COUNT(*) "
            "FROM orders WHERE spread_ticks IS NOT NULL "
            "AND spread_ticks <= ?"
            + self._day_clause()
            + " GROUP BY st ORDER BY st",
            (max_ticks,),
        )
        rows = cur.fetchall()
        conn.close()

        counts = pd.Series(
            {int(r[0]): int(r[1]) for r in rows}
        ).sort_index()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.scatter(counts.index, counts / counts.sum())
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Spread (ticks)"); ax.set_ylabel("Probability")
        ax.set_title("Spread PMF (log–log)")

        ax = axes[1]
        n_show = min(40, len(counts))
        ax.plot(counts.index[1:n_show], counts.values[1:n_show], marker="o")
        ax.set_xlabel("Spread (ticks)"); ax.set_ylabel("Count")
        ax.set_title("Spread frequency (first ticks)")

        plt.tight_layout(); plt.show()
        print(f"Modal spread (ticks): {counts.idxmax()}")
        return counts

    # --- Order Price Distribution ---
    def order_price_distribution(self, max_abs_tick=1000):
        """Signed relative price distribution for limit orders."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT CAST(ticks_from_best AS INTEGER) AS tfb, COUNT(*) "
            "FROM orders "
            "WHERE event_type = 'LO' AND ticks_from_best IS NOT NULL"
            + self._day_clause()
            + " GROUP BY tfb ORDER BY tfb"
        )
        rows = cur.fetchall()
        conn.close()

        counts = pd.Series(
            {int(r[0]): int(r[1]) for r in rows}
        ).sort_index()
        total = counts.sum()
        probs = counts / total

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter (symlog x)
        ax = axes[0]
        ax.scatter(probs.index, probs.values, s=3)
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("log")
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Ticks from best (signed)")
        ax.set_ylabel("Probability")
        ax.set_title("Relative price distribution (symlog)")

        # Histogram from grouped counts
        ax = axes[1]
        mask = (counts.index >= -max_abs_tick) & (counts.index <= max_abs_tick)
        sub = counts[mask]
        ax.bar(sub.index, sub.values / total, width=1.0, alpha=0.75)
        ax.set_yscale("log"); ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Ticks from best (signed)")
        ax.set_ylabel("Density (log)")
        ax.set_title("Signed relative price histogram")

        plt.tight_layout(); plt.show()

    # --- Inside-Spread Placement ---
    def inside_spread_placement(self, max_spread=20):
        """
        Inside-spread conditional PMF with linear-fit parameter *c*.

        Returns dict  ``{regime_label: c_value}``.
        """
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT spread_ticks, ticks_from_best FROM orders "
            "WHERE event_type = 'LO' AND spread_ticks IS NOT NULL "
            "  AND ticks_from_best IS NOT NULL"
            + self._day_clause()
        )
        rows = cur.fetchall()
        conn.close()

        spread = np.array([r[0] for r in rows], dtype=np.float64)
        tfb = np.array([r[1] for r in rows], dtype=np.float64)

        # tfb < 0 means the order improves on the own-side best (inside the
        # spread); x = -tfb is the improvement in ticks toward the mid.
        inside_mask = (tfb < 0) & (tfb > -spread)
        p_best = (tfb == 0).mean()
        p_passive = (tfb > 0).mean()
        print(f"P(best): {p_best:.4f}  |  P(passive): {p_passive:.4f}")
        print(f"Inside-spread orders: {inside_mask.sum():,} "
              f"({inside_mask.mean():.4f})")

        in_mask = inside_mask & (spread <= max_spread)
        x_all = -tfb[in_mask]
        s_all = spread[in_mask]
        u_all = x_all / s_all

        regimes = {
            "s = 3":   (s_all == 3),
            "s = 4":   (s_all == 4),
            "s = 5–8": (s_all >= 5) & (s_all <= 8),
            "s ≥ 9":   (s_all >= 9),
        }

        def _fit_c(u_sub):
            def neg_ll(c):
                vals = 1 - c * u_sub
                if np.any(vals <= 0) or (1 - c / 2) <= 0:
                    return 1e12
                return -(np.sum(np.log(vals))
                         - len(u_sub) * np.log(1 - c / 2))
            return minimize_scalar(
                neg_ll, bounds=(-1.99, 0.999), method="bounded"
            ).x

        regime_results = {}
        for label, mask_r in regimes.items():
            u_sub = u_all[mask_r]
            if len(u_sub) < 50:
                continue
            c_r = _fit_c(u_sub)
            regime_results[label] = c_r
            print(f"  {label:10s}: n={len(u_sub):>7,}, c={c_r:.6f}")

        c_global = _fit_c(u_all)
        print(f"  {'Global':10s}: n={len(u_all):>7,}, c={c_global:.6f}")

        # --- Per-regime PMF subplots ---
        fig, axes = plt.subplots(
            1, len(regimes), figsize=(5 * len(regimes), 5), sharey=True
        )
        for ax, (label, mask_r) in zip(axes, regimes.items()):
            c_r = regime_results.get(label, 0.0)
            s_sub, x_sub = s_all[mask_r], x_all[mask_r]
            for s in sorted(set(s_sub.astype(int))):
                m = s_sub == s
                if m.sum() < 100:
                    continue
                cnts = pd.Series(x_sub[m]).value_counts().sort_index()
                pr = cnts / cnts.sum()
                ax.plot(pr.index, pr.values, marker="o", ms=4,
                        label=f"s={s}", alpha=0.7)
                x_grid = np.arange(1, s)
                if len(x_grid) > 0 and c_r > 0:
                    p_fit = ((1 - c_r * x_grid / s)
                             / ((s - 1) * (1 - c_r / 2)))
                    ax.plot(x_grid, p_fit, "--", color="gray", alpha=0.5)
            ax.set_xlabel("Ticks from best")
            ax.set_title(f"{label}  (c = {c_r:.4f})")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Conditional probability")
        fig.suptitle("In-spread placement PMF per regime", y=1.02)
        plt.tight_layout(); plt.show()

        # --- Collapsed relative-position density ---
        fig2, axes2 = plt.subplots(
            1, len(regimes), figsize=(5 * len(regimes), 5), sharey=True
        )
        for ax, (label, mask_r) in zip(axes2, regimes.items()):
            c_r = regime_results.get(label, 0.0)
            u_sub, s_sub = u_all[mask_r], s_all[mask_r]
            for s in sorted(set(s_sub.astype(int))):
                m = s_sub == s
                if m.sum() < 100 or s < 4:
                    continue
                h, e = np.histogram(u_sub[m], bins=20, range=(0, 1),
                                    density=True)
                ctr = 0.5 * (e[:-1] + e[1:])
                v = h > 0
                ax.plot(ctr[v], h[v], ".", label=f"s={s}", alpha=0.7)
            u_grid = np.linspace(0.01, 0.99, 200)
            if c_r > 0:
                ax.plot(u_grid,
                        (1 - c_r * u_grid) / (1 - c_r / 2),
                        "k-", lw=2, label=f"fit (c={c_r:.4f})")
            else:
                ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform")
            ax.set_xlabel("u = x / s")
            ax.set_title(label)
            ax.legend(fontsize=7)
        axes2[0].set_ylabel("Density")
        fig2.suptitle("Collapsed in-spread density per regime", y=1.02)
        plt.tight_layout(); plt.show()

        return regime_results

    # --- Probability vs Spread ---
    def probability_vs_spread(self, max_spread=None):
        """P(best|s), P(inside|s), P(passive|s) vs spread.

        Sign convention (see ``extract.py``): ``ticks_from_best`` is measured
        from the *own-side* best, positive pointing away from the mid.  So
        ``0`` = at best, ``-1..-(s-1)`` = inside the spread, ``<= -s`` =
        crossing/marketable, ``> 0`` = deeper in the book.
        """
        conn = self._conn()
        cur = conn.cursor()
        if max_spread:
            max_clause = f" AND spread_ticks <= {int(max_spread)}"
        else:
            max_clause = ""
        cur.execute(
            "SELECT CAST(spread_ticks AS INTEGER) AS st, "
            "  COUNT(*) AS n, "
            "  SUM(CASE WHEN ticks_from_best = 0 THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN ticks_from_best < 0 "
            "       AND ticks_from_best > -spread_ticks THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN ticks_from_best > 0 THEN 1 ELSE 0 END) "
            "FROM orders "
            "WHERE event_type = 'LO' AND spread_ticks IS NOT NULL "
            "  AND ticks_from_best IS NOT NULL"
            + self._day_clause() + max_clause
            + " GROUP BY st ORDER BY st"
        )
        rows = cur.fetchall()
        conn.close()

        spread_values, p_best, p_inside, p_passive = [], [], [], []
        for st, n, n_best, n_inside, n_passive in rows:
            spread_values.append(int(st))
            if n < 200:
                p_best.append(np.nan)
                p_inside.append(np.nan)
                p_passive.append(np.nan)
            else:
                p_best.append(n_best / n)
                p_inside.append(n_inside / n)
                p_passive.append(n_passive / n)

        plt.figure(figsize=(8, 6))
        plt.plot(spread_values, p_best, marker="o", label="P(best | s)")
        plt.plot(spread_values, p_inside, marker="o", label="P(inside | s)")
        plt.plot(spread_values, p_passive, marker="o", label="P(passive | s)")
        plt.xlabel("Spread (ticks)"); plt.ylabel("Probability")
        plt.title("Order placement regime vs spread")
        plt.legend(); plt.tight_layout(); plt.show()

        return spread_values, p_best, p_inside, p_passive

    def piecewise_inside_fit(self, spread_values=None,
                             p_inside_values=None, max_spread=20):
        """Piecewise linear fit of P(inside | s)."""
        if spread_values is None or p_inside_values is None:
            spread_values, _, p_inside_values, _ = self.probability_vs_spread()

        s = np.array(spread_values, dtype=float)
        p = np.array(p_inside_values, dtype=float)
        mask = s <= max_spread
        s, p = s[mask], p[mask]

        m1 = (s >= 2) & (s <= 9)
        m2 = (s > 9) & (s <= max_spread)

        c1 = np.polyfit(s[m1], p[m1], 1)
        c2 = np.polyfit(s[m2], p[m2], 1)

        plt.figure(figsize=(8, 6))
        plt.scatter(s, p, label="Empirical", zorder=3)
        s1 = np.linspace(2, 9, 100)
        plt.plot(s1, c1[0] * s1 + c1[1], lw=2, label="Regime 1 (2–9)")
        s2 = np.linspace(9, max_spread, 100)
        plt.plot(s2, c2[0] * s2 + c2[1], lw=2, label=f"Regime 2 (9–{max_spread})")
        plt.xlabel("Spread (ticks)"); plt.ylabel("P(inside | s)")
        plt.legend(); plt.tight_layout(); plt.show()

        print(f"Regime 1 (2–9):   slope={c1[0]:.6f}, intercept={c1[1]:.6f}")
        print(f"Regime 2 (9–{max_spread}): slope={c2[0]:.6f}, intercept={c2[1]:.6f}")
        return c1, c2

    # --- Passive Depth Analysis ---
    def _power_law_fit(self, x, y):
        """Log-log OLS fit.  Returns (beta, r2, slope, intercept)."""
        logx, logy = np.log10(x), np.log10(y)
        coef = np.polyfit(logx, logy, 1)
        y_pred = coef[0] * logx + coef[1]
        ss_res = np.sum((logy - y_pred) ** 2)
        ss_tot = np.sum((logy - logy.mean()) ** 2)
        if ss_tot > 0:
            r2 = 1 - ss_res / ss_tot
        else:
            r2 = 0
        return -coef[0], r2, coef[0], coef[1]

    def _load_passive_depth(self):
        """Load passive LO depth data from DB.

        ``ticks_from_best > 0`` means the order sits behind the own-side
        best, and the depth in ticks behind the best is ``ticks_from_best``
        itself (matching ``Simulate.sample_passive_depth``, which prices a
        passive order at ``best -/+ depth`` ticks).
        """
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT ticks_from_best, spread_ticks, side, imbalance "
            "FROM orders "
            "WHERE event_type = 'LO' "
            "  AND ticks_from_best IS NOT NULL "
            "  AND spread_ticks IS NOT NULL "
            "  AND ticks_from_best > 0"
            + self._day_clause()
        )
        rows = cur.fetchall()
        conn.close()
        df = pd.DataFrame(rows,
                          columns=["ticks_from_best", "spread_ticks",
                                   "side", "imbalance"])
        df["depth"] = df["ticks_from_best"]
        return df

    def passive_depth_pmf(self, num_bins=80):
        """Passive depth PMF with overall power-law fit."""
        df = self._load_passive_depth()
        depth = df["depth"]

        bins = np.logspace(np.log10(depth.min()), np.log10(depth.max()),
                           num_bins)
        hist, edges = np.histogram(depth, bins=bins, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        m = hist > 0
        x, y = centers[m], hist[m]

        beta, r2, slope, intercept = self._power_law_fit(x, y)

        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        y_fit = 10 ** intercept * x_fit ** slope

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, s=8, label="Empirical")
        plt.plot(x_fit, y_fit, lw=2, label=f"Power-law (β = {beta:.3f})")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Passive depth (ticks beyond spread)")
        plt.ylabel("Density")
        plt.title("Passive depth PMF with power-law fit")
        plt.legend(); plt.tight_layout(); plt.show()

        print(f"Total passive orders: {len(depth):,}")
        print(f"β = {beta:.4f},  R² = {r2:.4f}")
        return beta, r2

    def passive_depth_by_side(self, num_bins=80):
        """Passive depth split by bid vs ask."""
        df = self._load_passive_depth()
        df = df[df["depth"] > 0]

        bins = np.logspace(np.log10(df["depth"].min()),
                           np.log10(df["depth"].max()), num_bins)

        plt.figure(figsize=(9, 6))
        results = []
        for label, side_val in [("Bid orders", 1), ("Ask orders", 2)]:
            d = df.loc[df["side"] == side_val, "depth"]
            if len(d) < 100:
                continue
            hist, edges = np.histogram(d, bins=bins, density=True)
            centers = np.sqrt(edges[:-1] * edges[1:])
            m = hist > 0
            x, y = centers[m], hist[m]
            plt.scatter(x, y, s=10, label=f"{label} (n={len(d):,})")

            beta, r2, slope, intercept = self._power_law_fit(x, y)
            results.append((label, beta, r2))
            xf = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
            plt.plot(xf, 10 ** intercept * xf ** slope, lw=2)

        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Passive depth (ticks)"); plt.ylabel("Density")
        plt.title("Passive depth: Bid vs Ask")
        plt.legend(); plt.tight_layout(); plt.show()

        for label, beta, r2 in results:
            print(f"{label}: β = {beta:.4f},  R² = {r2:.4f}")

    def passive_depth_by_spread(self, num_bins=80):
        """Passive depth by spread regime."""
        df = self._load_passive_depth()
        regimes = {
            "Spread = 1": df[df["spread_ticks"] == 1],
            "Spread 2-8": df[(df["spread_ticks"] >= 2)
                             & (df["spread_ticks"] <= 8)],
            "Spread ≥ 9": df[df["spread_ticks"] >= 9],
        }

        bins = np.logspace(np.log10(df["depth"].min()),
                           np.log10(df["depth"].max()), num_bins)

        plt.figure(figsize=(9, 6))
        for label, data in regimes.items():
            d = data["depth"]
            if len(d) < 100:
                continue
            hist, edges = np.histogram(d, bins=bins, density=True)
            centers = np.sqrt(edges[:-1] * edges[1:])
            m = hist > 0
            x, y = centers[m], hist[m]
            plt.scatter(x, y, s=8, label=f"{label} (n={len(d):,})")

            beta, _, slope, intercept = self._power_law_fit(x, y)
            xf = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
            plt.plot(xf, 10 ** intercept * xf ** slope, lw=2)
            print(f"{label}: β = {beta:.4f}")

        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Passive depth (ticks)"); plt.ylabel("Density")
        plt.title("Passive depth by spread regime")
        plt.legend(); plt.tight_layout(); plt.show()

    def passive_depth_by_imbalance(self, num_bins=80):
        """Passive depth by order-book imbalance × side."""
        df = self._load_passive_depth()
        df = df[df["depth"] > 0]

        imb_regimes = {
            "Ask heavy (I < -0.3)": df["imbalance"] < -0.3,
            "Neutral (-0.3 ≤ I ≤ 0.3)":
                (df["imbalance"] >= -0.3) & (df["imbalance"] <= 0.3),
            "Bid heavy (I > 0.3)": df["imbalance"] > 0.3,
        }

        bins = np.logspace(np.log10(df["depth"].min()),
                           np.log10(df["depth"].max()), num_bins)
        results = []

        for side_val, side_name in [(1, "Bid orders"), (2, "Ask orders")]:
            plt.figure(figsize=(9, 6))
            for label, mask in imb_regimes.items():
                d = df.loc[mask & (df["side"] == side_val), "depth"]
                if len(d) < 100:
                    continue
                hist, edges = np.histogram(d, bins=bins, density=True)
                centers = np.sqrt(edges[:-1] * edges[1:])
                m = hist > 0
                x, y = centers[m], hist[m]
                plt.scatter(x, y, s=10, label=f"{label} (n={len(d):,})")

                beta, r2, slope, intercept = self._power_law_fit(x, y)
                results.append((side_name, label, beta, r2))
                xf = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
                plt.plot(xf, 10 ** intercept * xf ** slope, lw=2)

            plt.xscale("log"); plt.yscale("log")
            plt.xlabel("Passive depth (ticks)"); plt.ylabel("Density")
            plt.title(f"Passive depth – {side_name}")
            plt.legend(); plt.tight_layout(); plt.show()

        for s, r, b, r2 in results:
            print(f"{s} | {r}: β = {b:.4f},  R² = {r2:.4f}")

    # --- Queue Position ---
    def queue_at_bbo(self, bins=150):
        """Queue-position histogram for LOs placed at best price."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT CAST(queue_ahead AS INTEGER) AS qa, COUNT(*) "
            "FROM orders "
            "WHERE event_type='LO' AND ticks_from_best=0 "
            "  AND queue_ahead IS NOT NULL"
            + self._day_clause()
            + " GROUP BY qa ORDER BY qa"
        )
        rows = cur.fetchall()
        conn.close()

        vals = np.array([r[0] for r in rows], dtype=np.float64)
        cnts = np.array([r[1] for r in rows], dtype=np.float64)
        total = cnts.sum()
        mean_qa = np.sum(vals * cnts) / total
        cs = np.cumsum(cnts)
        median_qa = vals[np.searchsorted(cs, total / 2)]

        plt.figure(figsize=(6, 4))
        plt.bar(vals, cnts, width=max(1, (vals[-1] - vals[0]) / bins),
                log=True)
        plt.xlabel("Queue ahead"); plt.ylabel("Frequency")
        plt.title("Queue position at BBO")
        plt.show()

        print(f"Mean: {mean_qa:.1f},  Median: {median_qa:.1f}")

    def queue_passive(self, bins=60):
        """Queue position for passive orders with power-law tail fit."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT CAST(queue_ahead AS INTEGER) AS qa, COUNT(*) "
            "FROM orders "
            "WHERE event_type='LO' AND ticks_from_best > 0 "
            "  AND queue_ahead IS NOT NULL AND queue_ahead > 0"
            + self._day_clause()
            + " GROUP BY qa ORDER BY qa"
        )
        rows = cur.fetchall()
        conn.close()

        vals = np.array([r[0] for r in rows], dtype=np.float64)
        cnts = np.array([r[1] for r in rows], dtype=np.float64)

        # Expand grouped counts into log-binned histogram
        log_bins = np.logspace(np.log10(vals.min()),
                               np.log10(vals.max()), bins)
        hist = np.zeros(len(log_bins) - 1)
        bin_idx = np.searchsorted(log_bins, vals, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, len(hist) - 1)
        for i, c in zip(bin_idx, cnts):
            hist[i] += c
        total = cnts.sum()
        widths = np.diff(log_bins)
        hist = hist / (total * widths)
        centers = np.sqrt(log_bins[:-1] * log_bins[1:])

        # Percentile-based tail fit from grouped data
        cs = np.cumsum(cnts)
        p90_idx = np.searchsorted(cs, 0.9 * total)
        xmin = float(vals[min(p90_idx, len(vals) - 1)])
        tail_mask = vals >= xmin
        tail_vals = vals[tail_mask]
        tail_cnts = cnts[tail_mask]
        n_tail = tail_cnts.sum()
        log_sum = np.sum(tail_cnts * np.log(tail_vals / xmin))
        if log_sum > 0:
            alpha = 1 + n_tail / log_sum
        else:
            alpha = 2.0

        plt.figure(figsize=(6, 4))
        m = hist > 0
        plt.loglog(centers[m], hist[m], "o", label="Empirical")
        x_fit = np.linspace(xmin, x.max(), 200)
        y_fit = (alpha - 1) / xmin * (x_fit / xmin) ** (-alpha)
        plt.loglog(x_fit, y_fit, "--",
                   label=f"Power law α ≈ {alpha:.2f}")
        plt.xlabel("Queue ahead"); plt.ylabel("Density")
        plt.title("Passive queue position")
        plt.legend(); plt.grid(True, alpha=0.3); plt.show()

        print(f"Estimated α = {alpha:.3f},  xmin = {xmin:.0f}")

    # --- Order Size Distributions ---
    def _plot_size_density(self, shares, title, n_bins,
                           mid_range, tail_start):
        """Internal: log-binned size density with two-regime fit."""
        shares = shares[np.isfinite(shares) & (shares > 0)]
        bins = np.logspace(np.log10(shares.min()),
                           np.log10(shares.max()), n_bins)
        hist, edges = np.histogram(shares, bins=bins, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        m = hist > 0
        centers, hist = centers[m], hist[m]

        logx, logy = np.log10(centers), np.log10(hist)

        mid_m = (centers > mid_range[0]) & (centers < mid_range[1])
        tail_m = centers > tail_start

        flat_slope, flat_int, *_ = linregress(logx[mid_m], logy[mid_m])
        tail_slope, tail_int, *_ = linregress(logx[tail_m], logy[tail_m])

        plt.figure(figsize=(7, 5))
        plt.loglog(centers, hist, "o", label="Empirical")

        xf = np.linspace(logx[mid_m].min(), logx[mid_m].max(), 100)
        plt.loglog(10 ** xf, 10 ** (flat_slope * xf + flat_int), "--",
                   label=f"Mid slope = {flat_slope:.2f}")
        xf = np.linspace(logx[tail_m].min(), logx[tail_m].max(), 100)
        plt.loglog(10 ** xf, 10 ** (tail_slope * xf + tail_int), "--",
                   label=f"Tail slope = {tail_slope:.2f}")
        plt.xlabel("Size"); plt.ylabel("Density")
        plt.title(f"{title} Size Density"); plt.legend()
        plt.show()

        print(f"Mid slope  ≈ {flat_slope:.3f}")
        print(f"Tail slope ≈ {tail_slope:.3f}  (α ≈ {-tail_slope:.3f})")

    def lo_size_distribution(self, n_log_bins=35):
        """Limit-order size density with two-regime fit."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT volume FROM orders "
            "WHERE event_type='LO' AND volume IS NOT NULL AND volume > 0"
            + self._day_clause()
        )
        shares = np.array([r[0] for r in cur.fetchall()], dtype=np.float64)
        conn.close()
        self._plot_size_density(shares, "Limit Order", n_log_bins,
                                mid_range=(2, 1000), tail_start=800)

    def mo_size_distribution(self, n_log_bins=35):
        """Market-order size density with two-regime fit."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT mo_volume FROM mo_orders "
            "WHERE mo_volume IS NOT NULL AND mo_volume > 0"
        )
        shares = np.array([r[0] for r in cur.fetchall()], dtype=np.float64)
        conn.close()
        self._plot_size_density(shares, "Market Order", n_log_bins,
                                mid_range=(2, 200), tail_start=200)

    def conditional_size_vs_distance(self, num_bins=80):
        """Mean order size vs distance from best price."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT ticks_from_best, volume FROM orders "
            "WHERE event_type='LO' AND volume > 0 "
            "  AND volume IS NOT NULL "
            "  AND ticks_from_best IS NOT NULL "
            "  AND ticks_from_best > 0"
            + self._day_clause()
        )
        rows = cur.fetchall()
        conn.close()

        delta = np.array([r[0] for r in rows], dtype=np.float64)
        vol = np.array([r[1] for r in rows], dtype=np.float64)

        bins = np.logspace(np.log10(delta.min()),
                           np.log10(delta.max()), num_bins)
        bi = np.clip(np.searchsorted(bins, delta, side="right") - 1,
                     0, len(bins) - 2)
        mean_v = np.empty(len(bins) - 1)
        cnt = np.empty(len(bins) - 1, dtype=int)
        for b in range(len(bins) - 1):
            m = bi == b
            cnt[b] = m.sum()
            if cnt[b] > 0:
                mean_v[b] = vol[m].mean()
            else:
                mean_v[b] = np.nan
        keep = cnt > 100
        centers = np.sqrt(bins[:-1] * bins[1:])[keep]
        mean_vol = mean_v[keep]
        m = mean_vol > 0

        plt.figure(figsize=(8, 6))
        plt.loglog(centers[m], mean_vol[m], "o", label="Binned mean")
        plt.xlabel("Distance from best (ticks)")
        plt.ylabel("Mean order size")
        plt.title("Conditional Mean Size vs Distance")
        plt.legend(); plt.tight_layout(); plt.show()

    # --- MO Size Models ---
    def _load_mo_size_depth(self):
        """Load (mo_volume, opp_depth_L0) as numpy arrays via cursor."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT mo_volume, opp_depth_L0 FROM mo_orders "
            "WHERE mo_volume IS NOT NULL AND mo_volume > 0 "
            "  AND opp_depth_L0 IS NOT NULL AND opp_depth_L0 > 0"
        )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return None, None
        size = np.array([r[0] for r in rows], dtype=np.float64)
        depth = np.array([r[1] for r in rows], dtype=np.float64)
        return size, depth

    _CUM_DEPTH_EXPR = " + ".join(
        [f"COALESCE(opp_depth_L{i}, 0)" for i in range(10)]
    )

    def _load_mo_cum_depth(self, extra_cols=""):
        """Load MO data with cumulative depth L0-L9 via cursor.

        Returns dict with numpy arrays: 'mo_volume', 'opp_depth_L0',
        'cum_depth', plus any extra columns requested.
        """
        sel = f"mo_volume, opp_depth_L0, ({self._CUM_DEPTH_EXPR}) AS cum_depth"
        if extra_cols:
            sel += f", {extra_cols}"
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            f"SELECT {sel} FROM mo_orders "
            f"WHERE mo_volume IS NOT NULL AND mo_volume > 0 "
            f"  AND opp_depth_L0 IS NOT NULL AND opp_depth_L0 > 0"
        )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return None
        n_base = 3
        result = {
            'mo_volume': np.array([r[0] for r in rows], dtype=np.float64),
            'opp_depth_L0': np.array([r[1] for r in rows], dtype=np.float64),
            'cum_depth': np.array([r[2] for r in rows], dtype=np.float64),
        }
        if extra_cols:
            for i, col in enumerate(extra_cols.split(",")):
                col = col.strip()
                result[col] = np.array(
                    [r[n_base + i] for r in rows], dtype=np.float64
                )
        mask = result['cum_depth'] > 0
        for k in result:
            result[k] = result[k][mask]
        return result

    def mo_size_vs_depth(self, num_bins=50):
        """MO size vs opposite L0 depth — power-law fit."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        size, depth = self._load_mo_size_depth()
        if size is None:
            print("No valid MO data."); return
        bins = np.logspace(np.log10(depth.min()),
                           np.log10(depth.max()), num_bins)
        bi = np.clip(np.searchsorted(bins, depth, side="right") - 1,
                     0, len(bins) - 2)
        mean_vol = np.empty(len(bins) - 1)
        median_vol = np.empty(len(bins) - 1)
        cnt = np.empty(len(bins) - 1, dtype=int)
        for b in range(len(bins) - 1):
            m = bi == b
            cnt[b] = m.sum()
            if cnt[b] > 0:
                mean_vol[b] = size[m].mean()
                median_vol[b] = np.median(size[m])
            else:
                mean_vol[b] = median_vol[b] = np.nan
        keep = cnt >= 30
        centers = np.sqrt(bins[:-1] * bins[1:])[keep]
        mean_vol = mean_vol[keep]
        median_vol = median_vol[keep]

        logx = np.log10(centers)
        logy = np.log10(mean_vol)
        slope, intercept, r_val, *_ = linregress(logx, logy)

        plt.figure(figsize=(8, 6))
        plt.loglog(centers, mean_vol, "o-", label="Mean", alpha=0.8)
        plt.loglog(centers, median_vol, "s-", label="Median", alpha=0.5)
        xf = np.linspace(logx.min(), logx.max(), 100)
        plt.loglog(10 ** xf, 10 ** (slope * xf + intercept), "--",
                   color="red", lw=2,
                   label=f"Fit: slope = {slope:.3f}")
        plt.xlabel("Opposite Depth L0 (shares)")
        plt.ylabel("Order Size (shares)")
        plt.title("MO Size vs Opposite Depth")
        plt.legend(); plt.tight_layout(); plt.show()

        print(f"Power-law slope: {slope:.3f}  (R² = {r_val**2:.3f})")
        return slope

    def mo_depth_ratio(self):
        """Distribution of MO size / opposite L0 depth."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        size, depth = self._load_mo_size_depth()
        if size is None:
            print("No valid MO data."); return

        ratio = size / depth
        ratio_pos = ratio[ratio > 0]

        bins_log = np.logspace(np.log10(ratio_pos.min()),
                               np.log10(ratio_pos.max()), 60)
        hist, edges = np.histogram(ratio_pos, bins=bins_log, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        m = hist > 0

        plt.figure(figsize=(10, 6))
        plt.loglog(centers[m], hist[m], "o", alpha=0.7)
        plt.xlabel("MO Size / Opposite L0 Depth")
        plt.ylabel("Density")
        plt.title("MO/Depth Ratio Distribution (log-log)")
        plt.tight_layout(); plt.show()

        print(f"Mean ratio: {ratio.mean():.3f},  "
              f"Median: {np.median(ratio):.3f}")
        print(f"P(r < 1): {(ratio < 1).mean():.1%},  "
              f"P(r ≥ 1): {(ratio >= 1).mean():.1%}")

    def mo_size_model(self, beta=0.295):
        """
        Calibrate parametric MO-size model  size = D^β · ε  (L0 depth).

        Returns (mu, sigma) of log(ε).
        """
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return None, None
        size, depth = self._load_mo_size_depth()
        if size is None:
            print("No valid MO data."); return None, None

        eps = size / (depth ** beta)
        log_eps = np.log(eps)
        mu_ln, sigma_ln = log_eps.mean(), log_eps.std()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1 — log(ε) histogram
        ax = axes[0]
        ax.hist(log_eps, bins=100, density=True, alpha=0.6,
                color="steelblue", label="Empirical log(ε)")
        xg = np.linspace(log_eps.min(), log_eps.max(), 300)
        ax.plot(xg, norm.pdf(xg, mu_ln, sigma_ln), "r-", lw=2,
                label=f"Normal(μ={mu_ln:.2f}, σ={sigma_ln:.2f})")
        ax.set_xlabel("log(ε)"); ax.set_ylabel("Density")
        ax.set_title("Distribution of log(ε)"); ax.legend()

        # 2 — ε log-log density
        ax = axes[1]
        eps_pos = eps[eps > 0]
        bl = np.logspace(np.log10(eps_pos.min()),
                         np.log10(eps_pos.max()), 60)
        h, e = np.histogram(eps_pos, bins=bl, density=True)
        c = np.sqrt(e[:-1] * e[1:]); m = h > 0
        ax.loglog(c[m], h[m], "o", alpha=0.7, color="steelblue")
        eg = np.logspace(np.log10(eps_pos.min()),
                         np.log10(eps_pos.max()), 300)
        ax.loglog(eg, lognorm.pdf(eg, s=sigma_ln, scale=np.exp(mu_ln)),
                  "r-", lw=2, label="Log-normal fit")
        ax.set_xlabel("ε = size / D^β"); ax.set_ylabel("Density")
        ax.set_title("Residual ε (log-log)"); ax.legend()

        # 3 — QQ-plot
        ax = axes[2]
        sle = np.sort(log_eps)
        n = len(sle); step = max(1, n // 2000)
        idx = np.arange(0, n, step)
        th = norm.ppf((idx + 0.5) / n, mu_ln, sigma_ln)
        ax.plot(th, sle[idx], ".", alpha=0.3, ms=2, color="steelblue")
        qr = [th.min(), th.max()]
        ax.plot(qr, qr, "r-", lw=2, label="Perfect fit")
        ax.set_xlabel("Normal quantiles")
        ax.set_ylabel("Empirical quantiles log(ε)")
        ax.set_title("QQ-plot"); ax.legend()

        plt.tight_layout(); plt.show()

        # Right-tail power-law fit
        eps_upper = eps_pos[eps_pos > np.median(eps_pos)]
        lbu = np.logspace(np.log10(eps_upper.min()),
                          np.log10(eps_upper.max()), 40)
        hu, eu = np.histogram(eps_upper, bins=lbu, density=True)
        cu = np.sqrt(eu[:-1] * eu[1:]); mu_h = hu > 0
        slope_t, _, r_t, *_ = linregress(
            np.log10(cu[mu_h]), np.log10(hu[mu_h])
        )
        print(f"β = {beta}")
        print(f"μ = {mu_ln:.4f},  σ = {sigma_ln:.4f}")
        print(f"Right-tail slope = {slope_t:.3f}  "
              f"(α ≈ {-slope_t - 1:.3f},  R² = {r_t**2:.3f})")
        print(f"\nRecipe: size = round(D^{beta} · ε),  "
              f"ε ~ LogNormal(μ, σ)")
        return mu_ln, sigma_ln

    def mo_cumulative_depth_model(self, beta=0.295):
        """
        Re-calibrate ε with cumulative depth L0-L9 (β fixed).

        Returns (mu, sigma) of log(ε).
        """
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return None, None
        data = self._load_mo_cum_depth()
        if data is None:
            print("No valid MO data."); return None, None

        size = data['mo_volume']
        d_cum = data['cum_depth']
        d_l0 = data['opp_depth_L0']

        eps_cum = size / (d_cum ** beta)
        log_eps = np.log(eps_cum)
        mu, sigma = log_eps.mean(), log_eps.std()

        # For comparison
        log_eps_old = np.log(size / (d_l0 ** beta))
        mu_old, sigma_old = log_eps_old.mean(), log_eps_old.std()

        print(f"β = {beta}  (FIXED)\n")
        print(f"{'':15s}  {'L0 only':>12s}  {'Cum L0–L9':>12s}")
        print(f"{'μ (log ε)':15s}  {mu_old:12.4f}  {mu:12.4f}")
        print(f"{'σ (log ε)':15s}  {sigma_old:12.4f}  {sigma:12.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.hist(log_eps, bins=100, density=True, alpha=0.6,
                color="steelblue")
        xg = np.linspace(log_eps.min(), log_eps.max(), 300)
        ax.plot(xg, norm.pdf(xg, mu, sigma), "r-", lw=2)
        ax.set_xlabel("log(ε)")
        ax.set_title(f"log(ε) — cum depth, β = {beta}")

        ax = axes[1]
        eps_pos = eps_cum[eps_cum > 0]
        bl = np.logspace(np.log10(eps_pos.min()),
                         np.log10(eps_pos.max()), 60)
        h, e = np.histogram(eps_pos, bins=bl, density=True)
        c = np.sqrt(e[:-1] * e[1:]); m = h > 0
        ax.loglog(c[m], h[m], "o", alpha=0.7, color="steelblue")
        eg = np.logspace(np.log10(eps_pos.min()),
                         np.log10(eps_pos.max()), 300)
        ax.loglog(eg, lognorm.pdf(eg, s=sigma, scale=np.exp(mu)),
                  "r-", lw=2)
        ax.set_xlabel("ε"); ax.set_title("Residual ε (log-log)")

        ax = axes[2]
        s = np.sort(log_eps)
        n = len(s); step = max(1, n // 2000)
        idx = np.arange(0, n, step)
        th = norm.ppf((idx + 0.5) / n, mu, sigma)
        ax.plot(th, s[idx], ".", alpha=0.3, ms=2, color="steelblue")
        qr = [th.min(), th.max()]
        ax.plot(qr, qr, "r-", lw=2)
        ax.set_xlabel("Normal quantiles"); ax.set_title("QQ-plot")

        plt.tight_layout(); plt.show()

        print(f"\nRecipe: D = sum(opp_depth L0..L9)")
        print(f"  log(ε) ~ Normal(μ={mu:.4f}, σ={sigma:.4f})")
        print(f"  size = max(1, round(D^{beta} · ε))")
        return mu, sigma

    def mo_ratio_cdfs(self, n_q=5, n_pts=1000, save_path=None):
        """
        Build empirical MO-size / cumulative-depth ratio CDFs
        per depth quintile.
        """
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return None, None
        data = self._load_mo_cum_depth()
        if data is None:
            print("No valid MO data."); return None, None

        mo_vol = data['mo_volume']
        cum_depth = data['cum_depth']
        all_r = mo_vol / cum_depth

        print(f"MOs with valid cumulative depth: {len(mo_vol):,}")

        quantiles = np.linspace(0, 1, n_q + 1)[1:-1]
        depth_bounds = np.quantile(cum_depth, quantiles)
        qi_arr = np.searchsorted(depth_bounds, cum_depth).astype(int)

        q_pts = np.linspace(1 / n_pts, 1 - 1 / n_pts, n_pts)
        r_quintiles = {}
        for qi in range(n_q):
            ratios = all_r[qi_arr == qi]
            r_quintiles[qi] = np.quantile(ratios, q_pts)
            print(f"  Q{qi}: n={len(ratios):>7,}  "
                  f"median={np.median(ratios):.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for qi in range(n_q):
            axes[0].plot(r_quintiles[qi], q_pts, label=f"Q{qi}")
            axes[1].semilogx(r_quintiles[qi], q_pts, label=f"Q{qi}")
        axes[0].set_xlim(0, 5)
        axes[0].set_xlabel("Ratio r"); axes[0].set_ylabel("CDF")
        axes[0].set_title("Empirical ratio CDF by depth quintile")
        axes[0].legend()
        axes[1].set_xlabel("Ratio r (log)"); axes[1].set_ylabel("CDF")
        axes[1].set_title("Ratio CDF (log scale)"); axes[1].legend()
        axes[2].hist(all_r[all_r < 5], bins=200, density=True,
                     alpha=0.7, color="steelblue")
        axes[2].axvline(1.0, color="red", ls="--", lw=2,
                        label="r = 1 (full sweep)")
        axes[2].set_xlabel("Ratio r"); axes[2].set_ylabel("Density")
        axes[2].set_title("Liquidity consumption ratio")
        axes[2].legend()

        plt.tight_layout(); plt.show()

        if save_path:
            save_dict = {"q_pts": q_pts, "depth_bounds": depth_bounds}
            for qi in range(n_q):
                save_dict[f"r_q{qi}"] = r_quintiles[qi]
            np.savez_compressed(save_path, **save_dict)
            print(f"\nSaved: {Path(save_path).name}")

        return r_quintiles, depth_bounds

    # --- Price Impact by Depth ---
    def price_impact_by_depth(self, n_quartiles=4, max_tw_plot=15):
        """Ticks-walked distribution by opposite-side depth quartile."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        data = self._load_mo_cum_depth(extra_cols="ticks_walked")
        if data is None:
            print("No valid MO data."); return

        tw_all = data['ticks_walked']
        valid = np.isfinite(tw_all)
        tw_all = tw_all[valid].astype(int)
        cum_depth = data['cum_depth'][valid]

        qb = np.quantile(cum_depth, np.linspace(0, 1, n_quartiles + 1)[1:-1])
        dq = np.searchsorted(qb, cum_depth).astype(int)

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        fig, axes = plt.subplots(n_quartiles, 2,
                                 figsize=(14, 4 * n_quartiles))

        for qi in range(n_quartiles):
            tw = tw_all[dq == qi]
            n = len(tw)
            if qi == 0:
                lo = 0
            else:
                lo = qb[qi - 1]
            if qi < n_quartiles - 1:
                hi = qb[qi]
            else:
                hi = cum_depth.max()
            label = (f"Q{qi + 1}: depth ∈ [{lo:,.0f}, {hi:,.0f}]  "
                     f"(n={n:,})")

            ax = axes[qi, 0]
            cnts = np.bincount(tw, minlength=max_tw_plot + 1
                               )[:max_tw_plot + 1]
            ax.bar(range(max_tw_plot + 1), cnts / n,
                   color=colors[qi % 4], edgecolor="white")
            ax.set_ylabel("Fraction"); ax.set_title(label)
            if qi == n_quartiles - 1:
                ax.set_xlabel("Ticks walked")

            ax = axes[qi, 1]
            max_k = min(int(tw.max()), 40)
            k_vals = np.arange(0, max_k + 1)
            survival = np.array([(tw >= k).mean() for k in k_vals])
            ax.semilogy(k_vals, survival, "o-",
                        color=colors[qi % 4], ms=3)
            ax.set_ylabel("P(tw ≥ k)"); ax.set_title(label)
            ax.grid(True, alpha=0.3)
            if qi == n_quartiles - 1:
                ax.set_xlabel("k (ticks)")

        plt.tight_layout(); plt.show()

        # Summary table
        print(f"\n{'Quartile':>10s}  {'n':>8s}  {'mean tw':>8s}  "
              f"{'P(>0)':>7s}  {'P(≥5)':>8s}")
        for qi in range(n_quartiles):
            tw_qi = tw_all[dq == qi]
            n_qi = len(tw_qi)
            print(f"{'Q' + str(qi + 1):>10s}  {n_qi:>8,}  "
                  f"{tw_qi.mean():>8.3f}  "
                  f"{(tw_qi > 0).mean():>7.1%}  "
                  f"{(tw_qi >= 5).mean():>8.2%}")

    def ticks_walked_cdfs(self, save_path=None):
        """Compute (and optionally save) ticks_walked CDFs per quartile."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        data = self._load_mo_cum_depth(extra_cols="ticks_walked")
        if data is None:
            print("No valid MO data."); return

        tw_all = data['ticks_walked']
        valid = np.isfinite(tw_all)
        tw_all = tw_all[valid].astype(int)
        cum_depth = data['cum_depth'][valid]

        qb = np.quantile(cum_depth, [0.25, 0.5, 0.75])
        dq = np.searchsorted(qb, cum_depth).astype(int)

        max_k = int(tw_all.max())
        save_dict = {
            "depth_quartile_bounds": qb,
            "max_k": np.array([max_k]),
        }

        for qi in range(4):
            tw = tw_all[dq == qi]
            pmf = np.bincount(tw, minlength=max_k + 1).astype(float)
            pmf /= pmf.sum()
            cdf = np.cumsum(pmf)
            save_dict[f"tw_cdf_q{qi}"] = cdf
            print(f"Q{qi}: P(0)={pmf[0]:.4f},  "
                  f"P(≥5)={1 - cdf[4]:.6f}")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_path, **save_dict)
            print(f"\nSaved: {Path(save_path).name}")

    # --- Time-series overview plots ---
    def _load_bbo_series(self, n_events=None, offset=0,
                         max_points=50_000):
        """Load BBO time series with optional window and subsampling.

        Parameters
        ----------
        n_events : int or None
            Number of events to include (``None`` = all).
        offset : int
            Skip this many events from the start.
        max_points : int
            Subsample within the window so plots stay responsive.
        """
        conn = self._conn()
        day_sql = self._day_clause()
        if n_events:
            limit_sql = f" LIMIT {n_events}"
        else:
            limit_sql = ""
        if offset:
            offset_sql = f" OFFSET {offset}"
        else:
            offset_sql = ""
        try:
            cur = conn.cursor()
            if self.has_bbo:
                inner = (
                    "SELECT rowid AS _rid, timestamp, best_bid, best_ask, "
                    "mid_price FROM bbo ORDER BY timestamp"
                    + limit_sql + offset_sql
                )
                cnt = cur.execute(
                    f"SELECT COUNT(*) FROM ({inner})"
                ).fetchone()[0]
                step = max(1, cnt // max_points)
                df = pd.read_sql(
                    f"SELECT * FROM ({inner}) WHERE _rid % {step} = 0",
                    conn,
                )
            else:
                inner = (
                    "SELECT rowid AS _rid, timestamp, best_bid, best_ask, "
                    "mid_price FROM orders "
                    "WHERE best_bid>0 AND best_ask>0" + day_sql
                    + " ORDER BY timestamp" + limit_sql + offset_sql
                )
                cnt = cur.execute(
                    f"SELECT COUNT(*) FROM ({inner})"
                ).fetchone()[0]
                step = max(1, cnt // max_points)
                df = pd.read_sql(
                    f"SELECT * FROM ({inner}) WHERE _rid % {step} = 0",
                    conn,
                )
            if len(df) == 0:
                return None

            # Convert string timestamps to float seconds
            ts = df["timestamp"].values
            if ts.dtype.kind in ("U", "O"):
                ts_dt = pd.to_datetime(df["timestamp"], utc=True)
                df["t_sec"] = (ts_dt - ts_dt.iloc[0]).dt.total_seconds().values
            else:
                df["t_sec"] = ts.astype(float)
            return df
        finally:
            conn.close()

    @staticmethod
    def _auto_time_label(t_sec):
        """Choose minutes or hours scale for time axis."""
        if len(t_sec):
            span = t_sec.max() - t_sec.min()
        else:
            span = 0
        if span > 7200:
            return t_sec / 3600.0, "Time (hours)", 3600.0
        elif span > 120:
            return t_sec / 60.0, "Time (minutes)", 60.0
        return t_sec, "Time (seconds)", 1.0

    @staticmethod
    def _collapse_trading_time(ts_dt):
        """Collapse multi-day datetimes into continuous trading seconds.

        Removes overnight / weekend gaps by stacking each trading day's
        elapsed seconds end-to-end.
        """
        if isinstance(ts_dt, pd.DatetimeIndex):
            ts_dt = pd.Series(ts_dt)
        dates = ts_dt.dt.date
        t_sec = np.zeros(len(ts_dt), dtype=np.float64)
        offset = 0.0
        for d in sorted(dates.unique()):
            mask = (dates == d).values
            day_ts = ts_dt[mask]
            day_secs = (day_ts - day_ts.iloc[0]).dt.total_seconds().values
            t_sec[mask] = day_secs + offset
            if len(day_secs) > 0:
                offset += day_secs[-1]
        return t_sec

    def plot_bbo_series(self, n_events=None, offset=0):
        """Plot best bid and best ask over time.

        Parameters
        ----------
        n_events : int or None
            Number of events to include (``None`` = all).
        offset : int
            Skip this many events from the start.
        """
        df = self._load_bbo_series(n_events=n_events, offset=offset)
        if df is None:
            print("No BBO data available."); return

        t, xlabel, _ = self._auto_time_label(df["t_sec"].values)

        plt.figure(figsize=(12, 5))
        plt.plot(t, df["best_bid"].values, lw=0.4, alpha=0.7,
                 label="Best bid")
        plt.plot(t, df["best_ask"].values, lw=0.4, alpha=0.7,
                 label="Best ask")
        plt.xlabel(xlabel)
        plt.ylabel("Price")
        plt.title("BBO over time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    def plot_spread(self, n_events=None, offset=0):
        """Plot spread over time + spread distribution.

        Parameters
        ----------
        n_events : int or None
            Number of events to include (``None`` = all).
        offset : int
            Skip this many events from the start.
        """
        df = self._load_bbo_series(n_events=n_events, offset=offset)
        if df is None:
            print("No BBO data available."); return

        spread = (df["best_ask"] - df["best_bid"]).values / self.tick_size
        t, xlabel, _ = self._auto_time_label(df["t_sec"].values)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4),
                                 gridspec_kw={"width_ratios": [3, 1]})

        ax = axes[0]
        ax.plot(t, spread, lw=0.4, alpha=0.7, color="tab:blue")
        ax.axhline(spread.mean(), ls="--", color="red", lw=1,
                   label=f"mean = {spread.mean():.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Spread (ticks)")
        ax.set_title("Spread over time")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        unique, counts = np.unique(np.round(spread).astype(int),
                                   return_counts=True)
        ax.barh(unique, counts / counts.sum(), height=0.8,
                color="tab:blue", alpha=0.7)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Spread (ticks)")
        ax.set_title("Spread distribution")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout(); plt.show()

        print(f"Spread stats:  mean={spread.mean():.2f}  "
              f"median={np.median(spread):.0f}  "
              f"min={spread.min():.0f}  max={spread.max():.0f}")

    def plot_depth(self, n_events=None, offset=0):
        """Plot total bid/ask depth over time.

        Parameters
        ----------
        n_events : int or None
            Number of events to include (``None`` = all).
        offset : int
            Skip this many events from the start.
        """
        conn = self._conn()
        day_sql = self._day_clause()
        if n_events:
            limit_sql = f" LIMIT {n_events}"
        else:
            limit_sql = ""
        if offset:
            offset_sql = f" OFFSET {offset}"
        else:
            offset_sql = ""
        try:
            inner = (
                "SELECT rowid AS _rid, timestamp, "
                "total_bid_depth, total_ask_depth "
                "FROM orders WHERE total_bid_depth IS NOT NULL"
                + day_sql + " ORDER BY timestamp"
                + limit_sql + offset_sql
            )
            cnt = conn.cursor().execute(
                f"SELECT COUNT(*) FROM ({inner})"
            ).fetchone()[0]
            step = max(1, cnt // 50_000)
            df = pd.read_sql(
                f"SELECT * FROM ({inner}) WHERE _rid % {step} = 0",
                conn,
            )
        finally:
            conn.close()

        if len(df) == 0:
            print("No depth data available."); return

        ts = df["timestamp"].values
        if ts.dtype.kind in ("U", "O"):
            ts_dt = pd.to_datetime(df["timestamp"], utc=True)
            if self.has_day:
                t_sec = self._collapse_trading_time(ts_dt)
            else:
                t_sec = (ts_dt - ts_dt.iloc[0]).dt.total_seconds().values
        else:
            t_sec = ts.astype(float)

        t, xlabel, _ = self._auto_time_label(t_sec)
        if self.has_day:
            xlabel = xlabel.replace("Time", "Trading time")

        plt.figure(figsize=(12, 5))
        plt.plot(t, df["total_bid_depth"].values, lw=0.4, alpha=0.7,
                 label="Bid depth")
        plt.plot(t, df["total_ask_depth"].values, lw=0.4, alpha=0.7,
                 label="Ask depth")
        plt.xlabel(xlabel)
        plt.ylabel("Total depth (shares)")
        plt.title("Order book depth over time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    _INTENSITY_COLS = (
        "mo_bid", "mo_ask", "lo_bid", "lo_ask", "cxl_bid", "cxl_ask",
    )

    def _load_intensities_series(self, n_events=None, offset=0,
                                 max_points=50_000):
        """Load Hawkes intensity time series from the ``intensities`` table."""
        if not self.has_intensities:
            return None

        conn = self._conn()
        if n_events:
            limit_sql = f" LIMIT {n_events}"
        else:
            limit_sql = ""
        if offset:
            offset_sql = f" OFFSET {offset}"
        else:
            offset_sql = ""
        try:
            inner = (
                "SELECT rowid AS _rid, timestamp, "
                + ", ".join(self._INTENSITY_COLS)
                + " FROM intensities ORDER BY timestamp"
                + limit_sql + offset_sql
            )
            cnt = conn.cursor().execute(
                f"SELECT COUNT(*) FROM ({inner})"
            ).fetchone()[0]
            if cnt == 0:
                return None
            step = max(1, cnt // max_points)
            df = pd.read_sql(
                f"SELECT * FROM ({inner}) WHERE _rid % {step} = 0",
                conn,
            )
        finally:
            conn.close()

        ts = df["timestamp"].values
        if ts.dtype.kind in ("U", "O"):
            ts_dt = pd.to_datetime(df["timestamp"], utc=True)
            df["t_sec"] = (ts_dt - ts_dt.iloc[0]).dt.total_seconds().values
        else:
            df["t_sec"] = ts.astype(float)
        return df

    def _replay_intensities(self, n_events=None, offset=0,
                            max_points=50_000, hawkes_filter=None):
        """Reconstruct λ(t) by replaying orders + MOs through a Hawkes filter."""
        from .hawkes_filter import classify_event, classify_mo

        hf = hawkes_filter.clone()
        hf.reset(t0=0.0)

        conn = self._conn()
        day_sql = self._day_clause()
        if n_events:
            limit_sql = f" LIMIT {n_events}"
        else:
            limit_sql = ""
        if offset:
            offset_sql = f" OFFSET {offset}"
        else:
            offset_sql = ""
        try:
            orders_df = pd.read_sql(
                "SELECT timestamp, event_type, side FROM orders "
                "WHERE 1=1" + day_sql
                + " ORDER BY timestamp" + limit_sql + offset_sql,
                conn,
            )
            if len(orders_df) == 0:
                return None

            ts = orders_df["timestamp"].values
            if ts.dtype.kind in ("U", "O"):
                o_times = pd.to_datetime(
                    orders_df["timestamp"], utc=True,
                ).astype("int64").values
                use_wall_clock = True
            elif self.has_day:
                o_times = orders_df["timestamp"].astype(np.int64).values
                use_wall_clock = True
            else:
                o_times = orders_df["timestamp"].astype(float).values
                use_wall_clock = False

            o_etype = orders_df["event_type"].values
            o_side = orders_df["side"].values

            if use_wall_clock:
                t_lo, t_hi = int(o_times[0]), int(o_times[-1])
                if self.day:
                    mos_df = pd.read_sql(
                        "SELECT first_time_ns AS time_ns, side AS mo_side "
                        "FROM mo_orders WHERE day = ? "
                        "AND first_time_ns >= ? AND first_time_ns <= ? "
                        "ORDER BY first_time_ns",
                        conn, params=(self.day, t_lo, t_hi),
                    )
                else:
                    mos_df = pd.read_sql(
                        "SELECT first_time_ns AS time_ns, side AS mo_side "
                        "FROM mo_orders "
                        "WHERE first_time_ns >= ? AND first_time_ns <= ? "
                        "ORDER BY first_time_ns",
                        conn, params=(t_lo, t_hi),
                    )
            else:
                t_min, t_max = float(o_times[0]), float(o_times[-1])
                mos_df = pd.read_sql(
                    "SELECT timestamp AS time_ns, side AS mo_side "
                    "FROM mo_orders WHERE timestamp BETWEEN ? AND ? "
                    "ORDER BY timestamp",
                    conn, params=(t_min, t_max),
                )
        finally:
            conn.close()

        if len(mos_df) > 0:
            m_times = mos_df["time_ns"].values.astype(np.int64 if use_wall_clock
                                                      else float)
            m_side = mos_df["mo_side"].values
            n_m = len(m_times)
        else:
            n_m = 0

        n_o = len(o_times)
        oi = mi = 0
        if use_wall_clock:
            t0 = int(o_times[0])
            if n_m > 0:
                t0 = min(t0, int(m_times[0]))
            int_max = np.iinfo(np.int64).max
            time_scale = 1e9
        else:
            t0 = 0.0
            int_max = float("inf")
            time_scale = 1.0

        total = n_o + n_m
        step = max(1, total // max_points)
        t_buf, lam_buf = [], []
        seen = 0

        while oi < n_o or mi < n_m:
            if oi < n_o:
                ot = o_times[oi]
            else:
                ot = int_max
            if mi < n_m:
                mt = m_times[mi]
            else:
                mt = int_max

            if ot <= mt:
                t = float(o_times[oi] - t0) / time_scale
                try:
                    hf.update(t, classify_event(o_etype[oi], o_side[oi]))
                except ValueError:
                    pass
                oi += 1
            else:
                t = float(m_times[mi] - t0) / time_scale
                try:
                    hf.update(t, classify_mo(m_side[mi]))
                except ValueError:
                    pass
                mi += 1

            if seen % step == 0:
                t_buf.append(t)
                lam_buf.append(hf.intensity(t))
            seen += 1

        if not t_buf:
            return None

        data = {"t_sec": np.asarray(t_buf, dtype=np.float64)}
        lam_arr = np.asarray(lam_buf, dtype=np.float64)
        for i, col in enumerate(self._INTENSITY_COLS):
            data[col] = lam_arr[:, i]
        return pd.DataFrame(data)

    def plot_intensities(self, n_events=None, offset=0, dims=None,
                         hawkes=None, max_points=50_000):
        """Plot Hawkes conditional intensities λ(t) over time.

        For simulation databases recorded with ``recording_mode='full'``,
        intensities are read directly from the ``intensities`` table (one
        row per simulated event).  Otherwise the event stream is replayed
        through an online :class:`~research_core.classes.hawkes_filter.HawkesFilter`
        (default: KGHM single-kernel multivariate).

        Parameters
        ----------
        n_events : int or None
            Number of order-flow events to include (``None`` = all).
        offset : int
            Skip this many events from the start (orders table window).
        dims : sequence of str or None
            Subset of intensity columns to plot.  Defaults to all six
            event types (``mo_bid`` … ``cxl_ask``).
        hawkes : None, bool, or HawkesFilter
            Filter used when replaying from ``orders`` / ``mo_orders``.
            Ignored when the database already contains an ``intensities``
            table.  ``True`` (default when replaying) uses the KGHM
            single-kernel default.
        max_points : int
            Subsample within the window so plots stay responsive.
        """
        if dims is None:
            dims = list(self._INTENSITY_COLS)
        else:
            unknown = set(dims) - set(self._INTENSITY_COLS)
            if unknown:
                raise ValueError(
                    f"Unknown intensity dims: {sorted(unknown)}; "
                    f"expected subset of {self._INTENSITY_COLS}"
                )

        if self.has_intensities:
            df = self._load_intensities_series(
                n_events=n_events, offset=offset, max_points=max_points,
            )
            source = "intensities table"
        else:
            if hawkes is False:
                print("No intensities table in this database.  Pass "
                      "hawkes=True (default) to replay via HawkesFilter, "
                      "or use a full-mode simulation DB.")
                return
            from .hawkes_filter import HawkesFilter, resolve_filter_factory

            if hawkes is None:
                factory = resolve_filter_factory(True)
            else:
                factory = resolve_filter_factory(hawkes)
            if factory is not None:
                hf = factory()
            else:
                hf = None
            if hf is None:
                print("Hawkes replay disabled (hawkes=False)."); return
            df = self._replay_intensities(
                n_events=n_events, offset=offset, max_points=max_points,
                hawkes_filter=hf,
            )
            source = "HawkesFilter replay"

        if df is None or len(df) == 0:
            print("No intensity data available."); return

        t, xlabel, _ = self._auto_time_label(df["t_sec"].values)
        if self.has_day and not self.has_intensities:
            xlabel = xlabel.replace("Time", "Trading time")

        colors = {
            "mo_bid": "tab:blue", "mo_ask": "tab:red",
            "lo_bid": "tab:cyan", "lo_ask": "tab:orange",
            "cxl_bid": "tab:green", "cxl_ask": "tab:purple",
        }
        labels = {
            "mo_bid": "MO bid", "mo_ask": "MO ask",
            "lo_bid": "LO bid", "lo_ask": "LO ask",
            "cxl_bid": "CXL bid", "cxl_ask": "CXL ask",
        }

        plt.figure(figsize=(12, 5))
        for col in dims:
            plt.plot(t, df[col].values, lw=0.5, alpha=0.8,
                     color=colors.get(col, None), label=labels[col])
        plt.xlabel(xlabel)
        plt.ylabel("Intensity λ(t)")
        plt.title(f"Hawkes intensities over time ({source})")
        plt.legend(ncol=3, fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    def plot_event_occurrences(self, n_events=None, offset=0):
        """Bar chart of event type counts.

        Parameters
        ----------
        n_events : int or None
            Number of events to include (``None`` = all).
        offset : int
            Skip this many events from the start.
        """
        conn = self._conn()
        day_sql = self._day_clause()
        if n_events is not None or offset:
            if n_events:
                limit_sql = f" LIMIT {n_events}"
            else:
                limit_sql = ""
            if offset:
                offset_sql = f" OFFSET {offset}"
            else:
                offset_sql = ""
            inner = (
                "SELECT event_type FROM orders"
                + (" WHERE 1=1" + day_sql if day_sql else "")
                + " ORDER BY timestamp" + limit_sql + offset_sql
            )
            df = pd.read_sql(
                f"SELECT event_type, COUNT(*) AS cnt "
                f"FROM ({inner}) GROUP BY event_type ORDER BY cnt DESC",
                conn,
            )
        else:
            df = pd.read_sql(
                "SELECT event_type, COUNT(*) AS cnt FROM orders"
                + (" WHERE 1=1" + day_sql if day_sql else "")
                + " GROUP BY event_type ORDER BY cnt DESC",
                conn,
            )
        conn.close()

        if len(df) == 0:
            print("No event data."); return

        plt.figure(figsize=(8, 5))
        plt.bar(df["event_type"], df["cnt"])
        plt.ylabel("Count")
        plt.title("Event type distribution")
        plt.xticks(rotation=45)
        plt.tight_layout(); plt.show()

        for _, row in df.iterrows():
            print(f"  {row['event_type']}: {int(row['cnt']):,}")

    def plot_candlestick(self, timeframe=60.0, ema_spans=None,
                         n_events=None, offset=0):
        """OHLC candlestick chart from fill/trade data.

        Parameters
        ----------
        timeframe : float
            Candle width in seconds (default 60 = one-minute candles).
        ema_spans : list[int] or None
            EMA spans in number of candles to overlay.
        n_events : int or None
            Number of fills to include (``None`` = all).
        offset : int
            Skip this many fills from the start.
        """
        conn = self._conn()
        day_sql = self._day_clause()
        if n_events:
            limit_sql = f" LIMIT {n_events}"
        else:
            limit_sql = ""
        if offset:
            offset_sql = f" OFFSET {offset}"
        else:
            offset_sql = ""
        try:
            if "fills" in self.tables:
                fill_cols = self._table_cols(conn, "fills")
                ts_col = ("time_ns" if "time_ns" in fill_cols
                          else "timestamp")
                df = pd.read_sql(
                    f"SELECT {ts_col} AS timestamp, price FROM fills"
                    + (" WHERE 1=1" + day_sql if day_sql else "")
                    + f" ORDER BY {ts_col}"
                    + limit_sql + offset_sql,
                    conn,
                )
            else:
                print("No fills table available."); return
        finally:
            conn.close()

        if len(df) < 2:
            print("Not enough trade data for candlestick."); return

        ts = df["timestamp"].values
        if ts.dtype.kind in ("U", "O"):
            ts_dt = pd.to_datetime(df["timestamp"], utc=True)
            if self.has_day:
                t_sec = self._collapse_trading_time(ts_dt)
            else:
                t_sec = (ts_dt - ts_dt.iloc[0]).dt.total_seconds().values
        elif ts.dtype.kind == "i":
            t_ns = ts.astype(np.float64)
            if self.has_day:
                ts_dt = pd.Series(pd.to_datetime(ts, unit="ns", utc=True))
                t_sec = self._collapse_trading_time(ts_dt)
            else:
                t_sec = (t_ns - t_ns[0]) / 1e9
        else:
            t_sec = ts.astype(float)

        prices = df["price"].values.astype(float)
        t_min, t_max = t_sec[0], t_sec[-1]
        edges = np.arange(t_min, t_max + timeframe, timeframe)
        n_candles = len(edges) - 1
        if n_candles < 2:
            print(f"Only {n_candles} candle(s) — try a smaller timeframe.")
            return

        opens  = np.empty(n_candles)
        highs  = np.empty(n_candles)
        lows   = np.empty(n_candles)
        closes = np.empty(n_candles)
        c_times = np.empty(n_candles)

        idx = 0
        for ci in range(n_candles):
            lo_t, hi_t = edges[ci], edges[ci + 1]
            c_times[ci] = lo_t
            start = idx
            while idx < len(t_sec) and t_sec[idx] < hi_t:
                idx += 1
            end = idx
            if start == end:
                if ci > 0:
                    val = closes[ci - 1]
                else:
                    val = prices[0]
                opens[ci] = highs[ci] = lows[ci] = closes[ci] = val
            else:
                seg = prices[start:end]
                if ci > 0:
                    opens[ci]  = closes[ci - 1]
                else:
                    opens[ci]  = seg[0]
                highs[ci]  = max(seg.max(), opens[ci])
                lows[ci]   = min(seg.min(), opens[ci])
                closes[ci] = seg[-1]

        c_t, xlabel, divisor = self._auto_time_label(c_times)
        candle_w = timeframe / divisor * 0.7

        ema_lines = {}
        if ema_spans:
            for span in ema_spans:
                alpha = 2.0 / (span + 1)
                ema = np.empty(n_candles)
                ema[0] = closes[0]
                for i in range(1, n_candles):
                    ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]
                ema_lines[span] = ema

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        up   = closes >= opens
        down = ~up
        col_up, col_down = "mediumseagreen", "tomato"

        ax.bar(c_t[up], closes[up] - opens[up], bottom=opens[up],
               width=candle_w, color=col_up, edgecolor=col_up, lw=0.5)
        ax.bar(c_t[down], opens[down] - closes[down], bottom=closes[down],
               width=candle_w, color=col_down, edgecolor=col_down, lw=0.5)
        ax.vlines(c_t[up], lows[up], highs[up], color=col_up, lw=0.6)
        ax.vlines(c_t[down], lows[down], highs[down], color=col_down, lw=0.6)

        ema_colors = ["darkorange", "dodgerblue", "purple", "cyan"]
        for i, (span, ema_v) in enumerate(ema_lines.items()):
            ax.plot(c_t, ema_v, lw=1.2,
                    color=ema_colors[i % len(ema_colors)],
                    label=f"EMA {span}")

        ax.set_ylabel("Price")
        tf_lbl = (f"{timeframe:.0f}s" if timeframe < 60 else
                  (f"{timeframe/60:.0f}min" if timeframe < 3600 else
                   f"{timeframe/3600:.1f}h"))
        ax.set_title(f"Candlestick chart  (timeframe={tf_lbl}, "
                     f"{n_candles} candles)")
        if ema_lines:
            ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        if self.has_day:
            xlabel = xlabel.replace("Time", "Trading time")
        ax.set_xlabel(xlabel)
        plt.tight_layout(); plt.show()

    # --- Post-fill markout (wall-clock) ---
    def _normalize_event_times(self, ts_raw):
        """Map raw DB timestamps to seconds on the ``_get_mid_prices`` axis."""
        if len(ts_raw) == 0:
            return np.array([], dtype=np.float64)
        if isinstance(ts_raw[0], str):
            ts_dt = pd.to_datetime(ts_raw, utc=True)
            return np.asarray(
                self._collapse_trading_time(pd.Series(ts_dt)), dtype=np.float64,
            )
        ts_num = np.asarray(ts_raw, dtype=np.float64)
        if self._numeric_ts_is_absolute_wall_clock(ts_num):
            unit = self._posix_unit_from_magnitude(ts_num)
            ts_dt = pd.to_datetime(ts_num, unit=unit, utc=True)
            return np.asarray(
                self._collapse_trading_time(pd.Series(ts_dt)), dtype=np.float64,
            )
        return ts_num.copy()

    def _load_mo_touch_events(self):
        """Load MO timestamps, sign (+1 buy / −1 sell), and touch mid."""
        conn = self._conn()
        day_sql = self._day_clause()
        try:
            if "mo_orders" not in self.tables:
                print("No mo_orders table."); return None
            mo_cols = self._table_cols(conn, "mo_orders")
            ts_col = ("first_time_ns" if "first_time_ns" in mo_cols
                      else "timestamp")
            cur = conn.cursor()
            cur.execute(
                f"SELECT {ts_col}, side, best_bid, best_ask "
                f"FROM mo_orders"
                + (" WHERE 1=1" + day_sql if day_sql else "")
                + f" ORDER BY {ts_col}"
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        if len(rows) < 100:
            print(f"Only {len(rows)} MOs — not enough."); return None

        t_sec = self._normalize_event_times([r[0] for r in rows])
        sign = np.where(np.array([r[1] for r in rows]) == "buy", 1.0, -1.0)
        mid0 = np.array([(r[2] + r[3]) / 2.0 for r in rows], dtype=np.float64)
        ok = np.isfinite(t_sec) & np.isfinite(mid0) & (mid0 > 0)
        if ok.sum() < 100:
            print("Not enough finite MO touch events."); return None
        return t_sec[ok], sign[ok], mid0[ok]

    @staticmethod
    def _locf_mids_at(ts, mids, t_query):
        """LOCF mid on *ts* at each time in *t_query*."""
        t_query = np.asarray(t_query, dtype=np.float64)
        idx = np.searchsorted(ts, t_query, side="right") - 1
        idx = np.clip(idx, 0, len(mids) - 1)
        return mids[idx]

    def post_fill_markout(self, tau_seconds=(1, 5, 10), plot=True):
        """Wall-clock post-fill markout via MO touch events.

        Uses aggressive MOs as a proxy for passive fills at the touch:
        sell MO → passive buy fill, buy MO → passive sell fill.

        Markout at horizon τ (seconds) is measured in ticks; **negative**
        means adverse selection for the passive side.

        Parameters
        ----------
        tau_seconds : sequence of float
            Horizons in seconds (default 1, 5, 10).
        plot : bool
            If True, bar chart of mean markout by side and τ.
        """
        bbo_ts, bbo_mids = self._get_mid_prices()
        mo = self._load_mo_touch_events()
        if bbo_ts is None or mo is None:
            return None
        t_sec, sign, mid0 = mo

        taus = [float(t) for t in tau_seconds if float(t) > 0]
        if not taus:
            print("No positive tau_seconds."); return None

        buy_mask = sign > 0   # passive sell filled by buy MO
        sell_mask = sign < 0  # passive buy filled by sell MO
        n_buy, n_sell = int(buy_mask.sum()), int(sell_mask.sum())

        rows = []
        buy_means, sell_means = [], []

        for tau in taus:
            mid_fut = self._locf_mids_at(bbo_ts, bbo_mids, t_sec + tau)
            # passive sell (buy MO): mid0 - mid_fut; passive buy: mid_fut - mid0
            mo_buy = (mid0[buy_mask] - mid_fut[buy_mask]) / self.tick_size
            mo_sell = (mid_fut[sell_mask] - mid0[sell_mask]) / self.tick_size
            buy_means.append(float(mo_buy.mean()) if len(mo_buy) else np.nan)
            sell_means.append(float(mo_sell.mean()) if len(mo_sell) else np.nan)
            rows.append(("passive sell (buy MO)", tau, n_buy, buy_means[-1]))
            rows.append(("passive buy (sell MO)", tau, n_sell, sell_means[-1]))

        if plot:
            x = np.arange(len(taus))
            w = 0.35
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(x - w / 2, buy_means, w, label="passive sell (buy MO)",
                   color="tab:orange", alpha=0.8)
            ax.bar(x + w / 2, sell_means, w, label="passive buy (sell MO)",
                   color="tab:blue", alpha=0.8)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{t:g} s" for t in taus])
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Mean markout (ticks)")
            ax.set_title("Post-fill markout (MO touch proxy)")
            ax.legend(fontsize=8)
            plt.tight_layout(); plt.show()

        print(f"MO touch events: {len(t_sec):,}  "
              f"(buy MO {n_buy:,}, sell MO {n_sell:,})")
        print(f"Tick size: {self.tick_size}")
        for label, tau, n, mean in rows:
            print(f"  {label:28s}  τ={tau:g}s  n={n:,}  "
                  f"mean={mean:+.4f} ticks")

        return {
            "tau_seconds": taus,
            "passive_sell_mean_ticks": buy_means,
            "passive_buy_mean_ticks": sell_means,
            "n_buy_mo": n_buy,
            "n_sell_mo": n_sell,
        }

    @staticmethod
    def markout_vector(markout_result):
        """Flatten ``post_fill_markout`` dict to [sell@τ…, buy@τ…]."""
        if markout_result is None:
            return None
        return np.array(
            list(markout_result["passive_sell_mean_ticks"])
            + list(markout_result["passive_buy_mean_ticks"]),
            dtype=float,
        )

    # --- Price-impact propagator ---
    def _load_propagator_data(self, max_horizon=1000):
        """Load MO-time mid-price series and signs for propagator plots.

        Returns
        -------
        ``(mid_mo, sign_all, day_labels, ticks_walked)`` or ``None``.

        * ``mid_mo[n]``  — ``(best_bid + best_ask) / 2`` from
          ``mo_orders`` row *n*.  Both simulation and real data use the
          same convention (the BBO recorded alongside each MO), so the
          bias cancels when computing ``mid_mo[n+τ] − mid_mo[n]``.
        * ``sign_all[n]`` — +1 for buy, −1 for sell.
        * ``day_labels[n]`` — day key for per-day computation (real
          data), or ``None`` for simulation.
        * ``ticks_walked[n]`` — number of price levels the MO consumed
          (0 = filled entirely at the best price).

        Notes
        -----
        The horizon τ counts **MOs**, not LO/CXL events.  This is the
        standard convention in the propagator literature (Bouchaud et al.
        2004).

        Cross-table ``searchsorted`` alignment is no longer needed
        because reference *and* future prices come from the same table
        with the same convention.  Per-day computation avoids overnight
        price-gap artefacts for real data.
        """
        conn = self._conn()
        day_sql = self._day_clause()
        try:
            if "mo_orders" not in self.tables:
                print("No mo_orders table."); return None

            mo_cols = self._table_cols(conn, "mo_orders")
            ts_col = ("first_time_ns" if "first_time_ns" in mo_cols
                      else "timestamp")
            has_day = "day" in mo_cols
            if has_day:
                day_sel = "day, "
            else:
                day_sel = ""

            cur = conn.cursor()
            cur.execute(
                f"SELECT {day_sel}{ts_col}, side, "
                f"best_bid, best_ask, ticks_walked "
                f"FROM mo_orders"
                + (" WHERE 1=1" + day_sql if day_sql else "")
                + f" ORDER BY {ts_col}"
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        if len(rows) < 100:
            print(f"Only {len(rows)} MOs — not enough."); return None

        if has_day:
            day_labels = np.array([r[0] for r in rows])
            mid_mo = np.array(
                [(r[3] + r[4]) / 2.0 for r in rows], dtype=np.float64
            )
            sign_all = np.where(
                np.array([r[2] for r in rows]) == "buy", 1.0, -1.0
            )
            tw = np.array([r[5] for r in rows], dtype=int)
        else:
            day_labels = None
            mid_mo = np.array(
                [(r[2] + r[3]) / 2.0 for r in rows], dtype=np.float64
            )
            sign_all = np.where(
                np.array([r[1] for r in rows]) == "buy", 1.0, -1.0
            )
            tw = np.array([r[4] for r in rows], dtype=int)

        # --- Validation: R(1) sanity check ---
        imm = sign_all[:-1] * np.diff(mid_mo)
        if day_labels is not None:
            same_day = day_labels[:-1] == day_labels[1:]
            imm = imm[same_day]

        n_pos = (imm > 0).sum()
        n_zero = (imm == 0).sum()
        n_neg = (imm < 0).sum()
        n_total = len(imm)
        frac_pos = n_pos / max(1, n_pos + n_neg)
        day_info = (f", {len(np.unique(day_labels))} days"
                    if day_labels is not None else "")
        print(f"Propagator data ({len(rows):,} MOs{day_info}):")
        print(f"  R(1) check: {n_pos:,} pos ({100*n_pos/n_total:.1f}%), "
              f"{n_zero:,} zero ({100*n_zero/n_total:.1f}%), "
              f"{n_neg:,} neg ({100*n_neg/n_total:.1f}%)")
        print(f"  Among price-moving: {frac_pos:.1%} positive "
              f"(expect >> 50%)")
        print(f"  Mean R(1): {np.mean(imm):.6f}")

        return mid_mo, sign_all, day_labels, tw

    # --- Helpers ---
    @staticmethod
    def _propagator_curve(mid_mo, sign, day_labels, horizons,
                          ref_shift=0, mask=None):
        """Mean & SE of signed return at each MO-time horizon (bps).

        Parameters
        ----------
        mid_mo : array
            Mid-price from ``mo_orders`` (one entry per MO).
        sign : array
            +1 for buy, -1 for sell.
        day_labels : array or None
            Day key per MO.  If not None, impact is computed per day
            and then averaged (avoids cross-day artefacts).
        horizons : array of int
            MO-horizon lags (tau values).
        ref_shift : int
            0 → raw impact:       E[ε_n (mid[n+τ] − mid[n]) / mid[n]]
            1 → adjusted impact:  E[ε_n (mid[n+τ] − mid[n+1]) / mid[n]]
        mask : bool array or None
            If given, only MOs where ``mask[n]`` is True are used as
            *starting points*, but the horizon τ still counts along the
            **full** MO timeline.  This ensures τ=1 always means "the
            next MO overall", not "the next MO in this regime".
        """
        if day_labels is not None:
            # --- Per-day computation ---
            days = np.unique(day_labels)
            means = np.full(len(horizons), np.nan)
            ses = np.zeros(len(horizons))
            for k, tau in enumerate(horizons):
                tau = int(tau)
                daily_means = []
                for d in days:
                    idx = np.where(day_labels == d)[0]
                    m = len(idx)
                    n_use = m - max(tau, ref_shift)
                    if n_use < 10:
                        continue
                    # Starting-point indices within this day
                    start = idx[:n_use]
                    if mask is not None:
                        start = start[mask[start]]
                    if len(start) < 5:
                        continue
                    s = sign[start]
                    ref = mid_mo[start + ref_shift]
                    fut = mid_mo[start + tau]
                    base = mid_mo[start]
                    r = s * (fut - ref) / base
                    daily_means.append(r.mean())
                if daily_means:
                    means[k] = np.mean(daily_means)
                    if len(daily_means) > 1:
                        ses[k] = (np.std(daily_means, ddof=1)
                                  / np.sqrt(len(daily_means)))
            return means * 1e4, ses * 1e4
        else:
            # --- Pooled (simulation — single run, no day boundaries) ---
            N = len(mid_mo)
            means = np.empty(len(horizons))
            ses = np.empty(len(horizons))
            for k, tau in enumerate(horizons):
                tau = int(tau)
                n_use = N - max(tau, ref_shift)
                if n_use < 10:
                    means[k] = np.nan; ses[k] = 0; continue
                # Starting-point indices
                start = np.arange(n_use)
                if mask is not None:
                    start = start[mask[:n_use]]
                if len(start) < 5:
                    means[k] = np.nan; ses[k] = 0; continue
                s = sign[start]
                ref = mid_mo[start + ref_shift]
                fut = mid_mo[start + tau]
                base = mid_mo[start]
                r = s * (fut - ref) / base
                means[k] = r.mean()
                ses[k] = r.std() / np.sqrt(len(start))
            return means * 1e4, ses * 1e4

    @staticmethod
    def _sign_autocorr(sign, horizons):
        """Order-sign autocorrelation C(τ) = E[ε_t · ε_{t+τ}]."""
        n = len(sign)
        acf = np.empty(len(horizons))
        for k, tau in enumerate(horizons):
            tau = int(tau)
            if tau >= n:
                acf[k] = 0.0
            else:
                acf[k] = np.mean(sign[:-tau] * sign[tau:])
        return acf

    # --- Plot methods ---
    def price_impact_propagator(self, max_horizon=100, n_points=80,
                                split_regimes=False):
        """Raw impact R(τ) = E[ε_n (mid[n+τ] − mid[n]) / mid[n]]  and
        normalised impact kernel G(τ) = R(τ) / C(τ).

        The horizon τ counts **MOs** (trade time), not LO/CXL events.
        For real data the curve is computed per day then averaged so that
        overnight price gaps do not contaminate the estimate.

        Parameters
        ----------
        max_horizon : int
            Maximum horizon in MOs (default 1000).
        n_points : int
            Number of log-spaced horizons to evaluate (default 80).
        split_regimes : bool
            If True, plot three curves: all MOs, MOs that walked ≥ 1
            tick, and MOs that filled at a single price level.
        """
        data = self._load_propagator_data(max_horizon)
        if data is None:
            return
        mid_mo, sign_all, day_labels, tw = data
        N = len(mid_mo)

        horizons = np.unique(
            np.geomspace(1, max_horizon, n_points).astype(int))

        # --- Raw impact R(τ) ---
        def _plot_one(ax, label, color, mask=None):
            m, se = self._propagator_curve(mid_mo, sign_all, day_labels,
                                           horizons, ref_shift=0,
                                           mask=mask)
            if mask is not None:
                n_mo = int(mask.sum())
            else:
                n_mo = N
            ax.plot(horizons, m, "o-", ms=3, lw=1.2, color=color,
                    label=f"{label} (n={n_mo:,})")
            ax.fill_between(horizons, m - se, m + se,
                            alpha=0.18, color=color)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Left: raw impact
        ax = axes[0]
        if split_regimes:
            walked = tw > 0
            _plot_one(ax, "All", "steelblue")
            _plot_one(ax, "Walked ticks", "tab:red", mask=walked)
            _plot_one(ax, "No tick walk", "tab:green", mask=~walked)
        else:
            _plot_one(ax, "All MOs", "steelblue")

        ax.axhline(0, ls="--", color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"Horizon $\tau$  (MOs)")
        ax.set_ylabel("Mean signed return  (bps)")
        ax.set_title(r"Raw impact  $R(\tau) = "
                     r"\mathbb{E}[\epsilon_n\,"
                     r"(m_{n+\tau} - m_n)/m_n]$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: normalised kernel G = R / C
        ax = axes[1]
        m_all, se_all = self._propagator_curve(
            mid_mo, sign_all, day_labels, horizons, ref_shift=0)
        acf = self._sign_autocorr(sign_all, horizons)
        acf_thresh = 0.01
        G = np.full_like(m_all, np.nan)
        acf_ok = np.abs(acf) > acf_thresh
        G[acf_ok] = m_all[acf_ok] / acf[acf_ok]

        ax.plot(horizons, m_all, "o-", ms=3, lw=1.2, color="steelblue",
                label=r"$R(\tau)$ raw")
        ax.plot(horizons[acf_ok], G[acf_ok], "s-", ms=3, lw=1.2,
                color="tab:orange",
                label=r"$G(\tau) = R/C$ normalised")
        ax.axhline(0, ls="--", color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"Horizon $\tau$  (MOs)")
        ax.set_ylabel("bps")
        ax.set_title(r"Normalised impact kernel  "
                     r"$G(\tau) = R(\tau)\,/\,C(\tau)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()

        # Summary
        walked = tw > 0
        print(f"MOs total: {N:,}")
        print(f"  walked ticks:    {walked.sum():,}")
        print(f"  no tick walk:    {(~walked).sum():,}")
        if acf_ok[0]:
            g1 = f"{G[0]:+.3f}"
        else:
            g1 = "n/a"
        if acf_ok[-1]:
            gl = f"{G[-1]:+.3f}"
        else:
            gl = "n/a"
        print(f"  R(1)  = {m_all[0]:+.3f} bps   "
              f"C(1)  = {acf[0]:.4f}   "
              f"G(1)  = {g1} bps")
        print(f"  R({horizons[-1]}) = {m_all[-1]:+.3f} bps   "
              f"C({horizons[-1]}) = {acf[-1]:.4f}   "
              f"G({horizons[-1]}) = {gl} bps")
        print(f"  |C| > {acf_thresh} for "
              f"{acf_ok.sum()}/{len(acf)} horizons")

    def adjusted_impact_propagator(self, max_horizon=100, n_points=80,
                                   split_regimes=False):
        """Adjusted impact: E[ε_n (mid[n+τ] − mid[n+1]) / mid[n]].

        Strips out the immediate (one-MO-step) impact to reveal whether
        the remaining price change is permanent or transient.

        Parameters
        ----------
        max_horizon : int
            Maximum horizon in MOs (default 1000).
        n_points : int
            Number of log-spaced horizons to evaluate (default 80).
        split_regimes : bool
            If True, split by whether the MO walked ≥ 1 tick.
        """
        data = self._load_propagator_data(max_horizon)
        if data is None:
            return
        mid_mo, sign_all, day_labels, tw = data
        N = len(mid_mo)

        horizons = np.unique(
            np.geomspace(1, max_horizon, n_points).astype(int))
        # τ ≥ 2 for adjusted (mid[n+1] is the reference)
        horizons = horizons[horizons >= 2]

        walked = tw > 0

        def _plot_one(ax, label, color, mask=None):
            m, se = self._propagator_curve(mid_mo, sign_all, day_labels,
                                           horizons, ref_shift=1,
                                           mask=mask)
            if mask is not None:
                n_mo = int(mask.sum())
            else:
                n_mo = N
            ax.plot(horizons, m, "o-", ms=3, lw=1.2, color=color,
                    label=f"{label} (n={n_mo:,})")
            ax.fill_between(horizons, m - se, m + se,
                            alpha=0.18, color=color)

        fig, ax = plt.subplots(figsize=(10, 5))

        if split_regimes:
            _plot_one(ax, "All", "steelblue")
            _plot_one(ax, "Walked ticks", "tab:red", mask=walked)
            _plot_one(ax, "No tick walk", "tab:green", mask=~walked)
        else:
            _plot_one(ax, "All MOs", "steelblue")

        ax.axhline(0, ls="--", color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"Horizon $\tau$  (MOs)")
        ax.set_ylabel("Mean signed return  (bps)")
        ax.set_title(r"Adjusted impact  $\mathbb{E}[\epsilon_n\,"
                     r"(m_{n+\tau} - m_{n+1})\,/\,m_n]$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

        m_all, _ = self._propagator_curve(
            mid_mo, sign_all, day_labels, horizons, ref_shift=1)
        print(f"MOs total: {N:,}")
        print(f"  walked ticks:    {walked.sum():,}")
        print(f"  no tick walk:    {(~walked).sum():,}")
        print(f"  tau = {horizons[0]:>5d}:  {m_all[0]:+.3f} bps")
        print(f"  tau = {horizons[-1]:>5d}:  {m_all[-1]:+.3f} bps")

    # --- Stylized Facts ---
    # --- 1. Fat-tailed returns ---
    def stylized_fat_tails(self, interval_minutes=30, bw_method=None,
                           label=None, ax=None, changes="level"):
        """Fat tails: KDE of fixed-interval price changes vs Gaussian.

        ``changes="level"`` uses Δ mid between LOCF samples on a trading-time
        grid (often very leptokurtic when the mid is sticky).  ``"log"`` uses
        log-return differences on the same grid.  This is *not* the same object
        as the legacy tick-by-tick log-return fat-tail figure (``_get_mid_returns``).

        Parameters
        ----------
        interval_minutes : float
            Aggregation window for price changes (default 30 min).
        bw_method : str or float, optional
            Bandwidth for scipy.stats.gaussian_kde.
        label : str, optional
            Legend label for the empirical curve. If *None*, defaults to
            ``"{interval_minutes} min changes"``.
        ax : matplotlib Axes, optional
            If provided, plot onto this axes (useful for overlaying
            simulation and empirical data on the same figure).
        changes : {"level", "log"}, optional
            ``"level"``: ``np.diff(sampled)``.  ``"log"``: ``np.diff(log(sampled))``.
        """
        interval_sec = interval_minutes * 60.0
        sampled = self._get_sampled_mid_prices(interval_sec)
        if sampled is None or len(sampled) < 20:
            print("Not enough data for fat-tails analysis."); return

        if changes == "level":
            chg = np.diff(sampled)
            xlabel = "Price change"
            title_kind = "price changes"
        elif changes == "log":
            if np.any(sampled <= 0) or not np.all(np.isfinite(sampled)):
                print("Non-positive or non-finite mids; cannot use log changes.")
                return
            chg = np.diff(np.log(sampled))
            xlabel = "Log return"
            title_kind = "log returns"
        else:
            print('changes must be "level" or "log".'); return

        if len(chg) < 10:
            print("Not enough intervals for KDE."); return

        mu, sigma = chg.mean(), chg.std()
        kurt = np.mean(((chg - mu) / sigma) ** 4) - 3.0

        kde = gaussian_kde(chg, bw_method=bw_method)
        x = np.linspace(chg.min(), chg.max(), 500)

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(x, norm.pdf(x, mu, sigma), "b-", lw=2, label="Gaussian")
        ax.plot(x, kde(x), "r-", lw=2, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(
            f"Density of {interval_minutes} minute {title_kind}\n"
            f"Excess kurtosis = {kurt:.2f}"
        )
        ax.legend()

        if own_fig:
            plt.tight_layout(); plt.show()

        print(f"Interval: {interval_minutes} min  |  "
              f"n_changes: {len(chg)}  |  "
              f"changes={changes!r}  |  "
              f"Excess kurtosis: {kurt:.2f}")

    def stylized_moment_instability(self, interval_minutes=5,
                                    min_n=50, max_n=7000, ax=None):
        """Second empirical moment of price changes as a function of sample size.

        For a process with finite variance (e.g. Gaussian), the sample
        second moment  M_2(n) = (1/n) Σ_{i=1}^{n} (Δp_i)²  converges
        smoothly and quickly to the population value as *n* grows.

        For heavy-tailed processes (tail exponent α ≈ 3, as is typical
        for financial returns), convergence is extremely slow and the
        estimator remains visibly erratic even at large *n*.  Individual
        extreme observations produce persistent jumps because they
        contribute disproportionately to the sum of squares, and the
        (1/n) averaging cannot damp them out quickly.

        The plot is constructed by evaluating M_2 at every prefix of the
        series: for each n from 1 to N, the second moment is computed
        using the first *n* observations only.  This is *not* a rolling
        window — each point uses all data up to that index, so the
        estimate can only become more stable (never "forgets" old data).
        The fact that it *still* wanders is precisely the diagnostic: it
        shows that moment estimation is unreliable for heavy-tailed data.

        Parameters
        ----------
        interval_minutes : float
            Aggregation window for price changes (default 5 min).
        min_n : int
            First sample size to display (trims the unstable left edge
            where a single extreme change dominates the plot).
        max_n : int
            Last sample size to display (keeps the x-axis range
            comparable to typical empirical datasets).
        ax : matplotlib Axes, optional
            If provided, plot onto this axes.
        """
        interval_sec = interval_minutes * 60.0
        sampled = self._get_sampled_mid_prices(interval_sec)
        if sampled is None or len(sampled) < 50:
            print("Not enough data for moment instability plot."); return

        changes = np.diff(sampled)
        if len(changes) < min_n:
            print("Not enough intervals."); return

        sq = changes ** 2
        running_m2 = np.cumsum(sq) / np.arange(1, len(sq) + 1)

        end = min(len(running_m2), max_n)
        plot_n = np.arange(min_n, end + 1)
        plot_m2 = running_m2[min_n - 1 : end]

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(plot_n, plot_m2, lw=1, color="tab:blue")
        ax.set_xlabel("Sample size")
        ax.set_ylabel("Second moment")
        ax.set_title(
            f"Second empirical moment of {interval_minutes} minute "
            f"price changes"
        )

        if own_fig:
            plt.tight_layout(); plt.show()

        print(f"Interval: {interval_minutes} min  |  "
              f"n_changes: {len(changes)}  |  "
              f"Final M₂: {running_m2[-1]:.6f}")

    # --- 2. Absence of return autocorrelation ---
    def stylized_return_autocorrelation(self, interval_minutes=1,
                                        max_lag=60):
        """
        ACF of fixed-interval log-returns with x-axis in minutes.

        Parameters
        ----------
        interval_minutes : float
            Sampling interval for mid-prices (default 1 min).
        max_lag : int
            Maximum number of lags to compute (each lag =
            *interval_minutes* minutes).  Default 60 covers one hour,
            well past the 20-minute efficiency threshold.

        Left panel : ACF vs lag in minutes.
        Right panel: Mean |ACF| before vs after the 20-minute threshold.
        """
        interval_sec = interval_minutes * 60.0
        sampled = self._get_sampled_mid_prices(interval_sec)
        if sampled is None or len(sampled) < 50:
            print("Not enough data for return analysis."); return

        rets = np.diff(np.log(sampled))
        rets = rets[np.isfinite(rets)]
        if len(rets) < 50:
            print("Not enough finite returns."); return

        n = len(rets)
        rets_dm = rets - rets.mean()
        var = np.dot(rets_dm, rets_dm)
        if var < 1e-30:
            print("Returns have zero variance."); return

        lags = np.arange(1, min(max_lag + 1, n))
        acf = np.array([np.dot(rets_dm[l:], rets_dm[:-l]) / var
                        for l in lags])
        lag_minutes = lags * interval_minutes
        ci = 1.96 / np.sqrt(n)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        colors = ["tab:red" if m <= 20 else "tab:blue"
                  for m in lag_minutes]
        ax.bar(lag_minutes, acf, width=interval_minutes * 0.8, alpha=0.7,
               color=colors)
        ax.axhline(ci, ls="--", color="red", alpha=0.5, label="95% CI")
        ax.axhline(-ci, ls="--", color="red", alpha=0.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(20.0, ls=":", color="green", lw=2, alpha=0.8,
                   label="20 min threshold")
        ax.bar([], [], color="tab:red", alpha=0.7, label="≤ 20 min")
        ax.bar([], [], color="tab:blue", alpha=0.7, label="> 20 min")
        ax.set_xlabel("Lag (minutes)"); ax.set_ylabel("Autocorrelation")
        ax.set_title("ACF of log-returns")
        ax.legend(fontsize=8, loc="upper right")

        ax = axes[1]
        bm = lag_minutes <= 20
        am = lag_minutes > 20
        if bm.any():
            mab = np.abs(acf[bm]).mean()
        else:
            mab = 0
        if am.any():
            maa = np.abs(acf[am]).mean()
        else:
            maa = 0
        n_before, n_after = int(bm.sum()), int(am.sum())

        bars = ax.bar(
            ["≤ 20 min\n(microstructure)", "> 20 min\n(efficient)"],
            [mab, maa],
            color=["tab:red", "tab:blue"], alpha=0.7, width=0.5,
        )
        ax.axhline(ci, ls="--", color="red", alpha=0.5, label="95% CI")
        ax.set_ylabel("Mean |ACF|")
        ax.set_title("Average absolute autocorrelation")
        ax.legend(fontsize=8)
        for bar, val, cnt in zip(bars, [mab, maa], [n_before, n_after]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}\n({cnt} lags)",
                    ha="center", fontsize=9)

        plt.suptitle(
            "Absence of return autocorrelation\n"
            "(ACF ≈ 0 for horizons ≳ 20 min)",
            fontsize=11,
        )
        plt.tight_layout(); plt.show()

        print(f"Sampling interval: {interval_minutes} min")
        print(f"Mean |ACF| ≤ 20 min: {mab:.4f}  ({n_before} lags)")
        print(f"Mean |ACF| > 20 min: {maa:.4f}  ({n_after} lags)")

    def stylized_return_autocorrelation_seconds(
        self, interval_seconds=1.0, max_lag=60,
    ):
        """
        ACF of fixed-interval log-returns with x-axis in seconds.

        Complements ``stylized_return_autocorrelation`` (minute bars) by
        resolving the sub-minute regime where MO Hawkes excitations and
        adverse selection are most visible.

        Parameters
        ----------
        interval_seconds : float
            Sampling interval for mid-prices (default 1 s).
        max_lag : int
            Maximum number of lags to compute (each lag =
            *interval_seconds* seconds).  Default 60 covers the first
            minute of the lag axis.
        """
        sampled = self._get_sampled_mid_prices(interval_seconds)
        if sampled is None or len(sampled) < 50:
            print("Not enough data for sub-minute return analysis."); return

        rets = np.diff(np.log(sampled))
        rets = rets[np.isfinite(rets)]
        if len(rets) < 50:
            print("Not enough finite returns."); return

        n = len(rets)
        rets_dm = rets - rets.mean()
        var = np.dot(rets_dm, rets_dm)
        if var < 1e-30:
            print("Returns have zero variance."); return

        lags = np.arange(1, min(max_lag + 1, n))
        acf = np.array([np.dot(rets_dm[l:], rets_dm[:-l]) / var
                        for l in lags])
        lag_seconds = lags * interval_seconds

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(lag_seconds, acf, width=interval_seconds * 0.8,
               alpha=0.7, color="tab:blue")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Lag (seconds)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(
            f"ACF of log-returns ({interval_seconds:g} s bars, "
            f"lags up to {max_lag} s)"
        )
        plt.tight_layout(); plt.show()

        print(f"Sampling interval: {interval_seconds:g} s")
        print(f"n_returns: {n:,}")

    # --- 3. Volatility clustering ---
    def stylized_volatility_clustering(self, interval_minutes=1,
                                       max_lag=100):
        """ACF of |returns| and returns² (slow decay = clustering).

        Parameters
        ----------
        interval_minutes : float
            Sampling interval for mid-prices (default 1 min).
        max_lag : int
            Maximum number of lags (each lag = *interval_minutes* min).
        """
        interval_sec = interval_minutes * 60.0
        sampled = self._get_sampled_mid_prices(interval_sec)
        if sampled is None or len(sampled) < 50:
            print("Not enough data for return analysis."); return

        rets = np.diff(np.log(sampled))
        rets = rets[np.isfinite(rets)]
        if len(rets) < 50:
            print("Not enough finite returns."); return

        n = len(rets)
        acf_abs = self._acf(np.abs(rets), max_lag)
        acf_sq = self._acf(rets ** 2, max_lag)
        lag_minutes = np.arange(1, len(acf_abs) + 1) * interval_minutes
        ci = 1.96 / np.sqrt(n)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

        axes[0].bar(lag_minutes, acf_abs, width=interval_minutes * 0.8,
                    alpha=0.7, color="tab:blue")
        axes[0].axhline(ci, ls="--", color="red", alpha=0.5)
        axes[0].axhline(-ci, ls="--", color="red", alpha=0.5)
        axes[0].axhline(0, color="black", lw=0.5)
        axes[0].set_xlabel("Lag (minutes)")
        axes[0].set_ylabel("Autocorrelation")
        axes[0].set_title("ACF of |returns|")

        axes[1].bar(lag_minutes, acf_sq, width=interval_minutes * 0.8,
                    alpha=0.7, color="tab:orange")
        axes[1].axhline(ci, ls="--", color="red", alpha=0.5)
        axes[1].axhline(-ci, ls="--", color="red", alpha=0.5)
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set_xlabel("Lag (minutes)")
        axes[1].set_title("ACF of returns²")

        plt.suptitle("Volatility clustering  "
                     "(slow decay → clustering present)")
        plt.tight_layout(); plt.show()

        if acf_abs[0] > 0:
            threshold = acf_abs[0] / np.e
            hl_idx = np.where(acf_abs < threshold)[0]
            hl_min = (hl_idx[0] + 1) * interval_minutes if len(hl_idx) > 0 \
                else f">{max_lag * interval_minutes}"
            print(f"|returns| ACF half-life ≈ {hl_min} min")
        print(f"|returns| ACF({interval_minutes} min) = {acf_abs[0]:.4f},  "
              f"ACF({10 * interval_minutes} min) = "
              f"{acf_abs[min(9, len(acf_abs) - 1)]:.4f}")

    # --- 4. Concave price impact ---
    def stylized_price_impact(self, n_bins=20):
        """Price impact (ticks walked) vs MO size with power-law fit."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT mo_volume, ticks_walked FROM mo_orders "
            "WHERE mo_volume IS NOT NULL AND mo_volume > 0 "
            "  AND ticks_walked IS NOT NULL"
        )
        rows = cur.fetchall()
        conn.close()

        sizes = np.array([r[0] for r in rows], dtype=np.float64)
        tw = np.array([r[1] for r in rows], dtype=np.float64)

        pcts = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(sizes, pcts))
        bi = np.clip(np.digitize(sizes, edges) - 1, 0, len(edges) - 2)

        ms, mi = [], []
        for b in range(len(edges) - 1):
            mask = bi == b
            if mask.sum() < 5:
                continue
            ms.append(sizes[mask].mean())
            mi.append(tw[mask].mean())
        ms, mi = np.array(ms), np.array(mi)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].scatter(sizes, tw, s=2, alpha=0.1)
        axes[0].set_xlabel("MO size"); axes[0].set_ylabel("Ticks walked")
        axes[0].set_title("Price impact: all MOs")
        axes[0].set_xscale("log")

        axes[1].scatter(ms, mi, s=30, zorder=3, label="Binned mean")
        pos = mi > 0
        delta = None
        if pos.sum() >= 2:
            coef = np.polyfit(np.log10(ms[pos]), np.log10(mi[pos]), 1)
            delta = coef[0]
            xf = np.logspace(np.log10(ms[pos].min()),
                             np.log10(ms[pos].max()), 100)
            axes[1].plot(xf, 10 ** coef[1] * xf ** delta, "r-", lw=2,
                         label=f"Power law: δ = {delta:.3f}")
        axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_xlabel("MO size")
        axes[1].set_ylabel("Mean ticks walked")
        axes[1].set_title("Concave price impact (√-law: δ ≈ 0.5)")
        axes[1].legend()

        plt.tight_layout(); plt.show()

        if delta is not None:
            print(f"Estimated impact exponent δ = {delta:.3f}  "
                  f"(empirical benchmark ≈ 0.5)")
        print(f"Total MOs: {len(sizes):,},  "
              f"fraction with impact > 0: "
              f"{(tw > 0).sum():,}/{len(tw):,} "
              f"({(tw > 0).mean():.1%})")

    # --- 5. Order sign autocorrelation (long memory) ---
    def stylized_order_sign_autocorrelation(self, max_lag_bar=50,
                                             max_lag_loglog=10_000,
                                             cap=800,
                                             n_log_bins=40):
        """ACF of MO signs with short-lag bar chart and long-lag
        log-log power-law fit.

        Parameters
        ----------
        max_lag_bar : int
            Maximum lag for the short-lag bar chart (left panel).
        max_lag_loglog : int
            Maximum lag for the log-log power-law fit (right panel).
        cap : int
            Upper lag cap used for the OLS power-law fit on [10^0, cap].
        n_log_bins : int
            Number of logarithmically spaced bins used in the log-log panel.
        """
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        conn = self._conn()
        mo_cols = self._table_cols(conn, "mo_orders")
        if "first_time_ns" in mo_cols:
            time_col = "first_time_ns"
        else:
            time_col = "timestamp"
        cur = conn.cursor()
        cur.execute(f"SELECT side FROM mo_orders ORDER BY {time_col}")
        sides_raw = [r[0] for r in cur.fetchall()]
        conn.close()

        signs = []
        for s in sides_raw:
            if s in ("buy", 1):
                signs.append(+1)
            elif s in ("sell", 2):
                signs.append(-1)
        signs = np.array(signs, dtype=float)

        if len(signs) < 50:
            print(f"Only {len(signs)} MOs — not enough."); return

        n = len(signs)
        buy_frac = (signs > 0).mean()
        print(f"Full sign series length: {n:,} MOs,  "
              f"buy fraction: {buy_frac:.3f}")

        # --- Panel 1: Short-lag ACF (bar chart) ---
        max_bar = min(max_lag_bar, n - 1)
        acf_short = self._acf_fft(signs, max_bar)   # [0..max_bar]
        lags_short = np.arange(1, max_bar + 1)
        ci = 1.96 / np.sqrt(n)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].bar(lags_short, acf_short[1:], width=0.8,
                    alpha=0.7, color="steelblue")
        axes[0].axhline(ci, ls="--", color="red", alpha=0.5,
                        label="95 % CI")
        axes[0].axhline(-ci, ls="--", color="red", alpha=0.5)
        axes[0].axhline(0, lw=0.3, color="k")
        axes[0].set_xlabel("Lag  $k$")
        axes[0].set_ylabel("Autocorrelation")
        axes[0].set_title("Signed-trade ACF (linear)")
        axes[0].legend(fontsize=8)

        # --- Panel 2: Log-log ACF with power-law fit ---
        max_ll = min(max_lag_loglog, n - 1)
        acf_long = self._acf_fft(signs, max_ll)     # [0..max_ll]
        lags_long = np.arange(1, max_ll + 1)
        acf_vals = acf_long[1:]                      # rho(1)..rho(max_ll)

        pos_mask = acf_vals > 0
        if pos_mask.sum() >= 10:
            # Log-spaced lag bins (instead of linear lag points on log axes).
            lags_pos = lags_long[pos_mask].astype(float)
            acf_pos = acf_vals[pos_mask]
            n_bins = int(max(5, n_log_bins))
            edges = np.unique(np.logspace(np.log10(lags_pos.min()),
                                          np.log10(lags_pos.max()),
                                          n_bins + 1))

            lag_bins, acf_bins = [], []
            for i in range(len(edges) - 1):
                if i == len(edges) - 2:
                    in_bin = ((lags_pos >= edges[i])
                              & (lags_pos <= edges[i + 1]))
                else:
                    in_bin = ((lags_pos >= edges[i])
                              & (lags_pos < edges[i + 1]))
                if in_bin.sum() == 0:
                    continue
                lag_bins.append(np.exp(np.mean(np.log(lags_pos[in_bin]))))
                acf_bins.append(acf_pos[in_bin].mean())

            lag_bins = np.asarray(lag_bins, dtype=float)
            acf_bins = np.asarray(acf_bins, dtype=float)

            # --- OLS fit in log-space on binned lags [10^0, cap] ---
            fit_lo = 10 ** 0
            fit_hi = min(float(cap), float(max_ll))
            fit_mask = ((lag_bins >= fit_lo) & (lag_bins <= fit_hi)
                        & (acf_bins > 0))
            if fit_mask.sum() >= 5:
                log_l = np.log(lag_bins[fit_mask])
                log_a = np.log(acf_bins[fit_mask])
                A = np.column_stack([np.ones_like(log_l), log_l])
                coefs = np.linalg.lstsq(A, log_a, rcond=None)[0]
                gamma_hat = -coefs[1]
                intercept = coefs[0]

                print(f"Power-law exponent:  "
                      f"\u03b3 = {gamma_hat:.4f}   "
                      f"(\u03c1(k) ~ k^{{-{gamma_hat:.4f}}})")
            else:
                gamma_hat = intercept = None
                print("Not enough positive log-binned points in fit range.")

            axes[1].scatter(lag_bins, acf_bins,
                            s=20, alpha=0.8, color="steelblue",
                            label="Log-binned ACF")
            axes[1].axvline(fit_lo, ls=":", lw=0.7, color="grey")
            axes[1].axvline(fit_hi, ls=":", lw=0.7, color="grey")

            if gamma_hat is not None:
                xf = np.logspace(0, np.log10(max_ll), 200)
                yf = np.exp(intercept) * xf ** (-gamma_hat)
                axes[1].plot(xf, yf, color="crimson", lw=1.5,
                             label=(rf"$\hat\rho(k) \propto "
                                    rf"k^{{-{gamma_hat:.3f}}}$  "
                                    rf"(fit $k\in[{fit_lo},{fit_hi}]$)"))

            axes[1].set_xscale("log"); axes[1].set_yscale("log")
            axes[1].set_xlabel("Lag  $k$")
            axes[1].set_ylabel(r"Autocorrelation  $\hat\rho(k)$")
            axes[1].set_title("Order-sign ACF power-law decay")
            axes[1].legend(fontsize=9)
        else:
            axes[1].text(0.5, 0.5,
                         "Not enough positive ACF values\n"
                         "for power-law fit",
                         transform=axes[1].transAxes, ha="center")
            axes[1].set_title("Log-log ACF")

        plt.suptitle("Order sign autocorrelation", fontsize=13)
        plt.tight_layout(); plt.show()

    # --- 6. Aggregational Gaussianity ---
    def stylized_aggregational_gaussianity(self, agg_minutes=None):
        """Excess kurtosis decays toward 0 at coarser timescales.

        For each horizon *mins*, uses the **same** construction as
        ``stylized_fat_tails(interval_minutes=mins)``:
        ``_get_sampled_mid_prices(mins * 60)`` then **level**
        ``np.diff(sampled)`` (price changes), standardized for κ and QQ.

        The default horizon list is ``[30, 60, 120]`` so every point matches
        that fat-tails definition (30+ minute level changes).  Shorter
        horizons (e.g. 1 min) are not included by default because level
        ΔP at very fine sampling is typically far more leptokurtic.

        Parameters
        ----------
        agg_minutes : list of float, optional
            Aggregation intervals in minutes.  Defaults to
            ``[30, 60, 120]``.  Pass a longer list explicitly if needed.
        """
        if agg_minutes is None:
            agg_minutes = [30, 60, 120]

        kurtoses, valid_levels = [], []
        returns_cache = {}
        for mins in agg_minutes:
            sampled = self._get_sampled_mid_prices(float(mins) * 60.0)
            if sampled is None or len(sampled) < 25:
                continue
            r = np.diff(np.asarray(sampled, dtype=np.float64))
            r = r[np.isfinite(r)]
            if len(r) < 20 or r.std() < 1e-15:
                continue
            k = np.mean(((r - r.mean()) / r.std()) ** 4) - 3.0
            kurtoses.append(k)
            valid_levels.append(mins)
            returns_cache[mins] = r

        if len(valid_levels) < 2:
            print("Not enough aggregation levels."); return

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.plot(valid_levels, kurtoses, "o-", lw=2, ms=8)
        ax.axhline(0, ls="--", color="grey", alpha=0.5,
                   label="Normal (κ = 0)")
        ax.set_xlabel("Aggregation interval (minutes)")
        ax.set_ylabel("Excess kurtosis")
        ax.set_title("Kurtosis decay with aggregation"); ax.legend()

        ax = axes[1]
        finest = valid_levels[0]
        coarsest = valid_levels[-1]
        for mins, col in [(finest, "tab:blue"), (coarsest, "tab:orange")]:
            r = returns_cache[mins]
            mu_r, sig_r = r.mean(), r.std()
            sr = np.sort((r - mu_r) / sig_r)
            nn = len(sr)
            th = np.array([self._ppf_normal((i + 0.5) / nn)
                           for i in range(nn)])
            ax.scatter(th, sr, s=2, alpha=0.4, color=col,
                       label=f"{mins} min")

        ax.plot([-4, 4], [-4, 4], "r--", lw=1, label="45° line")
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_xlabel("Normal quantiles")
        ax.set_ylabel("Standardised price-change quantiles")
        ax.set_title("QQ-plot: fine vs coarse aggregation")
        ax.legend()

        plt.suptitle("Aggregational Gaussianity  "
                     "(kurtosis → 0 at coarser scales)")
        plt.tight_layout(); plt.show()

        for mins, k in zip(valid_levels, kurtoses):
            print(f"  {mins:>5g} min:  excess kurtosis = {k:+.2f}")

    # --- Cancellation calibration ---
    def _cancel_sequential_pass(self, max_y=5, n_y_bins=24,
                                max_ticks_opp=100, max_ticks_same=50):
        """One sequential pass collecting all cancellation calibration data.

        Collects P(y), P(d_opp), P(d_same) via random active-order
        sampling, plus cancel lifetimes and queue positions.
        P(y|C), P(d_opp|C), P(d_same|C) are done vectorised from SQL.
        Results are cached on the instance.
        """
        if self._cancel_pass_done:
            return

        tick = self.tick_size
        day_sql = self._day_clause()

        # --- y-ratio bins ---
        y_bins = np.linspace(0, max_y, n_y_bins + 1)

        # --- Streaming P(·|C) from the database (cursor, no DataFrame) ---
        conn = self._conn()

        # P(y|C) — stream y_ratio for CXL rows, bin on the fly
        hist_cancel_y = np.zeros(len(y_bins) - 1)
        cur = conn.cursor()
        cur.execute(
            "SELECT y_ratio FROM orders WHERE event_type='CXL'"
            " AND y_ratio IS NOT NULL AND y_ratio>=0 AND y_ratio<=?"
            + day_sql, (max_y,)
        )
        for (yr,) in cur:
            idx = np.searchsorted(y_bins, yr, side="right") - 1
            if 0 <= idx < len(hist_cancel_y):
                hist_cancel_y[idx] += 1
        self._hist_cancel_y = hist_cancel_y

        # P(d_opp|C) and P(d_same|C) — stream CXL rows
        hist_cancel_opp = np.zeros(max_ticks_opp + 1)
        hist_cancel_same = np.zeros(max_ticks_same + 1)
        cur.execute(
            "SELECT order_price, side, best_bid, best_ask FROM orders "
            "WHERE event_type='CXL' AND order_price>0 "
            "AND best_bid>0 AND best_ask>0" + day_sql
        )
        for price, side, bb, ba in cur:
            if side == 1:
                d_opp = round((ba - price) / tick)
                d_same = round((bb - price) / tick)
            else:
                d_opp = round((price - bb) / tick)
                d_same = round((price - ba) / tick)
            if 0 <= d_opp <= max_ticks_opp:
                hist_cancel_opp[int(d_opp)] += 1
            if 0 <= d_same <= max_ticks_same:
                hist_cancel_same[int(d_same)] += 1
        self._hist_cancel_tick_opp = hist_cancel_opp
        self._hist_cancel_tick_same = hist_cancel_same

        conn.close()
        print("Streaming P(·|C) done.")

        # --- Sequential pass — P(y), P(d_opp), P(d_same), lifetimes, queue positions ---
        hist_total_y         = np.zeros(len(y_bins) - 1)
        hist_total_tick_opp  = np.zeros(max_ticks_opp + 1)
        hist_total_tick_same = np.zeros(max_ticks_same + 1)

        active = {}                     # oid → (price, side, delta0, ts_raw)
        price_queues = defaultdict(list) # price_key → [oid, …]
        lifetimes = []
        cancel_positions = []
        cancel_queue_sizes = []

        conn = self._conn()
        total_ev = conn.cursor().execute(
            "SELECT COUNT(*) FROM orders "
            "WHERE event_type IN ('LO','CXL')" + day_sql
        ).fetchone()[0]

        query = (
            "SELECT timestamp, event_type, order_id, side, "
            "       order_price, best_bid, best_ask, delta0 "
            "FROM orders WHERE event_type IN ('LO','CXL')"
            + day_sql + " ORDER BY timestamp"
        )

        cur = conn.cursor()
        cur.execute(query)
        chunk_size = 100_000
        processed = 0
        next_print = 0

        while True:
            rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            for row in rows:
                ts_raw, etype, oid, side, price, bb, ba, delta0 = row

                if total_ev:
                    pct = int(100 * processed / total_ev)
                else:
                    pct = 100
                if pct >= next_print:
                    print(f"  {pct}% ({processed:,} events)", flush=True)
                    next_print += 5

                # --- sample ONE random active order ---
                if active and bb and ba and bb > 0 and ba > 0:
                    s_oid = random.choice(tuple(active))
                    p, s, d0, _ = active[s_oid]
                    if p and p > 0:
                        # P(d_opp)
                        if s == 1:
                            d = round((ba - p) / tick)
                        else:
                            d = round((p - bb) / tick)
                        if 0 <= d <= max_ticks_opp:
                            hist_total_tick_opp[int(d)] += 1

                        # P(d_same)
                        if s == 1:
                            ds = round((bb - p) / tick)
                        else:
                            ds = round((p - ba) / tick)
                        if 0 <= ds <= max_ticks_same:
                            hist_total_tick_same[int(ds)] += 1

                        # P(y)
                        if d0 is not None and d0 != 0:
                            if s == 1:
                                delta_t = p - ba
                            else:
                                delta_t = bb - p
                            y = delta_t / d0
                            if np.isfinite(y) and 0 <= y <= max_y:
                                idx = np.searchsorted(y_bins, y,
                                                      side="right") - 1
                                if 0 <= idx < len(hist_total_y):
                                    hist_total_y[idx] += 1

                # --- LO: register ---
                if etype == "LO":
                    if price and price > 0:
                        active[oid] = (price, side, delta0, ts_raw)
                        price_queues[round(price / tick)].append(oid)

                # --- CXL: lifetime, queue pos, remove ---
                elif etype == "CXL" and oid in active:
                    p, s, d0, lo_ts = active[oid]
                    pk = round(p / tick)
                    queue = price_queues.get(pk)
                    if queue is not None:
                        try:
                            pos = queue.index(oid)
                            cancel_positions.append(pos)
                            cancel_queue_sizes.append(len(queue))
                            queue.pop(pos)
                        except ValueError:
                            pass
                    # lifetime
                    if isinstance(ts_raw, (int, float)):
                        lt = float(ts_raw) - float(lo_ts)
                    else:
                        try:
                            lt = (pd.Timestamp(ts_raw)
                                  - pd.Timestamp(lo_ts)).total_seconds()
                        except (TypeError, ValueError, OverflowError):
                            lt = None
                    if lt is not None and lt > 0:
                        lifetimes.append(lt)
                    del active[oid]

                processed += 1

        cur.close()
        conn.close()

        self._y_bins              = y_bins
        self._hist_total_y        = hist_total_y
        self._hist_total_tick_opp = hist_total_tick_opp
        self._hist_total_tick_same = hist_total_tick_same
        self._max_ticks_opp       = max_ticks_opp
        self._max_ticks_same      = max_ticks_same
        self._lifetimes_sec       = np.array(lifetimes)
        self._cancel_positions    = np.array(cancel_positions)
        self._cancel_queue_sizes  = np.array(cancel_queue_sizes)
        self._cancel_pass_done    = True

        print(f"\n100% done ({processed:,} events)")
        print(f"  Active orders remaining: {len(active):,}")
        print(f"  Lifetimes collected:     {len(self._lifetimes_sec):,}")
        print(f"  Queue-position samples:  {len(self._cancel_positions):,}")

    # --- P(C) computation ---
    def _compute_pc(self):
        """Compute global per-trade-timestep cancellation probability."""
        if hasattr(self, "_P_C"):
            return self._P_C

        conn = self._conn()
        cur = conn.cursor()
        day_sql = self._day_clause()

        row = cur.execute(
            "SELECT "
            "  SUM(CASE WHEN event_type='CXL' THEN 1 ELSE 0 END), "
            "  AVG(n_total) "
            "FROM orders WHERE n_total IS NOT NULL" + day_sql
        ).fetchone()

        cancellations = row[0]
        avg_n = row[1]

        if "mo_orders" in self.tables:
            n_trades = cur.execute(
                "SELECT COUNT(*) FROM mo_orders"
                + (" WHERE 1=1" + day_sql if day_sql else "")
            ).fetchone()[0]
        else:
            n_trades = cur.execute(
                "SELECT COUNT(*) FROM orders "
                "WHERE event_type LIKE 'MO%'" + day_sql
            ).fetchone()[0]

        conn.close()

        if avg_n and avg_n > 0 and n_trades and n_trades > 0:
            self._P_C = cancellations / (avg_n * n_trades)
        else:
            self._P_C = 0.0

        if n_trades:
            self._n_trades = int(n_trades)
        else:
            self._n_trades = 0
        print(f"n_trades = {self._n_trades:,}")
        print(f"P(C) per trade-timestep per order = {self._P_C:.8f}")
        return self._P_C

    # --- Public cancellation methods ---
    def cancel_event_counts(self):
        """Print LO / CXL event counts."""
        conn = self._conn()
        day_sql = self._day_clause()
        row = conn.cursor().execute(
            "SELECT "
            "  SUM(CASE WHEN event_type='LO'  THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN event_type='CXL' THEN 1 ELSE 0 END), "
            "  COUNT(*) "
            "FROM orders WHERE event_type IN ('LO','CXL')" + day_sql
        ).fetchone()
        conn.close()

        print(f"Limit orders:    {int(row[0]):>12,}")
        print(f"Cancellations:   {int(row[1]):>12,}")
        print(f"Total LO + CXL:  {int(row[2]):>12,}")

    def cancel_prob_y(self):
        """P(y), P(y|C), P(C|y) via Bayes — Mike-Farmer y-ratio."""
        self._cancel_sequential_pass()
        P_C = self._compute_pc()

        bins = self._y_bins
        centers = 0.5 * (bins[:-1] + bins[1:])
        h_tot = self._hist_total_y
        h_cxl = self._hist_cancel_y

        if h_tot.sum() > 0:
            p_y   = h_tot / h_tot.sum()
        else:
            p_y   = np.zeros_like(h_tot)
        if h_cxl.sum() > 0:
            p_y_c = h_cxl / h_cxl.sum()
        else:
            p_y_c = np.zeros_like(h_cxl)

        ok = (p_y > 0) & (p_y_c > 0)
        P_C_y = np.full_like(p_y, np.nan)
        P_C_y[ok] = P_C * p_y_c[ok] / p_y[ok]

        # --- plots ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        mask = ok
        ax.plot(centers[mask], np.log(p_y[mask]), "o-", ms=4,
                label="log P(y)")
        ax.plot(centers[mask], np.log(p_y_c[mask]), "o-", ms=4,
                label="log P(y|C)")
        ax.set_xlabel("y = δ(t) / δ₀")
        ax.set_ylabel("log probability")
        ax.set_title("P(y) vs P(y|C) — Mike-Farmer distributions")
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        valid = np.isfinite(P_C_y) & (P_C_y > 0)
        ax.plot(centers[valid], P_C_y[valid], "o-", ms=5)
        ax.set_xlabel("y = δ(t) / δ₀")
        ax.set_ylabel("P(C | y)")
        ax.set_title("Cancellation probability vs relative price distance")
        ax.grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()

        print(f"P(C|y) range: [{np.nanmin(P_C_y[valid]):.4f}, "
              f"{np.nanmax(P_C_y[valid]):.4f}]")

    def cancel_prob_tick_distance(self):
        """P(C|d) where d = tick distance to opposite best quote."""
        self._cancel_sequential_pass()
        P_C = self._compute_pc()

        h_tot = self._hist_total_tick_opp
        h_cxl = self._hist_cancel_tick_opp
        tick_x = np.arange(len(h_tot))

        if h_tot.sum() > 0:
            p_d   = h_tot / h_tot.sum()
        else:
            p_d   = np.zeros_like(h_tot)
        if h_cxl.sum() > 0:
            p_d_c = h_cxl / h_cxl.sum()
        else:
            p_d_c = np.zeros_like(h_cxl)

        P_C_d = np.full_like(p_d, np.nan)
        ok = (p_d > 0) & (p_d_c > 0)
        P_C_d[ok] = P_C * p_d_c[ok] / p_d[ok]
        valid = np.isfinite(P_C_d) & (P_C_d > 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(tick_x[valid], P_C_d[valid], "o-", ms=4)
        axes[0].set_xlabel("Tick distance to opposite best")
        axes[0].set_ylabel("P(C | d)")
        axes[0].set_title("Cancel prob vs tick distance (opposite best)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(tick_x[valid], P_C_d[valid], "o-", ms=4)
        axes[1].set_xlabel("Tick distance to opposite best")
        axes[1].set_ylabel("P(C | d)")
        axes[1].set_yscale("log")
        axes[1].set_title("Cancel prob vs tick distance (log)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()

        print(f"P(C|d) range: [{np.nanmin(P_C_d[valid]):.6f}, "
              f"{np.nanmax(P_C_d[valid]):.6f}]")
        print(f"  Book samples:   {h_tot.sum():.0f}")
        print(f"  Cancel samples: {h_cxl.sum():.0f}")

    def cancel_lifetime_distribution(self):
        """Distribution of canceled-order lifetimes."""
        self._cancel_sequential_pass()
        lt = self._lifetimes_sec
        if len(lt) == 0:
            print("No lifetime data."); return

        print(f"Canceled-order lifetimes: {len(lt):,}")
        print(f"  Median: {np.median(lt):.1f} s")
        print(f"  Mean:   {np.mean(lt):.1f} s")
        print(f"  Max:    {np.max(lt):.1f} s")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        q99 = np.percentile(lt, 99)
        axes[0].hist(lt[lt <= q99], bins=200, edgecolor="none", alpha=0.7)
        axes[0].set_xlabel("Lifetime (seconds)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Canceled-order lifetime distribution")
        axes[0].grid(True, alpha=0.3)

        log_bins = np.logspace(np.log10(max(lt.min(), 0.001)),
                               np.log10(lt.max()), 100)
        axes[1].hist(lt, bins=log_bins, edgecolor="none", alpha=0.7)
        axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_xlabel("Lifetime (seconds)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Canceled-order lifetime (log-log)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()

    def cancel_queue_position(self):
        """Queue position at cancellation + Beta fit."""
        self._cancel_sequential_pass()
        pos  = self._cancel_positions
        qsz  = self._cancel_queue_sizes
        if len(pos) == 0:
            print("No queue-position data."); return

        fracs = pos / qsz

        print(f"Queue-position samples: {len(pos):,}")
        print(f"  Median absolute:   {np.median(pos):.0f}")
        print(f"  Mean absolute:     {np.mean(pos):.1f}")
        print(f"  Median fractional: {np.median(fracs):.3f}")
        print(f"  Mean fractional:   {np.mean(fracs):.3f}")

        # Beta fit
        interior = fracs[(fracs > 0) & (fracs < 1)]
        try:
            a_fit, b_fit, _, _ = beta_dist.fit(interior, floc=0, fscale=1)
            print(f"\nFitted Beta(α={a_fit:.4f}, β={b_fit:.4f})")
        except (ValueError, RuntimeError):
            a_fit = b_fit = None
            print("Beta fit failed.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        max_pos = int(np.percentile(pos, 99))
        axes[0].hist(pos[pos <= max_pos],
                     bins=np.arange(0, max_pos + 2) - 0.5,
                     edgecolor="none", alpha=0.7)
        axes[0].set_xlabel("Queue position (# orders ahead)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Queue position at cancellation (absolute)")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(fracs, bins=50, density=True,
                     edgecolor="none", alpha=0.5, label="Empirical")
        if a_fit is not None:
            x_pdf = np.linspace(0.001, 0.999, 500)
            axes[1].plot(x_pdf, beta_dist.pdf(x_pdf, a_fit, b_fit), "r-",
                         lw=2, label=f"Beta(α={a_fit:.3f}, β={b_fit:.3f})")
        axes[1].set_xlabel("Fractional queue position  (pos / queue size)")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Queue position at cancellation (fraction)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()

    def cancel_touch_effect(self):
        """P(C|d_same) — touch-cancel effect with exponential & power-law fits."""
        self._cancel_sequential_pass()
        P_C = self._compute_pc()

        h_book = self._hist_total_tick_same
        h_cxl  = self._hist_cancel_tick_same
        max_d  = self._max_ticks_same

        book_sum = h_book.sum()
        if book_sum > 0:
            p_d = h_book / book_sum
        else:
            p_d = np.zeros_like(h_book)
        cxl_sum = h_cxl.sum()
        if cxl_sum > 0:
            p_d_cxl = h_cxl / cxl_sum
        else:
            p_d_cxl = np.zeros_like(h_cxl)

        P_C_ds = np.full(max_d + 1, np.nan)
        ok = (p_d > 0) & (p_d_cxl > 0)
        P_C_ds[ok] = P_C * p_d_cxl[ok] / p_d[ok]

        d_axis = np.arange(max_d + 1)
        valid  = np.isfinite(P_C_ds) & (P_C_ds > 0)

        # --- Exponential fit: base·(1 + c·exp(−d)) ---
        def exp_model(d, base, c):
            return base * (1.0 + c * np.exp(-d))

        try:
            popt_e, _ = curve_fit(
                exp_model, d_axis[valid], P_C_ds[valid],
                p0=[np.nanmedian(P_C_ds[valid]), 1.0],
                bounds=([0, 0], [1, 200]), maxfev=10_000)
            base_e, c_e = popt_e
            exp_ok = True
        except (ValueError, RuntimeError):
            exp_ok = False

        # --- Power-law fit on coarser bins: A·(1+d)^(−α) ---
        bin_edges = [0, 1, 2, 3, 5, 8, 12, 17, 25, 35, max_d + 1]
        # Trim edges to max_d+1
        bin_edges = [e for e in bin_edges if e <= max_d + 1]
        if bin_edges[-1] != max_d + 1:
            bin_edges.append(max_d + 1)
        n_bins = len(bin_edges) - 1

        binned_book  = np.zeros(n_bins)
        binned_cxl   = np.zeros(n_bins)
        bin_centers  = np.zeros(n_bins)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            binned_book[i]  = h_book[lo:hi].sum()
            binned_cxl[i]   = h_cxl[lo:hi].sum()
            w = h_book[lo:hi]
            ticks = np.arange(lo, hi)
            if w.sum() > 0:
                bin_centers[i] = np.average(ticks, weights=w)
            else:
                bin_centers[i] = 0.5 * (lo + hi - 1)

        binned_book_sum = binned_book.sum()
        if binned_book_sum > 0:
            pb = binned_book / binned_book_sum
        else:
            pb = np.zeros(n_bins)
        binned_cxl_sum = binned_cxl.sum()
        if binned_cxl_sum > 0:
            pc = binned_cxl / binned_cxl_sum
        else:
            pc = np.zeros(n_bins)
        P_bin = np.full(n_bins, np.nan)
        ok_b = (pb > 0) & (pc > 0)
        P_bin[ok_b] = P_C * pc[ok_b] / pb[ok_b]
        v_b = np.isfinite(P_bin) & (P_bin > 0)

        def power_law(d, A, alpha):
            return A * (1.0 + d) ** (-alpha)

        try:
            popt_p, _ = curve_fit(
                power_law, bin_centers[v_b], P_bin[v_b],
                p0=[0.2, 1.0], bounds=([0, 0], [10, 10]),
                maxfev=10_000)
            A_fit, alpha_fit = popt_p
            pw_ok = True
        except (ValueError, RuntimeError):
            pw_ok = False

        # --- Plots ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # panel 1: raw P(C|d_same)
        axes[0].plot(d_axis[valid], P_C_ds[valid], "o-", ms=4,
                     label="Empirical")
        if exp_ok:
            d_sm = np.linspace(0, max_d, 300)
            axes[0].plot(d_sm, exp_model(d_sm, base_e, c_e), "r--", lw=2,
                         label=f"base·(1+{c_e:.1f}·exp(−d))")
        axes[0].set_xlabel("Same-side distance (ticks)")
        axes[0].set_ylabel("P(C | d)")
        axes[0].set_title("Touch-cancel effect")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # panel 2: log-scale raw
        axes[1].plot(d_axis[valid], P_C_ds[valid], "o-", ms=4)
        axes[1].set_xlabel("Same-side distance (ticks)")
        axes[1].set_ylabel("P(C | d)")
        axes[1].set_yscale("log")
        axes[1].set_title("Touch-cancel effect (log)")
        axes[1].grid(True, alpha=0.3)

        # panel 3: coarse bins log-log + power law
        axes[2].plot(1 + bin_centers[v_b], P_bin[v_b], "o", ms=7,
                     label="Binned")
        if pw_ok:
            d_sm = np.linspace(0, max_d, 500)
            axes[2].plot(1 + d_sm, power_law(d_sm, A_fit, alpha_fit),
                         "r--", lw=2,
                         label=rf"$A(1+d)^{{-{alpha_fit:.2f}}}$")
        axes[2].set_xscale("log"); axes[2].set_yscale("log")
        axes[2].set_xlabel("1 + d")
        axes[2].set_ylabel("P(C | d)")
        axes[2].set_title("Touch-cancel (log-log, coarse bins)")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()

        if exp_ok:
            print(f"Exponential fit:  base={base_e:.6f},  c={c_e:.3f}")
            print(f"  → touch orders ~{1 + c_e:.1f}× more likely to cancel")
        if pw_ok:
            print(f"Power-law fit:  A={A_fit:.4f},  α={alpha_fit:.3f}")

    def cancel_vs_imbalance(self):
        """P(C | imbalance) with linear fit  (vectorised from DB)."""
        conn = self._conn()
        day_sql = self._day_clause()
        cur = conn.cursor()
        cur.execute(
            "SELECT imbalance, is_cancel, side, n_total "
            "FROM orders WHERE imbalance IS NOT NULL" + day_sql
        )
        rows = cur.fetchall()
        conn.close()

        if not rows:
            print("No imbalance data."); return

        imb = np.array([r[0] for r in rows], dtype=np.float64)
        is_cxl = np.array([int(r[1]) for r in rows], dtype=np.int32)
        side = np.array([r[2] for r in rows], dtype=np.int32)
        n_total = np.array([r[3] for r in rows], dtype=np.float64)

        imb_mf = np.where(side == 1, (1 + imb) / 2, (1 - imb) / 2)
        valid = (imb_mf >= 0) & (imb_mf <= 1)
        imb_mf = imb_mf[valid]
        is_cxl = is_cxl[valid]
        n_total = n_total[valid]

        n_bins = 20
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bi = np.clip(np.searchsorted(bin_edges, imb_mf, side="right") - 1,
                     0, n_bins - 1)
        x_vals, y_vals, ev_cnt = [], [], []
        for b in range(n_bins):
            m = bi == b
            cnt = m.sum()
            if cnt == 0:
                continue
            c_sum = is_cxl[m].sum()
            avg_n = n_total[m].mean()
            mid = (bin_edges[b] + bin_edges[b + 1]) / 2
            x_vals.append(mid)
            y_vals.append(c_sum / (avg_n * cnt) if avg_n > 0 else np.nan)
            ev_cnt.append(cnt)

        x = np.array(x_vals)
        y = np.array(y_vals)
        ev_cnt = np.array(ev_cnt)
        mask = (~np.isnan(y)) & (ev_cnt > 2000)
        x, y = x[mask], y[mask]

        coef = np.polyfit(x, y, 1)
        K2 = coef[0]
        if K2 != 0:
            B  = coef[1] / K2
        else:
            B  = 0

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, label="Empirical")
        x_fit = np.linspace(0, 1, 200)
        plt.plot(x_fit, K2 * (x_fit + B), "r--",
                 label=f"K2={K2:.4f}, B={B:.2f}")
        plt.xlabel("Mike–Farmer imbalance $n_{imb}$")
        plt.ylabel("P(C | imbalance)")
        plt.title("Cancellation probability vs order book imbalance")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

        print(f"K2 = {K2}")
        print(f"B  = {B}")

    def cancel_vs_book_size(self):
        """P(C | n_total) with K3/n fit  (vectorised from DB)."""
        conn = self._conn()
        day_sql = self._day_clause()
        cur = conn.cursor()
        cur.execute(
            "SELECT n_total, is_cancel FROM orders "
            "WHERE n_total IS NOT NULL" + day_sql
        )
        rows = cur.fetchall()
        conn.close()

        if not rows:
            print("No n_total data."); return

        n_tot = np.array([r[0] for r in rows], dtype=np.float64)
        is_cxl = np.array([int(r[1]) for r in rows], dtype=np.int32)

        bin_edges = np.arange(n_tot.min(), n_tot.max() + 5, 5)
        bi = np.clip(np.searchsorted(bin_edges, n_tot, side="right") - 1,
                     0, len(bin_edges) - 2)
        x_vals, y_vals = [], []
        for b in range(len(bin_edges) - 1):
            m = bi == b
            cnt = m.sum()
            if cnt == 0:
                continue
            c_sum = is_cxl[m].sum()
            avg = n_tot[m].mean()
            mid = (bin_edges[b] + bin_edges[b + 1]) / 2
            x_vals.append(mid)
            y_vals.append(c_sum / (avg * cnt) if avg > 0 else np.nan)

        x = np.array(x_vals)
        y = np.array(y_vals)
        mask = ~np.isnan(y)
        x, y = x[mask], y[mask]

        K3 = np.mean(y * x)

        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, s=20, label="Empirical")
        x_fit = np.linspace(x.min(), x.max(), 200)
        plt.plot(x_fit, K3 / x_fit, "r--", label=f"K3={K3:.3f}")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("$n_{tot}$")
        plt.ylabel("P(C | $n_{tot}$)")
        plt.title("Cancellation probability vs total orders in book")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

        print(f"K3 = {K3}")

    # --- Spike / gap diagnostics ---
    def diagnose_spike(self, offset, window=60, mid_threshold=None,
                       show_all_events=False):
        """Print an aligned event log around a suspicious BBO offset.

        Parameters
        ----------
        offset : int
            Row offset into the ``bbo`` table (``ORDER BY timestamp``).
        window : int
            Number of BBO rows to inspect starting from *offset*.
        mid_threshold : float or None
            Only highlight BBO transitions where ``|dp_mid|`` exceeds this
            value.  ``None`` → report every BBO change.
        show_all_events : bool
            If True, print *every* order/fill/MO row in the timestamp
            window, not just the ones coinciding with a BBO change.
        """
        conn = self._conn()
        try:
            bbo = pd.read_sql(
                "SELECT rowid AS rid, timestamp, best_bid, best_ask, "
                "mid_price FROM bbo ORDER BY timestamp "
                f"LIMIT {window} OFFSET {offset}",
                conn,
            )
            if bbo.empty:
                print("No BBO rows at that offset."); return

            bbo["d_bid"] = bbo["best_bid"].diff()
            bbo["d_ask"] = bbo["best_ask"].diff()
            bbo["d_mid"] = bbo["mid_price"].diff()

            changes = bbo[
                (bbo["d_bid"] != 0) | (bbo["d_ask"] != 0)
            ].iloc[0:]  # skip NaN row from diff
            if mid_threshold is not None:
                changes = changes[changes["d_mid"].abs() >= mid_threshold]

            if changes.empty:
                print(f"No BBO changes in the {window}-row window "
                      f"starting at offset {offset}.")
                return

            t_lo = bbo["timestamp"].iloc[0]
            t_hi = bbo["timestamp"].iloc[-1]

            orders = pd.read_sql(
                "SELECT rowid AS rid, timestamp, event_type, order_id, "
                "side, order_price, volume, best_bid, best_ask, mid_price, "
                "dp_mid, ticks_from_best, spread_ticks, imbalance, "
                "bid_depth_L0, bid_depth_L1, bid_depth_L2, "
                "ask_depth_L0, ask_depth_L1, ask_depth_L2 "
                "FROM orders WHERE timestamp BETWEEN ? AND ? "
                "ORDER BY timestamp, rowid",
                conn, params=(t_lo, t_hi),
            )
            mo = pd.read_sql(
                "SELECT rowid AS rid, timestamp, side, mo_volume, n_fills, "
                "min_price, max_price, ticks_walked, best_bid, best_ask "
                "FROM mo_orders WHERE timestamp BETWEEN ? AND ? "
                "ORDER BY timestamp, rowid",
                conn, params=(t_lo, t_hi),
            )
            fills = pd.read_sql(
                "SELECT rowid AS rid, timestamp, volume, price, side, "
                "best_bid, best_ask, ticks_from_bbo "
                "FROM fills WHERE timestamp BETWEEN ? AND ? "
                "ORDER BY timestamp, rowid",
                conn, params=(t_lo, t_hi),
            )
        finally:
            conn.close()

        print(f"\nSPIKE DIAGNOSTIC   offset={offset}   "
              f"window={window}   threshold={mid_threshold}")
        print(f"  Timestamp range: {t_lo:.6f} → {t_hi:.6f}")
        print(f"  BBO changes found: {len(changes)}\n")

        for _, ch in changes.iterrows():
            ts = ch["timestamp"]
            print(f"{'-' * 40}")
            print(f"  BBO row {int(ch['rid'])}  t={ts:.8f}")
            print(f"    bid {ch['best_bid']:.2f}  ask {ch['best_ask']:.2f}  "
                  f"mid {ch['mid_price']:.2f}")
            print(f"    Δbid={ch['d_bid']:+.2f}  Δask={ch['d_ask']:+.2f}  "
                  f"Δmid={ch['d_mid']:+.2f}")

            t_prev = bbo.loc[
                bbo["rid"] < ch["rid"], "timestamp"
            ]
            if not t_prev.empty:
                t_start = t_prev.iloc[-1]
            else:
                t_start = ts - 0.01

            if not show_all_events:
                nearby_orders = orders[
                    (orders["timestamp"] > t_start) & (orders["timestamp"] <= ts)
                ]
            else:
                nearby_orders = orders[
                    (orders["timestamp"] <= ts)
                ]
            nearby_mo = mo[
                (mo["timestamp"] > t_start) & (mo["timestamp"] <= ts)
            ]
            nearby_fills = fills[
                (fills["timestamp"] > t_start) & (fills["timestamp"] <= ts)
            ]

            if not nearby_mo.empty:
                print(f"\n    Market orders ({len(nearby_mo)}):")
                for _, m in nearby_mo.iterrows():
                    print(f"      {m['side']:4s}  vol={m['mo_volume']:.0f}  "
                          f"fills={m['n_fills']:.0f}  "
                          f"price=[{m['min_price']:.2f}–{m['max_price']:.2f}]  "
                          f"walked={m['ticks_walked']:.0f} ticks")

            if not nearby_fills.empty:
                print(f"\n    Fills ({len(nearby_fills)}):")
                for _, f in nearby_fills.iterrows():
                    print(f"      {f['side']:4s}  vol={f['volume']:.0f}  "
                          f"price={f['price']:.2f}  "
                          f"ticks_from_bbo={f['ticks_from_bbo']:.0f}")

            if not nearby_orders.empty:
                print(f"\n    Order events ({len(nearby_orders)}):")
                for _, o in nearby_orders.iterrows():
                    if o["side"] == 1:
                        side_str = "BID"
                    else:
                        side_str = "ASK"
                    print(
                        f"      {o['event_type']:3s} {side_str} "
                        f"id={int(o['order_id'])}  "
                        f"price={o['order_price']:.2f}  "
                        f"vol={o['volume']:.0f}  "
                        f"dp_mid={o['dp_mid']:+.1f}  "
                        f"spread={o['spread_ticks']:.0f}tk  "
                        f"depths B[{o['bid_depth_L0']:.0f},"
                        f"{o['bid_depth_L1']:.0f},"
                        f"{o['bid_depth_L2']:.0f}] "
                        f"A[{o['ask_depth_L0']:.0f},"
                        f"{o['ask_depth_L1']:.0f},"
                        f"{o['ask_depth_L2']:.0f}]"
                    )
            print()

        print("END OF DIAGNOSTIC\n")

    def trace_resting_order(self, order_price, side, before_timestamp,
                            limit=20):
        """Trace the life of resting orders at a given price level.

        Useful for understanding why a far-out quote was sitting in the
        book when it became the new best bid/ask during a spike.

        Parameters
        ----------
        order_price : float
            The price level to search for.
        side : int or str
            ``1`` / ``'bid'`` or ``2`` / ``'ask'``.
        before_timestamp : float
            Only look at order events up to this timestamp.
        limit : int
            Maximum number of matching order rows to return
            (most recent first).

        Returns
        -------
        pandas.DataFrame
            Matching rows from the ``orders`` table (newest first).
        """
        if isinstance(side, str):
            if side.lower() in ("bid", "buy", "b"):
                side = 1
            else:
                side = 2
        conn = self._conn()
        try:
            df = pd.read_sql(
                "SELECT rowid AS rid, timestamp, event_type, order_id, "
                "side, order_price, volume, best_bid, best_ask, "
                "mid_price, ticks_from_best "
                "FROM orders "
                "WHERE order_price = ? AND side = ? AND timestamp <= ? "
                "ORDER BY timestamp DESC, rowid DESC "
                f"LIMIT {limit}",
                conn,
                params=(order_price, side, before_timestamp),
            )
        finally:
            conn.close()

        if df.empty:
            print(f"No order events found at price={order_price} "
                  f"side={side} before t={before_timestamp}.")
            return df

        df = df.iloc[::-1].reset_index(drop=True)

        if side == 1:
            side_str = "BID"
        else:
            side_str = "ASK"
        print(f"\nRESTING ORDER TRACE   price={order_price}  "
              f"side={side_str}  before t={before_timestamp:.6f}")
        print(f"  Showing {len(df)} events (chronological):\n")

        for _, row in df.iterrows():
            bbo_str = (f"BBO=[{row['best_bid']:.2f}, "
                       f"{row['best_ask']:.2f}]")
            print(
                f"  rid={int(row['rid']):>10d}  t={row['timestamp']:.6f}  "
                f"{row['event_type']:3s}  id={int(row['order_id'])}  "
                f"vol={row['volume']:.0f}  {bbo_str}  "
                f"ticks_from_best={row['ticks_from_best']:.0f}"
            )

        lo_rows = df[df["event_type"] == "LO"]
        cxl_rows = df[df["event_type"] == "CXL"]
        if not lo_rows.empty:
            first = lo_rows.iloc[0]
            print(f"\n  First LO placement: t={first['timestamp']:.6f}  "
                  f"order_id={int(first['order_id'])}  "
                  f"volume={first['volume']:.0f}  "
                  f"ticks_from_best={first['ticks_from_best']:.0f}")
        still_live = len(lo_rows) - len(cxl_rows)
        if still_live > 0:
            print(f"  Uncancelled LOs at this price: {still_live}")
        return df

    # --- Mean-reversion signature & signal-conditioned MO reaction ---
    @staticmethod
    def _resil_signal_defaults():
        """Calibrated placement-mechanism parameters (resiliency_calibration).

        Reads ``data/resiliency_calibration.json`` (written by
        ``notebooks/resiliency_calibration.ipynb``); falls back to the last
        known calibrated values when the file is missing.
        """
        defaults = {"tau_s": 4.5, "flow_tau_s": 32.3,
                    "kappa": 0.0817, "phi": 0.0746}
        try:
            path = resolve_data_path("resiliency_calibration.json")
            if path.exists():
                with open(path) as f:
                    cal = json.load(f)
                defaults = {"tau_s": float(cal["resil_tau_s"]),
                            "flow_tau_s": float(cal["resil_flow_tau_s"]),
                            "kappa": float(cal["resil_kappa"]),
                            "phi": float(cal["resil_phi"])}
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        return defaults

    @staticmethod
    def _iso_to_ns(ts_series):
        """ISO-8601 (or numeric) timestamp Series -> int64 POSIX nanoseconds."""
        if pd.api.types.is_numeric_dtype(ts_series):
            return ts_series.to_numpy(dtype="int64")
        try:
            return pd.to_datetime(ts_series, format="ISO8601",
                                  utc=True).astype("int64").to_numpy()
        except ValueError:
            return pd.to_datetime(ts_series,
                                  utc=True).astype("int64").to_numpy()

    def _load_mid_mo_panel(self, n_days=50,
                           session_s=7.8 * 3600.0, warmup_s=None,
                           tail_s=None, verbose=True):
        """Load per-unit mid series (+ signed MOs) for panel statistics.

        A *unit* is the aggregation cell for across-unit (Fama-MacBeth style)
        means and standard errors:

        * empirical day-partitioned DB — one unit per trading day
          (``n_days`` selects the first *n* days in the DB, or a single day
          when ``self.day`` is set; the whole selection is fetched in one scan);
        * simulation DB — the run is split into consecutive sessions of
          ``session_s`` seconds (evidence-notebook convention).

        Returns a list of dicts with keys ``label``, ``t`` (seconds), ``mid``
        (PLN), ``mo_t``, ``mo_eps`` (+1 buy / -1 sell) and the analysis
        window ``t_lo``/``t_hi`` (mid history outside the window is kept so
        EMA signals can warm up). Default warmup/tail: 600 s / 120 s for
        empirical days, 60 s / 0 s for simulation runs.

        Loaded panels are cached on the instance (the empirical scan of a
        multi-GB DB takes minutes), so consecutive calls with the same
        selection are free.
        """
        if warmup_s is None:
            if self.has_day:
                warmup_s = 600.0
            else:
                warmup_s = 60.0
        if tail_s is None:
            if self.has_day:
                tail_s = 120.0
            else:
                tail_s = 0.0

        if not hasattr(self, "_panel_cache"):
            self._panel_cache = {}
        key = (n_days, float(session_s), float(warmup_s),
               float(tail_s), self.day)
        cached = self._panel_cache.get(key)
        if cached is not None:
            return cached

        conn = self._conn()
        units = []
        try:
            mo_cols = (self._table_cols(conn, "mo_orders")
                       if "mo_orders" in self.tables else [])
            if "first_time_ns" in mo_cols:
                mo_ts_col = "first_time_ns"
            else:
                mo_ts_col = "timestamp"

            if self.has_day:
                all_days = [r[0] for r in conn.execute(
                    "SELECT DISTINCT day FROM orders ORDER BY day")]
                if self.day:
                    days = [self.day]
                else:
                    days = all_days[:n_days]
                clause = ",".join(f"'{d}'" for d in days)
                if verbose:
                    print(f"Loading {len(days)} day(s) of mid/MO data "
                          f"(single scan of {self.db_path.name}) ...",
                          flush=True)
                orders = pd.read_sql_query(
                    f"SELECT day, timestamp, best_bid, best_ask FROM orders "
                    f"WHERE day IN ({clause}) AND best_bid > 0 "
                    f"AND best_ask > 0 AND best_ask >= best_bid "
                    f"ORDER BY day, timestamp", conn)
                if "mo_orders" in self.tables:
                    mo_all = pd.read_sql_query(
                        f"SELECT day, {mo_ts_col} AS ts, side FROM mo_orders "
                        f"WHERE day IN ({clause}) ORDER BY day, ts", conn)
                else:
                    mo_all = pd.DataFrame(columns=["day", "ts", "side"])

                for day in days:
                    o = orders[orders["day"] == day]
                    if len(o) < 1000:
                        continue
                    ns = self._iso_to_ns(o["timestamp"])
                    t = (ns - ns[0]) / 1e9
                    mid = 0.5 * (o["best_bid"].to_numpy(float)
                                 + o["best_ask"].to_numpy(float))
                    m = mo_all[mo_all["day"] == day]
                    mo_t = (m["ts"].to_numpy(np.int64) - ns[0]) / 1e9
                    mo_eps = np.where(
                        m["side"].astype(str).str.lower().str.startswith("b"),
                        1.0, -1.0)
                    units.append(dict(
                        label=day, t=t, mid=mid, mo_t=mo_t, mo_eps=mo_eps,
                        t_lo=float(warmup_s),
                        t_hi=float(t[-1] - tail_s)))
            else:
                if self.has_bbo:
                    mid_table = "bbo"
                else:
                    mid_table = "orders"
                rows = conn.execute(
                    f"SELECT timestamp, best_bid, best_ask FROM {mid_table} "
                    f"WHERE best_bid > 0 AND best_ask > 0 "
                    f"AND best_ask >= best_bid ORDER BY timestamp").fetchall()
                arr = np.asarray(rows, float)
                t_all = arr[:, 0]
                mid_all = 0.5 * (arr[:, 1] + arr[:, 2])
                if "mo_orders" in self.tables:
                    mo_rows = conn.execute(
                        f"SELECT {mo_ts_col}, side FROM mo_orders "
                        f"ORDER BY {mo_ts_col}").fetchall()
                    mo_t_all = np.asarray([r[0] for r in mo_rows], float)
                    if mo_t_all.size and np.nanmedian(mo_t_all) > 1e12:
                        mo_t_all = mo_t_all / 1e9  # ns clock -> seconds
                    mo_eps_all = np.asarray(
                        [1.0 if str(r[1]).lower().startswith("b") else -1.0
                         for r in mo_rows])
                else:
                    mo_t_all = np.array([])
                    mo_eps_all = np.array([])

                k = 0
                while True:
                    a = warmup_s + k * session_s
                    b = a + session_s
                    if b > float(t_all[-1]):
                        break
                    m_mask = (t_all >= a) & (t_all <= b)
                    if m_mask.sum() > 500:
                        mo_mask = (mo_t_all >= a) & (mo_t_all <= b)
                        units.append(dict(
                            label=f"session{k:02d}",
                            t=t_all[m_mask], mid=mid_all[m_mask],
                            mo_t=mo_t_all[mo_mask],
                            mo_eps=mo_eps_all[mo_mask],
                            t_lo=float(a), t_hi=float(b - tail_s)))
                    k += 1
        finally:
            conn.close()

        if verbose:
            n_mo = sum(len(u["mo_t"]) for u in units)
            if self.has_day:
                unit_word = "days"
            else:
                unit_word = "sessions"
            print(f"  {len(units)} {unit_word}, "
                  f"{sum(len(u['t']) for u in units):,} mid points, "
                  f"{n_mo:,} MOs", flush=True)
        self._panel_cache = {key: units}
        return units

    @staticmethod
    def _fm_stats(stat):
        """NaN-aware Fama-MacBeth mean / SE / t over axis 0 (units)."""
        stat = np.asarray(stat, float)
        mean = np.nanmean(stat, axis=0)
        n_eff = np.sum(np.isfinite(stat), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            se = (np.nanstd(stat, axis=0, ddof=1)
                  / np.sqrt(np.maximum(n_eff, 1)))
            t = mean / np.where(se > 0, se, np.nan)
        return mean, se, t

    def signature_plot(self, taus=None, normalize=True, n_days=50,
                       session_s=7.8 * 3600.0, warmup_s=None,
                       tail_s=None, compare_to=None, plot=True):
        """(Normalized) signature plot with across-unit SE bands.

        Computes the realized-variance rate ``C(tau)`` on a 1-s LOCF mid grid
        per unit (trading day for empirical DBs, 7.8 h session for
        simulation DBs), normalizes each unit's curve by its own ``C(1 s)``
        (``VR(tau) = C(tau)/C(1 s)``, the evidence-notebook methodology) and
        averages across units. ``VR < 1`` at a horizon means mean reversion;
        ``> 1`` means momentum.

        Parameters
        ----------
        taus : array-like or None
            Horizons in seconds (default dense ``1..300``). The first value
            is the normalization anchor.
        normalize : bool
            If False, plot raw ``C(tau)`` in ticks^2/min instead of VR.
        compare_to : str, Path or None
            Optional npz overlay, e.g. the 50-day empirical cache
            ``'_cache_signature_multiday_dense_300s.npz'`` when analysing a
            simulation DB. Accepts ``taus`` + per-day ``emp_linear`` curves
            or ``taus`` + ``vr_mean``/``vr_se``.

        Returns
        -------
        dict with ``taus``, ``vr_mean``, ``vr_se``, per-unit ``curves``
        (raw C in PLN^2/s), ``dip_min``, ``dip_argmin_s``, ``n_units``.
        """
        from .mean_reversion_metrics import signature_curve

        if taus is None:
            taus = np.arange(1, 301, dtype=float)
        taus = np.asarray(taus, float)

        units = self._load_mid_mo_panel(
            n_days=n_days, session_s=session_s,
            warmup_s=warmup_s, tail_s=tail_s)

        curves = []
        labels = []
        for u in units:
            m = (u["t"] >= u["t_lo"]) & (u["t"] <= u["t_hi"])
            if m.sum() < 500:
                continue
            c = signature_curve(u["t"][m], u["mid"][m], taus)
            if np.all(np.isfinite(c)) and c[0] > 0:
                curves.append(c)
                labels.append(u["label"])
        if not curves:
            print("No usable units for the signature plot.")
            return None
        curves = np.asarray(curves, float)

        if normalize:
            panel = curves / curves[:, [0]]
            ylabel = f"VR = C(τ) / C({taus[0]:g} s)"
        else:
            panel = curves * (60.0 / self.tick_size ** 2)
            ylabel = "C(τ)  (ticks²/min)"
        vr_mean, vr_se, _ = self._fm_stats(panel)

        dip_band = (taus >= 2) & (taus <= 15)
        if dip_band.any():
            dip_min = float(vr_mean[dip_band].min())
        else:
            dip_min = np.nan
        if dip_band.any():
            dip_arg = float(taus[dip_band][np.argmin(vr_mean[dip_band])])
        else:
            dip_arg = np.nan

        overlay = None
        if compare_to is not None:
            z = np.load(str(resolve_data_path(compare_to)),
                        allow_pickle=True)
            o_taus = np.asarray(z["taus"], float)
            if "emp_linear" in z.files:
                emp = np.asarray(z["emp_linear"], float)
                o_panel = (emp / emp[:, [0]] if normalize
                           else emp * (60.0 / self.tick_size ** 2))
                o_mean, o_se, _ = self._fm_stats(o_panel)
            else:
                o_mean = np.asarray(z["vr_mean"], float)
                o_se = np.asarray(z.get("vr_se",
                                        np.zeros_like(o_mean)), float)
            keep = o_taus <= taus.max()
            overlay = (o_taus[keep], o_mean[keep], o_se[keep])

        if plot:
            if self.has_day:
                src = "days"
            else:
                src = "sessions"
            fig, ax = plt.subplots(figsize=(11, 4.6))
            ax.plot(taus, vr_mean, lw=1.8, color="tab:red",
                    label=f"{self.db_path.name}  "
                          f"({len(curves)} {src})")
            ax.fill_between(taus, vr_mean - vr_se, vr_mean + vr_se,
                            color="tab:red", alpha=0.18, lw=0)
            if overlay is not None:
                ax.plot(overlay[0], overlay[1], lw=1.8, color="tab:blue",
                        label=f"overlay: {Path(str(compare_to)).name}")
                ax.fill_between(overlay[0], overlay[1] - overlay[2],
                                overlay[1] + overlay[2],
                                color="tab:blue", alpha=0.18, lw=0)
            if normalize:
                ax.axhline(1.0, color="black", lw=0.8)
            ax.set_xlabel("τ (s)")
            ax.set_ylabel(ylabel)
            ax.set_title("Signature plot"
                         + (" (normalized)" if normalize else ""))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        key_taus = [3, 5, 7, 10, 20, 30, 50, 75, 100, 150, 300]
        print(f"{'tau':>5} {'mean':>9} {'SE':>8}")
        for kt in key_taus:
            if kt > taus.max():
                continue
            i = int(np.argmin(np.abs(taus - kt)))
            print(f"{kt:>5} {vr_mean[i]:>9.4f} {vr_se[i]:>8.4f}")
        if normalize:
            print(f"dip min (2-15 s): {dip_min:.4f} at τ={dip_arg:g} s")

        return dict(taus=taus, vr_mean=vr_mean, vr_se=vr_se,
                    curves=curves, labels=labels, dip_min=dip_min,
                    dip_argmin_s=dip_arg, n_units=len(curves))

    def signal_conditional_markout(
            self, horizons=(1, 2, 3, 5, 7, 10, 15, 20, 30, 60, 120),
            tau_s=None, flow_tau_s=None, kappa=None, phi=None, xcap=4.0,
            n_days=50, session_s=7.8 * 3600.0, warmup_s=None,
            tail_s=None, base="post", signal_shift_s=0.0,
            dose_horizons=(10.0, 60.0), n_bins=5, min_mo_per_unit=50,
            plot=True):
        """Post-MO price reaction conditioned on the placement signal (d, s).

        Empirical validation of the state-dependent LO-placement mechanism at
        the *price* level: at each market order, sample the pre-MO band-pass
        displacement ``d`` and trend ``s`` (the same three-EMA signals that
        drive ``resil_kappa`` / ``resil_phi`` in the simulator), then measure
        the **signed markout** ``y(h) = eps * (mid(t0+h) - mid_base)`` in
        ticks, where ``eps`` is the MO direction (+1 buy / -1 sell).
        ``y > 0`` = the MO's move continues, ``y < 0`` = it reverts.

        The mechanism (conditional drift ``~ -(kappa*d - phi*s)``) predicts:

        * slope of ``y(h)`` on ``eps*d``: **negative** at ``h <~ 10-15 s``,
          washing out to ~0 by ``h ~ 60-120 s`` (zero-integral band-pass);
        * slope on ``eps*s``: **positive**, growing over ``h ~ 20-120 s``;
        * MOs classified "revert" by the calibrated tilt
          (``eps * (kappa*c(d) - phi*c(s)) > 0``) underperform MOs classified
          "continue".

        All statistics are per-unit (day / session) and aggregated
        Fama-MacBeth style (mean +/- SE across units), so serial correlation
        within a day never inflates the t-stats.

        Parameters
        ----------
        horizons : sequence of float
            Markout horizons in seconds.
        tau_s, flow_tau_s, kappa, phi : float or None
            Signal timescales and tilt amplitudes. Default (None) loads the
            calibrated values from ``data/resiliency_calibration.json``.
        base : {'post', 'pre'}
            Markout base mid: first event after the MO (pure reaction,
            default) or last event before it (includes the MO's own impact).
        signal_shift_s : float
            Placebo control: sample the signal this many seconds *before*
            the MO (e.g. 300). Genuine conditioning must collapse.
        dose_horizons : pair of float
            Horizons shown in the quantile dose-response panels.
        n_bins : int
            Number of quantile bins in the dose-response panels.

        Returns
        -------
        dict with the slope term structure (``b_d``, ``b_s`` +/- SE / t),
        tilt-classification means and difference, quadrant means, quantile
        dose-response curves, and bookkeeping (``n_units``, ``n_mo``).
        """
        from .empirical_placement_mle import _compute_signals

        cal = self._resil_signal_defaults()
        if tau_s is None:
            tau_s = cal["tau_s"]
        else:
            tau_s = float(tau_s)
        if flow_tau_s is None:
            flow_tau_s = cal["flow_tau_s"]
        else:
            flow_tau_s = float(flow_tau_s)
        if kappa is None:
            kappa = cal["kappa"]
        else:
            kappa = float(kappa)
        if phi is None:
            phi = cal["phi"]
        else:
            phi = float(phi)

        horizons = np.asarray(sorted(horizons), float)
        dose_horizons = [float(h) for h in dose_horizons]
        dose_idx = [int(np.argmin(np.abs(horizons - h)))
                    for h in dose_horizons]
        nH = len(horizons)
        h_max = float(horizons.max())

        units = self._load_mid_mo_panel(
            n_days=n_days, session_s=session_s,
            warmup_s=warmup_s, tail_s=tail_s)

        def _clip(x):
            return np.clip(x, -xcap, xcap)

        u_slopes, u_cls, u_quads = [], [], []
        u_dose_d, u_dose_s = [], []
        u_n = []
        q_grid = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]

        for u in units:
            t = np.asarray(u["t"], np.float64)
            mid_ticks = np.asarray(u["mid"], np.float64) / self.tick_size
            dt_arr = np.empty(len(t))
            dt_arr[0] = 0.0
            dt_arr[1:] = np.diff(t)
            d_sig, s_sig = _compute_signals(dt_arr, mid_ticks,
                                            tau_s, flow_tau_s)

            mo_t = np.asarray(u["mo_t"], float)
            eps = np.asarray(u["mo_eps"], float)
            idx_sig = np.searchsorted(t, mo_t - signal_shift_s,
                                      side="left") - 1
            idx_post = np.searchsorted(t, mo_t, side="right")
            # EMA warm-up guard: signals need ~3 slow time constants.
            t_lo_sig = max(u["t_lo"], float(t[0]) + 3.0 * flow_tau_s)
            valid = ((mo_t >= t_lo_sig) & (mo_t <= u["t_hi"] - h_max)
                     & (idx_sig >= 0) & (idx_post < len(t)))
            if valid.sum() < min_mo_per_unit:
                continue
            mo_t, eps = mo_t[valid], eps[valid]
            idx_sig, idx_post = idx_sig[valid], idx_post[valid]

            xd = eps * _clip(d_sig[idx_sig])
            xs = eps * _clip(s_sig[idx_sig])
            zeta = kappa * _clip(d_sig[idx_sig]) - phi * _clip(s_sig[idx_sig])
            pred_revert = (eps * zeta) > 0
            m_base = (mid_ticks[idx_post] if base == "post"
                      else mid_ticks[np.searchsorted(t, mo_t, "left") - 1])

            X = np.column_stack([np.ones_like(xd), xd, xs])
            beta_op = np.linalg.pinv(X)

            edges_d = np.quantile(xd, q_grid)
            edges_s = np.quantile(xs, q_grid)
            bin_d = np.searchsorted(edges_d, xd, side="right")
            bin_s = np.searchsorted(edges_s, xs, side="right")

            sl = np.empty((nH, 3))
            cls = np.empty((nH, 2))
            quads = np.empty((nH, 4))
            dd = np.full((len(dose_idx), n_bins), np.nan)
            ds = np.full((len(dose_idx), n_bins), np.nan)
            for j, h in enumerate(horizons):
                idx_h = np.searchsorted(t, mo_t + h, side="right") - 1
                idx_h = np.maximum(idx_h, idx_post)
                y = eps * (mid_ticks[idx_h] - m_base)
                sl[j] = beta_op @ y

                def _m(mask):
                    return float(y[mask].mean()) if mask.any() else np.nan

                cls[j, 0] = _m(pred_revert)
                cls[j, 1] = _m(~pred_revert)
                quads[j, 0] = _m((xd > 0) & (xs > 0))
                quads[j, 1] = _m((xd > 0) & (xs <= 0))
                quads[j, 2] = _m((xd <= 0) & (xs > 0))
                quads[j, 3] = _m((xd <= 0) & (xs <= 0))
                if j in dose_idx:
                    r = dose_idx.index(j)
                    for b in range(n_bins):
                        dd[r, b] = _m(bin_d == b)
                        ds[r, b] = _m(bin_s == b)

            u_slopes.append(sl)
            u_cls.append(cls)
            u_quads.append(quads)
            u_dose_d.append(dd)
            u_dose_s.append(ds)
            u_n.append(len(mo_t))

        if not u_slopes:
            print("No units with enough valid MOs.")
            return None

        sl_mean, sl_se, sl_t = self._fm_stats(np.asarray(u_slopes))
        cls_mean, cls_se, _ = self._fm_stats(np.asarray(u_cls))
        cls_arr = np.asarray(u_cls)
        diff_mean, diff_se, diff_t = self._fm_stats(
            cls_arr[:, :, 0] - cls_arr[:, :, 1])
        quad_mean, quad_se, _ = self._fm_stats(np.asarray(u_quads))
        dose_d_mean, dose_d_se, _ = self._fm_stats(np.asarray(u_dose_d))
        dose_s_mean, dose_s_se, _ = self._fm_stats(np.asarray(u_dose_s))
        n_units, n_mo = len(u_n), int(sum(u_n))

        if self.has_day:
            unit_word = "days"
        else:
            unit_word = "sessions"
        shift_note = (f"  [PLACEBO: signal shifted {signal_shift_s:g}s back]"
                      if signal_shift_s else "")
        print(f"\n{n_units} {unit_word}, {n_mo:,} MOs | "
              f"τ={tau_s:g}s τ_f={flow_tau_s:g}s κ={kappa:g} φ={phi:g} | "
              f"base={base}{shift_note}")
        print(f"\n{'h(s)':>6} {'b_d':>9} {'t':>6} {'b_s':>9} {'t':>6} "
              f"{'| revert':>9} {'continue':>9} {'diff':>8} {'t':>6}")
        for j, h in enumerate(horizons):
            print(f"{h:>6g} {sl_mean[j, 1]:>9.4f} {sl_t[j, 1]:>6.1f} "
                  f"{sl_mean[j, 2]:>9.4f} {sl_t[j, 2]:>6.1f} "
                  f"| {cls_mean[j, 0]:>7.4f} {cls_mean[j, 1]:>9.4f} "
                  f"{diff_mean[j]:>8.4f} {diff_t[j]:>6.1f}")

        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))

            ax = axes[0, 0]
            for col, lbl, colr in ((1, "b_d  (displacement, κ-channel)",
                                    "tab:red"),
                                   (2, "b_s  (trend, φ-channel)",
                                    "tab:blue")):
                ax.plot(horizons, sl_mean[:, col], "o-", ms=4, color=colr,
                        label=lbl)
                ax.fill_between(horizons,
                                sl_mean[:, col] - sl_se[:, col],
                                sl_mean[:, col] + sl_se[:, col],
                                color=colr, alpha=0.18, lw=0)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("horizon h (s)")
            ax.set_ylabel("slope (ticks / tick of signal)")
            ax.set_title("Signed markout ~ ε·d + ε·s   (per-horizon OLS)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            ax.plot(horizons, cls_mean[:, 0], "o-", ms=4, color="tab:green",
                    label="model says revert (ε·ζ > 0)")
            ax.plot(horizons, cls_mean[:, 1], "o-", ms=4, color="tab:orange",
                    label="model says continue (ε·ζ ≤ 0)")
            for col, colr in ((0, "tab:green"), (1, "tab:orange")):
                ax.fill_between(horizons,
                                cls_mean[:, col] - cls_se[:, col],
                                cls_mean[:, col] + cls_se[:, col],
                                color=colr, alpha=0.18, lw=0)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("horizon h (s)")
            ax.set_ylabel("mean signed markout (ticks)")
            ax.set_title("Reaction by calibrated tilt sign")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            for ax, dm, dse, sig_lbl in (
                    (axes[1, 0], dose_d_mean, dose_d_se,
                     "ε·d (displacement)"),
                    (axes[1, 1], dose_s_mean, dose_s_se, "ε·s (trend)")):
                x = np.arange(1, n_bins + 1)
                for r, h in enumerate(dose_horizons):
                    ax.errorbar(x, dm[r], yerr=dse[r], fmt="o-", ms=4,
                                capsize=3, label=f"h = {h:g} s")
                ax.axhline(0, color="black", lw=0.8)
                ax.set_xticks(x)
                ax.set_xlabel(f"quantile bin of {sig_lbl}  (low → high)")
                ax.set_ylabel("mean signed markout (ticks)")
                ax.set_title(f"Dose-response in {sig_lbl}")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            fig.suptitle(
                f"Signal-conditioned post-MO reaction — "
                f"{self.db_path.name} ({n_units} {unit_word}, "
                f"{n_mo:,} MOs){shift_note}", fontsize=11)
            plt.tight_layout()
            plt.show()

        return dict(
            horizons=horizons,
            b_d=sl_mean[:, 1], b_d_se=sl_se[:, 1], b_d_t=sl_t[:, 1],
            b_s=sl_mean[:, 2], b_s_se=sl_se[:, 2], b_s_t=sl_t[:, 2],
            intercept=sl_mean[:, 0],
            cls_revert=cls_mean[:, 0], cls_continue=cls_mean[:, 1],
            cls_diff=diff_mean, cls_diff_se=diff_se, cls_diff_t=diff_t,
            quad_mean=quad_mean, quad_se=quad_se,
            dose_horizons=dose_horizons,
            dose_d_mean=dose_d_mean, dose_d_se=dose_d_se,
            dose_s_mean=dose_s_mean, dose_s_se=dose_s_se,
            params=dict(tau_s=tau_s, flow_tau_s=flow_tau_s, kappa=kappa,
                        phi=phi, xcap=xcap, base=base,
                        signal_shift_s=signal_shift_s),
            n_units=n_units, n_mo=n_mo)

    def signal_pre_move_tick(
            self, tau_s=None, flow_tau_s=None, kappa=None, phi=None,
            xcap=4.0, n_days=50, session_s=7.8 * 3600.0, warmup_s=None,
            tail_s=None, one_tick_only=True, signal_shift_s=0.0, n_bins=5,
            min_moves_per_unit=100, plot=True):
        """Pre-move signal vs immediate mid tick direction.

        At each mid-changing event, read the band-pass displacement ``d`` and
        trend ``s`` at the **last event before the move** (index ``i-1`` when
        ``mid[i] != mid[i-1]``), then test whether the signed tick move
        ``y = mid[i] - mid[i-1]`` aligns with the resiliency mechanism:

        * slope on ``d``: **negative** (displacement → revert on next tick);
        * slope on ``s``: **positive** (trend → continue).

        Default ``one_tick_only=True`` keeps only ±1-tick steps. Statistics
        are aggregated Fama-MacBeth across days / sessions.

        Parameters
        ----------
        one_tick_only : bool
            If True (default), restrict to moves of exactly one tick.
        signal_shift_s : float
            Placebo: sample the signal this many seconds before ``t[i-1]``.
        min_moves_per_unit : int
            Minimum mid moves per unit to include in the panel.

        Returns
        -------
        dict with OLS slopes ``b_d``, ``b_s`` (+ SE / t), quantile
        dose-response curves, tilt-classification means, ``n_units``,
        ``n_moves``.
        """
        from .empirical_placement_mle import _compute_signals

        cal = self._resil_signal_defaults()
        if tau_s is None:
            tau_s = cal["tau_s"]
        else:
            tau_s = float(tau_s)
        if flow_tau_s is None:
            flow_tau_s = cal["flow_tau_s"]
        else:
            flow_tau_s = float(flow_tau_s)
        if kappa is None:
            kappa = cal["kappa"]
        else:
            kappa = float(kappa)
        if phi is None:
            phi = cal["phi"]
        else:
            phi = float(phi)

        units = self._load_mid_mo_panel(
            n_days=n_days, session_s=session_s,
            warmup_s=warmup_s, tail_s=tail_s)

        def _clip(x):
            return np.clip(x, -xcap, xcap)

        u_slopes, u_cls, u_dose_d, u_dose_s, u_n = [], [], [], [], []
        q_grid = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]

        for u in units:
            t = np.asarray(u["t"], np.float64)
            mid_ticks = np.rint(np.asarray(u["mid"], np.float64)
                                / self.tick_size)
            dt_arr = np.empty(len(t))
            dt_arr[0] = 0.0
            dt_arr[1:] = np.diff(t)
            d_sig, s_sig = _compute_signals(dt_arr, mid_ticks.astype(np.float64),
                                            tau_s, flow_tau_s)

            d_mid = np.diff(mid_ticks)
            move_idx = np.where(d_mid != 0.0)[0] + 1
            if one_tick_only:
                move_idx = move_idx[np.abs(d_mid[move_idx - 1]) == 1.0]
            if move_idx.size == 0:
                continue

            t_lo_sig = max(u["t_lo"], float(t[0]) + 3.0 * flow_tau_s)
            in_win = ((t[move_idx] >= t_lo_sig)
                      & (t[move_idx] <= u["t_hi"]))
            move_idx = move_idx[in_win]
            if move_idx.size < min_moves_per_unit:
                continue

            if signal_shift_s > 0.0:
                t_sig = t[move_idx - 1] - float(signal_shift_s)
                sig_idx = np.searchsorted(t, t_sig, side="left") - 1
            else:
                sig_idx = move_idx - 1
            ok = sig_idx >= 0
            move_idx, sig_idx = move_idx[ok], sig_idx[ok]
            if move_idx.size < min_moves_per_unit:
                continue

            y = mid_ticks[move_idx] - mid_ticks[move_idx - 1]
            xd = _clip(d_sig[sig_idx])
            xs = _clip(s_sig[sig_idx])
            zeta = kappa * xd - phi * xs
            pred_revert = (y * zeta) > 0.0

            X = np.column_stack([np.ones(len(y)), xd, xs])
            sl = np.linalg.pinv(X) @ y

            edges_d = np.quantile(xd, q_grid)
            edges_s = np.quantile(xs, q_grid)
            bin_d = np.searchsorted(edges_d, xd, side="right")
            bin_s = np.searchsorted(edges_s, xs, side="right")

            def _m(mask):
                return float(y[mask].mean()) if mask.any() else np.nan

            cls = np.array([_m(pred_revert), _m(~pred_revert)])
            dd = np.array([_m(bin_d == b) for b in range(n_bins)])
            ds = np.array([_m(bin_s == b) for b in range(n_bins)])

            u_slopes.append(sl)
            u_cls.append(cls)
            u_dose_d.append(dd)
            u_dose_s.append(ds)
            u_n.append(len(y))

        if not u_slopes:
            print("No units with enough valid mid moves.")
            return None

        sl_arr = np.asarray(u_slopes)
        sl_mean, sl_se, sl_t = self._fm_stats(sl_arr)
        cls_arr = np.asarray(u_cls)
        cls_mean, cls_se, _ = self._fm_stats(cls_arr)
        diff_mean, diff_se, diff_t = self._fm_stats(
            cls_arr[:, 0] - cls_arr[:, 1])
        dose_d_mean, dose_d_se, _ = self._fm_stats(np.asarray(u_dose_d))
        dose_s_mean, dose_s_se, _ = self._fm_stats(np.asarray(u_dose_s))
        n_units, n_moves = len(u_n), int(sum(u_n))

        if self.has_day:
            unit_word = "days"
        else:
            unit_word = "sessions"
        if one_tick_only:
            tick_note = "±1 tick"
        else:
            tick_note = "any tick size"
        shift_note = (f"  [PLACEBO: signal shifted {signal_shift_s:g}s back]"
                      if signal_shift_s else "")
        print(f"\n{n_units} {unit_word}, {n_moves:,} mid moves ({tick_note}) | "
              f"τ={tau_s:g}s τ_f={flow_tau_s:g}s κ={kappa:g} φ={phi:g}"
              f"{shift_note}")
        print(f"\n{'coef':>8} {'mean':>9} {'SE':>8} {'t':>6}")
        for lbl, j in (("b_d (d)", 1), ("b_s (s)", 2)):
            print(f"{lbl:>8} {sl_mean[j]:>9.4f} {sl_se[j]:>8.4f} "
                  f"{sl_t[j]:>6.1f}")
        print(f"{'revert':>8} {cls_mean[0]:>9.4f} {cls_se[0]:>8.4f} | "
              f"continue {cls_mean[1]:.4f}  diff {diff_mean:.4f} "
              f"(t={diff_t:.1f})")

        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))

            ax = axes[0, 0]
            labels = ["b_d\n(displacement)", "b_s\n(trend)"]
            cols = [1, 2]
            colors = ["tab:red", "tab:blue"]
            xpos = np.arange(2)
            for k, (j, lbl, colr) in enumerate(zip(cols, labels, colors)):
                ax.bar(xpos[k], sl_mean[j], yerr=sl_se[j], capsize=4,
                       color=colr, alpha=0.85, label=lbl)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(xpos)
            ax.set_xticklabels(labels)
            ax.set_ylabel("slope (ticks / tick of signal)")
            ax.set_title("Signed tick move ~ d + s  (immediate next step)")
            ax.grid(True, alpha=0.3, axis="y")

            ax = axes[0, 1]
            ax.bar([0, 1], cls_mean, yerr=cls_se, capsize=4,
                   color=["tab:green", "tab:orange"], alpha=0.85)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["y·ζ > 0", "y·ζ ≤ 0"])
            ax.set_ylabel("mean signed tick move")
            ax.set_title("Mean move by calibrated tilt sign")
            ax.grid(True, alpha=0.3, axis="y")

            for ax, dm, dse, sig_lbl in (
                    (axes[1, 0], dose_d_mean, dose_d_se, "d (displacement)"),
                    (axes[1, 1], dose_s_mean, dose_s_se, "s (trend)")):
                x = np.arange(1, n_bins + 1)
                ax.errorbar(x, dm, yerr=dse, fmt="o-", ms=5, capsize=3)
                ax.axhline(0, color="black", lw=0.8)
                ax.set_xticks(x)
                ax.set_xlabel(f"quantile bin of {sig_lbl}  (low → high)")
                ax.set_ylabel("mean signed tick move")
                ax.set_title(f"Dose-response in {sig_lbl}")
                ax.grid(True, alpha=0.3)

            fig.suptitle(
                f"Pre-move signal vs immediate tick — "
                f"{self.db_path.name} ({n_units} {unit_word}, "
                f"{n_moves:,} moves){shift_note}", fontsize=11)
            plt.tight_layout()
            plt.show()

        return dict(
            b_d=float(sl_mean[1]), b_d_se=float(sl_se[1]),
            b_d_t=float(sl_t[1]),
            b_s=float(sl_mean[2]), b_s_se=float(sl_se[2]),
            b_s_t=float(sl_t[2]),
            intercept=float(sl_mean[0]),
            cls_revert=float(cls_mean[0]), cls_continue=float(cls_mean[1]),
            cls_diff=float(diff_mean), cls_diff_se=float(diff_se),
            cls_diff_t=float(diff_t),
            dose_d_mean=dose_d_mean, dose_d_se=dose_d_se,
            dose_s_mean=dose_s_mean, dose_s_se=dose_s_se,
            params=dict(tau_s=tau_s, flow_tau_s=flow_tau_s, kappa=kappa,
                        phi=phi, xcap=xcap, one_tick_only=one_tick_only,
                        signal_shift_s=signal_shift_s),
            n_units=n_units, n_moves=n_moves)

    def _load_lo_panel(self, n_days=50, session_s=7.8 * 3600.0,
                       warmup_s=None, tail_s=None, verbose=True):
        """Load per-unit BBO stream + limit-order fields for LO-regime stats.

        One scan per trading day (cached on ``self._lo_panel_cache``). Each
        unit dict has ``t``/``mid`` for signal warm-up, LO arrays
        (``lo_t``, ``lo_side``, ``lo_spread``, ``lo_tfb``) and ``t_lo``/``t_hi``.
        """
        if warmup_s is None:
            if self.has_day:
                warmup_s = 600.0
            else:
                warmup_s = 60.0
        if tail_s is None:
            if self.has_day:
                tail_s = 120.0
            else:
                tail_s = 0.0

        if not hasattr(self, "_lo_panel_cache"):
            self._lo_panel_cache = {}
        key = (n_days, float(session_s), float(warmup_s),
               float(tail_s), self.day)
        cached = self._lo_panel_cache.get(key)
        if cached is not None:
            return cached

        conn = self._conn()
        units = []
        lo_sql_cols = (
            "timestamp, best_bid, best_ask, event_type, is_cancel, side, "
            "spread_ticks, ticks_from_best")
        try:
            if self.has_day:
                all_days = [r[0] for r in conn.execute(
                    "SELECT DISTINCT day FROM orders ORDER BY day")]
                if self.day:
                    days = [self.day]
                else:
                    days = all_days[:n_days]
                if verbose:
                    print(f"Loading {len(days)} day(s) of LO/BBO data "
                          f"(single scan of {self.db_path.name}) ...",
                          flush=True)
                for day in days:
                    df = pd.read_sql_query(
                        f"SELECT {lo_sql_cols} FROM orders WHERE day=? "
                        f"AND best_bid > 0 AND best_ask > 0 "
                        f"AND best_ask >= best_bid ORDER BY timestamp",
                        conn, params=(day,))
                    if len(df) < 1000:
                        continue
                    ns = self._iso_to_ns(df["timestamp"])
                    t = (ns - ns[0]) / 1e9
                    mid = 0.5 * (df["best_bid"].to_numpy(float)
                                 + df["best_ask"].to_numpy(float))
                    lo = df[
                        (df["event_type"] == "LO")
                        & (df["is_cancel"] == 0)
                        & df["spread_ticks"].notna()
                        & df["ticks_from_best"].notna()
                        & df["side"].isin([1, 2])
                    ]
                    if len(lo) < 100:
                        continue
                    lo_ns = self._iso_to_ns(lo["timestamp"])
                    lo_t = (lo_ns - ns[0]) / 1e9
                    units.append(dict(
                        label=day, t=t, mid=mid,
                        lo_t=lo_t,
                        lo_side=lo["side"].to_numpy(np.int8),
                        lo_spread=lo["spread_ticks"].to_numpy(np.int32),
                        lo_tfb=lo["ticks_from_best"].to_numpy(np.int32),
                        t_lo=float(warmup_s),
                        t_hi=float(t[-1] - tail_s)))
            else:
                df = pd.read_sql_query(
                    f"SELECT {lo_sql_cols} FROM orders "
                    f"WHERE best_bid > 0 AND best_ask > 0 "
                    f"AND best_ask >= best_bid ORDER BY timestamp", conn)
                if len(df) < 1000:
                    if verbose:
                        print("  insufficient rows for LO panel", flush=True)
                else:
                    ns = self._iso_to_ns(df["timestamp"])
                    t_all = (ns - ns[0]) / 1e9
                    mid_all = 0.5 * (df["best_bid"].to_numpy(float)
                                     + df["best_ask"].to_numpy(float))
                    k = 0
                    while True:
                        a = warmup_s + k * session_s
                        b = a + session_s
                        if b > float(t_all[-1]):
                            break
                        m = (t_all >= a) & (t_all <= b)
                        if m.sum() <= 500:
                            k += 1
                            continue
                        sub = df.loc[m]
                        lo = sub[
                            (sub["event_type"] == "LO")
                            & (sub["is_cancel"] == 0)
                            & sub["spread_ticks"].notna()
                            & sub["ticks_from_best"].notna()
                            & sub["side"].isin([1, 2])
                        ]
                        if len(lo) >= 100:
                            lo_ns = self._iso_to_ns(lo["timestamp"])
                            lo_t = (lo_ns - ns[0]) / 1e9
                            units.append(dict(
                                label=f"session{k:02d}",
                                t=t_all[m], mid=mid_all[m],
                                lo_t=lo_t,
                                lo_side=lo["side"].to_numpy(np.int8),
                                lo_spread=lo["spread_ticks"].to_numpy(
                                    np.int32),
                                lo_tfb=lo["ticks_from_best"].to_numpy(
                                    np.int32),
                                t_lo=float(a),
                                t_hi=float(b - tail_s)))
                        k += 1
        finally:
            conn.close()

        if verbose:
            n_lo = sum(len(u["lo_t"]) for u in units)
            if self.has_day:
                unit_word = "days"
            else:
                unit_word = "sessions"
            print(f"  {len(units)} {unit_word}, "
                  f"{sum(len(u['t']) for u in units):,} BBO points, "
                  f"{n_lo:,} LOs", flush=True)
        self._lo_panel_cache[key] = units
        return units

    @staticmethod
    def _lo_regime_labels(spread, tfb):
        """Canonical best / inside / passive from ``ticks_from_best``."""
        s = np.asarray(spread, np.int32)
        tfb = np.asarray(tfb, np.int32)
        reg = np.full(len(tfb), -1, dtype=np.int8)
        reg[tfb == 0] = 0
        reg[(tfb < 0) & (tfb > -s)] = 1
        reg[tfb > 0] = 2
        return reg

    def signal_conditional_lo_regime(
            self, tau_s=None, flow_tau_s=None, kappa=None, phi=None,
            xcap=4.0, n_days=50, warmup_s=None, tail_s=None,
            spread_bands=((2, 2), (3, 5), (6, 9), (10, 20)),
            max_spread=20, n_bins=5, min_lo_per_cell=50, min_lo_per_unit=500,
            signal_shift_s=0.0, sides=("bid", "ask"), plot=True):
        """Antisymmetric LO-placement tilt conditional on spread and signal.

        Empirical analogue of ``Simulate._resil_multiplier``. The sim's
        tilt is **rate-preserving**: at a given market state the two sides
        get ``1 ± tanh(ζ)``, so the *total* aggressive-placement rate is
        unchanged and only its **side composition** shifts. The mechanism
        is therefore a purely *antisymmetric* (between-side) prediction.

        Define the state signal in the ask frame
        ``ζ = κ·clip(d) − φ·clip(s)`` (side-independent; the bid frame is
        ``−ζ``). Own-side favorability is ``w = e·ζ`` with ``e = +1`` ask,
        ``−1`` bid. At matched state, ``P(aggr)`` decomposes as

            ``P(aggr | side) = μ(ζ)  +  e · g(ζ)``

        where ``μ`` is a **common-mode** term (both sides quote more/less
        aggressively together after a move — replenishment *or* trend
        continuation, exactly as liquidity can land on either side) and
        ``e·g(ζ)`` is the **antisymmetric mechanism** the sim implements.
        A per-side regression of ``P(aggr)`` on ``w`` conflates the two
        (``slope = g' ± c``); this method instead isolates them:

        * ``mech``  – FM slope on ``w`` from ``aggr ~ 1 + ζ + w`` (the
          mechanism ``g'``; the sim predicts ``mech > 0``);
        * ``common`` – FM slope on ``ζ`` (the co-movement ``c``);
        * ``D(ζ) = P(aggr|ask,ζ) − P(aggr|bid,ζ)`` at **matched ζ** (fixed
          global bins), whose slope is ``2·mech`` and which cancels the
          common mode exactly — the cleanest dose-response of the tilt.

        Regimes follow ``probability_vs_spread`` (``ticks_from_best``);
        signals are sampled at LO arrival (or time-shifted for placebo).
        A placebo (``signal_shift_s>0``) should flatten ``D`` and ``mech``.
        """
        from .empirical_placement_mle import _compute_signals

        cal = self._resil_signal_defaults()
        if tau_s is None:
            tau_s = cal["tau_s"]
        else:
            tau_s = float(tau_s)
        if flow_tau_s is None:
            flow_tau_s = cal["flow_tau_s"]
        else:
            flow_tau_s = float(flow_tau_s)
        if kappa is None:
            kappa = cal["kappa"]
        else:
            kappa = float(kappa)
        if phi is None:
            phi = cal["phi"]
        else:
            phi = float(phi)

        spread_bands = tuple(
            (int(a), int(b)) for a, b in spread_bands)
        band_labels = [
            f"s={lo}" if lo == hi else f"s={lo}–{hi}"
            for lo, hi in spread_bands]
        n_band = len(spread_bands)
        side_names = list(sides)

        units = self._load_lo_panel(
            n_days=n_days, warmup_s=warmup_s, tail_s=tail_s)

        def _clip(x):
            return np.clip(x, -xcap, xcap)

        # --- Pass 1: per-unit signal in the ask frame ---
        # ζ (=z_ask) is a property of the market STATE (side-independent);
        # own-side favorability is w = e·ζ with e = +1 ask / −1 bid.
        unit_data = []
        pool_spread, pool_reg = [], []
        band_pool_zeta = [[] for _ in range(n_band)]
        for u in units:
            t = np.asarray(u["t"], np.float64)
            mid_ticks = np.rint(np.asarray(u["mid"], np.float64)
                                / self.tick_size)
            dt_arr = np.empty(len(t))
            dt_arr[0] = 0.0
            dt_arr[1:] = np.diff(t)
            d_sig, s_sig = _compute_signals(
                dt_arr, mid_ticks.astype(np.float64), tau_s, flow_tau_s)

            lo_t = np.asarray(u["lo_t"], np.float64)
            lo_side = np.asarray(u["lo_side"], np.int8)
            lo_spread = np.asarray(u["lo_spread"], np.int32)
            lo_tfb = np.asarray(u["lo_tfb"], np.int32)
            reg = self._lo_regime_labels(lo_spread, lo_tfb)

            idx = np.searchsorted(t, lo_t - signal_shift_s, side="left") - 1
            t_lo_sig = max(u["t_lo"], float(t[0]) + 3.0 * flow_tau_s)
            valid = ((lo_t >= t_lo_sig) & (lo_t <= u["t_hi"])
                     & (reg >= 0) & (lo_spread <= max_spread)
                     & (idx >= 0))
            if valid.sum() < min_lo_per_unit:
                continue

            lo_side = lo_side[valid]
            lo_spread = lo_spread[valid]
            reg = reg[valid]
            idx = idx[valid]
            pool_spread.append(lo_spread)
            pool_reg.append(reg)

            # ζ = κ·clip(d) − φ·clip(s)  (matches sim's ask-side z)
            zeta = kappa * _clip(d_sig[idx]) - phi * _clip(s_sig[idx])
            e = np.where(lo_side == 2, 1.0, -1.0)   # +1 ask, −1 bid
            aggr = (reg <= 1).astype(np.float64)
            band_idx = np.full(len(zeta), -1, np.int8)
            for bi, (blo, bhi) in enumerate(spread_bands):
                bm = (lo_spread >= blo) & (lo_spread <= bhi)
                band_idx[bm] = bi
                if bm.any():
                    band_pool_zeta[bi].append(zeta[bm])
            unit_data.append(dict(zeta=zeta, e=e, aggr=aggr,
                                  band=band_idx, n=int(valid.sum())))

        if not unit_data:
            print("No units with enough valid LOs.")
            return None

        # --- Fixed global ζ-bin edges per spread band (shared across ---
        # sides and days so bid/ask are differenced at matched state) ─
        band_edges, band_centers = [], []
        for bi in range(n_band):
            allz = (np.concatenate(band_pool_zeta[bi])
                    if band_pool_zeta[bi] else np.empty(0))
            if allz.size < n_bins * 20 or np.std(allz) < 1e-12:
                band_edges.append(None)
                band_centers.append(np.full(n_bins, np.nan))
                continue
            q = np.quantile(allz, np.linspace(0.0, 1.0, n_bins + 1))
            inner = q[1:-1]
            bb = np.searchsorted(inner, allz, side="right")
            cen = np.array([allz[bb == k].mean() if np.any(bb == k)
                            else np.nan for k in range(n_bins)])
            band_edges.append(inner)
            band_centers.append(cen)

        # --- Pass 2: per-unit mechanism / common-mode / differential ---
        n_u = len(unit_data)
        u_mech = np.full((n_u, n_band), np.nan)   # slope on w (mechanism)
        u_comm = np.full((n_u, n_band), np.nan)   # slope on ζ (common-mode)
        u_D = np.full((n_u, n_band, n_bins), np.nan)   # P_ask − P_bid
        u_M = np.full((n_u, n_band, n_bins), np.nan)   # ½(P_ask + P_bid)
        u_n = []
        for ui, ud in enumerate(unit_data):
            u_n.append(ud["n"])
            for bi in range(n_band):
                m = ud["band"] == bi
                if m.sum() < min_lo_per_cell:
                    continue
                z = ud["zeta"][m]
                e = ud["e"][m]
                y = ud["aggr"][m]
                has_both = np.any(e > 0) and np.any(e < 0)
                # aggr ~ 1 + e + ζ + w   (e = side dummy absorbs any
                # baseline ask/bid asymmetry; β_ζ = common-mode,
                # β_w = antisymmetric mechanism)
                if has_both and y.size >= 30 and np.std(z) > 1e-12:
                    X = np.column_stack([np.ones_like(z), e, z, e * z])
                    coef = np.linalg.lstsq(X, y, rcond=None)[0]
                    u_comm[ui, bi] = coef[2]
                    u_mech[ui, bi] = coef[3]
                # matched-ζ between-side differential dose-response
                edges = band_edges[bi]
                if edges is None:
                    continue
                bins = np.searchsorted(edges, z, side="right")
                for b in range(n_bins):
                    ask_m = (bins == b) & (e > 0)
                    bid_m = (bins == b) & (e < 0)
                    if (ask_m.sum() >= min_lo_per_cell
                            and bid_m.sum() >= min_lo_per_cell):
                        pa = y[ask_m].mean()
                        pb = y[bid_m].mean()
                        u_D[ui, bi, b] = pa - pb
                        u_M[ui, bi, b] = 0.5 * (pa + pb)

        mech, mech_se, mech_t = self._fm_stats(u_mech)
        comm, comm_se, comm_t = self._fm_stats(u_comm)
        D, D_se, D_t = self._fm_stats(u_D)
        M, M_se, _ = self._fm_stats(u_M)
        band_centers = np.asarray(band_centers, float)
        n_units, n_lo = len(u_n), int(sum(u_n))

        pool_spread = np.concatenate(pool_spread)
        pool_reg = np.concatenate(pool_reg)
        p_marginal = {}
        for st in range(2, int(max_spread) + 1):
            m = pool_spread == st
            if m.sum() < 200:
                continue
            p_marginal[st] = dict(
                best=float((pool_reg[m] == 0).mean()),
                inside=float((pool_reg[m] == 1).mean()),
                passive=float((pool_reg[m] == 2).mean()),
                aggr=float((pool_reg[m] <= 1).mean()))

        if self.has_day:
            unit_word = "days"
        else:
            unit_word = "sessions"
        shift_note = (f"  [PLACEBO: signal shifted {signal_shift_s:g}s back]"
                      if signal_shift_s else "")
        print(f"\n{n_units} {unit_word}, {n_lo:,} LOs | "
              f"τ={tau_s:g}s τ_f={flow_tau_s:g}s κ={kappa:g} φ={phi:g}"
              f"{shift_note}")
        print("  mech = antisymmetric tilt (sim mechanism; want > 0)   "
              "common = both-sides co-movement")
        print(f"\n{'band':>8} {'mech':>8} {'t':>6} {'common':>8} {'t':>6} "
              f"{'D(ζ_lo)':>9} {'D(ζ_hi)':>9}")
        for bi, bl in enumerate(band_labels):
            print(f"{bl:>8} {mech[bi]:>8.4f} {mech_t[bi]:>6.1f} "
                  f"{comm[bi]:>8.4f} {comm_t[bi]:>6.1f} "
                  f"{D[bi, 0]:>9.4f} {D[bi, -1]:>9.4f}")

        if plot:
            # only bands with enough matched-ζ bins on both sides
            good = [bi for bi in range(n_band)
                    if np.isfinite(D[bi]).sum() >= 3]
            if good:
                ncol = len(good)
                fig, axes = plt.subplots(
                    2, ncol, figsize=(4.2 * ncol, 6.6), squeeze=False)
                dv = D[good][np.isfinite(D[good])]
                if dv.size:
                    dmax = max(np.nanmax(np.abs(dv)) * 1.25, 1e-3)
                else:
                    dmax = 0.05
                for ci, bi in enumerate(good):
                    cen = band_centers[bi]
                    axD = axes[0, ci]
                    axD.errorbar(cen, D[bi], yerr=D_se[bi], marker="o",
                                 capsize=3, color="tab:red")
                    axD.axhline(0, color="black", lw=0.8)
                    axD.axvline(0, color="gray", lw=0.6, ls=":")
                    axD.set_ylim(-dmax, dmax)
                    axD.set_title(band_labels[bi])
                    axD.set_xlabel("signal ζ (ask frame)")
                    axD.grid(True, alpha=0.3)
                    if ci == 0:
                        axD.set_ylabel("P(aggr|ask) − P(aggr|bid)\n"
                                       "antisymmetric tilt (mechanism)")
                    axM = axes[1, ci]
                    axM.errorbar(cen, M[bi], yerr=M_se[bi], marker="s",
                                 capsize=3, color="tab:gray")
                    axM.axvline(0, color="gray", lw=0.6, ls=":")
                    axM.set_xlabel("signal ζ (ask frame)")
                    axM.grid(True, alpha=0.3)
                    if ci == 0:
                        axM.set_ylabel("½[P(aggr|ask)+P(aggr|bid)]\n"
                                       "common-mode (both sides)")
                fig.suptitle(
                    f"LO aggressive-placement tilt vs signal — "
                    f"{self.db_path.name} ({n_units} {unit_word})"
                    f"{shift_note}\n"
                    "top row is the sim's mechanism: should rise through 0",
                    fontsize=10)
                plt.tight_layout()
                plt.show()

            # decomposition bars restricted to well-populated bands
            if good:
                gb = good
            else:
                gb = list(range(n_band))
            fig, ax = plt.subplots(figsize=(7.5, 4))
            xpos = np.arange(len(gb))
            w = 0.38
            ax.bar(xpos - w / 2, mech[gb], w, yerr=mech_se[gb], capsize=3,
                   label="mechanism (antisymmetric tilt)",
                   color="tab:red", alpha=0.85)
            ax.bar(xpos + w / 2, comm[gb], w, yerr=comm_se[gb], capsize=3,
                   label="common-mode (both sides)",
                   color="tab:gray", alpha=0.85)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(xpos)
            ax.set_xticklabels([band_labels[bi] for bi in gb])
            ax.set_ylabel("FM slope of P(aggr) on signal")
            ax.set_title(
                f"LO placement response decomposition{shift_note}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.show()

        return dict(
            spread_bands=spread_bands, band_labels=band_labels,
            n_bins=n_bins, sides=side_names,
            zeta_centers=band_centers,
            D=D, D_se=D_se, D_t=D_t, M=M, M_se=M_se,
            mech=mech, mech_se=mech_se, mech_t=mech_t,
            common=comm, common_se=comm_se, common_t=comm_t,
            p_marginal=p_marginal,
            n_units=n_units, n_lo=n_lo,
            params=dict(tau_s=tau_s, flow_tau_s=flow_tau_s,
                        kappa=kappa, phi=phi, xcap=xcap,
                        signal_shift_s=signal_shift_s))

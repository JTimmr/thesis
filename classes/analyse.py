"""
AnalyseMarket — unified analysis class for order-flow SQLite databases.

Works with both:
  - Empirical order-flow databases (partitioned by `day`, with `cls_method`)
  - Simulation output databases (continuous `timestamp`, with `bbo` and `intensities` tables)
"""

import sqlite3
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.stats import linregress, norm, lognorm, beta as beta_dist
from scipy.optimize import minimize_scalar, curve_fit

from .helpers import resolve_data_path


class AnalyseMarket:
    """Unified analysis class for order-flow SQLite databases."""

    # ═══════════════════════════════════════════════════════════════════
    # Construction
    # ═══════════════════════════════════════════════════════════════════

    def __init__(self, db_path, tick_size=0.05, day=None):
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
        """
        self.db_path = resolve_data_path(db_path)
        self.tick_size = tick_size
        self.day = day

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

        conn.close()

        # Cancellation calibration cache
        self._cancel_pass_done = False

        src = "real data" if self.has_day else "simulation"
        print(f"AnalyseMarket | {self.db_path.name} | {src}")
        print(f"  Tables: {sorted(self.tables)}")
        if self.day:
            print(f"  Day filter: {self.day}")

    # ═══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════════

    def _conn(self):
        """Return a fresh SQLite connection."""
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _table_cols(conn, table):
        cur = conn.cursor()
        try:
            cur.execute(f"PRAGMA table_info({table})")
            return [row[1] for row in cur.fetchall()]
        except Exception:
            return []

    def _day_clause(self, table_alias=""):
        """SQL fragment: `` AND day = '...' `` when day is set."""
        if self.day and self.has_day:
            col = f"{table_alias}.day" if table_alias else "day"
            return f" AND {col} = '{self.day}'"
        return ""

    # ── Mid-price helpers ──────────────────────────────────────────────

    def _get_mid_prices(self):
        """Return (timestamps, mid_prices) NumPy arrays.

        Timestamps are always returned as float64 seconds.
        For real data (ISO-8601 strings) they are converted to seconds
        since midnight of the first observation.
        """
        conn = self._conn()
        try:
            if self.has_bbo:
                df = pd.read_sql(
                    "SELECT timestamp, mid_price FROM bbo ORDER BY timestamp",
                    conn,
                )
            else:
                df = pd.read_sql(
                    "SELECT timestamp, mid_price FROM orders "
                    "WHERE mid_price IS NOT NULL"
                    + self._day_clause()
                    + " ORDER BY timestamp",
                    conn,
                )
            if len(df) < 5:
                return None, None

            ts = df["timestamp"].values
            # If timestamps are strings (ISO-8601), convert to float seconds
            if ts.dtype.kind in ("U", "O"):  # unicode or object
                ts_dt = pd.to_datetime(df["timestamp"], utc=True)
                # Seconds since the first timestamp
                ts = (ts_dt - ts_dt.iloc[0]).dt.total_seconds().values

            return ts.astype(np.float64), df["mid_price"].values
        finally:
            conn.close()

    def _get_mid_returns(self):
        """Return (timestamps, log_returns) or (None, None)."""
        ts, mids = self._get_mid_prices()
        if mids is None or len(mids) < 5:
            return None, None
        log_mids = np.log(mids)
        if not np.all(np.isfinite(log_mids)):
            return None, None
        return ts[1:], np.diff(log_mids)

    # ── Statistics helpers ─────────────────────────────────────────────

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

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════

    def summary(self):
        """Print database summary statistics."""
        conn = self._conn()

        n_orders = pd.read_sql(
            "SELECT COUNT(*) AS n FROM orders WHERE 1=1"
            + self._day_clause(), conn
        )["n"].iloc[0]

        if self.has_day:
            days_df = pd.read_sql(
                "SELECT DISTINCT day FROM orders ORDER BY day", conn
            )
            print(f"Orders: {n_orders:,} rows across {len(days_df)} days")
            print(f"Days: {days_df['day'].iloc[0]} … {days_df['day'].iloc[-1]}")
        else:
            print(f"Orders: {n_orders:,} rows")

        if "fills" in self.tables:
            n_fills = pd.read_sql(
                "SELECT COUNT(*) AS n FROM fills", conn
            )["n"].iloc[0]
            print(f"Fills: {n_fills:,}")

        if "mo_orders" in self.tables:
            n_mos = pd.read_sql(
                "SELECT COUNT(*) AS n FROM mo_orders", conn
            )["n"].iloc[0]
            print(f"Market orders: {n_mos:,}")

        if self.has_bbo:
            n_bbo = pd.read_sql(
                "SELECT COUNT(*) AS n FROM bbo", conn
            )["n"].iloc[0]
            print(f"BBO snapshots: {n_bbo:,}")

        conn.close()

    def trade_classification(self):
        """Print trade classification breakdown (real data with cls_method)."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table.")
            return

        conn = self._conn()
        n_fills = pd.read_sql("SELECT COUNT(*) AS n FROM fills", conn)["n"].iloc[0]
        n_mos = pd.read_sql("SELECT COUNT(*) AS n FROM mo_orders", conn)["n"].iloc[0]
        mo_buy = pd.read_sql(
            "SELECT COUNT(*) AS n FROM mo_orders WHERE side='buy'", conn
        )["n"].iloc[0]
        mo_sell = pd.read_sql(
            "SELECT COUNT(*) AS n FROM mo_orders WHERE side='sell'", conn
        )["n"].iloc[0]

        print(f"Raw fills: {n_fills:,}")
        print(f"Aggregated MOs: {n_mos:,}")
        print(f"  Buy: {mo_buy:,}  |  Sell: {mo_sell:,}")

        if self.has_cls_method:
            for method in ["quote", "midpoint", "tick", "unclassified"]:
                cnt = pd.read_sql(
                    f"SELECT COUNT(*) AS n FROM mo_orders "
                    f"WHERE cls_method='{method}'", conn
                )["n"].iloc[0]
                pct = 100 * cnt / n_mos if n_mos > 0 else 0
                print(f"  {method}: {cnt:,} ({pct:.2f}%)")

        conn.close()

    # ═══════════════════════════════════════════════════════════════════
    # Spread Analysis
    # ═══════════════════════════════════════════════════════════════════

    def spread_diagnostics(self, max_ticks=200):
        """Spread PMF (log-log) and linear frequency plot."""
        conn = self._conn()
        df = pd.read_sql(
            "SELECT spread_ticks FROM orders "
            "WHERE spread_ticks IS NOT NULL"
            + self._day_clause(),
            conn,
        )
        conn.close()

        df = df[df["spread_ticks"] <= max_ticks]
        counts = df["spread_ticks"].value_counts().sort_index()

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

    # ═══════════════════════════════════════════════════════════════════
    # Order Price Distribution
    # ═══════════════════════════════════════════════════════════════════

    def order_price_distribution(self, max_abs_tick=1000):
        """Signed relative price distribution for limit orders."""
        conn = self._conn()
        df = pd.read_sql(
            "SELECT ticks_from_best FROM orders "
            "WHERE event_type = 'LO' AND ticks_from_best IS NOT NULL"
            + self._day_clause(),
            conn,
        )
        conn.close()

        ticks = df["ticks_from_best"].values

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter (symlog x)
        ax = axes[0]
        counts = pd.Series(ticks).value_counts().sort_index()
        probs = counts / counts.sum()
        ax.scatter(probs.index, probs.values, s=3)
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("log")
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Ticks from best (signed)")
        ax.set_ylabel("Probability")
        ax.set_title("Relative price distribution (symlog)")

        # Histogram
        ax = axes[1]
        ticks_plot = ticks[np.abs(ticks) <= max_abs_tick]
        bins = np.arange(-max_abs_tick - 0.5, max_abs_tick + 1.5, 1)
        ax.hist(ticks_plot, bins=bins, density=True, alpha=0.75)
        ax.set_yscale("log"); ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Ticks from best (signed)")
        ax.set_ylabel("Density (log)")
        ax.set_title("Signed relative price histogram")

        plt.tight_layout(); plt.show()

    # ═══════════════════════════════════════════════════════════════════
    # Inside-Spread Placement
    # ═══════════════════════════════════════════════════════════════════

    def inside_spread_placement(self, max_spread=20):
        """
        Inside-spread conditional PMF with linear-fit parameter *c*.

        Returns dict  ``{regime_label: c_value}``.
        """
        conn = self._conn()
        df = pd.read_sql(
            "SELECT spread_ticks, ticks_from_best FROM orders "
            "WHERE event_type = 'LO' AND spread_ticks IS NOT NULL "
            "  AND ticks_from_best IS NOT NULL"
            + self._day_clause(),
            conn,
        )
        conn.close()

        inside_mask = (
            (df["ticks_from_best"] > 0)
            & (df["ticks_from_best"] < df["spread_ticks"])
        )
        p_best = (df["ticks_from_best"] == 0).mean()
        p_passive = (df["ticks_from_best"] > df["spread_ticks"]).mean()
        print(f"P(best): {p_best:.4f}  |  P(passive): {p_passive:.4f}")
        print(f"Inside-spread orders: {inside_mask.sum():,} "
              f"({inside_mask.mean():.4f})")

        df_in = df[inside_mask & (df["spread_ticks"] <= max_spread)].copy()
        x_all = df_in["ticks_from_best"].values.astype(float)
        s_all = df_in["spread_ticks"].values.astype(float)
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

        # ── Per-regime PMF subplots ──
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

        # ── Collapsed relative-position density ──
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

    # ═══════════════════════════════════════════════════════════════════
    # Probability vs Spread
    # ═══════════════════════════════════════════════════════════════════

    def probability_vs_spread(self, max_spread=None):
        """P(best|s), P(inside|s), P(passive|s) vs spread."""
        conn = self._conn()
        df = pd.read_sql(
            "SELECT spread_ticks, ticks_from_best FROM orders "
            "WHERE event_type = 'LO' AND spread_ticks IS NOT NULL "
            "  AND ticks_from_best IS NOT NULL"
            + self._day_clause(),
            conn,
        )
        conn.close()

        df["is_best"] = df["ticks_from_best"] == 0
        df["is_inside"] = (
            (df["ticks_from_best"] > 0)
            & (df["ticks_from_best"] < df["spread_ticks"])
        )
        df["is_passive"] = df["ticks_from_best"] > df["spread_ticks"]

        spread_values = sorted(df["spread_ticks"].unique())
        if max_spread:
            spread_values = [s for s in spread_values if s <= max_spread]

        p_best, p_inside, p_passive = [], [], []
        for s in spread_values:
            sub = df[df["spread_ticks"] == s]
            if len(sub) < 200:
                p_best.append(np.nan)
                p_inside.append(np.nan)
                p_passive.append(np.nan)
                continue
            n = len(sub)
            p_best.append(sub["is_best"].sum() / n)
            p_inside.append(sub["is_inside"].sum() / n)
            p_passive.append(sub["is_passive"].sum() / n)

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

    # ═══════════════════════════════════════════════════════════════════
    # Passive Depth Analysis
    # ═══════════════════════════════════════════════════════════════════

    def _power_law_fit(self, x, y):
        """Log-log OLS fit.  Returns (beta, r2, slope, intercept)."""
        logx, logy = np.log10(x), np.log10(y)
        coef = np.polyfit(logx, logy, 1)
        y_pred = coef[0] * logx + coef[1]
        ss_res = np.sum((logy - y_pred) ** 2)
        ss_tot = np.sum((logy - logy.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return -coef[0], r2, coef[0], coef[1]

    def _load_passive_depth(self):
        """Load passive LO depth data from DB."""
        conn = self._conn()
        df = pd.read_sql(
            "SELECT ticks_from_best, spread_ticks, side, imbalance "
            "FROM orders "
            "WHERE event_type = 'LO' "
            "  AND ticks_from_best IS NOT NULL "
            "  AND spread_ticks IS NOT NULL"
            + self._day_clause(),
            conn,
        )
        conn.close()
        df = df[df["ticks_from_best"] > df["spread_ticks"]].copy()
        df["depth"] = df["ticks_from_best"] - df["spread_ticks"]
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

    # ═══════════════════════════════════════════════════════════════════
    # Queue Position
    # ═══════════════════════════════════════════════════════════════════

    def queue_at_bbo(self, bins=150):
        """Queue-position histogram for LOs placed at best price."""
        conn = self._conn()
        df = pd.read_sql(
            "SELECT queue_ahead FROM orders "
            "WHERE event_type='LO' AND ticks_from_best=0 "
            "  AND queue_ahead IS NOT NULL"
            + self._day_clause(),
            conn,
        )
        conn.close()

        plt.figure(figsize=(6, 4))
        plt.hist(df["queue_ahead"], bins=bins, log=True)
        plt.xlabel("Queue ahead"); plt.ylabel("Frequency")
        plt.title("Queue position at BBO")
        plt.show()

        print(f"Mean: {df['queue_ahead'].mean():.1f},  "
              f"Median: {df['queue_ahead'].median():.1f}")

    def queue_passive(self, bins=60):
        """Queue position for passive orders with power-law tail fit."""
        conn = self._conn()
        df = pd.read_sql(
            "SELECT queue_ahead FROM orders "
            "WHERE event_type='LO' AND ticks_from_best > 0 "
            "  AND queue_ahead IS NOT NULL"
            + self._day_clause(),
            conn,
        )
        conn.close()

        x = df["queue_ahead"].values
        x = x[x > 0]

        log_bins = np.logspace(np.log10(x.min()), np.log10(x.max()), bins)
        hist, edges = np.histogram(x, bins=log_bins, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])

        xmin = np.percentile(x, 90)
        x_tail = x[x >= xmin]
        alpha = 1 + len(x_tail) / np.sum(np.log(x_tail / xmin))

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

    # ═══════════════════════════════════════════════════════════════════
    # Order Size Distributions
    # ═══════════════════════════════════════════════════════════════════

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
        df = pd.read_sql(
            "SELECT volume FROM orders "
            "WHERE event_type='LO' AND volume IS NOT NULL AND volume > 0"
            + self._day_clause(),
            conn,
        )
        conn.close()
        self._plot_size_density(df["volume"].astype(float).values,
                                "Limit Order", n_log_bins,
                                mid_range=(2, 1000), tail_start=800)

    def mo_size_distribution(self, n_log_bins=35):
        """Market-order size density with two-regime fit."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        conn = self._conn()
        df = pd.read_sql(
            "SELECT mo_volume AS volume FROM mo_orders "
            "WHERE mo_volume IS NOT NULL AND mo_volume > 0",
            conn,
        )
        conn.close()
        self._plot_size_density(df["volume"].astype(float).values,
                                "Market Order", n_log_bins,
                                mid_range=(2, 200), tail_start=200)

    def conditional_size_vs_distance(self, num_bins=80):
        """Mean order size vs distance from best price."""
        conn = self._conn()
        df = pd.read_sql(
            "SELECT ticks_from_best, volume FROM orders "
            "WHERE event_type='LO' AND volume > 0 "
            "  AND volume IS NOT NULL "
            "  AND ticks_from_best IS NOT NULL "
            "  AND ticks_from_best > 0"
            + self._day_clause(),
            conn,
        )
        conn.close()

        delta = df["ticks_from_best"].values
        bins = np.logspace(np.log10(delta.min()),
                           np.log10(delta.max()), num_bins)
        df["delta_bin"] = pd.cut(delta, bins=bins)
        grouped = (
            df.groupby("delta_bin", observed=False)
            .agg(mean_vol=("volume", "mean"),
                 count=("volume", "size"))
            .dropna()
        )
        grouped = grouped[grouped["count"] > 100]

        centers = np.sqrt(bins[:-1] * bins[1:])[:len(grouped)]
        mean_vol = grouped["mean_vol"].values
        m = mean_vol > 0

        plt.figure(figsize=(8, 6))
        plt.loglog(centers[m], mean_vol[m], "o", label="Binned mean")
        plt.xlabel("Distance from best (ticks)")
        plt.ylabel("Mean order size")
        plt.title("Conditional Mean Size vs Distance")
        plt.legend(); plt.tight_layout(); plt.show()

    # ═══════════════════════════════════════════════════════════════════
    # MO Size Models
    # ═══════════════════════════════════════════════════════════════════

    def mo_size_vs_depth(self, num_bins=50):
        """MO size vs opposite L0 depth — power-law fit."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        conn = self._conn()
        df = pd.read_sql(
            "SELECT mo_volume, opp_depth_L0 FROM mo_orders "
            "WHERE mo_volume IS NOT NULL AND mo_volume > 0 "
            "  AND opp_depth_L0 IS NOT NULL AND opp_depth_L0 > 0",
            conn,
        )
        conn.close()

        depth = df["opp_depth_L0"].values.astype(float)
        bins = np.logspace(np.log10(depth.min()),
                           np.log10(depth.max()), num_bins)
        df["depth_bin"] = pd.cut(depth, bins=bins)
        grouped = (
            df.groupby("depth_bin", observed=False)
            .agg(mean_vol=("mo_volume", "mean"),
                 median_vol=("mo_volume", "median"),
                 count=("mo_volume", "size"))
            .dropna()
        )
        grouped = grouped[grouped["count"] >= 30]
        centers = np.sqrt(bins[:-1] * bins[1:])[:len(grouped)]

        logx = np.log10(centers)
        logy = np.log10(grouped["mean_vol"].values)
        slope, intercept, r_val, *_ = linregress(logx, logy)

        plt.figure(figsize=(8, 6))
        plt.loglog(centers, grouped["mean_vol"], "o-",
                   label="Mean", alpha=0.8)
        plt.loglog(centers, grouped["median_vol"], "s-",
                   label="Median", alpha=0.5)
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
        conn = self._conn()
        df = pd.read_sql(
            "SELECT mo_volume, opp_depth_L0 FROM mo_orders "
            "WHERE mo_volume IS NOT NULL AND mo_volume > 0 "
            "  AND opp_depth_L0 IS NOT NULL AND opp_depth_L0 > 0",
            conn,
        )
        conn.close()

        ratio = (df["mo_volume"] / df["opp_depth_L0"]).values
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
        conn = self._conn()
        df = pd.read_sql(
            "SELECT mo_volume, opp_depth_L0 FROM mo_orders "
            "WHERE mo_volume IS NOT NULL AND mo_volume > 0 "
            "  AND opp_depth_L0 IS NOT NULL AND opp_depth_L0 > 0",
            conn,
        )
        conn.close()

        size = df["mo_volume"].values.astype(float)
        depth = df["opp_depth_L0"].values.astype(float)

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
        conn = self._conn()
        depth_cols = " + ".join(
            [f"COALESCE(opp_depth_L{i}, 0)" for i in range(10)]
        )
        df = pd.read_sql(
            f"SELECT mo_volume, opp_depth_L0, "
            f"       ({depth_cols}) AS cum_depth "
            f"FROM mo_orders "
            f"WHERE mo_volume IS NOT NULL AND mo_volume > 0 "
            f"  AND opp_depth_L0 IS NOT NULL AND opp_depth_L0 > 0",
            conn,
        )
        conn.close()
        df = df[df["cum_depth"] > 0].copy()

        size = df["mo_volume"].values.astype(float)
        d_cum = df["cum_depth"].values.astype(float)
        d_l0 = df["opp_depth_L0"].values.astype(float)

        eps_cum = size / (d_cum ** beta)
        log_eps = np.log(eps_cum)
        mu, sigma = log_eps.mean(), log_eps.std()

        # For comparison
        log_eps_old = np.log(size / (d_l0 ** beta))
        mu_old, sigma_old = log_eps_old.mean(), log_eps_old.std()

        print(f"β = {beta}  (FIXED)\n")
        print(f"{'':15s}  {'L0 only':>12s}  {'Cum L0–L9':>12s}")
        print("─" * 42)
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
        conn = self._conn()
        depth_expr = " + ".join(
            [f"COALESCE(opp_depth_L{i}, 0)" for i in range(10)]
        )
        df = pd.read_sql(
            f"SELECT mo_volume, ({depth_expr}) AS cum_depth "
            f"FROM mo_orders "
            f"WHERE mo_volume IS NOT NULL AND mo_volume > 0 "
            f"  AND opp_depth_L0 IS NOT NULL AND opp_depth_L0 > 0",
            conn,
        )
        conn.close()
        df = df[df["cum_depth"] > 0].copy()
        df["ratio"] = df["mo_volume"] / df["cum_depth"]

        print(f"MOs with valid cumulative depth: {len(df):,}")

        quantiles = np.linspace(0, 1, n_q + 1)[1:-1]
        depth_bounds = np.quantile(df["cum_depth"].values, quantiles)
        df["qi"] = np.searchsorted(
            depth_bounds, df["cum_depth"].values
        ).astype(int)

        q_pts = np.linspace(1 / n_pts, 1 - 1 / n_pts, n_pts)
        r_quintiles = {}
        for qi in range(n_q):
            ratios = df.loc[df["qi"] == qi, "ratio"].values
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

        all_r = df["ratio"].values
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

    # ═══════════════════════════════════════════════════════════════════
    # Price Impact by Depth
    # ═══════════════════════════════════════════════════════════════════

    def price_impact_by_depth(self, n_quartiles=4, max_tw_plot=15):
        """Ticks-walked distribution by opposite-side depth quartile."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        conn = self._conn()
        depth_expr = " + ".join(
            [f"COALESCE(opp_depth_L{i}, 0)" for i in range(10)]
        )
        df = pd.read_sql(
            f"SELECT ticks_walked, ({depth_expr}) AS cum_depth "
            f"FROM mo_orders "
            f"WHERE mo_volume IS NOT NULL AND mo_volume > 0 "
            f"  AND ticks_walked IS NOT NULL "
            f"  AND opp_depth_L0 IS NOT NULL AND opp_depth_L0 > 0",
            conn,
        )
        conn.close()
        df = df[df["cum_depth"] > 0].copy()

        qb = np.quantile(
            df["cum_depth"].values,
            np.linspace(0, 1, n_quartiles + 1)[1:-1],
        )
        df["dq"] = np.searchsorted(qb, df["cum_depth"].values).astype(int)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        fig, axes = plt.subplots(n_quartiles, 2,
                                 figsize=(14, 4 * n_quartiles))

        for qi in range(n_quartiles):
            tw = df.loc[df["dq"] == qi, "ticks_walked"].values.astype(int)
            n = len(tw)
            lo = 0 if qi == 0 else qb[qi - 1]
            hi = qb[qi] if qi < n_quartiles - 1 else df["cum_depth"].max()
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
        print("─" * 50)
        for qi in range(n_quartiles):
            sub = df.loc[df["dq"] == qi]
            print(f"{'Q' + str(qi + 1):>10s}  {len(sub):>8,}  "
                  f"{sub['ticks_walked'].mean():>8.3f}  "
                  f"{(sub['ticks_walked'] > 0).mean():>7.1%}  "
                  f"{(sub['ticks_walked'] >= 5).mean():>8.2%}")

    def ticks_walked_cdfs(self, save_path=None):
        """Compute (and optionally save) ticks_walked CDFs per quartile."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        conn = self._conn()
        depth_expr = " + ".join(
            [f"COALESCE(opp_depth_L{i}, 0)" for i in range(10)]
        )
        df = pd.read_sql(
            f"SELECT ticks_walked, ({depth_expr}) AS cum_depth "
            f"FROM mo_orders "
            f"WHERE mo_volume > 0 AND ticks_walked IS NOT NULL "
            f"  AND opp_depth_L0 IS NOT NULL AND opp_depth_L0 > 0",
            conn,
        )
        conn.close()
        df = df[df["cum_depth"] > 0].copy()

        qb = np.quantile(df["cum_depth"].values, [0.25, 0.5, 0.75])
        df["dq"] = np.searchsorted(qb, df["cum_depth"].values).astype(int)

        max_k = int(df["ticks_walked"].max())
        save_dict = {
            "depth_quartile_bounds": qb,
            "max_k": np.array([max_k]),
        }

        for qi in range(4):
            tw = df.loc[df["dq"] == qi, "ticks_walked"].values.astype(int)
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

    # ═══════════════════════════════════════════════════════════════════
    # Time-series overview plots
    # ═══════════════════════════════════════════════════════════════════

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
        limit_sql = f" LIMIT {n_events}" if n_events else ""
        offset_sql = f" OFFSET {offset}" if offset else ""
        try:
            if self.has_bbo:
                # Subquery to select the window, then subsample
                inner = (
                    "SELECT rowid AS _rid, timestamp, best_bid, best_ask, "
                    "mid_price FROM bbo ORDER BY timestamp"
                    + limit_sql + offset_sql
                )
                cnt = pd.read_sql(
                    f"SELECT COUNT(*) AS n FROM ({inner})", conn
                ).iloc[0]["n"]
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
                cnt = pd.read_sql(
                    f"SELECT COUNT(*) AS n FROM ({inner})", conn
                ).iloc[0]["n"]
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
        span = t_sec.max() - t_sec.min() if len(t_sec) else 0
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
        limit_sql = f" LIMIT {n_events}" if n_events else ""
        offset_sql = f" OFFSET {offset}" if offset else ""
        try:
            inner = (
                "SELECT rowid AS _rid, timestamp, "
                "total_bid_depth, total_ask_depth "
                "FROM orders WHERE total_bid_depth IS NOT NULL"
                + day_sql + " ORDER BY timestamp"
                + limit_sql + offset_sql
            )
            cnt = pd.read_sql(
                f"SELECT COUNT(*) AS n FROM ({inner})", conn
            ).iloc[0]["n"]
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
            limit_sql = f" LIMIT {n_events}" if n_events else ""
            offset_sql = f" OFFSET {offset}" if offset else ""
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
        limit_sql = f" LIMIT {n_events}" if n_events else ""
        offset_sql = f" OFFSET {offset}" if offset else ""
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
                val = closes[ci - 1] if ci > 0 else prices[0]
                opens[ci] = highs[ci] = lows[ci] = closes[ci] = val
            else:
                seg = prices[start:end]
                opens[ci]  = closes[ci - 1] if ci > 0 else seg[0]
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
        col_up, col_down = "#26a69a", "#ef5350"

        ax.bar(c_t[up], closes[up] - opens[up], bottom=opens[up],
               width=candle_w, color=col_up, edgecolor=col_up, lw=0.5)
        ax.bar(c_t[down], opens[down] - closes[down], bottom=closes[down],
               width=candle_w, color=col_down, edgecolor=col_down, lw=0.5)
        ax.vlines(c_t[up], lows[up], highs[up], color=col_up, lw=0.6)
        ax.vlines(c_t[down], lows[down], highs[down], color=col_down, lw=0.6)

        ema_colors = ["#ff9800", "#2196f3", "#9c27b0", "#00bcd4"]
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

    # ═══════════════════════════════════════════════════════════════════
    # Price-impact propagator
    # ═══════════════════════════════════════════════════════════════════

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
            day_sel = "day, " if has_day else ""

            mo_df = pd.read_sql(
                f"SELECT {day_sel}{ts_col} AS ts, side, "
                f"best_bid, best_ask, ticks_walked "
                f"FROM mo_orders"
                + (" WHERE 1=1" + day_sql if day_sql else "")
                + f" ORDER BY {ts_col}", conn)
        finally:
            conn.close()

        if len(mo_df) < 100:
            print(f"Only {len(mo_df)} MOs — not enough."); return None

        mid_mo = ((mo_df["best_bid"] + mo_df["best_ask"]) / 2.0
                  ).values.astype(np.float64)
        sign_all = np.where(mo_df["side"].values == "buy", 1.0, -1.0)
        day_labels = mo_df["day"].values if has_day else None
        tw = mo_df["ticks_walked"].values.astype(int)

        # ── Validation: R(1) sanity check ──
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
        print(f"Propagator data ({len(mo_df):,} MOs{day_info}):")
        print(f"  R(1) check: {n_pos:,} pos ({100*n_pos/n_total:.1f}%), "
              f"{n_zero:,} zero ({100*n_zero/n_total:.1f}%), "
              f"{n_neg:,} neg ({100*n_neg/n_total:.1f}%)")
        print(f"  Among price-moving: {frac_pos:.1%} positive "
              f"(expect >> 50%)")
        print(f"  Mean R(1): {np.mean(imm):.6f}")

        return mid_mo, sign_all, day_labels, tw

    # ── Helpers ──

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
            # ── Per-day computation ──
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
            # ── Pooled (simulation — single run, no day boundaries) ──
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

    # ── Plot methods ──

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

        # ── Raw impact R(τ) ──
        def _plot_one(ax, label, color, mask=None):
            m, se = self._propagator_curve(mid_mo, sign_all, day_labels,
                                           horizons, ref_shift=0,
                                           mask=mask)
            n_mo = int(mask.sum()) if mask is not None else N
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
        g1 = f"{G[0]:+.3f}" if acf_ok[0] else "n/a"
        gl = f"{G[-1]:+.3f}" if acf_ok[-1] else "n/a"
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
            n_mo = int(mask.sum()) if mask is not None else N
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

    # ═══════════════════════════════════════════════════════════════════
    # Stylized Facts
    # ═══════════════════════════════════════════════════════════════════

    # ── 1. Fat-tailed returns ──────────────────────────────────────────

    def stylized_fat_tails(self, n_bins=50):
        """Fat tails in return distribution: histogram + QQ-plot."""
        _, rets = self._get_mid_returns()
        if rets is None:
            print("Not enough data for return analysis."); return

        mu, sigma = rets.mean(), rets.std()
        kurt = np.mean(((rets - mu) / sigma) ** 4) - 3.0

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.hist(rets, bins=n_bins, density=True, alpha=0.6,
                label="Returns")
        xn = np.linspace(rets.min(), rets.max(), 300)
        ax.plot(xn, norm.pdf(xn, mu, sigma), "r-", lw=2,
                label="Normal fit")
        ax.set_xlabel("Log-return"); ax.set_ylabel("Density")
        ax.set_title("Return distribution vs Normal"); ax.legend()

        ax = axes[1]
        sr = np.sort(rets); n = len(sr)
        tq = np.array([mu + sigma * self._ppf_normal((i + 0.5) / n)
                        for i in range(n)])
        ax.scatter(tq, sr, s=1, alpha=0.4)
        lims = [min(tq.min(), sr.min()), max(tq.max(), sr.max())]
        ax.plot(lims, lims, "r--", lw=1, label="45° line")
        ax.set_xlabel("Normal theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title("QQ-plot (log-returns vs Normal)"); ax.legend()

        plt.suptitle(
            f"Fat tails — Excess kurtosis = {kurt:.2f}  (0 = Normal)",
            y=1.02,
        )
        plt.tight_layout(); plt.show()

        print(f"Mean return: {mu:.6e}")
        print(f"Std return:  {sigma:.6e}")
        print(f"Excess kurtosis: {kurt:.2f}  (> 0 → fat tails)")

    # ── 2. Absence of return autocorrelation ───────────────────────────

    def stylized_return_autocorrelation(self, max_lag=50):
        """
        ACF of raw log-returns with x-axis in minutes.

        Left panel : ACF vs lag in minutes.
        Right panel: Mean |ACF| before vs after the 20-minute threshold.
        """
        ts, rets = self._get_mid_returns()
        if rets is None:
            print("Not enough data for return analysis."); return

        # Average Δt in minutes
        dt_arr = np.diff(ts[:len(rets) + 1])
        avg_dt_min = np.mean(dt_arr) / 60.0

        n = len(rets)
        rets_dm = rets - rets.mean()
        var = np.dot(rets_dm, rets_dm)
        if var < 1e-30:
            print("Returns have zero variance."); return

        lags = np.arange(1, min(max_lag + 1, n))
        acf = np.array([np.dot(rets_dm[l:], rets_dm[:-l]) / var
                        for l in lags])
        lag_minutes = lags * avg_dt_min
        ci = 1.96 / np.sqrt(n)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        colors = ["tab:red" if m <= 20 else "tab:blue"
                  for m in lag_minutes]
        ax.bar(lag_minutes, acf, width=avg_dt_min * 0.8, alpha=0.7,
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
        mab = np.abs(acf[bm]).mean() if bm.any() else 0
        maa = np.abs(acf[am]).mean() if am.any() else 0
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

        print(f"Snapshot interval: {avg_dt_min:.2f} min")
        print(f"Mean |ACF| ≤ 20 min: {mab:.4f}  ({n_before} lags)")
        print(f"Mean |ACF| > 20 min: {maa:.4f}  ({n_after} lags)")

    # ── 3. Volatility clustering ───────────────────────────────────────

    def stylized_volatility_clustering(self, max_lag=100):
        """ACF of |returns| and returns² (slow decay = clustering)."""
        _, rets = self._get_mid_returns()
        if rets is None:
            print("Not enough data for return analysis."); return

        n = len(rets)
        acf_abs = self._acf(np.abs(rets), max_lag)
        acf_sq = self._acf(rets ** 2, max_lag)
        lags = np.arange(1, len(acf_abs) + 1)
        ci = 1.96 / np.sqrt(n)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

        axes[0].bar(lags, acf_abs, width=0.8, alpha=0.7, color="tab:blue")
        axes[0].axhline(ci, ls="--", color="red", alpha=0.5)
        axes[0].axhline(-ci, ls="--", color="red", alpha=0.5)
        axes[0].axhline(0, color="black", lw=0.5)
        axes[0].set_xlabel("Lag"); axes[0].set_ylabel("Autocorrelation")
        axes[0].set_title("ACF of |returns|")

        axes[1].bar(lags, acf_sq, width=0.8, alpha=0.7, color="tab:orange")
        axes[1].axhline(ci, ls="--", color="red", alpha=0.5)
        axes[1].axhline(-ci, ls="--", color="red", alpha=0.5)
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set_xlabel("Lag")
        axes[1].set_title("ACF of returns²")

        plt.suptitle("Volatility clustering  "
                     "(slow decay → clustering present)")
        plt.tight_layout(); plt.show()

        if acf_abs[0] > 0:
            threshold = acf_abs[0] / np.e
            hl_idx = np.where(acf_abs < threshold)[0]
            hl = hl_idx[0] + 1 if len(hl_idx) > 0 else ">100"
            print(f"|returns| ACF half-life ≈ {hl} lags")
        print(f"|returns| ACF(1) = {acf_abs[0]:.4f},  "
              f"ACF(10) = {acf_abs[min(9, len(acf_abs) - 1)]:.4f}")

    # ── 4. Concave price impact ────────────────────────────────────────

    def stylized_price_impact(self, n_bins=20):
        """Price impact (ticks walked) vs MO size with power-law fit."""
        if "mo_orders" not in self.tables:
            print("No mo_orders table."); return
        conn = self._conn()
        df = pd.read_sql(
            "SELECT mo_volume, ticks_walked FROM mo_orders "
            "WHERE mo_volume IS NOT NULL AND mo_volume > 0 "
            "  AND ticks_walked IS NOT NULL",
            conn,
        )
        conn.close()

        sizes = df["mo_volume"].values.astype(float)
        tw = df["ticks_walked"].values.astype(float)

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

    # ── 5. Order sign autocorrelation (long memory) ────────────────────

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
        time_col = "first_time_ns" if "first_time_ns" in mo_cols else "timestamp"
        df = pd.read_sql(
            f"SELECT side FROM mo_orders ORDER BY {time_col}", conn
        )
        conn.close()

        signs = []
        for s in df["side"]:
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

        # ── Panel 1: Short-lag ACF (bar chart) ───────────────────────
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

        # ── Panel 2: Log-log ACF with power-law fit ─────────────────
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

            # ── OLS fit in log-space on binned lags [10^0, cap] ───────
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

    # ── 6. Aggregational Gaussianity ───────────────────────────────────

    def stylized_aggregational_gaussianity(self, agg_levels=None):
        """Excess kurtosis decays toward 0 at coarser timescales."""
        ts, mids = self._get_mid_prices()
        if mids is None or len(mids) < 200:
            print("Not enough data for aggregational "
                  "Gaussianity analysis."); return

        if agg_levels is None:
            agg_levels = [1, 5, 10, 25, 50, 100]

        log_mids = np.log(mids)
        if not np.all(np.isfinite(log_mids)):
            print("Non-finite mid prices."); return

        kurtoses, valid_levels = [], []
        for agg in agg_levels:
            agg_prices = log_mids[::agg]
            if len(agg_prices) < 20:
                continue
            r = np.diff(agg_prices)
            if r.std() < 1e-15:
                continue
            k = np.mean(((r - r.mean()) / r.std()) ** 4) - 3.0
            kurtoses.append(k); valid_levels.append(agg)

        if len(valid_levels) < 2:
            print("Not enough aggregation levels."); return

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.plot(valid_levels, kurtoses, "o-", lw=2, ms=8)
        ax.axhline(0, ls="--", color="grey", alpha=0.5,
                   label="Normal (κ = 0)")
        ax.set_xlabel("Aggregation level (snapshots)")
        ax.set_ylabel("Excess kurtosis")
        ax.set_title("Kurtosis decay with aggregation"); ax.legend()

        ax = axes[1]
        for agg, lbl, col in [
            (valid_levels[0],  f"agg={valid_levels[0]}",  "tab:blue"),
            (valid_levels[-1], f"agg={valid_levels[-1]}", "tab:orange"),
        ]:
            r = np.diff(log_mids[::agg])
            mu_r, sig_r = r.mean(), r.std()
            if sig_r < 1e-15:
                continue
            sr = np.sort((r - mu_r) / sig_r)
            nn = len(sr)
            th = np.array([self._ppf_normal((i + 0.5) / nn)
                           for i in range(nn)])
            ax.scatter(th, sr, s=2, alpha=0.4, color=col, label=lbl)

        ax.plot([-4, 4], [-4, 4], "r--", lw=1, label="45° line")
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_xlabel("Normal quantiles")
        ax.set_ylabel("Standardised return quantiles")
        ax.set_title("QQ-plot: fine vs coarse aggregation")
        ax.legend()

        plt.suptitle("Aggregational Gaussianity  "
                     "(kurtosis → 0 at coarser scales)")
        plt.tight_layout(); plt.show()

        for lvl, k in zip(valid_levels, kurtoses):
            print(f"  agg={lvl:>4d}:  excess kurtosis = {k:+.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # Cancellation calibration
    # ═══════════════════════════════════════════════════════════════════

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

        # ── y-ratio bins ──────────────────────────────────────────────
        y_bins = np.linspace(0, max_y, n_y_bins + 1)

        # ══════════════════════════════════════════════════════════════
        #  Vectorised  P(·|C) from the database
        # ══════════════════════════════════════════════════════════════
        conn = self._conn()

        # P(y|C) from y_ratio on CXL rows
        y_cxl = pd.read_sql(
            "SELECT y_ratio FROM orders WHERE event_type='CXL'"
            " AND y_ratio IS NOT NULL AND y_ratio>=0 AND y_ratio<=?"
            + day_sql, conn, params=(max_y,)
        )["y_ratio"].values
        self._hist_cancel_y = np.histogram(y_cxl, bins=y_bins)[0].astype(float)
        del y_cxl

        # P(d_opp|C)  — tick distance to opposite best
        df_cxl = pd.read_sql(
            "SELECT order_price, side, best_bid, best_ask FROM orders "
            "WHERE event_type='CXL' AND order_price>0 "
            "AND best_bid>0 AND best_ask>0" + day_sql, conn)
        d_opp = np.where(
            df_cxl["side"] == 1,
            np.round((df_cxl["best_ask"] - df_cxl["order_price"]) / tick),
            np.round((df_cxl["order_price"] - df_cxl["best_bid"]) / tick),
        )
        d_v = d_opp[(d_opp >= 0) & (d_opp <= max_ticks_opp)].astype(int)
        self._hist_cancel_tick_opp = np.bincount(
            d_v, minlength=max_ticks_opp + 1).astype(float)

        # P(d_same|C)  — tick distance to same-side best
        d_same = np.where(
            df_cxl["side"] == 1,
            np.round((df_cxl["best_bid"] - df_cxl["order_price"]) / tick),
            np.round((df_cxl["order_price"] - df_cxl["best_ask"]) / tick),
        )
        d_sv = d_same[(d_same >= 0) & (d_same <= max_ticks_same)].astype(int)
        self._hist_cancel_tick_same = np.bincount(
            d_sv, minlength=max_ticks_same + 1).astype(float)
        del df_cxl, d_opp, d_v, d_same, d_sv

        conn.close()
        print("Vectorised P(·|C) done.")

        # ══════════════════════════════════════════════════════════════
        #  Sequential pass  — P(y), P(d_opp), P(d_same),
        #                      lifetimes, queue positions
        # ══════════════════════════════════════════════════════════════
        hist_total_y         = np.zeros(len(y_bins) - 1)
        hist_total_tick_opp  = np.zeros(max_ticks_opp + 1)
        hist_total_tick_same = np.zeros(max_ticks_same + 1)

        active = {}                     # oid → (price, side, delta0, ts_raw)
        price_queues = defaultdict(list) # price_key → [oid, …]
        lifetimes = []
        cancel_positions = []
        cancel_queue_sizes = []

        conn = self._conn()
        total_ev = pd.read_sql(
            "SELECT COUNT(*) AS cnt FROM orders "
            "WHERE event_type IN ('LO','CXL')" + day_sql, conn
        ).iloc[0]["cnt"]

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

                pct = int(100 * processed / total_ev) if total_ev else 100
                if pct >= next_print:
                    print(f"  {pct}% ({processed:,} events)", flush=True)
                    next_print += 5

                # ── sample ONE random active order ────────────────────
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

                # ── LO: register ──────────────────────────────────────
                if etype == "LO":
                    if price and price > 0:
                        active[oid] = (price, side, delta0, ts_raw)
                        price_queues[round(price / tick)].append(oid)

                # ── CXL: lifetime, queue pos, remove ──────────────────
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
                    try:
                        if isinstance(ts_raw, (int, float)):
                            lt = float(ts_raw) - float(lo_ts)
                        else:
                            lt = (pd.Timestamp(ts_raw)
                                  - pd.Timestamp(lo_ts)).total_seconds()
                        if lt > 0:
                            lifetimes.append(lt)
                    except Exception:
                        pass
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

    # ── P(C) computation ──────────────────────────────────────────────

    def _compute_pc(self):
        """Compute global per-trade-timestep cancellation probability."""
        if hasattr(self, "_P_C"):
            return self._P_C

        conn = self._conn()
        day_sql = self._day_clause()

        row = pd.read_sql(
            "SELECT "
            "  SUM(CASE WHEN event_type='CXL' THEN 1 ELSE 0 END) AS cxl, "
            "  AVG(n_total) AS avg_n "
            "FROM orders WHERE n_total IS NOT NULL" + day_sql, conn
        ).iloc[0]

        cancellations = row["cxl"]
        avg_n = row["avg_n"]

        # Number of trade-timesteps
        if "mo_orders" in self.tables:
            n_trades = pd.read_sql(
                "SELECT COUNT(*) AS cnt FROM mo_orders"
                + (" WHERE 1=1" + day_sql if day_sql else ""),
                conn,
            ).iloc[0]["cnt"]
        else:
            n_trades = pd.read_sql(
                "SELECT COUNT(*) AS cnt FROM orders "
                "WHERE event_type LIKE 'MO%'" + day_sql, conn
            ).iloc[0]["cnt"]

        conn.close()

        if avg_n and avg_n > 0 and n_trades and n_trades > 0:
            self._P_C = cancellations / (avg_n * n_trades)
        else:
            self._P_C = 0.0

        self._n_trades = int(n_trades) if n_trades else 0
        print(f"n_trades = {self._n_trades:,}")
        print(f"P(C) per trade-timestep per order = {self._P_C:.8f}")
        return self._P_C

    # ── Public cancellation methods ───────────────────────────────────

    def cancel_event_counts(self):
        """Print LO / CXL event counts."""
        conn = self._conn()
        day_sql = self._day_clause()
        row = pd.read_sql(
            "SELECT "
            "  SUM(CASE WHEN event_type='LO'  THEN 1 ELSE 0 END) AS lo, "
            "  SUM(CASE WHEN event_type='CXL' THEN 1 ELSE 0 END) AS cxl, "
            "  COUNT(*) AS total "
            "FROM orders WHERE event_type IN ('LO','CXL')" + day_sql, conn
        ).iloc[0]
        conn.close()

        print(f"Limit orders:    {int(row['lo']):>12,}")
        print(f"Cancellations:   {int(row['cxl']):>12,}")
        print(f"Total LO + CXL:  {int(row['total']):>12,}")

    def cancel_prob_y(self):
        """P(y), P(y|C), P(C|y) via Bayes — Mike-Farmer y-ratio."""
        self._cancel_sequential_pass()
        P_C = self._compute_pc()

        bins = self._y_bins
        centers = 0.5 * (bins[:-1] + bins[1:])
        h_tot = self._hist_total_y
        h_cxl = self._hist_cancel_y

        p_y   = h_tot / h_tot.sum() if h_tot.sum() > 0 else np.zeros_like(h_tot)
        p_y_c = h_cxl / h_cxl.sum() if h_cxl.sum() > 0 else np.zeros_like(h_cxl)

        ok = (p_y > 0) & (p_y_c > 0)
        P_C_y = np.full_like(p_y, np.nan)
        P_C_y[ok] = P_C * p_y_c[ok] / p_y[ok]

        # ── plots ──
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

        p_d   = h_tot / h_tot.sum() if h_tot.sum() > 0 else np.zeros_like(h_tot)
        p_d_c = h_cxl / h_cxl.sum() if h_cxl.sum() > 0 else np.zeros_like(h_cxl)

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
        except Exception:
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

        p_d     = h_book / h_book.sum() if h_book.sum() > 0 else np.zeros_like(h_book)
        p_d_cxl = h_cxl  / h_cxl.sum()  if h_cxl.sum()  > 0 else np.zeros_like(h_cxl)

        P_C_ds = np.full(max_d + 1, np.nan)
        ok = (p_d > 0) & (p_d_cxl > 0)
        P_C_ds[ok] = P_C * p_d_cxl[ok] / p_d[ok]

        d_axis = np.arange(max_d + 1)
        valid  = np.isfinite(P_C_ds) & (P_C_ds > 0)

        # ── Exponential fit: base·(1 + c·exp(−d)) ────────────────────
        def exp_model(d, base, c):
            return base * (1.0 + c * np.exp(-d))

        try:
            popt_e, _ = curve_fit(
                exp_model, d_axis[valid], P_C_ds[valid],
                p0=[np.nanmedian(P_C_ds[valid]), 1.0],
                bounds=([0, 0], [1, 200]), maxfev=10_000)
            base_e, c_e = popt_e
            exp_ok = True
        except Exception:
            exp_ok = False

        # ── Power-law fit on coarser bins: A·(1+d)^(−α) ──────────────
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
            bin_centers[i] = (np.average(ticks, weights=w)
                              if w.sum() > 0 else 0.5 * (lo + hi - 1))

        pb  = binned_book / binned_book.sum() if binned_book.sum() > 0 else np.zeros(n_bins)
        pc  = binned_cxl  / binned_cxl.sum()  if binned_cxl.sum()  > 0 else np.zeros(n_bins)
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
        except Exception:
            pw_ok = False

        # ── Plots ─────────────────────────────────────────────────────
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

        df = pd.read_sql(
            "SELECT imbalance, is_cancel, side, n_total "
            "FROM orders WHERE imbalance IS NOT NULL" + day_sql, conn)
        conn.close()

        if len(df) == 0:
            print("No imbalance data."); return

        df["is_cancel"] = df["is_cancel"].astype(int)

        # Mike-Farmer imbalance
        df["imb_mf"] = 0.0
        df.loc[df["side"] == 1, "imb_mf"] = (1 + df["imbalance"]) / 2
        df.loc[df["side"] == 2, "imb_mf"] = (1 - df["imbalance"]) / 2
        df = df[(df["imb_mf"] >= 0) & (df["imb_mf"] <= 1)]

        bins = np.linspace(0, 1, 21)
        df["bin"] = pd.cut(df["imb_mf"], bins)
        grp = df.groupby("bin", observed=True)

        cancels = grp["is_cancel"].sum()
        events  = grp.size()
        avg_n   = grp["n_total"].mean()
        p_c     = cancels / (avg_n * events)

        x = np.array([b.mid for b in p_c.index])
        y = p_c.values
        mask = (~np.isnan(y)) & (events.values > 2000)
        x, y = x[mask], y[mask]

        coef = np.polyfit(x, y, 1)
        K2 = coef[0]
        B  = coef[1] / K2 if K2 != 0 else 0

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

        df = pd.read_sql(
            "SELECT n_total, is_cancel FROM orders "
            "WHERE n_total IS NOT NULL" + day_sql, conn)
        conn.close()

        if len(df) == 0:
            print("No n_total data."); return

        bins = np.arange(df.n_total.min(), df.n_total.max() + 5, 5)
        df["bin"] = pd.cut(df["n_total"], bins)
        grp = df.groupby("bin", observed=True)

        cancels = grp["is_cancel"].sum()
        events  = grp.size()
        avg_n   = grp["n_total"].mean()
        p_c     = cancels / (avg_n * events)

        x = np.array([b.mid for b in p_c.index])
        y = p_c.values
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

    # ═══════════════════════════════════════════════════════════════════
    # Spike / gap diagnostics
    # ═══════════════════════════════════════════════════════════════════

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

        sep = "─" * 90
        print(f"\n{'═' * 90}")
        print(f"  SPIKE DIAGNOSTIC   offset={offset}   "
              f"window={window}   threshold={mid_threshold}")
        print(f"  Timestamp range: {t_lo:.6f} → {t_hi:.6f}")
        print(f"  BBO changes found: {len(changes)}")
        print(f"{'═' * 90}\n")

        for _, ch in changes.iterrows():
            ts = ch["timestamp"]
            print(sep)
            print(f"  BBO row {int(ch['rid'])}  t={ts:.8f}")
            print(f"    bid {ch['best_bid']:.2f}  ask {ch['best_ask']:.2f}  "
                  f"mid {ch['mid_price']:.2f}")
            print(f"    Δbid={ch['d_bid']:+.2f}  Δask={ch['d_ask']:+.2f}  "
                  f"Δmid={ch['d_mid']:+.2f}")

            t_prev = bbo.loc[
                bbo["rid"] < ch["rid"], "timestamp"
            ]
            t_start = t_prev.iloc[-1] if not t_prev.empty else ts - 0.01

            nearby_orders = orders[
                (orders["timestamp"] > t_start) & (orders["timestamp"] <= ts)
            ] if not show_all_events else orders[
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
                    side_str = "BID" if o["side"] == 1 else "ASK"
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

        print(f"{'═' * 90}")
        print("  END OF DIAGNOSTIC")
        print(f"{'═' * 90}\n")

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
            side = 1 if side.lower() in ("bid", "buy", "b") else 2
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

        side_str = "BID" if side == 1 else "ASK"
        print(f"\n{'═' * 80}")
        print(f"  RESTING ORDER TRACE   price={order_price}  "
              f"side={side_str}  before t={before_timestamp:.6f}")
        print(f"  Showing {len(df)} events (chronological):")
        print(f"{'═' * 80}\n")

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
        print(f"\n{'═' * 80}\n")
        return df

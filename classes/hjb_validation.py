"""Helpers for numerical HJB vs Guéant (ErgodicMM) validation backtests."""

from __future__ import annotations

import sqlite3
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter
from scipy import stats as sp_stats
from scipy.stats import wilcoxon

from .backtest import MMBacktester
from .market_maker import ErgodicMM, NumericalErgodicMM
from .mm_backtest_parallel import _load_day as load_day

QUOTE_COLUMNS = [
    "t", "bid", "ask", "bsz", "asz", "mid", "sigma", "res", "reason",
]


def quotes_df(agent) -> pd.DataFrame:
    """Build a DataFrame from an agent's ``quote_log``."""
    if not agent.quote_log:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    return pd.DataFrame(agent.quote_log, columns=QUOTE_COLUMNS)


def unique_quotes(qdf: pd.DataFrame) -> pd.DataFrame:
    if qdf.empty:
        return qdf
    return qdf.sort_values("t").drop_duplicates("t", keep="last")


def compare_quote_agreement(
    qa: pd.DataFrame,
    qn: pd.DataFrame,
    tick: float,
) -> Dict[str, Any]:
    """Compare analytical vs numerical quote streams on aligned timestamps."""
    qa_u = unique_quotes(qa)
    qn_u = unique_quotes(qn)
    merged = qa_u.merge(qn_u, on="t", how="inner", suffixes=("_an", "_nu"))
    n_aligned = len(merged)
    out: Dict[str, Any] = {
        "n_aligned": n_aligned,
        "bid_exact_pct": np.nan,
        "ask_exact_pct": np.nan,
        "max_bid_diff": np.nan,
        "max_ask_diff": np.nan,
        "max_da_diff": np.nan,
        "max_db_diff": np.nan,
        "merged": merged,
    }
    if n_aligned == 0:
        return out

    merged = merged.copy()
    merged["bid_diff"] = merged["bid_nu"] - merged["bid_an"]
    merged["ask_diff"] = merged["ask_nu"] - merged["ask_an"]
    merged["delta_b_an"] = merged["mid_an"] - merged["bid_an"]
    merged["delta_a_an"] = merged["ask_an"] - merged["mid_an"]
    merged["delta_b_nu"] = merged["mid_nu"] - merged["bid_nu"]
    merged["delta_a_nu"] = merged["ask_nu"] - merged["mid_nu"]
    merged["db_diff"] = merged["delta_b_nu"] - merged["delta_b_an"]
    merged["da_diff"] = merged["delta_a_nu"] - merged["delta_a_an"]
    out["merged"] = merged

    half = tick / 2.0
    out["bid_exact_pct"] = 100.0 * (merged["bid_diff"].abs() == 0).mean()
    out["ask_exact_pct"] = 100.0 * (merged["ask_diff"].abs() == 0).mean()
    out["bid_half_tick_pct"] = 100.0 * (merged["bid_diff"].abs() < half).mean()
    out["ask_half_tick_pct"] = 100.0 * (merged["ask_diff"].abs() < half).mean()
    out["max_bid_diff"] = float(merged["bid_diff"].abs().max())
    out["max_ask_diff"] = float(merged["ask_diff"].abs().max())
    out["max_da_diff"] = float(merged["da_diff"].abs().max())
    out["max_db_diff"] = float(merged["db_diff"].abs().max())
    return out


def erg_params_for_numerical(erg_params: Dict[str, Any]) -> Dict[str, Any]:
    """Strip ErgodicMM-only keys before constructing ``NumericalErgodicMM``.

    The numerical agent always quotes from EWMA event vol (``self.sigma``).
    Pair with ``vol_mode='ewma_event'`` on ``ErgodicMM`` so both agents share
    the same volatility definition (see ``mm_backtest`` S3a-iv).
    """
    mode = erg_params.get("vol_mode", "ewma_event")
    if mode != "ewma_event":
        warnings.warn(
            f"NumericalErgodicMM quotes from EWMA only; erg_params has "
            f"vol_mode={mode!r} for ErgodicMM - use 'ewma_event' for a fair "
            f"analytical vs numerical comparison.",
            stacklevel=2,
        )
    return {k: v for k, v in erg_params.items() if k != "vol_mode"}


def run_an_vs_nu_pair(
    bt: MMBacktester,
    day: str,
    *,
    gamma: float,
    erg_params: Dict[str, Any],
    size: int,
    solver_kw: Dict[str, Any],
    replay_start_s: Optional[float] = None,
    max_replay_s: Optional[float] = None,
    orders_df: Optional[pd.DataFrame] = None,
    mos_df: Optional[pd.DataFrame] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], ErgodicMM, NumericalErgodicMM]:
    """Run analytical and numerical agents on the same day/segment."""
    if orders_df is None or mos_df is None:
        conn = sqlite3.connect(str(bt.db_path))
        try:
            orders_df, mos_df = load_day(bt, conn, day)
        finally:
            conn.close()

    replay_kw: Dict[str, Any] = {}
    if replay_start_s is not None:
        replay_kw["replay_start_s"] = replay_start_s
    if max_replay_s is not None:
        replay_kw["max_replay_s"] = max_replay_s

    agent_an = ErgodicMM(**erg_params, gamma=gamma, size=size, verbose=False)
    stats_an = bt._replay(orders_df, mos_df, agent_an, **replay_kw)

    agent_nu = NumericalErgodicMM(
        **erg_params_for_numerical(erg_params),
        gamma=gamma,
        size=size,
        verbose=False,
        **solver_kw,
    )
    stats_nu = bt._replay(orders_df, mos_df, agent_nu, **replay_kw)
    return stats_an, stats_nu, agent_an, agent_nu


def validation_metrics_from_agents(
    agent_an: ErgodicMM,
    agent_nu: NumericalErgodicMM,
    stats_an: Dict[str, Any],
    stats_nu: Dict[str, Any],
    tick: float,
    *,
    source: str,
    day_label: str,
    gamma: float,
    setting: str,
) -> Dict[str, Any]:
    """One results-table row from a paired replay."""
    qa = quotes_df(agent_an)
    qn = quotes_df(agent_nu)
    agree = compare_quote_agreement(qa, qn, tick)

    pnl_an = float(stats_an.get("pnl", 0.0))
    pnl_nu = float(stats_nu.get("pnl", 0.0))
    trades_an = int(stats_an.get("n_trades", 0))
    trades_nu = int(stats_nu.get("n_trades", 0))
    max_inv_an = int(stats_an.get("max_inventory", 0))
    max_inv_nu = int(stats_nu.get("max_inventory", 0))
    n_solves = max(getattr(agent_nu, "n_solves", 0), 1)
    avg_iters = getattr(agent_nu, "total_iters", 0) / n_solves

    return {
        "source": source,
        "day_label": day_label,
        "gamma": gamma,
        "setting": setting,
        "n_aligned": agree["n_aligned"],
        "bid_exact_pct": agree["bid_exact_pct"],
        "ask_exact_pct": agree["ask_exact_pct"],
        "max_bid_diff": agree["max_bid_diff"],
        "max_ask_diff": agree["max_ask_diff"],
        "max_da_diff": agree["max_da_diff"],
        "pnl_an": pnl_an,
        "pnl_nu": pnl_nu,
        "d_pnl": pnl_nu - pnl_an,
        "trades_an": trades_an,
        "trades_nu": trades_nu,
        "d_trades": trades_nu - trades_an,
        "max_inv_an": max_inv_an,
        "max_inv_nu": max_inv_nu,
        "d_max_inv": max_inv_nu - max_inv_an,
        "avg_iters": avg_iters,
        "solver_time_s": getattr(agent_nu, "total_solve_time", 0.0),
    }


def run_validation_segment_gammas(
    segment: Dict[str, Any],
    gammas: List[float],
    *,
    source: str,
    db_path: Union[str, Path],
    tick_size: float,
    erg_params: Dict[str, Any],
    size: int,
    solver_kw: Dict[str, Any],
    setting: str,
    skip_opening: bool,
    hawkes: Union[bool, Any] = False,
    load_book_state: bool = False,
) -> List[Dict[str, Any]]:
    """Load one segment/day once, run all gammas, return metric rows."""
    day = segment["day"]
    day_label = segment.get("label") or day
    replay_start_s = segment.get("replay_start_s")
    max_replay_s = segment.get("max_replay_s")

    bt = MMBacktester(
        Path(db_path),
        tick_size=tick_size,
        hawkes=hawkes,
        load_book_state=load_book_state,
        skip_opening=skip_opening,
    )
    conn = sqlite3.connect(str(bt.db_path))
    try:
        orders_df, mos_df = load_day(bt, conn, day)
    finally:
        conn.close()

    n_events = len(orders_df) if orders_df is not None else 0
    print(
        f"[{day_label}] {setting} loaded {n_events:,} events, "
        f"{len(gammas)} gammas ({source})",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    seg_t0 = time.perf_counter()
    for gi, gamma in enumerate(gammas):
        g_t0 = time.perf_counter()
        stats_an, stats_nu, agent_an, agent_nu = run_an_vs_nu_pair(
            bt,
            day,
            gamma=gamma,
            erg_params=erg_params,
            size=size,
            solver_kw=solver_kw,
            replay_start_s=replay_start_s,
            max_replay_s=max_replay_s,
            orders_df=orders_df,
            mos_df=mos_df,
        )
        row = validation_metrics_from_agents(
            agent_an,
            agent_nu,
            stats_an,
            stats_nu,
            tick_size,
            source=source,
            day_label=day_label,
            gamma=gamma,
            setting=setting,
        )
        rows.append(row)
        elapsed = time.perf_counter() - g_t0
        print(
            f"  [{day_label}] {setting} gamma={gamma:.4f} ({gi + 1}/{len(gammas)}) "
            f"ask_exact={row['ask_exact_pct']:.1f}% "
            f"d_pnl={row['d_pnl']:+.2f} [{elapsed:.1f}s]",
            flush=True,
        )

    print(
        f"[{day_label}] {setting} done - {len(gammas)} gammas in "
        f"{time.perf_counter() - seg_t0:.0f}s",
        flush=True,
    )
    return rows


def append_results_csv(rows: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Append validation rows to CSV (create file + header if missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not path.is_file()
    df.to_csv(path, mode="a", header=write_header, index=False)


def aggregate_worst_case(df: pd.DataFrame) -> pd.DataFrame:
    """Worst-case metrics grouped by ``setting`` (MATCH / PROD)."""
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby("setting", sort=False)
        .agg(
            min_bid_exact_pct=("bid_exact_pct", "min"),
            min_ask_exact_pct=("ask_exact_pct", "min"),
            max_abs_bid_diff=("max_bid_diff", "max"),
            max_abs_ask_diff=("max_ask_diff", "max"),
            max_abs_da_diff=("max_da_diff", "max"),
            max_abs_d_pnl=("d_pnl", lambda s: s.abs().max()),
            max_abs_d_trades=("d_trades", lambda s: s.abs().max()),
            max_abs_d_max_inv=("d_max_inv", lambda s: s.abs().max()),
            n_rows=("gamma", "count"),
        )
        .reset_index()
    )


def print_validation_summary(df: pd.DataFrame) -> None:
    """Print a short paragraph suitable for thesis text."""
    if df.empty:
        print("No validation rows.")
        return
    wc = aggregate_worst_case(df)
    parts = []
    for _, r in wc.iterrows():
        setting = r["setting"]
        parts.append(
            f"{setting}: min ask exact match {r['min_ask_exact_pct']:.2f}%, "
            f"max |Δδ_a|={r['max_abs_da_diff']:.4f} PLN, "
            f"max |ΔPnL|={r['max_abs_d_pnl']:.2f}, "
            f"max |Δtrades|={int(r['max_abs_d_trades'])} "
            f"({int(r['n_rows'])} runs)."
        )
    print("\n".join(parts))


# --- Convergence study ---


def run_convergence_study(
    bt: MMBacktester,
    day: str,
    *,
    gamma: float,
    erg_params: Dict[str, Any],
    size: int,
    solver_ticks: List[float],
    base_solver_kw: Dict[str, Any],
    tick_size: float,
    replay_start_s: Optional[float] = None,
    max_replay_s: Optional[float] = None,
) -> pd.DataFrame:
    """Run analytical vs numerical at multiple solver_tick values.

    Returns a DataFrame with one row per solver_tick, containing error
    metrics (L-inf, RMSE, exact-match pct, MAE) for both bid and ask
    offsets.
    """
    conn = sqlite3.connect(str(bt.db_path))
    try:
        orders_df, mos_df = load_day(bt, conn, day)
    finally:
        conn.close()

    replay_kw: Dict[str, Any] = {}
    if replay_start_s is not None:
        replay_kw["replay_start_s"] = replay_start_s
    if max_replay_s is not None:
        replay_kw["max_replay_s"] = max_replay_s

    agent_an = ErgodicMM(**erg_params, gamma=gamma, size=size, verbose=False)
    bt._replay(orders_df, mos_df, agent_an, **replay_kw)
    qa = quotes_df(agent_an)

    rows: List[Dict[str, Any]] = []
    for st in sorted(solver_ticks, reverse=True):
        t0 = time.perf_counter()
        skw = {**base_solver_kw, "solver_tick": st}
        agent_nu = NumericalErgodicMM(
            **erg_params_for_numerical(erg_params),
            gamma=gamma, size=size, verbose=False, **skw,
        )
        bt._replay(orders_df, mos_df, agent_nu, **replay_kw)
        elapsed = time.perf_counter() - t0

        qn = quotes_df(agent_nu)
        agree = compare_quote_agreement(qa, qn, tick_size)
        merged = agree["merged"]

        if merged.empty:
            rows.append({"solver_tick": st})
            continue

        da_diff = merged["da_diff"].to_numpy()
        db_diff = merged["db_diff"].to_numpy()

        rows.append({
            "solver_tick": st,
            "max_da_diff": float(np.abs(da_diff).max()),
            "max_db_diff": float(np.abs(db_diff).max()),
            "rmse_da": float(np.sqrt((da_diff ** 2).mean())),
            "rmse_db": float(np.sqrt((db_diff ** 2).mean())),
            "mae_da": float(np.abs(da_diff).mean()),
            "mae_db": float(np.abs(db_diff).mean()),
            "ask_exact_pct": agree["ask_exact_pct"],
            "bid_exact_pct": agree["bid_exact_pct"],
            "n_aligned": agree["n_aligned"],
            "elapsed_s": elapsed,
        })
        print(
            f"  solver_tick={st:.1e}  "
            f"max|da|={rows[-1]['max_da_diff']:.6f}  "
            f"RMSE_da={rows[-1]['rmse_da']:.6f}  "
            f"exact_ask={rows[-1]['ask_exact_pct']:.2f}%  "
            f"[{elapsed:.1f}s]"
        )

    return pd.DataFrame(rows)


class MockSim:
    """Minimal shim passed to ``_compute_quotes`` in frozen-path mode."""

    def __init__(self, price_native_to_pln: float = 1.0,
                 bbo_in_tick_index: bool = False):
        self.price_native_to_pln = price_native_to_pln
        self.bbo_in_tick_index = bbo_in_tick_index


def frozen_path_segment_metrics(
    segment: Dict[str, Any],
    *,
    db_path: Union[str, Path],
    gamma: float,
    erg_params: Dict[str, Any],
    size: int,
    solver_ticks: List[float],
    base_solver_kw: Dict[str, Any],
    tick_size: float,
    hawkes: Union[bool, Any] = True,
    load_book_state: bool = True,
    skip_opening: bool = False,
    verbose: bool = True,
    preloaded_data: Optional[Dict[str, Any]] = None,
    warm_start: bool = True,
) -> List[Dict[str, Any]]:
    """Frozen-path metrics for one segment across all solver ticks.

    The analytical agent is run once to fix the realised (mid, sigma,
    inventory) trajectory.  Both agents are then evaluated at every frozen
    state, so inventory cannot drift and only the HJB solve quality varies.
    Each solver_tick yields one row with discretized (tick-rounded) and
    continuous (pre-rounding) offset-error metrics, all expressed in PLN.

    When ``preloaded_data`` is provided (dict with keys ``orders_df``,
    ``mos_df``, ``price_native_to_pln``, ``bbo_in_tick_index``), the DB is
    not touched — eliminating I/O contention in parallel runs.

    When ``warm_start=False``, the solver's value function ``phi`` is
    reset to zero before every quote evaluation, forcing a cold start
    at each state (useful for measuring the warm-start efficiency).
    """
    day = segment["day"]
    label = segment.get("label") or day
    replay_start_s = segment.get("replay_start_s")
    max_replay_s = segment.get("max_replay_s")

    bt = MMBacktester(
        Path(db_path), tick_size=tick_size, hawkes=hawkes,
        load_book_state=load_book_state, skip_opening=skip_opening,
    )

    if preloaded_data is not None:
        orders_df = preloaded_data["orders_df"]
        mos_df = preloaded_data["mos_df"]
    else:
        conn = sqlite3.connect(str(bt.db_path))
        try:
            orders_df, mos_df = load_day(bt, conn, day)
        finally:
            conn.close()

    replay_kw: Dict[str, Any] = {}
    if replay_start_s is not None:
        replay_kw["replay_start_s"] = replay_start_s
    if max_replay_s is not None:
        replay_kw["max_replay_s"] = max_replay_s

    agent_an = ErgodicMM(**erg_params, gamma=gamma, size=size, verbose=False)
    bt._replay(orders_df, mos_df, agent_an, **replay_kw)

    qa = quotes_df(agent_an)
    if qa.empty:
        return []
    qa = qa.sort_values("t").drop_duplicates("t", keep="last").reset_index(drop=True)

    n = len(qa)
    traj_mid = qa["mid"].to_numpy(dtype=np.float64)
    traj_sigma = qa["sigma"].to_numpy(dtype=np.float64)

    inv_at_quote = np.zeros(n, dtype=np.int64)
    tl = agent_an.trade_log
    if tl:
        t_quotes = qa["t"].to_numpy(dtype=np.float64)
        trade_times = np.array([float(e[0]) for e in tl], dtype=np.float64)
        trade_inv = np.array([int(e[6]) for e in tl], dtype=np.int64)
        j = np.searchsorted(trade_times, t_quotes, side="right") - 1
        valid = j >= 0
        inv_at_quote[valid] = trade_inv[j[valid]]

    ntp = float(bt.price_native_to_pln)
    mid_pln = traj_mid * ntp
    sigma_frac = np.where(mid_pln > 0, traj_sigma / np.where(mid_pln > 0, mid_pln, 1.0), 1e-4)
    mock_sim = MockSim(price_native_to_pln=ntp, bbo_in_tick_index=bt.bbo_in_tick_index)

    eval_an = ErgodicMM(**erg_params, gamma=gamma, size=size, verbose=False)
    an_bid = np.empty(n); an_ask = np.empty(n)
    an_db = np.empty(n); an_da = np.empty(n)
    for i in range(n):
        eval_an._ewma_var = sigma_frac[i] ** 2
        eval_an.inventory = int(inv_at_quote[i])
        b, a, _, _, db, da = eval_an._compute_quotes(traj_mid[i], mock_sim, return_deltas=True)
        an_bid[i] = b; an_ask[i] = a; an_db[i] = db; an_da[i] = da

    rows: List[Dict[str, Any]] = []
    for st in sorted(solver_ticks, reverse=True):
        t0 = time.perf_counter()
        agent_nu = NumericalErgodicMM(
            **erg_params_for_numerical(erg_params),
            gamma=gamma, size=size, verbose=False,
            **{**base_solver_kw, "solver_tick": st},
        )
        nu_bid = np.empty(n); nu_ask = np.empty(n)
        nu_db = np.empty(n); nu_da = np.empty(n)
        for i in range(n):
            if not warm_start:
                agent_nu._phi[:] = 0.0
            agent_nu._ewma_var = sigma_frac[i] ** 2
            agent_nu.inventory = int(inv_at_quote[i])
            b, a, _, _, db, da = agent_nu._compute_quotes(traj_mid[i], mock_sim, return_deltas=True)
            nu_bid[i] = b; nu_ask[i] = a; nu_db[i] = db; nu_da[i] = da
        elapsed = time.perf_counter() - t0

        disc_off = np.concatenate([(nu_ask - an_ask) * ntp, (nu_bid - an_bid) * ntp])
        cont_off = np.concatenate([nu_da - an_da, nu_db - an_db])
        abs_cont = np.abs(cont_off)

        n_solves = max(agent_nu.n_solves, 1)
        rows.append({
            "label": label,
            "solver_tick": float(st),
            "n_quotes": n,
            "ask_exact_pct": 100.0 * float((nu_ask == an_ask).sum()) / n,
            "bid_exact_pct": 100.0 * float((nu_bid == an_bid).sum()) / n,
            "rmse_disc": float(np.sqrt(np.mean(disc_off ** 2))),
            "mae_disc": float(np.mean(np.abs(disc_off))),
            "rmse_cont": float(np.sqrt(np.mean(cont_off ** 2))),
            "mae_cont": float(np.mean(np.abs(cont_off))),
            "elapsed_s": elapsed,
            "avg_iters": float(agent_nu.total_iters) / n_solves,
            "avg_solve_time_ms": 1000.0 * agent_nu.total_solve_time / n_solves,
            "p95_cont": float(np.percentile(abs_cont, 95)),
            "p99_cont": float(np.percentile(abs_cont, 99)),
            "p99_5_cont": float(np.percentile(abs_cont, 99.5)),
            "p99_9_cont": float(np.percentile(abs_cont, 99.9)),
            "max_cont": float(abs_cont.max()),
        })

        if verbose:
            print(
                f"  [{label}] tick={st:.1e} "
                f"ask={rows[-1]['ask_exact_pct']:.1f}% "
                f"rmse_c={rows[-1]['rmse_cont']:.2e} "
                f"[{elapsed:.1f}s]",
                flush=True,
            )

    if verbose and rows:
        total_s = sum(r["elapsed_s"] for r in rows)
        print(
            f"[{label}] DONE {len(rows)} ticks in {total_s:.0f}s total",
            flush=True,
        )
    return rows


def run_frozen_path_convergence_parallel(
    segments: List[Dict[str, Any]],
    *,
    db_path: Union[str, Path],
    gamma: float,
    erg_params: Dict[str, Any],
    size: int,
    solver_ticks: List[float],
    base_solver_kw: Dict[str, Any],
    tick_size: float,
    hawkes: Union[bool, Any] = True,
    load_book_state: bool = True,
    skip_opening: bool = False,
    n_jobs: int = 12,
    output_csv: Optional[Union[str, Path]] = None,
    warm_start: bool = True,
) -> pd.DataFrame:
    """Run the frozen-path study across segments in parallel.

    All referenced days are preloaded into memory first (sequential,
    no SQLite contention), then every segment is dispatched in a single
    ``Parallel`` call so workers never idle at day boundaries.
    Thread-based parallelism shares the preloaded DataFrames without
    copying.  When ``output_csv`` is given, completed segments are
    appended incrementally for crash resilience.
    """
    if output_csv is not None:
        output_csv = Path(output_csv)
        if output_csv.is_file():
            output_csv.unlink()

    bt = MMBacktester(
        Path(db_path), tick_size=tick_size, hawkes=hawkes,
        load_book_state=load_book_state, skip_opening=skip_opening,
    )

    needed_days = sorted({seg["day"] for seg in segments})
    day_data: Dict[str, Dict[str, Any]] = {}
    for day in needed_days:
        conn = sqlite3.connect(str(bt.db_path))
        try:
            orders_df, mos_df = load_day(bt, conn, day)
        finally:
            conn.close()
        day_data[day] = {
            "orders_df": orders_df,
            "mos_df": mos_df,
            "price_native_to_pln": bt.price_native_to_pln,
            "bbo_in_tick_index": bt.bbo_in_tick_index,
        }
        print(f"[preload] {day}: {len(orders_df):,} rows", flush=True)

    print(f"Dispatching {len(segments)} segments on {n_jobs} workers ...",
          flush=True)

    gen = Parallel(n_jobs=n_jobs, return_as="generator", verbose=5,
                   prefer="threads")(
        delayed(frozen_path_segment_metrics)(
            seg,
            db_path=db_path, gamma=gamma, erg_params=erg_params, size=size,
            solver_ticks=solver_ticks, base_solver_kw=base_solver_kw,
            tick_size=tick_size, hawkes=hawkes,
            load_book_state=load_book_state, skip_opening=skip_opening,
            preloaded_data=day_data[seg["day"]],
            warm_start=warm_start,
        )
        for seg in segments
    )

    rows: List[Dict[str, Any]] = []
    n_done = 0
    for batch in gen:
        if not batch:
            continue
        rows.extend(batch)
        n_done += 1
        if output_csv is not None:
            append_results_csv(batch, output_csv)
    print(f"  done: {n_done}/{len(segments)} segments", flush=True)

    return pd.DataFrame(rows)


def aggregate_frozen_convergence(
    df_long: pd.DataFrame,
    *,
    confidence: float = 0.95,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate per-segment metrics into mean + CI per solver_tick.

    For each solver_tick the cross-segment mean is reported with a two-sided
    Student-t ``confidence`` interval (columns ``<metric>_mean``,
    ``<metric>_lo``, ``<metric>_hi``).
    """
    if metrics is None:
        metrics = [
            "ask_exact_pct", "bid_exact_pct",
            "rmse_cont", "mae_cont", "rmse_disc", "mae_disc",
            "avg_iters", "avg_solve_time_ms",
            "p95_cont", "p99_cont", "p99_5_cont", "p99_9_cont", "max_cont",
        ]
        metrics = [m for m in metrics if m in df_long.columns]

    out_rows: List[Dict[str, Any]] = []
    for st, group in df_long.groupby("solver_tick"):
        nseg = len(group)
        tcrit = float(sp_stats.t.ppf(0.5 + confidence / 2.0, df=max(nseg - 1, 1)))
        row: Dict[str, Any] = {"solver_tick": float(st), "n_segments": int(nseg)}
        for m in metrics:
            values = group[m].to_numpy(dtype=np.float64)
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(nseg)) if nseg > 1 else 0.0
            row[f"{m}_mean"] = mean
            row[f"{m}_lo"] = mean - tcrit * sem
            row[f"{m}_hi"] = mean + tcrit * sem
        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values("solver_tick").reset_index(drop=True)


def setup_logx(ax) -> None:
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())


def setup_logx_decimal(ax) -> None:
    """Log-scaled x-axis with decimal tick labels (0.001, 0.01, 0.1)."""
    x_ticks = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3,
               0.01, 0.02, 0.05, 0.1]
    xlo, xhi = ax.get_xlim()
    visible = [v for v in x_ticks if xlo * 0.95 <= v <= xhi * 1.05]
    ax.set_xticks(visible)
    ax.set_xticklabels([f"{v:g}" for v in visible])
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_frozen_agreement_ci(
    df_agg: pd.DataFrame,
    tick_size: float,
) -> plt.Figure:
    """Quote disagreement percentage vs solver tick (log-log, mean and 95% CI)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    st = df_agg["solver_tick"].to_numpy()

    for side, color, marker in [("ask", "royalblue", "o"), ("bid", "firebrick", "s")]:
        mean = 100.0 - df_agg[f"{side}_exact_pct_mean"].to_numpy()
        hi = 100.0 - df_agg[f"{side}_exact_pct_lo"].to_numpy()
        lo = 100.0 - df_agg[f"{side}_exact_pct_hi"].to_numpy()
        lo = np.clip(lo, 1e-3, None)
        ax.fill_between(st, lo, hi, color=color, alpha=0.15)
        ax.loglog(st, mean, marker=marker, color=color, lw=1.4, ms=5, label=side.capitalize())

    ax.axvline(tick_size, ls="--", color="lightgray", alpha=0.6, lw=0.8)
    ax.set_xlabel("Solver tick (PLN)")
    ax.set_ylabel("Quote disagreement (%)")
    ax.set_title("Quote disagreement vs solver resolution")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.15)
    setup_logx(ax)

    pct_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ylo, yhi = ax.get_ylim()
    visible = [v for v in pct_ticks if ylo * 0.95 <= v <= yhi * 1.05]
    ax.set_yticks(visible)
    ax.set_yticklabels([f"{v:g}%" for v in visible])
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlim(ax.get_xlim()[0], 0.1)
    setup_logx_decimal(ax)
    ax.invert_xaxis()

    fig.tight_layout()
    return fig


def plot_frozen_offset_rmse_ci(
    df_agg: pd.DataFrame,
    tick_size: float,
) -> plt.Figure:
    """Quote-offset error RMSE vs solver tick (log-log, mean and 95% CI).

    Continuous (pre-rounding) offsets converge toward zero as the solver
    grid is refined, whereas tick-rounded offsets saturate at the market
    tick quantization floor.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    st = df_agg["solver_tick"].to_numpy()

    series = [
        ("rmse_cont", "royalblue", "o", "Continuous offsets"),
        ("rmse_disc", "firebrick", "s", "Tick-rounded offsets"),
    ]
    for key, color, marker, lbl in series:
        mean = df_agg[f"{key}_mean"].to_numpy()
        lo = np.clip(df_agg[f"{key}_lo"].to_numpy(), 1e-12, None)
        hi = np.clip(df_agg[f"{key}_hi"].to_numpy(), 1e-12, None)
        ax.fill_between(st, lo, hi, color=color, alpha=0.15)
        ax.loglog(st, mean, marker=marker, color=color, lw=1.4, ms=5, label=lbl)

    ax.axvline(tick_size, ls="--", color="lightgray", alpha=0.6, lw=0.8)
    ax.set_xlabel("Solver tick (PLN)")
    ax.set_ylabel("Offset error RMSE (PLN)")
    ax.set_title("Quote offset error vs solver resolution")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.15)
    setup_logx(ax)

    rmse_ticks = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
    ylo, yhi = ax.get_ylim()
    visible = [v for v in rmse_ticks if ylo * 0.95 <= v <= yhi * 1.05]
    ax.set_yticks(visible)
    ax.set_yticklabels([f"{v:.0e}" if v < 1e-3 else f"{v:g}" for v in visible])
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlim(ax.get_xlim()[0], 0.1)
    setup_logx_decimal(ax)
    ax.invert_xaxis()

    fig.tight_layout()
    return fig


def convergence_order_estimate(
    df_conv: pd.DataFrame,
    error_col: str = "rmse_da",
) -> Dict[str, float]:
    """Estimate convergence order p from log-log regression of error vs solver_tick.

    Returns dict with keys: order, intercept, r_squared, std_err.
    """
    mask = df_conv[error_col] > 0
    x = np.log10(df_conv.loc[mask, "solver_tick"].to_numpy())
    y = np.log10(df_conv.loc[mask, error_col].to_numpy())
    slope, intercept, r, _, stderr = sp_stats.linregress(x, y)
    return {
        "order": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r ** 2),
        "std_err": float(stderr),
    }


def convergence_bias_test(
    agent_an: ErgodicMM,
    agent_nu: NumericalErgodicMM,
    tick: float,
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test on quote offset residuals at fixed grid.

    Tests H0: median(delta_numerical - delta_analytical) = 0 for both sides.
    """
    qa = quotes_df(agent_an)
    qn = quotes_df(agent_nu)
    agree = compare_quote_agreement(qa, qn, tick)
    merged = agree["merged"]

    result: Dict[str, Any] = {
        "n_aligned": agree["n_aligned"],
        "ask_exact_pct": agree.get("ask_exact_pct", np.nan),
        "bid_exact_pct": agree.get("bid_exact_pct", np.nan),
    }

    if merged.empty or len(merged) < 10:
        result.update({"p_da": np.nan, "p_db": np.nan})
        return result

    da = merged["da_diff"].to_numpy()
    db = merged["db_diff"].to_numpy()

    da_nz = da[da != 0]
    db_nz = db[db != 0]

    if len(da_nz) >= 10:
        _, p_da = wilcoxon(da_nz, alternative="two-sided")
    else:
        p_da = 1.0
    if len(db_nz) >= 10:
        _, p_db = wilcoxon(db_nz, alternative="two-sided")
    else:
        p_db = 1.0

    result.update({
        "p_da": float(p_da),
        "p_db": float(p_db),
        "n_nonzero_da": len(da_nz),
        "n_nonzero_db": len(db_nz),
        "median_da": float(np.median(da)),
        "median_db": float(np.median(db)),
    })
    return result


def plot_convergence_study(
    df_conv: pd.DataFrame,
    tick_size: float,
    *,
    title_suffix: str = "",
) -> plt.Figure:
    """Log-log convergence plot with regression line and order annotation."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, label in [
        (axes[0], "rmse_da", "RMSE"),
        (axes[1], "mae_da", "MAE"),
    ]:
        mask = df_conv[col] > 0
        x = df_conv.loc[mask, "solver_tick"].to_numpy()
        y = df_conv.loc[mask, col].to_numpy()

        ax.loglog(x, y, "o-", color="royalblue", lw=1.4, ms=6, label=r"$\delta_a$ error")

        col_b = col.replace("_da", "_db")
        if col_b in df_conv.columns:
            mask_b = df_conv[col_b] > 0
            y_b = df_conv.loc[mask_b, col_b].to_numpy()
            x_b = df_conv.loc[mask_b, "solver_tick"].to_numpy()
            ax.loglog(x_b, y_b, "s--", color="firebrick", lw=1.2, ms=5, label=r"$\delta_b$ error")

        est = convergence_order_estimate(df_conv, error_col=col)
        x_fit = np.array([x.min(), x.max()])
        y_fit = 10 ** (est["intercept"] + est["order"] * np.log10(x_fit))
        ax.loglog(
            x_fit, y_fit, ":", color="gray", lw=1.5,
            label=f"slope = {est['order']:.2f} ($R^2$={est['r_squared']:.4f})",
        )

        ax.axvline(tick_size, ls="--", color="silver", alpha=0.5, lw=0.8)
        ax.text(
            tick_size * 1.15, ax.get_ylim()[0] * 2,
            "market tick", fontsize=7, color="gray", rotation=90, va="bottom",
        )

        ax.set_xlabel("Solver tick (PLN)")
        ax.set_ylabel(f"{label} of quote offset difference (PLN)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.15)
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())

    fig.suptitle(
        f"Grid convergence: numerical vs analytical quote offset error{title_suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_convergence_agreement(
    df_conv: pd.DataFrame,
    tick_size: float,
    *,
    title_suffix: str = "",
) -> plt.Figure:
    """Exact-match percentage and RMSE vs solver_tick with plateau annotation."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    st = df_conv["solver_tick"].to_numpy()

    ax = axes[0]
    ax.semilogx(st, df_conv["ask_exact_pct"], "o-", color="royalblue", lw=1.4, ms=6,
                label="Ask exact match")
    ax.semilogx(st, df_conv["bid_exact_pct"], "s--", color="firebrick", lw=1.2, ms=5,
                label="Bid exact match")
    ceiling_ask = df_conv["ask_exact_pct"].max()
    ceiling_bid = df_conv["bid_exact_pct"].max()
    ax.axhline(ceiling_ask, ls=":", color="royalblue", alpha=0.5, lw=0.8)
    ax.axhline(ceiling_bid, ls=":", color="firebrick", alpha=0.5, lw=0.8)
    ax.text(st.min() * 1.5, ceiling_ask - 1.5,
            f"ceiling = {ceiling_ask:.2f}%", fontsize=8, color="royalblue")
    ax.text(st.min() * 1.5, ceiling_bid - 1.5,
            f"ceiling = {ceiling_bid:.2f}%", fontsize=8, color="firebrick")
    ax.axvline(tick_size, ls="--", color="silver", alpha=0.5, lw=0.8)
    ax.set_xlabel("Solver tick (PLN)")
    ax.set_ylabel("Exact quote match (%)")
    ax.set_title("Quote agreement vs solver resolution")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.15)
    ax.set_ylim(60, 101)
    ax.invert_xaxis()

    ax = axes[1]
    mask_a = df_conv["rmse_da"] > 0
    mask_b = df_conv["rmse_db"] > 0
    ax.loglog(df_conv.loc[mask_a, "solver_tick"], df_conv.loc[mask_a, "rmse_da"],
              "o-", color="royalblue", lw=1.4, ms=6, label=r"RMSE $\delta_a$")
    if mask_b.any():
        ax.loglog(df_conv.loc[mask_b, "solver_tick"], df_conv.loc[mask_b, "rmse_db"],
                  "s--", color="firebrick", lw=1.2, ms=5, label=r"RMSE $\delta_b$")
    floor_rmse = df_conv.loc[df_conv["rmse_da"] > 0, "rmse_da"].min()
    ax.axhline(floor_rmse, ls=":", color="gray", alpha=0.6, lw=0.8,
               label=f"floor = {floor_rmse:.5f}")
    sat_tick = df_conv.loc[df_conv["rmse_da"] == floor_rmse, "solver_tick"].max()
    ax.axvline(sat_tick, ls="--", color="gray", alpha=0.4, lw=0.8)
    ax.text(sat_tick * 0.6, floor_rmse * 1.8,
            f"saturates at\nsolver_tick = {sat_tick:.1e}", fontsize=7, color="gray",
            ha="right")
    ax.set_xlabel("Solver tick (PLN)")
    ax.set_ylabel("RMSE of quote offset difference (PLN)")
    ax.set_title("Offset error vs solver resolution")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.15)

    fig.suptitle(
        f"Solver convergence to analytical solution{title_suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def load_bbo_for_validation_plot(
    db_path: Union[str, Path],
    day: str,
    *,
    replay_start_s: float = 0.0,
    max_replay_s: Optional[float] = None,
    time_in_hours: bool = False,
    tick_size: float = 0.05,
) -> pd.DataFrame:
    """Load BBO mid/bid/ask for the validation price panel."""
    conn = sqlite3.connect(str(db_path))
    try:
        day_orders = pd.read_sql_query(
            "SELECT timestamp, best_bid, best_ask FROM orders "
            "WHERE day = ? ORDER BY timestamp",
            conn,
            params=(day,),
        )
    finally:
        conn.close()

    if day_orders.empty:
        return day_orders

    try:
        ts = day_orders["timestamp"].to_numpy(dtype=np.float64)
        if ts.max() > 1e15:
            time_s = ts / 1e9
        else:
            time_s = ts
    except (ValueError, TypeError):
        time_s = pd.to_datetime(day_orders["timestamp"], utc=True).astype("int64").to_numpy() / 1e9
    t0 = float(time_s[0])
    day_orders = day_orders.copy()
    day_orders["time_s"] = time_s - t0

    t_lo = float(replay_start_s)
    t_hi = t_lo + float(max_replay_s) if max_replay_s is not None else np.inf
    day_orders = day_orders[
        (day_orders["time_s"] >= t_lo) & (day_orders["time_s"] <= t_hi)
    ].copy()
    day_orders["time_s"] -= t_lo
    bid_raw = day_orders["best_bid"].astype(float)
    ask_raw = day_orders["best_ask"].astype(float)
    day_orders["best_bid"] = (bid_raw / tick_size).round() * tick_size
    day_orders["best_ask"] = (ask_raw / tick_size).round() * tick_size
    day_orders["mid"] = (day_orders["best_bid"] + day_orders["best_ask"]) / 2.0
    if time_in_hours:
        day_orders["time_x"] = day_orders["time_s"] / 3600.0
    else:
        day_orders["time_x"] = day_orders["time_s"]
    return day_orders


def replay_opening_end_t(
    best_bid,
    best_ask,
    time_s,
    tick_size: float,
    thresh: int = 4,
) -> float:
    """First replay time where skip_opening would end (mid reversal count)."""
    prev_mid = None
    prev_dir = 0
    reversals = 0
    min_move = float(tick_size) * 0.5
    for i in range(len(best_bid)):
        bb, ba = float(best_bid[i]), float(best_ask[i])
        if not (np.isfinite(bb) and np.isfinite(ba)):
            continue
        mid = 0.5 * (bb + ba)
        if prev_mid is not None:
            diff = mid - prev_mid
            if abs(diff) >= min_move:
                cur_dir = 1 if diff > 0 else -1
                if prev_dir != 0 and cur_dir != prev_dir:
                    reversals += 1
                    if reversals >= thresh:
                        return float(time_s[i])
                prev_dir = cur_dir
        prev_mid = mid
    return float(time_s[0]) if len(time_s) else 0.0


def quote_arrays_plot(
    agent,
    *,
    t_plot0: float,
    t_plot_max: Optional[float],
    time_in_hours: bool,
):
    if not agent.quote_log:
        return [], [], [], []
    rows = [
        r for r in agent.quote_log
        if r[0] >= t_plot0 and (t_plot_max is None or r[0] <= t_plot_max)
    ]
    if time_in_hours:
        t = [(r[0] - t_plot0) / 3600.0 for r in rows]
    else:
        t = [r[0] - t_plot0 for r in rows]
    b = [r[1] for r in rows]
    a = [r[2] for r in rows]
    m = [r[5] for r in rows]
    return t, b, a, m


def plot_validation_reference(
    agent_an: ErgodicMM,
    agent_nu: NumericalErgodicMM,
    tick: float,
    *,
    title: str = "",
    style: str = "sim",
    t_max: Optional[float] = None,
    t_min: Optional[float] = None,
    t_plot0: Optional[float] = None,
    db_path: Optional[Union[str, Path]] = None,
    day: Optional[str] = None,
    replay_start_s: float = 0.0,
    gamma: Optional[float] = None,
    day_label: Optional[str] = None,
) -> plt.Figure:
    """Four-panel validation figure matching ``mm_backtest`` S3b (emp) / S3c (sim)."""
    style = style.lower()
    if style not in ("emp", "sim"):
        raise ValueError(f"style must be 'emp' or 'sim', got {style!r}")

    time_in_hours = style == "sim"
    qa = quotes_df(agent_an)
    qn = quotes_df(agent_nu)
    agree = compare_quote_agreement(qa, qn, tick)
    merged = agree["merged"].copy()

    t_plot_max = t_max
    bbo = pd.DataFrame()
    if db_path is not None and day is not None:
        bbo = load_bbo_for_validation_plot(
            db_path, day,
            replay_start_s=replay_start_s,
            max_replay_s=t_plot_max,
            time_in_hours=time_in_hours,
            tick_size=tick,
        )

    if t_plot0 is None:
        if style == "emp" and not bbo.empty:
            t_open_end = replay_opening_end_t(
                bbo["best_bid"].values, bbo["best_ask"].values,
                bbo["time_s"].values + float(replay_start_s), tick,
            )
            first_quotes = [
                agent.quote_log[0][0] for agent in (agent_an, agent_nu)
                if agent.quote_log
            ]
            t_plot0 = max(
                t_open_end,
                min(first_quotes) if first_quotes else t_open_end,
            )
        else:
            t_plot0 = float(t_min) if t_min is not None else 0.0

    if t_plot_max is not None:
        merged = merged[
            (merged["t"] >= t_plot0) & (merged["t"] <= t_plot_max)
        ].copy()
    else:
        merged = merged[merged["t"] >= t_plot0].copy()

    if time_in_hours:
        merged["time_x"] = (merged["t"] - t_plot0) / 3600.0
    else:
        merged["time_x"] = merged["t"] - t_plot0

    if not bbo.empty and t_plot0 > 0:
        bbo = bbo[bbo["time_s"] + float(replay_start_s) >= t_plot0].copy()
        bbo["time_x"] = (
            (bbo["time_s"] + float(replay_start_s) - t_plot0) / 3600.0
            if time_in_hours
            else bbo["time_s"] + float(replay_start_s) - t_plot0
        )

    t_an, b_an, a_an, m_an = quote_arrays_plot(
        agent_an, t_plot0=t_plot0, t_plot_max=t_plot_max, time_in_hours=time_in_hours,
    )
    t_nu, b_nu, a_nu, m_nu = quote_arrays_plot(
        agent_nu, t_plot0=t_plot0, t_plot_max=t_plot_max, time_in_hours=time_in_hours,
    )

    half_tick = tick / 2.0
    m_an_s = [round(v / half_tick) * half_tick for v in m_an] if m_an else []
    m_nu_s = [round(v / half_tick) * half_tick for v in m_nu] if m_nu else []

    if style == "sim":
        C_NU_BID, C_NU_ASK = "mediumblue", "crimson"
        C_AN_BID, C_AN_ASK = "darkseagreen", "rosybrown"
        Z_BACK, Z_FRONT = 2, 6
        nu_lw, an_lw = 1.0, 1.1
        nu_ls, an_ls = "-", "-"
    else:
        C_AN_BID, C_AN_ASK = "dimgray", "goldenrod"
        C_NU_BID, C_NU_ASK = "seagreen", "indianred"
        Z_BACK, Z_FRONT = 4, 3
        nu_lw, an_lw = 0.8, 0.8
        nu_ls, an_ls = "--", "-"

    fig, axes = plt.subplots(
        4, 1, figsize=(16, 15), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 2]},
    )
    if style == "sim":
        for ax_i in axes:
            ax_i.set_axisbelow(True)
            ax_i.grid(True, alpha=0.08, linewidth=0.35)

    ax = axes[0]
    if not bbo.empty:
        if style == "emp":
            ax.plot(bbo["time_x"], bbo["best_bid"], lw=0.3, alpha=0.45,
                    color="gray", label="BBO bid", zorder=1)
            ax.plot(bbo["time_x"], bbo["best_ask"], lw=0.3, alpha=0.45,
                    color="silver", label="BBO ask", zorder=1)
            ax.plot(bbo["time_x"], bbo["mid"], lw=0.3, alpha=0.6,
                    color="dimgray", label="Mid", zorder=1)
        else:
            ax.plot(bbo["time_x"], bbo["mid"], lw=0.35, alpha=0.45,
                    color="silver", label="Mid", zorder=1)
    elif t_an and m_an:
        ax.step(t_an, m_an_s, where="post", lw=0.35, alpha=0.5,
                color="silver", label="Mid", zorder=1)

    if style == "sim":
        if t_nu:
            ax.step(t_nu, b_nu, where="post", lw=nu_lw, alpha=1.0, color=C_NU_BID,
                    label="numerical bid", zorder=Z_BACK)
            ax.step(t_nu, a_nu, where="post", lw=nu_lw, alpha=1.0, color=C_NU_ASK,
                    label="numerical ask", zorder=Z_BACK)
        if t_an:
            ax.step(t_an, b_an, where="post", lw=an_lw, alpha=1.0, color=C_AN_BID,
                    label="analytical bid", zorder=Z_FRONT)
            ax.step(t_an, a_an, where="post", lw=an_lw, alpha=1.0, color=C_AN_ASK,
                    label="analytical ask", zorder=Z_FRONT)
    else:
        if t_an:
            ax.step(t_an, b_an, where="post", lw=an_lw, alpha=0.85, color=C_AN_BID,
                    label="analytical bid", zorder=Z_FRONT)
            ax.step(t_an, a_an, where="post", lw=an_lw, alpha=0.85, color=C_AN_ASK,
                    label="analytical ask", zorder=Z_FRONT)
        if t_nu:
            ax.step(t_nu, b_nu, where="post", lw=nu_lw, alpha=0.85, color=C_NU_BID,
                    ls=nu_ls, label="numerical bid", zorder=Z_BACK)
            ax.step(t_nu, a_nu, where="post", lw=nu_lw, alpha=0.85, color=C_NU_ASK,
                    ls=nu_ls, label="numerical ask", zorder=Z_BACK)

    ax.set_ylabel("Price (PLN)")
    if title:
        ax.set_title(title)
    elif style == "sim":
        hours = (t_plot_max or 0.0) / 3600.0
        ax.set_title(
            f"Analytical vs numerical Ergodic MM: "
            f"{day_label or day or 'sim'} sim, first {hours:.0f} h"
            + (f" (gamma={gamma})" if gamma is not None else "")
        )
    else:
        ax.set_title(
            f"Analytical vs numerical Ergodic MM vs BBO: "
            f"{day_label or day or 'emp'}"
            + (f" (gamma={gamma})" if gamma is not None else "")
        )
    ax.legend(loc="upper left", fontsize=8, ncol=4 if style == "emp" else 3)

    ax = axes[1]
    if style == "sim":
        if t_nu:
            db_nu_v = [-(m - b) for m, b in zip(m_nu_s, b_nu)]
            da_nu_v = [a - m for m, a in zip(m_nu_s, a_nu)]
            ax.step(t_nu, db_nu_v, where="post", lw=nu_lw, alpha=1.0, color=C_NU_BID,
                    label=r"numerical $-\delta_b$", zorder=Z_BACK)
            ax.step(t_nu, da_nu_v, where="post", lw=nu_lw, alpha=1.0, color=C_NU_ASK,
                    label=r"numerical $+\delta_a$", zorder=Z_BACK)
        if t_an:
            db_an_v = [-(m - b) for m, b in zip(m_an_s, b_an)]
            da_an_v = [a - m for m, a in zip(m_an_s, a_an)]
            ax.step(t_an, db_an_v, where="post", lw=an_lw, alpha=1.0, color=C_AN_BID,
                    label=r"analytical $-\delta_b$", zorder=Z_FRONT)
            ax.step(t_an, da_an_v, where="post", lw=an_lw, alpha=1.0, color=C_AN_ASK,
                    label=r"analytical $+\delta_a$", zorder=Z_FRONT)
    else:
        if t_an:
            db_an_v = [-(m - b) for m, b in zip(m_an_s, b_an)]
            da_an_v = [a - m for m, a in zip(m_an_s, a_an)]
            ax.step(t_an, db_an_v, where="post", lw=an_lw, alpha=0.85, color=C_AN_BID,
                    label=r"analytical $-\delta_b$")
            ax.step(t_an, da_an_v, where="post", lw=an_lw, alpha=0.85, color=C_AN_ASK,
                    label=r"analytical $+\delta_a$")
        if t_nu:
            db_nu_v = [-(m - b) for m, b in zip(m_nu_s, b_nu)]
            da_nu_v = [a - m for m, a in zip(m_nu_s, a_nu)]
            ax.step(t_nu, db_nu_v, where="post", lw=nu_lw, alpha=0.85, color=C_NU_BID,
                    ls=nu_ls, label=r"numerical $-\delta_b$")
            ax.step(t_nu, da_nu_v, where="post", lw=nu_lw, alpha=0.85, color=C_NU_ASK,
                    ls=nu_ls, label=r"numerical $+\delta_a$")
    ax.axhline(0, ls="--", color="grey", alpha=0.5)
    ax.yaxis.set_major_locator(MultipleLocator(tick))
    ax.set_ylabel("Quote offset from mid (PLN)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[2]
    if not merged.empty:
        ax.step(merged["time_x"], merged["db_diff"], where="post", lw=0.6, alpha=0.8,
                color=C_NU_BID, label=r"$\delta_b$ diff (num - an)")
        ax.step(merged["time_x"], merged["da_diff"], where="post", lw=0.6, alpha=0.8,
                color=C_NU_ASK, label=r"$\delta_a$ diff (num - an)")
    ax.axhline(0, ls="--", color="grey", alpha=0.5)
    ax.axhline(tick, ls=":", color="indianred", alpha=0.4, label="+/- 1 tick")
    ax.axhline(-tick, ls=":", color="indianred", alpha=0.4)
    ax.yaxis.set_major_locator(MultipleLocator(tick))
    ax.set_ylabel("Delta diff (PLN)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[3]
    if style == "sim":
        if agent_nu.trade_log:
            inv_t = [
                (r[0] - t_plot0) / 3600.0 for r in agent_nu.trade_log
                if t_plot0 <= r[0] and (t_plot_max is None or r[0] <= t_plot_max)
            ]
            inv_v = [
                r[-2] for r in agent_nu.trade_log
                if t_plot0 <= r[0] and (t_plot_max is None or r[0] <= t_plot_max)
            ]
            if inv_t:
                ax.step(inv_t, inv_v, where="post", lw=1.0, alpha=1.0, color=C_NU_BID,
                        label="numerical", zorder=Z_BACK)
        if agent_an.trade_log:
            inv_t = [
                (r[0] - t_plot0) / 3600.0 for r in agent_an.trade_log
                if t_plot0 <= r[0] and (t_plot_max is None or r[0] <= t_plot_max)
            ]
            inv_v = [
                r[-2] for r in agent_an.trade_log
                if t_plot0 <= r[0] and (t_plot_max is None or r[0] <= t_plot_max)
            ]
            if inv_t:
                ax.step(inv_t, inv_v, where="post", lw=1.1, alpha=1.0, color=C_AN_BID,
                        label="analytical", zorder=Z_FRONT)
    else:
        if agent_an.trade_log:
            inv_t = [r[0] - t_plot0 for r in agent_an.trade_log if r[0] >= t_plot0]
            inv_v = [r[-2] for r in agent_an.trade_log if r[0] >= t_plot0]
            if t_plot_max is not None:
                pairs = [(t, v) for t, v in zip(inv_t, inv_v) if t + t_plot0 <= t_plot_max]
                inv_t, inv_v = zip(*pairs) if pairs else ([], [])
            if inv_t:
                ax.step(inv_t, inv_v, where="post", lw=0.9, color=C_AN_BID,
                        label="analytical")
        if agent_nu.trade_log:
            inv_t = [r[0] - t_plot0 for r in agent_nu.trade_log if r[0] >= t_plot0]
            inv_v = [r[-2] for r in agent_nu.trade_log if r[0] >= t_plot0]
            if t_plot_max is not None:
                pairs = [(t, v) for t, v in zip(inv_t, inv_v) if t + t_plot0 <= t_plot_max]
                inv_t, inv_v = zip(*pairs) if pairs else ([], [])
            if inv_t:
                ax.step(inv_t, inv_v, where="post", lw=0.9, color=C_NU_BID,
                        ls="--", label="numerical")
    ax.axhline(0, ls="--", color="grey", alpha=0.5)
    ax.set_ylabel("Inventory (units)")
    ax.set_xlabel("Time (hours)" if time_in_hours else "Time (s)")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig

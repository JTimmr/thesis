"""Picklable workers for ``joblib.Parallel`` MM backtests (finite cash, carry_cash)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def get_day_span_s(db_path: Union[str, Path], day: str) -> float:
    """Return the replay span for *day* in seconds (same base as ``_replay``)."""
    import sqlite3

    from .backtest import MMBacktester

    bt = MMBacktester(Path(db_path), tick_size=0.05)
    conn = sqlite3.connect(str(db_path))
    try:
        orders_df, mos_df = bt._load_empirical_day(conn, day)
    finally:
        conn.close()
    if orders_df.empty:
        raise ValueError(f"No orders found for day={day!r} in {db_path}")
    order_times = orders_df["time_ns"].to_numpy(dtype=np.float64)
    t0 = order_times[0]
    if mos_df is not None and not mos_df.empty:
        t0 = min(t0, float(mos_df["time_ns"].iloc[0]))
    t_last = float(order_times[-1])
    if mos_df is not None and not mos_df.empty:
        t_last = max(t_last, float(mos_df["time_ns"].iloc[-1]))
    return (t_last - t0) / 1e9


def plan_8h_segments(db_path: Union[str, Path], day_keys: List[str], *, n_days: int = 10, segment_hours: float = 8.0) -> List[Dict[str, Any]]:
    """Plan ``n_days`` equal 8-hour segments from one or two sim files.

    The first source day is split into as many full segments as its span
    allows, using *equal* sub-intervals (e.g. 80 h → 10×8 h, 72 h → 9×8 h).
    Any remainder is taken from the start of the next day(s) in ``day_keys``.
    """
    segment_s = float(segment_hours) * 3600.0
    segments: List[Dict[str, Any]] = []
    remaining = int(n_days)

    for file_idx, day in enumerate(day_keys):
        if remaining <= 0:
            break
        span_s = get_day_span_s(db_path, day)
        n_fit = int(span_s // segment_s)
        if n_fit <= 0:
            continue

        if file_idx == 0:
            n_take = min(remaining, n_fit)
            for k in range(n_take):
                segments.append({
                    "day": day,
                    "segment_idx": len(segments),
                    "replay_start_s": k * segment_s,
                    "max_replay_s": segment_s,
                    "label": f"{day}_seg{k:02d}",
                    "source_file_idx": file_idx,
                })
        else:
            n_take = min(remaining, n_fit)
            for k in range(n_take):
                segments.append({
                    "day": day,
                    "segment_idx": len(segments),
                    "replay_start_s": k * segment_s,
                    "max_replay_s": segment_s,
                    "label": f"{day}_seg{k:02d}",
                    "source_file_idx": file_idx,
                })
        remaining -= n_take

    if remaining > 0:
        spans = {d: get_day_span_s(db_path, d) for d in day_keys}
        raise ValueError(
            f"Could not plan {n_days} segments of {segment_hours}h from "
            f"{day_keys}; spans={spans}, got {len(segments)}"
        )
    return segments


def single_multivariate_hawkes_factory(
    *, tick_size: float = 0.05,
    mo_self_scale: float = 1.0,
    mo_impact_scale: float = 1.0,
):
    """Picklable Hawkes factory for single-kernel sim backtests.

    ``mo_self_scale`` scales only MO self-excitation α;
    ``mo_impact_scale`` scales sampled ticks-walked per MO (see
    :class:`~research_core.classes.simulate.Simulate`).
    """
    from .hawkes_filter import HawkesFilter
    from .simulate import SimulateFast

    sim = SimulateFast(
        "hawkes_multivariate",
        T=1,
        tick_size=tick_size,
        mo_self_scale=float(mo_self_scale),
        mo_impact_scale=float(mo_impact_scale),
    )
    return HawkesFilter.from_simulate(sim)


def run_simple_cash_carry(
    run_index: int,
    db_path: str,
    *,
    tick_size: float,
    window_size: int,
    initial_cash: float,
    size: int,
    offset: int = 1,
    mm_sqlite_path: Optional[str] = None,
    mm_sqlite_side_is_string: bool = True,
) -> Optional[pd.DataFrame]:
    """One Simple MM backtest with ``initial_cash`` and ``carry_cash=True``.

    If ``mm_sqlite_path`` is set, ``MMBacktester.run_all`` writes merged
    ``trade_log`` / ``quote_log`` / ``pnl_snapshots`` for all windows to
    that SQLite file (same layout as ``to_sqlite``).
    """
    from .backtest import MMBacktester
    from .market_maker import SimpleMarketMaker

    bt = MMBacktester(Path(db_path), tick_size=tick_size)
    df = bt.run_all(
        lambda: SimpleMarketMaker(
            offset=offset,
            size=size,
            tick_size=tick_size,
            initial_cash=initial_cash,
            initial_inventory=0,
            verbose=False,
        ),
        seed=None,
        carry_cash=True,
        window_size=window_size,
        verbose=False,
        mm_sqlite_path=mm_sqlite_path,
        mm_sqlite_side_is_string=mm_sqlite_side_is_string,
    )
    if df is None or df.empty:
        return None
    out = df.copy()
    out["run"] = run_index
    return out


def run_as_cash_carry(
    run_index: int,
    db_path: str,
    *,
    tick_size: float,
    window_size: int,
    as_calib: Dict[str, Any],
    gamma: float,
    horizon: float,
    size: int,
    initial_cash: float,
    mm_sqlite_path: Optional[str] = None,
    mm_sqlite_side_is_string: bool = False,
) -> Optional[pd.DataFrame]:
    """One Avellaneda–Stoikov MM backtest (per-run *k* / *vol_halflife*), finite cash."""
    from .backtest import MMBacktester
    from .market_maker import AvellanedaStoikovMM

    calib = dict(as_calib)

    def factory() -> AvellanedaStoikovMM:
        return AvellanedaStoikovMM(
            **calib,
            gamma=gamma,
            horizon=horizon,
            size=size,
            initial_cash=initial_cash,
            initial_inventory=0,
            verbose=False,
        )

    bt = MMBacktester(Path(db_path), tick_size=tick_size)
    df = bt.run_all(
        factory,
        seed=None,
        carry_cash=True,
        window_size=window_size,
        verbose=False,
        mm_sqlite_path=mm_sqlite_path,
        mm_sqlite_side_is_string=mm_sqlite_side_is_string,
    )
    if df is None or df.empty:
        return None
    out = df.copy()
    out["run"] = run_index
    return out


def _load_nn_bundle(ckpt_path: str) -> Dict[str, Any]:
    """Load the NN fill-prob checkpoint into a reusable bundle.

    Returns a dict with the eval-mode ``model``, normalisation stats and a
    captured ``torch`` reference so the fill-law closures can run a forward
    pass without re-importing.  Defining the MLP class here keeps ``torch``
    a lazy dependency of these workers (importable without it).
    """
    import torch
    import torch.nn as nn

    class _FillProbMLP(nn.Module):
        def __init__(self, in_dim: int, hidden: list, dropout: float = 0.0):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dropout = ckpt.get("dropout", 0.0)
    temperature = ckpt.get("temperature", 1.0)
    model = _FillProbMLP(ckpt["in_dim"], ckpt["hidden"], dropout=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    feat_mean = ckpt["feat_mean"]
    return {
        "torch": torch,
        "model": model,
        "feat_mean": feat_mean,
        "feat_std": ckpt["feat_std"],
        "temperature": temperature,
        "n_feat": len(feat_mean) - 2,  # features + delta + queue_ahead
        # recent_vol training definition, if recorded by train_phantom_nn.
        # Absent on legacy (empirical) checkpoints -> resolved to "ewma_event".
        "vol_mode": ckpt.get("vol_mode", None),
        # queue_ahead feature transform applied before normalisation.
        # None on legacy checkpoints => raw linear (backward compatible).
        "queue_transform": ckpt.get("queue_transform", None),
    }


def assemble_fill_X(
    side_int: int,
    feat_mean,
    feat_std,
    n_feat: int,
    sim,
    delta,
    *,
    agent=None,
    use_realized: bool = False,
    t0: Optional[float] = None,
    span: Optional[float] = None,
    queue_transform: Optional[str] = None,
):
    """Assemble the normalised NN input matrix ``X`` for one quote side.

    Single source of truth for the fill-law feature vector: the 10 z-features
    (6 Hawkes intensities, ``spread_ticks``, ``l0_imbalance``, ``recent_vol``,
    ``tod_frac``) followed by normalised ``delta`` and ``queue_ahead`` and the
    raw ``side`` encoding.  Returns a ``(n_delta, n_feat + 3)`` float32 array,
    exactly the tensor the ``FillProbMLP`` consumes (pre-forward-pass).

    ``sim`` is the feature view handed to the fill closure (exposing ``ob``,
    ``hawkes_filter``, ``book_state``, ``tick_size``).  ``agent`` supplies the
    ``recent_vol`` feature (``sigma_realized`` / ``sigma``); when ``None`` the
    training mean is used.  Used by both the inference closure in
    :func:`_make_nn_h_fn` and the online RL feature logger so the logged
    training rows are byte-identical to what the model saw at quote time.
    """
    delta = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    n = delta.shape[0]
    side_enc = float(side_int - 1)

    hf = getattr(sim, "hawkes_filter", None)
    if hf is not None:
        t_now = hf._t_last
        hawkes_int = hf.intensity(t_now)
    else:
        hawkes_int = np.zeros(6, dtype=np.float64)
        t_now = 0.0

    z_raw = np.zeros(n_feat, dtype=np.float32)
    z_raw[:6] = hawkes_int

    bb, ba = sim.ob.get_bbo()
    tick = getattr(sim, "tick_size", 0.05)
    bs = getattr(sim, "book_state", None)

    if bb is not None and ba is not None and tick > 0:
        z_raw[6] = (ba - bb) / tick
    else:
        z_raw[6] = feat_mean[6]

    if bs is not None:
        z_raw[7] = float(bs.imbalance) if np.isfinite(bs.imbalance) else feat_mean[7]
    else:
        z_raw[7] = feat_mean[7]

    _vol_attr = "sigma_realized" if use_realized else "sigma"
    if agent is not None and hasattr(agent, _vol_attr):
        z_raw[8] = getattr(agent, _vol_attr)
    else:
        z_raw[8] = feat_mean[8]

    if span is not None and t0 is not None:
        z_raw[9] = min(max((t_now - t0) / span, 0.0), 1.0)
    else:
        z_raw[9] = feat_mean[9]

    z_norm = (z_raw - feat_mean[:n_feat]) / feat_std[:n_feat]
    d_norm = (delta.astype(np.float32) - feat_mean[n_feat]) / feat_std[n_feat]

    if bs is not None and bb is not None and ba is not None:
        half_spread = (ba - bb) / 2.0
        if side_int == 1:
            side_depths = np.nan_to_num(bs.bid_depths[:40], nan=0.0)
        else:
            side_depths = np.nan_to_num(bs.ask_depths[:40], nan=0.0)
        levels = np.floor((delta - half_spread) / tick).astype(int)
        cum_depth = np.cumsum(side_depths)
        levels_clamped = np.clip(levels, -1, len(cum_depth) - 1)
        queue_ahead = np.where(
            levels_clamped < 0, 0.0,
            cum_depth[np.clip(levels_clamped, 0, len(cum_depth) - 1)]
        )
        qa_val = queue_ahead.astype(np.float32)
        if queue_transform == "log1p":
            qa_val = np.log1p(qa_val)
        q_norm = (qa_val - feat_mean[n_feat + 1]) / feat_std[n_feat + 1]
    else:
        qa_fallback = np.float32(0.0)
        if queue_transform == "log1p":
            qa_fallback = np.log1p(qa_fallback)
        q_norm = np.full(n, (qa_fallback - feat_mean[n_feat + 1]) / feat_std[n_feat + 1], dtype=np.float32)

    z_tiled = np.tile(z_norm, (n, 1))
    d_col = d_norm.reshape(-1, 1)
    q_col = q_norm.reshape(-1, 1)
    s_col = np.full((n, 1), side_enc, dtype=np.float32)

    return np.hstack([z_tiled, d_col, q_col, s_col]).astype(np.float32)


def _make_nn_h_fn(
    side_int: int,
    bundle: Dict[str, Any],
    agent_holder: list,
    *,
    vol_feature_mode: str = "auto",
    day_t0_s: Optional[float] = None,
    day_span_s: Optional[float] = None,
):
    """Build a fill-law callback ``h_fn(sim, delta)`` for one quote side.

    The closure reads the *current* agent from ``agent_holder[0]`` (for the
    ``recent_vol`` feature) so the same callback can be reused across many
    agents (e.g. one per gamma) as long as ``agent_holder[0]`` is updated first.

    ``vol_feature_mode`` selects how the ``recent_vol`` feature (``z_raw[8]``)
    is reconstructed so it matches the definition the checkpoint was *trained*
    with:

    * ``"auto"`` (default) — use the checkpoint's recorded ``vol_mode``; legacy
      checkpoints without it fall back to ``"ewma_event"``.
    * ``"realized_time"`` — the cadence-independent trailing time-grid realised
      vol (``agent.sigma_realized``), matching phantom labels built with
      ``vol_mode='realized_time'``.
    * ``"ewma_event"`` — the legacy per-event EWMA vol (``agent.sigma``), for
      checkpoints trained on the original empirical labels.

    ``day_t0_s`` / ``day_span_s`` (replay time base, seconds) are used to
    reconstruct ``tod_frac`` (``z_raw[9]``) as the fraction of the day elapsed,
    matching the training definition ``(t - t_first) / (t_last - t_first)``.
    When ``day_span_s`` is unknown, ``tod_frac`` falls back to the training mean
    (in-distribution, no spurious extrapolation).
    """
    torch = bundle["torch"]
    model = bundle["model"]
    feat_mean = bundle["feat_mean"]
    feat_std = bundle["feat_std"]
    temperature = bundle["temperature"]
    n_feat = bundle["n_feat"]
    _queue_transform = bundle.get("queue_transform", None)
    _mode = str(vol_feature_mode)
    if _mode == "auto":
        _mode = str(bundle.get("vol_mode") or "ewma_event")
    use_realized = _mode == "realized_time"
    _t0 = None if day_t0_s is None else float(day_t0_s)
    _span = None if (day_span_s is None or day_span_s <= 0) else float(day_span_s)

    def h_fn(sim, delta):
        X = assemble_fill_X(
            side_int, feat_mean, feat_std, n_feat, sim, delta,
            agent=agent_holder[0], use_realized=use_realized,
            t0=_t0, span=_span, queue_transform=_queue_transform,
        )

        with torch.no_grad():
            logits = model(torch.from_numpy(X))
            probs = torch.sigmoid(logits / temperature).numpy()

        return probs

    return h_fn


def _make_numerical_nn_agent(gamma, bundle, agent_holder, *, erg_params,
                             size, solver_tick, poisson_tau, delta_lo,
                             max_iter, tol, vol_feature_mode="auto",
                             day_t0_s=None, day_span_s=None, max_delta=2.0):
    """Construct one ``NumericalErgodicMM`` with NN fill-law callbacks."""
    from .market_maker import NumericalErgodicMM

    # vol_mode is ErgodicMM-only; NN agent always uses EWMA event vol.
    erg_params = {k: v for k, v in erg_params.items() if k != "vol_mode"}

    h_bid = _make_nn_h_fn(1, bundle, agent_holder, vol_feature_mode=vol_feature_mode,
                          day_t0_s=day_t0_s, day_span_s=day_span_s)
    h_ask = _make_nn_h_fn(2, bundle, agent_holder, vol_feature_mode=vol_feature_mode,
                          day_t0_s=day_t0_s, day_span_s=day_span_s)
    agent = NumericalErgodicMM(
        **erg_params,
        gamma=gamma,
        size=size,
        verbose=False,
        solver_tick=solver_tick,
        h_b=h_bid,
        h_a=h_ask,
        poisson_tau=poisson_tau,
        delta_lo=delta_lo,
        max_delta=max_delta,
        max_iter=max_iter,
        tol=tol,
    )
    agent_holder[0] = agent
    return agent


def run_nn_ergodic_single_day(
    gamma: float,
    day: str,
    *,
    db_path: str,
    ckpt_path: str,
    tick_size: float,
    erg_params: Dict[str, Any],
    solver_tick: float,
    max_iter: int,
    tol: float,
    size: int,
    poisson_tau: float,
    delta_lo: float = 0.0,
    bbo_in_tick_index: Optional[bool] = None,
    price_native_to_pln: Optional[float] = None,
    skip_opening: bool = True,
    vol_feature_mode: str = "auto",
    day_t0_s: Optional[float] = None,
    day_span_s: Optional[float] = None,
    hawkes: Union[bool, Any] = True,
    max_delta: float = 2.0,
) -> Dict[str, Any]:
    """Run one NumericalErgodicMM backtest day with NN fill law.

    Reconstructs the NN model and callbacks from the checkpoint path so the
    function is fully picklable (safe for joblib/subprocess on Windows).

    ``bbo_in_tick_index`` / ``price_native_to_pln`` are forwarded to
    ``MMBacktester`` (``None`` keeps its auto-detection).  ``skip_opening``
    is exposed so simulated databases (no opening auction) can disable the
    empirical opening-skip heuristic.
    """
    from .backtest import MMBacktester

    bundle = _load_nn_bundle(ckpt_path)
    agent_holder: list = [None]

    bt = MMBacktester(
        Path(db_path), tick_size=tick_size, hawkes=hawkes,
        load_book_state=True, skip_opening=skip_opening,
        bbo_in_tick_index=bbo_in_tick_index,
        price_native_to_pln=price_native_to_pln,
    )

    agent = _make_numerical_nn_agent(
        gamma, bundle, agent_holder, erg_params=erg_params, size=size,
        solver_tick=solver_tick, poisson_tau=poisson_tau, delta_lo=delta_lo,
        max_iter=max_iter, tol=tol, vol_feature_mode=vol_feature_mode,
        day_t0_s=day_t0_s, day_span_s=day_span_s, max_delta=max_delta,
    )

    stats = bt.run_single(day, agent)
    return {
        "gamma": gamma,
        "day": day,
        "pnl": stats["pnl"],
        "trades": stats["n_trades"],
        "max_inv": stats["max_inventory"],
        "dd": stats["intraday_dd"],
    }


def _load_day(bt, conn, day):
    """Dispatch to the correct loader for ``bt``'s DB type (empirical/sim)."""
    if bt.db_type == "empirical":
        return bt._load_empirical_day(conn, day)
    start, end = day
    return bt._load_sim_window(conn, start, end)


def run_nn_ergodic_day_all_gammas(
    day: str,
    gammas,
    *,
    db_path: str,
    ckpt_path: str,
    tick_size: float,
    erg_params: Dict[str, Any],
    solver_tick: float,
    max_iter: int,
    tol: float,
    size: int,
    poisson_tau: float,
    delta_lo: float = 0.0,
    bbo_in_tick_index: Optional[bool] = None,
    price_native_to_pln: Optional[float] = None,
    skip_opening: bool = True,
    vol_feature_mode: str = "auto",
    hawkes: Union[bool, Any] = True,
    max_delta: float = 5.0,
    replay_start_s: Optional[float] = None,
    max_replay_s: Optional[float] = None,
    segment_label: Optional[str] = None,
) -> list:
    """Run a *full gamma sweep for one day* with the NN fill law.

    The (large) order/MO tables are loaded **once** and replayed once per
    gamma via ``MMBacktester._replay`` (which builds a fresh simulator and
    Hawkes filter each call and only reads the DataFrames).  The NN model is
    also loaded once.  This is the memory-friendly unit for joblib: spawn one
    worker per day instead of one per (gamma, day), so a day's data is loaded
    once rather than ``len(gammas)`` times.

    Returns a list of per-gamma stat dicts (same schema as
    ``run_nn_ergodic_single_day``).
    """
    import sqlite3
    import sys
    import time as _time

    from .backtest import MMBacktester

    bundle = _load_nn_bundle(ckpt_path)
    agent_holder: list = [None]

    bt = MMBacktester(
        Path(db_path), tick_size=tick_size, hawkes=hawkes,
        load_book_state=True, skip_opening=skip_opening,
        bbo_in_tick_index=bbo_in_tick_index,
        price_native_to_pln=price_native_to_pln,
    )

    conn = sqlite3.connect(str(bt.db_path))
    try:
        orders_df, mos_df = _load_day(bt, conn, day)
    finally:
        conn.close()

    n_events = len(orders_df) if orders_df is not None else 0
    print(f"[{day}] loaded {n_events:,} events, starting {len(gammas)} gammas",
          flush=True)

    # Day span in the *replay* time base (seconds), so tod_frac at inference
    # matches the training definition (fraction of [t_first, t_last] elapsed).
    # Empirical replay rebases t to seconds-from-open (t0=o_times[0], /1e9);
    # sim replay passes raw second timestamps (t0=0, scale=1).
    day_t0_s = day_span_s = None
    if orders_df is not None and not orders_df.empty:
        _tv = orders_df["time_ns"].to_numpy(dtype=np.float64)
        if bt.db_type == "empirical":
            day_t0_s = 0.0
            day_span_s = (_tv[-1] - _tv[0]) / 1e9
        else:
            day_t0_s = float(_tv[0])
            day_span_s = float(_tv[-1] - _tv[0])

    _day_t0 = _time.perf_counter()
    rows: list = []
    _out_day = segment_label or day
    for gi, gamma in enumerate(gammas):
        _g_t0 = _time.perf_counter()
        agent = _make_numerical_nn_agent(
            gamma, bundle, agent_holder, erg_params=erg_params, size=size,
            solver_tick=solver_tick, poisson_tau=poisson_tau, delta_lo=delta_lo,
            max_iter=max_iter, tol=tol, vol_feature_mode=vol_feature_mode,
            day_t0_s=day_t0_s, day_span_s=day_span_s, max_delta=max_delta,
        )
        stats = bt._replay(
            orders_df, mos_df, agent,
            replay_start_s=replay_start_s,
            max_replay_s=max_replay_s,
        )
        _elapsed = _time.perf_counter() - _g_t0
        _pnl = stats.get("pnl", 0.0)
        _trades = stats.get("n_trades", 0)
        _solves = getattr(agent, "n_solves", "?")
        print(f"  [{_out_day}] gamma={gamma:.4f} ({gi+1}/{len(gammas)}) "
              f"pnl={_pnl:+.2f} trades={_trades} solves={_solves} "
              f"[{_elapsed:.1f}s]", flush=True)
        rows.append({
            "gamma": gamma,
            "day": _out_day,
            "source_day": day,
            "replay_start_s": replay_start_s,
            "max_replay_s": max_replay_s,
            "pnl": _pnl,
            "trades": _trades,
            "max_inv": stats.get("max_inventory", 0),
            "dd": stats.get("intraday_dd", 0.0),
        })
    _day_total = _time.perf_counter() - _day_t0
    print(f"[{_out_day}] done — {len(gammas)} gammas in {_day_total:.0f}s", flush=True)
    return rows


def run_numerical_exp_day_all_gammas(
    day: str,
    gammas,
    *,
    db_path: str,
    tick_size: float,
    erg_params: Dict[str, Any],
    size: int,
    solver_tick: float,
    max_iter: int,
    tol: float,
    poisson_tau: float = 1.0,
    delta_lo: float = 0.0,
    max_delta: float = 2.0,
    skip_opening: bool = True,
    hawkes: Union[bool, Any] = False,
    load_book_state: bool = False,
    replay_start_s: Optional[float] = None,
    max_replay_s: Optional[float] = None,
    segment_label: Optional[str] = None,
) -> list:
    """Run a full ``NumericalErgodicMM`` gamma sweep with exponential fill law.

    Like ``run_ergodic_day_all_gammas`` but uses the discrete HJB solver with
    the default ``lambda = A exp(-k delta)`` fill law (no NN callbacks).
    """
    import sqlite3
    import time as _time

    from .backtest import MMBacktester
    from .market_maker import NumericalErgodicMM

    bt = MMBacktester(
        Path(db_path), tick_size=tick_size, hawkes=hawkes,
        load_book_state=load_book_state, skip_opening=skip_opening,
    )

    conn = sqlite3.connect(str(bt.db_path))
    try:
        orders_df, mos_df = _load_day(bt, conn, day)
    finally:
        conn.close()

    n_events = len(orders_df) if orders_df is not None else 0
    print(f"[{day}] loaded {n_events:,} events, starting {len(gammas)} gammas "
          f"(numerical exponential)", flush=True)

    _out_day = segment_label or day
    _day_t0 = _time.perf_counter()
    rows: list = []
    for gi, gamma in enumerate(gammas):
        _g_t0 = _time.perf_counter()
        agent = NumericalErgodicMM(
            **erg_params,
            gamma=gamma,
            size=size,
            verbose=False,
            solver_tick=solver_tick,
            poisson_tau=poisson_tau,
            delta_lo=delta_lo,
            max_delta=max_delta,
            max_iter=max_iter,
            tol=tol,
        )
        stats = bt._replay(
            orders_df, mos_df, agent,
            replay_start_s=replay_start_s,
            max_replay_s=max_replay_s,
        )
        _elapsed = _time.perf_counter() - _g_t0
        _pnl = stats.get("pnl", 0.0)
        _trades = stats.get("n_trades", 0)
        _solves = getattr(agent, "n_solves", "?")
        print(f"  [{_out_day}] gamma={gamma:.4f} ({gi+1}/{len(gammas)}) "
              f"pnl={_pnl:+.2f} trades={_trades} solves={_solves} "
              f"[{_elapsed:.1f}s]", flush=True)
        rows.append({
            "gamma": gamma,
            "day": _out_day,
            "source_day": day,
            "replay_start_s": replay_start_s,
            "max_replay_s": max_replay_s,
            "pnl": _pnl,
            "trades": _trades,
            "max_inv": stats.get("max_inventory", 0),
            "dd": stats.get("intraday_dd", 0.0),
        })
    _day_total = _time.perf_counter() - _day_t0
    print(f"[{_out_day}] done — {len(gammas)} gammas in {_day_total:.0f}s", flush=True)
    return rows


def run_ergodic_day_all_gammas(
    day: str,
    gammas,
    *,
    db_path: str,
    tick_size: float,
    erg_params: Dict[str, Any],
    size: int,
    skip_opening: bool = True,
    hawkes: Union[bool, Any] = False,
    replay_start_s: Optional[float] = None,
    max_replay_s: Optional[float] = None,
    segment_label: Optional[str] = None,
) -> list:
    """Run a full analytical ``ErgodicMM`` gamma sweep for one day.

    Like ``run_nn_ergodic_day_all_gammas`` but for the closed-form agent:
    load the day once, replay once per gamma.  Returns a list of per-gamma
    stat dicts using the analytical schema (``n_trades`` / ``intraday_dd``).
    """
    import sqlite3

    from .backtest import MMBacktester
    from .market_maker import ErgodicMM

    bt = MMBacktester(
        Path(db_path), tick_size=tick_size, hawkes=hawkes,
        load_book_state=False, skip_opening=skip_opening,
    )

    conn = sqlite3.connect(str(bt.db_path))
    try:
        orders_df, mos_df = _load_day(bt, conn, day)
    finally:
        conn.close()

    rows: list = []
    _out_day = segment_label or day
    for gamma in gammas:
        agent = ErgodicMM(**erg_params, gamma=gamma, size=size, verbose=False)
        stats = bt._replay(
            orders_df, mos_df, agent,
            replay_start_s=replay_start_s,
            max_replay_s=max_replay_s,
        )
        rows.append({
            "gamma": gamma,
            "day": _out_day,
            "source_day": day,
            "replay_start_s": replay_start_s,
            "max_replay_s": max_replay_s,
            "pnl": stats.get("pnl", 0.0),
            "n_trades": stats.get("n_trades", 0),
            "max_inventory": stats.get("max_inventory", 0),
            "intraday_dd": stats.get("intraday_dd", 0.0),
        })
    return rows


def run_ergodic_segment_vol_modes(
    segment: Dict[str, Any],
    gammas,
    vol_modes,
    *,
    bt_kw: Dict[str, Any],
    erg_params: Dict[str, Any],
) -> Dict[str, list]:
    """Run ErgodicMM γ-sweeps for each ``vol_mode`` on one day/segment."""
    seg_kw: Dict[str, Any] = {}
    if segment.get("replay_start_s") is not None:
        seg_kw["replay_start_s"] = segment["replay_start_s"]
    if segment.get("max_replay_s") is not None:
        seg_kw["max_replay_s"] = segment["max_replay_s"]
    if segment.get("label"):
        seg_kw["segment_label"] = segment["label"]

    rows_by_mode: Dict[str, list] = {}
    for mode in vol_modes:
        params = {**erg_params, "vol_mode": mode}
        rows_by_mode[mode] = run_ergodic_day_all_gammas(
            segment["day"], gammas, erg_params=params, **bt_kw, **seg_kw,
        )
    return rows_by_mode


def run_multi_vol_mode_ergodic_sweep(
    segments: List[Dict[str, Any]],
    gammas,
    vol_modes,
    *,
    bt_kw: Dict[str, Any],
    erg_params: Dict[str, Any],
    n_jobs: int = -1,
    backend: str = "loky",
) -> Dict[str, list]:
    """Parallel ErgodicMM γ-sweeps across segments, grouped by ``vol_mode``."""
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(run_ergodic_segment_vol_modes)(
            seg, gammas, vol_modes, bt_kw=bt_kw, erg_params=erg_params,
        )
        for seg in segments
    )
    rows_by_mode: Dict[str, list] = {m: [] for m in vol_modes}
    for res in results:
        for m in vol_modes:
            rows_by_mode[m].extend(res[m])
    return rows_by_mode


def run_segment_sweep_pair(
    segment: Dict[str, Any],
    gammas,
    *,
    an_kw: Dict[str, Any],
    nn_kw: Dict[str, Any],
) -> Dict[str, Any]:
    """Run analytical + NN gamma sweeps for one planned 8 h segment."""
    seg_kw = dict(
        replay_start_s=segment["replay_start_s"],
        max_replay_s=segment["max_replay_s"],
        segment_label=segment.get("label", segment["day"]),
    )
    rows_an = run_ergodic_day_all_gammas(
        segment["day"], gammas, **an_kw, **seg_kw,
    )
    rows_nn = run_nn_ergodic_day_all_gammas(
        segment["day"], gammas, **nn_kw, **seg_kw,
    )
    return {"segment": segment, "an": rows_an, "nn": rows_nn}


def run_multi_segment_gamma_sweep(
    segments: List[Dict[str, Any]],
    gammas,
    *,
    an_kw: Dict[str, Any],
    nn_kw: Dict[str, Any],
    n_jobs: int = -1,
    backend: str = "loky",
) -> List[Dict[str, Any]]:
    """Parallel γ-sweep over planned segments (one worker per segment)."""
    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(run_segment_sweep_pair)(seg, gammas, an_kw=an_kw, nn_kw=nn_kw)
        for seg in segments
    )


def run_ergodic_segment_gammas(
    segment: Dict[str, Any],
    gammas,
    *,
    bt_kw: Dict[str, Any],
    erg_params: Dict[str, Any],
) -> list:
    """ErgodicMM γ-sweep on one planned segment (analytical only)."""
    seg_kw: Dict[str, Any] = {}
    if segment.get("replay_start_s") is not None:
        seg_kw["replay_start_s"] = segment["replay_start_s"]
    if segment.get("max_replay_s") is not None:
        seg_kw["max_replay_s"] = segment["max_replay_s"]
    if segment.get("label"):
        seg_kw["segment_label"] = segment["label"]
    return run_ergodic_day_all_gammas(
        segment["day"], gammas, erg_params=erg_params, **bt_kw, **seg_kw,
    )


def run_multi_segment_ergodic_sweep(
    segments: List[Dict[str, Any]],
    gammas,
    *,
    bt_kw: Dict[str, Any],
    erg_params: Dict[str, Any],
    n_jobs: int = -1,
    backend: str = "loky",
) -> list:
    """Parallel ErgodicMM γ-sweep across segments (one worker per segment)."""
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(run_ergodic_segment_gammas)(
            seg, gammas, bt_kw=bt_kw, erg_params=erg_params,
        )
        for seg in segments
    )
    rows: list = []
    for part in results:
        rows.extend(part)
    return rows

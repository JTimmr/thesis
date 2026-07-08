"""Multi-agent NN market-maker competition sims (live/generative ``Simulate``).

This module wires the *replay-trained* NN fill-probability model into the
generative :class:`research_core.classes.simulate.Simulate` so that ``N``
identical NN ``NumericalErgodicMM`` agents can compete inside one simulation.

Two pieces make this work:

1. A small **feature view** (`_SimFeatureView`) that presents the live
   ``Simulate`` to the *unchanged* replay closure ``_make_nn_h_fn`` in the
   exact shape / units it was validated with:

   * ``ob.get_bbo()`` returns **PLN** (the generative book stores tick
     indices; the replay path feeds PLN prices), so ``spread_ticks`` and
     ``queue_ahead`` come out identical to training.
   * ``hawkes_filter.intensity(t)`` proxies ``sim._compute_intensities(t)[0]``
     (same 6-dim ``DEFAULT_LABELS`` order).
   * ``book_state`` exposes ``imbalance`` (= L0 bid share, matching
     ``Simulate._snapshot_pre_event``) and 40-level ``bid_depths`` /
     ``ask_depths`` for ``queue_ahead``.

   The view is handed to the agent as ``state_extractor`` so the HJB
   pricing path (which reads the *real* sim, tick-index -> PLN) is untouched.

2. :class:`CadenceNNErgodicMM`, a thin subclass of ``NumericalErgodicMM`` that
   re-quotes on a fixed **1-second cadence** (instead of on every BBO move),
   preserves queue priority on unchanged sides, and records per-quote
   *predicted* fill probability together with the *realized* filled fraction
   over the following 1s window (``fill_probe_log``), plus an inventory/PnL
   time series (``state_log``).

``run_competition_sim`` is a picklable joblib worker (one generative sim with
``n_agents`` agents).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .market_maker import NumericalErgodicMM
from .mm_backtest_parallel import _load_nn_bundle, _make_nn_h_fn

# Same value as PhantomLabelConfig.n_queue_levels used to build the sim labels.
DEFAULT_N_QUEUE_LEVELS = 40


# --- Live-sim feature view ---
class _LiveHawkesView:
    """Proxy exposing ``intensity(t)`` / ``_t_last`` from a live ``Simulate``."""

    __slots__ = ("_sim",)

    def __init__(self, sim):
        self._sim = sim

    @property
    def _t_last(self):
        return float(getattr(self._sim, "current_time", 0.0) or 0.0)

    def intensity(self, t):
        inten, _ = self._sim._compute_intensities(float(t))
        return np.asarray(inten, dtype=np.float64)


class _LiveBookStateView:
    """Proxy exposing ``imbalance`` / ``bid_depths`` / ``ask_depths``.

    Definitions match the sim DB the NN was trained on:
    ``imbalance = bb_size / (bb_size + ba_size)`` at L0 (see
    ``Simulate._snapshot_pre_event``); ``bid_depths[i]`` is the resting volume
    ``i`` ticks behind the best (``ask_depths`` symmetric).
    """

    __slots__ = ("_sim", "_n")

    def __init__(self, sim, n_levels=DEFAULT_N_QUEUE_LEVELS):
        self._sim = sim
        self._n = int(n_levels)

    @property
    def imbalance(self):
        ob = self._sim.ob
        bb, ba = ob.get_bbo()
        if bb is None or ba is None:
            return 0.5
        b = ob.bid_qty.get(bb, 0)
        a = ob.ask_qty.get(ba, 0)
        tot = b + a
        return (b / tot) if tot > 0 else 0.5

    @property
    def bid_depths(self):
        ob = self._sim.ob
        bb, _ = ob.get_bbo()
        if bb is None:
            return np.zeros(self._n, dtype=np.float64)
        return np.array(
            [ob.bid_qty.get(bb - i, 0) for i in range(self._n)], dtype=np.float64
        )

    @property
    def ask_depths(self):
        ob = self._sim.ob
        _, ba = ob.get_bbo()
        if ba is None:
            return np.zeros(self._n, dtype=np.float64)
        return np.array(
            [ob.ask_qty.get(ba + i, 0) for i in range(self._n)], dtype=np.float64
        )


class _BboPlnView:
    """``ob`` proxy whose ``get_bbo()`` returns PLN prices (tick_idx * scale)."""

    __slots__ = ("_sim", "_ntp")

    def __init__(self, sim):
        self._sim = sim
        self._ntp = float(getattr(sim, "price_native_to_pln", sim.tick_size) or 1.0)

    def get_bbo(self):
        bb, ba = self._sim.ob.get_bbo()
        if bb is None or ba is None:
            return None, None
        return bb * self._ntp, ba * self._ntp


class _SimFeatureView:
    """A ``sim``-shaped object the replay NN closure can read unchanged."""

    __slots__ = ("tick_size", "hawkes_filter", "book_state", "ob")

    def __init__(self, sim, n_queue_levels=DEFAULT_N_QUEUE_LEVELS):
        self.tick_size = float(getattr(sim, "tick_size", 0.05))
        self.hawkes_filter = _LiveHawkesView(sim)
        self.book_state = _LiveBookStateView(sim, n_queue_levels)
        self.ob = _BboPlnView(sim)


class SimFeatureExtractor:
    """Picklable ``state_extractor``: live ``Simulate`` -> :class:`_SimFeatureView`."""

    def __init__(self, n_queue_levels=DEFAULT_N_QUEUE_LEVELS):
        self.n_queue_levels = int(n_queue_levels)

    def __call__(self, sim):
        return _SimFeatureView(sim, self.n_queue_levels)


def make_sim_nn_h_fn(
    side_int: int,
    bundle: Dict[str, Any],
    agent_holder: list,
    *,
    day_t0_s: Optional[float] = None,
    day_span_s: Optional[float] = None,
    vol_feature_mode: str = "auto",
):
    """Fill-law callback for the live sim.

    Delegates to the validated replay closure ``_make_nn_h_fn``; the live
    ``Simulate`` is adapted to its expected shape by the agent's
    ``state_extractor`` (:class:`SimFeatureExtractor`), so the feature
    computation is byte-for-byte the same code path as the backtest.
    """
    return _make_nn_h_fn(
        side_int,
        bundle,
        agent_holder,
        vol_feature_mode=vol_feature_mode,
        day_t0_s=day_t0_s,
        day_span_s=day_span_s,
    )


# --- Cadence agent ---
class CadenceNNErgodicMM(NumericalErgodicMM):
    """``NumericalErgodicMM`` that re-quotes once per ``requote_cadence`` seconds.

    * Fills are processed on every event (inventory/cash stay correct).
    * At each cadence tick the previous 1s window is finalized (predicted ``h``
      at the quoted depth vs realized filled fraction) into ``fill_probe_log``,
      then quotes are recomputed; an unchanged side keeps its resting order
      (queue priority preserved).
    * ``state_log`` records ``(t, inventory, cash, mid, mtm)`` each tick.

    Intended for unconstrained agents (``initial_cash=None``).
    """

    def __init__(self, *args, requote_cadence: float = 1.0, n_agents: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.requote_cadence = float(requote_cadence)
        self.n_agents = int(n_agents)
        self._last_quote_sec = -1
        # (t, side, delta_ticks, pred_h, filled_qty, size, mid, spread_ticks,
        #  hawkes_sum, inventory, n_agents)
        self.fill_probe_log: List[tuple] = []
        # (t, inventory, cash, mid, mtm)
        self.state_log: List[tuple] = []
        self._win_bid_filled = 0.0
        self._win_ask_filled = 0.0
        self._pending: Optional[dict] = None
        # Optional online-RL feature logging.  Default ``None`` -> disabled, so
        # ordinary competition/backtest runs are completely unaffected.  A
        # subclass (e.g. RLFillMM) sets this to a list to collect per-window
        # (normalised NN input row, realized fill, predicted h) training tuples.
        self.feature_log: Optional[list] = None

    def _capture_features(self, feature_view, delta_b_pln, delta_a_pln):
        """Hook: return ``(X_b, X_a)`` normalised NN input rows for the quoted
        bid/ask deltas, or ``(None, None)``.  No-op in the base class; the RL
        subclass overrides it to log byte-identical training inputs."""
        return None, None

    # --- Window bookkeeping ---
    def _finalize_window(self):
        pending = self._pending
        if pending is None:
            return
        quote_size = max(1, int(self.size))
        self.fill_probe_log.append((
            pending["t"], 1, pending["delta_b_ticks"], pending["pred_hb"],
            float(self._win_bid_filled), quote_size, pending["mid"],
            pending["spread_ticks"], pending["hawkes_sum"], pending["inventory"],
            self.n_agents,
        ))
        self.fill_probe_log.append((
            pending["t"], 2, pending["delta_a_ticks"], pending["pred_ha"],
            float(self._win_ask_filled), quote_size, pending["mid"],
            pending["spread_ticks"], pending["hawkes_sum"], pending["inventory"],
            self.n_agents,
        ))
        if self.feature_log is not None:
            X_b = pending.get("X_b")
            X_a = pending.get("X_a")
            if X_b is not None:
                self.feature_log.append(
                    (X_b, float(self._win_bid_filled) / quote_size, pending["pred_hb"]))
            if X_a is not None:
                self.feature_log.append(
                    (X_a, float(self._win_ask_filled) / quote_size, pending["pred_ha"]))
        self._pending = None

    def _requote(self, sim, t, bid_price, ask_price, mid, sig_pln, res_nat, got_fill):
        """Per-side cancel/replace: keep a resting order if its price is unchanged."""
        need_bid = (self.bid_oid is None) or (bid_price != self._bid_price)
        need_ask = (self.ask_oid is None) or (ask_price != self._ask_price)

        if need_bid and self.bid_oid is not None:
            sim.agent_cancel_order(self.bid_oid, t)
            self.bid_oid = None
            self._bid_price = None
        if need_ask and self.ask_oid is not None:
            sim.agent_cancel_order(self.ask_oid, t)
            self.ask_oid = None
            self._ask_price = None

        bid_size = int(self.size)
        ask_size = int(self.size)

        if need_bid and bid_price > 0 and bid_size > 0:
            self.bid_oid = sim.agent_place_order(1, bid_price, bid_size, t)
            self._bid_price = bid_price
            self.n_quotes_bid += 1
        if need_ask and ask_price > 0 and ask_size > 0:
            self.ask_oid = sim.agent_place_order(2, ask_price, ask_size, t)
            self._ask_price = ask_price
            self.n_quotes_ask += 1

        if got_fill:
            reason = "fill"
        else:
            reason = "cadence"
        self.quote_log.append((
            float(t),
            float(bid_price) if self.bid_oid is not None else float("nan"),
            float(ask_price) if self.ask_oid is not None else float("nan"),
            int(bid_size) if self.bid_oid is not None else 0,
            int(ask_size) if self.ask_oid is not None else 0,
            float(mid),
            float(sig_pln) if sig_pln is not None else float("nan"),
            float(res_nat) if res_nat is not None else float("nan"),
            reason,
        ))

    # --- Main event hook ---
    def on_event(self, sim, t, fills):
        bb, ba = sim.ob.get_bbo()
        if bb is not None:
            bid_tick = float(bb)
        else:
            bid_tick = float("nan")
        if ba is not None:
            ask_tick = float(ba)
        else:
            ask_tick = float("nan")

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
                    bid_tick, ask_tick, int(self.inventory), float(self.cash)))
                if oid == self.bid_oid:
                    self._win_bid_filled += int(qty)
                    if oid not in sim.ob.order_map:
                        self.bid_oid = None
                        self._bid_price = None
            else:
                self.inventory -= int(qty)
                self.cash += float(px) * int(qty)
                self.trade_log.append((
                    float(t), -1, float(px), int(qty),
                    bid_tick, ask_tick, int(self.inventory), float(self.cash)))
                if oid == self.ask_oid:
                    self._win_ask_filled += int(qty)
                    if oid not in sim.ob.order_map:
                        self.ask_oid = None
                        self._ask_price = None

        if got_fill:
            self._snap_pnl(sim, t)

        if bb is None or ba is None:
            return

        mid = (bb + ba) / 2.0
        # Keep vol trackers fresh on every event (cheap, no HJB solve).
        self._update_vol(mid, t)

        cur_sec = int(math.floor(float(t) / self.requote_cadence))
        if cur_sec <= self._last_quote_sec:
            return

        self._finalize_window()
        self._win_bid_filled = 0.0
        self._win_ask_filled = 0.0
        self._last_quote_sec = cur_sec

        bid_price, ask_price, sig_pln, res_nat = self._compute_quotes(mid, sim)

        feature_view = self.state_extractor(sim)
        native_to_pln = float(getattr(sim, "price_native_to_pln", 1.0) or 1.0)
        mid_pln = mid * native_to_pln
        delta_b_pln = mid_pln - bid_price * native_to_pln
        delta_a_pln = ask_price * native_to_pln - mid_pln
        pred_hb = float(np.atleast_1d(
            self._fn_b(feature_view, np.array([delta_b_pln], dtype=np.float64)))[0])
        pred_ha = float(np.atleast_1d(
            self._fn_a(feature_view, np.array([delta_a_pln], dtype=np.float64)))[0])

        spread_ticks = float(ba - bb)
        inten, _ = sim._compute_intensities(float(t))
        hawkes_sum = float(np.sum(inten))
        tick = float(getattr(self, "tick_size", 0.05)) or 0.05

        self._pending = {
            "t": float(t), "mid": float(mid), "spread_ticks": spread_ticks,
            "hawkes_sum": hawkes_sum, "inventory": int(self.inventory),
            "delta_b_ticks": delta_b_pln / tick, "pred_hb": pred_hb,
            "delta_a_ticks": delta_a_pln / tick, "pred_ha": pred_ha,
        }

        if self.feature_log is not None:
            X_b, X_a = self._capture_features(feature_view, delta_b_pln, delta_a_pln)
            self._pending["X_b"] = X_b
            self._pending["X_a"] = X_a

        self._requote(sim, t, bid_price, ask_price, mid, sig_pln, res_nat, got_fill)

        mtm = self.cash + self.inventory * mid
        self.state_log.append(
            (float(t), int(self.inventory), float(self.cash), float(mid), float(mtm)))

    def liquidate(self, sim, t):
        # Close the final 1s window (passive fills only) before the MO close-out.
        self._finalize_window()
        super().liquidate(sim, t)
        bb, ba = sim.ob.get_bbo()
        if bb is not None and ba is not None:
            mid = (bb + ba) / 2.0
            self.state_log.append((
                float(t), int(self.inventory), float(self.cash),
                float(mid), float(self.cash + self.inventory * mid)))


# --- Joblib worker ---
# Column schemas for the per-agent logs (shared with the notebook).
FILL_COLS = ["t", "side", "delta_ticks", "pred_h", "filled_qty", "size",
             "mid", "spread_ticks", "hawkes_sum", "inventory", "n_agents"]
STATE_COLS = ["t", "inventory", "cash", "mid", "mtm"]
TRADE_COLS = ["t", "sign", "px", "qty", "bb", "ba", "inventory", "cash"]
QUOTE_COLS = ["t", "bid", "ask", "bid_sz", "ask_sz", "mid", "sigma", "res", "reason"]


def init_worker_seed(base_seed: int, run_id: int) -> int:
    """Seed numpy, stdlib random, and torch (if installed) for one worker."""
    import random as stdlib_random

    seed = (int(base_seed) + int(run_id) * 999_983) & 0xFFFF_FFFF
    np.random.seed(seed)
    stdlib_random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)
    except Exception:
        pass
    return seed


def write_competition_log_shards(agents, run_id, n_agents, out_dir):
    """Write this run's per-agent logs to parquet shards under out_dir/setup_NN/.

    One file per (log type, run) holding all agents (run_id / agent_id columns).
    Keeps joblib IPC tiny (only a summary is returned).
    """
    import pandas as pd
    from pathlib import Path

    setup_dir = Path(out_dir) / f"setup_{int(n_agents):02d}"
    setup_dir.mkdir(parents=True, exist_ok=True)

    logs = {"fill_probe": (FILL_COLS, "fill_probe_log", True),
            "state": (STATE_COLS, "state_log", False),
            "trades": (TRADE_COLS, "trade_log", False),
            "quotes": (QUOTE_COLS, "quote_log", False)}
    for log_name, (cols, attr, n_agents_in_cols) in logs.items():
        agent_frames = []
        for agent_idx, agent in enumerate(agents):
            rows = getattr(agent, attr)
            if not rows:
                continue
            frame = pd.DataFrame(rows, columns=cols)
            frame["run_id"] = run_id
            frame["agent_id"] = agent_idx
            if not n_agents_in_cols:
                frame["n_agents"] = n_agents
            agent_frames.append(frame)
        if agent_frames:
            pd.concat(agent_frames, ignore_index=True).to_parquet(
                setup_dir / f"{log_name}_run{int(run_id):02d}.parquet", index=False)


def run_competition_sim(
    run_id: int,
    n_agents: int,
    *,
    T: int,
    ckpt_path: str,
    snapshot_kwargs: Dict[str, Any],
    erg_params: Dict[str, Any],
    size: int,
    gamma: float,
    solver_tick: float,
    poisson_tau: float,
    delta_lo: float,
    max_iter: int,
    tol: float,
    max_delta: float = 2.0,
    arrival_mode: str = "hawkes_multivariate",
    drift_eps: float = 0.0,
    requote_cadence: float = 1.0,
    base_seed: int = 12345,
    day_span_s: Optional[float] = None,
    vol_feature_mode: str = "auto",
    agents_affect_kernels: bool = False,
    agents_affect_mo_sizing: bool = False,
    lo_inside_spread_scale: float = 0.0,
    resil_kappa: float = 0.0,
    resil_tau_s: float = 10.0,
    resil_phi: float = 0.0,
    resil_flow_tau_s: float = 40.0,
    out_dir: Optional[str] = None,
    solver_engine: str = "scan",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one generative ``Simulate`` with ``n_agents`` competing NN MMs.

    Picklable for ``joblib.Parallel`` (reconstructs the NN model and agents
    inside the worker).  ``snapshot_kwargs`` is forwarded to
    ``Simulate.load_real_orderbook_snapshot`` (asset/day_key/snapshot_time/
    tick_size).

    When ``out_dir`` is given the per-agent logs are written to parquet shards
    inside the worker and only a small summary is returned (recommended for the
    big sweeps).  When ``out_dir`` is ``None`` the full per-agent logs are
    returned in the result dict (handy for short validation runs).

    ``agents_affect_kernels`` (default ``False`` here) keeps the Hawkes order
    flow exogenous: the competing MMs trade through the book but their quotes /
    cancels do not excite arrival intensity, so the agent count does not
    mechanically inflate market activity.  ``agents_affect_mo_sizing`` (default
    ``False`` here) prevents agent depth from inflating the sampled MO target
    volume, so the exogenous MO flow is truly constant across N.
    ``lo_inside_spread_scale`` (default ``0.0`` here) removes background
    in-spread limit orders, leaving the inside of the spread to the agents.
    All default to the competition-appropriate settings.
    """
    import os
    import sys

    seed = init_worker_seed(base_seed, run_id)

    from .simulate import SimulateFast

    bundle = _load_nn_bundle(ckpt_path)
    extractor = SimFeatureExtractor(DEFAULT_N_QUEUE_LEVELS)
    tick_size = float(snapshot_kwargs.get("tick_size", 0.05))

    agents: List[CadenceNNErgodicMM] = []
    for agent_idx in range(int(n_agents)):
        holder: list = [None]
        h_bid = make_sim_nn_h_fn(
            1, bundle, holder, day_t0_s=0.0, day_span_s=day_span_s,
            vol_feature_mode=vol_feature_mode)
        h_ask = make_sim_nn_h_fn(
            2, bundle, holder, day_t0_s=0.0, day_span_s=day_span_s,
            vol_feature_mode=vol_feature_mode)
        agent = CadenceNNErgodicMM(
            **erg_params, gamma=gamma, size=size, verbose=False,
            solver_tick=solver_tick, h_b=h_bid, h_a=h_ask,
            poisson_tau=poisson_tau, delta_lo=delta_lo, max_delta=max_delta,
            max_iter=max_iter, tol=tol, state_extractor=extractor,
            solver_engine=solver_engine,
            agent_id=f"nn_{agent_idx}", requote_cadence=requote_cadence,
            n_agents=n_agents)
        holder[0] = agent
        agents.append(agent)

    # UTF-8 + replace so the snapshot loader's unicode prints never raise on a
    # cp1252 console (Windows) while stdout is redirected.
    _devnull = open(os.devnull, "w", encoding="utf-8", errors="replace")
    _old_stdout = sys.stdout
    if not verbose:
        sys.stdout = _devnull
    try:
        sim = SimulateFast(
            arrival_mode=arrival_mode, T=int(T),
            lightweight=True, agents_when_lightweight=True,
            agents=agents, shuffle_agents=True, drift_eps=drift_eps,
            agents_affect_kernels=agents_affect_kernels,
            agents_affect_mo_sizing=agents_affect_mo_sizing,
            lo_inside_spread_scale=lo_inside_spread_scale,
            resil_kappa=resil_kappa, resil_tau_s=resil_tau_s,
            resil_phi=resil_phi, resil_flow_tau_s=resil_flow_tau_s,
            tick_size=tick_size)
        sim.load_real_orderbook_snapshot(**snapshot_kwargs)
        sim.run()
        t_end = float(getattr(sim, "current_time", 0.0) or 0.0)
        for agent in agents:
            agent.liquidate(sim, t_end)
    finally:
        if not verbose:
            sys.stdout = _old_stdout
        _devnull.close()

    # Small per-agent summary (always returned; tiny IPC).
    summary = []
    for agent_idx, agent in enumerate(agents):
        if agent.state_log:
            last_mtm = agent.state_log[-1][4]
        else:
            last_mtm = float(agent.cash)
        summary.append({
            "n_agents": int(n_agents), "run_id": int(run_id), "agent_id": agent_idx,
            "final_cash": float(agent.cash), "final_inventory": int(agent.inventory),
            "realized_pnl": float(last_mtm),
            "n_solves": int(getattr(agent, "n_solves", 0)),
            "n_quotes_bid": int(getattr(agent, "n_quotes_bid", 0)),
            "n_quotes_ask": int(getattr(agent, "n_quotes_ask", 0)),
            "n_fill_probe": len(agent.fill_probe_log),
            "n_trades": len(agent.trade_log),
        })

    result: Dict[str, Any] = {
        "run_id": int(run_id), "n_agents": int(n_agents), "T": int(T),
        "seed": int(seed), "t_end": t_end, "summary": summary,
    }

    if out_dir is not None:
        write_competition_log_shards(agents, int(run_id), int(n_agents), out_dir)
    else:
        result["agents"] = [{
            "agent_id": agent_idx,
            "trade_log": list(agent.trade_log),
            "quote_log": list(agent.quote_log),
            "pnl_snapshots": list(agent.pnl_snapshots),
            "state_log": list(agent.state_log),
            "fill_probe_log": list(agent.fill_probe_log),
            "final_cash": float(agent.cash),
            "final_inventory": int(agent.inventory),
            "n_solves": int(getattr(agent, "n_solves", 0)),
            "n_quotes_bid": int(getattr(agent, "n_quotes_bid", 0)),
            "n_quotes_ask": int(getattr(agent, "n_quotes_ask", 0)),
        } for agent_idx, agent in enumerate(agents)]

    return result

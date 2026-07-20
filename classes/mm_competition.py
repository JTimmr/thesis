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

``run_coordinated_competition_sim`` adds the one-step cooperative benchmark:
individual HJB continuation values are frozen, then a shared controller
re-evaluates fill probabilities for the complete candidate queue configuration
and jointly maximises the agents' Hamiltonian sum.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from ._ergodic_solver import coordinated_side_argmax
from .market_maker import NumericalErgodicMM
from .mm_backtest_parallel import (
    _load_nn_bundle,
    _make_nn_h_fn,
    assemble_fill_X,
)

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
    queue_ahead_mode: str = "exact_fifo",
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
        queue_ahead_mode=queue_ahead_mode,
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

    def __init__(
        self,
        *args,
        requote_cadence: float = 1.0,
        n_agents: int = 1,
        externally_coordinated: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.requote_cadence = float(requote_cadence)
        self.n_agents = int(n_agents)
        self.externally_coordinated = bool(externally_coordinated)
        self._last_quote_sec = -1
        self._coord_got_fill = False
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
            if self.externally_coordinated:
                self._coord_got_fill = True

        if bb is None or ba is None:
            return

        mid = (bb + ba) / 2.0
        # Keep vol trackers fresh on every event (cheap, no HJB solve).
        self._update_vol(mid, t)
        if self.externally_coordinated:
            return

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


def coordinated_background_queue(sim, agents, side, candidate_prices):
    """Background volume ahead of fresh coordinated orders at each price.

    All currently resting orders owned by ``agents`` are removed
    counterfactually.  Background orders at a candidate price are ahead
    because each coordinated candidate is assumed to join the back of that
    FIFO queue.
    """
    prices = np.asarray(candidate_prices, dtype=np.float64)
    qty_by_price = sim.ob.bid_qty if int(side) == 1 else sim.ob.ask_qty

    coordinated_by_price: Dict[float, float] = {}
    for agent in agents:
        oid = agent.bid_oid if int(side) == 1 else agent.ask_oid
        order = sim.ob.order_map.get(oid)
        if order is None or int(order[0]) != int(side):
            continue
        price = float(order[1])
        coordinated_by_price[price] = (
            coordinated_by_price.get(price, 0.0) + float(order[2])
        )

    background_by_price = {
        float(price): max(
            0.0,
            float(quantity) - coordinated_by_price.get(float(price), 0.0),
        )
        for price, quantity in qty_by_price.items()
    }
    result = np.zeros(prices.shape, dtype=np.float64)
    for idx, candidate in enumerate(prices):
        if int(side) == 1:
            result[idx] = sum(
                quantity
                for price, quantity in background_by_price.items()
                if price >= candidate
            )
        else:
            result[idx] = sum(
                quantity
                for price, quantity in background_by_price.items()
                if price <= candidate
            )
    return result


def uniform_executable_price_grids(agent, mid, sim):
    """Map an agent's uniform delta grids to exchange-legal native prices.

    This is the vector counterpart of the historical uniform branch in
    ``NumericalErgodicMM._compute_quotes``: bid prices are rounded down and ask
    prices up.  When ``solver_tick`` is finer than the exchange tick, adjacent
    deltas can intentionally map to the same executable price.
    """
    native_to_pln = float(getattr(sim, "price_native_to_pln", 1.0) or 1.0)
    tick_indexed = bool(getattr(sim, "bbo_in_tick_index", False))
    mid_pln = float(mid) * native_to_pln

    bid_prices = np.empty(agent._dg_b.size, dtype=np.float64)
    ask_prices = np.empty(agent._dg_a.size, dtype=np.float64)
    if tick_indexed:
        for idx, delta in enumerate(agent._dg_b):
            raw_native = (mid_pln - float(delta)) / native_to_pln
            bid_prices[idx] = float(max(1.0, math.floor(raw_native)))
        for idx, delta in enumerate(agent._dg_a):
            raw_native = (mid_pln + float(delta)) / native_to_pln
            ask_prices[idx] = float(max(1.0, math.ceil(raw_native)))
    else:
        exchange_tick = float(agent.tick_size)
        for idx, delta in enumerate(agent._dg_b):
            raw_native = (mid_pln - float(delta)) / native_to_pln
            bid_prices[idx] = round(
                max(
                    exchange_tick,
                    math.floor(raw_native / exchange_tick) * exchange_tick,
                ),
                8,
            )
        for idx, delta in enumerate(agent._dg_a):
            raw_native = (mid_pln + float(delta)) / native_to_pln
            ask_prices[idx] = round(
                max(
                    exchange_tick,
                    math.ceil(raw_native / exchange_tick) * exchange_tick,
                ),
                8,
            )

    if np.any(np.diff(bid_prices) > 0.0):
        raise RuntimeError("uniform bid candidates are not ordered aggressive-to-passive")
    if np.any(np.diff(ask_prices) < 0.0):
        raise RuntimeError("uniform ask candidates are not ordered aggressive-to-passive")
    if bid_prices[0] > ask_prices[0]:
        raise RuntimeError(
            "uniform candidate sides cross; coordinated solving requires delta_lo >= 0"
        )
    return bid_prices, ask_prices


def coordinated_price_group_argmax(value_cube, candidate_prices):
    """Maximise a side while grouping deltas with the same executable price.

    ``coordinated_side_argmax`` operates on ordered price levels.  A uniform
    delta grid can contain several assessment deltas that snap to one exchange
    price.  Those alternatives must share one FIFO rank, not be treated as
    distinct queue levels.  For each agent and rank we retain the best delta
    within each executable-price group, solve over the groups, then map the
    result back to the original delta indices.
    """
    values = np.asarray(value_cube, dtype=np.float64)
    prices = np.asarray(candidate_prices, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("value_cube must be three-dimensional")
    if prices.ndim != 1 or prices.size != values.shape[1]:
        raise ValueError("candidate_prices must match value_cube's level axis")
    if prices.size == 0 or not np.isfinite(prices).all():
        raise ValueError("candidate_prices must be finite and non-empty")

    group_starts = [0]
    for level in range(1, prices.size):
        if prices[level] != prices[level - 1]:
            group_starts.append(level)
    group_ends = group_starts[1:] + [prices.size]

    n_agents = values.shape[0]
    n_ranks = values.shape[2]
    grouped_values = np.empty(
        (n_agents, len(group_starts), n_ranks), dtype=np.float64
    )
    best_original_level = np.empty(
        (n_agents, len(group_starts), n_ranks), dtype=np.int64
    )
    for group_idx, (start, end) in enumerate(zip(group_starts, group_ends)):
        group_slice = values[:, start:end, :]
        local_best = np.argmax(group_slice, axis=1)
        grouped_values[:, group_idx, :] = np.take_along_axis(
            group_slice, local_best[:, None, :], axis=1
        )[:, 0, :]
        best_original_level[:, group_idx, :] = start + local_best

    selected_groups, objective = coordinated_side_argmax(grouped_values)
    selected_levels = np.empty(n_agents, dtype=np.int64)
    for agent_idx in range(n_agents):
        group = int(selected_groups[agent_idx])
        rank = int(np.sum(selected_groups < group))
        rank += int(np.sum(selected_groups[:agent_idx] == group))
        selected_levels[agent_idx] = best_original_level[agent_idx, group, rank]
    return selected_levels, float(objective)


class CoordinatedQuoteController:
    """Synchronous one-step cooperative quote selector.

    Each managed agent first solves its ordinary individual HJB.  Those
    continuation values are then held fixed while the controller maximises the
    sum of candidate Hamiltonians.  NN intensities are re-evaluated using the
    queue ahead induced by the complete candidate quote vector.

    Bid and ask sides are separable when ``delta_lo >= 0``.  Each side is
    solved exactly by :func:`coordinated_side_argmax`; agents sharing a price
    receive FIFO priority in ascending ``agent_id`` order.  Uniform-grid
    deltas that snap to one exchange price are grouped before the solve, so
    they share the same FIFO queue while retaining every assessed delta.
    """

    def __init__(
        self,
        agents,
        bundle,
        *,
        requote_cadence: float,
        poisson_tau: float,
        day_span_s: Optional[float],
        vol_feature_mode: str,
    ):
        self.agents = list(agents)
        self.bundle = bundle
        self.requote_cadence = float(requote_cadence)
        self.poisson_tau = float(poisson_tau)
        self.day_span_s = (
            None if day_span_s is None or day_span_s <= 0 else float(day_span_s)
        )
        mode = str(vol_feature_mode)
        if mode == "auto":
            mode = str(bundle.get("vol_mode") or "ewma_event")
        self.use_realized_vol = mode == "realized_time"
        self._last_quote_sec = -1
        self.n_joint_solves = 0
        # (t, bid objective, ask objective, total objective)
        self.objective_log: List[tuple] = []

    def _predict_h_cube(
        self,
        feature_view,
        side,
        delta_grid,
        background_queue,
    ):
        """Return ``h[i, candidate, coordinated_rank]``."""
        bundle = self.bundle
        torch = bundle["torch"]
        n_agents = len(self.agents)
        n_levels = len(delta_grid)
        delta_rows = np.repeat(
            np.asarray(delta_grid, dtype=np.float64), n_agents
        )
        rank_offsets = np.tile(
            np.arange(n_agents, dtype=np.float64), n_levels
        )

        result = np.empty(
            (n_agents, n_levels, n_agents), dtype=np.float64
        )
        for agent_idx, agent in enumerate(self.agents):
            queue_rows = (
                np.repeat(
                    np.asarray(background_queue, dtype=np.float64), n_agents
                )
                + rank_offsets * float(agent.size)
            )
            X = assemble_fill_X(
                int(side),
                bundle["feat_mean"],
                bundle["feat_std"],
                bundle["n_feat"],
                feature_view,
                delta_rows,
                agent=agent,
                use_realized=self.use_realized_vol,
                t0=0.0,
                span=self.day_span_s,
                queue_transform=bundle.get("queue_transform", None),
                queue_ahead_override=queue_rows,
            )
            with torch.no_grad():
                logits = bundle["model"](torch.from_numpy(X))
                probs = torch.sigmoid(
                    logits / bundle["temperature"]
                ).numpy()
            result[agent_idx] = np.asarray(
                probs, dtype=np.float64
            ).reshape(n_levels, n_agents)
        return result

    @staticmethod
    def _continuation_difference(agent, side):
        q_lots = agent._inventory_lots()
        idx = agent._Q + q_lots
        if idx < 0 or idx >= agent._phi.size:
            raise RuntimeError("inventory is outside the individual HJB grid")
        if int(side) == 1:
            if idx >= agent._phi.size - 1:
                return None
            return float(agent._phi[idx] - agent._phi[idx + 1])
        if idx <= 0:
            return None
        return float(agent._phi[idx] - agent._phi[idx - 1])

    def _solve_side(self, sim, feature_view, side, delta_grid, price_grid):
        n_agents = len(self.agents)
        background_queue = coordinated_background_queue(
            sim, self.agents, side, price_grid
        )
        h_cube = self._predict_h_cube(
            feature_view, side, delta_grid, background_queue
        )
        value_cube = np.empty_like(h_cube)
        deltas = np.asarray(delta_grid, dtype=np.float64)[:, None]

        for agent_idx, agent in enumerate(self.agents):
            p_value = self._continuation_difference(agent, side)
            if p_value is None:
                value_cube[agent_idx].fill(-1e300)
                continue
            h_values = np.clip(
                h_cube[agent_idx],
                float(agent.h_clamp),
                1.0 - float(agent.h_clamp),
            )
            lam_values = -np.log1p(-h_values) / self.poisson_tau
            exponent = -float(agent.gamma) * (deltas - p_value)
            values = (
                lam_values
                / float(agent.gamma)
                * (1.0 - np.exp(np.minimum(exponent, 700.0)))
            )
            value_cube[agent_idx] = np.where(
                exponent > 700.0, -1e300, values
            )

        selected_levels, objective = coordinated_price_group_argmax(
            value_cube, price_grid
        )
        selected_h = np.empty(n_agents, dtype=np.float64)
        for agent_idx in range(n_agents):
            level = int(selected_levels[agent_idx])
            selected_price = float(np.asarray(price_grid)[level])
            if int(side) == 1:
                rank = int(np.sum(
                    np.asarray(price_grid)[selected_levels] > selected_price
                ))
            else:
                rank = int(np.sum(
                    np.asarray(price_grid)[selected_levels] < selected_price
                ))
            rank += int(np.sum(
                np.asarray(price_grid)[selected_levels[:agent_idx]]
                == selected_price
            ))
            selected_h[agent_idx] = h_cube[agent_idx, level, rank]

        return (
            np.asarray(price_grid, dtype=np.float64)[selected_levels],
            np.asarray(delta_grid, dtype=np.float64)[selected_levels],
            selected_h,
            float(objective),
        )

    def _cancel_managed_quotes(self, sim, t):
        for agent in self.agents:
            agent._cancel_all(sim, t)

    def _solve_non_crossing_sides(self, sim, feature_view, first):
        """Solve both sides exactly, enforcing strictly non-crossing quotes.

        With ``delta_lo == 0`` and an integer-tick midpoint, both side grids can
        contain that midpoint price.  Any feasible joint quote vector must then
        either exclude the midpoint from every bid or from every ask.  Solving
        those two alternatives and taking the better total is exact; all other
        bid and ask candidate prices are already strictly separated.
        """
        full_bid = self._solve_side(
            sim, feature_view, 1, first._dg_b, first._pg_b
        )
        full_ask = self._solve_side(
            sim, feature_view, 2, first._dg_a, first._pg_a
        )
        if np.max(full_bid[0]) < np.min(full_ask[0]):
            return full_bid, full_ask

        overlaps = np.intersect1d(
            np.unique(first._pg_b), np.unique(first._pg_a)
        )
        if overlaps.size != 1:
            raise RuntimeError(
                "coordinated candidate sides have an unsupported overlap"
            )
        overlap_price = float(overlaps[0])
        if (
            np.max(full_bid[0]) != overlap_price
            or np.min(full_ask[0]) != overlap_price
        ):
            raise RuntimeError("coordinated side solutions cross away from midpoint")

        ask_outside = np.asarray(first._pg_a) > overlap_price
        bid_outside = np.asarray(first._pg_b) < overlap_price
        if not ask_outside.any() or not bid_outside.any():
            raise RuntimeError("no non-crossing coordinated candidate pair exists")

        ask_restricted = self._solve_side(
            sim,
            feature_view,
            2,
            np.asarray(first._dg_a)[ask_outside],
            np.asarray(first._pg_a)[ask_outside],
        )
        bid_restricted = self._solve_side(
            sim,
            feature_view,
            1,
            np.asarray(first._dg_b)[bid_outside],
            np.asarray(first._pg_b)[bid_outside],
        )

        keep_bids_total = full_bid[3] + ask_restricted[3]
        keep_asks_total = bid_restricted[3] + full_ask[3]
        if keep_bids_total >= keep_asks_total:
            return full_bid, ask_restricted
        return bid_restricted, full_ask

    def on_event(self, sim, t, _fills):
        bb, ba = sim.ob.get_bbo()
        if bb is None or ba is None:
            return
        cur_sec = int(math.floor(float(t) / self.requote_cadence))
        if cur_sec <= self._last_quote_sec:
            return
        self._last_quote_sec = cur_sec

        for agent in self.agents:
            agent._finalize_window()
            agent._win_bid_filled = 0.0
            agent._win_ask_filled = 0.0
            agent._last_quote_sec = cur_sec

        mid = (bb + ba) / 2.0
        quote_metadata = []
        for agent in self.agents:
            computed = agent._compute_quotes(mid, sim, return_deltas=True)
            quote_metadata.append((computed[2], computed[3]))
            if agent.candidate_grid == "uniform":
                agent._pg_b, agent._pg_a = uniform_executable_price_grids(
                    agent, mid, sim
                )

        first = self.agents[0]
        for agent in self.agents[1:]:
            if (
                not np.array_equal(agent._dg_b, first._dg_b)
                or not np.array_equal(agent._dg_a, first._dg_a)
                or not np.array_equal(agent._pg_b, first._pg_b)
                or not np.array_equal(agent._pg_a, first._pg_a)
            ):
                raise RuntimeError(
                    "coordinated agents must share executable candidate grids"
                )

        feature_view = first.state_extractor(sim)
        bid_solution, ask_solution = self._solve_non_crossing_sides(
            sim, feature_view, first
        )
        bid_prices, bid_deltas, pred_hb, bid_objective = bid_solution
        ask_prices, ask_deltas, pred_ha, ask_objective = ask_solution
        self.n_joint_solves += 1
        self.objective_log.append((
            float(t),
            bid_objective,
            ask_objective,
            bid_objective + ask_objective,
        ))

        spread_ticks = float(ba - bb)
        intensities, _ = sim._compute_intensities(float(t))
        hawkes_sum = float(np.sum(intensities))

        for agent_idx, agent in enumerate(self.agents):
            tick = float(agent.tick_size) or 1.0
            agent._pending = {
                "t": float(t),
                "mid": float(mid),
                "spread_ticks": spread_ticks,
                "hawkes_sum": hawkes_sum,
                "inventory": int(agent.inventory),
                "delta_b_ticks": float(bid_deltas[agent_idx]) / tick,
                "pred_hb": float(pred_hb[agent_idx]),
                "delta_a_ticks": float(ask_deltas[agent_idx]) / tick,
                "pred_ha": float(pred_ha[agent_idx]),
            }

        # The objective assumes fresh orders.  Remove every managed quote
        # before placing the complete selected configuration.
        self._cancel_managed_quotes(sim, t)
        for agent_idx, agent in enumerate(self.agents):
            sig_pln, res_nat = quote_metadata[agent_idx]
            got_fill = bool(agent._coord_got_fill)
            agent._coord_got_fill = False
            agent._requote(
                sim,
                t,
                float(bid_prices[agent_idx]),
                float(ask_prices[agent_idx]),
                mid,
                sig_pln,
                res_nat,
                got_fill,
            )
            mtm = agent.cash + agent.inventory * mid
            agent.state_log.append((
                float(t),
                int(agent.inventory),
                float(agent.cash),
                float(mid),
                float(mtm),
            ))


# --- Joblib worker ---
# Column schemas for the per-agent logs (shared with the notebook).
FILL_COLS = ["t", "side", "delta_ticks", "pred_h", "filled_qty", "size",
             "mid", "spread_ticks", "hawkes_sum", "inventory", "n_agents"]
STATE_COLS = ["t", "inventory", "cash", "mid", "mtm"]
TRADE_COLS = ["t", "sign", "px", "qty", "bb", "ba", "inventory", "cash"]
QUOTE_COLS = ["t", "bid", "ask", "bid_sz", "ask_sz", "mid", "sigma", "res", "reason"]


def init_worker_seed(base_seed: int, run_id: int) -> int:
    """Seed the NumPy, Numba, and standard-library simulation RNGs."""
    import random as stdlib_random
    from .simulate import seed_numba_rng

    seed = (int(base_seed) + int(run_id) * 999_983) & 0xFFFF_FFFF
    np.random.seed(seed)
    seed_numba_rng(seed)
    stdlib_random.seed(seed)
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
    drift_eps: float = 0.0,
    requote_cadence: float = 1.0,
    base_seed: int = 12345,
    day_span_s: Optional[float] = None,
    vol_feature_mode: str = "auto",
    agents_affect_kernels: bool = False,
    agents_affect_mo_sizing: bool = False,
    rho_in: float = 0.0,
    resil_kappa: float = 0.0,
    resil_tau: float = 10.0,
    resil_varphi: float = 0.0,
    resil_tau_f: float = 40.0,
    out_dir: Optional[str] = None,
    solver_engine: str = "scan",
    candidate_grid: str = "legal",
    queue_ahead_mode: str = "exact_fifo",
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
    ``rho_in`` (default ``0.0`` here) removes background
    in-spread limit orders, leaving the inside of the spread to the agents.
    All default to the competition-appropriate settings.
    """
    import os
    import sys

    seed = init_worker_seed(base_seed, run_id)

    from .simulate import Simulate

    bundle = _load_nn_bundle(ckpt_path)
    extractor = SimFeatureExtractor(DEFAULT_N_QUEUE_LEVELS)
    tick_size = float(snapshot_kwargs.get("tick_size", 0.05))

    agents: List[CadenceNNErgodicMM] = []
    for agent_idx in range(int(n_agents)):
        holder: list = [None]
        h_bid = make_sim_nn_h_fn(
            1, bundle, holder, day_t0_s=0.0, day_span_s=day_span_s,
            vol_feature_mode=vol_feature_mode,
            queue_ahead_mode=queue_ahead_mode)
        h_ask = make_sim_nn_h_fn(
            2, bundle, holder, day_t0_s=0.0, day_span_s=day_span_s,
            vol_feature_mode=vol_feature_mode,
            queue_ahead_mode=queue_ahead_mode)
        agent = CadenceNNErgodicMM(
            **erg_params, gamma=gamma, size=size, verbose=False,
            solver_tick=solver_tick, candidate_grid=candidate_grid,
            h_b=h_bid, h_a=h_ask,
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
        sim = Simulate(
            T=int(T),
            lightweight=True, agents_when_lightweight=True,
            agents=agents, shuffle_agents=True, drift_eps=drift_eps,
            agents_affect_kernels=agents_affect_kernels,
            agents_affect_mo_sizing=agents_affect_mo_sizing,
            rho_in=rho_in,
            resil_kappa=resil_kappa, resil_tau=resil_tau,
            resil_varphi=resil_varphi, resil_tau_f=resil_tau_f,
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


def run_coordinated_competition_sim(
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
    drift_eps: float = 0.0,
    requote_cadence: float = 1.0,
    base_seed: int = 12345,
    day_span_s: Optional[float] = None,
    vol_feature_mode: str = "auto",
    agents_affect_kernels: bool = False,
    agents_affect_mo_sizing: bool = False,
    rho_in: float = 0.0,
    resil_kappa: float = 0.0,
    resil_tau: float = 10.0,
    resil_varphi: float = 0.0,
    resil_tau_f: float = 40.0,
    out_dir: Optional[str] = None,
    solver_engine: str = "scan",
    candidate_grid: str = "legal",
    queue_ahead_mode: str = "exact_fifo",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the one-step coordinated quoting benchmark.

    Individual HJB continuation values are solved exactly as in
    :func:`run_competition_sim` and then frozen.  At every cadence the complete
    bid and ask quote vectors jointly maximise the sum of agent Hamiltonians,
    with NN fill intensities recomputed for the induced FIFO queue positions.

    The exact side-wise dynamic program requires non-crossing candidate sides,
    hence both legal and uniform candidate grids require ``delta_lo >= 0``.
    Uniform deltas are snapped to executable prices exactly as in the ordinary
    MM path and duplicate prices are grouped into one FIFO level.
    """
    import os
    import sys

    if int(n_agents) < 1:
        raise ValueError("n_agents must be positive")
    if int(n_agents) > 12:
        raise ValueError("the exact coordinated solver supports at most 12 agents")
    if candidate_grid not in ("legal", "uniform"):
        raise ValueError("candidate_grid must be 'legal' or 'uniform'")
    if float(delta_lo) < 0.0:
        raise ValueError("the coordinated benchmark requires delta_lo >= 0")

    seed = init_worker_seed(base_seed, run_id)

    from .simulate import Simulate

    bundle = _load_nn_bundle(ckpt_path)
    extractor = SimFeatureExtractor(DEFAULT_N_QUEUE_LEVELS)
    tick_size = float(snapshot_kwargs.get("tick_size", 0.05))

    agents: List[CadenceNNErgodicMM] = []
    for agent_idx in range(int(n_agents)):
        holder: list = [None]
        h_bid = make_sim_nn_h_fn(
            1,
            bundle,
            holder,
            day_t0_s=0.0,
            day_span_s=day_span_s,
            vol_feature_mode=vol_feature_mode,
            queue_ahead_mode=queue_ahead_mode,
        )
        h_ask = make_sim_nn_h_fn(
            2,
            bundle,
            holder,
            day_t0_s=0.0,
            day_span_s=day_span_s,
            vol_feature_mode=vol_feature_mode,
            queue_ahead_mode=queue_ahead_mode,
        )
        agent = CadenceNNErgodicMM(
            **erg_params,
            gamma=gamma,
            size=size,
            verbose=False,
            solver_tick=solver_tick,
            candidate_grid=candidate_grid,
            h_b=h_bid,
            h_a=h_ask,
            poisson_tau=poisson_tau,
            delta_lo=delta_lo,
            max_delta=max_delta,
            max_iter=max_iter,
            tol=tol,
            state_extractor=extractor,
            solver_engine=solver_engine,
            agent_id=f"coordinated_nn_{agent_idx}",
            requote_cadence=requote_cadence,
            n_agents=n_agents,
            externally_coordinated=True,
        )
        holder[0] = agent
        agents.append(agent)

    controller = CoordinatedQuoteController(
        agents,
        bundle,
        requote_cadence=requote_cadence,
        poisson_tau=poisson_tau,
        day_span_s=day_span_s,
        vol_feature_mode=vol_feature_mode,
    )

    _devnull = open(os.devnull, "w", encoding="utf-8", errors="replace")
    _old_stdout = sys.stdout
    if not verbose:
        sys.stdout = _devnull
    try:
        # Managed agents process fills and update volatility first; the
        # controller then performs one synchronous quote update.
        sim = Simulate(
            T=int(T),
            lightweight=True,
            agents_when_lightweight=True,
            agents=[*agents, controller],
            shuffle_agents=False,
            drift_eps=drift_eps,
            agents_affect_kernels=agents_affect_kernels,
            agents_affect_mo_sizing=agents_affect_mo_sizing,
            rho_in=rho_in,
            resil_kappa=resil_kappa,
            resil_tau=resil_tau,
            resil_varphi=resil_varphi,
            resil_tau_f=resil_tau_f,
            tick_size=tick_size,
        )
        sim.load_real_orderbook_snapshot(**snapshot_kwargs)
        sim.run()
        t_end = float(getattr(sim, "current_time", 0.0) or 0.0)
        for agent in agents:
            agent.liquidate(sim, t_end)
    finally:
        if not verbose:
            sys.stdout = _old_stdout
        _devnull.close()

    if controller.objective_log:
        mean_joint_objective = float(np.mean(
            [row[3] for row in controller.objective_log]
        ))
    else:
        mean_joint_objective = float("nan")

    summary = []
    for agent_idx, agent in enumerate(agents):
        last_mtm = (
            agent.state_log[-1][4] if agent.state_log else float(agent.cash)
        )
        summary.append({
            "n_agents": int(n_agents),
            "run_id": int(run_id),
            "agent_id": agent_idx,
            "quote_mode": "coordinated",
            "final_cash": float(agent.cash),
            "final_inventory": int(agent.inventory),
            "realized_pnl": float(last_mtm),
            "n_solves": int(getattr(agent, "n_solves", 0)),
            "n_joint_solves": int(controller.n_joint_solves),
            "mean_joint_objective": mean_joint_objective,
            "n_quotes_bid": int(getattr(agent, "n_quotes_bid", 0)),
            "n_quotes_ask": int(getattr(agent, "n_quotes_ask", 0)),
            "n_fill_probe": len(agent.fill_probe_log),
            "n_trades": len(agent.trade_log),
        })

    result: Dict[str, Any] = {
        "run_id": int(run_id),
        "n_agents": int(n_agents),
        "T": int(T),
        "seed": int(seed),
        "t_end": t_end,
        "quote_mode": "coordinated",
        "n_joint_solves": int(controller.n_joint_solves),
        "mean_joint_objective": mean_joint_objective,
        "summary": summary,
    }

    if out_dir is not None:
        write_competition_log_shards(
            agents, int(run_id), int(n_agents), out_dir
        )
        if controller.objective_log:
            import pandas as pd
            from pathlib import Path

            setup_dir = Path(out_dir) / f"setup_{int(n_agents):02d}"
            pd.DataFrame(
                controller.objective_log,
                columns=[
                    "t",
                    "bid_objective",
                    "ask_objective",
                    "joint_objective",
                ],
            ).assign(
                run_id=int(run_id),
                n_agents=int(n_agents),
            ).to_parquet(
                setup_dir
                / f"coordinated_objective_run{int(run_id):02d}.parquet",
                index=False,
            )
    else:
        result["objective_log"] = list(controller.objective_log)
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

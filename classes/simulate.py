"""
Event-driven limit order book simulation (Numba-accelerated).

This is the single simulation engine for the project. Performance features:
- Numba JIT for Hawkes intensity computation and Ogata thinning
- BBO caching in the order book (``HeapOrderBookFast``)
- Price-to-OID index for O(1) market order fills
- Whole-LO market-order matching (rounds to the nearest resting order boundary)
- math.log for scalar calls

``SimulateFast`` is kept as a backwards-compatible alias for ``Simulate`` at the
bottom of this module (the old pure-NumPy fork has been merged into this file).

Four recording modes control the storage/performance trade-off:

- ``'full'``        – all SQLite tables (orders, fills, mo_orders, bbo, intensities).
- ``'medium'``      – ``bbo`` and ``mo_orders`` only.
- ``'bbo'``         – ``bbo`` table only (one row per event; no orders/fills/MO).
- ``'lightweight'`` – no SQLite; compact in-memory buffers only.

For calibration loops, set ``capture_mid=True`` to accumulate the per-event mid
series in memory (``get_mid_series()``) with no SQLite round-trip, and
``verbose=False`` to silence per-event progress prints.
"""

import os
import random
import sqlite3
from math import exp, log, tanh
from pathlib import Path
from typing import Optional, Union

import numba
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..data.schema import (
    SIM_CREATE_BBO,
    SIM_CREATE_FILLS,
    SIM_CREATE_INTENSITIES,
    SIM_CREATE_MO_ORDERS,
    SIM_CREATE_ORDERS,
    SIM_INSERT_BBO,
    SIM_INSERT_FILL,
    SIM_INSERT_INTENSITY,
    SIM_INSERT_MO,
    SIM_INSERT_ORDER,
    SIM_ORDER_DEPTH_LEVELS,
)
from .helpers import project_root
from .orderbook import HeapOrderBook


# --- Numba-accelerated helpers (module level, compiled once) ---

@numba.njit(cache=True)
def _compute_intensities_jit(baseline, A_stack, adj_stack, decay_stack, dt, n_kernels):
    """Compute intensity vector via recursive exponential decay (JIT)."""
    d = baseline.shape[0]
    intensities = baseline.copy()
    for k in range(n_kernels):
        for i in range(d):
            s = 0.0
            for j in range(d):
                decayed = A_stack[k, i, j] * np.exp(-decay_stack[k, i, j] * dt)
                s += adj_stack[k, i, j] * decay_stack[k, i, j] * decayed
            intensities[i] += s
    for i in range(d):
        if intensities[i] < 0.0:
            intensities[i] = 0.0
    return intensities


@numba.njit(cache=True)
def _decay_A_stack(A_stack, decay_stack, dt, n_kernels):
    """Decay the auxiliary matrix stack in-place."""
    d = A_stack.shape[1]
    for k in range(n_kernels):
        for i in range(d):
            for j in range(d):
                A_stack[k, i, j] *= np.exp(-decay_stack[k, i, j] * dt)


@numba.njit(cache=True)
def _record_event_jit(A_stack, decay_stack, t_last, t, dim_idx, n_kernels):
    """Update auxiliary state after an event (JIT). Returns updated t_last."""
    dt = t - t_last
    if dt < 0.0:
        dt = 0.0
    d = A_stack.shape[1]
    for k in range(n_kernels):
        for i in range(d):
            for j in range(d):
                A_stack[k, i, j] *= np.exp(-decay_stack[k, i, j] * dt)
        for i in range(d):
            A_stack[k, i, dim_idx] += 1.0
    return t


@numba.njit(cache=True)
def _sample_next_event_jit(baseline, A_stack, adj_stack, decay_stack, t_last, n_kernels):
    """Ogata thinning with JIT. Returns (t_event, dim_idx)."""
    t = t_last
    d = baseline.shape[0]

    for _ in range(100_000):
        # Compute intensity upper bound at current t
        dt = t - t_last
        if dt < 0.0:
            dt = 0.0
        intensities = baseline.copy()
        for k in range(n_kernels):
            for i in range(d):
                s = 0.0
                for j in range(d):
                    decayed = A_stack[k, i, j] * np.exp(-decay_stack[k, i, j] * dt)
                    s += adj_stack[k, i, j] * decay_stack[k, i, j] * decayed
                intensities[i] += s
        lambda_star = 0.0
        for i in range(d):
            if intensities[i] < 0.0:
                intensities[i] = 0.0
            lambda_star += intensities[i]

        if lambda_star < 1e-15:
            t += 0.1
            continue

        # Candidate inter-arrival
        u1 = np.random.random()
        dt_cand = -np.log(u1) / lambda_star
        t_cand = t + dt_cand

        # Actual intensity at candidate
        dt2 = t_cand - t_last
        if dt2 < 0.0:
            dt2 = 0.0
        intensities_cand = baseline.copy()
        for k in range(n_kernels):
            for i in range(d):
                s = 0.0
                for j in range(d):
                    decayed = A_stack[k, i, j] * np.exp(-decay_stack[k, i, j] * dt2)
                    s += adj_stack[k, i, j] * decay_stack[k, i, j] * decayed
                intensities_cand[i] += s
        lambda_cand = 0.0
        for i in range(d):
            if intensities_cand[i] < 0.0:
                intensities_cand[i] = 0.0
            lambda_cand += intensities_cand[i]

        # Accept/reject
        u2 = np.random.random()
        if u2 * lambda_star <= lambda_cand:
            # Categorical sample via cumulative sum
            u3 = np.random.random() * lambda_cand
            cumsum = 0.0
            dim_idx = d - 1
            for i in range(d):
                cumsum += intensities_cand[i]
                if u3 < cumsum:
                    dim_idx = i
                    break
            return t_cand, dim_idx, intensities_cand

        t = t_cand

    # Should not reach here
    return t, 0, baseline.copy()


class HeapOrderBookFast(HeapOrderBook):
    """HeapOrderBook with BBO caching for reduced overhead."""

    def __init__(self):
        super().__init__()
        self._bbo_dirty = True
        self._cached_bb = None
        self._cached_ba = None

    def clear(self):
        super().clear()
        self._bbo_dirty = True
        self._cached_bb = None
        self._cached_ba = None

    def add(self, order_id, side, price, volume):
        super().add(order_id, side, price, volume)
        self._bbo_dirty = True

    def modify(self, order_id, new_volume):
        result = super().modify(order_id, new_volume)
        self._bbo_dirty = True
        return result

    def delete(self, order_id):
        result = super().delete(order_id)
        self._bbo_dirty = True
        return result

    def get_bbo(self):
        if not self._bbo_dirty:
            return self._cached_bb, self._cached_ba
        self._clean_heaps()
        if self.bid_heap and self.ask_heap:
            self._cached_bb = -self.bid_heap[0]
            self._cached_ba = self.ask_heap[0]
        else:
            self._cached_bb = None
            self._cached_ba = None
        self._bbo_dirty = False
        return self._cached_bb, self._cached_ba


class Simulate:
    def __init__(self, arrival_mode, T,
                 agents=None,
                 db_path=None, flush_every=50_000,
                 lightweight=False, recording_mode=None,
                 guard_soft_ratio=0.25, guard_hard_ratio=0.10,
                 liquidity_guard=True, agents_when_lightweight=False,
                 shuffle_agents=False, agents_affect_kernels=True,
                 agents_affect_mo_sizing=True,
                 lo_inside_spread_scale=1.0,
                 lo_p_best=0.10,
                 lo_inside_c1=0.022021, lo_inside_c0=0.006944,
                 lo_inside_c1_hi=0.003620, lo_inside_c0_hi=0.163207,
                 lo_inside_break=9,
                 tick_size=0.05, alpha_scale=0.9, drift_eps=0.0,
                 mo_self_scale=1.0, mo_impact_scale=1.0,
                 resil_kappa=0.0, resil_tau_s=10.0, resil_phi=0.0,
                 resil_flow_tau_s=40.0, resil_xcap=4.0, resil_pmax=0.85,
                 verbose=True, capture_mid=False):
        """Initialise the simulator (Numba-accelerated variant).

        Parameters
        ----------
        arrival_mode : str
            ``'poisson'``, ``'hawkes_univariate'``, or ``'hawkes_multivariate'``.
        T : int
            Number of events to simulate.
        recording_mode : {'full', 'medium', 'bbo', 'lightweight'}, optional
            ``'full'``  – all SQLite tables.
            ``'medium'`` – ``bbo`` and ``mo_orders`` only.
            ``'bbo'``    – ``bbo`` only (timestamp, best bid/ask, mid).
            ``'lightweight'`` – no SQLite; use ``get_compact_results()``.
            Default ``'full'``.
        lightweight : bool, optional
            Deprecated; ``True`` is equivalent to
            ``recording_mode='lightweight'``.  Ignored when
            *recording_mode* is set.
        agents : list, optional
            Agent objects implementing ``on_event(sim, t, fills)``.
        drift_eps : float, optional
            Small directional drift applied to the buy market-order
            baseline: ``MO_bid`` is scaled by ``1 + drift_eps`` while
            ``MO_ask`` is left at its calibrated value.  ``drift_eps > 0``
            injects a mild upward price drift; ``< 0`` a downward one.
            Default ``0.0`` (no drift — the calibrated, near-driftless
            kernel).  Only the ``MO_bid`` baseline is touched, so depth,
            queue dynamics, and the impact propagator are left essentially
            unchanged.
        agents_affect_kernels : bool, optional
            When ``True`` (default), agent actions injected via
            :meth:`inject_event` excite the Hawkes intensity.  When
            ``False`` the agents still trade through the book but their
            order events do not feed the Hawkes excitation state.
        lo_inside_spread_scale : float, optional
            Multiplicative scale on the probability that a sampled
            background limit order is placed inside the spread.  Default
            ``1.0`` reproduces calibrated behaviour; ``0.0`` disables
            in-spread placement entirely.
        lo_p_best : float, optional
            Baseline probability that a background limit order joins the
            own-side best quote.  Empirical KGHM value (flat in spread
            over the well-populated range ``S <= 5``): ``0.10``.
        lo_inside_c1, lo_inside_c0, lo_inside_c1_hi, lo_inside_c0_hi,
        lo_inside_break : optional
            Piecewise-linear baseline inside-spread placement probability

            ``P(inside | S) = 0``                     for ``S < 2``
            ``P(inside | S) = c1 * S + c0``           for ``2 <= S <= break``
            ``P(inside | S) = c1_hi * S + c0_hi``     for ``S > break``

            (clamped to ``[0, 0.5]``).  Defaults are the corrected
            empirical KGHM piecewise fit from
            ``AnalyseMarket.piecewise_inside_fit`` (regime 1 on
            ``2 <= S <= 9``, regime 2 on ``9 < S <= 20``).  The previous
            hard-coded single line (``0.03318 * S + 0.01994``) was fitted
            with an inverted ``ticks_from_best`` sign convention, which
            misclassified shallow *deep* orders as inside-spread
            placements.
        resil_kappa : float, optional
            Strength of the order-book *resiliency* (stimulated-refill)
            mechanism.  The mid is tracked against a band-pass anchor built
            from two EMAs (time constants ``resil_tau_s`` and
            ``2 * resil_tau_s``): ``dev = mid - 2*EMA_tau + EMA_2tau``.
            ``dev`` responds fully to a sudden displacement and decays
            within ~``1.4 * resil_tau_s``, but is exactly zero for a
            steadily trending mid, so persistent (momentum) moves are not
            dragged.  When             ``dev`` is ``x`` ticks, background limit orders
            on the side that would revert the displacement have their
            at-best / inside-spread placement probabilities multiplied by
            ``1 + tanh(resil_kappa * x)`` (momentum side by
            ``1 - tanh(resil_kappa * x)``), a rate-preserving reshuffle of
            aggressive placement across the two sides.  This is the
            microscopic analogue
            of transient price impact decaying: liquidity providers
            re-quote inside the widened spread after a sudden price move,
            which produces short-horizon mean reversion in the mid
            (signature plot dip) while leaving long-horizon dynamics
            intact.  Default ``0.0`` disables the mechanism entirely
            (calibrated legacy behaviour).
        resil_tau_s : float, optional
            Fast EMA time constant (seconds) of the resiliency anchor: sets
            the horizon at which reversion pressure acts (dip trough near
            ``~tau``).  Default ``10.0``.
        resil_phi : float, optional
            Strength of the complementary *trend-chasing* placement bias.
            A smoothed trend signal ``s = EMA_tau - EMA_tau_flow`` (zero at
            the instant of a price jump, builds only when a move persists,
            steady-state ``v * (tau_flow - tau)`` along a trend of speed
            ``v``) shifts the momentum side's at-best/inside placement
            probabilities up by the same rate-preserving
            ``1 +/- tanh(resil_phi * s)`` reshuffle.  This propagates
            persistent moves
            (liquidity providers re-quote in the direction of the
            prevailing flow), producing the medium-horizon variance-ratio
            rise above 1 seen empirically, without touching the
            short-horizon dip created by ``resil_kappa``.  Default ``0.0``
            (off).
        resil_flow_tau_s : float, optional
            Slow EMA time constant (seconds) of the trend signal.
            Default ``40.0``.
        resil_xcap : float, optional
            Clip on the tick deviations used in the placement multipliers
            (guards against runaway multipliers after large moves).
            Default ``4.0``.
        resil_pmax : float, optional
            Upper bound on the boosted ``p_best + p_inside`` placement
            probability mass.  Default ``0.85``.
        """
        self.arrival_mode = arrival_mode

        if recording_mode is not None:
            if recording_mode not in ('full', 'medium', 'bbo', 'lightweight'):
                raise ValueError(
                    f"Unknown recording_mode: {recording_mode!r}. "
                    "Use 'full', 'medium', 'bbo', or 'lightweight'.")
            self.recording_mode = recording_mode
        elif lightweight:
            self.recording_mode = 'lightweight'
        else:
            self.recording_mode = 'full'
        self.tick_size = tick_size
        self.bbo_in_tick_index = True
        self.price_native_to_pln = float(tick_size)
        self.alpha_scale = alpha_scale
        self.drift_eps = float(drift_eps)
        self.T = T

        # --- Liquidity guard ---
        self._guard_soft_ratio = guard_soft_ratio
        self._guard_hard_ratio = guard_hard_ratio
        self._guard_stats = {
            'soft_mo_remapped': 0, 'soft_cxl_remapped': 0,
            'hard_mo_blocked': 0, 'hard_cxl_blocked': 0,
        }
        self.liquidity_guard = liquidity_guard
        self.agents_when_lightweight = agents_when_lightweight
        self.shuffle_agents = bool(shuffle_agents)
        self.agents_affect_kernels = bool(agents_affect_kernels)
        self.agents_affect_mo_sizing = bool(agents_affect_mo_sizing)
        self.lo_inside_spread_scale = float(lo_inside_spread_scale)
        self.lo_p_best = float(lo_p_best)
        self.lo_inside_c1 = float(lo_inside_c1)
        self.lo_inside_c0 = float(lo_inside_c0)
        self.lo_inside_c1_hi = float(lo_inside_c1_hi)
        self.lo_inside_c0_hi = float(lo_inside_c0_hi)
        self.lo_inside_break = int(lo_inside_break)
        self.mo_self_scale = float(mo_self_scale)
        self.mo_impact_scale = float(mo_impact_scale)
        # --- Resiliency placement bias ---
        self.resil_kappa = float(resil_kappa)
        self.resil_tau_s = float(resil_tau_s)
        self.resil_phi = float(resil_phi)
        self.resil_flow_tau_s = float(resil_flow_tau_s)
        self.resil_xcap = float(resil_xcap)
        self.resil_pmax = float(resil_pmax)
        self._resil_ema = None      # fast EMA (tau) of the mid, tick units
        self._resil_ema2 = None     # slow EMA (2*tau) of the mid, tick units
        self._resil_ema_flow = None  # trend EMA (flow_tau) of the mid
        self._resil_t = 0.0         # time of the last EMA update
        self.verbose = bool(verbose)
        # When True, run() accumulates the per-event mid series in memory
        # (no SQLite needed) for calibration loops; see get_mid_series().
        self.capture_mid = bool(capture_mid)
        self._mid_t = []
        self._mid_v = []

        # --- Compact data buffers (lightweight mode) ---
        self._fills_compact = []
        self._mo_compact = []

        # Bin edges for y ∈ [0, 5]
        self.cancel_y_bins = np.linspace(0, 5, 25)

        # Found by Mike & Farmer
        self.cancel_prob_by_y = 0.012 * (1 - np.exp(-1 * self.cancel_y_bins))

        self.cancel_prob_y_min = self.cancel_prob_by_y[0]
        self.cancel_prob_y_max = self.cancel_prob_by_y[-1]

        # Passive-depth power law, refit with the corrected ticks_from_best
        # convention (depth = ticks behind the own-side best, all tfb >= 1).
        self.beta_depth = 2.1524
        self.xmin_depth = 1

        self.queue_cancel_alpha = 8.1029
        self.queue_cancel_beta  = 0.6585

        self.queue_uniform_max = 3500
        self.queue_tail_alpha = 3.99
        self.queue_max_retries = 20

        # Limit Order parameters
        self.lo_mid_min = 2
        self.lo_mid_max = 4000
        self.lo_mid_slope = -0.41
        self.lo_tail_slope = -2.31
        self.lo_tail_max = 200_000

        # Market Order parameters
        self.mo_mid_min = 1
        self.mo_mid_max = 200
        self.mo_mid_slope = -0.3
        self.mo_tail_slope = -2.68
        self.mo_tail_max = 155_000

        # Ticks-walked CDFs per depth quartile
        _tw_path = Path(__file__).resolve().parent.parent / "data" / "mo_depth_data" / "KGHM_tw_quartiles.npz"
        if _tw_path.exists():
            ticks_walked_data = np.load(_tw_path)
            self._tw_depth_bounds = ticks_walked_data["depth_quartile_bounds"]
            self._tw_cdfs = [ticks_walked_data[f"tw_cdf_q{i}"] for i in range(4)]
            self._tw_loaded = True
        else:
            print(f"WARNING: {_tw_path.name} not found — falling back to static MO sizes")
            self._tw_loaded = False

        self._tw_depth_history = []
        self._tw_adaptive_bounds = None
        self._tw_warmup = 2000
        self._tw_recalib_interval = 500

        # Poisson baseline rates
        self.poisson_rates = {
            "MO_bid": 0.071652,
            "MO_ask": 0.066922,
            "LO_bid": 0.656950,
            "LO_ask": 0.652339,
            "CXL_bid": 0.656051,
            "CXL_ask": 0.651098
        }

        # Univariate Hawkes parameters
        self.univariate_baseline = np.array([
            0.019840, 0.018044, 0.164048, 0.165068, 0.205264, 0.204799
        ])

        self.univariate_adjacency = np.diag([
            0.724101, 0.730989, 0.750187, 0.746101, 0.687174, 0.685461
        ])

        self.univariate_decays = np.diag([
            19.977277, 19.981433, 10.111358, 10.046270, 19.986916, 19.982077
        ])

        # Multivariate Hawkes parameters
        self.multivariate_adjacency = np.array([
        [0.428726,0.040907,0.000000,0.017025,0.000000,0.000000],
        [0.057898,0.493649,0.000000,0.013028,0.000000,0.000000],
        [0.165819,0.000000,0.000000,0.000000,1.104000,0.000000],
        [0.000000,0.000000,0.000000,0.000000,0.000000,1.110977],
        [0.810592,2.148012,0.203659,0.068113,0.221105,0.149018],
        [2.174030,0.000000,0.000000,0.095280,0.775037,0.000000]
        ])

        self.multivariate_decays = np.array([
        [1.14739,1.884366,3.273707,16.521833,0.283018,1.488311],
        [1.527990,0.621115,0.101905,7.647492,0.355405,1.115153],
        [9.681731,2.675103,1.777656,0.441333,19.969653,2.545340],
        [1.014375,0.154112,0.516323,4.309765,0.296790,19.964264],
        [4.509318,18.311254,8.092670,4.106688,19.547920,11.121506],
        [17.760429,0.574812,17.071582,0.289585,19.931405,8.093308]
        ])

        self.multivariate_baseline = np.array([
        0.01263952, 0.01149326, 0.00000000, 0.00000000, 0.09788585, 0.00000000
        ])

        self.labels = ["MO_bid","MO_ask","LO_bid","LO_ask","CXL_bid","CXL_ask"]

        # cancellation statistics
        self.cancel_stats = {
            'bid_attempts': 0, 'bid_success': 0,
            'ask_attempts': 0, 'ask_success': 0,
        }
        self.mo_stats = {
            'bid_attempts': 0, 'bid_filled': 0,
            'ask_attempts': 0, 'ask_filled': 0,
        }
        self.lo_stats = []
        self.mo_sizes = []
        self.mo_ticks_walked = []

        self.ob = HeapOrderBookFast()

        self.order_id_counter = 0
        self.lifetimes = []

        # --- Array-based order storage (per side) ---
        initial_order_capacity = 50_000
        self._bid_log_prices = np.empty(initial_order_capacity)
        self._bid_delta0s    = np.empty(initial_order_capacity)
        self._bid_times      = np.empty(initial_order_capacity)
        self._bid_oids       = np.empty(initial_order_capacity, dtype=np.int64)
        self._bid_n          = 0
        self._bid_cap        = initial_order_capacity
        self._bid_oid_idx    = {}

        self._ask_log_prices = np.empty(initial_order_capacity)
        self._ask_delta0s    = np.empty(initial_order_capacity)
        self._ask_times      = np.empty(initial_order_capacity)
        self._ask_oids       = np.empty(initial_order_capacity, dtype=np.int64)
        self._ask_n          = 0
        self._ask_cap        = initial_order_capacity
        self._ask_oid_idx    = {}

        # --- Price-to-OID index for O(1) fills ---
        # Maps price -> insertion-ordered dict {oid: True} used as an
        # ordered set.  Insertion order == arrival order, so iterating
        # yields the FIFO queue (price-time priority) that whole-LO MO
        # matching and the phantom-fill queue model both assume.
        self._bid_price_oids = {}  # price -> {oid: True} (FIFO)
        self._ask_price_oids = {}  # price -> {oid: True} (FIFO)

        # Distribution diagnostics
        self.cancel_y_log = []
        self.cancel_f_log = []
        self.cancel_dsame_log = []

        self.last_log_best_ask = None
        self.last_log_best_bid = None

        # --- Agent infrastructure ---
        if agents is None:
            self.agents = []
        else:
            self.agents = agents
        self.agent_oids = set()
        self._agent_fills = []
        self.last_trade_price = None

        self.current_time = 0.0
        self.current_index = 0
        self.current_stamp = 0.0

        # --- Event database infrastructure ---
        self.db_path = db_path
        self._flush_every = flush_every
        self._db_conn = None
        self._db_cursor = None
        self._orders_buf = []
        self._fills_buf = []
        self._mo_buf = []
        self._bbo_buf = []
        self._intensities_buf = []
        self._prev_event_time = None
        self._prev_mid = None
        self._event_detail = None
        self._mo_fill_log = []

        # --- Unified intensity parameters (stacked 3D arrays for Numba) ---
        d = len(self.labels)
        if self.arrival_mode == "poisson":
            rates = [self.poisson_rates[l] for l in self.labels]
            self._baseline = np.array(rates)
            self._n_kernels = 1
            self._adjacency_list = [np.zeros((d, d))]
            self._decays_list = [np.ones((d, d))]

        elif self.arrival_mode == "hawkes_univariate":
            self._baseline = self.univariate_baseline.copy()
            self._n_kernels = 1
            self._adjacency_list = [self.univariate_adjacency.copy()]
            self._decays_list = [self.univariate_decays.copy()]

        elif self.arrival_mode == "hawkes_multivariate":
            self._baseline = self.multivariate_baseline.copy()
            self._n_kernels = 1
            self._adjacency_list = [self.multivariate_adjacency.copy()]
            self._decays_list = [self.multivariate_decays.copy()]
        else:
            raise ValueError(f"Unknown arrival_mode: {self.arrival_mode}")

        # Symmetric MO self-excitation tuning (single-kernel multivariate only).
        # Default 1.0 leaves calibrated matrices unchanged.
        if self.mo_self_scale != 1.0:
            if self.arrival_mode == "hawkes_multivariate":
                mo_bid = self.labels.index("MO_bid")
                mo_ask = self.labels.index("MO_ask")
                adj = self._adjacency_list[0]
                s = self.mo_self_scale
                adj[mo_bid, mo_bid] *= s
                adj[mo_ask, mo_ask] *= s
            else:
                raise ValueError(
                    "mo_self_scale applies only to hawkes_multivariate"
                )

        # --- Optional directional drift on the buy market-order baseline ---
        # MO_bid (idx 0) is the buy-trade arrival that walks the ask up, so
        # scaling only its baseline injects a mild upward price drift while
        # leaving MO_ask and all LO/CXL (depth, queue, impact) untouched.
        if self.drift_eps != 0.0:
            mo_bid = self.labels.index("MO_bid")
            self._baseline[mo_bid] *= (1.0 + self.drift_eps)
            if self.verbose:
                print(f"drift_eps={self.drift_eps:+.3f}: MO_bid baseline "
                      f"x{1.0 + self.drift_eps:.3f} (MO_ask unchanged)")

        # Stack into contiguous 3D arrays for Numba
        self._adj_stack = np.stack(self._adjacency_list)
        self._decay_stack = np.stack(self._decays_list)
        self._A_stack = np.zeros((self._n_kernels, d, d))
        self._t_last = 0.0

    def load_single_hawkes_params(self, baseline, adjacency, decays):
        """Inject explicit single-kernel (K=1) Hawkes params.

        Used by the mean-reversion calibration search to swap in a perturbed
        cross-excitation matrix. ``baseline`` is ``(d,)``, ``adjacency`` and
        ``decays`` are ``(d, d)``. ``drift_eps`` is re-applied to the
        ``MO_bid`` baseline exactly as in ``__init__``, so pass the *unscaled*
        calibrated baseline. The Numba stacks are rebuilt so ``run()`` sees
        the new params.
        """
        d = len(self.labels)
        bl = np.asarray(baseline, dtype=float).ravel()
        adj = np.asarray(adjacency, dtype=float)
        dec = np.asarray(decays, dtype=float)
        if bl.size != d:
            raise ValueError(f"Expected baseline length {d}, got {bl.shape}")
        if adj.shape != (d, d):
            raise ValueError(f"Expected adjacency shape {(d, d)}, got {adj.shape}")
        if dec.shape != (d, d):
            raise ValueError(f"Expected decays shape {(d, d)}, got {dec.shape}")
        self._baseline = bl.copy()
        self._n_kernels = 1
        self._adjacency_list = [adj.astype(float, copy=True)]
        self._decays_list = [dec.astype(float, copy=True)]
        if self.drift_eps != 0.0:
            mo_bid = self.labels.index("MO_bid")
            self._baseline[mo_bid] *= (1.0 + self.drift_eps)
        self._adj_stack = np.stack(self._adjacency_list)
        self._decay_stack = np.stack(self._decays_list)
        self._A_stack = np.zeros((self._n_kernels, d, d))

    def load_multi_mo_hawkes_params(self, baseline, adjacency, decays,
                                     cross_alphas, self_alphas, betas):
        """Inject a K=(1+N) hybrid kernel: base + N extra MO timescales.

        Kernel 0 is the full calibrated single kernel (all d x d pairs).
        Kernels 1..N each add symmetric MO cross- and self-excitation at a
        specific decay rate (timescale), used to reproduce the multi-timescale
        mean-reversion signature.

        Parameters
        ----------
        baseline : array (d,)
            Unscaled baseline intensities (drift_eps is re-applied internally).
        adjacency : array (d, d)
            Base single-kernel adjacency (branching) matrix.
        decays : array (d, d)
            Base single-kernel decay matrix.
        cross_alphas : list of N floats
            MO cross-excitation adjacency per additional timescale.
            Applied symmetrically: MO_bid<->MO_ask.
        self_alphas : list of N floats
            MO self-excitation adjacency per additional timescale.
            Applied symmetrically: MO_bid->MO_bid and MO_ask->MO_ask.
        betas : list of N floats
            Decay rate for each additional timescale component.
        """
        d = len(self.labels)
        bl = np.asarray(baseline, dtype=float).ravel().copy()
        adj0 = np.array(adjacency, dtype=float, copy=True)
        dec0 = np.array(decays, dtype=float, copy=True)
        if bl.size != d:
            raise ValueError(f"Expected baseline length {d}, got {bl.shape}")
        if adj0.shape != (d, d):
            raise ValueError(f"Expected adjacency shape {(d, d)}, got {adj0.shape}")

        adj_list = [adj0]
        dec_list = [dec0]
        mo_bid, mo_ask = 0, 1

        for alpha_c, alpha_s, beta in zip(cross_alphas, self_alphas, betas):
            adj_k = np.zeros((d, d))
            dec_k = np.ones((d, d))
            adj_k[mo_ask, mo_bid] = alpha_c
            adj_k[mo_bid, mo_ask] = alpha_c
            adj_k[mo_bid, mo_bid] = alpha_s
            adj_k[mo_ask, mo_ask] = alpha_s
            dec_k[mo_ask, mo_bid] = beta
            dec_k[mo_bid, mo_ask] = beta
            dec_k[mo_bid, mo_bid] = beta
            dec_k[mo_ask, mo_ask] = beta
            adj_list.append(adj_k)
            dec_list.append(dec_k)

        self._baseline = bl
        self._n_kernels = len(adj_list)
        self._adjacency_list = adj_list
        self._decays_list = dec_list
        if self.drift_eps != 0.0:
            mo_bid_idx = self.labels.index("MO_bid")
            self._baseline[mo_bid_idx] *= (1.0 + self.drift_eps)
        self._adj_stack = np.stack(self._adjacency_list)
        self._decay_stack = np.stack(self._decays_list)
        self._A_stack = np.zeros((self._n_kernels, d, d))
        self._A_seeded = None

    @property
    def lightweight(self):
        """True iff ``recording_mode == 'lightweight'``."""
        return self.recording_mode == 'lightweight'

    def _compute_intensities(self, t):
        """Compute intensity vector λ(t) via Numba JIT."""
        dt = max(t - self._t_last, 0.0)
        intensities = _compute_intensities_jit(
            self._baseline, self._A_stack, self._adj_stack,
            self._decay_stack, dt, self._n_kernels
        )
        return intensities, None

    # --- Liquidity guard helpers ---

    def _liquidity_state(self):
        bid_depth = self.ob.total_bid_depth
        ask_depth = self.ob.total_ask_depth

        if bid_depth <= 0 or ask_depth <= 0:
            return None

        max_depth = max(bid_depth, ask_depth)
        depth_ratio = min(bid_depth, ask_depth) / max_depth
        if bid_depth < ask_depth:
            thin_side = 'bid'
        else:
            thin_side = 'ask'

        return (thin_side, depth_ratio)

    def _guard_event(self, label):
        state = self._liquidity_state()
        if state is None:
            return label

        thin_side, depth_ratio = state

        if thin_side == 'ask':
            dangerous_mo = 'MO_bid'
            dangerous_cxl = 'CXL_ask'
            safe_lo = 'LO_ask'
        else:
            dangerous_mo = 'MO_ask'
            dangerous_cxl = 'CXL_bid'
            safe_lo = 'LO_bid'

        if label not in (dangerous_mo, dangerous_cxl):
            return label

        if depth_ratio < self._guard_hard_ratio:
            if label == dangerous_mo:
                self._guard_stats['hard_mo_blocked'] += 1
            else:
                self._guard_stats['hard_cxl_blocked'] += 1
            return safe_lo

        if depth_ratio < self._guard_soft_ratio:
            span = self._guard_soft_ratio - self._guard_hard_ratio
            if span > 0:
                p_remap = (self._guard_soft_ratio - depth_ratio) / span
            else:
                p_remap = 1.0
            if random.random() < p_remap:
                if label == dangerous_mo:
                    self._guard_stats['soft_mo_remapped'] += 1
                else:
                    self._guard_stats['soft_cxl_remapped'] += 1
                return safe_lo

        return label

    def _open_db(self, overwrite=False):
        if self.db_path is None:
            return
        if os.path.exists(self.db_path) and not overwrite:
            raise FileExistsError(
                f"Database '{self.db_path}' already exists!\n"
                f"  To use a different name:  sim.db_path = 'new_name.sqlite'\n"
                f"  To overwrite:             sim.run(overwrite=True)"
            )
        if os.path.exists(self.db_path) and overwrite:
            Path(self.db_path).unlink(missing_ok=True)
            print(f"Overwriting existing database: {Path(self.db_path).name}")
        self._db_conn = sqlite3.connect(self.db_path)
        self._db_cursor = self._db_conn.cursor()
        self._db_cursor.execute("PRAGMA journal_mode=WAL")
        self._db_cursor.execute("PRAGMA synchronous=NORMAL")
        if self.recording_mode == 'full':
            self._db_cursor.execute(SIM_CREATE_ORDERS)
            self._db_cursor.execute(SIM_CREATE_FILLS)
            self._db_cursor.execute(SIM_CREATE_INTENSITIES)
        if self.recording_mode != 'bbo':
            self._db_cursor.execute(SIM_CREATE_MO_ORDERS)
        self._db_cursor.execute(SIM_CREATE_BBO)
        self._db_conn.commit()

    def _flush_db(self):
        if self._db_conn is None:
            return
        if self.recording_mode == 'full':
            if self._orders_buf:
                self._db_cursor.executemany(SIM_INSERT_ORDER, self._orders_buf)
                self._orders_buf.clear()
            if self._fills_buf:
                self._db_cursor.executemany(SIM_INSERT_FILL, self._fills_buf)
                self._fills_buf.clear()
            if self._intensities_buf:
                self._db_cursor.executemany(SIM_INSERT_INTENSITY, self._intensities_buf)
                self._intensities_buf.clear()
        if self.recording_mode != 'bbo' and self._mo_buf:
            self._db_cursor.executemany(SIM_INSERT_MO, self._mo_buf)
            self._mo_buf.clear()
        if self._bbo_buf:
            self._db_cursor.executemany(SIM_INSERT_BBO, self._bbo_buf)
            self._bbo_buf.clear()
        self._db_conn.commit()

    def _close_db(self):
        self._flush_db()
        if self._db_conn:
            if self.recording_mode == 'full':
                self._db_cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(timestamp)")
                self._db_cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(timestamp)")
                self._db_cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_int_ts ON intensities(timestamp)")
            if self.recording_mode != 'bbo':
                self._db_cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_mo_ts ON mo_orders(timestamp)")
            self._db_cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_bbo_ts ON bbo(timestamp)")
            self._db_conn.commit()
            self._db_conn.close()
            self._db_conn = None
            self._db_cursor = None

    def _snapshot_pre_event(self, bb, ba):
        if bb is None or ba is None:
            return None
        mid = (bb + ba) / 2.0
        bb_size = self.ob.bid_qty.get(bb, 0)
        ba_size = self.ob.ask_qty.get(ba, 0)
        total_sz = bb_size + ba_size
        return {
            'bb': bb, 'ba': ba, 'mid': mid,
            'spread': ba - bb,
            'bb_size': bb_size, 'ba_size': ba_size,
            'microprice': (bb_size * ba + ba_size * bb) / total_sz
                          if total_sz > 0 else mid,
            'imbalance': bb_size / total_sz if total_sz > 0 else 0.5,
            'total_bid': self.ob.total_bid_depth,
            'total_ask': self.ob.total_ask_depth,
            'n_bid': len(self.ob.bid_qty),
            'n_ask': len(self.ob.ask_qty),
            'n_total': len(self.ob.order_map),
            'bid_depths': [self.ob.bid_qty.get(bb - i, 0) for i in range(5)],
            'ask_depths': [self.ob.ask_qty.get(ba + i, 0) for i in range(5)],
            'opp_ask_10': [self.ob.ask_qty.get(ba + i, 0) for i in range(10)],
            'opp_bid_10': [self.ob.bid_qty.get(bb - i, 0) for i in range(10)],
            # Wide same-side depth for the orders table (phantom queue_ahead);
            # MO/fills rows keep the 5-level arrays above.
            'bid_depths_ord': [self.ob.bid_qty.get(bb - i, 0)
                               for i in range(SIM_ORDER_DEPTH_LEVELS)],
            'ask_depths_ord': [self.ob.ask_qty.get(ba + i, 0)
                               for i in range(SIM_ORDER_DEPTH_LEVELS)],
        }

    def _record_to_db(self, t, label, pre, counter):
        if pre is None:
            return

        ts = self.tick_size
        bb = pre['bb']; ba = pre['ba']; mid = pre['mid']
        spread = pre['spread']
        dt_prev = ((t - self._prev_event_time)
                    if self._prev_event_time is not None else 0.0)
        dp_mid = ((mid - self._prev_mid)
                   if self._prev_mid is not None else 0.0)

        self._bbo_buf.append((t, bb * ts, ba * ts, mid * ts))

        detail = self._event_detail
        if detail is None:
            self._prev_event_time = t
            self._prev_mid = mid
            return

        etype = detail.get('type')

        if etype in ('LO', 'CXL'):
            if self.recording_mode == 'full':
                side = detail['side']
                price = detail['price']
                if side == 1:
                    best_same = bb
                else:
                    best_same = ba
                ticks_from_mid = int(round(price - mid))
                if etype == 'CXL':
                    is_cancel = 1
                else:
                    is_cancel = 0

                row = (
                    t, etype, detail['oid'], side, price * ts,
                    bb * ts, ba * ts, best_same * ts, pre['bb_size'], pre['ba_size'],
                    pre['total_bid'], pre['total_ask'], mid * ts,
                    ticks_from_mid, spread * ts, int(spread), pre['imbalance'],
                    detail.get('ticks_from_best', 0),
                    detail.get('queue_ahead', 0),
                    detail.get('volume', 0),
                    detail.get('delta0', 0.0),
                    detail.get('delta_t', detail.get('delta0', 0.0)),
                    detail.get('y_ratio', 1.0),
                    dt_prev, pre['n_total'], is_cancel, pre['microprice'] * ts,
                    pre['n_bid'], pre['n_ask'], dp_mid * ts,
                    *pre['bid_depths_ord'], *pre['ask_depths_ord'],
                )
                self._orders_buf.append(row)

        elif etype == 'MO':
            side_text = detail['side_text']
            side_int = detail['side_int']
            fills = detail.get('fills', [])
            mo_vol = detail['mo_volume']
            tw = detail['ticks_walked']

            if side_int == 1:
                opp_10 = pre['opp_ask_10']
                L0_depth = pre['ba_size']
            else:
                opp_10 = pre['opp_bid_10']
                L0_depth = pre['bb_size']

            ratio_L0 = (mo_vol / L0_depth if L0_depth > 0
                        else float('inf'))

            if fills:
                fill_prices = [p * ts for p, _ in fills]
            else:
                fill_prices = [0]
            mo_row = (
                t, side_text, mo_vol, len(fills),
                min(fill_prices), max(fill_prices),
                bb * ts, ba * ts, tw, ratio_L0, pre['microprice'] * ts,
                *opp_10, *pre['bid_depths'], *pre['ask_depths'],
            )
            self._mo_buf.append(mo_row)

            if self.recording_mode == 'full':
                for fill_price, fill_vol in fills:
                    if side_int == 1:
                        ticks_from_bbo = int(fill_price - ba)
                    else:
                        ticks_from_bbo = int(bb - fill_price)
                    fill_row = (
                        t, fill_vol, fill_price * ts, side_text,
                        bb * ts, ba * ts, ticks_from_bbo, pre['microprice'] * ts,
                        *opp_10, *pre['bid_depths'], *pre['ask_depths'],
                    )
                    self._fills_buf.append(fill_row)

        self._prev_event_time = t
        self._prev_mid = mid

        if counter % self._flush_every == 0:
            self._flush_db()

    def _record_event(self, t, dim_idx):
        """Update auxiliary state after an event of type *dim_idx* at time *t*."""
        self._t_last = _record_event_jit(
            self._A_stack, self._decay_stack, self._t_last, t, dim_idx, self._n_kernels
        )

    def _sample_next_event(self):
        """Sample the next event using Ogata's thinning (Numba JIT)."""
        t_cand, dim_idx, intensities_cand = _sample_next_event_jit(
            self._baseline, self._A_stack, self._adj_stack,
            self._decay_stack, self._t_last, self._n_kernels
        )
        # Update auxiliary state
        self._t_last = _record_event_jit(
            self._A_stack, self._decay_stack, self._t_last, t_cand, dim_idx, self._n_kernels
        )
        return t_cand, self.labels[dim_idx], intensities_cand

    def inject_event(self, t, label):
        """Inject an external event into the Hawkes excitation state.

        When ``self.agents_affect_kernels`` is ``False`` this is a no-op.
        """
        if not self.agents_affect_kernels:
            return
        dim_idx = self.labels.index(label)
        self._record_event(t, dim_idx)

    def next_id(self):
        self.order_id_counter += 1
        return self.order_id_counter

    def _add_order(self, side, oid, log_price, delta0_log, t):
        """Append an order to the compact array storage. O(1) amortized."""
        if side == 1:
            n = self._bid_n
            if n >= self._bid_cap:
                self._bid_cap *= 2
                c = self._bid_cap
                self._bid_log_prices = np.resize(self._bid_log_prices, c)
                self._bid_delta0s    = np.resize(self._bid_delta0s, c)
                self._bid_times      = np.resize(self._bid_times, c)
                self._bid_oids       = np.resize(self._bid_oids, c)
            self._bid_log_prices[n] = log_price
            self._bid_delta0s[n]    = delta0_log
            self._bid_times[n]      = t
            self._bid_oids[n]       = oid
            self._bid_oid_idx[oid]  = n
            self._bid_n = n + 1
            # Price-to-OID index
            if oid in self.ob.order_map:
                price = self.ob.order_map[oid][1]
            else:
                price = None
            if price is not None:
                if price not in self._bid_price_oids:
                    self._bid_price_oids[price] = {}
                self._bid_price_oids[price][oid] = True
        else:
            n = self._ask_n
            if n >= self._ask_cap:
                self._ask_cap *= 2
                c = self._ask_cap
                self._ask_log_prices = np.resize(self._ask_log_prices, c)
                self._ask_delta0s    = np.resize(self._ask_delta0s, c)
                self._ask_times      = np.resize(self._ask_times, c)
                self._ask_oids       = np.resize(self._ask_oids, c)
            self._ask_log_prices[n] = log_price
            self._ask_delta0s[n]    = delta0_log
            self._ask_times[n]      = t
            self._ask_oids[n]       = oid
            self._ask_oid_idx[oid]  = n
            self._ask_n = n + 1
            # Price-to-OID index
            if oid in self.ob.order_map:
                price = self.ob.order_map[oid][1]
            else:
                price = None
            if price is not None:
                if price not in self._ask_price_oids:
                    self._ask_price_oids[price] = {}
                self._ask_price_oids[price][oid] = True

    def _remove_order(self, side, oid):
        """Remove an order via swap-and-pop. O(1). Returns placement time."""
        if side == 1:
            idx_map = self._bid_oid_idx
            if oid not in idx_map:
                return None
            i = idx_map.pop(oid)
            t_placed = float(self._bid_times[i])
            # Remove from price-to-oid index
            if oid in self.ob.order_map:
                price = self.ob.order_map[oid][1]
                s = self._bid_price_oids.get(price)
                if s:
                    s.pop(oid, None)
                    if not s:
                        del self._bid_price_oids[price]
            else:
                # Order already removed from book; scan price index
                for price, s in list(self._bid_price_oids.items()):
                    if oid in s:
                        s.pop(oid, None)
                        if not s:
                            del self._bid_price_oids[price]
                        break
            self._bid_n -= 1
            last = self._bid_n
            if i < last:
                moved_oid = int(self._bid_oids[last])
                self._bid_log_prices[i] = self._bid_log_prices[last]
                self._bid_delta0s[i]    = self._bid_delta0s[last]
                self._bid_times[i]      = self._bid_times[last]
                self._bid_oids[i]       = moved_oid
                idx_map[moved_oid] = i
            return t_placed
        else:
            idx_map = self._ask_oid_idx
            if oid not in idx_map:
                return None
            i = idx_map.pop(oid)
            t_placed = float(self._ask_times[i])
            # Remove from price-to-oid index
            if oid in self.ob.order_map:
                price = self.ob.order_map[oid][1]
                s = self._ask_price_oids.get(price)
                if s:
                    s.pop(oid, None)
                    if not s:
                        del self._ask_price_oids[price]
            else:
                for price, s in list(self._ask_price_oids.items()):
                    if oid in s:
                        s.pop(oid, None)
                        if not s:
                            del self._ask_price_oids[price]
                        break
            self._ask_n -= 1
            last = self._ask_n
            if i < last:
                moved_oid = int(self._ask_oids[last])
                self._ask_log_prices[i] = self._ask_log_prices[last]
                self._ask_delta0s[i]    = self._ask_delta0s[last]
                self._ask_times[i]      = self._ask_times[last]
                self._ask_oids[i]       = moved_oid
                idx_map[moved_oid] = i
            return t_placed

    def _get_order_data(self, side, oid):
        if side == 1:
            idx_map = self._bid_oid_idx
        else:
            idx_map = self._ask_oid_idx
        if oid not in idx_map:
            return (0.0, 1.0)
        i = idx_map[oid]
        if side == 1:
            return (float(self._bid_log_prices[i]), float(self._bid_delta0s[i]))
        else:
            return (float(self._ask_log_prices[i]), float(self._ask_delta0s[i]))

    # --- Agent interaction helpers ---

    def agent_place_order(self, side, price, volume, t):
        """Place a limit order on behalf of an agent."""
        oid = self.next_id()
        self.ob.add(oid, side, price, volume)
        self.agent_oids.add(oid)

        bb, ba = self.ob.get_bbo()
        if price > 0:
            log_price = log(price)
        else:
            log_price = 0.0

        if side == 1:
            if ba and ba > 0 and price > 0:
                delta0_log = log(ba) - log_price
            else:
                delta0_log = 1.0
            self._add_order(1, oid, log_price, delta0_log, t)
            self.inject_event(t, "LO_bid")
        else:
            if bb and bb > 0 and price > 0:
                delta0_log = log_price - log(bb)
            else:
                delta0_log = 1.0
            self._add_order(2, oid, log_price, delta0_log, t)
            self.inject_event(t, "LO_ask")

        return oid

    def agent_cancel_order(self, oid, t):
        """Cancel an agent's resting order."""
        if oid not in self.ob.order_map:
            self.agent_oids.discard(oid)
            self._remove_order(1, oid)
            self._remove_order(2, oid)
            return False

        side, price, vol = self.ob.order_map[oid]
        self.ob.delete(oid)
        self.agent_oids.discard(oid)
        self._remove_order(side, oid)

        if side == 1:
            self.inject_event(t, "CXL_bid")
        else:
            self.inject_event(t, "CXL_ask")

        return True

    def agent_market_order(self, side, volume, t):
        """Execute a market order on behalf of an agent."""
        fills = []
        remaining = volume

        if side == 1:  # buy — consume asks
            self.inject_event(t, "MO_bid")
            while remaining > 0:
                bb, ba = self.ob.get_bbo()
                if ba is None:
                    break
                oids_at_ba = self._ask_price_oids.get(ba)
                if not oids_at_ba:
                    break
                oid = next(iter(oids_at_ba))
                if oid not in self.ob.order_map:
                    oids_at_ba.pop(oid, None)
                    self._remove_order(2, oid)
                    continue
                s, p, vol = self.ob.order_map[oid]
                trade = min(vol, remaining)
                remaining -= trade
                self.last_trade_price = p
                fills.append((p, trade))
                self.ob.modify(oid, vol - trade)
                if vol - trade <= 0:
                    self._remove_order(2, oid)
                    self.agent_oids.discard(oid)

        else:  # sell — consume bids
            self.inject_event(t, "MO_ask")
            while remaining > 0:
                bb, ba = self.ob.get_bbo()
                if bb is None:
                    break
                oids_at_bb = self._bid_price_oids.get(bb)
                if not oids_at_bb:
                    break
                oid = next(iter(oids_at_bb))
                if oid not in self.ob.order_map:
                    oids_at_bb.pop(oid, None)
                    self._remove_order(1, oid)
                    continue
                s, p, vol = self.ob.order_map[oid]
                trade = min(vol, remaining)
                remaining -= trade
                self.last_trade_price = p
                fills.append((p, trade))
                self.ob.modify(oid, vol - trade)
                if vol - trade <= 0:
                    self._remove_order(1, oid)
                    self.agent_oids.discard(oid)

        return fills

    def _compute_cancel_weights(self, side):
        if side == 1:
            n = self._bid_n
            oids = self._bid_oids[:n]
        else:
            n = self._ask_n
            oids = self._ask_oids[:n]

        if n == 0:
            return np.empty(0, dtype=np.int64), np.array([]), np.array([]), np.array([])

        # Evict stale entries using set operations
        oid_list = oids.tolist()
        order_map_keys = self.ob.order_map
        stale = [o for o in oid_list if o not in order_map_keys]
        if stale:
            for so in stale:
                self._remove_order(side, so)
            if side == 1:
                n = self._bid_n
            else:
                n = self._ask_n
            if n == 0:
                return np.empty(0, dtype=np.int64), np.array([]), np.array([]), np.array([])

        if side == 1:
            log_prices = self._bid_log_prices[:n]
            delta0s    = self._bid_delta0s[:n]
            times      = self._bid_times[:n]
            oids       = self._bid_oids[:n]
        else:
            log_prices = self._ask_log_prices[:n]
            delta0s    = self._ask_delta0s[:n]
            times      = self._ask_times[:n]
            oids       = self._ask_oids[:n]

        if side == 1:
            opp_log = self.last_log_best_ask
        else:
            opp_log = self.last_log_best_bid

        if opp_log is not None:
            if side == 1:
                y_vals = np.where(delta0s != 0, (opp_log - log_prices) / delta0s, 1.0)
            else:
                y_vals = np.where(delta0s != 0, (log_prices - opp_log) / delta0s, 1.0)
        else:
            y_vals = np.ones(n)

        y_clipped = np.clip(y_vals, 0.0, 5.0)
        bin_idx = np.digitize(y_clipped, self.cancel_y_bins) - 1
        np.clip(bin_idx, 0, len(self.cancel_prob_by_y) - 1, out=bin_idx)
        pcy_weights = self.cancel_prob_by_y[bin_idx]
        pcy_weights = np.where(np.isfinite(y_vals) & (y_vals > 0), pcy_weights, self.cancel_prob_y_min)

        rank_order = np.argsort(times)
        f_values = np.empty(n)
        f_values[rank_order] = (np.arange(n, dtype=np.float64) + 0.5) / n

        a = self.queue_cancel_alpha - 1.0
        b = self.queue_cancel_beta - 1.0
        queue_weights = f_values ** a * (1.0 - f_values) ** b

        combined = pcy_weights * queue_weights
        return oids.copy(), combined, y_vals, f_values

    def plot_book(self, title: str = "Initial Order Book") -> None:
        ts = self.tick_size
        bids = {}
        asks = {}
        for side, price, vol in self.ob.order_map.values():
            if side == 1:
                bids[price * ts] = bids.get(price * ts, 0) + vol
            else:
                asks[price * ts] = asks.get(price * ts, 0) + vol

        bids = pd.Series(bids).sort_index()
        asks = pd.Series(asks).sort_index()

        bb, ba = self.ob.get_bbo()
        if bb is None or ba is None:
            raise ValueError("Order book has no valid BBO; nothing to plot.")

        window = 50
        bb_pln, ba_pln = bb * ts, ba * ts
        bids_plot = bids[bids.index >= bb_pln - window * ts]
        asks_plot = asks[asks.index <= ba_pln + window * ts]

        w = ts * 0.85
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(bids_plot.index, bids_plot.values, width=w,
               color="steelblue", label="Bids")
        ax.bar(asks_plot.index, asks_plot.values, width=w,
               color="darkorange", label="Asks")
        ax.axvline(bb_pln, color="grey", ls="--", lw=0.8)
        ax.axvline(ba_pln, color="grey", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Price (PLN)")
        ax.set_ylabel("Volume")
        ax.legend()
        plt.tight_layout()
        plt.show()

        print(f"Best bid: {bb_pln:.2f}")
        print(f"Best ask: {ba_pln:.2f}")
        print(f"Spread: {(ba - bb) * ts:.2f} ({ba - bb} ticks)")

    def load_real_orderbook_snapshot(
        self,
        asset: str = "KGHM",
        day_key: str = "d20170110",
        snapshot_time: str = "10:00:00",
        tick_size: float = None,
        orders_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Load a real orderbook snapshot from WSE HDF5 data."""
        if orders_dir is None:
            orders_dir = project_root() / "data" / "WSELOB-2017" / "orders"
        else:
            orders_dir = Path(orders_dir)
        orders_file = orders_dir / f"{asset}_lob_2017_zlib.h5"

        if self.verbose:
            print(f"Loading {asset} orders for {day_key}...")

        df = pd.read_hdf(orders_file, f"/{day_key}")
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True).dt.tz_convert(
            "Europe/Warsaw"
        )
        df["price"] = df["price"] / (10 ** df["price_level"])
        df["action_type"] = df["action_type"].astype(str)

        df = df.sort_values("time").reset_index(drop=True)

        day_start = df["time"].iloc[0].normalize()
        cutoff_time = day_start + pd.to_timedelta(snapshot_time)

        if self.verbose:
            print(f"Replaying orders up to {cutoff_time}...")

        real_ob = HeapOrderBook()

        placement_info = {}
        hawkes_events = []

        for row in df.itertuples():
            if row.time > cutoff_time:
                break

            sim_time = -(cutoff_time - row.time).total_seconds()

            if row.action_type in ("A", "Y"):
                bb, ba = real_ob.get_bbo()
                if row.side == 1:
                    opp_best = ba
                else:
                    opp_best = bb
                placement_info[row.order_id] = (row.time, opp_best)

                if row.side == 1:
                    if ba is not None and row.price >= ba:
                        hawkes_events.append((sim_time, 0))
                    else:
                        hawkes_events.append((sim_time, 2))
                else:
                    if bb is not None and row.price <= bb:
                        hawkes_events.append((sim_time, 1))
                    else:
                        hawkes_events.append((sim_time, 3))

            elif row.action_type == "D":
                if row.side == 1:
                    hawkes_events.append((sim_time, 4))
                else:
                    hawkes_events.append((sim_time, 5))

            real_ob.apply_action(
                row.action_type, row.order_id, row.side, row.price, row.volume
            )

        real_bb, real_ba = real_ob.get_bbo()
        if real_bb is None or real_ba is None:
            raise ValueError("Real orderbook has no valid BBO at snapshot time!")

        if self.verbose:
            print(
                f"Real BBO at snapshot: bid={real_bb:.2f}, ask={real_ba:.2f}, "
                f"spread={real_ba - real_bb:.2f}"
            )
            print(f"Real book has {len(real_ob.order_map)} orders")

        self.ob.clear()
        self._bid_n = 0
        self._bid_oid_idx.clear()
        self._ask_n = 0
        self._ask_oid_idx.clear()
        self._bid_price_oids.clear()
        self._ask_price_oids.clear()

        max_real_oid = 0
        n_bids = 0
        n_asks = 0

        first_event = 0.0

        ba_ticks_snap = int(round(real_ba / tick_size))
        bb_ticks_snap = int(round(real_bb / tick_size))

        y_values = []

        for real_oid, (side, price, volume) in real_ob.order_map.items():
            price_ticks = int(round(price / tick_size))
            if price_ticks > 0:
                log_price = log(price_ticks)
            else:
                log_price = 0.0

            oid = real_oid
            max_real_oid = max(max_real_oid, oid)
            self.ob.add(oid, side, price_ticks, volume)

            p_time, opp_best_at_placement = placement_info.get(
                real_oid, (cutoff_time, None)
            )

            age_seconds = (cutoff_time - p_time).total_seconds()
            order_time = first_event - age_seconds

            if opp_best_at_placement is not None:
                opp_best_ticks_0 = int(round(opp_best_at_placement / tick_size))
                log_opp_0 = log(max(opp_best_ticks_0, 1))
            else:
                log_opp_0 = None

            if side == 1:
                if log_opp_0 is not None:
                    delta0_log = log_opp_0 - log_price
                    delta_now = log(ba_ticks_snap) - log_price
                else:
                    delta0_log = log(ba_ticks_snap) - log_price
                    delta_now = delta0_log

                if delta0_log > 0:
                    y = delta_now / delta0_log
                else:
                    y = 1.0

                self._add_order(1, oid, log_price, delta0_log, order_time)
                n_bids += 1
            else:
                if log_opp_0 is not None:
                    delta0_log = log_price - log_opp_0
                    delta_now = log_price - log(bb_ticks_snap)
                else:
                    delta0_log = log_price - log(bb_ticks_snap)
                    delta_now = delta0_log

                if delta0_log > 0:
                    y = delta_now / delta0_log
                else:
                    y = 1.0

                self._add_order(2, oid, log_price, delta0_log, order_time)
                n_asks += 1

            y_values.append(y)

        self.order_id_counter = max_real_oid + 1

        bb_final, ba_final = self.ob.get_bbo()
        if ba_final is not None and ba_final > 0:
            self.last_log_best_ask = log(ba_final)
        if bb_final is not None and bb_final > 0:
            self.last_log_best_bid = log(bb_final)

        d = len(self.labels)
        self._A_stack = np.zeros((self._n_kernels, d, d))
        if hawkes_events:
            self._t_last = hawkes_events[0][0]
            for ev_time, ev_dim in hawkes_events:
                self._t_last = _record_event_jit(
                    self._A_stack, self._decay_stack, self._t_last, ev_time, ev_dim, self._n_kernels
                )
            dt_to_zero = 0.0 - self._t_last
            if dt_to_zero > 0:
                _decay_A_stack(self._A_stack, self._decay_stack, dt_to_zero, self._n_kernels)
            self._t_last = 0.0
        else:
            self._t_last = 0.0

        self._A_seeded = self._A_stack.copy()

        intensities_0, _ = self._compute_intensities(0.0)

        event_counts = np.bincount([dim for _, dim in hawkes_events], minlength=d)

        if self.verbose:
            print(f"\nLoaded {n_bids} bids, {n_asks} asks into simulation")
            print(f"order_id_counter starts at {self.order_id_counter}")

            bid_depth = sum(
                self.ob.order_map[oid][2]
                for oid in self._bid_oid_idx
                if oid in self.ob.order_map
            )
            ask_depth = sum(
                self.ob.order_map[oid][2]
                for oid in self._ask_oid_idx
                if oid in self.ob.order_map
            )
            print(f"Bid depth: {bid_depth:,}, Ask depth: {ask_depth:,}")

            y_arr = np.array(y_values)
            print("\nPre-existing order y distribution:")
            print(
                f"  mean={y_arr.mean():.2f}, median={np.median(y_arr):.2f}, "
                f"min={y_arr.min():.2f}, max={y_arr.max():.2f}"
            )
            print(f"  y < 0.5 (near execution): {(y_arr < 0.5).sum()}")
            print(
                f"  0.5 ≤ y ≤ 1.5 (near placement): "
                f'{((y_arr >= 0.5) & (y_arr <= 1.5)).sum()}'
            )
            print(f"  y > 1.5 (stranded): {(y_arr > 1.5).sum()}")

            print(f"\nHawkes seeded with {len(hawkes_events):,} real events:")
            for i, label in enumerate(self.labels):
                print(f"  {label:>10}: {event_counts[i]:>7,} events")
            print("\nInitial intensities at t=0 (baseline → seeded):")
            for i, label in enumerate(self.labels):
                print(
                    f"  λ({label:>7}) = {intensities_0[i]:.6f}  "
                    f"(baseline {self._baseline[i]:.6f},  "
                    f"excitation +{intensities_0[i] - self._baseline[i]:.6f})"
                )

    def sample_passive_depth(self):
        u = random.random()
        depth = int(self.xmin_depth * (1 - u) ** (-1 / (self.beta_depth - 1)))
        return max(1, depth)

    def order_regime(self, spread, resil_mult=1.0):
        p_best = self.lo_p_best
        if spread < 2:
            p_inside = 0.0  # no interior tick exists at a 1-tick spread
        else:
            if spread <= self.lo_inside_break:
                raw = spread * self.lo_inside_c1 + self.lo_inside_c0
            else:
                raw = spread * self.lo_inside_c1_hi + self.lo_inside_c0_hi
            p_inside = self.lo_inside_spread_scale * min(0.5, max(0.0, raw))

        if resil_mult != 1.0:
            p_best *= resil_mult
            p_inside *= resil_mult
            tot = p_best + p_inside
            if tot > self.resil_pmax:
                shrink = self.resil_pmax / tot
                p_best *= shrink
                p_inside *= shrink

        r = random.random()

        if r < p_best:
            return "best"
        elif r < p_best + p_inside:
            return "inside"
        else:
            return "passive"

    def _resil_multiplier(self, side, bb, ba):
        """Placement-aggressiveness multiplier ``1 + tanh(z)`` for the
        resiliency (stimulated-refill) and trend-chasing placement biases.

        ``z = resil_kappa * x - resil_phi * sgn * trend`` where ``x`` is the
        side-signed band-pass displacement ``mid - 2*EMA_tau + EMA_2tau``
        (responds fully to a sudden move, decays over ~1.4*tau, identically
        zero along a steady trend) and ``trend = EMA_tau - EMA_flow`` (zero
        at a jump, builds only when a move persists).  ``x > 0`` means
        placing on *side* at/inside the touch would revert the displacement.
        The ``1 +/- tanh(z)`` pair keeps the two-side aggregate aggressive
        placement rate unchanged.  Returns 1.0 when disabled.
        """
        if self._resil_ema is None:
            return 1.0
        if side == "ask":
            side_sign = 1.0
        else:
            side_sign = -1.0
        z = 0.0
        if self.resil_kappa != 0.0:
            dev = (bb + ba) * 0.5 - 2.0 * self._resil_ema + self._resil_ema2
            x = side_sign * dev
            if x > self.resil_xcap:
                x = self.resil_xcap
            elif x < -self.resil_xcap:
                x = -self.resil_xcap
            z += self.resil_kappa * x
        if self.resil_phi != 0.0:
            trend = self._resil_ema - self._resil_ema_flow
            # trend > 0 (rising): boost bid-side aggression, suppress ask
            trend_signal = -side_sign * trend
            y = trend_signal
            if y > self.resil_xcap:
                y = self.resil_xcap
            elif y < -self.resil_xcap:
                y = -self.resil_xcap
            z += self.resil_phi * y
        if z == 0.0:
            return 1.0
        # Rate-preserving form: the two sides get (1 + tanh z) and
        # (1 - tanh z), so the *total* at-best/inside placement rate is
        # unchanged and only its side composition shifts.  A plain exp(z)
        # would inflate the aggregate aggressive-placement rate (Jensen),
        # inflating C(1s) flicker variance and depressing every VR ratio.
        return 1.0 + tanh(z)

    def _compute_regime_masses(self, x_min, x_trans, x_max, alpha1, alpha2):
        a1 = alpha1 + 1
        a2 = alpha2 + 1

        if abs(a1) < 1e-10:
            I_mid = log(x_trans) - log(x_min)
        else:
            I_mid = (x_trans**a1 - x_min**a1) / a1

        if abs(a2) < 1e-10:
            I_tail_raw = log(x_max) - log(x_trans)
        else:
            I_tail_raw = (x_max**a2 - x_trans**a2) / a2

        I_tail = x_trans**(a1 - a2) * I_tail_raw

        total = I_mid + I_tail
        return I_mid / total, I_tail / total

    def _sample_power_law(self, x_min, x_max, alpha):
        u = random.random()

        if abs(alpha + 1) < 1e-10:
            log_min = log(x_min)
            log_max = log(x_max)
            return int(np.exp(log_min + u * (log_max - log_min)))

        a = alpha + 1
        x_min_a = x_min ** a
        x_max_a = x_max ** a

        x = (x_min_a + u * (x_max_a - x_min_a)) ** (1.0 / a)
        return max(1, int(x))

    def sample_order_size(self, order_type: str = "LO") -> int:
        if order_type == "MO":
            mid_min = self.mo_mid_min
            mid_max = self.mo_mid_max
            mid_slope = self.mo_mid_slope
            tail_slope = self.mo_tail_slope
            tail_max = self.mo_tail_max
        else:
            mid_min = self.lo_mid_min
            mid_max = self.lo_mid_max
            mid_slope = self.lo_mid_slope
            tail_slope = self.lo_tail_slope
            tail_max = self.lo_tail_max

        p_mid, p_tail = self._compute_regime_masses(mid_min, mid_max, tail_max, mid_slope, tail_slope)

        r = random.random()

        if r < p_mid:
            return self._sample_power_law(mid_min, mid_max, mid_slope)
        else:
            return self._sample_power_law(mid_max, tail_max, tail_slope)

    # --- Ticks-walked MO size model helpers ---

    def _background_qty(self, qty_dict, side_int):
        """Return a copy of *qty_dict* with agent order volumes subtracted."""
        if not self.agent_oids:
            return qty_dict
        agent_vol = {}
        for oid in self.agent_oids:
            entry = self.ob.order_map.get(oid)
            if entry is not None and entry[0] == side_int:
                p = entry[1]
                agent_vol[p] = agent_vol.get(p, 0) + entry[2]
        if not agent_vol:
            return qty_dict
        result = dict(qty_dict)
        for p, av in agent_vol.items():
            if p in result:
                result[p] = max(0, result[p] - av)
                if result[p] == 0:
                    del result[p]
        return result

    def _opposite_level_volumes(self, qty_dict, best_price, side, n_levels=10):
        if best_price is None or not qty_dict:
            return []
        if side == "ask":
            direction = +1
        else:
            direction = -1
        return [qty_dict.get(best_price + i * direction, 0) for i in range(n_levels)]

    def _sample_truncated_mo(self, lo, hi):
        if lo > hi:
            return lo

        x_trans = self.mo_mid_max

        if hi <= x_trans:
            return self._sample_power_law(lo, hi, self.mo_mid_slope)
        elif lo >= x_trans:
            return self._sample_power_law(lo, hi, self.mo_tail_slope)
        else:
            a1 = self.mo_mid_slope + 1
            a2 = self.mo_tail_slope + 1
            if abs(a1) > 1e-10:
                I_mid = (x_trans ** a1 - lo ** a1) / a1
            else:
                I_mid = log(x_trans / lo)
            if abs(a2) > 1e-10:
                I_tail_raw = (hi ** a2 - x_trans ** a2) / a2
            else:
                I_tail_raw = log(hi / x_trans)
            I_tail = x_trans ** (a1 - a2) * I_tail_raw
            total = I_mid + I_tail
            if total < 1e-30:
                return max(lo, min(hi, int(round(
                    lo + random.random() * (hi - lo)))))
            if random.random() < I_mid / total:
                return self._sample_power_law(lo, x_trans, self.mo_mid_slope)
            else:
                return self._sample_power_law(x_trans, hi, self.mo_tail_slope)

    def sample_MO_size(self, qty_dict, best_price, side):
        """Sample MO size using the ticks-walked model."""
        if not self._tw_loaded or best_price is None:
            return self.sample_order_size("MO"), 0

        depth_levels = self._opposite_level_volumes(qty_dict, best_price, side, n_levels=10)
        if not depth_levels or depth_levels[0] <= 0:
            return self.sample_order_size("MO"), 0

        cum_depth = sum(depth_levels)
        if cum_depth <= 0:
            return self.sample_order_size("MO"), 0

        self._tw_depth_history.append(cum_depth)

        n_obs = len(self._tw_depth_history)
        if n_obs >= self._tw_warmup:
            if (self._tw_adaptive_bounds is None
                    or n_obs % self._tw_recalib_interval == 0):
                self._tw_adaptive_bounds = np.percentile(
                    self._tw_depth_history, [25, 50, 75],
                )
            bounds = self._tw_adaptive_bounds
        else:
            bounds = self._tw_depth_bounds

        qi = int(np.searchsorted(bounds, cum_depth))

        cdf = self._tw_cdfs[qi]
        max_k = len(cdf)
        levels = self._opposite_level_volumes(qty_dict, best_price, side, n_levels=max_k)
        for _ in range(100):
            k = int(np.searchsorted(cdf, random.random()))
            k = min(k, len(levels) - 1)

            if k == 0:
                if levels[0] <= 1:
                    continue
                lo = 1
                hi = levels[0] - 1
                break

            lo = max(1, sum(levels[:k]))
            hi = sum(levels[:k + 1]) - 1
            if hi >= lo:
                break
        else:
            k = 0
            lo = 1
            hi = max(1, levels[0] - 1)

        size = self._sample_truncated_mo(lo, hi)
        k = max(0, int(round(k * self.mo_impact_scale)))
        return size, k

    def _queue_accept_prob(self, queue_ahead):
        if queue_ahead <= self.queue_uniform_max:
            return 1.0
        return (self.queue_uniform_max / queue_ahead) ** self.queue_tail_alpha

    def _sample_passive_with_queue_accept(self, ob, side, bb, ba):
        depth = self.sample_passive_depth()
        for _ in range(self.queue_max_retries):
            depth = self.sample_passive_depth()
            if side == "bid":
                price = bb - depth
                queue = ob.bid_qty.get(price, 0)
            else:
                price = ba + depth
                queue = ob.ask_qty.get(price, 0)
            if random.random() < self._queue_accept_prob(queue):
                return depth
        return depth

    def place_limit_price(self, ob, side):
        bb, ba = ob.get_bbo()

        if bb is None or ba is None:
            return None

        spread = ba - bb
        regime = self.order_regime(spread, self._resil_multiplier(side, bb, ba))

        if side == "bid":
            if regime == "best":
                return 0
            elif regime == "inside":
                if spread >= 2:
                    return -1 * np.random.randint(1, spread)
                else:
                    return 0
            else:
                return self._sample_passive_with_queue_accept(ob, side, bb, ba)
        else:
            if regime == "best":
                return 0
            elif regime == "inside":
                if spread >= 2:
                    return -1 * np.random.randint(1, spread)
                else:
                    return 0
            else:
                return self._sample_passive_with_queue_accept(ob, side, bb, ba)

    def _normalize_cancel_sample_weights(self, weights):
        weights = np.asarray(weights, dtype=float)
        finite = np.isfinite(weights)
        if not finite.all():
            weights = np.where(finite, weights, 0.0)
        weights = np.maximum(weights, 0.0)
        wsum = weights.sum()
        if wsum <= 0.0:
            n = len(weights)
            return np.full(n, 1.0 / n)
        return weights / wsum

    def _execute_mo(self, side_int):
        """Execute a market order using price-to-oid index for O(1) fills."""
        bb, ba = self.ob.get_bbo()
        if side_int == 1:
            if ba is None:
                return 2
            self.mo_stats['bid_attempts'] += 1
            sizing_qty = (self._background_qty(self.ob.ask_qty, 2)
                          if not self.agents_affect_mo_sizing else self.ob.ask_qty)
            target, max_ticks = self.sample_MO_size(sizing_qty, ba, "ask")
            self.mo_sizes.append(target)
            initial_ba = ba
            tick_limit = initial_ba + max_ticks
            self._mo_fill_log = []
            consumed = 0

            # Whole-LO matching: consume entire resting orders in price-time
            # priority until the cumulative consumed volume reaches the
            # sampled target.  The order that crosses the target is taken in
            # full (round-up), so an MO never partially fills any LO and the
            # recorded volume always lands on an order boundary -- which is
            # what keeps the phantom fill predicate exact for any quote size.
            while consumed < target and self._ask_n > 0:
                bb, ba = self.ob.get_bbo()
                if ba is None or ba > tick_limit:
                    break

                oids_at_ba = self._ask_price_oids.get(ba)
                if not oids_at_ba:
                    break
                oid = next(iter(oids_at_ba))
                if oid not in self.ob.order_map:
                    oids_at_ba.pop(oid, None)
                    self._remove_order(2, oid)
                    continue

                _, price, vol = self.ob.order_map[oid]
                # Round-to-nearest order boundary, including zero: take this
                # order only if doing so lands cumulative volume at least as
                # close to the sampled target as stopping now would.  No
                # consumed>0 guard, so an MO whose target is <= half the front
                # order rounds to *zero* impact -- it removes nothing and leaves
                # the best quote untouched (the whole-LO analogue of the old
                # small partial nibble).  The event still excited the Hawkes
                # kernels in _sample_next_event, so this only changes the fill,
                # not the arrival process.  Keeps whole-LO matching exact and
                # makes realized volume unbiased w.r.t. target.
                if (consumed + vol / 2.0) >= target:
                    break
                consumed += vol
                self.last_trade_price = price
                self._mo_fill_log.append((price, vol))
                if oid in self.agent_oids:
                    self._agent_fills.append((oid, price, vol, 2))
                # Remove from compact arrays / price index first (fast path
                # while oid is still in order_map), then drop it from the book.
                start = self._remove_order(2, oid)
                self.ob.delete(oid)
                if start is not None:
                    duration = self.current_stamp - start
                else:
                    duration = self.current_stamp - self.current_stamp
                self.lifetimes.append((duration, 'executed'))
                self.agent_oids.discard(oid)

            _, final_ba = self.ob.get_bbo()
            if final_ba is None:
                raise RuntimeError(
                    f"MO_bid (target {target}) depleted the ask book at t={self.current_time}")
            if initial_ba is not None:
                tw = int(final_ba - initial_ba)
            else:
                tw = 0
            self.mo_ticks_walked.append(max(0, tw))
            self._event_detail = {
                'type': 'MO', 'side_text': 'buy', 'side_int': 1,
                'mo_volume': consumed, 'ticks_walked': max(0, tw),
                'fills': list(self._mo_fill_log),
                '_pre_bb': bb, '_pre_ba': initial_ba,
            }
            if consumed >= target:
                self.mo_stats['bid_filled'] += 1
            return

        if side_int == 2:
            if bb is None:
                return 2
            self.mo_stats['ask_attempts'] += 1
            sizing_qty = (self._background_qty(self.ob.bid_qty, 1)
                          if not self.agents_affect_mo_sizing else self.ob.bid_qty)
            target, max_ticks = self.sample_MO_size(sizing_qty, bb, "bid")
            self.mo_sizes.append(target)
            initial_bb = bb
            tick_limit = initial_bb - max_ticks
            self._mo_fill_log = []
            consumed = 0

            # Whole-LO matching (see the buy-side branch above for rationale).
            while consumed < target and self._bid_n > 0:
                bb, ba = self.ob.get_bbo()
                if bb is None or bb < tick_limit:
                    break

                oids_at_bb = self._bid_price_oids.get(bb)
                if not oids_at_bb:
                    break
                oid = next(iter(oids_at_bb))
                if oid not in self.ob.order_map:
                    oids_at_bb.pop(oid, None)
                    self._remove_order(1, oid)
                    continue

                _, price, vol = self.ob.order_map[oid]
                # Round-to-nearest order boundary, including zero (see buy-side
                # branch above for the full rationale).
                if (consumed + vol / 2.0) >= target:
                    break
                consumed += vol
                self.last_trade_price = price
                self._mo_fill_log.append((price, vol))
                if oid in self.agent_oids:
                    self._agent_fills.append((oid, price, vol, 1))
                start = self._remove_order(1, oid)
                self.ob.delete(oid)
                if start is not None:
                    duration = self.current_stamp - start
                else:
                    duration = self.current_stamp - self.current_stamp
                self.lifetimes.append((duration, 'executed'))
                self.agent_oids.discard(oid)

            final_bb, _ = self.ob.get_bbo()
            if final_bb is None:
                raise RuntimeError(
                    f"MO_ask (target {target}) depleted the bid book at t={self.current_time}")
            if initial_bb is not None:
                tw = int(initial_bb - final_bb)
            else:
                tw = 0
            self.mo_ticks_walked.append(max(0, tw))
            self._event_detail = {
                'type': 'MO', 'side_text': 'sell', 'side_int': 2,
                'mo_volume': consumed, 'ticks_walked': max(0, tw),
                'fills': list(self._mo_fill_log),
                '_pre_bb': initial_bb, '_pre_ba': ba,
            }
            if consumed >= target:
                self.mo_stats['ask_filled'] += 1
            return

        raise ValueError(f"_execute_mo: side_int must be 1 or 2, got {side_int!r}")

    def _execute_lo_bid(self, spread):
        relative_price = self.place_limit_price(self.ob, "bid")
        size = self.sample_order_size("LO")

        if relative_price is None:
            return 2

        bb, ba = self.ob.get_bbo()
        price = int(bb - relative_price)

        if ba is not None and price >= ba:
            price = int(ba - 1)

        price = max(1, int(price))

        if ba is not None and ba > 0 and price > 0:
            log_price = log(price)
            delta0_log = log(ba) - log_price
        else:
            if price > 0:
                log_price = log(price)
            else:
                log_price = 0.0
            delta0_log = 1.0
        oid = self.next_id()
        self.ob.add(oid, 1, price, size)
        self._add_order(1, oid, log_price, delta0_log, self.current_stamp)

        self.lo_stats.append({
            "ticks_from_best": relative_price,
            "spread_ticks": spread,
            "size": size
        })
        self._event_detail = {
            'type': 'LO', 'oid': oid, 'side': 1, 'price': price,
            'volume': size, 'delta0': delta0_log,
            'ticks_from_best': relative_price,
            'queue_ahead': max(0, self.ob.bid_qty.get(price, 0) - size),
        }

    def _execute_lo_ask(self, spread):
        relative_price = self.place_limit_price(self.ob, "ask")
        size = self.sample_order_size("LO")
        if relative_price is None:
            return 2

        bb, ba = self.ob.get_bbo()
        price = int(ba + relative_price)
        if bb is not None and price <= bb:
            price = int(bb + 1)

        price = max(1, int(price))

        if bb is not None and bb > 0 and price > 0:
            log_price = log(price)
            delta0_log = log_price - log(bb)
        else:
            if price > 0:
                log_price = log(price)
            else:
                log_price = 0.0
            delta0_log = 1.0
        oid = self.next_id()
        self.ob.add(oid, 2, price, size)
        self._add_order(2, oid, log_price, delta0_log, self.current_stamp)

        self.lo_stats.append({
            "ticks_from_best": relative_price,
            "spread_ticks": spread,
            "size": size
        })
        self._event_detail = {
            'type': 'LO', 'oid': oid, 'side': 2, 'price': price,
            'volume': size, 'delta0': delta0_log,
            'ticks_from_best': relative_price,
            'queue_ahead': max(0, self.ob.ask_qty.get(price, 0) - size),
        }

    def _execute_cxl(self, side, bb, ba):
        if side == 1:
            self.cancel_stats['bid_attempts'] += 1
        elif side == 2:
            self.cancel_stats['ask_attempts'] += 1
        else:
            raise ValueError(f"_execute_cxl: side must be 1 or 2, got {side!r}")

        if side == 1:
            n_side = self._bid_n
        else:
            n_side = self._ask_n
        if n_side == 0:
            return

        valid_orders, weights, y_arr, f_arr = \
            self._compute_cancel_weights(side=side)
        if len(valid_orders) == 0:
            return

        if self.agent_oids:
            mask = np.array([int(oid) not in self.agent_oids for oid in valid_orders])
            if not mask.any():
                return
            valid_orders = valid_orders[mask]
            weights = weights[mask]
            y_arr = y_arr[mask]
            f_arr = f_arr[mask]

        weights = self._normalize_cancel_sample_weights(weights)

        chosen_idx = np.random.choice(len(valid_orders), p=weights)
        chosen_oid = int(valid_orders[chosen_idx])

        self.cancel_y_log.append(y_arr[chosen_idx])
        self.cancel_f_log.append(f_arr[chosen_idx])

        if chosen_oid in self.ob.order_map:
            _, p_ticks, _ = self.ob.order_map[chosen_oid]
            bb_now, ba_now = self.ob.get_bbo()
            if side == 1:
                if bb_now is not None:
                    self.cancel_dsame_log.append(max(0, bb_now - p_ticks))
            else:
                if ba_now is not None:
                    self.cancel_dsame_log.append(max(0, p_ticks - ba_now))

        if chosen_oid in self.ob.order_map:
            _, cxl_price, cxl_vol = self.ob.order_map[chosen_oid]
        else:
            cxl_price, cxl_vol = 0, 0
        cxl_log_p, cxl_d0 = self._get_order_data(side, chosen_oid)
        if side == 1:
            cxl_dt = ((self.last_log_best_ask - cxl_log_p)
                      if self.last_log_best_ask is not None else cxl_d0)
            cxl_tfb = max(0, int((bb or 0) - cxl_price))
            cxl_qa = self.ob.bid_qty.get(cxl_price, 0)
        else:
            cxl_dt = ((cxl_log_p - self.last_log_best_bid)
                      if self.last_log_best_bid is not None else cxl_d0)
            cxl_tfb = max(0, int(cxl_price - (ba or 0)))
            cxl_qa = self.ob.ask_qty.get(cxl_price, 0)

        self.ob.delete(chosen_oid)

        start = self._remove_order(side, chosen_oid)
        if start is not None:
            duration = self.current_stamp - start
        else:
            duration = self.current_stamp - self.current_stamp
        self.lifetimes.append((duration, 'canceled'))
        if side == 1:
            self.cancel_stats['bid_success'] += 1
        else:
            self.cancel_stats['ask_success'] += 1
        self._event_detail = {
            'type': 'CXL', 'oid': chosen_oid, 'side': side,
            'price': cxl_price, 'volume': cxl_vol,
            'delta0': cxl_d0, 'delta_t': cxl_dt,
            'y_ratio': y_arr[chosen_idx],
            'ticks_from_best': cxl_tfb, 'queue_ahead': cxl_qa,
        }

    def process_event(self, event_type, bb, ba):
        self._event_detail = None

        spread = None
        if bb is not None and ba is not None:
            spread = ba - bb

        if ba is not None and ba > 0:
            self.last_log_best_ask = log(ba)
        if bb is not None and bb > 0:
            self.last_log_best_bid = log(bb)

        if event_type == "MO_bid":
            r = self._execute_mo(1)
            if r == 2:
                return 2
        elif event_type == "MO_ask":
            r = self._execute_mo(2)
            if r == 2:
                return 2
        elif event_type == "LO_bid":
            r = self._execute_lo_bid(spread)
            if r == 2:
                return 2
        elif event_type == "LO_ask":
            r = self._execute_lo_ask(spread)
            if r == 2:
                return 2
        elif event_type == "CXL_bid":
            self._execute_cxl(1, bb, ba)
        elif event_type == "CXL_ask":
            self._execute_cxl(2, bb, ba)

    def run(self, overwrite=False, verbose=None):
        """Execute the simulation for ``T`` events.

        ``verbose`` overrides ``self.verbose`` for this call when not ``None``;
        when falsy, the liquidity-guard summary is suppressed (used by the
        calibration search).
        """
        if verbose is None:
            run_verbose = self.verbose
        else:
            run_verbose = bool(verbose)
        self.lifetimes.clear()
        self.lo_stats.clear()
        self.mo_sizes.clear()
        self.mo_ticks_walked.clear()
        self._tw_depth_history.clear()
        self._tw_adaptive_bounds = None
        self.cancel_stats = {'bid_attempts': 0, 'bid_success': 0,
                             'ask_attempts': 0, 'ask_success': 0}
        self.mo_stats = {'bid_attempts': 0, 'bid_filled': 0,
                         'ask_attempts': 0, 'ask_filled': 0}
        self.cancel_y_log.clear()
        self.cancel_f_log.clear()
        self.cancel_dsame_log.clear()

        for k in self._guard_stats:
            self._guard_stats[k] = 0

        self._fills_compact.clear()
        self._mo_compact.clear()
        self._mid_t = []
        self._mid_v = []
        _capture_mid = self.capture_mid
        self._resil_ema = None
        self._resil_ema2 = None
        self._resil_ema_flow = None
        self._resil_t = 0.0
        _resil_on = (self.resil_kappa != 0.0) or (self.resil_phi != 0.0)

        if not self.lightweight:
            self._mo_buf.clear()
            self._bbo_buf.clear()
            if self.recording_mode == 'full':
                self._orders_buf.clear()
                self._fills_buf.clear()
                self._intensities_buf.clear()
            self._prev_event_time = None
            self._prev_mid = None
            self._open_db(overwrite=overwrite)

        self.last_trade_price = None

        d = len(self.labels)
        if hasattr(self, '_A_seeded') and self._A_seeded is not None:
            self._A_stack = self._A_seeded.copy()
        else:
            self._A_stack = np.zeros((self._n_kernels, d, d))
        self._t_last = 0.0

        for counter in range(1, self.T + 1):

            t, label, intensities = self._sample_next_event()
            if self.liquidity_guard:
                label = self._guard_event(label)

            if self.recording_mode == 'full' and self._db_conn is not None:
                self._intensities_buf.append((t, *intensities))

            self.current_time = t
            self.current_index = counter
            self.current_stamp = t + counter * 1e-9

            bb_pre, ba_pre = self.ob.get_bbo()

            if _capture_mid and bb_pre is not None and ba_pre is not None:
                self._mid_t.append(t)
                self._mid_v.append((bb_pre + ba_pre) * 0.5 * self.tick_size)

            if _resil_on and bb_pre is not None and ba_pre is not None:
                # EMAs of the piecewise-constant mid: the pre-event mid held
                # over (last update, t], so relax all anchors toward it.
                mid_pre_ticks = (bb_pre + ba_pre) * 0.5
                if self._resil_ema is None:
                    self._resil_ema = mid_pre_ticks
                    self._resil_ema2 = mid_pre_ticks
                    self._resil_ema_flow = mid_pre_ticks
                else:
                    dtr = t - self._resil_t
                    if dtr > 0.0:
                        self._resil_ema = mid_pre_ticks + (
                            (self._resil_ema - mid_pre_ticks)
                            * exp(-dtr / self.resil_tau_s))
                        self._resil_ema2 = mid_pre_ticks + (
                            (self._resil_ema2 - mid_pre_ticks)
                            * exp(-dtr / (2.0 * self.resil_tau_s)))
                        self._resil_ema_flow = mid_pre_ticks + (
                            (self._resil_ema_flow - mid_pre_ticks)
                            * exp(-dtr / self.resil_flow_tau_s))
                self._resil_t = t

            if self.recording_mode == 'full':
                _pre = (self._snapshot_pre_event(bb_pre, ba_pre)
                        if self._db_conn is not None else None)
            elif self.recording_mode == 'medium':
                _pre = (self._snapshot_pre_event(bb_pre, ba_pre)
                        if self._db_conn is not None and label.startswith('MO')
                        else None)
            elif self.recording_mode == 'bbo':
                _pre = None

            self._agent_fills.clear()
            if self.process_event(label, bb_pre, ba_pre) == 2:
                continue
            self.current_time = t

            if self.agents and (not self.lightweight or self.agents_when_lightweight):
                fills = list(self._agent_fills)
                if self.shuffle_agents and len(self.agents) > 1:
                    agent_iter = list(self.agents)
                    random.shuffle(agent_iter)
                else:
                    agent_iter = self.agents
                for agent in agent_iter:
                    agent.on_event(self, t, fills)

            if self.lightweight:
                detail = self._event_detail
                if detail is not None and detail.get('type') == 'MO':
                    bb_now, ba_now = self.ob.get_bbo()
                    ts = self.tick_size
                    self._mo_compact.append((
                        t,
                        detail['side_text'],
                        detail.get('_pre_bb', bb_now or 0) * ts,
                        detail.get('_pre_ba', ba_now or 0) * ts,
                        detail['ticks_walked'],
                    ))
                    for fill_price, fill_vol in detail.get('fills', []):
                        self._fills_compact.append((t, fill_price * ts))
            elif self.recording_mode == 'medium':
                if self._db_conn is not None and bb_pre is not None and ba_pre is not None:
                    if _pre is not None:
                        self._record_to_db(t, label, _pre, counter)
                    else:
                        ts = self.tick_size
                        mid_pre = (bb_pre + ba_pre) / 2.0
                        self._bbo_buf.append((t, bb_pre * ts, ba_pre * ts, mid_pre * ts))
                        if counter % self._flush_every == 0:
                            self._flush_db()
            elif self.recording_mode == 'bbo':
                if self._db_conn is not None and bb_pre is not None and ba_pre is not None:
                    ts = self.tick_size
                    mid_pre = (bb_pre + ba_pre) / 2.0
                    self._bbo_buf.append((t, bb_pre * ts, ba_pre * ts, mid_pre * ts))
                    if counter % self._flush_every == 0:
                        self._flush_db()
            else:
                if _pre is not None:
                    self._record_to_db(t, label, _pre, counter)

                bb, ba = self.ob.get_bbo()
                if bb is None or ba is None or ba <= bb:
                    continue

        if not self.lightweight:
            self._close_db()

        total_interventions = sum(self._guard_stats.values())
        if run_verbose and self.liquidity_guard and total_interventions > 0:
            pct = 100.0 * total_interventions / self.T
            print(f"\nLiquidity guard: {total_interventions} interventions "
                  f"({pct:.2f}% of {self.T} events)")
            for k, v in self._guard_stats.items():
                if v > 0:
                    print(f"  {k}: {v}")

    def get_compact_results(self):
        """Return compact in-memory results for lightweight runs."""
        fills = self._fills_compact
        mos = self._mo_compact
        return {
            'fills_ts':         np.array([f[0] for f in fills], dtype=np.float64) if fills else np.array([], dtype=np.float64),
            'fills_price':      np.array([f[1] for f in fills], dtype=np.float64) if fills else np.array([], dtype=np.float64),
            'mo_ts':            np.array([m[0] for m in mos], dtype=np.float64) if mos else np.array([], dtype=np.float64),
            'mo_side':          np.array([m[1] for m in mos]) if mos else np.array([], dtype='U4'),
            'mo_best_bid':      np.array([m[2] for m in mos], dtype=np.float64) if mos else np.array([], dtype=np.float64),
            'mo_best_ask':      np.array([m[3] for m in mos], dtype=np.float64) if mos else np.array([], dtype=np.float64),
            'mo_ticks_walked':  np.array([m[4] for m in mos], dtype=np.int64) if mos else np.array([], dtype=np.int64),
        }

    def get_mid_series(self):
        """Return the per-event mid series captured in memory during ``run()``.

        Requires ``capture_mid=True`` at construction. Returns ``(t, mid)`` as
        float64 arrays where ``t`` is event time (s) and ``mid`` is the
        pre-event mid price in PLN -- the same series the ``bbo`` SQLite table
        records, but with no disk round-trip (used by the calibration search).
        """
        return (
            np.asarray(self._mid_t, dtype=np.float64),
            np.asarray(self._mid_v, dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Backwards-compatible alias. The Numba engine used to live in a separate
# ``simulate_fast.py`` as ``SimulateFast``; it is now the one and only engine.
# Existing imports (``from .simulate import SimulateFast``) keep working.
# ---------------------------------------------------------------------------
SimulateFast = Simulate

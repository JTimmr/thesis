"""
Event-driven limit order book simulation.

Control flags:
- ``liquidity_guard``: when False, liquidity guard remapping is disabled.
- ``agents_when_lightweight``: when True, agents receive ``on_event`` in lightweight
  mode; when False, agents run only in full recording mode (default).
"""

import numpy as np
import pandas as pd
import random
import sqlite3
import os
from matplotlib import pyplot as plt
from .orderbook import HeapOrderBook
from pathlib import Path
from typing import Optional, Union

from .helpers import project_root
from ..data.schema import (
    SIM_CREATE_ORDERS, SIM_CREATE_FILLS, SIM_CREATE_MO_ORDERS,
    SIM_CREATE_BBO, SIM_CREATE_INTENSITIES,
    SIM_INSERT_ORDER, SIM_INSERT_FILL, SIM_INSERT_MO,
    SIM_INSERT_BBO, SIM_INSERT_INTENSITY,
)

class Simulate:
    def __init__(self, arrival_mode, T,
                 agents=None, kernel_mode="single",
                 db_path=None, flush_every=50_000,
                 lightweight=False,
                 guard_soft_ratio=0.25, guard_hard_ratio=0.10,
                 liquidity_guard=True, agents_when_lightweight=False,
                 tick_size=0.05, alpha_scale=0.9):
        """Initialize the simulator.

        Parameters
        ----------
        arrival_mode : str
            One of 'poisson', 'hawkes_univariate', or 'hawkes_multivariate'.
        T : int
            Number of events to generate (roughly the trading horizon).
        lightweight : bool, optional
            If True, skip ALL heavy recording (SQLite, frames, diagnostics)
            and only collect compact fills (timestamp, price) and compact
            MO data (timestamp, side, best_bid, best_ask, ticks_walked).
            Designed for parallel batch runs where only candlestick and
            propagator analysis are needed afterwards.
        agents : list, optional
            List of agent objects that interact with the simulated market.
            Each agent must implement an ``on_event(sim, t)`` method that
            is called after every background event.  Agents inspect the
            order book via ``sim.ob`` and place/cancel orders through the
            ``agent_place_order``, ``agent_cancel_order``, and
            ``agent_market_order`` helper methods.  If an agent's
            ``on_event`` does nothing, the simulation simply continues.

            Agent protocol::

                class MyAgent:
                    def on_event(self, sim, t, fills):
                        # sim.ob  = current order book
                        # t       = simulation time
                        # fills   = [(oid, price, qty, side), ...]
                        #           fills on agent orders this cycle
                        pass
        """
        self.arrival_mode = arrival_mode
        self.kernel_mode = kernel_mode
        self.lightweight = lightweight
        self.tick_size = tick_size
        self.alpha_scale = alpha_scale
        if kernel_mode not in ("single", "triple"):
            raise ValueError(f"Unknown kernel_mode: {kernel_mode}. Use 'single' or 'triple'.")
        self.T = T

        # ── Liquidity guard ─────────────────────────────────────────────
        self._guard_soft_ratio = guard_soft_ratio
        self._guard_hard_ratio = guard_hard_ratio
        self._guard_stats = {
            'soft_mo_remapped': 0, 'soft_cxl_remapped': 0,
            'hard_mo_blocked': 0, 'hard_cxl_blocked': 0,
        }
        self.liquidity_guard = liquidity_guard
        self.agents_when_lightweight = agents_when_lightweight

        # ── Compact data buffers (lightweight mode) ────────────────────
        # Always initialised; populated only when self.lightweight is True.
        self._fills_compact = []   # [(timestamp, price), ...]
        self._mo_compact = []      # [(timestamp, side_str, best_bid, best_ask, ticks_walked), ...]

        # Empirical
        # self.P_C_GIVEN_Y = np.array([
        #             0.012, 0.025, 0.035, 0.052, 0.180,
        #             0.088, 0.035, 0.028, 0.012, 0.035,
        #             0.005, 0.008, 0.008, 0.003, 0.025,
        #             0.002, 0.004, 0.002, 0.001, 0.017,
        #             0.002, 0.002, 0.002, 0.005
        #         ])

        # Bin edges for y ∈ [0, 5]
        self.P_C_Y_BINS = np.linspace(0, 5, 25)

        # Found by Mike & Farmer
        self.P_C_GIVEN_Y = 0.012*(1-np.exp(-1*self.P_C_Y_BINS))
        

        self.P_C_Y_CENTERS = 0.5 * (self.P_C_Y_BINS[:-1] + self.P_C_Y_BINS[1:])

        # Fallback for y outside [0, 5]: use boundary values
        self.P_C_Y_MIN = self.P_C_GIVEN_Y[0]   # for y < 0
        self.P_C_Y_MAX = self.P_C_GIVEN_Y[-1]  # for y > 5

        self.beta_depth = 2.145 # 2.1
        self.xmin_depth = 1

        # Queue-position cancellation weights (fitted Beta distribution)
        # From empirical WSE data: orders at the back of the queue
        # (recently placed, high fractional position) are canceled far
        # more often than orders at the front.
        self.queue_cancel_alpha = 8.1029
        self.queue_cancel_beta  = 0.6585

        # Queue-size acceptance for passive LO placement (accept-reject)
        # Empirical queue-ahead distribution is ~uniform up to a threshold,
        # then decays as a steep power law.  Orders into large queues are
        # accepted with lower probability to match the empirical pattern.
        self.QUEUE_UNIFORM_MAX = 3500      # threshold (shares) for uniform regime
        self.QUEUE_TAIL_ALPHA  = 3.99      # power-law exponent for tail decay
        self.QUEUE_MAX_RETRIES = 20        # max resamples before accepting anyway

        # Limit Order parameters (from order_prices_calibration_KGHM.ipynb)
        self.LO_MID_MIN = 2           # minimum size in mid regime
        self.LO_MID_MAX = 4000        # transition point to tail regime
        self.LO_MID_SLOPE = -0.41     # slope in log-log (near-uniform)
        self.LO_TAIL_SLOPE = -2.31    # tail regime slope (steeper decay)
        self.LO_TAIL_MAX = 200_000    # empirical max (from calibration data)

        # Market Order parameters (from MO calibration)
        self.MO_MID_MIN = 1           # minimum size in mid regime
        self.MO_MID_MAX = 200         # transition point to tail regime  
        self.MO_MID_SLOPE = -0.3      # slope in log-log (adjust from MO calibration)
        self.MO_TAIL_SLOPE = -2.68    # tail regime slope (adjust from MO calibration)
        self.MO_TAIL_MAX = 155_000    # empirical max: 154,779 (from calibration data)

        # Ticks-walked CDFs per depth quartile (from order_prices_calibration)
        _tw_path = Path(__file__).resolve().parent.parent / "data" / "mo_depth_data" / "KGHM_tw_quartiles.npz"
        if _tw_path.exists():
            _td = np.load(_tw_path)
            self._tw_depth_bounds = _td["depth_quartile_bounds"]  # shape (3,)
            self._tw_cdfs = [_td[f"tw_cdf_q{i}"] for i in range(4)]
            self._tw_loaded = True
        else:
            print(f"WARNING: {_tw_path.name} not found — falling back to static MO sizes")
            self._tw_loaded = False

        self._tw_depth_history = []
        self._tw_adaptive_bounds = None
        self._tw_warmup = 2000
        self._tw_recalib_interval = 500

        # Poisson baseline rates (from KGHM calibration)
        self.poisson_rates = {
            "MO_bid": 0.071652,
            "MO_ask": 0.066922,
            "LO_bid": 0.656950,
            "LO_ask": 0.652339,
            "CXL_bid": 0.656051,
            "CXL_ask": 0.651098
        }


        # Univariate Hawkes parameters (calibrated to KGHM data)
        self.univariate_baseline = np.array([
            0.019840,
            0.018044,
            0.164048,
            0.165068,
            0.205264,
            0.204799
        ])

        self.univariate_adjacency = np.diag([
            0.724101,
            0.730989,
            0.750187,
            0.746101,
            0.687174,
            0.685461
        ])

        self.univariate_decays = np.diag([
            19.977277,
            19.981433,
            10.111358,
            10.046270,
            19.986916,
            19.982077
        ])

        # Multivariate Hawkes parameters (calibrated to KGHM data)

        self.multivariate_adjacency = np.array([
        [0.428726,0.040907,0.000000,0.017025,0.000000,0.000000],
        [0.057898,0.493649,0.000000,0.013028,0.000000,0.000000],
        [0.165819,0.000000,0.000000,0.000000,1.101833,0.000000],
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
        0.01263952,
        0.01149326,
        0.00000000,
        0.00000000,
        0.09788585,
        0.00000000
        ])

        # ── Triple-exponential Hawkes parameters ─────────────────────────
        # Optimised decay rates for sum-of-three-exponentials kernel
        # (Optuna 500 trials, multivariate tau-time, KGHM, ρ=0.95)
        self.BETA_FAST = 99.9990
        self.BETA_MID  = 3.3090
        self.BETA_SLOW = 0.0012

        # Multivariate triple-exp — manually-tuned values, FULLY symmetrized
        # Every α[i][j] averaged with α[mirror(i)][mirror(j)] where
        # mirror swaps bid↔ask: 0↔1, 2↔3, 4↔5.
        # Base values are the manually-tuned params (commented out below).
        self.multivariate_triple_baseline = np.array([
            0.005766, 0.005766, 0.000000, 0.000000, 0.000000, 0.000000
        ])

        self.multivariate_triple_adjacency_fast = np.array([
            [0.068545, 0.012797, 0.000000, 0.002306, 0.013029, 0.000000],
            [0.012797, 0.068545, 0.002306, 0.000000, 0.000000, 0.013029],
            [0.012417, 0.000000, 0.000000, 0.000000, 1.231247, 0.045555],
            [0.000000, 0.012417, 0.000000, 0.000000, 0.045555, 1.231247],
            [0.052225, 1.627118, 0.043475, 0.054523, 0.004591, 0.318264],
            [1.627118, 0.052225, 0.054523, 0.043475, 0.318264, 0.004591],
        ])
        self.multivariate_triple_adjacency_mid = np.array([
            [0.344115, 0.000207, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.000207, 0.344115, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.375713, 0.000000, 0.154554, 0.000000, 0.137881, 0.005443],
            [0.000000, 0.375713, 0.000000, 0.154554, 0.005443, 0.137881],
        ])
        self.multivariate_triple_adjacency_slow = np.array([
            [0.142300, 0.110700, 0.000000, 0.000039, 0.000000, 0.000001],
            [0.110700, 0.142300, 0.000039, 0.000000, 0.000001, 0.000000],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.021933, 0.000000, 0.025680, 0.000000, 0.032611, 0.000000],
            [0.000000, 0.021933, 0.000000, 0.025680, 0.000000, 0.032611],
        ])

        # ── Previous manually-tuned triple-exp params (kept for reference) ──
        # self.multivariate_triple_baseline = np.array([
        #     0.005480, 0.006052, 0.000000, 0.000000, 0.000000, 0.000000
        # ])
        # self.multivariate_triple_adjacency_fast = np.array([
        #     [0.065926, 0.016842, 0.000000, 0.001814, 0.013351, 0.000000],
        #     [0.008752, 0.071164, 0.002797, 0.000000, 0.000000, 0.012707],
        #     [0.024833, 0.000000, 0.000000, 0.000000, 1.267371, 0.000000],
        #     [0.000000, 0.000000, 0.000000, 0.000000, 0.091109, 1.195123],
        #     [0.104449, 1.616449, 0.086950, 0.109046, 0.009182, 0.030889],
        #     [1.637787, 0.000000, 0.000000, 0.000000, 0.605639, 0.000000],
        # ])
        # self.multivariate_triple_adjacency_mid = np.array([
        #     [0.367898, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        #     [0.000413, 0.320332, 0.000000, 0.000000, 0.000000, 0.000000],
        #     [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        #     [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        #     [0.751425, 0.000000, 0.211381, 0.000000, 0.219718, 0.010885],
        #     [0.000000, 0.000000, 0.000000, 0.097726, 0.000000, 0.056044],
        # ])
        # self.multivariate_triple_adjacency_slow = np.array([
        #     [0.1445, 0.1186, 0.000000, 0.000077, 0.000000, 0.000001],
        #     [0.1028, 0.1401, 0.000000, 0.000000, 0.000000, 0.000000],
        #     [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        #     [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        #     [0.000000, 0.000000, 0.000000, 0.000000, 0.065221, 0.000000],
        #     [0.000000, 0.043866, 0.000000, 0.051359, 0.000000, 0.000000],
        # ])

        self.labels = ["MO_bid","MO_ask","LO_bid","LO_ask","CXL_bid","CXL_ask"]

        # cancellation statistics
        self.cancel_stats = {
            'bid_attempts': 0,
            'bid_success': 0,
            'ask_attempts': 0,
            'ask_success': 0,
        }
        # track MO outcomes for diagnostics
        self.mo_stats = {
            'bid_attempts': 0,
            'bid_filled': 0,
            'ask_attempts': 0,
            'ask_filled': 0,
        }
        self.lo_stats = []
        self.mo_sizes = []           # store market order sizes for plotting
        self.mo_ticks_walked = []    # store ticks walked per MO for diagnostics

        self.ob = HeapOrderBook()

        self.order_id_counter = 0
        self.lifetimes = []          # list of (duration, outcome) tuples

        # ── Array-based order storage (per side) ──────────────────────
        _CAP = 50_000
        self._bid_log_prices = np.empty(_CAP)
        self._bid_delta0s    = np.empty(_CAP)
        self._bid_times      = np.empty(_CAP)
        self._bid_oids       = np.empty(_CAP, dtype=np.int64)
        self._bid_n          = 0
        self._bid_cap        = _CAP
        self._bid_oid_idx    = {}   # oid -> array index

        self._ask_log_prices = np.empty(_CAP)
        self._ask_delta0s    = np.empty(_CAP)
        self._ask_times      = np.empty(_CAP)
        self._ask_oids       = np.empty(_CAP, dtype=np.int64)
        self._ask_n          = 0
        self._ask_cap        = _CAP
        self._ask_oid_idx    = {}   # oid -> array index

        # Distribution diagnostics (populated during run)
        self.cancel_y_log = []       # y values at each cancellation
        self.cancel_f_log = []       # fractional queue position f at each cancel
        self.cancel_dsame_log = []   # same-side distance (ticks) at each cancel

        # Track last known best prices (in log space) to detect changes
        self.last_log_best_ask = None
        self.last_log_best_bid = None

        # ── Agent infrastructure ──
        self.agents = agents if agents is not None else []
        self.agent_oids = set()  # order IDs belonging to agents (protected from background CXL)
        self._agent_fills = []   # filled agent orders this event cycle: [(oid, price, qty, side)]
        self.last_trade_price = None  # last executed trade price (for candlestick charts)

        # Safe defaults used by process_event before run() sets event-time state.
        # before run() sets event-time state.
        self.current_time = 0.0
        self.current_index = 0
        self.current_stamp = 0.0

        self.frames = []
        self.executed_events = []  # record events that actually modify the book

        # ── Event database infrastructure ──────────────────────────────
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
        self._event_detail = None   # set by process_event for DB recording
        self._mo_fill_log = []      # individual MO fills for DB recording

        # -----------------------------------------------------------------
        # Unified intensity parameters for on-the-fly event generation.
        # All three modes (poisson, hawkes_univariate, hawkes_multivariate)
        # use the same Ogata thinning sampler.
        #
        # Internal state uses *lists* of adjacency / decay / auxiliary
        # matrices so that single-exp (K=1) and double-exp (K=2) kernels
        # share the same code path.
        # -----------------------------------------------------------------
        d = len(self.labels)
        if self.arrival_mode == "poisson":
            rates = [self.poisson_rates[l] for l in self.labels]
            self._baseline = np.array(rates)
            self._n_kernels = 1
            self._adjacency_list = [np.zeros((d, d))]
            self._decays_list = [np.ones((d, d))]

        elif self.arrival_mode == "hawkes_univariate":
            if self.kernel_mode == "single":
                self._baseline = self.univariate_baseline.copy()
                self._n_kernels = 1
                self._adjacency_list = [self.univariate_adjacency.copy()]
                self._decays_list = [self.univariate_decays.copy()]
            else:  # triple
                raise NotImplementedError(
                    "Univariate triple-exponential kernel is not yet calibrated. "
                    "Calibrate the parameters and set univariate_triple_* attributes "
                    "before using this mode."
                )

        elif self.arrival_mode == "hawkes_multivariate":
            if self.kernel_mode == "single":
                self._baseline = self.multivariate_baseline.copy()
                self._n_kernels = 1
                self._adjacency_list = [self.multivariate_adjacency.copy()]
                self._decays_list = [self.multivariate_decays.copy()]
            else:  # triple
                self._baseline = self.multivariate_triple_baseline.copy()
                self._n_kernels = 3
                self._adjacency_list = [
                    self.multivariate_triple_adjacency_fast.copy() * self.alpha_scale,
                    self.multivariate_triple_adjacency_mid.copy() * self.alpha_scale,
                    self.multivariate_triple_adjacency_slow.copy() * self.alpha_scale,
                ]
                self._decays_list = [
                    np.full((d, d), self.BETA_FAST),
                    np.full((d, d), self.BETA_MID),
                    np.full((d, d), self.BETA_SLOW),
                ]
        else:
            raise ValueError(f"Unknown arrival_mode: {self.arrival_mode}")

        # Auxiliary matrices for intensity computation (one per kernel)
        # Updated recursively: no event history stored
        self._A_list = [np.zeros((d, d)) for _ in range(self._n_kernels)]
        self._t_last = 0.0  # time of last Hawkes state update


    def _compute_intensities(self, t):
        """Compute intensity vector λ(t) via recursive exponential decay.

        For K kernel components:
            λ_i(t) = μ_i + Σ_k Σ_j α^(k)_ij · β^(k)_ij · A^(k)_ij(t)
        where  A^(k)_ij(t) = A^(k)_ij(t_last) · exp(-β^(k)_ij · Δt)

        Returns
        -------
        intensities : ndarray (d,)
        None        : (kept for API compatibility)
        """
        dt = max(t - self._t_last, 0.0)
        intensities = self._baseline.copy()
        for k in range(self._n_kernels):
            A_k_decayed = self._A_list[k] * np.exp(-self._decays_list[k] * dt)
            intensities += np.sum(
                self._adjacency_list[k] * self._decays_list[k] * A_k_decayed,
                axis=1,
            )
        np.maximum(intensities, 0.0, out=intensities)  # float safety
        return intensities, None


    # ── Liquidity guard helpers ─────────────────────────────────────

    def _liquidity_state(self):
        """Compute current liquidity metrics for the guard.

        Returns ``(thin_side, depth_ratio)`` or ``None`` when the book
        is one-sided.
        """
        bid_depth = self.ob.total_bid_depth
        ask_depth = self.ob.total_ask_depth

        if bid_depth <= 0 or ask_depth <= 0:
            return None

        max_depth = max(bid_depth, ask_depth)
        depth_ratio = min(bid_depth, ask_depth) / max_depth
        thin_side = 'bid' if bid_depth < ask_depth else 'ask'

        return (thin_side, depth_ratio)

    def _guard_event(self, label):
        """Apply liquidity guard to a sampled background event label.

        Returns the (possibly remapped) label.  Dangerous events that would
        further deplete the thin side of the book are remapped to a
        replenishing limit order on that side.

        Two regimes:
          * **Hard** (``depth_ratio < hard_ratio``): always remap.
          * **Soft** (``depth_ratio < soft_ratio``): remap with probability
            that increases linearly as the ratio drops toward the hard floor.
        """
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

        # Hard regime: deterministic remap
        if depth_ratio < self._guard_hard_ratio:
            if label == dangerous_mo:
                self._guard_stats['hard_mo_blocked'] += 1
            else:
                self._guard_stats['hard_cxl_blocked'] += 1
            return safe_lo

        # Soft regime: probabilistic remap (linear ramp from 0 at soft_ratio
        # to 1 at hard_ratio)
        if depth_ratio < self._guard_soft_ratio:
            span = self._guard_soft_ratio - self._guard_hard_ratio
            p_remap = (self._guard_soft_ratio - depth_ratio) / span if span > 0 else 1.0
            if random.random() < p_remap:
                if label == dangerous_mo:
                    self._guard_stats['soft_mo_remapped'] += 1
                else:
                    self._guard_stats['soft_cxl_remapped'] += 1
                return safe_lo

        return label

    def _open_db(self, overwrite=False):
        """Create / open the SQLite database and set up tables."""
        if self.db_path is None:
            return
        if os.path.exists(self.db_path) and not overwrite:
            raise FileExistsError(
                f"Database '{self.db_path}' already exists!\n"
                f"  To use a different name:  sim.db_path = 'new_name.sqlite'\n"
                f"  To overwrite:             sim.run(overwrite=True)"
            )
        if os.path.exists(self.db_path) and overwrite:
            os.remove(self.db_path)
            print(f"Overwriting existing database: {Path(self.db_path).name}")
        self._db_conn = sqlite3.connect(self.db_path)
        self._db_cursor = self._db_conn.cursor()
        self._db_cursor.execute("PRAGMA journal_mode=WAL")
        self._db_cursor.execute("PRAGMA synchronous=NORMAL")
        self._db_cursor.execute(SIM_CREATE_ORDERS)
        self._db_cursor.execute(SIM_CREATE_FILLS)
        self._db_cursor.execute(SIM_CREATE_MO_ORDERS)
        self._db_cursor.execute(SIM_CREATE_BBO)
        self._db_cursor.execute(SIM_CREATE_INTENSITIES)
        self._db_conn.commit()

    def _flush_db(self):
        """Write accumulated buffers to SQLite and clear them."""
        if self._db_conn is None:
            return
        if self._orders_buf:
            self._db_cursor.executemany(SIM_INSERT_ORDER, self._orders_buf)
            self._orders_buf.clear()
        if self._fills_buf:
            self._db_cursor.executemany(SIM_INSERT_FILL, self._fills_buf)
            self._fills_buf.clear()
        if self._mo_buf:
            self._db_cursor.executemany(SIM_INSERT_MO, self._mo_buf)
            self._mo_buf.clear()
        if self._bbo_buf:
            self._db_cursor.executemany(SIM_INSERT_BBO, self._bbo_buf)
            self._bbo_buf.clear()
        if self._intensities_buf:
            self._db_cursor.executemany(SIM_INSERT_INTENSITY, self._intensities_buf)
            self._intensities_buf.clear()
        self._db_conn.commit()

    def _close_db(self):
        """Final flush, create indices, and close the database."""
        self._flush_db()
        if self._db_conn:
            self._db_cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(timestamp)")
            self._db_cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(timestamp)")
            self._db_cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_mo_ts ON mo_orders(timestamp)")
            self._db_cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_bbo_ts ON bbo(timestamp)")
            self._db_cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_int_ts ON intensities(timestamp)")
            self._db_conn.commit()
            self._db_conn.close()
            self._db_conn = None
            self._db_cursor = None

    def _snapshot_pre_event(self, bb, ba):
        """Snapshot book state *before* an event modifies it."""
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
        }

    def _record_to_db(self, t, label, pre, counter):
        """Build DB rows from pre-event state + event detail, append to buffers."""
        if pre is None:
            return

        ts = self.tick_size
        bb = pre['bb']; ba = pre['ba']; mid = pre['mid']
        spread = pre['spread']
        dt_prev = ((t - self._prev_event_time)
                    if self._prev_event_time is not None else 0.0)
        dp_mid = ((mid - self._prev_mid)
                   if self._prev_mid is not None else 0.0)

        # Always record BBO (in PLN)
        self._bbo_buf.append((t, bb * ts, ba * ts, mid * ts))

        detail = self._event_detail
        if detail is None:
            self._prev_event_time = t
            self._prev_mid = mid
            return

        etype = detail.get('type')

        if etype in ('LO', 'CXL'):
            side = detail['side']
            price = detail['price']
            best_same = bb if side == 1 else ba
            ticks_from_mid = int(round(price - mid))
            is_cancel = 1 if etype == 'CXL' else 0

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
                *pre['bid_depths'], *pre['ask_depths'],
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

            fill_prices = [p * ts for p, _ in fills] if fills else [0]
            mo_row = (
                t, side_text, mo_vol, len(fills),
                min(fill_prices), max(fill_prices),
                bb * ts, ba * ts, tw, ratio_L0, pre['microprice'] * ts,
                *opp_10, *pre['bid_depths'], *pre['ask_depths'],
            )
            self._mo_buf.append(mo_row)

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
        dt = max(t - self._t_last, 0.0)
        for k in range(self._n_kernels):
            self._A_list[k] *= np.exp(-self._decays_list[k] * dt)
            self._A_list[k][:, dim_idx] += 1.0
        self._t_last = t

    def _sample_next_event(self):
        """Sample the next event using Ogata's thinning algorithm.

        Returns
        -------
        (t, label, intensities) : tuple
            Event time, type label, and intensity vector at event time.
        """
        t = self._t_last

        for _ in range(100_000):
            # Intensity upper bound (intensity only decays between events)
            intensities, _ = self._compute_intensities(t)
            lambda_star = intensities.sum()

            if lambda_star < 1e-15:
                t += 0.1
                continue

            # Candidate inter-arrival time
            dt = np.random.exponential(1.0 / lambda_star)
            t_cand = t + dt

            # Actual intensity at candidate time (decayed)
            intensities_cand, _ = self._compute_intensities(t_cand)
            lambda_cand = intensities_cand.sum()

            # Accept / reject
            if random.random() * lambda_star <= lambda_cand:
                probs = intensities_cand / lambda_cand
                dim_idx = np.random.choice(len(self.labels), p=probs)
                self._record_event(t_cand, dim_idx)
                return t_cand, self.labels[dim_idx], intensities_cand

            # Rejected — advance time (tighter upper bound next iteration)
            t = t_cand

        raise RuntimeError(
            "Ogata thinning failed to produce an event after 100 000 iterations"
        )

    def inject_event(self, t, label):
        """Inject an external event into the Hawkes excitation state.

        Call this from a trading bot so that its action excites future
        event intensities.  The caller is responsible for any order-book
        modifications — this method only updates the intensity state.

        Parameters
        ----------
        t : float
            Event time (must be ≥ current simulation time).
        label : str
            Event type (one of self.labels).
        """
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

    def _remove_order(self, side, oid):
        """Remove an order via swap-and-pop. O(1). Returns placement time."""
        if side == 1:
            idx_map = self._bid_oid_idx
            if oid not in idx_map:
                return None
            i = idx_map.pop(oid)
            t_placed = float(self._bid_times[i])
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
        """Return (log_price, delta0_log) for an order, or (0.0, 1.0) if missing."""
        idx_map = self._bid_oid_idx if side == 1 else self._ask_oid_idx
        if oid not in idx_map:
            return (0.0, 1.0)
        i = idx_map[oid]
        if side == 1:
            return (float(self._bid_log_prices[i]), float(self._bid_delta0s[i]))
        else:
            return (float(self._ask_log_prices[i]), float(self._ask_delta0s[i]))

    # ═══════════════════════════════════════════════════════════════
    # Agent interaction helpers
    # ═══════════════════════════════════════════════════════════════

    def agent_place_order(self, side, price, volume, t):
        """Place a limit order on behalf of an agent.

        Registers the order in the book and agent_oids (so background
        CXLs skip it).
        Injects the corresponding LO event into the Hawkes process.

        Parameters
        ----------
        side : int   — 1 = bid, 2 = ask.
        price : int  — Price in ticks.
        volume : int — Order size (shares).
        t : float    — Current simulation time.

        Returns
        -------
        int — The new order ID.
        """
        oid = self.next_id()
        self.ob.add(oid, side, price, volume)
        self.agent_oids.add(oid)

        bb, ba = self.ob.get_bbo()
        log_price = np.log(price) if price > 0 else 0.0

        if side == 1:
            delta0_log = (np.log(ba) - log_price) if (ba and ba > 0 and price > 0) else 1.0
            self._add_order(1, oid, log_price, delta0_log, t)
            self.inject_event(t, "LO_bid")
        else:
            delta0_log = (log_price - np.log(bb)) if (bb and bb > 0 and price > 0) else 1.0
            self._add_order(2, oid, log_price, delta0_log, t)
            self.inject_event(t, "LO_ask")

        return oid

    def agent_cancel_order(self, oid, t):
        """Cancel an agent's resting order.

        Removes from book, agent_oids, and injects CXL.

        Returns True if the order was found and removed.
        """
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
        """Execute a market order on behalf of an agent.

        Walks the opposite side of the book.  Injects MO into Hawkes.

        Parameters
        ----------
        side : int   — 1 = buy (lifts asks), 2 = sell (hits bids).
        volume : int — Order size (shares).
        t : float    — Current simulation time.

        Returns
        -------
        list of (price, qty) — Fills achieved.
        """
        fills = []
        remaining = volume

        if side == 1:  # buy — consume asks
            self.inject_event(t, "MO_bid")
            while remaining > 0:
                bb, ba = self.ob.get_bbo()
                if ba is None:
                    break
                filled_any = False
                for oid in list(self._ask_oid_idx.keys()):
                    if oid not in self.ob.order_map:
                        self._remove_order(2, oid)
                        continue
                    s, p, vol = self.ob.order_map[oid]
                    if p == ba:
                        trade = min(vol, remaining)
                        remaining -= trade
                        self.last_trade_price = p
                        fills.append((p, trade))
                        self.ob.modify(oid, vol - trade)
                        if vol - trade <= 0:
                            self._remove_order(2, oid)
                            self.agent_oids.discard(oid)
                        filled_any = True
                        break
                if not filled_any:
                    break

        else:  # sell — consume bids
            self.inject_event(t, "MO_ask")
            while remaining > 0:
                bb, ba = self.ob.get_bbo()
                if bb is None:
                    break
                filled_any = False
                for oid in list(self._bid_oid_idx.keys()):
                    if oid not in self.ob.order_map:
                        self._remove_order(1, oid)
                        continue
                    s, p, vol = self.ob.order_map[oid]
                    if p == bb:
                        trade = min(vol, remaining)
                        remaining -= trade
                        self.last_trade_price = p
                        fills.append((p, trade))
                        self.ob.modify(oid, vol - trade)
                        if vol - trade <= 0:
                            self._remove_order(1, oid)
                            self.agent_oids.discard(oid)
                        filled_any = True
                        break
                if not filled_any:
                    break

        return fills


    
    def get_cancel_weight(self, y: float) -> float:
        """
        Look up P(C|y) from calibrated distribution.
        
        Parameters
        ----------
        y : float
            Relative depth ratio δ(t)/δ₀
            
        Returns
        -------
        float
            Cancellation probability weight P(C|y)
        """
        if not np.isfinite(y):
            return self.P_C_Y_MIN
        
        if y <= 0:
            return self.P_C_Y_MIN
        elif y >= 5:
            return self.P_C_Y_MAX
        else:
            # Find bin index
            bin_idx = np.digitize([y], self.P_C_Y_BINS)[0] - 1
            bin_idx = max(0, min(bin_idx, len(self.P_C_GIVEN_Y) - 1))
            return self.P_C_GIVEN_Y[bin_idx]

    def _queue_position_weight(self, f):
        """Unnormalized Beta(α, β) kernel for queue-position weighting.

        Parameters
        ----------
        f : float
            Fractional queue position in (0, 1).  0 = front, 1 = back.
        """
        a = self.queue_cancel_alpha - 1.0   # 7.1029
        b = self.queue_cancel_beta  - 1.0   # -0.3415
        return f ** a * (1.0 - f) ** b

    def _compute_cancel_weights(self, side):
        """Compute P(C|y) x Q(f) weights for all orders on one side.

        Pure numpy — no Python loops. Order data lives in pre-allocated arrays;
        queue ranking uses numpy argsort (C-level O(n log n)).
        """
        if side == 1:
            n = self._bid_n
            oids = self._bid_oids[:n]
        else:
            n = self._ask_n
            oids = self._ask_oids[:n]

        if n == 0:
            return np.empty(0, dtype=np.int64), np.array([]), np.array([]), np.array([])

        # Evict stale entries (oid no longer in the order book).
        # swap-and-pop invalidates the slice, so re-read afterwards.
        stale = [int(o) for o in oids if int(o) not in self.ob.order_map]
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

        opp_log = self.last_log_best_ask if side == 1 else self.last_log_best_bid

        # Vectorized y computation
        if opp_log is not None:
            if side == 1:
                y_vals = np.where(delta0s != 0, (opp_log - log_prices) / delta0s, 1.0)
            else:
                y_vals = np.where(delta0s != 0, (log_prices - opp_log) / delta0s, 1.0)
        else:
            y_vals = np.ones(n)

        # Vectorized P(C|y) lookup
        y_clipped = np.clip(y_vals, 0.0, 5.0)
        bin_idx = np.digitize(y_clipped, self.P_C_Y_BINS) - 1
        np.clip(bin_idx, 0, len(self.P_C_GIVEN_Y) - 1, out=bin_idx)
        pcy_weights = self.P_C_GIVEN_Y[bin_idx]
        pcy_weights = np.where(np.isfinite(y_vals) & (y_vals > 0), pcy_weights, self.P_C_Y_MIN)

        # Queue position via numpy argsort (C-level, ~50x faster than Python iteration)
        rank_order = np.argsort(times)
        f_values = np.empty(n)
        f_values[rank_order] = (np.arange(n, dtype=np.float64) + 0.5) / n

        # Vectorized queue-position weight
        a = self.queue_cancel_alpha - 1.0
        b = self.queue_cancel_beta - 1.0
        queue_weights = f_values ** a * (1.0 - f_values) ** b

        combined = pcy_weights * queue_weights
        return oids.copy(), combined, y_vals, f_values

    def plot_book(self, title: str = "Initial Order Book") -> None:
        """Bar chart of aggregate volume per price level in a window around the BBO."""
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
               color="#1f77b4", label="Bids")
        ax.bar(asks_plot.index, asks_plot.values, width=w,
               color="#ff7f0e", label="Asks")
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
        """
        Load a real orderbook snapshot from WSE HDF5 data and initialize the simulation.

        Requires **PyTables** (``pip install tables``) for ``pandas.read_hdf``.

        Parameters
        ----------
        asset : str
            Asset name (KGHM, PKNORLEN, PKOBP, etc.)
        day_key : str
            Day key in HDF5 file (e.g. ``'d20170110'``)
        snapshot_time : str
            Time to snapshot the book (e.g. ``'10:00:00'``)
        tick_size : float, optional
            Tick size for price conversion. If omitted, uses ``self.tick_size``.
        orders_dir : path-like, optional
            Directory containing ``{asset}_lob_2017_zlib.h5``. If omitted, uses
            ``data/WSELOB-2017/orders`` inside the package root.
        """
        if tick_size is None:
            tick_size = self.tick_size
        if orders_dir is None:
            orders_dir = project_root() / "data" / "WSELOB-2017" / "orders"
        else:
            orders_dir = Path(orders_dir)
        orders_file = orders_dir / f"{asset}_lob_2017_zlib.h5"

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

        max_real_oid = 0
        n_bids = 0
        n_asks = 0

        first_event = 0.0

        ba_ticks_snap = int(round(real_ba / tick_size))
        bb_ticks_snap = int(round(real_bb / tick_size))

        y_values = []

        for real_oid, (side, price, volume) in real_ob.order_map.items():
            price_ticks = int(round(price / tick_size))
            log_price = np.log(price_ticks)

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
                log_opp_0 = np.log(max(opp_best_ticks_0, 1))
            else:
                log_opp_0 = None

            if side == 1:
                if log_opp_0 is not None:
                    delta0_log = log_opp_0 - log_price
                    delta_now = np.log(ba_ticks_snap) - log_price
                else:
                    delta0_log = np.log(ba_ticks_snap) - log_price
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
                    delta_now = log_price - np.log(bb_ticks_snap)
                else:
                    delta0_log = log_price - np.log(bb_ticks_snap)
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
            self.last_log_best_ask = np.log(ba_final)
        if bb_final is not None and bb_final > 0:
            self.last_log_best_bid = np.log(bb_final)

        d = len(self.labels)
        self._A_list = [np.zeros((d, d)) for _ in range(self._n_kernels)]
        if hawkes_events:
            self._t_last = hawkes_events[0][0]
            for ev_time, ev_dim in hawkes_events:
                self._record_event(ev_time, ev_dim)
            dt_to_zero = 0.0 - self._t_last
            if dt_to_zero > 0:
                for k in range(self._n_kernels):
                    self._A_list[k] *= np.exp(-self._decays_list[k] * dt_to_zero)
            self._t_last = 0.0
        else:
            self._t_last = 0.0

        self._A_seeded = [A.copy() for A in self._A_list]

        intensities_0, _ = self._compute_intensities(0.0)

        event_counts = np.bincount([dim for _, dim in hawkes_events], minlength=d)

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
        """Sample relative price distance (ticks behind best) from power-law."""
        u = random.random()
        depth = int(self.xmin_depth * (1 - u) ** (-1 / (self.beta_depth - 1)))
        return max(1, depth)
    
    def order_regime(self, spread):

        # probabilities approximated from empirical calibration results
        p_best = 0.1
        p_inside = min(0.5, max(0, spread * 0.03317993 + 0.01994101))

        r = random.random()

        if r < p_best:
            return "best"
        elif r < p_best + p_inside:
            return "inside"
        else:
            return "passive"

    def _compute_regime_masses(self, x_min: float, x_trans: float, x_max: float, alpha1: float, alpha2: float):
        """
        Compute the probability mass in each regime for a continuous 
        two-regime TRUNCATED power-law distribution.
        
        PDF: p(x) ∝ x^α₁ for x ∈ [x_min, x_trans]
            p(x) ∝ x^α₂ for x ∈ [x_trans, x_max]  (with continuity at x_trans)
        
        The PDF is continuous at x_trans, so:
        c₁ * x_trans^α₁ = c₂ * x_trans^α₂
        
        Returns
        -------
        p_mid : float
            Probability mass in mid regime [x_min, x_trans]
        p_tail : float
            Probability mass in tail regime [x_trans, x_max]
        """
        a1 = alpha1 + 1
        a2 = alpha2 + 1
        
        # Integral of x^α₁ over [x_min, x_trans]
        if abs(a1) < 1e-10:
            I_mid = np.log(x_trans) - np.log(x_min)
        else:
            I_mid = (x_trans**a1 - x_min**a1) / a1
        
        # Integral of (x_trans^(α₁-α₂)) * x^α₂ over [x_trans, x_max]
        # The continuity factor is x_trans^(α₁-α₂)
        # Integral of x^α₂ over [x_trans, x_max] = (x_max^(α₂+1) - x_trans^(α₂+1)) / (α₂+1)
        if abs(a2) < 1e-10:
            I_tail_raw = np.log(x_max) - np.log(x_trans)
        else:
            I_tail_raw = (x_max**a2 - x_trans**a2) / a2
        
        # Apply continuity factor: c₁/c₂ = x_trans^(α₂-α₁)
        # The integral in terms of the mid-regime normalisation:
        I_tail = x_trans**(a1 - a2) * I_tail_raw
        
        total = I_mid + I_tail
        return I_mid / total, I_tail / total

    def _sample_power_law(self, x_min: float, x_max: float, alpha: float) -> int:
        """
        Sample from a truncated power-law distribution.
        
        PDF: p(x) ∝ x^α  for x in [x_min, x_max]
        
        Parameters
        ----------
        x_min : float
            Minimum value (lower bound)
        x_max : float
            Maximum value (upper bound)
        alpha : float
            Power-law exponent (slope in log-log space)
            
        Returns
        -------
        int
            Sampled value (rounded to integer)
        """
        u = random.random()
        
        if abs(alpha + 1) < 1e-10:
            # Special case: α ≈ -1 → log-uniform
            log_min = np.log(x_min)
            log_max = np.log(x_max)
            return int(np.exp(log_min + u * (log_max - log_min)))
        
        # General power-law inverse CDF
        # For truncated power-law: x = [x_min^(α+1) + u*(x_max^(α+1) - x_min^(α+1))]^(1/(α+1))
        a = alpha + 1
        x_min_a = x_min ** a
        x_max_a = x_max ** a
        
        x = (x_min_a + u * (x_max_a - x_min_a)) ** (1.0 / a)
        return max(1, int(x))

    def sample_order_size(self, order_type: str = "LO") -> int:
        """
        Sample order size from a continuous two-regime TRUNCATED power-law distribution.
        
        The probability of being in each regime is DERIVED from the slopes
        and transition point (not an arbitrary parameter).
        
        Both regimes are truncated (capped at empirical max from calibration data).
        
        Parameters
        ----------
        order_type : str
            'LO' for limit orders, 'MO' for market orders
            
        Returns
        -------
        int
            Sampled order size (number of shares)
        """
        
        # Select parameters based on order type
        if order_type == "MO":
            mid_min = self.MO_MID_MIN
            mid_max = self.MO_MID_MAX
            mid_slope = self.MO_MID_SLOPE
            tail_slope = self.MO_TAIL_SLOPE
            tail_max = self.MO_TAIL_MAX
        else:  # LO (default)
            mid_min = self.LO_MID_MIN
            mid_max = self.LO_MID_MAX
            mid_slope = self.LO_MID_SLOPE
            tail_slope = self.LO_TAIL_SLOPE
            tail_max = self.LO_TAIL_MAX
        
        # Compute probability mass in each regime (derived from distribution shape)
        p_mid, p_tail = self._compute_regime_masses(mid_min, mid_max, tail_max, mid_slope, tail_slope)
        
        r = random.random()
        
        # -------------------------
        # Mid regime (truncated power-law with slope α_mid)
        # -------------------------
        if r < p_mid:
            volume =  self._sample_power_law(mid_min, mid_max, mid_slope)
        
        # -------------------------
        # Tail regime (truncated power-law with slope α_tail)
        # -------------------------
        else:
            volume =  self._sample_power_law(mid_max, tail_max, tail_slope)
        
        return volume

    # ── helpers for the ticks-walked MO size model ──────────────

    def _opposite_level_volumes(self, qty_dict, best_price, side, n_levels=10):
        """Return a list of volumes at the first *n_levels* on the given side.

        Parameters
        ----------
        qty_dict : dict
            Price → volume mapping (e.g. ``self.ob.ask_qty``).
        best_price : int or float
            Current best price on this side.
        side : str
            ``"ask"`` (prices increase) or ``"bid"`` (prices decrease).
        n_levels : int
            Number of levels to collect.

        Returns
        -------
        list[int]
            Volumes at each level (0-indexed from the best).
        """
        if best_price is None or not qty_dict:
            return []
        direction = +1 if side == "ask" else -1
        return [qty_dict.get(best_price + i * direction, 0) for i in range(n_levels)]

    def _sample_truncated_mo(self, lo, hi):
        """Sample from the unconditional MO size distribution truncated to [lo, hi].

        Uses direct power-law inverse CDF on the interval, which avoids the
        catastrophic precision loss that occurs when routing through the
        full-distribution CDF/PPF in the extreme tail.
        """
        if lo > hi:
            return lo  # degenerate range

        x_trans = self.MO_MID_MAX  # regime boundary (200)

        if hi <= x_trans:
            # Entirely in mid regime
            return self._sample_power_law(lo, hi, self.MO_MID_SLOPE)
        elif lo >= x_trans:
            # Entirely in tail regime
            return self._sample_power_law(lo, hi, self.MO_TAIL_SLOPE)
        else:
            # Spans both regimes — compute mass in each and pick regime
            a1 = self.MO_MID_SLOPE + 1
            a2 = self.MO_TAIL_SLOPE + 1
            I_mid = (x_trans ** a1 - lo ** a1) / a1 if abs(a1) > 1e-10 \
                else np.log(x_trans / lo)
            I_tail_raw = (hi ** a2 - x_trans ** a2) / a2 if abs(a2) > 1e-10 \
                else np.log(hi / x_trans)
            I_tail = x_trans ** (a1 - a2) * I_tail_raw
            total = I_mid + I_tail
            if total < 1e-30:
                return max(lo, min(hi, int(round(
                    lo + random.random() * (hi - lo)))))
            if random.random() < I_mid / total:
                return self._sample_power_law(lo, x_trans, self.MO_MID_SLOPE)
            else:
                return self._sample_power_law(x_trans, hi, self.MO_TAIL_SLOPE)

    def sample_MO_size(self, qty_dict, best_price, side):
        """Sample MO size using the ticks-walked model.

        Returns ``(size, max_ticks)`` so the execution loop can enforce
        a hard cap on the number of price levels walked.

        Falls back to ``(sample_order_size("MO"), 0)`` when calibration
        data is missing or the book is empty (tick cap = 0, best-level only).
        """
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

        qi = int(np.searchsorted(bounds, cum_depth))  # 0..3

        cdf = self._tw_cdfs[qi]
        max_k = len(cdf)
        levels = self._opposite_level_volumes(qty_dict, best_price, side, n_levels=max_k)
        for _ in range(100):
            k = int(np.searchsorted(cdf, random.random()))
            k = min(k, len(levels) - 1)

            if k == 0:
                if levels[0] <= 1:
                    # Can't partially fill a 1-share level; treat as degenerate
                    continue
                lo = 1
                hi = levels[0] - 1
                break

            # k≥1: MO must clear L0…L(k-1), partial fill at Lk
            lo = max(1, sum(levels[:k]))
            hi = sum(levels[:k + 1]) - 1
            if hi >= lo:          # levels[k] > 0 → valid bounds
                break
            # levels[k] is empty → reject, resample k
        else:
            # all attempts gave degenerate bounds → fall back to k=0
            k = 0
            lo = 1
            hi = max(1, levels[0] - 1)

        # ── sample from truncated unconditional MO distribution ──
        size = self._sample_truncated_mo(lo, hi)
        return size, k

    def _queue_accept_prob(self, queue_ahead):
        """Acceptance probability for placing into a queue of given size (shares).

        Uniform (p=1) for queues ≤ QUEUE_UNIFORM_MAX, then power-law decay.
        """
        if queue_ahead <= self.QUEUE_UNIFORM_MAX:
            return 1.0
        return (self.QUEUE_UNIFORM_MAX / queue_ahead) ** self.QUEUE_TAIL_ALPHA

    def _sample_passive_with_queue_accept(self, ob, side, bb, ba):
        """Sample passive depth with accept-reject conditioning on queue size.

        Repeatedly draws a candidate tick-depth from the power-law price
        distribution, checks the queue already present at that level, and
        accepts the placement with probability given by the empirical
        queue-ahead distribution.  Falls back after QUEUE_MAX_RETRIES.
        """
        depth = self.sample_passive_depth()          # fallback value
        for _ in range(self.QUEUE_MAX_RETRIES):
            depth = self.sample_passive_depth()
            if side == "bid":
                price = bb - depth
                queue = ob.bid_qty.get(price, 0)
            else:
                price = ba + depth
                queue = ob.ask_qty.get(price, 0)
            if random.random() < self._queue_accept_prob(queue):
                return depth
        return depth                                 # accept last sample

    def place_limit_price(self, ob, side):
        """
        Two-stage order placement:
        1. Select regime (best / inside / passive)
        2. For passive: accept-reject on queue size at candidate level
        """
        bb, ba = ob.get_bbo()

        if bb is None or ba is None:
            return None

        spread = ba - bb
        regime = self.order_regime(spread)

        # -------------------
        # BID SIDE
        # -------------------
        if side == "bid":

            if regime == "best":
                return 0

            elif regime == "inside":
                if spread >= 2:
                    # Improve by 1 tick (penny-jumping), consistent with
                    # empirical HFT behaviour on most equity LOBs.
                    return -1 * np.random.randint(1, spread)
                else:
                    return 0  # Can't go inside, place at best

            else:  # passive
                return self._sample_passive_with_queue_accept(ob, side, bb, ba)

        # -------------------
        # ASK SIDE
        # -------------------
        else:

            if regime == "best":
                return 0

            elif regime == "inside":
                if spread >= 2:
                    return -1 * np.random.randint(1, spread)
                else:
                    return 0  # Can't go inside, place at best

            else:  # passive
                return self._sample_passive_with_queue_accept(ob, side, bb, ba)


    def _normalize_cancel_sample_weights(self, weights):
        """Return a valid probability vector for np.random.choice."""
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
        """Execute a market order.

        Parameters
        ----------
        side_int : int
            ``1`` = buy MO (walk the ask book); ``2`` = sell MO (walk the bid book).
        """
        bb, ba = self.ob.get_bbo()
        if side_int == 1:
            if ba is None:
                return 2
            self.mo_stats['bid_attempts'] += 1
            size, max_ticks = self.sample_MO_size(self.ob.ask_qty, ba, "ask")
            self.mo_sizes.append(size)
            init_size = size
            initial_ba = ba
            tick_limit = initial_ba + max_ticks
            self._mo_fill_log = []

            while size > 0 and self._ask_n > 0:
                bb, ba = self.ob.get_bbo()
                if ba is None:
                    break
                if ba > tick_limit:
                    break

                filled = False
                for oid in list(self._ask_oid_idx.keys()):
                    if oid not in self.ob.order_map:
                        self._remove_order(2, oid)
                        continue

                    side, price, vol = self.ob.order_map[oid]
                    if price == ba:
                        trade = min(vol, size)
                        size -= trade
                        self.last_trade_price = price
                        self.ob.modify(oid, vol - trade)
                        self._mo_fill_log.append((price, trade))
                        if oid in self.agent_oids:
                            self._agent_fills.append((oid, price, trade, 2))
                        if vol - trade <= 0:
                            start = self._remove_order(2, oid)
                            duration = self.current_stamp - (start if start is not None else self.current_stamp)
                            self.lifetimes.append((duration, 'executed'))
                            self.agent_oids.discard(oid)
                        filled = True
                        break

                if not filled:
                    break

            _, final_ba = self.ob.get_bbo()
            if final_ba is None:
                raise RuntimeError(
                    f"MO_bid of size {init_size} depleted the ask book at t={self.current_time}")
            tw = int(final_ba - initial_ba) if initial_ba is not None else 0
            self.mo_ticks_walked.append(max(0, tw))
            self._event_detail = {
                'type': 'MO', 'side_text': 'buy', 'side_int': 1,
                'mo_volume': init_size, 'ticks_walked': max(0, tw),
                'fills': list(self._mo_fill_log),
                '_pre_bb': bb, '_pre_ba': initial_ba,
            }
            if size <= 0:
                self.mo_stats['bid_filled'] += 1
            return

        if side_int == 2:
            if bb is None:
                return 2
            self.mo_stats['ask_attempts'] += 1
            size, max_ticks = self.sample_MO_size(self.ob.bid_qty, bb, "bid")
            self.mo_sizes.append(size)
            init_size = size
            initial_bb = bb
            tick_limit = initial_bb - max_ticks
            self._mo_fill_log = []

            while size > 0 and self._bid_n > 0:
                bb, ba = self.ob.get_bbo()
                if bb is None:
                    break
                if bb < tick_limit:
                    break

                filled = False
                for oid in list(self._bid_oid_idx.keys()):
                    if oid not in self.ob.order_map:
                        self._remove_order(1, oid)
                        continue

                    side, price, vol = self.ob.order_map[oid]
                    if price == bb:
                        trade = min(vol, size)
                        size -= trade
                        self.last_trade_price = price
                        self.ob.modify(oid, vol - trade)
                        self._mo_fill_log.append((price, trade))
                        if oid in self.agent_oids:
                            self._agent_fills.append((oid, price, trade, 1))
                        if vol - trade <= 0:
                            start = self._remove_order(1, oid)
                            duration = self.current_stamp - (start if start is not None else self.current_stamp)
                            self.lifetimes.append((duration, 'executed'))
                            self.agent_oids.discard(oid)
                        filled = True
                        break

                if not filled:
                    break

            final_bb, _ = self.ob.get_bbo()
            if final_bb is None:
                raise RuntimeError(
                    f"MO_ask of size {init_size} depleted the bid book at t={self.current_time}")
            tw = int(initial_bb - final_bb) if initial_bb is not None else 0
            self.mo_ticks_walked.append(max(0, tw))
            self._event_detail = {
                'type': 'MO', 'side_text': 'sell', 'side_int': 2,
                'mo_volume': init_size, 'ticks_walked': max(0, tw),
                'fills': list(self._mo_fill_log),
                '_pre_bb': initial_bb, '_pre_ba': ba,
            }
            if size <= 0:
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
            print(f"Adjusting bid price from {price * self.tick_size:.2f} to {(ba - 1) * self.tick_size:.2f} to avoid crossing")
            price = int(ba - 1)

        price = max(1, int(price))

        if ba is not None and ba > 0 and price > 0:
            log_price = np.log(price)
            delta0_log = np.log(ba) - log_price
        else:
            log_price = np.log(price) if price > 0 else 0.0
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
            print(f"Adjusting ask price from {price * self.tick_size:.2f} to {(bb + 1) * self.tick_size:.2f} to avoid crossing")
            price = int(bb + 1)

        price = max(1, int(price))

        if bb is not None and bb > 0 and price > 0:
            log_price = np.log(price)
            delta0_log = log_price - np.log(bb)
        else:
            log_price = np.log(price) if price > 0 else 0.0
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
        """Cancel a resting order on ``side`` (1 = bid, 2 = ask)."""
        if side == 1:
            self.cancel_stats['bid_attempts'] += 1
        elif side == 2:
            self.cancel_stats['ask_attempts'] += 1
        else:
            raise ValueError(f"_execute_cxl: side must be 1 or 2, got {side!r}")

        n_side = self._bid_n if side == 1 else self._ask_n
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

        removed = self.ob.delete(chosen_oid)
        if not removed:
            print(f"cancellation: oid {chosen_oid} not found in book")

        start = self._remove_order(side, chosen_oid)
        duration = self.current_stamp - (start if start is not None else self.current_stamp)
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
        """
        Event processing with P(C|y)-weighted cancellation.

        For cancellations, each order is weighted by P(C|y) where
        y = δ(t) / δ₀, with δ(t) the current distance from the opposite best
        quote and δ₀ the distance when the order was placed.
        P(C|y) is computed on the fly in _compute_cancel_weights.
        """
        self._event_detail = None  # reset; set below if event modifies book

        spread = None
        if bb is not None and ba is not None:
            spread = ba - bb

        # Track log BBO for cancel-weight y computation
        if ba is not None and ba > 0:
            self.last_log_best_ask = np.log(ba)
        if bb is not None and bb > 0:
            self.last_log_best_bid = np.log(bb)

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

    def run(self, overwrite=False):
        # clear previous diagnostics so repeated calls start fresh
        self.lifetimes.clear()
        self.lo_stats.clear()
        self.mo_sizes.clear()
        self.mo_ticks_walked.clear()
        self._tw_depth_history.clear()
        self._tw_adaptive_bounds = None
        self.frames.clear()
        self.executed_events.clear()
        self.cancel_stats = {'bid_attempts': 0, 'bid_success': 0,
                             'ask_attempts': 0, 'ask_success': 0}
        self.mo_stats = {'bid_attempts': 0, 'bid_filled': 0,
                         'ask_attempts': 0, 'ask_filled': 0}
        self.cancel_y_log.clear()
        self.cancel_f_log.clear()
        self.cancel_dsame_log.clear()

        # Reset liquidity-guard counters
        for k in self._guard_stats:
            self._guard_stats[k] = 0

        # Clear compact buffers
        self._fills_compact.clear()
        self._mo_compact.clear()

        if not self.lightweight:
            # -- Event database (heavy mode only) --
            self._orders_buf.clear()
            self._fills_buf.clear()
            self._mo_buf.clear()
            self._bbo_buf.clear()
            self._intensities_buf.clear()
            self._prev_event_time = None
            self._prev_mid = None
            self._open_db(overwrite=overwrite)

        self.last_trade_price = None  # reset trade tracker

        # Restore Hawkes auxiliary state (seeded state if set, else cold).
        d = len(self.labels)
        if hasattr(self, '_A_seeded') and self._A_seeded is not None:
            self._A_list = [A.copy() for A in self._A_seeded]
        else:
            self._A_list = [np.zeros((d, d)) for _ in range(self._n_kernels)]
        self._t_last = 0.0

        _diag_evt = {"LO_bid": 0, "LO_ask": 0, "CXL_bid": 0, "CXL_ask": 0,
                     "MO_bid": 0, "MO_ask": 0}
        _diag_guard_remap = 0

        for counter in range(1, self.T + 1):

            # Sample next event on the fly (Ogata thinning)
            t, label, intensities = self._sample_next_event()
            raw_label = label
            if self.liquidity_guard:
                label = self._guard_event(label)
                if label != raw_label:
                    _diag_guard_remap += 1

            _diag_evt[label] = _diag_evt.get(label, 0) + 1

            if counter % 50_000 == 0:
                print(f"[{counter:>7d}]  bid_d={self.ob.total_bid_depth:>10,.0f}  "
                      f"ask_d={self.ob.total_ask_depth:>10,.0f}  "
                      f"n_bid={self._bid_n:>5d}  n_ask={self._ask_n:>5d}  "
                      f"LO={_diag_evt['LO_bid']+_diag_evt['LO_ask']:>6d}  "
                      f"CXL={_diag_evt['CXL_bid']+_diag_evt['CXL_ask']:>6d}  "
                      f"MO={_diag_evt['MO_bid']+_diag_evt['MO_ask']:>6d}  "
                      f"guard_remap={_diag_guard_remap}")

            if not self.lightweight and self._db_conn is not None:
                self._intensities_buf.append((t, *intensities))

            # expose current time and index for lifetime logging
            self.current_time = t
            self.current_index = counter
            self.current_stamp = t + counter * 1e-9

            bb_pre, ba_pre = self.ob.get_bbo()

            if not self.lightweight:
                # -- Pre-event state for DB --
                _pre = (self._snapshot_pre_event(bb_pre, ba_pre)
                        if self._db_conn is not None else None)

            # process event
            self._agent_fills.clear()
            if self.process_event(label, bb_pre, ba_pre) == 2:
                continue
            self.current_time = t

            if self.agents and (not self.lightweight or self.agents_when_lightweight):
                fills = list(self._agent_fills)
                for agent in self.agents:
                    agent.on_event(self, t, fills)

            if self.lightweight:
                # ── Lightweight recording: compact fills & MO data ──
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
            else:
                self.executed_events.append(label)

                # -- Record event to database --
                if _pre is not None:
                    self._record_to_db(t, label, _pre, counter)

                bb, ba = self.ob.get_bbo()
                if bb is None or ba is None or ba <= bb:
                    continue

        # -- Finalize database --
        if not self.lightweight:
            self._close_db()

        # -- Liquidity-guard summary --
        total_interventions = sum(self._guard_stats.values())
        if self.liquidity_guard and total_interventions > 0:
            pct = 100.0 * total_interventions / self.T
            print(f"\nLiquidity guard: {total_interventions} interventions "
                  f"({pct:.2f}% of {self.T} events)")
            for k, v in self._guard_stats.items():
                if v > 0:
                    print(f"  {k}: {v}")

    def get_compact_results(self):
        """Return compact in-memory results for lightweight runs.

        Returns
        -------
        dict with keys:
            'fills_ts'      : np.ndarray of fill timestamps
            'fills_price'   : np.ndarray of fill prices
            'mo_ts'         : np.ndarray of MO timestamps
            'mo_side'       : np.ndarray of MO sides ('buy'/'sell')
            'mo_best_bid'   : np.ndarray of best bid at MO time
            'mo_best_ask'   : np.ndarray of best ask at MO time
            'mo_ticks_walked' : np.ndarray of ticks walked per MO
        """
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

"""Discrete-tick ergodic HJB solver kernels.

Slim copy of the discrete-mode kernels from
``notebooks/numerical_as_numba.py``, used by ``NumericalErgodicMM`` in
:mod:`research_core.classes.market_maker`.  The full kernel set
(continuous fine grid, h-mode fill probabilities, golden-section search,
warm-up routine) lives only in the notebook because it is research code
used by the validation sweep.

Functions provided:

- :func:`precompute_lam_grid_discrete` -- pure-Python builder for a
  tick-spaced ``(delta_grid, lam_grid)`` table from an arrival-rate
  callable.  Supports negative ``delta_lo`` so the grid can include
  aggressive quotes inside the spread.
- :func:`precompute_lam_grid_discrete_from_h` -- same as above but
  built from a fill-probability callable using the Poisson link
  ``lambda = -ln(1 - h) / tau``.
- :func:`ham_val` -- the Hamiltonian
  ``H(delta, p) = (lam / gamma) * (1 - exp(-gamma * (delta - p)))``.
- :func:`discrete_max_H` -- O(n) linear scan for the global maximum
  of ``H`` on a discrete tick grid.
- :func:`solve_ergodic_discrete` -- pure-Numba relaxation loop with
  reflecting inventory boundaries.
- :func:`optimal_deltas_discrete` -- extracts ``(delta_b, delta_a)`` at
  a given inventory from the converged value function.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import numba
import numpy as np


def build_delta_grid(
    tick_size: float, max_delta: float, delta_lo: float
) -> np.ndarray:
    """Tick-spaced ``δ`` grid from ``delta_lo`` to ``max_delta``."""
    tick = float(tick_size)
    lo = float(delta_lo)
    hi = float(max_delta)
    j_lo = int(math.ceil(lo / tick)) if lo < tick else 1
    j_hi = int(math.floor(hi / tick))
    n = max(1, j_hi - j_lo + 1)
    return tick * np.arange(j_lo, j_lo + n, dtype=np.int64).astype(np.float64)


def precompute_lam_grid_discrete(
    lam_fn: Callable[[Any, float], float],
    z: Any,
    tick_size: float,
    max_delta: float,
    delta_lo: float = 0.0,
    lam_floor: float = 0.0,
    delta_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tabulate ``lam(z, delta)`` at every tick from ``delta_lo`` to ``max_delta``.

    When ``delta_lo < 0``, the grid includes negative tick levels (aggressive
    quotes inside the spread).  Grid spacing equals ``tick_size`` exactly,
    so the inner loop can use a direct argmax instead of golden-section search.

    Performance: tries one *vectorised* call ``lam_fn(z, delta_grid)``
    first; falls back to a scalar Python loop if the callable returns
    something other than an array of matching shape (e.g. an old-style
    scalar-only function).  Vectorised callbacks are strongly preferred
    when the grid is rebuilt per quote (state-dependent fill laws).
    """
    if delta_grid is None:
        delta_grid = build_delta_grid(tick_size, max_delta, delta_lo)
    else:
        delta_grid = np.asarray(delta_grid, dtype=np.float64)
        if delta_grid.ndim != 1 or delta_grid.size == 0:
            raise ValueError("delta_grid must be a non-empty 1D array")
        if not np.isfinite(delta_grid).all():
            raise ValueError("delta_grid must contain only finite values")
        if np.any(np.diff(delta_grid) <= 0.0):
            raise ValueError("delta_grid must be strictly increasing")
    n = delta_grid.shape[0]
    floor = float(lam_floor)

    try:
        candidate = np.asarray(lam_fn(z, delta_grid), dtype=np.float64)
        if candidate.shape == delta_grid.shape:
            mask = ~np.isfinite(candidate)
            if mask.any():
                candidate = np.where(mask, floor, candidate)
            np.maximum(candidate, floor, out=candidate)
            np.minimum.accumulate(candidate, out=candidate)
            return delta_grid, candidate
    except Exception:
        pass

    lam_grid = np.empty(n, dtype=np.float64)
    for j_idx in range(n):
        d = float(delta_grid[j_idx])
        val = float(lam_fn(z, d))
        lam_grid[j_idx] = max(val, floor) if math.isfinite(val) else floor
    np.minimum.accumulate(lam_grid, out=lam_grid)
    return delta_grid, lam_grid


def precompute_lam_grid_discrete_from_h(
    h_fn: Callable[[Any, float], float],
    z: Any,
    tau: float,
    tick_size: float,
    max_delta: float,
    delta_lo: float = 0.0,
    h_clamp: float = 1e-9,
    lam_floor: float = 0.0,
    delta_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tabulate arrival rate at tick-spaced deltas from a fill-probability ``h(z, d)``.

    Uses the Poisson link ``lambda = -ln(1 - h) / tau``.  Values are
    clamped to ``[h_clamp, 1 - h_clamp]`` for numerical stability.  Grid
    geometry matches :func:`precompute_lam_grid_discrete`.

    Vectorised when ``h_fn(z, delta_grid)`` returns an array; otherwise
    falls back to a per-tick scalar loop.
    """
    if delta_grid is None:
        delta_grid = build_delta_grid(tick_size, max_delta, delta_lo)
    else:
        delta_grid = np.asarray(delta_grid, dtype=np.float64)
        if delta_grid.ndim != 1 or delta_grid.size == 0:
            raise ValueError("delta_grid must be a non-empty 1D array")
        if not np.isfinite(delta_grid).all():
            raise ValueError("delta_grid must contain only finite values")
        if np.any(np.diff(delta_grid) <= 0.0):
            raise ValueError("delta_grid must be strictly increasing")
    n = delta_grid.shape[0]
    tau_f = float(tau)
    h_eps = float(h_clamp)
    floor = float(lam_floor)

    try:
        h_arr = np.asarray(h_fn(z, delta_grid), dtype=np.float64)
        if h_arr.shape == delta_grid.shape:
            h_arr = np.where(np.isfinite(h_arr), h_arr, 0.5)
            np.clip(h_arr, h_eps, 1.0 - h_eps, out=h_arr)
            lam_grid = -np.log1p(-h_arr) / tau_f
            np.maximum(lam_grid, floor, out=lam_grid)
            np.minimum.accumulate(lam_grid, out=lam_grid)
            return delta_grid, lam_grid
    except Exception:
        pass

    lam_grid = np.empty(n, dtype=np.float64)
    for j_idx in range(n):
        d = float(delta_grid[j_idx])
        h = float(h_fn(z, d))
        if not math.isfinite(h):
            h = 0.5
        h = min(max(h, h_eps), 1.0 - h_eps)
        lam_grid[j_idx] = max(-math.log(1.0 - h) / tau_f, floor)
    np.minimum.accumulate(lam_grid, out=lam_grid)
    return delta_grid, lam_grid


@numba.njit
def ham_val(delta: float, p: float, gamma: float, lam: float) -> float:
    """``H(delta, p) = (lam / gamma) * (1 - exp(-gamma * (delta - p)))``."""
    arg = -gamma * (delta - p)
    if arg > 700.0:
        return -1e300
    exp_val = math.exp(arg)
    return (lam / gamma) * (1.0 - exp_val)


@numba.njit
def _mask_popcount(mask: int) -> int:
    """Numba-compatible population count for the joint quote DP."""
    count = 0
    while mask:
        count += mask & 1
        mask >>= 1
    return count


@numba.njit
def coordinated_side_argmax(value_cube: np.ndarray) -> Tuple[np.ndarray, float]:
    """Exactly maximise one side of the coordinated Hamiltonian.

    ``value_cube[i, j, r]`` is agent ``i``'s Hamiltonian contribution when it
    quotes candidate level ``j`` and has ``r`` coordinated orders ahead of it.
    Candidate levels run from most to least aggressive.  Agents sharing a
    level receive FIFO priority in ascending agent-index order.

    The dynamic program processes one price level at a time and assigns any
    subset of the remaining agents to that level.  Its complexity is
    ``O(n_levels * 3**n_agents)`` rather than enumerating
    ``n_levels**n_agents`` complete quote vectors.
    """
    n_agents = value_cube.shape[0]
    n_levels = value_cube.shape[1]
    if value_cube.ndim != 3 or value_cube.shape[2] < n_agents:
        raise ValueError(
            "value_cube must have shape (n_agents, n_levels, >=n_agents)"
        )
    if n_agents < 1 or n_agents > 12:
        raise ValueError("coordinated_side_argmax supports 1..12 agents")

    n_masks = 1 << n_agents
    full_mask = n_masks - 1
    neg_inf = -1e300

    dp = np.full(n_masks, neg_inf, dtype=np.float64)
    dp[0] = 0.0
    parent = np.full((n_levels + 1, n_masks), -1, dtype=np.int64)
    choice = np.zeros((n_levels + 1, n_masks), dtype=np.int64)
    parent[0, 0] = 0

    for level in range(n_levels):
        next_dp = np.full(n_masks, neg_inf, dtype=np.float64)
        for assigned in range(n_masks):
            base = dp[assigned]
            if base <= neg_inf:
                continue

            remaining = full_mask ^ assigned
            subset = remaining
            while True:
                rank = _mask_popcount(assigned)
                gain = 0.0
                for agent_idx in range(n_agents):
                    if subset & (1 << agent_idx):
                        gain += value_cube[agent_idx, level, rank]
                        rank += 1

                new_mask = assigned | subset
                score = base + gain
                if score > next_dp[new_mask]:
                    next_dp[new_mask] = score
                    parent[level + 1, new_mask] = assigned
                    choice[level + 1, new_mask] = subset

                if subset == 0:
                    break
                subset = (subset - 1) & remaining
        dp = next_dp

    selected_levels = np.full(n_agents, -1, dtype=np.int64)
    mask = full_mask
    for level_plus_one in range(n_levels, 0, -1):
        subset = choice[level_plus_one, mask]
        for agent_idx in range(n_agents):
            if subset & (1 << agent_idx):
                selected_levels[agent_idx] = level_plus_one - 1
        mask = parent[level_plus_one, mask]

    return selected_levels, dp[full_mask]


@numba.njit
def discrete_max_H(
    p: float,
    gamma: float,
    lam_grid: np.ndarray,
    delta_grid: np.ndarray,
) -> Tuple[float, float]:
    """Find the global maximum of ``H`` on a discrete tick grid via linear scan.

    Uses O(n) evaluations to guarantee the correct global optimum
    regardless of the shape of ``lam_grid`` (handles cliff-shaped,
    non-smooth, or plateau lambda profiles from NN fill laws).
    """
    n = lam_grid.shape[0]
    best_j = 0
    best_H = ham_val(delta_grid[0], p, gamma, lam_grid[0])
    for j in range(1, n):
        H_j = ham_val(delta_grid[j], p, gamma, lam_grid[j])
        if H_j > best_H:
            best_H = H_j
            best_j = j
    return delta_grid[best_j], best_H


@numba.njit
def build_H_envelope(
    lam_grid: np.ndarray,
    delta_grid: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Upper envelope of the Hamiltonian lines for a frozen lambda grid.

    For a fixed grid, ``H_j(p) = a_j - b_j * x`` with ``x = exp(gamma * p)``,
    ``a_j = lam_j / gamma`` and ``b_j = a_j * exp(-gamma * delta_j)`` -- i.e.
    every candidate delta is a straight line in ``x`` and the discrete max
    over the grid is the upper envelope of those lines.  Building the
    envelope once per solve turns each ``discrete_max_H`` linear scan
    (O(n) ``exp`` calls) into a binary search (O(log n), one ``exp``).

    ``lam_grid`` is non-increasing (enforced by the grid builders), so the
    lines arrive sorted by slope ``-b_j`` and a single monotone pass
    suffices.  The result is *exactly* the same maximiser as the scan --
    no approximation is involved.

    Returns ``(env_idx, env_x, m)``: hull line indices into the grid,
    crossover points (``env_x[k]`` is where line ``k`` starts to dominate
    line ``k-1``; ``env_x[0]`` is unused), and the hull size ``m``.
    """
    n = lam_grid.shape[0]
    env_idx = np.empty(n, dtype=np.int64)
    env_x = np.empty(n, dtype=np.float64)

    m = 0
    for j in range(n):
        a_j = lam_grid[j] / gamma
        b_j = a_j * math.exp(-gamma * delta_grid[j])

        k = env_idx[m - 1] if m > 0 else -1
        if m > 0:
            a_k = lam_grid[k] / gamma
            b_k = a_k * math.exp(-gamma * delta_grid[k])
            if b_j == b_k:
                # Parallel lines: a is non-increasing, so j never exceeds
                # the incumbent -- skip (keeps the scan's first-index
                # tie-break for identical lines).
                continue

        while m > 0:
            k = env_idx[m - 1]
            a_k = lam_grid[k] / gamma
            b_k = a_k * math.exp(-gamma * delta_grid[k])
            # b_j < b_k here, so the crossover is well defined and >= 0.
            x_cross = (a_j - a_k) / (b_j - b_k)
            x_start_k = env_x[m - 1] if m >= 2 else 0.0
            if x_cross <= x_start_k:
                m -= 1  # line k never dominates once j exists
            else:
                env_x[m] = x_cross
                break

        if m == 0:
            env_x[0] = 0.0
        env_idx[m] = j
        m += 1

    return env_idx, env_x, m


@numba.njit
def query_H_envelope(
    p: float,
    gamma: float,
    env_idx: np.ndarray,
    env_x: np.ndarray,
    m: int,
    lam_grid: np.ndarray,
    delta_grid: np.ndarray,
) -> Tuple[float, float]:
    """Evaluate ``max_j H_j(p)`` via the precomputed envelope.

    Falls back to the exact linear scan when ``exp(gamma * p)`` would
    overflow (never happens for realistic ``phi`` differences, but keeps
    the kernel total).  At exact crossover points the *lower* index wins
    (strict ``<``), matching the scan's first-maximum tie-break.
    """
    arg = gamma * p
    if arg > 690.0:
        return discrete_max_H(p, gamma, lam_grid, delta_grid)
    x = math.exp(arg)

    lo = 0
    hi = m - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if env_x[mid] < x:
            lo = mid
        else:
            hi = mid - 1

    # Evaluate H at the winning line via the original ham_val formula so
    # the returned value is bit-identical to the linear scan whenever the
    # argmax agrees (keeps relaxation trajectories exactly reproducible).
    j = env_idx[lo]
    return delta_grid[j], ham_val(delta_grid[j], p, gamma, lam_grid[j])


@numba.njit
def solve_ergodic_discrete_hull(
    S: np.ndarray,
    lam_grid_b: np.ndarray,
    lam_grid_a: np.ndarray,
    delta_grid_b: np.ndarray,
    delta_grid_a: np.ndarray,
    gamma: float,
    relax_step: float,
    tol: float,
    max_iter: int,
    Q: int,
    phi0: np.ndarray,
) -> Tuple[np.ndarray, float, int, bool, float]:
    """Drop-in replacement for :func:`solve_ergodic_discrete` that uses the
    line-envelope maximisation instead of per-query linear scans.

    Identical relaxation scheme, boundaries, pinning, tolerance and return
    contract; only the inner ``max_j H_j`` is computed via
    :func:`query_H_envelope` (same maximum, computed in O(log n)).
    """
    n = 2 * Q + 1
    idx0 = Q

    env_idx_b, env_x_b, m_b = build_H_envelope(lam_grid_b, delta_grid_b, gamma)
    env_idx_a, env_x_a, m_a = build_H_envelope(lam_grid_a, delta_grid_a, gamma)

    phi = phi0.copy()
    Hb = np.empty(n)
    Ha = np.empty(n)

    g = 0.0
    delta_phi = 1e300
    n_iter = 0

    for it in range(max_iter):
        for i in range(n):
            p_b = phi[i] - phi[i + 1] if i < n - 1 else 0.0
            p_a = phi[i] - phi[i - 1] if i > 0 else 0.0

            if i == 0:
                _, hb_val = query_H_envelope(
                    p_b, gamma, env_idx_b, env_x_b, m_b,
                    lam_grid_b, delta_grid_b)
                Hb[i] = hb_val
                Ha[i] = 0.0
            elif i == n - 1:
                Hb[i] = 0.0
                _, ha_val = query_H_envelope(
                    p_a, gamma, env_idx_a, env_x_a, m_a,
                    lam_grid_a, delta_grid_a)
                Ha[i] = ha_val
            else:
                _, hb_val = query_H_envelope(
                    p_b, gamma, env_idx_b, env_x_b, m_b,
                    lam_grid_b, delta_grid_b)
                _, ha_val = query_H_envelope(
                    p_a, gamma, env_idx_a, env_x_a, m_a,
                    lam_grid_a, delta_grid_a)
                Hb[i] = hb_val
                Ha[i] = ha_val

        g = S[idx0] + Hb[idx0] + Ha[idx0]

        delta_phi = 0.0
        for i in range(n):
            R_i = -g + S[i] + Hb[i] + Ha[i]
            new_phi_i = phi[i] + relax_step * R_i
            diff = abs(new_phi_i - phi[i])
            if diff > delta_phi:
                delta_phi = diff
            phi[i] = new_phi_i

        anchor = phi[idx0]
        for i in range(n):
            phi[i] -= anchor

        n_iter = it + 1

        if delta_phi < tol:
            return phi, g, n_iter, True, delta_phi

    p_b0 = phi[idx0] - phi[idx0 + 1] if idx0 < n - 1 else 0.0
    p_a0 = phi[idx0] - phi[idx0 - 1] if idx0 > 0 else 0.0
    _, hb0 = discrete_max_H(p_b0, gamma, lam_grid_b, delta_grid_b)
    _, ha0 = discrete_max_H(p_a0, gamma, lam_grid_a, delta_grid_a)
    g = S[idx0] + hb0 + ha0

    return phi, g, n_iter, False, delta_phi


@numba.njit
def solve_ergodic_discrete(
    S: np.ndarray,
    lam_grid_b: np.ndarray,
    lam_grid_a: np.ndarray,
    delta_grid_b: np.ndarray,
    delta_grid_a: np.ndarray,
    gamma: float,
    relax_step: float,
    tol: float,
    max_iter: int,
    Q: int,
    phi0: np.ndarray,
) -> Tuple[np.ndarray, float, int, bool, float]:
    """Pure-Numba ergodic relaxation solver using discrete tick-spaced grids.

    Reflecting inventory boundaries: at ``q=-Q`` only the bid side quotes
    (``Ha=0``); at ``q=+Q`` only the ask side quotes (``Hb=0``).  The
    ergodic constant ``g`` is pinned at ``q=0``.

    Returns ``(phi, g, n_iter, converged, last_delta_phi)``.
    """
    n = 2 * Q + 1
    idx0 = Q

    phi = phi0.copy()
    Hb = np.empty(n)
    Ha = np.empty(n)

    g = 0.0
    delta_phi = 1e300
    n_iter = 0

    for it in range(max_iter):
        for i in range(n):
            p_b = phi[i] - phi[i + 1] if i < n - 1 else 0.0
            p_a = phi[i] - phi[i - 1] if i > 0 else 0.0

            if i == 0:
                _, hb_val = discrete_max_H(p_b, gamma, lam_grid_b, delta_grid_b)
                Hb[i] = hb_val
                Ha[i] = 0.0
            elif i == n - 1:
                Hb[i] = 0.0
                _, ha_val = discrete_max_H(p_a, gamma, lam_grid_a, delta_grid_a)
                Ha[i] = ha_val
            else:
                _, hb_val = discrete_max_H(p_b, gamma, lam_grid_b, delta_grid_b)
                _, ha_val = discrete_max_H(p_a, gamma, lam_grid_a, delta_grid_a)
                Hb[i] = hb_val
                Ha[i] = ha_val

        g = S[idx0] + Hb[idx0] + Ha[idx0]

        delta_phi = 0.0
        for i in range(n):
            R_i = -g + S[i] + Hb[i] + Ha[i]
            new_phi_i = phi[i] + relax_step * R_i
            diff = abs(new_phi_i - phi[i])
            if diff > delta_phi:
                delta_phi = diff
            phi[i] = new_phi_i

        anchor = phi[idx0]
        for i in range(n):
            phi[i] -= anchor

        n_iter = it + 1

        if delta_phi < tol:
            return phi, g, n_iter, True, delta_phi

    p_b0 = phi[idx0] - phi[idx0 + 1] if idx0 < n - 1 else 0.0
    p_a0 = phi[idx0] - phi[idx0 - 1] if idx0 > 0 else 0.0
    _, hb0 = discrete_max_H(p_b0, gamma, lam_grid_b, delta_grid_b)
    _, ha0 = discrete_max_H(p_a0, gamma, lam_grid_a, delta_grid_a)
    g = S[idx0] + hb0 + ha0

    return phi, g, n_iter, False, delta_phi


@numba.njit
def optimal_deltas_discrete(
    inventory: int,
    phi: np.ndarray,
    gamma: float,
    lam_grid_b: np.ndarray,
    lam_grid_a: np.ndarray,
    delta_grid_b: np.ndarray,
    delta_grid_a: np.ndarray,
    Q: int,
) -> Tuple[float, float]:
    """Optimal ``(delta_b, delta_a)`` from converged phi using discrete tick grids."""
    n = 2 * Q + 1
    idx0 = Q
    i = idx0 + inventory
    p_b = phi[i] - phi[i + 1] if i < n - 1 else 0.0
    p_a = phi[i] - phi[i - 1] if i > 0 else 0.0

    if i == 0:
        db, _ = discrete_max_H(p_b, gamma, lam_grid_b, delta_grid_b)
        return db, 0.0
    if i == n - 1:
        da, _ = discrete_max_H(p_a, gamma, lam_grid_a, delta_grid_a)
        return 0.0, da

    db, _ = discrete_max_H(p_b, gamma, lam_grid_b, delta_grid_b)
    da, _ = discrete_max_H(p_a, gamma, lam_grid_a, delta_grid_a)
    return db, da

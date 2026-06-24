#!/usr/bin/env python3
"""Build 100 calibration quintets via simulated annealing.

Loads an empirical order-flow SQLite DB, takes ``n_days`` keys after skipping
``skip`` leading days, computes per-day session returns from first/last
``orders.mid_price`` (continuous session is already applied at extraction),
then partitions 500 day-slots into 100 bundles of 5 **distinct** days with
multiplicity 2 or 3 per day.

**Thesis / default window:** ``skip=6``, ``n_days=229``. If the DB has at least
235 distinct days, days ``all_keys[6:235]`` are used. If the DB has fewer
(e.g. exactly 229 days), ``skip`` is **clamped** to ``max(0, len(all_keys)-n_days)``
so all DB days can still be used (typically ``skip_effective=0``).

With 229 evaluation days the multiplicities are
forced to **187*2 + 42*3 = 500** slot assignments (42 days appear in three
quintets each, 187 in two). Other ``n_days`` in [167, 250] are supported for
tests only.

Energy: sum over bundles of (mean return in bundle)^2.

See project plan ``Quintet SA sampling`` for data conventions.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from research_core.classes.helpers import (
    list_day_keys_from_sqlite,
    resolve_data_path,
)

# Evaluation calendar (thesis): skip first 6 DB days, then 229 days.
EVAL_SKIP_FIRST_DAYS = 6
EVAL_N_DAYS = 229


def load_evaluation_day_keys(
    db_path: Path,
    skip: int,
    n_days: int,
    *,
    strict_skip: bool = False,
) -> Tuple[List[str], int, int]:
    """Return ``(day_keys, skip_effective, n_distinct_in_db)``.

    If ``len(all_keys) < skip + n_days``, by default ``skip`` is reduced so
    ``all_keys[skip_effective : skip_effective + n_days]`` still fits and uses
    as many DB days as requested (exactly ``n_days`` keys). With a 229-day DB
    and ``n_days=229``, this yields ``skip_effective=0`` and all calendar days.

    Set ``strict_skip=True`` to require ``skip + n_days`` days instead
    (original behaviour).
    """
    all_keys = list_day_keys_from_sqlite(db_path)
    if n_days > len(all_keys):
        raise ValueError(
            f"n_days={n_days} exceeds distinct days in DB ({len(all_keys)})"
        )
    need = skip + n_days
    if len(all_keys) < need:
        if strict_skip:
            raise ValueError(
                f"strict_skip: need at least {need} distinct days in DB, "
                f"got {len(all_keys)}"
            )
        skip_effective = max(0, len(all_keys) - n_days)
        if skip_effective != skip:
            print(
                f"Note: DB has {len(all_keys)} days; clamping skip "
                f"{skip} -> {skip_effective} so n_days={n_days} fits.",
                file=sys.stderr,
            )
    else:
        skip_effective = skip
    return (
        all_keys[skip_effective : skip_effective + n_days],
        skip_effective,
        len(all_keys),
    )


def daily_returns_from_sqlite(
    db_path: Path,
    day_keys: Sequence[str],
    mode: str = "log",
) -> np.ndarray:
    """First-to-last mid log-return (or simple return) per day from ``orders``.

    Uses **one** pass over ``orders`` via window functions (SQLite 3.25+):
    first and last ``mid_price`` by ``timestamp`` per ``day`` among rows with
    non-null mid. Tie-break with ``rowid`` so first/last are deterministic.
    """
    if not day_keys:
        return np.array([], dtype=np.float64)

    placeholders = ",".join("?" * len(day_keys))
    sql = f"""
WITH filtered AS (
    SELECT day, mid_price, timestamp, rowid
    FROM orders
    WHERE day IN ({placeholders})
      AND mid_price IS NOT NULL
),
ranked AS (
    SELECT day, mid_price,
        ROW_NUMBER() OVER (
            PARTITION BY day ORDER BY timestamp ASC, rowid ASC
        ) AS rn_a,
        ROW_NUMBER() OVER (
            PARTITION BY day ORDER BY timestamp DESC, rowid DESC
        ) AS rn_z
    FROM filtered
)
SELECT day,
    MAX(CASE WHEN rn_a = 1 THEN mid_price END) AS first_mid,
    MAX(CASE WHEN rn_z = 1 THEN mid_price END) AS last_mid
FROM ranked
GROUP BY day
"""
    print(
        f"Loading daily returns: 1 query, {len(day_keys)} days ...",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.time()
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(sql, tuple(day_keys))
        rows = cur.fetchall()
    finally:
        conn.close()

    by_day = {str(r[0]): (float(r[1]), float(r[2])) for r in rows}
    elapsed = time.time() - t0
    print(
        f"  daily returns: {len(rows)} days in {elapsed:.2f}s",
        file=sys.stderr,
        flush=True,
    )

    out = np.empty(len(day_keys), dtype=np.float64)
    for i, day in enumerate(day_keys):
        pair = by_day.get(day)
        if pair is None:
            raise ValueError(f"No mid_price rows for day {day}")
        a, b = pair
        if a <= 0 or b <= 0:
            raise ValueError(f"Non-positive mid for {day}: first={a}, last={b}")
        if mode == "log":
            out[i] = math.log(b) - math.log(a)
        elif mode == "simple":
            out[i] = (b - a) / a
        else:
            raise ValueError(f"Unknown return mode: {mode}")
    return out


def _multiplicity_vector(n_days: int, n_triple: int, rng: np.random.Generator) -> np.ndarray:
    """f[d] in {2,3} with exactly n_triple days at 3 and rest at 2.

    For n_days=229, n_triple=42: 42*3 + 187*2 = 500.
    """
    if n_triple * 3 + (n_days - n_triple) * 2 != 500:
        raise ValueError(
            f"Incompatible n_days={n_days}, n_triple={n_triple}: "
            "need 3*n_triple + 2*(n_days-n_triple) == 500"
        )
    f = np.full(n_days, 2, dtype=np.int32)
    triple_days = rng.choice(n_days, size=n_triple, replace=False)
    f[triple_days] = 3
    return f


def _row_unique(state: np.ndarray, b: int) -> bool:
    return len(np.unique(state[b])) == 5


def swap_maintains_unique_rows(
    state: np.ndarray,
    b1: int,
    j1: int,
    b2: int,
    j2: int,
) -> bool:
    if b1 == b2:
        return j1 != j2
    d1_new = int(state[b2, j2])
    d2_new = int(state[b1, j1])
    row1 = list(state[b1])
    row1[j1] = d1_new
    row2 = list(state[b2])
    row2[j2] = d2_new
    return len(set(row1)) == 5 and len(set(row2)) == 5


def shuffle_repair_initial_quintets(
    f: np.ndarray,
    rng: np.random.Generator,
    max_reshuffles: int = 500,
    max_repair: int = 100_000,
) -> np.ndarray:
    """Build (100, 5) day indices: shuffle multiset of tokens, repair row duplicates by swaps."""
    n_days = len(f)
    tokens = np.concatenate([np.full(int(f[d]), d, dtype=np.int32) for d in range(n_days)])
    assert tokens.size == 500

    for _ in range(max_reshuffles):
        rng.shuffle(tokens)
        state = tokens.reshape(100, 5).copy()
        for _rep in range(max_repair):
            bad = [b for b in range(100) if not _row_unique(state, b)]
            if not bad:
                return state
            b1 = int(rng.choice(bad))
            row = state[b1]
            vals, counts = np.unique(row, return_counts=True)
            dup_vals = vals[counts >= 2]
            d_dup = int(rng.choice(dup_vals))
            js = np.flatnonzero(row == d_dup)
            j1 = int(rng.choice(js))
            improved = False
            for _try in range(500):
                b2 = int(rng.integers(0, 100))
                j2 = int(rng.integers(0, 5))
                if b1 == b2 and j1 == j2:
                    continue
                if swap_maintains_unique_rows(state, b1, j1, b2, j2):
                    state[b1, j1], state[b2, j2] = state[b2, j2], state[b1, j1]
                    improved = True
                    break
            if not improved:
                break
    raise RuntimeError(
        "shuffle_repair_initial_quintets: could not obtain 100 unique rows; "
        "increase max_repair or reshuffles"
    )


def energy_quintets(state: np.ndarray, r: np.ndarray) -> float:
    """Sum_b (mean_{d in bundle b} r_d)^2."""
    means = r[state].mean(axis=1)
    return float(np.dot(means, means))


def grid_mean_return(state: np.ndarray, r: np.ndarray) -> float:
    """Mean log-return (or simple return) over all 500 grid cells ``r[state[b,j]]``."""
    return float(np.mean(r[state]))


def _assignment_row_is_unique(
    state: np.ndarray,
    bs: np.ndarray,
    js: np.ndarray,
    row_idx: int,
    assignment: dict,
) -> bool:
    """Check row uniqueness after applying a partial slot assignment."""
    seen: set = set()
    for col_idx in range(5):
        value = int(state[row_idx, col_idx])
        for assignment_idx, assigned_value in assignment.items():
            if (
                int(bs[int(assignment_idx)]) == row_idx
                and int(js[int(assignment_idx)]) == col_idx
            ):
                value = int(assigned_value)
                break
        if value in seen:
            return False
        seen.add(value)
    return True


def _assignment_row_sum(
    state: np.ndarray,
    r: np.ndarray,
    row_sums: np.ndarray,
    bs: np.ndarray,
    js: np.ndarray,
    row_idx: int,
    assignment: dict,
) -> float:
    """Return a row's return sum after applying a partial assignment."""
    row_sum = float(row_sums[row_idx])
    for assignment_idx, assigned_value in assignment.items():
        slot_idx = int(assignment_idx)
        if int(bs[slot_idx]) != row_idx:
            continue
        old_day = int(state[int(bs[slot_idx]), int(js[slot_idx])])
        row_sum += float(r[int(assigned_value)]) - float(r[old_day])
    return row_sum


def _partial_assignment_energy(
    state: np.ndarray,
    r: np.ndarray,
    row_sums: np.ndarray,
    bs: np.ndarray,
    js: np.ndarray,
    rows_set: Sequence[int],
    assignment: dict,
) -> float:
    """Score affected rows for a partial reassignment."""
    total = 0.0
    inv5 = 0.2
    for row_idx in rows_set:
        if not _assignment_row_is_unique(state, bs, js, row_idx, assignment):
            return float("inf")
        row_mean = _assignment_row_sum(
            state, r, row_sums, bs, js, row_idx, assignment
        ) * inv5
        total += row_mean * row_mean
    return total


def _greedy_multiset_assign(
    state: np.ndarray,
    r: np.ndarray,
    row_sums: np.ndarray,
    bs: np.ndarray,
    js: np.ndarray,
    slots_per_move: int,
    old_vals: np.ndarray,
    rows: np.ndarray,
) -> Optional[np.ndarray]:
    """Assign the ``slots_per_move`` day ids to those slots to reduce local energy.

    Slots are filled in descending order of |bundle mean| of the affected row
    (rows that look most ``off'' first). Each step picks a remaining day id
    that minimizes the sum of squared bundle means over affected rows,
    subject to row uniqueness (5 distinct days). Returns shape
    ``(slots_per_move,)`` new values aligned with ``bs, js``, or ``None``.

    ``row_sums[b]`` must equal ``r[state[b]].sum()`` for all rows ``b``.
    """
    rows_set = [int(x) for x in np.unique(rows)]
    inv5 = 0.2

    slot_order = sorted(
        range(slots_per_move),
        key=lambda k: (
            -abs(float(row_sums[int(bs[k])]) * inv5),
            int(bs[k]),
            int(js[k]),
        ),
    )

    remaining = [int(x) for x in old_vals.tolist()]
    assignment: dict = {}

    for k in slot_order:
        best_i: Optional[int] = None
        best_obj = float("inf")
        best_v = 0
        for i, v in enumerate(remaining):
            trial = dict(assignment)
            trial[int(k)] = int(v)
            obj = _partial_assignment_energy(
                state, r, row_sums, bs, js, rows_set, trial
            )
            if obj >= float("inf"):
                continue
            if best_i is None or obj < best_obj - 1e-18:
                best_obj = obj
                best_i = i
                best_v = int(v)
            elif abs(obj - best_obj) <= 1e-18:
                if abs(r[int(v)]) < abs(r[int(best_v)]) or (
                    abs(r[int(v)]) == abs(r[int(best_v)]) and int(v) < int(best_v)
                ):
                    best_obj = obj
                    best_i = i
                    best_v = int(v)
        if best_i is None:
            return None
        assignment[int(k)] = best_v
        remaining.pop(int(best_i))

    new_vals = np.empty(slots_per_move, dtype=np.int32)
    for k in range(slots_per_move):
        new_vals[k] = int(assignment[k])
    return new_vals


def _try_random_slot_shuffle(
    state: np.ndarray,
    r: np.ndarray,
    row_sums: np.ndarray,
    rng: np.random.Generator,
    slots_per_move: int,
    max_attempts: int = 40,
) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Propose reassigning ``slots_per_move`` distinct cells' day indices.

    Picks ``slots_per_move`` distinct linear indices uniformly, then **greedily**
    permutes the multiset of day ids occupying those cells: slots in rows
    with the largest |bundle mean| are decided first; each step picks the
    remaining day id that minimizes the sum of squared bundle means on the
    affected rows, while keeping 5 distinct days per affected quintet.

    ``row_sums[b]`` must equal ``r[state[b]].sum()`` for all ``b``.

    Returns ``(delta_E, bs, js, new_vals)`` on success; ``None`` if no valid
    greedy completion within ``max_attempts`` random slot draws.
    """
    if slots_per_move < 2 or slots_per_move > 500:
        raise ValueError("slots_per_move must be in [2, 500]")

    inv5 = 0.2
    for _ in range(max_attempts):
        idx = rng.choice(500, size=slots_per_move, replace=False)
        bs = (idx // 5).astype(np.int32)
        js = (idx % 5).astype(np.int32)
        old_vals = state[bs, js].astype(np.int64).copy()
        rows = np.unique(bs)

        new_vals_arr = _greedy_multiset_assign(
            state, r, row_sums, bs, js, slots_per_move, old_vals, rows
        )
        if new_vals_arr is None:
            continue
        new_vals = new_vals_arr
        if np.array_equal(old_vals, new_vals):
            continue

        delta = 0.0
        for b in rows:
            b = int(b)
            m_old = float(row_sums[b] * inv5)
            S_new = float(row_sums[b])
            for k in range(slots_per_move):
                if int(bs[k]) != b:
                    continue
                S_new += float(r[int(new_vals[k])]) - float(
                    r[int(state[int(bs[k]), int(js[k])])]
                )
            m_new = S_new * inv5
            delta += m_new * m_new - m_old * m_old

        return delta, bs, js, new_vals.astype(np.int32)

    return None


def simulated_annealing(
    state: np.ndarray,
    r: np.ndarray,
    rng: np.random.Generator,
    n_steps: int,
    T0: float,
    T_min: float,
    alpha: float,
    log_every: int,
    *,
    quiet: bool = False,
    slots_per_move: int = 6,
) -> Tuple[np.ndarray, float, list]:
    """Metropolis on random multi-slot proposals; returns (best_state, best_E, history).

    Each proposal picks ``slots_per_move`` distinct positions in the 100×5
    grid and **greedily** reassigns the multiset of day indices on those cells
    to reduce local sum of squared bundle means (see
    ``_try_random_slot_shuffle``), preserving global multiplicities.
    """
    state = state.copy()
    row_sums = np.asarray(r[state], dtype=np.float64).sum(axis=1)
    cur_e = energy_quintets(state, r)
    best_state = state.copy()
    best_e = cur_e
    history: list = []

    T = T0
    t0 = time.time()
    n_accept_win = 0
    n_valid_win = 0

    if not quiet:
        print(
            f"SA start  steps={n_steps}  log_every={log_every}  "
            f"slots_per_move={slots_per_move}  "
            f"E={cur_e:.6g}  best_E={best_e:.6g}  T0={T0}",
            flush=True,
        )

    accept = False
    for step in range(n_steps):
        if not quiet and log_every and step > 0 and step % log_every == 0:
            p_acc = (n_accept_win / n_valid_win) if n_valid_win else float("nan")
            elapsed = time.time() - t0
            pct = 100.0 * step / max(n_steps, 1)
            print(
                f"SA  step {step}/{n_steps}  ({pct:.1f}%)  "
                f"T={T:.4g}  E={cur_e:.6g}  best={best_e:.6g}  "
                f"p_acc(valid)={p_acc:.3f}  elapsed={elapsed:.1f}s",
                flush=True,
            )
            n_accept_win = 0
            n_valid_win = 0

        prop = _try_random_slot_shuffle(
            state,
            r,
            row_sums,
            rng,
            slots_per_move=slots_per_move,
        )
        if prop is None:
            if log_every and step % log_every == 0:
                history.append({"step": step, "T": T, "E": cur_e, "accept": False})
            T = max(T * alpha, T_min)
            continue

        delta, bs, js, new_vals = prop
        accept = delta <= 0.0 or rng.random() < math.exp(-delta / T)
        n_valid_win += 1
        if accept:
            n_accept_win += 1
            for k in range(slots_per_move):
                state[int(bs[k]), int(js[k])] = int(new_vals[k])
            for b in np.unique(bs):
                b = int(b)
                row_sums[b] = float(r[state[b]].sum())
            cur_e += delta
            if cur_e < best_e:
                best_e = cur_e
                best_state = state.copy()

        if log_every and step % log_every == 0:
            history.append(
                {
                    "step": step,
                    "T": T,
                    "E": cur_e,
                    "best_E": best_e,
                    "accept": bool(accept),
                }
            )
        T = max(T * alpha, T_min)

    if not quiet and log_every and n_steps > 0 and (n_valid_win > 0 or n_accept_win > 0):
        p_acc = (n_accept_win / n_valid_win) if n_valid_win else float("nan")
        elapsed = time.time() - t0
        print(
            f"SA  step {n_steps}/{n_steps}  (100.0%)  "
            f"T={T:.4g}  E={cur_e:.6g}  best={best_e:.6g}  "
            f"p_acc(valid, tail)={p_acc:.3f}  elapsed={elapsed:.1f}s",
            flush=True,
        )

    history.append(
        {
            "step": n_steps,
            "wall_s": time.time() - t0,
            "final_E": cur_e,
            "best_E": best_e,
        }
    )
    if not quiet:
        print(
            f"SA done  wall={history[-1]['wall_s']:.2f}s  "
            f"final_E={cur_e:.6g}  best_E={best_e:.6g}",
            flush=True,
        )
    return best_state, best_e, history


def slot_multiplicities(state: np.ndarray, n_days: int) -> np.ndarray:
    """Per-day counts in the 100×5 grid (each should be 2 or 3 for valid layouts)."""
    return np.bincount(state.ravel().astype(np.int64), minlength=n_days).astype(
        np.int32
    )


def _try_multiplicity_swap(
    state: np.ndarray,
    r: np.ndarray,
    row_sums: np.ndarray,
    n_days: int,
    rng: np.random.Generator,
    *,
    row_bias_power: float = 2.0,
    uniform_mix: float = 0.15,
    max_attempts: int = 160,
) -> Optional[Tuple[float, int, int, int, int]]:
    """Replace one cell holding a triple-count day A with double-count day B.

    Counts change 3→2 for ``A`` and 2→3 for ``B``; only one row is modified.
    Returns ``(delta_E, b, j, A, B)`` or ``None``.
    """
    counts = slot_multiplicities(state, n_days)
    double_ids = np.flatnonzero(counts == 2)
    if double_ids.size == 0:
        return None

    inv5 = 0.2
    w_row = (np.abs(row_sums * inv5) + 1e-18) ** float(row_bias_power)
    w_row = np.asarray(w_row, dtype=np.float64)
    w_row /= float(w_row.sum())

    for _ in range(max_attempts):
        if rng.random() < float(uniform_mix):
            b = int(rng.integers(0, 100))
        else:
            b = int(rng.choice(100, p=w_row))
        j = int(rng.integers(0, 5))
        A = int(state[b, j])
        if counts[A] != 3:
            continue
        B = int(rng.choice(double_ids))
        row = state[b].astype(np.int64).copy()
        row[j] = B
        if len(np.unique(row)) != 5:
            continue

        m_old = float(row_sums[b] * inv5)
        S_new = float(row_sums[b]) + float(r[B]) - float(r[A])
        m_new = S_new * inv5
        delta = m_new * m_new - m_old * m_old
        return delta, b, j, A, B

    return None


def simulated_annealing_multiplicity_swaps(
    state: np.ndarray,
    r: np.ndarray,
    rng: np.random.Generator,
    n_days: int,
    n_steps: int,
    T0: float,
    T_min: float,
    alpha: float,
    log_every: int,
    *,
    quiet: bool = False,
    row_bias_power: float = 2.0,
    uniform_mix: float = 0.15,
    metropolis_objective: str = "energy",
) -> Tuple[np.ndarray, float, list]:
    """Metropolis on **multiplicity** moves: 3× day ↔ 2× day at one grid cell.

    Each proposal picks a row biased toward large |bundle mean|, replaces a
    cell that currently holds a day with global count 3 by a day with count 2,
    subject to row uniqueness. Recomputes ``triple_mask``-style counts via
    ``slot_multiplicities`` after the run if needed.

    ``metropolis_objective`` selects what Metropolis minimizes (accept if
    ``Δ ≤ 0`` or ``rng < exp(-Δ/T)`` with ``Δ = objective_after - objective_before``):

    - ``"energy"`` (default): :math:`\\sum_b \\bar r_b^2` (``energy_quintets``).
      Second return value is best energy.
    - ``"grid_mean"``: mean of ``r[state]`` over all 500 cells. Use when the
      grid-average return should move **down** (or explore around it).
      Second return value is the **lowest mean** achieved.
    - ``"abs_grid_mean"``: absolute value of that grid mean (toward **zero**).
      Second return value is the **lowest |mean|** achieved.

    Bundle energy is still updated on each accept for logging; ``best_state`` /
    ``best_e`` refer to the best layout under the chosen objective.

    Returns ``(best_state, best_score, history)`` (``best_score`` is ``best_E``
    in ``"energy"`` mode, else best mean or best |mean| as above).
    """
    if metropolis_objective not in ("energy", "grid_mean", "abs_grid_mean"):
        raise ValueError(
            "metropolis_objective must be 'energy', 'grid_mean', or 'abs_grid_mean'"
        )

    state = state.copy()
    row_sums = np.asarray(r[state], dtype=np.float64).sum(axis=1)
    cur_e = energy_quintets(state, r)
    cur_mean = grid_mean_return(state, r)
    best_state = state.copy()
    if metropolis_objective == "energy":
        best_e = cur_e
    elif metropolis_objective == "grid_mean":
        best_e = float(cur_mean)
    else:
        best_e = float(abs(cur_mean))
    history: list = []

    T = T0
    t0 = time.time()
    n_accept_win = 0
    n_valid_win = 0

    if not quiet:
        print(
            f"SA(mult) start  steps={n_steps}  log_every={log_every}  "
            f"E={cur_e:.6g}  best_E={best_e:.6g}  mean={cur_mean:.6g}  T0={T0}  "
            f"objective={metropolis_objective}  "
            f"row_bias_power={row_bias_power}  uniform_mix={uniform_mix}",
            flush=True,
        )

    accept = False
    for step in range(n_steps):
        if not quiet and log_every and step > 0 and step % log_every == 0:
            p_acc = (n_accept_win / n_valid_win) if n_valid_win else float("nan")
            elapsed = time.time() - t0
            pct = 100.0 * step / max(n_steps, 1)
            print(
                f"SA(mult)  step {step}/{n_steps}  ({pct:.1f}%)  "
                f"T={T:.4g}  E={cur_e:.6g}  best={best_e:.6g}  mean={cur_mean:.6g}  "
                f"p_acc(valid)={p_acc:.3f}  elapsed={elapsed:.1f}s",
                flush=True,
            )
            n_accept_win = 0
            n_valid_win = 0

        prop = _try_multiplicity_swap(
            state,
            r,
            row_sums,
            n_days,
            rng,
            row_bias_power=row_bias_power,
            uniform_mix=uniform_mix,
        )
        if prop is None:
            if log_every and step % log_every == 0:
                history.append(
                    {
                        "step": step,
                        "T": T,
                        "E": cur_e,
                        "grid_mean": cur_mean,
                        "accept": False,
                        "phase": "mult",
                    }
                )
            T = max(T * alpha, T_min)
            continue

        delta, b, j, A, B = prop
        d_mean = (float(r[int(B)]) - float(r[int(A)])) / 500.0
        if metropolis_objective == "energy":
            delta_metro = float(delta)
        elif metropolis_objective == "grid_mean":
            delta_metro = d_mean
        else:
            delta_metro = abs(cur_mean + d_mean) - abs(cur_mean)

        accept = delta_metro <= 0.0 or rng.random() < math.exp(-delta_metro / T)
        n_valid_win += 1
        if accept:
            n_accept_win += 1
            state[int(b), int(j)] = int(B)
            row_sums[int(b)] = float(r[state[int(b)]].sum())
            cur_e += float(delta)
            cur_mean += d_mean
            if metropolis_objective == "energy":
                if cur_e < best_e:
                    best_e = cur_e
                    best_state = state.copy()
            elif metropolis_objective == "grid_mean":
                if cur_mean < best_e:
                    best_e = float(cur_mean)
                    best_state = state.copy()
            else:
                if abs(cur_mean) < best_e:
                    best_e = float(abs(cur_mean))
                    best_state = state.copy()

        if log_every and step % log_every == 0:
            history.append(
                {
                    "step": step,
                    "T": T,
                    "E": cur_e,
                    "grid_mean": cur_mean,
                    "best_E": best_e,
                    "accept": bool(accept),
                    "phase": "mult",
                    "metropolis_objective": metropolis_objective,
                }
            )
        T = max(T * alpha, T_min)

    if not quiet and log_every and n_steps > 0 and (n_valid_win > 0 or n_accept_win > 0):
        p_acc = (n_accept_win / n_valid_win) if n_valid_win else float("nan")
        elapsed = time.time() - t0
        print(
            f"SA(mult)  step {n_steps}/{n_steps}  (100.0%)  "
            f"T={T:.4g}  E={cur_e:.6g}  best={best_e:.6g}  mean={cur_mean:.6g}  "
            f"p_acc(valid, tail)={p_acc:.3f}  elapsed={elapsed:.1f}s",
            flush=True,
        )

    history.append(
        {
            "step": n_steps,
            "wall_s": time.time() - t0,
            "final_E": cur_e,
            "final_grid_mean": cur_mean,
            "best_E": best_e,
            "phase": "mult",
            "metropolis_objective": metropolis_objective,
        }
    )
    if not quiet:
        print(
            f"SA(mult) done  wall={history[-1]['wall_s']:.2f}s  "
            f"final_E={cur_e:.6g}  best_score={best_e:.6g}  final_mean={cur_mean:.6g}  "
            f"objective={metropolis_objective}",
            flush=True,
        )
    return best_state, best_e, history


def _bundle_means(state: np.ndarray, r: np.ndarray) -> np.ndarray:
    return r[state].mean(axis=1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db",
        type=str,
        default="KGHM_order_flow.sqlite",
        help="SQLite path (relative to data_dir if not absolute)",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=EVAL_SKIP_FIRST_DAYS,
        help=f"Skip first N calendar days in DB order (default {EVAL_SKIP_FIRST_DAYS})",
    )
    p.add_argument(
        "--n-days",
        type=int,
        default=EVAL_N_DAYS,
        help=f"Number of evaluation days after skip (default {EVAL_N_DAYS} for thesis window)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--T0", type=float, default=1.0)
    p.add_argument("--T-min", type=float, default=1e-6)
    p.add_argument("--alpha", type=float, default=0.99995)
    p.add_argument(
        "--log-every",
        type=int,
        default=None,
        metavar="N",
        help="Progress + history every N steps (default: steps//100, i.e. 1%%; "
        "e.g. 2000 for --steps 200000). Use <=0 for same auto.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress SA progress lines (JSON unchanged)",
    )
    p.add_argument(
        "--slots-per-move",
        type=int,
        default=6,
        metavar="K",
        help="Metropolis proposal: greedy reassignment on K distinct cells (default 6)",
    )
    p.add_argument(
        "--return-mode",
        choices=("log", "simple"),
        default="log",
        help="log: log(last/first); simple: (last-first)/first",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (default: data_dir/calibration_quintets.json)",
    )
    p.add_argument(
        "--strict-skip",
        action="store_true",
        help="Fail if DB has fewer than skip+n_days days (no skip clamping)",
    )
    args = p.parse_args(argv)

    if args.log_every is None or args.log_every <= 0:
        log_every = max(1, args.steps // 100)
    else:
        log_every = args.log_every

    if args.slots_per_move < 2 or args.slots_per_move > 500:
        p.error("--slots-per-move must be between 2 and 500")

    n_triple = 500 - 2 * args.n_days
    if not (0 <= n_triple <= args.n_days):
        p.error(
            f"need 0 <= 500 - 2*n_days <= n_days; for n_days={args.n_days} "
            f"got n_triple={n_triple} (valid n_days range is 167..250; "
            f"thesis default is {EVAL_N_DAYS})"
        )

    db_path = resolve_data_path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    day_keys, skip_effective, n_db_days = load_evaluation_day_keys(
        db_path, args.skip, args.n_days, strict_skip=args.strict_skip,
    )
    r = daily_returns_from_sqlite(db_path, day_keys, mode=args.return_mode)

    n_days = args.n_days
    f = _multiplicity_vector(n_days, n_triple, rng)
    triple_mask = f == 3

    state = shuffle_repair_initial_quintets(f, rng)
    e0 = energy_quintets(state, r)
    best, best_e, hist = simulated_annealing(
        state,
        r,
        rng,
        n_steps=args.steps,
        T0=args.T0,
        T_min=args.T_min,
        alpha=args.alpha,
        log_every=log_every,
        quiet=args.quiet,
        slots_per_move=args.slots_per_move,
    )

    means = _bundle_means(best, r)
    out_path = Path(args.out) if args.out else resolve_data_path("calibration_quintets.json")

    quintets_keys: List[List[str]] = [
        [day_keys[int(best[b, j])] for j in range(5)] for b in range(100)
    ]
    triple_day_keys = [day_keys[i] for i in np.where(triple_mask)[0].tolist()]

    payload = {
        "db": str(db_path),
        "n_distinct_days_in_db": n_db_days,
        "skip_requested": args.skip,
        "skip_effective": skip_effective,
        "n_days": n_days,
        "day_keys": list(day_keys),
        "return_mode": args.return_mode,
        "r_d": r.tolist(),
        "n_triple_slots": int(n_triple),
        "triple_day_keys": triple_day_keys,
        "quintets": quintets_keys,
        "energy_initial": e0,
        "energy_final": energy_quintets(best, r),
        "energy_best": best_e,
        "bundle_mean_returns": means.tolist(),
        "bundle_mean_abs_max": float(np.max(np.abs(means))),
        "sa": {
            "seed": args.seed,
            "steps": args.steps,
            "T0": args.T0,
            "T_min": args.T_min,
            "alpha": args.alpha,
            "slots_per_move": args.slots_per_move,
            "log_every_requested": args.log_every,
            "log_every": log_every,
            "history_tail": hist[-5:] if len(hist) > 5 else hist,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Wrote {out_path}")
    print(
        f"  DB days={n_db_days}  n_days={n_days}  "
        f"skip {args.skip}->{skip_effective}  "
        f"energy {e0:.6g} -> {best_e:.6g}  "
        f"max|bundle mean| = {payload['bundle_mean_abs_max']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Exact live-simulation queue-ahead features for NN market makers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def resolve_live_sim(feature_view: Any) -> Optional[Any]:
    """Return the underlying generative simulator, if one is available."""
    direct = getattr(feature_view, "underlying_sim", None)
    if direct is not None:
        return direct
    book_state = getattr(feature_view, "book_state", None)
    candidate = getattr(book_state, "_sim", None)
    if candidate is not None:
        return candidate
    if (
        hasattr(feature_view, "_bid_price_oids")
        and hasattr(feature_view, "_ask_price_oids")
    ):
        return feature_view
    return None


def _candidate_prices(agent, side: int, deltas, sim) -> np.ndarray:
    """Map assessed deltas to the exact native prices used by the HJB."""
    delta_values = np.atleast_1d(np.asarray(deltas, dtype=np.float64))
    suffix = "b" if side == 1 else "a"
    delta_grid = getattr(agent, f"_legal_dg_{suffix}", None)
    price_grid = getattr(agent, f"_pg_{suffix}", None)

    if delta_grid is not None and price_grid is not None:
        delta_grid = np.asarray(delta_grid, dtype=np.float64)
        price_grid = np.asarray(price_grid, dtype=np.float64)
        prices = np.empty(delta_values.shape, dtype=np.float64)
        matched = True
        for i, delta in enumerate(delta_values):
            idx = int(np.argmin(np.abs(delta_grid - delta)))
            if not np.isclose(
                delta_grid[idx], delta, rtol=0.0, atol=1e-10,
            ):
                matched = False
                break
            prices[i] = price_grid[idx]
        if matched:
            return prices

    bb, ba = sim.ob.get_bbo()
    if bb is None or ba is None:
        return np.full(delta_values.shape, np.nan, dtype=np.float64)
    native_to_pln = float(
        getattr(sim, "price_native_to_pln", 1.0) or 1.0
    )
    mid_pln = 0.5 * (float(bb) + float(ba)) * native_to_pln
    if side == 1:
        prices = (mid_pln - delta_values) / native_to_pln
    else:
        prices = (mid_pln + delta_values) / native_to_pln
    if bool(getattr(sim, "bbo_in_tick_index", False)):
        # Match NumericalErgodicMM._build_grids exactly: bids snap down and
        # asks snap up so that the assessed quote never improves past delta.
        prices = np.floor(prices) if side == 1 else np.ceil(prices)
    return prices


def _fifo_volume_before(price_oids, order_map, price, own_oid) -> float:
    """Remaining same-price volume ahead of ``own_oid`` in FIFO order."""
    queue = price_oids.get(price)
    if not queue or own_oid not in queue:
        return 0.0
    ahead = 0.0
    for oid in queue:
        if oid == own_oid:
            break
        order = order_map.get(oid)
        if order is not None:
            ahead += float(order[2])
    return ahead


def queue_ahead_from_live_sim(
    feature_view: Any,
    side: int,
    deltas,
    *,
    agent=None,
) -> Optional[np.ndarray]:
    """Return exact counterfactual queue ahead for live HJB candidates.

    At the agent's currently resting price, the order is retained and its
    actual FIFO priority is used. At every other candidate, the current order
    is counterfactually cancelled before the new order joins the back.
    ``None`` means that exact live-book state is unavailable.
    """
    sim = resolve_live_sim(feature_view)
    if sim is None:
        return None
    if not (
        hasattr(sim, "_bid_price_oids")
        and hasattr(sim, "_ask_price_oids")
        and hasattr(sim.ob, "order_map")
    ):
        return None

    delta_values = np.atleast_1d(np.asarray(deltas, dtype=np.float64))
    prices = _candidate_prices(agent, side, delta_values, sim)
    qty_by_price = sim.ob.bid_qty if side == 1 else sim.ob.ask_qty
    price_oids = (
        sim._bid_price_oids if side == 1 else sim._ask_price_oids
    )

    own_oid = None
    own_price = None
    if agent is not None:
        if side == 1:
            own_oid = getattr(agent, "bid_oid", None)
            own_price = getattr(agent, "_bid_price", None)
        else:
            own_oid = getattr(agent, "ask_oid", None)
            own_price = getattr(agent, "_ask_price", None)

    own_order = sim.ob.order_map.get(own_oid)
    own_valid = (
        own_order is not None
        and int(own_order[0]) == int(side)
        and own_price == own_order[1]
    )
    own_volume = float(own_order[2]) if own_valid else 0.0

    result = np.zeros(delta_values.shape, dtype=np.float64)
    for i, candidate_price in enumerate(prices):
        if not np.isfinite(candidate_price) or candidate_price <= 0:
            continue

        retained = (
            own_valid
            and candidate_price == own_price
            and own_oid in price_oids.get(candidate_price, {})
        )
        if side == 1:
            better = sum(
                float(volume)
                for price, volume in qty_by_price.items()
                if price > candidate_price
            )
            own_is_counted = own_valid and own_price >= candidate_price
        else:
            better = sum(
                float(volume)
                for price, volume in qty_by_price.items()
                if price < candidate_price
            )
            own_is_counted = own_valid and own_price <= candidate_price

        if retained:
            same_price_ahead = _fifo_volume_before(
                price_oids, sim.ob.order_map, candidate_price, own_oid,
            )
            result[i] = max(0.0, better + same_price_ahead)
        else:
            queue = better + float(qty_by_price.get(candidate_price, 0.0))
            if own_is_counted:
                queue -= own_volume
            result[i] = max(0.0, queue)
    return result

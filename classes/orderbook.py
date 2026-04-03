"""Heap-based order book for WSE LOB data.

Provides:
- HeapOrderBook      : lightweight order book with O(1) BBO queries
- compute_bbo_series : compute best-bid / best-ask at every order message
- find_continuous_trading_start : detect the end of the opening-auction artifact
"""

import heapq

import numpy as np
import pandas as pd


class HeapOrderBook:
    """
    Lightweight order book using heaps for O(1) best-bid / best-ask queries.

    Maintains per-price aggregate quantities and per-order state.
    """

    def __init__(self):
        self.order_map = {}   # order_id -> [side, price, volume]
        self.bid_qty = {}     # price -> total qty on bid side
        self.ask_qty = {}     # price -> total qty on ask side
        self.bid_heap = []    # max-heap (stored as negated prices)
        self.ask_heap = []    # min-heap
        self.total_bid_depth = 0   # rolling total bid volume
        self.total_ask_depth = 0   # rolling total ask volume

    def clear(self):
        """
        F action: wipe the book.
        """

        self.order_map.clear()
        self.bid_qty.clear()
        self.ask_qty.clear()
        self.bid_heap.clear()
        self.ask_heap.clear()
        self.total_bid_depth = 0
        self.total_ask_depth = 0

    def add(self, order_id, side, price, volume):
        """
        A / Y action: add an order (skip duplicates).
        """

        if side not in (1, 2, 5):
            return
        if order_id in self.order_map:
            return  # ignore duplicate snapshot replay

        self.order_map[order_id] = [side, price, volume]
        book = self.bid_qty if side == 1 else self.ask_qty
        book[price] = book.get(price, 0) + volume

        if side == 1:
            heapq.heappush(self.bid_heap, -price)
            self.total_bid_depth += volume
        else:
            heapq.heappush(self.ask_heap, price)
            self.total_ask_depth += volume

    def modify(self, order_id, new_volume):
        """
        M action: partial fill (new_volume = remaining qty).
        """

        if order_id not in self.order_map:
            return False

        old_side, old_price, old_volume = self.order_map[order_id]
        delta = old_volume - new_volume
        book = self.bid_qty if old_side == 1 else self.ask_qty

        if old_price in book:
            book[old_price] -= delta
            if book[old_price] <= 0:
                del book[old_price]

        if old_side == 1:
            self.total_bid_depth -= delta
        else:
            self.total_ask_depth -= delta

        if new_volume <= 0:
            del self.order_map[order_id]
        else:
            self.order_map[order_id][2] = new_volume
        return True

    def delete(self, order_id):
        """
        D action: remove order.
        """

        if order_id not in self.order_map:
            return False

        old_side, old_price, old_volume = self.order_map.pop(order_id)
        book = self.bid_qty if old_side == 1 else self.ask_qty
        if old_price in book:
            book[old_price] -= old_volume
            if book[old_price] <= 0:
                del book[old_price]
        if old_side == 1:
            self.total_bid_depth -= old_volume
        else:
            self.total_ask_depth -= old_volume
        return True

    def _clean_heaps(self):
        while self.bid_heap and (-self.bid_heap[0] not in self.bid_qty):
            heapq.heappop(self.bid_heap)
        while self.ask_heap and (self.ask_heap[0] not in self.ask_qty):
            heapq.heappop(self.ask_heap)

    def get_bbo(self):
        """
        Return (best_bid, best_ask) or (None, None) if book is empty.
        """

        self._clean_heaps()
        if self.bid_heap and self.ask_heap:
            return -self.bid_heap[0], self.ask_heap[0]
        return None, None

    def has_both_sides(self):
        self._clean_heaps()
        return bool(self.bid_heap) and bool(self.ask_heap)

    def apply_action(self, action_type, order_id, side, price, volume):
        """
        Apply a single order-book message.  Returns True on success,
        False if the order was not found (for M/D).
        """

        if action_type == "F":
            self.clear()
            return True
        elif action_type in ("A", "Y"):
            self.add(order_id, side, price, volume)
            return True
        elif action_type == "M":
            return self.modify(order_id, volume)
        elif action_type == "D":
            return self.delete(order_id)
        return False


# ---------------------------------------------------------------------------
#  BBO series computation
# ---------------------------------------------------------------------------

def compute_bbo_series(df: pd.DataFrame):
    """
    Compute best-bid / best-ask at every order message.

    Parameters
    ----------
    df : DataFrame with columns [time, action_type, side, price, volume, order_id]
         (already price-scaled, timestamps as datetime).

    Returns
    -------
    timestamps : ndarray of datetime64
    best_bid   : ndarray of float (NaN where book is one-sided)
    best_ask   : ndarray of float
    """

    df = df.sort_values("time")
    n = len(df)
    best_bid = np.full(n, np.nan)
    best_ask = np.full(n, np.nan)
    timestamps = df["time"].values

    ob = HeapOrderBook()
    for i, row in enumerate(df.itertuples(index=False)):
        ob.apply_action(row.action_type, row.order_id, row.side, row.price, row.volume)
        bb, ba = ob.get_bbo()
        if bb is not None:
            best_bid[i] = bb
            best_ask[i] = ba

    return timestamps, best_bid, best_ask


def find_continuous_trading_start(best_bid, best_ask, search_fraction=0.1, min_search=1000):
    """
    Detect the end of the opening-auction artifact in a BBO series.

    During the opening auction the book is temporarily crossed (bid >= ask)
    and/or has an extremely wide spread.  This function finds the last such
    anomalous tick in the initial portion of the day and returns the index
    of the first "clean" tick.

    Parameters
    ----------
    best_bid, best_ask : arrays (NaN-free, already filtered to market hours)
    search_fraction    : fraction of the day to search (default 10 %)
    min_search         : minimum number of ticks to search

    Returns
    -------
    stable_idx : int -- first tick of continuous trading
    info       : dict with diagnostic counts
    """

    n = len(best_bid)
    search_end = min(max(int(n * search_fraction), min_search), n)

    crossed = best_bid[:search_end] >= best_ask[:search_end]
    spread = best_ask[:search_end] - best_bid[:search_end]
    wide = spread > 1.0

    is_artifact = crossed | wide
    indices = np.where(is_artifact)[0]

    if len(indices) > 0:
        stable_idx = int(indices[-1]) + 1
    else:
        stable_idx = 0

    return stable_idx, {"crossed": int(crossed.sum()), "wide_spread": int(wide.sum())}

# Collusive Dynamics from Evolutionary Market-Making Strategies in a Calibrated Limit Order Book Simulation

This repository provides the simulation, calibration, and analysis toolkit for studying market microstructure and market-maker behaviour in a calibrated limit order book environment. Built around empirical data from the Warsaw Stock Exchange (WSE) WSELOB-2017 dataset.

## Overview

- **Simulation** (`classes/simulate.py`) -- Event-driven LOB simulator with Hawkes process arrivals (Poisson, univariate self-exciting, multivariate mutually exciting). Single- and triple-exponential kernels, empirical placement/cancellation distributions. Three recording modes (`recording_mode`): `'full'` (all SQLite tables), `'medium'` (`bbo` + `mo_orders` only), `'lightweight'` (in-memory only, no SQLite).
- **Calibration** (`classes/calibrate.py`) - Hawkes process parameter estimation via MLE with Optuna optimisation. Handles seasonality adjustment (raw time and tau-time), goodness-of-fit testing via the time-rescaling theorem, and parallel multi-worker calibration.
- **Order book** (`classes/orderbook.py`) - Heap-based order book implementation optimised for efficient simulation.
- **Extraction** (`classes/extract.py`) - Transforms raw WSE HDF5 order and trade data into structured SQLite databases with full book state snapshots at each event.
- **Analysis** (`classes/analyse.py`) - Statistical analysis of simulated and empirical order flow: stylised facts (fat tails, volatility clustering, order sign autocorrelation, price impact) and comparative diagnostics.
- **Market makers** (`classes/market_maker.py`) - Agent-based market-making strategies (simple symmetric, compact, Avellaneda-Stoikov) that plug into the simulator.
- **Backtesting** (`classes/backtest.py`) - Multi-window market-maker backtesting with PnL, inventory, and fill statistics.

## Installation

```bash
pip install -e .
```

Verify:

```bash
python -c "from research_core.classes import Simulate; print('OK')"
```

### Dependencies

All dependencies, including **tick** (used for Hawkes process MLE fitting), are declared in `pyproject.toml` and installed automatically.

## Package layout

| Path | Description |
|------|-------------|
| `classes/` | Core domain logic: simulation engine, calibration, analysis, order book, market makers, backtesting |
| `data/` | Database schema definitions (`schema.py`) and extraction helpers |
| `validation/` | Smoke tests (`python -m research_core.validation.smoke_test`) |
| `notebooks/` | Jupyter notebooks for calibration, simulation, analysis, and backtesting workflows |

## Usage

### Running a simulation

```python
from research_core.classes import Simulate

sim = Simulate(
    arrival_mode="hawkes_multivariate",
    T=184300,
    kernel_mode="triple",
    db_path="sim_events.sqlite",
    recording_mode="medium",   # 'full' | 'medium' | 'lightweight'
)
sim.load_real_orderbook_snapshot(
    asset="KGHM",
    day_key="d20170110",
    snapshot_time="10:00:00",
)
sim.run()
```

### Extracting empirical data

```python
from research_core.classes import run_full_extraction
from pathlib import Path

run_full_extraction(
    asset="KGHM",
    orders_h5=Path("data/WSELOB-2017/orders/KGHM_lob_2017_zlib.h5"),
    trades_h5=Path("data/WSELOB-2017/trades/KGHM_trades_2017_zlib.h5"),
    db_path=Path("data/KGHM_order_flow.sqlite"),
)
```

### Querying a database

```python
from research_core.classes import list_day_keys_from_sqlite, load_day_events_from_sqlite
import sqlite3

days = list_day_keys_from_sqlite("data/KGHM_order_flow.sqlite")
conn = sqlite3.connect("data/KGHM_order_flow.sqlite")
events = load_day_events_from_sqlite(conn, days[0], "09:00:00", ["LO", "CXL", "MO"])
conn.close()
```

---

## SQLite database schema

The package uses two families of SQLite databases with closely related but distinct schemas. Both are defined in `data/schema.py`.

### 1. Empirical order-flow database

Produced by `classes/extract.py` via `run_full_extraction()`. Each database covers a single asset (e.g. `KGHM_order_flow.sqlite`) and contains all trading days extracted from the raw WSE HDF5 files.

#### Table: `orders`

Every limit-order submission and cancellation observed in the LOB during continuous trading. Each row captures the book state **before** the event was applied.

| Column | Type | Description |
|--------|------|-------------|
| `day` | TEXT | Trading day key, e.g. `"d20170110"` |
| `timestamp` | TEXT | Intraday timestamp string |
| `event_type` | TEXT | `"LO"` (limit order) or `"CXL"` (cancellation) |
| `order_id` | INTEGER | Exchange-assigned order identifier |
| `side` | INTEGER | `1` = bid, `-1` = ask |
| `order_price` | REAL | Price of this order (PLN) |
| `best_bid` | REAL | Best bid before the event |
| `best_ask` | REAL | Best ask before the event |
| `best_same_side` | REAL | Best price on the same side as this order |
| `best_bid_size` | REAL | Volume at the best bid |
| `best_ask_size` | REAL | Volume at the best ask |
| `total_bid_depth` | REAL | Aggregate bid-side depth |
| `total_ask_depth` | REAL | Aggregate ask-side depth |
| `mid_price` | REAL | `(best_bid + best_ask) / 2` |
| `ticks_from_mid` | INTEGER | Signed distance from mid in ticks |
| `spread` | REAL | `best_ask - best_bid` (PLN) |
| `spread_ticks` | INTEGER | Spread in tick units |
| `imbalance` | REAL | `best_bid_size / (best_bid_size + best_ask_size)` |
| `ticks_from_best` | INTEGER | Distance from best same-side price in ticks |
| `queue_ahead` | REAL | Volume queued ahead of this order at its price level |
| `volume` | REAL | Order size |
| `delta0` | REAL | Hawkes kernel contribution at the event time |
| `delta_t` | REAL | Time-decayed kernel contribution |
| `y_ratio` | REAL | Depth ratio used in placement sampling |
| `dt_prev_event` | REAL | Time elapsed since previous event (seconds) |
| `n_total` | INTEGER | Total number of live orders in the book |
| `is_cancel` | INTEGER | `1` if cancellation, `0` if limit order |
| `microprice` | REAL | Volume-weighted mid: `(bb_size * ask + ba_size * bid) / (bb_size + ba_size)` |
| `n_bid` | INTEGER | Number of distinct bid price levels |
| `n_ask` | INTEGER | Number of distinct ask price levels |
| `dp_mid` | REAL | Change in mid-price since previous event |
| `bid_depth_L0..L4` | REAL | Bid-side depth at levels 0 (best) through 4 |
| `ask_depth_L0..L4` | REAL | Ask-side depth at levels 0 (best) through 4 |

#### Table: `fills`

Individual fills generated by market orders walking through the book. A single market order that sweeps multiple price levels produces one row per fill.

| Column | Type | Description |
|--------|------|-------------|
| `day` | TEXT | Trading day key |
| `time_ns` | INTEGER | Fill timestamp (nanoseconds) |
| `volume` | REAL | Filled volume |
| `price` | REAL | Execution price |
| `side` | TEXT | `"BUY"` or `"SELL"` (aggressor side) |
| `cls_method` | TEXT | Classification method used |
| `best_bid` | REAL | Best bid before the fill |
| `best_ask` | REAL | Best ask before the fill |
| `ticks_from_bbo` | INTEGER | How many ticks this fill is from BBO |
| `microprice` | REAL | Microprice before the fill |
| `opp_depth_L0..L9` | REAL | Opposite-side depth at 10 levels (the side being consumed) |
| `bid_depth_L0..L4` | REAL | Bid depth at 5 levels |
| `ask_depth_L0..L4` | REAL | Ask depth at 5 levels |

#### Table: `mo_orders`

Aggregated market-order records. One row per market order (which may comprise multiple fills if it walks the book).

| Column | Type | Description |
|--------|------|-------------|
| `day` | TEXT | Trading day key |
| `first_time_ns` | INTEGER | Timestamp of first fill (nanoseconds) |
| `side` | TEXT | `"BUY"` or `"SELL"` |
| `cls_method` | TEXT | Classification method |
| `mo_volume` | REAL | Total market-order volume |
| `n_fills` | INTEGER | Number of fills this MO generated |
| `min_price` | REAL | Lowest fill price |
| `max_price` | REAL | Highest fill price |
| `best_bid` | REAL | Best bid before execution |
| `best_ask` | REAL | Best ask before execution |
| `ticks_walked` | INTEGER | Price levels consumed beyond BBO |
| `ratio_L0` | REAL | `mo_volume / L0_depth`; fraction of top-of-book consumed |
| `microprice` | REAL | Microprice before execution |
| `opp_depth_L0..L9` | REAL | Opposite-side depth at 10 levels |
| `bid_depth_L0..L4` | REAL | Bid depth at 5 levels |
| `ask_depth_L0..L4` | REAL | Ask depth at 5 levels |

---

### 2. Simulation database

Produced by `Simulate.run()` when a `db_path` is supplied. Each run creates a fresh SQLite file. The database uses WAL journaling and `PRAGMA synchronous=NORMAL` for write performance, and creates timestamp indices on close.

The tables written depend on `recording_mode`:

| Mode | Tables | Use case |
|------|--------|----------|
| `'full'` (default) | `orders`, `fills`, `mo_orders`, `bbo`, `intensities` | Event-level analysis, order placement studies |
| `'medium'` | `mo_orders`, `bbo` | Stylised-facts analysis at reduced storage and compute |
| `'lightweight'` | *(none)* | Parallel batch runs; in-memory results via `get_compact_results()` |

#### Table: `orders` *(full mode only)*

Mirrors the empirical `orders` table with simulation-native types. The `day` column is dropped (simulations have no day boundaries) and `timestamp` is `REAL` (Hawkes process time, not wall-clock).

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | REAL | Simulation time (Hawkes process clock) |
| `event_type` | TEXT | `"LO"` or `"CXL"` |
| `order_id` | INTEGER | Simulation-internal order ID |
| `side` | INTEGER | `1` = bid, `-1` = ask |
| `order_price` | REAL | Price in PLN (`tick_price * tick_size`) |
| `best_bid` | REAL | Best bid before event (PLN) |
| `best_ask` | REAL | Best ask before event (PLN) |
| `best_same_side` | REAL | Best price on same side (PLN) |
| `best_bid_size` | REAL | Volume at best bid |
| `best_ask_size` | REAL | Volume at best ask |
| `total_bid_depth` | REAL | Total bid depth |
| `total_ask_depth` | REAL | Total ask depth |
| `mid_price` | REAL | Mid-price (PLN) |
| `ticks_from_mid` | INTEGER | Signed distance from mid |
| `spread` | REAL | Spread (PLN) |
| `spread_ticks` | INTEGER | Spread in ticks |
| `imbalance` | REAL | Bid / total BBO volume |
| `ticks_from_best` | INTEGER | Distance from best same-side |
| `queue_ahead` | REAL | Volume ahead in queue |
| `volume` | REAL | Order volume |
| `delta0` | REAL | Hawkes kernel value at event |
| `delta_t` | REAL | Time-decayed kernel value |
| `y_ratio` | REAL | Depth ratio for placement |
| `dt_prev_event` | REAL | Inter-event time |
| `n_total` | INTEGER | Total live orders |
| `is_cancel` | INTEGER | `1` if CXL, `0` if LO |
| `microprice` | REAL | Volume-weighted mid (PLN) |
| `n_bid` | INTEGER | Distinct bid levels |
| `n_ask` | INTEGER | Distinct ask levels |
| `dp_mid` | REAL | Mid-price change since previous event (PLN) |
| `bid_depth_L0..L4` | REAL | Bid depth at 5 levels |
| `ask_depth_L0..L4` | REAL | Ask depth at 5 levels |

**Index:** `idx_orders_ts ON orders(timestamp)`

#### Table: `fills` *(full mode only)*

One row per fill from a simulated market order.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | REAL | Simulation time |
| `volume` | REAL | Fill volume |
| `price` | REAL | Execution price (PLN) |
| `side` | TEXT | `"BUY"` or `"SELL"` |
| `best_bid` | REAL | Pre-fill best bid (PLN) |
| `best_ask` | REAL | Pre-fill best ask (PLN) |
| `ticks_from_bbo` | INTEGER | Ticks from BBO |
| `microprice` | REAL | Pre-fill microprice (PLN) |
| `opp_depth_L0..L9` | REAL | Opposite-side depth at 10 levels |
| `bid_depth_L0..L4` | REAL | Bid depth at 5 levels |
| `ask_depth_L0..L4` | REAL | Ask depth at 5 levels |

**Index:** `idx_fills_ts ON fills(timestamp)`

#### Table: `mo_orders` *(full and medium modes)*

Aggregated market orders. Drops `day`, `first_time_ns`, and `cls_method` relative to the empirical schema.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | REAL | Simulation time |
| `side` | TEXT | `"BUY"` or `"SELL"` |
| `mo_volume` | REAL | Total MO volume |
| `n_fills` | INTEGER | Number of fills |
| `min_price` | REAL | Lowest fill price (PLN) |
| `max_price` | REAL | Highest fill price (PLN) |
| `best_bid` | REAL | Pre-event best bid (PLN) |
| `best_ask` | REAL | Pre-event best ask (PLN) |
| `ticks_walked` | INTEGER | Levels consumed beyond BBO |
| `ratio_L0` | REAL | `mo_volume / L0_depth` |
| `microprice` | REAL | Pre-event microprice (PLN) |
| `opp_depth_L0..L9` | REAL | Opposite-side depth at 10 levels |
| `bid_depth_L0..L4` | REAL | Bid depth at 5 levels |
| `ask_depth_L0..L4` | REAL | Ask depth at 5 levels |

**Index:** `idx_mo_ts ON mo_orders(timestamp)`

#### Table: `bbo` *(full and medium modes)*

Best-bid/offer snapshot recorded on every event (including events that do not produce an `orders` row).

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | REAL | Simulation time |
| `best_bid` | REAL | Best bid (PLN) |
| `best_ask` | REAL | Best ask (PLN) |
| `mid_price` | REAL | Mid-price (PLN) |

**Index:** `idx_bbo_ts ON bbo(timestamp)`

#### Table: `intensities` *(full mode only)*

Hawkes process intensity snapshot at each event time.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | REAL | Simulation time |
| `mo_bid` | REAL | Market-order bid intensity |
| `mo_ask` | REAL | Market-order ask intensity |
| `lo_bid` | REAL | Limit-order bid intensity |
| `lo_ask` | REAL | Limit-order ask intensity |
| `cxl_bid` | REAL | Cancellation bid intensity |
| `cxl_ask` | REAL | Cancellation ask intensity |

**Index:** `idx_int_ts ON intensities(timestamp)`

---

### Key differences between empirical and simulation schemas

| Aspect | Empirical | Simulation |
|--------|-----------|------------|
| Time column | `timestamp TEXT` (wall-clock string) | `timestamp REAL` (Hawkes time) |
| Day partitioning | `day TEXT` column on every table | No day column (single continuous run) |
| MO timestamp | `first_time_ns INTEGER` (nanoseconds) | `timestamp REAL` |
| Fill timestamp | `time_ns INTEGER` (nanoseconds) | `timestamp REAL` |
| Classification | `cls_method TEXT` on fills/MO tables | Not applicable |
| Additional tables | -- | `bbo`, `intensities` |
| Write strategy | Bulk insert per day, then commit | Buffered (flush every N events), WAL mode |

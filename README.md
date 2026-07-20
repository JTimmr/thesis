# Collusive Dynamics from Adaptive Market-Making Strategies in a Calibrated Limit Order Book Simulation

Simulation, calibration, and analysis toolkit for studying market-maker behaviour in a limit order book calibrated to Warsaw Stock Exchange (WSE) data ([WSELOB-2017](https://data.mendeley.com/datasets/3g4mhdp899/1)).

The package installs as `research_core`. Work is organised around five notebooks in `notebooks/`.

## Install

```bash
pip install -e .
```

Verify:

```bash
python -c "from research_core.classes import Simulate; print('OK')"
```

Declared dependencies live in `pyproject.toml` (including `tick` for Hawkes MLE). Some notebooks also need **PyTorch**, **joblib**, and **Optuna** (Hawkes calibration and signature search); install those in the same environment if you run the empirical or fill-belief pipelines.

## Data setup

Paths resolve under `data/` via `research_core.classes.helpers.resolve_data_path`.

1. Download WSELOB-2017 from Mendeley and place the HDF5 files as:

```text
data/WSELOB-2017/
├── orders/
│   ├── KGHM_lob_2017_zlib.h5
│   └── …
└── trades/
    ├── KGHM_trades_2017_zlib.h5
    └── …
```

2. Run `notebooks/calibration.ipynb` (or call `run_full_extraction`) to build per-asset SQLite databases such as `data/KGHM_order_flow.sqlite`.

The repo already ships a few small calibration artefacts used by the simulator (e.g. `data/mo_depth_data/KGHM_tw_quartiles.npz`, seasonality cache). Large SQLite / HDF5 inputs are gitignored and must be supplied locally.

**Source:** Marszałek, Adam (2023), *WSELOB-2017*, Mendeley Data, V1, doi:[10.17632/3g4mhdp899.1](https://data.mendeley.com/datasets/3g4mhdp899/1).

## Notebooks

Run in this order when reproducing the thesis pipeline:

| Notebook | Role |
|----------|------|
| `calibration.ipynb` | Extract WSE order flow; fit the 6-D Hawkes model (production: single-exponential kernel) |
| `simulation_stylized_facts.ipynb` | Run the calibrated LOB simulator; check stylised facts vs empirical |
| `hjb_solver_validation.ipynb` | Frozen-path check that `NumericalErgodicMM` converges to Guéant `ErgodicMM` |
| `empirical_pipeline.ipynb` | Phantom fill labels, fill-probability NN, empirical market-maker analysis |
| `simulation_fill_belief_pipeline.ipynb` | Population fill-belief adaptation and competition simulations (live MM queue-ahead: exact FIFO priority, not full level size) |

Large runtime artefacts are local and gitignored: SQLite databases, simulation shards, fill-belief checkpoints, and report caches under the notebook data paths. They are not shipped in the repo; regenerate them or copy them locally to reproduce thesis figures.

## Package layout

| Path | Description |
|------|-------------|
| `classes/` | Domain logic: simulation, Hawkes calibration, analysis, order book, market makers, backtests, fill-belief / HJB helpers |
| `data/` | Schema (`schema.py`), small shipped calibration artefacts; runtime databases go here too |
| `notebooks/` | End-to-end workflows above |
| `pyproject.toml` | Package metadata and dependencies |

Import name is always `research_core` (see `package-dir` in `pyproject.toml`).

### Main modules

- **`simulate.py`** — Event-driven LOB with Hawkes arrivals. Recording modes: `full`, `medium`, `lightweight`.
- **`calibrate.py` / `parallelisation.py`** — Hawkes MLE via Optuna (+ `tick`), with multi-worker support.
- **`extract.py`** — WSE HDF5 → SQLite order-flow databases.
- **`analyse.py`** — Stylised-fact and comparative diagnostics on empirical or simulated DBs.
- **`market_maker.py` / `_ergodic_solver.py`** — Market-making agents, including Guéant ergodic and numerical HJB.
- **`backtest.py` / `mm_backtest_parallel.py` / `mm_competition.py`** — Backtests and multi-agent competition sims.
- **`phantom_labels.py` / `fill_rl.py` / `fill_belief_*` / `mm_pipeline_*`** — Fill labels, belief adaptation, and reporting.

Full SQLite table definitions for empirical and simulation databases are in [`data/schema.py`](data/schema.py).

## Quick usage

### Simulation

```python
from research_core.classes import Simulate

sim = Simulate(
    T=184300,
    db_path="sim_events.sqlite",
    recording_mode="medium",  # 'full' | 'medium' | 'lightweight'
)
sim.load_real_orderbook_snapshot(
    asset="KGHM",
    day_key="d20170110",
    snapshot_time="10:00:00",
)
sim.run()
```

### Extraction

```python
from pathlib import Path
from research_core.classes import run_full_extraction

run_full_extraction(
    asset="KGHM",
    orders_h5=Path("data/WSELOB-2017/orders/KGHM_lob_2017_zlib.h5"),
    trades_h5=Path("data/WSELOB-2017/trades/KGHM_trades_2017_zlib.h5"),
    db_path=Path("data/KGHM_order_flow.sqlite"),
)
```

### Query an empirical database

```python
import sqlite3
from research_core.classes import list_day_keys_from_sqlite, load_day_events_from_sqlite

days = list_day_keys_from_sqlite("data/KGHM_order_flow.sqlite")
conn = sqlite3.connect("data/KGHM_order_flow.sqlite")
events = load_day_events_from_sqlite(conn, days[0], "09:00:00", ["LO", "CXL", "MO"])
conn.close()
```

Simulation recording modes:

| Mode | Tables written | Typical use |
|------|----------------|-------------|
| `full` | `orders`, `fills`, `mo_orders`, `bbo`, `intensities` | Event-level analysis |
| `medium` | `mo_orders`, `bbo` | Stylised facts, lighter I/O |
| `lightweight` | none (in-memory) | Parallel batch runs |

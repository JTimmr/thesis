"""
DDL strings for empirical (WSE) and simulation SQLite databases.

Empirical tables match ``research_core.classes.extract``.
Simulation tables match ``research_core.classes.simulate`` (event DB written by ``Simulate.run``).
"""

# ── Empirical order-flow DB ───────────────────────────────────────────────

CREATE_ORDERS_TABLE = """
CREATE TABLE IF NOT EXISTS orders (
    day              TEXT,
    timestamp        TEXT,
    event_type       TEXT,
    order_id         INTEGER,
    side             INTEGER,

    order_price      REAL,
    best_bid         REAL,
    best_ask         REAL,
    best_same_side   REAL,

    best_bid_size    REAL,
    best_ask_size    REAL,

    total_bid_depth  REAL,
    total_ask_depth  REAL,

    mid_price        REAL,
    ticks_from_mid   INTEGER,

    spread           REAL,
    spread_ticks     INTEGER,

    imbalance        REAL,

    ticks_from_best  INTEGER,
    queue_ahead      REAL,

    volume           REAL,

    delta0           REAL,
    delta_t          REAL,
    y_ratio          REAL,

    dt_prev_event    REAL,

    n_total          INTEGER,
    is_cancel        INTEGER,

    microprice       REAL,
    n_bid            INTEGER,
    n_ask            INTEGER,
    dp_mid           REAL,

    bid_depth_L0     REAL,
    bid_depth_L1     REAL,
    bid_depth_L2     REAL,
    bid_depth_L3     REAL,
    bid_depth_L4     REAL,

    ask_depth_L0     REAL,
    ask_depth_L1     REAL,
    ask_depth_L2     REAL,
    ask_depth_L3     REAL,
    ask_depth_L4     REAL
)
"""

CREATE_FILLS_TABLE = """
CREATE TABLE IF NOT EXISTS fills (
    day              TEXT,
    time_ns          INTEGER,
    volume           REAL,
    price            REAL,
    side             TEXT,
    cls_method       TEXT,
    best_bid         REAL,
    best_ask         REAL,
    ticks_from_bbo   INTEGER,
    microprice       REAL,

    opp_depth_L0     REAL,
    opp_depth_L1     REAL,
    opp_depth_L2     REAL,
    opp_depth_L3     REAL,
    opp_depth_L4     REAL,
    opp_depth_L5     REAL,
    opp_depth_L6     REAL,
    opp_depth_L7     REAL,
    opp_depth_L8     REAL,
    opp_depth_L9     REAL,

    bid_depth_L0     REAL,
    bid_depth_L1     REAL,
    bid_depth_L2     REAL,
    bid_depth_L3     REAL,
    bid_depth_L4     REAL,

    ask_depth_L0     REAL,
    ask_depth_L1     REAL,
    ask_depth_L2     REAL,
    ask_depth_L3     REAL,
    ask_depth_L4     REAL
)
"""

CREATE_MO_ORDERS_TABLE = """
CREATE TABLE IF NOT EXISTS mo_orders (
    day              TEXT,
    first_time_ns    INTEGER,
    side             TEXT,
    cls_method       TEXT,
    mo_volume        REAL,
    n_fills          INTEGER,
    min_price        REAL,
    max_price        REAL,
    best_bid         REAL,
    best_ask         REAL,
    ticks_walked     INTEGER,
    ratio_L0         REAL,
    microprice       REAL,

    opp_depth_L0     REAL,
    opp_depth_L1     REAL,
    opp_depth_L2     REAL,
    opp_depth_L3     REAL,
    opp_depth_L4     REAL,
    opp_depth_L5     REAL,
    opp_depth_L6     REAL,
    opp_depth_L7     REAL,
    opp_depth_L8     REAL,
    opp_depth_L9     REAL,

    bid_depth_L0     REAL,
    bid_depth_L1     REAL,
    bid_depth_L2     REAL,
    bid_depth_L3     REAL,
    bid_depth_L4     REAL,

    ask_depth_L0     REAL,
    ask_depth_L1     REAL,
    ask_depth_L2     REAL,
    ask_depth_L3     REAL,
    ask_depth_L4     REAL
)
"""

# ── Simulation DB (from Simulate) ───────────────────────────────────────

SIM_CREATE_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    timestamp        REAL,
    event_type       TEXT,
    order_id         INTEGER,
    side             INTEGER,
    order_price      REAL,
    best_bid         REAL,
    best_ask         REAL,
    best_same_side   REAL,
    best_bid_size    REAL,
    best_ask_size    REAL,
    total_bid_depth  REAL,
    total_ask_depth  REAL,
    mid_price        REAL,
    ticks_from_mid   INTEGER,
    spread           REAL,
    spread_ticks     INTEGER,
    imbalance        REAL,
    ticks_from_best  INTEGER,
    queue_ahead      REAL,
    volume           REAL,
    delta0           REAL,
    delta_t          REAL,
    y_ratio          REAL,
    dt_prev_event    REAL,
    n_total          INTEGER,
    is_cancel        INTEGER,
    microprice       REAL,
    n_bid            INTEGER,
    n_ask            INTEGER,
    dp_mid           REAL,
    bid_depth_L0     REAL,
    bid_depth_L1     REAL,
    bid_depth_L2     REAL,
    bid_depth_L3     REAL,
    bid_depth_L4     REAL,
    ask_depth_L0     REAL,
    ask_depth_L1     REAL,
    ask_depth_L2     REAL,
    ask_depth_L3     REAL,
    ask_depth_L4     REAL
)"""

SIM_CREATE_FILLS = """
CREATE TABLE IF NOT EXISTS fills (
    timestamp        REAL,
    volume           REAL,
    price            REAL,
    side             TEXT,
    best_bid         REAL,
    best_ask         REAL,
    ticks_from_bbo   INTEGER,
    microprice       REAL,
    opp_depth_L0     REAL,
    opp_depth_L1     REAL,
    opp_depth_L2     REAL,
    opp_depth_L3     REAL,
    opp_depth_L4     REAL,
    opp_depth_L5     REAL,
    opp_depth_L6     REAL,
    opp_depth_L7     REAL,
    opp_depth_L8     REAL,
    opp_depth_L9     REAL,
    bid_depth_L0     REAL,
    bid_depth_L1     REAL,
    bid_depth_L2     REAL,
    bid_depth_L3     REAL,
    bid_depth_L4     REAL,
    ask_depth_L0     REAL,
    ask_depth_L1     REAL,
    ask_depth_L2     REAL,
    ask_depth_L3     REAL,
    ask_depth_L4     REAL
)"""

SIM_CREATE_MO_ORDERS = """
CREATE TABLE IF NOT EXISTS mo_orders (
    timestamp        REAL,
    side             TEXT,
    mo_volume        REAL,
    n_fills          INTEGER,
    min_price        REAL,
    max_price        REAL,
    best_bid         REAL,
    best_ask         REAL,
    ticks_walked     INTEGER,
    ratio_L0         REAL,
    microprice       REAL,
    opp_depth_L0     REAL,
    opp_depth_L1     REAL,
    opp_depth_L2     REAL,
    opp_depth_L3     REAL,
    opp_depth_L4     REAL,
    opp_depth_L5     REAL,
    opp_depth_L6     REAL,
    opp_depth_L7     REAL,
    opp_depth_L8     REAL,
    opp_depth_L9     REAL,
    bid_depth_L0     REAL,
    bid_depth_L1     REAL,
    bid_depth_L2     REAL,
    bid_depth_L3     REAL,
    bid_depth_L4     REAL,
    ask_depth_L0     REAL,
    ask_depth_L1     REAL,
    ask_depth_L2     REAL,
    ask_depth_L3     REAL,
    ask_depth_L4     REAL
)"""

SIM_CREATE_BBO = """
CREATE TABLE IF NOT EXISTS bbo (
    timestamp        REAL,
    best_bid         REAL,
    best_ask         REAL,
    mid_price        REAL
)"""

SIM_CREATE_INTENSITIES = """
CREATE TABLE IF NOT EXISTS intensities (
    timestamp        REAL,
    mo_bid           REAL,
    mo_ask           REAL,
    lo_bid           REAL,
    lo_ask           REAL,
    cxl_bid          REAL,
    cxl_ask          REAL
)"""

# ── Prepared-statement helpers (simulation) ──────────────────────────────

SIM_N_ORDER_COLS = 40
SIM_INSERT_ORDER = (
    "INSERT INTO orders VALUES (" + ",".join(["?"] * SIM_N_ORDER_COLS) + ")"
)

SIM_N_FILL_COLS = 28
SIM_INSERT_FILL = (
    "INSERT INTO fills VALUES (" + ",".join(["?"] * SIM_N_FILL_COLS) + ")"
)

SIM_N_MO_COLS = 31
SIM_INSERT_MO = (
    "INSERT INTO mo_orders VALUES (" + ",".join(["?"] * SIM_N_MO_COLS) + ")"
)

SIM_INSERT_BBO = "INSERT INTO bbo VALUES (?,?,?,?)"

SIM_INSERT_INTENSITY = "INSERT INTO intensities VALUES (?,?,?,?,?,?,?)"

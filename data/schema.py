"""
DDL strings for empirical (WSE) and simulation SQLite databases.

Empirical tables match ``research_core.classes.extract``.
Simulation tables match ``research_core.classes.simulate`` (event DB written by ``Simulate.run``).
"""

# ── Empirical order-flow DB ───────────────────────────────────────────────
#
# Side conventions (written by research_core.classes.extract, read by
# research_core.classes.helpers):
#   * orders.side                 INTEGER: 1 = bid side, 2 = ask side
#                                           (LO/CXL events).
#   * fills.side, mo_orders.side  TEXT: 'buy' / 'sell', the Lee-Ready
#                                           trade direction.
# Calibration marks map onto these as: MO_bid is a buy market order ('buy'),
# MO_ask is a sell market order ('sell'); LO_bid / CXL_bid use side = 1, and
# LO_ask / CXL_ask use side = 2.

ORDERS_DEPTH_LEVELS = 40

_ORDERS_BASE_COLS = """\
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
    dp_mid           REAL"""

_ORDERS_N_BASE_COLS = 31


def _build_create_orders_ddl(n_levels: int) -> str:
    bid_cols = ",\n".join(f"    bid_depth_L{i:<3d}  REAL" for i in range(n_levels))
    ask_cols = ",\n".join(f"    ask_depth_L{i:<3d}  REAL" for i in range(n_levels))
    return (
        "CREATE TABLE IF NOT EXISTS orders (\n"
        + _ORDERS_BASE_COLS + ",\n\n"
        + bid_cols + ",\n\n"
        + ask_cols + "\n)"
    )


CREATE_ORDERS_TABLE = _build_create_orders_ddl(ORDERS_DEPTH_LEVELS)

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

# Same-side book depth levels recorded per event in the orders table.  This
# sets the deepest phantom that can be labelled: max_delta = N * tick.  The
# empirical extraction uses 40 levels (=> 2.0 PLN at tick 0.05); match it so
# the simulation labels cover the same delta grid.
SIM_ORDER_DEPTH_LEVELS = 40

_SIM_ORDERS_SCALAR_DDL = """    timestamp        REAL,
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
    dp_mid           REAL"""

_SIM_ORDERS_N_SCALAR_COLS = 30  # number of columns in _SIM_ORDERS_SCALAR_DDL


def _sim_create_orders_ddl(n_levels: int) -> str:
    bid = ",\n".join(f"    bid_depth_L{i:<3d}  REAL" for i in range(n_levels))
    ask = ",\n".join(f"    ask_depth_L{i:<3d}  REAL" for i in range(n_levels))
    return (
        "\nCREATE TABLE IF NOT EXISTS orders (\n"
        + _SIM_ORDERS_SCALAR_DDL + ",\n"
        + bid + ",\n"
        + ask + "\n)"
    )


SIM_CREATE_ORDERS = _sim_create_orders_ddl(SIM_ORDER_DEPTH_LEVELS)

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

SIM_N_ORDER_COLS = _SIM_ORDERS_N_SCALAR_COLS + 2 * SIM_ORDER_DEPTH_LEVELS
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

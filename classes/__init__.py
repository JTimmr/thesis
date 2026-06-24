"""Public API for domain logic."""

from .analyse import AnalyseMarket
from .backtest import MMBacktester, SweepResult
from .calibrate import (
    HawkesCalibration,
    create_average_time_transformer,
    get_average_seasonality_shape,
    plot_all_seasonality_patterns,
)
from . import helpers
from .extract import (
    extract_events_for_day,
    infer_trade_side,
    list_day_keys_hdf,
    load_orders_day,
    load_trades_day,
    run_full_extraction,
    summarize_side_stats,
)
from .hawkes_filter import (
    DEFAULT_LABELS as HAWKES_LABELS,
    HawkesFilter,
    classify_event as classify_hawkes_event,
    classify_mo as classify_hawkes_mo,
)
from .market_maker import (
    AvellanedaStoikovMM,
    CompactMarketMaker,
    ErgodicMM,
    NumericalErgodicMM,
    SimpleMarketMaker,
)
from .orderbook import HeapOrderBook
from .phantom_labels import (
    PHANTOM_PER_T_FEATURES,
    PhantomLabelConfig,
    PhantomLabeller,
    realized_vol_time_grid,
    write_day_parquet,
    write_feature_schema,
    write_manifest,
)
from .simulate import Simulate
from .mo_sim_calibrate import (
    DEFAULT_MO_SELF_SCALE,
    LAYER_A_TOL_TICKS,
    LAYER_C_TOL_TICKS,
    calibrate_and_validate_mo_sim,
    calibrate_mo_impact,
    mo_cal_tag,
    run_layer_c_comparison,
)
from .simulate_fast import SimulateFast
from .simulate_uncond_mo import SimulateUncondMO
from .sim_stylized_metrics import score_quintet_db, score_quintet_manifest
from .helpers import plot_mm_result_compact

load_day_events_from_sqlite = helpers.load_day_events_from_sqlite
load_events_from_sqlite_bulk = helpers.load_events_from_sqlite_bulk
list_day_keys_from_sqlite = helpers.list_day_keys_from_sqlite
list_day_keys = helpers.list_day_keys
compute_end_times = helpers.compute_end_times
estimate_seasonality_profiles = helpers.estimate_seasonality_profiles
project_root = helpers.project_root
data_dir = helpers.data_dir
resolve_data_path = helpers.resolve_data_path

__all__ = [
    "AnalyseMarket",
    "AvellanedaStoikovMM",
    "CompactMarketMaker",
    "ErgodicMM",
    "HAWKES_LABELS",
    "HawkesCalibration",
    "HawkesFilter",
    "HeapOrderBook",
    "DEFAULT_MO_SELF_SCALE",
    "LAYER_A_TOL_TICKS",
    "LAYER_C_TOL_TICKS",
    "MMBacktester",
    "NumericalErgodicMM",
    "PHANTOM_PER_T_FEATURES",
    "PhantomLabelConfig",
    "PhantomLabeller",
    "realized_vol_time_grid",
    "SimpleMarketMaker",
    "write_day_parquet",
    "write_feature_schema",
    "write_manifest",
    "Simulate",
    "SimulateFast",
    "SimulateUncondMO",
    "SweepResult",
    "calibrate_and_validate_mo_sim",
    "calibrate_mo_impact",
    "classify_hawkes_event",
    "classify_hawkes_mo",
    "score_quintet_db",
    "score_quintet_manifest",
    "compute_end_times",
    "create_average_time_transformer",
    "data_dir",
    "estimate_seasonality_profiles",
    "extract_events_for_day",
    "get_average_seasonality_shape",
    "helpers",
    "infer_trade_side",
    "list_day_keys",
    "mo_cal_tag",
    "list_day_keys_from_sqlite",
    "list_day_keys_hdf",
    "load_day_events_from_sqlite",
    "load_events_from_sqlite_bulk",
    "load_orders_day",
    "load_trades_day",
    "plot_all_seasonality_patterns",
    "plot_mm_result_compact",
    "resolve_data_path",
    "run_full_extraction",
    "summarize_side_stats",
    "run_layer_c_comparison",
    "project_root",
]

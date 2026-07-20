"""Public API for domain logic."""

from . import helpers
from .extract import (
    extract_events_for_day,
    infer_trade_side,
    list_day_keys_hdf,
    load_extraction_stats,
    load_orders_day,
    load_trades_day,
    run_full_extraction,
    summarize_side_stats,
)
from .orderbook import HeapOrderBook
from .analyse import AnalyseMarket
from .helpers import plot_mm_result_compact
from .simulate import Simulate

load_day_events_from_sqlite = helpers.load_day_events_from_sqlite
load_events_from_sqlite_bulk = helpers.load_events_from_sqlite_bulk
list_day_keys_from_sqlite = helpers.list_day_keys_from_sqlite
list_day_keys = helpers.list_day_keys
compute_end_times = helpers.compute_end_times
estimate_seasonality_profiles = helpers.estimate_seasonality_profiles
project_root = helpers.project_root
data_dir = helpers.data_dir
resolve_data_path = helpers.resolve_data_path

_phantom_exports = {
    "PhantomLabelConfig",
    "PhantomLabeller",
    "write_day_parquet",
    "write_feature_schema",
    "write_manifest",
}
_calibrate_exports = {
    "HawkesCalibration",
    "get_average_seasonality_shape",
    "plot_all_seasonality_patterns",
}


def __getattr__(name):
    if name in _phantom_exports:
        from . import phantom_labels

        value = getattr(phantom_labels, name)
        globals()[name] = value
        return value
    if name in _calibrate_exports:
        from . import calibrate

        value = getattr(calibrate, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnalyseMarket",
    "HawkesCalibration",
    "HeapOrderBook",
    "compute_end_times",
    "data_dir",
    "estimate_seasonality_profiles",
    "extract_events_for_day",
    "get_average_seasonality_shape",
    "helpers",
    "infer_trade_side",
    "list_day_keys",
    "list_day_keys_from_sqlite",
    "list_day_keys_hdf",
    "load_extraction_stats",
    "load_day_events_from_sqlite",
    "load_events_from_sqlite_bulk",
    "load_orders_day",
    "load_trades_day",
    "PhantomLabelConfig",
    "PhantomLabeller",
    "plot_all_seasonality_patterns",
    "plot_mm_result_compact",
    "resolve_data_path",
    "run_full_extraction",
    "Simulate",
    "summarize_side_stats",
    "write_day_parquet",
    "write_feature_schema",
    "write_manifest",
    "project_root",
]

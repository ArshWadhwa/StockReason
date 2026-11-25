"""StockReason Evaluation Package."""

from src.evaluation.metrics import compute_forecasting_metrics
from src.evaluation.comparison import build_model_comparison_table
from src.evaluation.bootstrap import (
    run_paired_date_block_bootstrap,
    compute_date_block_bootstrap_sensitivity,
)
from src.evaluation.visualization import plot_return_distribution, plot_error_distribution

__all__ = [
    "compute_forecasting_metrics",
    "build_model_comparison_table",
    "run_paired_date_block_bootstrap",
    "compute_date_block_bootstrap_sensitivity",
    "plot_return_distribution",
    "plot_error_distribution",
]

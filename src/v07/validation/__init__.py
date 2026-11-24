"""Validation subpackage for StockReason v0.7.

Provides:
- PurgedWalkForward: Standard expanding walk-forward splitter with purging and embargo.
- CombinatorialPurgedCV: Optional/advanced combinatorial purged cross-validation.
- assert_no_leakage: Verification function guarding against silent data leakage.
- LeakageError: Exception raised on any leakage detection.
- stitch_backtest_paths: Utility to reconstruct historical backtest paths from CPCV.
"""

try:
    from src.v07.validation.cpcv import CombinatorialPurgedCV, stitch_backtest_paths
    from src.v07.validation.leakage_check import LeakageError, assert_no_leakage
    from src.v07.validation.purged_walk_forward import PurgedWalkForward
except ImportError:
    from v07.validation.cpcv import CombinatorialPurgedCV, stitch_backtest_paths
    from v07.validation.leakage_check import LeakageError, assert_no_leakage
    from v07.validation.purged_walk_forward import PurgedWalkForward

__all__ = [
    "PurgedWalkForward",
    "CombinatorialPurgedCV",
    "assert_no_leakage",
    "LeakageError",
    "stitch_backtest_paths",
]

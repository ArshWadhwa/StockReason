"""
StockReason v0.7 — Signals & Portfolio Module
"""

from src.v07.signals.engine import (
    SignalEngineV2,
    SignalEngineConfig,
    SignalResult,
    HeuristicSignalEngine,
)
from src.v07.signals.portfolio import (
    PortfolioSizer,
    PortfolioConfig,
    PortfolioAllocationResult,
    PositionAllocation,
    DEFAULT_SECTOR_MAP,
)
from src.v07.signals.sensitivity import (
    SensitivityAnalyzer,
    SensitivityReport,
    PerturbationResult,
)
from src.v07.signals.brackets import (
    BracketManager,
    BracketConfig,
    TradeBracket,
    SimulatedTradeResult,
    BracketComparisonReport,
)
from src.v07.signals.attribution import (
    FeatureAttributionEngine,
    FeatureAttributionResult,
    FeatureAttributionItem,
)
from src.v07.signals.explanations import (
    SignalExplanation,
    SignalExplanationGenerator,
    InvalidationCondition,
)

__all__ = [
    "SignalEngineV2",
    "SignalEngineConfig",
    "SignalResult",
    "HeuristicSignalEngine",
    "PortfolioSizer",
    "PortfolioConfig",
    "PortfolioAllocationResult",
    "PositionAllocation",
    "DEFAULT_SECTOR_MAP",
    "SensitivityAnalyzer",
    "SensitivityReport",
    "PerturbationResult",
    "BracketManager",
    "BracketConfig",
    "TradeBracket",
    "SimulatedTradeResult",
    "BracketComparisonReport",
    "FeatureAttributionEngine",
    "FeatureAttributionResult",
    "FeatureAttributionItem",
    "SignalExplanation",
    "SignalExplanationGenerator",
    "InvalidationCondition",
]

"""
StockReason v0.7 — WS9 Comprehensive Test Suite
================================================
Tests for Signal Engine v2, Portfolio Rules, Sensitivity Analysis,
ATR Brackets, Explanations, and Feature Attribution.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb

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
    DEFAULT_SECTOR_MAP,
)
from src.v07.signals.sensitivity import (
    SensitivityAnalyzer,
    SensitivityReport,
)
from src.v07.signals.brackets import (
    BracketManager,
    BracketConfig,
    TradeBracket,
)
from src.v07.signals.attribution import (
    FeatureAttributionEngine,
    FeatureAttributionResult,
)
from src.v07.signals.explanations import (
    SignalExplanation,
    SignalExplanationGenerator,
)


# =============================================================================
# 1. Signal Engine v2 Tests
# =============================================================================

def test_no_buy_when_calibrated_prob_below_floor():
    """Mandate: No BUY signal is generated when calibrated prob < floor (0.52)."""
    engine = SignalEngineV2(SignalEngineConfig(min_calibrated_prob_buy=0.52))

    # Even with huge expected return and top rank, sub-floor prob MUST veto BUY
    res = engine.evaluate_single(
        ticker="RELIANCE.NS",
        calibrated_probability=0.49,     # Below 0.52 floor
        expected_return=0.10,            # Huge +10% expected return
        cross_sectional_rank=0.95,       # Top 5% in universe
        model_agreement=True,
    )
    assert res.signal != "BUY", f"Expected signal not to be BUY, got {res.signal}"
    assert res.signal == "HOLD"
    assert res.is_prob_floor_binding is True
    assert any("below BUY floor" in r for r in res.reasons)


def test_signal_engine_buy_case():
    """Verify BUY signal when all conviction, probability, rank, and cost conditions pass."""
    engine = SignalEngineV2(SignalEngineConfig(min_calibrated_prob_buy=0.52, cost_bps=20.0))

    res = engine.evaluate_single(
        ticker="INFY.NS",
        calibrated_probability=0.62,
        expected_return=0.035,           # 3.5% gross - 20 bps = 3.3% net (> 1.5% hurdle)
        cross_sectional_rank=0.85,       # Top 15%
        model_agreement=True,
        regime="low_vol_trending",
    )
    assert res.signal == "BUY"
    assert res.expected_net_return == pytest.approx(0.035 - 0.002, abs=1e-5)
    assert res.signal_score >= 65.0
    assert res.is_prob_floor_binding is False


def test_signal_engine_cost_hurdle():
    """Verify that transaction costs are properly deducted from expected return."""
    # 50 bps cost = 0.50%
    engine = SignalEngineV2(SignalEngineConfig(cost_bps=50.0, min_net_return_buy=0.015))

    # Gross return 1.8% - 0.5% = 1.3% net, which fails 1.5% hurdle -> HOLD
    res = engine.evaluate_single(
        ticker="TCS.NS",
        calibrated_probability=0.60,
        expected_return=0.018,
        cross_sectional_rank=0.80,
        model_agreement=True,
    )
    assert res.signal == "HOLD"
    assert res.expected_net_return == pytest.approx(0.018 - 0.005, abs=1e-5)


def test_signal_engine_model_disagreement_downgrade():
    """Verify that model disagreement prevents BUY signal."""
    engine = SignalEngineV2(SignalEngineConfig(allow_buy_on_disagreement=False))

    res = engine.evaluate_single(
        ticker="DISPUTE.NS",
        calibrated_probability=0.65,
        expected_return=0.04,
        cross_sectional_rank=0.88,
        model_agreement=False,           # Disagreement!
    )
    assert res.signal == "HOLD"
    assert any("disagreement" in r.lower() for r in res.reasons)


# =============================================================================
# 2. Portfolio Rules & Position Sizing Tests
# =============================================================================

def test_portfolio_sector_cap_and_count_enforced():
    """
    Mandate:
    1. Max 30% per sector (sum of weights in same sector <= 0.30)
    2. Max 3 BUYs from same sector (at most 3 BUY positions per sector)
    3. Single-name cap (max 10% per ticker)
    """
    sizer = PortfolioSizer(PortfolioConfig(
        max_sector_weight=0.30,
        max_buys_per_sector=3,
        max_single_weight=0.10,
    ))

    # Provide 6 banking stocks all with BUY signals
    banking_tickers = [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS"
    ]
    signals = [
        {
            "ticker": t,
            "signal": "BUY",
            "calibrated_probability": 0.60,
            "expected_net_return": 0.03,
            "signal_score": 75.0 + i,   # Varying scores
            "volatility_20d": 0.22,
        }
        for i, t in enumerate(banking_tickers)
    ]

    sector_map = {t: "Banking" for t in banking_tickers}
    res = sizer.allocate(signals, sector_map=sector_map)

    # 1. Sector count check: exactly 3 positions allowed
    assert res.sector_counts.get("Banking", 0) == 3
    assert len(res.weights) == 3

    # 2. Sector weight check: total Banking weight <= 0.30
    banking_tot = res.sector_weights.get("Banking", 0.0)
    assert banking_tot <= 0.30 + 1e-5
    assert res.is_sector_cap_satisfied is True
    assert res.is_sector_count_satisfied is True

    # 3. Single-name cap check: each stock <= 0.10
    for t, w in res.weights.items():
        assert w <= 0.10 + 1e-5
    assert res.is_single_name_cap_satisfied is True


def test_portfolio_weights_sum_le_one():
    """Mandate: sum of all weights <= 1.0 (remainder in cash)."""
    sizer = PortfolioSizer()

    # 15 tickers across different sectors
    test_signals = [
        {
            "ticker": f"STOCK_{i}.NS",
            "signal": "BUY",
            "calibrated_probability": 0.60,
            "expected_net_return": 0.025,
            "signal_score": 70.0,
            "volatility_20d": 0.20 + (i * 0.01),
        }
        for i in range(15)
    ]
    # Spread evenly across sectors
    sec_map = {f"STOCK_{i}.NS": f"Sector_{i % 5}" for i in range(15)}

    res = sizer.allocate(test_signals, sector_map=sec_map)

    assert res.total_equity_weight <= 1.0 + 1e-5
    assert res.cash_weight >= -1e-6
    assert (res.total_equity_weight + res.cash_weight) == pytest.approx(1.0, abs=1e-5)
    assert res.is_weight_sum_satisfied is True


def test_volatility_scaled_weights_and_targeting():
    """Verify inverse volatility weighting and volatility targeting scalar."""
    sizer = PortfolioSizer(PortfolioConfig(
        target_annualized_vol=0.15,
        use_volatility_targeting=True,
    ))

    # Low vol vs high vol stock in different sectors
    signals = [
        {"ticker": "LOW_VOL.NS", "signal": "BUY", "volatility_20d": 0.12, "signal_score": 70.0, "calibrated_probability": 0.6, "expected_net_return": 0.02},
        {"ticker": "HIGH_VOL.NS", "signal": "BUY", "volatility_20d": 0.36, "signal_score": 70.0, "calibrated_probability": 0.6, "expected_net_return": 0.02},
    ]
    sec_map = {"LOW_VOL.NS": "SecA", "HIGH_VOL.NS": "SecB"}

    res = sizer.allocate(signals, sector_map=sec_map)
    # Low vol asset should receive higher or equal raw weight than high vol asset
    pos_map = {p.ticker: p for p in res.positions}
    assert pos_map["LOW_VOL.NS"].raw_weight > pos_map["HIGH_VOL.NS"].raw_weight


def test_fractional_kelly_sizer():
    """Verify Fractional-Kelly sizer, off-by-default flag, and docstring documentation."""
    sizer = PortfolioSizer(PortfolioConfig(use_kelly=False))
    assert sizer.config.use_kelly is False

    # Inspect docstring for calibrated probability requirement
    doc = PortfolioSizer.compute_fractional_kelly_weight.__doc__
    assert "CALIBRATED PROBABILITIES" in doc or "calibrated" in doc.lower()

    # When prob <= 0.50, Kelly weight MUST be 0
    w_zero = sizer.compute_fractional_kelly_weight(calibrated_probability=0.48, expected_net_return=0.02)
    assert w_zero == 0.0

    # When prob > 0.50, Fractional Kelly produces positive size
    w_pos = sizer.compute_fractional_kelly_weight(calibrated_probability=0.65, expected_net_return=0.04)
    assert w_pos > 0.0
    assert w_pos <= 1.0


# =============================================================================
# 3. Sensitivity Analysis Tests
# =============================================================================

def test_sensitivity_runner_runs_end_to_end():
    """Mandate: Sensitivity runner runs end-to-end and outputs fragility analysis."""
    # Synthetic evaluation data
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["RELIANCE.NS"] * n,
        "calibrated_probability": np.random.uniform(0.48, 0.65, size=n),
        "expected_return": np.random.normal(0.02, 0.015, size=n),
        "realized_return": np.random.normal(0.015, 0.03, size=n),
        "cross_sectional_rank": np.random.uniform(0.50, 0.90, size=n),
    })

    analyzer = SensitivityAnalyzer()
    cfg = SignalEngineConfig(min_calibrated_prob_buy=0.52, min_net_return_buy=0.015)
    report = analyzer.run_sensitivity(cfg, df, perturbation_delta=0.20)

    assert isinstance(report, SensitivityReport)
    assert len(report.perturbations) > 0
    assert isinstance(report.is_fragile, bool)
    assert 0.0 <= report.fragility_score <= 1.0
    d = report.to_dict()
    assert "base_sharpe" in d
    assert "perturbation_details" in d


# =============================================================================
# 4. Explanations & Invalidation Tests
# =============================================================================

def test_explanation_object_origin_date_and_model_version():
    """Mandate: Explanation object contains origin date and model version."""
    gen = SignalExplanationGenerator(
        model_name="StockReason Ensemble v2",
        model_version="v0.7.0-dev",
    )

    exp = gen.explain(
        ticker="RELIANCE.NS",
        signal="BUY",
        forecast_origin_date="2024-03-15",
        current_price=2950.0,
        expected_net_return=0.032,
        calibrated_probability=0.64,
        horizon="21D",
        prices_context={"open": 2940.0, "high": 2965.0, "low": 2930.0, "sma_50": 2890.0},
    )

    assert exp.forecast_origin_date == "2024-03-15"
    assert exp.model_version == "v0.7.0-dev"
    assert exp.model_name == "StockReason Ensemble v2"
    assert "close" in exp.prices_used
    assert exp.prices_used["close"] == 2950.0
    assert exp.oos_rank_ic > 0
    assert exp.oos_directional_accuracy > 0.50
    assert len(exp.invalidation_conditions) >= 3

    # Check serialization
    d = exp.to_dict()
    assert d["forecast_origin_date"] == "2024-03-15"
    assert d["model_version"] == "v0.7.0-dev"
    md = exp.to_markdown()
    assert "StockReason Signal Audit" in md
    assert "v0.7.0-dev" in md


# =============================================================================
# 5. ATR Brackets & Fill Comparison Tests
# =============================================================================

def test_brackets_creation_and_fill_simulation():
    """Verify ATR bracket creation and realistic limit order fill simulation."""
    mgr = BracketManager(BracketConfig(entry_atr_mult=0.5, stop_atr_mult=2.0, target_atr_mult=3.0))
    bracket = mgr.create_brackets("TCS.NS", current_price=4000.0, atr_14=50.0, use_pullback=True)

    # Entry = 4000 - 25 = 3975
    # Stop = 3975 - 100 = 3875
    # Target = 3975 + 150 = 4125
    assert bracket.entry_price == 3975.0
    assert bracket.stop_loss == 3875.0
    assert bracket.take_profit == 4125.0
    assert bracket.risk_reward_ratio == pytest.approx(1.5, abs=0.01)

    # Forward bars: bar 0 dips to 3970 (limit filled!), bar 1 hits take profit 4130
    bars = pd.DataFrame([
        {"date": "2024-03-16", "open": 3990.0, "high": 4010.0, "low": 3970.0, "close": 3985.0},
        {"date": "2024-03-17", "open": 3990.0, "high": 4140.0, "low": 3980.0, "close": 4130.0},
    ])
    trade = mgr.simulate_trade_with_brackets(bracket, bars)
    assert trade.is_filled is True
    assert trade.exit_reason == "TAKE_PROFIT"
    assert trade.net_return > 0


# =============================================================================
# 6. Feature Attribution Naming Tests
# =============================================================================

def test_feature_attribution_naming_and_methods():
    """Mandate: Rename gradient x input to feature_attribution with method: 'gradient_x_input'."""
    # 1. PyTorch model test
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

        def forward(self, x):
            return self.linear(x)

    torch.manual_seed(42)
    py_model = SimpleLinear()
    x = torch.randn(1, 4)
    feat_names = ["rsi_14", "volatility_20d", "macd_line_12_26", "sentiment_mean"]

    attr_res = FeatureAttributionEngine.compute_gradient_x_input(
        model=py_model,
        x_tensor=x,
        feature_names=feat_names,
        ticker="RELIANCE.NS",
        target_idx=2,
    )
    assert attr_res.method == "gradient_x_input"
    assert len(attr_res.attributions) == 4
    for a in attr_res.attributions:
        assert a.method == "gradient_x_input"

    # 2. LightGBM Tree SHAP test
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    X_train = np.random.randn(50, 4)
    y_train = np.random.choice([0, 1], size=50)
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {"objective": "binary", "verbose": -1, "min_data_in_leaf": 5, "num_threads": 1}
    booster = lgb.train(params, train_data, num_boost_round=5)

    tree_res = FeatureAttributionEngine.compute_tree_shap(
        booster=booster,
        features_row=X_train[0],
        feature_names=feat_names,
        ticker="RELIANCE.NS",
    )
    assert tree_res.method == "tree_shap"
    assert len(tree_res.attributions) == 4
    for a in tree_res.attributions:
        assert a.method == "tree_shap"


# =============================================================================
# 7. Comparison: Signal Engine v2 vs Legacy Heuristic Engine
# =============================================================================

def test_old_vs_new_engine_cost_awareness():
    """Compares old heuristic engine with SignalEngineV2 net of costs."""
    old_engine = HeuristicSignalEngine(buy_return_threshold=0.025)
    new_engine = SignalEngineV2(SignalEngineConfig(min_calibrated_prob_buy=0.52, cost_bps=20.0))

    # Old engine only looks at gross return and uncalibrated confidence
    old_res = old_engine.evaluate(
        ticker="MARGINAL.NS",
        current_price=1000.0,
        expected_return_1M=0.026,        # Barely clears 2.5%
        confidence=0.70,
        sentiment_score=0.10,
    )
    # New engine checks net return after 20 bps cost and cross-sectional rank
    new_res = new_engine.evaluate_single(
        ticker="MARGINAL.NS",
        calibrated_probability=0.50,     # Below 0.52 floor!
        expected_return=0.026,
        cross_sectional_rank=0.55,       # Below 0.65 rank cutoff!
    )

    # Old engine says BUY because it ignores costs, rank, and prob calibration
    assert old_res["signal"] == "BUY"
    # New engine avoids false positive and correctly holds
    assert new_res.signal == "HOLD"
    assert new_res.is_prob_floor_binding is True

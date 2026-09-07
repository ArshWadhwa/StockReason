import pytest
from src.m4_dashboard.backend.signal_engine import RuleBasedSignalEngine

@pytest.fixture
def engine():
    return RuleBasedSignalEngine()

def test_signal_engine_buy_case(engine):
    # High return, high confidence, positive sentiment, models agree -> BUY
    res = engine.evaluate(
        ticker="RELIANCE.NS",
        current_price=3000.0,
        expected_return_1M=0.045,
        confidence=0.85,
        sentiment_score=0.45,
        regime="low_vol_trending",
        lstm_return_1M=0.045,
        sma_50=2900.0,
        rsi_14=60.0
    )
    assert res["signal"] == "BUY"
    assert res["signal_score"] >= 60.0
    assert res["ensemble_agreement"] is True
    assert len(res["reasoning"]) >= 3

def test_signal_engine_avoid_on_negative_return(engine):
    # Negative return -> AVOID
    res = engine.evaluate(
        ticker="WEAK.NS",
        current_price=500.0,
        expected_return_1M=-0.03,
        confidence=0.75,
        sentiment_score=-0.1,
        regime="high_vol_choppy"
    )
    assert res["signal"] == "AVOID"
    assert res["signal_score"] < 40.0

def test_signal_engine_low_confidence_downgrade(engine):
    # Positive return, but confidence below minimum threshold -> HOLD
    res = engine.evaluate(
        ticker="LOWCONF.NS",
        current_price=1000.0,
        expected_return_1M=0.03,
        confidence=0.50, # below 0.65 threshold
        sentiment_score=0.1,
        regime="low_vol_trending",
    )
    assert res["signal"] in ["HOLD", "AVOID"]

def test_signal_engine_neutral_case(engine):
    # Modest return -> HOLD
    res = engine.evaluate(
        ticker="TCS.NS",
        current_price=4100.0,
        expected_return_1M=0.012,
        confidence=0.68,
        sentiment_score=0.05,
        regime="mean_reverting"
    )
    assert res["signal"] == "HOLD"

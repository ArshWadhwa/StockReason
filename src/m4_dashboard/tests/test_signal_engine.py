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
        secondary_return_1M=0.042,  # Real ensemble: close agreement
        sma_50=2900.0,
        rsi_14=60.0,
        atr_14=45.0,
    )
    assert res["signal"] == "BUY"
    assert res["signal_score"] >= 60.0
    assert res["ensemble_agreement"] is True
    assert len(res["reasoning"]) >= 3
    # Verify entry/exit brackets are present
    assert res["entry_exit"] is not None
    assert res["entry_exit"]["entry_price"] > 0
    assert res["entry_exit"]["stop_loss"] < res["entry_exit"]["entry_price"]
    assert res["entry_exit"]["take_profit"] > res["entry_exit"]["entry_price"]

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

def test_signal_engine_ensemble_disagreement_downgrade(engine):
    # Positive return from primary LSTM, but large disagreement from secondary -> conviction lowered
    res = engine.evaluate(
        ticker="DISPUTE.NS",
        current_price=1000.0,
        expected_return_1M=0.03,
        confidence=0.70,
        sentiment_score=0.1,
        regime="low_vol_trending",
        lstm_return_1M=0.03,
        secondary_return_1M=-0.04,  # 7% divergence (>5% threshold)
        atr_14=15.0,
    )
    assert res["ensemble_agreement"] is False
    assert any("Disagreement" in r for r in res["reasoning"])

def test_signal_engine_low_confidence_downgrade(engine):
    # Positive return, but confidence below minimum threshold -> HOLD
    res = engine.evaluate(
        ticker="LOWCONF.NS",
        current_price=1000.0,
        expected_return_1M=0.03,
        confidence=0.50,  # below 0.65 threshold
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

def test_signal_engine_entry_exit_brackets(engine):
    # Verify entry/exit bracket computation
    res = engine.evaluate(
        ticker="TEST.NS",
        current_price=1000.0,
        expected_return_1M=0.05,
        confidence=0.80,
        sentiment_score=0.20,
        regime="low_vol_trending",
        lstm_return_1M=0.05,
        secondary_return_1M=0.048,
        atr_14=20.0,
        natr_14=2.0,
        sma_50=980.0,
        rsi_14=55.0,
    )
    assert res["signal"] == "BUY"
    ee = res["entry_exit"]
    assert ee is not None
    assert ee["bracket_type"] == "pullback_entry"
    assert ee["risk_reward_ratio"] > 0
    assert ee["atr_14"] == 20.0

def test_signal_engine_no_atr_no_brackets(engine):
    # Without ATR data, entry_exit should be None
    res = engine.evaluate(
        ticker="NOATR.NS",
        current_price=500.0,
        expected_return_1M=0.04,
        confidence=0.75,
        sentiment_score=0.15,
        regime="low_vol_trending",
    )
    assert res["entry_exit"] is None

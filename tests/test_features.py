import pandas as pd
import numpy as np
from src.m1_price_data.technical_features import generate_technical_features
from src.m1_price_data.regime_detection import detect_market_regime

def test_technical_features_generation():
    dates = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame({
        "date": dates,
        "ticker": "INFY.NS",
        "open": np.random.uniform(1400, 1500, 100),
        "high": np.random.uniform(1500, 1550, 100),
        "low": np.random.uniform(1350, 1400, 100),
        "close": np.random.uniform(1410, 1510, 100),
        "volume": np.random.randint(100000, 500000, 100)
    })
    feat_df = generate_technical_features(df)
    assert "rsi_14" in feat_df.columns, "RSI feature missing"
    assert "macd_line_12_26" in feat_df.columns, "MACD feature missing"
    assert "bb_upper_20" in feat_df.columns, "Bollinger Bands missing"
    assert "atr_14" in feat_df.columns, "ATR feature missing"
    assert len(feat_df.columns) > 100, f"Expected >100 features, got {len(feat_df.columns)}"

def test_regime_detection():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "vix_close": [12.0, 14.0, 20.0, 15.0, 13.0],
        "adx_14": [18.0, 30.0, 15.0, 28.0, 14.0],
        "natr_14": [1.2, 1.5, 3.5, 1.8, 1.1]
    })
    reg_df = detect_market_regime(df)
    assert "market_regime" in reg_df.columns
    assert reg_df.iloc[2]["market_regime"] == "VOLATILE", "High VIX should trigger VOLATILE regime"
    assert reg_df.iloc[1]["market_regime"] == "TRENDING", "High ADX with moderate VIX should trigger TRENDING"

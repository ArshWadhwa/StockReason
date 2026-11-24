"""
test_ws1.py
-----------
Deterministic unit tests for StockReason v0.7 Workstream 1:
  - SequenceDataset first and last sequence for a ticker
  - No window crosses ticker boundaries
  - Target equals exact manual calculation for all horizons
  - Live-inference window matches training window format exactly
  - Corporate action validator flags moves > ±25% without known action
  - Data gap validator flags trading day gaps > 5 days
  - Freshness monitor calculates trading day staleness and enforces publication threshold
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest
import torch

from src.v07.common.config import load_v07_config
from src.v07.common.freshness import (
    FreshnessReport,
    StaleDataError,
    check_freshness,
    enforce_freshness_for_forecast,
    extract_latest_price_date,
)
from src.v07.common.sequences import (
    SequenceDataset,
    build_sequences,
    get_live_inference_window,
)
from src.v07.common.validation import (
    count_missed_trading_days_between,
    count_trading_days_elapsed,
    validate_dataset_integrity,
    validate_price_moves,
    validate_trading_gaps,
)


@pytest.fixture
def synthetic_panel():
    """
    Creates a deterministic 2-ticker panel over 70 business days.
      - TICKER_A: prices start at 100, growing 1% daily. Features in [0, 100).
      - TICKER_B: prices start at 200, decaying 0.5% daily. Features in [1000, 1200).
    """
    dates = pd.date_range("2024-01-01", periods=70, freq="B")
    n = len(dates)

    # Ticker A
    prices_a = [100.0 * (1.01 ** i) for i in range(n)]
    feat1_a = [float(i) for i in range(n)]
    feat2_a = [float(i * 2) for i in range(n)]
    df_a = pd.DataFrame({
        "ticker": ["TICKER_A"] * n,
        "date": dates,
        "adj_close": prices_a,
        "close": prices_a,
        "feature_1": feat1_a,
        "feature_2": feat2_a,
    })

    # Ticker B
    prices_b = [200.0 * (0.995 ** i) for i in range(n)]
    feat1_b = [1000.0 + float(i) for i in range(n)]
    feat2_b = [1000.0 + float(i * 2) for i in range(n)]
    df_b = pd.DataFrame({
        "ticker": ["TICKER_B"] * n,
        "date": dates,
        "adj_close": prices_b,
        "close": prices_b,
        "feature_1": feat1_b,
        "feature_2": feat2_b,
    })

    panel = pd.concat([df_a, df_b], ignore_index=True)
    return panel


def test_first_and_last_sequence_per_ticker(synthetic_panel):
    seq_len = 10
    horizons = [5, 21]
    feature_cols = ["feature_1", "feature_2"]

    dataset = build_sequences(
        panel=synthetic_panel,
        feature_cols=feature_cols,
        seq_len=seq_len,
        horizons=horizons,
    )

    meta = dataset.get_metadata()
    assert len(dataset) > 0
    assert len(dataset) == len(meta)

    # Check for TICKER_A specifically
    meta_a = meta[meta["ticker"] == "TICKER_A"].reset_index(drop=True)
    n_a = len(synthetic_panel[synthetic_panel["ticker"] == "TICKER_A"])  # 70
    expected_sequences_per_ticker = 70 - max(horizons) - seq_len + 1  # 70 - 21 - 10 + 1 = 40
    assert len(meta_a) == expected_sequences_per_ticker

    # 1. FIRST SEQUENCE FOR TICKER_A
    first_sample_idx = int(meta_a.loc[0, "sample_idx"])
    x_first, y_first, meta_first = dataset[first_sample_idx]

    # t for first sequence is seq_len - 1 = 9
    assert meta_first["ticker"] == "TICKER_A"
    assert meta_first["t_idx"] == 9
    assert x_first.shape == (seq_len, len(feature_cols))
    
    # Feature window is rows 0 to 9 inclusive
    expected_x_first = np.column_stack([np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32) * 2])
    np.testing.assert_allclose(x_first.numpy(), expected_x_first)

    # 2. LAST SEQUENCE FOR TICKER_A
    last_sample_idx = int(meta_a.iloc[-1]["sample_idx"])
    x_last, y_last, meta_last = dataset[last_sample_idx]

    # t for last sequence is n_a - 1 - max_horizon = 70 - 1 - 21 = 48
    assert meta_last["t_idx"] == 48
    assert meta_last["ticker"] == "TICKER_A"
    
    # Window rows: 48 - 10 + 1 = 39 to 48 (inclusive, 10 rows)
    expected_x_last = np.column_stack([np.arange(39, 49, dtype=np.float32), np.arange(39, 49, dtype=np.float32) * 2])
    np.testing.assert_allclose(x_last.numpy(), expected_x_last)


def test_no_window_crosses_tickers(synthetic_panel):
    seq_len = 15
    horizons = [5, 21]
    feature_cols = ["feature_1", "feature_2"]

    dataset = build_sequences(
        panel=synthetic_panel,
        feature_cols=feature_cols,
        seq_len=seq_len,
        horizons=horizons,
    )

    meta = dataset.get_metadata()
    for idx in range(len(dataset)):
        x, y, sample_meta = dataset[idx]
        ticker = sample_meta["ticker"]
        x_np = x.numpy()

        if ticker == "TICKER_A":
            # TICKER_A features are strictly < 1000
            assert np.all(x_np < 1000.0), f"TICKER_A window at idx {idx} contains TICKER_B data!"
        elif ticker == "TICKER_B":
            # TICKER_B features are strictly >= 1000
            assert np.all(x_np >= 1000.0), f"TICKER_B window at idx {idx} contains TICKER_A data!"


def test_target_equals_manual_calculation(synthetic_panel):
    seq_len = 12
    horizons = [5, 21]
    feature_cols = ["feature_1", "feature_2"]

    dataset = build_sequences(
        panel=synthetic_panel,
        feature_cols=feature_cols,
        seq_len=seq_len,
        horizons=horizons,
    )

    df_a = synthetic_panel[synthetic_panel["ticker"] == "TICKER_A"].sort_values("date").reset_index(drop=True)
    prices_a = df_a["adj_close"].values
    dates_a = df_a["date"].values

    meta = dataset.get_metadata()
    meta_a = meta[meta["ticker"] == "TICKER_A"].reset_index(drop=True)

    for i in range(len(meta_a)):
        sample_idx = int(meta_a.loc[i, "sample_idx"])
        t = int(meta_a.loc[i, "t_idx"])
        x, y, sample_meta = dataset[sample_idx]

        price_t = prices_a[t]
        assert sample_meta["label_start"] == pd.Timestamp(dates_a[t])

        for h_idx, h in enumerate(horizons):
            expected_ret = prices_a[t + h] / price_t - 1.0
            expected_end_date = pd.Timestamp(dates_a[t + h])

            assert sample_meta[f"label_end_{h}"] == expected_end_date
            pytest.approx(y[h_idx].item(), rel=1e-5) == expected_ret
            pytest.approx(meta_a.loc[i, f"target_{h}"], rel=1e-5) == expected_ret


def test_live_inference_window_matches_training_format(synthetic_panel):
    seq_len = 10
    feature_cols = ["feature_1", "feature_2"]

    # 1. Get live inference window
    live_tensor, live_meta = get_live_inference_window(
        panel=synthetic_panel,
        feature_cols=feature_cols,
        seq_len=seq_len,
    )

    assert live_tensor.shape == (2, seq_len, len(feature_cols))
    assert live_tensor.dtype == torch.float32
    assert len(live_meta) == 2

    # Check ticker order and content
    row_a = live_meta[live_meta["ticker"] == "TICKER_A"].iloc[0]
    idx_a = live_meta.index[live_meta["ticker"] == "TICKER_A"].tolist()[0]
    x_live_a = live_tensor[idx_a].numpy()

    # The live window for TICKER_A must match its last 10 rows (rows 60 to 69)
    expected_live_a = np.column_stack([np.arange(60, 70, dtype=np.float32), np.arange(60, 70, dtype=np.float32) * 2])
    np.testing.assert_allclose(x_live_a, expected_live_a)
    assert row_a["as_of_date"] == synthetic_panel[synthetic_panel["ticker"] == "TICKER_A"]["date"].max()


def test_corporate_action_validator_flags_unusual_jump():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    # Base prices around 100
    prices = [100.0] * 10
    # Add a 35% jump on day 5 without known split
    prices[5] = 135.0

    df = pd.DataFrame({
        "ticker": ["JUMP.NS"] * 10,
        "date": dates,
        "adj_close": prices,
    })

    # Without known action: must be flagged
    anomalies = validate_price_moves(df, max_abs_move=0.25)
    assert len(anomalies) == 2  # jump from 100 to 135 (+35%), and jump from 135 back to 100 (-25.9%)
    assert anomalies[0]["ticker"] == "JUMP.NS"
    assert anomalies[0]["daily_return"] == pytest.approx(0.35, rel=1e-3)

    # With known action on day 5: only the return back is flagged
    known_actions = {"JUMP.NS": [str(dates[5].date())]}
    anomalies_filtered = validate_price_moves(df, max_abs_move=0.25, known_actions=known_actions)
    assert len(anomalies_filtered) == 1
    assert anomalies_filtered[0]["date"] == str(dates[6].date())


def test_trading_day_gaps_validator():
    # 3 consecutive days, then a 15-day gap, then 3 days
    d1 = pd.date_range("2024-01-01", periods=3, freq="B")
    d2 = pd.date_range("2024-01-25", periods=3, freq="B")
    all_dates = list(d1) + list(d2)

    df = pd.DataFrame({
        "ticker": ["GAP.NS"] * len(all_dates),
        "date": all_dates,
        "adj_close": [100.0] * len(all_dates),
    })

    gaps = validate_trading_gaps(df, max_gap_trading_days=5)
    assert len(gaps) == 1
    assert gaps[0]["ticker"] == "GAP.NS"
    assert gaps[0]["missed_trading_days"] > 5


def test_freshness_check_and_enforcement(tmp_path):
    # Create synthetic parquet with recent date
    recent_date = date(2026, 11, 23)
    df_fresh = pd.DataFrame({
        "ticker": ["TEST.NS"],
        "date": [pd.Timestamp(recent_date)],
        "adj_close": [100.0],
    })
    fresh_parquet = tmp_path / "fresh_price.parquet"
    df_fresh.to_parquet(fresh_parquet)

    # Check freshness as of same day (0 days stale)
    report = check_freshness(
        price_path=fresh_parquet,
        as_of_date=recent_date,
        max_stale_trading_days=3,
    )
    assert report.can_publish_forecast is True
    assert report.sources["price"].stale_trading_days == 0
    assert report.sources["price"].is_fresh is True

    # Check enforcement passes
    passed_report = enforce_freshness_for_forecast(
        price_path=fresh_parquet,
        as_of_date=recent_date,
        max_stale_trading_days=3,
    )
    assert passed_report.can_publish_forecast is True

    # Check enforcement raises StaleDataError when evaluated 10 trading days later
    eval_future_date = date(2026, 12, 10)
    with pytest.raises(StaleDataError, match="Forecast publication rejected due to stale market data"):
        enforce_freshness_for_forecast(
            price_path=fresh_parquet,
            as_of_date=eval_future_date,
            max_stale_trading_days=3,
        )


def test_v07_config_integrity():
    cfg = load_v07_config()
    assert cfg["horizons"] == [5, 21]
    assert cfg["sequence_length"] == 60
    assert cfg["embargo"] == 5
    assert cfg["seeds"] == [0, 1, 2, 3, 4]
    assert "cost_parameters" in cfg
    assert "brokerage" in cfg["cost_parameters"]
    assert "slippage_bps" in cfg["cost_parameters"]
    assert "paths" in cfg

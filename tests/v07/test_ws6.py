"""
Unit and integration tests for WS6 (Data expansion and scaling experiment).

Verifies:
1. Downloader handles failed tickers gracefully without unhandled exceptions.
2. Cleaner flags injected fake split (±25% jump).
3. Liquidity filter excludes illiquid synthetic ticker (< ₹5 Cr/day).
4. Calendar alignment forward-fills macro without leaking future data.
5. Universe tiers exist, are strictly nested, and document survivorship bias.
6. Zero volume runs and minimum history filters function deterministically.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.v07.data_expansion.universe import (
    get_universe,
    get_universe_details,
    get_universe_metadata,
    UNIVERSE_TIERS,
    SURVIVORSHIP_SAFE,
    BENCHMARK_INDICES,
    MACRO_TICKERS,
)
from src.v07.data_expansion.download import (
    fetch_ticker_ohlcv,
    standardize_ohlcv_dataframe,
    sanitize_ticker_for_filename,
    unsanitize_filename_to_ticker,
)
from src.v07.data_expansion.cleaner import (
    detect_split_bonus_anomalies,
    detect_zero_volume_runs,
    filter_minimum_history,
    filter_liquidity,
    align_to_nse_calendar,
    enforce_price_boundaries,
)
from src.v07.data_expansion.metadata_builder import build_universe_metadata_table
from src.v07.data_expansion.run_scaling_experiment import (
    compute_features_and_targets,
    evaluate_metrics,
)


class TestUniverseDefinitions:
    """Verifies tiered universe specification, nesting, and metadata."""

    def test_universe_tiers_exist_and_counts(self):
        u16 = get_universe("U16")
        u100 = get_universe("U100")
        u200 = get_universe("U200")
        u300 = get_universe("U300")

        assert len(u16) == 16, f"Expected 16 tickers in U16, got {len(u16)}"
        assert len(u100) == 100, f"Expected 100 tickers in U100, got {len(u100)}"
        assert len(u200) == 200, f"Expected 200 tickers in U200, got {len(u200)}"
        assert len(u300) == 300, f"Expected 300 tickers in U300, got {len(u300)}"

    def test_universe_nesting(self):
        """U16 should be subset of U100, U100 subset of U200, U200 subset of U300."""
        u16 = set(get_universe("U16"))
        u100 = set(get_universe("U100"))
        u200 = set(get_universe("U200"))
        u300 = set(get_universe("U300"))

        assert u16.issubset(u100), "U16 must be a strict subset of U100"
        assert u100.issubset(u200), "U100 must be a strict subset of U200"
        assert u200.issubset(u300), "U200 must be a strict subset of U300"

    def test_survivorship_safe_metadata(self):
        """Metadata must prominently document survivorship bias."""
        for tier in UNIVERSE_TIERS:
            meta = get_universe_metadata(tier)
            assert meta["survivorship_safe"] is False
            assert "survivorship bias" in meta["survivorship_bias_note"].lower()

            details_df = get_universe_details(tier)
            assert details_df.attrs["survivorship_safe"] is False


class TestDownloader:
    """Verifies downloader error resilience, sanitization, and standardization."""

    def test_downloader_handles_failed_ticker_gracefully(self, tmp_path):
        """Invalid ticker must not throw an unhandled exception and must return None."""
        invalid_ticker = "INVALID_NONEXISTENT_TICKER_XYZ999.NS"
        result = fetch_ticker_ohlcv(
            ticker=invalid_ticker,
            period="5d",
            max_retries=1,
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path),
            force_refresh=True,
            log_path=str(tmp_path / "download_log.json"),
        )
        assert result is None or result.empty

        # Verify log recorded failure
        import json
        with open(tmp_path / "download_log.json", "r") as f:
            log_data = json.load(f)
        assert invalid_ticker in log_data
        assert log_data[invalid_ticker]["status"] == "failed"

    def test_sanitize_ticker_roundtrip(self):
        tickers = ["^NSEI", "^NSEBANK", "USDINR=X", "CL=F", "RELIANCE.NS"]
        for t in tickers:
            san = sanitize_ticker_for_filename(t)
            assert "^" not in san
            assert "=" not in san
            restored = unsanitize_filename_to_ticker(san)
            assert restored == t


class TestCleaner:
    """Verifies anomaly detection, liquidity filter, and calendar alignment."""

    def test_cleaner_flags_injected_fake_split(self):
        """Injected 50% split in unadjusted price without ratio adjustment must be flagged."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "ticker": ["SPLIT_TEST.NS"] * 10,
            "date": dates,
            "open": [200.0] * 5 + [100.0] * 5,
            "high": [205.0] * 5 + [102.0] * 5,
            "low": [198.0] * 5 + [98.0] * 5,
            "close": [200.0] * 5 + [100.0] * 5,
            "adj_close": [100.0] * 10,  # smoothly adjusted
            "volume": [1_000_000] * 10,
        })

        cleaned = detect_split_bonus_anomalies(df, threshold=0.25)
        assert "split_bonus_anomaly" in cleaned.columns
        # Day 5 is the jump from 200 to 100 (-50%)
        assert cleaned.iloc[5]["split_bonus_anomaly"] == True
        assert cleaned["split_bonus_anomaly"].sum() == 1

    def test_liquidity_filter_excludes_illiquid_synthetic_ticker(self):
        """Tickers with median 60-day traded value < ₹5 Cr/day must be excluded."""
        dates = pd.date_range("2024-01-01", periods=70, freq="D")
        # Liquid stock: close=1000, vol=100,000 -> traded_val = 100,000,000 (10 Cr/day)
        liquid_df = pd.DataFrame({
            "ticker": ["LIQUID.NS"] * 70,
            "date": dates,
            "open": [1000.0] * 70,
            "high": [1010.0] * 70,
            "low": [990.0] * 70,
            "close": [1000.0] * 70,
            "adj_close": [1000.0] * 70,
            "volume": [100_000.0] * 70,
        })

        # Illiquid stock: close=50, vol=1,000 -> traded_val = 50,000 (0.005 Cr/day)
        illiquid_df = pd.DataFrame({
            "ticker": ["PENNY.NS"] * 70,
            "date": dates,
            "open": [50.0] * 70,
            "high": [51.0] * 70,
            "low": [49.0] * 70,
            "close": [50.0] * 70,
            "adj_close": [50.0] * 70,
            "volume": [1_000.0] * 70,
        })

        combined = pd.concat([liquid_df, illiquid_df], ignore_index=True)
        filtered, excluded = filter_liquidity(combined, min_median_value_inr=50_000_000.0)

        assert "PENNY.NS" in excluded
        assert "LIQUID.NS" not in excluded
        assert "low_liquidity" in excluded["PENNY.NS"]["reason"]
        assert set(filtered["ticker"].unique()) == {"LIQUID.NS"}

    def test_calendar_alignment_forward_fills_macro_without_future_leakage(self):
        """Macro series must be forward filled strictly, with no back-filling or future leakage."""
        calendar_dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        stock_dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")

        stock_df = pd.DataFrame({
            "ticker": ["STOCK.NS"] * 10,
            "date": stock_dates,
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [100.0] * 10,
            "adj_close": [100.0] * 10,
            "volume": [50000] * 10,
        })

        # Macro series has only a few sparse points (e.g., day 1 and day 5)
        macro_df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")],
            "usdinr": [83.0, 84.0],
        })

        aligned_stocks, aligned_macro = align_to_nse_calendar(
            stocks_df=stock_df,
            macro_df=macro_df,
            calendar_dates=calendar_dates,
        )

        assert aligned_macro is not None
        # Days 1, 2, 3, 4 must have usdinr == 83.0 (from day 1)
        # Day 4 MUST NOT know day 5's value of 84.0!
        day4_macro = aligned_macro[aligned_macro["date"] == pd.Timestamp("2024-01-04")]["usdinr"].iloc[0]
        assert day4_macro == 83.0, f"Day 4 leaked future value: {day4_macro}"

        # Day 5 takes 84.0
        day5_macro = aligned_macro[aligned_macro["date"] == pd.Timestamp("2024-01-05")]["usdinr"].iloc[0]
        assert day5_macro == 84.0

        # Day 6 forward fills 84.0
        day6_macro = aligned_macro[aligned_macro["date"] == pd.Timestamp("2024-01-06")]["usdinr"].iloc[0]
        assert day6_macro == 84.0

    def test_zero_volume_run_detection(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "ticker": ["HALT.NS"] * 10,
            "date": dates,
            "close": [100.0] * 10,
            "volume": [1000, 1000, 0, 0, 0, 0, 0, 1000, 1000, 1000],
        })
        res = detect_zero_volume_runs(df, min_consecutive=4)
        assert res["zero_vol_run"].sum() == 2  # days with streak >= 4 (days index 5 and 6)

    def test_minimum_history_filter(self):
        dates_short = pd.date_range("2024-01-01", periods=100, freq="D")
        dates_long = pd.date_range("2020-01-01", periods=800, freq="D")

        short_df = pd.DataFrame({
            "ticker": ["NEW_IPO.NS"] * 100,
            "date": dates_short,
            "close": [100.0] * 100,
        })
        long_df = pd.DataFrame({
            "ticker": ["VETERAN.NS"] * 800,
            "date": dates_long,
            "close": [500.0] * 800,
        })

        combined = pd.concat([short_df, long_df], ignore_index=True)
        filtered, excluded = filter_minimum_history(combined, min_trading_days=750)

        assert "NEW_IPO.NS" in excluded
        assert "insufficient_history" in excluded["NEW_IPO.NS"]["reason"]
        assert "VETERAN.NS" not in excluded
        assert set(filtered["ticker"].unique()) == {"VETERAN.NS"}


class TestMetadataAndScalingHelpers:
    """Verifies metadata builder and metric evaluation."""

    def test_metadata_builder(self, tmp_path):
        out_parquet = str(tmp_path / "universe.parquet")
        df = build_universe_metadata_table(
            panel_df=None,
            excluded_info={"TEST_EX.NS": {"rows": 50, "reason": "test"}},
            tier="U16",
            output_path=out_parquet,
        )
        assert os.path.exists(out_parquet)
        assert len(df) == 16
        assert "ticker" in df.columns
        assert "survivorship_safe" in df.columns

    def test_evaluate_metrics(self):
        y_true = np.array([0.05, -0.03, 0.02, -0.01])
        y_pred = np.array([0.04, -0.02, 0.01, -0.02])

        metrics = evaluate_metrics(y_true, y_pred)
        assert metrics["hit_rate"] == 1.0  # All signs matched!
        assert metrics["ic"] > 0.9
        assert metrics["rmse"] > 0.0
        assert metrics["count"] == 4

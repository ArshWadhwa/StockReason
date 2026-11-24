"""
StockReason v0.7 Target Engineering Module.

Implements stationary, volatility-scaled, cross-sectionally normalized excess return
targets for multi-horizon equity price prediction.

Target Construction Specification (h in {5, 21}):
1. Raw forward return:
     y_h_raw = adj_close[t+h] / adj_close[t] - 1
2. Benchmark excess return:
     y_h_excess = y_h_raw - nifty_return_h
   where nifty_return_h is the forward return of NIFTY 50 over [t, t+h].
3. Realized volatility scaling:
     sigma_h = sigma_60d * sqrt(h / 252)
   where sigma_60d is the trailing 60-day annualized realized return volatility of the stock.
4. Outlier clipping:
     y_h_clipped = clip(y_h_excess / (sigma_h + eps), -5.0, 5.0)
5. Cross-sectional rank normalization:
     y_h_ranked = (rank(y_h_clipped) - 1) / (N_t - 1) - 0.5 in [-0.5, 0.5]

Note on 126D Horizon:
The 126-day (~6-month) forward horizon suffers from extreme macroeconomic regime shifts
in Indian equities, severe sample overlap autocorrelation (effectively reducing independent
degrees of freedom by ~95%), and lookahead vulnerability during structural transitions.
It is disabled by default via config (enable_126d=False) but fully implemented for
scientific ablation and historical replication.
"""

from typing import List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_trailing_realized_volatility(
    prices: pd.Series,
    window: int = 60,
    annualization_factor: int = 252,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Compute trailing annualized realized return volatility for a price series.

    Parameters
    ----------
    prices : pd.Series
        Stock price series (adj_close or close).
    window : int, default 60
        Rolling window in trading days.
    annualization_factor : int, default 252
        Number of trading days per year.
    min_periods : Optional[int], default None
        Minimum observations required for rolling calculation (defaults to min(20, window)).

    Returns
    -------
    pd.Series
        Annualized realized volatility sigma_60d = std(daily_return, 60) * sqrt(252).
    """
    if min_periods is None:
        actual_min_periods = min(20, window)
    else:
        actual_min_periods = min(min_periods, window)

    daily_returns = prices.pct_change(1)
    rolling_std = daily_returns.rolling(window=window, min_periods=actual_min_periods).std()
    annualized_vol = rolling_std * np.sqrt(annualization_factor)
    return annualized_vol


def rank_normalize_cross_section(
    series: pd.Series,
    dates: pd.Series,
    out_bounds: Tuple[float, float] = (-0.5, 0.5),
) -> pd.Series:
    """
    Rank-normalize values cross-sectionally per date into [min_bound, max_bound].
    Preserves NaNs. Yields zero-mean cross-sectional distribution per date.

    Parameters
    ----------
    series : pd.Series
        Target or feature values to rank normalize.
    dates : pd.Series
        Corresponding date series for cross-sectional grouping.
    out_bounds : Tuple[float, float], default (-0.5, 0.5)
        Target output range [min_val, max_val].

    Returns
    -------
    pd.Series
        Rank-normalized series.
    """
    df_temp = pd.DataFrame({"val": series, "date": dates})
    min_b, max_b = out_bounds
    span = max_b - min_b

    def _rank_group(grp: pd.Series) -> pd.Series:
        valid_mask = grp.notna()
        n = valid_mask.sum()
        if n <= 1:
            res = pd.Series(0.0 if n == 1 else np.nan, index=grp.index)
            res[~valid_mask] = np.nan
            return res
        
        # Average ranking for ties
        ranks = grp[valid_mask].rank(method="average", ascending=True)
        norm_ranks = ((ranks - 1.0) / (n - 1.0)) * span + min_b
        res = pd.Series(np.nan, index=grp.index)
        res[valid_mask] = norm_ranks
        return res

    ranked = df_temp.groupby("date")["val"].transform(_rank_group)
    return ranked


def compute_targets(
    df: pd.DataFrame,
    horizons: Tuple[int, ...] = (5, 21),
    enable_126d: bool = False,
    vol_window: int = 60,
    annualization_factor: int = 252,
    clip_bounds: Tuple[float, float] = (-5.0, 5.0),
    benchmark_col: str = "nifty_50_close",
    price_col: str = "adj_close",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Calculate target variables for StockReason v0.7.

    Produces:
      - y_{h}d_raw: Raw forward return
      - y_{h}d_excess: Excess forward return over Nifty 50
      - y_{h}d_clipped: Volatility-scaled and clipped excess return
      - y_{h}d_ranked: Cross-sectionally rank-normalized [-0.5, 0.5] target
      - y_{h}d: Alias for y_{h}d_ranked (standard target for ML training)
      - label_start: Date t at decision/feature observation
      - label_end_{h}d: Date t+h at forward realization

    Parameters
    ----------
    df : pd.DataFrame
        Input price panel dataframe.
    horizons : Tuple[int, ...], default (5, 21)
        Prediction horizons in trading days.
    enable_126d : bool, default False
        Whether to calculate 126-day (~6 month) forward target.
    vol_window : int, default 60
        Trailing window in trading days for realized volatility.
    annualization_factor : int, default 252
        Trading days per year.
    clip_bounds : Tuple[float, float], default (-5.0, 5.0)
        Min and max clipping bounds for volatility-scaled return.
    benchmark_col : str, default 'nifty_50_close'
        Column representing benchmark index price level.
    price_col : str, default 'adj_close'
        Stock price column used for return calculation (falls back to 'close').
    date_col : str, default 'date'
        Date column name.
    ticker_col : str, default 'ticker'
        Ticker column name.

    Returns
    -------
    pd.DataFrame
        Dataframe with all target and label timing columns appended.
    """
    if df.empty:
        return df.copy()

    result = df.copy()

    # 1. Validate dates and price columns
    if date_col not in result.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe.")
    result[date_col] = pd.to_datetime(result[date_col]).dt.tz_localize(None)

    actual_price_col = price_col
    if actual_price_col not in result.columns:
        if "close" in result.columns:
            logger.warning(f"Price column '{price_col}' not found; falling back to 'close'.")
            actual_price_col = "close"
        else:
            raise ValueError(f"Neither '{price_col}' nor 'close' found in dataframe.")

    has_ticker = ticker_col in result.columns
    if not has_ticker:
        result[ticker_col] = "ALL"

    # Sort deterministically
    result = result.sort_values([ticker_col, date_col]).reset_index(drop=True)

    # 2. Label start timing column
    result["label_start"] = result[date_col]

    # Active horizon set
    active_horizons = list(horizons)
    if enable_126d and 126 not in active_horizons:
        active_horizons.append(126)
    elif not enable_126d and 126 in active_horizons:
        active_horizons.remove(126)

    # 3. Compute benchmark forward return lookup by date
    has_benchmark = benchmark_col in result.columns
    bench_returns_by_h = {}
    if has_benchmark:
        bench_by_date = (
            result.groupby(date_col)[benchmark_col]
            .first()
            .sort_index()
            .dropna()
        )
        for h in active_horizons:
            fwd_bench = bench_by_date.shift(-h)
            bench_ret = fwd_bench / bench_by_date - 1.0
            bench_returns_by_h[h] = bench_ret
    else:
        logger.warning(
            f"Benchmark column '{benchmark_col}' not found. "
            "Excess returns will equal raw returns."
        )

    # 4. Groupwise calculations per ticker
    grouped_dfs = []
    for ticker, grp in result.groupby(ticker_col):
        grp = grp.sort_values(date_col).copy()
        prices = grp[actual_price_col]

        # Trailing realized volatility (sigma_60d annualized)
        ann_vol = compute_trailing_realized_volatility(
            prices=prices,
            window=vol_window,
            annualization_factor=annualization_factor,
        )
        grp["realized_vol_60d"] = ann_vol

        for h in active_horizons:
            # Raw forward return
            fwd_prices = prices.shift(-h)
            y_raw = fwd_prices / prices - 1.0
            grp[f"y_{h}d_raw"] = y_raw

            # Label timing
            grp[f"label_end_{h}d"] = grp[date_col].shift(-h)

            # Benchmark return
            if has_benchmark and h in bench_returns_by_h:
                nifty_ret = grp[date_col].map(bench_returns_by_h[h])
            else:
                nifty_ret = 0.0
            
            # Excess return
            y_excess = y_raw - nifty_ret
            grp[f"y_{h}d_excess"] = y_excess

            # Horizon-scaled realized volatility
            # sigma_h = ann_vol * sqrt(h / annualization_factor)
            sigma_h = ann_vol * np.sqrt(h / annualization_factor)

            # Volatility-scaled return (clamped floor to prevent division blowout)
            denom = np.maximum(sigma_h, 1e-4)
            y_scaled = y_excess / denom

            # Outlier clipping at clip_bounds (+/- 5.0)
            y_clipped = np.clip(y_scaled, clip_bounds[0], clip_bounds[1])
            # If sigma_h or y_excess was NaN, keep NaN
            y_clipped = np.where(y_excess.isna() | sigma_h.isna(), np.nan, y_clipped)
            grp[f"y_{h}d_clipped"] = y_clipped

        grouped_dfs.append(grp)

    combined = pd.concat(grouped_dfs, ignore_index=True)

    # 5. Cross-sectional rank normalization across stocks per date
    for h in active_horizons:
        clipped_col = f"y_{h}d_clipped"
        ranked_col = f"y_{h}d_ranked"
        standard_target_col = f"y_{h}d"

        combined[ranked_col] = rank_normalize_cross_section(
            series=combined[clipped_col],
            dates=combined[date_col],
            out_bounds=(-0.5, 0.5),
        )
        # Standard target alias
        combined[standard_target_col] = combined[ranked_col]

    if not has_ticker and ticker_col in combined.columns:
        combined = combined.drop(columns=[ticker_col])

    return combined

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _compute_single_ticker_indicators(df_t: pd.DataFrame) -> pd.DataFrame:
    df = df_t.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_p = df["open"]
    volume = df["volume"]

    # 1. Simple Moving Averages (SMA) & Exponential Moving Averages (EMA)
    for p in [5, 10, 15, 20, 30, 50, 100, 150, 200]:
        df[f"sma_{p}"] = close.rolling(window=p).mean()
        df[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()
        df[f"dist_sma_{p}"] = (close - df[f"sma_{p}"]) / df[f"sma_{p}"]
        df[f"dist_ema_{p}"] = (close - df[f"ema_{p}"]) / df[f"ema_{p}"]

    # 2. Moving Average Cross Ratios
    df["sma_5_20_ratio"] = df["sma_5"] / df["sma_20"]
    df["sma_20_50_ratio"] = df["sma_20"] / df["sma_50"]
    df["sma_50_200_ratio"] = df["sma_50"] / df["sma_200"]

    # 3. Price Spreads & Candlesticks
    df["hl_spread"] = high - low
    df["hl_spread_pct"] = df["hl_spread"] / close
    df["co_spread"] = close - open_p
    df["co_spread_pct"] = df["co_spread"] / open_p
    df["upper_shadow"] = high - np.maximum(close, open_p)
    df["lower_shadow"] = np.minimum(close, open_p) - low

    # 4. Momentum & Returns
    for p in [1, 2, 3, 5, 10, 15, 21, 42, 63]:
        df[f"return_{p}d"] = close.pct_change(p)
        df[f"log_return_{p}d"] = np.log(close / close.shift(p))
        df[f"roc_{p}"] = ((close - close.shift(p)) / close.shift(p)) * 100
        df[f"mom_{p}"] = close - close.shift(p)

    # 5. Relative Strength Index (RSI)
    for p in [7, 14, 21, 28]:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
        rs = gain / (loss + 1e-10)
        df[f"rsi_{p}"] = 100 - (100 / (1 + rs))

    # 6. MACD
    for fast, slow, sig in [(12, 26, 9), (8, 17, 9), (5, 35, 5)]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=sig, adjust=False).mean()
        macd_hist = macd_line - signal_line

        suffix = f"_{fast}_{slow}"
        df[f"macd_line{suffix}"] = macd_line
        df[f"macd_signal{suffix}"] = signal_line
        df[f"macd_hist{suffix}"] = macd_hist

    # 7. Bollinger Bands (BB)
    for p, std_dev in [(10, 2), (20, 2), (50, 2)]:
        rolling_mean = close.rolling(window=p).mean()
        rolling_std = close.rolling(window=p).std()
        upper = rolling_mean + (rolling_std * std_dev)
        lower = rolling_mean - (rolling_std * std_dev)

        df[f"bb_upper_{p}"] = upper
        df[f"bb_lower_{p}"] = lower
        df[f"bb_mid_{p}"] = rolling_mean
        df[f"bb_width_{p}"] = (upper - lower) / rolling_mean
        df[f"bb_pct_b_{p}"] = (close - lower) / (upper - lower + 1e-10)

    # 8. ATR & Volatility Metrics
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    for p in [7, 14, 21]:
        atr = tr.rolling(window=p).mean()
        df[f"atr_{p}"] = atr
        df[f"natr_{p}"] = (atr / close) * 100

    ret_1d = df["return_1d"]
    for p in [5, 10, 20, 60, 120]:
        df[f"volatility_{p}d"] = ret_1d.rolling(window=p).std() * np.sqrt(252)

    df["volatility_parkinson_20"] = np.sqrt(
        (1 / (4 * np.log(2))) * ((np.log(high / low)) ** 2).rolling(window=20).mean()
    ) * np.sqrt(252)

    df["volatility_garman_klass_20"] = np.sqrt(
        (0.5 * (np.log(high / low)) ** 2 - (2 * np.log(2) - 1) * (np.log(close / open_p)) ** 2)
        .rolling(window=20).mean()
    ) * np.sqrt(252)

    # 9. Stochastic Oscillator
    for p in [14, 21]:
        lowest_low = low.rolling(window=p).min()
        highest_high = high.rolling(window=p).max()
        stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
        stoch_d = stoch_k.rolling(window=3).mean()
        df[f"stoch_k_{p}"] = stoch_k
        df[f"stoch_d_{p}"] = stoch_d

    # 10. Williams %R
    for p in [7, 14, 28]:
        lowest_low = low.rolling(window=p).min()
        highest_high = high.rolling(window=p).max()
        df[f"williams_r_{p}"] = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))

    # 11. Commodity Channel Index (CCI)
    tp = (high + low + close) / 3
    for p in [14, 20]:
        sma_tp = tp.rolling(window=p).mean()
        mad_tp = tp.rolling(window=p).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df[f"cci_{p}"] = (tp - sma_tp) / (0.015 * mad_tp + 1e-10)

    # 12. Donchian & Keltner Channels
    for p in [10, 20]:
        donchian_high = high.rolling(window=p).max()
        donchian_low = low.rolling(window=p).min()
        df[f"donchian_high_{p}"] = donchian_high
        df[f"donchian_low_{p}"] = donchian_low
        df[f"donchian_mid_{p}"] = (donchian_high + donchian_low) / 2
        df[f"donchian_pos_{p}"] = (close - donchian_low) / (donchian_high - donchian_low + 1e-10)

    ema_20 = close.ewm(span=20, adjust=False).mean()
    atr_10 = tr.rolling(window=10).mean()
    df["keltner_upper_20"] = ema_20 + (2 * atr_10)
    df["keltner_lower_20"] = ema_20 - (2 * atr_10)
    df["keltner_pos_20"] = (close - df["keltner_lower_20"]) / (df["keltner_upper_20"] - df["keltner_lower_20"] + 1e-10)

    # 13. Volume & Money Flow
    for p in [5, 10, 20, 50]:
        df[f"vol_sma_{p}"] = volume.rolling(window=p).mean()
        df[f"vol_ratio_{p}"] = volume / (df[f"vol_sma_{p}"] + 1e-10)

    direction = np.sign(close.diff().fillna(0))
    df["obv"] = (direction * volume).cumsum()
    df["obv_ema_20"] = df["obv"].ewm(span=20, adjust=False).mean()

    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
    mf_volume = mf_multiplier * volume
    df["cmf_20"] = mf_volume.rolling(window=20).sum() / (volume.rolling(window=20).sum() + 1e-10)
    df["pvt"] = (df["return_1d"] * volume).fillna(0).cumsum()

    # 14. ADX & Directional Movement Index (DMI 14)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_14 = tr.rolling(window=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / (atr_14 + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / (atr_14 + 1e-10))
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))

    df["plus_di_14"] = plus_di
    df["minus_di_14"] = minus_di
    df["adx_14"] = dx.rolling(window=14).mean()

    # 15. Trend Slopes
    for p in [5, 10, 20]:
        df[f"close_slope_{p}"] = (close - close.shift(p)) / p

    return df


def generate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating 100+ technical indicators...")
    if df.empty:
        return df

    if "ticker" in df.columns:
        feature_dfs = []
        for ticker, df_group in df.groupby("ticker"):
            df_group_sorted = df_group.sort_values("date").reset_index(drop=True)
            df_feat = _compute_single_ticker_indicators(df_group_sorted)
            feature_dfs.append(df_feat)
        result = pd.concat(feature_dfs, ignore_index=True)
    else:
        df_sorted = df.sort_values("date").reset_index(drop=True)
        result = _compute_single_ticker_indicators(df_sorted)

    num_generated = len([c for c in result.columns if c not in df.columns])
    logger.info(f"Successfully generated {num_generated} technical indicators.")
    return result

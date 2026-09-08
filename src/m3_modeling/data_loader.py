import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import joblib


def load_data_from_directory(data_dir):
    """
    Loads all Parquet files in a directory and merges them on date and ticker.
    If a single file is passed, loads just that file.
    """
    if os.path.isfile(data_dir):
        if data_dir.endswith('.parquet'):
            return pd.read_parquet(data_dir)
        return pd.read_csv(data_dir, index_col=0, parse_dates=True)

    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")

    df_merged = None
    # Prioritize price_features.parquet as the base dataframe
    parquet_files = sorted(parquet_files, key=lambda x: 0 if 'price_features' in x else 1)

    for file in parquet_files:
        # Skip predictions.json or non-feature files
        if 'predictions' in os.path.basename(file):
            continue
        df = pd.read_parquet(file)
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        if df_merged is None:
            df_merged = df
        else:
            # Merge on date and ticker
            merge_cols = [c for c in ['date', 'ticker'] if c in df.columns and c in df_merged.columns]
            if merge_cols:
                df_merged = pd.merge(df_merged, df, on=merge_cols, how='left')
            else:
                # Fallback to concat if no common keys
                df_merged = pd.concat([df_merged, df], axis=1)

    # Fill missing sentiment or other left-joined features with 0 (neutral/baseline)
    df_merged.fillna(0, inplace=True)

    return df_merged


def _get_feature_cols(df):
    """Select numeric feature columns, excluding targets and identifiers."""
    exclude_cols = {'Target_1D', 'Target_5D', 'Target_21D', 'Target_126D', 'ticker', 'date', 'market_regime'}
    return [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]


def _build_sequences_per_ticker(df, feature_cols, sequence_length=60):
    """
    Build (X, Y) sequences PER TICKER to prevent cross-ticker contamination.
    Returns raw (unscaled) feature arrays and target arrays, plus a date array
    for chronological sorting.

    Each sequence is a window of `sequence_length` consecutive rows for a
    single ticker. The target is the forward-looking return at the end of
    the window.
    """
    all_x = []
    all_y = []
    all_dates = []

    target_cols = ['Target_1D', 'Target_5D', 'Target_21D', 'Target_126D']

    if 'ticker' in df.columns:
        for ticker, grp in df.groupby('ticker'):
            grp = grp.sort_values('date').reset_index(drop=True)
            features = grp[feature_cols].values
            targets = grp[target_cols].values
            dates = grp['date'].values

            for i in range(sequence_length, len(features)):
                all_x.append(features[i - sequence_length:i, :])
                all_y.append(targets[i, :])
                all_dates.append(dates[i])
    else:
        df = df.sort_values('date').reset_index(drop=True) if 'date' in df.columns else df
        features = df[feature_cols].values
        targets = df[target_cols].values
        dates = df['date'].values if 'date' in df.columns else np.arange(len(df))

        for i in range(sequence_length, len(features)):
            all_x.append(features[i - sequence_length:i, :])
            all_y.append(targets[i, :])
            all_dates.append(dates[i])

    x = np.array(all_x)
    y = np.array(all_y)
    dates = np.array(all_dates)

    # Sort chronologically so train/test split is temporal
    sort_idx = np.argsort(dates)
    x = x[sort_idx]
    y = y[sort_idx]
    dates = dates[sort_idx]

    return x, y, dates


def prepare_multi_horizon_data(data_dir, sequence_length=60, split_ratio=0.8, batch_size=32):
    """
    Loads data, generates targets, builds per-ticker sequences, splits
    chronologically, then fits scalers on TRAIN ONLY to prevent data leakage.

    Returns DataLoaders and the fitted scalers.
    """
    df = load_data_from_directory(data_dir)

    # Target Generation (forward-looking returns based on close price)
    if 'close' not in df.columns:
        raise ValueError("Data must contain a 'close' column to calculate targets.")

    if 'ticker' in df.columns:
        df['Target_1D'] = df.groupby('ticker')['close'].shift(-1) / df['close'] - 1
        df['Target_5D'] = df.groupby('ticker')['close'].shift(-5) / df['close'] - 1
        df['Target_21D'] = df.groupby('ticker')['close'].shift(-21) / df['close'] - 1
        df['Target_126D'] = df.groupby('ticker')['close'].shift(-126) / df['close'] - 1
    else:
        df['Target_1D'] = df['close'].shift(-1) / df['close'] - 1
        df['Target_5D'] = df['close'].shift(-5) / df['close'] - 1
        df['Target_21D'] = df['close'].shift(-21) / df['close'] - 1
        df['Target_126D'] = df['close'].shift(-126) / df['close'] - 1

    # Drop rows with NaNs (from shifting and technical indicators)
    df.dropna(inplace=True)

    # Dynamic Feature Selection
    feature_cols = _get_feature_cols(df)

    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    print(f"Detected {len(feature_cols)} features for training.")

    # Build sequences PER TICKER (fixes cross-ticker contamination)
    x, y, dates = _build_sequences_per_ticker(df, feature_cols, sequence_length)

    # ── Chronological Train / Test Split ──────────────────────────────
    split_idx = int(len(x) * split_ratio)
    x_train_raw, y_train_raw = x[:split_idx], y[:split_idx]
    x_test_raw, y_test_raw = x[split_idx:], y[split_idx:]

    # ── Fit scalers on TRAIN ONLY (fixes data leakage) ────────────────
    # Reshape sequences to 2D for scaler fitting: (n_samples * seq_len, n_features)
    n_train, seq_len, n_feat = x_train_raw.shape
    n_test = x_test_raw.shape[0]

    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler = MinMaxScaler(feature_range=(-1, 1))

    # Fit on train data only
    train_flat = x_train_raw.reshape(-1, n_feat)
    feature_scaler.fit(train_flat)

    target_scaler.fit(y_train_raw)

    # Transform both train and test
    x_train_scaled = feature_scaler.transform(x_train_raw.reshape(-1, n_feat)).reshape(n_train, seq_len, n_feat)
    x_test_scaled = feature_scaler.transform(x_test_raw.reshape(-1, n_feat)).reshape(n_test, seq_len, n_feat)

    y_train_scaled = target_scaler.transform(y_train_raw)
    y_test_scaled = target_scaler.transform(y_test_raw)

    # Convert to PyTorch tensors
    x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

    # DataLoaders
    train_dataset = TensorDataset(x_train_t, y_train_t)
    test_dataset = TensorDataset(x_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_features = len(feature_cols)

    return train_loader, test_loader, feature_scaler, target_scaler, x_test_raw, y_test_raw, num_features


def prepare_walk_forward_folds(data_dir, sequence_length=60, n_folds=5, batch_size=32):
    """
    Walk-forward (expanding window) cross-validation for financial time series.

    Returns a list of dicts, each containing:
        - train_loader, val_loader
        - feature_scaler, target_scaler (fit on that fold's train data only)
        - x_val_raw, y_val_raw (unscaled validation data for evaluation)
        - num_features
        - fold_idx

    Each fold: train on all data up to cutoff, validate on cutoff → cutoff+val_size.
    The final fold uses the most recent data as validation (production-representative).
    """
    df = load_data_from_directory(data_dir)

    # Target Generation
    if 'close' not in df.columns:
        raise ValueError("Data must contain a 'close' column to calculate targets.")

    if 'ticker' in df.columns:
        df['Target_1D'] = df.groupby('ticker')['close'].shift(-1) / df['close'] - 1
        df['Target_5D'] = df.groupby('ticker')['close'].shift(-5) / df['close'] - 1
        df['Target_21D'] = df.groupby('ticker')['close'].shift(-21) / df['close'] - 1
        df['Target_126D'] = df.groupby('ticker')['close'].shift(-126) / df['close'] - 1
    else:
        df['Target_1D'] = df['close'].shift(-1) / df['close'] - 1
        df['Target_5D'] = df['close'].shift(-5) / df['close'] - 1
        df['Target_21D'] = df['close'].shift(-21) / df['close'] - 1
        df['Target_126D'] = df['close'].shift(-126) / df['close'] - 1

    df.dropna(inplace=True)

    feature_cols = _get_feature_cols(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    print(f"[Walk-Forward] Detected {len(feature_cols)} features.")

    # Build sequences per ticker
    x, y, dates = _build_sequences_per_ticker(df, feature_cols, sequence_length)

    total_samples = len(x)
    # Reserve 20% of total data for validation in each fold
    val_size = total_samples // (n_folds + 1)
    # Minimum training size: 40% of total data
    min_train_size = int(total_samples * 0.4)

    folds = []
    n_feat = x.shape[2]
    seq_len = x.shape[1]

    for fold_idx in range(n_folds):
        # Expanding window: each fold adds more training data
        train_end = min_train_size + fold_idx * val_size
        val_end = min(train_end + val_size, total_samples)

        if train_end >= total_samples or val_end <= train_end:
            break

        x_train_raw = x[:train_end]
        y_train_raw = y[:train_end]
        x_val_raw = x[train_end:val_end]
        y_val_raw = y[train_end:val_end]

        # Fit scalers on this fold's training data only
        f_scaler = MinMaxScaler(feature_range=(-1, 1))
        t_scaler = MinMaxScaler(feature_range=(-1, 1))

        n_train = x_train_raw.shape[0]
        n_val = x_val_raw.shape[0]

        f_scaler.fit(x_train_raw.reshape(-1, n_feat))
        t_scaler.fit(y_train_raw)

        x_train_s = f_scaler.transform(x_train_raw.reshape(-1, n_feat)).reshape(n_train, seq_len, n_feat)
        x_val_s = f_scaler.transform(x_val_raw.reshape(-1, n_feat)).reshape(n_val, seq_len, n_feat)
        y_train_s = t_scaler.transform(y_train_raw)
        y_val_s = t_scaler.transform(y_val_raw)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(x_train_s, dtype=torch.float32),
                          torch.tensor(y_train_s, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(x_val_s, dtype=torch.float32),
                          torch.tensor(y_val_s, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False
        )

        folds.append({
            'fold_idx': fold_idx,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'feature_scaler': f_scaler,
            'target_scaler': t_scaler,
            'x_val_raw': x_val_raw,
            'y_val_raw': y_val_raw,
            'num_features': len(feature_cols),
            'train_size': n_train,
            'val_size': n_val,
        })

        print(f"  Fold {fold_idx + 1}/{n_folds}: train={n_train}, val={n_val}")

    return folds

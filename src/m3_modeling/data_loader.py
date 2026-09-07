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
    num_cols = df_merged.select_dtypes(include=[np.number]).columns
    df_merged[num_cols] = df_merged[num_cols].fillna(0)
                
    return df_merged

def prepare_multi_horizon_data(data_dir, sequence_length=60, split_ratio=0.8, batch_size=32):
    """
    Loads data, dynamically selects features, generates targets, and scales.
    Returns DataLoaders and the fitted scalers.
    """
    df = load_data_from_directory(data_dir)
    
    # Target Generation (Forward looking returns based on close price)
    # We do this first before dropping NA so we don't mess up the shift
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
    
    # Drop rows with NaNs (due to shifting and M1 technical indicators)
    df.dropna(inplace=True)
    
    # Dynamic Feature Selection
    # We exclude non-numeric columns, targets, and identifier columns
    exclude_cols = ['Target_1D', 'Target_5D', 'Target_21D', 'Target_126D', 'ticker', 'date']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")
        
    print(f"Detected {len(feature_cols)} features for training.")
    
    # Prepare Arrays
    features = df[feature_cols].values
    targets = df[['Target_1D', 'Target_5D', 'Target_21D', 'Target_126D']].values
    
    # Scale Features and Targets
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_targets = target_scaler.fit_transform(targets)
    
    # Generate Sequences
    x, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        x.append(scaled_features[i-sequence_length:i, :])
        y.append(scaled_targets[i, :])
        
    x, y = np.array(x), np.array(y)
    
    # Train / Test split chronologically
    split_idx = int(len(x) * split_ratio)
    x_train, y_train = x[:split_idx], y[:split_idx]
    x_test, y_test = x[split_idx:], y[split_idx:]
    
    # Convert to PyTorch tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # DataLoaders
    train_dataset = TensorDataset(x_train_t, y_train_t)
    test_dataset = TensorDataset(x_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_features = len(feature_cols)
    
    return train_loader, test_loader, feature_scaler, target_scaler, x_test, y_test, num_features

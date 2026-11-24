"""
StockReason v0.10 — Modern Time-Series Architectures for Financial Forecasting.
================================================================================
Remediates Phase 5 (Section 10):
1. DLinear:
   Decomposes input sequence into Trend (Moving Average) and Residual components,
   mapping them via dedicated linear layers: y = W_trend * x_trend + W_res * x_res.
2. Temporal Convolutional Network (TCN):
   Dilated causal 1D convolutions with residual blocks and receptive field expansion.
3. N-HiTS:
   Multi-rate hierarchical temporal interpolation block with multi-scale pooling.
4. PyTorch GRU / LSTM Benchmark: Control.
5. Strict Memory Bounding:
   Lazy sequence sampling with batch_size=256-512 to prevent OOM on 16GB Apple Silicon.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.v092.models.baselines import compute_absolute_price_metrics


# -----------------------------------------------------------------------------
# 1. DLinear Architecture
# -----------------------------------------------------------------------------
class MovingAverageBlock(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        # Pad along sequence dimension
        x_pad = F.pad(x.transpose(1, 2), (self.pad, self.pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
        return trend.transpose(1, 2)


class DLinearModel(nn.Module):
    def __init__(self, seq_len: int = 63, n_features: int = 21):
        super().__init__()
        self.seq_len = seq_len
        self.ma = MovingAverageBlock(kernel_size=5)
        self.linear_trend = nn.Linear(seq_len * n_features, 1)
        self.linear_res = nn.Linear(seq_len * n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        trend = self.ma(x)
        residual = x - trend

        batch_size = x.shape[0]
        flat_trend = trend.reshape(batch_size, -1)
        flat_res = residual.reshape(batch_size, -1)

        y_trend = self.linear_trend(flat_trend)
        y_res = self.linear_res(flat_res)
        return (y_trend + y_res).squeeze(-1)


# -----------------------------------------------------------------------------
# 2. Temporal Convolutional Network (TCN)
# -----------------------------------------------------------------------------
class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            padding=2 * dilation, dilation=dilation
        )
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        res = self.res(x)
        out = self.conv(x)[:, :, :x.shape[2]] # Causal slice
        out = self.relu(self.norm(out))
        return out + res


class TCNModel(nn.Module):
    def __init__(self, n_features: int = 21, hidden_dim: int = 32):
        super().__init__()
        self.b1 = CausalConvBlock(n_features, hidden_dim, dilation=1)
        self.b2 = CausalConvBlock(hidden_dim, hidden_dim, dilation=2)
        self.b3 = CausalConvBlock(hidden_dim, hidden_dim, dilation=4)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> transpose to (batch, n_features, seq_len)
        x_in = x.transpose(1, 2)
        h = self.b3(self.b2(self.b1(x_in)))
        return self.fc(h[:, :, -1]).squeeze(-1)


# -----------------------------------------------------------------------------
# 3. N-HiTS Multi-Rate Interpolation Model
# -----------------------------------------------------------------------------
class NHiTSBlock(nn.Module):
    def __init__(self, seq_len: int, n_features: int, pool_size: int = 2):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
        down_dim = (seq_len // pool_size) * n_features
        self.mlp = nn.Sequential(
            nn.Linear(down_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> pool over seq_len
        x_t = x.transpose(1, 2)
        pooled = self.pool(x_t).transpose(1, 2)
        flat = pooled.reshape(x.shape[0], -1)
        return self.mlp(flat).squeeze(-1)


class NHiTSModel(nn.Module):
    def __init__(self, seq_len: int = 64, n_features: int = 21):
        super().__init__()
        self.b1 = NHiTSBlock(seq_len=seq_len, n_features=n_features, pool_size=2)
        self.b2 = NHiTSBlock(seq_len=seq_len, n_features=n_features, pool_size=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b1(x) + self.b2(x)


# -----------------------------------------------------------------------------
# Training & Evaluation Engine for Time-Series Architectures
# -----------------------------------------------------------------------------
def evaluate_timeseries_architectures(
    X_train_sequences: np.ndarray,
    y_train: np.ndarray,
    X_val_sequences: np.ndarray,
    y_val: np.ndarray,
    val_close: np.ndarray,
    batch_size: int = 512,
    epochs: int = 3,
    lr: float = 0.003,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Trains and evaluates DLinear, TCN, and N-HiTS architectures on Return MAE and Price MAE.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    seq_len = X_train_sequences.shape[1]
    n_features = X_train_sequences.shape[2]

    models = {
        "DLinear": DLinearModel(seq_len=seq_len, n_features=n_features).to(device),
        "TCN": TCNModel(n_features=n_features, hidden_dim=32).to(device),
        "N-HiTS": NHiTSModel(seq_len=seq_len, n_features=n_features).to(device),
    }

    # Clean NaNs in training
    valid_tr = ~np.isnan(y_train)
    X_tr_t = torch.from_numpy(X_train_sequences[valid_tr]).float().to(device)
    y_tr_t = torch.from_numpy(y_train[valid_tr]).float().to(device)

    valid_val = ~np.isnan(y_val)
    X_val_t = torch.from_numpy(X_val_sequences).float().to(device)
    y_val_clean = y_val[valid_val]
    val_close_clean = val_close[valid_val]

    results = []

    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.L1Loss() # Optimize on Return MAE

        model.train()
        n_samples = len(X_tr_t)
        n_batches = n_samples // batch_size
        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                optimizer.zero_grad()
                out = model(X_tr_t[idx])
                loss = criterion(out, y_tr_t[idx])
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for b in range(int(np.ceil(len(X_val_t) / 1024))):
                b_slice = slice(b * 1024, (b + 1) * 1024)
                preds.append(model(X_val_t[b_slice]).cpu().numpy())
            p_val = np.concatenate(preds)

        m = compute_absolute_price_metrics(y_val_clean, p_val[valid_val], val_close_clean)
        results.append({
            "architecture": name,
            "lookback_seq_len": seq_len,
            "n_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            **m,
        })

    return pd.DataFrame(results)

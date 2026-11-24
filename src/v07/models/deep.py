"""
Deep learning sequence forecasters for StockReason v0.7.

Implements:
- LSTMForecaster: wraps MultiHorizonLSTM/BaselineLSTM architecture with 5 seeds averaged,
  early stopping, and gradient clipping.
- GRUForecaster: 1-layer GRU (hidden 32-64) with 5 seeds averaged, early stopping,
  and gradient clipping.
- DLinearForecaster: series decomposition into trend and remainder with dual linear layers
  per DLinear paper (Zeng et al., AAAI 2023).
"""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

try:
    from src.v07.models.base import BaseForecaster
except ImportError:
    from v07.models.base import BaseForecaster


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Architectures
# ─────────────────────────────────────────────────────────────────────────────

class MultiHorizonLSTMModule(nn.Module):
    """2-layer LSTM with MC Dropout and linear head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out.squeeze(-1)


class BaselineLSTMModule(nn.Module):
    """1-layer LSTM with larger hidden dimension."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        output_dim: int = 1,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out.squeeze(-1)


class GRUModule(nn.Module):
    """1-layer GRU with hidden 32-64 and linear head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 48,
        num_layers: int = 1,
        output_dim: int = 1,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out.squeeze(-1)


class MovingAverage(nn.Module):
    """Moving average block for series decomposition in DLinear."""

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Seq_len, Features] -> permute to [Batch, Features, Seq_len]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_perm = x_pad.permute(0, 2, 1)
        out = self.avg(x_perm)
        return out.permute(0, 2, 1)


class DLinearModule(nn.Module):
    """
    DLinear decomposition model:
    Trend = MovingAverage(X)
    Remainder = X - Trend
    Y = Linear_Trend(Trend) + Linear_Remainder(Remainder)
    """

    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        kernel_size: int = 13,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.decompsition = MovingAverage(kernel_size=kernel_size)
        self.linear_trend = nn.Linear(seq_len, 1)
        self.linear_remainder = nn.Linear(seq_len, 1)
        self.head = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, D]
        if x.size(1) == 1:
            # Single time step fallback
            trend = x
            rem = torch.zeros_like(x)
        else:
            k = min(self.decompsition.kernel_size, x.size(1))
            if k % 2 == 0:
                k = max(1, k - 1)
            decomps = MovingAverage(kernel_size=k)
            trend = decomps(x)
            rem = x - trend

        # Trend & Remainder projection along time dimension
        # trend.permute(0, 2, 1): [B, D, T]
        trend_proj = self.linear_trend(trend.permute(0, 2, 1)).squeeze(-1)  # [B, D]
        rem_proj = self.linear_remainder(rem.permute(0, 2, 1)).squeeze(-1)  # [B, D]

        combined = self.dropout(trend_proj + rem_proj)
        out = self.head(combined)  # [B, 1]
        return out.squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: PyTorch Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def _train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 30,
    lr: float = 0.002,
    patience: int = 6,
    max_norm: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, float]:
    """Train single PyTorch model with Adam, early stopping, and grad clipping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            total_val = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = model(bx)
                    val_loss += criterion(pred, by).item() * len(by)
                    total_val += len(by)
            avg_val = val_loss / max(1, total_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        else:
            best_weights = copy.deepcopy(model.state_dict())
            best_val_loss = 0.0

    model.load_state_dict(best_weights)
    return model, best_val_loss


def _ensure_3d(X: np.ndarray) -> np.ndarray:
    """Ensure array is 3D (N, T, D). If 2D (N, D), reshapes to (N, 1, D)."""
    if X.ndim == 2:
        return X[:, np.newaxis, :]
    elif X.ndim == 1:
        return X[:, np.newaxis, np.newaxis]
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Forecaster Classes
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster(BaseForecaster):
    """
    LSTM forecaster wrapping MultiHorizonLSTM / BaselineLSTM architecture.
    Trained per fold with 5 seeds averaged, early stopping, and gradient clipping.
    """

    def __init__(
        self,
        name: str = "lstm",
        seed: int = 42,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        epochs: int = 30,
        lr: float = 0.002,
        patience: int = 6,
        batch_size: int = 64,
        n_seeds: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.n_seeds = n_seeds

        self.scaler: Optional[StandardScaler] = None
        self.models: List[nn.Module] = []
        self.device = torch.device("cpu")

    def _scale_transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        N, T, D = X.shape
        flat = X.reshape(-1, D)
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.scaler = StandardScaler()
            flat_scaled = self.scaler.fit_transform(flat)
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler not fit.")
            flat_scaled = self.scaler.transform(flat)
        return flat_scaled.reshape(N, T, D)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMForecaster":
        X_tr = _ensure_3d(X_train)
        y_tr = np.asarray(y_train, dtype=np.float32).flatten()
        X_tr_scaled = self._scale_transform(X_tr, fit=True)

        train_ds = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = _ensure_3d(X_val)
            y_v = np.asarray(y_val, dtype=np.float32).flatten()
            X_v_scaled = self._scale_transform(X_v, fit=False)
            val_ds = TensorDataset(
                torch.tensor(X_v_scaled, dtype=torch.float32),
                torch.tensor(y_v, dtype=torch.float32),
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False
            )

        input_dim = X_tr.shape[2]
        self.models = []

        # Train 5 seeds averaged
        for s_idx in range(self.n_seeds):
            torch.manual_seed(self.seed + s_idx)
            np.random.seed(self.seed + s_idx)

            module = MultiHorizonLSTMModule(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=1,
                dropout_rate=self.dropout_rate,
            )
            trained_model, _ = _train_pytorch_model(
                model=module,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.epochs,
                lr=self.lr,
                patience=self.patience,
                max_norm=1.0,
                device=self.device,
            )
            trained_model.eval()
            self.models.append(trained_model)

        self.is_fitted = True
        train_preds = self.predict(X_train)
        self.train_residuals = y_tr - train_preds

        if X_val is not None and y_val is not None and len(y_val) > 0:
            val_preds = self.predict(X_val)
            self.val_residuals = np.asarray(y_val).flatten() - val_preds

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or not self.models:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        X_3d = _ensure_3d(X)
        X_scaled = self._scale_transform(X_3d, fit=False)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        preds_list = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                p = m(x_tensor).cpu().numpy().flatten()
                preds_list.append(p)

        return np.mean(preds_list, axis=0)


class GRUForecaster(BaseForecaster):
    """
    1-layer GRU forecaster (hidden 32-64) with 5 seeds averaged,
    early stopping, and gradient clipping.
    """

    def __init__(
        self,
        name: str = "gru",
        seed: int = 42,
        hidden_dim: int = 48,
        epochs: int = 30,
        lr: float = 0.002,
        patience: int = 6,
        batch_size: int = 64,
        n_seeds: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.n_seeds = n_seeds

        self.scaler: Optional[StandardScaler] = None
        self.models: List[nn.Module] = []
        self.device = torch.device("cpu")

    def _scale_transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        N, T, D = X.shape
        flat = X.reshape(-1, D)
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.scaler = StandardScaler()
            flat_scaled = self.scaler.fit_transform(flat)
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler not fit.")
            flat_scaled = self.scaler.transform(flat)
        return flat_scaled.reshape(N, T, D)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "GRUForecaster":
        X_tr = _ensure_3d(X_train)
        y_tr = np.asarray(y_train, dtype=np.float32).flatten()
        X_tr_scaled = self._scale_transform(X_tr, fit=True)

        train_ds = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = _ensure_3d(X_val)
            y_v = np.asarray(y_val, dtype=np.float32).flatten()
            X_v_scaled = self._scale_transform(X_v, fit=False)
            val_ds = TensorDataset(
                torch.tensor(X_v_scaled, dtype=torch.float32),
                torch.tensor(y_v, dtype=torch.float32),
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False
            )

        input_dim = X_tr.shape[2]
        self.models = []

        # Train 5 seeds averaged
        for s_idx in range(self.n_seeds):
            torch.manual_seed(self.seed + s_idx)
            np.random.seed(self.seed + s_idx)

            module = GRUModule(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=1,
                output_dim=1,
                dropout_rate=0.2,
            )
            trained_model, _ = _train_pytorch_model(
                model=module,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.epochs,
                lr=self.lr,
                patience=self.patience,
                max_norm=1.0,
                device=self.device,
            )
            trained_model.eval()
            self.models.append(trained_model)

        self.is_fitted = True
        train_preds = self.predict(X_train)
        self.train_residuals = y_tr - train_preds

        if X_val is not None and y_val is not None and len(y_val) > 0:
            val_preds = self.predict(X_val)
            self.val_residuals = np.asarray(y_val).flatten() - val_preds

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or not self.models:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        X_3d = _ensure_3d(X)
        X_scaled = self._scale_transform(X_3d, fit=False)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        preds_list = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                p = m(x_tensor).cpu().numpy().flatten()
                preds_list.append(p)

        return np.mean(preds_list, axis=0)


class DLinearForecaster(BaseForecaster):
    """
    DLinear forecaster per Zeng et al. (AAAI 2023).
    Decomposes series into Trend and Remainder components and applies
    dual linear layers with early stopping and gradient clipping.
    """

    def __init__(
        self,
        name: str = "dlinear",
        seed: int = 42,
        kernel_size: int = 13,
        epochs: int = 30,
        lr: float = 0.003,
        patience: int = 6,
        batch_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(name=name, seed=seed)
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cpu")

    def _scale_transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        N, T, D = X.shape
        flat = X.reshape(-1, D)
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.scaler = StandardScaler()
            flat_scaled = self.scaler.fit_transform(flat)
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler not fit.")
            flat_scaled = self.scaler.transform(flat)
        return flat_scaled.reshape(N, T, D)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "DLinearForecaster":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_tr = _ensure_3d(X_train)
        y_tr = np.asarray(y_train, dtype=np.float32).flatten()
        X_tr_scaled = self._scale_transform(X_tr, fit=True)

        train_ds = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            X_v = _ensure_3d(X_val)
            y_v = np.asarray(y_val, dtype=np.float32).flatten()
            X_v_scaled = self._scale_transform(X_v, fit=False)
            val_ds = TensorDataset(
                torch.tensor(X_v_scaled, dtype=torch.float32),
                torch.tensor(y_v, dtype=torch.float32),
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False
            )

        seq_len = X_tr.shape[1]
        input_dim = X_tr.shape[2]

        module = DLinearModule(
            seq_len=seq_len,
            input_dim=input_dim,
            kernel_size=self.kernel_size,
        )

        self.model, _ = _train_pytorch_model(
            model=module,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            lr=self.lr,
            patience=self.patience,
            max_norm=1.0,
            device=self.device,
        )
        self.model.eval()
        self.is_fitted = True

        train_preds = self.predict(X_train)
        self.train_residuals = y_tr - train_preds

        if X_val is not None and y_val is not None and len(y_val) > 0:
            val_preds = self.predict(X_val)
            self.val_residuals = np.asarray(y_val).flatten() - val_preds

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        X_3d = _ensure_3d(X)
        X_scaled = self._scale_transform(X_3d, fit=False)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            self.model.eval()
            pred = self.model(x_tensor).cpu().numpy().flatten()
        return pred

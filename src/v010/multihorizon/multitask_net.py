"""
StockReason v0.10 — Multi-Task Neural Network for Joint Multi-Horizon Forecasting.
===================================================================================
Remediates Phase 4 (Section 9):
1. Shared Feature Encoder:
   Input (21 features) -> Linear(64) -> ReLU -> LayerNorm -> Dropout -> Linear(32) -> ReLU
2. Five Independent Horizon Output Heads:
   - Head 1D
   - Head 5D
   - Head 21D (Primary)
   - Head 63D
   - Head 126D
3. Normalized Multi-Horizon Loss:
   Normalizes L1 loss by standard deviation scaling sqrt(H / 252) so longer horizons
   do not dominate gradient backpropagation.
4. Target Masking:
   Ignores observations where a specific horizon's forward target is NaN / masked.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import torch.nn as nn

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.v092.models.baselines import compute_absolute_price_metrics


HORIZONS = [1, 5, 21, 63, 126]


class MultiTaskHorizonNet(nn.Module):
    def __init__(self, input_dim: int = 21, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            str(h): nn.Linear(latent_dim, 1) for h in HORIZONS
        })

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        latent = self.encoder(x)
        outputs = {}
        for h in HORIZONS:
            outputs[h] = self.heads[str(h)](latent).squeeze(-1)
        return outputs


def train_multitask_neural_network(
    X_train: np.ndarray,
    y_train_by_horizon: Dict[int, np.ndarray],
    X_val: np.ndarray,
    y_val_by_horizon: Dict[int, np.ndarray],
    val_close: np.ndarray,
    batch_size: int = 2048,
    epochs: int = 4,
    lr: float = 0.003,
    sample_limit: int = 60000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains the multi-task neural network on normalized multi-horizon return targets.
    Evaluates out-of-sample metrics for each horizon independently.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    input_dim = X_train.shape[1]
    net = MultiTaskHorizonNet(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Subsample training data for memory bounded fast convergence
    n_tr = len(X_train)
    sub_size = min(sample_limit, n_tr)
    sub_idx = np.random.choice(n_tr, size=sub_size, replace=False)

    X_tr_t = torch.from_numpy(X_train[sub_idx]).float().to(device)

    # Target tensors & valid masks
    y_tr_tensors = {}
    masks_tensors = {}
    vol_scalers = {}
    for h in HORIZONS:
        arr = y_train_by_horizon[h][sub_idx]
        valid = ~np.isnan(arr)
        clean_arr = np.where(valid, arr, 0.0)
        y_tr_tensors[h] = torch.from_numpy(clean_arr).float().to(device)
        masks_tensors[h] = torch.from_numpy(valid.astype(np.float32)).to(device)
        # Normalization scale: annual vol scaled by sqrt(h / 252)
        vol_scalers[h] = float(np.sqrt(h / 252.0))

    net.train()
    n_batches = sub_size // batch_size
    for epoch in range(epochs):
        for b in range(n_batches):
            idx = slice(b * batch_size, (b + 1) * batch_size)
            optimizer.zero_grad()
            outputs = net(X_tr_t[idx])

            loss = torch.tensor(0.0, device=device)
            for h in HORIZONS:
                diff = torch.abs(outputs[h] - y_tr_tensors[h][idx])
                masked_diff = diff * masks_tensors[h][idx]
                n_valid = torch.sum(masks_tensors[h][idx]) + 1e-5
                # Normalized by vol_scalers so 126d doesn't overwhelm 1d
                norm_loss_h = torch.sum(masked_diff) / (n_valid * vol_scalers[h])
                loss = loss + norm_loss_h

            loss.backward()
            optimizer.step()

    # Out-of-sample evaluation across all horizons
    net.eval()
    val_preds_by_horizon = {h: [] for h in HORIZONS}
    with torch.no_grad():
        for b in range(int(np.ceil(len(X_val) / 4096))):
            b_slice = slice(b * 4096, (b + 1) * 4096)
            X_v_b = torch.from_numpy(X_val[b_slice]).float().to(device)
            out_b = net(X_v_b)
            for h in HORIZONS:
                val_preds_by_horizon[h].append(out_b[h].cpu().numpy())

    final_preds = {h: np.concatenate(val_preds_by_horizon[h]) for h in HORIZONS}

    metrics_table = []
    for h in HORIZONS:
        y_true = y_val_by_horizon[h]
        y_pred = final_preds[h]
        valid = ~np.isnan(y_true)
        if valid.sum() > 0:
            m = compute_absolute_price_metrics(y_true[valid], y_pred[valid], val_close[valid])
            metrics_table.append({
                "model": "Multi-Task Neural Network",
                "horizon": f"{h}d",
                "valid_eval_obs": int(valid.sum()),
                **m,
            })

    return {
        "metrics_table": pd.DataFrame(metrics_table),
        "predictions": final_preds,
        "model": net,
    }

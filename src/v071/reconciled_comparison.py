"""
StockReason v0.7.1 - Workstream A4: Reconciled Model Comparison on Large Universe
=================================================================================
Runs 8 purged expanding-window walk-forward folds across U100 (96 tickers) and U16
on non-lockbox data. Evaluates all models with cross-sectional per-date Rank IC,
reconciles with scaling experiment, and computes the scaling grid.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.v07.common.lockbox import load_panel, get_lockbox_date_bounds
from src.v07.models.purged_cv import PurgedWalkForwardCV
from src.v071.metrics import compute_ic, summarize_ic, compute_directional_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("A4_Comparison")


# Deep Learning PyTorch Models
class SimpleGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, 1, input_dim)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class SimpleDLinear(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x[:, -1, :]).squeeze(-1)


def fit_torch_model(
    model_type: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.003,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_dim = X_tr.shape[1]
    if model_type == "gru":
        net = SimpleGRU(input_dim)
    elif model_type == "lstm":
        net = SimpleLSTM(input_dim)
    elif model_type == "dlinear":
        net = SimpleDLinear(input_dim)
    else:
        raise ValueError(model_type)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Reshape to (batch, 1, input_dim)
    t_X_tr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1)
    t_y_tr = torch.tensor(y_tr, dtype=torch.float32)
    t_X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    t_y_val = torch.tensor(y_val, dtype=torch.float32)
    t_X_te = torch.tensor(X_te, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(t_X_tr, t_y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        net.train()
        for bx, by in loader:
            optimizer.zero_grad()
            pred = net(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            v_pred = net(t_X_val)
            v_loss = criterion(v_pred, t_y_val).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_weights = {k: v.clone() for k, v in net.state_dict().items()}

    if best_weights is not None:
        net.load_state_dict(best_weights)
    net.eval()
    with torch.no_grad():
        test_pred = net(t_X_te).numpy()
    return test_pred


def run_reconciled_benchmark(
    panel_path: str = "data/v0.7.1/panel_u100.parquet",
    horizons: List[int] = [5, 21],
    n_folds: int = 8,
) -> Dict[str, Any]:
    logger.info("Loading U100 panel from %s...", panel_path)
    df = pd.read_parquet(panel_path)
    df["date"] = pd.to_datetime(df["date"])

    # Enforce non-lockbox cutoff
    lockbox_start, embargo_start, max_date = get_lockbox_date_bounds(df)
    logger.info("Excluding lockbox: dropping dates >= %s. Panel: %d rows.", embargo_start.strftime("%Y-%m-%d"), len(df))
    clean_df = df[df["date"] < embargo_start].copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info("Non-lockbox data: %d rows across %d dates and %d tickers.", len(clean_df), clean_df["date"].nunique(), clean_df["ticker"].nunique())

    feature_cols = [
        "return_1d", "return_2d", "return_3d", "return_5d", "return_10d", "return_21d",
        "vol_20d", "vol_60d", "natr_14", "natr_20", "dist_sma_10", "dist_sma_20",
        "dist_sma_50", "dist_sma_100", "dist_sma_200", "rsi_14", "rsi_28",
        "bb_pct_b_20", "bb_width_20", "bb_width_50", "sma_50_200_ratio", "vol_ratio_20", "high_low_spread"
    ]
    feature_cols = [c for c in feature_cols if c in clean_df.columns]

    cv = PurgedWalkForwardCV(n_folds=n_folds, val_ratio=0.10, test_ratio=0.10, purge_window=21, min_train_ratio=0.40)
    splits = list(cv.split_indices(clean_df["date"]))
    logger.info("Generated %d purged walk-forward folds.", len(splits))

    models = ["zero", "market", "ticker_mean", "ridge", "lightgbm", "catboost", "dlinear", "gru", "lstm"]
    all_results = {}

    for h in horizons:
        target_col = f"y_{h}d"
        logger.info("\n======================== EVALUATING HORIZON %dD ========================", h)
        sub_df = clean_df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
        h_splits = list(cv.split_indices(sub_df["date"]))

        fold_preds = {m: [] for m in models}
        fold_ics = {m: [] for m in models}
        fold_test_dates = []

        for fold_id, (tr_idx, val_idx, te_idx) in enumerate(h_splits):
            logger.info("Fold %d/%d: Train=%d, Val=%d, Test=%d", fold_id + 1, len(h_splits), len(tr_idx), len(val_idx), len(te_idx))
            tr = sub_df.iloc[tr_idx]
            val = sub_df.iloc[val_idx]
            te = sub_df.iloc[te_idx]

            X_tr = tr[feature_cols].fillna(0).to_numpy(dtype=np.float32)
            y_tr = tr[target_col].to_numpy(dtype=np.float32)
            X_val = val[feature_cols].fillna(0).to_numpy(dtype=np.float32)
            y_val = val[target_col].to_numpy(dtype=np.float32)
            X_te = te[feature_cols].fillna(0).to_numpy(dtype=np.float32)
            y_te = te[target_col].to_numpy(dtype=np.float32)
            te_dates = te["date"].values

            # 1. Zero Forecaster
            p_zero = np.zeros(len(te), dtype=float)
            fold_preds["zero"].append(pd.DataFrame({"date": te_dates, "pred": p_zero, "target": y_te, "fold": fold_id + 1}))

            # 2. Market Forecaster (strictly constant cross-sectionally)
            mkt_mean = float(np.mean(y_tr))
            p_mkt = np.full(len(te), mkt_mean, dtype=float)
            fold_preds["market"].append(pd.DataFrame({"date": te_dates, "pred": p_mkt, "target": y_te, "fold": fold_id + 1}))

            # 3. Ticker Mean Forecaster
            tck_means = tr.groupby("ticker")[target_col].mean().to_dict()
            p_tck = te["ticker"].map(tck_means).fillna(0.0).to_numpy(dtype=float)
            fold_preds["ticker_mean"].append(pd.DataFrame({"date": te_dates, "pred": p_tck, "target": y_te, "fold": fold_id + 1}))

            # 4. Ridge
            clf_ridge = Ridge(alpha=10.0)
            clf_ridge.fit(X_tr, y_tr)
            p_ridge = clf_ridge.predict(X_te)
            fold_preds["ridge"].append(pd.DataFrame({"date": te_dates, "pred": p_ridge, "target": y_te, "fold": fold_id + 1}))

            # 5. LightGBM (3 seeds)
            lgb_seeds = [42, 101, 2024]
            p_lgb_seeds = []
            for s in lgb_seeds:
                clf_lgb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=31, random_state=s, n_jobs=1, verbose=-1)
                clf_lgb.fit(X_tr, y_tr)
                p_lgb_seeds.append(clf_lgb.predict(X_te))
            p_lgb = np.mean(p_lgb_seeds, axis=0)
            fold_preds["lightgbm"].append(pd.DataFrame({"date": te_dates, "pred": p_lgb, "target": y_te, "fold": fold_id + 1}))

            # 6. CatBoost (3 seeds)
            cb_seeds = [42, 101, 2024]
            p_cb_seeds = []
            for s in cb_seeds:
                clf_cb = CatBoostRegressor(iterations=200, learning_rate=0.03, depth=5, random_seed=s, thread_count=1, verbose=0)
                clf_cb.fit(X_tr, y_tr)
                p_cb_seeds.append(clf_cb.predict(X_te))
            p_cb = np.mean(p_cb_seeds, axis=0)
            fold_preds["catboost"].append(pd.DataFrame({"date": te_dates, "pred": p_cb, "target": y_te, "fold": fold_id + 1}))

            # 7. DLinear
            p_dl = fit_torch_model("dlinear", X_tr, y_tr, X_val, y_val, X_te, seed=42)
            fold_preds["dlinear"].append(pd.DataFrame({"date": te_dates, "pred": p_dl, "target": y_te, "fold": fold_id + 1}))

            # 8. GRU
            p_gru = fit_torch_model("gru", X_tr, y_tr, X_val, y_val, X_te, seed=42)
            fold_preds["gru"].append(pd.DataFrame({"date": te_dates, "pred": p_gru, "target": y_te, "fold": fold_id + 1}))

            # 9. LSTM
            p_lstm = fit_torch_model("lstm", X_tr, y_tr, X_val, y_val, X_te, seed=42)
            fold_preds["lstm"].append(pd.DataFrame({"date": te_dates, "pred": p_lstm, "target": y_te, "fold": fold_id + 1}))

        # Aggregate across folds
        all_results[f"{h}d"] = {}
        for m in models:
            df_concat = pd.concat(fold_preds[m], ignore_index=True)
            ic_df = compute_ic(df_concat, min_names_per_date=20)
            sum_m = summarize_ic(ic_df, horizon=h)
            dir_res = compute_directional_accuracy(df_concat)
            mae = float(np.mean(np.abs(df_concat["target"] - df_concat["pred"])))
            rmse = float(np.sqrt(np.mean((df_concat["target"] - df_concat["pred"]) ** 2)))

            # Per fold breakdown
            per_fold_ic = {}
            for f_id in range(1, len(h_splits) + 1):
                f_sub = df_concat[df_concat["fold"] == f_id]
                f_ic_df = compute_ic(f_sub, min_names_per_date=20)
                per_fold_ic[f"fold_{f_id}"] = float(f_ic_df["ic"].mean()) if not f_ic_df.empty else 0.0

            all_results[f"{h}d"][m] = {
                "mean_ic": sum_m["mean_ic"],
                "std_ic": sum_m["std_ic"],
                "icir_daily": sum_m["icir_daily"],
                "icir_annualized": sum_m["icir_annualized"],
                "t_stat": sum_m["t_stat"],
                "p_val": sum_m["p_val"],
                "mae": mae,
                "rmse": rmse,
                "dir_acc": dir_res["dir_acc"],
                "base_rate": dir_res["pos_base_rate"],
                "n_dates": sum_m["n_dates"],
                "per_fold_ic": per_fold_ic,
            }

    # Compare U16 vs U100 on CatBoost 21D
    logger.info("Computing U16 vs U100 comparison...")
    u16_tickers = [
        "AUBANK.NS", "AXISBANK.NS", "BANKBARODA.NS", "BHARTIARTL.NS", "HDFCBANK.NS",
        "ICICIBANK.NS", "IDFCFIRSTB.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
        "KOTAKBANK.NS", "LT.NS", "PNB.NS", "RELIANCE.NS", "SBIN.NS", "TCS.NS"
    ]
    df_u16 = clean_df[clean_df["ticker"].isin(u16_tickers)].dropna(subset=["y_21d"] + feature_cols).reset_index(drop=True)
    splits_u16 = list(cv.split_indices(df_u16["date"]))
    u16_cb_preds = []
    for f_id, (tr_idx, val_idx, te_idx) in enumerate(splits_u16):
        tr = df_u16.iloc[tr_idx]
        te = df_u16.iloc[te_idx]
        X_tr = tr[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        y_tr = tr["y_21d"].to_numpy(dtype=np.float32)
        X_te = te[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        y_te = te["y_21d"].to_numpy(dtype=np.float32)
        clf = CatBoostRegressor(iterations=200, learning_rate=0.03, depth=5, random_seed=42, thread_count=1, verbose=0)
        clf.fit(X_tr, y_tr)
        u16_cb_preds.append(pd.DataFrame({"date": te["date"].values, "pred": clf.predict(X_te), "target": y_te}))
    df_u16_cb = pd.concat(u16_cb_preds, ignore_index=True)
    ic_u16_df = compute_ic(df_u16_cb, min_names_per_date=10)
    u16_ic_summary = summarize_ic(ic_u16_df, horizon=21)

    all_results["u16_vs_u100_comparison"] = {
        "u16_catboost_21d_mean_ic": u16_ic_summary["mean_ic"],
        "u16_catboost_21d_icir_ann": u16_ic_summary["icir_annualized"],
        "u100_catboost_21d_mean_ic": all_results["21d"]["catboost"]["mean_ic"],
        "u100_catboost_21d_icir_ann": all_results["21d"]["catboost"]["icir_annualized"],
    }

    # Data fraction curve (25%, 50%, 75%, 100%)
    logger.info("Computing data fraction scaling curve...")
    data_fraction_res = {}
    last_split = splits[-1]
    tr_idx, _, te_idx = last_split
    te = clean_df.iloc[te_idx].dropna(subset=["y_21d"] + feature_cols)
    X_te = te[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y_te = te["y_21d"].to_numpy(dtype=np.float32)

    for frac in [0.25, 0.50, 0.75, 1.0]:
        n_sample = int(len(tr_idx) * frac)
        sub_tr = clean_df.iloc[tr_idx[:n_sample]].dropna(subset=["y_21d"] + feature_cols)
        X_tr = sub_tr[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        y_tr = sub_tr["y_21d"].to_numpy(dtype=np.float32)
        clf = Ridge(alpha=10.0)
        clf.fit(X_tr, y_tr)
        p = clf.predict(X_te)
        ic_res = float(compute_ic(pd.DataFrame({"date": te["date"].values, "pred": p, "target": y_te}), min_names_per_date=20)["ic"].mean())
        data_fraction_res[f"{int(frac*100)}%"] = ic_res
    all_results["data_fraction_curve"] = data_fraction_res

    os.makedirs("artifacts/v0.7.1", exist_ok=True)
    with open("artifacts/v0.7.1/reconciled_model_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved reconciled results to artifacts/v0.7.1/reconciled_model_comparison.json")

    return all_results


if __name__ == "__main__":
    run_reconciled_benchmark()

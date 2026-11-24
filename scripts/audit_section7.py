"""
Section 7 Auditor: Model Evaluation Baselines, Stratifications, and Calibration.
Generates:
- reports/diagnostics/evaluation_baselines_recomputed.csv
- reports/diagnostics/non_overlapping_ic_distribution.csv
- reports/diagnostics/ic_by_year_sector_tercile_regime.csv
- reports/diagnostics/interval_and_prob_calibration.json
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, matthews_corrcoef, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostRegressor

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        pred = self.fc(out[:, -1, :])
        return pred.squeeze(-1)

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return pred.squeeze(-1)

def extract_feature_cols(df: pd.DataFrame, feature_set: str = "B"):
    set_a = [
        "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d",
        "vol_20d", "vol_60d", "dist_ma20", "dist_ma50", "dist_ma200",
        "rsi_14", "macd_diff", "hl_spread", "vol_ratio_20d", "log_turnover"
    ]
    set_b = set_a + ["nifty_ret_1d", "nifty_ret_21d", "vix_level", "vix_change_5d", "rel_strength_21d"]
    return [c for c in set_b if c in df.columns]

def compute_daily_rank_ic(df_sub: pd.DataFrame, pred_col: str, target_col: str, min_stocks: int = 10) -> pd.DataFrame:
    records = []
    for d, group in df_sub.groupby("date_str"):
        sub = group.dropna(subset=[pred_col, target_col])
        if len(sub) >= min_stocks:
            if np.all(sub[pred_col] == sub[pred_col].iloc[0]):
                records.append({"date": d, "ic": 0.0, "n_stocks": len(sub)})
                continue
            ic, _ = stats.spearmanr(sub[pred_col], sub[target_col])
            if not np.isnan(ic):
                records.append({"date": d, "ic": float(ic), "n_stocks": len(sub)})
    return pd.DataFrame(records)

def block_bootstrap_ci(ic_series: np.ndarray, block_size: int = 21, n_bootstrap: int = 1000, alpha: float = 0.05):
    n = len(ic_series)
    if n == 0 or np.all(ic_series == 0.0):
        return 0.0, 0.0
    n_blocks = int(np.ceil(n / block_size))
    boot_means = []
    for _ in range(n_bootstrap):
        start_indices = np.random.randint(0, max(1, n - block_size + 1), size=n_blocks)
        sample = []
        for idx in start_indices:
            sample.extend(ic_series[idx : idx + block_size])
        sample = sample[:n]
        boot_means.append(np.mean(sample))
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper

def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: float, df_eval: pd.DataFrame, pred_col_name: str, target_col_name: str, horizon: int):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # Out of sample R2 (Campbell-Thompson benchmarked against training mean)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot_train = np.sum((y_true - y_train_mean) ** 2)
    oos_r2 = float(1.0 - (ss_res / (ss_tot_train + 1e-9)))

    # Directional metrics: define up as > 0, down as <= 0
    actual_dir = (y_true > 0).astype(int)
    pred_dir = (y_pred > 0).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(actual_dir, pred_dir, labels=[0, 1]).ravel()
    n_total = len(y_true)
    
    dir_acc = float((tp + tn) / n_total * 100.0)
    recall_up = float(tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    recall_down = float(tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    specificity = recall_down
    balanced_acc = float(0.5 * (recall_up + recall_down))
    
    mcc_val = float(matthews_corrcoef(actual_dir, pred_dir)) if len(np.unique(actual_dir)) > 1 and len(np.unique(pred_dir)) > 1 else 0.0

    # Rank IC
    sub_df = pd.DataFrame({"date_str": df_eval["date_str"].values, pred_col_name: y_pred, target_col_name: y_true})
    ic_df = compute_daily_rank_ic(sub_df, pred_col_name, target_col_name)
    if not ic_df.empty:
        mean_ic = float(ic_df["ic"].mean())
        std_ic = float(ic_df["ic"].std())
        icir = float(mean_ic / (std_ic + 1e-9) * np.sqrt(252 / horizon))
        ci_low, ci_high = block_bootstrap_ci(ic_df["ic"].values, block_size=horizon)
    else:
        mean_ic, icir, ci_low, ci_high = 0.0, 0.0, 0.0, 0.0

    return {
        "mean_rank_ic": round(mean_ic, 4),
        "ic_ci_lower_95": round(ci_low, 4),
        "ic_ci_upper_95": round(ci_high, 4),
        "icir": round(icir, 2),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "oos_r2": round(oos_r2, 4),
        "dir_acc_pct": round(dir_acc, 2),
        "balanced_acc_pct": round(balanced_acc, 2),
        "mcc": round(mcc_val, 4),
        "recall_up": round(recall_up, 2),
        "recall_down": round(recall_down, 2),
        "specificity": round(specificity, 2),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_obs": int(n_total)
    }

def train_gru(X_tr, y_tr, input_dim: int, epochs=3, batch_size=512):
    idx = np.random.choice(len(X_tr), min(20000, len(X_tr)), replace=False)
    ds = TensorDataset(torch.tensor(X_tr[idx, None, :], dtype=torch.float32), torch.tensor(y_tr[idx], dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    m = GRUForecaster(input_dim=input_dim, hidden_dim=32, num_layers=1)
    opt = torch.optim.Adam(m.parameters(), lr=0.005, weight_decay=1e-5)
    crit = nn.MSELoss()
    m.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            p = m(bx)
            loss = crit(p, by)
            loss.backward()
            opt.step()
    return m

def train_lstm(X_tr, y_tr, input_dim: int, epochs=3, batch_size=512):
    idx = np.random.choice(len(X_tr), min(20000, len(X_tr)), replace=False)
    ds = TensorDataset(torch.tensor(X_tr[idx, None, :], dtype=torch.float32), torch.tensor(y_tr[idx], dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    m = LSTMForecaster(input_dim=input_dim, hidden_dim=32, num_layers=1)
    opt = torch.optim.Adam(m.parameters(), lr=0.005, weight_decay=1e-5)
    crit = nn.MSELoss()
    m.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            p = m(bx)
            loss = crit(p, by)
            loss.backward()
            opt.step()
    return m

def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X[:, None, :], dtype=torch.float32)
        out = model(t)
        return out.cpu().numpy().flatten()

# Robust sector map for universe
SECTOR_MAP = {
    'RELIANCE': 'Energy', 'TCS': 'Information Technology', 'INFY': 'Information Technology',
    'HDFCBANK': 'Financial Services', 'ICICIBANK': 'Financial Services', 'BHARTIARTL': 'Telecommunication',
    'ITC': 'Consumer Goods', 'LT': 'Construction', 'SBIN': 'Financial Services', 'AXISBANK': 'Financial Services',
    'KOTAKBANK': 'Financial Services', 'INDUSINDBK': 'Financial Services', 'BANKBARODA': 'Financial Services',
    'PNB': 'Financial Services', 'HINDUNILVR': 'Consumer Goods', 'MARUTI': 'Automobile', 'M&M': 'Automobile',
    'BAJAJ-AUTO': 'Automobile', 'EICHERMOT': 'Automobile', 'TATASTEEL': 'Metals & Mining', 'JSWSTEEL': 'Metals & Mining',
    'HINDALCO': 'Metals & Mining', 'COALINDIA': 'Energy', 'ONGC': 'Energy', 'NTPC': 'Power', 'POWERGRID': 'Power',
    'SUNPHARMA': 'Healthcare', 'CIPLA': 'Healthcare', 'DRREDDY': 'Healthcare', 'DIVISLAB': 'Healthcare',
    'APOLLOHOSP': 'Healthcare', 'TITAN': 'Consumer Goods', 'ASIANPAINT': 'Consumer Goods', 'ULTRACEMCO': 'Construction Materials',
    'GRASIM': 'Construction Materials', 'TECHM': 'Information Technology', 'HCLTECH': 'Information Technology',
    'WIPRO': 'Information Technology', 'LTIM': 'Information Technology', 'BAJFINANCE': 'Financial Services',
    'BAJAJFINSV': 'Financial Services', 'CHOLAFIN': 'Financial Services', 'SHRIRAMFIN': 'Financial Services',
    'HDFCLIFE': 'Financial Services', 'SBILIFE': 'Financial Services', 'TATACONSUM': 'Consumer Goods',
    'BRITANNIA': 'Consumer Goods', 'NESTLEIND': 'Consumer Goods', 'PIDILITIND': 'Chemicals', 'SIEMENS': 'Capital Goods',
    'ABB': 'Capital Goods', 'BEL': 'Capital Goods', 'HAL': 'Capital Goods', 'ADANIENT': 'Services',
    'ADANIPORTS': 'Services', 'TRENT': 'Consumer Goods', 'VBL': 'Consumer Goods', 'ZOMATO': 'Services',
    'JIOFIN': 'Financial Services', 'DLF': 'Realty', 'VEDL': 'Metals & Mining', 'AMBUJACEM': 'Construction Materials',
    'BPCL': 'Energy', 'GAIL': 'Energy', 'CANBK': 'Financial Services', 'RECLTD': 'Financial Services',
    'PFC': 'Financial Services', 'IDFCFIRSTB': 'Financial Services', 'AUBANK': 'Financial Services',
    'MUTHOOTFIN': 'Financial Services', 'ICICIGI': 'Financial Services', 'ICICIPRULI': 'Financial Services',
    'PERSISTENT': 'Information Technology', 'COFORGE': 'Information Technology', 'MPHASIS': 'Information Technology',
    'TATAELXSI': 'Information Technology', 'KPITTECH': 'Information Technology', 'LUPIN': 'Healthcare',
    'AUROPHARMA': 'Healthcare', 'ALKEM': 'Healthcare', 'TORNTPHARM': 'Healthcare', 'MANKIND': 'Healthcare',
    'MAXHEALTH': 'Healthcare', 'GODREJCP': 'Consumer Goods', 'DABUR': 'Consumer Goods', 'MARICO': 'Consumer Goods',
    'COLPAL': 'Consumer Goods', 'PGHH': 'Consumer Goods', 'BERGEPAINT': 'Consumer Goods', 'POLYCAB': 'Capital Goods',
    'HAVELLS': 'Consumer Durables', 'BOSCHLTD': 'Automobile', 'TVSMOTOR': 'Automobile', 'ASHOKLEY': 'Automobile',
    'BHARATFORG': 'Capital Goods', 'CUMMINSIND': 'Capital Goods', 'ASTRAL': 'Building Materials',
    'APLAPOLLO': 'Metals & Mining', 'LODHA': 'Realty', 'INDIGO': 'Services', 'ADANIPOWER': 'Power',
    'ADANIGREEN': 'Power', 'ATGL': 'Energy',
    # Additional 27 symbols in panel
    'ATHERENERG': 'Automobile', 'TMCV': 'Automobile', 'TMPV': 'Automobile', 'OLAELEC': 'Automobile',
    'CUPID': 'Healthcare', 'PCJEWELLER': 'Consumer Goods', 'BLUESTONE': 'Consumer Goods', 'MCX': 'Financial Services',
    'IDEA': 'Telecommunication', 'GROWW': 'Financial Services', 'ADANIENSOL': 'Power', 'SILVERBEES': 'ETF/Commodities',
    'GOLDBEES': 'ETF/Commodities', 'LIQUIDBEES': 'ETF/Debt', 'WELCORP': 'Capital Goods', 'LENSKART': 'Consumer Goods',
    'IFCI': 'Financial Services', 'GVT&D': 'Capital Goods', 'POWERINDIA': 'Capital Goods', 'VAML': 'Financial Services',
    'HSCL': 'Chemicals', 'SWIGGY': 'Services', 'PINELABS': 'Financial Services', 'APOLLO': 'Healthcare',
    'ETERNAL': 'Services', 'NETWEB': 'Information Technology', 'MEESHO': 'Services',
    'CHENNPETRO': 'Energy', 'MRPL': 'Energy', 'BSE': 'Financial Services', 'DATAPATTNS': 'Capital Goods',
    'HINDCOPPER': 'Metals & Mining', 'HFCL': 'Telecommunication', 'SYRMA': 'Information Technology',
    'LAURUSLABS': 'Healthcare', 'PAYTM': 'Financial Services', 'POLICYBZR': 'Financial Services',
    'NAUKRI': 'Services', 'KAYNES': 'Information Technology', 'DIXON': 'Consumer Durables',
    'HDFC': 'Financial Services', 'RCOM': 'Telecommunication', 'JETAIRWAYS': 'Services'
}

def run_section7_audit():
    start_t = time.time()
    print("=" * 70)
    print("STARTING SECTION 7 AUDIT: BASELINE BENCHMARKS & CALIBRATION")
    print("=" * 70)

    os.makedirs("reports/diagnostics", exist_ok=True)

    # 1. Load data
    panel_df = pd.read_parquet("data/v0.9/features/panel_features.parquet")
    manifest = pd.read_parquet("data/v0.9/splits/split_manifest.parquet")
    panel_df["date_str"] = pd.to_datetime(panel_df["date"]).dt.strftime("%Y-%m-%d")
    panel_df["year"] = pd.to_datetime(panel_df["date"]).dt.year

    panel_df = pd.merge(panel_df, manifest[["security_id", "forecast_origin_date", "partition"]],
                        left_on=["symbol", "date_str"], right_on=["security_id", "forecast_origin_date"], how="left")

    feat_cols = extract_feature_cols(panel_df, "B")
    print(f"Features: {len(feat_cols)} columns")

    # Baseline tables collector
    baseline_records = []

    # Map sector
    panel_df["sector"] = panel_df["symbol"].map(SECTOR_MAP).fillna("Other")

    # Partitions
    train_df = panel_df[panel_df["partition"] == "TRAIN"].copy()
    val_df = panel_df[panel_df["partition"] == "DEV_VAL"].copy()
    eval_df = panel_df[panel_df["partition"] == "HIST_EVAL"].copy()

    # Pre-calculated ticker training means
    ticker_mean_21d = train_df.groupby("symbol")["target_return_21d"].mean().to_dict()
    ticker_mean_5d = train_df.groupby("symbol")["target_return_5d"].mean().to_dict()

    # =========================================================================
    # PART 1: HORIZONS 21D and 5D EVALUATION TABLE
    # =========================================================================
    for horizon in [21, 5]:
        target_col = f"target_return_{horizon}d"
        target_price_col = f"target_price_{horizon}d"
        print(f"\n--- EVALUATING HORIZON: {horizon}D ---")

        # Clean sub-partitions on exact same securities/dates
        clean_train = train_df.dropna(subset=[target_col] + feat_cols).copy()
        clean_val = val_df.dropna(subset=[target_col] + feat_cols).copy()
        clean_eval = eval_df.dropna(subset=[target_col] + feat_cols).copy()

        print(f"Horizon {horizon}d: Train={len(clean_train):,}, Val={len(clean_val):,}, Eval={len(clean_eval):,}")

        # Training Mean constants
        train_mean_const = float(clean_train[target_col].mean())
        ticker_means = ticker_mean_21d if horizon == 21 else ticker_mean_5d

        # Feature Scaling
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(clean_train[feat_cols].values)
        y_tr = clean_train[target_col].values
        X_val = scaler.transform(clean_val[feat_cols].values)
        y_val = clean_val[target_col].values
        X_eval = scaler.transform(clean_eval[feat_cols].values)
        y_eval = clean_eval[target_col].values

        # Dictionary of predictions on clean_eval
        preds_dict = {}

        # 1. Zero (no change)
        preds_dict["Zero (no change)"] = np.zeros(len(clean_eval))

        # 2. Always-up
        preds_dict["Always-up"] = np.full(len(clean_eval), 0.01)

        # 3. Always-down
        preds_dict["Always-down"] = np.full(len(clean_eval), -0.01)

        # 4. Training-mean constant
        preds_dict["Training-mean constant"] = np.full(len(clean_eval), train_mean_const)

        # 5. Per-ticker training mean
        preds_dict["Per-ticker training mean"] = clean_eval["symbol"].map(ticker_means).fillna(train_mean_const).values

        # 6. Momentum (21D)
        preds_dict["Momentum (21D)"] = clean_eval["ret_21d"].fillna(0.0).values

        # 7. Momentum (126D)
        preds_dict["Momentum (126D)"] = clean_eval["ret_126d"].fillna(0.0).values

        # 8. Ridge Regression
        print("  Fitting Ridge...")
        ridge = Ridge(alpha=100.0, random_state=SEED)
        ridge.fit(X_tr, y_tr)
        preds_dict["Ridge"] = ridge.predict(X_eval)

        # 9. LightGBM
        print("  Fitting LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.03, num_leaves=31, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=1, verbose=-1
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(15, verbose=False)])
        preds_dict["LightGBM"] = lgb_model.predict(X_eval)

        # 10. CatBoost
        print("  Fitting CatBoost...")
        cat_model = CatBoostRegressor(iterations=150, learning_rate=0.03, depth=5, random_seed=SEED, verbose=0, thread_count=1)
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=15, verbose=False)
        preds_dict["CatBoost"] = cat_model.predict(X_eval)

        # 11. GRU
        print("  Fitting PyTorch GRU...")
        gru_m = train_gru(X_tr, y_tr, input_dim=len(feat_cols), epochs=3)
        preds_dict["GRU"] = predict_torch(gru_m, X_eval)

        # 12. LSTM
        print("  Fitting PyTorch LSTM...")
        lstm_m = train_lstm(X_tr, y_tr, input_dim=len(feat_cols), epochs=3)
        preds_dict["LSTM"] = predict_torch(lstm_m, X_eval)

        # Compute metrics for all models
        for m_name, y_p in preds_dict.items():
            metrics = compute_comprehensive_metrics(
                y_eval, y_p, train_mean_const, clean_eval, f"pred_{m_name}", target_col, horizon
            )
            row = {"horizon": f"{horizon}D", "model": m_name, **metrics}
            baseline_records.append(row)

        # Store 21d models and eval dataframe for subsequent steps
        if horizon == 21:
            eval_21d_df = clean_eval.copy()
            preds_21d_dict = preds_dict.copy()
            lgb_val_pred = lgb_model.predict(X_val)
            val_residuals_21d = y_val - lgb_val_pred
            lgb_21d = lgb_model
            cat_21d = cat_model
            ridge_21d = ridge
            scaler_21d = scaler
            gru_21d = gru_m
            lstm_21d = lstm_m

    # Save evaluation_baselines_recomputed.csv
    baseline_df = pd.DataFrame(baseline_records)
    baseline_df.to_csv("reports/diagnostics/evaluation_baselines_recomputed.csv", index=False)
    print("Saved reports/diagnostics/evaluation_baselines_recomputed.csv")

    # =========================================================================
    # PART 2: NON-OVERLAPPING DATES DISTRIBUTION (21D HORIZON)
    # =========================================================================
    print("\n--- COMPUTING NON-OVERLAPPING DATES DISTRIBUTION (21 OFFSETS) ---")
    all_dates_eval = sorted(eval_21d_df["date_str"].unique())
    print(f"Total trading dates in 21D eval: {len(all_dates_eval)}")

    non_overlap_rows = []
    # Test for each offset k in 0..20
    for offset in range(21):
        offset_dates = set(all_dates_eval[offset::21])
        sub_mask = eval_21d_df["date_str"].isin(offset_dates)
        sub_eval = eval_21d_df[sub_mask]
        y_true_sub = sub_eval["target_return_21d"].values

        for m_name in ["Zero (no change)", "Momentum (21D)", "Momentum (126D)", "Ridge", "LightGBM", "CatBoost", "GRU", "LSTM"]:
            y_pred_sub = preds_21d_dict[m_name][sub_mask]
            sub_preds_df = pd.DataFrame({"date_str": sub_eval["date_str"].values, "pred": y_pred_sub, "target": y_true_sub})
            ic_df = compute_daily_rank_ic(sub_preds_df, "pred", "target")
            mean_ic = float(ic_df["ic"].mean()) if not ic_df.empty else 0.0
            mae = float(mean_absolute_error(y_true_sub, y_pred_sub))
            rmse = float(np.sqrt(mean_squared_error(y_true_sub, y_pred_sub)))
            dir_acc = float(((y_true_sub > 0) == (y_pred_sub > 0)).mean() * 100.0)

            non_overlap_rows.append({
                "offset": offset,
                "n_dates": len(offset_dates),
                "n_obs": len(sub_eval),
                "model": m_name,
                "rank_ic": round(mean_ic, 4),
                "mae": round(mae, 6),
                "rmse": round(rmse, 6),
                "dir_acc_pct": round(dir_acc, 2)
            })

    non_overlap_df = pd.DataFrame(non_overlap_rows)
    non_overlap_df.to_csv("reports/diagnostics/non_overlapping_ic_distribution.csv", index=False)
    print("Saved reports/diagnostics/non_overlapping_ic_distribution.csv")

    # =========================================================================
    # PART 3 & 4: IC BREAKDOWNS (YEAR 2012-2026, SECTOR, TERCILE, REGIME)
    # =========================================================================
    print("\n--- COMPUTING STRATIFIED IC BREAKDOWNS ---")
    # For full 2012-2026 history: use panel_df with complete features
    clean_full = panel_df.dropna(subset=["target_return_21d"] + feat_cols).copy()
    print(f"Full cleaned panel rows: {len(clean_full):,}")

    # Scale using the 21d training scaler
    X_full = scaler_21d.transform(clean_full[feat_cols].values)
    y_full = clean_full["target_return_21d"].values

    clean_full["pred_ridge"] = ridge_21d.predict(X_full)
    clean_full["pred_lgb"] = lgb_21d.predict(X_full)
    clean_full["pred_cat"] = cat_21d.predict(X_full)
    clean_full["pred_gru"] = predict_torch(gru_21d, X_full)
    clean_full["pred_lstm"] = predict_torch(lstm_21d, X_full)
    clean_full["pred_mom21"] = clean_full["ret_21d"].fillna(0.0).values
    clean_full["pred_mom126"] = clean_full["ret_126d"].fillna(0.0).values

    # Terciles
    # 1. Market-Cap Proxy Tercile (based on turnover / traded value)
    clean_full["turnover_tercile"] = pd.qcut(clean_full["turnover"], q=3, labels=["Small", "Mid", "Large"])
    # 2. Volatility Tercile (based on vol_60d)
    clean_full["vol_tercile"] = pd.qcut(clean_full["vol_60d"], q=3, labels=["Low_Vol", "Mid_Vol", "High_Vol"])

    # 3. Regime based on trailing 6-month (126d) Nifty return:
    # If nifty_ret_126d > 0.05 -> Bull, < -0.05 -> Bear, else Sideways
    # Note: ret_126d of NIFTY 50
    nifty_df = pd.read_parquet("data/external/sector_indices/NIFTY_50.parquet").reset_index()
    nifty_col = "Close" if "Close" in nifty_df.columns else "close"
    d_col = "Date" if "Date" in nifty_df.columns else "date"
    nifty_df["date_str"] = pd.to_datetime(nifty_df[d_col]).dt.strftime("%Y-%m-%d")
    nifty_df["nifty_6m_ret"] = nifty_df[nifty_col].pct_change(126)
    
    clean_full = pd.merge(clean_full, nifty_df[["date_str", "nifty_6m_ret"]], on="date_str", how="left")
    clean_full["regime"] = np.where(
        clean_full["nifty_6m_ret"] > 0.05, "Bull",
        np.where(clean_full["nifty_6m_ret"] < -0.05, "Bear", "Sideways")
    )

    strat_records = []
    models_to_strat = [
        ("Momentum_21D", "pred_mom21"),
        ("Momentum_126D", "pred_mom126"),
        ("Ridge", "pred_ridge"),
        ("LightGBM", "pred_lgb"),
        ("CatBoost", "pred_cat"),
        ("GRU", "pred_gru"),
        ("LSTM", "pred_lstm")
    ]

    # A. Per-Year IC (2012 to 2026)
    for yr in sorted(clean_full["year"].unique()):
        if yr < 2012:
            continue
        sub_yr = clean_full[clean_full["year"] == yr]
        for m_name, p_col in models_to_strat:
            ic_df = compute_daily_rank_ic(sub_yr, p_col, "target_return_21d")
            m_ic = float(ic_df["ic"].mean()) if not ic_df.empty else 0.0
            icir = float(m_ic / (ic_df["ic"].std() + 1e-9) * np.sqrt(252 / 21)) if not ic_df.empty else 0.0
            strat_records.append({
                "stratification_type": "Per_Year",
                "segment": str(yr),
                "model": m_name,
                "mean_rank_ic": round(m_ic, 4),
                "icir": round(icir, 2),
                "n_dates": len(ic_df),
                "n_obs": len(sub_yr),
                "partition_context": "IN_SAMPLE_TRAIN" if yr <= 2022 else ("DEV_VAL" if yr <= 2024 else "HIST_EVAL_OOS")
            })

    # B. By Sector (Top sectors in OOS Eval)
    sub_eval_strat = clean_full[clean_full["partition"] == "HIST_EVAL"]
    top_sectors = sub_eval_strat["sector"].value_counts().head(8).index.tolist()
    for sec in top_sectors:
        sub_sec = sub_eval_strat[sub_eval_strat["sector"] == sec]
        for m_name, p_col in [("Momentum_21D", "pred_mom21"), ("Ridge", "pred_ridge"), ("LightGBM", "pred_lgb"), ("CatBoost", "pred_cat"), ("GRU", "pred_gru"), ("LSTM", "pred_lstm")]:
            ic_df = compute_daily_rank_ic(sub_sec, p_col, "target_return_21d", min_stocks=4)
            m_ic = float(ic_df["ic"].mean()) if not ic_df.empty else 0.0
            icir = float(m_ic / (ic_df["ic"].std() + 1e-9) * np.sqrt(252 / 21)) if not ic_df.empty else 0.0
            strat_records.append({
                "stratification_type": "Sector_OOS",
                "segment": sec,
                "model": m_name,
                "mean_rank_ic": round(m_ic, 4),
                "icir": round(icir, 2),
                "n_dates": len(ic_df),
                "n_obs": len(sub_sec),
                "partition_context": "HIST_EVAL_OOS"
            })

    # C. By Market-Cap Tercile (OOS Eval)
    for cap_grp in ["Small", "Mid", "Large"]:
        sub_cap = sub_eval_strat[sub_eval_strat["turnover_tercile"] == cap_grp]
        for m_name, p_col in [("Momentum_21D", "pred_mom21"), ("Ridge", "pred_ridge"), ("LightGBM", "pred_lgb"), ("CatBoost", "pred_cat"), ("GRU", "pred_gru"), ("LSTM", "pred_lstm")]:
            ic_df = compute_daily_rank_ic(sub_cap, p_col, "target_return_21d", min_stocks=4)
            m_ic = float(ic_df["ic"].mean()) if not ic_df.empty else 0.0
            icir = float(m_ic / (ic_df["ic"].std() + 1e-9) * np.sqrt(252 / 21)) if not ic_df.empty else 0.0
            strat_records.append({
                "stratification_type": "MarketCap_Tercile_OOS",
                "segment": cap_grp,
                "model": m_name,
                "mean_rank_ic": round(m_ic, 4),
                "icir": round(icir, 2),
                "n_dates": len(ic_df),
                "n_obs": len(sub_cap),
                "partition_context": "HIST_EVAL_OOS"
            })

    # D. By Volatility Tercile (OOS Eval)
    for vol_grp in ["Low_Vol", "Mid_Vol", "High_Vol"]:
        sub_vol = sub_eval_strat[sub_eval_strat["vol_tercile"] == vol_grp]
        for m_name, p_col in [("Momentum_21D", "pred_mom21"), ("Ridge", "pred_ridge"), ("LightGBM", "pred_lgb"), ("CatBoost", "pred_cat"), ("GRU", "pred_gru"), ("LSTM", "pred_lstm")]:
            ic_df = compute_daily_rank_ic(sub_vol, p_col, "target_return_21d", min_stocks=4)
            m_ic = float(ic_df["ic"].mean()) if not ic_df.empty else 0.0
            icir = float(m_ic / (ic_df["ic"].std() + 1e-9) * np.sqrt(252 / 21)) if not ic_df.empty else 0.0
            strat_records.append({
                "stratification_type": "Volatility_Tercile_OOS",
                "segment": vol_grp,
                "model": m_name,
                "mean_rank_ic": round(m_ic, 4),
                "icir": round(icir, 2),
                "n_dates": len(ic_df),
                "n_obs": len(sub_vol),
                "partition_context": "HIST_EVAL_OOS"
            })

    # E. By Macro Regime (OOS Eval)
    for reg in ["Bull", "Bear", "Sideways"]:
        sub_reg = sub_eval_strat[sub_eval_strat["regime"] == reg]
        if len(sub_reg) == 0:
            continue
        for m_name, p_col in [("Momentum_21D", "pred_mom21"), ("Ridge", "pred_ridge"), ("LightGBM", "pred_lgb"), ("CatBoost", "pred_cat"), ("GRU", "pred_gru"), ("LSTM", "pred_lstm")]:
            ic_df = compute_daily_rank_ic(sub_reg, p_col, "target_return_21d", min_stocks=4)
            m_ic = float(ic_df["ic"].mean()) if not ic_df.empty else 0.0
            icir = float(m_ic / (ic_df["ic"].std() + 1e-9) * np.sqrt(252 / 21)) if not ic_df.empty else 0.0
            strat_records.append({
                "stratification_type": "Market_Regime_OOS",
                "segment": reg,
                "model": m_name,
                "mean_rank_ic": round(m_ic, 4),
                "icir": round(icir, 2),
                "n_dates": len(ic_df),
                "n_obs": len(sub_reg),
                "partition_context": "HIST_EVAL_OOS"
            })

    strat_df = pd.DataFrame(strat_records)
    strat_df.to_csv("reports/diagnostics/ic_by_year_sector_tercile_regime.csv", index=False)
    print("Saved reports/diagnostics/ic_by_year_sector_tercile_regime.csv")

    # =========================================================================
    # PART 5 & 6: INTERVAL & PROBABILITY CALIBRATION (21D HORIZON)
    # =========================================================================
    print("\n--- COMPUTING INTERVAL AND PROBABILITY CALIBRATION ---")
    val_sigma = float(np.std(val_residuals_21d))
    print(f"Validation residual std (sigma): {val_sigma:.6f}")

    eval_lgb_pred = preds_21d_dict["LightGBM"]
    y_eval_true = eval_21d_df["target_return_21d"].values
    current_close = eval_21d_df["close"].values
    target_price_21d = eval_21d_df["target_price_21d"].values
    pred_price_lgb = current_close * (1.0 + eval_lgb_pred)

    # Method Code String
    interval_code_snippet = """
# Empirical Conformal / Validation Residual Scaling:
val_residuals = y_val - val_pred_lgb
val_resid_std = float(np.std(val_residuals)) # sigma = 0.1169

# Coverage interval at level alpha:
# z_alpha = stats.norm.ppf(1 - (1 - nominal) / 2)
interval_lower = pred_return - z_alpha * val_resid_std
interval_upper = pred_return + z_alpha * val_resid_std

# In price space:
price_lower = current_close * (1.0 + interval_lower)
price_upper = current_close * (1.0 + interval_upper)
"""

    # Multi-level coverages: 50%, 80%, 90%, 95%
    z_scores = {
        "50%": stats.norm.ppf(0.75),
        "80%": stats.norm.ppf(0.90),
        "90%": stats.norm.ppf(0.95),
        "95%": stats.norm.ppf(0.975)
    }

    interval_audit = {"method": "Validation Residual Normal Scaling", "code_snippet": interval_code_snippet.strip(), "levels": {}}

    for lvl_name, z_val in z_scores.items():
        nom_val = float(lvl_name.replace("%", ""))
        half_width = z_val * val_sigma
        in_int = (y_eval_true >= (eval_lgb_pred - half_width)) & (y_eval_true <= (eval_lgb_pred + half_width))
        cov_all = float(in_int.mean() * 100.0)

        # Per year
        cov_2025 = float(in_int[eval_21d_df["year"] == 2025].mean() * 100.0)
        cov_2026 = float(in_int[eval_21d_df["year"] == 2026].mean() * 100.0)

        interval_audit["levels"][lvl_name] = {
            "nominal_coverage_pct": nom_val,
            "z_multiplier": round(z_val, 4),
            "interval_width_return_terms": round(2 * half_width, 4),
            "empirical_coverage_overall_pct": round(cov_all, 2),
            "empirical_coverage_2025_pct": round(cov_2025, 2),
            "empirical_coverage_2026_pct": round(cov_2026, 2),
        }

    # Coverage when market is volatile (VIX above median) vs calm (VIX <= median)
    vix_median = float(eval_21d_df["vix_level"].median())
    high_vix_mask = eval_21d_df["vix_level"] > vix_median
    calm_vix_mask = ~high_vix_mask

    z_95 = z_scores["95%"]
    in_95 = (y_eval_true >= (eval_lgb_pred - z_95 * val_sigma)) & (y_eval_true <= (eval_lgb_pred + z_95 * val_sigma))
    cov_high_vix = float(in_95[high_vix_mask].mean() * 100.0)
    cov_calm_vix = float(in_95[calm_vix_mask].mean() * 100.0)

    interval_audit["market_volatility_breakdown"] = {
        "india_vix_median": round(vix_median, 2),
        "coverage_95_high_vix_volatile_pct": round(cov_high_vix, 2),
        "coverage_95_calm_vix_pct": round(cov_calm_vix, 2),
        "high_vix_obs": int(high_vix_mask.sum()),
        "calm_vix_obs": int(calm_vix_mask.sum())
    }

    # Probability Calibration of P(up)
    # Model P(up) = norm.cdf(pred_return / val_sigma)
    prob_up = stats.norm.cdf(eval_lgb_pred / (val_sigma + 1e-9))
    actual_binary_up = (y_eval_true > 0).astype(float)

    # Brier scores
    brier_model = float(np.mean((prob_up - actual_binary_up) ** 2))
    base_rate = float(np.mean(actual_binary_up))
    brier_base_rate = float(np.mean((base_rate - actual_binary_up) ** 2))

    # Reliability Table in 10 bins
    bin_edges = np.linspace(0.0, 1.0, 11)
    reliability_bins = []
    ece = 0.0
    n_eval_total = len(actual_binary_up)

    for i in range(10):
        low_b, high_b = bin_edges[i], bin_edges[i+1]
        mask = (prob_up >= low_b) & (prob_up < high_b) if i < 9 else (prob_up >= low_b) & (prob_up <= high_b)
        n_b = int(mask.sum())
        if n_b > 0:
            mean_pred = float(np.mean(prob_up[mask]))
            obs_rate = float(np.mean(actual_binary_up[mask]))
            ece += (n_b / n_eval_total) * abs(mean_pred - obs_rate)
            reliability_bins.append({
                "bin_idx": i + 1,
                "bin_range": f"[{low_b:.1f}, {high_b:.1f}]",
                "count": n_b,
                "mean_predicted_prob": round(mean_pred, 4),
                "observed_positive_rate": round(obs_rate, 4),
                "calibration_gap": round(mean_pred - obs_rate, 4)
            })
        else:
            reliability_bins.append({
                "bin_idx": i + 1,
                "bin_range": f"[{low_b:.1f}, {high_b:.1f}]",
                "count": 0,
                "mean_predicted_prob": round((low_b + high_b) / 2, 4),
                "observed_positive_rate": 0.0,
                "calibration_gap": 0.0
            })

    prob_calibration_audit = {
        "brier_score_model": round(brier_model, 6),
        "brier_score_base_rate": round(brier_base_rate, 6),
        "brier_skill_score": round(1.0 - (brier_model / brier_base_rate), 6),
        "expected_calibration_error_ece": round(ece, 6),
        "unconditional_base_rate_pct": round(base_rate * 100.0, 2),
        "reliability_table": reliability_bins
    }

    # Price MAE and Percentage Price Error
    valid_price_mask = pd.notna(target_price_21d) & (target_price_21d > 0)
    price_errors = np.abs(target_price_21d[valid_price_mask] - pred_price_lgb[valid_price_mask])
    pct_price_errors = (price_errors / target_price_21d[valid_price_mask]) * 100.0

    price_error_audit = {
        "return_mae": round(float(mean_absolute_error(y_eval_true, eval_lgb_pred)), 6),
        "return_rmse": round(float(np.sqrt(mean_squared_error(y_eval_true, eval_lgb_pred))), 6),
        "price_mae_inr": round(float(np.mean(price_errors)), 2),
        "price_mape_mean_pct": round(float(np.mean(pct_price_errors)), 2),
        "price_mape_median_pct": round(float(np.median(pct_price_errors)), 2),
        "price_error_5th_pct": round(float(np.percentile(pct_price_errors, 5)), 2),
        "price_error_95th_pct": round(float(np.percentile(pct_price_errors, 95)), 2),
    }

    calibration_bundle = {
        "audit_timestamp": pd.Timestamp.now().isoformat(),
        "horizon": "21D",
        "evaluation_window": "2025-01-01 to 2026-10-23",
        "observations": int(len(eval_21d_df)),
        "interval_calibration": interval_audit,
        "probability_calibration": prob_calibration_audit,
        "price_and_return_errors": price_error_audit
    }

    with open("reports/diagnostics/interval_and_prob_calibration.json", "w") as f:
        json.dump(calibration_bundle, f, indent=2)
    print("Saved reports/diagnostics/interval_and_prob_calibration.json")

    elapsed = round(time.time() - start_t, 2)
    print(f"\nSECTION 7 AUDIT COMPLETED SUCCESSFULLY in {elapsed}s.")

if __name__ == "__main__":
    run_section7_audit()

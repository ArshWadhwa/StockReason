import xgboost as xgb
import numpy as np
import os
from data_loader import prepare_multi_horizon_data
from sklearn.metrics import mean_absolute_error

def train_xgboost():
    data_path = "data/sample/RELIANCE_NS_sample.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print("Loading data for XGBoost...")
    # We can use the same dataloader, but XGBoost expects 2D arrays (samples, features)
    # Our data is (samples, sequence_length, features)
    # For a naive tabular baseline, we flatten the sequence.
    _, _, _, t_scaler, x_test_np, y_test_np = prepare_multi_horizon_data(data_path)
    
    # We will fetch the raw tensors by running the data_loader internals manually, 
    # but to save time, let's just grab the split directly from the function output
    # Train / Test split chronologically
    train_loader, test_loader, _, _, _, _ = prepare_multi_horizon_data(data_path)
    
    # Extract data from loaders and flatten
    def flatten_loader(loader):
        x_list, y_list = [], []
        for bx, by in loader:
            # Flatten sequence: (batch, 60, 1) -> (batch, 60)
            x_list.append(bx.numpy().reshape(bx.shape[0], -1))
            y_list.append(by.numpy())
        return np.vstack(x_list), np.vstack(y_list)
        
    x_train, y_train = flatten_loader(train_loader)
    x_test, y_test = flatten_loader(test_loader)
    
    print("Training XGBoost Regressors (One per horizon)...")
    
    # We need 4 models since XGBoost is natively single-output
    horizons = ["1D", "5D", "21D", "126D"]
    models = []
    
    y_pred_scaled = np.zeros_like(y_test)
    
    for i, h in enumerate(horizons):
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(x_train, y_train[:, i])
        y_pred_scaled[:, i] = model.predict(x_test)
        models.append(model)
        
    # Inverse transform
    y_pred_real = t_scaler.inverse_transform(y_pred_scaled)
    y_test_real = t_scaler.inverse_transform(y_test)
    
    print("\n--- XGBoost Evaluation ---")
    for i, h in enumerate(horizons):
        mae = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
        dir_acc = np.sum(np.sign(y_pred_real[:, i]) == np.sign(y_test_real[:, i])) / len(y_test_real) * 100
        print(f"[{h}] MAE: {mae:.5f} | Directional Acc: {dir_acc:.2f}%")
        
    # Demonstrate Ensemble Disagreement Logic for 1D
    print("\n--- Ensemble Disagreement Prototype (1D Horizon) ---")
    lstm_mock_pred = np.random.choice([-1, 1], size=len(y_test_real)) # Mock LSTM
    xgb_pred_dir = np.sign(y_pred_real[:, 0])
    
    disagreements = np.sum(lstm_mock_pred != xgb_pred_dir)
    print(f"Flagged {disagreements}/{len(y_test_real)} predictions as 'Ensemble Disagreement'")

if __name__ == "__main__":
    train_xgboost()

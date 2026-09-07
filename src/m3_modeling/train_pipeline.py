import argparse
import os
import torch
import torch.nn as nn
import joblib
import xgboost as xgb
import numpy as np

from src.m3_modeling.data_loader import prepare_multi_horizon_data
from src.m3_modeling.lstm_model import MultiHorizonLSTM, mc_dropout_inference
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_xgboost_ensemble(train_loader, test_loader, t_scaler, output_dir):
    def flatten_loader(loader):
        x_list, y_list = [], []
        for bx, by in loader:
            x_list.append(bx.numpy().reshape(bx.shape[0], -1))
            y_list.append(by.numpy())
        return np.vstack(x_list), np.vstack(y_list)
        
    x_train, y_train = flatten_loader(train_loader)
    x_test, y_test = flatten_loader(test_loader)
    
    print("Training XGBoost Ensemble Models...")
    horizons = ["1D", "5D", "21D", "126D"]
    models = {}
    
    for i, h in enumerate(horizons):
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(x_train, y_train[:, i])
        models[h] = model
        
    # Save models
    xgb_path = os.path.join(output_dir, "xgboost_ensemble.pkl")
    joblib.dump(models, xgb_path)
    print(f"Saved XGBoost models to {xgb_path}")

def main():
    parser = argparse.ArgumentParser(description="M3 Model Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/sample", help="Directory containing CSV files")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=64, help="LSTM hidden dimensions")
    parser.add_argument("--layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    args = parser.parse_args()
    
    # Ensure artifacts directory exists
    model_dir = "artifacts/models"
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"--- Starting M3 Pipeline on {args.data_dir} ---")
    
    # 1. Load Data
    try:
        train_loader, test_loader, f_scaler, t_scaler, x_test_np, y_test_np, num_features = prepare_multi_horizon_data(args.data_dir)
    except Exception as e:
        print(f"Data Loading Error: {e}")
        return
        
    # Save Scalers
    joblib.dump(f_scaler, os.path.join(model_dir, "feature_scaler.pkl"))
    joblib.dump(t_scaler, os.path.join(model_dir, "target_scaler.pkl"))
    print(f"Saved scalers to {model_dir}")
    
    # 2. Train LSTM
    model = MultiHorizonLSTM(input_dim=num_features, hidden_dim=args.hidden_dim, num_layers=args.layers, dropout_rate=args.dropout, output_dim=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    best_model_path = os.path.join(model_dir, "best_lstm.pth")
    
    print("\n--- Training LSTM ---")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Eval on test set for checkpointing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                y_pred = model(batch_x)
                test_loss += criterion(y_pred, batch_y).item()
                
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss/len(train_loader):.5f} | Test Loss: {avg_test_loss:.5f}")
        
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_path)
            
    print(f"Saved best LSTM model to {best_model_path}")
    
    # 3. Train XGBoost
    train_xgboost_ensemble(train_loader, test_loader, t_scaler, model_dir)
    
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    main()

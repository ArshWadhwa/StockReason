import torch
import torch.nn as nn
import numpy as np
import os
from src.m3_modeling.data_loader import prepare_multi_horizon_data
from sklearn.metrics import mean_absolute_error, mean_squared_error

class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=4, dropout_rate=0.3):
        super(MultiHorizonLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM with dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        
        # MC Dropout layer before linear output
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer for 4 horizons: 1D, 5D, 21D, 126D
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.linear(out) 
        return out

def mc_dropout_inference(model, x, num_passes=50):
    """
    Runs Monte Carlo Dropout inference by enabling dropout during evaluation.
    Returns the mean prediction and the standard deviation (uncertainty).
    """
    model.train() # Force dropout to be active
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            predictions.append(model(x).numpy())
            
    predictions = np.array(predictions)
    
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred

def train_and_evaluate():
    data_path = "data/sample/RELIANCE_NS_sample.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print("Loading data and preparing multi-horizon targets...")
    train_loader, test_loader, f_scaler, t_scaler, x_test_np, y_test_np = prepare_multi_horizon_data(data_path)
    
    # Init model: input=1 (feature), output=4 (horizons)
    model = MultiHorizonLSTM(input_dim=1, output_dim=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    print("Training Multi-Horizon LSTM...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.5f}")
        
    print("\nRunning Monte Carlo Dropout Inference on Test Set...")
    # Get a batch from test set for demonstration
    x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32)
    mean_pred, std_pred = mc_dropout_inference(model, x_test_tensor, num_passes=30)
    
    # Inverse transform to get real percentages
    mean_pred_real = t_scaler.inverse_transform(mean_pred)
    y_test_real = t_scaler.inverse_transform(y_test_np)
    
    # Evaluate 1D Horizon (Index 0)
    mae_1d = mean_absolute_error(y_test_real[:, 0], mean_pred_real[:, 0])
    dir_acc_1d = np.sum(np.sign(mean_pred_real[:, 0]) == np.sign(y_test_real[:, 0])) / len(y_test_real) * 100
    
    print(f"\n--- 1D Horizon Evaluation ---")
    print(f"MAE: {mae_1d:.5f}")
    print(f"Directional Accuracy: {dir_acc_1d:.2f}%")
    print(f"Average Confidence (Inverse Variance): {1.0 / (np.mean(std_pred[:, 0]) + 1e-6):.2f}")

    # Evaluate 21D Horizon (Index 2)
    mae_21d = mean_absolute_error(y_test_real[:, 2], mean_pred_real[:, 2])
    dir_acc_21d = np.sum(np.sign(mean_pred_real[:, 2]) == np.sign(y_test_real[:, 2])) / len(y_test_real) * 100
    
    print(f"\n--- 21D Horizon (1 Month) Evaluation ---")
    print(f"MAE: {mae_21d:.5f}")
    print(f"Directional Accuracy: {dir_acc_21d:.2f}%")
    
if __name__ == "__main__":
    train_and_evaluate()

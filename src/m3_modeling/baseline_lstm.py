import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

class BaselineLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, output_dim=1):
        super(BaselineLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :]) 
        return out

def prepare_data(df, sequence_length=60, split_ratio=0.8):
    # Use 'Close' price and calculate percentage returns
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    
    data = df['Return'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
        
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    # Train / Test split chronologically
    split_idx = int(len(x) * split_ratio)
    x_train, y_train = x[:split_idx], y[:split_idx]
    x_test, y_test = x[split_idx:], y[split_idx:]
    
    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return x_train, y_train, x_test, y_test, scaler

def train_baseline_model(data_path, epochs=5, batch_size=32):
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please run data_fetcher.py first.")
        return
        
    x_train, y_train, x_test, y_test, scaler = prepare_data(df)
    
    # Create DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    model = BaselineLSTM()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting baseline training...")
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
        
    print("Baseline training complete!")
    evaluate_model(model, x_test, y_test, scaler)

def evaluate_model(model, x_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        
    # Inverse transform to get actual returns
    y_pred_np = scaler.inverse_transform(y_pred.numpy())
    y_test_np = scaler.inverse_transform(y_test.numpy())
    
    mae = mean_absolute_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
    
    # Directional Accuracy
    correct_direction = np.sum(np.sign(y_pred_np) == np.sign(y_test_np))
    dir_acc = correct_direction / len(y_test_np) * 100
    
    print("\n--- Evaluation Metrics ---")
    print(f"MAE: {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"Directional Accuracy: {dir_acc:.2f}%")

if __name__ == "__main__":
    sample_data_path = "data/sample/RELIANCE_NS_sample.csv"
    train_baseline_model(sample_data_path, epochs=10)

"""
MultiHorizonLSTM — Primary model for multi-horizon stock return prediction.
2-layer LSTM with MC Dropout for uncertainty estimation.
"""
import torch
import torch.nn as nn
import numpy as np


class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=4, dropout_rate=0.3):
        super(MultiHorizonLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM with dropout between layers
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
    model.train()  # Force dropout to be active
    predictions = []

    with torch.no_grad():
        for _ in range(num_passes):
            predictions.append(model(x).numpy())

    predictions = np.array(predictions)

    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)

    return mean_pred, std_pred

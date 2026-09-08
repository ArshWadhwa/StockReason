"""
BaselineLSTM — Secondary ensemble model.
Simpler architecture (1-layer, larger hidden) for diversity in ensemble predictions.
Supports multi-horizon output and MC Dropout inference.
"""
import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=1, output_dim=4, dropout_rate=0.2):
        super(BaselineLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Single-layer LSTM (different architecture from primary for ensemble diversity)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0.0)

        # MC Dropout layer before linear output
        self.dropout = nn.Dropout(dropout_rate)

        # Output layer for multi-horizon prediction
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out

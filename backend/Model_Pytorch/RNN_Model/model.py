# model.py

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Simple LSTM model for single-station time series forecasting.
    We assume:
      - x has shape (batch_size, seq_len, in_channels)
      - We will pass x directly into an LSTM (batch_first=True).
      - Then we'll take the last time-step output and pass it through
        a final fully connected layer to produce a single output.
    """
    def __init__(
        self,
        in_channels,   # number of input features
        hidden_dim=64,
        num_layers=2,
        label_width=1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.label_width = label_width
        
        # LSTM (batch_first=True means input is (batch, seq, features))
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Final linear layer to produce a single output (or label_width outputs)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """
        x shape: (batch_size, seq_len, in_channels).
        LSTM output shape: (batch_size, seq_len, hidden_dim)
        We'll take the output of the last time step: out[:, -1, :]
        Then pass that through fc_out -> (batch_size, 1)
        """
        out, (h, c) = self.lstm(x)           # out -> (batch, seq_len, hidden_dim)
        last_timestep = out[:, -1, :]        # (batch, hidden_dim)
        prediction = self.fc_out(last_timestep)  # (batch, 1)
        return prediction

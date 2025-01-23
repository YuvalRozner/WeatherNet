# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModule(nn.Module):
    def __init__(self, in_channels, d_model, cnn_kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=cnn_kernel_size,
            padding=(cnn_kernel_size // 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        return x

class TransformerModule(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer(x)
        return x

class CNNTransformerModel(nn.Module):
    """
    Model for single-station time series forecasting:
      - 1D CNN over time, producing embeddings for each time step.
      - Transformer encoder to model long-range temporal dependencies.
      - Final dense layer to predict target temperature.
    """
    def __init__(
        self,
        in_channels,      # number of input features
        d_model=64,       # dimension for transformer
        nhead=4,
        num_transformer_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        cnn_kernel_size=3,
        label_width=1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.d_model = d_model
        self.label_width = label_width
        
        # 1) Convolution that processes each feature over time,
        #    output dimension = d_model
        #    We'll keep the time dimension so we get an embedding per time step.
        
        # For simplicity, let's define a single Conv1d that uses stride=1, padding=1
        # so we get an output shape of (batch_size, d_model, seq_len).
        self.cnn = CNNModule(in_channels, d_model, cnn_kernel_size)
        
        # 2) Transformer Encoder
        self.transformer = TransformerModule(d_model, nhead, num_transformer_layers, dim_feedforward, dropout)
        
        # 3) Final linear to produce the forecast
        #    The transformer's output has shape (batch, seq_len, d_model).
        #    We want to map to label_width * 1 (if predicting only 1 feature).
        #    If label_width=1, we might just take the last time step.
        
        self.fc_out = nn.Linear(d_model, 1)  # predict single value (temp)
        
    def forward(self, x):
        """
        x shape: (batch_size, seq_len, in_channels)
            if your dataset returns (seq_len, in_channels), we might need to permute:
            we want (batch_size, in_channels, seq_len) for the Conv1d usually.
        """
        # We might need to rearrange x:
        # Currently x: (batch_size, seq_len, in_channels)
        # For Conv1D we want: (batch_size, in_channels, seq_len)
        x = x.permute(0, 2, 1)  # => (batch_size, in_channels, seq_len)
        
        # CNN
        # out shape => (batch_size, d_model, seq_len)
        x = self.cnn(x)
        x = F.relu(x)
        
        # Now for the transformer, we want (batch_size, seq_len, d_model)
        x = x.permute(0, 2, 1)  # => (batch_size, seq_len, d_model)
        
        # Transformer
        x = self.transformer(x)  # => (batch_size, seq_len, d_model)
        
        # For a single-step forecast (label_width=1), we can focus on the last time step
        # e.g. x[:, -1, :] => (batch_size, d_model)
        # Then feed into fc_out => (batch_size, 1)
        
        out = self.fc_out(x[:, -1, :])  # take last time step embedding
        return out

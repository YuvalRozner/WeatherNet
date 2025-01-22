import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
################################################################################
# 1) CNN modules
################################################################################
class StationCNN(nn.Module):
    """
    Shared CNN for non-target stations.
    Expects input shape: (batch_size, in_channels, seq_len).
    """
    def __init__(self, in_channels, cnn_embed_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(64, cnn_embed_dim)

    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(dim=-1)  # (batch_size, 64)
        x = self.fc(x)                    # (batch_size, cnn_embed_dim)
        return x


class TargetStationCNN(nn.Module):
    """
    Specialized CNN for the *target* station.
    Similar structure but separate weights.
    """
    def __init__(self, in_channels, cnn_embed_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(64, cnn_embed_dim)

    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(dim=-1)  # (batch_size, 64)
        x = self.fc(x)                    # (batch_size, cnn_embed_dim)
        return x


################################################################################
# 2) Location embedding
################################################################################
class LocationEmbedding(nn.Module):
    """
    Embeds station location (x, y) into a learnable vector of dimension loc_emb_dim.
    """
    def __init__(self, loc_in_dim=2, loc_emb_dim=32):
        super().__init__()
        self.fc = nn.Linear(loc_in_dim, loc_emb_dim)

    def forward(self, loc):
        # loc shape: (batch_size, 2)
        x = self.fc(loc)
        return F.relu(x)  # (batch_size, loc_emb_dim)


################################################################################
# 3) Transformer encoder
################################################################################
class MultiStationTransformer(nn.Module):
    """
    Applies a TransformerEncoder over station embeddings (batch_first=True).
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch_size, num_stations, d_model)
        x = self.transformer_encoder(x)
        return x


################################################################################
# 4) Full model
################################################################################
class TemperatureForecastModel(nn.Module):
    """
    Architecture:
      - Shared CNN for non-target stations
      - Dedicated CNN for the target station
      - Location embeddings for all stations
      - Transformer across station embeddings
      - Predict target station temperature
    """
    def __init__(
        self,
        num_stations,
        target_station_idx,
        in_channels_per_station,
        seq_len,
        cnn_embed_dim=64,
        loc_emb_dim=32,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        
        # Station CNN for non-target stations
        self.station_cnn = StationCNN(
            in_channels=in_channels_per_station,
            cnn_embed_dim=cnn_embed_dim
        )
        
        # Dedicated CNN for target station
        self.target_station_cnn = TargetStationCNN(
            in_channels=in_channels_per_station,
            cnn_embed_dim=cnn_embed_dim
        )
        
        # Location embedding
        self.location_embedding = LocationEmbedding(loc_in_dim=2, loc_emb_dim=loc_emb_dim)
        
        # Combine CNN + location => project to d_model
        self.combined_dim = cnn_embed_dim + loc_emb_dim
        self.proj_to_d_model = nn.Linear(self.combined_dim, d_model)
        
        # Transformer
        self.transformer = MultiStationTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Final linear layer for target station temperature
        self.fc_out = nn.Linear(d_model, 1)
        
        self.num_stations = num_stations
        self.target_station_idx = target_station_idx

    def forward(self, station_timeseries, station_locations):
        """
        Args:
            station_timeseries: shape (batch_size, num_stations, in_channels, seq_len)
            station_locations:  shape (batch_size, num_stations, 2)
            
        Returns:
            Predictions for the target station's temperature, shape (batch_size, 1).
        """
        batch_size, num_stations, in_ch, seq_len = station_timeseries.shape
        
        station_embs = []
        
        for s in range(num_stations):
            ts_data = station_timeseries[:, s, :, :]  # (batch_size, in_channels, seq_len)
            loc_data = station_locations[:, s, :]     # (batch_size, 2)
            
            # 1) CNN embedding
            if s == self.target_station_idx:
                # Use the dedicated target station CNN
                cnn_embed = self.target_station_cnn(ts_data)
            else:
                # Use shared CNN
                cnn_embed = self.station_cnn(ts_data)
            
            # 2) Location embedding
            loc_embed = self.location_embedding(loc_data)
            
            # 3) Combine CNN + location
            combined = torch.cat([cnn_embed, loc_embed], dim=-1)  # (batch_size, cnn_embed_dim + loc_emb_dim)
            
            # 4) Project to d_model
            station_d_model = self.proj_to_d_model(combined)  # (batch_size, d_model)
            
            station_embs.append(station_d_model)
        
        # Stack => (batch_size, num_stations, d_model)
        station_embs = torch.stack(station_embs, dim=1)
        
        # 5) Transformer
        transformer_out = self.transformer(station_embs)
        
        # 6) Extract the target station's vector
        target_emb = transformer_out[:, self.target_station_idx, :]  # (batch_size, d_model)
        
        # 7) Predict final temperature
        out = self.fc_out(target_emb)  # (batch_size, 1)
        
        return out





################################################################################
# 5) Example usage
################################################################################
if __name__ == "__main__":
    # Hypothetical example
    batch_size = 8
    num_stations = 5
    target_station_idx = 0
    in_channels_per_station = 10  # e.g. 10 meteorological features
    seq_len = 24                  # e.g. 24-hour window
    
    # Random input
    station_timeseries = torch.randn(batch_size, num_stations, in_channels_per_station, seq_len)
    station_locations  = torch.randn(batch_size, num_stations, 2)
    
    # Define model
    model = TemperatureForecastModel(
        num_stations=num_stations,
        target_station_idx=target_station_idx,
        in_channels_per_station=in_channels_per_station,
        seq_len=seq_len,
        cnn_embed_dim=64,
        loc_emb_dim=32,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    
    # Forward pass
    prediction = model(station_timeseries, station_locations)
    print("Prediction shape:", prediction.shape)  # (batch_size, 1)

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# 1) CNN for per-station short-term feature extraction
################################################################################
class StationCNN(nn.Module):
    """
    Example CNN to extract embeddings from a single station's time series.
    Expects input shape: (batch_size, in_channels, seq_len).
    """
    def __init__(self, in_channels, cnn_embed_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(64, cnn_embed_dim)  # Project to desired embedding dimension

    def forward(self, x):
        # x shape = (batch_size, in_channels, seq_len)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # Global pooling over time dimension
        x = self.pool(x)               # -> (batch_size, 64, 1)
        x = x.squeeze(dim=-1)          # -> (batch_size, 64)
        x = self.fc(x)                 # -> (batch_size, cnn_embed_dim)
        return x


################################################################################
# 2) Location embedding
################################################################################
class LocationEmbedding(nn.Module):
    """
    Small MLP to embed location coordinates, e.g., (x, y) in "New Network of Israel" coordinates.
    """
    def __init__(self, loc_in_dim=2, loc_emb_dim=32):
        super().__init__()
        self.fc = nn.Linear(loc_in_dim, loc_emb_dim)

    def forward(self, loc):
        # loc shape = (batch_size, loc_in_dim)
        x = self.fc(loc)               # -> (batch_size, loc_emb_dim)
        return F.relu(x)


################################################################################
# 3) Multi-station Transformer encoder
################################################################################
class MultiStationTransformer(nn.Module):
    """
    Applies a standard TransformerEncoder over station embeddings.
    Expects shape (batch_size, num_stations, d_model).
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # If using PyTorch >= 1.10, we can specify batch_first directly
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape = (batch_size, num_stations, d_model)
        x = self.transformer_encoder(x)  # -> same shape
        return x


################################################################################
# 4) Full model that glues everything together
################################################################################
class TemperatureForecastModel(nn.Module):
    """
    1) Runs a CNN on each station's time series data.
    2) Embeds station location.
    3) Concatenates CNN embedding + location embedding, then projects to d_model.
    4) Runs a Transformer across stations.
    5) Predicts temperature for the target station.
    """
    def __init__(
        self,
        num_stations: int,
        in_channels_per_station: int,
        seq_len: int,
        cnn_embed_dim: int = 64,
        loc_emb_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 1) Single CNN (shared) or a separate CNN per station?
        #    Often you'd share parameters if stations have the same features.
        #    Below is a single shared CNN, repeatedly applied.
        self.station_cnn = StationCNN(
            in_channels=in_channels_per_station,
            cnn_embed_dim=cnn_embed_dim
        )
        
        # 2) Location embedding (shared by all stations, shape = (2,) => (loc_emb_dim,))
        self.location_embedding = LocationEmbedding(loc_in_dim=2, loc_emb_dim=loc_emb_dim)
        
        # Combine CNN + location embeddings
        self.combined_dim = cnn_embed_dim + loc_emb_dim
        self.proj_to_d_model = nn.Linear(self.combined_dim, d_model)
        
        # 3) Transformer to handle station embeddings
        self.transformer = MultiStationTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 4) Final linear layer for temperature prediction
        self.fc_out = nn.Linear(d_model, 1)  # Predict a single value (temperature)
        
    def forward(self, station_timeseries, station_locations, target_station_idx=0):
        """
        station_timeseries: list or tensor of shape (batch_size, num_stations, in_channels, seq_len)
            - e.g., for each station, we have a feature matrix (in_channels x seq_len).
        station_locations: shape (batch_size, num_stations, 2)
            - (x, y) coordinates of each station.
        target_station_idx: which station's temperature to predict.
            - If you always predict the same station, you can fix this or pass 0.

        Returns:
            Predicted temperature, shape = (batch_size, 1).
        """
        batch_size, num_stations, in_ch, seq_len = station_timeseries.shape
        
        # We'll collect embeddings from each station in a list
        station_embs = []
        
        for s in range(num_stations):
            # Extract the time series for station s
            ts_data = station_timeseries[:, s, :, :]  # -> (batch_size, in_channels, seq_len)
            
            # CNN embedding
            cnn_embed = self.station_cnn(ts_data)
            
            # Location embedding
            loc_data = station_locations[:, s, :]      # -> (batch_size, 2)
            loc_embed = self.location_embedding(loc_data)
            
            # Combine CNN + location
            combined = torch.cat([cnn_embed, loc_embed], dim=-1)  # (batch_size, cnn_embed_dim + loc_emb_dim)
            
            # Project to d_model
            station_d_model = self.proj_to_d_model(combined)      # (batch_size, d_model)
            
            station_embs.append(station_d_model)
        
        # Stack along station dimension => (batch_size, num_stations, d_model)
        station_embs = torch.stack(station_embs, dim=1)
        
        # Pass through Transformer
        # shape stays (batch_size, num_stations, d_model)
        transformer_out = self.transformer(station_embs)
        
        # Extract the embedding for the target station
        # E.g. if target_station_idx = 0, we pick out the first station's vector
        target_emb = transformer_out[:, target_station_idx, :]  # -> (batch_size, d_model)
        
        # Final prediction
        out = self.fc_out(target_emb)  # -> (batch_size, 1)
        
        return out


###############################################################################
# 5) Example usage
###############################################################################
if __name__ == "__main__":
    # Hypothetical example:
    batch_size = 8
    num_stations = 5
    in_channels_per_station = 10  # e.g. 10 meteorological features
    seq_len = 24                  # e.g. 24-hour window
    
    # Create random input
    station_timeseries = torch.randn(batch_size, num_stations, in_channels_per_station, seq_len)
    station_locations  = torch.randn(batch_size, num_stations, 2)  # (x, y) coords
    
    # Define the model
    model = TemperatureForecastModel(
        num_stations=num_stations,
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
    target_station_idx = 0  # Suppose the "target station" is station index 0
    prediction = model(station_timeseries, station_locations, target_station_idx=target_station_idx)
    print("Prediction shape:", prediction.shape)  # (batch_size, 1)

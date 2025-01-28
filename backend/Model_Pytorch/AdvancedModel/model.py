# model.py

import torch
import torch.nn as nn
import math

class StationCNN(nn.Module):
    def __init__(self, input_channels, cnn_channels, kernel_size, feature_dim):
        super(StationCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels,
                               out_channels=cnn_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=cnn_channels,
                               out_channels=feature_dim,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        """
        x: [batch_size, input_channels, time_steps]
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x  # [batch_size, feature_dim, time_steps]


class CoordinatePositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(CoordinatePositionalEncoding, self).__init__()
        # Assuming two coordinates: X and Y
        self.lat_linear = nn.Linear(1, d_model // 2)
        self.lon_linear = nn.Linear(1, d_model // 2)
        self.activation = nn.ReLU()

    def forward(self, lat, lon):
        """
        lat: [num_stations, 1] - Normalized X coordinates
        lon: [num_stations, 1] - Normalized Y coordinates
        """
        lat_enc = self.lat_linear(lat)  # [num_stations, d_model//2]
        lon_enc = self.lon_linear(lon)  # [num_stations, d_model//2]
        spatial_emb = self.activation(torch.cat([lat_enc, lon_enc], dim=1))  # [num_stations, d_model]
        return spatial_emb


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TemporalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, num_stations, time_steps, d_model]
        """
        x = x + self.pe[:, :x.size(2), :].unsqueeze(1)  # [batch, num_stations, time_steps, d_model]
        return x


class TargetedWeatherPredictionModel(nn.Module):
    def __init__(self, num_stations, time_steps, feature_dim, cnn_channels, kernel_size,
                 d_model, nhead, num_layers, target_station_idx, label_width=1):
        super(TargetedWeatherPredictionModel, self).__init__()
        self.num_stations = num_stations
        self.time_steps = time_steps
        self.target_station_idx = target_station_idx
        self.label_width = label_width

        # Initialize separate CNNs for each station
        self.station_cnns = nn.ModuleList([
            StationCNN(input_channels=feature_dim,
                       cnn_channels=cnn_channels,
                       kernel_size=kernel_size,
                       feature_dim=feature_dim)
            for _ in range(num_stations)
        ])

        # Coordinate Positional Encoding
        self.coord_pos_encoding = CoordinatePositionalEncoding(d_model=d_model)

        # Linear layer to map CNN features to d_model
        self.feature_mapping = nn.Linear(feature_dim, d_model)

        # Temporal Positional Encoding
        self.temporal_pos_encoding = TemporalPositionalEncoding(d_model=d_model, max_len=time_steps)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final prediction layer
        self.fc_out = nn.Linear(d_model, label_width)  # Output label_width predictions

    def forward(self, x, lat, lon):
        """
        x: [batch_size, num_stations, time_steps, feature_dim]
        lat: [num_stations, 1] - Normalized X coordinates
        lon: [num_stations, 1] - Normalized Y coordinates
        """
        batch_size, num_stations, time_steps, feature_dim = x.size()

        # Extract temporal features for each station
        temporal_features = []
        for i in range(num_stations):
            station_data = x[:, i, :, :]  # [batch_size, time_steps, feature_dim]
            station_data = station_data.permute(0, 2, 1)  # [batch_size, feature_dim, time_steps]
            cnn_out = self.station_cnns[i](station_data)  # [batch_size, feature_dim, time_steps]
            cnn_out = cnn_out.permute(0, 2, 1)  # [batch_size, time_steps, feature_dim]
            temporal_features.append(cnn_out)

        # Stack temporal features: [batch_size, num_stations, time_steps, feature_dim]
        temporal_features = torch.stack(temporal_features, dim=1)

        # Spatial positional encoding using coordinates
        spatial_emb = self.coord_pos_encoding(lat, lon)  # [num_stations, d_model]
        spatial_emb = spatial_emb.unsqueeze(0).unsqueeze(2)  # [1, num_stations, 1, d_model]

        # Map temporal features to d_model
        temporal_features = self.feature_mapping(temporal_features)  # [batch_size, num_stations, time_steps, d_model]

        # Apply temporal positional encoding
        temporal_features = self.temporal_pos_encoding(temporal_features)  # [batch, num_stations, time_steps, d_model]

        # Combine temporal and spatial features
        combined_features = temporal_features + spatial_emb  # [batch, num_stations, time_steps, d_model]

        # Reshape for Transformer: [batch_size, num_stations * time_steps, d_model]
        combined_features = combined_features.view(batch_size, num_stations * time_steps, -1)

        # Transpose for Transformer: [sequence_length, batch_size, d_model]
        combined_features = combined_features.permute(1, 0, 2)

        # Transformer expects [sequence_length, batch_size, d_model]
        transformer_out = self.transformer_encoder(combined_features)  # [sequence_length, batch_size, d_model]

        # Reshape back: [batch_size, num_stations, time_steps, d_model]
        transformer_out = transformer_out.permute(1, 0, 2)
        transformer_out = transformer_out.view(batch_size, num_stations, time_steps, -1)

        # Select target station's features: [batch_size, time_steps, d_model]
        target_features = transformer_out[:, self.target_station_idx, :, :]  # [batch_size, time_steps, d_model]

        # Instead of mean pooling, you might retain temporal information or use other aggregation
        # For multi-step prediction, you can use the entire sequence or specific parts
        # Here, we'll simply take the last time step's features for simplicity
        # Alternatively, more sophisticated methods can be employed
        last_time_step_features = target_features[:, -1, :]  # [batch_size, d_model]

        # Final prediction
        prediction = self.fc_out(last_time_step_features)  # [batch_size, label_width]

        return prediction
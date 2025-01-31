# model.py

import torch
import torch.nn as nn
import math

class StationCNN(nn.Module):
    def __init__(self,
                 input_features=15,
                 output_per_feature=3,
                 kernel_size=3,
                 use_batch_norm=False,
                 use_residual=False):
        """
        Args:
            input_features (int): Number of input features per station.
            output_per_feature (int): Number of output channels per feature.
            kernel_size (int): Size of the convolutional kernel.
            use_batch_norm (bool): Whether to use Batch Normalization.
            use_residual (bool): Whether to use residual connections.
        """
        super(StationCNN, self).__init__()
        self.output_per_feature = output_per_feature
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # Total out_channels = input_features * output_per_feature
        self.out_channels = input_features * output_per_feature

        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=input_features,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=input_features  # Depthwise convolution
        )
        self.relu1 = nn.ReLU()

        # Optional Batch Normalization
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(self.out_channels)

        # Second convolutional layer (optional for deeper CNN)
        self.conv2 = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=input_features  # Maintain feature independence
        )
        self.relu2 = nn.ReLU()

        # Optional Batch Normalization
        if self.use_batch_norm:
            self.bn2 = nn.BatchNorm1d(self.out_channels)

        # Optional Residual Connection
        if self.use_residual:
            self.residual_conv = nn.Conv1d(
                in_channels=input_features,
                out_channels=self.out_channels,
                kernel_size=1,
                groups=input_features  # Depthwise 1x1 convolution
            )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_features, time_steps]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_per_feature, time_steps, input_features]
        """
        b, f, t = x.shape  # [batch_size, input_features, time_steps]

        # First convolution
        out = self.conv1(x)  # [batch_size, input_features * output_per_feature, time_steps]
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu1(out)

        # Second convolution
        out = self.conv2(out)  # [batch_size, input_features * output_per_feature, time_steps]
        if self.use_batch_norm:
            out = self.bn2(out)
        out = self.relu2(out)

        # Optional Residual Connection
        if self.use_residual:
            residual = self.residual_conv(x)  # [batch_size, input_features * output_per_feature, time_steps]
            out += residual
            out = self.relu2(out)

        # Reshape to [batch_size, output_per_feature, input_features, time_steps]
        out = out.view(b, self.output_per_feature, f, t)
        # Permute to [batch_size, output_per_feature, time_steps, input_features]
        out = out.permute(0, 1, 3, 2)  # [batch_size, output_per_feature, time_steps, features]

        return out  # [batch_size, output_per_feature, time_steps, features]

class CoordinatePositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(CoordinatePositionalEncoding, self).__init__()
        # Assuming two coordinates: X and Y
        self.lat_linear = nn.Linear(1, d_model // 2)
        self.lon_linear = nn.Linear(1, d_model // 2)
        self.activation = nn.ReLU()

    def forward(self, lat, lon):
        """
        Args:
            lat (torch.Tensor): [num_stations, 1] - Normalized X coordinates
            lon (torch.Tensor): [num_stations, 1] - Normalized Y coordinates
        Returns:
            torch.Tensor: [num_stations, d_model]
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
        Args:
            x (torch.Tensor): [batch_size, num_stations, time_steps, d_model]
        Returns:
            torch.Tensor: [batch_size, num_stations, time_steps, d_model]
        """
        x = x + self.pe[:, :x.size(2), :].unsqueeze(1)  # [batch, num_stations, time_steps, d_model]
        return x

class TargetedWeatherPredictionModel(nn.Module):
    def __init__(self, num_stations, time_steps, feature_dim, kernel_size,
                 d_model, nhead, num_layers, target_station_idx, label_width=1,
                 output_per_feature=3, use_batch_norm=False, use_residual=False):
        """
        Args:
            num_stations (int): Number of stations.
            time_steps (int): Number of time steps in the sliding window.
            feature_dim (int): Number of features per station.
            kernel_size (int): Size of the CNN kernel.
            d_model (int): Dimension of the model (for Transformer).
            nhead (int): Number of attention heads in the Transformer.
            num_layers (int): Number of Transformer encoder layers.
            target_station_idx (int): Index of the target station.
            label_width (int): Number of prediction steps.
            output_per_feature (int): Number of output channels per feature in CNN.
            use_batch_norm (bool): Whether to use Batch Normalization in CNNs.
            use_residual (bool): Whether to use residual connections in CNNs.
        """
        super(TargetedWeatherPredictionModel, self).__init__()
        self.num_stations = num_stations
        self.time_steps = time_steps
        self.target_station_idx = target_station_idx
        self.label_width = label_width
        self.output_per_feature = output_per_feature

        # Initialize separate CNNs for each station
        self.station_cnns = nn.ModuleList([
            StationCNN(
                input_features=feature_dim,
                output_per_feature=output_per_feature,
                kernel_size=kernel_size,
                use_batch_norm=use_batch_norm,
                use_residual=use_residual
            )
            for _ in range(num_stations)
        ])

        # Coordinate Positional Encoding
        self.coord_pos_encoding = CoordinatePositionalEncoding(d_model=d_model)

        # Linear layer to map CNN features to d_model
        # New feature_dim after CNN: output_per_feature * original feature_dim
        self.feature_mapping = nn.Linear(feature_dim * output_per_feature, d_model)

        # Temporal Positional Encoding
        self.temporal_pos_encoding = TemporalPositionalEncoding(d_model=d_model, max_len=time_steps)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final prediction layer
        self.fc_out = nn.Linear(d_model, label_width)  # Output label_width predictions

    def forward(self, x, lat, lon):
        """
        Args:
            x (torch.Tensor): [batch_size, num_stations, time_steps, feature_dim]
            lat (torch.Tensor): [num_stations, 1] - Normalized X coordinates
            lon (torch.Tensor): [num_stations, 1] - Normalized Y coordinates
        Returns:
            torch.Tensor: [batch_size, label_width]
        """
        batch_size, num_stations, time_steps, feature_dim = x.size()

        # Extract temporal features for each station
        # Initialize a list to collect CNN outputs
        temporal_features = []
        for i in range(num_stations):
            station_data = x[:, i, :, :]  # [batch_size, time_steps, feature_dim]
            station_data = station_data.permute(0, 2, 1)  # [batch_size, feature_dim, time_steps]
            cnn_out = self.station_cnns[i](station_data)  # [batch_size, output_per_feature, time_steps, feature_dim]
            temporal_features.append(cnn_out)

        # Stack temporal features: [batch_size, num_stations, output_per_feature, time_steps, feature_dim]
        temporal_features = torch.stack(temporal_features, dim=1)  # [batch, num_stations, output_per_feature, time_steps, features]

        # Reshape to combine output_per_feature and features dimensions
        # New shape: [batch_size, num_stations, time_steps, output_per_feature * feature_dim]
        temporal_features = temporal_features.view(batch_size, num_stations, self.output_per_feature, time_steps, feature_dim)
        temporal_features = temporal_features.permute(0, 1, 3, 2, 4)  # [batch, num_stations, time_steps, output_per_feature, features]
        temporal_features = temporal_features.contiguous().view(batch_size, num_stations, time_steps, self.output_per_feature * feature_dim)  # [batch, num_stations, time_steps, output_per_feature * features]

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
        combined_features = combined_features.permute(1, 0, 2)  # [num_stations * time_steps, batch_size, d_model]

        # Transformer expects [sequence_length, batch_size, d_model]
        transformer_out = self.transformer_encoder(combined_features)  # [sequence_length, batch_size, d_model]

        # Reshape back: [batch_size, num_stations, time_steps, d_model]
        transformer_out = transformer_out.permute(1, 0, 2)  # [batch_size, sequence_length, d_model]
        transformer_out = transformer_out.view(batch_size, num_stations, time_steps, -1)  # [batch_size, num_stations, time_steps, d_model]

        # Select target station's features: [batch_size, time_steps, d_model]
        target_features = transformer_out[:, self.target_station_idx, :, :]  # [batch_size, time_steps, d_model]

        # Instead of mean pooling, retain temporal information or use other aggregation
        # Here, we'll take the last time step's features for simplicity
        last_time_step_features = target_features[:, -1, :]  # [batch_size, d_model]

        # Final prediction
        prediction = self.fc_out(last_time_step_features)  # [batch_size, label_width]

        return prediction

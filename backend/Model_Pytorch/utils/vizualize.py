# visualize.py

import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torchinfo import summary
from torchviz import make_dot
import netron

from Backend.Model_Pytorch.common.data import (
    load_pkl_file,
    timeEncode,
    normalize_coordinates,
    drop_nan_rows_multiple
)
from Backend.Model_Pytorch.AdvancedModel.model import TargetedWeatherPredictionModel
from Backend.Model_Pytorch.AdvancedModel.parameters import PARAMS, WINDOW_PARAMS, ADVANCED_MODEL_PARAMS, \
    STATIONS_COORDINATES
from Backend.Model_Pytorch.common.analyze import analyze

# Define dictionaries to store feature maps and attention weights
cnn_feature_maps = {}
attention_weights = {}


def get_cnn_hook(name):
    """
    Creates a hook to capture CNN feature maps.

    Args:
        name (str): Unique name for the CNN layer.

    Returns:
        function: The hook function.
    """

    def hook(module, input, output):
        cnn_feature_maps[name] = output.detach().cpu()

    return hook


def load_scalers(scaler_dir='./output/scalers'):
    """
    Load the previously saved scalers for each station.

    Args:
        scaler_dir (str): Directory where scaler files are stored.

    Returns:
        list: List of scalers for each station.
    """
    scalers = []
    num_stations = ADVANCED_MODEL_PARAMS['num_stations']
    for i in range(num_stations):
        scaler_path = os.path.join(scaler_dir, f'scaler_station_{i}.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            scalers.append(scaler)
        print(f"Scaler for Station {i} loaded from {scaler_path}")
    return scalers


def load_model_for_inference(checkpoint_path, model_params, device='cpu'):
    """
    Create a model with the same architecture, load checkpoint, and return it in eval mode.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        model_params (dict): Parameters to initialize the model.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        nn.Module: The loaded model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    model = TargetedWeatherPredictionModel(**model_params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def load_window_multi_station(data_np, window_size, shift, label_width, scalers, target_column_index, idx=0):
    """
    Load a window of data encompassing all stations, apply individual scalers, and prepare tensors.

    Args:
        data_np (np.ndarray): Raw data of shape (T, num_stations, num_features).
        window_size (int): Number of time steps in the input window.
        shift (int): How far ahead the prediction starts after the input.
        label_width (int): Number of time steps to predict.
        scalers (list): List of scalers, one per station.
        target_column_index (int): Index of the target feature/column.
        idx (int): Starting index for the window.

    Returns:
        torch.Tensor: Scaled input tensor of shape (1, time_steps, num_stations, feature_dim).
        np.ndarray: Actual target values in original scale, shape (label_width,)
    """
    total_window_size = window_size + shift - 1 + label_width
    if idx + total_window_size > len(data_np):
        raise ValueError(f"Index {idx} with window size {total_window_size} exceeds data length {len(data_np)}.")

    # Extract input window: shape (window_size, num_stations, num_features)
    window = data_np[idx:idx + window_size, :, :]  # (window_size, num_stations, num_features)

    # Extract target: shape (label_width, )
    target_start = idx + window_size + shift - 1
    target_end = target_start + label_width
    actual_target = data_np[target_start:target_end, ADVANCED_MODEL_PARAMS['target_station_idx'], target_column_index]
    # No aggregation

    # Apply individual scalers to each station's data
    scaled_window = []
    num_stations = window.shape[1]
    for station_idx in range(num_stations):
        station_data = window[:, station_idx, :]  # (window_size, num_features)
        scaler = scalers[station_idx]
        station_data_scaled = scaler.transform(station_data)  # (window_size, num_features)
        scaled_window.append(station_data_scaled)

    # Stack scaled data: shape (window_size, num_stations, num_features)
    scaled_window = np.stack(scaled_window, axis=1)

    # Convert to torch.Tensor and reshape to (1, time_steps, num_stations, feature_dim)
    window_tensor = torch.from_numpy(scaled_window).float().unsqueeze(0)
    window_tensor = window_tensor.permute(0, 2, 1, 3)  # [1, time_steps, num_stations, feature_dim]

    return window_tensor, actual_target


@torch.no_grad()
def predict(model, input_window, lat, lon, device='cpu'):
    """
    Perform prediction using the multi-station model.

    Args:
        model (torch.nn.Module): Trained TargetedWeatherPredictionModel.
        input_window (torch.Tensor): Input data of shape (1, num_stations, time_steps, feature_dim).
        lat (torch.Tensor): Normalized latitude coordinates of shape (num_stations, 1).
        lon (torch.Tensor): Normalized longitude coordinates of shape (num_stations, 1).
        device (str): Device to perform inference on.
        return_attn (bool): Whether to return attention weights.

    Returns:
        tuple:
            - np.ndarray: Predicted values in original scale (e.g., Temperature in °C), shape (label_width, 1)
            - torch.Tensor or None: Attention weights if `return_attn` is True, else None.
    """
    input_tensor = input_window.to(device)
    lat = lat.to(device)
    lon = lon.to(device)

    # Model prediction without attention weights
    output_scaled = model(input_tensor, lat, lon)
    attn_weights = None

    # Convert to numpy
    output_scaled_np = output_scaled.squeeze(-1).cpu().numpy().reshape(-1, 1)  # Shape: (label_width, 1)

    return output_scaled_np, attn_weights


def visualize_feature_with_all_feature_maps(original_data, feature_maps, feature_name, station_idx,
                                            layer='conv2', num_features=15):
    """
    Visualize the original feature data alongside all its CNN-generated feature maps.

    Args:
        original_data (np.ndarray): Original time series data for the feature. Shape: [time_steps]
        feature_maps (dict): Dictionary containing feature maps.
        feature_names (list): List of feature names corresponding to each feature map.
        feature_name (str): The specific feature to visualize (e.g., 'Rain').
        station_idx (int): Index of the station.
        layer (str): CNN layer name ('conv1' or 'conv2').
        num_features (int): Number of feature maps to display.
    """
    # Validate the feature_name
    feature_names = [
        'Rain', 'RH (%)', 'TD (degC)', 'TDmax (degC)', 'TDmin (degC)',
        'STDwd (deg)', 'Year', 'Wind_x', 'Wind_y', 'Gust_x', 'Gust_y',
        'Day sin', 'Day cos', 'Year sin', 'Year cos'
    ]
    if feature_name not in feature_names:
        print(f"Feature '{feature_name}' not found in feature_names list.")
        return

    feature_idx = feature_names.index(feature_name)

    layer_name = f'station_{station_idx}_{layer}'
    if layer_name not in feature_maps:
        print(f"No feature maps found for layer '{layer_name}'.")
        return

    # Get all feature maps for the feature
    # Assuming multiple feature maps per feature
    # Adjust based on your architecture
    feature_map = feature_maps[layer_name][0, feature_idx, :]  # Shape: [time_steps]

    # Determine the number of feature maps
    total_feature_maps = feature_maps[layer_name].shape[1]

    # Adjust num_features if necessary
    num_features = min(num_features, total_feature_maps)

    # Set up the plot grid
    rows = 4  # Number of rows for feature maps
    cols = 4  # Number of columns for feature maps
    total_plots = rows * cols

    plt.figure(figsize=(20, 20))

    # Plot Original Feature
    plt.subplot(rows + 1, cols, 1)
    plt.plot(original_data, color='blue')
    plt.title(f'Original Feature: {feature_name}')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{feature_name} Value')
    plt.grid(True)

    # Plot Feature Maps
    for i in range(num_features):
        plt.subplot(rows + 1, cols, i + 2)
        plt.plot(feature_maps[layer_name][0, i, :].numpy(), color='red')
        plt.title(f'Feature Map {i + 1} from {layer_name}')
        plt.xlabel('Time Steps')
        plt.ylabel('Activation')
        plt.grid(True)

    # Hide any unused subplots
    for j in range(num_features + 1, rows * cols + 1):
        plt.subplot(rows + 1, cols, j + 1)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_cnn_feature_maps(feature_maps, station_idx, layer='conv2'):
    """
    Visualize CNN feature maps for a specific station and layer.

    Args:
        feature_maps (dict): Dictionary containing feature maps.
        station_idx (int): Index of the station.
        layer (str): Layer name ('conv1' or 'conv2').
        num_features (int): Number of feature maps to display.
    """
    num_features=15
    feature_names = [
        'Rain', 'RH (%)', 'TD (degC)', 'TDmax (degC)', 'TDmin (degC)',
        'STDwd (deg)', 'Year', 'Wind_x', 'Wind_y', 'Gust_x', 'Gust_y',
        'Day sin', 'Day cos', 'Year sin', 'Year cos'
        ]
  
    layer_name = f'station_{station_idx}_{layer}'
    if layer_name not in feature_maps:
        print(f"No feature maps found for {layer_name}")
        return

    # Get the feature maps tensor: [batch_size, channels, time_steps]
    feature_map = feature_maps[layer_name][0]  # Assuming batch_size=1

    # Select the first `num_features` feature maps
    feature_map = feature_map[:num_features]

    # Plotting
    plt.figure(figsize=(15, 15))
    for i in range(num_features):
        plt.subplot(5, 3, i + 1)
        plt.plot(feature_map[i].numpy())
        plt.title(feature_names[i])
        plt.xlabel('Time Steps')
        plt.ylabel('Activation')
    plt.tight_layout()
    plt.show()


def visualize_attention_weights(attn_weights, layer_idx=0, head_idx=0):
    """
    Visualize attention weights for a specific layer and head.

    Args:
        attn_weights (torch.Tensor): Attention weights tensor of shape [num_layers, batch_size, nhead, seq_len, seq_len]
        layer_idx (int): Index of the Transformer layer.
        head_idx (int): Index of the attention head.
    """
    # Select attention weights for the specified layer and head
    attn_matrix = attn_weights[layer_idx, 0, head_idx].numpy()  # [seq_len, seq_len]

    plt.figure(figsize=(8, 6))
    plt.imshow(attn_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}')
    plt.xlabel('Key Sequence Position')
    plt.ylabel('Query Sequence Position')
    plt.show()


def export_model_to_onnx(model, device, filepath='model.onnx'):
    """
    Export the PyTorch model to ONNX format for visualization with Netron.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        device (str): Device to perform inference on.
        filepath (str): Path to save the ONNX model.
    """
    # Create dummy inputs matching the model's expected input
    dummy_input = torch.randn(1, ADVANCED_MODEL_PARAMS['num_stations'],
                              WINDOW_PARAMS['input_width'],
                              ADVANCED_MODEL_PARAMS['feature_dim']).to(device)
    #dummy_lat = torch.from_numpy(east_normalized).float().unsqueeze(1).to(device)
    #dummy_lon = torch.from_numpy(north_normalized).float().unsqueeze(1).to(device)

    # Export the model
    torch.onnx.export(model,
                      (dummy_input, east_normalized, north_normalized),
                      filepath,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input_window', 'lat', 'lon'],
                      output_names=['output'],
                      dynamic_axes={'input_window': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
    print(f"Model exported to {filepath}")

    # Optionally, automatically open Netron for visualization
    # Uncomment the following line if you want Netron to open automatically
    # netron.start(filepath)
    print("You can visualize the model architecture using Netron by running the following command:")
    print(f"    netron {filepath}")
    print("Or open the ONNX file in the Netron desktop application or web interface at https://netron.app/")


def visualize_model_summary(model,east_normalized,north_normalized, device):
    """
    Print the model summary using torchsummary.

    Args:
        model (torch.nn.Module): The PyTorch model.
        device (str): Device to load the model on ('cpu' or 'cuda').
    """
    # Define the input size: (num_stations, time_steps, feature_dim)
    input_size = [(32,
        ADVANCED_MODEL_PARAMS['num_stations'],
                  WINDOW_PARAMS['input_width'],
                  ADVANCED_MODEL_PARAMS['feature_dim']),
                  (ADVANCED_MODEL_PARAMS['num_stations'],1),
                  (ADVANCED_MODEL_PARAMS['num_stations'],1)
                  ]
    print("Model Summary:")
    summary(model, input_size)


def visualize_computational_graph(model, device,east_normalized,north_normalized, filepath='computational_graph'):
    """
    Visualize the computational graph using torchviz.

    Args:
        model (torch.nn.Module): The PyTorch model.
        device (str): Device to perform inference on.
        filepath (str): Path to save the computational graph visualization.
    """
    # Create dummy inputs
    dummy_input = torch.randn(1, ADVANCED_MODEL_PARAMS['num_stations'],
                              WINDOW_PARAMS['input_width'],
                              ADVANCED_MODEL_PARAMS['feature_dim']).to(device)
    # Forward pass
    output = model(dummy_input, east_normalized.to(device), north_normalized.to(device))

    # Generate the computational graph
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png'  # Supported formats: pdf, png, svg, etc.
    dot.render(filepath)
    print(f"Computational graph saved as '{filepath}.png'")


if __name__ == "__main__":
    # Device configuration
    device = PARAMS['device']

    east = []
    north = []
    # 1. Load DataFrames from CSVs
    filenames = PARAMS['fileNames']  # List of filenames
    dfs = []
    for filename in filenames:
        df = load_pkl_file(filename)
        dfs.append(df)
        east.append(STATIONS_COORDINATES[filename][0])
        north.append(STATIONS_COORDINATES[filename][1])

    east = np.array(east)
    north = np.array(north)
    east_normalized, north_normalized = normalize_coordinates(east, north)

    print("Original size of data:")
    for df in dfs:
        print(df.shape)

    timeEncode(dfs)

    df_cleaned_list = drop_nan_rows_multiple(dfs)

    print("Size of data after drop_nan_rows_multiple:")
    for i, df in enumerate(df_cleaned_list):
        print(f"Station {i}: {df.shape}")

    # Extract Feature Values (Use Cleaned Data)
    list_of_values = [df.values for df in df_cleaned_list]

    # Train/Validation Split per Station
    train_size = int(0.8 * len(list_of_values[0]))
    list_of_train_data = []
    list_of_val_data = []
    for values in list_of_values:
        train_data = values[:train_size]
        val_data = values[train_size:]
        list_of_train_data.append(train_data)
        list_of_val_data.append(val_data)

    # Combine Data into 3D Arrays
    combined_train_data = np.stack(list_of_train_data, axis=1)  # (T_train, num_stations, num_features)
    combined_val_data = np.stack(list_of_val_data, axis=1)  # (T_val, num_stations, num_features)

    input_width = WINDOW_PARAMS['input_width']
    label_width = WINDOW_PARAMS['label_width']
    shift = WINDOW_PARAMS['shift']

    # Ensure consistent column indexing
    representative_df = df_cleaned_list[0]
    column_indices = {name: i for i, name in enumerate(representative_df.columns)}
    label_columns = [column_indices[WINDOW_PARAMS['label_columns'][0]]]

    # Define target station index
    target_station_idx = PARAMS['target_station_id']  # Ensure this is 0-based and within range

    scaler_dir = os.path.join(os.path.dirname(__file__), 'output', 'scalers')
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'output', 'checkpoints', 'best_checkpoint.pth')

    # Load scalers
    scalers = load_scalers(scaler_dir=scaler_dir)

    # Load model parameters from ADVANCED_MODEL_PARAMS
    model_params = ADVANCED_MODEL_PARAMS.copy()

    # Load model
    model = load_model_for_inference(checkpoint_path, model_params, device=device)

    # Register hooks to CNN layers
    for station_idx, station_cnn in enumerate(model.station_cnns):
        conv1_name = f'station_{station_idx}_conv1'
        conv2_name = f'station_{station_idx}_conv2'
        station_cnn.conv1.register_forward_hook(get_cnn_hook(conv1_name))
        station_cnn.conv2.register_forward_hook(get_cnn_hook(conv2_name))

    # Visualize model summary
    #visualize_model_summary(model,east_normalized.to(device),north_normalized.to(device), device)

    # Visualize computational graph
    #visualize_computational_graph(model, device,east_normalized,north_normalized, filepath='computational_graph')

    # Export model to ONNX for visualization with Netron
    #export_model_to_onnx(model, device, filepath='model.onnx')

    # Define prediction mode
    target_col_index = label_columns[0]

    input_window, actual_temp = load_window_multi_station(
        data_np=combined_val_data,
        window_size=input_width,
        shift=shift,
        label_width=label_width,
        scalers=scalers,
        target_column_index=target_col_index,
        idx=0
    )
    y_pred_scaled, attn_weights = predict(model, input_window, east_normalized, north_normalized,
                                          device=device)
    # reduce input_window from (1, 2, 72, 15) to selectinf the first station and the first feature
    input_window = input_window[:, 0, :, 2].unsqueeze(1)
    # reduce input_window from (1, 1, 72) to np.array (72,)
    input_window = input_window.squeeze(0).squeeze(0).numpy()
    visualize_feature_with_all_feature_maps(
        original_data=input_window,
        feature_maps=cnn_feature_maps,
        feature_name='TD (degC)',
        station_idx=0,
        layer='conv2',
        num_features=16  # Adjust based on grid layout (e.g., 4x4=16)
    )

    prediction_mode = 'analyze'  # Options: 'single', 'batch', 'analyze'

    if prediction_mode == 'analyze':
        # Comprehensive analysis over validation data
        total_window_size = input_width + shift - 1 + label_width
        end = len(combined_val_data) - total_window_size
        end = 30
        predictions = []
        actual_temps = []
        all_attention_weights = []  # To store attention weights for visualization
        for i in tqdm(range(0, end), desc="Predicting"):
            try:
                input_window, actual_temp = load_window_multi_station(
                    data_np=combined_val_data,
                    window_size=input_width,
                    shift=shift,
                    label_width=label_width,
                    scalers=scalers,
                    target_column_index=target_col_index,
                    idx=i
                )
                y_pred_scaled, attn_weights = predict(model, input_window, east_normalized, north_normalized,
                                                      device=device)

                # Inverse transform
                target_scaler = scalers[ADVANCED_MODEL_PARAMS['target_station_idx']]
                dummy = np.zeros((y_pred_scaled.shape[0], target_scaler.mean_.shape[0]))
                dummy[:, target_col_index] = y_pred_scaled[:, 0]
                y_pred_original = target_scaler.inverse_transform(dummy)[:, target_col_index]

                if len(y_pred_original) != len(actual_temp):
                    continue
                predictions.extend(y_pred_original)
                actual_temps.extend(actual_temp)

                # Store attention weights (optional)
                if attn_weights is not None:
                    all_attention_weights.append(attn_weights.cpu())
            except ValueError as ve:
                print(f"Skipping index {i}: {ve}")
                continue

        # Perform analysis (assuming analyze is a custom function)
        #analyze(predictions, actual_temps, WINDOW_PARAMS['label_width'])

        # Plot predictions vs actual
        plt.figure(figsize=(15, 7))
        plt.plot(actual_temps, label='Actual Temperature', color='blue')
        plt.plot(predictions, label='Predicted Temperature', color='red')
        plt.xlabel('Time Steps')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Prediction Analysis')
        plt.legend()
        plt.show()

        # Calculate evaluation metrics
        mae = mean_absolute_error(actual_temps, predictions)
        rmse = np.sqrt(mean_squared_error(actual_temps, predictions))
        print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")

        input_window, actual_temp = load_window_multi_station(
            data_np=combined_val_data,
            window_size=input_width,
            shift=shift,
            label_width=label_width,
            scalers=scalers,
            target_column_index=target_col_index,
            idx=0
        )
        y_pred_scaled, attn_weights = predict(model, input_window, east_normalized, north_normalized,
                                              device=device)
        # reduce input_window from (1, 2, 72, 15) to selectinf the first station and the first feature
        input_window = input_window[:, 0, :, 2].unsqueeze(1)
        # reduce input_window from (1, 1, 72) to np.array (72,)
        input_window = input_window.squeeze(0).squeeze(0).numpy()

        visualize_feature_with_all_feature_maps(
            original_data=input_window,
            feature_maps=cnn_feature_maps,
            feature_name='TD (degC)',
            station_idx=0,
            layer='conv2',
            num_features=16  # Adjust based on grid layout (e.g., 4x4=16)
        )

        # Visualize CNN Feature Maps (Example for Station 0, conv1)
        visualize_cnn_feature_maps(cnn_feature_maps, station_idx=0, layer='conv1')

        # Visualize Attention Weights (Example for first sample, first layer, first head)
        if all_attention_weights:
            sample_attn = all_attention_weights[0]  # [num_layers, batch_size, nhead, seq_len, seq_len]
            visualize_attention_weights(sample_attn, layer_idx=0, head_idx=0)

    else:
        print(f"Invalid prediction mode: {prediction_mode}. Choose from 'single', 'batch', 'analyze'.")

    print("\nAll visualizations completed successfully.")

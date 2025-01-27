# inference.py

import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  

from backend.Model_Pytorch.common.data import (
    load_pkl_file ,
    timeEncode,
    normalize_coordinates, 
    drop_nan_rows_multiple
)

from backend.Model_Pytorch.AdvancedModel.model import TargetedWeatherPredictionModel
from backend.Model_Pytorch.AdvancedModel.parameters import PARAMS, WINDOW_PARAMS, ADVANCED_MODEL_PARAMS, STATIONS_COORDINATES
from backend.Model_Pytorch.common.analyze import analyze

def load_scalers(scaler_dir='./output/scalers'):
    """
    Load the previously saved scalers for each station.
    Assumes scalers are saved as 'scaler_station0.pkl', 'scaler_station1.pkl', etc.
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
    Create a model with the same architecture,
    load checkpoint, and return it in eval mode.
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
   
    total_window_size = window_size + shift - 1 + label_width
    if idx + total_window_size > len(data_np):
        raise ValueError(f"Index {idx} with window size {total_window_size} exceeds data length {len(data_np)}.")

    # Extract input window: shape (window_size, num_stations, num_features)
    window = data_np[idx:idx + window_size, :, :]  # (window_size, num_stations, num_features)

    # Extract target: shape (label_width, )
    target_start = idx + window_size + shift - 1
    target_end = target_start + label_width
    actual_target = data_np[target_start:target_end, ADVANCED_MODEL_PARAMS['target_station_idx'], target_column_index]
    #actual_target_mean = actual_target.mean()  # Aggregate if label_width >1

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

    # Convert to torch.Tensor and reshape to (1, num_stations, time_steps, feature_dim)
    window_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0)
   # batch_size, num_stations, time_steps, feature_dim = x.size()

    window_tensor = window_tensor.permute(0, 2, 1, 3)

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

    Returns:
        float: Predicted target value in original scale (e.g., Temperature in °C).
    """
    input_tensor = input_window.to(device)
    lat = lat.to(device)
    lon = lon.to(device)

    # Model prediction (scaled)
    output_scaled = model(input_tensor, lat, lon)

    # Convert to numpy
    output_scaled_np = output_scaled.squeeze(-1).cpu().numpy().reshape(-1, 1)  # Scalar
    return output_scaled_np


if __name__ == "__main__":
    east = []
    north = []
    # 3. Load DataFrames from CSVs
    filenames = PARAMS['fileNames']  # List of 5 filenames
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
    device = PARAMS['device']
    model = load_model_for_inference(checkpoint_path, model_params, device=device)

    # Define prediction mode
    prediction_mode = 'analyze'  # Options: 'single', 'batch', 'analyze'
    
    if prediction_mode == 'single':
        # Predict a single window
        idx = 500  # Example index
        input_window, actual_temp = load_window_multi_station(
            data_np=combined_val_data,
            window_size=input_width,
            shift=shift,
            label_width=label_width,
            scalers=scalers,
            target_column_index=label_columns,
            idx=idx
        )
        y_pred_scaled = predict(model, input_window, east_normalized, north_normalized, device=device)
        
        # Inverse transform the prediction for the target station
        target_scaler = scalers[ADVANCED_MODEL_PARAMS['target_station_idx']]
        # Create a dummy array for inverse_transform
        dummy = np.zeros((y_pred_scaled.shape[0], target_scaler.mean_.shape[0]))
        dummy[:, label_columns] = y_pred_scaled  # Assign predicted values to the target column
        y_pred_original = target_scaler.inverse_transform(dummy)[:, label_columns]
        
        print(f"Predicted temperatures (°C): {y_pred_original}")
        print(f"Actual temperatures (°C): {actual_temp}")
    
    elif prediction_mode == 'batch':
        # Predict in batch over validation data
        total_window_size = input_width + shift - 1 + label_width
        end = len(combined_val_data) - total_window_size
        predictions = []
        actual_temps = []
        for i in tqdm(range(0, end, label_width), desc="Predicting"):
            try:
                input_window, actual_temp = load_window_multi_station(
                    data_np=combined_val_data,
                    window_size=input_width,
                    shift=shift,
                    label_width=label_width,
                    scalers=scalers,
                    target_column_index=label_columns,
                    idx=i
                )
                y_pred_scaled = predict(model, input_window, east_normalized, north_normalized, device=device)
                
                # Inverse transform
                target_scaler = scalers[ADVANCED_MODEL_PARAMS['target_station_idx']]
                dummy = np.zeros((y_pred_scaled.shape[0], target_scaler.mean_.shape[0]))
                dummy[:, label_columns] = y_pred_scaled
                y_pred_original = target_scaler.inverse_transform(dummy)[:, label_columns]
                
                predictions.extend(y_pred_original)
                actual_temps.extend(actual_temp)
            except ValueError as ve:
                print(f"Skipping index {i}: {ve}")
                continue
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.plot(actual_temps, label='Actual', color='blue')
        plt.plot(predictions, label='Predicted', color='red')
        plt.xlabel('Time Steps')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Prediction')
        plt.legend()
        plt.show()
    
    elif prediction_mode == 'analyze':
        # Comprehensive analysis over validation data
        total_window_size = input_width + shift - 1 + label_width
        end = len(combined_val_data) - total_window_size
        end = 30
        predictions = []
        actual_temps = []
        for i in tqdm(range(0, end), desc="Predicting"):
            try:
                input_window, actual_temp = load_window_multi_station(
                    data_np=combined_val_data,
                    window_size=input_width,
                    shift=shift,
                    label_width=label_width,
                    scalers=scalers,
                    target_column_index=label_columns,
                    idx=i
                )
                y_pred_scaled = predict(model, input_window, east_normalized, north_normalized,  device=device)
                
                # Inverse transform
                target_scaler = scalers[ADVANCED_MODEL_PARAMS['target_station_idx']]
                dummy = np.zeros((y_pred_scaled.shape[0], target_scaler.mean_.shape[0]))
                dummy[:, label_columns] = y_pred_scaled
                y_pred_original = target_scaler.inverse_transform(dummy)[:, label_columns]

                if len(y_pred_original) != len(actual_temp):
                    continue
                predictions.append(y_pred_original)
                actual_temps.append(actual_temp)
            except ValueError as ve:
                print(f"Skipping index {i}: {ve}")
                continue
        analyze(predictions, actual_temps,WINDOW_PARAMS['label_width'])

        # Perform analysis (assuming analyze is a custom function)
        # You may need to adjust this based on your actual analyze function
        # For demonstration, let's plot the results
        plt.figure(figsize=(15, 7))
        plt.plot(actual_temps, label='Actual Temperature', color='blue')
        plt.plot(predictions, label='Predicted Temperature', color='red')
        plt.xlabel('Time Steps')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Prediction Analysis')
        plt.legend()
        plt.show()
        
        # You can also calculate metrics like MAE, RMSE, etc.
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(actual_temps, predictions)
        rmse = np.sqrt(mean_squared_error(actual_temps, predictions))
        print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")
    
    else:
        print(f"Invalid prediction mode: {prediction_mode}. Choose from 'single', 'batch', 'analyze'.")

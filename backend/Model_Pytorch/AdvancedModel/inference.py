# inference.py

import torch
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
from backend.Model_Pytorch.AdvancedModel.parameters import PARAMS, WINDOW_PARAMS, ADVANCED_MODEL_PARAMS, STATIONS_COORDINATES, STATIONS_LIST
from backend.Model_Pytorch.common.analyze import analyze
from backend.Model_Pytorch.common.import_and_process_data import get_prccessed_latest_data_by_hour_and_station

import numpy as np
import json
from datetime import datetime, timedelta

def generate_forecast_json(city_name, date_str, starting_hour, temperatures, output_file):
    """
    Generates a JSON file containing forecast data.

    Parameters:
    - city_name (str): Name of the city.
    - date_str (str): Starting date in "YYYY-MM-DD" format.
    - starting_hour (int): Starting hour in 24-hour format (0-23).
    - temperatures (np.ndarray): NumPy array of temperature readings.
    - output_file (str): Path to the output JSON file.
    """
    
    # Validate inputs
    if not isinstance(city_name, str):
        raise TypeError("city_name must be a string.")
    
    try:
        current_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("date_str must be in 'YYYY-MM-DD' format.")
    
    if not (0 <= starting_hour <= 23):
        raise ValueError("starting_hour must be between 0 and 23.")
    
    if not isinstance(temperatures, np.ndarray):
        raise TypeError("temperatures must be a NumPy array.")
    
    # Initialize forecast_data dictionary
    forecast_data = {}
    
    current_hour = starting_hour
    
    for temp in temperatures:
        # Format the current date
        date_key = current_date.strftime("%Y-%m-%d")
        
        # Initialize the date entry if it doesn't exist
        if date_key not in forecast_data:
            forecast_data[date_key] = {"hourly": {}}
        
        # Format the current time
        time_str = "{:02d}:00".format(current_hour)
        
        # Assign the temperature, formatted to one decimal place
        forecast_data[date_key]["hourly"][time_str] = {
            "temperature": "{:.1f}".format(temp)
        }
        
        # Increment the hour
        current_hour += 1
        
        # If hour exceeds 23, reset to 0 and move to the next day
        if current_hour > 23:
            current_hour = 0
            current_date += timedelta(days=1)
    
    # Construct the final JSON structure
    data = {
        "data": {
            "title": city_name,
            "forecast_data": forecast_data
        }
    }
    
    # Write the data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Forecast data successfully written to {output_file}")

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


def load_window_multi_station_return_only_input_window(data_np, window_size, shift, scalers, idx=0):
    total_window_size = window_size
    if idx + total_window_size > len(data_np):
        raise ValueError(f"Index {idx} with window size {total_window_size} exceeds data length {len(data_np)}.")

    # Extract input window: shape (window_size, num_stations, num_features)
    window = data_np[idx:idx + window_size, :, :]  # (window_size, num_stations, num_features)

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

    return window_tensor
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

    print("Size of data after drop_nan_rows_multiple:")
    for i, df in enumerate(dfs):
        print(f"Station {i}: {df.shape}")

    # Extract Feature Values (Use Cleaned Data)
    list_of_values = [df.values for df in dfs]

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
    representative_df = dfs[0]
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
    prediction_mode = 'analyze'  # Options: 'live', 'analyze'
    target_col_index = label_columns[0]

    
    if prediction_mode == 'analyze':
        # Comprehensive analysis over validation data
        total_window_size = input_width + shift - 1 + label_width
        end = len(combined_val_data) - total_window_size
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
                    target_column_index=target_col_index,
                    idx=i
                )
                y_pred_scaled = predict(model, input_window, east_normalized, north_normalized,  device=device)

                # Inverse transform
                target_scaler = scalers[ADVANCED_MODEL_PARAMS['target_station_idx']]
                dummy = np.zeros((y_pred_scaled.shape[0], target_scaler.mean_.shape[0]))
                dummy[:, target_col_index] = y_pred_scaled[:, 0]
                y_pred_original = target_scaler.inverse_transform(dummy)[:, target_col_index]

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
    elif prediction_mode == 'live':
        # getting the window to predict from function
        # need to implement in the future get_window_live(input_width)
        
        dataframes, last_hour, last_date, success = get_prccessed_latest_data_by_hour_and_station(STATIONS_LIST, input_width)
        last_hour = int(last_hour.split(':')[0])
        print(f"len of df: {len(dataframes)}")
        print(f"len of df[0]: {len(dataframes[list(dataframes.keys())[0]])}")
        print(f"len of df[1]: {len(dataframes[list(dataframes.keys())[1]])}")
        print(f"Last hour: {last_hour}")
        print(f"Last hour date: {last_date}")
        print(f"success: {success}")
        # datafreames is a dictionary with the station name as key and the dataframe as value - convering it into a list of dataframes
        dataframes_list = [dataframes[station] for station in STATIONS_LIST]

        list_of_values = [df.values for df in dataframes_list]
        combined_window = np.stack(list_of_values, axis=1) 
        input_window = load_window_multi_station_return_only_input_window(
            data_np=combined_window,
            window_size=input_width,
            shift=shift,
            scalers=scalers
        )

        y_pred_scaled = predict(model, input_window, east_normalized, north_normalized,  device=device)
        target_scaler = scalers[ADVANCED_MODEL_PARAMS['target_station_idx']]
        dummy = np.zeros((y_pred_scaled.shape[0], target_scaler.mean_.shape[0]))
        dummy[:, target_col_index] = y_pred_scaled[:, 0]
        y_pred_original = target_scaler.inverse_transform(dummy)[:, target_col_index]
        generate_forecast_json(PARAMS['target_station_desplay_name'], last_date, last_hour+1, y_pred_original, "forecast.json")


    else:
        print(f"Invalid prediction mode: {prediction_mode}. Choose from 'single', 'batch', 'analyze'.")

# inference.py

import torch
import pickle
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  
import numpy as np
import json
from datetime import datetime, timedelta
import importlib.util
import sys
from pathlib import Path

from backend.Model_Pytorch.common.data import load_pkl_file ,normalize_coordinates
from backend.Model_Pytorch.AdvancedModel.model import TargetedWeatherPredictionModel
from backend.Model_Pytorch.AdvancedModel.parameters import PARAMS, WINDOW_PARAMS, ADVANCED_MODEL_PARAMS, STATIONS_COORDINATES, STATIONS_LIST,INFERENCE_PARAMS
from backend.Model_Pytorch.common.import_and_process_data import get_prccessed_latest_data_by_hour_and_station

def load_params(params_path):
    # Convert path to absolute if it's relative
    params_path = Path(params_path).resolve()

    # Create a module name dynamically (avoid conflicts)
    module_name = "params_module"

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, params_path)
    params_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = params_module
    spec.loader.exec_module(params_module)

    return params_module  # Now you can access its attributes


def flatten_data(predictions, actuals):
    flat_predictions = [temp for window in predictions for temp in window]
    flat_actuals = [temp for window in actuals for temp in window]

    data = pd.DataFrame({
        'Predicted': flat_predictions,
        'Actual': flat_actuals
    })

    data['Error'] = data['Predicted'] - data['Actual']
    return data

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


def load_window_multi_station_return_only_input_window(data_np, window_size, scalers, idx=0):
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
    """
    1. define INFERENCE_PARAMS in your file - all the INFERENCE_PARAMS are mendatory in addittion to that all the parameters that are in section 2

    2. there is an assumption that yours parameters file (not what define in the infarance parameters nor what in the folders)
        has the same values in these parameters:

        PARAMS['fileNames'] - list of the stations names, if you have different names it wont make sense 
        PARAMS['target_station_id'] - the index of the target station in the list of stations
        ADVANCED_MODEL_PARAMS['target_station_idx'] - the index of the target station in the list of stations
        WINDOW_PARAMS['label_columns'] - the label column
        PARAMS['device']
    3.  WINDOW_PARAMS['input_width'] - the window size - support defrrent sizes - didnt checkd yet


    """
    inference_mode = 'live'  # Options: 'live', 'analyze'
    analyze_stop_at = 0  # Number of predictions to analyze

    parameters_files = [] # load parameters files
    for path in INFERENCE_PARAMS['params_path']:
        parameters_files.append(load_params(path))

    east = []
    north = []
    filenames = PARAMS['fileNames'] 
    for filename in filenames: # get the coordinates of the stations
        east.append(STATIONS_COORDINATES[filename][0])
        north.append(STATIONS_COORDINATES[filename][1])

    east = np.array(east)
    north = np.array(north)
    east_normalized, north_normalized = normalize_coordinates(east, north)
    
    scalers = load_scalers(scaler_dir=INFERENCE_PARAMS['scaler_folder_path'])
    #model_params = ADVANCED_MODEL_PARAMS.copy()

    
    model_params = []
    for params_file in parameters_files:
        model_params.append(params_file.ADVANCED_MODEL_PARAMS)

    models = []
    for i, weights_path in enumerate(INFERENCE_PARAMS['weights_paths']):
        model = load_model_for_inference(weights_path, model_params[i], device=PARAMS['device'])
        models.append(model)

    window_params = []
    for params_file in parameters_files:
        window_params.append(params_file.WINDOW_PARAMS)
    
    max_input_width = max([window_param['input_width'] for window_param in window_params])

    device = PARAMS['device']
    target_station_idx = PARAMS['target_station_id']  

    
    if inference_mode == 'live':        
        dataframes, last_hour, last_date, success = get_prccessed_latest_data_by_hour_and_station(STATIONS_LIST, max_input_width)
        last_hour = int(last_hour.split(':')[0])
        if False:
            print(f"len of df: {len(dataframes)}")
            print(f"len of df[0]: {len(dataframes[list(dataframes.keys())[0]])}")
            print(f"len of df[1]: {len(dataframes[list(dataframes.keys())[1]])}")
            print(f"Last hour: {last_hour}")
            print(f"Last hour date: {last_date}")
            print(f"success: {success}")
        # datafreames is a dictionary with the station name as key and the dataframe as value - convering it into a list of dataframes
        dataframes_list = [dataframes[station] for station in STATIONS_LIST]
        
        representative_df = dataframes_list[0]
        column_indices = {name: i for i, name in enumerate(representative_df.columns)}
        label_columns = [column_indices[WINDOW_PARAMS['label_columns'][0]]]
        target_col_index = label_columns[0]

        list_of_values = [df.values for df in dataframes_list]
        combined_window = np.stack(list_of_values, axis=1) 
    
        target_scaler = scalers[ADVANCED_MODEL_PARAMS['target_station_idx']]
        predictions_of_models = []
        for i, model in enumerate(models):
            input_width = parameters_files[i].WINDOW_PARAMS['input_width']
            input_window = load_window_multi_station_return_only_input_window(
                data_np=    combined_window[-input_width:],
                window_size=input_width,
                scalers=    scalers
            )
            y_pred_scaled = predict(model, input_window, east_normalized, north_normalized,  device=device)
            dummy = np.zeros((y_pred_scaled.shape[0], target_scaler.mean_.shape[0]))
            dummy[:, target_col_index] = y_pred_scaled[:, 0]
            y_pred_original = target_scaler.inverse_transform(dummy)[:, target_col_index]
            predictions_of_models.append(y_pred_original)

        generate_forecast_json(PARAMS['target_station_desplay_name'], last_date, last_hour+1, np.concatenate(predictions_of_models), "forecast.json")

    elif inference_mode == 'analyze':
        dfs = []
        for filename in filenames:
            df = load_pkl_file(filename)
            dfs.append(df)

        print("Original size of data:")
        for i, df in enumerate(dfs):
            print(f"Station {i}: {df.shape}")

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

        # Ensure consistent column indexing
        representative_df = dfs[0]
        column_indices = {name: i for i, name in enumerate(representative_df.columns)}
        label_columns = [column_indices[WINDOW_PARAMS['label_columns'][0]]]

        # Define target station index
        target_col_index = label_columns[0]

        for i, model in enumerate(models):
            input_width = parameters_files[i].WINDOW_PARAMS['input_width']
            shift = parameters_files[i].WINDOW_PARAMS['shift']
            label_width = parameters_files[i].WINDOW_PARAMS['label_width']

            total_window_size = input_width + shift - 1 + label_width
            end = len(combined_val_data) - total_window_size if analyze_stop_at == 0 else min(analyze_stop_at, len(combined_val_data) - total_window_size)
            
            predictions = []
            actual_temps = []
            for j in tqdm(range(0, end), desc="Predicting"):
                try:
                    input_window, actual_temp = load_window_multi_station(
                        data_np=combined_val_data,
                        window_size=    input_width,
                        shift=          shift,
                        label_width=    label_width,
                        scalers=        scalers,
                        target_column_index=target_col_index,
                        idx=    j
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
                    print(f"Skipping index {j}: {ve}")
                    continue
            
            output_dir_for_all = os.path.join(os.path.dirname(__file__), INFERENCE_PARAMS['inference_output_path'])
            os.makedirs(output_dir_for_all, exist_ok=True)
            output_dir_per_folder = os.path.join(os.path.dirname(__file__), INFERENCE_PARAMS['inference_output_path_per_model'][i])
            os.makedirs(output_dir_per_folder, exist_ok=True)
            predictions_actuals_df = flatten_data(predictions, actual_temps)
            predictions_actuals_df['input_width'] = input_width
            predictions_actuals_df['label_width'] = label_width
            predictions_actuals_df.to_csv(os.path.join(output_dir_per_folder, f'{i}_predictions_{i}.csv'), index=False)
            predictions_actuals_df.to_csv(os.path.join(output_dir_for_all, f'{i}_predictions_{i}.csv'), index=False)
    else:
        print(f"Invalid inference_mode : {inference_mode}.")
# main.py
import os
import pandas as pd
import numpy as np

from backend.Model_Pytorch.common.window_generator_multiple_stations import WindowGeneratorMultipleStations
from backend.Model_Pytorch.common.data import (
    load_pkl_file , 
    normalize_data_collective, 
    normalize_data_independent,
    normalize_coordinates, 
    timeEncode, 
    drop_nan_rows_multiple
)

from backend.Model_Pytorch.AdvancedModel.model import TargetedWeatherPredictionModel
from backend.Model_Pytorch.AdvancedModel.train import train_model
from backend.Model_Pytorch.AdvancedModel.parameters import PARAMS, WINDOW_PARAMS, ADVANCED_MODEL_PARAMS, STATIONS_COORDINATES

if __name__ == "__main__":
    east = []
    north = []
    # 3. Load DataFrames from CSVs
    filenames = PARAMS['fileNames']
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
    combined_val_data = np.stack(list_of_val_data, axis=1)      # (T_val, num_stations, num_features)

    # Normalize Data Independently per Station
    scaler_directory = os.path.join(os.path.dirname(__file__), 'output', 'scalers')
    train_data_scaled, val_data_scaled, scalers = normalize_data_independent(
        train_data=combined_train_data,
        val_data=combined_val_data,
        scaler_dir=scaler_directory
    )

    # Create Datasets
    input_width = WINDOW_PARAMS['input_width']
    label_width = WINDOW_PARAMS['label_width']
    shift = WINDOW_PARAMS['shift']

    # Ensure consistent column indexing
    representative_df = df_cleaned_list[0]
    column_indices = {name: i for i, name in enumerate(representative_df.columns)}
    label_columns = [column_indices[WINDOW_PARAMS['label_columns'][0]]]

    # Define target station index
    target_station_idx = PARAMS['target_station_id']  # Ensure this is 0-based and within range

    # Instantiate Datasets
    train_dataset = WindowGeneratorMultipleStations(
        data=train_data_scaled, 
        input_width=input_width, 
        label_width=label_width, 
        shift=shift, 
        label_columns=label_columns,
        target_station_idx=target_station_idx
    )

    val_dataset = WindowGeneratorMultipleStations(
        data=val_data_scaled, 
        input_width=input_width, 
        label_width=label_width, 
        shift=shift, 
        label_columns=label_columns,
        target_station_idx=target_station_idx
    )

    device = PARAMS['device']
    print(f"Using device: {device}")

    model = TargetedWeatherPredictionModel(ADVANCED_MODEL_PARAMS.copy())

    # 14. Train the Model
    train_model(
        train_dataset,
        val_dataset,
        model,
        epochs=PARAMS['epochs'],
        batch_size=32,
        lr=1e-5,
        checkpoint_dir=os.path.join(os.path.dirname(__file__),'output','checkpoints'),
        resume=PARAMS['resume'],
        device=device,
        coord=[east_normalized, north_normalized]  # Ensure these are on the correct device
    )
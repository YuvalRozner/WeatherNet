# main.py
import os
import numpy as np

from backend.Model_Pytorch.common.window_generator_multiple_stations import WindowGeneratorMultipleStations
from backend.Model_Pytorch.common.data import (
    load_pkl_file , 
    normalize_data_independent,
    normalize_coordinates
)
from backend.Model_Pytorch.AdvancedModel.model import TargetedWeatherPredictionModel
from backend.Model_Pytorch.AdvancedModel.train import train_model
from backend.Model_Pytorch.AdvancedModel.parameters import PARAMS, WINDOW_PARAMS,TRAIN_PARAMS, ADVANCED_MODEL_PARAMS, STATIONS_COORDINATES

if __name__ == "__main__":
    if os.path.exists(PARAMS['output_path']) and TRAIN_PARAMS['resume'] is False:
        print(f"Error: Directory {PARAMS['output_path']} already exists. Please remove it or set resume=True.")
        exit(1)

    print(f"Using device: {PARAMS['device']}")
    east = []
    north = []
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

    print("size of data:")
    for i, df in enumerate(dfs):
        print(f"Station {i}: {df.shape}")

    list_of_values = [df.values for df in dfs]    # Extract Feature Values

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
    train_data_scaled, val_data_scaled, scalers = normalize_data_independent(
        train_data=combined_train_data,
        val_data=combined_val_data,
        scaler_dir=PARAMS['scalers_path']
    )
    representative_df = dfs[0]
    column_indices = {name: i for i, name in enumerate(representative_df.columns)}
    label_columns = [column_indices[WINDOW_PARAMS['label_columns'][0]]]

    # Instantiate Datasets
    train_dataset = WindowGeneratorMultipleStations(
        data=train_data_scaled, 
        input_width=WINDOW_PARAMS['input_width'],
        label_width=WINDOW_PARAMS['label_width'],
        shift=      WINDOW_PARAMS['shift'],
        label_columns=label_columns,
        target_station_idx=PARAMS['target_station_id']
    )

    val_dataset = WindowGeneratorMultipleStations(
        data=val_data_scaled, 
        input_width=WINDOW_PARAMS['input_width'],
        label_width=WINDOW_PARAMS['label_width'],
        shift=      WINDOW_PARAMS['shift'],
        label_columns=label_columns,
        target_station_idx=PARAMS['target_station_id']
    )

    model = TargetedWeatherPredictionModel(**ADVANCED_MODEL_PARAMS.copy())

    train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=      model,
        coordinates=[east_normalized, north_normalized],  # Ensure these are on the correct device
        epochs=     TRAIN_PARAMS['epochs'],
        batch_size= TRAIN_PARAMS['batch_size'],
        lr=         TRAIN_PARAMS['lr'],
        checkpoint_dir=TRAIN_PARAMS['checkpoint_dir'],
        resume=     TRAIN_PARAMS['resume'],
        device=     TRAIN_PARAMS['device'],
        early_stopping_patience= TRAIN_PARAMS['early_stopping_patience'],
        scheduler_patience=TRAIN_PARAMS['scheduler_patience'],
        scheduler_factor=TRAIN_PARAMS['scheduler_factor'],
        min_lr=TRAIN_PARAMS['min_lr']
    )
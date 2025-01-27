import pickle 
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import torch
from tqdm import tqdm
"""
this file let you load the data of the stations from the pkl files 
and the coordinates of the stations from the json file
"""

def normalize_coordinates(x_coords, y_coords):
    """
    Normalize the X and Y coordinates to the range [0, 1].

    Args:
        x_coords (numpy.ndarray): Array of X coordinates in meters.
        y_coords (numpy.ndarray): Array of Y coordinates in meters.

    Returns:
        tuple: Normalized X and Y coordinates as torch tensors.
    """
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_normalized = (x_coords - x_min) / (x_max - x_min)
    y_normalized = (y_coords - y_min) / (y_max - y_min)

    # Convert to torch tensors
    x_normalized = torch.tensor(x_normalized, dtype=torch.float32).unsqueeze(1)  # [num_stations, 1]
    y_normalized = torch.tensor(y_normalized, dtype=torch.float32).unsqueeze(1)  # [num_stations, 1]

    return x_normalized, y_normalized 

def drop_nan_rows_multiple(df_list, reset_indices=True):
    """
    Removes rows from all DataFrames in the list where any DataFrame has NaN in any column.
    
    Parameters:
    df_list (List[pd.DataFrame]): List of DataFrames to process.
    reset_indices (bool): Whether to reset the index after dropping rows. Defaults to True.
    
    Returns:
    List[pd.DataFrame]: List of cleaned DataFrames.
    """
    if not df_list:
        raise ValueError("The list of DataFrames is empty.")
    
    # Ensure all DataFrames have the same number of rows
    num_rows = df_list[0].shape[0]
    for df in df_list:
        if df.shape[0] != num_rows:
            raise ValueError("All DataFrames must have the same number of rows.")
    
    # Step 1: Identify rows with any NaN in each DataFrame
    nan_indices_list = [df.isnull().any(axis=1) for df in df_list]
    
    # Step 2: Combine the indices where NaNs are present in any DataFrame
    combined_nan = pd.Series([False] * num_rows, index=df_list[0].index)
    for nan_mask in nan_indices_list:
        combined_nan = combined_nan | nan_mask
    
    # Get the indices to drop
    indices_to_drop = combined_nan[combined_nan].index
    
    # Step 3: Drop the identified indices from all DataFrames
    cleaned_df_list = []
    for df in tqdm(df_list, desc="Dropping NaN rows"):
        cleaned_df = df.drop(indices_to_drop)
        if reset_indices:
            cleaned_df = cleaned_df.reset_index(drop=True)
        cleaned_df_list.append(cleaned_df)
    
    return cleaned_df_list

# Define the normalization function
def normalize_coordinates(x_coords, y_coords):
    """
    Normalize the X and Y coordinates to the range [0, 1].
    """
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_normalized = (x_coords - x_min) / (x_max - x_min)
    y_normalized = (y_coords - y_min) / (y_max - y_min)

    # Convert to torch tensors
    x_normalized = torch.tensor(x_normalized, dtype=torch.float32).unsqueeze(1)  # [num_stations, 1]
    y_normalized = torch.tensor(y_normalized, dtype=torch.float32).unsqueeze(1)  # [num_stations, 1]

    return x_normalized, y_normalized

def timeEncode(dataframes):
    day = 24*60*60
    year = (365.2425)*day

    for df in dataframes:
        if 'Date Time' in df.columns:
            timestamp_s = df['Date Time'].map(pd.Timestamp.timestamp)
            df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
            df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
            df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
            df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
            df.drop(columns=['Date Time'], inplace=True)


def preprocessing_tensor_df(df):
    """
    Apply the same preprocessing steps as during training.
    """
    print("preproccessing data...")
    # Slice the DataFrame and create a copy to avoid SettingWithCopyWarning
    df = df[5::6].copy()
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    # Handle 'wv (m/s)'
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    df.loc[bad_wv, 'wv (m/s)'] = 0.0  # Use .loc to modify the original DataFrame
    wv = df.pop('wv (m/s)')

    # Handle 'max. wv (m/s)'
    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    df.loc[bad_max_wv, 'max. wv (m/s)'] = 0.0  # Use .loc to modify the original DataFrame
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Calculate wind x and y components using .loc
    df.loc[:, 'Wx'] = wv * np.cos(wd_rad)
    df.loc[:, 'Wy'] = wv * np.sin(wd_rad)
    df.loc[:, 'max Wx'] = max_wv * np.cos(wd_rad)
    df.loc[:, 'max Wy'] = max_wv * np.sin(wd_rad)

    # Time-based features
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day

    df.loc[:, 'Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df.loc[:, 'Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df.loc[:, 'Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df.loc[:, 'Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df

def normalize_data(train_data, val_data, scaler_path='./scaler.pkl'):
    """
    Fit a StandardScaler on the training data and transform both train and val data.
    Save the scaler to disk for future use.

    Args:
        train_data (np.ndarray): Training data.
        val_data (np.ndarray): Validation data.
        scaler_path (str): Path to save the scaler.

    Returns:
        train_data_scaled (np.ndarray): Scaled training data.
        val_data_scaled (np.ndarray): Scaled validation data.
        scaler (StandardScaler): Fitted scaler object.
    """
    scaler = StandardScaler()
    scaler.fit(train_data)

    train_data_scaled = scaler.transform(train_data)
    val_data_scaled = scaler.transform(val_data)

    # Save the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Scaler saved to {scaler_path}")

    return train_data_scaled, val_data_scaled, scaler

def preprocessing_our_df(df):
    """
    Apply the same preprocessing steps as during training.
    """
    print("preproccessing data...")
    df = df[5::6].copy()
    # drop nan
    df = df.dropna()
    return df

def return_and_save_scaler_normalize_data(train_data, val_data, scaler_path='./scaler.pkl'):
   
    scaler = StandardScaler()
    scaler.fit(train_data)
    
    # Save the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Scaler saved to {scaler_path}")
    
    return scaler

def normalize_data_independent(train_data, val_data, scaler_dir='./scalers'):
    """
    Fit a StandardScaler per station on the training data and transform both train and val data.
    Save each scaler to disk for future use.
    
    Args:
        train_data (np.ndarray): Training data of shape (T_train, num_stations, num_features).
        val_data (np.ndarray): Validation data of shape (T_val, num_stations, num_features).
        scaler_dir (str): Directory path to save the scalers.
        
    Returns:
        train_data_scaled (np.ndarray): Scaled training data of shape (T_train, num_stations, num_features).
        val_data_scaled (np.ndarray): Scaled validation data of shape (T_val, num_stations, num_features).
        scalers (list of StandardScaler): List containing a scaler for each station.
    """
    if not os.path.exists(scaler_dir):
        os.makedirs(scaler_dir)
    
    T_train, num_stations, num_features = train_data.shape
    T_val = val_data.shape[0]
    
    # Initialize arrays to hold scaled data
    train_data_scaled = np.zeros_like(train_data)
    val_data_scaled = np.zeros_like(val_data)
    
    scalers = []
    
    for station_idx in range(num_stations):
        scaler = StandardScaler()
        
        # Extract training data for the current station
        train_station_data = train_data[:, station_idx, :]  # Shape: (T_train, num_features)
        
        # Fit the scaler on training data
        scaler.fit(train_station_data)
        scalers.append(scaler)
        
        # Transform training and validation data for the current station
        train_data_scaled[:, station_idx, :] = scaler.transform(train_station_data)
        val_data_scaled[:, station_idx, :] = scaler.transform(val_data[:, station_idx, :])
        
        # Save the scaler for the current station
        scaler_path = os.path.join(scaler_dir, f'scaler_station_{station_idx}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler for Station {station_idx} saved to {scaler_path}")
    
    return train_data_scaled, val_data_scaled, scalers

def normalize_data_collective(train_data, val_data, scaler_path='./scaler.pkl'):
    """
    Fit a single StandardScaler across all stations and features.
    
    Args:
        train_data (np.ndarray): Training data of shape (T_train, num_stations, num_features).
        val_data (np.ndarray): Validation data of shape (T_val, num_stations, num_features).
        scaler_path (str): Path to save the scaler.
        
    Returns:
        train_scaled (np.ndarray), val_scaled (np.ndarray), scaler (StandardScaler)
    """
    T_train, num_stations, num_features = train_data.shape
    T_val = val_data.shape[0]
    
    # Reshape to (T_train*num_stations, num_features)
    train_reshaped = train_data.reshape(-1, num_features)
    val_reshaped = val_data.reshape(-1, num_features)
    
    scaler = StandardScaler()
    scaler.fit(train_reshaped)
    
    train_scaled = scaler.transform(train_reshaped).reshape(train_data.shape)
    val_scaled = scaler.transform(val_reshaped).reshape(val_data.shape)
    
    # Save the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    return train_scaled, val_scaled, scaler

def load_pkl_file(station_name):
    current_path = os.path.dirname(__file__)
    file_path = f"{current_path}\\..\\..\\..\\data\\{station_name}.pkl"
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"data succsesfuly loaded from {file_path}")
        return data
    except Exception as e:
        print(f"Failed to load file:\n{e}")
        return None

def openJsonFile():
    current_path = os.path.dirname(__file__)
    file_path = f"{current_path}\\..\\..\\data code files\\stations_details_updated.json"
    with open(file_path) as file:
        stations = json.load(file)
    return stations

def loadCoordinatesNewIsraelData(stations_details, station_name):
    for station_id, station_details in stations_details.items():
        if station_details["name"] == station_name:
            return station_details["coordinates_in_a_new_israe"]["east"], station_details["coordinates_in_a_new_israe"]["north"]

def loadData(station_names):
    stations_data = {}
    stations_details = openJsonFile()
    for station in station_names:
        stations_csv = load_pkl_file(station)
        station_coordinates = loadCoordinatesNewIsraelData(stations_details, station)
        stations_data[station] = stations_csv, station_coordinates
    return stations_data
"""
# example of use for this file
if __name__ == "__main__":
    # Load the data
    stations_data = loadData(["Afeq","Harashim"])
    if "Afeq" in stations_data:
        print("Data of Afeq:")
        print(stations_data["Afeq"][0].head())

        print("Coordinate of Afeq:")
        print(stations_data["Afeq"][1])

        print("First coordinate of Afeq:")
        print(stations_data["Afeq"][1][0])

        print("Second coordinate of Afeq:")
        print(stations_data["Afeq"][1][1])
    else:
        print("Afeq data not found")

    print("yey")

"""
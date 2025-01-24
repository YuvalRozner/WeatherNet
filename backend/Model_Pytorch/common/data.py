import pickle 
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
"""
this file let you load the data of the stations from the pkl files 
and the coordinates of the stations from the json file
"""

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


def load_pkl_file(station_name):
    current_path = os.path.dirname(__file__)
    file_path = f"{current_path}\\..\\..\\data\\{station_name}.pkl"
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


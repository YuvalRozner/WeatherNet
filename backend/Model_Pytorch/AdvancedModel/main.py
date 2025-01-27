# main.py

import torch
import pandas as pd

from backend.Model_Pytorch.common.window_generator_multiple_stations import WindowGeneratorMultipleStations
from backend.Model_Pytorch.common.data import preprocessing_our_df, load_pkl_file , normalize_data_collective, normalize_data_independent
from backend.Model_Pytorch.AdvancedModel.model import TargetedWeatherPredictionModel, normalize_coordinates
from backend.Model_Pytorch.AdvancedModel.train import train_model
from backend.Model_Pytorch.AdvancedModel.parameters import PARAMS, WINDOW_PARAMS, LSTM_MODEL_PARAMS
import os

import pandas as pd
import numpy as np
from tqdm import tqdm


def find_first_match( df1: pd.DataFrame, df2: pd.DataFrame, tolerance: float = 1e-6,):
    """
    Find the first row indices (i, j) where df1 and df2 match on:
      - 'Year' (exact integer comparison)
      - 'Day sin', 'Day cos', 'Year sin', 'Year cos' (within a tolerance)

    Returns:
    --------
    (i, j) : tuple of int or None
        - i is the row index in df1 (after sorting/resetting index) for the first match
        - j is the row index in df2 for the first match
        If no match is found, returns None.
    """
    required_cols = ['Year', 'Day sin', 'Day cos', 'Year sin', 'Year cos']

    # 1) Check if required columns exist
    for col in required_cols:
        if col not in df1.columns or col not in df2.columns:
            raise ValueError(f"Missing column '{col}' in one of the DataFrames.")

    # 3) Nested loop to find the first match
    for i, row1 in df1.iterrows():
        # We'll destructure for readability
        y1, ds1, dc1, ys1, yc1 = row1['Year'], row1['Day sin'], row1['Day cos'], row1['Year sin'], row1['Year cos']

        for j, row2 in df2.iterrows():
            y2, ds2, dc2, ys2, yc2 = row2['Year'], row2['Day sin'], row2['Day cos'], row2['Year sin'], row2['Year cos']

            # Check integer match for Year
            if y1 == y2:
                # Check approximate match for sine/cosine
                if (np.isclose(ds1, ds2, atol=tolerance) and np.isclose(dc1, dc2, atol=tolerance) and
                        np.isclose(ys1, ys2, atol=tolerance) and np.isclose(yc1, yc2, atol=tolerance)):
                    # Found the first match, return (i, j)
                    return (i, j)

    # If we exit the loops without returning, no match was found
    return None


def find_last_match( df1: pd.DataFrame, df2: pd.DataFrame, tolerance: float = 1e-6 ):
    """
    Find the last row indices (i, j) where df1 and df2 match on:
      - 'Year' (exact integer comparison)
      - 'Day sin', 'Day cos', 'Year sin', 'Year cos' (within a tolerance)

    Returns:
    --------
    (i, j) : tuple of int or None
        - i is the row index in df1 for the last match (from the end)
        - j is the row index in df2 for the last match (from the end)
        If no match is found, returns None.
    """
    required_cols = ['Year', 'Day sin', 'Day cos', 'Year sin', 'Year cos']

    # 1) Check that both DataFrames have the required columns
    for col in required_cols:
        if col not in df1.columns or col not in df2.columns:
            raise ValueError(f"Missing column '{col}' in one of the DataFrames.")

    # 2) Traverse df1 from the end to the start
    for i in range(len(df1) - 1, -1, -1):
        row1 = df1.iloc[i]
        y1, ds1, dc1, ys1, yc1 = (
            row1['Year'],
            row1['Day sin'],
            row1['Day cos'],
            row1['Year sin'],
            row1['Year cos']
        )

        # 3) For each row in df2 (also from the end to the start), compare
        for j in range(len(df2) - 1, -1, -1):
            row2 = df2.iloc[j]
            y2, ds2, dc2, ys2, yc2 = (
                row2['Year'],
                row2['Day sin'],
                row2['Day cos'],
                row2['Year sin'],
                row2['Year cos']
            )

            # Check integer match for 'Year'
            if y1 == y2:
                # Check approximate match for cyclical features
                if (np.isclose(ds1, ds2, atol=tolerance) and np.isclose(dc1, dc2, atol=tolerance) and np.isclose(ys1, ys2, atol=tolerance) and np.isclose(yc1, yc2, atol=tolerance)):
                    # As soon as we find a match in reverse, return (i, j)
                    return (i, j)

    # If no match was found, return None
    return None

def sliceDf(df1, df2):
    """
    example of use of the function - put the example in main
    data1 = {
        'Year': [2020, 2020, 2020, 2020],
        'Day sin': [0.0, 1.0, 0.0, -1.0],
        'Day cos': [1.0, 0.0, -1.0, 0.0],
        'Year sin': [0.0, 0.0, 0.0, 0.0],
        'Year cos': [1.0, 1.0, 1.0, 1.0],
        'Value': [10, 20, 30, 40]
    }

    data2 = {
        'Year': [ 2020, 2020],
        'Day sin': [ 0.0, -1.0],
        'Day cos': [ -1.0, 0.0],
        'Year sin': [ 0.0, 0.0],
        'Year cos': [ 1.0, 1.0],
        'Value': [ 3000, 4000]
    }

    data3 = {
        'Year': [ 2020, 2020],
        'Day sin': [ 1.0, 0.0],
        'Day cos': [0.0, -1.0],
        'Year sin': [ 0.0, 0.0],
        'Year cos': [ 1.0, 1.0],
        'Value': [ 2000, 3000]
    }

    # Create DataFrames
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)

    # List of DataFrames
    dataframes = [df1, df2, df3]


    # use sliceDf() on df1 with all the rest of the dataframes in the list
    for df in dataframes[1:]:
        dataframes[0] = sliceDf(dataframes[0], df)
    
    # Display synchronized DataFrames
    for idx, df_sync in enumerate(dataframes):
        print(f"\nSynchronized DataFrame {idx+1}:")
        print(df_sync)
    """
    result = find_first_match(df1, df2)
    if result is not None:
        i, j = result
    print(f"find_first_match: i:{i},j:{j}")
    df1 = df1.iloc[i:]
    df2 = df2.iloc[j:]
    # reset index
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    result = find_last_match(df1, df2)
    if result is not None:
        i, j = result
    print(f"find_last_match : i:{i},j:{j}")
    df1 = df1.iloc[:i+1]
    df2 = df2.iloc[:j+1]
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    return df1, df2
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


def slice_six(df):
    """
    Apply the same preprocessing steps as during training.
    """
    print("preproccessing data...")
    df = df[5::6].copy()

    return df

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

if __name__ == "__main__":

    east = np.array([217010, 238440])  # new israel coordination system
    north = np.array([734820, 734540])  # new israel coordination system

    east_normalized, north_normalized = normalize_coordinates(east, north)

    # 3. Load DataFrames from CSVs
    filenames = PARAMS['fileNames']  # List of 5 filenames
    dfs = []
    for filename in filenames:
        df = load_pkl_file(filename)  # Replace with actual data loading function
        dfs.append(df)

    print("Original size of data:")
    for df in dfs:
        print(df.shape)

    # 4. Slice DataFrames Based on Year (Ensure Independent Slicing)
    """
    print("Filtering data from year 2005 onwards:")
    
    for i in range(len(dfs)):
        if 'Year' not in dfs[i].columns:
            raise ValueError(f"'Year' column not found in DataFrame {i}.")
        dfs[i] = dfs[i][dfs[i]['Year'] > 2004]
        print(f"Station {i} data shape after filtering: {dfs[i].shape}")

    # 5. Apply `sliceDf` on All Pairs of DataFrames (Ensure Correct Functionality)
    for i in tqdm(range(len(dfs)), desc="Slicing DataFrames"):
        for j in range(i+1, len(dfs)):
            dfs[i], dfs[j] = sliceDf(dfs[i], dfs[j])  # Ensure `sliceDf` correctly processes each pair

    print("Size of data after sliceDf:")
    for i, df in enumerate(dfs):
        print(f"Station {i}: {df.shape}")
"""
    # 6. Slice Every 6th Row
    for i in tqdm(range(len(dfs)), desc="Slicing every 6th row"):
        dfs[i] = slice_six(dfs[i])  # Ensure `slice_six` correctly slices the DataFrame

    print("Size of data after slice_six:")
    for i, df in enumerate(dfs):
        print(f"Station {i}: {df.shape}")

    # 7. Drop NaN Rows
    df_cleaned_list = drop_nan_rows_multiple(dfs)  # Ensure this function returns a list of cleaned DataFrames

    print("Size of data after drop_nan_rows_multiple:")
    for i, df in enumerate(df_cleaned_list):
        print(f"Station {i}: {df.shape}")

    # 8. Extract Feature Values (Use Cleaned Data)
    list_of_values = [df.values for df in df_cleaned_list]  # Each element is (T, num_features)

    # 9. Train/Validation Split per Station
    train_size = int(0.8 * len(list_of_values[0]))
    list_of_train_data = []
    list_of_val_data = []
    for values in list_of_values:
        train_data = values[:train_size]
        val_data = values[train_size:]
        list_of_train_data.append(train_data)
        list_of_val_data.append(val_data)

    # 10. Combine Data into 3D Arrays
    combined_train_data = np.stack(list_of_train_data, axis=1)  # (T_train, num_stations, num_features)
    combined_val_data = np.stack(list_of_val_data, axis=1)      # (T_val, num_stations, num_features)

    # 11. Normalize Data Independently per Station
    scaler_directory = os.path.join(os.path.dirname(__file__), 'output', 'scalers')
    train_data_scaled, val_data_scaled, scalers = normalize_data_independent(
        train_data=combined_train_data,
        val_data=combined_val_data,
        scaler_dir=scaler_directory
    )

    # 12. Create Datasets
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

    # 13. Instantiate Model (Ensure Correct Parameters)
    #in_channels = len(label_columns)  # Number of features to predict

    device = PARAMS['device']
    print(f"Using device: {device}")
    num_stations = len(df_cleaned_list)
    model = TargetedWeatherPredictionModel(
        num_stations=num_stations,
        time_steps=input_width,  # Corrected to input_width
        feature_dim= PARAMS['in_channels'],
        cnn_channels=16,
        kernel_size=3,
        d_model=128,
        nhead=8,
        num_layers=4,
        target_station_idx=target_station_idx
    )

    # 14. Train the Model
    train_model(
        train_dataset,
        val_dataset,
        model,
        epochs=PARAMS['epochs'],
        batch_size=32,
        lr=1e-3,
        checkpoint_dir=os.path.join(os.path.dirname(__file__),'output','checkpoints'),
        resume=PARAMS['resume'],
        device=device,
        coord=[east_normalized, north_normalized]  # Ensure these are on the correct device
    )
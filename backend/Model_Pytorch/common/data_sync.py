import pandas as pd
from typing import Tuple
from tqdm import tqdm
import numpy as np
import os

def find_first_matching_indices(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    atol: float = 1e-8,
    rtol: float = 1e-5
) -> Tuple[int, int]:
    """
    Finds the first pair of starting indices (i, j) in df1 and df2 such that
    df1.iloc[i:] is approximately equal to df2.iloc[j:] within a given tolerance.

    Parameters:
    df1 (pd.DataFrame): First DataFrame.
    df2 (pd.DataFrame): Second DataFrame.
    atol (float): Absolute tolerance parameter for np.allclose.
    rtol (float): Relative tolerance parameter for np.allclose.

    Returns:
    Tuple[int, int]: (i, j) starting indices in df1 and df2 respectively.
                     Returns (-1, -1) if no such pair exists.
    """
    # Ensure both DataFrames have the same columns in the same order
    if list(df1.columns) != list(df2.columns):
        raise ValueError("Both DataFrames must have the same columns in the same order.")

    len1 = len(df1)
    len2 = len(df2)

    # Determine the maximum possible suffix length
    max_possible_suffix = min(len1, len2)

    # Iterate through possible suffix lengths from largest to smallest
    for suffix_len in tqdm(range(max_possible_suffix, 0, -1), desc="Checking suffix lengths"):
        # Extract the last 'suffix_len' rows from both DataFrames
        suffix_df1 = df1.iloc[-suffix_len:].reset_index(drop=True)
        suffix_df2 = df2.iloc[-suffix_len:].reset_index(drop=True)

        # Check if all elements are approximately equal within the specified tolerance
        if np.allclose(suffix_df1.values, suffix_df2.values, atol=atol, rtol=rtol):
            # Calculate starting indices
            i = len1 - suffix_len
            j = len2 - suffix_len
            return (i, j)

    # If no matching suffix found, return (-1, -1)
    return (-1, -1)

def find_last_matching_indices(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    atol: float = 1e-8,
    rtol: float = 1e-5
) -> Tuple[int, int]:
    """
    Finds the last pair of starting indices (i, j) in df1 and df2 such that
    df1.iloc[i:] is approximately equal to df2.iloc[j:] within a given tolerance.

    This corresponds to finding the **shortest** matching suffix.

    Parameters:
    df1 (pd.DataFrame): First DataFrame.
    df2 (pd.DataFrame): Second DataFrame.
    atol (float): Absolute tolerance parameter for np.allclose.
    rtol (float): Relative tolerance parameter for np.allclose.

    Returns:
    Tuple[int, int]: (i, j) starting indices in df1 and df2 respectively.
                     Returns (-1, -1) if no such pair exists.
    """
    # Ensure both DataFrames have the same columns in the same order
    if list(df1.columns) != list(df2.columns):
        raise ValueError("Both DataFrames must have the same columns in the same order.")

    len1 = len(df1)
    len2 = len(df2)

    # Determine the maximum possible suffix length
    max_possible_suffix = min(len1, len2)

    # Iterate through possible suffix lengths from smallest to largest
    for suffix_len in tqdm(range(1, max_possible_suffix + 1), desc="Checking suffix lengths"):
        # Extract the last 'suffix_len' rows from both DataFrames
        suffix_df1 = df1.iloc[-suffix_len:].reset_index(drop=True)
        suffix_df2 = df2.iloc[-suffix_len:].reset_index(drop=True)

        # Check if all elements are approximately equal within the specified tolerance
        if np.allclose(suffix_df1.values, suffix_df2.values, atol=atol, rtol=rtol):
            # Calculate starting indices
            i = len1 - suffix_len
            j = len2 - suffix_len
            return (i, j)

    # If no matching suffix found, return (-1, -1)
    return (-1, -1)


if __name__ == "__main__":
    # Load pickle files
    folder_path = r'C:\Users\dorsha\Documents\GitHub\WeatherNet\backend\Model_Pytorch\input'  # Update this path as needed
    # List to hold DataFrames
    dfs = []



    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_pickle(file_path)
                dfs.append(df)
                print(f"Loaded {filename} successfully.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not dfs:
        print("No DataFrames loaded. Please check the folder path and contents.")
        exit()
    for i in range(len(dfs)):
        dfs[i] = dfs[i][dfs[i]['Year'] >= 2005]
    dfs_for_start_slice = []

    year_to_check = 2005

    selected_columns = ['Day sin', 'Day cos', 'Year sin', 'Year cos']  # Replace with your actual column names
    for idx, df in enumerate(dfs):
        df_tmp = df.copy()
        if 'Year' not in df_tmp.columns:
            print(f"'Year' column not found in DataFrame {idx}. Skipping this DataFrame.")
            continue
        df_tmp = df_tmp[df_tmp['Year'] == year_to_check]
        missing_cols = [col for col in selected_columns if col not in df_tmp.columns]
        if missing_cols:
            print(f"Missing columns {missing_cols} in DataFrame {idx}. Skipping this DataFrame.")
            continue
        df_tmp = df_tmp.loc[:, selected_columns]
        if df_tmp.empty:
            print(f"No data for year {year_to_check} in DataFrame {idx}. Skipping this DataFrame.")
            continue
        dfs_for_start_slice.append(df_tmp)
        print(f"Preprocessed DataFrame {idx} successfully.")

    if not dfs_for_start_slice:
        print("No DataFrames available after preprocessing. Exiting.")
        exit()

    # Define a fixed reference DataFrame (first in the list)
    reference_df = dfs_for_start_slice[0].copy()
    reference_idx = 0  # Index of the reference DataFrame in 'dfs'

    for idx, current_df in enumerate(dfs_for_start_slice[1:], start=1):
        print(f"--- Comparing Reference DF with DF {idx} ---")

        # Find the longest matching suffix
        i_first, j_first = find_first_matching_indices(
            reference_df,
            current_df,
            atol=1e-8,  # Adjust as needed
            rtol=1e-5   # Adjust as needed
        )
        print(f"Longest matching suffix starts at index {i_first} in Reference DF and index {j_first} in DF {idx}.")

        # Slice based on the longest matching suffix
        if i_first != -1 and j_first != -1:
            sliced_reference_first = reference_df.iloc[i_first:].reset_index(drop=True)
            sliced_current_first = current_df.iloc[j_first:].reset_index(drop=True)

        if i_first != -1 and j_first != -1:
            reference_df = sliced_reference_first.copy()
            dfs[0] = dfs[0].iloc[i_first:].reset_index(drop=True)
            dfs[idx] = dfs[idx].iloc[j_first:].reset_index(drop=True)

    # find the shortest df and cut the rest from dfs
    min_len = min([len(df) for df in dfs])
    for idx, df in enumerate(dfs):
        dfs[idx] = df.iloc[:min_len].reset_index(drop=True)
    
    # Save the cleaned DataFrames
    cleaned_folder_path = r'C:\Users\dorsha\Documents\GitHub\WeatherNet\backend\Model_Pytorch\input\cleaned'  # Update this path as needed
    os.makedirs(cleaned_folder_path, exist_ok=True)
    for idx, df in enumerate(dfs):
        save_path = os.path.join(cleaned_folder_path, f'df_{idx}.pkl')
        df.to_pickle(save_path)
        print(f"Saved cleaned DataFrame {idx} to {save_path}.")
    
    print("Data cleaning completed successfully.")

    # print head of each df, shape and last cuople rows
    for idx, df in enumerate(dfs):
        print(f"DataFrame {idx}")
        print(df.head())
        print(df.shape)
        print(df.tail(2))
        print()
    
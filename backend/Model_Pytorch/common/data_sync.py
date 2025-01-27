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


def main():   # Load pickle files
    folder_path = r'C:\Users\dorsh\Documents\GitHub\WeatherNet\backend\Model_Pytorch\input'  # Update this path as needed
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
        dfs[i] = dfs[i][dfs[i]['Year'] >= 2006].reset_index(drop=True)
    dfs_for_start_slice = []

    year_to_check = 2006

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
            reference_df = reference_df.iloc[i_first:].reset_index(drop=True)
            #current_df = current_df.iloc[j_first:].reset_index(drop=True)
            dfs[0] = dfs[0].iloc[i_first:].reset_index(drop=True)
            dfs[idx] = dfs[idx].iloc[j_first:].reset_index(drop=True)

    columns_to_print = ['Day sin', 'Day cos', 'Year sin', 'Year cos']  # Replace with your actual column names

    # find the shortest df and cut the rest from dfs
    min_len = min([len(df) for df in dfs])
    for i in tqdm(range(min_len)):
        if dfs[0][columns_to_print].iloc[i] != dfs[1][columns_to_print].iloc[i] or dfs[0][columns_to_print].iloc[i] != dfs[2][columns_to_print].iloc[i]:

            print(f"in hereeeeeee {i}")
        #print(f"{dfs[0][columns_to_print].iloc[i]}")
        #print(f"{dfs[1][columns_to_print].iloc[i]}")
        #print(f"{dfs[2][columns_to_print].iloc[i]}")


    for idx, df in enumerate(dfs):
        dfs[idx] = df.iloc[:min_len]
    
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
        print(df[columns_to_print].head())
        print(df[columns_to_print].shape)
        print(df[columns_to_print].tail(2))
        print()
    




import pandas as pd

def run(df1, df2,atol):
    """
    Identifies the indexes of rows to delete from df1 and df2 to make them equal.

    Parameters:
    - df1: First Pandas DataFrame.
    - df2: Second Pandas DataFrame.

    Returns:
    - delete_from_df1: List of indexes to delete from df1.
    - delete_from_df2: List of indexes to delete from df2.
    """
    i = 0  # Pointer for df1
    j = 0  # Pointer for df2
    delete_from_df1 = []
    delete_from_df2 = []

    # Convert DataFrame indexes to lists for faster access
    df1_indices = df1.index.tolist()
    df2_indices = df2.index.tolist()
    max1 = max(len(df1),len(df2))
    while i < len(df1) and j < len(df2):
        progress = (((i + j) / 2) / max1) * 100
        print(f"\rProgress: {progress:.2f}% , len(delete_from_df1): {len(delete_from_df1)} ", end="")

        row_df1 = df1.iloc[i]
        row_df2 = df2.iloc[j]

        if np.allclose(row_df1.values, row_df2.values, atol=atol):
            # Rows match within the tolerance; move both pointers forward
            i += 1
            j += 1
        else:
            # Attempt to find the next matching row in df2 for df1.iloc[i]
            # Create a boolean Series where each row in df2[j:] is compared to df1.iloc[i]
            def rows_match(row):
                return np.allclose(row.values, row_df1.values, atol=atol)

            # Apply the comparison function to rows in df2 starting from index j
            matches = df2.iloc[j:].apply(rows_match, axis=1)
            
            if matches.any():
                # Match found in df2
                next_match_relative_idx = matches.idxmax()  # Get the index label
                next_match_idx = df2_indices.index(next_match_relative_idx)  # Convert to positional index

                # Rows in df2 from j to next_match_idx -1 are to be deleted
                rows_to_delete = df2_indices[j:next_match_idx]
                delete_from_df2.extend(rows_to_delete)

                # Move j to the position after the matched row
                j = next_match_idx + 1

                # Move i forward as we've found a match for df1.iloc[i]
                i += 1
            else:
                # No match found in df2; mark df1.iloc[i] for deletion
                delete_from_df1.append(df1_indices[i])
                i += 1

    # After the loop, handle any remaining rows in df1
    if i < len(df1):
        remaining_df1 = df1_indices[i:]
        delete_from_df1.extend(remaining_df1)

    # Handle any remaining rows in df2
    if j < len(df2):
        remaining_df2 = df2_indices[j:]
        delete_from_df2.extend(remaining_df2)

    return delete_from_df1, delete_from_df2


if __name__ == "__main__":
    data1 = {
        'Year': [2020, 2020, 2020 ],
        'Day sin': [0.0, 1.0, 0.0],
        'Day cos': [1.0, 0.0, -1.0 ],
        'Year sin': [0.0, 0.0, 0.0],
        'Year cos': [1.0, 1.0, 1.0 ],
        'Value': [10, 20, 30 ]
    }

    data2 = {
        'Year': [2020, 2020],
        'Day sin': [0.0, -1.0],
        'Day cos': [-1.0, 0.0],
        'Year sin': [0.0, 0.0],
        'Year cos': [1.0, 1.0],
        'Value': [3000, 4000]
    }
    #df1 = pd.DataFrame(data1)
    #df2 = pd.DataFrame(data2)

    df1 = pd.read_pickle(r"C:\Users\dorsh\Documents\GitHub\WeatherNet\backend\Model_Pytorch\input\Newe Yaar.pkl")
    df2 = pd.read_pickle(r"C:\Users\dorsh\Documents\GitHub\WeatherNet\backend\Model_Pytorch\input\Tavor Kadoorie.pkl")
    selected_columns = ['Day sin', 'Day cos', 'Year sin', 'Year cos', 'Year']
    df1_col = df1.loc[:, selected_columns]
    df2_col = df2.loc[:, selected_columns]
    delete_from_df1_col, delete_from_df2_col = run(df1_col[df1_col['Year']==2005], df2_col[df2_col['Year']==2005])
    print(f"Rows to delete from df1: {delete_from_df1_col}")
    print(f"Rows to delete from df2: {delete_from_df2_col}")
    print(f"Total rows to delete from df1: {len(delete_from_df1_col)}")
    print(f"Total rows to delete from df2: {len(delete_from_df2_col)}")
    print()
    # delete the rows from df1
    df1_cleaned = df1.drop(delete_from_df1_col)
    df2_cleaned = df2.drop(delete_from_df2_col)
    print(f"Cleaned DataFrame 1 shape: {df1_cleaned.shape}")
    print(f"Cleaned DataFrame 2 shape: {df2_cleaned.shape}")
    print()
    print(f"Cleaned DataFrame 1 head:")
    print(df1_cleaned.head())
    print()
    print(f"Cleaned DataFrame 2 head:")
    print(df2_cleaned.head())
    print()
    print(f"Cleaned DataFrame 1 tail:")
    print(df1_cleaned.tail())
    print()
    print(f"Cleaned DataFrame 2 tail:")
    print(df2_cleaned.tail())
    print()

    # df1_cleaned.to_pickle(r"path_to_save_cleaned_df1.pkl")
    # df2_cleaned.to_pickle(r"path_to_save_cleaned_df2.pkl")
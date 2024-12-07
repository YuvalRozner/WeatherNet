"""
utils.py
This module provides utility functions for handling data files in PKL and CSV formats. 
It includes functions to load, save, and convert data files, as well as to list files in a directory.
Functions:
    load_all_pkl_files():
        Loads all PKL files from the data directory and returns a dictionary of DataFrames.

    load_pkl_file(file_name):
        Loads a single PKL file from the data directory and returns a DataFrame.

    load_all_csv_files():
        Loads all CSV files from the data directory and returns a dictionary of DataFrames.

    load_csv_file(file_name):
        Loads a single CSV file from the data directory and returns a DataFrame.

    save_to_pkl(df, file_name):
        Saves a DataFrame to a PKL file in the data directory.

    save_to_csv(df, file_name):
        Saves a DataFrame to a CSV file in the data directory.

    get_files_in_folder(folder_path=DATA_DIRECTORY):
        Returns a list of full paths of all files in the specified folder.

    get_files_names_in_folder(folder_path=DATA_DIRECTORY):
        Returns a list of names of all files in the specified folder.
        
    pkl_to_csv(file_name):
        Converts a PKL file to a CSV file and saves it in the 'csv' subdirectory of the data directory.
"""
import pandas as pd

import os
from tqdm import tqdm


DATA_DIRECTORY = "../data/"
na_values = ['None', 'null', '-', '', ' ', 'NaN', 'nan', 'NAN']

# Load all pkl files
def load_all_pkl_files():
    pkl_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.pkl')]
    dataframes = {}
    for file in tqdm(pkl_files, desc="Loading PKL files"):
        file_path = os.path.join(DATA_DIRECTORY, file)
        df_name = os.path.splitext(file)[0]
        dataframes[df_name] = pd.read_pickle(file_path)
    return dataframes

# load single pkl file  
def load_pkl_file(file_name):
    file_path = os.path.join(DATA_DIRECTORY, file_name)
    return pd.read_pickle(file_path)

# Load all csv files
def load_all_csv_files():
    csv_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.csv')]
    dataframes = {}
    for file in tqdm(csv_files, desc="Loading CSV files"):
        file_path = os.path.join(DATA_DIRECTORY, file)
        df_name = os.path.splitext(file)[0]
        dataframes[df_name] = pd.read_csv(file_path, na_values=na_values)
    return

# load single csv file
def load_csv_file(file_name):
    file_path = os.path.join(DATA_DIRECTORY, file_name)
    return pd.read_csv(file_path, na_values=na_values)

# Save dataframe to pkl file
def save_to_pkl(df, file_name):
    file_path = os.path.join(DATA_DIRECTORY, file_name)
    df.to_pickle(file_path)
    print(f"Dataframe saved to {file_path}")

# Save dataframe to csv file
def save_to_csv(df, file_name):
    file_path = os.path.join(DATA_DIRECTORY, file_name)
    df.to_csv(file_path, index=False)
    print(f"Dataframe saved to {file_path}")

# return a list of all files full paths in a folder 
def get_files_in_folder(folder_path=DATA_DIRECTORY):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

# return a list of all files names in a folder 
def get_files_names_in_folder(folder_path=DATA_DIRECTORY):
    return [f for f in os.listdir(folder_path)]

# from pkl to csv
def pkl_to_csv(file_name):
    pkl_file_path = os.path.join(DATA_DIRECTORY, file_name + '.pkl')
    csv_folder_path = os.path.join(DATA_DIRECTORY, 'csv')
    csv_file_path = os.path.join(csv_folder_path, file_name + '.csv')
    
    # Create csv directory if it doesn't exist
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)
    
    # Load pkl file
    df = pd.read_pickle(pkl_file_path)
    
    # Save as csv
    df.to_csv(csv_file_path, index=False)
    print(f"Converted {pkl_file_path} to {csv_file_path}")



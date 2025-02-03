import pandas as pd# type: ignore
import requests# type: ignore
import json
from datetime import datetime
import os
import numpy as np# type: ignore
import matplotlib.pyplot as plt # type: ignore
from tqdm import tqdm # type: ignore
from Model_Pytorch.common.constantsParams import *

##    function used for the first time to get the data from the IMS ##
######################################################################################################################
def fetch_weather_data(station_id, start_date, end_date):
    url = f"https://ims.gov.il/he/envista_station_all_data_time_range/{station_id}/BP%26DiffR%26Grad%26NIP%26RH%26TD%26TDmax%26TDmin%26TW%26WD%26WDmax%26WS%26WS1mm%26Ws10mm%26Ws10maxEnd%26WSmax%26STDwd%26Rain/{start_date}/{end_date}/1/S"
    response = requests.get(url)
    data = json.loads(response.content)
    return data

def fetch_data_for_station(station_id, start_year, end_year):
    all_data = []
    for year in tqdm(range(start_year, end_year + 1), desc="Fetching data by year"):
        today_fore0 = f"{year}" + BEGINING_OF_YEAR
        today_fore23 = f"{year}" + ENDING_OF_YEAR
        data = fetch_weather_data(station_id, today_fore0, today_fore23)
        process_data(data)
        # Convert the data to a DataFrame and append to the list
        df = pd.DataFrame(data['data']['records'])
        all_data.append(df)
    # Concatenate all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def fetch_data_for_station_manual_time_range(station_id, start_year, end_year, start_date, end_date):
    all_data = []
    for year in tqdm(range(start_year, end_year + 1), desc="Fetching data by year"):
        today_fore0 = f"{year}" + start_date
        today_fore23 = f"{year}" + end_date
        data = fetch_weather_data(station_id, today_fore0, today_fore23)
        process_data(data)
        # Convert the data to a DataFrame and append to the list
        df = pd.DataFrame(data['data']['records'])
        all_data.append(df)
    # Concatenate all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def get_station_data(station_id, start_year, end_year):
    # Get all data for the station
    combined_df = fetch_data_for_station(station_id, start_year, end_year)
    # Convert the DataFrame back to the dictionary format expected by process_data
    data = {'data': {'records': combined_df.to_dict(orient='records')}}
    # Process the data
    process_data(data)
    return data

def get_station_data_manual_time_range(station_id, start_year, end_year, start_date, end_date):
    # Get all data for the station
    combined_df = fetch_data_for_station_manual_time_range(station_id, start_year, end_year, start_date, end_date)
    # Convert the DataFrame back to the dictionary format expected by process_data
    data = {'data': {'records': combined_df.to_dict(orient='records')}}
    # Process the data
    process_data(data)
    return data

def remove_unwanted_keys(data):
    # Remove 'sid', 'sname', and 'date_for_sort' from each record in data
    for record in data['data']['records']:
        # if 'date_for_sort' in record:
        #     del record['date_for_sort']
        if 'sid' in record:
            del record['sid']
        if 'TW' in record:
            del record['TW']
        if 'sname' in record:
            del record['sname']

def replace_column_names(data):
    # Replace the names of the columns by the pairs in COLUMN_PAIRS
    for record in data['data']['records']:
        for new_name, old_name in COLUMN_PAIRS:
            if new_name in record:
                record[old_name] = record.pop(new_name)

def process_data(data):
    remove_unwanted_keys(data)
    replace_column_names(data)

def get_data_of_stations_from_ims_by_constants_params(StationsList, startYear, endYear):
    dataframes = {}
    for station_name, station_id in StationsList.items():
      print(f"\n Downloading data for {station_name}")
      try:
          data = get_station_data(station_id, START_YEAR, END_YEAR)
          df = pd.DataFrame(data['data']['records'])
          dataframes[station_name] = df
      except IndexError as e:
          print(f"Error processing data for {station_name}: {e}")
    return dataframes

def get_data_of_stations_from_ims_manual_time_range(StationsList, startYear, endYear, startDate, endDate):
    dataframes = {}
    for station_name, station_id in StationsList.items():
      print(f"\n Downloading data for {station_name}")
      try:
          data = get_station_data_manual_time_range(station_id, startYear, endYear, startDate, endDate)
          df = pd.DataFrame(data['data']['records'])
          dataframes[station_name] = df
      except IndexError as e:
          print(f"Error processing data for {station_name}: {e}")
    return dataframes
######################################################################################################################


##    functions used for loading and saving the data to pickles   ##
######################################################################################################################
def save_dataframes_to_pickles(dataframes, DATA_DIRECTORY):
  for df_name, df in dataframes.items():
      file_path = os.path.join(DATA_DIRECTORY, f"{df_name}.pkl")
      df.to_pickle(file_path)
      print(f"Saved {df_name} to {file_path}")

def load_dataframes_from_pickles(DATA_DIRECTORY):
    data_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.pkl')]

    dataframes = {}
    for file in tqdm(data_files, desc="Loading Pickle files of year data"):
        file_path = os.path.join(DATA_DIRECTORY, file)
        df_name = os.path.splitext(file)[0]
        dataframes[df_name] = pd.read_pickle(file_path)
    
    return dataframes
######################################################################################################################


##    function used for displaying ##
######################################################################################################################
def display_dataframes_heads(dataframes):
    for df_name, df in dataframes.items():
        print(f"Heads of {df_name}:")
        print(df.head())
        print("\n")

def display_wind_before_vectorize(dataframes):
    plt.figure(figsize=(14, 6))
    # Assuming 'dataframes' is a dictionary of DataFrames and we take the first one
    first_df_name = list(dataframes.keys())[0]
    first_df = dataframes[first_df_name]

    first_df['WD (deg)'] = first_df['WD (deg)'].replace(NA_VALUES, np.nan).infer_objects(copy=False)
    first_df['WS (m/s)'] = first_df['WS (m/s)'].replace(NA_VALUES, np.nan).infer_objects(copy=False)
    first_df['WDmax (deg)'] = first_df['WDmax (deg)'].replace(NA_VALUES, np.nan).infer_objects(copy=False)
    first_df['WSmax (m/s)'] = first_df['WSmax (m/s)'].replace(NA_VALUES, np.nan).infer_objects(copy=False)

    # Convert columns to numeric, forcing errors to NaN
    first_df['WD (deg)'] = pd.to_numeric(first_df['WD (deg)'], errors='coerce')
    first_df['WS (m/s)'] = pd.to_numeric(first_df['WS (m/s)'], errors='coerce')
    first_df['WDmax (deg)'] = pd.to_numeric(first_df['WDmax (deg)'], errors='coerce')
    first_df['WSmax (m/s)'] = pd.to_numeric(first_df['WSmax (m/s)'], errors='coerce')

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Mask to filter out NaN values for wind
    mask_wind = first_df['WD (deg)'].notna() & first_df['WS (m/s)'].notna()

    # Create the 2D histogram plot for wind
    hist_wind = ax[0].hist2d(
        first_df.loc[mask_wind, 'WD (deg)'],
        first_df.loc[mask_wind, 'WS (m/s)'],
        bins=(50, 50),
        vmax=400
    )
    fig.colorbar(hist_wind[3], ax=ax[0])
    ax[0].set_xlabel('Wind Direction [deg]')
    ax[0].set_ylabel('Wind Velocity [m/s]')
    ax[0].set_title(f'2D Histogram of Wind for {first_df_name}')

    # Mask to filter out NaN values for gust
    mask_gust = first_df['WDmax (deg)'].notna() & first_df['WSmax (m/s)'].notna()

    # Create the 2D histogram plot for gust
    hist_gust = ax[1].hist2d(
        first_df.loc[mask_gust, 'WDmax (deg)'],
        first_df.loc[mask_gust, 'WSmax (m/s)'],
        bins=(50, 50),
        vmax=400
    )
    fig.colorbar(hist_gust[3], ax=ax[1])
    ax[1].set_xlabel('Gust Direction [deg]')
    ax[1].set_ylabel('Gust Velocity [m/s]')
    ax[1].set_title(f'2D Histogram of Gust for {first_df_name}')

    plt.tight_layout()
    plt.show()

def display_wind_after_vectorize(dataframes):
  # Plot 2D histogram plots of the wind ('Wind_x', 'Wind_y') and gust ('Gust_x', 'Gust_y') for the first dataframe
  first_df_name = list(dataframes.keys())[0]
  first_df = dataframes[first_df_name]

  fig, ax = plt.subplots(1, 2, figsize=(14, 6))

  # Mask to filter out NaN values for wind
  mask_wind = first_df['Wind_x'].notna() & first_df['Wind_y'].notna()

  # Create the 2D histogram plot for wind
  hist_wind = ax[0].hist2d(first_df.loc[mask_wind, 'Wind_x'], first_df.loc[mask_wind, 'Wind_y'], bins=(50, 50), vmax=400)
  fig.colorbar(hist_wind[3], ax=ax[0])
  ax[0].set_xlabel('Wind X Component')
  ax[0].set_ylabel('Wind Y Component')
  ax[0].set_title(f'2D Histogram of Wind Components for {first_df_name}')

  # Mask to filter out NaN values for gust
  mask_gust = first_df['Gust_x'].notna() & first_df['Gust_y'].notna()

  # Create the 2D histogram plot for gust
  hist_gust = ax[1].hist2d(first_df.loc[mask_gust, 'Gust_x'], first_df.loc[mask_gust, 'Gust_y'], bins=(50, 50), vmax=400)
  fig.colorbar(hist_gust[3], ax=ax[1])
  ax[1].set_xlabel('Gust X Component')
  ax[1].set_ylabel('Gust Y Component')
  ax[1].set_title(f'2D Histogram of Gust Components for {first_df_name}')

  plt.tight_layout()
  plt.show()

def print_length_of_dataframes(dataframes):
   print("\n  Length of dataframes:")
   for df_name, df in dataframes.items():
    print("Length of dataframe {}: {}".format(df_name, len(df)))
######################################################################################################################


##    function used for syncing the dataframes ##
######################################################################################################################
def sort_dataframes(dataframes):
  # Sort each dataframe by the column 'date_for_sort'
  for df_name, df in dataframes.items():
    dataframes[df_name] = df.sort_values(by='date_for_sort')

def slice_dataframes_beginning(dataframes, begin_date):
   for df_name, df in dataframes.items():
    index_to_keep = df[df['Date Time'] == begin_date].index
    if not index_to_keep.empty:
      index_to_keep = index_to_keep[0]
      print(f"Index to keep from dataframe {df_name}: {index_to_keep}")
      dataframes[df_name] = df.loc[index_to_keep:].reset_index(drop=True)
    else:
      print(f"{begin_date} not found in dataframe {df_name}")

def delete_rows_not_existing_in_all_dataframes(dataframes):
  # Find common 'Date Time' keys present in all dataframes
  common_keys = set.intersection(*(set(df['Date Time']) for df in dataframes.values()))
  # Initialize a dictionary to store the number of deleted rows
  deleted_rows = {}
  # Initialize a variable to store the latest deleted date across all dataframes
  latest_deleted_date = None
  # Remove rows not in common_keys and count deletions
  for df_name, df in dataframes.items():
    initial_count = len(df)
    df_filtered = df[df['Date Time'].isin(common_keys)].reset_index(drop=True)
    deleted = initial_count - len(df_filtered)
    dataframes[df_name] = df_filtered
    deleted_rows[df_name] = deleted
    if deleted > 0:
      max_deleted_date = df[~df['Date Time'].isin(common_keys)]['Date Time'].max()
      if latest_deleted_date is None or max_deleted_date > latest_deleted_date:
        latest_deleted_date = max_deleted_date

  # Return the number of rows deleted from each dataframe and the latest deleted date
  return deleted_rows, latest_deleted_date
######################################################################################################################


##    function used for preprocessing the dataframes ##
######################################################################################################################
def remove_unecessery_columns(dataframes, columns_to_remove):
  for df_name, df in dataframes.items():
    dataframes[df_name] = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

def format_the_time_column(dataframes):
  for df_name, df in dataframes.items():
    if 'Date Time' in df.columns:
      df['Date Time'] = pd.to_datetime(df.pop('Date Time'), format="%d/%m/%Y %H:%M")
      df['Year'] = df['Date Time'].dt.year

def take_round_hours(dataframes):
  for df_name, df in dataframes.items():
    if 'Date Time' in df.columns:
      df = df[df['Date Time'].dt.minute == 0]
      dataframes[df_name] = df

def fill_1_missing_values(dataframes, values_to_fill, should_print=False):
  for df_name, df in dataframes.items():
    for value in values_to_fill:
      if value in df.columns:
        df[value] = df[value].replace(NA_VALUES, np.nan)
        
        # Fill NaN values wrapped with two non-NaN values:
        nan_wrapped_count = 0
        value_values = df[value].values
        for i in range(1, len(value_values) - 1):
          if pd.isna(value_values[i]) and not pd.isna(value_values[i - 1]) and not pd.isna(value_values[i + 1]):
            try:
              value_values[i] = (float(value_values[i - 1]) + float(value_values[i + 1])) / 2
              nan_wrapped_count += 1
            except ValueError as e:
              print(f"ValueError encountered in {df_name} at index {i} for column {value}: {e}")
        if should_print:
          print(f"Number of NaN values wrapped with two non-NaN values and filled in {df_name} station for column {value}: {nan_wrapped_count} which is {nan_wrapped_count / len(df) * 100}% of the data")

def fill_2_missing_values(dataframes, values_to_fill, should_print=False):
  for value in values_to_fill:
    for df_name, df in dataframes.items():
      if value in df.columns:
        df[value] = df[value].replace(NA_VALUES, np.nan)
        
        # Fill two consecutive NaN values wrapped with two non-NaN values:
        nan_wrapped_count = 0
        value_values = df[value].values
        i = 2  # Start from index 2 to ensure i-2 is valid
        while i < len(value_values) - 2:
          if (pd.isna(value_values[i]) and pd.isna(value_values[i+1]) and
            not pd.isna(value_values[i - 2]) and not pd.isna(value_values[i - 1]) and
            not pd.isna(value_values[i + 2]) and not pd.isna(value_values[i + 3])):
            
            val1 = float(value_values[i - 2])
            val2 = float(value_values[i - 1])
            val3 = float(value_values[i + 2])
            val4 = float(value_values[i + 3])
            
            # Determine trends
            trend_before = val2 < val1
            trend_after = val4 < val3
            if trend_before == trend_after:
              try:
                diff = val3 - val2
                value_values[i] = val2 + diff / 3
                value_values[i + 1] = val2 + diff * 2 / 3
                nan_wrapped_count += 2
                i += 2  # Skip the next index as it's already processed
                continue
              except ValueError as e:
                print(f"ValueError encountered in {df_name} at indices {i} and {i+1} for column {value}: {e}")
          i += 1
        if should_print:
          print(f"Number of NaN values wrapped with two non-NaN values and filled in {df_name} station for column {value}: {nan_wrapped_count} which is {nan_wrapped_count / len(df) * 100:.2f}% of the data")
    
def fill_3_missing_values(dataframes, values_to_fill, should_print=False):
  for value in values_to_fill:
    for df_name, df in dataframes.items():
      if value in df.columns:
        df[value] = df[value].replace(NA_VALUES, np.nan)
        
        # Fill three consecutive NaN values wrapped with two non-NaN values:
        nan_wrapped_count = 0
        value_values = df[value].values
        i = 2  # Start from index 2 to ensure i-2 is valid
        while i < len(value_values) - 4:
          if (pd.isna(value_values[i]) and pd.isna(value_values[i+1]) and pd.isna(value_values[i+2]) and
            not pd.isna(value_values[i - 2]) and not pd.isna(value_values[i - 1]) and
            not pd.isna(value_values[i + 3]) and not pd.isna(value_values[i + 4])):
            
            val1 = float(value_values[i - 2])
            val2 = float(value_values[i - 1])
            val3 = float(value_values[i + 3])
            val4 = float(value_values[i + 4])
            
            # Determine trends
            trend_before = val2 < val1
            trend_after = val4 < val3
            if trend_before == trend_after:
              try:
                diff = val3 - val2
                value_values[i] = val2 + diff / 4
                value_values[i + 1] = val2 + (diff * 2) / 4
                value_values[i + 2] = val2 + (diff * 3) / 4
                nan_wrapped_count += 3
                i += 3  # Skip the next indices as they're already processed
                continue
              except ValueError as e:
                print(f"ValueError encountered in {df_name} at indices {i}, {i+1}, and {i+2} for column {value}: {e}")
          i += 1
        if should_print:
          print(f"Number of NaN values wrapped with three non-NaN values and filled in {df_name} station for column {value}: {nan_wrapped_count} which is {nan_wrapped_count / len(df) * 100:.2f}% of the data")

def replace_time_with_cyclic_representation(dataframes):
  day = 24*60*60
  year = (365.2425)*day

  for df_name, df in dataframes.items():
    if 'Date Time' in df.columns:
      timestamp_s = df['Date Time'].map(pd.Timestamp.timestamp)
      df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
      df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
      df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
      df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
      dataframes[df_name] = df.drop(columns=['Date Time'])

def vectorize_wind(dataframes):
  print("vectorizing wind.")
  for df_name, df in dataframes.items():
    try:
      wind_speed = pd.to_numeric(df.pop('WS (m/s)'), errors='coerce')
      wind_direction_rad = pd.to_numeric(df.pop('WD (deg)'), errors='coerce') * np.pi / 180
      if wind_speed is not None and wind_direction_rad is not None:
        mask_wind = wind_speed.notna() & wind_direction_rad.notna()
        df['Wind_x'] = wind_speed * np.cos(wind_direction_rad)
        df['Wind_y'] = wind_speed * np.sin(wind_direction_rad)
        df.loc[~mask_wind, ['Wind_x', 'Wind_y']] = np.nan

      gust_speed = pd.to_numeric(df.pop('WSmax (m/s)'), errors='coerce')
      gust_direction_rad = pd.to_numeric(df.pop('WDmax (deg)'), errors='coerce') * np.pi / 180
      if gust_speed is not None and gust_direction_rad is not None:
        mask_gust = gust_speed.notna() & gust_direction_rad.notna()
        df['Gust_x'] = gust_speed * np.cos(gust_direction_rad)
        df['Gust_y'] = gust_speed * np.sin(gust_direction_rad)
        df.loc[~mask_gust, ['Gust_x', 'Gust_y']] = np.nan
    except KeyError as e:
      print(f"KeyError encountered in {df_name}: {e}")
    except TypeError as e:
      print(f"TypeError encountered in {df_name}: {e}")

def drop_nan_rows_multiple(dataframes, reset_indices=True):
    """
    Removes rows from all DataFrames in the list where any DataFrame has NaN in any column.
    
    Parameters:
    df_list (List[pd.DataFrame]): List of DataFrames to process.
    reset_indices (bool): Whether to reset the index after dropping rows. Defaults to True.
    
    Returns:
    Tuple[List[pd.DataFrame], Optional[pd.Timestamp]]: List of cleaned DataFrames and the latest 'Date Time' of removed rows.
    """
    if not dataframes:
        raise ValueError("The list of DataFrames is empty.")
    
    # Step 1: Identify rows with any NaN in each DataFrame
    nan_indices_list = [df.isnull().any(axis=1) for df in dataframes.values()]
    
    # Step 2: Combine the indices where NaNs are present in any DataFrame
    combined_nan = pd.Series([False] * len(dataframes[list(dataframes.keys())[0]]), index=dataframes[list(dataframes.keys())[0]].index)
    for nan_mask in nan_indices_list:
        combined_nan = combined_nan | nan_mask
    
    # Get the indices to drop
    indices_to_drop = combined_nan[combined_nan].index
    print(" number of rows with NaNs to drop: ", len(indices_to_drop))
    
    # Step 3: Find the latest 'Date Time' of the rows to be removed
    latest_removed_date = None
    for df in dataframes.values():
        if 'Date Time' in df.columns:
            removed_dates = df.loc[indices_to_drop, 'Date Time']
            if not removed_dates.empty:
                max_date = pd.to_datetime(removed_dates, errors='coerce').max()
                if latest_removed_date is None or (max_date is not pd.NaT and max_date > latest_removed_date):
                    latest_removed_date = max_date
    
    # Step 4: Drop the identified indices from all DataFrames
    cleaned_df_list = []
    for df in dataframes.values():
        cleaned_df = df.drop(indices_to_drop)
        if reset_indices:
            cleaned_df = cleaned_df.reset_index(drop=True)
        cleaned_df_list.append(cleaned_df)
    
    return cleaned_df_list, latest_removed_date
######################################################################################################################

def get_prccessed_latest_data_by_hour_and_station(stations_list, hours_back, begin_forecast_time=datetime.now()):
  success = True
  end_datetime = begin_forecast_time.replace(minute=0, second=0, microsecond=0)
  start_datetime = end_datetime - pd.Timedelta(days=7) # 7 days back
  startYear, endYear = start_datetime.year, end_datetime.year
  startDate, endDate = start_datetime.strftime("%m%d%H%M"), end_datetime.strftime("%m%d%H%M")
  print(f"startDate: {startDate}, endDate: {endDate}")

  dataframes = get_data_of_stations_from_ims_manual_time_range(stations_list, startYear, endYear, startDate, endDate)
  if len(dataframes) == 0:
    success = False
  
  sort_dataframes(dataframes)
  latest_deleted_date = delete_rows_not_existing_in_all_dataframes(dataframes)
  if latest_deleted_date and isinstance(latest_deleted_date, datetime) and latest_deleted_date > end_datetime:
    success = False
  remove_unecessery_columns(dataframes, COLUMNS_TO_REMOVE)
  format_the_time_column(dataframes)
  fill_1_missing_values(dataframes, VALUES_TO_FILL)
  fill_2_missing_values(dataframes, VALUES_TO_FILL)
  fill_3_missing_values(dataframes, VALUES_TO_FILL)
  take_round_hours(dataframes)
  fill_1_missing_values(dataframes, VALUES_TO_FILL)
  fill_2_missing_values(dataframes, VALUES_TO_FILL)
  last_datetime = pd.to_datetime(dataframes[list(dataframes.keys())[0]]['Date Time'].iloc[-1], format="%d/%m/%Y %H:%M")
  last_hour = last_datetime.strftime("%H:%M")
  last_date = last_datetime.strftime("%Y-%m-%d")
  print(f"last hour: {last_hour}, last hour date: {last_date}")

  replace_time_with_cyclic_representation(dataframes)
  vectorize_wind(dataframes)
  latest_deleted_date = drop_nan_rows_multiple(dataframes)
  if latest_deleted_date and isinstance(latest_deleted_date, datetime) and latest_deleted_date > end_datetime:
    success = False
    
  for df_name, df in dataframes.items(): # return only the last hours_back hours
      if len(df) > hours_back:
          dataframes[df_name] = df.iloc[-hours_back:].reset_index(drop=True)
  return dataframes, last_hour, last_date, success

## how to use this function:
    # stations_list = STATIONS_LIST
    # hours_back = 72
    # dataframes, last_hour, last_date, success = get_prccessed_latest_data_by_hour_and_station(stations_list, hours_back)
    # print(f"len of df: {len(dataframes)}")
    # print(f"len of df[0]: {len(dataframes[list(dataframes.keys())[0]])}")
    # print(f"len of df[1]: {len(dataframes[list(dataframes.keys())[1]])}")
    # print(f"len of df[2]: {len(dataframes[list(dataframes.keys())[2]])}")
    # print(f"Last hour: {last_hour}")
    # print(f"Last hour date: {last_date}")
    # print(f"success: {success}")


def main():
  #menue: 
  ## load from:
  get_data_from_ims = True # (including the first process)
  load_data_from_directory = not get_data_from_ims and True
  ## save to:
  save_data_to_pickles_in_the_end = True
  ## sync:
  sync = True #master switch
  should_sort_dataframes = sync and True
  should_slice_dataframes_beginning = sync and True
  should_delete_rows_not_existing_in_all_dataframes = sync and True
  ## preprocessing:
  preprocess = True
  should_remove_unecessery_columns = preprocess and True
  should_format_the_time_column = preprocess and True
  data_imputation = preprocess and True
  should_fill_data_1_missing_value = data_imputation and True
  should_fill_data_2_missing_values = data_imputation and True
  should_fill_data_3_missing_values = data_imputation and True
  should_take_round_hours = preprocess and True
  should_replace_time_with_cyclic_representation = preprocess and True #leave it false
  should_vectorize_wind = preprocess and True
  should_drop_nan_rows = preprocess and True
    ## display:
  should_display_heads_of_dataframes = False
  should_print_length_of_dataframes = True
  should_display_wind_before_vectorize = False
  should_display_wind_after_vectorize = should_vectorize_wind and False


  if get_data_from_ims:
    dataframes = get_data_of_stations_from_ims_by_constants_params(STATIONS_LIST, START_YEAR, END_YEAR)

  if load_data_from_directory:
    dataframes = load_dataframes_from_pickles(DATA_DIRECTORY)

  if should_sort_dataframes:
    sort_dataframes(dataframes)

  if should_slice_dataframes_beginning:
    slice_dataframes_beginning(dataframes, '01/04/2005 00:00')
  
  if should_print_length_of_dataframes:
    print_length_of_dataframes(dataframes)

  if should_delete_rows_not_existing_in_all_dataframes:
    delete_rows_not_existing_in_all_dataframes(dataframes)

  if sync and should_print_length_of_dataframes:
    print_length_of_dataframes(dataframes)

  if should_remove_unecessery_columns:
    remove_unecessery_columns(dataframes, COLUMNS_TO_REMOVE)

  if should_format_the_time_column:
    format_the_time_column(dataframes)

  if data_imputation: # imputation before rounding hours
    if should_fill_data_1_missing_value:
      fill_1_missing_values(dataframes, VALUES_TO_FILL, should_print=True)
    if should_fill_data_2_missing_values:
      fill_2_missing_values(dataframes, VALUES_TO_FILL, should_print=True)
    if should_fill_data_3_missing_values:
      fill_3_missing_values(dataframes, VALUES_TO_FILL, should_print=True)

  if should_take_round_hours:
    take_round_hours(dataframes)

  if data_imputation: # imputation after rounding hours
    if should_fill_data_1_missing_value:
      fill_1_missing_values(dataframes, VALUES_TO_FILL, should_print=True)
    if should_fill_data_2_missing_values:
      fill_2_missing_values(dataframes, VALUES_TO_FILL, should_print=True)

  if should_replace_time_with_cyclic_representation:
    replace_time_with_cyclic_representation(dataframes)

  if should_display_wind_before_vectorize:
    display_wind_before_vectorize(dataframes)

  if should_vectorize_wind:
    vectorize_wind(dataframes)

  if should_display_wind_after_vectorize:
    display_wind_after_vectorize(dataframes)
  
  if should_drop_nan_rows:
    drop_nan_rows_multiple(dataframes)

  if should_display_heads_of_dataframes:
    display_dataframes_heads(dataframes)

  if save_data_to_pickles_in_the_end:
    save_dataframes_to_pickles(dataframes, DATA_DIRECTORY)
"""
if __name__ == "__main__":
    main()
"""
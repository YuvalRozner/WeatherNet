import pickle 
import pandas as pd
import os
import json

"""
this file let you load the data of the stations from the pkl files 
and the coordinates of the stations from the json file
"""

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

